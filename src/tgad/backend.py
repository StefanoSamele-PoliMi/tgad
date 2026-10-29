"""Anomaly-map backends: the official pre-trained AnomalyGPT and AA-CLIP adapters."""

from __future__ import annotations

import importlib
import subprocess
import sys
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np
from PIL import Image

from .config import AAClipModelConfig, AnomalyGPTModelConfig
from .exceptions import BackendConfigurationError
from .utils import pushd, sha256_json


# Supervision regime: 0-shot, 1-shot, or "full" (every defect-free training image).
Shot = int | str
FULL_SHOT = "full"
SHOT_ORDER: tuple[Shot, ...] = (0, 1, FULL_SHOT)


def parse_shot(value: object) -> Shot:
    """Parse a shot regime from the CLI or YAML: ``0``, ``1``, or ``full``."""

    shot = FULL_SHOT if str(value).strip().lower() == FULL_SHOT else int(str(value))
    if shot not in SHOT_ORDER:
        raise ValueError(f"Unknown shot regime {value!r}; expected one of {list(SHOT_ORDER)}")
    return shot


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Protocol properties of one evaluated model."""

    display_name: str
    # How the benchmark text reaches the model, recorded in run metadata.
    prompt_handling: str
    # Key of IMAGE_SCORE_RULES used for the image-level decision.
    image_score_rule: str
    # Supervision regimes the model is evaluated in (Section 3.2).
    shots: tuple[Shot, ...]
    # Regime of Scenario 2 and Per-Error AD, where the paper reports one regime per model.
    default_shot: Shot
    # MVTec 1-shot reference: train/good/{4 * round:03d}.png.
    mvtec_reference_round: int = 3


_LOGSAD_PROMPTS = (
    "foreground and background keywords; the released LogSAD pipeline does not use them for "
    "the structural anomaly map"
)
_LOGSAD_FMS_PROMPTS = (
    "foreground and background keywords, each expanded into upstream's template ensemble, "
    "select the per-patch foreground mask of Equation (2) (focus similarity above the maximum "
    "contrast similarity) applied to the structural anomaly map"
)

MODEL_SPECS: dict[str, ModelSpec] = {
    "anomalygpt": ModelSpec(
        display_name="AnomalyGPT",
        prompt_handling="verbatim prompt passed unchanged to AnomalyGPT",
        image_score_rule="map_max",
        shots=(0, 1),
        default_shot=0,
    ),
    "aaclip": ModelSpec(
        display_name="AA-CLIP",
        prompt_handling=(
            "verbatim keyword inserted unchanged into the Appendix C.3 normal/abnormal "
            "states and templates, encoded by the official AA-CLIP code"
        ),
        image_score_rule="aaclip_fusion",
        shots=(0,),
        default_shot=0,
    ),
    # LogSAD's evaluation script takes the first training images as references (round 0).
    "logsad": ModelSpec(
        display_name="LogSAD",
        prompt_handling=_LOGSAD_PROMPTS,
        image_score_rule="map_max",
        shots=(1, FULL_SHOT),
        default_shot=FULL_SHOT,
        mvtec_reference_round=0,
    ),
    "logsad_fms": ModelSpec(
        display_name="LogSAD (FMS)",
        prompt_handling=_LOGSAD_FMS_PROMPTS,
        image_score_rule="map_max",
        shots=(1, FULL_SHOT),
        default_shot=FULL_SHOT,
        mvtec_reference_round=0,
    ),
}

IMAGE_SCORE_RULES = {
    "map_max": "maximum anomaly-map value",
    "aaclip_fusion": (
        "0.5 * min-max-normalized anomaly-map maximum + 0.5 * min-max-normalized global "
        "detection score, both normalized over the evaluation set (official AA-CLIP test.py)"
    ),
}


@dataclass(frozen=True, slots=True)
class BackendOutput:
    anomaly_map: np.ndarray
    response_text: str | None = None
    # Image-level score from a dedicated model head (AA-CLIP's pooled detection token).
    detection_score: float | None = None


def _min_max(values: np.ndarray, low: float, high: float) -> np.ndarray:
    span = high - low
    return (values - low) / span if span > 0 else np.zeros_like(values)


def image_level_scores(
    model_name: str,
    anomaly_maps: Sequence[np.ndarray],
    detection_scores: Sequence[float | None],
) -> list[float]:
    """Image-level scores of one evaluation set, following the model's official test code.

    AA-CLIP min-max normalizes its pixel maps and detection scores over the whole set
    before fusing them, so a score depends on the set it is computed in: the same image
    can score differently under EV1 and EV2, or in two APD categories.
    """

    rule = MODEL_SPECS[model_name].image_score_rule
    if rule == "map_max":
        return [float(np.max(anomaly_map)) for anomaly_map in anomaly_maps]
    if any(score is None for score in detection_scores):
        raise ValueError(f"{model_name} image scores require a detection score per image")
    maps = np.stack([np.asarray(anomaly_map, dtype=np.float64) for anomaly_map in anomaly_maps])
    map_max = _min_max(maps.max(axis=(1, 2)), float(maps.min()), float(maps.max()))
    detection = np.asarray(detection_scores, dtype=np.float64)
    detection = _min_max(detection, float(detection.min()), float(detection.max()))
    return [float(score) for score in 0.5 * map_max + 0.5 * detection]


def _path_signature(path: Path) -> dict[str, object]:
    """Cheap cache-invalidating signature for a model file or directory."""

    resolved = Path(path).resolve()
    if resolved.is_file():
        stat = resolved.stat()
        return {
            "path": str(resolved),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    if resolved.is_dir():
        entries: list[tuple[str, int, int]] = []
        for candidate in sorted(resolved.glob("*")):
            if candidate.is_file():
                stat = candidate.stat()
                entries.append((candidate.name, stat.st_size, stat.st_mtime_ns))
        return {"path": str(resolved), "files": entries}
    return {"path": str(resolved), "missing": True}


def _repository_revision(repository: Path, fallback_file: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return sha256_json(_path_signature(fallback_file))


def _import_from_checkout(name: str, checkout: Path) -> Any:
    """Import ``name`` and check it resolves inside ``checkout``.

    The official repositories use generic top-level names (``model``, ``utils``), so a
    package of the same name loaded earlier in the process would silently shadow them.
    """

    top_level = sys.modules.get(name.split(".")[0])
    if top_level is not None:
        # Namespace packages (AA-CLIP's ``model``) have no __file__, only __path__.
        file = getattr(top_level, "__file__", None)
        origins = [Path(file)] if file else [Path(p) for p in getattr(top_level, "__path__", [])]
        if not any(checkout in origin.resolve().parents for origin in origins):
            raise BackendConfigurationError(
                f"A different top-level Python module named {name.split('.')[0]!r} is already "
                f"loaded from {[str(origin) for origin in origins]}. "
                "Start a clean process for this backend."
            )
    module = importlib.import_module(name)
    module_path = Path(module.__file__).resolve()
    if checkout not in module_path.parents:
        raise BackendConfigurationError(f"Imported {name} from the wrong checkout: {module_path}")
    return module


def make_prompt_text_encoder(
    module: Any,
    upstream_encoder: Callable[..., Any],
    tokenizer: Any,
    prompt: str,
) -> Callable[..., Any]:
    """Drop-in replacement for upstream ``encode_text_with_prompt_ensemble``.

    Upstream returns ``[len(obj), 2, D]`` unit-norm features: the normal ensemble at
    index 0, the abnormal ensemble at index 1, both built from a class word. Here the
    normal entry becomes the ImageBind embedding of the whole prompt; the abnormal entry
    stays the upstream ensemble for the class word upstream picked.
    """

    eot = tokenizer.encoder["<|endoftext|>"]

    def encode(model: Any, obj: Sequence[str], device: Any) -> Any:
        features = upstream_encoder(model, obj, device).clone()
        tokens = tokenizer(prompt).to(device)
        if not bool((tokens == eot).any()):
            # The tokenizer truncates at its context length and drops the end token.
            # ImageBind pools at the end token (highest id), so put it back.
            warnings.warn(
                f"Prompt exceeds the ImageBind text context ({tokens.numel()} tokens) "
                "and was truncated for anomaly-map conditioning",
                stacklevel=2,
            )
            tokens[-1] = eot
        text = module.ModalityType.TEXT
        embedding = model({text: tokens.unsqueeze(0)})[text][0]
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        features[:, 0, :] = embedding
        return features

    return encode


class AnomalyMapBackend(ABC):
    """Minimal interface required by all benchmark scenarios."""

    # Key of MODEL_SPECS: selects the prompt resources and the image-score rule.
    model_name: str = "anomalygpt"
    # How the prompt conditions the anomaly map; None when there is no such option.
    text_conditioning: str | None = None

    @property
    def spec(self) -> ModelSpec:
        return MODEL_SPECS[self.model_name]

    @property
    @abstractmethod
    def fingerprint(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        image_path: Path,
        prompt: str,
        normal_image_paths: Sequence[Path] = (),
        shots: Shot | None = None,
    ) -> BackendOutput:
        """Return a float32 anomaly map, plus a response or detection score if any.

        ``shots`` names the supervision regime of the references, for models whose
        reference handling depends on it (LogSAD); the other models ignore it.
        """


class MockAnomalyBackend(AnomalyMapBackend):
    """Lightweight deterministic backend for tests and pipeline dry-runs.

    Zero-shot returns image luminance. One-shot returns absolute luminance
    difference from the mean reference image. With ``model='aaclip'`` the mean of the
    map stands in for the detection score. It is not a scientific baseline.
    """

    def __init__(self, target_size: int = 224, model: str = "anomalygpt") -> None:
        if target_size <= 0:
            raise ValueError("target_size must be positive")
        if model not in MODEL_SPECS:
            raise ValueError(f"Unknown model {model!r}; expected one of {sorted(MODEL_SPECS)}")
        self.target_size = target_size
        self.model_name = model

    @property
    def fingerprint(self) -> str:
        # The anomaly map does not depend on the model; only the detection score does.
        suffix = "" if self.model_name == "anomalygpt" else f"-{self.model_name}"
        return f"mock-luminance-v1-{self.target_size}{suffix}"

    def _load(self, path: Path) -> np.ndarray:
        with Image.open(path) as image:
            gray = image.convert("L").resize(
                (self.target_size, self.target_size), Image.Resampling.BILINEAR
            )
        return np.asarray(gray, dtype=np.float32) / 255.0

    def predict(
        self,
        image_path: Path,
        prompt: str,
        normal_image_paths: Sequence[Path] = (),
        shots: Shot | None = None,
    ) -> BackendOutput:
        del prompt, shots
        query = self._load(image_path)
        if normal_image_paths:
            references = np.stack([self._load(path) for path in normal_image_paths], axis=0)
            anomaly_map = np.abs(query - references.mean(axis=0))
        else:
            anomaly_map = query
        detection_score = float(anomaly_map.mean()) if self.model_name == "aaclip" else None
        return BackendOutput(
            anomaly_map.astype(np.float32, copy=False), detection_score=detection_score
        )


class OfficialAnomalyGPTBackend(AnomalyMapBackend):
    """Adapter around the official ``OpenLLAMAPEFTModel`` implementation.

    This repository deliberately does not reimplement AnomalyGPT. The adapter imports
    ``model.openllama`` from the pinned official checkout and loads the same ImageBind,
    Vicuna-v0, PandaGPT and ``train_visa`` deltas used by ``test_mvtec.py``.

    The benchmark prompt is forwarded verbatim as ``inputs["prompt"]``; no text is
    extracted, rewritten, or appended on this side.

    Upstream builds the zero-shot map from fixed normal/abnormal templates filled with
    the first ``CLASS_NAMES`` entry found in the prompt (or ``'object'``), so no other
    prompt text reaches it. With ``text_conditioning='prompt'`` (default) the adapter
    swaps ``encode_text_with_prompt_ensemble`` for the duration of each call so that the
    normal text embedding is the embedding of the full prompt
    (:func:`make_prompt_text_encoder`). ``'upstream'`` leaves the official module
    untouched. The 1-shot map is unaffected either way: upstream replaces it with visual
    similarity to the reference images.

    ``inference_path='extract'`` calls the official anomaly-map path directly and skips
    autoregressive response generation. The map is identical to the ``pixel_output``
    produced by ``generate``. Select ``generate`` only when response text is needed.
    """

    model_name = "anomalygpt"

    def __init__(self, config: AnomalyGPTModelConfig) -> None:
        config.validate_paths()
        self.config = config
        self.text_conditioning = config.text_conditioning
        self._model = None
        self._module = None
        self._tokenizer = None
        self._torch = None
        self._code_dir = (config.official_repo / "code").resolve()

    @property
    def fingerprint(self) -> str:
        payload = {
            "backend": "official-anomalygpt",
            "official_repo_revision": _repository_revision(
                self.config.official_repo,
                self.config.official_repo / "code" / "model" / "openllama.py",
            ),
            "imagebind_checkpoint": _path_signature(self.config.imagebind_checkpoint),
            "vicuna_checkpoint": _path_signature(self.config.vicuna_checkpoint),
            "pandagpt_delta": _path_signature(self.config.pandagpt_delta),
            "anomalygpt_delta": _path_signature(self.config.anomalygpt_delta),
            "device": self.config.device,
            "inference_path": self.config.inference_path,
            "max_target_length": self.config.max_target_length,
            "top_p": self.config.top_p,
            "temperature": self.config.temperature,
            "adapter_version": 3,
        }
        # Omitted for 'upstream' so caches written before this option stay valid.
        if self.config.text_conditioning != "upstream":
            payload["text_conditioning"] = self.config.text_conditioning
        return "official-anomalygpt-" + sha256_json(payload)[:20]

    @staticmethod
    def _parse_cuda_device(device: str) -> int:
        if device == "cuda":
            return 0
        if device.startswith("cuda:"):
            try:
                return int(device.split(":", 1)[1])
            except ValueError as exc:
                raise BackendConfigurationError(f"Invalid CUDA device: {device}") from exc
        raise BackendConfigurationError(
            "The official AnomalyGPT implementation requires a CUDA device; "
            f"received {device!r}."
        )

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
        except ImportError as exc:
            raise BackendConfigurationError("PyTorch is required for the official backend") from exc

        device_index = self._parse_cuda_device(self.config.device)
        if not torch.cuda.is_available():
            raise BackendConfigurationError(
                "CUDA is unavailable. Use a CUDA environment matching the official "
                "AnomalyGPT requirements."
            )
        torch.cuda.set_device(device_index)
        # Inference contains no stochastic augmentation; disabling autotuning avoids
        # run-to-run kernel selection differences without changing the model path.
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        code_text = str(self._code_dir)
        if code_text not in sys.path:
            sys.path.insert(0, code_text)
        try:
            # The official module creates ImageBind text tensors at import time and
            # uses a relative BPE path, so import from its code directory.
            with pushd(self._code_dir):
                existing_model = sys.modules.get("model")
                if existing_model is not None:
                    origin = Path(getattr(existing_model, "__file__", "") or ".").resolve()
                    if self._code_dir not in origin.parents:
                        raise BackendConfigurationError(
                            "A different top-level Python package named 'model' is already loaded. "
                            "Start a clean process for the official AnomalyGPT backend."
                        )
                module = importlib.import_module("model.openllama")
                module_path = Path(module.__file__).resolve()
                if self._code_dir not in module_path.parents:
                    raise BackendConfigurationError(
                        f"Imported model.openllama from the wrong checkout: {module_path}"
                    )
                model_class = module.OpenLLAMAPEFTModel
                args = {
                    "model": "openllama_peft",
                    "imagebind_ckpt_path": str(self.config.imagebind_checkpoint),
                    "vicuna_ckpt_path": str(self.config.vicuna_checkpoint),
                    "anomalygpt_ckpt_path": str(self.config.anomalygpt_delta),
                    "delta_ckpt_path": str(self.config.pandagpt_delta),
                    "stage": 2,
                    # Constructor value from official test_mvtec.py. Generation length
                    # is supplied independently per request below.
                    "max_tgt_len": 128,
                    "lora_r": 32,
                    "lora_alpha": 32,
                    "lora_dropout": 0.1,
                }
                model = model_class(**args)
                panda_state = torch.load(
                    self.config.pandagpt_delta, map_location=torch.device("cpu")
                )
                model.load_state_dict(panda_state, strict=False)
                anomaly_state = torch.load(
                    self.config.anomalygpt_delta, map_location=torch.device("cpu")
                )
                model.load_state_dict(anomaly_state, strict=False)
                model = model.eval().half().cuda(device_index)
                # Same tokenizer and relative BPE path upstream uses for its templates.
                tokenizer = module.data.SimpleTokenizer(bpe_path=module.data.BPE_PATH)
        except BackendConfigurationError:
            raise
        except Exception as exc:
            raise BackendConfigurationError(
                "Failed to initialize the official AnomalyGPT model. Verify the pinned "
                "official checkout, exact inference dependencies, and all checkpoint paths."
            ) from exc

        self._torch = torch
        self._model = model
        self._module = module
        self._tokenizer = tokenizer

    @contextmanager
    def _conditioned_on(self, prompt: str) -> Iterator[None]:
        if self.config.text_conditioning == "upstream":
            yield
            return
        module = self._module
        upstream_encoder = module.encode_text_with_prompt_ensemble
        module.encode_text_with_prompt_ensemble = make_prompt_text_encoder(
            module, upstream_encoder, self._tokenizer, prompt
        )
        try:
            yield
        finally:
            module.encode_text_with_prompt_ensemble = upstream_encoder

    def predict(
        self,
        image_path: Path,
        prompt: str,
        normal_image_paths: Sequence[Path] = (),
        shots: Shot | None = None,
    ) -> BackendOutput:
        del shots  # the regime follows from the references
        self._load_model()
        assert self._model is not None and self._torch is not None
        inputs = {
            "prompt": prompt,
            "image_paths": [str(Path(image_path).resolve())],
            "audio_paths": [],
            "video_paths": [],
            "thermal_paths": [],
            "normal_img_paths": [str(Path(path).resolve()) for path in normal_image_paths],
            "top_p": self.config.top_p,
            "temperature": self.config.temperature,
            "max_tgt_len": self.config.max_target_length,
            "modality_embeds": [],
        }
        response_text: str | None = None
        with (
            self._torch.inference_mode(),
            pushd(self._code_dir),
            self._conditioned_on(prompt),
        ):
            if self.config.inference_path == "extract":
                _, anomaly_map = self._model.extract_multimodal_feature(inputs, web_demo=False)
            else:
                response, anomaly_map = self._model.generate(inputs)
                response_text = str(response)
        array = anomaly_map.detach().float().cpu().numpy().squeeze()
        if array.ndim != 2:
            raise RuntimeError(f"Official backend returned an unexpected map shape: {array.shape}")
        if not np.isfinite(array).all():
            raise RuntimeError("Official backend returned non-finite anomaly-map values")
        return BackendOutput(array.astype(np.float32, copy=False), response_text=response_text)


# Normalization of OpenAI CLIP, as in the upstream AA-CLIP dataset transforms.
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# Normal/abnormal states and sentence templates of the AA-CLIP text anchors (Appendix C.3).
# They replace the upstream ensemble of dataset/constants.py (normal "{}", "a {}", "the {}";
# five abnormal states; templates "{}." and "a photo of {}.").
AACLIP_PROMPT_STATES = {
    "normal": ("a {}", "a clean {}"),
    "abnormal": ("a damaged {}", "a {} with defect"),
}
AACLIP_PROMPT_TEMPLATES = ("{}.", "a photo of {}.", "this is {}.")


def use_paper_aaclip_prompts(forward_utils: Any) -> None:
    """Point upstream ``forward_utils`` at the Appendix C.3 states and templates.

    ``get_adapted_single_class_text_embedding`` reads the module globals ``prompt_state``
    and ``prompt_templates``, so replacing them changes the anchors without editing the
    pinned checkout.
    """

    forward_utils.prompt_state = [
        list(AACLIP_PROMPT_STATES["normal"]),
        list(AACLIP_PROMPT_STATES["abnormal"]),
    ]
    forward_utils.prompt_templates = list(AACLIP_PROMPT_TEMPLATES)


class OfficialAAClipBackend(AnomalyMapBackend):
    """Adapter around the official AA-CLIP ``AdaptedCLIP`` model.

    As for AnomalyGPT, nothing is reimplemented: the adapter imports ``model.clip``,
    ``model.adapter`` and ``forward_utils`` from the pinned official checkout and follows
    the zero-shot inference of upstream ``test.py``:

    - the benchmark text is AA-CLIP's class keyword. Upstream
      ``get_adapted_single_class_text_embedding`` inserts it verbatim into the normal
      (``"a {}"``, ``"a clean {}"``) and abnormal (``"a damaged {}"``,
      ``"a {} with defect"``) states and the templates of Appendix C.3
      (:data:`AACLIP_PROMPT_STATES`, :data:`AACLIP_PROMPT_TEMPLATES`), and encodes the
      ensemble with the adapted text encoder into two text anchors;
    - the anomaly map is the sum over the four patch-token levels of the blurred
      patch-to-anchor similarity map (``calculate_similarity_map``);
    - the detection score is ``(det_token . abnormal_anchor + 1) / 2``, from the pooled
      detection token. The image-level decision fuses it with the map maximum over the
      evaluation set (:func:`image_level_scores`).

    AA-CLIP is evaluated zero-shot only and takes no reference images.
    """

    model_name = "aaclip"
    # The keyword is registered under these names in upstream's class-name tables so that
    # upstream's own text-embedding function can be called with it.
    _DATASET = "TGAD"
    _CLASS = "tgad_keyword"

    def __init__(self, config: AAClipModelConfig) -> None:
        config.validate_paths()
        self.config = config
        self._code_dir = config.official_repo.resolve()
        self._model = None
        self._torch = None
        self._device = None
        self._transform = None
        self._forward_utils = None
        self._constants = None
        self._anchors: dict[str, Any] = {}

    @property
    def fingerprint(self) -> str:
        payload = {
            "backend": "official-aaclip",
            "official_repo_revision": _repository_revision(
                self.config.official_repo, self.config.official_repo / "model" / "adapter.py"
            ),
            "clip_checkpoint": _path_signature(self.config.clip_checkpoint),
            "text_adapter": _path_signature(self.config.text_adapter),
            "image_adapter": _path_signature(self.config.image_adapter),
            "device": self.config.device,
            "model_name": self.config.model_name,
            "image_size": self.config.image_size,
            "relu": self.config.relu,
            "text_adapt_weight": self.config.text_adapt_weight,
            "image_adapt_weight": self.config.image_adapt_weight,
            "text_adapt_until": self.config.text_adapt_until,
            "image_adapt_until": self.config.image_adapt_until,
            "prompt_states": AACLIP_PROMPT_STATES,
            "prompt_templates": AACLIP_PROMPT_TEMPLATES,
            "adapter_version": 2,
        }
        return "official-aaclip-" + sha256_json(payload)[:20]

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from torchvision import transforms
        except ImportError as exc:
            raise BackendConfigurationError(
                "PyTorch and torchvision are required for the official AA-CLIP backend"
            ) from exc

        device = torch.device(self.config.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise BackendConfigurationError(
                    f"CUDA is unavailable but the AA-CLIP config requests {self.config.device}"
                )
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        code_text = str(self._code_dir)
        if code_text not in sys.path:
            sys.path.insert(0, code_text)
        try:
            _import_from_checkout("utils", self._code_dir)
            clip_module = _import_from_checkout("model.clip", self._code_dir)
            adapter_module = _import_from_checkout("model.adapter", self._code_dir)
            constants = _import_from_checkout("dataset.constants", self._code_dir)
            forward_utils = _import_from_checkout("forward_utils", self._code_dir)
            use_paper_aaclip_prompts(forward_utils)
            # Upstream create_model reads the OpenAI weights from this table, which
            # points inside the checkout by default.
            clip_module._MODEL_CKPT_PATHS[self.config.model_name] = self.config.clip_checkpoint
            clip_model = clip_module.create_model(
                model_name=self.config.model_name,
                img_size=self.config.image_size,
                device=device,
                pretrained="openai",
                require_pretrained=True,
            )
            clip_model.eval()
            model = adapter_module.AdaptedCLIP(
                clip_model=clip_model,
                text_adapt_weight=self.config.text_adapt_weight,
                image_adapt_weight=self.config.image_adapt_weight,
                text_adapt_until=self.config.text_adapt_until,
                image_adapt_until=self.config.image_adapt_until,
                relu=self.config.relu,
            ).to(device)
            model.eval()
            text_state = torch.load(self.config.text_adapter, map_location=device)
            model.text_adapter.load_state_dict(text_state["text_adapter"])
            image_state = torch.load(self.config.image_adapter, map_location=device)
            model.image_adapter.load_state_dict(image_state["image_adapter"])
        except BackendConfigurationError:
            raise
        except Exception as exc:
            raise BackendConfigurationError(
                "Failed to initialize the official AA-CLIP model. Verify the pinned official "
                "checkout, its dependencies, and the CLIP and adapter checkpoint paths."
            ) from exc

        size = self.config.image_size
        self._transform = transforms.Compose(
            [
                transforms.Resize((size, size), transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
            ]
        )
        self._torch = torch
        self._device = device
        self._forward_utils = forward_utils
        self._constants = constants
        self._model = model

    def _text_anchors(self, keyword: str) -> Any:
        """Return the ``[D, 2]`` normal/abnormal text anchors for a keyword."""

        anchors = self._anchors.get(keyword)
        if anchors is None:
            self._constants.CLASS_NAMES[self._DATASET] = [self._CLASS]
            self._constants.REAL_NAMES[self._DATASET] = {self._CLASS: keyword}
            anchors = self._forward_utils.get_adapted_single_class_text_embedding(
                self._model, self._DATASET, self._CLASS, self._device
            )
            self._anchors[keyword] = anchors
        return anchors

    def predict(
        self,
        image_path: Path,
        prompt: str,
        normal_image_paths: Sequence[Path] = (),
        shots: Shot | None = None,
    ) -> BackendOutput:
        del shots
        if normal_image_paths:
            raise BackendConfigurationError(
                "AA-CLIP is evaluated zero-shot only and takes no reference images"
            )
        self._load_model()
        assert self._model is not None and self._torch is not None
        with Image.open(image_path) as image:
            tensor = self._transform(image.convert("RGB")).unsqueeze(0).to(self._device)
        with self._torch.no_grad():
            anchors = self._text_anchors(prompt)
            patch_features, det_feature = self._model(tensor)
            detection = det_feature @ anchors
            detection_score = float((detection[0, 1] + 1) / 2)
            # MVTec AD and APD are both in upstream's "Industrial" domain (blur sigma 1).
            level_maps = [
                self._forward_utils.calculate_similarity_map(
                    features, anchors, self.config.image_size, test=True, domain="Industrial"
                )
                for features in patch_features
            ]
            anomaly_map = self._torch.cat(level_maps, dim=1).sum(1)[0]
        array = anomaly_map.detach().float().cpu().numpy()
        if array.ndim != 2:
            raise RuntimeError(f"AA-CLIP backend returned an unexpected map shape: {array.shape}")
        if not np.isfinite(array).all() or not np.isfinite(detection_score):
            raise RuntimeError("AA-CLIP backend returned non-finite scores")
        return BackendOutput(array.astype(np.float32, copy=False), detection_score=detection_score)
