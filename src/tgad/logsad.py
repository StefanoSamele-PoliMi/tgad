"""LogSAD backend: LogSAD's structural pathway and the TGAD foreground-masking extension.

Upstream LogSAD (``zhang0jhon/LogSAD``) is written for the five MVTec LOCO categories: its
``MyModel`` hard-codes per-category query words, cluster counts and score statistics, and
its logical (histogram/instance) branch exists only for those categories. On MVTec AD and
on the APD the paper's experiments use LogSAD's structural, PatchCore-style pathway, which
this adapter reproduces from the pinned upstream components:

- ``open_clip_local``: the DataComp CLIP ViT-L/14 returning intermediate patch tokens;
- ``KCenterGreedy`` from the vendored anomalib: coreset subsampling of the memory bank;
- the feature, memory-bank and matching steps of ``MyModel.forward_one_sample`` and
  ``MyModel.process_k_shot`` at their upstream settings: layers 6/12/18/24 at 448x448,
  32x32 tokens upsampled to 64x64 (bilinear, aligned corners), cosine-similarity matching
  per layer, anomaly = 1 - best match, averaged over layers, image score = map maximum.

Inputs follow the paper runs: the image is resized to 448 (antialiased, as anomalib's
datamodule does), down to 224, zero-padded back to 448, and CLIP-normalized; the 64x64 map
is upsampled to 448 and its central 224x224 content is returned.

Supervision regimes follow upstream's two protocols. 1-shot keeps every patch feature of
the reference images (``model_ensemble_few_shot``). Full-shot splits the references into
chunks of four, keeps a 25% ``KCenterGreedy`` coreset of each chunk, and concatenates them
(``compute_coreset.py`` with ``model_ensemble``); like upstream, it drops the last
``n % 4`` images.

The foreground-masking extension (FMS) is Equations (2)-(3) of Section 3.1. Each keyword is
expanded into upstream's template ensemble and encoded with CLIP (``encode_obj_text``); the
foreground keywords are averaged into the focus embedding, and each background keyword is a
contrast embedding. A patch of the 64x64 grid is foreground when the cosine similarity of
its projected CLIP feature to the focus embedding exceeds its maximum similarity to the
contrast embeddings, and the mask multiplies the structural map. The keywords never reach
the map without FMS.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from .backend import (
    _CLIP_MEAN,
    _CLIP_STD,
    FULL_SHOT,
    AnomalyMapBackend,
    BackendOutput,
    Shot,
    _import_from_checkout,
    _repository_revision,
)
from .config import LogSADModelConfig
from .exceptions import BackendConfigurationError
from .utils import sha256_json


# Upstream MyModel settings.
FEATURE_LAYERS = (6, 12, 18, 24)
TOKEN_GRID = 32  # 448 / 14
FEATURE_GRID = 64
INPUT_SIZE = 448
CHUNK_IMAGES = 4
CORESET_RATIO = 0.25
# Paper runs: 224x224 content zero-padded to the 448x448 input.
CONTENT_SIZE = 224
# Rows of the memory bank matched at once; the result does not depend on it.
_MATCH_CHUNK = 32768


@contextlib.contextmanager
def _quiet() -> Any:
    """Silence upstream progress output (rich and tqdm bars)."""

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def parse_logsad_prompt(prompt: str) -> tuple[list[str], list[str]]:
    """Return the foreground and background keywords of a LogSAD prompt resource entry."""

    data = json.loads(prompt)
    foreground = [str(keyword) for keyword in data["foreground"]]
    background = [str(keyword) for keyword in data["background"]]
    if not foreground or not background:
        raise ValueError(f"A LogSAD prompt needs foreground and background keywords: {prompt}")
    return foreground, background


def keyword_groups(foreground: Sequence[str], background: Sequence[str]) -> list[Any]:
    """Query words for upstream ``encode_obj_text``: row 0 focus, rows 1.. contrast.

    ``encode_obj_text`` averages the template embeddings of a list entry into one row, so
    the foreground keywords form the single focus embedding of Equation (2) and each
    background keyword its own contrast embedding.
    """

    return [list(foreground), *background]


def foreground_mask(patch_features: Any, anchors: Any) -> Any:
    """Equation (2): patches closer to the focus embedding than to every contrast embedding.

    ``patch_features`` are L2-normalized ``[N, D]`` patch embeddings, ``anchors`` the
    L2-normalized ``[1 + K, D]`` text embeddings (focus first); returns a bool ``[N]`` mask.
    """

    similarity = patch_features @ anchors.T
    return similarity[:, 0] > similarity[:, 1:].max(dim=1).values


def _load_upstream_file(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise BackendConfigurationError(f"Cannot load upstream module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_kcenter_greedy(checkout: Path) -> Any:
    """Load upstream ``KCenterGreedy`` from the vendored anomalib source files.

    Importing it through the package would execute ``anomalib/models/__init__.py``, which
    imports every anomalib model and Lightning. The sampler needs only its random
    projection, which is registered under the module name it imports.
    """

    components = checkout / "anomalib" / "models" / "components"
    dependency = "anomalib.models.components.dimensionality_reduction"
    previous = sys.modules.get(dependency)
    _load_upstream_file(
        dependency, components / "dimensionality_reduction" / "random_projection.py"
    )
    try:
        module = _load_upstream_file(
            "tgad_logsad_k_center_greedy", components / "sampling" / "k_center_greedy.py"
        )
    finally:
        if previous is None:
            sys.modules.pop(dependency, None)
        else:
            sys.modules[dependency] = previous
    return module.KCenterGreedy


@dataclass(slots=True)
class _ReferenceState:
    key: tuple[Any, ...]
    # Per-layer L2-normalized memory banks.
    memory: list[Any]


class OfficialLogSADBackend(AnomalyMapBackend):
    """LogSAD's structural pathway, optionally with the foreground-masking extension."""

    def __init__(self, config: LogSADModelConfig, *, foreground_masking: bool = False) -> None:
        config.validate_paths()
        self.config = config
        self.foreground_masking = foreground_masking
        self.model_name = "logsad_fms" if foreground_masking else "logsad"
        self._code_dir = config.official_repo.resolve()
        self._torch = None
        self._F = None
        self._TF = None
        self._device = None
        self._clip = None
        self._tokenizer = None
        self._kcenter_greedy = None
        self._encode_obj_text = None
        self._state: _ReferenceState | None = None
        self._anchors: dict[tuple[tuple[str, ...], tuple[str, ...]], Any] = {}

    @property
    def fingerprint(self) -> str:
        payload = {
            "backend": "official-logsad",
            "official_repo_revision": _repository_revision(
                self.config.official_repo, self.config.official_repo / "model_ensemble.py"
            ),
            "clip_model": self.config.clip_model,
            "device": self.config.device,
            "seed": self.config.seed,
            "foreground_masking": self.foreground_masking,
            "adapter_version": 1,
        }
        if self.foreground_masking:
            payload["foreground_mask"] = "equation-2-template-ensemble"
        return f"official-{self.model_name}-" + sha256_json(payload)[:20]

    # ------------------------------------------------------------------ loading
    def _load_model(self) -> None:
        if self._clip is not None:
            return
        try:
            import torch
            import torch.nn.functional as F
            import torchvision.transforms.functional as TF
        except ImportError as exc:
            raise BackendConfigurationError(
                "PyTorch and torchvision are required for the LogSAD backend"
            ) from exc

        device = torch.device(self.config.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise BackendConfigurationError(
                    f"CUDA is unavailable but the LogSAD config requests {self.config.device}"
                )
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        code_text = str(self._code_dir)
        if code_text not in sys.path:
            sys.path.insert(0, code_text)
        try:
            open_clip = _import_from_checkout("open_clip_local", self._code_dir)
            clip, _, _ = open_clip.create_model_and_transforms(self.config.clip_model)
            tokenizer = open_clip.get_tokenizer(self.config.clip_model)
            self._kcenter_greedy = load_kcenter_greedy(self._code_dir)
            if self.foreground_masking:
                prompt_ensemble = _import_from_checkout("prompt_ensemble", self._code_dir)
                self._encode_obj_text = prompt_ensemble.encode_obj_text
        except BackendConfigurationError:
            raise
        except Exception as exc:
            raise BackendConfigurationError(
                "Failed to initialize the LogSAD components. Verify the pinned official "
                "checkout, the LogSAD environment, and Hugging Face access "
                f"to {self.config.clip_model}."
            ) from exc

        self._torch, self._F, self._TF = torch, F, TF
        self._device = device
        self._tokenizer = tokenizer
        self._clip = clip.to(device).eval()

    # ------------------------------------------------------------------ features
    def _load_image(self, path: Path) -> Any:
        """Return the normalized ``[1, 3, 448, 448]`` input of the paper runs."""

        F, TF = self._F, self._TF
        with Image.open(path) as image:
            tensor = TF.to_tensor(image.convert("RGB")).unsqueeze(0)
        tensor = TF.resize(
            tensor,
            [INPUT_SIZE, INPUT_SIZE],
            interpolation=TF.InterpolationMode.BILINEAR,
            antialias=True,
        )
        tensor = F.interpolate(
            tensor, size=(CONTENT_SIZE, CONTENT_SIZE), mode="bilinear", align_corners=False
        )
        pad = (INPUT_SIZE - CONTENT_SIZE) // 2
        tensor = F.pad(tensor, (pad, pad, pad, pad), mode="constant", value=0.0)
        tensor = TF.normalize(tensor, mean=_CLIP_MEAN, std=_CLIP_STD)
        return tensor.to(self._device)

    def _upsample(self, tokens: Any, batch_size: int) -> Any:
        """32x32 tokens to a 64x64 grid (bilinear, aligned corners), then L2-normalized."""

        F = self._F
        grid = tokens.reshape(batch_size, TOKEN_GRID, TOKEN_GRID, -1).permute(0, 3, 1, 2)
        grid = F.interpolate(
            grid, size=(FEATURE_GRID, FEATURE_GRID), mode="bilinear", align_corners=True
        )
        flat = grid.permute(0, 2, 3, 1).reshape(-1, grid.shape[1])
        return F.normalize(flat, p=2, dim=-1)

    def _features(self, batch: Any) -> tuple[Any, Any]:
        """Patch features (4 layers) and projected patch tokens, both on the 64x64 grid."""

        torch = self._torch
        batch_size = batch.shape[0]
        with torch.no_grad():
            _, patch_tokens, projected = self._clip.encode_image(batch, list(FEATURE_LAYERS))
            tokens = [
                p[:, 1:, :].reshape(batch_size * TOKEN_GRID * TOKEN_GRID, p.shape[-1])
                for p in patch_tokens
            ]
            patch = self._upsample(torch.cat(tokens, dim=-1), batch_size)
            projected = self._upsample(
                projected.reshape(batch_size * TOKEN_GRID * TOKEN_GRID, -1), batch_size
            )
        return patch, projected

    # ------------------------------------------------------------------ references
    def _seed(self) -> None:
        # Memory coresets draw from the global NumPy and torch generators.
        np.random.seed(self.config.seed)
        self._torch.manual_seed(self.config.seed)

    def _reference_state(self, references: tuple[Path, ...], coreset: bool) -> _ReferenceState:
        key = (references, coreset)
        if self._state is not None and self._state.key == key:
            return self._state
        torch, F = self._torch, self._F
        self._state = None
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
        self._seed()
        chunk = min(CHUNK_IMAGES, len(references))
        # A full-shot bank reaches several GB (e.g. 391 hazelnut images: ~400k x 4096),
        # so per-layer pieces are collected on the CPU and moved to the device once.
        layers: list[list[Any]] = [[] for _ in FEATURE_LAYERS]

        def collect(bank: Any) -> None:
            for index, part in enumerate(bank.chunk(len(FEATURE_LAYERS), dim=-1)):
                layers[index].append(F.normalize(part, dim=-1).cpu())

        with torch.no_grad():
            if coreset:
                for start in range(0, len(references) // chunk * chunk, chunk):
                    batch = torch.cat([self._load_image(p) for p in references[start : start + chunk]])
                    patch, _ = self._features(batch)
                    sampler = self._kcenter_greedy(embedding=patch, sampling_ratio=CORESET_RATIO)
                    with _quiet():
                        collect(sampler.sample_coreset())
            else:
                for path in references:
                    collect(self._features(self._load_image(path))[0])
            per_layer = []
            for pieces in layers:
                per_layer.append(torch.cat(pieces, dim=0).to(self._device))
                pieces.clear()
        self._state = _ReferenceState(key=key, memory=per_layer)
        return self._state

    # ------------------------------------------------------------------ scoring
    def _structural_map(self, patch: Any, memory: Sequence[Any]) -> Any:
        torch, F = self._torch, self._F
        maps = []
        for feature, bank in zip(patch.chunk(len(FEATURE_LAYERS), dim=-1), memory):
            feature = F.normalize(feature, dim=-1)
            best = torch.cat(
                [
                    (feature @ part.T).max(dim=1, keepdim=True)[0]
                    for part in bank.split(_MATCH_CHUNK, dim=0)
                ],
                dim=1,
            ).max(dim=1)[0]
            maps.append(1 - best)
        return torch.stack(maps).mean(0).reshape(FEATURE_GRID, FEATURE_GRID)

    def _text_anchors(self, foreground: Sequence[str], background: Sequence[str]) -> Any:
        """Focus and contrast embeddings of upstream's template ensemble (``[1 + K, D]``)."""

        key = (tuple(foreground), tuple(background))
        anchors = self._anchors.get(key)
        if anchors is None:
            anchors = self._encode_obj_text(
                self._clip,
                keyword_groups(foreground, background),
                self._tokenizer,
                self._device,
            )
            self._anchors[key] = anchors
        return anchors

    def predict(
        self,
        image_path: Path,
        prompt: str,
        normal_image_paths: Sequence[Path] = (),
        shots: Shot | None = None,
    ) -> BackendOutput:
        foreground, background = parse_logsad_prompt(prompt)
        if not normal_image_paths:
            raise BackendConfigurationError(
                "LogSAD needs defect-free reference images (1-shot or full-shot)"
            )
        self._load_model()
        torch, F = self._torch, self._F
        references = tuple(Path(path).resolve() for path in normal_image_paths)
        coreset = shots == FULL_SHOT if shots is not None else len(references) > 1
        state = self._reference_state(references, coreset)

        image = self._load_image(Path(image_path))
        patch, projected = self._features(image)
        with torch.no_grad():
            anomaly_map = self._structural_map(patch, state.memory)
            if self.foreground_masking:
                mask = foreground_mask(projected, self._text_anchors(foreground, background))
                anomaly_map = anomaly_map * mask.reshape(FEATURE_GRID, FEATURE_GRID).to(anomaly_map)
            full = F.interpolate(
                anomaly_map[None, None],
                size=(INPUT_SIZE, INPUT_SIZE),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
        start = (INPUT_SIZE - CONTENT_SIZE) // 2
        array = full[start : start + CONTENT_SIZE, start : start + CONTENT_SIZE].float().cpu().numpy()
        if not np.isfinite(array).all():
            raise RuntimeError("LogSAD backend returned non-finite anomaly-map values")
        return BackendOutput(array.astype(np.float32, copy=False))
