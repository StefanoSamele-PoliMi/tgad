"""Configuration and packaged-resource loading."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Callable, Sequence

import yaml

from .exceptions import BackendConfigurationError


RESOURCE_PACKAGE = "tgad.resources"


def load_yaml(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_resource_yaml(name: str) -> Any:
    traversable = files(RESOURCE_PACKAGE).joinpath(name)
    with traversable.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _model_section(
    path: str | Path, key: str, required: Sequence[str]
) -> tuple[dict[str, Any], Callable[[Any], Path]]:
    """Return the ``key`` mapping of a model YAML and a resolver for its paths.

    Relative paths are resolved against the directory of the YAML file.
    """

    raw = load_yaml(path)
    if not isinstance(raw, dict):
        raise BackendConfigurationError(f"Model config must be a mapping: {path}")
    model = raw.get(key, raw)
    if not isinstance(model, dict):
        raise BackendConfigurationError(f"The '{key}' entry must be a mapping: {path}")
    missing = [name for name in required if not model.get(name)]
    if missing:
        raise BackendConfigurationError(
            f"Missing required keys in model config {path}: {', '.join(missing)}"
        )
    base = Path(path).expanduser().resolve().parent

    def resolve(value: Any) -> Path:
        candidate = Path(str(value)).expanduser()
        if not candidate.is_absolute():
            candidate = base / candidate
        return candidate.resolve()

    return model, resolve


@dataclass(frozen=True, slots=True)
class AnomalyGPTModelConfig:
    official_repo: Path
    imagebind_checkpoint: Path
    vicuna_checkpoint: Path
    pandagpt_delta: Path
    anomalygpt_delta: Path
    device: str = "cuda:0"
    inference_path: str = "extract"
    text_conditioning: str = "prompt"
    max_target_length: int = 512
    top_p: float = 0.1
    temperature: float = 1.0

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AnomalyGPTModelConfig":
        model, resolve = _model_section(
            path,
            "anomalygpt",
            [
                "official_repo",
                "imagebind_checkpoint",
                "vicuna_checkpoint",
                "pandagpt_delta",
                "anomalygpt_delta",
            ],
        )
        return cls(
            official_repo=resolve(model["official_repo"]),
            imagebind_checkpoint=resolve(model["imagebind_checkpoint"]),
            vicuna_checkpoint=resolve(model["vicuna_checkpoint"]),
            pandagpt_delta=resolve(model["pandagpt_delta"]),
            anomalygpt_delta=resolve(model["anomalygpt_delta"]),
            device=str(model.get("device", "cuda:0")),
            inference_path=str(model.get("inference_path", "extract")),
            text_conditioning=str(model.get("text_conditioning", "prompt")),
            max_target_length=int(model.get("max_target_length", 512)),
            top_p=float(model.get("top_p", 0.1)),
            temperature=float(model.get("temperature", 1.0)),
        )

    def validate_paths(self) -> None:
        checks = {
            "official AnomalyGPT repository": self.official_repo,
            "ImageBind checkpoint": self.imagebind_checkpoint,
            "Vicuna checkpoint directory": self.vicuna_checkpoint,
            "PandaGPT delta": self.pandagpt_delta,
            "AnomalyGPT delta": self.anomalygpt_delta,
        }
        missing = [f"{label}: {path}" for label, path in checks.items() if not path.exists()]
        if missing:
            raise BackendConfigurationError(
                "Required model resources are missing:\n  - " + "\n  - ".join(missing)
            )
        code_dir = self.official_repo / "code"
        if not (code_dir / "model" / "openllama.py").is_file():
            raise BackendConfigurationError(
                f"The official repository does not look like AnomalyGPT: {self.official_repo}"
            )
        if self.inference_path not in {"extract", "generate"}:
            raise BackendConfigurationError(
                f"inference_path must be 'extract' or 'generate', got {self.inference_path!r}"
            )
        if self.text_conditioning not in {"prompt", "upstream"}:
            raise BackendConfigurationError(
                "text_conditioning must be 'prompt' or 'upstream', "
                f"got {self.text_conditioning!r}"
            )
        if not self.device.startswith("cuda"):
            raise BackendConfigurationError(
                "The official AnomalyGPT implementation is CUDA-only; configure device as cuda or cuda:N"
            )
        if self.max_target_length <= 0:
            raise BackendConfigurationError("max_target_length must be positive")
        if not 0.0 < self.top_p <= 1.0:
            raise BackendConfigurationError("top_p must be in (0, 1]")
        if self.temperature <= 0.0:
            raise BackendConfigurationError("temperature must be positive")


@dataclass(frozen=True, slots=True)
class AAClipModelConfig:
    """Paths and test-time hyper-parameters of the official AA-CLIP model.

    Defaults are the values of the paper runs: OpenAI ViT-L-14-336 adapted on VisA at
    224x224 (``train.py --img_size 224``) with the upstream adapter settings.
    ``image_size`` must equal the resolution the adapters were trained at.
    """

    official_repo: Path
    clip_checkpoint: Path
    text_adapter: Path
    image_adapter: Path
    device: str = "cuda:0"
    model_name: str = "ViT-L-14-336"
    image_size: int = 224
    relu: bool = False
    text_adapt_weight: float = 0.1
    image_adapt_weight: float = 0.1
    text_adapt_until: int = 3
    image_adapt_until: int = 6

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AAClipModelConfig":
        model, resolve = _model_section(
            path,
            "aaclip",
            ["official_repo", "clip_checkpoint", "text_adapter", "image_adapter"],
        )
        return cls(
            official_repo=resolve(model["official_repo"]),
            clip_checkpoint=resolve(model["clip_checkpoint"]),
            text_adapter=resolve(model["text_adapter"]),
            image_adapter=resolve(model["image_adapter"]),
            device=str(model.get("device", "cuda:0")),
            model_name=str(model.get("model_name", "ViT-L-14-336")),
            image_size=int(model.get("image_size", 224)),
            relu=bool(model.get("relu", False)),
            text_adapt_weight=float(model.get("text_adapt_weight", 0.1)),
            image_adapt_weight=float(model.get("image_adapt_weight", 0.1)),
            text_adapt_until=int(model.get("text_adapt_until", 3)),
            image_adapt_until=int(model.get("image_adapt_until", 6)),
        )

    def validate_paths(self) -> None:
        checks = {
            "official AA-CLIP repository": self.official_repo,
            "CLIP checkpoint": self.clip_checkpoint,
            "AA-CLIP text adapter": self.text_adapter,
            "AA-CLIP image adapter": self.image_adapter,
        }
        missing = [f"{label}: {path}" for label, path in checks.items() if not path.exists()]
        if missing:
            raise BackendConfigurationError(
                "Required model resources are missing:\n  - " + "\n  - ".join(missing)
            )
        if not (self.official_repo / "model" / "adapter.py").is_file() or not (
            self.official_repo / "forward_utils.py"
        ).is_file():
            raise BackendConfigurationError(
                f"The official repository does not look like AA-CLIP: {self.official_repo}"
            )
        if not (self.device == "cpu" or self.device.startswith("cuda")):
            raise BackendConfigurationError(
                f"device must be cpu, cuda, or cuda:N, got {self.device!r}"
            )
        # Upstream reshapes the patch tokens into a square grid of 14-pixel patches.
        if self.image_size <= 0 or self.image_size % 14:
            raise BackendConfigurationError("image_size must be a positive multiple of 14")


@dataclass(frozen=True, slots=True)
class LogSADModelConfig:
    """Paths and settings of the official LogSAD components.

    ``clip_model`` is the OpenCLIP identifier LogSAD loads (DataComp ViT-L/14, fetched
    from the Hugging Face Hub). The foreground-masking extension (``--model logsad_fms``)
    uses the same CLIP model and needs no further resources.
    """

    official_repo: Path
    clip_model: str = "hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"
    device: str = "cuda:0"
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LogSADModelConfig":
        model, resolve = _model_section(path, "logsad", ["official_repo"])
        return cls(
            official_repo=resolve(model["official_repo"]),
            clip_model=str(model.get("clip_model", cls.clip_model)),
            device=str(model.get("device", "cuda:0")),
            seed=int(model.get("seed", 42)),
        )

    def validate_paths(self) -> None:
        checks = {"official LogSAD repository": self.official_repo}
        missing = [f"{label}: {path}" for label, path in checks.items() if not path.exists()]
        if missing:
            raise BackendConfigurationError(
                "Required model resources are missing:\n  - " + "\n  - ".join(missing)
            )
        sampler = self.official_repo / "anomalib/models/components/sampling/k_center_greedy.py"
        templates = self.official_repo / "prompt_ensemble.py"
        if not (self.official_repo / "open_clip_local").is_dir() or not (
            sampler.is_file() and templates.is_file()
        ):
            raise BackendConfigurationError(
                f"The official repository does not look like LogSAD: {self.official_repo}"
            )
        if not (self.device == "cpu" or self.device.startswith("cuda")):
            raise BackendConfigurationError(
                f"device must be cpu, cuda, or cuda:N, got {self.device!r}"
            )
