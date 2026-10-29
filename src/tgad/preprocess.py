"""Spatial transforms used by the benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from .utils import ensure_dir, file_identity, sha256_json


SpatialStrategy = Literal["direct_resize", "pad_to_square"]
MaskInterpolation = Literal["nearest", "bilinear_threshold_0.1"]


@dataclass(frozen=True, slots=True)
class SpatialGeometry:
    source_width: int
    source_height: int
    target_size: int
    resized_width: int
    resized_height: int
    left: int
    top: int


def spatial_geometry(width: int, height: int, strategy: SpatialStrategy, target_size: int) -> SpatialGeometry:
    """Return the exact resize/padding geometry used for an image and its mask."""

    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive")
    if target_size <= 0:
        raise ValueError("target_size must be positive")
    if strategy == "direct_resize":
        return SpatialGeometry(width, height, target_size, target_size, target_size, 0, 0)
    if strategy != "pad_to_square":
        raise ValueError(f"Unknown spatial strategy: {strategy}")
    scale = min(target_size / width, target_size / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    return SpatialGeometry(
        width,
        height,
        target_size,
        resized_width,
        resized_height,
        (target_size - resized_width) // 2,
        (target_size - resized_height) // 2,
    )


def _as_binary_mask(mask: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(mask, Image.Image):
        return mask.convert("L")
    array = np.asarray(mask)
    if array.ndim == 3:
        array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Mask must be two-dimensional after squeeze; got {array.shape}")
    binary = (array > 0).astype(np.uint8) * 255
    return Image.fromarray(binary, mode="L")


def transform_image(
    image: Image.Image,
    strategy: SpatialStrategy,
    target_size: int = 224,
) -> Image.Image:
    """Transform an RGB image according to the Scenario 3 protocol."""

    image = image.convert("RGB")
    geometry = spatial_geometry(*image.size, strategy, target_size)
    resized = image.resize(
        (geometry.resized_width, geometry.resized_height),
        Image.Resampling.BICUBIC,
    )
    if strategy == "direct_resize":
        return resized
    canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    canvas.paste(resized, (geometry.left, geometry.top))
    return canvas


def transform_binary_mask(
    mask: Image.Image | np.ndarray,
    strategy: SpatialStrategy,
    target_size: int = 224,
    *,
    interpolation: MaskInterpolation = "nearest",
) -> np.ndarray:
    """Apply an image strategy to a binary mask and return bool ``[H, W]``."""

    mask_image = _as_binary_mask(mask)
    if interpolation not in {"nearest", "bilinear_threshold_0.1"}:
        raise ValueError(f"Unknown mask interpolation mode: {interpolation}")
    resample = (
        Image.Resampling.NEAREST
        if interpolation == "nearest"
        else Image.Resampling.BILINEAR
    )
    geometry = spatial_geometry(*mask_image.size, strategy, target_size)
    resized = mask_image.resize(
        (geometry.resized_width, geometry.resized_height),
        resample,
    )
    if strategy == "direct_resize":
        transformed = resized
    else:
        transformed = Image.new("L", (target_size, target_size), 0)
        transformed.paste(resized, (geometry.left, geometry.top))

    array = np.asarray(transformed, dtype=np.float32) / 255.0
    threshold = 0.1 if interpolation == "bilinear_threshold_0.1" else 0.5
    return array > threshold


def load_mvtec_mask(
    mask_path: Path | None,
    image_path: Path,
    target_size: int = 224,
) -> np.ndarray:
    """Match official AnomalyGPT's MVTec mask transform.

    ``image_path`` is accepted for a uniform loader signature and for clearer call
    sites; normal images simply receive an all-zero mask.
    """

    del image_path
    if mask_path is None:
        return np.zeros((target_size, target_size), dtype=bool)
    with Image.open(mask_path) as mask:
        return transform_binary_mask(
            mask,
            "direct_resize",
            target_size,
            interpolation="bilinear_threshold_0.1",
        )


@dataclass(slots=True)
class PreparedImageCache:
    """On-disk cache of square APD inputs passed to the official model."""

    root: Path
    target_size: int = 224

    def __post_init__(self) -> None:
        if self.target_size <= 0:
            raise ValueError("target_size must be positive")
        self.root = ensure_dir(self.root)

    def prepare(self, image_path: Path, strategy: SpatialStrategy) -> Path:
        identity = {
            "source": file_identity(image_path),
            "strategy": strategy,
            "target_size": self.target_size,
            "format_version": 2,
        }
        key = sha256_json(identity)
        destination = self.root / strategy / key[:2] / f"{key}.png"
        if destination.is_file():
            return destination
        ensure_dir(destination.parent)
        with Image.open(image_path) as image:
            transformed = transform_image(image, strategy, self.target_size)
            temp = destination.with_suffix(".tmp.png")
            transformed.save(temp, format="PNG", optimize=False)
            temp.replace(destination)
        return destination
