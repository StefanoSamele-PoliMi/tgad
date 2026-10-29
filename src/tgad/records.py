"""Shared immutable records used by all three scenarios."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ImageSample:
    """A single image and its benchmark metadata.

    Masks are loaded lazily by the dataset-specific loader so that large datasets do
    not reside in memory during model inference.
    """

    sample_id: str
    image_path: Path
    is_anomalous: bool
    class_name: str | None = None
    defect_type: str | None = None
    mask_path: Path | None = None
    labels: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)


@dataclass(frozen=True, slots=True)
class Prediction:
    """One anomaly-map prediction."""

    anomaly_map_path: Path
    image_score: float
    cache_key: str
    response_text: str | None = None
