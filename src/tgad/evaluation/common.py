"""Shared evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image

from ..backend import AnomalyMapBackend, Shot, parse_shot
from ..metrics import MetricBundle
from ..utils import ensure_dir, write_csv, write_json


def select_shots(
    backend: AnomalyMapBackend,
    requested: Sequence[Shot] | None,
    default: Sequence[Shot],
) -> tuple[Shot, ...]:
    """Return the requested shot regimes (or ``default``), checked against the model."""

    selected = tuple(parse_shot(shot) for shot in (default if requested is None else requested))
    if not selected:
        raise ValueError("At least one shot regime is required")
    spec = backend.spec
    unsupported = sorted(set(selected) - set(spec.shots))
    if unsupported:
        raise ValueError(
            f"{spec.display_name} is evaluated in the {list(spec.shots)}-shot regime(s) only; "
            f"requested {unsupported}"
        )
    return selected


def reference_record(references: Sequence[Path]) -> list[str] | dict[str, Any]:
    """Reference images of a prediction; full-shot sets are summarized, not listed."""

    if len(references) <= 4:
        return [str(path) for path in references]
    return {
        "count": len(references),
        "directory": str(Path(references[0]).parent),
        "first": str(references[0]),
        "last": str(references[-1]),
    }


def resize_anomaly_map(anomaly_map: np.ndarray, target_size: int = 224) -> np.ndarray:
    array = np.asarray(anomaly_map, dtype=np.float32)
    if array.shape == (target_size, target_size):
        return array
    image = Image.fromarray(array, mode="F")
    resized = image.resize((target_size, target_size), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def metric_row(bundle: MetricBundle, **metadata: Any) -> dict[str, Any]:
    return {**metadata, **bundle.as_dict(percentage=True)}


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def finalize_output(
    output_dir: Path,
    *,
    per_unit_rows: Sequence[dict[str, Any]],
    summary_rows: Sequence[dict[str, Any]],
    prediction_rows: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
    per_unit_name: str,
) -> dict[str, Path]:
    ensure_dir(output_dir)
    per_unit_path = output_dir / per_unit_name
    summary_path = output_dir / "summary.csv"
    predictions_path = output_dir / "predictions.jsonl"
    metadata_path = output_dir / "run_metadata.json"
    write_csv(per_unit_path, list(per_unit_rows))
    write_csv(summary_path, list(summary_rows))
    write_jsonl(predictions_path, prediction_rows)
    write_json(metadata_path, metadata)
    return {
        "per_unit": per_unit_path,
        "summary": summary_path,
        "predictions": predictions_path,
        "metadata": metadata_path,
    }
