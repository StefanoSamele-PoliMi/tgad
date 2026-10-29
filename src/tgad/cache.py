"""Content-addressed anomaly-map cache for expensive model inference."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .backend import AnomalyMapBackend, Shot
from .utils import ensure_dir, file_identity, sha256_json, write_json


@dataclass(frozen=True, slots=True)
class PredictionResult:
    anomaly_map: np.ndarray
    image_score: float
    cache_key: str
    cache_path: Path
    response_text: str | None
    cache_hit: bool
    detection_score: float | None = None


class PredictionCache:
    def __init__(self, root: str | Path, backend: AnomalyMapBackend, *, enabled: bool = True) -> None:
        self.root = ensure_dir(root)
        self.backend = backend
        self.enabled = enabled

    def _key(
        self,
        image_path: Path,
        prompt: str,
        normal_image_paths: Sequence[Path],
        context: dict[str, Any] | None,
    ) -> tuple[str, dict[str, Any]]:
        payload = {
            "format_version": 1,
            "backend": self.backend.fingerprint,
            "image": file_identity(image_path),
            "prompt": prompt,
            "normal_images": [file_identity(path) for path in normal_image_paths],
            "context": context or {},
        }
        return sha256_json(payload), payload

    def predict(
        self,
        image_path: Path,
        prompt: str,
        normal_image_paths: Sequence[Path] = (),
        *,
        shots: Shot | None = None,
        context: dict[str, Any] | None = None,
    ) -> PredictionResult:
        """Return the cached prediction or run the backend.

        ``shots`` is forwarded to the backend only; runners record the regime in
        ``context``, which enters the cache key.
        """
        image_path = Path(image_path).resolve()
        references = tuple(Path(path).resolve() for path in normal_image_paths)
        key, payload = self._key(image_path, prompt, references, context)
        npz_path = self.root / key[:2] / f"{key}.npz"
        json_path = npz_path.with_suffix(".json")
        if self.enabled and npz_path.is_file():
            with np.load(npz_path, allow_pickle=False) as loaded:
                anomaly_map = loaded["anomaly_map"].astype(np.float32, copy=False)
                image_score = float(loaded["image_score"])
                detection_score = (
                    float(loaded["detection_score"]) if "detection_score" in loaded.files else None
                )
            metadata = {}
            if json_path.is_file():
                with json_path.open("r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
            return PredictionResult(
                anomaly_map=anomaly_map,
                image_score=image_score,
                cache_key=key,
                cache_path=npz_path,
                response_text=metadata.get("response_text"),
                cache_hit=True,
                detection_score=detection_score,
            )

        output = self.backend.predict(image_path, prompt, references, shots=shots)
        anomaly_map = np.asarray(output.anomaly_map, dtype=np.float32)
        if anomaly_map.ndim != 2:
            raise ValueError(f"Backend map must be 2-D, got {anomaly_map.shape}")
        image_score = float(np.max(anomaly_map))
        ensure_dir(npz_path.parent)
        if self.enabled:
            arrays = {"anomaly_map": anomaly_map, "image_score": np.float32(image_score)}
            if output.detection_score is not None:
                arrays["detection_score"] = np.float64(output.detection_score)
            with tempfile.NamedTemporaryFile(dir=npz_path.parent, suffix=".npz", delete=False) as handle:
                temp_path = Path(handle.name)
            np.savez_compressed(temp_path, **arrays)
            os.replace(temp_path, npz_path)
            write_json(
                json_path,
                {
                    "cache_key": key,
                    "request": payload,
                    "response_text": output.response_text,
                    "map_shape": list(anomaly_map.shape),
                    "image_score": image_score,
                    "detection_score": output.detection_score,
                },
            )
        return PredictionResult(
            anomaly_map=anomaly_map,
            image_score=image_score,
            cache_key=key,
            cache_path=npz_path,
            response_text=output.response_text,
            cache_hit=False,
            detection_score=output.detection_score,
        )
