"""Image-level and localization metrics used in the TGAD benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy import ndimage
from sklearn.metrics import auc, roc_auc_score


@dataclass(frozen=True, slots=True)
class MetricBundle:
    i_auroc: float
    p_auroc: float
    aupro: float
    n_images: int
    n_positive_images: int
    n_regions: int

    def as_dict(self, *, percentage: bool = True) -> dict[str, float | int]:
        scale = 100.0 if percentage else 1.0
        return {
            "i_auroc": self.i_auroc * scale,
            "p_auroc": self.p_auroc * scale,
            "aupro": self.aupro * scale,
            "n_images": self.n_images,
            "n_positive_images": self.n_positive_images,
            "n_regions": self.n_regions,
        }


def _stack_2d(values: Sequence[np.ndarray], name: str) -> np.ndarray:
    if not values:
        raise ValueError(f"{name} must not be empty")
    arrays = [np.asarray(value) for value in values]
    shape = arrays[0].shape
    if len(shape) != 2:
        raise ValueError(f"{name} entries must be 2-D; got {shape}")
    if any(array.shape != shape for array in arrays):
        raise ValueError(f"All {name} entries must have one common shape")
    return np.stack(arrays, axis=0)


def image_auroc(labels: Sequence[int | bool], scores: Sequence[float]) -> float:
    y_true = np.asarray(labels, dtype=np.uint8)
    y_score = np.asarray(scores, dtype=np.float64)
    if y_true.size != y_score.size:
        raise ValueError("Image labels and scores must have equal length")
    if np.unique(y_true).size != 2:
        raise ValueError("I-AUROC requires at least one normal and one anomalous image")
    return float(roc_auc_score(y_true, y_score))


def pixel_auroc(masks: Sequence[np.ndarray], anomaly_maps: Sequence[np.ndarray]) -> float:
    y_true = _stack_2d(masks, "masks").astype(bool, copy=False).ravel()
    y_score = _stack_2d(anomaly_maps, "anomaly_maps").astype(np.float64, copy=False).ravel()
    if np.unique(y_true).size != 2:
        raise ValueError("P-AUROC requires both anomalous and normal pixels")
    return float(roc_auc_score(y_true, y_score))


def count_regions(masks: Sequence[np.ndarray]) -> int:
    structure = np.ones((3, 3), dtype=np.uint8)
    total = 0
    for mask in masks:
        _, count = ndimage.label(np.asarray(mask, dtype=bool), structure=structure)
        total += int(count)
    return total


def aupro(
    masks: Sequence[np.ndarray],
    anomaly_maps: Sequence[np.ndarray],
    *,
    max_fpr: float = 0.30,
    num_thresholds: int = 200,
) -> float:
    """Compute normalized area under the PRO-FPR curve.

    PRO is averaged over connected ground-truth anomaly regions (8-connectivity),
    so every region contributes equally irrespective of area. FPR is computed on
    normal pixels only. The curve is integrated up to ``max_fpr`` and divided by
    that cap, yielding a score in [0, 1]. This is the conventional MVTec-style
    AUPRO@0.30 protocol used by the benchmark.
    """

    if not 0 < max_fpr <= 1:
        raise ValueError("max_fpr must be in (0, 1]")
    if num_thresholds < 2:
        raise ValueError("num_thresholds must be at least 2")

    mask_stack = _stack_2d(masks, "masks").astype(bool, copy=False)
    map_stack = _stack_2d(anomaly_maps, "anomaly_maps").astype(np.float64, copy=False)
    if mask_stack.shape != map_stack.shape:
        raise ValueError("Masks and anomaly maps must have equal shape")

    structure = np.ones((3, 3), dtype=np.uint8)
    labeled_masks: list[tuple[np.ndarray, int]] = []
    region_count = 0
    for mask in mask_stack:
        labeled, n_regions = ndimage.label(mask, structure=structure)
        labeled_masks.append((labeled, int(n_regions)))
        region_count += int(n_regions)
    if region_count == 0:
        raise ValueError("AUPRO requires at least one anomalous ground-truth region")

    normal_pixels = ~mask_stack
    normal_pixel_count = int(normal_pixels.sum())
    if normal_pixel_count == 0:
        raise ValueError("AUPRO requires at least one normal pixel")

    finite_maps = np.nan_to_num(
        map_stack,
        nan=-np.inf,
        posinf=np.finfo(np.float64).max,
        neginf=np.finfo(np.float64).min,
    )
    score_min = float(np.min(finite_maps))
    score_max = float(np.max(finite_maps))
    if score_max == score_min:
        thresholds = np.array([np.inf, score_max, -np.inf], dtype=np.float64)
    else:
        thresholds = np.concatenate(
            (
                np.array([np.inf]),
                np.linspace(score_max, score_min, num_thresholds, endpoint=True),
                np.array([-np.inf]),
            )
        )

    fprs: list[float] = []
    pros: list[float] = []
    for threshold in thresholds:
        predicted = finite_maps > threshold
        false_positive = int(np.logical_and(predicted, normal_pixels).sum())
        fpr = false_positive / normal_pixel_count

        overlaps: list[float] = []
        for predicted_image, (labeled, n_regions) in zip(predicted, labeled_masks):
            for region_id in range(1, n_regions + 1):
                region = labeled == region_id
                overlaps.append(float(predicted_image[region].mean()))
        pros.append(float(np.mean(overlaps)))
        fprs.append(float(fpr))

    fpr_array = np.asarray(fprs, dtype=np.float64)
    pro_array = np.asarray(pros, dtype=np.float64)
    order = np.argsort(fpr_array, kind="mergesort")
    fpr_array = fpr_array[order]
    pro_array = pro_array[order]

    # Multiple thresholds may produce the same FPR. Retain the greatest PRO at
    # each FPR so the sampled curve is monotone in achievable overlap.
    unique_fpr: list[float] = []
    unique_pro: list[float] = []
    for fpr_value in np.unique(fpr_array):
        unique_fpr.append(float(fpr_value))
        unique_pro.append(float(np.max(pro_array[fpr_array == fpr_value])))
    x = np.asarray(unique_fpr, dtype=np.float64)
    y = np.maximum.accumulate(np.asarray(unique_pro, dtype=np.float64))

    if x[0] > 0:
        x = np.insert(x, 0, 0.0)
        y = np.insert(y, 0, 0.0)
    if x[-1] < max_fpr:
        cap_y = y[-1]
        x = np.append(x, max_fpr)
        y = np.append(y, cap_y)
    else:
        cap_y = float(np.interp(max_fpr, x, y))
        below = x < max_fpr
        x = np.concatenate((x[below], np.array([max_fpr])))
        y = np.concatenate((y[below], np.array([cap_y])))

    return float(np.clip(auc(x, y) / max_fpr, 0.0, 1.0))


def compute_metrics(
    labels: Sequence[int | bool],
    scores: Sequence[float],
    masks: Sequence[np.ndarray],
    anomaly_maps: Sequence[np.ndarray],
    *,
    max_fpr: float = 0.30,
    num_thresholds: int = 200,
) -> MetricBundle:
    return MetricBundle(
        i_auroc=image_auroc(labels, scores),
        p_auroc=pixel_auroc(masks, anomaly_maps),
        aupro=aupro(masks, anomaly_maps, max_fpr=max_fpr, num_thresholds=num_thresholds),
        n_images=len(labels),
        n_positive_images=int(np.asarray(labels, dtype=np.uint8).sum()),
        n_regions=count_regions(masks),
    )


def macro_mean(rows: Iterable[MetricBundle]) -> MetricBundle:
    values = list(rows)
    if not values:
        raise ValueError("Cannot average an empty metric collection")
    return MetricBundle(
        i_auroc=float(np.mean([row.i_auroc for row in values])),
        p_auroc=float(np.mean([row.p_auroc for row in values])),
        aupro=float(np.mean([row.aupro for row in values])),
        n_images=sum(row.n_images for row in values),
        n_positive_images=sum(row.n_positive_images for row in values),
        n_regions=sum(row.n_regions for row in values),
    )
