"""Scenario 3: industrial robustness on the Assembled Panel Dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from tqdm import tqdm

from ..backend import (
    FULL_SHOT,
    IMAGE_SCORE_RULES,
    SHOT_ORDER,
    AnomalyMapBackend,
    Shot,
    image_level_scores,
    parse_shot,
)
from ..cache import PredictionCache
from ..datasets.apd_coco import APD_A_IDS, APD_A_ID_TO_ERROR, APDIndex
from ..metrics import MetricBundle, compute_metrics, macro_mean
from ..preprocess import PreparedImageCache, SpatialStrategy, transform_binary_mask
from ..prompts import SPATIAL_STRATEGIES, Scenario3Prompts
from .common import (
    finalize_output,
    metric_row,
    reference_record,
    resize_anomaly_map,
    select_shots,
)


StrategyShotVariants = Mapping[str, Mapping[Shot, Sequence[str]]]


def _select_apd_reference(
    index: APDIndex,
    explicit_reference: str | Path | None,
) -> tuple[Path, str]:
    if explicit_reference is not None:
        path = Path(explicit_reference).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"APD one-shot reference not found: {path}")
        return path, "explicit"
    normals = index.normal_samples()
    if not normals:
        raise ValueError("APD one-shot evaluation requires at least one normal image")
    # Default reference: first normal image by relative sample id, recorded in
    # run_metadata.json. Pass --one-shot-reference to choose another one.
    selected = sorted(normals, key=lambda sample: sample.sample_id)[0]
    return selected.image_path, "first_lexicographic_normal"


def _variant_matrix(
    prompts: Scenario3Prompts,
    *,
    by_shot: Mapping[int, Sequence[str]] | None,
    by_strategy_and_shot: StrategyShotVariants | None,
) -> dict[str, dict[int, tuple[str, ...]]]:
    # Default: the Standard AD sweep reported in Appendix Tables 10 and 11, stored with
    # the model's prompts.
    matrix: dict[str, dict[int, tuple[str, ...]]] = {
        strategy: dict(prompts.standard_variants[strategy]) for strategy in SPATIAL_STRATEGIES
    }
    if by_shot:
        for strategy in matrix:
            for shot, variants in by_shot.items():
                matrix[strategy][parse_shot(shot)] = tuple(variants)
    if by_strategy_and_shot:
        for strategy, shot_mapping in by_strategy_and_shot.items():
            if strategy not in matrix:
                raise ValueError(f"Unknown preprocessing strategy in variant matrix: {strategy}")
            for shot, variants in shot_mapping.items():
                matrix[strategy][parse_shot(shot)] = tuple(variants)
    for strategy, shot_mapping in matrix.items():
        for shot, variants in shot_mapping.items():
            if shot not in SHOT_ORDER:
                raise ValueError(f"Unsupported shot regime {shot} for {strategy}")
            unknown = sorted(set(variants) - set(prompts.standard))
            if unknown:
                raise ValueError(
                    f"Unknown Scenario 3 Standard prompt variants for {strategy}/{shot}: {unknown}"
                )
    return matrix


def run_scenario3(
    *,
    apd_root: str | Path,
    coco_annotations: str | Path,
    backend: AnomalyMapBackend,
    output_dir: str | Path,
    cache_dir: str | Path,
    prepared_cache_dir: str | Path,
    prompt_config: str | Path | None = None,
    modes: Sequence[str] = ("standard", "per_error"),
    preprocessing: Sequence[SpatialStrategy] = ("direct_resize", "pad_to_square"),
    standard_shots: Sequence[Shot] | None = None,
    per_error_shots: Sequence[Shot] | None = None,
    standard_variants_by_shot: Mapping[Shot, Sequence[str]] | None = None,
    standard_variants_by_strategy_and_shot: StrategyShotVariants | None = None,
    one_shot_reference: str | Path | None = None,
    full_shot_references: Sequence[str | Path] | None = None,
    dataset_dir_name: str = "apd",
    target_size: int = 224,
    aupro_max_fpr: float = 0.30,
    aupro_thresholds: int = 200,
    cache_enabled: bool = True,
    validate_protocol_counts: bool = True,
) -> dict[str, Path]:
    """Run Standard and Per-Error AD.

    ``standard_shots=None`` runs every swept regime; ``per_error_shots=None`` runs the
    model's regime (0-shot, or full-shot for LogSAD). The APD has no training split:
    full-shot references default to the 1-shot reference alone, as in the paper runs, so
    LogSAD's full-shot differs from its 1-shot only by the coreset memory bank.
    """

    normalized_modes = tuple(mode.lower() for mode in modes)
    if not normalized_modes or not set(normalized_modes) <= {"standard", "per_error"}:
        raise ValueError("modes must contain standard and/or per_error")
    selected_preprocessing = tuple(preprocessing)
    if not selected_preprocessing:
        raise ValueError("At least one preprocessing strategy is required")
    for strategy in selected_preprocessing:
        if strategy not in SPATIAL_STRATEGIES:
            raise ValueError(f"Unknown preprocessing strategy: {strategy}")

    prompts = Scenario3Prompts.load(prompt_config, model=backend.model_name)
    variants = _variant_matrix(
        prompts,
        by_shot=standard_variants_by_shot,
        by_strategy_and_shot=standard_variants_by_strategy_and_shot,
    )
    swept_shots = sorted(
        {shot for strategy in selected_preprocessing for shot, v in variants[strategy].items() if v},
        key=SHOT_ORDER.index,
    )
    standard_shots = (
        select_shots(backend, standard_shots, swept_shots)
        if "standard" in normalized_modes
        else ()
    )
    per_error_shots = (
        select_shots(backend, per_error_shots, (backend.spec.default_shot,))
        if "per_error" in normalized_modes
        else ()
    )
    for strategy in selected_preprocessing:
        for shot in standard_shots:
            if not variants[strategy].get(shot):
                raise ValueError(
                    f"No Scenario 3 Standard AD prompt variants for {strategy}/{shot}-shot"
                )
    index = APDIndex(
        apd_root,
        coco_annotations,
        dataset_dir_name=dataset_dir_name,
        strict=True,
    )
    protocol_counts = index.validate_protocol_counts() if validate_protocol_counts else None
    prediction_cache = PredictionCache(cache_dir, backend, enabled=cache_enabled)
    prepared_cache = PreparedImageCache(Path(prepared_cache_dir), target_size)
    output = Path(output_dir)

    reference_source: str | None = None
    raw_reference: Path | None = None
    raw_full_references: list[Path] = []
    used_shots = set(standard_shots) | set(per_error_shots)
    if used_shots & {1, FULL_SHOT}:
        raw_reference, reference_source = _select_apd_reference(index, one_shot_reference)
    if FULL_SHOT in used_shots:
        raw_full_references = [
            Path(path).expanduser().resolve() for path in full_shot_references or [raw_reference]
        ]
        missing = [str(path) for path in raw_full_references if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"APD full-shot reference(s) not found: {missing}")

    per_config_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for strategy in selected_preprocessing:
        references_by_shot: dict[Shot, list[Path]] = {0: []}
        if raw_reference is not None:
            references_by_shot[1] = [prepared_cache.prepare(raw_reference, strategy)]
        if raw_full_references:
            references_by_shot[FULL_SHOT] = [
                prepared_cache.prepare(path, strategy) for path in raw_full_references
            ]

        if "standard" in normalized_modes:
            samples = sorted(index.samples(), key=lambda sample: sample.sample_id)
            for shot in standard_shots:
                references = references_by_shot[shot]
                for variant in variants[strategy][shot]:
                    prompt = prompts.standard[variant]
                    labels: list[int] = []
                    detection_scores: list[float | None] = []
                    masks: list[np.ndarray] = []
                    maps: list[np.ndarray] = []
                    set_rows: list[dict[str, object]] = []
                    iterator = tqdm(
                        samples,
                        desc=f"S3 standard {strategy} {shot}-shot {variant}",
                        leave=False,
                    )
                    for sample in iterator:
                        prepared_image = prepared_cache.prepare(sample.image_path, strategy)
                        result = prediction_cache.predict(
                            prepared_image,
                            prompt,
                            references,
                            shots=shot,
                            context={
                                "scenario": 3,
                                "mode": "standard",
                                "strategy": strategy,
                                "shots": shot,
                                "prompt_variant": variant,
                                "source_image": str(sample.image_path),
                                "target_size": target_size,
                            },
                        )
                        anomaly_map = resize_anomaly_map(result.anomaly_map, target_size)
                        source_mask = index.load_mask(sample)
                        mask = transform_binary_mask(
                            source_mask,
                            strategy,
                            target_size,
                            interpolation="nearest",
                        )
                        labels.append(int(sample.is_anomalous))
                        detection_scores.append(result.detection_score)
                        masks.append(mask)
                        maps.append(anomaly_map)
                        set_rows.append(
                            {
                                "scenario": 3,
                                "mode": "standard",
                                "preprocessing": strategy,
                                "shots": shot,
                                "prompt_variant": variant,
                                "sample_id": sample.sample_id,
                                "is_anomalous": sample.is_anomalous,
                                "labels": list(sample.labels),
                                "source_image_path": str(sample.image_path),
                                "prepared_image_path": str(prepared_image),
                                "prompt": prompt,
                                "reference_images": reference_record(references),
                                "detection_score": result.detection_score,
                                "cache_key": result.cache_key,
                                "cache_path": str(result.cache_path),
                                "cache_hit": result.cache_hit,
                                "response_text": result.response_text,
                            }
                        )
                    scores = image_level_scores(backend.model_name, maps, detection_scores)
                    for row, score in zip(set_rows, scores):
                        row["image_score"] = score
                    prediction_rows.extend(set_rows)
                    metric = compute_metrics(
                        labels,
                        scores,
                        masks,
                        maps,
                        max_fpr=aupro_max_fpr,
                        num_thresholds=aupro_thresholds,
                    )
                    row = metric_row(
                        metric,
                        scenario=3,
                        mode="standard",
                        unit_type="dataset",
                        preprocessing=strategy,
                        shots=shot,
                        prompt_variant=variant,
                        category="",
                    )
                    per_config_rows.append(row)
                    summary_rows.append(
                        {
                            **row,
                            "aggregation": "all_images",
                            "n_categories": "",
                        }
                    )

        if "per_error" in normalized_modes:
            for shot in per_error_shots:
                references = references_by_shot[shot]
                category_metrics: list[MetricBundle] = []
                for a_id in APD_A_IDS:
                    error_name = APD_A_ID_TO_ERROR[a_id]
                    prompt = prompts.per_error[a_id]
                    positives = index.per_error_samples(a_id)
                    if not positives:
                        raise ValueError(f"APD category {a_id} has no positive images")
                    samples = sorted(
                        positives + index.normal_samples(), key=lambda sample: sample.sample_id
                    )
                    positive_ids = {sample.sample_id for sample in positives}
                    labels: list[int] = []
                    detection_scores: list[float | None] = []
                    masks: list[np.ndarray] = []
                    maps: list[np.ndarray] = []
                    set_rows: list[dict[str, object]] = []
                    iterator = tqdm(
                        samples,
                        desc=f"S3 per-error {strategy} {shot}-shot {a_id}",
                        leave=False,
                    )
                    for sample in iterator:
                        prepared_image = prepared_cache.prepare(sample.image_path, strategy)
                        result = prediction_cache.predict(
                            prepared_image,
                            prompt,
                            references,
                            shots=shot,
                            context={
                                "scenario": 3,
                                "mode": "per_error",
                                "category": a_id,
                                "strategy": strategy,
                                "shots": shot,
                                "source_image": str(sample.image_path),
                                "target_size": target_size,
                            },
                        )
                        anomaly_map = resize_anomaly_map(result.anomaly_map, target_size)
                        is_positive = sample.sample_id in positive_ids
                        source_mask = (
                            index.load_mask(sample, target_error=error_name)
                            if is_positive
                            else index.empty_mask(sample)
                        )
                        mask = transform_binary_mask(
                            source_mask,
                            strategy,
                            target_size,
                            interpolation="nearest",
                        )
                        labels.append(int(is_positive))
                        detection_scores.append(result.detection_score)
                        masks.append(mask)
                        maps.append(anomaly_map)
                        set_rows.append(
                            {
                                "scenario": 3,
                                "mode": "per_error",
                                "preprocessing": strategy,
                                "shots": shot,
                                "category": a_id,
                                "coco_error": error_name,
                                "sample_id": sample.sample_id,
                                "is_positive_for_category": is_positive,
                                "source_labels": list(sample.labels),
                                "source_image_path": str(sample.image_path),
                                "prepared_image_path": str(prepared_image),
                                "prompt": prompt,
                                "reference_images": reference_record(references),
                                "detection_score": result.detection_score,
                                "cache_key": result.cache_key,
                                "cache_path": str(result.cache_path),
                                "cache_hit": result.cache_hit,
                                "response_text": result.response_text,
                            }
                        )
                    scores = image_level_scores(backend.model_name, maps, detection_scores)
                    for row, score in zip(set_rows, scores):
                        row["image_score"] = score
                    prediction_rows.extend(set_rows)
                    metric = compute_metrics(
                        labels,
                        scores,
                        masks,
                        maps,
                        max_fpr=aupro_max_fpr,
                        num_thresholds=aupro_thresholds,
                    )
                    category_metrics.append(metric)
                    per_config_rows.append(
                        metric_row(
                            metric,
                            scenario=3,
                            mode="per_error",
                            unit_type="category",
                            preprocessing=strategy,
                            shots=shot,
                            prompt_variant="category_specific",
                            category=a_id,
                            coco_error=error_name,
                            n_category_positives=len(positives),
                        )
                    )
                aggregate = macro_mean(category_metrics)
                summary_rows.append(
                    metric_row(
                        aggregate,
                        scenario=3,
                        mode="per_error",
                        unit_type="macro_category",
                        preprocessing=strategy,
                        shots=shot,
                        prompt_variant="category_specific",
                        category="",
                        aggregation="macro_category",
                        n_categories=len(category_metrics),
                    )
                )

    metadata = {
        "scenario": 3,
        "dataset": "Assembled Panel Dataset",
        "dataset_dir_name": dataset_dir_name,
        "modes": list(normalized_modes),
        "preprocessing": list(selected_preprocessing),
        "target_size": target_size,
        "standard_shots": list(standard_shots),
        "per_error_shots": list(per_error_shots),
        "standard_variant_matrix": {
            strategy: {
                str(shot): list(variants[strategy][shot]) for shot in standard_shots
            }
            for strategy in selected_preprocessing
        },
        "one_shot_reference": str(raw_reference) if raw_reference else None,
        "one_shot_reference_source": reference_source,
        "full_shot_references": [str(path) for path in raw_full_references],
        "image_transform": (
            "Direct Resize uses bicubic square resize; Pad to Square preserves aspect ratio, "
            "centers the resized image, and fills padding with zero-valued pixels"
        ),
        "mask_transform": (
            "same Direct Resize or Pad to Square geometry as image; nearest-neighbor binary mask"
        ),
        "per_error_negative_set": "all defect-free images only",
        "per_error_localization": (
            "target-category mask only; other defects in a multi-defect positive image are background"
        ),
        "per_error_aggregation": "unweighted macro-average over 14 categories",
        "model": backend.model_name,
        "image_score": IMAGE_SCORE_RULES[backend.spec.image_score_rule],
        "prompt_handling": backend.spec.prompt_handling,
        "text_conditioning": backend.text_conditioning,
        "protocol_counts": protocol_counts,
        "backend_fingerprint": backend.fingerprint,
        "aupro": {"max_fpr": aupro_max_fpr, "thresholds": aupro_thresholds},
    }
    return finalize_output(
        output,
        per_unit_rows=per_config_rows,
        summary_rows=summary_rows,
        prediction_rows=prediction_rows,
        metadata=metadata,
        per_unit_name="per_configuration.csv",
    )
