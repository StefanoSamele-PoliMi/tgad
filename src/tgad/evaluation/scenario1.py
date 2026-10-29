"""Scenario 1: controlled prompt sensitivity on MVTec AD."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

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
from ..datasets.mvtec import MVTEC_CLASSES, MVTecDatasetIndex
from ..metrics import MetricBundle, compute_metrics, macro_mean
from ..preprocess import load_mvtec_mask
from ..prompts import Scenario1Prompts
from .common import (
    finalize_output,
    metric_row,
    reference_record,
    resize_anomaly_map,
    select_shots,
)


def run_scenario1(
    *,
    mvtec_root: str | Path,
    backend: AnomalyMapBackend,
    output_dir: str | Path,
    cache_dir: str | Path,
    classes: Sequence[str] | None = None,
    shots: Sequence[Shot] | None = None,
    variants_by_shot: Mapping[Shot, Sequence[str]] | None = None,
    prompt_config: str | Path | None = None,
    reference_round: int | None = None,
    target_size: int = 224,
    aupro_max_fpr: float = 0.30,
    aupro_thresholds: int = 200,
    cache_enabled: bool = True,
) -> dict[str, Path]:
    """Run the prompt-sensitivity sweep; ``shots=None`` runs every regime with variants.

    1-shot uses ``train/good/{4 * reference_round:03d}.png`` (default: the model's own
    protocol); full-shot uses every ``train/good`` image of the class.
    """

    prompts = Scenario1Prompts.load(prompt_config, model=backend.model_name)
    selected_classes = tuple(classes or MVTEC_CLASSES)
    if reference_round is None:
        reference_round = backend.spec.mvtec_reference_round

    variants_for_shot: dict[Shot, tuple[str, ...]] = dict(prompts.variants_by_shot)
    if variants_by_shot:
        variants_for_shot.update(
            {parse_shot(key): tuple(value) for key, value in variants_by_shot.items()}
        )
    selected_shots = select_shots(
        backend, shots, [shot for shot in SHOT_ORDER if variants_for_shot.get(shot)]
    )
    for shot in selected_shots:
        if not variants_for_shot.get(shot):
            raise ValueError(f"No Scenario 1 prompt variants are defined for {shot}-shot")
        unknown = sorted(set(variants_for_shot[shot]) - set(prompts.variants))
        if unknown:
            raise ValueError(f"Unknown Scenario 1 prompt variants: {unknown}")

    dataset = MVTecDatasetIndex(mvtec_root, selected_classes)
    prediction_cache = PredictionCache(cache_dir, backend, enabled=cache_enabled)
    output = Path(output_dir)

    def references_for(class_name: str, shot: Shot) -> list[Path]:
        if shot == 0:
            return []
        if shot == FULL_SHOT:
            return dataset.train_good_images(class_name)
        return dataset.official_reference_images(class_name, round_index=reference_round, k_shot=1)

    # Classes form the outer loop so that a model's per-class reference state (LogSAD's
    # memory bank) is built once per regime; rows are written in (regime, variant, class)
    # order as before.
    class_rows_by_unit: dict[tuple[Shot, str], list[dict[str, object]]] = defaultdict(list)
    prediction_rows_by_unit: dict[tuple[Shot, str], list[dict[str, object]]] = defaultdict(list)
    bundles: dict[tuple[Shot, str], list[MetricBundle]] = defaultdict(list)

    for class_name in selected_classes:
        samples = dataset.samples(class_name)
        for shot in selected_shots:
            references = references_for(class_name, shot)
            for variant in variants_for_shot[shot]:
                prompt = prompts.variants[variant][class_name]
                labels: list[int] = []
                detection_scores: list[float | None] = []
                masks = []
                maps = []
                set_rows: list[dict[str, object]] = []
                iterator = tqdm(
                    samples,
                    desc=f"S1 {shot}-shot {variant} {class_name}",
                    leave=False,
                )
                for sample in iterator:
                    result = prediction_cache.predict(
                        sample.image_path,
                        prompt,
                        references,
                        shots=shot,
                        context={
                            "scenario": 1,
                            "shots": shot,
                            "prompt_variant": variant,
                            "class_name": class_name,
                            "target_size": target_size,
                        },
                    )
                    anomaly_map = resize_anomaly_map(result.anomaly_map, target_size)
                    mask = load_mvtec_mask(sample.mask_path, sample.image_path, target_size)
                    labels.append(int(sample.is_anomalous))
                    detection_scores.append(result.detection_score)
                    masks.append(mask)
                    maps.append(anomaly_map)
                    set_rows.append(
                        {
                            "scenario": 1,
                            "shots": shot,
                            "prompt_variant": variant,
                            "class_name": class_name,
                            "sample_id": sample.sample_id,
                            "is_anomalous": sample.is_anomalous,
                            "image_path": str(sample.image_path),
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
                prediction_rows_by_unit[(shot, variant)].extend(set_rows)
                metric = compute_metrics(
                    labels,
                    scores,
                    masks,
                    maps,
                    max_fpr=aupro_max_fpr,
                    num_thresholds=aupro_thresholds,
                )
                bundles[(shot, variant)].append(metric)
                class_rows_by_unit[(shot, variant)].append(
                    metric_row(
                        metric,
                        scenario=1,
                        shots=shot,
                        prompt_variant=variant,
                        class_name=class_name,
                    )
                )

    # Preserve the paper/config order rather than lexicographically sorting variants.
    units = [(shot, variant) for shot in selected_shots for variant in variants_for_shot[shot]]
    per_class_rows = [row for unit in units for row in class_rows_by_unit[unit]]
    prediction_rows = [row for unit in units for row in prediction_rows_by_unit[unit]]
    summary_rows = [
        metric_row(
            macro_mean(bundles[(shot, variant)]),
            scenario=1,
            aggregation="macro_class",
            shots=shot,
            prompt_variant=variant,
            n_classes=len(bundles[(shot, variant)]),
        )
        for shot, variant in units
    ]

    metadata = {
        "scenario": 1,
        "dataset": "MVTec AD original test split",
        "classes": list(selected_classes),
        "shots": list(selected_shots),
        "variants_by_shot": {
            str(shot): list(variants_for_shot[shot]) for shot in selected_shots
        },
        "target_size": target_size,
        "mask_transform": (
            f"direct resize to {target_size}x{target_size}, bilinear, threshold 0.1 "
            "(official test script)"
        ),
        "model": backend.model_name,
        "image_score": IMAGE_SCORE_RULES[backend.spec.image_score_rule],
        "aggregation": "per-class metrics followed by an unweighted class mean",
        "one_shot_reference_policy": {
            "name": "official_round",
            "round": reference_round,
            "k_shot": 1,
            "default_filename": f"{reference_round * 4:03d}.png",
        },
        "full_shot_references": "every train/good image of the class",
        "prompt_handling": backend.spec.prompt_handling,
        "text_conditioning": backend.text_conditioning,
        "backend_fingerprint": backend.fingerprint,
        "aupro": {"max_fpr": aupro_max_fpr, "thresholds": aupro_thresholds},
    }
    return finalize_output(
        output,
        per_unit_rows=per_class_rows,
        summary_rows=summary_rows,
        prediction_rows=prediction_rows,
        metadata=metadata,
        per_unit_name="per_class.csv",
    )
