"""Scenario 2: text-conditioned per-part inspection on MVTec AD."""

from __future__ import annotations

import csv
import hashlib
import random
from collections import defaultdict
from importlib.resources import files
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from tqdm import tqdm

from ..backend import FULL_SHOT, IMAGE_SCORE_RULES, AnomalyMapBackend, Shot, image_level_scores
from ..cache import PredictionCache
from ..datasets.mvtec import MVTecDatasetIndex
from ..datasets.mvtec_parts import PART_CLASSES, MVTecPartTags, SubclassSplit
from ..exceptions import DataValidationError
from ..metrics import MetricBundle, aupro, count_regions, image_auroc, macro_mean, pixel_auroc
from ..preprocess import load_mvtec_mask
from ..prompts import Scenario2Prompts
from .common import finalize_output, metric_row, resize_anomaly_map, select_shots


PromptSelector = Callable[[str, str, str, Sequence[str]], str]
LOCALIZATION_SCOPES = ("evaluation_set", "in_scope_only")
DEFAULT_LOCALIZATION_SCOPE = "evaluation_set"


def _load_default_tags() -> MVTecPartTags:
    resource = files("tgad.resources").joinpath("mvtec_test_part_tags_raw.txt")
    return MVTecPartTags.from_text(resource.read_text(encoding="utf-8"), source=str(resource))


def _make_prompt_selector(seed: int, mode: str, setting: str | None = None) -> PromptSelector:
    if mode == "stable_hash":

        def choose(
            class_name: str,
            subclass: str,
            sample_id: str,
            pool: Sequence[str],
        ) -> str:
            payload = (
                f"{seed}|{setting or 'shared'}|{class_name}|{subclass}|{sample_id}"
            ).encode("utf-8")
            index = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % len(pool)
            return str(pool[index])

        return choose
    if mode in {"global_sequence", "per_setting_sequence"}:
        rng = random.Random(seed)

        def choose(
            class_name: str,
            subclass: str,
            sample_id: str,
            pool: Sequence[str],
        ) -> str:
            del class_name, subclass, sample_id
            return str(rng.choice(list(pool)))

        return choose
    raise ValueError(
        "prompt_sampling must be stable_hash, global_sequence, or per_setting_sequence"
    )


def _validate_table6(splits: Sequence[SubclassSplit]) -> None:
    resource = files("tgad.resources").joinpath(
        "scenario2_expected_cardinalities.csv"
    )
    with resource.open("r", encoding="utf-8", newline="") as handle:
        expected_rows = list(csv.DictReader(handle))
    expected = {(row["class_name"], row["subclass"]): row for row in expected_rows}
    errors: list[str] = []
    for split in splits:
        row = expected.get((split.class_name, split.subclass))
        if row is None:
            errors.append(f"unexpected subclass {split.class_name}/{split.subclass}")
            continue
        actual = {
            "part_count": split.part_count,
            "n1_good": len(split.normal_good),
            "a_exact": len(split.positives),
            "n2_disjoint": len(split.irrelevant_defects),
            "excluded_overlap": len(split.excluded),
        }
        for field, value in actual.items():
            if int(row[field]) != value:
                errors.append(
                    f"{split.class_name}/{split.subclass} {field}: "
                    f"expected {row[field]}, got {value}"
                )
    actual_keys = {(split.class_name, split.subclass) for split in splits}
    if len(actual_keys) != len(splits):
        errors.append("subclass inventory contains duplicate entries")
    missing_keys = sorted(set(expected) - actual_keys)
    extra_keys = sorted(actual_keys - set(expected))
    if missing_keys:
        errors.append(f"missing Appendix Table 6 subclasses: {missing_keys[:10]}")
    if extra_keys:
        errors.append(f"unexpected subclasses not in Appendix Table 6: {extra_keys[:10]}")
    if errors:
        raise DataValidationError(
            "Scenario 2 cardinalities do not match Appendix Table 6:\n  - "
            + "\n  - ".join(errors[:30])
        )


def _bundle_from_scopes(
    image_labels: Sequence[int],
    image_scores: Sequence[float],
    localization_masks: Sequence[np.ndarray],
    localization_maps: Sequence[np.ndarray],
    *,
    max_fpr: float,
    thresholds: int,
) -> MetricBundle:
    return MetricBundle(
        i_auroc=image_auroc(image_labels, image_scores),
        p_auroc=pixel_auroc(localization_masks, localization_maps),
        aupro=aupro(
            localization_masks,
            localization_maps,
            max_fpr=max_fpr,
            num_thresholds=thresholds,
        ),
        n_images=len(image_labels),
        n_positive_images=int(sum(image_labels)),
        n_regions=count_regions(localization_masks),
    )


def run_scenario2(
    *,
    mvtec_root: str | Path,
    backend: AnomalyMapBackend,
    output_dir: str | Path,
    cache_dir: str | Path,
    tags_file: str | Path | None = None,
    prompt_config: str | Path | None = None,
    classes: Sequence[str] | None = None,
    settings: Sequence[str] = ("ev1", "ev2"),
    seed: int = 0,
    prompt_sampling: str = "stable_hash",
    localization_scope: str = DEFAULT_LOCALIZATION_SCOPE,
    target_size: int = 224,
    aupro_max_fpr: float = 0.30,
    aupro_thresholds: int = 200,
    cache_enabled: bool = True,
    validate_cardinalities: bool = True,
    shots: Shot | None = None,
    reference_round: int | None = None,
) -> dict[str, Path]:
    """Run the component-conditioned benchmark.

    Localization (P-AUROC, AUPRO) is computed separately for EV1 and EV2 on the same
    image set used for I-AUROC (``evaluation_set``): ``A(Ts)`` keeps its ground-truth
    masks, while every image admitted as normal (``N1`` under EV1, ``N1 + N2`` under
    EV2) contributes an all-zero target mask. Under EV2 a model is therefore penalized
    for highlighting defects outside the instructed component. ``in_scope_only`` is an
    optional diagnostic that scores localization on ``A(Ts)`` alone, which is identical
    for EV1 and EV2.

    ``shots`` is the single supervision regime of the run (default: the model's, 0-shot
    for AnomalyGPT and AA-CLIP, full-shot for LogSAD). 1-shot uses
    ``train/good/{4 * reference_round:03d}.png``; full-shot uses every ``train/good`` image.
    """

    selected_classes = tuple(classes or PART_CLASSES)
    unknown_classes = sorted(set(selected_classes) - set(PART_CLASSES))
    if unknown_classes:
        raise ValueError(
            f"Scenario 2 supports only the nine annotated classes: {unknown_classes}"
        )
    normalized_settings = tuple(setting.lower() for setting in settings)
    if not normalized_settings or not set(normalized_settings) <= {"ev1", "ev2"}:
        raise ValueError("settings must contain ev1 and/or ev2")
    if localization_scope not in LOCALIZATION_SCOPES:
        raise ValueError("localization_scope must be evaluation_set or in_scope_only")

    prompts = Scenario2Prompts.load(prompt_config, model=backend.model_name)
    tags = MVTecPartTags.from_file(tags_file) if tags_file else _load_default_tags()
    dataset = MVTecDatasetIndex(mvtec_root, selected_classes)
    tags.validate_against_dataset(dataset)
    splits = [
        tags.build_split(dataset, class_name, tag_set)
        for class_name in selected_classes
        for tag_set in tags.subclasses(class_name)
    ]
    missing_prompt_pools = sorted(
        split.subclass for split in splits if split.subclass not in prompts.pools
    )
    if missing_prompt_pools:
        raise DataValidationError(
            "Scenario 2 prompt config lacks subclass pool(s): "
            + ", ".join(missing_prompt_pools)
        )
    all_nine_classes = (
        len(selected_classes) == len(PART_CLASSES)
        and set(selected_classes) == set(PART_CLASSES)
    )
    if validate_cardinalities and all_nine_classes:
        _validate_table6(splits)

    (shot,) = select_shots(backend, None if shots is None else [shots], [backend.spec.default_shot])
    if reference_round is None:
        reference_round = backend.spec.mvtec_reference_round
    references_by_class: dict[str, list[Path]] = {
        class_name: (
            []
            if shot == 0
            else dataset.train_good_images(class_name)
            if shot == FULL_SHOT
            else dataset.official_reference_images(
                class_name, round_index=reference_round, k_shot=1
            )
        )
        for class_name in selected_classes
    }
    # Only non-zero regimes enter the cache context, so earlier 0-shot caches stay valid.
    shot_context = {} if shot == 0 else {"shots": shot}

    prediction_cache = PredictionCache(cache_dir, backend, enabled=cache_enabled)
    output = Path(output_dir)
    per_subclass_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    bundles: dict[tuple[str, int, str], list[tuple[str, MetricBundle]]] = defaultdict(list)

    shared_selector = (
        _make_prompt_selector(seed, prompt_sampling)
        if prompt_sampling != "per_setting_sequence"
        else None
    )

    for setting in normalized_settings:
        selector = shared_selector or _make_prompt_selector(seed, prompt_sampling, setting)
        assert selector is not None
        for split in splits:
            pool = prompts.pools[split.subclass]
            evaluation_samples = sorted(
                split.image_level_samples(setting), key=lambda sample: sample.sample_id
            )
            positive_ids = {sample.sample_id for sample in split.positives}
            good_ids = {sample.sample_id for sample in split.normal_good}
            sample_maps: dict[str, np.ndarray] = {}
            detection_scores: list[float | None] = []
            subclass_rows: list[dict[str, object]] = []
            iterator = tqdm(
                evaluation_samples,
                desc=f"S2 {setting} {split.class_name}/{split.subclass}",
                leave=False,
            )
            for sample in iterator:
                prompt = selector(
                    split.class_name,
                    split.subclass,
                    sample.sample_id,
                    pool,
                )
                result = prediction_cache.predict(
                    sample.image_path,
                    prompt,
                    references_by_class[split.class_name],
                    shots=shot,
                    context={
                        "scenario": 2,
                        "setting": (
                            setting if prompt_sampling == "per_setting_sequence" else "shared"
                        ),
                        "class_name": split.class_name,
                        "subclass": split.subclass,
                        "seed": seed,
                        "prompt_sampling": prompt_sampling,
                        "target_size": target_size,
                        **shot_context,
                    },
                )
                sample_maps[sample.sample_id] = resize_anomaly_map(result.anomaly_map, target_size)
                detection_scores.append(result.detection_score)
                role = (
                    "positive"
                    if sample.sample_id in positive_ids
                    else "n1_good"
                    if sample.sample_id in good_ids
                    else "n2_irrelevant"
                )
                subclass_rows.append(
                    {
                        "scenario": 2,
                        "setting": setting,
                        "class_name": split.class_name,
                        "subclass": split.subclass,
                        "part_count": split.part_count,
                        "sample_id": sample.sample_id,
                        "role": role,
                        "is_anomalous_for_subclass": role == "positive",
                        "image_path": str(sample.image_path),
                        "prompt": prompt,
                        "prompt_index": list(pool).index(prompt),
                        "shots": shot,
                        "detection_score": result.detection_score,
                        "cache_key": result.cache_key,
                        "cache_path": str(result.cache_path),
                        "cache_hit": result.cache_hit,
                        "response_text": result.response_text,
                    }
                )

            image_labels = [
                int(sample.sample_id in positive_ids) for sample in evaluation_samples
            ]
            # Scored over this setting's image set: for AA-CLIP the same image can score
            # differently under EV1 and EV2 (see image_level_scores).
            image_scores = image_level_scores(
                backend.model_name,
                [sample_maps[sample.sample_id] for sample in evaluation_samples],
                detection_scores,
            )
            for row, score in zip(subclass_rows, image_scores):
                row["image_score"] = score
            prediction_rows.extend(subclass_rows)

            if localization_scope == "evaluation_set":
                localization_samples = evaluation_samples
                localization_masks = [
                    load_mvtec_mask(sample.mask_path, sample.image_path, target_size)
                    if sample.sample_id in positive_ids
                    else np.zeros((target_size, target_size), dtype=bool)
                    for sample in localization_samples
                ]
            else:
                localization_samples = sorted(
                    split.positives, key=lambda sample: sample.sample_id
                )
                localization_masks = [
                    load_mvtec_mask(sample.mask_path, sample.image_path, target_size)
                    for sample in localization_samples
                ]
            localization_maps = [sample_maps[sample.sample_id] for sample in localization_samples]
            metric = _bundle_from_scopes(
                image_labels,
                image_scores,
                localization_masks,
                localization_maps,
                max_fpr=aupro_max_fpr,
                thresholds=aupro_thresholds,
            )
            bundles[(setting, split.part_count, split.class_name)].append(
                (split.subclass, metric)
            )
            per_subclass_rows.append(
                metric_row(
                    metric,
                    scenario=2,
                    setting=setting,
                    localization_scope=localization_scope,
                    class_name=split.class_name,
                    subclass=split.subclass,
                    part_count=split.part_count,
                    a_exact=len(split.positives),
                    n1_good=len(split.normal_good),
                    n2_disjoint=len(split.irrelevant_defects),
                    excluded_overlap=len(split.excluded),
                    n_localization_images=len(localization_samples),
                )
            )

    part_counts = sorted({split.part_count for split in splits})
    for setting in normalized_settings:
        for part_count in part_counts:
            subclass_metrics = [
                metric
                for (bundle_setting, bundle_parts, _), entries in bundles.items()
                if bundle_setting == setting and bundle_parts == part_count
                for _, metric in entries
            ]
            if subclass_metrics:
                summary_rows.append(
                    metric_row(
                        macro_mean(subclass_metrics),
                        scenario=2,
                        setting=setting,
                        localization_scope=localization_scope,
                        part_count=part_count,
                        aggregation="macro_subclass",
                        n_units=len(subclass_metrics),
                    )
                )

            # Equal class weighting: first average subclasses of this cardinality
            # within each class, then average the class means.
            class_means: list[MetricBundle] = []
            for class_name in selected_classes:
                entries = bundles.get((setting, part_count, class_name), [])
                if entries:
                    class_means.append(macro_mean([metric for _, metric in entries]))
            if class_means:
                summary_rows.append(
                    metric_row(
                        macro_mean(class_means),
                        scenario=2,
                        setting=setting,
                        localization_scope=localization_scope,
                        part_count=part_count,
                        aggregation="macro_class",
                        n_units=len(class_means),
                    )
                )

    metadata = {
        "scenario": 2,
        "dataset": "MVTec AD per-part extension",
        "classes": list(selected_classes),
        "settings": list(normalized_settings),
        "target_size": target_size,
        "seed": seed,
        "prompt_sampling": prompt_sampling,
        "prompt_sampling_note": (
            "stable_hash gives a deterministic uniform draw per sample and shares it across "
            "EV1/EV2; sequence modes draw from a seeded random.Random stream instead"
        ),
        "model": backend.model_name,
        "shots": shot,
        "references": (
            "none"
            if shot == 0
            else "every train/good image of the class"
            if shot == FULL_SHOT
            else f"train/good/{reference_round * 4:03d}.png"
        ),
        "prompt_handling": backend.spec.prompt_handling,
        "text_conditioning": backend.text_conditioning,
        "localization_scope": localization_scope,
        "localization_scope_note": (
            "evaluation_set: P-AUROC/AUPRO computed per setting on the image-level set, with "
            "ground-truth masks for A(Ts) and all-zero masks for N1 (EV1) or N1+N2 (EV2); "
            "in_scope_only: localization on A(Ts) only, identical across settings"
        ),
        "image_score": IMAGE_SCORE_RULES[backend.spec.image_score_rule],
        "aggregation_note": (
            "macro_subclass weights all subclasses of a part count equally; macro_class first "
            "averages subclasses within each class, then averages the class means"
        ),
        "backend_fingerprint": backend.fingerprint,
        "subclass_inventory": (
            tags.validate_inventory()
            if all_nine_classes
            else tags.inventory_summary(selected_classes)
        ),
        "aupro": {"max_fpr": aupro_max_fpr, "thresholds": aupro_thresholds},
    }
    return finalize_output(
        output,
        per_unit_rows=per_subclass_rows,
        summary_rows=summary_rows,
        prediction_rows=prediction_rows,
        metadata=metadata,
        per_unit_name="per_subclass.csv",
    )
