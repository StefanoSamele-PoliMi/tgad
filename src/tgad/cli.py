"""Command-line interface for the three TGAD benchmark scenarios."""

from __future__ import annotations

import argparse
import json
import os
from importlib.resources import files
from pathlib import Path
from typing import Sequence

from .backend import (
    MODEL_SPECS,
    MockAnomalyBackend,
    OfficialAAClipBackend,
    OfficialAnomalyGPTBackend,
    Shot,
    parse_shot,
)
from .config import AAClipModelConfig, AnomalyGPTModelConfig, LogSADModelConfig
from .datasets.apd_coco import APDIndex
from .datasets.mvtec import MVTecDatasetIndex
from .datasets.mvtec_parts import PART_CLASSES, MVTecPartTags
from .evaluation.scenario1 import run_scenario1
from .evaluation.scenario2 import DEFAULT_LOCALIZATION_SCOPE, LOCALIZATION_SCOPES, run_scenario2
from .evaluation.scenario3 import run_scenario3
from .utils import write_json


def _csv_or_space(values: Sequence[str] | None) -> list[str] | None:
    if values is None:
        return None
    result: list[str] = []
    for value in values:
        result.extend(item.strip() for item in value.split(",") if item.strip())
    return result


def _parse_shots(values: Sequence[str] | None) -> list[Shot] | None:
    """Parse shot regimes (0, 1, full); ``None`` (option not given) selects the model's defaults."""

    if values is None:
        return None
    return [parse_shot(value) for value in _csv_or_space(values) or []]


# Model config file and environment-variable fallback of each model; both LogSAD
# variants share one config.
_MODEL_CONFIG = {
    "anomalygpt": ("anomalygpt", "TGAD_ANOMALYGPT_MODEL_CONFIG"),
    "aaclip": ("aaclip", "TGAD_AACLIP_MODEL_CONFIG"),
    "logsad": ("logsad", "TGAD_LOGSAD_MODEL_CONFIG"),
    "logsad_fms": ("logsad", "TGAD_LOGSAD_MODEL_CONFIG"),
}


def _model_config_path(args: argparse.Namespace, command: str) -> str:
    name, variable = _MODEL_CONFIG[args.model]
    config_path = args.model_config or os.environ.get(variable)
    if not config_path:
        raise SystemExit(
            f"{command} requires --model-config or {variable}. "
            f"Copy configs/{name}.example.yaml and fill in checkpoint paths."
        )
    return config_path


def _backend_from_args(args: argparse.Namespace):
    if args.backend == "mock":
        return MockAnomalyBackend(target_size=args.target_size, model=args.model)
    config_path = _model_config_path(args, "The official backend")
    if args.model == "aaclip":
        return OfficialAAClipBackend(AAClipModelConfig.from_yaml(config_path))
    if args.model in {"logsad", "logsad_fms"}:
        # Imported here: the LogSAD backend runs in its own environment.
        from .logsad import OfficialLogSADBackend

        return OfficialLogSADBackend(
            LogSADModelConfig.from_yaml(config_path),
            foreground_masking=args.model == "logsad_fms",
        )
    return OfficialAnomalyGPTBackend(AnomalyGPTModelConfig.from_yaml(config_path))


def _add_model_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_SPECS),
        default="anomalygpt",
        help="Evaluated model (default: anomalygpt)",
    )


def _add_backend_args(parser: argparse.ArgumentParser) -> None:
    _add_model_arg(parser)
    parser.add_argument("--backend", choices=("official", "mock"), default="official")
    parser.add_argument("--model-config", help="YAML containing official repository/checkpoint paths")
    parser.add_argument("--target-size", type=int, default=224)
    parser.add_argument("--no-cache", action="store_true", help="Disable prediction-map cache")


def _output_dir(args: argparse.Namespace, scenario: str) -> str:
    """Default output directory; other models get their own so they never overwrite AnomalyGPT's."""

    if args.output_dir:
        return args.output_dir
    return f"outputs/{scenario}" if args.model == "anomalygpt" else f"outputs/{args.model}/{scenario}"


def _print_paths(paths: dict[str, Path]) -> None:
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2))


def cmd_scenario1(args: argparse.Namespace) -> None:
    backend = _backend_from_args(args)
    shots = _parse_shots(args.shots)
    variants = _csv_or_space(args.variants)
    variants_by_shot = None
    if variants:
        variants_by_shot = {shot: variants for shot in shots or MODEL_SPECS[args.model].shots}
    paths = run_scenario1(
        mvtec_root=args.mvtec_root,
        backend=backend,
        output_dir=_output_dir(args, "scenario1"),
        cache_dir=args.cache_dir,
        classes=_csv_or_space(args.classes),
        shots=shots,
        variants_by_shot=variants_by_shot,
        prompt_config=args.prompt_config,
        reference_round=args.reference_round,
        target_size=args.target_size,
        aupro_max_fpr=args.aupro_max_fpr,
        aupro_thresholds=args.aupro_thresholds,
        cache_enabled=not args.no_cache,
    )
    _print_paths(paths)


def cmd_scenario2(args: argparse.Namespace) -> None:
    backend = _backend_from_args(args)
    paths = run_scenario2(
        mvtec_root=args.mvtec_root,
        backend=backend,
        output_dir=_output_dir(args, "scenario2"),
        cache_dir=args.cache_dir,
        tags_file=args.tags_file,
        prompt_config=args.prompt_config,
        classes=_csv_or_space(args.classes),
        settings=_csv_or_space(args.settings) or ["ev1", "ev2"],
        seed=args.seed,
        prompt_sampling=args.prompt_sampling,
        localization_scope=args.localization_scope,
        target_size=args.target_size,
        aupro_max_fpr=args.aupro_max_fpr,
        aupro_thresholds=args.aupro_thresholds,
        cache_enabled=not args.no_cache,
        validate_cardinalities=not args.skip_cardinality_validation,
        shots=parse_shot(args.shots) if args.shots is not None else None,
        reference_round=args.reference_round,
    )
    _print_paths(paths)


def cmd_scenario3(args: argparse.Namespace) -> None:
    backend = _backend_from_args(args)
    variants = _csv_or_space(args.standard_variants)
    standard_shots = _parse_shots(args.standard_shots)
    variants_by_shot = None
    if variants:
        variants_by_shot = {
            shot: variants for shot in standard_shots or MODEL_SPECS[args.model].shots
        }
    paths = run_scenario3(
        apd_root=args.apd_root,
        coco_annotations=args.coco_annotations,
        backend=backend,
        output_dir=_output_dir(args, "scenario3"),
        cache_dir=args.cache_dir,
        prepared_cache_dir=args.prepared_cache_dir,
        prompt_config=args.prompt_config,
        modes=_csv_or_space(args.modes) or ["standard", "per_error"],
        preprocessing=_csv_or_space(args.preprocessing) or ["direct_resize", "pad_to_square"],
        standard_shots=standard_shots,
        per_error_shots=_parse_shots(args.per_error_shots),
        standard_variants_by_shot=variants_by_shot,
        one_shot_reference=args.one_shot_reference,
        full_shot_references=args.full_shot_references,
        dataset_dir_name=args.dataset_dir_name,
        target_size=args.target_size,
        aupro_max_fpr=args.aupro_max_fpr,
        aupro_thresholds=args.aupro_thresholds,
        cache_enabled=not args.no_cache,
        validate_protocol_counts=not args.skip_protocol_count_validation,
    )
    _print_paths(paths)


def cmd_validate(args: argparse.Namespace) -> None:
    resource = files("tgad.resources").joinpath("mvtec_test_part_tags_raw.txt")
    tags = MVTecPartTags.from_text(resource.read_text(encoding="utf-8"), source=str(resource))
    report: dict[str, object] = {"part_tags": tags.validate_inventory()}
    if args.mvtec_root:
        dataset = MVTecDatasetIndex(args.mvtec_root, PART_CLASSES)
        report["mvtec"] = {
            "summary": dataset.summary(),
            "part_tags": tags.validate_against_dataset(dataset),
        }
    if args.apd_root or args.coco_annotations:
        if not args.apd_root or not args.coco_annotations:
            raise SystemExit("APD validation needs both --apd-root and --coco-annotations")
        apd = APDIndex(
            args.apd_root,
            args.coco_annotations,
            dataset_dir_name=args.dataset_dir_name,
            strict=True,
        )
        report["apd"] = apd.validate_protocol_counts()
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def cmd_doctor(args: argparse.Namespace) -> None:
    config_path = _model_config_path(args, "doctor")
    if args.model in {"logsad", "logsad_fms"}:
        logsad = LogSADModelConfig.from_yaml(config_path)
        logsad.validate_paths()
        report = {
            "status": "paths_ok",
            "model": args.model,
            "official_repo": str(logsad.official_repo),
            "official_checkout_has_git_head": (logsad.official_repo / ".git" / "HEAD").is_file(),
            "clip_model": logsad.clip_model,
            "device": logsad.device,
            "note": (
                "Path validation only; run in the LogSAD environment. The first experiment "
                "downloads the CLIP weights and performs the model import test."
            ),
        }
        print(json.dumps(report, indent=2))
        return
    if args.model == "aaclip":
        aaclip = AAClipModelConfig.from_yaml(config_path)
        aaclip.validate_paths()
        report = {
            "status": "paths_ok",
            "model": "aaclip",
            "official_repo": str(aaclip.official_repo),
            "official_checkout_has_git_head": (aaclip.official_repo / ".git" / "HEAD").is_file(),
            "clip_checkpoint": str(aaclip.clip_checkpoint),
            "text_adapter": str(aaclip.text_adapter),
            "image_adapter": str(aaclip.image_adapter),
            "device": aaclip.device,
            "image_size": aaclip.image_size,
            "note": "Path validation only; the first experiment performs the model import test.",
        }
        print(json.dumps(report, indent=2))
        return
    config = AnomalyGPTModelConfig.from_yaml(config_path)
    config.validate_paths()
    revision_file = config.official_repo / ".git" / "HEAD"
    report = {
        "status": "paths_ok",
        "official_repo": str(config.official_repo),
        "official_checkout_has_git_head": revision_file.is_file(),
        "imagebind_checkpoint": str(config.imagebind_checkpoint),
        "vicuna_checkpoint": str(config.vicuna_checkpoint),
        "pandagpt_delta": str(config.pandagpt_delta),
        "anomalygpt_delta": str(config.anomalygpt_delta),
        "device": config.device,
        "inference_path": config.inference_path,
        "text_conditioning": config.text_conditioning,
        "note": "Path validation only; the first experiment performs the CUDA/model import test.",
    }
    print(json.dumps(report, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tgad",
        description=(
            "Run the three TGAD benchmark scenarios with pre-trained AnomalyGPT, AA-CLIP, or LogSAD."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    output_help = "default: outputs/<scenario> for anomalygpt, outputs/<model>/<scenario> otherwise"

    s1 = subparsers.add_parser("scenario1", help="MVTec prompt-sensitivity experiment")
    _add_backend_args(s1)
    s1.add_argument("--mvtec-root", required=True)
    s1.add_argument("--output-dir", help=output_help)
    s1.add_argument("--cache-dir", default="outputs/cache/predictions")
    s1.add_argument("--classes", nargs="+")
    s1.add_argument(
        "--shots",
        nargs="+",
        help=(
            "Shot regimes: 0, 1, full (default: every regime the model is evaluated in; "
            "anomalygpt 0 1, aaclip 0, logsad 1 full)"
        ),
    )
    s1.add_argument("--variants", nargs="+", help="Override the default variants for every selected shot")
    s1.add_argument("--prompt-config")
    s1.add_argument(
        "--reference-round",
        type=int,
        help="1-shot reference train/good/{4*round:03d}.png (default: 3 for anomalygpt, 0 for logsad)",
    )
    s1.add_argument("--aupro-max-fpr", type=float, default=0.30)
    s1.add_argument("--aupro-thresholds", type=int, default=200)
    s1.set_defaults(func=cmd_scenario1)

    s2 = subparsers.add_parser("scenario2", help="MVTec per-part EV1/EV2 experiment")
    _add_backend_args(s2)
    s2.add_argument("--mvtec-root", required=True)
    s2.add_argument("--output-dir", help=output_help)
    s2.add_argument("--cache-dir", default="outputs/cache/predictions")
    s2.add_argument("--tags-file")
    s2.add_argument("--prompt-config")
    s2.add_argument("--classes", nargs="+")
    s2.add_argument("--settings", nargs="+", default=["ev1", "ev2"])
    s2.add_argument(
        "--shots",
        help="Supervision regime: 0, 1, or full (default: 0, or full for logsad and logsad_fms)",
    )
    s2.add_argument(
        "--reference-round",
        type=int,
        help="1-shot reference train/good/{4*round:03d}.png (default: the model's)",
    )
    s2.add_argument("--seed", type=int, default=0)
    s2.add_argument(
        "--prompt-sampling",
        choices=("global_sequence", "per_setting_sequence", "stable_hash"),
        default="stable_hash",
    )
    s2.add_argument(
        "--localization-scope",
        choices=LOCALIZATION_SCOPES,
        default=DEFAULT_LOCALIZATION_SCOPE,
        help=(
            "evaluation_set (default): P-AUROC/AUPRO per setting on the EV1 or EV2 image set, "
            "with zero masks for images admitted as normal; in_scope_only: on A(Ts) only."
        ),
    )
    s2.add_argument("--skip-cardinality-validation", action="store_true")
    s2.add_argument("--aupro-max-fpr", type=float, default=0.30)
    s2.add_argument("--aupro-thresholds", type=int, default=200)
    s2.set_defaults(func=cmd_scenario2)

    s3 = subparsers.add_parser("scenario3", help="Assembled Panel Standard/Per-Error experiment")
    _add_backend_args(s3)
    s3.add_argument(
        "--apd-root",
        default="data",
        help="Parent of the APD directory (default: data, i.e. data/apd/{normal,anomalous})",
    )
    s3.add_argument(
        "--coco-annotations",
        default="data/apd/assembled_panel_annotations.json",
        help="APD COCO annotation file (default: data/apd/assembled_panel_annotations.json)",
    )
    s3.add_argument("--dataset-dir-name", default="apd")
    s3.add_argument("--output-dir", help=output_help)
    s3.add_argument("--cache-dir", default="outputs/cache/predictions")
    s3.add_argument("--prepared-cache-dir", default="outputs/cache/prepared_apd")
    s3.add_argument("--prompt-config")
    s3.add_argument("--modes", nargs="+", default=["standard", "per_error"])
    s3.add_argument("--preprocessing", nargs="+", default=["direct_resize", "pad_to_square"])
    s3.add_argument(
        "--standard-shots",
        nargs="+",
        help="Standard AD shot regimes (default: every swept regime; anomalygpt 0 1, aaclip 0, logsad 1 full)",
    )
    s3.add_argument(
        "--per-error-shots",
        nargs="+",
        help="Per-Error AD shot regimes (default: 0, or full for logsad and logsad_fms)",
    )
    s3.add_argument("--standard-variants", nargs="+")
    s3.add_argument("--one-shot-reference")
    s3.add_argument(
        "--full-shot-references",
        nargs="+",
        help="Normal images of the full-shot memory bank (default: the 1-shot reference)",
    )
    s3.add_argument("--skip-protocol-count-validation", action="store_true")
    s3.add_argument("--aupro-max-fpr", type=float, default=0.30)
    s3.add_argument("--aupro-thresholds", type=int, default=200)
    s3.set_defaults(func=cmd_scenario3)

    validate = subparsers.add_parser("validate", help="Validate benchmark annotations and dataset layouts")
    validate.add_argument("--mvtec-root")
    validate.add_argument("--apd-root")
    validate.add_argument("--coco-annotations")
    validate.add_argument("--dataset-dir-name", default="apd")
    validate.add_argument("--output")
    validate.set_defaults(func=cmd_validate)

    doctor = subparsers.add_parser("doctor", help="Validate official repository/checkpoint paths")
    _add_model_arg(doctor)
    doctor.add_argument("--model-config")
    doctor.set_defaults(func=cmd_doctor)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
