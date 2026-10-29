"""Typed accessors for the benchmark's verbatim prompt resources.

Prompts are loaded exactly as written in the paper appendices and passed to the model
unchanged: the benchmark never extracts nouns from, rewrites, or appends text to them.
Each model has its own resources, matching its textual interface: free-form prompts for
AnomalyGPT, class keywords for AA-CLIP, foreground and background keyword sets for LogSAD.
A keyword set is written as a ``{foreground: [...], background: [...]}`` mapping and passed
to the model as its canonical JSON string.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .backend import Shot, parse_shot
from .config import load_resource_yaml, load_yaml
from .exceptions import DataValidationError


# Packaged prompt resource of each (model, scenario).
_RESOURCE_PREFIX = {
    "anomalygpt": "",
    "aaclip": "aaclip_",
    "logsad": "logsad_",
    "logsad_fms": "logsad_",
}
SPATIAL_STRATEGIES = ("direct_resize", "pad_to_square")
STANDARD_LEVELS = ("minimal", "simple", "detailed", "very_detailed")


def _load_raw(path: str | Path | None, model: str, scenario: int) -> Any:
    if path:
        return load_yaml(path)
    if model not in _RESOURCE_PREFIX:
        raise ValueError(f"No prompt resources for model {model!r}")
    return load_resource_yaml(f"{_RESOURCE_PREFIX[model]}scenario{scenario}_prompts.yaml")


def _prompt_text(value: Any) -> str:
    """Return a prompt string: text as written, or a keyword set as canonical JSON."""

    if not isinstance(value, Mapping):
        return str(value)
    if set(value) != {"foreground", "background"}:
        raise DataValidationError(f"A keyword set needs foreground and background lists: {value}")
    keywords = {key: [str(word) for word in value[key]] for key in ("foreground", "background")}
    if not keywords["foreground"] or not keywords["background"]:
        raise DataValidationError(f"A keyword set needs non-empty keyword lists: {value}")
    return json.dumps(keywords, ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class Scenario1Prompts:
    variants: Mapping[str, Mapping[str, str]]
    class_order: tuple[str, ...]
    zero_shot_variants: tuple[str, ...]
    one_shot_variants: tuple[str, ...]
    full_shot_variants: tuple[str, ...] = ()

    @classmethod
    def load(cls, path: str | Path | None = None, model: str = "anomalygpt") -> "Scenario1Prompts":
        raw = _load_raw(path, model, 1)
        if not isinstance(raw, dict):
            raise DataValidationError("Scenario 1 prompt configuration must be a mapping")
        metadata = raw["metadata"]
        result = cls(
            variants={
                variant: {name: _prompt_text(prompt) for name, prompt in prompts.items()}
                for variant, prompts in raw["variants"].items()
            },
            class_order=tuple(raw["class_order"]),
            zero_shot_variants=tuple(metadata.get("zero_shot_variants") or ()),
            one_shot_variants=tuple(metadata.get("one_shot_variants") or ()),
            full_shot_variants=tuple(metadata.get("full_shot_variants") or ()),
        )
        result.validate()
        return result

    @property
    def variants_by_shot(self) -> dict[Shot, tuple[str, ...]]:
        return {0: self.zero_shot_variants, 1: self.one_shot_variants, "full": self.full_shot_variants}

    def validate(self) -> None:
        expected = set(self.class_order)
        if not self.variants:
            raise DataValidationError("Scenario 1 requires at least one prompt variant")
        for variant, prompts in self.variants.items():
            if set(prompts) != expected:
                raise DataValidationError(f"Scenario 1 prompt classes are incomplete for {variant}")
            empty = [name for name, prompt in prompts.items() if not str(prompt).strip()]
            if empty:
                raise DataValidationError(f"Scenario 1 has empty prompts for {variant}: {empty}")
        listed = set(self.zero_shot_variants) | set(self.one_shot_variants) | set(
            self.full_shot_variants
        )
        unknown = listed - set(self.variants)
        if unknown:
            raise DataValidationError(
                f"Scenario 1 metadata references unknown variants: {sorted(unknown)}"
            )


@dataclass(frozen=True, slots=True)
class Scenario2Prompts:
    pools: Mapping[str, Sequence[str]]
    # Three LLM-written paraphrases for AnomalyGPT (Appendix D.1), one keyword for AA-CLIP.
    prompts_per_subclass: int = 3

    @classmethod
    def load(cls, path: str | Path | None = None, model: str = "anomalygpt") -> "Scenario2Prompts":
        raw = _load_raw(path, model, 2)
        if not isinstance(raw, dict):
            raise DataValidationError("Scenario 2 prompt configuration must be a mapping")
        metadata = raw.get("metadata") or {}
        result = cls(
            pools={
                name: [_prompt_text(prompt) for prompt in pool]
                for name, pool in raw["pools"].items()
            },
            prompts_per_subclass=int(metadata.get("prompts_per_subclass", 3)),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if len(self.pools) != 42:
            raise DataValidationError(f"Scenario 2 requires 42 prompt pools, got {len(self.pools)}")
        bad = {
            name: len(pool)
            for name, pool in self.pools.items()
            if len(pool) != self.prompts_per_subclass
        }
        if bad:
            raise DataValidationError(
                f"Every Scenario 2 subclass needs {self.prompts_per_subclass} prompt(s): {bad}"
            )
        empty = {
            name: [index for index, prompt in enumerate(pool) if not str(prompt).strip()]
            for name, pool in self.pools.items()
            if any(not str(prompt).strip() for prompt in pool)
        }
        if empty:
            raise DataValidationError(f"Scenario 2 contains empty prompt strings: {empty}")


@dataclass(frozen=True, slots=True)
class Scenario3Prompts:
    standard: Mapping[str, str]
    per_error: Mapping[str, str]
    coco_error_to_paper_id: Mapping[str, str]
    # Default Standard AD sweep: spatial strategy -> shot regime -> prompt variants.
    standard_variants: Mapping[str, Mapping[Shot, tuple[str, ...]]]

    @classmethod
    def load(cls, path: str | Path | None = None, model: str = "anomalygpt") -> "Scenario3Prompts":
        raw = _load_raw(path, model, 3)
        if not isinstance(raw, dict):
            raise DataValidationError("Scenario 3 prompt configuration must be a mapping")
        result = cls(
            standard={name: _prompt_text(prompt) for name, prompt in raw["standard"].items()},
            per_error={name: _prompt_text(prompt) for name, prompt in raw["per_error"].items()},
            coco_error_to_paper_id=raw["coco_error_to_paper_id"],
            standard_variants={
                strategy: {parse_shot(shot): tuple(variants) for shot, variants in by_shot.items()}
                for strategy, by_shot in raw["metadata"]["standard_variants"].items()
            },
        )
        result.validate()
        return result

    def validate(self) -> None:
        if not self.standard or not set(self.standard) <= set(STANDARD_LEVELS):
            raise DataValidationError(
                f"Scenario 3 Standard AD variants must be among {list(STANDARD_LEVELS)}"
            )
        expected = {f"A{index}" for index in range(1, 15)}
        if set(self.per_error) != expected:
            raise DataValidationError("Scenario 3 Per-Error AD requires A1 through A14")
        identity = {f"err_{index}": f"A{index}" for index in range(1, 15)}
        if dict(self.coco_error_to_paper_id) != identity:
            raise DataValidationError(
                "APD categories must follow the convention err_k -> A_k for k = 1..14"
            )
        if any(not str(prompt).strip() for prompt in self.standard.values()):
            raise DataValidationError("Scenario 3 Standard AD contains an empty prompt")
        if any(not str(prompt).strip() for prompt in self.per_error.values()):
            raise DataValidationError("Scenario 3 Per-Error AD contains an empty prompt")
        if set(self.standard_variants) != set(SPATIAL_STRATEGIES):
            raise DataValidationError(
                f"Scenario 3 standard_variants must list {list(SPATIAL_STRATEGIES)}"
            )
        for strategy, by_shot in self.standard_variants.items():
            for shot, variants in by_shot.items():
                unknown = sorted(set(variants) - set(self.standard))
                if unknown:
                    raise DataValidationError(
                        f"Scenario 3 standard_variants for {strategy}/{shot} reference unknown "
                        f"variants: {unknown}"
                    )
