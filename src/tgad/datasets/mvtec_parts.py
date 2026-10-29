"""Parser and EV1/EV2 construction for the component-tagged MVTec extension."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import yaml

from ..exceptions import DataValidationError
from ..records import ImageSample
from ..utils import natural_key
from .mvtec import MVTecDatasetIndex


PART_CLASSES: tuple[str, ...] = (
    "cable",
    "capsule",
    "hazelnut",
    "metal_nut",
    "pill",
    "screw",
    "toothbrush",
    "transistor",
    "zipper",
)

# Appendix A uses metal_nut1/metal_nut2, while the supplied raw annotation
# artifact spells the values metalnut1/metalnut2. Keep the raw artifact intact
# and normalize that one documented notation mismatch at load time.
TAG_ALIASES: Mapping[str, Mapping[str, str]] = {
    "metal_nut": {
        "metalnut1": "metal_nut1",
        "metalnut2": "metal_nut2",
    }
}

COMPONENT_TAGS: Mapping[str, tuple[str, ...]] = {
    "cable": ("cable1", "cable2", "cable3", "cable4"),
    "capsule": ("capsule1", "capsule2", "capsule3"),
    "hazelnut": ("hazelnut1", "hazelnut2"),
    "metal_nut": ("metal_nut1", "metal_nut2"),
    "pill": ("pill1", "pill2"),
    "screw": ("screw1", "screw2", "screw3"),
    "toothbrush": ("toothbrush1", "toothbrush2", "toothbrush3"),
    "transistor": ("transistor1", "transistor2"),
    "zipper": ("zipper1", "zipper2", "zipper3"),
}


def canonical_tag(class_name: str, tag: str) -> str:
    value = str(tag).strip()
    return TAG_ALIASES.get(class_name, {}).get(value, value)


def tag_sort_key(tag: str) -> tuple[object, ...]:
    return natural_key(tag)


def subclass_name(tags: Iterable[str]) -> str:
    return "_".join(sorted(set(tags), key=tag_sort_key))


@dataclass(frozen=True, slots=True)
class SubclassSplit:
    class_name: str
    subclass: str
    tags: frozenset[str]
    positives: tuple[ImageSample, ...]
    normal_good: tuple[ImageSample, ...]
    irrelevant_defects: tuple[ImageSample, ...]
    excluded: tuple[ImageSample, ...]

    @property
    def part_count(self) -> int:
        return len(self.tags)

    def image_level_samples(self, setting: str) -> tuple[ImageSample, ...]:
        setting = setting.lower()
        if setting == "ev1":
            return self.positives + self.normal_good
        if setting == "ev2":
            return self.positives + self.normal_good + self.irrelevant_defects
        raise ValueError(f"Unknown evaluation setting: {setting}")


class MVTecPartTags:
    """Normalized view of the supplied per-image part-tag annotation."""

    def __init__(self, annotations: Mapping[tuple[str, str, str], frozenset[str]]) -> None:
        self.annotations = dict(annotations)
        unknown_classes = sorted({key[0] for key in self.annotations} - set(PART_CLASSES))
        if unknown_classes:
            raise DataValidationError(
                f"Unexpected classes in part tags: {', '.join(unknown_classes)}"
            )
        invalid_tags: list[str] = []
        for (class_name, defect_type, filename), tags in self.annotations.items():
            allowed = set(COMPONENT_TAGS[class_name])
            unexpected = sorted(set(tags) - allowed, key=tag_sort_key)
            if unexpected:
                invalid_tags.append(
                    f"{class_name}/{defect_type}/{filename}: {', '.join(unexpected)}"
                )
        if invalid_tags:
            raise DataValidationError(
                "Unexpected component tag(s):\n  - " + "\n  - ".join(invalid_tags[:20])
            )

    @classmethod
    def from_file(cls, path: str | Path) -> "MVTecPartTags":
        return cls.from_text(Path(path).read_text(encoding="utf-8"), source=str(path))

    @classmethod
    def from_text(cls, text: str, *, source: str = "<text>") -> "MVTecPartTags":
        lines = text.splitlines()
        if lines and not lines[0].rstrip().endswith(":"):
            # The supplied artifact starts with a human-readable title.
            lines = lines[1:]
        body = "\n".join(lines).replace("\t", "    ")
        raw = yaml.safe_load(body)
        if not isinstance(raw, dict):
            raise DataValidationError(f"Could not parse part-tag annotation: {source}")
        annotations: dict[tuple[str, str, str], frozenset[str]] = {}
        for raw_class_name, defects in raw.items():
            class_name = str(raw_class_name)
            if class_name not in PART_CLASSES:
                raise DataValidationError(f"Unexpected class in part tags: {class_name}")
            if not isinstance(defects, dict):
                raise DataValidationError(f"Invalid defect mapping for class {class_name}")
            for defect_type, images in defects.items():
                if not isinstance(images, dict):
                    raise DataValidationError(
                        f"Invalid image mapping for {class_name}/{defect_type}"
                    )
                for filename, tag_value in images.items():
                    tags = frozenset(
                        canonical_tag(class_name, item)
                        for item in str(tag_value).split(",")
                        if item.strip()
                    )
                    if not tags:
                        raise DataValidationError(
                            f"Empty tag set for {class_name}/{defect_type}/{filename}"
                        )
                    key = (class_name, str(defect_type), str(filename))
                    if key in annotations:
                        raise DataValidationError(f"Duplicate part-tag entry: {key}")
                    annotations[key] = tags
        return cls(annotations)

    def tags_for(self, sample: ImageSample) -> frozenset[str]:
        if not sample.is_anomalous:
            return frozenset()
        key = (str(sample.class_name), str(sample.defect_type), sample.image_path.name)
        try:
            return self.annotations[key]
        except KeyError as exc:
            raise DataValidationError(
                f"No part tags found for defective sample {sample.sample_id}"
            ) from exc

    def subclasses(self, class_name: str) -> list[frozenset[str]]:
        unique = {
            tags
            for (annotation_class, _, _), tags in self.annotations.items()
            if annotation_class == class_name
        }
        return sorted(unique, key=lambda tags: (len(tags), subclass_name(tags)))

    def subclass_inventory(
        self, classes: Sequence[str] | None = None
    ) -> list[dict[str, object]]:
        selected = tuple(classes or PART_CLASSES)
        rows: list[dict[str, object]] = []
        for class_name in selected:
            for tags in self.subclasses(class_name):
                rows.append(
                    {
                        "class_name": class_name,
                        "subclass": subclass_name(tags),
                        "part_count": len(tags),
                        "tags": sorted(tags, key=tag_sort_key),
                    }
                )
        return rows

    def inventory_summary(
        self, classes: Sequence[str] | None = None
    ) -> dict[str, object]:
        selected = tuple(classes or PART_CLASSES)
        inventory = self.subclass_inventory(selected)
        cardinalities = Counter(int(row["part_count"]) for row in inventory)
        selected_set = set(selected)
        return {
            "classes": len(selected),
            "class_names": list(selected),
            "subclasses": len(inventory),
            "cardinality_counts": dict(sorted(cardinalities.items())),
            "tagged_defective_images": sum(
                key[0] in selected_set for key in self.annotations
            ),
        }

    def build_split(
        self,
        dataset: MVTecDatasetIndex,
        class_name: str,
        tags: frozenset[str],
    ) -> SubclassSplit:
        positives: list[ImageSample] = []
        irrelevant: list[ImageSample] = []
        excluded: list[ImageSample] = []
        for sample in dataset.defective_samples(class_name):
            image_tags = self.tags_for(sample)
            if image_tags == tags:
                positives.append(sample)
            elif image_tags.isdisjoint(tags):
                irrelevant.append(sample)
            else:
                excluded.append(sample)
        return SubclassSplit(
            class_name=class_name,
            subclass=subclass_name(tags),
            tags=tags,
            positives=tuple(positives),
            normal_good=tuple(dataset.normal_test_samples(class_name)),
            irrelevant_defects=tuple(irrelevant),
            excluded=tuple(excluded),
        )

    def all_splits(self, dataset: MVTecDatasetIndex) -> list[SubclassSplit]:
        splits: list[SubclassSplit] = []
        for class_name in dataset.classes:
            if class_name not in PART_CLASSES:
                continue
            for tags in self.subclasses(class_name):
                splits.append(self.build_split(dataset, class_name, tags))
        return splits

    def validate_inventory(self) -> dict[str, object]:
        summary = self.inventory_summary(PART_CLASSES)
        expected = {1: 24, 2: 16, 3: 2}
        if (
            summary["subclasses"] != 42
            or summary["cardinality_counts"] != expected
            or summary["tagged_defective_images"] != 813
        ):
            raise DataValidationError(
                "Expected 42 subclasses (24/16/2 by cardinality) and 813 tagged "
                f"defective images; got {summary}"
            )
        return summary

    def validate_against_dataset(self, dataset: MVTecDatasetIndex) -> dict[str, object]:
        missing: list[str] = []
        selected_classes = tuple(
            class_name for class_name in PART_CLASSES if class_name in dataset.classes
        )
        extra_keys = {key for key in self.annotations if key[0] in selected_classes}
        for class_name in selected_classes:
            for sample in dataset.defective_samples(class_name):
                key = (class_name, str(sample.defect_type), sample.image_path.name)
                if key not in self.annotations:
                    missing.append(sample.sample_id)
                extra_keys.discard(key)
        if missing:
            raise DataValidationError(
                f"{len(missing)} defective MVTec images lack part tags; first: {missing[:5]}"
            )
        if extra_keys:
            preview = sorted(extra_keys)[:5]
            raise DataValidationError(
                f"{len(extra_keys)} part-tag entries do not match MVTec images; first: {preview}"
            )
        return self.inventory_summary(selected_classes)
