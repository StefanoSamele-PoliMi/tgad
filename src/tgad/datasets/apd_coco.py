"""Assembled Panel Dataset reader for COCO-annotated APD images."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from ..exceptions import DataValidationError
from ..records import ImageSample
from ..utils import natural_key


# Paper category A_k is the APD error code err_k (A1 = err_1, ..., A14 = err_14); the
# meaning of each code is DEFAULT_ANOMALY_DESCRIPTIONS below.
APD_ERROR_TO_A_ID: dict[str, str] = {f"err_{index}": f"A{index}" for index in range(1, 15)}
APD_A_ID_TO_ERROR: dict[str, str] = {value: key for key, value in APD_ERROR_TO_A_ID.items()}
APD_A_IDS: tuple[str, ...] = tuple(f"A{index}" for index in range(1, 15))

# Counts of the released annotation file (data/apd/assembled_panel_annotations.json).
APD_EXPECTED_NORMAL_IMAGES = 89
APD_EXPECTED_ANOMALOUS_IMAGES = 53
APD_EXPECTED_MULTI_DEFECT_IMAGES = 10
APD_EXPECTED_CATEGORY_COUNTS: dict[str, int] = {
    "A1": 5,
    "A2": 4,
    "A3": 6,
    "A4": 1,
    "A5": 5,
    "A6": 5,
    "A7": 5,
    "A8": 7,
    "A9": 5,
    "A10": 7,
    "A11": 5,
    "A12": 6,
    "A13": 1,
    "A14": 2,
}

DEFAULT_ANOMALY_DESCRIPTIONS: Dict[str, str] = {
    "err_1": "Swapped colors in power cables",
    "err_2": "Swapped large cables in EV port",
    "err_3": "Swapped small cables in EV port",
    "err_4": "Swapped same color IO power cables",
    "err_5": "Misplaced label",
    "err_6": "Missing label",
    "err_7": "Missing rubber ring",
    "err_8": "Missing red cap",
    "err_9": "Missing wire(s) on power connections",
    "err_10": "Missing wire(s) on EV port connections",
    "err_11": "Missing screw(s)",
    "err_12": "Wires passing through current sensor",
    "err_13": "Scratched or broken PCB",
    "err_14": "Current sensor unplugged cable",
}

IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)


def _normalize_extensions(values: Sequence[str]) -> frozenset[str]:
    result = {
        value.lower() if str(value).startswith(".") else f".{str(value).lower()}"
        for value in values
    }
    if not result:
        raise ValueError("At least one image extension is required")
    return frozenset(result)


def canonical_apd_error(label: str) -> str:
    """Normalize a COCO category name to ``err_N``.

    The APD COCO export uses ``err_1`` through ``err_14``. Paper identifiers
    ``A1`` through ``A14`` are accepted as a convenience.
    """

    value = str(label).strip()
    match = re.fullmatch(r"err[_\- ]?(\d{1,2})", value, flags=re.IGNORECASE)
    if match:
        result = f"err_{int(match.group(1))}"
    else:
        match = re.fullmatch(r"A(\d{1,2})", value, flags=re.IGNORECASE)
        if not match:
            raise DataValidationError(
                f"Unsupported APD category name {label!r}; expected err_1..err_14 or A1..A14"
            )
        a_id = f"A{int(match.group(1))}"
        try:
            result = APD_A_ID_TO_ERROR[a_id]
        except KeyError as exc:
            raise DataValidationError(f"Unsupported APD category identifier: {label}") from exc
    if result not in APD_ERROR_TO_A_ID:
        raise DataValidationError(f"Unsupported APD category identifier: {label}")
    return result


class APDIndex:
    """Deterministic APD image/COCO index with combined and category masks."""

    def __init__(
        self,
        root: str | Path,
        coco_annotation_file: str | Path,
        *,
        dataset_dir_name: str = "apd",
        image_extensions: Sequence[str] = IMAGE_EXTENSIONS,
        strict: bool = True,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.dataset_dir = self.root / dataset_dir_name
        self.normal_dir = self.dataset_dir / "normal"
        self.anomalous_dir = self.dataset_dir / "anomalous"
        self.coco_annotation_file = Path(coco_annotation_file).expanduser().resolve()
        self.image_extensions = _normalize_extensions(image_extensions)
        self.strict = strict
        for label, folder in (("normal", self.normal_dir), ("anomalous", self.anomalous_dir)):
            if not folder.is_dir():
                raise FileNotFoundError(f"APD {label} image directory not found: {folder}")
        if not self.coco_annotation_file.is_file():
            raise FileNotFoundError(
                f"APD COCO annotation file not found: {self.coco_annotation_file}"
            )

        with self.coco_annotation_file.open("r", encoding="utf-8") as handle:
            coco = json.load(handle)
        if not isinstance(coco, dict):
            raise DataValidationError("APD COCO root must be a JSON object")
        self._coco = coco

        self.category_id_to_raw_name: dict[int, str] = {}
        self.category_id_to_error: dict[int, str] = {}
        seen_errors: set[str] = set()
        for category in coco.get("categories", []):
            try:
                category_id = int(category["id"])
                raw_name = str(category["name"])
            except (KeyError, TypeError, ValueError) as exc:
                raise DataValidationError(f"Invalid APD COCO category record: {category!r}") from exc
            if category_id in self.category_id_to_raw_name:
                raise DataValidationError(f"Duplicate APD COCO category id: {category_id}")
            error_name = canonical_apd_error(raw_name)
            if error_name in seen_errors:
                raise DataValidationError(
                    f"Multiple APD COCO categories map to {error_name}; mapping is ambiguous"
                )
            seen_errors.add(error_name)
            self.category_id_to_raw_name[category_id] = raw_name
            self.category_id_to_error[category_id] = error_name

        image_info_by_id: dict[int, dict] = {}
        self.coco_by_basename: dict[str, dict] = {}
        for image_info in coco.get("images", []):
            try:
                image_id = int(image_info["id"])
                basename = Path(str(image_info["file_name"])).name
            except (KeyError, TypeError, ValueError) as exc:
                raise DataValidationError(f"Invalid APD COCO image record: {image_info!r}") from exc
            if not basename:
                raise DataValidationError(f"APD COCO image {image_id} has an empty filename")
            if image_id in image_info_by_id:
                raise DataValidationError(f"Duplicate APD COCO image id: {image_id}")
            if basename in self.coco_by_basename:
                raise DataValidationError(
                    f"Duplicate COCO image basename makes matching ambiguous: {basename}"
                )
            image_info_by_id[image_id] = image_info
            self.coco_by_basename[basename] = {
                "image_info": image_info,
                "annotations": [],
            }

        annotation_ids: set[int] = set()
        basename_by_image_id = {
            image_id: Path(str(info["file_name"])).name
            for image_id, info in image_info_by_id.items()
        }
        for annotation in coco.get("annotations", []):
            try:
                annotation_id = int(annotation["id"])
                image_id = int(annotation["image_id"])
                category_id = int(annotation["category_id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise DataValidationError(
                    f"Invalid APD COCO annotation record: {annotation!r}"
                ) from exc
            if annotation_id in annotation_ids:
                raise DataValidationError(f"Duplicate APD COCO annotation id: {annotation_id}")
            annotation_ids.add(annotation_id)
            if image_id not in image_info_by_id:
                raise DataValidationError(
                    f"APD COCO annotation {annotation_id} references unknown image id {image_id}"
                )
            if category_id not in self.category_id_to_error:
                raise DataValidationError(
                    f"APD COCO annotation {annotation_id} references unknown category id {category_id}"
                )
            self.coco_by_basename[basename_by_image_id[image_id]]["annotations"].append(annotation)

        normal_paths = self._find_images(self.normal_dir)
        anomalous_paths = self._find_images(self.anomalous_dir)
        anomalous_basenames = [path.name for path in anomalous_paths]
        duplicate_disk_basenames = sorted(
            name for name, count in Counter(anomalous_basenames).items() if count > 1
        )
        if duplicate_disk_basenames:
            raise DataValidationError(
                "Anomalous disk images contain duplicate basenames, but COCO matching is basename-only: "
                + ", ".join(duplicate_disk_basenames[:10])
            )

        self._samples: list[ImageSample] = []
        for path in normal_paths:
            relative = path.relative_to(self.dataset_dir).as_posix()
            self._samples.append(
                ImageSample(
                    sample_id=relative,
                    image_path=path,
                    is_anomalous=False,
                    labels=(),
                    extra={"coco": None},
                )
            )

        unmatched: list[Path] = []
        empty_annotations: list[Path] = []
        used_coco_basenames: set[str] = set()
        for path in anomalous_paths:
            relative = path.relative_to(self.dataset_dir).as_posix()
            entry = self.coco_by_basename.get(path.name)
            if entry is None:
                unmatched.append(path)
                labels: tuple[str, ...] = ()
            else:
                used_coco_basenames.add(path.name)
                if not entry["annotations"]:
                    empty_annotations.append(path)
                labels = tuple(
                    sorted(
                        {
                            self.category_id_to_error[int(annotation["category_id"])]
                            for annotation in entry["annotations"]
                        },
                        key=lambda value: int(value.split("_")[1]),
                    )
                )
            self._samples.append(
                ImageSample(
                    sample_id=relative,
                    image_path=path,
                    is_anomalous=True,
                    labels=labels,
                    extra={"coco": entry},
                )
            )

        unused_coco = sorted(set(self.coco_by_basename) - used_coco_basenames)
        errors: list[str] = []
        if unmatched:
            preview = ", ".join(path.name for path in unmatched[:10])
            errors.append(
                f"{len(unmatched)} anomalous images are absent from COCO by basename: {preview}"
            )
        if empty_annotations:
            preview = ", ".join(path.name for path in empty_annotations[:10])
            errors.append(
                f"{len(empty_annotations)} anomalous images have no COCO annotations: {preview}"
            )
        if unused_coco:
            errors.append(
                f"{len(unused_coco)} COCO image records do not match anomalous disk files: "
                + ", ".join(unused_coco[:10])
            )
        if errors and strict:
            raise DataValidationError("APD image/COCO mismatch:\n  - " + "\n  - ".join(errors))
        if not self._samples:
            raise DataValidationError(f"No APD images found under {self.dataset_dir}")

    def _find_images(self, folder: Path) -> list[Path]:
        return sorted(
            (
                path
                for path in folder.rglob("*")
                if path.is_file() and path.suffix.lower() in self.image_extensions
            ),
            key=lambda path: natural_key(path.relative_to(folder).as_posix()),
        )

    def samples(self) -> list[ImageSample]:
        return list(self._samples)

    def normal_samples(self) -> list[ImageSample]:
        return [sample for sample in self._samples if not sample.is_anomalous]

    def anomalous_samples(self) -> list[ImageSample]:
        return [sample for sample in self._samples if sample.is_anomalous]

    def per_error_samples(self, a_id_or_error: str) -> list[ImageSample]:
        error = canonical_apd_error(a_id_or_error)
        return [sample for sample in self.anomalous_samples() if error in sample.labels]

    @staticmethod
    def _polygon_sequence(segmentation: object) -> list[list[float]]:
        if not isinstance(segmentation, (list, tuple)):
            return []
        values = list(segmentation)
        if not values:
            return []
        # Be permissive with a single flat polygon, although canonical COCO stores
        # a list of polygons.
        if all(isinstance(value, (int, float)) for value in values):
            return [[float(value) for value in values]]
        polygons: list[list[float]] = []
        for polygon in values:
            if isinstance(polygon, (list, tuple)):
                polygons.append([float(value) for value in polygon])
        return polygons

    def _draw_segmentation(
        self,
        mask_image: Image.Image,
        annotation: dict,
        *,
        scale_x: float,
        scale_y: float,
    ) -> None:
        segmentation = annotation.get("segmentation", [])
        if isinstance(segmentation, dict):
            try:
                from pycocotools import mask as mask_utils  # type: ignore
            except ImportError as exc:
                raise NotImplementedError(
                    "RLE COCO segmentation encountered. Install pycocotools for RLE support."
                ) from exc
            decoded = mask_utils.decode(segmentation)
            if decoded.ndim == 3:
                decoded = np.any(decoded, axis=2)
            rle_image = Image.fromarray(decoded.astype(np.uint8) * 255, mode="L")
            if rle_image.size != mask_image.size:
                rle_image = rle_image.resize(mask_image.size, Image.Resampling.NEAREST)
            mask_image.paste(Image.new("L", mask_image.size, 255), mask=rle_image)
            return

        draw = ImageDraw.Draw(mask_image)
        for polygon in self._polygon_sequence(segmentation):
            if len(polygon) < 6:
                continue
            if len(polygon) % 2:
                raise DataValidationError(
                    f"Odd polygon coordinate count in annotation {annotation.get('id')}"
                )
            points = [
                (polygon[index] * scale_x, polygon[index + 1] * scale_y)
                for index in range(0, len(polygon), 2)
            ]
            draw.polygon(points, outline=255, fill=255)

    @staticmethod
    def empty_mask(sample: ImageSample) -> np.ndarray:
        with Image.open(sample.image_path) as image:
            width, height = image.size
        return np.zeros((height, width), dtype=bool)

    def load_mask(
        self,
        sample: ImageSample,
        *,
        target_error: str | None = None,
    ) -> np.ndarray:
        """Rasterize a combined mask or one target-category mask at source size."""

        with Image.open(sample.image_path) as image:
            width, height = image.size
        if not sample.is_anomalous or sample.extra.get("coco") is None:
            return np.zeros((height, width), dtype=bool)
        target = canonical_apd_error(target_error) if target_error is not None else None
        entry = sample.extra["coco"]
        image_info = entry["image_info"]
        coco_width = float(image_info.get("width", width)) or float(width)
        coco_height = float(image_info.get("height", height)) or float(height)
        scale_x = width / coco_width
        scale_y = height / coco_height
        mask_image = Image.new("L", (width, height), 0)
        for annotation in entry["annotations"]:
            error = self.category_id_to_error[int(annotation["category_id"])]
            if target is not None and error != target:
                continue
            self._draw_segmentation(
                mask_image,
                annotation,
                scale_x=scale_x,
                scale_y=scale_y,
            )
        return np.asarray(mask_image, dtype=np.uint8) > 0

    def validate_protocol_counts(self) -> dict[str, object]:
        normals = len(self.normal_samples())
        anomalies = len(self.anomalous_samples())
        assignments = Counter(
            APD_ERROR_TO_A_ID[label]
            for sample in self.anomalous_samples()
            for label in sample.labels
        )
        multi_defect = sum(len(sample.labels) > 1 for sample in self.anomalous_samples())
        assignment_total = sum(assignments.values())
        errors: list[str] = []
        expected_total = sum(APD_EXPECTED_CATEGORY_COUNTS.values())
        if normals != APD_EXPECTED_NORMAL_IMAGES:
            errors.append(
                f"normal images: expected {APD_EXPECTED_NORMAL_IMAGES}, got {normals}"
            )
        if anomalies != APD_EXPECTED_ANOMALOUS_IMAGES:
            errors.append(
                f"anomalous images: expected {APD_EXPECTED_ANOMALOUS_IMAGES}, got {anomalies}"
            )
        if multi_defect != APD_EXPECTED_MULTI_DEFECT_IMAGES:
            errors.append(
                f"multi-defect images: expected {APD_EXPECTED_MULTI_DEFECT_IMAGES}, "
                f"got {multi_defect}"
            )
        if assignment_total != expected_total:
            errors.append(
                f"category assignments: expected {expected_total}, got {assignment_total}"
            )
        for a_id, expected in APD_EXPECTED_CATEGORY_COUNTS.items():
            if assignments[a_id] != expected:
                errors.append(f"{a_id}: expected {expected}, got {assignments[a_id]}")
        if errors:
            raise DataValidationError("APD protocol-count mismatch:\n  - " + "\n  - ".join(errors))
        return {
            "normal_images": normals,
            "anomalous_images": anomalies,
            "multi_defect_images": multi_defect,
            "category_assignments": assignment_total,
            "category_counts": dict(
                sorted(assignments.items(), key=lambda item: int(item[0][1:]))
            ),
        }


class APDCocoDataset(Dataset):
    """PyTorch dataset compatible with the supplied APD class.

    It preserves the original return fields and adds ``image_path``. The mask is
    the union of all defect annotations; category-specific masks are available
    through :class:`APDIndex` for Scenario 3 Per-Error evaluation.
    """

    DEFAULT_EXTENSIONS = IMAGE_EXTENSIONS
    DEFAULT_ANOMALY_DESCRIPTIONS = DEFAULT_ANOMALY_DESCRIPTIONS

    def __init__(
        self,
        root: str | Path,
        coco_annotation_file: str | Path,
        dataset_dir_name: str = "apd",
        image_extensions: Sequence[str] = DEFAULT_EXTENSIONS,
        joint_transform: Optional[
            Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
        ] = None,
        anomaly_descriptions: Optional[Dict[str, str]] = None,
        strict: bool = True,
    ) -> None:
        self.index = APDIndex(
            root,
            coco_annotation_file,
            dataset_dir_name=dataset_dir_name,
            image_extensions=image_extensions,
            strict=strict,
        )
        self.samples = self.index.samples()
        self.joint_transform = joint_transform
        self.anomaly_descriptions = dict(DEFAULT_ANOMALY_DESCRIPTIONS)
        if anomaly_descriptions:
            self.anomaly_descriptions.update(
                {str(key): str(value) for key, value in anomaly_descriptions.items()}
            )

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _pil_rgb_to_tensor(image: Image.Image) -> torch.Tensor:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            image_tensor = self._pil_rgb_to_tensor(image)
        mask = self.index.load_mask(sample)
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        if self.joint_transform is not None:
            image_tensor, mask_tensor = self.joint_transform(image_tensor, mask_tensor)
        labels = list(sample.labels)
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "is_anomalous": sample.is_anomalous,
            "anomaly_label": labels,
            "anomaly_description": [
                self.anomaly_descriptions.get(label, "unspecified defect") for label in labels
            ],
            "file_name": sample.image_path.name,
            "image_path": str(sample.image_path),
        }
