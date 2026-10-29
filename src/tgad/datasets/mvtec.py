"""MVTec AD test-split indexing and official one-shot references."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


from ..exceptions import DataValidationError
from ..records import ImageSample
from ..utils import natural_key


MVTEC_CLASSES: tuple[str, ...] = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class MVTecDatasetIndex:
    """A deterministic index over MVTec AD's original test split."""

    def __init__(
        self,
        root: str | Path,
        classes: Sequence[str] | None = None,
        *,
        strict: bool = True,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.classes = tuple(classes or MVTEC_CLASSES)
        self.strict = strict
        if not self.root.is_dir():
            raise FileNotFoundError(f"MVTec root not found: {self.root}")
        unknown = sorted(set(self.classes) - set(MVTEC_CLASSES))
        if unknown:
            raise DataValidationError(f"Unknown MVTec classes: {', '.join(unknown)}")
        self._samples_by_class = {class_name: self._scan_class(class_name) for class_name in self.classes}

    @staticmethod
    def _images(folder: Path) -> list[Path]:
        return sorted(
            (path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
            key=lambda path: natural_key(path.name),
        )

    def _find_mask(self, class_name: str, defect_type: str, image_path: Path) -> Path:
        folder = self.root / class_name / "ground_truth" / defect_type
        expected = folder / f"{image_path.stem}_mask.png"
        if expected.is_file():
            return expected
        candidates = sorted(folder.glob(f"{image_path.stem}_mask.*"))
        if len(candidates) == 1:
            return candidates[0]
        raise DataValidationError(
            f"Ground-truth mask not found unambiguously for {image_path}; expected {expected}"
        )

    def _scan_class(self, class_name: str) -> list[ImageSample]:
        test_dir = self.root / class_name / "test"
        if not test_dir.is_dir():
            if self.strict:
                raise DataValidationError(f"Missing MVTec test directory: {test_dir}")
            return []
        samples: list[ImageSample] = []
        defect_dirs = sorted((path for path in test_dir.iterdir() if path.is_dir()), key=lambda p: natural_key(p.name))
        for defect_dir in defect_dirs:
            defect_type = defect_dir.name
            is_anomalous = defect_type != "good"
            for image_path in self._images(defect_dir):
                mask_path = self._find_mask(class_name, defect_type, image_path) if is_anomalous else None
                samples.append(
                    ImageSample(
                        sample_id=f"{class_name}/{defect_type}/{image_path.name}",
                        image_path=image_path,
                        mask_path=mask_path,
                        is_anomalous=is_anomalous,
                        class_name=class_name,
                        defect_type=defect_type,
                    )
                )
        if self.strict and not samples:
            raise DataValidationError(f"No test images found for MVTec class {class_name}")
        return samples

    def samples(self, class_name: str | None = None) -> list[ImageSample]:
        if class_name is not None:
            return list(self._samples_by_class[class_name])
        return [sample for name in self.classes for sample in self._samples_by_class[name]]

    def defective_samples(self, class_name: str) -> list[ImageSample]:
        return [sample for sample in self._samples_by_class[class_name] if sample.is_anomalous]

    def normal_test_samples(self, class_name: str) -> list[ImageSample]:
        return [sample for sample in self._samples_by_class[class_name] if not sample.is_anomalous]

    def official_reference_images(
        self,
        class_name: str,
        *,
        round_index: int = 3,
        k_shot: int = 1,
    ) -> list[Path]:
        """Return the same filename sequence used by official ``test_mvtec.py``.

        For round ``r``, the official script uses ``r*4`` through ``r*4+3`` and
        then truncates to ``k_shot``. The benchmark's 1-shot control therefore
        defaults to ``train/good/012.png`` (round 3).
        """

        if round_index < 0:
            raise ValueError("round_index must be non-negative")
        if not 1 <= k_shot <= 4:
            raise ValueError("k_shot must be between 1 and 4 for the official policy")
        train_good = self.root / class_name / "train" / "good"
        references = [train_good / f"{round_index * 4 + offset:03d}.png" for offset in range(4)]
        selected = references[:k_shot]
        missing = [path for path in selected if not path.is_file()]
        if missing:
            raise DataValidationError(
                "Official one-shot reference file(s) are missing: " + ", ".join(str(path) for path in missing)
            )
        return selected

    def train_good_images(self, class_name: str) -> list[Path]:
        """All defect-free training images of a class, in natural filename order."""

        train_good = self.root / class_name / "train" / "good"
        if not train_good.is_dir():
            raise DataValidationError(f"Missing MVTec training directory: {train_good}")
        images = self._images(train_good)
        if not images:
            raise DataValidationError(f"No training images found in {train_good}")
        return images

    def summary(self) -> list[dict[str, int | str]]:
        rows: list[dict[str, int | str]] = []
        for class_name in self.classes:
            samples = self._samples_by_class[class_name]
            rows.append(
                {
                    "class_name": class_name,
                    "normal_test": sum(not sample.is_anomalous for sample in samples),
                    "anomalous_test": sum(sample.is_anomalous for sample in samples),
                    "total_test": len(samples),
                }
            )
        return rows
