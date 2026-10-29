"""Dataset readers used by the benchmark scenarios."""

from .apd_coco import APDCocoDataset, APDIndex
from .mvtec import MVTEC_CLASSES, MVTecDatasetIndex
from .mvtec_parts import MVTecPartTags

__all__ = [
    "APDCocoDataset",
    "APDIndex",
    "MVTEC_CLASSES",
    "MVTecDatasetIndex",
    "MVTecPartTags",
]
