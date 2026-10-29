# Dataset layouts

## MVTec AD: Scenarios 1 and 2

Point `--mvtec-root` at the directory containing the 15 class directories:

```text
mvtec_anomaly_detection/
  cable/
    train/good/000.png ...
    test/good/*.png
    test/bent_wire/*.png
    ground_truth/bent_wire/*_mask.png
  ...
```

Scenario 1 uses all 15 classes and the original test split. Its 1-shot control follows the filename policy in the official `test_mvtec.py`: at round `r`, candidate references are `r*4` through `r*4+3`; the default round 3 and `k=1` selects `train/good/012.png` for every class.

Scenario 2 uses nine object classes: cable, capsule, hazelnut, metal nut, pill, screw, toothbrush, transistor, and zipper. The per-part annotation is released in:

- `src/tgad/resources/mvtec_test_part_tags_raw.txt` (the file the loader reads)
- `src/tgad/resources/mvtec_test_part_tags.json` (the same tags in normalized JSON form)

The loader normalizes the one notation discrepancy `metalnut1/2` to the Appendix spelling `metal_nut1/2`. It validates all 813 annotated defective images, 42 subclasses, and Appendix Table 6 cardinalities before a full run.

## APD: Scenario 3

The APD is distributed as a gated Hugging Face dataset, [StefanoSamele-PoliMi/assembled-panel-dataset](https://huggingface.co/datasets/StefanoSamele-PoliMi/assembled-panel-dataset), under the APD Research-Only License (non-commercial research and education, no redistribution, citation required). To download it:

1. Open the dataset page, log in, and accept the terms. Access is granted immediately.
2. Authenticate locally with `hf auth login`, or export `HF_TOKEN`.
3. Run `bash scripts/download_apd.sh`. It needs `huggingface_hub` (`pip install huggingface_hub`).

The script downloads into `data/apd/` (pass another directory as its first argument), checks the 89/53 image counts, and prints the matching `validate` command. Do not commit the downloaded images; the license forbids redistribution.

By default the APD is read from `data/apd/` in the repository root:

```text
data/
  apd/
    normal/
      *.png                              (89 defect-free images)
    anomalous/
      *.png                              (53 anomalous images)
    assembled_panel_annotations.json     (COCO annotations of the anomalous images)
    ImageAcquisitionProtocolAssembledPanel.pdf, error_code_for_anomalies.pdf, LICENSE, README.md
```

`--apd-root` is the parent of the `apd/` directory (default `data`); `--dataset-dir-name` overrides `apd`. The defaults are relative paths, so run the commands from the repository root or pass absolute paths.

The COCO JSON is passed independently with `--coco-annotations` (default `data/apd/assembled_panel_annotations.json`). Matching is by image basename because Label Studio exports may contain absolute paths. The implementation supports polygon segmentations and, when `pycocotools` is installed, COCO RLE. It rejects duplicate basenames because they make matching ambiguous.

The COCO categories are named `err_1` … `err_14`; category A_k is `err_k`. Standard AD uses the union of all annotations in an anomalous image. Per-Error AD uses only the target category's mask; other defects in the same positive image count as background for that category-specific localization evaluation.

The APD loader is `src/tgad/datasets/apd_coco.py`. Besides the image and the union mask of each sample, it validates the annotation counts and builds category-specific masks for Per-Error AD.
