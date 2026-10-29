# TGAD: A Structured Benchmark for Text-Guided Anomaly Detection

Official code for **A Structured Benchmark for Text-Guided Anomaly Detection: When Language Stops Conditioning the Decision** (Samele, Lomurno, Jovanovic, Shivakumar Manohar, Crivellaro, Matteucci — BMVC 2026).

TGAD measures whether the textual input of a multimodal anomaly detector actually conditions its decision. It has three scenarios, and in each one language carries a larger share of the inspection decision:

| Scenario | Data | What changes | AnomalyGPT | AA-CLIP | LogSAD |
|---|---|---|---|---|---|
| 1. Controlled text sensitivity | MVTec AD | Prompt variants on fixed images (Appendix C) | 0-shot (5 prompts), 1-shot control (2 prompts) | 0-shot (3 keywords) | 1-shot and full-shot (3 keywords) |
| 2. Per-part anomaly detection | Component-tagged MVTec AD extension | Instruction to inspect only a tagged component; EV1 vs. EV2 (Appendix A, D) | 0-shot | 0-shot | full-shot, released and with FMS |
| 3. Industrial robustness | Assembled Panel Dataset (APD) | Standard AD and Per-Error AD (14 categories), Direct Resize and Pad to Square (Appendix B, E) | 0-shot and 1-shot Standard AD (4 prompts); 0-shot Per-Error AD | 0-shot Standard AD (3 keywords) and Per-Error AD | 1-shot and full-shot Standard AD (3 keywords); full-shot Per-Error AD, released and with FMS |

**Release scope.** This release contains the AnomalyGPT, AA-CLIP, and LogSAD pipelines for all three scenarios. Select the model with `--model anomalygpt` (default), `--model aaclip`, `--model logsad` (released LogSAD), or `--model logsad_fms` (LogSAD with the foreground-masking extension, FMS).

## How prompts are used

Every prompt is read verbatim from the appendix prompt files in `src/tgad/resources/` and passed **unchanged** to the model. The benchmark code does not extract nouns, rewrite text, or append text to any prompt. Each model has its own files, matching its textual interface: `scenario{1,2,3}_prompts.yaml` for AnomalyGPT, `aaclip_scenario{1,2,3}_prompts.yaml` for AA-CLIP, `logsad_scenario{1,2,3}_prompts.yaml` for both LogSAD variants.

**AnomalyGPT** receives the prompt as its `prompt` input. The released model, left to itself, uses only one class word found in the prompt (or `object`) to build its zero-shot anomaly map, through fixed normal/abnormal templates. With the default `text_conditioning: prompt` in the model config, the adapter instead uses the ImageBind embedding of the full prompt as the normal text embedding, and keeps the upstream abnormal templates. `text_conditioning: upstream` restores the released behavior. 1-shot maps do not depend on text in either mode.

**AA-CLIP** receives a keyword (a class name, a description, `object`, or a component name). The keyword is inserted into the normal (`a {}`, `a clean {}`) and abnormal (`a damaged {}`, `a {} with defect`) states and the templates `{}.`, `a photo of {}.`, `this is {}.` of Appendix C.3. The official code encodes these sentences with the adapted text encoder into a normal and an abnormal text anchor. The anomaly map and the image-level score are both computed against these anchors.

**LogSAD** receives foreground and background keyword sets (Appendix C.2, D.2, E.2), passed as a JSON string such as `{"foreground": ["bottle"], "background": ["background"]}`. Upstream LogSAD is written for MVTec LOCO only, so the benchmark runs its structural (PatchCore-style) pathway, built from the pinned upstream components (see [docs/PROTOCOL.md](docs/PROTOCOL.md)). In the released pipeline the keywords never reach the anomaly map, so `--model logsad` is invariant to them by construction. With `--model logsad_fms`, in any scenario, they select the per-patch foreground mask of Equation (2) applied to the map.

Each prediction record stores the exact prompt that was sent.

## Image-level score

AnomalyGPT's and LogSAD's image score is the maximum of the anomaly map, as in their released test code. AA-CLIP's image score follows its official `test.py`: `0.5 * minmax(max of the anomaly map) + 0.5 * minmax(detection score)`, where the detection score comes from the pooled detection token and both min-max normalizations run over the evaluation set. An AA-CLIP image score therefore depends on its set: the same image can score differently under EV1 and EV2, or in two APD categories. `predictions.jsonl` records both the raw `detection_score` and the fused `image_score` of each set.

## Repository layout

```text
configs/                    Model config templates
scripts/                    Upstream checkouts, APD download, and all-scenario launcher
src/tgad/                   Benchmark package and CLI
src/tgad/resources/         Appendix prompt files, per-part MVTec tags, Table 6 cardinalities
models/                     Pinned upstream AnomalyGPT, AA-CLIP, and LogSAD revisions
docs/                       Setup, protocol, and dataset layouts
```

## Quick start

Create the environment and install this package:

```bash
conda env create -f environment.yml
conda activate tgad
pip install -e .
```

Fetch the official AnomalyGPT source (pinned to `f21c51b`) and point the model config at the checkpoints:

```bash
bash scripts/fetch_anomalygpt.sh
cp configs/anomalygpt.example.yaml configs/anomalygpt.local.yaml
# Edit configs/anomalygpt.local.yaml with your checkpoint paths.
tgad doctor --model-config configs/anomalygpt.local.yaml
```

Download the APD (Scenario 3) into `data/apd/`. It is a gated Hugging Face dataset under a research-only license: accept its terms on [the dataset page](https://huggingface.co/datasets/StefanoSamele-PoliMi/assembled-panel-dataset) and log in with `hf auth login` first. MVTec AD is obtained separately from MVTec.

```bash
bash scripts/download_apd.sh
```

Validate the annotations against your dataset copies:

```bash
tgad validate --mvtec-root /data/mvtec_anomaly_detection
tgad validate --apd-root data --coco-annotations data/apd/assembled_panel_annotations.json
```

Run every experiment:

```bash
bash scripts/run_all.sh \
  /data/mvtec_anomaly_detection \
  data \
  data/apd/assembled_panel_annotations.json \
  configs/anomalygpt.local.yaml \
  outputs
```

or one scenario at a time:

```bash
tgad scenario1 --mvtec-root /data/mvtec_anomaly_detection \
  --model-config configs/anomalygpt.local.yaml

tgad scenario2 --mvtec-root /data/mvtec_anomaly_detection \
  --model-config configs/anomalygpt.local.yaml

tgad scenario3 --apd-root data \
  --coco-annotations data/apd/assembled_panel_annotations.json \
  --model-config configs/anomalygpt.local.yaml
```

`--backend mock` runs the pipeline without a model. Use it only for dry runs; its numbers are not results.

### AA-CLIP

Fetch the official AA-CLIP source (pinned to `53db195`), prepare the OpenAI ViT-L-14-336 weights and the VisA-trained adapters (see [docs/SETUP.md](docs/SETUP.md)), and fill in the model config:

```bash
bash scripts/fetch_aaclip.sh
cp configs/aaclip.example.yaml configs/aaclip.local.yaml
# Edit configs/aaclip.local.yaml with your checkpoint paths.
tgad doctor --model aaclip --model-config configs/aaclip.local.yaml
```

Then add `--model aaclip` to any scenario command, or set `MODEL=aaclip` for `run_all.sh`:

```bash
tgad scenario1 --model aaclip --mvtec-root /data/mvtec_anomaly_detection \
  --model-config configs/aaclip.local.yaml

MODEL=aaclip bash scripts/run_all.sh \
  /data/mvtec_anomaly_detection \
  data \
  data/apd/assembled_panel_annotations.json \
  configs/aaclip.local.yaml
```

AA-CLIP results go to `outputs/aaclip/<scenario>` by default, so they never overwrite AnomalyGPT's. AA-CLIP is evaluated zero-shot only. With `--model aaclip` the shot options default to `0`, and 1-shot requests are rejected.

### LogSAD

LogSAD runs in its own environment, because its vendored anomalib code needs torch 2 while AnomalyGPT pins torch 1.13:

```bash
conda env create -f environment-logsad.yml
conda activate tgad-logsad
bash scripts/fetch_logsad.sh          # official LogSAD, pinned to 06aed1a
cp configs/logsad.example.yaml configs/logsad.local.yaml
tgad doctor --model logsad_fms --model-config configs/logsad.local.yaml
```

The DataComp CLIP weights are downloaded from the Hugging Face Hub on the first run. Then:

```bash
MODEL=logsad bash scripts/run_all.sh \
  /data/mvtec_anomaly_detection data \
  data/apd/assembled_panel_annotations.json \
  configs/logsad.local.yaml

tgad scenario2 --model logsad_fms --mvtec-root /data/mvtec_anomaly_detection \
  --model-config configs/logsad.local.yaml
tgad scenario3 --model logsad_fms --modes per_error --apd-root data \
  --coco-annotations data/apd/assembled_panel_annotations.json \
  --model-config configs/logsad.local.yaml
```

LogSAD needs defect-free references, so it runs 1-shot and full-shot (`--shots 1 full`), never 0-shot.
- **References.** 1-shot uses the first training image (`train/good/000.png`), as LogSAD's evaluation script does. Full-shot uses every `train/good` image. Scenario 2 and Per-Error AD default to full-shot (`--shots`, `--per-error-shots`).
- **APD.** The APD has no training split, so its full-shot memory bank defaults to the 1-shot reference alone, as in the paper runs (`--full-shot-references` overrides it). LogSAD's full-shot then differs from its 1-shot only by the coreset subsampling of the memory bank.
- **Output.** Results go to `outputs/logsad/` and `outputs/logsad_fms/`.

## Scenario 2 evaluation settings

For an instructed tag set `Ts`: `A(Ts)` contains the images whose tag set equals `Ts`; `N1` contains the defect-free test images; `N2` contains defective images whose tags are disjoint from `Ts`. Images that overlap `Ts` only partially are excluded.

- **EV1** scores `A(Ts)` against `N1`.
- **EV2** scores `A(Ts)` against `N1 ∪ N2`.

All three metrics (I-AUROC, P-AUROC, AUPRO) are computed separately for EV1 and EV2, on the image set of that setting. `A(Ts)` keeps its ground-truth masks, and every image admitted as normal contributes an all-zero target mask. Under EV2, a model that highlights a defect outside the instructed component is therefore penalized in localization as well as in detection. `--localization-scope in_scope_only` is an optional diagnostic that computes localization on `A(Ts)` alone; with that option EV1 and EV2 localization values are identical.

## Assembled Panel categories

In this repository, the APD annotations name the 14 anomaly categories by error code `err_1` … `err_14`. Category **A_k is error code `err_k`** throughout the benchmark: in the annotation files, the dataset loader, the Per-Error prompts.

| ID | Code | Category | Type | Visual signature |
|---|---|---|---|---|
| A1 | `err_1` | swapped power-cable colors | 1 | color permutation at left/right power ports |
| A2 | `err_2` | swapped large cables in EV ports | 1 | green/blue large cables exchanged between left/right EV ports |
| A3 | `err_3` | swapped small cables in EV ports | 1 | orange/white small cables exchanged between left/right EV ports |
| A4 | `err_4` | swapped same-color IO power cables | 1 | upper/lower same-color power cables exchanged at terminal block |
| A5 | `err_5` | misplaced label | 2 | identification label present but in non-standard position |
| A6 | `err_6` | missing label | 2 | absent identification sticker on PCB black box |
| A7 | `err_7` | missing rubber ring near mounting screw | 2 | absent ring around panel-mounting screw head |
| A8 | `err_8` | missing red cap | 2 | absent red protective cap near EV port |
| A9 | `err_9` | missing power-connection wire(s) | 1 | absent wire at top/bottom power port |
| A10 | `err_10` | missing wire(s) on EV port connections | 1 | absent wire at left/right EV port connection |
| A11 | `err_11` | missing screw(s) | 2 | absent screw at panel or PCB mounting position |
| A12 | `err_12` | incorrect wire routing through current sensor | 1 | wire count through right current sensor differs from two |
| A13 | `err_13` | scratched or broken PCB | 0 | surface scratch, fracture, or missing component on PCB |
| A14 | `err_14` | unplugged wire(s) on current sensor | 1 | disconnected wire at the right current sensor |

The annotation file for Scenario 3 is `data/apd/assembled_panel_annotations.json` (the default of `--coco-annotations`). It contains 89 normal and 53 anomalous images, 10 of which carry more than one category; `tgad validate` checks these counts.

The APD acquisition protocol (camera, enclosure, and the five tested setups; Setup 5 is the one used) is in [`ImageAcquisitionProtocolAssembledPanel.pdf`](https://huggingface.co/datasets/StefanoSamele-PoliMi/assembled-panel-dataset/blob/main/ImageAcquisitionProtocolAssembledPanel.pdf), and representative images of each error code are in [`error_code_for_anomalies.pdf`](https://huggingface.co/datasets/StefanoSamele-PoliMi/assembled-panel-dataset/blob/main/error_code_for_anomalies.pdf), both shipped with the dataset. Per-category binary masks are rasterized from the COCO polygons at load time.

## Outputs

Each scenario output directory is self-contained:

- `summary.csv`: aggregate metrics in percentage points. Scenario 1 rows are class-macro prompt/regime summaries; Scenario 2 rows identify the setting (EV1/EV2), part count, localization scope, and aggregation; Scenario 3 rows identify mode, preprocessing, shot regime, and prompt variant.
- `per_class.csv` (Scenario 1), `per_subclass.csv` (Scenario 2), or `per_configuration.csv` (Scenario 3): unit-level metrics with sample and region counts and protocol-specific fields (exact positives, disjoint negatives, excluded overlaps, category-positive counts).
- `predictions.jsonl`: one record per model request, with the image identity and protocol role, the exact prompt (AnomalyGPT), keyword (AA-CLIP), or keyword set (LogSAD, as JSON) sent to the model, the normal references (full-shot sets summarized by count, directory, first and last file), `image_score` (the score used for I-AUROC), `detection_score` (AA-CLIP's raw detection-token score, `null` otherwise), the cache key, path, and hit flag, and the generated response when `inference_path: generate` is used. Scenario 2 evaluates the same image repeatedly under different subclass instructions by design.
- `run_metadata.json`: model, classes or categories, transform rules, prompt handling, image-score rule, prompt sampling method and seed, localization scope, aggregation definitions, reference selection, backend fingerprint, and AUPRO settings. Keep it with every reported run.

Anomaly maps (and AA-CLIP detection scores) are cached as compressed NPZ files, keyed by backend fingerprint, model resource identities, image identity, prompt, normal references, and protocol context. APD square inputs are cached separately by source identity, strategy, and size. Delete the caches after changing upstream code outside the recorded fingerprint.

## Citation

```bibtex
@inproceedings{samele2026tgad,
  title     = {A Structured Benchmark for Text-Guided Anomaly Detection: When Language Stops Conditioning the Decision},
  author    = {Samele, Stefano and Lomurno, Eugenio and Jovanovic, Teodora and Shivakumar Manohar, Sanjay and Crivellaro, Alberto and Matteucci, Matteo},
  booktitle = {British Machine Vision Conference (BMVC)},
  year      = {2026}
}
```

Please also cite AnomalyGPT (Gu et al., AAAI 2024), AA-CLIP (Ma et al., CVPR 2025), or LogSAD (Zhang et al., CVPR 2025) when using the corresponding pipeline.

## Licensing

The code in this repository is MIT licensed. The official AnomalyGPT, AA-CLIP, and LogSAD sources are fetched separately and are governed by their upstream licenses (CC BY-NC-SA 4.0 for AnomalyGPT, Apache 2.0 for AA-CLIP; LogSAD declares none, and its bundled components keep theirs). Model checkpoints and datasets have their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The APD is released under the APD Research-Only License (non-commercial research and education only, no redistribution); see its [dataset page](https://huggingface.co/datasets/StefanoSamele-PoliMi/assembled-panel-dataset).
