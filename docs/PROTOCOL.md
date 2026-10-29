# Benchmark protocol

## Shared scoring and metrics

Each model returns one continuous anomaly map per image. The image-level score follows each model's official test code:

- AnomalyGPT: the maximum map value, matching the released MVTec test script;
- AA-CLIP: `0.5 * minmax(map maximum) + 0.5 * minmax(detection score)`, matching the official `test.py` (`metrics_eval`, Industrial domain). The detection score is `(det_token . abnormal_anchor + 1) / 2` from the pooled detection token. Both min-max normalizations run over the evaluation set: a Scenario 1 class, a Scenario 2 subclass under one setting, or a Scenario 3 Standard AD configuration or Per-Error category. The same image can therefore score differently in different sets.

LogSAD's image score is the maximum of its anomaly map. The map is LogSAD's structural (PatchCore-style) score, computed as follows:

- the image is resized to 448 (antialiased), down to 224, zero-padded to 448, and CLIP-normalized;
- the DataComp CLIP ViT-L/14 produces patch tokens at layers 6, 12, 18, and 24, upsampled from 32x32 to 64x64;
- each layer's anomaly is `1 - max cosine similarity` to the memory bank, and the four layers are averaged;
- the 64x64 map is upsampled to 448, and its central 224x224 content is the returned map.

The AA-CLIP anomaly map is the sum over the four patch-token levels (layers 6, 12, 18, 24) of the blurred patch-to-anchor similarity map produced by upstream `calculate_similarity_map`. P-AUROC and AUPRO use these raw maps; the min-max normalization in the official test code is monotone over the set and does not change them.

Metrics are reported as percentages:

- I-AUROC: AUROC over image labels and image-level scores;
- P-AUROC: AUROC over flattened binary masks and anomaly maps;
- AUPRO@0.30: area under the per-region-overlap/FPR curve up to normal-pixel FPR 0.30, normalized by 0.30, discretized over 200 thresholds. Regions use 8-connectivity and contribute equally.

MVTec masks follow the official AnomalyGPT test transform: bilinear resize to 224x224 and threshold at 0.1. APD masks use the same geometry as the image and nearest-neighbor interpolation. The ground truth is the same for every model.

## Scenario 1: controlled text sensitivity

The original MVTec train/test split and all 15 classes are fixed while prompts vary. The exact Appendix C.1 AnomalyGPT text is stored in `src/tgad/resources/scenario1_prompts.yaml`.

Default matrix:

- 0-shot: `original`, `new_description`, `no_anomaly_description`, `no_normality_description`, `minimal`;
- 1-shot: `original`, `minimal`.

Metrics are computed per class, then averaged equally across the 15 classes. One-shot references follow the official round policy documented in `docs/DATASETS.md`.

LogSAD is evaluated 1-shot and full-shot with the three Appendix C.2 foreground keyword forms (`resources/logsad_scenario1_prompts.yaml`; background keyword `background`). Its 1-shot reference is `train/good/000.png` (round 0: LogSAD's evaluation script takes the first training images); full-shot uses every `train/good` image of the class.

AA-CLIP is evaluated 0-shot with the three Appendix C.3 keyword forms (`resources/aaclip_scenario1_prompts.yaml`): `class_name` (Table 13 Simple), `detailed` (Very Detailed), and `generic` (`object`).

## Scenario 2: per-part MVTec

For an instructed tag set `Ts` and an image tag set `Ti`:

```text
A(Ts)  = images with Ti == Ts
N1(Ts) = defect-free test images
N2(Ts) = defective images with Ti disjoint from Ts
```

EV1 compares `A(Ts)` with `N1(Ts)`. EV2 compares `A(Ts)` with `N1(Ts) union N2(Ts)`. Images that overlap the instruction but do not exactly equal it are excluded.

Each subclass has three semantically equivalent prompts from Appendix D.1, used exactly as written (no suffix is appended). The default `stable_hash` sampler makes a uniform deterministic choice from `(seed, class, subclass, sample_id)` with seed 0 and shares that choice across EV1 and EV2, so both settings score the same anomaly map for a given image. `global_sequence` and `per_setting_sequence` draw from a seeded `random.Random` stream instead.

AA-CLIP receives one component keyword per subclass (Appendix D.3, `resources/aaclip_scenario2_prompts.yaml`); multi-part subclasses join components with `AND`. With a single keyword the prompt sampler is moot. EV1 and EV2 still share the anomaly map of an image, but AA-CLIP's image score is normalized over each setting's image set (see above).

LogSAD is evaluated full-shot (`--shots` changes the regime), released (`--model logsad`) and with the foreground-masking extension (`--model logsad_fms`). Each subclass has one foreground/background keyword set from Appendix D.2 (`resources/logsad_scenario2_prompts.yaml`).

Localization (P-AUROC and AUPRO) is computed separately for EV1 and EV2 by default (`--localization-scope evaluation_set`). It uses the same images as I-AUROC for that setting: images in `A(Ts)` keep their MVTec ground-truth masks, and every image admitted as normal (`N1` under EV1, `N1 ∪ N2` under EV2) contributes an all-zero target mask. Under EV2, a response to a defect outside the instructed component therefore counts as a false positive in localization as well as in detection.

`--localization-scope in_scope_only` is an optional diagnostic that computes localization on `A(Ts)` alone. That set does not depend on the setting, so EV1 and EV2 localization values are identical under this option.

`summary.csv` reports two aggregations for each setting and part count: `macro_subclass` (every subclass of that cardinality weighted equally) and `macro_class` (subclasses averaged within each class, then classes averaged).

## Scenario 3: APD industrial robustness

All model inputs are 224x224. Two spatial strategies are evaluated:

- `direct_resize`: bicubic stretch to 224x224;
- `pad_to_square`: preserve aspect ratio, resize into 224x224, center, and fill unused pixels with zeros.

### Standard AD

Every normal and anomalous image is evaluated together with one shared prompt. Four Appendix E.1 prompts are provided: `minimal`, `simple`, `detailed`, and `very_detailed`.

The default sweep mirrors Appendix Tables 10 and 11 and is stored as `metadata.standard_variants` in each model's Scenario 3 prompt file:

- AnomalyGPT, Direct Resize, 0-shot: all four prompts;
- AnomalyGPT, Direct Resize, 1-shot: very detailed and minimal;
- AnomalyGPT, Pad to Square, 0-shot: very detailed and minimal;
- AnomalyGPT, Pad to Square, 1-shot: very detailed and minimal;
- AA-CLIP, both strategies, 0-shot: the three Appendix E.3 keywords `minimal`, `simple`, and `very_detailed`;
- LogSAD, both strategies, 1-shot and full-shot: the three Appendix E.2 foreground keywords, each against the two E.2 background keywords `background` and `industrial environment`.

The APD has no training split. The 1-shot reference is the first normal image (see below), and the full-shot memory bank defaults to that same image (`--full-shot-references` sets another list), as in the paper runs. LogSAD's two regimes therefore differ only in their memory bank: every patch feature (1-shot) or a 25% coreset (full-shot).

### Per-Error AD

For each A1-A14 category, its positive images are compared with all 89 normal images. Metrics are computed separately for each category and then averaged equally across the 14 categories. Category-specific masks are used. The regime is the model's: 0-shot for AnomalyGPT and AA-CLIP, full-shot for LogSAD. AA-CLIP uses the Appendix E.3 per-category keywords, and LogSAD the Appendix E.2 per-category foreground/background keywords, both keyed by the defect they describe.

The APD 1-shot reference is set with `--one-shot-reference`. Without it, the first normal sample by lexicographic sample ID is used. The selected file is recorded in `run_metadata.json`.

## Prompt handling

Every prompt is passed to AnomalyGPT unchanged, as the `prompt` field of the official `OpenLLAMAPEFTModel` inference call. The benchmark does not extract, rewrite, or append text. The released model would condition the zero-shot map only on the first word from its `CLASS_NAMES` list found in the prompt (falling back to `object`), filled into fixed normal and abnormal templates. Under the default `text_conditioning: prompt`, the adapter replaces the normal side with the ImageBind embedding of the whole prompt and keeps the upstream abnormal templates for that class word. Prompts over ImageBind's 77-token text context are truncated, keeping the end token. `text_conditioning: upstream` runs the released code unmodified. In the 1-shot regime, the upstream model scores patches by visual similarity to the reference image, and that score replaces the zero-shot text-conditioned map.

Every AA-CLIP keyword is passed unchanged to upstream `get_adapted_single_class_text_embedding`, which fills it into the normal states (`a {}`, `a clean {}`) and abnormal states (`a damaged {}`, `a {} with defect`), wraps each in the templates `{}.`, `a photo of {}.`, and `this is {}.`, encodes the ensemble with the adapted text encoder, and averages it into one normal and one abnormal anchor. These are the Appendix C.3 states and templates (`AACLIP_PROMPT_STATES`, `AACLIP_PROMPT_TEMPLATES` in `backend.py`). The adapter swaps them into upstream `forward_utils` at load time in place of upstream's own ensemble in `dataset/constants.py`, so the checkout itself stays unmodified. The adapter registers the keyword in upstream's class-name tables so that this function can be called with it. AA-CLIP takes no reference images and is evaluated zero-shot only.

## LogSAD

Upstream LogSAD is written for MVTec LOCO: its model class hard-codes per-category query words, cluster counts, and score statistics for those five categories, and its logical branch exists only for them. On MVTec AD and the APD, the paper's experiments use LogSAD's structural pathway, which `src/tgad/logsad.py` reproduces from the pinned upstream components (`open_clip_local`, anomalib's `KCenterGreedy`) at the upstream settings. The keywords do not reach this pathway, so released LogSAD is invariant to them by construction.

Memory banks follow upstream's two protocols:

- **1-shot**: every 64x64 patch feature of the reference image (`model_ensemble_few_shot`);
- **full-shot**: the references are taken in chunks of four, a 25% `KCenterGreedy` coreset is kept per chunk, and the coresets are concatenated (`compute_coreset.py`). Like upstream, the last `n % 4` images are left out.

Coresets are seeded (`seed: 42`, as upstream) before each memory bank is built.

### Foreground-masking extension (FMS)

`--model logsad_fms` multiplies the structural map by the foreground mask of Equations (2)-(3), built from the keyword set:

1. each keyword is expanded into upstream's template ensemble (the 80 `openai_imagenet_template` sentences) and encoded with LogSAD's CLIP text encoder, using upstream `prompt_ensemble.encode_obj_text`: the template embeddings are L2-normalized, averaged, and normalized again;
2. the foreground keywords together form the focus embedding `t_focus` (their template embeddings are averaged into one, as `encode_obj_text` does for a list of words); each background keyword is one contrast embedding `t_contrast_k`;
3. for each patch `j` of the 64x64 grid, `p_j` is its projected CLIP patch embedding (L2-normalized), and the patch is foreground when `cos(p_j, t_focus) > max_k cos(p_j, t_contrast_k)`;
4. the mask multiplies the 64x64 structural map, which is then upsampled and cropped as without FMS.

The mask depends only on the test image and the keywords; it applies in every scenario. With `--model logsad_fms` the Scenario 1 and Scenario 3 Standard AD keywords therefore reach the anomaly map (focus: the Table 12 or E.2 foreground keyword; contrast: `background`, and on the APD also `industrial environment`). The paper's Scenario 1 and Standard AD numbers use the released pipeline (`--model logsad`).
