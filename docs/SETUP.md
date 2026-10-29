# Setup

## 1. Hardware and operating system

The paper reports experiments on one NVIDIA Quadro RTX 6000 with 24 GB. The official released AnomalyGPT stack is CUDA-only. A Linux host with an NVIDIA GPU and enough memory for Vicuna-7B, ImageBind-Huge, the projection modules, and intermediate features is required for faithful inference.

## 2. Python environment

The environment file uses Python 3.10 and the CUDA 11.7 generation of PyTorch used by the upstream project:

```bash
conda env create -f environment.yml
conda activate tgad
pip install -e .
```

The benchmark package declares its own dependencies in `pyproject.toml`. `requirements-anomalygpt-inference.txt` records the older upstream inference stack separately, so the model's compatibility constraints stay explicit. AA-CLIP runs in the same environment; `requirements-aaclip-inference.txt` adds the one package its upstream code needs on top of that stack (pandas). LogSAD needs its own environment (Section 6).

## 3. Official source

```bash
bash scripts/fetch_anomalygpt.sh
```

The script clones `CASIA-LMC-Lab/AnomalyGPT`, checks out the revision in `models/ANOMALYGPT_REVISION`, and refuses to continue if the resolved revision differs. The checkout is ignored by Git/ZIP packaging.

## 4. Official checkpoints

Follow the upstream AnomalyGPT README to prepare:

1. ImageBind-Huge checkpoint;
2. Vicuna-v0 7B base model;
3. PandaGPT 7B delta/checkpoint;
4. AnomalyGPT `train_visa` delta (`code/ckpt/train_visa/pytorch_model.pt`).

The MVTec evaluation uses the VisA-trained AnomalyGPT delta, matching the upstream `test_mvtec.py` cross-dataset protocol. No checkpoint is bundled here.

Copy and edit the template:

```bash
cp configs/anomalygpt.example.yaml configs/anomalygpt.local.yaml
```

Relative paths in this YAML are resolved relative to the YAML file itself. The default `inference_path: extract` calls the official anomaly-map extraction path while skipping autoregressive response generation. Set `inference_path: generate` only when textual responses are required; both paths use the same official anomaly map. `text_conditioning` selects how the prompt conditions the zero-shot map: `prompt` (default) or `upstream` (released class-word scan); see `docs/PROTOCOL.md`.

Validate paths:

```bash
tgad doctor --model-config configs/anomalygpt.local.yaml
```

The first real run performs the CUDA import and model-load check.

## 5. AA-CLIP

### Official source

```bash
bash scripts/fetch_aaclip.sh
```

The script clones `Mwxinnn/AA-CLIP` and checks out the revision in `models/AACLIP_REVISION` (`53db195`). The adapter imports `model.clip`, `model.adapter`, and `forward_utils` from this checkout unmodified.

### Checkpoints

1. OpenAI CLIP ViT-L/14@336px weights, `ViT-L-14-336px.pt` (SHA-256 `3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02`), from `https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt`. Upstream expects them at `models/AA-CLIP/model/ViT-L-14-336px.pt`; any location set as `clip_checkpoint` works.
2. AA-CLIP text and image adapters trained on VisA at 224x224. The paper uses adapters trained with the upstream `train.py` defaults. Run it from a separate clone of the official repository at the same revision. Upstream ships the VisA metadata in `dataset/metadata/VisA/`; point `DATA_PATH["VisA"]` in `dataset/constants.py` at your VisA copy:

   ```bash
   python train.py --dataset VisA --training_mode full_shot --img_size 224 --save_path visa224
   ```

   This runs 5 text epochs (lr 1e-5) and 20 image epochs (lr 5e-4), seed 111, without `--relu`. The benchmark uses `visa224/text_adapter.pth` and the image adapter of the last epoch, `visa224/image_adapter_20.pth`. VisA is disjoint from MVTec AD and APD, so the evaluation stays zero-shot.

The adapters are resolution-dependent: `image_size` in the config must equal the training resolution. Copy and edit the template, then validate the paths:

```bash
cp configs/aaclip.example.yaml configs/aaclip.local.yaml
tgad doctor --model aaclip --model-config configs/aaclip.local.yaml
```

AA-CLIP runs on `cuda`, `cuda:N`, or `cpu`. The first real run performs the model import and load check.

## 6. LogSAD

### Environment

The vendored anomalib code of LogSAD needs torch 2, while AnomalyGPT pins torch 1.13, so LogSAD has its own environment. `requirements-logsad-inference.txt` follows the upstream LogSAD pins for every package the adapter uses (torch 2.2.2+cu121, numpy 1.23.1, open-clip-torch 2.24.0):

```bash
conda env create -f environment-logsad.yml
conda activate tgad-logsad
```

### Official source and checkpoints

```bash
bash scripts/fetch_logsad.sh
```

The script clones `zhang0jhon/LogSAD` and checks out the revision in `models/LOGSAD_REVISION` (`06aed1a`). The adapter imports `open_clip_local`, anomalib's `KCenterGreedy`, and, for the foreground-masking extension (`--model logsad_fms`), the template ensemble of `prompt_ensemble.py` from this checkout. The CLIP model (`hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K`) is downloaded from the Hugging Face Hub on the first run; no other checkpoint is needed. LogSAD is training-free: its memory banks are built from the reference images at run time.

```bash
cp configs/logsad.example.yaml configs/logsad.local.yaml
tgad doctor --model logsad_fms --model-config configs/logsad.local.yaml
```

### Resources

The full-shot memory bank of an MVTec class holds a 25% coreset of every training image's 64x64 patch features: up to about 400k vectors of 4096 values (6.5 GB for hazelnut). It is built on the GPU, one chunk of four images at a time, and kept on the GPU while the class is evaluated. A 24 GB GPU holds it together with CLIP.

## 7. Dataset validation

```bash
tgad validate --mvtec-root /data/mvtec_anomaly_detection

tgad validate \
  --apd-root data \
  --coco-annotations data/apd/assembled_panel_annotations.json
```

The first command checks that all part-tag records resolve to MVTec test images and validates the Appendix subclass inventory. The second verifies the APD counts: 89 normal images, 53 anomalous images, 10 multi-defect images, 64 category assignments, and all per-category counts (category A_k = code `err_k`; see the README table of Assembled Panel categories).

## 8. Determinism

The benchmark sorts all files naturally, uses deterministic prompt selection by default, fingerprints all model resources and input files, and disables cuDNN benchmarking. GPU kernels and the old upstream dependency stack can still introduce platform-level variation. Preserve `run_metadata.json`, the model YAML, the upstream commit, and exact checkpoint files with every reported run.
