# Third-party notices

## AnomalyGPT

The repository uses, but does not redistribute, the official AnomalyGPT source from `CASIA-LMC-Lab/AnomalyGPT`. At the pinned revision, the upstream project declares the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International** license. `scripts/fetch_anomalygpt.sh` downloads that code into `models/AnomalyGPT`; the upstream license and notices govern it and may impose non-commercial/share-alike restrictions that do not apply to this repository's MIT-licensed files.

## AA-CLIP

The repository uses, but does not redistribute, the official AA-CLIP source from `Mwxinnn/AA-CLIP`, which declares the **Apache License 2.0** at the pinned revision. `scripts/fetch_aaclip.sh` downloads that code into `models/AA-CLIP`; the upstream license and notices govern it.

## LogSAD

The repository uses, but does not redistribute, the official LogSAD source from `zhang0jhon/LogSAD`. At the pinned revision it declares no license. It bundles adapted copies of OpenCLIP (`open_clip_local`, MIT), anomalib (Apache 2.0), and DINOv2 (Apache 2.0), which keep their own licenses. `scripts/fetch_logsad.sh` downloads that code into `models/LogSAD`.

## Model components and checkpoints

ImageBind, Vicuna, PandaGPT, AnomalyGPT deltas, OpenAI CLIP weights, AA-CLIP adapters, the DataComp CLIP checkpoint used by LogSAD, Hugging Face model files, and any tokenizer/model assets retain their respective licenses and use restrictions. They are not included.

## Datasets

MVTec AD is not included. Obtain it from its authorized source and comply with its terms.

The Assembled Panel Dataset (APD) is not stored in this repository. It is released with the TGAD benchmark as a gated Hugging Face dataset, [StefanoSamele-PoliMi/assembled-panel-dataset](https://huggingface.co/datasets/StefanoSamele-PoliMi/assembled-panel-dataset), under the APD Research-Only License: non-commercial research and educational use only, no redistribution, and citation of the TGAD paper required. Access requires accepting these terms on the dataset page; `scripts/download_apd.sh` then fetches it into `data/apd/`. The MIT license of this repository does not apply to the APD.

The per-part MVTec tag annotation is released as part of the TGAD benchmark.
