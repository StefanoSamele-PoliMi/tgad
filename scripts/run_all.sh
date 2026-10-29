#!/usr/bin/env bash
set -euo pipefail

# Usage: [MODEL=anomalygpt|aaclip|logsad|logsad_fms] run_all.sh MVTec_ROOT APD_ROOT APD_COCO_JSON MODEL_CONFIG [OUTPUT_ROOT]
# OUTPUT_ROOT defaults to outputs (AnomalyGPT) or outputs/<model> (other models).

if [[ $# -lt 4 ]]; then
  echo "Usage: [MODEL=anomalygpt|aaclip|logsad|logsad_fms] $0 MVTec_ROOT APD_ROOT APD_COCO_JSON MODEL_CONFIG [OUTPUT_ROOT]" >&2
  exit 2
fi

MODEL="${MODEL:-anomalygpt}"
MVTEC_ROOT="$1"
APD_ROOT="$2"
APD_COCO="$3"
MODEL_CONFIG="$4"
if [[ "${MODEL}" == "anomalygpt" ]]; then
  OUTPUT_ROOT="${5:-outputs}"
else
  OUTPUT_ROOT="${5:-outputs/${MODEL}}"
fi

COMMON=(--model "${MODEL}" --backend official --model-config "${MODEL_CONFIG}")

tgad scenario1 \
  --mvtec-root "${MVTEC_ROOT}" \
  --output-dir "${OUTPUT_ROOT}/scenario1" \
  --cache-dir "${OUTPUT_ROOT}/cache/predictions" \
  "${COMMON[@]}"

tgad scenario2 \
  --mvtec-root "${MVTEC_ROOT}" \
  --output-dir "${OUTPUT_ROOT}/scenario2" \
  --cache-dir "${OUTPUT_ROOT}/cache/predictions" \
  "${COMMON[@]}"

tgad scenario3 \
  --apd-root "${APD_ROOT}" \
  --coco-annotations "${APD_COCO}" \
  --output-dir "${OUTPUT_ROOT}/scenario3" \
  --cache-dir "${OUTPUT_ROOT}/cache/predictions" \
  --prepared-cache-dir "${OUTPUT_ROOT}/cache/prepared_apd" \
  "${COMMON[@]}"
