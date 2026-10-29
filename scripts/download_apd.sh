#!/usr/bin/env bash
set -euo pipefail

REPO_ID="StefanoSamele-PoliMi/assembled-panel-dataset"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-${ROOT_DIR}/data/apd}"
PYTHON="${PYTHON:-python}"

if ! "${PYTHON}" -c "import huggingface_hub" >/dev/null 2>&1; then
  echo "huggingface_hub is required: pip install huggingface_hub" >&2
  exit 1
fi

"${PYTHON}" - "${REPO_ID}" "${TARGET}" <<'EOF'
import sys

from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

repo_id, target = sys.argv[1], sys.argv[2]
try:
    snapshot_download(
        repo_id,
        repo_type="dataset",
        local_dir=target,
        allow_patterns=["normal/*", "anomalous/*", "assembled_panel_annotations.json", "*.pdf", "LICENSE", "README.md"],
    )
except (GatedRepoError, RepositoryNotFoundError):
    sys.exit(
        f"Access to {repo_id} is gated. Accept the terms at https://huggingface.co/datasets/{repo_id}, "
        "then log in with `hf auth login` or set HF_TOKEN."
    )
EOF

NORMAL="$(find "${TARGET}/normal" -maxdepth 1 -name '*.png' | wc -l)"
ANOMALOUS="$(find "${TARGET}/anomalous" -maxdepth 1 -name '*.png' | wc -l)"
if [[ "${NORMAL}" -ne 89 || "${ANOMALOUS}" -ne 53 || ! -f "${TARGET}/assembled_panel_annotations.json" ]]; then
  echo "Incomplete APD download in ${TARGET}: ${NORMAL}/89 normal, ${ANOMALOUS}/53 anomalous images" >&2
  exit 1
fi

echo "APD downloaded to ${TARGET} (${NORMAL} normal, ${ANOMALOUS} anomalous images)."
echo "Next: tgad validate --apd-root $(dirname "${TARGET}") --coco-annotations ${TARGET}/assembled_panel_annotations.json"
