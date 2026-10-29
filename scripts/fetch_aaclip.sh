#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/Mwxinnn/AA-CLIP.git"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REVISION_FILE="${ROOT_DIR}/models/AACLIP_REVISION"
TARGET="${ROOT_DIR}/models/AA-CLIP"

if [[ ! -f "${REVISION_FILE}" ]]; then
  echo "Pinned revision file is missing: ${REVISION_FILE}" >&2
  exit 1
fi
PINNED_COMMIT="$(tr -d '[:space:]' < "${REVISION_FILE}")"
if [[ -z "${PINNED_COMMIT}" ]]; then
  echo "Pinned revision file is empty: ${REVISION_FILE}" >&2
  exit 1
fi
if ! command -v git >/dev/null 2>&1; then
  echo "git is required" >&2
  exit 1
fi

if [[ -d "${TARGET}/.git" ]]; then
  git -C "${TARGET}" fetch --tags origin
else
  if [[ -d "${TARGET}" ]]; then
    rm -rf "${TARGET}"
  fi
  git clone "${REPO_URL}" "${TARGET}"
fi

git -C "${TARGET}" checkout --detach "${PINNED_COMMIT}"
ACTUAL="$(git -C "${TARGET}" rev-parse --short=7 HEAD)"
EXPECTED="$(git -C "${TARGET}" rev-parse --short=7 "${PINNED_COMMIT}")"
if [[ "${ACTUAL}" != "${EXPECTED}" ]]; then
  echo "Expected ${EXPECTED}, checked out ${ACTUAL}" >&2
  exit 1
fi

echo "Official AA-CLIP checked out at ${TARGET} (${ACTUAL})."
echo "Next: prepare the OpenAI ViT-L-14-336 weights and the VisA-trained AA-CLIP adapters per docs/SETUP.md."
