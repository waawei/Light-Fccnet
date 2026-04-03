#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WORK_DIR="$(cd -- "${PACKAGE_DIR}/.." && pwd)"

cd "${WORK_DIR}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"

configs=(
  "gwhd/config_gwhd_light_full.yaml"
  "mtc/config_mtc_light_full.yaml"
  "urc/config_urc_light_full.yaml"
)

for cfg in "${configs[@]}"; do
  echo "==== Running ${cfg} ===="
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m pack.train --config "pack/config/${cfg}" --device cuda
done
