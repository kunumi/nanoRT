#!/usr/bin/env bash
# Install frozen artifacts and launch the single-GPU training recipe.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if [[ $# -ne 0 ]]; then
  echo "run.sh accepts configuration through environment variables only." >&2
  echo "Example: NANORT_SEED=1 ./run.sh" >&2
  exit 2
fi

# Model and optimizer defaults are defined once in rt/config.py.
export WANDB_MODE="${WANDB_MODE:-offline}"
export NANORT_ARTIFACT_ROOT="${NANORT_ARTIFACT_ROOT:-${HOME}/.cache/modded-nanort/artifacts}"
export NANORT_PRE_ROOT="${NANORT_PRE_ROOT:-${HOME}/.cache/modded-nanort/pre}"
NANORT_HF_REPO="${NANORT_HF_REPO:-${HF_REPO:-GabrielVeloso/nanort-artifacts}}"
NANORT_HF_REVISION="${NANORT_HF_REVISION:-${HF_REVISION:-8e6289edd0aa1d61b3b4c2ec1e1beed48224cd2c}}"

pixi install --locked
(
  cd rustler
  pixi run maturin develop --uv --release
)
pixi run python data/download.py \
  --repo-id "${NANORT_HF_REPO}" \
  --revision "${NANORT_HF_REVISION}"
pixi run python -u scripts/nanort_train.py
