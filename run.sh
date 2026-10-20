#!/usr/bin/env bash
# Launch the released configuration on one NVIDIA H200.
set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "run.sh accepts configuration through environment variables only." >&2
  echo "Example: NANORT_HF_REPO=namespace/artifacts NANORT_SEED=1 ./run.sh" >&2
  exit 2
fi

# Training configuration. Override defaults through environment variables.
export NANORT_HOLDOUT_DB="${NANORT_HOLDOUT_DB:-rel-avito}"
export NANORT_BATCH_SIZE="${NANORT_BATCH_SIZE:-256}"
export NANORT_OPTIMIZER="${NANORT_OPTIMIZER:-muon}"
export NANORT_LR="${NANORT_LR:-0.001}"
export NANORT_MUON_LR="${NANORT_MUON_LR:-0.005}"
export NANORT_MUON_WARMUP_STEPS="${NANORT_MUON_WARMUP_STEPS:-3000}"
export NANORT_MAX_STEPS="${NANORT_MAX_STEPS:-1000000}"
export NANORT_LR_TOTAL_STEPS="${NANORT_LR_TOTAL_STEPS:-0}"

# ``simplicial`` is the legacy code name for typed trilinear attention.
export NANORT_SIMPLICIAL_LAYERS="${NANORT_SIMPLICIAL_LAYERS:-all}"
export NANORT_UPDATE_RELATIONS="${NANORT_UPDATE_RELATIONS:-0}"
export NANORT_QK_NORM="${NANORT_QK_NORM:-1}"
export NANORT_SIMPLICIAL_LAYERSCALE_INIT="${NANORT_SIMPLICIAL_LAYERSCALE_INIT:-1.0}"

export NANORT_SIGREG_LAYER="${NANORT_SIGREG_LAYER:-0}"
export NANORT_SIGREG_LAMBDA_ROW="${NANORT_SIGREG_LAMBDA_ROW:-0.0}"
export NANORT_SIGREG_LAMBDA_COL="${NANORT_SIGREG_LAMBDA_COL:-0.001}"
export NANORT_SIGREG_N_PROJ="${NANORT_SIGREG_N_PROJ:-64}"
export NANORT_SIGREG_MIN_GROUP="${NANORT_SIGREG_MIN_GROUP:-8}"
export NANORT_SIGREG_COL_POOL="${NANORT_SIGREG_COL_POOL:-1}"

# Step 0 is diagnostic and untimed. The first record-eligible evaluation is at
# step 100, matching the archived timing boundary.
export NANORT_EVAL_AT_STEP0="${NANORT_EVAL_AT_STEP0:-1}"
export NANORT_EVAL_AT_STEP1="${NANORT_EVAL_AT_STEP1:-0}"
export NANORT_VAL_EVAL_FREQ="${NANORT_VAL_EVAL_FREQ:-100}"
export NANORT_VAL_EVAL_DENSE_UNTIL="${NANORT_VAL_EVAL_DENSE_UNTIL:-0}"
export NANORT_TIME_BUDGET_MINUTES="${NANORT_TIME_BUDGET_MINUTES:-45}"

# Local logging makes the quick start non-interactive. Set WANDB_MODE=online
# explicitly to publish a run.
export WANDB_MODE="${WANDB_MODE:-offline}"

export NANORT_ARTIFACT_ROOT="${NANORT_ARTIFACT_ROOT:-${HOME}/.cache/modded-nanort/artifacts}"
export NANORT_PRE_ROOT="${NANORT_PRE_ROOT:-${HOME}/.cache/modded-nanort/pre}"
NANORT_HF_REPO="${NANORT_HF_REPO:-${HF_REPO:-GabrielVeloso/nanort-artifacts}}"
NANORT_HF_REVISION="${NANORT_HF_REVISION:-${HF_REVISION:-8e6289edd0aa1d61b3b4c2ec1e1beed48224cd2c}}"

if [[ -z "${NANORT_HF_REPO}" ]]; then
  echo "NANORT_HF_REPO is required." >&2
  echo "Set it to the relation-aware artifact dataset repository." >&2
  exit 2
fi

echo "modded-nanoRT record configuration"
echo "  holdout=${NANORT_HOLDOUT_DB} optimizer=${NANORT_OPTIMIZER} seed=${NANORT_SEED:-0}"
echo "  typed_trilinear_layers=${NANORT_SIMPLICIAL_LAYERS} update_relations=${NANORT_UPDATE_RELATIONS} qk_norm=${NANORT_QK_NORM}"
echo "  sigreg_layer=${NANORT_SIGREG_LAYER} lambda_col=${NANORT_SIGREG_LAMBDA_COL}"
echo "  val_freq=${NANORT_VAL_EVAL_FREQ} step0=${NANORT_EVAL_AT_STEP0} step1=${NANORT_EVAL_AT_STEP1}"

pixi install --locked

(
  cd rustler
  pixi run maturin develop --uv --release
)

pixi run python data/download.py \
  --repo-id "${NANORT_HF_REPO}" \
  --revision "${NANORT_HF_REVISION}" \
  --holdout "${NANORT_HOLDOUT_DB}" \
  --artifact-root "${NANORT_ARTIFACT_ROOT}" \
  --install-root "${NANORT_PRE_ROOT}"

pixi run python -u scripts/nanort_train.py
