#!/usr/bin/env bash
# Reproduce record #4 (305.63 s / 5m05.6s) on one NVIDIA H200.
#
# Configuration is environment-only; scripts/nanort_train.py intentionally has
# no CLI parser. Override any NANORT_* value before invoking this script.
set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "run.sh accepts configuration through environment variables only." >&2
  echo "Example: NANORT_SEED=1 HF_REPO=user/nanort-artifacts ./run.sh" >&2
  exit 2
fi

export NANORT_HOLDOUT_DB="${NANORT_HOLDOUT_DB:-rel-avito}"
export NANORT_BATCH_SIZE="${NANORT_BATCH_SIZE:-256}"
export NANORT_OPTIMIZER="${NANORT_OPTIMIZER:-muon}"
export NANORT_LR="${NANORT_LR:-0.001}"
export NANORT_MUON_LR="${NANORT_MUON_LR:-0.005}"
export NANORT_MUON_WARMUP_STEPS="${NANORT_MUON_WARMUP_STEPS:-3000}"
export NANORT_MAX_STEPS="${NANORT_MAX_STEPS:-1000000}"
export NANORT_LR_TOTAL_STEPS="${NANORT_LR_TOTAL_STEPS:-0}"

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

# The official timer includes scheduled validation. Step 0 is an untimed
# diagnostic; step 1 is skipped and the first eligible record eval is step 100.
export NANORT_EVAL_AT_STEP0="${NANORT_EVAL_AT_STEP0:-1}"
export NANORT_EVAL_AT_STEP1="${NANORT_EVAL_AT_STEP1:-0}"
export NANORT_VAL_EVAL_FREQ="${NANORT_VAL_EVAL_FREQ:-100}"
export NANORT_VAL_EVAL_DENSE_UNTIL="${NANORT_VAL_EVAL_DENSE_UNTIL:-0}"
export NANORT_TIME_BUDGET_MINUTES="${NANORT_TIME_BUDGET_MINUTES:-45}"

HF_REPO="${HF_REPO:-gabrielveloso/nanort-artifacts}"
HF_REVISION_REQUESTED="${HF_REVISION:-main}"
NANORT_ARTIFACT_ROOT="${NANORT_ARTIFACT_ROOT:-${HOME}/scratch/nanort_artifacts}"
NANORT_PRE_ROOT="${NANORT_PRE_ROOT:-${HOME}/scratch/pre}"
ARTIFACT_PREFIX="ra-holdout-${NANORT_HOLDOUT_DB}"

echo "nanoRT record #4 configuration"
echo "  holdout=${NANORT_HOLDOUT_DB} optimizer=${NANORT_OPTIMIZER} seed=${NANORT_SEED:-0}"
echo "  simplicial=${NANORT_SIMPLICIAL_LAYERS} update_relations=${NANORT_UPDATE_RELATIONS} qk_norm=${NANORT_QK_NORM}"
echo "  sigreg_layer=${NANORT_SIGREG_LAYER} lambda_row=${NANORT_SIGREG_LAMBDA_ROW} lambda_col=${NANORT_SIGREG_LAMBDA_COL}"
echo "  val_freq=${NANORT_VAL_EVAL_FREQ} step0=${NANORT_EVAL_AT_STEP0} step1=${NANORT_EVAL_AT_STEP1}"

pixi install --locked

# Resolve a mutable branch/tag once, then download and log the immutable Hub
# commit. Set HF_REVISION to the recorded commit to reproduce a paper run later.
HF_REVISION_RESOLVED="$(
  pixi run python - "${HF_REPO}" "${HF_REVISION_REQUESTED}" <<'PY'
import sys

from huggingface_hub import HfApi

repo_id, revision = sys.argv[1:]
print(HfApi().repo_info(repo_id=repo_id, repo_type="dataset", revision=revision).sha)
PY
)"
export NANORT_ARTIFACT_REPO="${HF_REPO}"
export NANORT_ARTIFACT_REVISION="${HF_REVISION_RESOLVED}"
echo "  artifact=${NANORT_ARTIFACT_REPO}@${NANORT_ARTIFACT_REVISION}"

(
  cd rustler
  pixi run maturin develop --uv --release
)

mkdir -p "${NANORT_ARTIFACT_ROOT}" "${NANORT_PRE_ROOT}"
pixi run hf download "${HF_REPO}" \
  --repo-type dataset \
  --revision "${HF_REVISION_RESOLVED}" \
  --include "${ARTIFACT_PREFIX}/train/*" \
  --include "${ARTIFACT_PREFIX}/val/*" \
  --local-dir "${NANORT_ARTIFACT_ROOT}"

TRAIN_ROOT="${NANORT_ARTIFACT_ROOT}/${ARTIFACT_PREFIX}/train"
VAL_ROOT="${NANORT_ARTIFACT_ROOT}/${ARTIFACT_PREFIX}/val/${NANORT_HOLDOUT_DB}"
if [[ ! -d "${TRAIN_ROOT}" || ! -d "${VAL_ROOT}" ]]; then
  echo "Downloaded artifact layout is incomplete under ${ARTIFACT_PREFIX}." >&2
  exit 1
fi

for source_dir in "${TRAIN_ROOT}"/*; do
  [[ -d "${source_dir}" ]] || continue
  db_name="$(basename "${source_dir}")"
  ln -sfn "${source_dir}" "${NANORT_PRE_ROOT}/${db_name}"
done
ln -sfn "${VAL_ROOT}" "${NANORT_PRE_ROOT}/${NANORT_HOLDOUT_DB}"

pixi run python -u scripts/nanort_train.py
