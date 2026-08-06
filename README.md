# nanoRT

nanoRT is an open, single-GPU speedrun for zero-shot relational foundation model
pretraining. The goal is to reach a fixed **pooled instance-level AUROC of
0.81621** on the held-out `rel-avito` domain using one NVIDIA H200.

The current record is **305.63 seconds (5m05.6s; 306 s rounded)**. It combines:

- [Muon](https://github.com/KellerJordan/Muon) for matrix parameters, with AdamW
  for embeddings, biases, normalization parameters, and SIGReg projectors;
- column-view [SIGReg](SIGREG.md) at the first transformer block;
- relation-conditioned trilinear neighbor attention, called *simplicial
  attention* in the implementation;
- a static text-derived relation stream and per-head QK normalization.

The code is derived from the
[Relational Transformer](https://arxiv.org/abs/2510.06377) and follows the open
record-lineage style of
[modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt) and
[Speedrunning Tabular Foundation Model Pretraining](https://arxiv.org/abs/2606.03681).

## Leaderboard

All entries use the same H200 hardware class, five-database training mixture,
frozen `rel-avito` validation artifact, pooled AUROC implementation, target, and
historical `TrainTimer` boundary.

| # | Time to target | Date | Incremental recipe | Contributor |
|---:|---:|---:|---|---|
| 1 | 45m00s | baseline | AdamW Relational Transformer | — |
| 2 | 12m27s | 2026-05-05 | + Muon | Gabriel Veloso |
| 3 | 7m26s | 2026-06-24 | + column-view SIGReg | Gabriel Veloso |
| 4 | **5m05.6s (305.63 s)** | 2026-07-28 | + relation-conditioned trilinear neighbor attention | Gabriel Veloso |

Record #4 is **1.46× faster** than record #3 and **8.83× faster** than the
45-minute baseline under the historical timing rule.

The archived evidence is W&B run
[`ltlneqhn`](https://wandb.ai/franciscokunumi-universidade-federal-de-minas-gerais/nanort-train/runs/ltlneqhn):
the first eligible evaluation, at step 100, obtained pooled AUROC `0.8713918`
and logged `train/time_to_target=305.6257`. The corresponding checkpoint is
archived under
[`checkpoints/rel-avito/nanort-simplicial-staticrel`](https://huggingface.co/GabrielVeloso/NanoRT/tree/main/checkpoints/rel-avito/nanort-simplicial-staticrel).
See the [record manifest](docs/RECORD_04_MANIFEST.md) for the resolved
configuration, validation curve, and timing decomposition.

### Initialization audit

The record configuration also has a strong initialization prior. A diagnostic
run measured pooled AUROC `0.8651` at step 0, before any optimizer update. nanoRT
therefore logs step 0 outside the timer, prevents it from selecting a checkpoint
or record, and reports post-initialization gains separately. The official 305.63 s
entry is the first eligible post-update evaluation under the frozen leaderboard
rule; the paper must report `AUROC@0` alongside time-to-target.

This distinction matters: nanoRT measures **time to a verified solution under a
fixed protocol**, not pure optimizer convergence time.

## Reproduce record #4

Requirements:

- Linux x86-64;
- one NVIDIA H200 with a CUDA 12.4-compatible driver;
- access to the public nanoRT artifact repository;
- [pixi](https://pixi.sh/).

```bash
git clone https://github.com/kunumi/nanoRT.git
cd nanoRT
./run.sh
```

`run.sh` installs the locked environment, builds the Rust sampler, downloads the
relation-aware holdout artifact, creates the expected per-database links, and
starts the record configuration. Override settings through environment variables:

```bash
NANORT_SEED=1 \
WANDB_PROJECT=nanort-reproduction \
HF_REPO=gabrielveloso/nanort-artifacts \
HF_REVISION=YOUR_IMMUTABLE_COMMIT_SHA \
./run.sh
```

The record defaults are centralized in [`rt/config.py`](rt/config.py) and mirrored
by [`plx_nanort_train.yaml`](plx_nanort_train.yaml):

| setting | value |
|---|---:|
| optimizer | Muon + AdamW |
| `muon_lr` | `5e-3` |
| `muon_warmup_steps` | `3000` |
| SIGReg | layer `0`, column `1e-3`, global pooling |
| simplicial layers | all 12 blocks |
| relation stream | static (`update_relations=0`) |
| QK norm | enabled |
| batch size / sequence length | `256` / `1024` |
| official eval cadence | every `100` steps |
| step 0 / step 1 eval | diagnostic on / official off |
| maximum timed budget | 45 minutes |

For an ablation, override only the relevant variable. Examples:

```bash
# Remove the simplicial path.
NANORT_SIMPLICIAL_LAYERS=none ./run.sh

# Keep the architecture but update the relation stream between blocks.
NANORT_UPDATE_RELATIONS=1 ./run.sh

# Disable SIGReg.
NANORT_SIGREG_LAYER=-1 NANORT_SIGREG_LAMBDA_COL=0 ./run.sh
```

## What the model does

RT samples a temporally valid relational neighborhood around each seed row,
masks the target cell, and encodes cell values together with table, column, and
relation text metadata. The model alternates schema-aware attention patterns and
predicts the masked value with a shared backbone.

The new sparse neighbor operator scores a parent query `q`, child-cell key `k`,
and relation key `r` jointly:

```text
score(q, k, r) = sum_d q_d * k_d * r_d
value(v, r)    = v * r
```

It is a factorized trilinear/FiLM interaction over typed foreign-key edges, not
an explicit simplicial complex. See
[`docs/simplicial_relational_attention.md`](docs/simplicial_relational_attention.md)
for the data flow, sparse edge representation, and controls.

## Evaluation protocol

### Official metric

The callback concatenates predictions and labels from the eligible binary
`rel-avito` tasks and calls `roc_auc_score` once. This is **pooled
instance-level AUROC**, not macro or mean per-task AUROC.

Equivalently, it is the probability that a positive drawn from the complete
empirical validation population is scored above a negative, with half credit for
ties. This makes within-task ranking and cross-task score alignment part of the
official estimand. The metric remains fixed to preserve the historical record
lineage.

Every evaluation additionally logs:

- AUROC and prevalence for each task;
- unweighted macro AUROC;
- exact within-task and cross-task pair contributions;
- a task-prior oracle diagnostic;
- score duplication and positive/negative tie rates;
- evaluation duration and the step-0 baseline.

Only `val/auroc_pooled`—also logged under the historical key
`val/auroc_zero_shot`—can trigger the target.

### Timer

The historical timer starts immediately **after the first optimizer update**.
Compilation, compile warm-up, and step-0 evaluation are untimed. Once started,
the timer is wall-clock time and includes subsequent data loading, scheduled
validation, logging, and checkpoint overhead. This boundary is intentionally
preserved for leaderboard comparability; separate timing metrics expose its
components for the paper.

Because evaluation is discrete, time-to-target is the timestamp of the first
eligible evaluation at or above `0.81621`. It should not be interpreted with
precision finer than the evaluation cadence.

### Step 0

Step 0 evaluates the initialized model before any optimizer update. It:

- runs outside the official timer;
- cannot trigger `train/time_to_target`;
- cannot select or save the best checkpoint;
- measures architectural/schema priors and enables `AUROC(t) - AUROC(0)`.

## Holdout guarantee

For a `rel-avito` run, the training artifact contains exactly the other five
eligible databases: `rel-amazon`, `rel-hm`, `rel-stack`, `rel-trial`, and
`rel-f1`. The entry point fails if the resolved training set differs or includes
the holdout. Validation loads only the frozen `rel-avito` artifact.

The Rust sampler enforces temporal cutoffs, masks the prediction target, and
removes declared leakage columns. Seeds, three-hop extraction, schema text
embeddings, and the validation artifact are frozen across records.

Both launchers resolve the requested HuggingFace revision to an immutable commit
before downloading. That revision, the Git commit and dirty-state flag, Python,
PyTorch/CUDA, and GPU identity are stored in the W&B run config. Paper
reproductions should pass the recorded artifact commit through `HF_REVISION` (or
the equivalent Polyaxon input), rather than relying on a mutable `main` branch.

`rel-avito` is a **gradient holdout and zero-shot development domain**. Because
its validation results have guided repeated engineering decisions, it should not
be described as a statistically untouched test set.

## Tests and validation

CPU-only protocol tests:

```bash
python -m pytest tests
```

Full environment checks:

```bash
pixi install --locked
pixi run python -m pytest
pixi run python -m compileall -q rt scripts tests
pixi run cargo test --manifest-path rustler/Cargo.toml
bash -n run.sh
```

The H200 run remains the required integration test for `torch.compile`, bf16,
the Rust sampler, sparse scatter kernels, and the exact timing result.

## Paper material

- [Paper execution plan](docs/PAPER_EXECUTION_PLAN.md)
- [Introduction and Related Work draft](docs/PAPER_DRAFT.md)
- [Record #4 evidence manifest](docs/RECORD_04_MANIFEST.md)
- [Polyaxon paper sweep plan](docs/POLYAXON_SWEEP_PLAN.md)
- [Simplicial attention design](docs/simplicial_relational_attention.md)
- [SIGReg implementation](SIGREG.md)
- [Ablation and metric audit](docs/ABLATION_PLAN.md)

## References

1. [Robinson et al. RelBench: A Benchmark for Deep Learning on Relational Databases.](https://arxiv.org/abs/2407.20060)
2. [Ranjan et al. Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data.](https://arxiv.org/abs/2510.06377)
3. [Ozturk et al. Speedrunning Tabular Foundation Model Pretraining.](https://arxiv.org/abs/2606.03681)
4. [Pfefferle et al. nanoTabPFN.](https://arxiv.org/abs/2511.03634)
5. [Jordan et al. modded-nanoGPT and Muon.](https://github.com/KellerJordan/modded-nanogpt)
