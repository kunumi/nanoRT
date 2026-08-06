# nanoRT record #4 evidence manifest

Status: historical leaderboard evidence for the paper. Values in this document
come from the archived W&B run; rounded values are identified explicitly.

## Canonical references

- W&B run: [`ltlneqhn` / `balmy-wave-99`](https://wandb.ai/franciscokunumi-universidade-federal-de-minas-gerais/nanort-train/runs/ltlneqhn)
- Checkpoint archive: [`checkpoints/rel-avito/nanort-simplicial-staticrel`](https://huggingface.co/GabrielVeloso/NanoRT/tree/main/checkpoints/rel-avito/nanort-simplicial-staticrel)
- Run date: 2026-07-28
- Run state: finished

Use `305.6257 s` in machine-readable result tables, `305.63 s` in prose, and
`306 s (5m06s)` only when an integer-second headline is required.

## Verified result

The first scheduled and record-eligible validation occurred at step 100:

| field | value |
|---|---:|
| target pooled AUROC | `0.81621` |
| step-100 pooled AUROC | `0.8713918113` |
| `train/time_to_target` | `305.6257489 s` |
| rounded time | `305.63 s` / `5m05.6s` |
| previous record | `446 s` / `7m26s` |
| speedup over previous record | `1.4593x` |
| speedup over 45-minute baseline | `8.8343x` |

The run remained above the target at every scheduled evaluation from step 100
through step 700, fell below it at steps 800 and 900, and finished above it at
step 1000. This supports the first-crossing record while also showing that the
quality curve is not monotonic.

| step | pooled AUROC |
|---:|---:|
| 100 | 0.871392 |
| 200 | 0.853380 |
| 300 | 0.817586 |
| 400 | 0.835204 |
| 500 | 0.848178 |
| 600 | 0.849546 |
| 700 | 0.825661 |
| 800 | 0.781553 |
| 900 | 0.784179 |
| 1000 | 0.833696 |

## Resolved configuration

| setting | value |
|---|---|
| hardware | 1x NVIDIA H200 |
| OS / CUDA / Python | Linux x86-64 / CUDA 12.8 / CPython 3.12 |
| holdout | `rel-avito` |
| training databases | `rel-amazon`, `rel-hm`, `rel-stack`, `rel-trial`, `rel-f1` |
| optimizer | Muon for matrix parameters; AdamW for the remaining groups |
| AdamW / Muon LR | `0.001` / `0.005` |
| Muon warmup | `3000` steps |
| batch / sequence length | `256` / `1024` |
| model | 12 blocks, `d_model=256`, 8 heads, `d_ff=1024` |
| simplicial layers | all 12 blocks (`simplicial_layers=null` in the historical config) |
| relation stream | static (`update_relations=false`) |
| QK norm | enabled |
| LayerScale init | `1.0` |
| SIGReg | layer 0, column view only, `lambda_col=0.001` |
| SIGReg projections | 64, global column pooling, minimum group size 8 |
| validation cadence | every 100 steps |
| timed budget | 45 minutes |
| seed | 0 |

The output log confirms that SIGReg was active and used 659,584 projector
parameters. The checkpoint name predates the final evidence manifest and does
not encode every enabled component; the W&B config is authoritative.

## Timing boundary

The historical timer starts after the first optimizer update. It excludes setup,
data/model construction, compile warm-up, and the later step-0 diagnostic, but
includes subsequent training, validation, logging, and checkpoint selection.

At the step-100 crossing:

| component | seconds | share of official time |
|---|---:|---:|
| elapsed before the scheduled validation | 122.67 | 40.1% |
| validation and target verification | 182.96 | 59.9% |
| official timed total | 305.63 | 100% |
| W&B runtime from process start to the logged result | 404.32 | not part of the historical timer |

The paper must report the official timer and the end-to-end runtime separately.

## Initialization audit

The historical run predates explicit step-0 logging. A later instrumented run
with the same relevant model and optimizer configuration,
[`d1kv95bw` / `vague-eon-4`](https://wandb.ai/franciscokunumi-universidade-federal-de-minas-gerais/nanort-ablation/runs/d1kv95bw),
measured the following:

| metric | step 0 | step 100 | delta |
|---|---:|---:|---:|
| pooled AUROC | 0.865064 | 0.872058 | +0.006994 |
| within-task pooled AUROC | 0.570710 | 0.584723 | +0.014013 |
| mean per-task AUROC | 0.568182 | 0.578928 | +0.010746 |
| cross-task pooled AUROC | 0.914633 | 0.920445 | +0.005812 |

The cross-task pair weight is `0.855872`. Because the initialized model already
exceeds `0.81621`, record #4 is a **time to the first eligible verified solution**,
not a crossing from below. The within-task improvement is evidence of learned
signal, but it must be replicated across the pre-registered seeds before it is a
paper claim.

## Provenance and remaining verification

The historical W&B metadata does not contain a Git commit or immutable data
revision. The archived config, logs, environment metadata, and checkpoint are
the evidence for the leaderboard entry. The maintained release contains the
same record recipe, while all new Polyaxon paper runs resolve the dataset to an
immutable revision and log the code state automatically.

Before submission:

- confirm anonymous download access to the checkpoint archive;
- record the checkpoint file names, sizes, and hashes;
- execute the paired-seed sweep in [`POLYAXON_SWEEP_PLAN.md`](POLYAXON_SWEEP_PLAN.md);
- archive logits and labels for metric recomputation;
- replace this section with links to the final sweep manifest.

## Claim boundary

Supported now:

> Under the historical nanoRT protocol, the Muon + column-SIGReg + static
> relation-conditioned trilinear attention configuration logged its first
> eligible pooled-AUROC verification above 0.81621 in 305.63 seconds on one H200.

Not yet supported:

- that each component independently causes a speedup in the final architecture;
- a confidence interval over seeds;
- that 305.63 seconds is the time required to learn the target from an
  initialization below it;
- generalization of the speedup to a second held-out database.
