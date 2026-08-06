# Polyaxon sweep plan for the nanoRT paper

Status: pre-registered execution plan. Do not change the arms, seeds, target,
evaluation schedule, or analysis endpoints after inspecting sweep results.

The runnable three-seed screening matrix is
[`plx_nanort_paper_sweep.yaml`](../plx_nanort_paper_sweep.yaml). This document
defines the promotion rule, analysis, and later mechanism controls. Launching
the sweep is intentionally out of scope until the dataset revision and H200
allocation are confirmed.

## 1. Questions

The sweep separates the cumulative leaderboard story from causal attribution:

1. Does Muon improve the RT baseline under the current, audited protocol?
2. Does column-view SIGReg improve Muon without simplicial attention?
3. Does simplicial attention improve Muon + SIGReg?
4. Does SIGReg still contribute once simplicial attention is enabled?
5. Does simplicial attention depend on Muon?

Record #4 itself is already documented in
[`RECORD_04_MANIFEST.md`](RECORD_04_MANIFEST.md). The sweep estimates variation
and component effects; it does not retroactively change the historical record.

## 2. Frozen protocol

These values are shared by every arm:

| setting | frozen value |
|---|---|
| hardware | 1x NVIDIA H200 |
| holdout | `rel-avito` |
| training databases | the five non-holdout databases |
| batch / sequence length | `256` / `1024` |
| target | pooled AUROC `0.81621` |
| validation cadence | every 100 steps |
| step 0 / step 1 | diagnostic on / eligible eval off |
| dense early eval | off |
| timed budget | 45 minutes |
| AdamW LR / weight decay | `0.001` / `0.1` |
| Muon LR / warmup | `0.005` / `3000` steps |
| relation stream | static |
| QK norm | on |
| simplicial LayerScale | `1.0` |
| SIGReg champion | layer 0, column `0.001`, row `0`, 64 projections |

Before launch, replace the dataset revision placeholder with a single immutable
Hugging Face commit. Every child run must resolve to that same commit. Upload
checkpoints to `GabrielVeloso/NanoRT` using an arm-specific prefix.

## 3. Main factorial matrix

| arm | optimizer | SIGReg | simplicial | estimand |
|---|---|---:|---:|---|
| A | AdamW | off | off | maintained RT baseline |
| B | Muon | off | off | Muon effect: B - A |
| C | Muon | column | off | SIGReg before simplicial: C - B |
| D | Muon | column | all | cumulative champion: D - C |
| E | Muon | off | all | SIGReg in final architecture: D - E |
| F | AdamW | off | all | optimizer interaction: E - F |

Arm D is the record recipe. Arm E is mandatory because the historical paired
seed-0 runs with and without SIGReg were close at the first eligible evaluation;
the final paper must not attribute an additional simplicial-era speedup to SIGReg
without replicated evidence.

### Stage 1: screening

- Seeds: `0, 1, 2`, paired across all six arms.
- Runs: 18.
- Maximum allocation: 13.5 H200-hours.
- Polyaxon concurrency: 3.
- Run the complete matrix even if an early arm looks unfavorable.

Launch only after setting the immutable dataset revision:

```bash
polyaxon run -f plx_nanort_paper_sweep.yaml -u \
  -P hf_artifacts_revision="$HF_DATASET_SHA" \
  -P hf_token="$HF_TOKEN"
```

### Stage 2: paper table

After Stage 1 passes integrity checks, append seeds `3` and `4` to **every** arm,
not only the best-looking arm.

- Additional runs: 12.
- Additional maximum allocation: 9 H200-hours.
- Final matrix: 30 runs, five paired seeds per arm, 22.5 H200-hours maximum.

Five seeds support transparent paired effect estimates, but not strong
small-sample significance claims. Report every seed, median, range/IQR, and
paired differences. If a formal small-p claim is required, budget at least 10
paired seeds and choose the test before launching them.

## 4. Pre-registered endpoints

### Historical endpoint

- `train/time_to_target`: first eligible scheduled evaluation with pooled AUROC
  at or above `0.81621`.
- Runs that do not cross within 45 minutes are right-censored failures, never
  dropped or encoded as ordinary `-1` values in an average.

### Scientific endpoints

At step 100, report:

- pooled AUROC and `pooled@100 - pooled@0`;
- within-task pooled AUROC and `within@100 - within@0`;
- mean per-task AUROC and its delta from step 0;
- cross-task AUROC and cross-pair weight;
- per-task AUROCs, base rates, ties, and score duplication;
- active time immediately before evaluation and after verification.

The main attribution endpoint is the **paired change in within-task AUROC from
step 0 to step 100**. The historical pooled metric remains the leaderboard rule,
but it is already above target at initialization and therefore cannot by itself
identify learning speed.

At the end of the timed budget, also report the full validation trajectory,
best observed values, final values, success rate, throughput, peak memory, and
the time split among training, evaluation, logging, and checkpointing.

## 5. Integrity gates

Do not analyze method effects until all runs satisfy:

- same immutable artifact revision;
- expected five training databases and `rel-avito` only in validation;
- expected H200 identity and world size 1;
- identical seed within each paired block;
- identical batch, sampler, validation examples, and evaluation cadence;
- step 0 cannot trigger a record or checkpoint;
- no NaN/Inf in losses, logits, optimizer state, or gradients;
- logits, labels, resolved config, environment, timing data, and checkpoint
  manifest archived.

Any failed infrastructure run may be retried with the same arm and seed. A model
failure remains part of the result.

## 6. Analysis table

The script that consumes W&B histories should produce one row per arm and seed:

| field | description |
|---|---|
| `arm`, `seed`, W&B URL | run identity |
| code/data revision | provenance |
| `auroc_pooled_0`, `auroc_pooled_100` | official metric and delta source |
| `auroc_within_0`, `auroc_within_100` | relational discrimination endpoint |
| `auroc_mean_task_0`, `auroc_mean_task_100` | macro diagnostic |
| `auroc_cross_0`, `auroc_cross_100` | between-task contribution |
| `time_to_target`, `crossed` | censored leaderboard endpoint |
| `train_seconds_100`, `eval_seconds_100` | time decomposition |
| `examples_per_second`, `edges_per_second` | throughput |
| `peak_memory_bytes`, `parameters` | system cost |
| checkpoint/logit locations | artifact audit |

For each contrast in Section 3, compute within-seed differences first and only
then aggregate. Show the raw five points behind every summary.

## 7. Mechanism controls after the main matrix

These controls require new, tested configuration switches and are not part of
the immediately runnable YAML:

1. correct relation text;
2. one constant relation embedding for every edge;
3. a fixed within-database permutation of relation identities;
4. bilinear neighbor attention without the relation factor in the score;
5. relation in score only;
6. relation in value only;
7. QK norm off;
8. dynamic relation updates on.

Run one seed for implementation validation, then seeds `0,1,2` for every control
that passes. Keep parameter count and sparse edge support matched where possible.
These runs answer whether the simplicial result comes from relation semantics,
the trilinear mechanism, or merely added capacity.

## 8. Deliverables

- W&B report containing all runs and paired panels;
- immutable CSV/Parquet export of summary and history metrics;
- main results table and paired-seed plot;
- pooled within/cross decomposition figure;
- timing-stack figure;
- Hugging Face checkpoint and logits manifest;
- appendix table with every arm, seed, URL, failure, and retry.
