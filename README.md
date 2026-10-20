# modded-nanoRT

**Authors:** Gabriel Wedson Mendonça de Souza Veloso<sup>1,2</sup>,
Luan Borges Teodoro Reis Sena<sup>1,2</sup>,
Denis Oliveira Correa<sup>1</sup>,
Francisco Galuppo Azevedo<sup>1,2</sup>.

<sup>1</sup> Kunumi Institute, Belo Horizonte, Minas Gerais, Brazil<br>
<sup>2</sup> Universidade Federal de Minas Gerais, Belo Horizonte, Minas Gerais, Brazil

`{gabriel.veloso, luan.borges, denis, francisco}@kunumi.com`

Our paper, *Speedrunning Relational Foundation Model Pretraining*, was accepted
at the **CODEC-FM Workshop at NeurIPS 2026**.

This repository hosts the *nanoRT speedrun*, in which we collaboratively and
competitively search for the fastest way to train a Relational Transformer on
**1 NVIDIA H200** to reach **0.81621 pooled zero-shot AUROC** on the frozen
[`rel-avito`](https://relbench.stanford.edu/) validation set.

The target comes from a 45-minute calibration run of the upstream
[Relational Transformer](https://github.com/snap-stanford/relational-transformer),
which reaches it at **41m20s**.
The code descends from that implementation and follows the cumulative speedrun
format of [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt) and
[modded-nanoTabPFN](https://github.com/borawhocodess/modded-nanotabpfn).

The best archived record reaches the target in **5m05.6s (305.63 seconds)**,
an **8.11× speedup** over the reference time-to-target, using:

- [Muon](https://github.com/KellerJordan/Muon) for transformer block matrices,
  with AdamW for the remaining parameters;
- column-view **SIGReg** after the first transformer block, with global
  column pooling and `lambda_col=1e-3`;
- **typed trilinear attention** over foreign-key edges, with static text-derived
  relation states and per-head QK normalization.

## Running the current record

Requirements: Linux x86-64, one NVIDIA H200, a CUDA 12.4-compatible driver,
and [pixi](https://pixi.sh/).

```bash
git clone https://github.com/kunumi/nanoRT.git
cd nanoRT
./run.sh
```

`run.sh` installs the locked environment, builds the Rust sampler, downloads
the frozen artifacts, checks holdout isolation, and starts training.
The first invocation downloads tens of gigabytes and compiles the environment.
Setup and compile warm-up are outside the historical record timer.

The public [artifact dataset](https://huggingface.co/datasets/GabrielVeloso/nanort-artifacts)
is pinned to commit `8e6289edd0aa1d61b3b4c2ec1e1beed48224cd2c`.
See the [data guide](data/README.md) for installation and layout details.

Training defaults live in [`rt/config.py`](rt/config.py). Override them through
environment variables:

```bash
NANORT_SEED=1 ./run.sh
```

W&B logs stay offline by default. Set `WANDB_MODE=online` and
`WANDB_PROJECT=your-project` to publish your own run.

| Setting | Record recipe |
|---|---|
| Model | 12 blocks, width 256, 8 heads |
| Batch size / context length | `256` / `1024` |
| AdamW / Muon learning rate | `1e-3` / `5e-3` |
| Muon warm-up | 3,000 steps |
| Trilinear attention | all blocks; static relations; QK normalization |
| SIGReg | block 0, column view, weight `1e-3` |
| Eligible evaluation | every 100 updates |
| Timed budget | 45 minutes |

In the released implementation, Muon warms up linearly for 3,000 updates and
then keeps a constant learning rate. The backbone AdamW group uses a linear
OneCycle schedule with a 20% warm-up. The SIGReg projector has a separate
AdamW optimizer at `1e-3`, without weight decay or a learning-rate schedule.

The code retains `simplicial` in class and configuration names for compatibility
with the record runs.

## Record history

Historical progression for the following task:

> Train on five RelBench databases and reach at least 0.81621 pooled zero-shot
> AUROC on the frozen `rel-avito` holdout using 1× NVIDIA H200.

| # | Record | Date | Description | Links | Contributor |
|---:|---|---|---|---|---|
| 1 | 41m20s | — | AdamW Relational Transformer, 45-minute calibration budget | [protocol](docs/BENCHMARK.md) | — |
| 2 | 12m27s | 2026-05-05 | + Muon | [code](https://github.com/kunumi/nanoRT/commit/c50df4c) | Gabriel Veloso |
| 3 | 7m26s | 2026-06-24 | + column-view SIGReg | [code](https://github.com/kunumi/nanoRT/commit/49df323) | Gabriel Veloso |
| 4 | **5m05.6s** | 2026-07-28 | + typed trilinear attention | [log](https://wandb.ai/franciscokunumi-universidade-federal-de-minas-gerais/nanort-train/runs/ltlneqhn), [checkpoint](https://huggingface.co/GabrielVeloso/NanoRT/tree/main/checkpoints/rel-avito/nanort-simplicial-staticrel) | Gabriel Veloso |

Record #4 first verifies the target at step 100, with pooled AUROC `0.8713918`
and time-to-target `305.6257 s`. It is **1.46× faster** than record #3.
These are archived results; the streamlined release still needs a fresh H200
run to measure its own wall-clock time.

## Rules

New records must:

- use one NVIDIA H200 and start trainable weights from a declared random seed;
- use the prescribed RelBench v1 binary-classification training artifact:
  `rel-amazon`, `rel-hm`, `rel-stack`, `rel-trial`, and `rel-f1`, with
  `rel-avito` held out from gradient updates and `rel-event` excluded;
- keep the frozen temperature-sampled mixture, with database seed shares
  proportional to `n(d)^0.5`; this subsampling is applied when building the
  artifact, rather than reweighted during training;
- preserve the validation artifact, target masking, temporal cutoffs, and
  declared leakage-column exclusions;
- preserve the validation pipeline, pooled AUROC implementation, target,
  evaluation cadence, and timer boundary;
- load no model checkpoint; frozen text embeddings belong to the data artifact;
- keep the timed budget at 45 minutes;
- report runs that miss the target as right-censored failures.

Architecture, optimization, regularization, sampling, precision, and systems
implementation are open. See the [benchmark specification](docs/BENCHMARK.md)
for the complete protocol and required evidence.

## Evaluation details

The official evaluator concatenates predictions and labels from both eligible
`rel-avito` tasks, `user-visits` and `user-clicks`, and computes **one pooled
instance-level AUROC**. Only this pooled score determines record eligibility;
it weights examples rather than averaging task AUROCs. Cross-task score
alignment contributes to this metric. The artifact fixes the validation seed
cap at 50,000, hop radius, and text embeddings.
Each evaluation also reports per-task AUROC, prevalence, within-task and
cross-task contributions, and prediction tie rates.

Step 0 evaluates the initialized model outside the timer. It cannot establish
a record or select a checkpoint. The initialized recipe can already exceed the
pooled target, so the historical record measures time to the first eligible
verified solution under this protocol.

The clock starts after the first optimizer update and includes subsequent
training, scheduled validation, logging, and checkpointing. The first eligible
evaluation is at step 100. Time-to-target is an upper bound at that evaluation
cadence.

The reference calibration was evaluated every 500 updates and reached the
target at step 10,500 (41m20s). The archived record was evaluated every 100
updates and first verified the target at step 100 (305.6257 seconds). Both
times are upper bounds on their respective evaluation grids.

## References

1. [RelBench: A Benchmark for Deep Learning on Relational Databases.](https://arxiv.org/abs/2407.20060)
2. [Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data.](https://arxiv.org/abs/2510.06377)
3. [modded-nanoGPT.](https://github.com/KellerJordan/modded-nanogpt)
4. [modded-nanoTabPFN.](https://github.com/borawhocodess/modded-nanotabpfn)
5. [Muon optimizer.](https://github.com/KellerJordan/Muon)
