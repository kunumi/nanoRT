# modded-nanoRT

This repository hosts the nanoRT speedrun: train a relational foundation model
on **1 NVIDIA H200** to reach **0.81621 pooled zero-shot AUROC** on `rel-avito`.
The code builds on [Relational Transformer](https://github.com/snap-stanford/relational-transformer),
following [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt) and
[modded-nanoTabPFN](https://github.com/borawhocodess/modded-nanotabpfn).

The archived record reaches the target in **5m05.6s**, an **8.11× speedup** over
the reference's **41m20s**, using:

- **Muon** for backbone matrices, with AdamW for other parameters.
- **SIGReg** after the first block, with column grouping and weight `1e-3`.
- **Typed trilinear attention**, with static foreign-key relation states.

## Running the current record

Requirements: Linux x86-64, one NVIDIA H200, a CUDA 12.4-compatible driver,
and [pixi](https://pixi.sh/).

```bash
git clone https://github.com/kunumi/nanoRT.git
cd nanoRT
./run.sh
```

The launcher installs the locked environment, builds the Rust sampler, downloads
the [frozen artifacts](data/README.md), and starts training. W&B runs offline by
default. Override [training settings](rt/config.py) through environment variables:

```bash
NANORT_SEED=1 ./run.sh
```

## Record history

> Reach at least 0.81621 pooled zero-shot AUROC on the frozen `rel-avito`
> holdout using one NVIDIA H200.

| # | Record | Date | Description | Links | Contributor |
|---:|---|---|---|---|---|
| 1 | 41m20s | — | AdamW Relational Transformer | [protocol](docs/BENCHMARK.md) | — |
| 2 | 12m27s | 2026-05-05 | + Muon | [code](https://github.com/kunumi/nanoRT/commit/c50df4c) | Gabriel Veloso |
| 3 | 7m26s | 2026-06-24 | + column-view SIGReg | [code](https://github.com/kunumi/nanoRT/commit/49df323) | Gabriel Veloso |
| 4 | **5m05.6s** | 2026-07-28 | + typed trilinear attention | [log](https://wandb.ai/franciscokunumi-universidade-federal-de-minas-gerais/nanort-train/runs/ltlneqhn), [checkpoint](https://huggingface.co/GabrielVeloso/NanoRT/tree/main/checkpoints/rel-avito/nanort-simplicial-staticrel) | Gabriel Veloso |

The record first verifies the target at step 100: AUROC `0.8713918`,
`305.6257 s`. These are archived timings; the streamlined release needs a fresh
H200 run to measure its own time.

## Rules

New records must:

- Use one H200, a declared random seed, and no model checkpoint.
- Keep the prescribed five-database training artifact and its frozen mixture
  (`alpha=0.5`); exclude `rel-avito` from training and `rel-event` entirely.
- Preserve the validation pipeline, target masking, temporal cutoffs, leakage
  exclusions, metric, target, evaluation schedule, and timing boundary.
- Use a 45-minute budget and report missed targets as right-censored failures.

Architecture, optimization, regularization, training context sampling, precision,
and systems implementation are open. See the [full protocol](docs/BENCHMARK.md)
for artifact identity, recipe details, and submission evidence.

## Evaluation details

- **Holdout:** frozen `rel-avito` tasks `user-visits` and `user-clicks`, with
  fixed seed cap, hop radius, and text embeddings.
- **Metric:** concatenate both tasks' predictions and labels and compute one
  pooled AUROC. Only this score determines eligibility; per-task scores and
  within/cross-task contributions are diagnostics.
- **Schedule:** step 0 is untimed and cannot establish a record or select a
  checkpoint; eligible evaluations occur every 100 updates.
- **Clock:** starts after the first optimizer update and includes subsequent
  loading, training, validation, logging, and checkpointing. Setup and compile
  warm-up are untimed.
- **Reference:** evaluated every 500 updates; reaches the target at step 10,500
  (41m20s). Reported times are upper bounds on their respective evaluation grids.

## Paper

*Speedrunning Relational Foundation Model Pretraining*, accepted at the
**CODEC-FM Workshop at NeurIPS 2026**.

**Authors:** Gabriel Wedson Mendonça de Souza Veloso<sup>1,2</sup>,
Luan Borges Teodoro Reis Sena<sup>1,2</sup>,
Denis Oliveira Correa<sup>1</sup>,
Francisco Galuppo Azevedo<sup>1,2</sup>.

<sup>1</sup> Kunumi Institute, Belo Horizonte, Minas Gerais, Brazil<br>
<sup>2</sup> Universidade Federal de Minas Gerais, Belo Horizonte, Minas Gerais, Brazil

`{gabriel.veloso, luan.borges, denis, francisco}@kunumi.com`

## References

1. [RelBench](https://arxiv.org/abs/2407.20060).
2. [Relational Transformer](https://arxiv.org/abs/2510.06377).
3. [Muon](https://github.com/KellerJordan/Muon).
