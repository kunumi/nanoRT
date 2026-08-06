# nanoRT: Speedrunning Zero-Shot Relational Foundation Model Pretraining on One GPU

Working draft for a six-page CODEC-FM @ NeurIPS 2026 submission. The text below is
intentionally limited to claims that do not depend on the pending experiments.
`TODO-RESULT` marks passages that must not be completed from memory or a single run.

## 1. Introduction

Relational databases organize much of the world's operational data, yet most machine
learning pipelines still reduce them to manually designed flat tables. This reduction
requires task-specific joins, aggregations, and feature engineering, and it discards
the schema that describes how entities interact. Recent benchmarks such as
[RelBench](https://arxiv.org/abs/2407.20060) have made end-to-end learning on relational
databases easier to study, while the
[Relational Transformer (RT)](https://arxiv.org/abs/2510.06377) suggests that a single
schema-aware model can transfer across databases and answer previously unseen
relational prediction tasks. These models offer a path away from bespoke pipelines,
but their training cost and system complexity make rapid, reproducible iteration
difficult.

We study this problem through a speedrun: a constrained search for the shortest time
needed to reach a quality threshold fixed before the interventions under comparison
were evaluated in nanoRT. The speedrun fixes the model family, training mixture,
held-out target domain, validation artifact, hardware, and metric, while permitting changes to the
training algorithm and architecture. This time-to-quality framing complements
training-at-scale studies. It asks not only whether a method eventually reaches a
strong score, but whether it makes relational foundation model research accessible on
a single accelerator and supports a tight cycle of hypothesis, implementation, and
verification.

This framing also turns efficient pretraining into open research infrastructure.
Lowering the cost of a complete experiment allows independent contributors to replay
records, isolate one change, and add improvements to a shared lineage. nanoRT does not
propose a decentralized training algorithm; it provides a reproducible, accessible
unit of relational foundation model development that fits within a broader open and
collaborative ecosystem.

The resulting benchmark, **nanoRT**, trains on five relational databases and evaluates
zero-shot on a frozen `rel-avito` artifact using one NVIDIA H200. The target domain is
excluded from gradient-based training, and the official objective is the historical
pooled instance-level AUROC target of `0.81621`. Pooling is deliberate: RT is a single
schema-aware predictor, and the benchmark evaluates its ranking over the empirical
population of validation examples, including score alignment across task contexts.
Because this estimand can contain both within-task discrimination and between-task
separation, we accompany it with per-task AUROC and an exact decomposition of
within- and cross-task pairs. The additional diagnostics clarify the source of a
record without changing the pre-existing rule after observing results.

We investigate three complementary interventions. First, **Muon** changes the
optimization geometry of matrix-valued parameters while retaining AdamW for
embeddings, biases, and other non-matrix parameters. Second, **SIGReg** regularizes the
geometry of intermediate representations during training, targeting collapse without
adding inference-time computation. Third, we introduce a relation-conditioned
neighbor-attention operator in which a query, a neighboring cell, and the edge type
interact through a factorized trilinear score and relation-modulated value. We refer to
this operator as *simplicial attention* in the implementation, while distinguishing it
from architectures that explicitly construct and attend over simplicial complexes.

Our study is organized around the following contributions:

1. We specify an open, single-H200 time-to-quality protocol for zero-shot relational
   pretraining, including a frozen gradient holdout, artifact manifests, and a
   reproducible timing boundary.
2. We evaluate how an optimizer, a representation-geometric regularizer, and a
   relation-conditioned inductive bias contribute separately and jointly to
   time-to-quality.
3. We make initialization an explicit control through `AUROC@0`, separating
   architectural or schema priors from gains acquired by optimization.
4. We retain the benchmark's historical pooled metric while exposing its within- and
   cross-task components, providing a more auditable interpretation of speed records.

Under the historical leaderboard protocol, the current simplicial configuration
records its first eligible verification above the fixed target in **305.63 seconds
(5m05.6s)**, a 1.46x improvement over the 7m26s SIGReg record and an 8.83x
improvement over the 45-minute baseline. The corresponding step-100 pooled AUROC
is `0.8713918`.

The historical result is archived as W&B run
[`ltlneqhn`](https://wandb.ai/franciscokunumi-universidade-federal-de-minas-gerais/nanort-train/runs/ltlneqhn),
with its checkpoint under
[`nanort-simplicial-staticrel`](https://huggingface.co/GabrielVeloso/NanoRT/tree/main/checkpoints/rel-avito/nanort-simplicial-staticrel).

<!-- TODO-RESULT: Add paired verification seeds from the pre-registered Polyaxon
sweep before submission. -->

## 2. Background and Related Work

### 2.1 Relational learning and relational foundation models

Learning directly from multi-table data has been approached through propositionalization,
graph neural networks, and models specialized for database schemas. RelBench provides
a standardized collection of relational databases, temporal splits, and predictive
tasks, allowing relational methods to be compared without reducing every dataset to a
single handcrafted table. Its task construction and temporal evaluation also make
data leakage a first-class concern rather than an implementation detail.

RT treats a sampled relational neighborhood as a structured collection of cells and
alternates attention patterns that reflect feature and relational connectivity. Text
embeddings of table, column, and relation metadata give the model a shared interface
across heterogeneous schemas and enable zero-shot evaluation on a database excluded
from pretraining. nanoRT keeps this schema-aware setting and studies a different axis:
the time required to obtain a fixed level of downstream quality. Concurrent work such
as [PluRel](https://arxiv.org/abs/2602.04029) further illustrates the growing interest
in pretrained models that operate across relational datasets; our focus is the
optimization and systems regime in which such models can be developed with modest
compute.

### 2.2 Time-to-quality benchmarks and training speedruns

End-to-end training time is not determined by FLOPs alone. Kernel efficiency, data
movement, compilation, evaluation frequency, optimizer state, and the number of
updates needed to reach useful quality all matter.
[DAWNBench](https://arxiv.org/abs/1806.01427) helped popularize
time-to-accuracy as an end-to-end criterion, while
[MLPerf Training](https://arxiv.org/abs/1910.01500) formalized benchmark-specific
quality targets and controlled submissions. These efforts motivate nanoRT's fixed
target and explicit separation of active training, evaluation, compilation, and wall
clock time.

Small-compute training studies have shown that fixed budgets can be scientifically
productive rather than merely restrictive.
[Cramming](https://arxiv.org/abs/2212.14034) asks how much language-model pretraining
can be accomplished on a single GPU in one day, revealing that many gains arise from
the interaction of architecture, optimization, and implementation. The open
[modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt) project turns this
idea into a continuously improving record lineage. In tabular foundation models,
[nanoTabPFN](https://arxiv.org/abs/2511.03634) and
[Speedrunning Tabular Foundation Model Pretraining](https://arxiv.org/abs/2606.03681)
apply a closely related time-to-quality methodology to compact transformer
pretraining. nanoRT transfers this experimental style to relational databases, where
ragged neighborhoods, schema-conditioned inputs, database mixing, and temporal
holdouts introduce distinct bottlenecks and failure modes.

A speedrun is useful only when its target remains stable. For this reason, we preserve
the original nanoRT validation artifact and pooled AUROC rule across the record
lineage. We treat richer metrics as diagnostics rather than retroactive replacements,
and report the interval between validation events in which a threshold is crossed.
This matters for short runs: a coarse evaluation schedule can otherwise be mistaken
for an optimization improvement.

### 2.3 Optimization and representation geometry

AdamW is a strong default for transformer training, but it applies elementwise updates
to parameters whose roles are often matrix-valued.
[Muon](https://github.com/KellerJordan/Muon) instead constructs
approximately orthogonalized updates for eligible matrices and is commonly paired
with AdamW for embeddings, biases, and scale parameters. The nanoRT speedrun tests
whether this parameter-aware update geometry reduces time-to-quality in a relational
transformer, rather than assuming that gains observed in language models transfer
unchanged.

Optimization speed also depends on the geometry of learned representations. Highly
anisotropic or collapsed features can reduce the effective dimension available to a
downstream prediction head.
[SIGReg](https://openreview.net/forum?id=IYnHchzvYB) uses sketched one-dimensional
projections and an Epps--Pulley statistic to encourage intermediate representations
toward an isotropic Gaussian geometry. In nanoRT, relational samples do not form a
conventional dense tabular grid: rows have ragged neighborhoods, cells carry schema metadata, and
the network contains multiple aggregation streams. We therefore treat the choice of
SIGReg view and layer as part of the method, and ablate column- and row-oriented views
separately. This regularizer changes training but adds no operation to the deployed
forward pass.

### 2.4 Relation-aware and higher-order attention

Standard dot-product attention scores a pair consisting of a query and a key. A
relational edge provides a third object: its type specifies how the neighboring entity
is connected to the target. Prior work injects typed relations through
relation-specific message transformations, as in
[R-GCN](https://arxiv.org/abs/1703.06103), or through key/value representations in
[relation-aware self-attention](https://aclanthology.org/N18-2074/). Our operator uses
the relation in both compatibility and content. For head
\(h\), query \(q_i^h\), neighboring cell key \(k_j^h\), and relation key \(r_{ij}^h\),
the unnormalized score has the factorized trilinear form

\[
a_{ij}^h = \sum_d q_{i,d}^h k_{j,d}^h r_{ij,d}^h,
\]

and the message value is modulated elementwise by a relation value. This construction
is inexpensive relative to a dense tensor interaction and makes the edge type affect
individual messages rather than only adding a scalar logit bias.

The three-way interaction is related in spirit to work on
[2-simplicial attention](https://arxiv.org/abs/1909.00668), which extends attention
beyond pairs, and to efficient higher-order designs such as
[Fast and Simplex](https://arxiv.org/abs/2507.02754). The objects are nevertheless
different. Those models reason over tuples of entities or explicit higher-order
structures; nanoRT's third factor is the typed database relation attached to a
query-neighbor edge. We use the term *simplicial* to emphasize the ternary interaction,
not to claim construction of a simplicial complex.

## 3. Protocol notes ready to promote into the paper

This section is working material for the future Methods section. It is included now
because it does not depend on the final results.

### 3.1 What is RT?

RT is a transformer for sampled relational neighborhoods. An example begins at a seed
row associated with a prediction task. A sampler follows database relations to collect
a bounded, temporally valid neighborhood, masks the target, and represents cells using
their values together with textual table, column, and relation metadata. Specialized
attention masks expose different structural views of this neighborhood, allowing the
network to exchange information within records and along foreign-key connections. A
shared model is pretrained across databases and applied to the held-out schema without
gradient updates.

Before finalizing this paragraph, verify every architectural term against the current
branch and add a diagram of: seed row → temporal neighborhood sampler → cell encoder →
feature/relational blocks → masked-cell prediction.

### 3.2 What is a speedrun?

In nanoRT, a speedrun is a controlled optimization problem, not an unrestricted demo.
It fixes the hardware, data stream, target-domain artifact, metric, quality threshold,
and maximum budget, then minimizes the time at which an eligible post-update
evaluation first reaches the threshold. Compilation and initialization are recorded
separately; the historical active-training timer remains available for record
continuity. A valid submission must also reproduce the crossing across pre-declared
seeds and expose the evaluation interval containing it.

The benchmark is closer to a race against a frozen quality constraint than to a claim
that one model is universally better. Its scientific value comes from the record
lineage: each change is tested against the previous configuration under the same
constraints, making negative and positive interventions comparable.

### 3.3 Holdout wording

`rel-avito` is a gradient holdout: its examples are absent from the five-database
training mixture and are never used to update parameters. Validation seeds, temporal
extraction rules, text embeddings, and the three-hop artifact are frozen. Because the
community has repeatedly inspected validation results while developing nanoRT, we do
not describe this domain as a statistically untouched test set. It is a fixed
zero-shot development benchmark; final external validity should be assessed on an
additional domain selected before evaluation.

### 3.4 Metric wording

The official score concatenates predictions and labels across the eligible binary
tasks and computes one AUROC. Equivalently, it estimates the probability that a
random positive example from the empirical validation population is ranked above a
random negative example, with half credit for ties. This instance-level weighting is
appropriate to the challenge's single-predictor, global-ranking objective and keeps
the original threshold comparable across records. It is not a macro-average over
tasks. We therefore report macro and per-task AUROC, base rates, score distributions,
and the exact contribution of within- and cross-task pairs alongside the official
number.

### 3.5 Initialization wording

We evaluate every arm at step zero, before any parameter update. This evaluation is
outside the official timer and cannot trigger a record. `AUROC@0` controls for
schema- and architecture-induced separation, verifies paired initialization, and
allows us to distinguish absolute quality from quality acquired through optimization.

## 4. Result-dependent placeholders

- `TODO-RESULT`: verified record table with one row per cumulative intervention.
- `TODO-RESULT`: paired-seed curves and crossing intervals.
- `TODO-RESULT`: `AUROC@0` distribution and narrative branch.
- `TODO-RESULT`: pooled AUROC decomposition into within/cross-task pairs.
- `TODO-RESULT`: simplicial controls (text, constant, permutation, bilinear).
- `TODO-RESULT`: SIGReg geometry measurement and row/column ablation.
- `TODO-RESULT`: throughput, memory, compilation, evaluation, and wall-clock costs.
- `TODO-RESULT`: external holdout if completed.

## 5. Citation queue

Convert these links to BibTeX keys when the LaTeX project is created:

- [RelBench](https://arxiv.org/abs/2407.20060)
- [Relational Transformer](https://arxiv.org/abs/2510.06377)
- [PluRel](https://arxiv.org/abs/2602.04029)
- [Speedrunning Tabular Foundation Model Pretraining](https://arxiv.org/abs/2606.03681)
- [nanoTabPFN](https://arxiv.org/abs/2511.03634)
- [Cramming](https://arxiv.org/abs/2212.14034)
- [MLPerf Training](https://arxiv.org/abs/1910.01500)
- [DAWNBench analysis](https://arxiv.org/abs/1806.01427)
- [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt)
- [Muon](https://github.com/KellerJordan/Muon)
- [Muon is Scalable for LLM Training](https://openreview.net/pdf/1bff28e2c4dc62d54931563338d4dd25112a0203.pdf)
- [SIGReg](https://openreview.net/forum?id=IYnHchzvYB)
- [LeJEPA](https://arxiv.org/abs/2511.08544)
- [2-simplicial attention](https://arxiv.org/abs/1909.00668)
- [Fast and Simplex](https://arxiv.org/abs/2507.02754)
- [R-GCN](https://arxiv.org/abs/1703.06103)
- [Relation-aware self-attention](https://aclanthology.org/N18-2074/)
