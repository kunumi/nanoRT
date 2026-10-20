# Benchmark specification

modded-nanoRT measures time to a fixed zero-shot quality target on one NVIDIA
H200. This document defines the public submission contract.

## Fixed quantities

- Training bundle: the five non-holdout RelBench databases in the pinned
  relation-aware artifact.
- Holdout: `rel-avito`.
- Validation tasks: `user-visits/num_click` and `user-clicks/num_click`.
- Official metric: pooled instance-level ROC AUC over both tasks.
- Target: pooled AUROC at least `0.81621`.
- Evaluation: step 0 for diagnosis, then every 100 optimizer updates.
- Hardware: one NVIDIA H200.
- Timed budget: 45 minutes.

The exact artifact repository and immutable revision must be logged with every
submission. The artifact bundle, not a mutable HF branch, defines the data.

## Open quantities

Submissions may change the architecture, optimizer, regularization, context
sampling algorithm, numerical precision, compilation strategy, and systems
implementation. They must disclose every change needed to reproduce the run.

## Validity rules

A valid run:

1. starts all trainable model weights from a declared random seed;
2. uses exactly one H200;
3. uses the prescribed training and validation artifacts;
4. excludes the holdout database from gradient updates;
5. preserves target masking and temporal cutoffs;
6. preserves the official metric, target, evaluation cadence, and timer;
7. loads no pretrained model checkpoint;
8. reports failures that do not reach the target as right-censored runs.

The preprocessed text embeddings are part of the fixed data artifact, not a
pretrained model checkpoint.

## Metric

At each scheduled evaluation, the evaluator concatenates predictions and
labels from all eligible validation examples and computes one ROC AUC. This
weights tasks by their positive-negative pair counts and includes comparisons
between examples from different tasks.

The following diagnostics must accompany the official pooled value:

- AUROC and prevalence for each task;
- unweighted mean-task AUROC;
- exact within-task and cross-task pair contributions;
- cross-task pair weight;
- task-prior oracle;
- prediction duplicate and tie rates.

Only pooled AUROC determines eligibility. The diagnostics prevent a pooled
score from being misread as uniform per-task quality.

## Timer

The historical timer starts immediately after the first optimizer update.
Compilation, warm-up, artifact installation, and step-0 evaluation are untimed.
After it starts, the clock includes data loading, training, scheduled
validation, logging, and checkpointing.

The score is the timer value at the first eligible evaluation with pooled
AUROC at or above `0.81621`. Since evaluation is discrete, the score is an upper
bound located on the evaluation grid.

## Required submission evidence

- source snapshot or commit;
- clean/dirty state;
- artifact repository and immutable revision;
- resolved configuration;
- seed, GPU model, CUDA, PyTorch, and Python versions;
- complete metric history through the first crossing or budget exhaustion;
- timing decomposition for startup, training, and validation;
- checkpoint from the best eligible post-update evaluation, when saved.
