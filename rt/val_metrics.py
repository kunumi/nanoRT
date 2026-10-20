"""Pooled validation AUROC, task diagnostics, and evaluation cadence."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import roc_auc_score


def eval_is_due(
    step: int,
    *,
    eval_every: int,
    eval_at_step1: bool,
    dense_until: int = 0,
    dense_freq: int = 5,
) -> bool:
    """Schedule step 0, periodic evaluations, and optional dense early evaluations."""
    if step == 0:
        return True
    period = max(1, dense_freq) if step <= dense_until else eval_every
    return step % period == 0 or (step == 1 and eval_at_step1)


def safe_auroc(labels_bin, preds) -> float:
    """AUROC, or nan when the group holds a single class."""
    if len(set(labels_bin)) < 2:
        return float("nan")
    return roc_auc_score(labels_bin, preds)


def auroc_between_ceiling(counts: dict) -> float:
    """Pooled AUROC attainable by ordering constant task scores by prevalence."""
    order = sorted(
        counts.values(),
        key=lambda c: c[0] / max(c[0] + c[1], 1),
        reverse=True,
    )
    n_pos = sum(p for p, _ in order)
    n_neg = sum(n for _, n in order)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Positives of a higher-ranked task beat negatives of every lower-ranked one.
    num = sum(
        p_i * sum(n_j for _, n_j in order[i + 1 :]) for i, (p_i, _) in enumerate(order)
    )
    num += 0.5 * sum(p * n for p, n in order)  # within-task ties
    return num / (n_pos * n_neg)


def pooled_auroc_parts(preds, labels, task_ids) -> dict:
    """Split pooled AUROC into within-task and cross-task Mann–Whitney contributions."""
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    task_ids = np.asarray(task_ids)
    tasks = np.unique(task_ids)
    pos = {t: np.sort(preds[(task_ids == t) & (labels == 1)]) for t in tasks}
    neg = {t: np.sort(preds[(task_ids == t) & (labels == 0)]) for t in tasks}
    n_pos = sum(v.size for v in pos.values())
    n_neg = sum(v.size for v in neg.values())
    nan = float("nan")
    if n_pos == 0 or n_neg == 0:
        return {
            "val/auroc_pooled_within": nan,
            "val/auroc_pooled_cross": nan,
            "val/pooled_cross_weight": nan,
        }

    w_within = u_within = w_cross = u_cross = 0.0
    for g in tasks:
        p = pos[g]
        if p.size == 0:
            continue
        for h in tasks:
            n = neg[h]
            if n.size == 0:
                continue
            lo = np.searchsorted(n, p, side="left")  # negatives strictly below
            hi = np.searchsorted(n, p, side="right")  # ... and the tied ones
            u = float(lo.sum()) + 0.5 * float((hi - lo).sum())
            w = float(p.size * n.size)
            if g == h:
                w_within, u_within = w_within + w, u_within + u
            else:
                w_cross, u_cross = w_cross + w, u_cross + u

    return {
        "val/auroc_pooled_within": u_within / w_within if w_within else nan,
        "val/auroc_pooled_cross": u_cross / w_cross if w_cross else nan,
        "val/pooled_cross_weight": w_cross / (n_pos * n_neg),
    }


def decompose_val_auroc(preds, labels_bin, task_ids, task_names) -> dict:
    """Report task AUROCs, prevalence, pooled contributions, and score ties."""
    preds = np.asarray(preds)
    labels = np.asarray(labels_bin)
    task_ids = np.asarray(task_ids)
    out, per_task, counts = {}, [], {}

    for tid in np.unique(task_ids):
        sel = task_ids == tid
        name = task_names.get(int(tid), f"col{int(tid)}")
        y, p = labels[sel], preds[sel]
        n_pos, n_neg = int(y.sum()), int((1 - y).sum())
        counts[name] = (n_pos, n_neg)

        auroc_t = safe_auroc(y.tolist(), p)
        out[f"val/auroc/{name}"] = auroc_t
        out[f"val/base_rate/{name}"] = n_pos / max(n_pos + n_neg, 1)
        out[f"val/n_pos/{name}"] = float(n_pos)
        out[f"val/n_neg/{name}"] = float(n_neg)
        if not math.isnan(auroc_t):
            per_task.append(auroc_t)

    out["val/auroc_mean_per_task"] = (
        sum(per_task) / len(per_task) if per_task else float("nan")
    )
    task_prior_oracle = auroc_between_ceiling(counts)
    out["val/auroc_task_prior_oracle"] = task_prior_oracle
    # Backward-compatible name used by the initial audit runs.
    out["val/auroc_between"] = task_prior_oracle
    out.update(pooled_auroc_parts(preds, labels, task_ids))
    out["val/n_tasks"] = float(len(counts))
    # Count task IDs missing from the name map.
    out["val/n_unmapped_tasks"] = float(
        sum(1 for tid in np.unique(task_ids) if int(tid) not in task_names)
    )
    n_unique = int(np.unique(preds).size)
    out["val/pred_unique"] = float(n_unique)
    out["val/pred_duplicate_fraction"] = 1.0 - n_unique / max(preds.size, 1)

    pos_scores = preds[labels == 1]
    neg_scores = preds[labels == 0]
    if pos_scores.size and neg_scores.size:
        pos_values, pos_counts = np.unique(pos_scores, return_counts=True)
        neg_values, neg_counts = np.unique(neg_scores, return_counts=True)
        _, pos_idx, neg_idx = np.intersect1d(
            pos_values, neg_values, assume_unique=True, return_indices=True
        )
        tied_pairs = np.multiply(
            pos_counts[pos_idx].astype(np.int64),
            neg_counts[neg_idx].astype(np.int64),
        ).sum()
        tie_rate = float(tied_pairs) / float(pos_scores.size * neg_scores.size)
    else:
        tie_rate = float("nan")
    out["val/pred_pos_neg_tie_rate"] = tie_rate
    # Backward-compatible key; now defined by the AUROC-relevant pairwise rate.
    out["val/pred_tie_frac"] = tie_rate
    return out
