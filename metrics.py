"""Patient-level evaluation with bootstrap confidence intervals.

Everything is reported at PATIENT level. Slice-level numbers are optimistic and
are not the clinical unit of analysis; include them in an appendix at most.

With ~60 subjects and a smallest class of ~11, the intervals will be wide.
Report them anyway. A reviewer who finds an unstated interval distrusts the
whole manuscript; one who is told upfront reads the rest as credible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, f1_score, balanced_accuracy_score,
    confusion_matrix, classification_report,
)


def aggregate_to_patient(subject_ids, probs, labels, how: str = "mean"):
    """Pool slice probabilities into one prediction per subject."""
    df = pd.DataFrame(probs, columns=[f"p{i}" for i in range(probs.shape[1])])
    df["subject_id"] = subject_ids
    df["y"] = labels
    pool = df.groupby("subject_id").agg(
        {**{c: how for c in df.columns if c.startswith("p")}, "y": "first"})
    p = pool[[c for c in pool.columns if c.startswith("p")]].to_numpy()
    p = p / p.sum(axis=1, keepdims=True)
    return pool.index.to_numpy(), p, pool["y"].to_numpy()


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, classes) -> dict:
    y_pred = probs.argmax(axis=1)
    present = np.unique(y_true)
    out = {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "n_subjects": int(len(y_true)),
    }
    # macro one-vs-rest AUC, guarded because a fold may miss a rare class
    if len(present) > 1:
        try:
            out["macro_auc"] = roc_auc_score(
                y_true, probs, multi_class="ovr", average="macro",
                labels=np.arange(len(classes)))
        except ValueError:
            out["macro_auc"] = float("nan")
    else:
        out["macro_auc"] = float("nan")

    per = f1_score(y_true, y_pred, average=None,
                   labels=np.arange(len(classes)), zero_division=0)
    for i, c in enumerate(classes):
        out[f"f1_{c}"] = per[i]
        out[f"n_{c}"] = int((y_true == i).sum())
    return out


def bootstrap_ci(y_true, probs, classes, metric="macro_f1",
                 reps=1000, seed=42, alpha=0.05):
    """Patient-level resampling CI. Returns (point, lo, hi)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = compute_metrics(y_true, probs, classes)[metric]
    vals = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        v = compute_metrics(y_true[idx], probs[idx], classes)[metric]
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def report(y_true, probs, classes) -> str:
    y_pred = probs.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    txt = classification_report(y_true, y_pred, labels=np.arange(len(classes)),
                                target_names=list(classes), zero_division=0)
    return f"Confusion matrix (rows=true):\n{cm}\n\n{txt}"
