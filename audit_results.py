
from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score

RESULTS = Path("results")
CLASSES = ("mild", "moderate", "severe")

RUNS = {
    "swin_only":  ("Backbone only",            "original"),
    "classical4": ("Classical 4-D bottleneck", "original"),
    "q4_L1":      ("Quantum L=1",              "original"),
}

SECONDARY = {
    "q4_L1_new": ("Quantum L=1, refined encoding", "batchnorm"),
}


def load(name: str):
    """Return (y_true, probs, source_path) from a saved predictions file."""
    for p in [
        RESULTS / f"{name}_predictions.npz",
        Path("runs") / name / "predictions.npz",
        RESULTS / "archive" / name / "predictions.npz",
    ]:
        if p.exists():
            d = np.load(p)
            return d["y"], d["p"], p

    return None


def metrics(y, p):
    pred = p.argmax(1)

    return {
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(
            y, pred,
            average="macro",
            zero_division=0
        ),
        "macro_auc": roc_auc_score(
            y,
            p,
            multi_class="ovr",
            average="macro",
            labels=np.arange(len(CLASSES))
        ),
        "n": int(len(y)),
    }


def boot_ci(y, p, metric, reps=1000, seed=42):
    rng = np.random.default_rng(seed)
    vals = []

    for _ in range(reps):
        i = rng.integers(0, len(y), len(y))

        if len(np.unique(y[i])) < 2:
            continue

        try:
            vals.append(metrics(y[i], p[i])[metric])
        except ValueError:
            continue

    lo, hi = np.percentile(vals, [2.5, 97.5])

    return (
        metrics(y, p)[metric],
        float(lo),
        float(hi)
    )


def paired_auc_test(yA, pA, yB, pB, reps=10000, seed=42):
    """
    Bootstrap the difference in macro-AUC between two arms
    evaluated on the same subjects.
    """

    assert np.array_equal(
        yA, yB
    ), "Arms must be aligned on the same subjects"

    rng = np.random.default_rng(seed)

    obs = (
        metrics(yA, pA)["macro_auc"]
        - metrics(yB, pB)["macro_auc"]
    )

    diffs = []

    for _ in range(reps):
        i = rng.integers(0, len(yA), len(yA))

        if len(np.unique(yA[i])) < 2:
            continue

        try:
            d = (
                metrics(yA[i], pA[i])["macro_auc"]
                - metrics(yB[i], pB[i])["macro_auc"]
            )
        except ValueError:
            continue

        diffs.append(d)

    diffs = np.array(diffs)

    lo, hi = np.percentile(
        diffs,
        [2.5, 97.5]
    )

    p = 2 * min(
        (diffs <= 0).mean(),
        (diffs >= 0).mean()
    )

    return (
        obs,
        float(lo),
        float(hi),
        float(min(p, 1.0))
    )


def main():

    print("=" * 72)
    print("AUTHORITATIVE NUMBERS — COPY THESE INTO THE MANUSCRIPT")
    print("=" * 72)

    store = {}

    for name, (label, arch) in {
        **RUNS,
        **SECONDARY
    }.items():

        got = load(name)

        if got is None:
            print(
                f"\n[MISSING] {name} — "
                "no predictions file found"
            )
            continue

        y, p, path = got

        store[name] = (y, p)

        m = metrics(y, p)

        print(f"\n{label}   [{arch} architecture]")
        print(f"  source:   {path}")
        print(f"  subjects: {m['n']}")

        for k in (
            "balanced_accuracy",
            "macro_f1",
            "macro_auc"
        ):

            pt, lo, hi = boot_ci(
                y,
                p,
                k
            )

            print(
                f"  {k:18s} "
                f"{pt:.3f}  "
                f"[{lo:.3f}, {hi:.3f}]"
            )

        per = f1_score(
            y,
            p.argmax(1),
            average=None,
            labels=np.arange(len(CLASSES)),
            zero_division=0
        )

        for i, c in enumerate(CLASSES):

            print(
                f"  F1 {c:9s} "
                f"{per[i]:.3f}   "
                f"(n={int((y == i).sum())})"
            )


    print("\n" + "=" * 72)
    print("PAIRED COMPARISON — CLASSICAL VS QUANTUM")
    print("=" * 72)

    if "classical4" in store and "q4_L1" in store:

        (yc, pc) = store["classical4"]
        (yq, pq) = store["q4_L1"]

        obs, lo, hi, pval = paired_auc_test(
            yc,
            pc,
            yq,
            pq
        )

        print(
            f"  macro-AUC difference "
            f"(classical - quantum): {obs:+.3f}"
        )

        print(
            f"  bootstrap 95% CI "
            f"of the difference: "
            f"[{lo:+.3f}, {hi:+.3f}]"
        )

        print(
            f"  two-sided p "
            f"(difference = 0): {pval:.4f}"
        )

        print()

        if lo > 0 or hi < 0:

            print(
                "  -> CI excludes zero: "
                "the difference IS resolved."
            )

            print(
                "     Report this test in the manuscript."
            )

        else:

            print(
                "  -> CI contains zero: "
                "the difference is NOT resolved."
            )

            print(
                "     The difference is not statistically resolved."
            )

    else:

        print(
            "  [SKIPPED] Need both "
            "classical4 and q4_L1 predictions."
        )


    print("\n" + "=" * 72)
    print("DATASET CHECK")
    print("=" * 72)

    index_paths = [
        Path("/content/cache/index.csv"),
        Path("cache/index.csv")
    ]

    for idx in index_paths:

        if idx.exists():

            import pandas as pd

            d = pd.read_csv(idx)

            print(f"  index:   {idx}")
            print(
                f"  subjects: "
                f"{d.subject_id.nunique()}"
            )

            print(
                f"  slices:   "
                f"{len(d)}"
            )

            print(
                d.groupby("y")
                 .subject_id
                 .nunique()
                 .rename("subjects per class")
            )

            break

    else:

        print(
            "  [NOT FOUND] Cache index was not found."
        )

        print(
            "  Rebuild the cache to confirm the slice count."
        )


if __name__ == "__main__":
    main()
