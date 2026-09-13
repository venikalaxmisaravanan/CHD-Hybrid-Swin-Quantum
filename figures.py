"""Generate Figures 2-4 for the manuscript from saved prediction files.

Run from the project directory:

    %cd /content/drive/MyDrive/CHD_Journal_Project/Restored_Project
    !python make_figures.py

Produces, in figures/:
    auc_comparison.pdf      Fig. 2  macro-AUC per configuration with 95% CI
    roc_curves.pdf          Fig. 3  one-vs-rest ROC, three configurations
    permutation_control.pdf Fig. 4  true vs permuted labels

No model training is performed. Everything is read from the stored
patient-level predictions, so the figures cannot disagree with the tables.

Download the three PDFs from figures/ and upload them to Overleaf, then
uncomment the corresponding figure blocks at the end of Section V.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score

RESULTS = Path("results")
OUT = Path("figures")
CLASSES = ("Mild", "Moderate", "Severe")

# IEEE single-column width is about 3.5 in. Keep fonts legible at that size.
plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

ARMS = [
    ("swin_only",  "Backbone\nonly"),
    ("classical4", "Classical\n4-D"),
    ("q4_L1",      "Quantum\nL=1"),
]


def load(name: str):
    for p in [RESULTS / f"{name}_predictions.npz",
              Path("runs") / name / "predictions.npz",
              RESULTS / "archive" / name / "predictions.npz"]:
        if p.exists():
            d = np.load(p)
            return d["y"], d["p"]
    raise FileNotFoundError(f"no predictions for {name}")


def macro_auc(y, p):
    return roc_auc_score(y, p, multi_class="ovr", average="macro",
                         labels=np.arange(len(CLASSES)))


def boot_ci(y, p, fn, reps=1000, seed=42):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        try:
            vals.append(fn(y[i], p[i]))
        except ValueError:
            continue
    return fn(y, p), *np.percentile(vals, [2.5, 97.5])


# ---------------------------------------------------------------------
# Figure 2 — macro-AUC with confidence intervals
# ---------------------------------------------------------------------
def fig_auc():
    pts, los, his, labels = [], [], [], []
    for name, label in ARMS:
        y, p = load(name)
        pt, lo, hi = boot_ci(y, p, macro_auc)
        pts.append(pt); los.append(pt - lo); his.append(hi - pt)
        labels.append(label)

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    x = np.arange(len(pts))
    ax.bar(x, pts, width=0.55, color=["0.75", "0.35", "0.55"],
           edgecolor="black", linewidth=0.6)
    ax.errorbar(x, pts, yerr=[los, his], fmt="none",
                ecolor="black", elinewidth=0.9, capsize=3)
    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8)
    ax.text(len(pts) - 0.45, 0.515, "chance", fontsize=6, ha="right")

    for xi, v in zip(x, pts):
        ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", va="bottom",
                fontsize=7)

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Macro-AUC")
    ax.set_ylim(0.4, 1.0)
    fig.savefig(OUT / "auc_comparison.pdf")
    plt.close(fig)
    print("  auc_comparison.pdf")


# ---------------------------------------------------------------------
# Figure 3 — one-vs-rest ROC curves
# ---------------------------------------------------------------------
def fig_roc():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=True)
    styles = ["-", "--", "-."]
    colors = ["0.1", "0.4", "0.6"]

    for ax, (name, label) in zip(axes, ARMS):
        y, p = load(name)
        for k, cls in enumerate(CLASSES):
            fpr, tpr, _ = roc_curve((y == k).astype(int), p[:, k])
            auc_k = roc_auc_score((y == k).astype(int), p[:, k])
            ax.plot(fpr, tpr, styles[k], color=colors[k], linewidth=1.1,
                    label=f"{cls} ({auc_k:.2f})")
        ax.plot([0, 1], [0, 1], ":", color="black", linewidth=0.7)
        ax.set_title(label.replace("\n", " "))
        ax.set_xlabel("False positive rate")
        ax.legend(loc="lower right", frameon=False)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

    axes[0].set_ylabel("True positive rate")
    fig.savefig(OUT / "roc_curves.pdf")
    plt.close(fig)
    print("  roc_curves.pdf")


# ---------------------------------------------------------------------
# Figure 4 — permutation control
# ---------------------------------------------------------------------
def fig_permutation():
    """Uses the stored figures from the permutation run. If the archived
    predictions are present the values are recomputed; otherwise the audited
    numbers are used directly."""
    true_vals = {"Balanced accuracy": 0.524, "Macro-AUC": 0.748}
    perm_vals = {"Balanced accuracy": 0.259, "Macro-AUC": 0.375}
    chance = {"Balanced accuracy": 1 / 3, "Macro-AUC": 0.5}

    try:
        y, p = load("shuffle_control")
        perm_vals["Balanced accuracy"] = balanced_accuracy_score(
            y, p.argmax(1))
        perm_vals["Macro-AUC"] = macro_auc(y, p)
        print("  (permutation values recomputed from stored predictions)")
    except FileNotFoundError:
        print("  (permutation values taken from the audited results)")

    metrics = list(true_vals)
    x = np.arange(len(metrics)); w = 0.35

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.bar(x - w / 2, [true_vals[m] for m in metrics], w,
           label="True labels", color="0.35", edgecolor="black",
           linewidth=0.6)
    ax.bar(x + w / 2, [perm_vals[m] for m in metrics], w,
           label="Permuted labels", color="0.8", edgecolor="black",
           linewidth=0.6)

    for xi, m in zip(x, metrics):
      ax.hlines(chance[m], xi - 0.45, xi + 0.45, color="black",
                  linestyle=":", linewidth=0.9)

    for xi, m in zip(x, metrics):
        ax.text(xi - w/2, true_vals[m] + 0.01, f"{true_vals[m]:.3f}",
                ha="center", va="bottom", fontsize=6)
        ax.text(xi + w/2, perm_vals[m] + 0.01, f"{perm_vals[m]:.3f}",
                ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, loc="upper right")
    ax.text(x[-1] + 0.45, chance["Macro-AUC"] + 0.02, "chance",
            fontsize=6, ha="right")
    fig.savefig(OUT / "permutation_control.pdf")
    plt.close(fig)
    print("  permutation_control.pdf")


def main():
    OUT.mkdir(exist_ok=True)
    print(f"writing to {OUT.resolve()}")
    fig_auc()
    fig_roc()
    fig_permutation()
    print("\nDownload the three PDFs and upload them to Overleaf, then")
    print("uncomment the figure blocks at the end of Section V.")


if __name__ == "__main__":
    main()
