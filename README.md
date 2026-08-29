# HVSMR-2.0 — Quantum-Inspired Swin Transformer for CHD Severity Classification

Scaffold for a three-class (mild / moderate / severe) severity classification
study on the HVSMR-2.0 cardiac MR dataset, with a simulated parameterized
quantum circuit as a feature bottleneck and Grad-CAM for interpretability.

**This code has been syntax-checked but never executed.** The container it was
written in has no network access, so `torch`, `timm`, `pennylane` and `nibabel`
could not be installed or run. Expect to fix small things on the first run —
particularly the timm attribute paths in `explain.py`.

---

## 1. Setup

```bash
pip install torch torchvision timm pennylane nibabel \
            scikit-learn pandas grad-cam matplotlib seaborn
```

Download HVSMR-2.0 from figshare (open access, CC BY, no application required):
https://doi.org/10.6084/m9.figshare.c.7074755.v2

Arrange as:

```
hvsmr2/
  images/    subject NIfTI volumes
  labels/    matching segmentation masks
metadata.csv
```

`metadata.csv` needs two columns:

```csv
subject_id,severity
pat0,severe
pat1,mild
...
```

Build this from the dataset documentation. The published class distribution is
mild 12, moderate 11, severe 37.

---

## 2. Run order

```bash
# 1. Sanity check the index before training anything
python -c "from config import CFG; from data import build_index; build_index(CFG)"

# 2. One configuration end to end
python train.py --bottleneck quantum --n_qubits 4 --n_layers 3 --epochs 3

# 3. Full ablation once step 2 is clean
bash run_ablation.sh
```

Step 1 prints subjects per class and total slices. If subject counts don't match
12 / 11 / 37, your metadata mapping is wrong — fix it before training.

Start with `--epochs 3` to confirm the pipeline runs, then raise it.

---

## 3. Design decisions worth defending in the paper

| Decision | Reason |
|---|---|
| Patient-level folds | Slice-level splitting puts the same heart in train and test. This is the single fastest way to lose credibility. |
| Percentile normalisation | MRI has no absolute intensity scale. HU windowing is a CT operation and does not apply here. |
| Mask-guided slice selection | Avoids training on empty slices above and below the heart. |
| 2.5D stacks | Gives a principled reason for 3-channel input to an ImageNet backbone rather than replicating grayscale. |
| No horizontal flip | Left-right orientation is diagnostic in CHD (dextrocardia, situs anomalies, TGA). Flipping corrupts the label. |
| Class weights over subjects | Subjects with more slices would otherwise dominate the weighting. |
| Mean pooling to patient | Max pooling over a long volume amplifies single-slice false positives. |
| 5-fold CV | n=60 is far too small for a single held-out split to be stable. |

---

## 4. Known pitfalls

**Grad-CAM target layer.** `model.backbone.layers[-1].blocks[-1].norm1` varies
across timm versions. Run `print(model.backbone)` and confirm before trusting
any heatmap. A wrong layer produces a plausible-looking but meaningless map —
it fails silently, not loudly.

**Parameter-shift is slow.** Two extra circuit evaluations per parameter per
step. At 6 qubits × 3 layers that is 18 parameters, so 36 extra evaluations per
batch. Budget accordingly, and expect the quantum arms to take substantially
longer than the classical ones.

**Volume caching.** `HVSMRSliceDataset` caches four volumes in memory. With
`num_workers > 0` each worker keeps its own cache. Reduce workers if you hit
memory limits.

**A fold may miss a class.** With 11 subjects in the smallest class, some folds
will have very few. `compute_metrics` guards against this and returns NaN for
AUC rather than crashing, but check the per-fold output rather than only the
pooled numbers.

---

## 5. What to report

Patient level, always:

- Balanced accuracy, macro-F1, macro one-vs-rest AUC, each with bootstrap 95% CI
- Per-class F1 with the number of subjects stated beside it
- Confusion matrix
- Ablation table: quantum vs. matched classical bottleneck at each width
- Grad-CAM overlap precision against the `mask_frac` chance baseline

With 60 subjects the intervals will be wide and a small accuracy difference will
not be statistically separable. Say so directly. An honest small study is
publishable; an overclaimed one is not.
