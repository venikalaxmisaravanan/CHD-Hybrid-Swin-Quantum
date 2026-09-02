# CHD-Hybrid-Swin-Quantum

## Hybrid Swin Transformer and Quantum Feature Learning for Congenital Heart Disease Severity Classification

A research implementation of a hybrid classical–quantum deep learning framework for
cardiac MRI analysis. The project combines a pretrained **Swin Transformer** with a
compact **4-qubit parameterized quantum feature layer** and a classical
classification head.

The implementation is evaluated using **patient-level cross-validation** on the
HVSMR-2.0 cardiac MRI dataset.

> **Project status:** Experimental validation in progress. Classical baselines and
> the quantum L=1 experiment are complete. L=2 and L=3 are being re-evaluated after
> correcting the quantum-layer initialization. L=1 must be rerun under the corrected
> initialization before the final depth ablation is reported.

---

## 1. Project Overview

The objective is to investigate whether a quantum feature transformation provides
useful information beyond a conventional low-dimensional classical bottleneck.

```text
Cardiac MRI
     │
     ▼
Preprocessing / Cached MRI slices
     │
     ▼
Swin Transformer Backbone
     │
     ▼
768-dimensional feature representation
     │
     ▼
4-dimensional latent projection
     │
     ├─────────────── Classical 4-D bottleneck
     │
     └─────────────── Quantum Feature Layer
                              │
                              ▼
                         4 Qubits
                              │
                    Angle Embedding (Rx)
                              │
                    BasicEntanglerLayers × L
                              │
                    Pauli-Z measurements
                              │
                              ▼
                     Classification Head
                              │
                              ▼
                    Mild / Moderate / Severe
```

CHD severity is treated as a **three-class classification problem**:

* Class 0: Mild
* Class 1: Moderate
* Class 2: Severe

---

## 2. Research Questions

### RQ1 — Does the quantum feature transformation provide an advantage over a classical bottleneck?

Configurations compared:

1. Swin Transformer only
2. Swin Transformer + classical 4-D projection
3. Swin Transformer + 4-qubit quantum feature layer

The classical 4-D projection is the critical control. It holds bottleneck width
fixed while removing the quantum structure, so the comparison asks whether the
quantum transformation adds value beyond compression alone. Without this arm, any
improvement over the plain backbone is uninterpretable.

### RQ2 — How does quantum circuit depth affect performance?

Four qubits with varying entangling depth: **L = 1, 2, 3**. Only circuit depth
changes across this ablation.

---

## 3. Dataset

### HVSMR-2.0

Cropped HVSMR-2.0 cardiac MRI data with clinical metadata.

| Class     | Subjects |
| --------- | -------: |
| Mild      |       12 |
| Moderate  |       11 |
| Severe    |       36 |
| **Total** |   **59** |

The indexed dataset contains **59 subjects**. The nominal dataset size is 60; one
subject is absent from the current index. This is stated rather than assumed away.

All experiments are performed at the **subject level** so that slices from the same
patient never appear across different cross-validation folds.

---

## 4. Data Processing

1. Load cropped cardiac MRI volumes (NIfTI).
2. Associate scans with clinical metadata by exact subject-ID matching.
3. Percentile-clip intensities (1st–99th) and rescale to [0, 1]. MRI has no
   absolute intensity scale, so normalization is per volume. Hounsfield windowing
   does not apply.
4. Retain only slices intersecting annotated cardiac structures.
5. Build 2.5D inputs: three contiguous axial slices mapped to three channels.
6. Resize to **224 × 224**; standardize with ImageNet channel statistics.
7. Write a slice-level memmap cache with subject identifiers preserved.

> **Subject-ID matching.** An earlier version used substring globbing, which caused
> `pat1` to match `pat10`–`pat19` and silently pair five subjects with the wrong
> volumes. Matching is now anchored and rejects ambiguous IDs. Any result predating
> this fix is invalid.

The cache is stored outside the Git repository:

```text
/content/cache/
├── stacks.npy      (~2 GB, float16)
└── index.csv
```

Current cache: approximately **7,814 slices** from **59 subjects**.

---

## 5. Patient-Level Cross-Validation

**5-fold patient-level cross-validation**, stratified by severity. Folds are
constructed over subject IDs, never over individual slices.

```text
Training subjects → Model training → Held-out subjects → Fold predictions → Fold metrics
```

Held-out predictions are aggregated across all five folds. Slice-level probabilities
are pooled to patient level by mean before any metric is computed.

---

## 6. Model Architecture

### 6.1 Swin Transformer Backbone

Pretrained **Swin Transformer Tiny** (`swin_tiny_patch4_window7_224`, ImageNet
weights, fully fine-tuned), producing a 768-dimensional pooled embedding.

### 6.2 Classical 4-D Bottleneck (control arm)

```text
Swin feature → 768-D → Classical projection → 4-D → Classification head
```

Determines whether performance changes are caused by dimensionality reduction alone.

### 6.3 Quantum Feature Layer

```text
4-D latent → AngleEmbedding → BasicEntanglerLayers × L → Pauli-Z expectations → 4-D
```

Implemented with **PennyLane** via `qml.qnn.TorchLayer`, simulated on
`default.qubit`.

---

## 7. Quantum Circuit

* Qubits: **4**
* Encoding: **AngleEmbedding** (Rx rotations)
* Entangling block: **BasicEntanglerLayers** (ring-topology CNOTs)
* Measurement: **Pauli-Z expectation values**
* Depths evaluated: **L = 1, 2, 3**

Circuit parameters are trained jointly with the classical network.

### Quantum initialization

Rotation parameters are initialized from a small normal distribution:

```python
torch.nn.init.normal_(t, mean=0.0, std=0.1)
```

Introduced after diagnostics showed severe compression of quantum output variance at
deeper circuit depths (see Section 14).

---

## 8. Optimization

**AdamW.** Base learning rate `1e-4` for all classical components.

Quantum circuit parameters use a separate rate:

```text
Classical parameters: 1e-4
Quantum parameters:   1e-4 × 100 = 1e-2
```

The separation is necessary because circuit rotation angles are parameterized on a
2π scale while network weights operate two orders of magnitude smaller; a shared
rate leaves the angles effectively frozen.

**This applies only to circuit rotation angles, which have no counterpart in the
classical arm.** Backbone, projection, and head remain at `1e-4` in every
configuration, so the classical-versus-quantum comparison stays matched.

---

## 9. Quantum Differentiation

```python
diff_method="backprop"
```

Changed from parameter-shift, which failed on broadcasted inputs in this
configuration. For a simulated circuit, backpropagation through the state vector
yields gradients mathematically identical to parameter-shift.

**Manuscript note:** the parameter-shift rule would be required on physical
hardware, where backpropagation through a quantum device is not possible. The
methods section must describe what was actually used, with that distinction stated.

---

## 10. Hardware and Runtime

Google Colab with GPU acceleration for classical components. The PennyLane simulator
runs on CPU; the quantum layer is pinned to CPU and the 4-D bottleneck vector
crosses the device boundary inside `forward()`. Autograd tracks the transfer, so
gradients flow normally. A 4-qubit state has 16 amplitudes, so the CPU cost is
negligible beside the backbone.

---

## 11. Training Configuration

| Parameter               | Current value           |
| ----------------------- | ----------------------- |
| Backbone                | Swin Transformer Tiny   |
| Input size              | 224 × 224 (2.5D stack)  |
| Bottleneck size         | 4                       |
| Number of qubits        | 4                       |
| Quantum layers          | 1 / 2 / 3               |
| Quantum weight init     | N(0, 0.1)               |
| Base learning rate      | 1e-4                    |
| Quantum LR multiplier   | ×100                    |
| Optimizer               | AdamW                   |
| Loss                    | CrossEntropyLoss (class-weighted) |
| Cross-validation        | 5-fold patient-level    |
| Training epochs         | 8                       |
| Quantum simulator       | PennyLane default.qubit |
| Quantum differentiation | Backpropagation         |

Finalized after all depth experiments complete.

---

## 12. Experimental Configurations

`--bottleneck` accepts `none`, `classical`, or `quantum`. These are *modes*, not run
names; the run name (`swin_only`, `classical4`, `q4_L1`, …) is derived automatically.

### Baseline 1 — Swin only

```bash
python train.py --bottleneck none --epochs 8
```

Performance of the backbone without bottleneck or quantum transformation.

### Baseline 2 — Classical 4-D

```bash
python train.py --bottleneck classical --n_qubits 4 --epochs 8
```

Tests whether 4-D compression alone explains any improvement.

### Quantum L=1

```bash
python train.py --bottleneck quantum --n_qubits 4 --n_layers 1 --epochs 8
```

### Quantum L=2

Two-epoch diagnostic first:

```bash
python train.py --bottleneck quantum --n_qubits 4 --n_layers 2 --epochs 2
```

The full 8-epoch run starts only after the diagnostic confirms training loss falls
clearly below ln(3) ≈ 1.0986.

### Quantum L=3

```bash
python train.py --bottleneck quantum --n_qubits 4 --n_layers 3 --epochs 8
```

Evaluated after L=2 is finalized.

---

## 13. Current Validated Results

Patient-level, 5-fold cross-validation, bootstrap 95% confidence intervals
(1000 replicates, patient-level resampling).

| Configuration | Balanced Accuracy [95% CI] | Macro F1 | Macro AUC [95% CI] |
| ------------- | -------------------------: | -------: | -----------------: |
| Swin Only     |     0.524 [0.401, 0.656]   |    0.516 | 0.748 [0.645, 0.860] |
| Classical 4-D |     0.540 [0.429, 0.652]   |    0.521 | 0.832 [0.746, 0.915] |
| Quantum L=1   |     0.449 [0.336, 0.568]   |    0.456 | 0.722 [0.612, 0.821] |
| Quantum L=2   |                    Pending |  Pending |            Pending |
| Quantum L=3   |                    Pending |  Pending |            Pending |

Intervals are wide because the cohort contains 59 subjects with 11 in the smallest
class. Differences between configurations are **not statistically resolved** at this
sample size and must not be reported as such.

### Per-class F1

| Class    | Swin Only | Classical 4-D | Quantum L=1 |
| -------- | --------: | ------------: | ----------: |
| Mild     |     0.462 |         0.538 |       0.353 |
| Moderate |     0.211 |         0.118 |       0.190 |
| Severe   |     0.877 |         0.907 |       0.825 |

Severe is recovered well across all configurations; mild and moderate are not. The
model largely learns a severe-versus-not-severe distinction, reflecting both class
imbalance and the difficulty of the mild/moderate boundary.

### Interpretation

The L=1 quantum arm sits **below** the classical 4-D control (0.722 vs 0.832 macro
AUC). The present evidence does **not** support claiming that the quantum feature
layer improves performance over a matched classical bottleneck. Remaining depth
experiments are required before any conclusion is drawn.

---

## 14. Validation Controls

### Label-permutation control

Severity labels were randomly permuted across subjects, holding all else fixed, and
the pipeline retrained.

| Metric            | Real labels | Permuted labels | Chance |
| ----------------- | ----------: | --------------: | -----: |
| Balanced accuracy |       0.524 |           0.259 |  0.333 |
| Macro AUC         |       0.748 |           0.375 |  0.500 |
| F1 mild           |       0.462 |           0.000 |      — |
| F1 moderate       |       0.211 |           0.000 |      — |

Permuted-label macro AUC: 0.375, 95% CI [0.263, 0.506] — the interval contains 0.50.
Training loss reached 0.017, showing the network memorised the permuted training
labels completely without any transfer to held-out subjects.

**This confirms the absence of patient-level leakage between folds.** Results are
stored in `results/archive/shuffle_control/`.

### Quantum initialization investigation

During the L=2 experiment, training loss remained at approximately **1.0975** across
consecutive epochs. For three classes, ln(3) ≈ 1.0986, so the classifier was
producing an uninformative class-prior solution and had not trained at all.

Gradient diagnostics were healthy (circuit gradient norm 0.129, projection 0.382),
ruling out a barren plateau. However, output variance collapsed through the circuit:

```text
projection output std: [0.095, 0.078, 0.095, 0.033]
quantum   output std: [0.017, 0.050, 0.035, 0.017]
```

`BasicEntanglerLayers` initializes rotation angles uniformly in [0, 2π], placing the
circuit in a strongly mixing regime. At L=1 enough signal survives; at L≥2 the
measurement output is nearly constant across inputs and the head has nothing to
separate. Initialization was therefore changed to N(0, 0.1), which starts the
entangling block near identity.

This is an **optimization diagnostic**, not a performance claim. A model that cannot
fit its own training data has not been trained, and its test metrics measure an
optimization failure rather than the architecture. Such runs are archived, never
reported.

---

## 15. Evaluation Metrics

Recorded per configuration: accuracy, balanced accuracy, macro F1, per-class F1,
macro ROC-AUC, confusion matrix, fold-level predictions, bootstrap confidence
intervals.

Balanced accuracy and macro F1 are reported alongside raw accuracy because the class
distribution is uneven — predicting "severe" for every subject yields 61% raw
accuracy but 0.333 balanced accuracy.

---

## 16. Statistical Validation

Patient-level cross-validation with held-out predictions aggregated across folds.
Bootstrap confidence intervals (patient-level resampling, ≥1000 replicates) are
computed for balanced accuracy, macro F1, and macro AUC.

Paired comparison between configurations will use DeLong's test per class with
Benjamini–Hochberg correction, finalized once all ablation arms are complete.

---

## 17. Explainability

Grad-CAM analysis at the final Swin stage, computed per severity class. A Swin
`reshape_transform` is required, since pytorch-grad-cam expects channel-first
activations while Swin emits tokens.

HVSMR-2.0 ships substructure segmentation masks, enabling a **quantitative** check:
overlap between the Grad-CAM peak region and annotated cardiac anatomy, compared
against the chance baseline given by the mask's area fraction. Reported separately
from the quantitative ablation.

---

## 18. Repository Structure

```text
CHD-Hybrid-Swin-Quantum/
│
├── config.py
├── data.py
├── cached_data.py
├── prepare_cache.py
├── model.py
├── train.py
├── metrics.py
├── explain.py
├── run_ablation.sh
├── README.md
│
├── results/
│   ├── swin_only_results.json
│   ├── swin_only_predictions.npz
│   ├── classical4_results.json
│   ├── classical4_predictions.npz
│   ├── q4_L1_results.json
│   └── q4_L1_predictions.npz
│
└── results/archive/
    ├── shuffle_control/          (label-permutation validation)
    ├── q4_L3_initial/            (failed: single LR, no convergence)
    └── q4_L2_uniform_init/       (failed: uniform init, no convergence)
```

Datasets, cached tensors, and checkpoints are excluded from version control.

---

## 19. Reproducing the Experiments

### Step 1 — Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 2 — Enter the project directory

```python
%cd /content/drive/MyDrive/CHD_Journal_Project/Restored_Project
```

### Step 3 — Install dependencies

```python
!pip install -q timm pennylane nibabel grad-cam
```

PennyLane is required only for quantum runs but is not preinstalled in Colab and is
wiped on every runtime restart.

### Step 4 — Build the cache

The cache lives on local disk and is wiped each session:

```python
!ls -lh /content/cache 2>/dev/null || python prepare_cache.py
```

Verify:

```python
import pandas as pd
idx = pd.read_csv("/content/cache/index.csv")
print("Subjects:", idx.subject_id.nunique())
print(idx.groupby("y").subject_id.nunique())
```

Expected:

```text
Subjects: 59
y
0    12
1    11
2    36
```

### Step 5 — Check GPU

```python
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
```

If this reports CPU, stop. Training on CPU is roughly 30–50× slower.

### Step 6 — Run an experiment

```python
!python train.py --bottleneck none --epochs 8
!python train.py --bottleneck classical --n_qubits 4 --epochs 8
!python train.py --bottleneck quantum --n_qubits 4 --n_layers 1 --epochs 8
```

### Step 7 — Check convergence before trusting any result

Training loss must fall clearly below **ln(3) ≈ 1.0986**. A run ending near that
value has not trained, and its test metrics are meaningless regardless of whether
the script exited cleanly.

---

## 20. Experiment Output

```text
runs/
└── q4_L1/
    ├── fold0_preds.npz … fold4_preds.npz
    ├── folds_partial.json
    ├── predictions.npz
    └── results.json
```

Final result files are copied into `results/` for version control. Model checkpoints
are written to `/content/ckpt/` (local disk) to avoid consuming Drive quota.

---

## 21. Resume and Checkpointing

Fold-level predictions are written as each fold completes. Rerunning a command skips
folds already present, which matters in Colab where sessions disconnect during long
runs.

**When changing any configuration or initialization, delete the run directory
first:**

```python
!rm -rf runs/q4_L2
```

Otherwise stale folds are silently reused and the result mixes two configurations.

---

## 22. Experimental Principle

The final depth ablation must use identical settings across L=1, L=2, and L=3,
differing only in entangling depth. The goal is to isolate the effect of circuit
depth, not to compare independently tuned models.

**The L=1 experiment was completed under the previous initialization and therefore
must be rerun under the corrected N(0, 0.1) initialization before the final depth
comparison is reported.** The current L=1 figures in Section 13 stand as a valid
result for that configuration, but they are not comparable to L=2 and L=3 once those
use different initialization.

---

## 23. Current Experimental Status

### Completed

* [x] HVSMR-2.0 data organization
* [x] Subject-level metadata indexing with anchored ID matching
* [x] Slice cache generation
* [x] 5-fold patient-level cross-validation pipeline
* [x] **Label-permutation leakage control (passed)**
* [x] Swin-only baseline
* [x] Classical 4-D bottleneck
* [x] Quantum L=1 (previous initialization)
* [x] Quantum circuit backpropagation implementation
* [x] Quantum-specific learning-rate configuration
* [x] Small quantum-weight initialization implementation

### In progress

* [ ] L=2 two-epoch diagnostic
* [ ] L=2 full 8-epoch experiment
* [ ] L=3 corrected experiment
* [ ] Rerun L=1 under the final common initialization
* [ ] Final quantum-depth ablation table
* [ ] DeLong paired statistical comparison
* [ ] Grad-CAM analysis and mask-overlap scoring
* [ ] Final journal figures
* [ ] Final manuscript experimental section

---

## 24. Scientific Reporting Policy

This repository deliberately distinguishes between:

1. **Previously reported results** (from earlier drafts, not treated as valid)
2. **Currently validated results** (from this pipeline, post-leakage-check)
3. **Pending experiments**
4. **Archived failed or obsolete runs**

Results are not treated as final merely because they appeared in an earlier
manuscript draft. A run is reported only if the model demonstrably converged and the
configuration is documented.

The journal manuscript will be updated only after the current reproducible
patient-level experiments are complete and verified.

---

## 25. Disclaimer

A research prototype for experimental evaluation of hybrid quantum-classical machine
learning on cardiac MRI. **Not a clinically validated diagnostic system.** Not for
use in medical decision-making.

## Author
S.Venikalaxmi (Integrated MTech at VIT, Vellore )
