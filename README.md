# CHD-Hybrid-Swin-Quantum

## Isolating the Contribution of a Quantum Feature Layer in Hybrid Transformer Models: A Controlled Study on Congenital Heart Disease Severity Classification From Cardiac MRI

A controlled study of hybrid quantum-classical deep learning for congenital heart
disease severity classification from cardiac MRI. A pretrained **Swin Transformer**
backbone is paired with three interchangeable bottlenecks — none, a four-dimensional
classical projection, and a four-qubit simulated parameterised circuit of identical
width — to test whether entangling structure adds value beyond the dimensionality
reduction that quantum encoding necessarily imposes.

Evaluated with **five-fold patient-level cross-validation** on the public HVSMR-2.0
dataset.

**Author:** S. Venikalaxmi — Integrated M.Tech, Vellore Institute of Technology

---

## Headline finding

**The four-qubit entangling bottleneck did not outperform a matched classical
projection.** The compression itself helped: macro-AUC rose from 0.748 to 0.832 when
a four-dimensional classical bottleneck was introduced. Replacing that projection
with the quantum circuit reduced performance to 0.722.

A paired bootstrap of the difference gives **+0.111 in favour of the classical
bottleneck, 95% CI [+0.017, +0.212]**, excluding zero. A second, structurally
different encoding configuration shifted the metric profile across severity classes
without closing the gap.

The contribution is the controlled experimental design and the transparency of its
reporting, not a performance improvement.

---

## 1. Motivation

Encoding a classical representation into an *n*-qubit circuit requires first
compressing that representation to *n* dimensions. On the small cohorts typical of
clinical imaging, a narrow bottleneck is itself a powerful regulariser.

When a hybrid model is compared only against an unmodified backbone — as is standard
in the hybrid medical-imaging literature — any observed improvement is ambiguous
between two explanations: the quantum transformation, and the compression that
necessarily precedes it. The two are introduced together and cannot be separated
after the fact.

This project resolves that ambiguity by holding bottleneck width constant and varying
only the presence of the quantum circuit.

```text
Cardiac MRI (2.5D slice stack, 224 × 224 × 3)
     │
     ▼
Swin Transformer Tiny backbone
     │
     ▼
768-dimensional embedding
     │
     ├──── (a) no bottleneck ─────────────────────────────┐
     │                                                     │
     ▼                                                     │
Shared 4-D projection: Linear(768 → 4) + tanh              │
     │                                                     │
     ├──── (b) classical 4-D, matched control ─────────────┤
     │                                                     │
     └──── (c) quantum                                     │
                 │                                         │
        4-qubit parameterized circuit                      │
        AngleEmbedding → Entangler × L → Pauli-Z           │
                 │                                         │
                 └─────────────────────────────────────────┤
                                                           ▼
                                            Classification head
                                     (identical architecture,
                                       trained separately)
                                                           │
                                                           ▼
                                          Mild / Moderate / Severe
```

---

## 2. Research Question

**Does a quantum feature transformation provide an advantage over a classical
bottleneck of identical width?**

Three configurations, identical in every respect except the bottleneck:

1. Swin Transformer only
2. Swin Transformer + classical four-dimensional projection **(the control)**
3. Swin Transformer + four-qubit quantum feature layer

Preprocessing, augmentation, optimiser, schedule and cross-validation folds are held
constant across all three, so that the bottleneck is the only variable.

A secondary question — how performance varies with entangling depth (L = 1, 2, 3) —
could not be answered: circuits at L ≥ 2 failed to converge under four separate
optimisation configurations. The cause was diagnosed and is reported as an
optimisation finding (Section 9.2).

---

## 3. Dataset

Cropped HVSMR-2.0 cardiac MRI (Pace et al., 2024), via figshare:
https://doi.org/10.6084/m9.figshare.c.7074755.v2

| Class     | Subjects |
| --------- | -------: |
| Mild      |       12 |
| Moderate  |       11 |
| Severe    |       36 |
| **Total** |   **59** |

The nominal release contains 60 scans; one subject is absent from the current index
and is reported as excluded rather than assumed away.

All experiments are at the **subject level** — slices from one patient never appear
across different folds.

---

## 4. Data Processing

1. Load cropped cardiac MRI volumes (NIfTI).
2. Associate scans with metadata by **anchored** subject-ID matching.
3. Percentile-clip intensities (1st–99th), rescale to [0, 1], per volume. MRI has no
   absolute intensity scale; Hounsfield windowing does not apply.
4. Retain only slices intersecting annotated cardiac structures.
5. Build 2.5D inputs: three contiguous axial slices → three channels.
6. Resize to 224 × 224; standardise with ImageNet channel statistics.
7. Write a slice-level memmap cache with subject identifiers preserved.

Augmentation: rotation ±15°, translation ±10%, intensity jitter. **Horizontal
flipping is excluded** — left–right orientation is diagnostically meaningful in CHD
(dextrocardia, situs anomalies, transposition), so flipping corrupts the label.

> **Subject-ID matching bug (fixed).** An earlier implementation matched IDs by
> substring, causing `pat1` to match `pat10`–`pat19` and silently pairing five
> subjects with the wrong volumes. Matching is now anchored and rejects ambiguous
> IDs. All results below postdate this fix.

Cache (outside version control): `stacks.npy` (~2 GB, float16) and `index.csv`.
**7,814 slices** from **59 subjects**, approximately 132 slices per subject.

---

## 5. Cross-Validation

**Five-fold patient-level cross-validation**, stratified by severity, partitioned
over subject IDs — never over individual slices.

Slice-level probabilities are pooled to patient level by mean before any metric is
computed. Held-out predictions are aggregated across all five folds. Each evaluation
partition contains eleven or twelve subjects.

---

## 6. Model Architectures

Two bottleneck designs were evaluated. **The first produced the main ablation; the
second is a secondary configuration reported separately.** Both are retained in the
repository.

### 6.1 Original design — `model.py`

Used for all three arms of the main ablation.

```text
Swin → Linear(768→4) → tanh → [circuit] → head
```

No normalisation, no angle scaling. Circuit weights use PennyLane's default
`BasicEntanglerLayers` initialisation, uniform in [0, 2π].

### 6.2 Modified design — `model_batchnorm_variant.py`

Developed while investigating non-convergence at L ≥ 2.

```text
Swin → Linear(768→4) → BatchNorm → tanh → × angle_scale → [circuit] → head
```

Adds BatchNorm after the projection and a learnable `angle_scale` (initialised π/2)
multiplying the tanh output before angle encoding. Circuit weights initialised
N(0, 0.1).

> **These are different architectures.** Results from one cannot be placed in the
> same comparison table as results from the other. The main ablation uses 6.1
> throughout; the 6.2 result is reported separately with that limitation stated.

---

## 7. Quantum Circuit

* Qubits: **4** (sixteen-dimensional state space)
* Encoding: **AngleEmbedding** (Rx rotations)
* Entangling block: **BasicEntanglerLayers** (ring-topology CNOTs)
* Measurement: **Pauli-Z expectation values**
* Depth reported: **L = 1** (L ≥ 2 did not converge — Section 9.2)
* Trainable circuit parameters: n × L

**Terminology.** The circuit is simulated classically on PennyLane's
`default.qubit`. No quantum hardware was used at any stage and no hardware claim is
made. At four qubits no asymptotic advantage is available, and none is claimed. The
question is empirical: does the entangling structure of a small simulated circuit
outperform a classical layer of identical output width on this task?

### Differentiation

```python
diff_method="backprop"
```

Changed from parameter-shift, which failed on broadcasted inputs in this
configuration. For a circuit evaluated on a classical simulator, backpropagation
through the state vector yields values identical to the parameter-shift rule.
Parameter-shift, or an equivalent hardware-compatible scheme, would be required on
physical hardware where the internal state is not accessible.

### Optimisation

**AdamW**, base learning rate `1e-4` for all classical components. Circuit rotation
parameters use a separate rate of `1e-2` (base × 100), because rotation angles are
parameterised on a 2π scale while network weights operate two orders of magnitude
smaller; a shared rate leaves the angles effectively frozen.

**This applies only to circuit parameters, which have no counterpart in the classical
arm.** Backbone, projection and head remain at `1e-4` in every configuration, so the
classical-versus-quantum comparison stays matched in all shared components.

### Device placement

The PennyLane simulator allocates its state vector on CPU; the quantum layer is
pinned to CPU and the four-dimensional bottleneck vector crosses the device boundary
inside `forward()`. Autograd tracks the transfer. A four-qubit state has sixteen
amplitudes, so the CPU cost is negligible beside the backbone on GPU.

---

## 8. Results

### 8.1 Main ablation — original architecture (Section 6.1)

Patient-level, five-fold cross-validation, bootstrap 95% confidence intervals
(1000 replicates, patient-level resampling). All three arms share preprocessing,
augmentation, optimiser, schedule and folds; **only the bottleneck differs.**

| Configuration | Balanced Accuracy [95% CI] | Macro F1 | Macro AUC [95% CI]       |
| ------------- | -------------------------: | -------: | -----------------------: |
| Backbone only |       0.524 [0.401, 0.656] |    0.516 |     0.748 [0.645, 0.860] |
| Classical 4-D |       0.540 [0.429, 0.652] |    0.521 | **0.832 [0.746, 0.915]** |
| Quantum L=1   |       0.449 [0.336, 0.568] |    0.456 |     0.722 [0.612, 0.821] |

**Paired comparison.** Because every configuration was evaluated on the same 59
subjects under identical folds, the difference in macro-AUC was bootstrapped
directly, resampling subjects and recomputing both arms on each resample
(10,000 replicates):

```text
classical − quantum = +0.111,  95% CI [+0.017, +0.212]   (excludes zero)
```

Marginal intervals overlap, but they are conservative here: both arms were evaluated
on the same subjects, so their errors are correlated and treating the intervals as
independent discards that structure. The paired analysis retains it.

**Per-class F1**

| Class    |  n | Backbone | Classical 4-D | Quantum L=1 |
| -------- | -: | -------: | ------------: | ----------: |
| Mild     | 12 |    0.462 |         0.538 |       0.353 |
| Moderate | 11 |    0.211 |         0.118 |       0.190 |
| Severe   | 36 |    0.877 |         0.907 |       0.825 |

**Per-class one-vs-rest AUC**

| Class    | Backbone | Classical 4-D | Quantum L=1 |
| -------- | -------: | ------------: | ----------: |
| Mild     |     0.78 |          0.84 |        0.79 |
| Moderate |     0.58 |          0.71 |        0.58 |
| Severe   |     0.89 |          0.95 |        0.80 |

**Confusion matrices** (rows = true: mild / moderate / severe)

```text
Backbone only      Classical 4-D       Quantum L=1
[[ 6  4  2]        [[ 7  4  1]         [[ 3  5  4]
 [ 6  2  3]         [ 6  1  4]          [ 2  2  7]
 [ 2  2 32]]        [ 1  1 34]]         [ 0  3 33]]
```

All row sums are 12 / 11 / 36, totalling 59. Balanced accuracy, macro F1 and every
per-class F1 above are reproducible from these matrices.

### 8.2 Secondary configuration — modified architecture (Section 6.2)

| Configuration | Balanced Accuracy [95% CI] | Macro F1 | Macro AUC [95% CI]   |
| ------------- | -------------------------: | -------: | -------------------: |
| Quantum L=1 (BatchNorm + angle scale) | 0.492 [0.350, 0.639] | 0.492 | 0.683 [0.564, 0.804] |

Per-class F1: mild 0.455, moderate 0.286, severe 0.735.

```text
Confusion matrix
[[ 5  3  4]
 [ 4  4  3]
 [ 1 10 25]]
```

**Interpretation — stated precisely.** Relative to the original quantum
configuration, the refined encoding **raised balanced accuracy (0.449 → 0.492) and
macro F1 (0.456 → 0.492)**, with a notable gain in minority-class recovery
(moderate F1 0.190 → 0.286). **Macro AUC declined (0.722 → 0.683)** and severe-class
F1 fell (0.825 → 0.735). The change redistributed performance across classes rather
than improving it uniformly, and is not described as an improvement.

**This configuration is not architecturally matched to the classical control**, which
lacks BatchNorm, so it is reported separately rather than as a row in the main table.
Its value is that two structurally distinct quantum configurations both fell short of
the classical control — stronger evidence than a single run.

### 8.3 What the results show

**Compression alone improves discrimination.** Introducing a four-dimensional
classical bottleneck raised macro-AUC from 0.748 to 0.832 (+0.084) over the plain
backbone. On a 59-subject cohort, compressing 768 dimensions to four acts as a strong
regulariser. This is the confound the ablation was built to expose: without this arm,
any quantum improvement over the backbone would have been uninterpretable. The
magnitude of this effect is comparable to or larger than differences commonly
reported as evidence of quantum advantage in work lacking a matched-width control.

**The quantum bottleneck did not outperform its matched control.** At 0.722 macro-AUC
it fell 0.111 below the classical projection of identical width, and 0.026 below the
plain backbone. Under the refined encoding it reached 0.683.

**Scope.** The result applies to four qubits at entangling depth one, under noiseless
classical simulation, on a single 59-subject public cohort. It does not generalise to
wider circuits, greater depth, physical hardware, or other datasets.

**Class-wise behaviour.** Severe is recovered reliably (F1 0.735–0.907); moderate is
not (0.118–0.286). All configurations largely learn a severe-versus-not-severe
distinction. Moderate is misclassified as mild in the backbone and classical arms
(six of eleven in each), suggesting the mild/moderate boundary is the difficult one.

---

## 9. Validation Controls and Diagnostics

### 9.1 Label-permutation control

Severity labels were randomly permuted across subjects, holding all else fixed, and
the full cross-validation procedure repeated.

| Metric            | True labels | Permuted labels | Chance |
| ----------------- | ----------: | --------------: | -----: |
| Balanced accuracy |       0.524 |           0.259 |  0.333 |
| Macro AUC         |       0.748 |           0.375 |  0.500 |
| F1 mild           |       0.462 |           0.000 |      — |
| F1 moderate       |       0.211 |           0.000 |      — |
| F1 severe         |       0.877 |           0.444 |      — |

Permuted-label macro-AUC 0.375, 95% CI [0.263, 0.506] — **the interval contains
0.500.** Training loss reached 0.017, showing the network memorised the permuted
training labels essentially completely with no transfer to held-out subjects. Mild
and moderate received F1 of exactly 0.000, the majority-class behaviour of an
uninformative model.

**This confirms the absence of patient-level leakage between folds.** Archived in
`results/archive/shuffle_control/`.

### 9.2 Non-convergence at entangling depth L ≥ 2

Circuits at L ≥ 2 could not be trained to convergence. Across four configurations,
training loss remained at approximately **1.098** — equal to ln(3), the cross-entropy
of an uninformative three-class classifier — meaning these models failed to fit even
their own training data.

| Configuration                                            | Final train loss | Converged |
| -------------------------------------------------------- | ---------------: | --------- |
| Shared learning rate (1e-4)                               |           1.0963 | No        |
| Separate circuit rate (1e-2)                              |           1.0975 | No        |
| Separate circuit rate (5e-2)                              |           1.0975 | No        |
| Small init N(0, 0.1) + BatchNorm + learnable angle scale  |           1.0973 | No        |

**Mechanism.** Gradient diagnostics ruled out a barren plateau — circuit gradient
norm 0.129 against 0.382 for the projection layer. The cause was variance collapse
through the angle encoding:

```text
projection output std: [0.095, 0.078, 0.095, 0.033]
circuit    output std: [0.017, 0.050, 0.035, 0.017]
```

`AngleEmbedding` interprets inputs as rotation angles **in radians**. With projection
outputs of magnitude ≈ 0.09, each input produces an Rx rotation of ≈ 0.09 rad, so
⟨Z⟩ = cos(0.09) ≈ 0.996 nearly independently of the input. The unparameterised CNOT
ring compounds this contraction at each layer — which is why L = 1 trained and L ≥ 2
did not.

Adding BatchNorm and a learnable angle scale restored variance transmission (ratio of
circuit-output to projection-output standard deviation rose from ≈ 0.19 to ≈ 0.62)
and permitted convergence over the first two epochs at L = 2, but training
destabilised over the full schedule and returned to the class prior.

**This is an optimisation finding for the evaluated implementation, not a general
property of parameterized circuits.** A model that cannot fit its own training data
has not been trained, and its test metrics measure an optimisation failure rather
than an architecture. Such runs are archived, never reported as performance.

The failure is silent under conventional monitoring: training proceeds without error,
gradients remain non-zero, and the loss curve is smooth. Only its value, sitting at
the class-prior entropy, distinguishes it from slow convergence. **Monitoring
activation variance through the encoding stage is recommended as a routine
diagnostic.**

---

## 10. Evaluation Metrics and Statistical Treatment

Recorded per configuration: accuracy, balanced accuracy, macro F1, per-class F1,
macro one-vs-rest ROC-AUC, confusion matrix, fold-level predictions, bootstrap
confidence intervals.

Balanced accuracy and macro F1 are emphasised over raw accuracy because the class
distribution is uneven — predicting "severe" for every subject yields approximately
61% raw accuracy but 0.333 balanced accuracy, which is chance.

Marginal confidence intervals use patient-level bootstrap resampling with 1000
replicates. Between-configuration comparisons use a **paired** bootstrap of the
difference with 10,000 replicates, resampling subjects and recomputing both arms on
each resample.

The resampling procedure characterises variability arising from cohort composition.
It does not account for variability arising from repeated model training, such as
initialisation and data ordering, so the intervals are correspondingly narrower than
intervals incorporating both sources.

---

## 11. Training Configuration

| Parameter               | Value                             |
| ----------------------- | --------------------------------- |
| Backbone                | Swin Transformer Tiny             |
| Input size              | 224 × 224 (2.5D three-slice stack)|
| Bottleneck width        | 4                                 |
| Number of qubits        | 4                                 |
| Entangling layers L     | 1                                 |
| Base learning rate      | 1e-4                              |
| Circuit learning rate   | 1e-2 (base × 100)                 |
| Optimizer               | AdamW                             |
| Loss                    | CrossEntropyLoss (class-weighted) |
| Epochs                  | 8                                 |
| Cross-validation        | 5-fold, patient-level             |
| Bootstrap replicates    | 1000 (marginal), 10,000 (paired)  |
| Quantum simulator       | PennyLane default.qubit           |
| Circuit gradients       | Backpropagation                   |

Class weights are computed over **subjects**, not slices, so that subjects with
larger cardiac extents do not dominate the objective.

Eight epochs was fixed in advance from the convergence behaviour of the backbone-only
arm and was not tuned per configuration.

---

## 12. Reported Configurations

`--bottleneck` accepts `none`, `classical`, or `quantum`. These are *modes*, not run
names; the run name (`swin_only`, `classical4`, `q4_L1`) is derived automatically.

```bash
# Main ablation — with model.py (original architecture)
python train.py --bottleneck none      --epochs 8
python train.py --bottleneck classical --n_qubits 4 --epochs 8
python train.py --bottleneck quantum   --n_qubits 4 --n_layers 1 --epochs 8

# Secondary configuration — with model_batchnorm_variant.py
python train.py --bottleneck quantum   --n_qubits 4 --n_layers 1 --epochs 8
```

**Verify results against the saved prediction files** rather than against console
output:

```bash
python audit_results.py
```

This recomputes every reported metric from `results/*_predictions.npz`, performs the
paired comparison, and prints the cohort composition.

---

## 13. Figures

Generated from stored predictions; no training required.

```bash
python make_figures.py
```

| Figure | Content |
| ------ | ------- |
| 1 | Architecture: three bottleneck configurations (`architecture.drawio.xml`) |
| 2 | Macro-AUC across configurations with bootstrap 95% CIs |
| 3 | One-vs-rest ROC curves, three configurations, per-class AUC in legends |
| 4 | True versus permuted labels (leakage control) |

---

## 14. Explainability (implemented, not run)

Grad-CAM at the final Swin stage, computed per severity class. A Swin
`reshape_transform` is required, since pytorch-grad-cam expects channel-first
activations while Swin emits tokens.

HVSMR-2.0 ships substructure segmentation masks, enabling a quantitative check:
overlap between the Grad-CAM peak region and annotated cardiac anatomy, against the
chance baseline given by mask area fraction. Implemented in `explain.py`; not
included in the reported results.

---

## 15. Repository Structure

```text
CHD-Hybrid-Swin-Quantum/
│
├── config.py
├── data.py
├── cached_data.py
├── prepare_cache.py
├── model.py                        (original — main ablation)
├── model_batchnorm_variant.py      (modified — secondary configuration)
├── train.py
├── metrics.py
├── explain.py
├── audit_results.py                (recompute all metrics from predictions)
├── make_figures.py                 (generate Figures 2–4)
├── architecture.drawio.xml         (Figure 1 source)
├── run_ablation.sh
├── README.md
│
├── results/
│   ├── swin_only_results.json      / swin_only_predictions.npz
│   ├── classical4_results.json     / classical4_predictions.npz
│   └── q4_L1_results.json          / q4_L1_predictions.npz
│
├── figures/
│   ├── auc_comparison.pdf
│   ├── roc_curves.pdf
│   └── permutation_control.pdf
│
└── results/archive/
    ├── shuffle_control/            (label-permutation validation — passed)
    ├── q4_L1_new/                  (secondary configuration — Section 8.2)
    ├── q4_L3_shared_lr/            (failed: no convergence, loss 1.0963)
    ├── q4_L2_uniform_init/         (failed: no convergence, loss 1.0975)
    ├── q4_L2_high_lr/              (failed: no convergence, loss 1.0975)
    └── q4_L2_batchnorm/            (failed: diverged, loss 1.0973)
```

Datasets, cached tensors and model checkpoints are excluded from version control.

---

## 16. Reproducing the Experiments

**Step 1 — Mount Drive and enter the project directory**

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/CHD_Journal_Project/Restored_Project
```

**Step 2 — Install dependencies** (wiped on every Colab restart)

```python
!pip install -q timm pennylane nibabel grad-cam
```

**Step 3 — Build the cache** (local disk, wiped each session)

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

Expected: 59 subjects; 12 / 11 / 36; 7,814 slices.

**Step 4 — Check GPU**

```python
import torch
print("CUDA available:", torch.cuda.is_available())
```

If this reports CPU, stop. Training on CPU is roughly 30–50× slower.

**Step 5 — Run the configurations** (Section 12).

**Step 6 — Check convergence before trusting any result.** Training loss must fall
clearly below **ln(3) ≈ 1.0986**. A run ending near that value has not trained, and
its test metrics are meaningless regardless of whether the script exited cleanly.

**Step 7 — Audit.** Run `audit_results.py` and confirm the printed metrics match
those in Section 8 before citing any figure.

---

## 17. Experiment Output and Resume

```text
runs/
└── q4_L1/
    ├── fold0_preds.npz … fold4_preds.npz
    ├── folds_partial.json
    ├── predictions.npz
    └── results.json
```

Fold-level predictions are written as each fold completes, so a disconnect costs one
fold rather than a whole run. Rerunning a command skips folds already present.

**When changing any configuration or model file, delete the run directory first:**

```bash
rm -rf runs/q4_L1
```

Otherwise stale folds are silently reused and the result mixes two configurations.

Final result files are copied into `results/` for version control. Checkpoints go to
`/content/ckpt/` (local disk) to avoid consuming Drive quota.

---

## 18. Scientific Reporting Policy

This repository deliberately distinguishes between:

1. **Currently validated results** — from this pipeline, post-leakage-check, with
   every number reproducible from the stored prediction files
2. **Secondary configurations** — valid but not architecturally matched to the main
   ablation, reported separately with that limitation stated
3. **Archived failed or non-converged runs** — retained with the reason recorded
4. **Results from earlier drafts** — not treated as valid and not carried forward

A run is reported only if the model demonstrably converged and the configuration is
documented. A negative outcome is reported as readily as a positive one, and a
partial improvement is not described as a uniform one.

---

## 19. Manuscript

Prepared for submission to IEEE Access using the official IEEE Access LaTeX template.
Sections I–VII complete, with four figures, nine tables, and 29 references. Every
reported figure traces to a file in `results/`.

Outstanding before submission:

* Verification of the provenance of the HVSMR-2.0 severity grading
* Author details and corresponding-author information
* Final proofing pass

---

## 20. Disclaimer

A research prototype for experimental evaluation of hybrid quantum-classical machine
learning on cardiac MRI. **Not a clinically validated diagnostic system.** Not for
use in medical decision-making.
