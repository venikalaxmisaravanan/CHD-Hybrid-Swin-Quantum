"""Central configuration for the HVSMR-2.0 severity classification experiments."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ---- paths -------------------------------------------------------------
    data_root: Path = Path("./hvsmr2")          # contains images/ and labels/
    image_dir: str = "images"                    # *.nii.gz volumes
    label_dir: str = "labels"                    # *.nii.gz segmentation masks
    metadata_csv: Path = Path("./metadata.csv")  # subject_id,severity
    output_dir: Path = Path("./runs")

    # ---- data --------------------------------------------------------------
    img_size: int = 224
    stack_depth: int = 3            # 2.5D: number of contiguous slices -> channels
    min_fg_voxels: int = 200        # slice kept if mask foreground exceeds this
    clip_percentiles: tuple = (1.0, 99.0)   # MRI intensity clipping (NOT HU windowing)
    classes: tuple = ("mild", "moderate", "severe")

    # ---- model -------------------------------------------------------------
    backbone: str = "swin_tiny_patch4_window7_224"
    embed_dim: int = 768
    bottleneck: str = "quantum"     # {"quantum", "classical", "none"}
    n_qubits: int = 4               # swept in ablation: 4, 6
    n_layers: int = 3               # entangling depth: 1, 2, 3

    # ---- training ----------------------------------------------------------
    n_folds: int = 5
    epochs: int = 25
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 42
    num_workers: int = 2  # Colab gives 2 CPUs; more workers hurts

    # ---- augmentation ------------------------------------------------------
    # NOTE: horizontal flip is deliberately excluded. Left-right orientation is
    # diagnostically meaningful in CHD (dextrocardia, situs anomalies, TGA).
    rotation_deg: float = 15.0
    translate_frac: float = 0.10
    intensity_jitter: float = 0.10

    # ---- evaluation --------------------------------------------------------
    bootstrap_reps: int = 1000
    aggregate: str = "mean"         # patient-level pooling of slice probabilities

    def run_name(self) -> str:
        if self.bottleneck == "quantum":
            return f"q{self.n_qubits}_L{self.n_layers}"
        if self.bottleneck == "classical":
            return f"classical{self.n_qubits}"
        return "swin_only"


CFG = Config()
