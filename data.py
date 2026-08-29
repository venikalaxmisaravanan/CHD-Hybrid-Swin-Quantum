"""HVSMR-2.0 data pipeline.

Key design points, each of which a reviewer will check:

1. Splitting is at PATIENT level. Slices from one subject never appear in two
   folds. Slice-level splitting lets the model recognise the patient instead of
   the pathology and invalidates every reported number.
2. Intensity handling is percentile clipping, NOT Hounsfield windowing. HVSMR is
   MRI; there is no absolute intensity scale, so each volume is normalised
   against its own distribution.
3. Slices are kept only where the provided segmentation mask has cardiac
   foreground, so the model is not trained on empty superior/inferior slices.
4. Input is a 2.5D stack of contiguous slices mapped to the three input
   channels. This gives a principled reason for 3-channel input to an
   ImageNet-pretrained backbone, rather than replicating grayscale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import re
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold

try:
    import nibabel as nib
except ImportError:  # pragma: no cover
    nib = None


# ---------------------------------------------------------------------------
# volume-level helpers
# ---------------------------------------------------------------------------

def load_volume(path: Path) -> np.ndarray:
    """Load a NIfTI volume as float32, axis order (H, W, S)."""
    if nib is None:
        raise ImportError("nibabel is required: pip install nibabel")
    return np.asarray(nib.load(str(path)).dataobj).astype(np.float32)


def normalise_mri(vol: np.ndarray, pct: tuple = (1.0, 99.0)) -> np.ndarray:
    """Percentile-clip then scale to [0, 1]. Per-volume, because MRI intensity
    has no absolute scale across scans."""
    lo, hi = np.percentile(vol, pct)
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((vol - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def foreground_slices(mask: np.ndarray, min_voxels: int) -> list[int]:
    """Indices of axial slices whose mask contains enough cardiac foreground."""
    counts = (mask > 0).reshape(-1, mask.shape[-1]).sum(axis=0)
    return [int(i) for i in np.where(counts >= min_voxels)[0]]


# ---------------------------------------------------------------------------
# index construction
# ---------------------------------------------------------------------------
def _find_file(directory: Path, sid: str) -> Path | None:
    """Match sid only when not followed by another digit."""
    cands = sorted(directory.glob(f"*{sid}*.nii*"))
    exact = [
        p for p in cands
        if re.match(rf"^{re.escape(sid)}(?!\d)", p.name)
    ]

    if len(exact) == 1:
        return exact[0]

    if len(exact) > 1:
        raise ValueError(
            f"ambiguous ID {sid}: {[p.name for p in exact]}"
        )

    return None

def build_index(cfg) -> pd.DataFrame:
    """Scan the dataset once and return one row per usable slice.

    metadata.csv must contain: subject_id, severity
    where severity is one of cfg.classes. You construct this from the
    dataset's accompanying documentation.
    """
    meta = pd.read_csv(cfg.metadata_csv)
    missing = set(meta["severity"]) - set(cfg.classes)
    if missing:
        raise ValueError(f"Unknown severity labels in metadata: {missing}")

    cls_to_idx = {c: i for i, c in enumerate(cfg.classes)}
    img_dir = cfg.data_root / cfg.image_dir
    lbl_dir = cfg.data_root / cfg.label_dir

    rows = []
    for _, r in meta.iterrows():
        sid = str(r["subject_id"])
        img_p = _find_file(img_dir, sid)
        lbl_p = _find_file(lbl_dir, sid)
        if img_p is None or lbl_p is None:
            print(f"[warn] missing volume or mask for {sid}; skipped")
            continue
        mask = load_volume(lbl_p)
        for s in foreground_slices(mask, cfg.min_fg_voxels):
            rows.append({
                "subject_id": sid,
                "image_path": str(img_p),
                "label_path": str(lbl_p),
                "slice": s,
                "y": cls_to_idx[r["severity"]],
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No usable slices found. Check paths and mask values.")
    print(f"[index] {df.subject_id.nunique()} subjects, {len(df)} slices")
    print(df.groupby('y').subject_id.nunique().rename('subjects per class'))
    return df


def patient_folds(df: pd.DataFrame, n_folds: int, seed: int):
    """Stratified K-fold over SUBJECTS, yielding slice-level index arrays."""
    subj = df.groupby("subject_id")["y"].first().reset_index()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr_i, te_i in skf.split(subj["subject_id"], subj["y"]):
        tr_s = set(subj.subject_id.iloc[tr_i])
        te_s = set(subj.subject_id.iloc[te_i])
        assert not (tr_s & te_s), "patient leakage between folds"
        yield (df.index[df.subject_id.isin(tr_s)].to_numpy(),
               df.index[df.subject_id.isin(te_s)].to_numpy())


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class HVSMRSliceDataset(Dataset):
    """Returns (tensor[3,H,W], label, subject_id) for one 2.5D stack."""

    def __init__(self, df: pd.DataFrame, cfg, train: bool = False):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.train = train
        self._cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.df)

    def _volume(self, path: str) -> np.ndarray:
        if path not in self._cache:
            # keep the cache small; volumes are large
            if len(self._cache) > 4:
                self._cache.pop(next(iter(self._cache)))
            v = normalise_mri(load_volume(Path(path)), self.cfg.clip_percentiles)
            if v is None:
                raise RuntimeError(f"normalise_mri returned None for {path}")
            self._cache[path] = v
        return self._cache[path]

    def __getitem__(self, i: int):
        import torch.nn.functional as F

        r = self.df.iloc[i]
        vol = self._volume(r["image_path"])
        s, n_s = int(r["slice"]), vol.shape[-1]
        half = self.cfg.stack_depth // 2
        idxs = [min(max(s + o, 0), n_s - 1) for o in range(-half, half + 1)]
        stack = np.stack([vol[..., j] for j in idxs], axis=0).astype(np.float32)

        x = torch.from_numpy(stack).unsqueeze(0)
        x = F.interpolate(x, size=(self.cfg.img_size, self.cfg.img_size),
                          mode="bilinear", align_corners=False).squeeze(0)

        if self.train:
            x = self._augment(x)

        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        x = (x - mean) / std
        return x, int(r["y"]), r["subject_id"]

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Rotation, translation, intensity jitter. NO horizontal flip."""
        import torchvision.transforms.functional as TF

        c = self.cfg
        ang = float(np.random.uniform(-c.rotation_deg, c.rotation_deg))
        max_t = c.translate_frac * c.img_size
        tx = int(np.random.uniform(-max_t, max_t))
        ty = int(np.random.uniform(-max_t, max_t))
        x = TF.affine(x, angle=ang, translate=[tx, ty], scale=1.0, shear=[0.0])
        if c.intensity_jitter > 0:
            f = 1.0 + float(np.random.uniform(-c.intensity_jitter, c.intensity_jitter))
            x = torch.clamp(x * f, 0.0, 1.0)
        return x


def class_weights(df: pd.DataFrame, n_classes: int) -> torch.Tensor:
    """Inverse-frequency weights computed over SUBJECTS, not slices, so that
    subjects with many slices do not dominate the weighting."""
    per_subj = df.groupby("subject_id")["y"].first()
    counts = np.bincount(per_subj.to_numpy(), minlength=n_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32)
