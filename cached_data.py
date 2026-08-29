"""Dataset backed by the memmap produced by prepare_cache.py.

Drop-in replacement for HVSMRSliceDataset. Same output signature:
    (tensor[3,H,W], label, subject_id)

Add to train.py:
    from cached_data import CachedSliceDataset, load_cached_index
and swap the two HVSMRSliceDataset(...) constructions for
    CachedSliceDataset(tr_df, cfg, train=True)
    CachedSliceDataset(te_df, cfg, train=False)

build_index() in main() is replaced by load_cached_index().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset

from data import IMAGENET_MEAN, IMAGENET_STD

CACHE_DIR = Path("cache")


def load_cached_index() -> pd.DataFrame:
    """Index rows aligned with the memmap row order."""
    df = pd.read_csv(CACHE_DIR / "index.csv")
    df["row"] = np.arange(len(df))
    print(f"[cache] {df.subject_id.nunique()} subjects, {len(df)} slices")
    print(df.groupby("y").subject_id.nunique().rename("subjects per class"))
    return df


class CachedSliceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg, train: bool = False):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.train = train
        self._arr = None            # opened lazily, per worker process

    def __len__(self) -> int:
        return len(self.df)

    @property
    def arr(self) -> np.ndarray:
        if self._arr is None:
            self._arr = np.load(CACHE_DIR / "stacks.npy", mmap_mode="r")
        return self._arr

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        x = torch.from_numpy(
            np.asarray(self.arr[int(r["row"])], dtype=np.float32))
        if self.train:
            x = self._augment(x)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return (x - mean) / std, int(r["y"]), r["subject_id"]

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Rotation, translation, intensity jitter. NO horizontal flip —
        left-right orientation is diagnostic in CHD."""
        import torchvision.transforms.functional as TF

        c = self.cfg
        ang = float(np.random.uniform(-c.rotation_deg, c.rotation_deg))
        m = c.translate_frac * c.img_size
        tx, ty = int(np.random.uniform(-m, m)), int(np.random.uniform(-m, m))
        x = TF.affine(x, angle=ang, translate=[tx, ty], scale=1.0, shear=[0.0])
        if c.intensity_jitter > 0:
            f = 1.0 + float(np.random.uniform(-c.intensity_jitter,
                                              c.intensity_jitter))
            x = torch.clamp(x * f, 0.0, 1.0)
        return x
