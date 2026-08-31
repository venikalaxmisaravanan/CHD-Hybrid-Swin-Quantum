"""One-time pre-extraction of 2.5D slice stacks into a flat memmap.

Why this exists: the on-the-fly loader re-reads a full NIfTI volume almost every
batch, because shuffled sampling defeats any small volume cache. Extracting once
turns each __getitem__ into a small contiguous read.

Run once:
    python prepare_cache.py

Produces:
    cache/stacks.npy   float16, shape (N, 3, img_size, img_size)
    cache/index.csv    row order, subject_id, y
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path

from config import CFG
from data import build_index, load_volume, normalise_mri


def main():
    cfg = CFG
    out_dir = Path("/content/cache")
    out_dir.mkdir(exist_ok=True)

    df = build_index(cfg).reset_index(drop=True)
    n, sz = len(df), cfg.img_size
    print(f"[cache] extracting {n} stacks at {sz}x{sz}")

    arr = np.lib.format.open_memmap(
        out_dir / "stacks.npy", mode="w+",
        dtype=np.float16, shape=(n, 3, sz, sz))

    half = cfg.stack_depth // 2
    # process one volume at a time so each is loaded exactly once
    for vol_path, grp in df.groupby("image_path"):
        vol = normalise_mri(load_volume(Path(vol_path)), cfg.clip_percentiles)
        n_s = vol.shape[-1]
        for row_i, r in grp.iterrows():
            s = int(r["slice"])
            idxs = [min(max(s + o, 0), n_s - 1) for o in range(-half, half + 1)]
            stack = np.stack([vol[..., j] for j in idxs], 0).astype(np.float32)
            t = torch.from_numpy(stack).unsqueeze(0)
            t = F.interpolate(t, size=(sz, sz), mode="bilinear",
                              align_corners=False).squeeze(0)
            arr[row_i] = t.numpy().astype(np.float16)
        print(f"  done {Path(vol_path).name} ({len(grp)} slices)", flush=True)

    arr.flush()
    df[["subject_id", "y"]].to_csv(out_dir / "index.csv", index=False)
    gb = arr.nbytes / 1e9
    print(f"[cache] wrote {out_dir/'stacks.npy'} ({gb:.2f} GB)")


if __name__ == "__main__":
    main()
