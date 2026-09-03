"""Cross-validated training for HVSMR-2.0 severity classification.

Usage:
    python train.py --bottleneck quantum   --n_qubits 4 --n_layers 3
    python train.py --bottleneck classical --n_qubits 4
    python train.py --bottleneck none
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

from config import CFG
from data import patient_folds, class_weights
from cached_data import CachedSliceDataset, load_cached_index
from model import HybridSwin, count_params
from metrics import aggregate_to_patient, compute_metrics, bootstrap_ci, report


def set_seed(s: int):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    P, Y, S = [], [], []
    for x, y, sid in loader:
        logits = model(x.to(device))
        P.append(torch.softmax(logits, dim=1).cpu().numpy())
        Y.append(y.numpy())
        S.extend(list(sid))
    return np.concatenate(P), np.concatenate(Y), np.array(S)


def train_one_fold(cfg, df, tr_idx, te_idx, device, fold: int):
    tr_df, te_df = df.loc[tr_idx], df.loc[te_idx]
    tr_ds = CachedSliceDataset(tr_df, cfg, train=True)
    te_ds = CachedSliceDataset(te_df, cfg, train=False)

    tr_dl = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True,
                       num_workers=cfg.num_workers, drop_last=False)
    te_dl = DataLoader(te_ds, batch_size=cfg.batch_size, shuffle=False,
                       num_workers=cfg.num_workers)

    model = HybridSwin(cfg, n_classes=len(cfg.classes)).to(device)
    w = class_weights(tr_df, len(cfg.classes)).to(device)
    crit = nn.CrossEntropyLoss(weight=w)
    q_params = model.quantum_parameters() if model.quantum is not None else []
    q_ids = {id(p) for p in q_params}
    other = [p for p in model.parameters() if id(p) not in q_ids]
    opt = torch.optim.AdamW(
        [{"params": other, "lr": cfg.lr},
         {"params": q_params, "lr": cfg.lr * 100}],
        weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    history = []
    for ep in range(cfg.epochs):
        model.train()
        tot, n = 0.0, 0
        for bi, (x, y, _) in enumerate(tr_dl):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * x.size(0); n += x.size(0)
            if bi % 25 == 0:
                print(f"    fold {fold} ep {ep+1} batch {bi}/{len(tr_dl)} "
                      f"loss {loss.item():.4f}", flush=True)
        sched.step()
        history.append({"epoch": ep + 1, "train_loss": tot / max(n, 1)})
        print(f"  fold {fold} epoch {ep+1}/{cfg.epochs} loss {tot/max(n,1):.4f}")

    probs, ys, sids = predict(model, te_dl, device)
    _, p_pat, y_pat = aggregate_to_patient(sids, probs, ys, cfg.aggregate)
    return model, history, y_pat, p_pat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bottleneck", choices=["quantum", "classical", "none"])
    ap.add_argument("--n_qubits", type=int)
    ap.add_argument("--n_layers", type=int)
    ap.add_argument("--epochs", type=int)
    a = ap.parse_args()

    cfg = CFG
    for k in ("bottleneck", "n_qubits", "n_layers", "epochs"):
        if getattr(a, k) is not None:
            setattr(cfg, k, getattr(a, k))

    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[run] {cfg.run_name()} on {device}")

    df = load_cached_index()
    out = cfg.output_dir / cfg.run_name()
    out.mkdir(parents=True, exist_ok=True)

    all_y, all_p, folds = [], [], []
    for fold, (tr, te) in enumerate(patient_folds(df, cfg.n_folds, cfg.seed)):
        fold_file = out / f"fold{fold}_preds.npz"
        if fold_file.exists():
            print(f"[skip] fold {fold} already done")
            d = np.load(fold_file)
            y_pat, p_pat = d["y"], d["p"]
            m = compute_metrics(y_pat, p_pat, cfg.classes)
            m["fold"] = fold
            folds.append(m)
            all_y.append(y_pat); all_p.append(p_pat)
            continue

        model, hist, y_pat, p_pat = train_one_fold(cfg, df, tr, te, device, fold)
        m = compute_metrics(y_pat, p_pat, cfg.classes)
        m["fold"] = fold
        folds.append(m)
        all_y.append(y_pat); all_p.append(p_pat)

        ckpt_dir = Path("/content/ckpt") / cfg.run_name()
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_dir / f"fold{fold}.pt")

        np.savez(fold_file, y=y_pat, p=p_pat)
        with open(out / "folds_partial.json", "w") as f:
            json.dump(folds, f, indent=2, default=float)
        print(f"[fold {fold}] {m}")

    Y = np.concatenate(all_y); P = np.concatenate(all_p)
    pooled = compute_metrics(Y, P, cfg.classes)
    for met in ("macro_f1", "macro_auc", "balanced_accuracy"):
        pt, lo, hi = bootstrap_ci(Y, P, cfg.classes, met, cfg.bootstrap_reps,
                                  cfg.seed)
        pooled[f"{met}_ci"] = [pt, lo, hi]

    print("\n=== POOLED (patient level, all folds) ===")
    print(json.dumps(pooled, indent=2, default=float))
    print(report(Y, P, cfg.classes))

    np.savez(out / "predictions.npz", y=Y, p=P)
    with open(out / "results.json", "w") as f:
        json.dump({"config": cfg.run_name(), "folds": folds,
                   "pooled": pooled,
                   "params": count_params(HybridSwin(cfg, pretrained=False))},
                  f, indent=2, default=float)
    print(f"[saved] {out/'results.json'}")


if __name__ == "__main__":
    main()
