"""Day 4: train the CNN or the temporal Transformer once (outside all systems), export ONNX.

    python scripts/06_train.py --model cnn --seed 0
    python scripts/06_train.py --model transformer --seed 0
    (repeat for seeds 1 and 2)   -- or on Kaggle, see docs/RUNBOOK.md

Writes models/<tag>/<model>_seed<k>.{pt,onnx}, _history.csv, _meta.json
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from edc import config  # noqa: E402
from edc.dataset import normalise  # noqa: E402
from edc.metrics import confusion, scores  # noqa: E402
from edc.models import build, n_params  # noqa: E402


class Tiles(Dataset):
    def __init__(self, X, Y, mean, std, augment):
        self.X, self.Y, self.m, self.s, self.aug = X, Y, mean, std, augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = normalise(self.X[i], self.m, self.s)          # (T,B,h,w)
        x = x.reshape(-1, *x.shape[-2:])                  # (T*B,h,w)
        y = self.Y[i].astype(np.int64)
        if self.aug:
            k = np.random.randint(4)
            x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k)
            if np.random.rand() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))


class Pixels(Dataset):
    def __init__(self, X, Y, mean, std):
        self.X = torch.from_numpy(normalise(X, mean, std))
        self.Y = torch.from_numpy(Y.astype(np.int64))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


def class_weights(y, k):
    c = np.bincount(y[y != 255].ravel(), minlength=k)[:k].astype(np.float64)
    w = 1.0 / np.sqrt(np.maximum(c, 1))
    w[c == 0] = 0
    return torch.tensor(w / w[c > 0].mean(), dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, k, device):
    model.eval()
    cm = np.zeros((k, k), np.int64)
    for x, y in loader:
        p = model(x.to(device)).argmax(1).cpu().numpy()
        cm += confusion(y.numpy(), p, k)
    return cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "transformer"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aoi", default=None)
    ap.add_argument("--size-km", type=float, default=50)
    ap.add_argument("--epochs", type=int, default=None)
    # Windows copies the training arrays into every worker process -> use 0 there; Kaggle (Linux) -> 2
    ap.add_argument("--workers", type=int, default=0 if os.name == "nt" else 2)
    a = ap.parse_args()

    wf = config.workflow()
    aoi = config.get_aoi(a.aoi)
    tag = config.tag(aoi["name"], a.size_km)
    ddir, mdir = config.dataset_dir(tag), config.models_dir(tag)
    st = json.load(open(ddir / "stats.json"))
    mean, std = np.array(st["mean"], np.float32), np.array(st["std"], np.float32)
    months, bands, names = st["months"], st["bands"], st["class_names"]
    k = len(names)
    tc = wf["training"]
    hp = tc[a.model]
    epochs = a.epochs or tc["epochs"]

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")

    def load(split, kind):
        return (np.load(ddir / f"{kind}_{split}_X.npy", mmap_mode="r"),
                np.load(ddir / f"{kind}_{split}_Y.npy"))

    if a.model == "cnn":
        Xtr, Ytr = load("train", "tiles"); Xva, Yva = load("val", "tiles")
        tr = Tiles(Xtr, Ytr, mean, std, augment=True); va = Tiles(Xva, Yva, mean, std, augment=False)
    else:
        Xtr, Ytr = load("train", "pixels"); Xva, Yva = load("val", "pixels")
        tr = Pixels(Xtr, Ytr, mean, std); va = Pixels(Xva, Yva, mean, std)
    nw = a.workers if a.model == "cnn" else 0
    bs = min(hp["batch_size"], len(tr))           # tiny test AOIs have few tiles
    tl = DataLoader(tr, bs, shuffle=True, num_workers=nw, drop_last=len(tr) > bs)
    vl = DataLoader(va, hp["batch_size"] * 2, shuffle=False, num_workers=nw)

    model = build(a.model, len(months), len(bands), months, k, wf).to(device)
    print(f"{a.model}: {n_params(model):,} parameters; {len(tr)} train / {len(va)} val samples")
    crit = nn.CrossEntropyLoss(weight=class_weights(Ytr, k).to(device), ignore_index=255)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    steps, warm = epochs * len(tl), tc["warmup_epochs"] * len(tl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    best, hist, stem = -1.0, [], f"{a.model}_seed{a.seed}"
    t_start = time.time()
    for ep in range(1, epochs + 1):
        model.train(); t0 = time.time(); tot = 0.0
        for x, y in tl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        s = scores(evaluate(model, vl, k, device), names)
        hist.append({"epoch": ep, "train_loss": tot / len(tl), "val_oa": s["overall_accuracy"],
                     "val_macro_f1": s["macro_f1"], "lr": sched.get_last_lr()[0], "sec": time.time() - t0})
        print(f"ep {ep:3d} loss {tot/len(tl):.4f}  val OA {s['overall_accuracy']:.4f}  macroF1 {s['macro_f1']:.4f}")
        if s["macro_f1"] > best:
            best = s["macro_f1"]
            torch.save(model.state_dict(), mdir / f"{stem}.pt")
            best_ep, best_scores = ep, s
    pd.DataFrame(hist).to_csv(mdir / f"{stem}_history.csv", index=False)

    # ---- ONNX export of the best checkpoint (CPU, fp32)
    model.load_state_dict(torch.load(mdir / f"{stem}.pt", map_location="cpu")); model = model.cpu().eval()
    T, B = len(months), len(bands)
    if a.model == "cnn":
        dummy = torch.zeros(1, T * B, 64, 64)
        axes = {"x": {0: "n", 2: "h", 3: "w"}, "logits": {0: "n", 2: "h", 3: "w"}}
    else:
        dummy = torch.zeros(8, T, B)
        axes = {"x": {0: "n"}, "logits": {0: "n"}}
    onnx_path = mdir / f"{stem}.onnx"
    torch.onnx.export(model, dummy, onnx_path, input_names=["x"], output_names=["logits"],
                      dynamic_axes=axes, opset_version=17, dynamo=False)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    test_in = torch.randn(*((2, T * B, 128, 128) if a.model == "cnn" else (300, T, B)))
    diff = np.abs(sess.run(None, {"x": test_in.numpy()})[0] - model(test_in).detach().numpy()).max()
    print(f"ONNX export ok, max |torch - onnx| = {diff:.2e}")

    meta = {"model": a.model, "seed": a.seed, "params": n_params(model), "epochs": epochs,
            "best_epoch": best_ep, "val": best_scores, "train_minutes": (time.time() - t_start) / 60,
            "device": device, "torch": torch.__version__, "months": months, "bands": bands,
            "onnx_max_abs_diff": float(diff), "hyperparameters": hp}
    json.dump(meta, open(mdir / f"{stem}_meta.json", "w"), indent=1)
    print(f"best val macro-F1 {best:.4f} at epoch {best_ep}; saved {onnx_path}")


if __name__ == "__main__":
    main()
