"""Day 3: spatial block split + training data (CNN tiles, pixel samples) + normalisation stats.

    python scripts/05_make_dataset.py --size-km 50

Writes data/datasets/<tag>/:
  split.json, stats.json,
  tiles_{train,val,test}_X.npy (N,T,B,64,64 uint16) + _Y.npy (N,64,64 uint8)
  pixels_{train,val,test}_X.npy (N,T,B uint16) + _Y.npy (N, uint8)
and outputs/figures/split_<tag>.png
Only the reference (cloud-native) cube is used for training. Models are trained ONCE.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

from edc import config  # noqa: E402
from edc.cloudnative import open_cube  # noqa: E402
from edc.dataset import (SPLITS, RunningStats, assign_blocks, block_slices,  # noqa: E402
                         split_mask, tiles_in_block)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", default=None)
    ap.add_argument("--size-km", type=float, default=50)
    ap.add_argument("--system", default="xarray")
    a = ap.parse_args()

    wf = config.workflow()
    aoi = config.get_aoi(a.aoi)
    tag = config.tag(aoi["name"], a.size_km)
    out = config.dataset_dir(tag)
    months, bands = wf["months"], wf["bands"]
    names = wf["labels"]["class_names"]
    k = len(names)
    sp, dsc = wf["split"], wf["dataset"]

    cube, _ = open_cube(config.cube_dir(a.system, tag), months, bands)
    lab = xr.open_zarr(config.labels_path(tag))["label"].values
    H, W = lab.shape
    assert cube.shape[-2:] == (H, W), f"cube {cube.shape} and labels {lab.shape} grids differ!"

    block_px = int(sp["block_km"] * 1000 / wf["grid"]["resolution"])
    if H < 3 * block_px:  # small test AOI: use smaller blocks so all 3 splits exist
        block_px = max(dsc["tile_size"] * 2 + 2 * sp["margin_px"], H // 3)
        print(f"small AOI -> block size reduced to {block_px} px (test only)")
    split = assign_blocks(lab, block_px, sp["ratios"], sp["seed"], k)
    (out / "split.json").write_text(json.dumps(split))
    print({s: split["split"].count(s) for s in SPLITS}, "blocks; min class share", round(split["min_class_share"], 3))

    rng = np.random.default_rng(sp["seed"])
    T, B = len(months), len(bands)
    stats = RunningStats((T, B))
    acc = {s: {"px": [], "py": []} for s in SPLITS}
    margin, tile = sp["margin_px"], dsc["tile_size"]

    def good_tiles(ys, xs):
        """Tiles fully inside the block interior with at least 50% labelled pixels."""
        return [(y, x) for (y, x) in tiles_in_block(ys, xs, tile, margin)
                if (lab[y:y + tile, x:x + tile] != 255).mean() >= 0.5]

    # Pass 1 (labels only, fast): count tiles per split and pre-allocate the .npy files ON DISK,
    # so the ~3 GB of tiles never has to fit in RAM twice.
    n_tiles = {s: 0 for s in SPLITS}
    for b, s in enumerate(split["split"]):
        n_tiles[s] += len(good_tiles(*block_slices(b, split["nbx"], block_px, H, W)))
    TX = {s: np.lib.format.open_memmap(out / f"tiles_{s}_X.npy", "w+", np.uint16, (n_tiles[s], T, B, tile, tile))
          for s in SPLITS}
    TY = {s: np.lib.format.open_memmap(out / f"tiles_{s}_Y.npy", "w+", np.uint8, (n_tiles[s], tile, tile))
          for s in SPLITS}
    pos = {s: 0 for s in SPLITS}
    print("tiles per split:", n_tiles)

    for b, s in enumerate(split["split"]):
        ys, xs = block_slices(b, split["nbx"], block_px, H, W)
        X = cube.isel(y=ys, x=xs).values            # (T,B,h,w) uint16, loaded block by block
        Y = lab[ys, xs]
        if s == "train":
            stats.add(X.reshape(T, B, -1)[:, :, ::7])   # subsample pixels for the stats
        # CNN tiles
        for (y, x) in good_tiles(ys, xs):
            ly, lx = y - ys.start, x - xs.start
            TX[s][pos[s]] = X[:, :, ly:ly + tile, lx:lx + tile]
            TY[s][pos[s]] = Y[ly:ly + tile, lx:lx + tile]
            pos[s] += 1
        # pixel samples (interior only)
        inner = np.zeros_like(Y, bool)
        inner[margin:-margin or None, margin:-margin or None] = True
        cand = np.flatnonzero(inner & (Y != 255))
        if len(cand):
            pick = rng.choice(cand, min(len(cand), dsc["pixels_per_block"]), replace=False)
            py, px = np.unravel_index(pick, Y.shape)
            acc[s]["px"].append(X[:, :, py, px].transpose(2, 0, 1))  # (n,T,B)
            acc[s]["py"].append(Y[py, px])
        print(f"block {b:3d} {s:5s}: {pos[s]}/{n_tiles[s]} tiles")

    mean, std = stats.result()
    json.dump({"months": months, "bands": bands, "class_names": names,
               "mean": mean.tolist(), "std": std.tolist(), "block_px": block_px},
              open(out / "stats.json", "w"), indent=1)

    cap = dsc["max_train_pixels_per_class"]
    for s in SPLITS:
        TX[s].flush(); TY[s].flush()
        px = np.concatenate(acc[s]["px"]); py = np.concatenate(acc[s]["py"])
        if s == "train":  # cap very frequent classes (class imbalance)
            keep = []
            for c in range(k):
                idx = np.flatnonzero(py == c)
                keep.append(rng.choice(idx, min(len(idx), cap), replace=False))
            keep = rng.permutation(np.concatenate(keep))
            px, py = px[keep], py[keep]
        np.save(out / f"pixels_{s}_X.npy", px); np.save(out / f"pixels_{s}_Y.npy", py)
        print(f"{s:5s}: {n_tiles[s]} tiles, {len(px)} pixels, pixel classes {np.bincount(py, minlength=k).tolist()}")

    # split figure
    smap = np.full((H, W), np.nan)
    for i, s in enumerate(SPLITS):
        smap[split_mask(H, W, split, s, margin)] = i
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    step = max(1, W // 800)
    ax.imshow(smap[::step, ::step], cmap=matplotlib.colors.ListedColormap(["#3b7dd8", "#f2a541", "#d84a3b"]),
              vmin=-0.5, vmax=2.5, interpolation="nearest")
    ax.set_title("Spatial blocks: train (blue) / val (orange) / test (red)", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(config.figures_dir() / f"split_{tag}.png", dpi=170, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
