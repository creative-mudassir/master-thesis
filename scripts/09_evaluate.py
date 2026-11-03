"""Day 5+ / Day 11: accuracy of every map on the TEST blocks + agreement between maps.

    python scripts/09_evaluate.py                      # all maps found in outputs/maps/*/
    python scripts/09_evaluate.py --size-km 50

Outputs:
  outputs/tables/accuracy_<tag>.csv     one row per (system, model): OA, kappa, macro-F1, per-class F1
  outputs/tables/agreement_<tag>.csv    pairwise agreement between systems for the same model
  outputs/figures/disagreement_<a>_vs_<b>_<model>.png
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
import xarray as xr  # noqa: E402

from edc import config  # noqa: E402
from edc.dataset import split_mask  # noqa: E402
from edc.metrics import agreement, confusion, scores  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", default=None)
    ap.add_argument("--size-km", type=float, default=50)
    a = ap.parse_args()
    wf = config.workflow()
    aoi = config.get_aoi(a.aoi)
    tag = config.tag(aoi["name"], a.size_km)
    names = wf["labels"]["class_names"]
    k = len(names)
    lab = xr.open_zarr(config.labels_path(tag))["label"].values
    split = json.load(open(config.dataset_dir(tag) / "split.json"))
    H, W = lab.shape
    # same buffer as for training data (workflow.yaml split.margin_px, decision D9)
    test = split_mask(H, W, split, "test", wf["split"]["margin_px"])

    maps = {}
    for tif in sorted((config.ROOT / "outputs" / "maps").glob(f"*/{tag}_*.tif")):
        system, model = tif.parent.name, tif.stem[len(tag) + 1:]
        with rasterio.open(tif) as src:
            maps[(system, model)] = src.read(1)

    rows = []
    for (system, model), m in maps.items():
        s = scores(confusion(lab[test], m[test], k), names)
        rows.append({"system": system, "model": model, **s})
    acc = pd.DataFrame(rows)
    acc.to_csv(config.tables_dir() / f"accuracy_{tag}.csv", index=False)
    print(acc[["system", "model", "overall_accuracy", "kappa", "macro_f1"]].to_string(index=False))

    rows = []
    models = sorted({m for _, m in maps})
    for model in models:
        systems = sorted(s for s, m in maps if m == model)
        for sa, sb in itertools.combinations(systems, 2):
            A, Bm = maps[(sa, model)], maps[(sb, model)]
            g = agreement(A, Bm, k)              # whole AOI
            rows.append({"model": model, "system_a": sa, "system_b": sb, **g})
            dis = (A != Bm) & (A != 255) & (Bm != 255)
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            step = max(1, W // 900)
            ax.imshow(dis[::step, ::step], cmap="Reds", interpolation="nearest")
            ax.set_title(f"{model}: {sa} vs {sb}\n{g['disagree_pct']:.2f}% pixels differ", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            fig.savefig(config.figures_dir() / f"disagreement_{sa}_vs_{sb}_{model}.png", dpi=170, bbox_inches="tight")
            plt.close(fig)
    if rows:
        ag = pd.DataFrame(rows)
        ag.to_csv(config.tables_dir() / f"agreement_{tag}.csv", index=False)
        print(ag.to_string(index=False))
    else:
        print("(only one system so far: no agreement table yet)")


if __name__ == "__main__":
    main()
