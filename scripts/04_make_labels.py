"""Day 2: ESA WorldCover 2021 labels on exactly the same grid as the cube.

    python scripts/04_make_labels.py --size-km 10

Writes data/labels/<aoi>_<size>km.zarr (variables: worldcover, label),
outputs/figures/labels_<tag>.png and outputs/tables/label_shares_<tag>.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from edc import config  # noqa: E402
from edc.aoi import aoi_geobox  # noqa: E402
from edc.bench import keep_awake  # noqa: E402
from edc.labels import class_colors, load_worldcover, remap  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", default=None)
    ap.add_argument("--size-km", type=float, default=10)
    a = ap.parse_args()
    keep_awake()                     # no idle sleep during the run

    wf = config.workflow()
    aoi = config.get_aoi(a.aoi)
    tag = config.tag(aoi["name"], a.size_km)
    gbox = aoi_geobox(aoi["lat"], aoi["lon"], a.size_km, wf["grid"]["crs"], wf["grid"]["resolution"])

    wc = load_worldcover(gbox, wf["year"]).compute()
    lab = remap(wc.values, wf["labels"]["remap"])
    ds = xr.Dataset({"worldcover": wc.astype("uint8"), "label": (wc.dims, lab)}, coords=wc.coords)
    ds.to_zarr(config.labels_path(tag), mode="w")

    names = wf["labels"]["class_names"]
    k = len(names)
    cnt = np.bincount(lab[lab != 255], minlength=k)[:k]
    df = pd.DataFrame({"class": names, "pixels": cnt, "share_pct": 100 * cnt / max(1, cnt.sum())})
    df.to_csv(config.tables_dir() / f"label_shares_{tag}.csv", index=False)
    print(df.to_string(index=False))

    cols = class_colors(wf["labels"]["remap"], k)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1.1, 1]})
    show = np.where(lab == 255, np.nan, lab).astype(float)
    step = max(1, lab.shape[1] // 1000)
    ax1.imshow(show[::step, ::step], cmap=ListedColormap(cols), vmin=-0.5, vmax=k - 0.5, interpolation="nearest")
    ax1.set_title(f"WorldCover 2021 labels, {tag}", fontsize=10)
    ax1.set_xticks([]); ax1.set_yticks([])
    ax2.barh(names[::-1], df.share_pct[::-1], color=cols[::-1])
    ax2.set_xlabel("Share of pixels (%)")
    ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(config.figures_dir() / f"labels_{tag}.png", dpi=170)
    print("saved", config.labels_path(tag))


if __name__ == "__main__":
    main()
