"""Day 1: compare candidate AOIs (configs/aois.yaml) with real data.

For each AOI (50 x 50 km):
  * Sentinel-2 L2A acquisition dates per month in the year (all months!),
    and how many of them are "clear" (scene cloud cover < 20 %)
  * WorldCover 2021 class shares (100 m quick look)
Outputs:
  outputs/tables/aoi_scene_counts.csv, outputs/tables/aoi_class_shares.csv
  outputs/figures/aoi_clear_dates.png, outputs/figures/aoi_class_shares.png
  outputs/aois.geojson  (open it on https://geojson.io to see the boxes on a map)

    python scripts/02_aoi_report.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from edc import config  # noqa: E402
from edc.aoi import aoi_grid, geojson_feature, lonlat_bbox  # noqa: E402
from edc.cloudnative import open_catalog  # noqa: E402
from edc.labels import WC_COLORS, WC_NAMES, load_worldcover  # noqa: E402

CLEAR = 20  # % scene cloud cover counted as "clear"


def scene_counts(bbox, year):
    items = open_catalog().search(collections=["sentinel-2-l2a"], bbox=bbox,
                                  datetime=f"{year}-01-01/{year}-12-31").item_collection()
    rows = [{"date": it.datetime.date(), "month": it.datetime.month,
             "cloud": it.properties.get("eo:cloud_cover", 100),
             "tile": it.properties.get("s2:mgrs_tile", "")} for it in items]
    df = pd.DataFrame(rows)
    # one acquisition date can have several MGRS tiles: use the best (min cloud) per date
    per_date = df.groupby(["date", "month"], as_index=False)["cloud"].min()
    out = []
    for m in range(1, 13):
        d = per_date[per_date.month == m]
        out.append({"month": m, "dates": len(d), "clear_dates": int((d.cloud < CLEAR).sum()),
                    "median_cloud": float(d.cloud.median()) if len(d) else np.nan})
    return pd.DataFrame(out), sorted(df.tile.unique())


def main():
    wf = config.workflow()
    cfg = config.aois()
    year = wf["year"]
    counts, shares, feats = [], [], []
    for name, a in cfg["aois"].items():
        print(f"--- {name}")
        gb = aoi_grid(a, a["size_km"], wf)
        bbox = lonlat_bbox(gb)
        feats.append(geojson_feature(gb, name))
        sc, tiles = scene_counts(bbox, year)
        sc.insert(0, "aoi", name)
        counts.append(sc)
        print(sc.to_string(index=False), "\nMGRS tiles:", tiles)

        wc = load_worldcover(gb, year, coarse=True).compute().values
        vals, n = np.unique(wc[wc > 0], return_counts=True)
        for v, c in zip(vals, n):
            shares.append({"aoi": name, "code": int(v), "class": WC_NAMES.get(int(v), str(v)),
                           "share_pct": 100 * c / n.sum()})

    counts = pd.concat(counts)
    shares = pd.DataFrame(shares)
    tdir, fdir = config.tables_dir(), config.figures_dir()
    counts.to_csv(tdir / "aoi_scene_counts.csv", index=False)
    shares.to_csv(tdir / "aoi_class_shares.csv", index=False)
    with open(config.path("outputs", "aois.geojson"), "w") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f)

    # ---- figure 1: clear dates per month
    names = list(cfg["aois"])
    fig, ax = plt.subplots(figsize=(9, 3.6))
    w = 0.8 / len(names)
    for i, nm in enumerate(names):
        d = counts[counts.aoi == nm]
        ax.bar(d.month + (i - (len(names) - 1) / 2) * w, d.clear_dates, w, label=nm)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel(f"Month ({year})")
    ax.set_ylabel(f"Dates with scene cloud < {CLEAR}%")
    ax.legend(frameon=False, ncol=len(names))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fdir / "aoi_clear_dates.png", dpi=200)

    # ---- figure 2: class shares
    piv = shares.pivot_table(index="aoi", columns="code", values="share_pct", fill_value=0).loc[names]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    left = np.zeros(len(piv))
    for code in piv.columns:
        ax.barh(piv.index, piv[code], left=left, color=WC_COLORS.get(code, "#999"),
                label=WC_NAMES.get(code, code), edgecolor="white", linewidth=0.5)
        left += piv[code].values
    ax.set_xlabel("Share of AOI (%)")
    ax.legend(frameon=False, ncol=5, bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fdir / "aoi_class_shares.png", dpi=200)

    # ---- summary to choose
    months = wf["months"]
    summ = []
    for nm in names:
        c = counts[(counts.aoi == nm) & counts.month.isin(months)]
        s = piv.loc[nm]
        s = s[s > 0] / 100
        summ.append({"aoi": nm, f"clear_dates_m{months[0]}-{months[-1]}": int(c.clear_dates.sum()),
                     "min_clear_in_a_month": int(c.clear_dates.min()),
                     "classes_over_2pct": int((s > 0.02).sum()),
                     "class_entropy": round(float(-(s * np.log(s)).sum()), 2),
                     "cropland_pct": round(float(piv.loc[nm].get(40, 0)), 1),
                     "builtup_pct": round(float(piv.loc[nm].get(50, 0)), 1)})
    summ = pd.DataFrame(summ)
    summ.to_csv(tdir / "aoi_summary.csv", index=False)
    print("\n================ SUMMARY (higher = better) ================")
    print(summ.to_string(index=False))
    print("\nPick an AOI with: no month at 0 clear dates, high entropy (balanced classes),"
          "\nand set `selected:` in configs/aois.yaml. Figures in outputs/figures/.")


if __name__ == "__main__":
    main()
