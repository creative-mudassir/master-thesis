"""Day 1: snow and clouds per month for the selected AOI -> evidence for decision D2 (months).

Uses only STAC item metadata (no pixel download):
  s2:snow_ice_percentage  - % of the scene (whole MGRS tile) classified as snow/ice
  eo:cloud_cover          - % of the scene covered by clouds
All Sentinel-2 L2A items of the year over the AOI are used (no cloud filter).

Outputs:
  outputs/tables/<aoi>_snow_by_month.csv
  outputs/figures/<aoi>_snow_by_month.png

    python scripts/02b_snow_report.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from edc import config  # noqa: E402
from edc.aoi import aoi_grid, lonlat_bbox  # noqa: E402
from edc.cloudnative import open_catalog  # noqa: E402

CLEAR = 20  # % scene cloud cover counted as "clear" (same as 02_aoi_report.py)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", default=None)
    a = ap.parse_args()
    wf = config.workflow()
    aoi = config.get_aoi(a.aoi)
    year, used = wf["year"], wf["months"]
    gb = aoi_grid(aoi, aoi["size_km"], wf)

    items = open_catalog().search(collections=["sentinel-2-l2a"], bbox=lonlat_bbox(gb),
                                  datetime=f"{year}-01-01/{year}-12-31").item_collection()
    df = pd.DataFrame([{"date": it.datetime.date(), "month": it.datetime.month,
                        "tile": it.properties.get("s2:mgrs_tile", ""),
                        "cloud": it.properties.get("eo:cloud_cover"),
                        "snow": it.properties.get("s2:snow_ice_percentage")} for it in items])
    print(f"{len(df)} items, {df.snow.isna().sum()} without s2:snow_ice_percentage")

    rows = []
    for m in range(1, 13):
        d = df[df.month == m]
        clear_items = d[d.cloud < CLEAR]
        per_date = d.groupby("date")["cloud"].min()           # best tile per date, as in 02_aoi_report
        rows.append({
            "month": m,
            "used_in_workflow": m in used,
            "n_items": len(d),
            "median_snow_pct": d.snow.median(),
            # snow on clear scenes: the land surface is visible, so this is what a composite would show
            "n_clear_items": len(clear_items),
            "median_snow_pct_clear_items": clear_items.snow.median(),
            "clear_dates": int((per_date < CLEAR).sum()),
            "median_cloud_pct": d.cloud.median(),
        })
    out = pd.DataFrame(rows).round(2)
    tab = config.tables_dir() / f"{aoi['name']}_snow_by_month.csv"
    out.to_csv(tab, index=False)
    print(out.to_string(index=False))

    # figure: two panels on the same month axis (no dual y-axis)
    blue, orange, ink, muted, band = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#ecebe8"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 4.6), sharex=True,
                                   gridspec_kw={"height_ratios": [1.2, 1], "hspace": 0.25})
    for ax in (ax1, ax2):
        ax.axvspan(min(used) - 0.5, max(used) + 0.5, color=band, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(muted)
        ax.tick_params(colors=muted, labelsize=8)
        ax.grid(axis="y", color=band, linewidth=0.8, zorder=0)
    # all scenes: clouds hide snow, so cloudy months look snow-free; clear scenes show the real surface
    w = 0.38
    ax1.bar(out.month - w / 2 - 0.01, out.median_snow_pct, w, color=blue, zorder=2, label="all scenes")
    ax1.bar(out.month + w / 2 + 0.01, out.median_snow_pct_clear_items, w, color=orange, zorder=2,
            label=f"clear scenes (cloud < {CLEAR}%)")
    for _, r in out[out.n_clear_items == 0].iterrows():
        ax1.text(r.month, 8, "no clear\nscene", fontsize=6.5, color=muted, ha="center", va="bottom")
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Median snow/ice\n% of scene", fontsize=8, color=ink)
    ax1.legend(frameon=False, fontsize=8, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    ax1.text(min(used), 62, "months used (D2)", fontsize=8, color=muted)
    ax1.set_title(f"{aoi['name'].capitalize()} {year}: snow and clear dates per month "
                  f"(Sentinel-2 L2A metadata, {len(df)} scenes)", fontsize=9, color=ink, loc="left")
    ax2.bar(out.month, out.clear_dates, 0.6, color=blue, zorder=2)
    ax2.set_ylabel(f"Dates with\ncloud < {CLEAR}%", fontsize=8, color=ink)
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    fig.savefig(config.figures_dir() / f"{aoi['name']}_snow_by_month.png", dpi=200, bbox_inches="tight")
    print("saved", tab)


if __name__ == "__main__":
    main()
