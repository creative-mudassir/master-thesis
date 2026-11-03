"""Experiment (not part of the workflow): effect of the scene cloud filter on the monthly composites.

Rebuilds months of the cloud-native cube with several values of max_scene_cloud_cover and
compares the composites. Optionally shows which SCL classes removed the water pixels.
Does NOT change configs/workflow.yaml and does NOT touch the real cube or timings.csv.

    python scripts/03b_test_scene_filter.py --size-km 10 --months 9 --filters 60 80 100 --scl-water
    python scripts/03b_test_scene_filter.py --size-km 10 --months 4 5 6 7 8 9 10 --filters 60 80

A failed build (network error) is retried up to --retries times; every attempt is logged.

Outputs (<mm> = m09 for one month, m04-m10 for several):
  data/cubes/cloudnative_scenefilter_test/<tag>/mXX_cc<F>.zarr     test composites
  outputs/tables/scene_filter_test_<mm>_<tag>.csv                    one row per (month, filter) + total rows
  outputs/tables/scene_filter_test_<mm>_<tag>_scl_water.csv          SCL classes of water pixels (--scl-water)
  outputs/benchmarks/scene_filter_test.csv                           timings (separate from timings.csv)
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

from edc import config  # noqa: E402
from edc.aoi import aoi_geobox, lonlat_bbox  # noqa: E402
from edc.bench import new_run_id, stage  # noqa: E402
from edc.cloudnative import load_month, mask_and_median, offset_mask_for, search_s2  # noqa: E402

SCL_NAMES = {0: "no data", 1: "saturated", 2: "dark/shadow area", 3: "cloud shadow", 4: "vegetation",
             5: "not vegetated", 6: "water", 7: "unclassified", 8: "cloud medium", 9: "cloud high",
             10: "thin cirrus", 11: "snow"}
WATER, BUILT = 6, 4      # training class ids (workflow.yaml labels.remap)


def build(m, f, items, gbox, wf, zpath, common):
    """Download + mask + median + write one test composite (one timed stage). Returns the loaded dataset."""
    with stage(f"W2W3_m{m:02d}_maxcloud{f}", **common) as info:
        ds = load_month(items, gbox, wf["bands"], wf["scl_band"], wf["resampling"], wf["dask_chunks"])
        off = offset_mask_for(ds, items) if wf["harmonise_baseline_04"] else None
        comp = mask_and_median(ds, wf["bands"], wf["scl_band"], wf["cloud_mask"]["remove_scl_classes"], off)
        shutil.rmtree(zpath, ignore_errors=True)
        comp.to_zarr(zpath, mode="w")
        info["note"] = f"{len(items)} items, {ds.sizes['time']} dates"
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", default=None)
    ap.add_argument("--size-km", type=float, default=10)
    ap.add_argument("--months", type=int, nargs="+", default=[9])
    ap.add_argument("--filters", type=int, nargs="+", default=[60, 80, 100])
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--scl-water", action="store_true", help="SCL classes of water pixels (largest filter)")
    a = ap.parse_args()

    wf = config.workflow()
    aoi = config.get_aoi(a.aoi)
    tag = config.tag(aoi["name"], a.size_km)
    gbox = aoi_geobox(aoi["lat"], aoi["lon"], a.size_km, wf["grid"]["crs"], wf["grid"]["resolution"])
    out = config.cube_dir("cloudnative_scenefilter_test", tag)
    bench_csv = config.path("outputs", "benchmarks", "scene_filter_test.csv")
    lab = xr.open_zarr(config.labels_path(tag))["label"].values
    water, not_built = lab == WATER, (lab != BUILT) & (lab != 255)
    run = new_run_id()
    common = dict(system="cloudnative_test", aoi=aoi["name"], area_km=a.size_km, n_months=1,
                  run_id=run, csv_path=bench_csv)

    rows, scl_rows = [], []
    for m in a.months:
        for f in a.filters:
            # STAC query is "cloud cover < f": for 100 use no limit, so 100%-cloudy scenes are included too
            limit = 101 if f >= 100 else f
            items = search_s2(lonlat_bbox(gbox), wf["year"], [m], limit)
            zpath = out / f"m{m:02d}_cc{f}.zarr"
            for attempt in range(1, a.retries + 2):          # 1 try + up to `retries` retries
                try:
                    ds = build(m, f, items, gbox, wf, zpath, common)
                    break
                except Exception as e:                        # already logged as "error" by stage()
                    print(f"m{m:02d} cc{f} attempt {attempt} failed: {type(e).__name__}: {str(e)[:150]}", flush=True)
                    if attempt == a.retries + 1:
                        raise
            sec = pd.read_csv(bench_csv).iloc[-1]["seconds"]

            res = xr.open_zarr(zpath)
            cnt, b2 = res.valid_count.values, res.B02.values
            rows.append({
                "month": m, "max_scene_cloud_cover": f, "n_items": len(items), "n_dates": int(ds.sizes["time"]),
                "scene_cloud_pct": ";".join(f"{it.properties['eo:cloud_cover']:.0f}" for it in
                                            sorted(items, key=lambda i: i.datetime)),
                "no_valid_pct": round(100 * float((cnt == 0).mean()), 2),
                "water_empty_pct": round(100 * float((cnt[water] == 0).mean()), 2),
                "bright_b02_gt2000_nonbuilt_pct": round(100 * float(((b2 > 2000) & not_built).sum() / not_built.sum()), 2),
                "median_valid_count": float(np.median(cnt)),
                "seconds_download_composite_write": sec,
                "attempts": attempt,
            })
            print(rows[-1], flush=True)

            # SCL class of every water pixel on every date (only for the largest filter: it has all dates)
            if a.scl_water and f == max(a.filters):
                scl = ds[wf["scl_band"]].compute()
                for t in scl.time.values:
                    v = scl.sel(time=t).values[water].astype("float64")
                    v[~np.isfinite(v) | (v < 0) | (v > 11)] = 0      # outside the scene footprint -> "no data"
                    h = np.bincount(v.astype(np.int64), minlength=12)
                    date = str(np.datetime64(t, "D"))
                    cc = [it.properties["eo:cloud_cover"] for it in items if str(it.datetime.date()) == date]
                    for c in range(12):
                        if h[c]:
                            scl_rows.append({"date": date, "scene_cloud_pct": round(max(cc), 1) if cc else None,
                                             "scl_class": c, "scl_name": SCL_NAMES[c],
                                             "removed_by_mask": c in wf["cloud_mask"]["remove_scl_classes"],
                                             "water_pixels_pct": round(100 * h[c] / len(v), 2)})

    t1 = pd.DataFrame(rows)
    # total row per filter: sums for counts/time; percentages are means over months
    # (every month has the same pixels, so the mean of the % equals the % over all month-pixels)
    if len(a.months) > 1:
        tot = t1.groupby("max_scene_cloud_cover").agg(
            n_items=("n_items", "sum"), n_dates=("n_dates", "sum"),
            no_valid_pct=("no_valid_pct", "mean"), water_empty_pct=("water_empty_pct", "mean"),
            bright_b02_gt2000_nonbuilt_pct=("bright_b02_gt2000_nonbuilt_pct", "mean"),
            seconds_download_composite_write=("seconds_download_composite_write", "sum"),
            attempts=("attempts", "sum")).round(2).reset_index()
        tot["month"] = "total"
        t1 = pd.concat([t1, tot], ignore_index=True)
    mm = f"m{a.months[0]:02d}" if len(a.months) == 1 else f"m{min(a.months):02d}-m{max(a.months):02d}"
    t1.to_csv(config.tables_dir() / f"scene_filter_test_{mm}_{tag}.csv", index=False)
    pd.set_option("display.width", 220)
    print(t1.drop(columns="scene_cloud_pct").to_string(index=False))
    if a.scl_water:
        t2 = pd.DataFrame(scl_rows)
        t2.to_csv(config.tables_dir() / f"scene_filter_test_{mm}_{tag}_scl_water.csv", index=False)
        print(t2.to_string(index=False))
    sys.stdout.flush()
    os._exit(0)   # GDAL/dask network threads can keep the process alive after the work is done


if __name__ == "__main__":
    main()
