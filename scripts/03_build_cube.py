"""Day 2: cloud-native reference cube (W1 discovery, W2 cube, W3 mask + monthly median).

    python scripts/03_build_cube.py --size-km 10          # FIRST: quick test (~5-10 min)
    python scripts/03_build_cube.py --size-km 50          # full AOI (long; run overnight)

Writes data/cubes/cloudnative/<aoi>_<size>km/m04.zarr ... m10.zarr + grid.json,
a true-colour quick look per month and the valid-observation counts.
Every stage is timed into outputs/benchmarks/timings.csv.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

from edc import config  # noqa: E402
from edc.aoi import aoi_geobox, grid_spec, lonlat_bbox  # noqa: E402
from edc.bench import new_run_id, stage  # noqa: E402
from edc.cloudnative import (items_by_month, load_month, mask_and_median,  # noqa: E402
                             offset_mask_for, open_cube, search_s2)

SYSTEM = "cloudnative"


def quicklook(cube, count, months, out_png):
    n = len(months)
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.6))
    axes = np.atleast_2d(axes)
    step = max(1, cube.sizes["x"] // 600)            # downsample for the figure
    for i, m in enumerate(months):
        rgb = cube.sel(month=m, band=["B04", "B03", "B02"]).isel(
            y=slice(None, None, step), x=slice(None, None, step)).values.astype("float32")
        rgb = np.moveaxis(rgb, 0, -1) / 3000.0       # simple stretch
        axes[0, i].imshow(np.clip(rgb, 0, 1))
        axes[0, i].set_title(f"month {m:02d}", fontsize=9)
        c = count.sel(month=m).isel(y=slice(None, None, step), x=slice(None, None, step)).values
        im = axes[1, i].imshow(c, vmin=0, vmax=max(1, int(count.max())), cmap="viridis")
        for ax in axes[:, i]:
            ax.set_xticks([]); ax.set_yticks([])
    axes[1, 0].set_ylabel("valid obs.")
    fig.colorbar(im, ax=axes[1, :].tolist(), shrink=0.8, label="valid observations")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", default=None)
    ap.add_argument("--size-km", type=float, default=10)
    ap.add_argument("--months", type=int, nargs="*", default=None, help="override months (scaling runs)")
    ap.add_argument("--overwrite", action="store_true",
                    help="rebuild months that already exist (REQUIRED for benchmark repeats)")
    a = ap.parse_args()

    wf = config.workflow()
    aoi = config.get_aoi(a.aoi)
    months = a.months or wf["months"]
    tag = config.tag(aoi["name"], a.size_km)
    out = config.cube_dir(SYSTEM, tag)
    gbox = aoi_geobox(aoi["lat"], aoi["lon"], a.size_km, wf["grid"]["crs"], wf["grid"]["resolution"])
    (out / "grid.json").write_text(json.dumps(grid_spec(gbox), indent=2))
    print(f"AOI {tag}: grid {gbox.shape.y} x {gbox.shape.x} px, months {months}")

    run = new_run_id()
    common = dict(system=SYSTEM, aoi=aoi["name"], area_km=a.size_km, n_months=len(months), run_id=run)

    with stage("W1_discovery", **common) as info:
        items = search_s2(lonlat_bbox(gbox), wf["year"], months, wf["scene_filter"]["max_scene_cloud_cover"])
        by_m = items_by_month(items)
        info["note"] = f"{len(items)} items"
    print({m: len(by_m.get(m, [])) for m in months})

    for m in months:
        zpath = out / f"m{m:02d}.zarr"
        if zpath.exists() and not a.overwrite:
            print(f"skip month {m} (exists)")
            continue
        its = by_m.get(m, [])
        if not its:
            print(f"WARNING: no scenes in month {m}; writing an all no-data month (gap-fill rule applies)")
            from odc.geo.xr import xr_zeros
            empty = {b: xr_zeros(gbox, dtype="uint16") for b in wf["bands"]}
            empty["valid_count"] = xr_zeros(gbox, dtype="uint8")
            xr.Dataset(empty).to_zarr(zpath, mode="w")
            continue
        with stage(f"W2W3_cube_composite_m{m:02d}", **common) as info:
            # odc-stac is lazy: loading (W2) and mask/median (W3) run together on compute,
            # so they are timed as one stage (note this in the paper).
            ds = load_month(its, gbox, wf["bands"], wf["scl_band"], wf["resampling"], wf["dask_chunks"])
            off = offset_mask_for(ds, its) if wf["harmonise_baseline_04"] else None
            comp = mask_and_median(ds, wf["bands"], wf["scl_band"],
                                   wf["cloud_mask"]["remove_scl_classes"], off)
            comp.attrs.update({"month": m, "n_items": len(its), "system": SYSTEM})
            # write to a temporary store first, so a failed/interrupted month never
            # leaves a half-written m0X.zarr that a rerun would skip
            tmp = zpath.with_name(zpath.name + ".tmp")
            shutil.rmtree(tmp, ignore_errors=True)
            comp.to_zarr(tmp, mode="w")
            shutil.rmtree(zpath, ignore_errors=True)   # --overwrite: remove the old month first
            tmp.rename(zpath)
            info["note"] = f"{len(its)} items, {ds.sizes['time']} dates"

    cube, count = open_cube(out, months, wf["bands"])
    quicklook(cube, count, months, config.figures_dir() / f"cube_quicklook_{SYSTEM}_{tag}.png")
    empty = {int(m): float((count.sel(month=m) == 0).mean()) for m in months}
    print("share of pixels with NO valid observation per month:", empty)
    (out / "qa.json").write_text(json.dumps({"no_valid_share": empty}, indent=2))
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
