"""Day 2: reference cube, system `xarray` (W1 discovery, W2 cube, W3 mask + monthly median).

    python scripts/03a_download_archive.py --size-km 10   # once: local archive (D16)
    python scripts/03_build_cube.py --size-km 10 --archive-km 10   # FIRST: quick test
    python scripts/03_build_cube.py --size-km 50          # full AOI (reads the 50 km archive)

Reads ONLY the local archive data/archive/sentinel-2-l2a/<aoi>_<archive-km>km/ (no network), and
writes data/cubes/xarray/<aoi>_<size>km/m04.zarr ... m10.zarr + grid.json,
a true-colour quick look per month and the valid-observation counts.

  * W1: the scenes are found in the local STAC catalogue (catalog.json), by month, scene filter, footprint;
  * each month is built in square spatial PARTS (--part-km, default 10 km = 1000 x 1000 px):
    each part is loaded, masked, composited and written into its own region of the month's Zarr
    (keeps the memory small at 50 km);
  * finished parts are listed in m0X.parts.json: a rerun (WITHOUT --overwrite) continues
    where it stopped; a month only becomes m0X.zarr when all its parts are written.
Every part is timed into outputs/benchmarks/timings.csv (stage W2W3_cube_composite_m0X_pNN);
the month time = sum of its part rows.
(The network safety nets - watchdog, new tokens, retries - are in 03a; local reads do not need them,
and a child process per part would add start-up time to every timed part.)
"""
import argparse
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from odc.geo.xr import xr_zeros  # noqa: E402

from edc import config  # noqa: E402
from edc.aoi import aoi_grid, grid_spec, lonlat_bbox  # noqa: E402
from edc.bench import keep_awake, new_run_id, stage  # noqa: E402
from edc.archive import open_archive  # noqa: E402
from edc.cloudnative import build_part, items_by_month, items_over, open_cube  # noqa: E402

SYSTEM = "xarray"


def quicklook(cube, count, months, out_png):
    n = len(months)
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.6), squeeze=False)   # 2 x n even for 1 month
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


def split_parts(gbox, part_px):
    """Square parts of the grid: list of (y0, y1, x0, x1) pixel windows (edge parts may be smaller)."""
    ny, nx = gbox.shape.y, gbox.shape.x
    return [(y0, min(y0 + part_px, ny), x0, min(x0 + part_px, nx))
            for y0 in range(0, ny, part_px) for x0 in range(0, nx, part_px)]


def write_empty_month(path, gbox, bands, part_px, attrs):
    """Month store full of zeros (= no data), chunked so that every part is exactly one Zarr chunk."""
    ch = (part_px, part_px)
    data = {b: xr_zeros(gbox, dtype="uint16", chunks=ch) for b in bands}
    data["valid_count"] = xr_zeros(gbox, dtype="uint8", chunks=ch)
    xr.Dataset(data, attrs=attrs).to_zarr(path, mode="w")


def save_progress(path, state):
    """Write the list of finished parts atomically (a crash mid-write must not corrupt it)."""
    tmp = path.with_name(path.name + ".new")
    tmp.write_text(json.dumps(state))
    os.replace(tmp, path)


def build_month(m, its, gbox, wf, out, part_px, common):
    zpath = out / f"m{m:02d}.zarr"
    tmp = zpath.with_name(zpath.name + ".tmp")
    progress = out / f"m{m:02d}.parts.json"
    parts = split_parts(gbox, part_px)
    # what this month's result depends on: if any of it changed, unfinished parts are not reused
    recipe = {"items": sorted(it.id for it in its), "part_px": part_px, "grid": grid_spec(gbox),
              "bands": wf["bands"], "remove_scl_classes": wf["cloud_mask"]["remove_scl_classes"],
              "harmonise_baseline_04": wf["harmonise_baseline_04"]}
    state = json.loads(progress.read_text()) if progress.exists() and tmp.exists() else None
    if state is None or state["recipe"] != recipe:
        shutil.rmtree(tmp, ignore_errors=True)
        write_empty_month(tmp, gbox, wf["bands"], part_px, {"month": m, "n_items": len(its), "system": SYSTEM})
        state = {"recipe": recipe, "done": []}
        save_progress(progress, state)
    elif state["done"]:
        print(f"month {m}: resuming, {len(state['done'])}/{len(parts)} parts already written")

    for k, (y0, y1, x0, x1) in enumerate(parts):
        if k in state["done"]:
            continue
        sub = gbox[y0:y1, x0:x1]
        its_p = items_over(its, lonlat_bbox(sub))     # only scenes that touch this part
        if its_p:                                      # (no scene -> the zeros = no data stay)
            # odc-stac is lazy: loading (W2) and mask/median (W3) run together on compute,
            # so they are timed as one stage (D7).
            with stage(f"W2W3_cube_composite_m{m:02d}_p{k:02d}", **common) as info:
                comp, n_dates = build_part(its_p, sub, wf, local=True)
                comp = comp.drop_vars(list(comp.coords))          # grid coords are in the store already
                comp.to_zarr(tmp, region={"y": slice(y0, y1), "x": slice(x0, x1)})
                info["note"] = f"{len(its_p)} items, {n_dates} dates"
        state["done"].append(k)
        save_progress(progress, state)
        print(f"month {m}: part {k + 1}/{len(parts)} written ({len(its_p)} items)", flush=True)

    shutil.rmtree(zpath, ignore_errors=True)
    tmp.rename(zpath)           # the month only "exists" when every part is written
    progress.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", default=None)
    ap.add_argument("--size-km", type=float, default=10)
    ap.add_argument("--months", type=int, nargs="*", default=None, help="override months (scaling runs)")
    ap.add_argument("--overwrite", action="store_true",
                    help="rebuild months that already exist (REQUIRED for benchmark repeats). "
                         "After a crash, rerun WITHOUT it to continue.")
    ap.add_argument("--part-km", type=float, default=10, help="side of one spatial part")
    ap.add_argument("--archive-km", type=float, default=50, help="which local archive to read (its box size)")
    ap.add_argument("--out-system", default=SYSTEM,
                    help="output folder name in data/cubes/ (tests only; timings then go to parts_test.csv)")
    a = ap.parse_args()
    keep_awake()                     # idle sleep would pause a long run

    wf = config.workflow()
    aoi = config.get_aoi(a.aoi)
    months = a.months or wf["months"]
    tag = config.tag(aoi["name"], a.size_km)
    out = config.cube_dir(a.out_system, tag)
    gbox = aoi_grid(aoi, a.size_km, wf)
    part_px = int(round(a.part_km * 1000 / wf["grid"]["resolution"]))
    (out / "grid.json").write_text(json.dumps(grid_spec(gbox), indent=2))
    print(f"AOI {tag}: grid {gbox.shape.y} x {gbox.shape.x} px, months {months}, "
          f"{len(split_parts(gbox, part_px))} parts of {part_px} px per month")

    run = new_run_id()
    common = dict(system=SYSTEM if a.out_system == SYSTEM else a.out_system, aoi=aoi["name"],
                  area_km=a.size_km, n_months=len(months), run_id=run)
    if a.out_system != SYSTEM:
        common["csv_path"] = config.path("outputs", "benchmarks", "parts_test.csv")

    if a.overwrite:
        # remove ALL months to be rebuilt first: if the run then crashes, a rerun without
        # --overwrite cannot mistake an old (e.g. different scene filter) month for a finished one
        for m in months:
            for p in (out / f"m{m:02d}.zarr", out / f"m{m:02d}.zarr.tmp"):
                shutil.rmtree(p, ignore_errors=True)
            (out / f"m{m:02d}.parts.json").unlink(missing_ok=True)

    adir = config.archive_dir(config.tag(aoi["name"], a.archive_km))
    max_cloud = wf["scene_filter"]["max_scene_cloud_cover"]
    with stage("W1_discovery", **common) as info:
        items, ainfo = open_archive(adir)
        # the archive must contain everything this cube needs
        if ainfo["aoi"] != aoi["name"] or ainfo["size_km"] < a.size_km:
            raise ValueError(f"archive {adir.name} is for {ainfo['aoi']} {ainfo['size_km']} km")
        if not set(months) <= set(ainfo["months"]):
            raise ValueError(f"archive has months {ainfo['months']}, this cube needs {months}")
        if ainfo["max_scene_cloud_cover"] < max_cloud:
            raise ValueError("the archive was filled with a stricter scene filter than workflow.yaml")
        items = [it for it in items if it.datetime.year == wf["year"] and it.datetime.month in months
                 and it.properties["eo:cloud_cover"] < max_cloud]
        items = items_over(items, lonlat_bbox(gbox))
        by_m = items_by_month(items)
        info["note"] = f"{len(items)} items from local archive {adir.name}"
    print({m: len(by_m.get(m, [])) for m in months})

    for m in months:
        if (out / f"m{m:02d}.zarr").exists():
            print(f"skip month {m} (exists)")
            continue
        its = by_m.get(m, [])
        if not its:
            print(f"WARNING: no scenes in month {m}; writing an all no-data month (gap-fill rule applies)")
        build_month(m, its, gbox, wf, out, part_px, common)

    cube, count = open_cube(out, months, wf["bands"])
    quicklook(cube, count, months, config.figures_dir() / f"cube_quicklook_{a.out_system}_{tag}.png")
    empty = {int(m): float((count.sel(month=m) == 0).mean()) for m in months}
    print("share of pixels with NO valid observation per month:", empty)
    (out / "qa.json").write_text(json.dumps({"no_valid_share": empty}, indent=2))
    print(f"done -> {out}")


if __name__ == "__main__":
    code = 0
    try:
        main()                       # returns only after every month and qa.json are written
    except KeyboardInterrupt:
        code = 130
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    # Forced exit: GDAL/dask threads sometimes kept the process alive after "done"
    # (observed 2026-11-03). All Zarr writes are finished by now, so nothing is lost.
    os._exit(code)
