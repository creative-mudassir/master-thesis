"""Day 2: cloud-native reference cube (W1 discovery, W2 cube, W3 mask + monthly median).

    python scripts/03_build_cube.py --size-km 10          # FIRST: quick test (~5-10 min)
    python scripts/03_build_cube.py --size-km 50          # full AOI (long; run overnight)

Writes data/cubes/cloudnative/<aoi>_<size>km/m04.zarr ... m10.zarr + grid.json,
a true-colour quick look per month and the valid-observation counts.

Robust against the unstable network:
  * each month is built in square spatial PARTS (--part-km, default 10 km = 1000 x 1000 px);
    each part is downloaded, masked, composited and written into its own region of the month's Zarr;
  * the STAC items get NEW download tokens before every part attempt (PC tokens expire ~45 min);
  * each part runs in its own child process (watchdog): if it takes longer than --part-timeout-min
    (default 20 min) the process is killed and the part retried. A killed process is the only way
    that really works on Windows: a thread stuck inside a GDAL download cannot be stopped;
  * a failed part is retried up to --retries times, waiting a bit longer each time;
  * finished parts are listed in m0X.parts.json: a rerun (WITHOUT --overwrite) continues
    where it stopped; a month only becomes m0X.zarr when all its parts are written.
Every part attempt is timed into outputs/benchmarks/timings.csv (stage W2W3_cube_composite_m0X_pNN);
the month time = sum of its "ok" part rows. Failed attempts are logged as "error" rows.
The row time includes starting the child process (a few s); the note has the child's own time.
"""
import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
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
from edc.aoi import aoi_geobox, grid_spec, lonlat_bbox  # noqa: E402
from edc.bench import keep_awake, new_run_id, stage  # noqa: E402
from edc.cloudnative import (items_by_month, items_over, open_cube, part_worker,  # noqa: E402
                             resign, search_s2)

SYSTEM = "cloudnative"
WAIT_S = 30          # wait before retry k = WAIT_S * k seconds (30, 60, 90, ...)


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


def run_part(its, sub, wf, zarr_path, window, timeout_s, stall_s=0):
    """Build + write one part in a child process; kill it if it runs longer than timeout_s.

    Returns (n_dates, child_seconds). Raises TimeoutError or RuntimeError so the caller retries.
    """
    ctx = mp.get_context("spawn")                     # Windows only supports "spawn"
    parent, child = ctx.Pipe(duplex=False)
    p = ctx.Process(target=part_worker, daemon=True,
                    args=(child, [it.to_dict() for it in its], sub, wf, str(zarr_path), window, stall_s))
    p.start()
    child.close()
    try:
        if not parent.poll(timeout_s):                # nothing back in time -> frozen
            raise TimeoutError(f"part did not finish within {timeout_s:.0f} s; child process killed")
        msg = parent.recv()
    except EOFError:                                  # child died without sending anything
        raise RuntimeError(f"child process died (exit code {p.exitcode})")
    finally:
        p.join(5)
        if p.is_alive():
            p.kill()                                  # TerminateProcess on Windows
            p.join(10)
    if msg[0] != "ok":
        raise RuntimeError(msg[1])
    return msg[1], msg[2]


def build_month(m, its, gbox, wf, out, part_px, retries, common, timeout_s, test_stall=None):
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
            for attempt in range(1, retries + 2):     # 1 try + up to `retries` retries
                try:
                    # odc-stac is lazy: loading (W2) and mask/median (W3) run together on compute,
                    # so they are timed as one stage (D7).
                    stall = 10**6 if test_stall == (m, k) and attempt == 1 else 0   # stall test only
                    with stage(f"W2W3_cube_composite_m{m:02d}_p{k:02d}", **common) as info:
                        n_dates, child_s = run_part(resign(its_p), sub, wf, tmp, (y0, y1, x0, x1),
                                                    timeout_s, stall)            # fresh tokens every attempt
                        info["note"] = f"{len(its_p)} items, {n_dates} dates, attempt {attempt}, child {child_s} s"
                    break
                except Exception as e:                 # already logged as an "error" row by stage()
                    print(f"m{m:02d} part {k} attempt {attempt} failed: {type(e).__name__}: {str(e)[:200]}",
                          flush=True)
                    if attempt == retries + 1:
                        raise
                    time.sleep(WAIT_S * attempt)
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
    ap.add_argument("--retries", type=int, default=5, help="retries per failed part")
    ap.add_argument("--part-timeout-min", type=float, default=20, help="watchdog: kill + retry a part after this")
    ap.add_argument("--test-stall", default=None, metavar="MONTH:PART",
                    help="TEST ONLY: first attempt of this part freezes (checks the watchdog)")
    ap.add_argument("--out-system", default=SYSTEM,
                    help="output folder name in data/cubes/ (tests only; timings then go to parts_test.csv)")
    a = ap.parse_args()
    keep_awake()                     # idle sleep would pause the run and expire the tokens

    wf = config.workflow()
    aoi = config.get_aoi(a.aoi)
    months = a.months or wf["months"]
    tag = config.tag(aoi["name"], a.size_km)
    out = config.cube_dir(a.out_system, tag)
    gbox = aoi_geobox(aoi["lat"], aoi["lon"], a.size_km, wf["grid"]["crs"], wf["grid"]["resolution"])
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

    for attempt in range(1, a.retries + 2):            # the STAC search fails on connection resets too
        try:
            with stage("W1_discovery", **common) as info:
                items = search_s2(lonlat_bbox(gbox), wf["year"], months,
                                  wf["scene_filter"]["max_scene_cloud_cover"])
                by_m = items_by_month(items)
                info["note"] = f"{len(items)} items, attempt {attempt}"
            break
        except Exception as e:
            print(f"discovery attempt {attempt} failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
            if attempt == a.retries + 1:
                raise
            time.sleep(WAIT_S * attempt)
    print({m: len(by_m.get(m, [])) for m in months})

    for m in months:
        if (out / f"m{m:02d}.zarr").exists():
            print(f"skip month {m} (exists)")
            continue
        its = by_m.get(m, [])
        if not its:
            print(f"WARNING: no scenes in month {m}; writing an all no-data month (gap-fill rule applies)")
        test_stall = tuple(int(v) for v in a.test_stall.split(":")) if a.test_stall else None
        build_month(m, its, gbox, wf, out, part_px, a.retries, common, 60 * a.part_timeout_min, test_stall)

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
    # Forced exit: GDAL/dask network threads sometimes keep the process alive after "done"
    # (observed 2026-11-03). All Zarr writes are finished by now, so nothing is lost.
    os._exit(code)
