"""W0 (once): download the Sentinel-2 L2A scenes of the AOI into the local archive (D16).

    python scripts/03a_download_archive.py --size-km 10      # FIRST: small test archive
    python scripts/03a_download_archive.py                   # 50 km archive (serves 10/25/50 km cubes)

For every scene of the year/months that passes the scene filter (workflow.yaml scene_filter), the
10 bands + SCL are cut to the AOI box + archive.buffer_m and saved as local COGs, in the scene's own
CRS and pixel grid (no resampling). Result (see src/edc/archive.py):
    data/archive/sentinel-2-l2a/<aoi>_<size>km/catalog.json, archive_info.json, <scene id>/...
All systems then read only these files, so their timings do not depend on the network.

Robust against the unstable network (same safety nets as the old network cube build):
  * one scene = one unit of work, run in its own child process (watchdog, edc.robust): if it takes
    longer than --timeout-min (default 20 min) it is killed and retried;
  * every attempt gets NEW download tokens (PC tokens expire after ~45 min);
  * GDAL stall limits turn a frozen download into an error (edc.cloudnative.GDAL_NET);
  * a failed scene is retried up to --retries times, waiting 30, 60, 90 ... s;
  * finished scenes (their STAC item JSON exists) are skipped: just rerun after a crash.
Every attempt is timed into outputs/benchmarks/timings.csv (system "archive", stage W0_download).
This time is NOT part of the system comparison.
"""
import argparse
import datetime as dt
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rasterio.warp import transform_bounds  # noqa: E402

from edc import config  # noqa: E402
from edc.aoi import aoi_grid  # noqa: E402
from edc.archive import clip_box, download_scene, scene_done, write_catalog  # noqa: E402
from edc.bench import keep_awake, new_run_id, stage  # noqa: E402
from edc.cloudnative import resign, search_s2  # noqa: E402
from edc.robust import run_in_child  # noqa: E402

WAIT_S = 30          # wait before retry k = WAIT_S * k seconds (30, 60, 90, ...)


def with_retries(label, retries, fn):
    """fn(attempt) with up to `retries` retries; every failure is printed (stage() logs it too)."""
    for attempt in range(1, retries + 2):
        try:
            return fn(attempt)
        except Exception as e:
            print(f"{label} attempt {attempt} failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
            if attempt == retries + 1:
                raise
            time.sleep(WAIT_S * attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", default=None)
    ap.add_argument("--size-km", type=float, default=50, help="box of the archive (50 km serves all sizes)")
    ap.add_argument("--months", type=int, nargs="*", default=None)
    ap.add_argument("--retries", type=int, default=5, help="retries per failed scene")
    ap.add_argument("--timeout-min", type=float, default=20, help="watchdog: kill + retry a scene after this")
    ap.add_argument("--overwrite", action="store_true", help="delete this archive first and download again")
    ap.add_argument("--test-stall", type=int, default=None, metavar="N",
                    help="TEST ONLY: the first attempt of the N-th scene freezes (checks the watchdog)")
    a = ap.parse_args()
    keep_awake()                     # idle sleep would pause the download and expire the tokens

    wf = config.workflow()
    aoi = config.get_aoi(a.aoi)
    months = a.months or wf["months"]
    tag = config.tag(aoi["name"], a.size_km)
    adir = config.archive_dir(tag)
    if a.overwrite:
        shutil.rmtree(adir)
        adir = config.archive_dir(tag)
    gbox = aoi_grid(aoi, a.size_km, wf)
    crs, bounds = clip_box(gbox, wf["archive"]["buffer_m"])
    assets = list(wf["bands"]) + [wf["scl_band"]]
    max_cloud = wf["scene_filter"]["max_scene_cloud_cover"]
    print(f"archive {adir}\nbox {bounds} ({crs}), months {months}, scenes with cloud < {max_cloud}%")

    common = dict(system="archive", aoi=aoi["name"], area_km=a.size_km, n_months=len(months), run_id=new_run_id())

    def search(attempt):
        with stage("W0_search", **common) as info:
            found = search_s2(transform_bounds(crs, "EPSG:4326", *bounds, densify_pts=21),
                              wf["year"], months, max_cloud)
            info["note"] = f"{len(found)} items, attempt {attempt}"
        return sorted(found, key=lambda it: it.datetime)
    items = with_retries("search", a.retries, search)
    todo = [it for it in items if not scene_done(adir, it.id)]
    print(f"{len(items)} scenes, {len(items) - len(todo)} already in the archive, {len(todo)} to download")

    for n, it in enumerate(todo, 1):
        def download(attempt):
            stall = 10**6 if a.test_stall == n and attempt == 1 else 0      # stall test only
            with stage("W0_download", **common) as info:
                signed = resign([it])[0]                                     # fresh tokens every attempt
                mb, child_s = run_in_child(download_scene,
                                           (signed.to_dict(), assets, crs, bounds, str(adir / it.id)),
                                           60 * a.timeout_min, stall)
                info["note"] = f"{it.id}, {mb} MB, attempt {attempt}, child {child_s} s"
            return mb
        mb = with_retries(f"scene {it.id}", a.retries, download)
        print(f"scene {n}/{len(todo)} {it.id}: {mb} MB", flush=True)

    write_catalog(adir, [it.id for it in items], {
        "aoi": aoi["name"], "size_km": a.size_km, "grid_crs": crs, "box": list(bounds),
        "buffer_m": wf["archive"]["buffer_m"], "year": wf["year"], "months": months,
        "max_scene_cloud_cover": max_cloud, "assets": assets, "n_items": len(items),
        "item_ids": [it.id for it in items], "source": "Planetary Computer STAC sentinel-2-l2a",
        "written": dt.datetime.now().isoformat(timespec="seconds")})
    mb = sum(f.stat().st_size for f in adir.rglob("*.tif")) / 2**20
    print(f"done -> {adir / 'catalog.json'}: {len(items)} scenes, {mb:.0f} MB")


if __name__ == "__main__":
    code = 0
    try:
        main()                       # returns only after every scene and the catalogue are written
    except KeyboardInterrupt:
        code = 130
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)                   # GDAL network threads can keep the process alive after "done"
