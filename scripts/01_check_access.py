"""Day 1: check the data source and the locally installed systems.

    python scripts/01_check_access.py                 # all
    python scripts/01_check_access.py --only pc       # Planetary Computer (download source of the local archive)
    python scripts/01_check_access.py --only docker   # Open Data Cube + rasdaman run in Docker
    python scripts/01_check_access.py --only gdalcubes
    python scripts/01_check_access.py --only openeo_local   # optional system; works only in an env that
                                                            # has openeo-processes-dask AND this repo's packages

Only the Planetary Computer needs internet: it is used ONCE to fill the local archive (D16).
Result table -> outputs/tables/access_check.csv (attach this to the supervisor update).
"""
import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from edc import config  # noqa: E402
from edc.aoi import aoi_grid, lonlat_bbox  # noqa: E402


def check_pc(bbox):
    import planetary_computer
    import pystac_client
    import rasterio
    from rasterio.windows import Window
    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                    modifier=planetary_computer.sign_inplace)
    items = cat.search(collections=["sentinel-2-l2a"], bbox=bbox,
                       datetime="2021-07-01/2021-07-31").item_collection()
    n = len(items)
    with rasterio.open(items[0].assets["B04"].href) as src:
        arr = src.read(1, window=Window(0, 0, 256, 256))
    wc = cat.search(collections=["esa-worldcover"], bbox=bbox,
                    datetime="2021-01-01/2021-12-31").item_collection()
    return f"{n} S2 items in July 2021; read 256x256 B04 window (mean DN {arr.mean():.0f}); {len(wc)} WorldCover items"


def run(cmd):
    """Run a command, return its first output line (raises if it is missing or fails)."""
    import subprocess
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip()[:300])
    return (r.stdout or r.stderr).strip().splitlines()[0]


def check_docker():
    """Docker runs the Open Data Cube (PostgreSQL + datacube) and rasdaman containers."""
    return run(["docker", "version", "--format", "client {{.Client.Version}}, server {{.Server.Version}}"])


def check_gdalcubes():
    """gdalcubes is used through its R package."""
    return run(["Rscript", "-e", "cat('gdalcubes', as.character(packageVersion('gdalcubes')))"])


def check_openeo_local():
    """Optional: openEO processes executed locally (openeo-processes-dask), no cloud back-end."""
    import importlib.metadata as md
    import openeo_processes_dask  # noqa: F401
    return f"openeo-processes-dask {md.version('openeo-processes-dask')}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["pc", "docker", "gdalcubes", "openeo_local"])
    a = ap.parse_args()

    checks = {"pc": lambda: check_pc(lonlat_bbox(aoi_grid(config.get_aoi(), 10, config.workflow()))),
              "docker": check_docker,
              "gdalcubes": check_gdalcubes,
              "openeo_local": check_openeo_local}
    rows = []
    for name, fn in checks.items():
        if a.only and name != a.only:
            continue
        t0 = time.time()
        try:
            msg, ok = fn(), "OK"
        except Exception as e:
            msg, ok = f"{type(e).__name__}: {e}", "FAILED"
            traceback.print_exc()
        rows.append({"system": name, "status": ok, "seconds": round(time.time() - t0, 1), "detail": msg})
        print(f"\n== {name}: {ok}\n   {msg}\n")

    out = config.tables_dir() / "access_check.csv"
    df = pd.DataFrame(rows)
    if out.exists():
        old = pd.read_csv(out)
        df = pd.concat([old[~old.system.isin(df.system)], df])
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
