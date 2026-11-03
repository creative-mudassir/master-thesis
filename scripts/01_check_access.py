"""Day 1: test access to every platform with a tiny Sentinel-2 query.

    python scripts/01_check_access.py                 # all
    python scripts/01_check_access.py --only pc       # Planetary Computer only
    python scripts/01_check_access.py --only gee --gee-project my-ee-project
    python scripts/01_check_access.py --only openeo

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
from edc.aoi import aoi_geobox, lonlat_bbox  # noqa: E402


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


def check_gee(bbox, project):
    import ee
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()            # opens browser once
        ee.Initialize(project=project)
    geom = ee.Geometry.Rectangle(list(bbox))
    n = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
         .filterBounds(geom).filterDate("2021-07-01", "2021-08-01").size().getInfo())
    return f"{n} images in COPERNICUS/S2_SR_HARMONIZED, July 2021 (project={project})"


def check_openeo(bbox):
    import openeo
    con = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()  # browser/device login
    meta = con.describe_collection("SENTINEL2_L2A")
    try:
        acc = con.describe_account()
        who = acc.get("name") or acc.get("user_id", "")
    except Exception:
        who = "?"
    bands = [b["name"] for b in meta.get("cube:dimensions", {}).get("bands", {}).get("values", [])] or \
            meta.get("summaries", {}).get("eo:bands", [])
    return f"logged in as {who}; SENTINEL2_L2A available ({len(bands)} bands listed)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["pc", "gee", "openeo"])
    ap.add_argument("--gee-project", default=None, help="your Google Cloud project id registered for Earth Engine")
    a = ap.parse_args()

    aoi = config.get_aoi()
    gb = aoi_geobox(aoi["lat"], aoi["lon"], 10)
    bbox = lonlat_bbox(gb)

    checks = {"pc": lambda: check_pc(bbox),
              "gee": lambda: check_gee(bbox, a.gee_project),
              "openeo": lambda: check_openeo(bbox)}
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
