"""Cloud-native reference system: STAC (Planetary Computer) + odc-stac + xarray/dask + Zarr."""
from collections import defaultdict

import numpy as np
import xarray as xr

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"


# ---------------------------------------------------------------- W1 discovery
def open_catalog():
    import planetary_computer
    import pystac_client
    return pystac_client.Client.open(PC_STAC, modifier=planetary_computer.sign_inplace)


def search_s2(bbox_ll, year, months, max_cloud):
    """All Sentinel-2 L2A items over bbox in the selected months of `year`."""
    cat = open_catalog()
    items = cat.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox_ll,
        datetime=f"{year}-01-01/{year}-12-31",
        query={"eo:cloud_cover": {"lt": max_cloud}},
    ).item_collection()
    return [it for it in items if it.datetime.month in months]


def resign(items):
    """Fresh copies of `items` with NEW download tokens (PC SAS tokens expire after ~45 min).

    planetary_computer.sign() skips URLs that already carry a token, so the old token is
    removed from every asset URL first, and the token cache is emptied so a new one is fetched.
    """
    import planetary_computer
    import planetary_computer.sas
    planetary_computer.sas.TOKEN_CACHE.clear()
    out = []
    for it in items:
        c = it.clone()
        for asset in c.assets.values():
            asset.href = asset.href.split("?")[0]
        out.append(planetary_computer.sign(c, copy=False))
    return out


def items_over(items, bbox_ll):
    """Only the items whose footprint touches bbox_ll (west, south, east, north in degrees)."""
    from shapely.geometry import box, shape
    b = box(*bbox_ll)
    return [it for it in items if shape(it.geometry).intersects(b)]


def items_by_month(items):
    out = defaultdict(list)
    for it in items:
        out[it.datetime.month].append(it)
    return dict(out)


def baseline(item):
    try:
        return float(item.properties.get("s2:processing_baseline", "0"))
    except ValueError:
        return 0.0


# --------------------------------------------- W2 cube + W3 mask & composite
def mask_and_median(ds, bands, scl_band, remove_classes, offset_mask=None):
    """Pure xarray function (testable without internet).

    ds: Dataset with dims (time, y, x), integer DN bands + SCL.
    offset_mask: optional bool DataArray over `time`: True where DN - 1000 must be applied.
    Returns Dataset: median DN per band (uint16, 0 = no data) + valid_count (uint8).
    """
    scl = ds[scl_band]
    valid = ~scl.isin(remove_classes)
    valid = valid & (ds[bands[0]] > 0)
    out = {}
    for b in bands:
        v = ds[b].astype("float32")
        if offset_mask is not None:
            v = xr.where(offset_mask, v - 1000.0, v)
        v = v.where(valid & (v > 0))
        med = v.chunk({"time": -1}).median("time", skipna=True) if v.chunks else v.median("time", skipna=True)
        out[b] = med.fillna(0).round().clip(0, 65535).astype("uint16")
    out["valid_count"] = valid.sum("time").clip(0, 255).astype("uint8")
    res = xr.Dataset(out)
    for c in ("spatial_ref",):
        if c in ds.coords:
            res = res.assign_coords({c: ds.coords[c]})
    return res


def load_month(items, gbox, bands, scl_band, resampling, chunk):
    """Lazy (dask) cube for one month on the exact common grid."""
    import odc.stac
    import planetary_computer
    # Retry flaky downloads inside GDAL instead of silently dropping scenes.
    # fail_on_error=True: if a scene still cannot be read, the month FAILS (and is logged as
    # an error stage) rather than producing a composite with missing scenes. Just rerun the
    # script: finished months are skipped.
    odc.stac.configure_rio(cloud_defaults=True, GDAL_HTTP_MAX_RETRY=10, GDAL_HTTP_RETRY_DELAY=2)
    ds = odc.stac.load(
        items,
        bands=list(bands) + [scl_band],
        geobox=gbox,
        groupby="solar_day",
        chunks={"x": chunk, "y": chunk},
        resampling={"*": resampling["spectral"], scl_band: resampling["scl"]},
        patch_url=planetary_computer.sign,
        fail_on_error=True,
    )
    return ds


def offset_mask_for(ds, items):
    """True for time steps whose scenes have processing baseline >= 04.00."""
    by_day = defaultdict(list)
    for it in items:
        by_day[np.datetime64(it.datetime.date(), "D")].append(baseline(it) >= 4.0)
    flags = [any(by_day.get(np.datetime64(t, "D"), [False])) for t in ds.time.values]
    return xr.DataArray(np.array(flags), dims="time", coords={"time": ds.time})


# ---------------------------------------------------------------- reading
def open_cube(cube_dir, months, bands):
    """Read monthly Zarr stores -> DataArray (month, band, y, x) uint16 + valid_count (month, y, x)."""
    import pandas as pd
    arrs, counts = [], []
    for m in months:
        ds = xr.open_zarr(cube_dir / f"m{m:02d}.zarr")
        arrs.append(ds[list(bands)].to_array("band"))
        counts.append(ds["valid_count"])
    idx = pd.Index(list(months), name="month")
    cube = xr.concat(arrs, dim=idx)
    count = xr.concat(counts, dim=idx)
    return cube.transpose("month", "band", "y", "x"), count
