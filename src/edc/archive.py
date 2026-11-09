"""Local Sentinel-2 archive (D16): download once, then every system reads the same local files.

Layout of one archive (one AOI, one size, e.g. data/archive/sentinel-2-l2a/austria_tulln_50km/):
    catalog.json                        STAC catalogue (relative links -> works inside Docker too)
    archive_info.json                   what was downloaded (AOI, box, months, scene filter, scene ids)
    <scene id>/<scene id>.json          STAC item of the scene, assets point to the local files
    <scene id>/B02.tif ... SCL.tif      one Cloud-Optimized GeoTIFF per band

Each file is a WINDOW of the original Planetary Computer COG: same CRS, same pixel grid, same values
(no resampling, no change of data type). Only the part of the scene that covers the AOI box + buffer
is kept. Reprojection to the common grid (W2) is still done by every system itself.
"""
import json
import math
import os
from pathlib import Path


def clip_box(gbox, buffer_m):
    """(crs, (left, bottom, right, top)) of the grid box + buffer, in the grid CRS."""
    b = gbox.boundingbox
    return str(gbox.crs), (b.left - buffer_m, b.bottom - buffer_m, b.right + buffer_m, b.top + buffer_m)


def _window(src, bounds):
    """Pixel window of `src` that covers bounds (in src CRS), rounded OUTWARDS to whole pixels."""
    from rasterio.windows import Window
    t = src.transform                                  # t.a = pixel width, t.e = -pixel height
    l, b, r, top = bounds
    c0 = max(0, math.floor((l - t.c) / t.a))
    c1 = min(src.width, math.ceil((r - t.c) / t.a))
    r0 = max(0, math.floor((t.f - top) / -t.e))
    r1 = min(src.height, math.ceil((t.f - b) / -t.e))
    if c1 <= c0 or r1 <= r0:
        raise ValueError("the scene does not overlap the archive box")
    return Window(c0, r0, c1 - c0, r1 - r0)


def _write_cog(path, arr, profile):
    """Write one band as a Cloud-Optimized GeoTIFF (deflate, 512 px tiles, no overviews).

    No overviews: every system must read the full-resolution pixels. Written to a .part file
    first, so an interrupted download never leaves a half file under the real name.
    """
    from rasterio.io import MemoryFile
    from rasterio.shutil import copy as rio_copy
    tmp = path.with_name(path.name + ".part")
    with MemoryFile() as mem:
        with mem.open(**profile) as d:
            d.write(arr, 1)
            rio_copy(d, str(tmp), driver="COG", COMPRESS="DEFLATE", PREDICTOR="2",
                     BLOCKSIZE="512", OVERVIEWS="NONE")
    os.replace(tmp, path)


def download_scene(item_dict, assets, clip_crs, clip_bounds, scene_dir):
    """Download the AOI window of every asset of one scene and write its local STAC item.

    Runs in a child process (edc.robust.run_in_child), so item_dict is a signed STAC item as dict.
    The item JSON is written LAST: a scene folder without it is unfinished and is redone on rerun.
    Returns the size of the scene folder in MB.
    """
    import pystac
    import rasterio
    from rasterio.transform import array_bounds
    from rasterio.warp import transform_bounds
    from shapely.geometry import box, mapping, shape

    from .cloudnative import GDAL_NET

    item = pystac.Item.from_dict(item_dict)
    scene_dir = Path(scene_dir)
    scene_dir.mkdir(parents=True, exist_ok=True)
    size = 0
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", **GDAL_NET):
        for name in assets:
            with rasterio.open(item.assets[name].href) as src:
                # the box is defined in the grid CRS; a scene can be in another UTM zone
                win = _window(src, transform_bounds(clip_crs, src.crs, *clip_bounds, densify_pts=21))
                arr = src.read(1, window=win)
                tr = src.window_transform(win)
                profile = dict(driver="GTiff", width=arr.shape[1], height=arr.shape[0], count=1,
                               dtype=arr.dtype, crs=src.crs, transform=tr, nodata=src.nodata)
            out = scene_dir / f"{name}.tif"
            _write_cog(out, arr, profile)
            size += out.stat().st_size
            a = item.assets[name]
            a.href = f"./{name}.tif"                   # relative to the item JSON
            a.media_type = pystac.MediaType.COG
            a.extra_fields["proj:shape"] = [arr.shape[0], arr.shape[1]]
            a.extra_fields["proj:transform"] = list(tr)[:6]
            a.extra_fields["proj:bbox"] = list(array_bounds(arr.shape[0], arr.shape[1], tr))

    for k in list(item.assets):                         # keep only the downloaded bands
        if k not in assets:
            del item.assets[k]
    item.clear_links()                                  # no links back to Planetary Computer
    ll = transform_bounds(clip_crs, "EPSG:4326", *clip_bounds, densify_pts=21)
    foot = shape(item.geometry).intersection(box(*ll))  # footprint inside the archive box
    item.geometry, item.bbox = mapping(foot), list(foot.bounds)
    item.properties["edc:archive_box"] = {"crs": clip_crs, "bounds": list(clip_bounds)}

    tmp = scene_dir / f"{item.id}.json.part"
    tmp.write_text(json.dumps(item.to_dict(include_self_link=False, transform_hrefs=False), indent=1))
    os.replace(tmp, scene_dir / f"{item.id}.json")
    return round(size / 2**20, 1)


def scene_done(adir, item_id):
    return (Path(adir) / item_id / f"{item_id}.json").exists()


def write_catalog(adir, item_ids, info):
    """catalog.json over the finished scenes (relative links) + archive_info.json."""
    import pystac
    adir = Path(adir)
    cat = pystac.Catalog(id=adir.name, description=f"Local Sentinel-2 L2A archive {adir.name} (EDC thesis, D16)")
    for i in item_ids:
        cat.add_item(pystac.Item.from_file(str(adir / i / f"{i}.json")))
    cat.normalize_hrefs(str(adir))                      # -> <id>/<id>.json, as written by download_scene
    cat.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)
    (adir / "archive_info.json").write_text(json.dumps(info, indent=2))


def open_archive(adir):
    """(items, info) of a local archive; asset hrefs become absolute local paths."""
    import pystac
    adir = Path(adir)
    if not (adir / "catalog.json").exists():
        raise FileNotFoundError(f"no local archive in {adir}: run scripts/03a_download_archive.py first")
    items = list(pystac.Catalog.from_file(str(adir / "catalog.json")).get_items(recursive=True))
    for it in items:
        for a in it.assets.values():
            a.href = a.get_absolute_href()
    return items, json.loads((adir / "archive_info.json").read_text())
