"""ESA WorldCover 2021 labels on the common grid."""
import numpy as np

# Official WorldCover colours (for figures)
WC_COLORS = {
    10: "#006400", 20: "#ffbb22", 30: "#ffff4c", 40: "#f096ff", 50: "#fa0000",
    60: "#b4b4b4", 70: "#f0f0f0", 80: "#0064c8", 90: "#0096a0", 95: "#00cf75", 100: "#fae6a0",
}
WC_NAMES = {
    10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland", 50: "Built-up",
    60: "Bare/sparse", 70: "Snow/ice", 80: "Water", 90: "Herb. wetland", 95: "Mangroves",
    100: "Moss/lichen",
}


def remap(wc, remap_dict):
    """WorldCover codes -> class ids (0..K-1), everything else -> 255."""
    lut = np.full(256, 255, dtype=np.uint8)
    for code, cid in remap_dict.items():
        if int(code) < 256:
            lut[int(code)] = int(cid)
    wc = np.asarray(wc)
    out = np.full(wc.shape, 255, dtype=np.uint8)
    ok = (wc >= 0) & (wc < 256)
    out[ok] = lut[wc[ok].astype(np.int64)]
    return out


def class_colors(remap_dict, n_classes):
    """Colour per training class id, taken from WorldCover colours."""
    cols = ["#000000"] * n_classes
    for code, cid in remap_dict.items():
        if int(cid) < n_classes:
            cols[int(cid)] = WC_COLORS[int(code)]
    return cols


def load_worldcover(gbox, year=2021, coarse=False):
    """WorldCover map on gbox (nearest). coarse=True -> 100 m quick look (AOI report)."""
    import odc.stac
    import planetary_computer
    from .cloudnative import open_catalog

    bbox = gbox.extent.to_crs("EPSG:4326").boundingbox
    items = open_catalog().search(
        collections=["esa-worldcover"],
        bbox=(bbox.left, bbox.bottom, bbox.right, bbox.top),
        datetime=f"{year}-01-01/{year}-12-31",
    ).item_collection()
    if len(items) == 0:
        raise RuntimeError("No WorldCover items found for this AOI/year")
    g = gbox.zoom_to(resolution=100) if coarse else gbox
    ds = odc.stac.load(items, bands=["map"], geobox=g, resampling="nearest",
                       patch_url=planetary_computer.sign, chunks={"x": 2048, "y": 2048})
    return ds["map"].max("time")  # one 2021 layer; max merges tile borders
