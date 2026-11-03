"""AOI -> exact pixel grid (GeoBox). Every system must use this same grid."""
import json

from odc.geo.geobox import GeoBox
from pyproj import Transformer


def aoi_geobox(lat, lon, size_km, crs="EPSG:32637", res=10):
    """Square grid of size_km around (lat, lon), snapped to `res` metres.

    The centre is snapped first, so 10/25/50 km grids around the same centre
    are nested and pixel-aligned.
    """
    t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    cx, cy = t.transform(lon, lat)
    cx = round(cx / res) * res
    cy = round(cy / res) * res
    half = size_km * 1000 / 2
    bbox = (cx - half, cy - half, cx + half, cy + half)
    return GeoBox.from_bbox(bbox, crs=crs, resolution=res)


def lonlat_bbox(gbox):
    """(west, south, east, north) in degrees, for STAC / GEE / openEO queries."""
    b = gbox.extent.to_crs("EPSG:4326").boundingbox
    return (b.left, b.bottom, b.right, b.top)


def grid_spec(gbox):
    """Plain-dict description of the grid (saved next to outputs, used by GEE/openEO)."""
    b = gbox.boundingbox
    return {
        "crs": str(gbox.crs),
        "resolution": float(gbox.resolution.x),
        "bounds_utm": [b.left, b.bottom, b.right, b.top],
        "shape_yx": [int(gbox.shape.y), int(gbox.shape.x)],
        "bounds_lonlat": list(lonlat_bbox(gbox)),
    }


def geojson_feature(gbox, name):
    geom = gbox.extent.to_crs("EPSG:4326").geom
    return {
        "type": "Feature",
        "properties": {"name": name},
        "geometry": json.loads(json.dumps(geom.__geo_interface__)),
    }
