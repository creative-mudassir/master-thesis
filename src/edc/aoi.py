"""AOI -> exact pixel grid (GeoBox). Every system must use this same grid."""
import json

from odc.geo.geobox import GeoBox
from pyproj import Transformer


def utm_epsg(lat, lon):
    """EPSG code of the UTM zone that contains (lat, lon): 326xx north, 327xx south.

    Zones are 6 degrees wide, zone 1 starts at 180W. (The special zones around Norway/Svalbard
    are ignored; they do not matter for a study area outside them.)
    """
    zone = min(60, int((lon + 180) // 6) + 1)
    return (32600 if lat >= 0 else 32700) + zone


def grid_crs(lat, lon, rule):
    """workflow.yaml grid.crs: "auto_utm" -> UTM zone of the AOI centre; anything else is used as is."""
    return f"EPSG:{utm_epsg(lat, lon)}" if rule == "auto_utm" else rule


def aoi_grid(aoi, size_km, wf):
    """THE common grid of an AOI (from configs/aois.yaml) at a given size. Every system uses this."""
    crs = grid_crs(aoi["lat"], aoi["lon"], wf["grid"]["crs"])
    return aoi_geobox(aoi["lat"], aoi["lon"], size_km, crs, wf["grid"]["resolution"])


def aoi_geobox(lat, lon, size_km, crs, res=10):
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
    """(west, south, east, north) in degrees, for STAC queries."""
    b = gbox.extent.to_crs("EPSG:4326").boundingbox
    return (b.left, b.bottom, b.right, b.top)


def grid_spec(gbox):
    """Plain-dict description of the grid (saved next to every cube; other systems can read it)."""
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
