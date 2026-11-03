"""Load configs and build standard paths."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def workflow():
    return load_yaml(ROOT / "configs" / "workflow.yaml")


def aois():
    return load_yaml(ROOT / "configs" / "aois.yaml")


def get_aoi(name=None):
    cfg = aois()
    name = name or cfg["selected"]
    if name not in cfg["aois"]:
        raise KeyError(f"AOI '{name}' not in configs/aois.yaml: {list(cfg['aois'])}")
    a = dict(cfg["aois"][name])
    a["name"] = name
    return a


def tag(aoi_name, size_km):
    """Standard id used in all file names, e.g. 'kolomna_10km'."""
    return f"{aoi_name}_{int(size_km)}km"


def path(*parts, mkdir_parent=True):
    p = ROOT.joinpath(*parts)
    if mkdir_parent:
        p.parent.mkdir(parents=True, exist_ok=True)
    return p


# Standard locations -------------------------------------------------------
def cube_dir(system, aoi_tag):
    """Folder with one Zarr store per month: m04.zarr, m05.zarr, ..."""
    d = ROOT / "data" / "cubes" / system / aoi_tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def labels_path(aoi_tag):
    return path("data", "labels", f"{aoi_tag}.zarr")


def dataset_dir(aoi_tag):
    d = ROOT / "data" / "datasets" / aoi_tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def models_dir(aoi_tag):
    d = ROOT / "models" / aoi_tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def maps_dir(system):
    d = ROOT / "outputs" / "maps" / system
    d.mkdir(parents=True, exist_ok=True)
    return d


def figures_dir():
    d = ROOT / "outputs" / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def tables_dir():
    d = ROOT / "outputs" / "tables"
    d.mkdir(parents=True, exist_ok=True)
    return d


def timings_csv():
    return path("outputs", "benchmarks", "timings.csv")
