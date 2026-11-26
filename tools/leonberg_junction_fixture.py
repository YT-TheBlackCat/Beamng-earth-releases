"""Capture the map data round one junction as a small offline fixture (tests/kreuzfixtures.py loads it).

    python tools/leonberg_junction_fixture.py NAME LAT LON [--keep-m 250] [--cache ~/.cache/beamng-earth] [--host cloud]
    -> server/tests/fixtures/kreuz_NAME.npz

Stores the roads, infrastructure and road_routes layers of every z14 cell within --keep-m of the point, only the
features within keep_m of it (interchanges._keep_near), plus the sources: whether the lane tags (OSM, LaneTags) were
there, which the capture host could not always fetch. No elevation: the junction tests run on flat ground.
The cache holds the downloads (as in capture_interchange_fixture.py); nothing else is written.
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "server" / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import geodesy  # noqa: E402
import mapcells  # noqa: E402
import mapdata as mapdatamod  # noqa: E402
import netfetch  # noqa: E402
from capture_interchange_fixture import plain  # noqa: E402
from interchanges import _keep_near  # noqa: E402

LAYERS = (("roads", "roads"), ("infrastructure", "infra"), ("road_routes", "routes"))
LANE_KEYS = ("osm_lane_parts", "osm_priority_road")


def has_lane_tags(features):
    """True when a roads feature carries an OSM lane or priority tag (the LaneTags join worked)."""
    for k in LANE_KEYS:
        col = features.attrs.get(k)
        if col is not None and any(x is not None for x in col):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("lat", type=float)
    ap.add_argument("lon", type=float)
    ap.add_argument("--keep-m", type=float, default=250.0)
    ap.add_argument("--cache", default="~/.cache/beamng-earth")
    ap.add_argument("--host", choices=("cloud", "local"), default="cloud")
    a = ap.parse_args()
    cache = str(Path(a.cache).expanduser().resolve())
    md = mapdatamod.MapData(cache, fetcher=netfetch.Fetcher())
    pad = a.keep_m / 111320.0
    keys = sorted(mapcells.cells_for_points(np.array([a.lat]), np.array([a.lon]), pad,
                                            pad / np.cos(np.radians(a.lat))), key=str)
    frame = geodesy.EnuFrame(a.lat, a.lon)
    data = {"origin": np.array([a.lat, a.lon]), "keep_m": np.array(a.keep_m), "cells": np.array([str(k) for k in keys]),
            "release": np.array(md.release)}
    lane_tags = False
    for i, key in enumerate(keys):
        cell = md.cell(key, layers=tuple(layer for layer, _ in LAYERS))
        for layer, tag in LAYERS:
            f = cell.layers.get(layer)
            n = 0 if f is None else len(f)
            print(f"cell {key} {layer}: {n} features", flush=True)
            if not n:
                continue
            if layer == "roads":
                lane_tags = lane_tags or has_lane_tags(f)
            arrays = f.to_arrays(f"{tag}{i}.")
            arrays.pop(f"{tag}{i}.en", None)           # localized again on load
            arrays = _keep_near(arrays, f"{tag}{i}.", frame, a.keep_m)
            if len(arrays[f"{tag}{i}.ids"]):
                data.update(arrays)
    sources = {"lane_tags": bool(lane_tags), "captured": date.today().isoformat(), "host": a.host}
    data["sources"] = np.array(json.dumps(sources))
    out = ROOT / "server" / "tests" / "fixtures" / f"kreuz_{a.name}.npz"
    np.savez_compressed(out, **plain(data))
    print(f"{out}: {len(keys)} cells, {out.stat().st_size / 1e3:.0f} KB, sources {sources}")


if __name__ == "__main__":
    main()
