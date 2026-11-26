"""Junction fixtures: the map data round one junction (fixtures/kreuz_NAME.npz, tools/leonberg_junction_fixture.py).

    kreuzfixtures.load("K1") -> SimpleNamespace(origin, frame, roads, infra, routes, sources, keep_m)

roads, infra and routes are lists of mapgeom.Features, one per z14 cell, localized to the cell's own ENU frame (the
layout Features.localize gives them). sources is the capture's JSON: lane_tags, captured, host.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import geodesy  # noqa: E402
import interchanges  # noqa: E402  (_structured: the object columns read back)
import mapcells  # noqa: E402
import mapgeom  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
LAYERS = (("roads", "roads"), ("infrastructure", "infra"), ("road_routes", "routes"))


def load(name):
    with np.load(FIXTURES / f"kreuz_{name}.npz", allow_pickle=False) as a:
        data = {k: interchanges._structured(k, a[k]) for k in a.files}
    lat, lon = (float(v) for v in data["origin"])
    out = {tag: [] for _, tag in LAYERS}
    for i, key in enumerate(data["cells"]):
        cell = mapcells.CellKey(*(int(v) for v in str(key).split("/")))
        cell_frame = geodesy.EnuFrame(*cell.center())
        for layer, tag in LAYERS:
            prefix = f"{tag}{i}."
            if f"{prefix}ids" in data:
                out[tag].append(mapgeom.Features.from_arrays(layer, data, prefix).localize(cell_frame))
    return SimpleNamespace(origin=(lat, lon), frame=geodesy.EnuFrame(lat, lon), roads=out["roads"],
                           infra=out["infra"], routes=out["routes"],
                           sources=json.loads(str(data["sources"])), keep_m=float(data["keep_m"]))
