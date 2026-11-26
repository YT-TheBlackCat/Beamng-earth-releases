#!/usr/bin/env python3
"""Triangles per building tile (256 m) around the Leonberg test spots, with LoD2 geometry on or off.

    python3 tools/leonberg_tile_triangles.py [--spots T1,T3] [--cache DIR]      # runs both variants
    python3 tools/leonberg_tile_triangles.py --kit [--spots T1,T3]             # facade kit off / on

Each variant runs in its own process (BNE_LOD2_GEOM is read when server/lod2geom.py is imported). Per spot the tiles
within 1 tile of the spot centre: tile count, median, 90th percentile and maximum of the triangles per tile
(buildings.build_parts, all pieces of a tile summed). Reads the cache like the other Leonberg tools.
"""
import argparse
import json
import logging
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "server"))


def measure(spot_ids, cache):
    os.environ.setdefault("BNE_ROOF_PHOTO", "off")
    import numpy as np
    import blockworker as bw
    import buildings as B
    import geodesy
    import leonberg_buildings_report as R
    import roads

    logging.disable(logging.CRITICAL)
    st = bw._State(cache, False, True, True, True, True)
    bw._s = st
    out = {}
    for sid, _name, lat, lon in [s for s in R.SPOTS if s[0] in spot_ids]:
        frame = geodesy.EnuFrame(lat, lon)
        counts = []
        for tx in range(-1, 2):
            for ty in range(-1, 2):
                items, _ = B.road_tile_items(frame, tx, ty, st.md, R._ground(frame, st),
                                             urban_fn=st.surface.urban if st.surface is not None else None,
                                             widths=st.widths)
                counts.append(sum(len(m[2]) for m in B.build_parts(items)))
        c = np.array(counts)
        out[sid] = {"tiles": len(c), "median": int(np.median(c)), "p90": int(np.percentile(c, 90)), "max": int(c.max())}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spots", default="T1,T2,T3,T4,T5,T6,T7,T8")
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/beamng-earth"))
    ap.add_argument("--kit", action="store_true", help="compare the facade kit off / on (BNE_FACADE_KIT) instead of LoD2")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    ids = [s.strip() for s in a.spots.split(",") if s.strip()]
    if a.child:
        print(json.dumps(measure(ids, a.cache)))
        return
    res = {}
    variants = (("ohne Fassadenbausteine", {"BNE_FACADE_KIT": "off"}), ("mit Fassadenbausteinen", {"BNE_FACADE_KIT": "on"})) \
        if a.kit else (("ohne LoD2-Geometrie", {"BNE_LOD2_GEOM": "off"}), ("mit LoD2-Geometrie", {"BNE_LOD2_GEOM": "on"}))
    for label, extra in variants:
        env = dict(os.environ, **extra)
        p = subprocess.run([sys.executable, __file__, "--child", "--spots", a.spots, "--cache", a.cache],
                           env=env, capture_output=True, text=True, check=True)
        res[label] = json.loads(p.stdout.strip().splitlines()[-1])
    print("| Spot | Variante | Kacheln | Median | P90 | Max |\n| --- | --- | --- | --- | --- | --- |")
    for sid in ids:
        for label, d in res.items():
            r = d[sid]
            print(f"| {sid} | {label} | {r['tiles']} | {r['median']} | {r['p90']} | {r['max']} |")


if __name__ == "__main__":
    main()
