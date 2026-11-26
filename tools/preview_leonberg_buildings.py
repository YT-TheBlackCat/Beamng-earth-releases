#!/usr/bin/env python3
"""Render the generated Leonberg building meshes outside BeamNG (CPU z-buffer, no textures).

    python3 tools/preview_leonberg_buildings.py [--spots T1,T3] [--radius 150] [--size 1600x1000]
                                                [--out DIR] [--obj] [--label TEXT] [--no-photo]

Per spot, the meshes buildings.build_parts makes from buildings.road_tile_items (the same items as the
building report), shifted into the spot's ENU frame, rendered from three fixed cameras:
  nw   oblique from azimuth 315 (north-west), elevation 35
  se   oblique from azimuth 135 (south-east), elevation 35
  top  from above, elevation 80
Output: <out>/<id>_<camera>.png, and with --obj <out>/<id>.obj (+ .mtl, one colour per material, Y-up:
x=east, y=up, z=-north). Triangles whose centroid is beyond radius+20 m from the spot are left out.
Roof photos stay on (the default); --no-photo sets BNE_ROOF_PHOTO=off before the generator is imported.
Default OUT: ~/bnge-tests/leonberg/preview.
"""
import argparse
import logging
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "server"))

CAMERAS = {"nw": (315.0, 35.0), "se": (135.0, 35.0), "top": (0.0, 80.0)}
BACKGROUND = 0.92
GROUND_GREY = 0.80
LIGHT = np.array([-.4, -.55, 1.]) / np.linalg.norm([-.4, -.55, 1.])
MARGIN = 20.0
FACADE_MEAN = (0.80, 0.78, 0.74)   # mean colour of the plaster/stucco facade textures (no stock measurement)


def camera_axes(azimuth_deg, elevation_deg):
    """(3,3) matrix with columns right, up, towards-viewer (ENU: x east, y north, z up); orthonormal."""
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    v = np.array([math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)])
    forward = -v
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return np.array([right, up, v]).T


def render(P, T, C, axes, size, scale, center):
    """Z-buffer render of triangles (vertex colours C, 0..255) with Lambert shading; a grey ground plane at the
    lowest height of P. scale: pixels per metre; center: the world point at the image centre."""
    width, height = size
    P = np.asarray(P, float).reshape(-1, 3)
    T = np.asarray(T, np.int64).reshape(-1, 3)
    C = np.asarray(C, float).reshape(-1, C.shape[-1])[:, :3] / 255.0 if len(C) else np.zeros((0, 3))
    center = np.asarray(center, float)
    gz = float(P[:, 2].min()) if len(P) else float(center[2])
    half = 1.5 * max(width, height) / scale
    gx, gy = center[0], center[1]
    ground = np.array([[gx - half, gy - half, gz], [gx + half, gy - half, gz],
                       [gx + half, gy + half, gz], [gx - half, gy + half, gz]])
    n = len(P)
    P = np.concatenate([P, ground])
    C = np.concatenate([C, np.full((4, 3), GROUND_GREY)])
    T = np.concatenate([T, [[n, n + 1, n + 2], [n, n + 2, n + 3]]])

    q = (P - center) @ axes
    screen = np.c_[width / 2 + q[:, 0] * scale, height / 2 - q[:, 1] * scale]
    view = axes[:, 2]
    image = np.full((height, width, 3), BACKGROUND)
    depth = np.full((height, width), -1e9)
    for tri in T:
        pp = P[tri]
        normal = np.cross(pp[1] - pp[0], pp[2] - pp[0])
        length = np.linalg.norm(normal)
        if length < 1e-9 or np.dot(normal, view) <= 0:
            continue
        normal /= length
        ss = screen[tri]
        if max(np.ptp(ss[:, 0]), np.ptp(ss[:, 1])) < 0.3:
            continue
        lo = np.maximum(np.floor(ss.min(axis=0)).astype(int), 0)
        hi = np.minimum(np.ceil(ss.max(axis=0)).astype(int), [width - 1, height - 1])
        if (hi < lo).any():
            continue
        yy, xx = np.mgrid[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
        a, bb, cc = ss
        den = (bb[1] - cc[1]) * (a[0] - cc[0]) + (cc[0] - bb[0]) * (a[1] - cc[1])
        if abs(den) < 1e-9:
            continue
        w0 = ((bb[1] - cc[1]) * (xx + .5 - cc[0]) + (cc[0] - bb[0]) * (yy + .5 - cc[1])) / den
        w1 = ((cc[1] - a[1]) * (xx + .5 - cc[0]) + (a[0] - cc[0]) * (yy + .5 - cc[1])) / den
        w2 = 1 - w0 - w1
        z = w0 * q[tri[0], 2] + w1 * q[tri[1], 2] + w2 * q[tri[2], 2]
        mask = (w0 >= -1e-7) & (w1 >= -1e-7) & (w2 >= -1e-7) & (z > depth[yy, xx] + 1e-7)
        if not mask.any():
            continue
        weights = np.stack([w0, w1, w2], axis=-1)
        color = weights @ C[tri] * (.55 + .45 * max(0., float(np.dot(normal, LIGHT))))
        image[yy[mask], xx[mask]] = color[mask]
        depth[yy[mask], xx[mask]] = z[mask]
    return Image.fromarray(np.clip(image * 255, 0, 255).astype(np.uint8))


def _mat_name(names, m):
    if m < 0 or m >= len(names):
        return "none"
    return f"m{m}_" + re.sub(r"\W", "_", str(names[m]))


def write_obj(path, P, T, M, C, names=None):
    """OBJ (Y-up: x east, y up, z -north) with a .mtl beside it: one material per M index, Kd = mean vertex colour.
    Faces are 1-based; triangles with M < 0 get the material 'none'."""
    path = Path(path)
    P = np.asarray(P, float).reshape(-1, 3)
    T = np.asarray(T, np.int64).reshape(-1, 3)
    M = np.asarray(M, np.int64).reshape(-1)
    C = np.asarray(C, float).reshape(len(P), -1)[:, :3] / 255.0
    if names is None:
        names = [f"mat{i}" for i in range(int(M.max()) + 1 if len(M) and M.max() >= 0 else 0)]
    mtl = path.with_suffix(".mtl")
    mats = sorted(set(M.tolist()))
    with open(mtl, "w", encoding="utf-8") as f:
        for m in mats:
            verts = np.unique(T[M == m])
            kd = C[verts].mean(0) if len(verts) else np.full(3, .5)
            f.write(f"newmtl {_mat_name(names, m)}\nKd {kd[0]:.4f} {kd[1]:.4f} {kd[2]:.4f}\n\n")
    lines = [f"mtllib {mtl.name}", f"o {path.stem}"]
    lines += [f"v {x:.4f} {y:.4f} {z:.4f}" for x, y, z in zip(P[:, 0], P[:, 2], -P[:, 1])]
    last = None
    for t in np.argsort(M, kind="stable"):
        if M[t] != last:
            lines.append(f"usemtl {_mat_name(names, int(M[t]))}")
            last = M[t]
        a, b, c = T[t] + 1
        lines.append(f"f {a} {b} {c}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def within(items, x0, y0, reach):
    """The (spec, gmin, gmax) items whose footprint centroid (ring mean + (x0, y0), ENU) is at most reach m from (0, 0).
    Whole buildings only: a building is kept or dropped as a unit, never cut."""
    out = []
    for item in items:
        c = np.asarray(item[0].ring, np.float64)[:, :2].mean(0) + (x0, y0)
        if math.hypot(c[0], c[1]) <= reach:
            out.append(item)
    return out


def _select(P, T, M, C, reach, UV=None):
    """Keep the triangles whose centroid is within reach m of (0,0); drop the vertices they do not use. build_spot
    passes reach=inf (no cut), so this only compacts the vertex list."""
    cen = P[T][:, :, :2].mean(1) if len(T) else np.zeros((0, 2))
    keep = np.hypot(cen[:, 0], cen[:, 1]) <= reach
    T, M = T[keep], M[keep]
    used, inv = np.unique(T, return_inverse=True)
    if UV is not None:
        return P[used], np.asarray(inv).reshape(-1, 3), M, C[used], np.asarray(UV, float).reshape(-1, 2)[used]
    return P[used], np.asarray(inv).reshape(-1, 3), M, C[used]


def _albedo(name):
    """Texture mean of a material: buildings.ROOF_TEX_MEAN, else the facade mean for facade/stucco/plaster names,
    else a neutral grey. The game multiplies the vertex tint by the texture, so the preview does too."""
    import buildings as B
    if name in B.ROOF_TEX_MEAN:
        return np.array(B.ROOF_TEX_MEAN[name], float)
    if any(s in str(name).lower() for s in ("fac", "stucco", "plaster")):
        return np.array(FACADE_MEAN, float)
    return np.array((0.75, 0.75, 0.75))


GLASS_RGB = (0.20, 0.25, 0.30)      # a window's glass in the preview (the sheet's window cells are textures: no colour)


def _glass_triangles(UV, T, M, names):
    """Boolean per triangle: its three UVs lie in the plaster sheet's window cells (buildings.WIN_CELLS x WIN_UV rows) of a
    facade material: a window (the laid-over windows of the old walls and the facade kit's)."""
    import buildings as B
    UV = np.asarray(UV, float).reshape(-1, 2)
    T = np.asarray(T, np.int64).reshape(-1, 3)
    v0, v1 = B.WIN_UV[2] - 1e-4, B.WIN_UV[3] + 1e-4
    inside = np.zeros(len(UV), bool)
    for u0, u1 in B.WIN_CELLS:
        inside |= (UV[:, 0] >= u0 - 1e-4) & (UV[:, 0] <= u1 + 1e-4)
    inside &= (UV[:, 1] >= v0) & (UV[:, 1] <= v1)
    fac = np.array([n == B.FACADE or str(n).startswith(B.FACADE + "|") or str(n) == "bne_bld_facade" for n in names] + [False])
    mat_ok = fac[np.clip(np.asarray(M, np.int64), -1, len(names) - 1)] if len(names) else np.zeros(len(T), bool)
    return inside[T].all(axis=1) & mat_ok


def _tinted(P, T, M, C, names, UV=None):
    """Each triangle gets its own three vertices, coloured Vertexcolour/255 x the texture mean of its material (capped at
    1). Triangles with no material (M < 0) keep their vertex colour. Returns (P, T, M, C) with C as 0..255 RGBA."""
    T = np.asarray(T, np.int64).reshape(-1, 3)
    M = np.asarray(M, np.int64).reshape(-1)
    P = np.asarray(P, float).reshape(-1, 3)
    C = np.asarray(C, float).reshape(len(P), -1)
    fac = np.ones((len(T), 3))
    for t, m in enumerate(M):
        if 0 <= m < len(names):
            fac[t] = _albedo(names[m])
    rgb = np.clip(C[T][:, :, :3] / 255.0 * fac[:, None, :], 0.0, 1.0) * 255.0
    if UV is not None:
        glass = _glass_triangles(UV, T, M, names)
        rgb[glass] = np.array(GLASS_RGB) * 255.0 * np.clip(C[T][glass][:, :, :3].mean(axis=(1, 2)) / 255.0, 0.5, 1.0)[:, None, None]
    C2 = np.full((3 * len(T), 4), 255.0)
    C2[:, :3] = rgb.reshape(-1, 3)
    return P[T].reshape(-1, 3), np.arange(3 * len(T)).reshape(-1, 3), M, C2


def _merge(parts):
    """Join (P, T, M, C, names) meshes into one, materials matched by name."""
    Ps, Ts, Ms, Cs, index = [], [], [], [], {}
    off = 0
    for P, T, M, C, names in parts:
        gm = np.array([index.setdefault(str(nm), len(index)) for nm in names], np.int64)
        Mg = np.where(M >= 0, gm[np.clip(M, 0, max(len(gm) - 1, 0))] if len(gm) else -1, -1)
        Ps.append(P)
        Ts.append(T + off)
        Ms.append(Mg)
        Cs.append(C)
        off += len(P)
    if not Ps:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros(0, np.int64), np.zeros((0, 4), np.uint8), []
    return (np.concatenate(Ps), np.concatenate(Ts), np.concatenate(Ms), np.concatenate(Cs), list(index))


def build_spot(st, spot, radius):
    """(P, T, M, C, names, n_specs) of the spot's buildings in its ENU frame. Buildings whose footprint centroid is
    within radius + MARGIN m are kept whole (no triangle cut at the edge)."""
    import buildings as B
    import geodesy
    import roads
    import leonberg_buildings_report as R

    sid, name, lat, lon = spot
    frame = geodesy.EnuFrame(lat, lon)
    reach = radius + MARGIN
    k_lo, k_hi = math.floor(-reach / roads.TILE), math.floor(reach / roads.TILE)
    parts, n_specs = [], 0
    for tx in range(k_lo, k_hi + 1):
        for ty in range(k_lo, k_hi + 1):
            try:
                x0, y0 = roads.tile_bbox(tx, ty)[:2]
                items, _custom = B.road_tile_items(frame, tx, ty, st.md, R._ground(frame, st),
                                                   urban_fn=st.surface.urban if st.surface is not None else None,
                                                   widths=st.widths)
                items = within(items, x0, y0, reach)
                n_specs += len(items)
                if not items:
                    continue
                for mesh in B.build_parts(items):
                    P = np.asarray(mesh[0], float).copy()
                    P[:, 0] += x0
                    P[:, 1] += y0
                    T, M, mats, C = mesh[2], mesh[3], mesh[4], mesh[5]
                    Ps, Ts, Ms, Cs, UVs = _select(P, np.asarray(T, np.int64), np.asarray(M, np.int64),
                                                  np.asarray(C), float("inf"), UV=mesh[1])
                    Ps, Ts, Ms, Cs = _tinted(Ps, Ts, Ms, Cs, list(mats), UV=UVs)
                    parts.append((Ps, Ts, Ms, Cs, list(mats)))
            except Exception:
                logging.exception("tile %s,%s of %s failed", tx, ty, sid)
                print(f"{sid} tile {tx},{ty} FAILED, skipped", flush=True)
    P, T, M, C, names = _merge(parts)
    return P, T, M, C, names, n_specs


def _label(img, text):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=26)
    except TypeError:
        font = ImageFont.load_default()
    draw.rectangle([8, 8, 14 + draw.textlength(text, font=font), 14 + 30], fill=(255, 255, 255))
    draw.text((11, 11), text, fill=(20, 20, 20), font=font)
    return img


def parse_size(text):
    w, h = text.lower().split("x")
    return int(w), int(h)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spots", default=None, help="comma list of spot ids (default all)")
    ap.add_argument("--radius", type=float, default=150.0)
    ap.add_argument("--size", default="1600x1000", help="image size WxH")
    ap.add_argument("--out", default=os.path.expanduser("~/bnge-tests/leonberg/preview"))
    ap.add_argument("--obj", action="store_true", help="also write <id>.obj (+ .mtl)")
    ap.add_argument("--label", default=None, help="text in the top left corner, e.g. 'vorher' or 'nachher'")
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/beamng-earth"))
    ap.add_argument("--no-photo", action="store_true", help="no roof photos (BNE_ROOF_PHOTO=off)")
    ap.add_argument("--close", action="append", default=[], metavar="DX,DY,R",
                    help="also close-ups (cameras nw, se) of the radius R m round the ENU offset DX,DY m from the spot, "
                         "<id>_close<k>_<cam>.png (repeatable)")
    ap.add_argument("--elev", type=float, default=None, help="elevation of the close-ups' cameras (default 22)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    if args.no_photo:
        os.environ["BNE_ROOF_PHOTO"] = "off"
    else:
        os.environ.setdefault("BNE_ROOF_PHOTO", "on")
    import leonberg_buildings_report as R   # (after the env setting: it defaults BNE_ROOF_PHOTO to off)
    import blockworker as bw

    spots = R.SPOTS
    if args.spots:
        want = [s.strip() for s in args.spots.split(",") if s.strip()]
        known = {s[0] for s in R.SPOTS}
        bad = [s for s in want if s not in known]
        if bad:
            ap.error(f"unknown spot(s): {', '.join(bad)}")
        spots = [s for s in R.SPOTS if s[0] in want]
    size = parse_size(args.size)
    os.makedirs(args.out, exist_ok=True)
    st = bw._State(args.cache, False, True, True, True, True)
    bw._s = st
    scale = min(size) / (2.0 * args.radius)
    for spot in spots:
        sid, name = spot[0], spot[1]
        t0 = time.time()
        try:
            P, T, M, C, names, n_specs = build_spot(st, spot, args.radius)
        except Exception as ex:
            logging.exception("%s failed", sid)
            print(f"{sid} FAILED {type(ex).__name__}: {ex}", flush=True)
            continue
        center = np.array([0., 0., float(np.median(P[T][:, :, 2].min(1))) if len(T) else 0.])
        for cam, (az, el) in CAMERAS.items():
            img = render(P, T, C, camera_axes(az, el), size, scale, center)
            if args.label:
                img = _label(img, args.label)
            img.save(os.path.join(args.out, f"{sid}_{cam}.png"))
        for k, spec_ in enumerate(args.close):
            dx, dy, rr = (float(v) for v in spec_.split(","))
            near = np.hypot(P[:, 0] - dx, P[:, 1] - dy) <= rr * 1.6
            gz = float(np.median(P[near][:, 2].min())) if near.any() else float(center[2])
            if near.any():
                gz = float(np.percentile(P[near][:, 2], 3))
            for cam, (az, el) in (("nw", (315.0, args.elev or 22.0)), ("se", (135.0, args.elev or 22.0))):
                img = render(P, T, C, camera_axes(az, el), size, min(size) / (2.0 * rr), np.array([dx, dy, gz + 4.0]))
                if args.label:
                    img = _label(img, args.label)
                img.save(os.path.join(args.out, f"{sid}_close{k}_{cam}.png"))
        if args.obj:
            write_obj(os.path.join(args.out, f"{sid}.obj"), P, T, M, C, names)
        print(f"{sid} {name}: {n_specs} Gebäude-Specs, {len(T)} Dreiecke, {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
