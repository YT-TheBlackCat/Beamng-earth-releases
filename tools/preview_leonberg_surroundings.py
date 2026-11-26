#!/usr/bin/env python3
"""Leonberg test spot as a top-down PNG: sidewalks, street lamps, street trees, fences and noise walls.

    .venv/bin/python tools/preview_leonberg_surroundings.py --spot U1 [--radius 150] [--cache DIR] [--out DIR] [--eu on|off]

Offline: the generators run on the cache (~/.cache/beamng-earth, or --cache) and nothing is published. For PRs, a
before/after pair: run once with --eu off and once with --eu on (or on two checkouts) and compare the PNGs.
Writes <out>/<spot>-<eu|base>.png (default out ~/bnge-tests/leonberg/preview).

What is drawn (frame around the spot, the spot at (0, 0), +-radius m): the walks of the road tiles (lines, width as
drawn, colour by surface), the props of the tiles: street lamps (yellow dot), street trees (green circle), chain-link
(thin black line in its yaw), noise walls (thick red line), other props (grey dot). Buildings, crown vegetation and
roads are not drawn.
Fences, garden walls and hedges of Germany (fences.german_rows: groups fr_fence_<style> and fr_hedge) are drawn as
coloured lines by model (hedges as dark green circles); buildings (grey fill) and roads (dark lines, as wide as they are)
give the context to check that nothing stands on them. --boundaries on|off sets BNE_DE_BOUNDARIES (the switch of
those rows) and adds -grenzen / -ohne-grenzen to the file name, for a before / after pair.
"""
import argparse
import importlib.util
import logging
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "server"))

DEFAULT_OUT = os.path.expanduser("~/bnge-tests/leonberg/preview")
DEFAULT_CACHE = os.path.expanduser("~/.cache/beamng-earth")
DEFAULT_WALK_W = 2.0                    # m, when a walk has no width

YELLOW = "#f2c200"
GREEN = "#2e8b3a"
RED = "#d62728"
GREY = "#8c8c8c"
BLACK = "#111111"
HEDGE_GREEN = "#0b3d0b"
BOUNDARY_COLOURS = {"limestone": "#e6007e", "dark_stone": "#000000", "brick": "#c0392b", "iron": "#0033cc",
                    "mesh": "#00b7c3", "wood": "#ff7f0e", "slats": "#7f3fbf", "rails": "#2ca02c"}
FR_HEIGHTS = {"limestone": .95, "dark_stone": .94, "brick": 1.05, "iron": 1.23, "mesh": 1.14, "wood": 1.16,
              "slats": 1.12, "rails": 1.19}     # fences.FR_HEIGHTS (the panel's length is height / this * 2.2 m)
PANEL_M = 2.2                                   # fences.PANEL_M

SURFACE_COLOURS = {
    "asphalt": "#4a4a4a",
    "light_asphalt": "#8a8a8a",
    "concrete": "#c4c4c4",
    "pavers": "#d9c7a3",
    "slabs": "#a9a9a9",
    "stone": "#8b6b4a",
    "us_slab": "#cfcfcf",
}
SURFACE_DEFAULT = "#b0b0b0"
CHAIN_LEN = 2.04                        # fences.CHAIN_LEN, m
WALL_LEN = 4.0                          # fences.WALL_LEN, m


def _report_module():
    """tools/leonberg_surroundings_report.py, loaded by path (SPOTS, _walks_in_frame); its top level imports numpy only."""
    path = os.path.join(HERE, "leonberg_surroundings_report.py")
    spec = importlib.util.spec_from_file_location("leonberg_surroundings_report", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def symbol_for(group):
    """(kind, colour) of a prop group: kind is 'lamp', 'tree', 'chain', 'wall' or 'other'."""
    g = str(group)
    if g in ("streetlight", "pole"):
        return "lamp", YELLOW
    if g in ("deciduous", "tree"):
        return "tree", GREEN
    if g == "chainlink":
        return "chain", BLACK
    if g == "noisewall":
        return "wall", RED
    if g in ("fr_hedge", "fr_fence_hedge"):
        return "hedge", HEDGE_GREEN
    if g.startswith("fr_fence_") and g[len("fr_fence_"):] in BOUNDARY_COLOURS:
        return "boundary", BOUNDARY_COLOURS[g[len("fr_fence_"):]]
    return "other", GREY


def surface_colour(surface):
    """Colour of a walk's surface (None is the US slab)."""
    if surface is None:
        return SURFACE_COLOURS["us_slab"]
    return SURFACE_COLOURS.get(str(surface), SURFACE_DEFAULT)


def props_points(rows, radius, with_height=False):
    """[(group, x, y, yaw)] of the prop rows [group, x, y, z, yaw, scale, height, rank] within radius (m) of (0, 0);
    with_height: (group, x, y, yaw, height) (the row's height, 1.0 when it has none)."""
    out = []
    for r in rows or []:
        if len(r) < 3:
            continue
        try:
            x, y = float(r[1]), float(r[2])
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(x) and math.isfinite(y)) or math.hypot(x, y) > radius:
            continue
        try:
            yaw = float(r[4]) if len(r) > 4 else 0.0
        except (TypeError, ValueError):
            yaw = 0.0
        yaw = yaw if math.isfinite(yaw) else 0.0
        if with_height:
            try:
                height = float(r[6]) if len(r) > 6 else 1.0
            except (TypeError, ValueError):
                height = 1.0
            out.append((str(r[0]), x, y, yaw, height if math.isfinite(height) else 1.0))
        else:
            out.append((str(r[0]), x, y, yaw))
    return out


def walks_lines(walks):
    """[(points (k, 2) list of (x, y), width m, surface)] of the walk records with 'pts'; discs and corners are skipped."""
    out = []
    for w in walks or []:
        if not isinstance(w, dict) or "disc" in w or "corner" in w:
            continue
        pts = w.get("pts")
        if pts is None or len(pts) < 2:
            continue
        xy = [(float(p[0]), float(p[1])) for p in pts]
        width = w.get("w")
        try:
            width = float(width) if width is not None else DEFAULT_WALK_W
        except (TypeError, ValueError):
            width = DEFAULT_WALK_W
        out.append((xy, width, w.get("surface")))
    return out


def _collect(R, spot, radius, cache, eu_on):
    """(walks, props, rings, roads) of the road tiles around the spot: walks in the frame, as the report tool
    collects them; the buildings' outer rings and the roads' [(centreline (k, 2), width m)] for the picture."""
    import blockworker as bw
    import furniture
    import geodesy
    import roads

    sid, name, lat, lon = spot
    st = bw._State(cache, False, True, True, True, True)
    bw._s = st
    frame = geodesy.EnuFrame(lat, lon)
    k_lo, k_hi = math.floor(-radius / roads.TILE), math.floor(radius / roads.TILE)
    lamp = furniture.lamp_region("DE")
    import buildings as buildingsmod
    walks, props, rings, lines = [], [], [], []
    for tx in range(k_lo, k_hi + 1):
        for ty in range(k_lo, k_hi + 1):
            t = (tx, ty)
            try:
                P = bw._tile_parts(lambda *a, **k: None, frame, t, 15, None, True, False, False, False, 0.0, lamp,
                                   "DE", None, ("leonberg", lat, lon), (lat, lon), cache_dir=cache)
            except Exception as ex:         # (one tile failing: the others go on)
                logging.exception("%s tile %s failed", sid, t)
                print(f"{sid} tile {t} FAILED {type(ex).__name__}: {ex}", flush=True)
                continue
            tw, how = R._walks_in_frame(P)
            walks += tw
            props += list(P.props or [])
            lines += [(c["n"], float(c["w"])) for c in P.chunks if isinstance(c, dict) and "n" in c and "w" in c
                      and "poly" not in c and "frontage" not in c and "layer" not in c]
            try:
                rings += buildingsmod.near_rings(frame, tx, ty, st.md, margin=2.0)
            except Exception as ex:         # (the picture without buildings)
                logging.warning("%s tile %s: buildings: %s", sid, t, ex)
            logging.info("%s tile %s: %d walks (%s), %d props", sid, t, len(tw), how, len(P.props or []))
    return walks, props, rings, lines


def render(spot, walks, props, radius, eu_on, out_path, rings=(), lines=()):
    """Draw the frame to a 1200 x 1200 px PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    sid, name = spot[0], spot[1]
    mode = "eu" if eu_on else "base"
    fig = plt.figure(figsize=(12, 12), dpi=100, facecolor="white")
    ax = fig.add_axes([0.05, 0.15, 0.90, 0.77])
    ax.set_facecolor("white")
    ax.set_aspect("equal")
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#999999")

    # 1 m in points on this axes (the axes is 0.90 of the 12 in figure wide)
    pt_per_m = 0.90 * 12 * 72.0 / (2 * radius)
    for ring in rings:
        ax.fill([p[0] for p in ring], [p[1] for p in ring], color="#dddddd", ec="#aaaaaa", lw=0.4, zorder=0)
    for pts, width in lines:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color="#555555", lw=max(0.5, width * pt_per_m), alpha=0.35,
                solid_capstyle="butt", zorder=0.5)
    for xy, width, surface in walks_lines(walks):
        xs, ys = zip(*xy)
        ax.plot(xs, ys, color=surface_colour(surface), lw=max(0.5, width * pt_per_m), solid_capstyle="butt", alpha=0.55, zorder=1)

    for group, x, y, yaw, height in props_points(props, radius, with_height=True):
        kind, colour = symbol_for(group)
        if kind == "lamp":
            ax.plot(x, y, "o", color=colour, ms=4, mec="#6b5600", mew=0.5, zorder=4)
        elif kind == "tree":
            ax.add_patch(plt.Circle((x, y), 2.0, color=colour, alpha=0.75, zorder=3))
        elif kind == "chain":
            a = math.radians(yaw)
            dx, dy = math.cos(a) * CHAIN_LEN / 2, math.sin(a) * CHAIN_LEN / 2
            ax.plot([x - dx, x + dx], [y - dy, y + dy], color=colour, lw=0.8, zorder=5)
        elif kind == "wall":
            a = math.radians(yaw)
            dx, dy = math.cos(a) * WALL_LEN / 2, math.sin(a) * WALL_LEN / 2
            ax.plot([x - dx, x + dx], [y - dy, y + dy], color=colour, lw=3.5, solid_capstyle="butt", zorder=5)
        elif kind == "hedge":
            ax.add_patch(plt.Circle((x, y), 0.8, color=colour, alpha=0.8, zorder=3))
        elif kind == "boundary":
            a = math.radians(yaw)
            half = height / FR_HEIGHTS[group[len("fr_fence_"):]] * PANEL_M / 2
            dx, dy = math.cos(a) * half, math.sin(a) * half
            ax.plot([x - dx, x + dx], [y - dy, y + dy], color=colour, lw=3.0, solid_capstyle="butt", zorder=6)
        else:
            ax.plot(x, y, "o", color=colour, ms=2, zorder=4)

    ax.plot(0, 0, "+", color="black", ms=10, mew=1.2, zorder=6)

    # north arrow (top-right of the axes)
    nx, ny = radius * 0.86, radius * 0.80
    ax.annotate("", xy=(nx, ny + radius * 0.10), xytext=(nx, ny - radius * 0.02),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5))
    ax.text(nx, ny + radius * 0.14, "N", ha="center", va="bottom", fontsize=12, fontweight="bold")

    # scale bar: 50 m
    bx, by = -radius * 0.80, -radius * 0.88
    ax.plot([bx, bx + 50], [by, by], color="black", lw=3, solid_capstyle="butt")
    ax.text(bx + 25, by - radius * 0.04, "50 m", ha="center", va="top", fontsize=10)

    fig.text(0.05, 0.955, f"{sid} {name} ({mode})", fontsize=16, fontweight="bold", va="bottom")
    fig.text(0.05, 0.935, f"Radius {radius:g} m, Mittelpunkt = Testpunkt. EU-Straßendetails: {'an' if eu_on else 'aus'}",
             fontsize=9, color="#444444", va="bottom")

    legend = [
        Line2D([], [], marker="o", ls="", color=YELLOW, mec="#6b5600", label="Laterne"),
        Line2D([], [], marker="o", ls="", ms=9, color=GREEN, alpha=0.75, label="Straßenbaum"),
        Line2D([], [], color=BLACK, lw=1, label="Maschendrahtzaun"),
        Line2D([], [], color=RED, lw=3.5, label="Lärmschutz"),
        Line2D([], [], color=BOUNDARY_COLOURS["wood"], lw=2.2, label="Zaun Holz"),
        Line2D([], [], color=BOUNDARY_COLOURS["mesh"], lw=2.2, label="Zaun Gitter"),
        Line2D([], [], color=BOUNDARY_COLOURS["iron"], lw=2.2, label="Zaun Eisen"),
        Line2D([], [], color=BOUNDARY_COLOURS["limestone"], lw=2.2, label="Mauer"),
        Line2D([], [], color=BOUNDARY_COLOURS["rails"], lw=2.2, label="Zaun Metall"),
        Line2D([], [], color=BOUNDARY_COLOURS["slats"], lw=2.2, label="Zaun Latten"),
        Line2D([], [], marker="o", ls="", ms=8, color=HEDGE_GREEN, alpha=0.8, label="Hecke"),
        Line2D([], [], color=surface_colour("asphalt"), lw=4, label="Gehweg Asphalt"),
        Line2D([], [], color=surface_colour("concrete"), lw=4, label="Gehweg Beton"),
        Line2D([], [], color=surface_colour("pavers"), lw=4, label="Gehweg Pflaster"),
    ]
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(0.0, -0.025), ncol=4, fontsize=7, frameon=False)
    fig.text(0.05, 0.012, "Gehwege, Laternen, Bäume (Straßen-Props), Zäune, Mauern, Hecken, Lärmschutz; "
             "Gebäude und Straßen grau als Umgebung; ohne Kronenvegetation.", fontsize=8, color="#444444")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=100, facecolor="white")
    plt.close(fig)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spot", required=True, help="spot id from the report tool's SPOTS, e.g. U1")
    ap.add_argument("--radius", type=float, default=150.0)
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--eu", choices=["on", "off"], default="off", help="BNE_EU_ROAD_DETAILS for the generators")
    ap.add_argument("--boundaries", choices=["on", "off"], default=None,
                    help="BNE_DE_BOUNDARIES (Germany's fences, walls, hedges); adds -grenzen / -ohne-grenzen to the name")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    # the env var is read by the generators, so it is set before they are imported
    os.environ["BNE_EU_ROAD_DETAILS"] = "on" if args.eu == "on" else "off"
    if args.boundaries is not None:
        os.environ["BNE_DE_BOUNDARIES"] = args.boundaries
    R = _report_module()
    spots = {s[0]: s for s in R.SPOTS}
    if args.spot not in spots:
        ap.error(f"unknown spot {args.spot}; known: {', '.join(spots)}")
    spot = spots[args.spot]
    eu_on = args.eu == "on"

    walks, props, rings, lines = _collect(R, spot, args.radius, os.path.expanduser(args.cache), eu_on)
    suffix = {None: "", "on": "-grenzen", "off": "-ohne-grenzen"}[args.boundaries]
    out_path = os.path.join(os.path.expanduser(args.out), f"{spot[0]}-{'eu' if eu_on else 'base'}{suffix}.png")
    render(spot, walks, props, args.radius, eu_on, out_path, rings, lines)
    print(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
