"""Live conditions for the GPS: road closures to route around and traffic delays on the way (the user, 2026-11-26:
"check that the navigation can do traffic and closures"). The route itself comes from OSRM (routing.py), which knows
only OpenStreetMap's fixed data: no jam, no roadworks. This module is the interface a live source (traffic and
closure feeds, later together with the realistic traffic) fills, and the pure logic that applies it to planned routes.

A closure:   {"id", "name"?, "lat", "lon", "radius_m"?}  or  {"id", "name"?, "points": [[lat, lon], ...], "radius_m"?}
A delay:     {"id", "extra_s", ...the same place fields}   (a jam: this many seconds more if the route passes it)
radius_m is the half width of the blocked road or jam (default HALF_M). Everything here is pure except Store.
"""
import math

import dmath

HALF_M = 25.0
MAX_ITEMS = 500


def _xy(points, lat_ref):
    k = dmath.cos(math.radians(lat_ref))
    return [(lo * 111320.0 * k, la * 111320.0) for la, lo in points]


def _seg_dist(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 <= 0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2))
    return math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy)


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _segs_dist(a, b, c, d):
    """The least distance between segments ab and cd (0 when they cross)."""
    if _cross(a, b, c) * _cross(a, b, d) < 0 and _cross(c, d, a) * _cross(c, d, b) < 0:
        return 0.0
    return min(_seg_dist(a, c, d), _seg_dist(b, c, d), _seg_dist(c, a, b), _seg_dist(d, a, b))


def _place(item):
    """(points [(lat, lon)...], radius_m) of a closure/delay, or None when it has no valid place."""
    try:
        raw = item.get("points") or [[item["lat"], item["lon"]]]
        pts = [(float(p[0]), float(p[1])) for p in raw]
        r = float(item.get("radius_m", HALF_M))
    except (AttributeError, KeyError, TypeError, ValueError, IndexError):
        return None
    if not pts or not all(-90 <= a <= 90 and -180 <= b <= 180 for a, b in pts) or not 0 < r <= 2000:
        return None
    return pts, r


def touches(route, item):
    """Whether the route ([(lat, lon)...]) passes the closure or delay item (within its radius)."""
    place = _place(item)
    if place is None or not route:
        return False
    pts, r = place
    lat_ref = route[0][0]
    rt, it = _xy(route, lat_ref), _xy(pts, lat_ref)
    if len(rt) == 1:
        rt = rt * 2
    if len(it) == 1:
        it = it * 2
    return any(_segs_dist(a, b, c, d) <= r for a, b in zip(rt, rt[1:]) for c, d in zip(it, it[1:]))


def check(route, closures=(), delays=()):
    """{"blocked": [closure ids the route passes], "delay": extra seconds from the delays it passes}. Pure."""
    blocked = [str(c.get("id", i)) for i, c in enumerate(closures) if touches(route, c)]
    delay = sum(max(0.0, float(d.get("extra_s") or 0)) for d in delays if touches(route, d))
    return {"blocked": blocked, "delay": delay}


def choose(candidates, closures=(), delays=()):
    """The best of the planned routes ({"points": [[lat, lon]...], "dur"}; OSRM's first answer and its alternatives):
    the ones passing no closure by time including delays; if every one is blocked, the one with the fewest closures.
    Returns that route plus "delay" (s, already in "dur"), "blocked" ([ids]) and "avoided" (closure ids the first,
    fastest candidate passed that this one does not). Pure."""
    best = None
    first = None
    for i, c in enumerate(candidates):
        r = check([(p[0], p[1]) for p in c["points"]], closures, delays)
        key = (len(r["blocked"]), float(c.get("dur") or 0) + r["delay"], i)
        if i == 0:
            first = r["blocked"]
        if best is None or key < best[0]:
            best = (key, c, r)
    if best is None:
        raise ValueError("no route")
    _, c, r = best
    out = dict(c, dur=float(c.get("dur") or 0) + r["delay"])
    if r["delay"] or closures or delays:
        out["delay"] = r["delay"]
    out["blocked"] = r["blocked"]
    out["avoided"] = [b for b in (first or []) if b not in r["blocked"]]
    return out


def detours(route, blocked_items, offsets_m=(300.0, 600.0)):
    """Detour waypoints [(lat, lon)] for a route that still passes closures: to either side of the first blocked
    item, at each offset, perpendicular to the route there; nearest first. OSRM snaps them to roads. Pure."""
    if not route or not blocked_items:
        return []
    place = _place(blocked_items[0])
    if place is None:
        return []
    clat, clon = place[0][0]
    lat_ref = route[0][0]
    rt = _xy(route, lat_ref)
    (cx, cy), = _xy([(clat, clon)], lat_ref)
    i = min(range(max(1, len(rt) - 1)), key=lambda k: _seg_dist((cx, cy), rt[k], rt[min(k + 1, len(rt) - 1)]))
    a, b = rt[i], rt[min(i + 1, len(rt) - 1)]
    n = math.hypot(b[0] - a[0], b[1] - a[1]) or 1.0
    nx, ny = -(b[1] - a[1]) / n, (b[0] - a[0]) / n
    k = dmath.cos(math.radians(lat_ref))
    return [(clat + s * off * ny / 111320.0, clon + s * off * nx / (111320.0 * k))
            for off in offsets_m for s in (1.0, -1.0)]


def clean(items):
    """The valid items of a feed message's list (dicts with a place), at most MAX_ITEMS. Pure."""
    return [i for i in (items if isinstance(items, list) else ())[:MAX_ITEMS] if isinstance(i, dict) and _place(i)]


class Store:
    """The conditions now in force. A live source calls set(); version goes up on every change, so a running GPS
    can tell that its route must be checked again."""

    def __init__(self):
        self.closures, self.delays, self.version = [], [], 0

    def set(self, closures=None, delays=None):
        c = clean(closures) if closures is not None else self.closures
        d = clean(delays) if delays is not None else self.delays
        if (c, d) != (self.closures, self.delays):
            self.closures, self.delays, self.version = c, d, self.version + 1
        return self.version

    def active(self):
        return bool(self.closures or self.delays)
