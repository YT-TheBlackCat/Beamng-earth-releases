"""Road profiles round the Leonberg places (drivability.replay): do the roads join without an edge, and does a motorway
get smoothed while the town streets follow the ground? Offline, on the replays (fixtures/drive_leonberg_*.npz).

    python server/tests/roadprofile.py [NAME ...]         # per site: the counts of each check, the 5 worst of each
    python server/tests/roadprofile.py --json [NAME ...]  # one JSON line per site (measure()'s result)

measure(name) reports, for the road parts within RADIUS_M of the place (samples beside a structure, within 20 m of a
bridge deck or tunnel floor, are left out):
  nodes   where two or more road parts meet (ends within 1 m): the excess of the graded step over the raw ground's
          (2nd differences, 1 m samples, max(0, dg - dr)) on the way through the node, over every pair of arms (10 m
          each); step_raw is the raw ground's own step there; grade_jump: the difference of the arms' grades
  edge    an artificial edge across a road: the graded ground steps where the raw ground does not (cross-sections
          every 5 m, offsets over the road and 2 m beyond it)
  follow  city and town roads: |graded - raw| along the centre line (p95 and max)
  radius  motorways and trunks: the smallest vertical radius (chord 2 * 25 m) of the graded and the raw ground, and
          the steepest grade over 25 m; each piece of the centre line measured on its own
checks(result) lists the lines of each limit (LIMITS) that a site breaks; an empty list is ok.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
import drivability as dv  # noqa: E402
import dnp  # noqa: E402
import mapgeom  # noqa: E402

RADIUS_M = dv.RADIUS_M
SITES = ("leonberg_dreieck", "leonberg_altstadt", "leonberg_a8_g3", "leonberg_a81_link")
CHECKS = ("nodes", "edges", "follow", "motorway")
MOTORWAY = {"motorway", "trunk"}
CITY = {"residential", "living_street", "unclassified", "tertiary", "service"}
TOWN = {"primary", "secondary"}
LIMITS = {
    "node_step_m": 0.10,                                        # excess of the graded step over the raw one, through a node
    "edge_m": 0.12,                                             # artificial edge: road ground -> raw ground
    "follow_p95_m": {"city": 0.30, "town": 0.50},               # |graded - raw| along the centre line
    "follow_max_m": {"city": 0.80, "town": 1.20},
    "radius_min_m": {"motorway": 3000.0, "trunk": 2000.0},      # smallest vertical radius, chord 2 * 25 m
    "grade_max": {"motorway": 0.06, "trunk": 0.08},             # over 25 m
}
NEAR_STRUCT_M = 20.0
NODE_TOL_M = 1.0
ARM_M = 10              # m of each arm of a node
GRADE_ARM_M = 5         # m: the grade of an arm, for grade_jump
EDGE_EVERY_M = 5        # m between the cross-sections of a road part
EDGE_OFFSET_M = 0.5     # m between the samples of a cross-section
EDGE_STEP_M = 0.12      # m per step: a graded step this big ...
EDGE_EXCESS_M = 0.10    # ... that the raw ground does not have (|dg - dr|) is an artificial edge
CHORD_M = 25            # m: the chord of the vertical radius and the run of the grade


def node_step(z):
    """max |z[i-1] - 2 z[i] + z[i+1]| over 1 m samples: a ledge or a kink; 0 below three samples."""
    z = np.asarray(z, np.float64)
    if len(z) < 3:
        return 0.0
    return float(np.abs(z[:-2] - 2 * z[1:-1] + z[2:]).max())


def node_excess(graded, raw):
    """max over the middle samples of max(0, |dg| - |dr|) (the 2nd differences of the graded and the raw ground, as in
    _edge_excess): the step the road makes beyond the raw ground's; a NaN raw sample counts as none, as there."""
    g = np.asarray(graded, np.float64)
    r = np.asarray(raw, np.float64)
    if len(g) < 3:
        return 0.0
    ex = np.abs(g[:-2] - 2 * g[1:-1] + g[2:]) - np.abs(r[:-2] - 2 * r[1:-1] + r[2:])
    return float(np.where(ex > 0, ex, 0.0).max())


def vertical_radius(z, chord=CHORD_M):
    """(smallest radius m, index of its middle sample) of a 1 m profile z: kappa = |z(s-L) - 2 z(s) + z(s+L)| / L^2,
    R = 1 / kappa; (inf, None) when the profile is too short or flat."""
    z = np.asarray(z, np.float64)
    n = len(z) - 2 * chord
    if n <= 0:
        return float("inf"), None
    kappa = np.abs(z[:n] - 2 * z[chord:chord + n] + z[2 * chord:2 * chord + n]) / chord ** 2
    i = int(np.argmax(kappa))
    if kappa[i] == 0:
        return float("inf"), None
    return float(1.0 / kappa[i]), i + chord


def _grades(z, run=CHORD_M):
    z = np.asarray(z, np.float64)
    if len(z) <= run:
        return np.zeros(0)
    return np.abs(z[run:] - z[:-run]) / run


def grade_max(z, run=CHORD_M):
    """max |z(s + run) - z(s)| / run over a 1 m profile; 0 when shorter than run + 1 samples."""
    g = _grades(z, run)
    return float(g.max()) if g.size else 0.0


def follow(graded, raw):
    """(p95, max) of |graded - raw|."""
    d = np.abs(np.asarray(graded, np.float64) - np.asarray(raw, np.float64))
    return (float(np.percentile(d, 95)), float(d.max())) if d.size else (0.0, 0.0)


def _edge_excess(graded, raw):
    """Per middle sample of a cross-section: the graded second difference where it is a step over EDGE_STEP_M and the
    raw ground does not have it (the difference over EDGE_EXCESS_M); 0 elsewhere."""
    g = np.asarray(graded, np.float64)
    r = np.asarray(raw, np.float64)
    if len(g) < 3:
        return np.zeros(0)
    dg = np.abs(g[:-2] - 2 * g[1:-1] + g[2:])
    dr = np.abs(r[:-2] - 2 * r[1:-1] + r[2:])
    return np.where((dg > EDGE_STEP_M) & (dg - dr > EDGE_EXCESS_M), dg, 0.0)


def edge_profile(graded, raw, step=0.5):
    """max over a cross-section's samples of _edge_excess. step: the m between its samples (the test is per sample, so
    it does not enter the value)."""
    ex = _edge_excess(graded, raw)
    return float(ex.max()) if ex.size else 0.0


def _xy(p):
    return [round(float(p[0]), 1), round(float(p[1]), 1)]


def _raw(r, xy):
    """The raw (ungraded) ground height at the m xy (n, 2)."""
    la, lo, _ = r.site.frame.enu_to_geodetic(xy[:, 0], xy[:, 1], 0.0)
    return np.asarray(r.site.sample(la, lo), np.float64)


def _pieces(mask, min_len):
    """Index arrays of the runs of True in mask, of at least min_len samples."""
    out, start = [], None
    for i, v in enumerate(list(mask) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                out.append(np.arange(start, i))
            start = None
    return out


def _ok_fn(r):
    """ok(xy) -> bool per point: inside RADIUS_M and more than NEAR_STRUCT_M from every bridge deck or tunnel floor."""
    pts = [np.asarray(s.xy, np.float64) for s in r.structures]
    tree = cKDTree(np.concatenate(pts)) if pts else None

    def ok(xy):
        xy = np.asarray(xy, np.float64).reshape(-1, 2)
        good = dv._inside(xy)
        if tree is not None:
            good &= tree.query(xy)[0] > NEAR_STRUCT_M
        return good
    return ok


def _parts(r):
    """Every ground road part near the place: key, fid, cls, width, centre (1 m samples), raw (the polyline), normal."""
    out, seen = [], set()
    for feats in r.cells:
        for i, spec in enumerate(r.roads.road_specs(feats)):
            if spec is None:
                continue
            fid = str(feats.ids[i])
            for j, ring in enumerate(r.roads.ground_parts(feats, i)):
                if (fid, j) in seen:
                    continue
                seen.add((fid, j))
                e, n = mapgeom.enu_xy(feats, r.site.frame, ring)
                raw = np.c_[e, n]
                centre = dv._resample(raw)
                if centre is None or dnp.hypot(centre[:, 0], centre[:, 1]).min() > RADIUS_M + spec.width:
                    continue
                d = np.gradient(centre, axis=0)
                normal = np.c_[-d[:, 1], d[:, 0]] / np.maximum(dnp.hypot(d[:, 0], d[:, 1]), 1e-9)[:, None]
                out.append(SimpleNamespace(key=f"ground:{fid}" if j == 0 else f"ground:{fid}#{j}", fid=fid,
                                           cls=spec.cls, width=float(spec.width), centre=centre, raw=raw,
                                           normal=normal))
    return out


def _arm(node, out):
    """The first ARM_M m of a part's centre line out from the node (1 m samples, the first at the node), None when the
    part is shorter than 3 m."""
    pts = out if dnp.hypot(*(out[0] - node)) < 1e-6 else np.r_[node[None], out]
    arm = dv._resample(pts)
    return None if arm is None else arm[:ARM_M + 1]


def _grade_out(r, arm):
    """The grade of an arm over its first GRADE_ARM_M m, outwards from the node; None when too short."""
    z = r.ground(arm)
    return float((z[GRADE_ARM_M] - z[0]) / GRADE_ARM_M) if len(z) > GRADE_ARM_M else None


def _project(p, line):
    """(the point of line (m, 2) nearest to p, the index s of its segment line[s]..line[s+1], the distance)."""
    A, B = line[:-1], line[1:]
    d = B - A
    t = np.clip(np.einsum("ij,ij->i", p - A, d) / np.maximum(np.einsum("ij,ij->i", d, d), 1e-12), 0.0, 1.0)
    q = A + t[:, None] * d
    dist = np.hypot(*(p - q).T)
    s = int(np.argmin(dist))
    return q[s], s, float(dist[s])


def find_nodes(lines):
    """The junctions of road part polylines (each (m, 2) m xy): where two part ends meet (within NODE_TOL_M), or where
    a part's end lies on another part's centre line (a T: the through part gives two arms, forward and backward).
    Returns [(xy, [(part index, arm xy), ...]), ...] for each node with at least two arms; an arm is up to ARM_M m of
    a part out from the node (1 m samples, the first at the node, none when the part there is shorter than 3 m)."""
    cent = [None if len(l) < 2 else dv._resample(np.asarray(l, np.float64)) for l in lines]
    touches = []    # (position, part index, the part's line outwards from there)
    for k, line in enumerate(lines):
        c = cent[k]
        ends = (np.asarray(line[0], np.float64), np.asarray(line[-1], np.float64))
        if c is not None:      # (a part under 3 m gives no arm of its own, but its ends still meet others)
            touches.append((ends[0], k, c))
            touches.append((ends[1], k, c[::-1]))
        for e in ends:
            for j, cj in enumerate(cent):
                if j == k or cj is None:
                    continue
                q, s, dist = _project(e, cj)
                if dist <= NODE_TOL_M:
                    touches.append((q, j, cj[s + 1:]))          # forward from q
                    touches.append((q, j, cj[:s + 1][::-1]))    # backward from q
    clusters = []   # [position, [(part index, outward line), ...]]
    for pos, k, out in touches:
        if len(out) == 0:
            continue
        for c in clusters:
            if dnp.hypot(*(c[0] - pos)) <= NODE_TOL_M:
                c[1].append((k, out))
                break
        else:
            clusters.append([pos, [(k, out)]])
    nodes = []
    for pos, members in clusters:
        arms, seen = [], set()
        for k, out in members:
            key = (k, round(float(out[0][0]), 1), round(float(out[0][1]), 1))
            if key in seen:
                continue
            seen.add(key)
            arm = _arm(pos, out)
            if arm is not None:
                arms.append((k, arm))
        if len(arms) >= 2:
            nodes.append((pos, arms))
    return nodes


def _nodes(r, parts, ok):
    """({count, bad}, node_step_max, node_step_raw_max, node_grade_jump_max): the junctions (find_nodes) of the parts;
    step_m is the excess of the graded step over the raw one (node_excess), step_raw_m the raw ground's own step."""
    bad, count, step_max, raw_max, jump_max = [], 0, 0.0, 0.0, 0.0
    for node, arms in find_nodes([p.raw for p in parts]):
        measured = False
        for x in range(len(arms)):
            for y in range(x + 1, len(arms)):
                (ka, a), (kb, b) = arms[x], arms[y]
                path = np.r_[a[::-1], b[1:]]          # from a's far end, through the node, to b's far end
                if not ok(path).all():
                    continue
                measured = True
                rw = _raw(r, path)
                step = node_excess(r.ground(path), rw)
                raw_step = float(np.nan_to_num(node_step(rw)))
                step_max = max(step_max, step)
                raw_max = max(raw_max, raw_step)
                if step > LIMITS["node_step_m"]:
                    bad.append({"at": _xy(node), "step_m": round(step, 4), "step_raw_m": round(raw_step, 4),
                                "parts": [parts[ka].fid, parts[kb].fid]})
                ga, gb = _grade_out(r, a), _grade_out(r, b)
                if ga is not None and gb is not None:
                    jump_max = max(jump_max, abs(ga + gb))  # (the two outward grades: a straight rise gives 0)
        count += measured
    return {"count": count, "bad": bad}, step_max, raw_max, jump_max


def _motorway(m, gz, rz, c, segs):
    """radius_min_m / radius_raw_m / grade_max of a motorway part, each piece of its centre line on its own."""
    rad_g = rad_r = grade = None
    for seg in segs:
        if len(seg) >= 2 * CHORD_M + 1:
            R, i = vertical_radius(gz[seg])
            if rad_g is None or R < rad_g[0]:
                rad_g = (R, seg[i])
            R, _ = vertical_radius(rz[seg])
            if rad_r is None or R < rad_r[0]:
                rad_r = (R, None)
        g = _grades(gz[seg])
        if g.size and (grade is None or g.max() > grade[0]):
            grade = (float(g.max()), seg[int(np.argmax(g))])
    if rad_g is not None:
        m["radius_min_m"] = round(rad_g[0], 4) if np.isfinite(rad_g[0]) else None
        m["radius_min_at"] = _xy(c[rad_g[1]]) if rad_g[1] is not None else None
        m["radius_raw_m"] = round(rad_r[0], 4) if np.isfinite(rad_r[0]) else None
    if grade is not None:
        m["grade_max"] = round(grade[0], 4)
        m["grade_max_at"] = _xy(c[grade[1]])


def _part_metrics(r, p, ok):
    """The metrics of one road part (a dict without its key); empty when nothing of it lies within the limits' scope."""
    c = p.centre
    valid = ok(c)
    gz, rz = r.ground(c), _raw(r, c)
    m = {"class": p.cls}
    offs = np.arange(-(p.width / 2 + 2), p.width / 2 + 2 + 1e-9, EDGE_OFFSET_M)
    idx = [i for i in range(0, len(c), EDGE_EVERY_M) if valid[i]]
    if idx:
        P = c[idx][:, None, :] + offs[None, :, None] * p.normal[idx][:, None, :]
        flat = P.reshape(-1, 2)
        G = r.ground(flat).reshape(len(idx), len(offs))
        R = _raw(r, flat).reshape(len(idx), len(offs))
        best, at = 0.0, c[idx[0]]
        for a in range(len(idx)):
            ex = _edge_excess(G[a], R[a])
            if ex.size and ex.max() > best:
                best, at = float(ex.max()), P[a, int(np.argmax(ex)) + 1]
        m["edge_m"] = round(best, 4)
        m["edge_at"] = _xy(at)
    segs = _pieces(valid, 3)
    if p.cls in CITY | TOWN and segs:
        cat = np.concatenate(segs)
        g, rr = gz[cat], rz[cat]
        p95, mx = follow(g, rr)
        m["follow_p95_m"] = round(p95, 4)
        m["follow_max_m"] = round(mx, 4)
        m["follow_max_at"] = _xy(c[cat[int(np.argmax(np.abs(g - rr)))]])
    if p.cls in MOTORWAY:
        _motorway(m, gz, rz, c, segs)
    return m


def measure(name):
    """The site's road profiles (the module's docstring): {site, nodes, node_step_max, node_grade_jump_max, parts}."""
    with dv.replay(name) as r:
        ok = _ok_fn(r)
        parts = _parts(r)
        nodes, step_max, raw_max, jump_max = _nodes(r, parts, ok)
        out = {}
        for p in parts:
            m = _part_metrics(r, p, ok)
            if len(m) > 1:
                out[p.key] = m
    return {"site": name, "nodes": nodes, "node_step_max": round(step_max, 4),
            "node_step_raw_max": round(raw_max, 4), "node_grade_jump_max": round(jump_max, 4), "parts": out}


def checks(result):
    """{nodes, edges, follow, motorway}: the lines of each check's broken limits, the worst first; [] = ok."""
    found = {c: [] for c in CHECKS}
    for b in result["nodes"]["bad"]:
        limit = LIMITS["node_step_m"]
        found["nodes"].append((b["step_m"] / limit, f"node {b['at']}: step excess {b['step_m']} m (raw step "
                                                    f"{b.get('step_raw_m')} m) between {b['parts'][0]} "
                                                    f"and {b['parts'][1]} (limit {limit} m)"))
    for key, p in result["parts"].items():
        cls = p["class"]
        if p.get("edge_m", 0.0) > LIMITS["edge_m"]:
            found["edges"].append((p["edge_m"] / LIMITS["edge_m"],
                                   f"{key} ({cls}): edge {p['edge_m']} m at {p['edge_at']}"))
        group = "city" if cls in CITY else "town" if cls in TOWN else None
        if group and "follow_p95_m" in p:
            for stat in ("follow_p95_m", "follow_max_m"):
                limit = LIMITS[stat][group]
                if p[stat] > limit:
                    found["follow"].append((p[stat] / limit, f"{key} ({cls}): {stat} {p[stat]} m > {limit} m "
                                                             f"(worst at {p.get('follow_max_at')})"))
        if cls in MOTORWAY:
            rmin = p.get("radius_min_m")
            limit = LIMITS["radius_min_m"][cls]
            if rmin is not None and rmin < limit:
                found["motorway"].append((limit / rmin, f"{key} ({cls}): radius {rmin} m < {limit} m at "
                                                         f"{p.get('radius_min_at')}"))
            gm = p.get("grade_max")
            limit = LIMITS["grade_max"][cls]
            if gm is not None and gm > limit:
                found["motorway"].append((gm / limit, f"{key} ({cls}): grade {gm} > {limit} at "
                                                      f"{p.get('grade_max_at')}"))
    return {c: [t for _, t in sorted(v, key=lambda x: -x[0])] for c, v in found.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="sites (default: all)")
    ap.add_argument("--json", action="store_true", help="one JSON line per site")
    a = ap.parse_args()
    unknown = sorted(set(a.names) - set(SITES))
    if unknown:
        ap.error(f"unknown sites: {', '.join(unknown)} (known: {', '.join(SITES)})")
    for name in a.names or SITES:
        t0 = time.perf_counter()
        result = measure(name)
        took = time.perf_counter() - t0
        if a.json:
            print(json.dumps(result), flush=True)
            continue
        found = checks(result)
        print(f"{name}: {took:.1f} s, {result['nodes']['count']} nodes, node_step_max {result['node_step_max']} m "
              f"(raw {result['node_step_raw_max']} m), grade_jump_max {result['node_grade_jump_max']}, "
              f"{len(result['parts'])} parts")
        for c in CHECKS:
            print(f"  {c}: {len(found[c])}")
            for line in found[c][:5]:
                print(f"    {line}")


if __name__ == "__main__":
    main()
