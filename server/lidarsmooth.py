"""Adaptive smoothing of the roads on lidar: as little as the data allows, no edge the road has not.

On lidar (grading's lidar mode) a town's roads are not graded: their carriageway is the DTM itself (grading's lidar
keeps, LIDAR_UNGRADED). The DTM is good, but not on every metre: a bridge's hole runs onto its approach, the
interpolation between scan lines leaves seams, a wall or a hedge bleeds into a lane, a stair-stepped lane in the
Altstadt rises in 0.2-0.4 m ledges. Each of these is a ledge a car with 4 cm of ground clearance hits (2026-11-24:
75 861 m of wheel track round Leonberg had a 1 m kink over 2 cm, 551 m one over 0.3 m, the worst 3 m).

The filter (profile()):
  1. A curvature cap: the profile's second difference may not exceed kappa (1 / the class's least crest or sag radius,
     MIN_RADIUS_M) anywhere. Wherever the lidar keeps within it, it is kept to the millimetre: a steady grade, a ramp, a
     hill, a real crest or sag pass unchanged. Where it does not, the least change that makes it: a parabolic opening
     (sharp crests, spikes) and closing (sharp sags, pits) with the curvature kappa, both orders averaged (each
     order keeps -kappa <= z'' <= kappa, and so does their mean). A morphological filter takes a spike off whole, it
     does not spread it into a bump as a mean would; a ledge becomes an S of curvature +-kappa, no longer than it must.
  2. Adaptive: where the lidar breaks the cap on many metres close together (a washboard, a stair-stepped lane, the
     ragged edge of a hole), the capped profile is smoothed further, up to ZONE_SIGMA_M, then capped again: there the
     data is noise, not the road. A single ledge in a clean profile gets no more than step 1's S.
  3. Ends: past its ends a profile goes on along the line fitted to its last END_FREE_M (_pad): an end is no anchor
     (the lidar at a junction's node is as rough as anywhere; held there, a side road carried a ledge into the main
     road). grading ranks the roads at a junction by class (priority()). An end at a bridge or a tunnel (end_fix)
     first gets the DTM's hole or the embankment over a culvert replaced by the trend of the road before it: the lidar
     has no road there, the structure carries it on.
  4. Holes: where a way or a stream passes under the road (hole_fix), the DTM's floor or bed under it is no road
     either: the road's trend over it (a sag within the limit stays).

A keep's carriageway (surface()): the lidar across it (columns every ~1 m), each column's structure ends and holes
mended on its own (3., 4.); the centre's height (a trimmed least-squares line across each station) filtered as above,
its correction added to every column (the lidar's crossfall and crown stay), each column capped again, so every wheel
track keeps the limit too. At a junction (grading.keep_ground) the higher road (class, then length) keeps its surface
and the lesser takes it where their carriageways overlap (pin(): eased off onto its own profile past it); on the margin
beside a carriageway the DTM moved by the correction at its edge.

Pure functions on numpy arrays (+ - * / min max, sqrt and the native smooth_profile: the same bits on every platform).
Switches: BNE_LIDAR_SMOOTH (on; off: the keeps are the raw lidar again), BNE_LIDAR_SMOOTH_RADIUS_SCALE (1: all radii
times this), BNE_LIDAR_SMOOTH_ZONE (ZONE_SIGMA_M; 0: no adaptive zones), BNE_LIDAR_SMOOTH_END (on: the bridge / tunnel
ends' fix), BNE_LIDAR_SMOOTH_HOLE (on: the holes' fix), BNE_LIDAR_SMOOTH_PIN (on: junctions as above; off: their mean
by weight), BNE_LIDAR_SMOOTH_ACROSS (0: the cap along only), grading's BNE_LIDAR_SMOOTH_ROUND (0.7: a keep's corners
rounded). Measured with tools/leonberg_profile_bench.py.
"""
import math
import os

import numpy as np


def _env(name, default):
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return 0.0 if v.lower() in ("off", "false", "no") else default


ON = os.environ.get("BNE_LIDAR_SMOOTH", "on").lower() not in ("0", "off", "false", "no")

# The least crest / sag radius (m) a road of each class may have where the lidar does not show it is real; 1 / R is the
# limit of the profile's second difference per square metre (a 1 m step's kink). A steady grade has none, so every ramp
# and hill keeps its heights. The floor, DEFAULT_RADIUS_M (50 m): a 1 m kink of 2 cm at most, a 2.6 m wheelbase sinks
# 1.7 cm into such a sag (a car with 4 cm of clearance clears it). The faster classes more: a crest of 80 m at 50 km/h
# lifts a car 0.25 g, a motorway's 400 m at 130 km/h 0.33 g (the plans of motorways and trunks are graded anyway; this
# is their safety net).
MIN_RADIUS_M = {
    "motorway": 400.0, "trunk": 250.0, "motorway_link": 150.0, "trunk_link": 150.0,
    "primary": 80.0, "primary_link": 60.0, "secondary": 70.0, "secondary_link": 60.0,
    "tertiary": 60.0, "tertiary_link": 60.0, "busway": 60.0,
}
DEFAULT_RADIUS_M = 50.0
UNSMOOTHED = {"steps"}          # real stairs
RADIUS_SCALE = max(_env("BNE_LIDAR_SMOOTH_RADIUS_SCALE", 1.0), 1e-3)

END_FREE_M = 8.0                # m at an end whose straight line the profile goes on along past it (_pad)
CAP_DEPTH_M = 6.0               # m: the deepest pit / highest spike the cap bends round in one piece (its window)
CAP_WINDOW_MAX = 80             # samples: the window's limit (a motorway's 400 m: 70)
ZONE_SIGMA_M = _env("BNE_LIDAR_SMOOTH_ZONE", 6.0)  # m: the most smoothing in a zone of many kinks (0: none)
ZONE_HALF_M = 5.0               # m either side over which the share of kinks is counted ...
ZONE_FROM, ZONE_FULL = 0.15, 0.5    # ... no zone smoothing up to this share of the metres, all of it from this one
ZONE_LEVELS_M = (0.0, 1.0, 2.0, 4.0, 6.0, 8.0)     # the Gaussians blended between (sigma by station)
ZONE_EASE_M = 4.0               # m: the zone's sigma eased along the road (no seam where a zone starts)
END_FIX_ON = _env("BNE_LIDAR_SMOOTH_END", 1.0) > 0
END_FIT_M = (12.0, 40.0)        # m from a structure's end: the stretch whose straight line is the road's trend there
END_DEV_M = 0.25                # m below (a bridge's hole) or above (an embankment over a culvert) the trend: no road
END_MAX_M = 30.0                # m: the longest stretch at an end replaced by the trend
END_GRADE_MAX = 0.15            # the trend's steepest grade
TRIM_M = 0.15                   # m: a column this far off a station's plane is left out of its fit (a wall, a hedge)
# Across the road the same limit times this: a curb or a wall's foot inside the carriageway would be no ledge a car
# turning in or out meets, a steady crossfall stays, a crown rounds over ~2 m. 0, the default: the lidar across as it
# is (1 measured worse: more edges where junctions meet, more off the lidar).
ACROSS_SCALE = max(_env("BNE_LIDAR_SMOOTH_ACROSS", 0.0), 0.0)
ACROSS_ROUNDS = 2               # alternations of the across and the along cap
PIN_ON = _env("BNE_LIDAR_SMOOTH_PIN", 1.0) > 0      # a junction: the lesser road meets the greater one's surface
PIN_EASE_M = (3.0, 40.0)        # m: the shortest and longest run over which it eases off to its own
# A way or a stream passing under the road (a pedestrian underpass, a culvert, a stream past a bridge mapped too short):
# the lidar's ground there is its floor or bed (to the classification the road over it is a bridge), so the road sank
# into it with its banks (Leonberg 2026-11-24: Eltinger Strasse 1.9 m over the Rathaus underpass, Renninger Strasse
# 1.3 m on one side past the Glems bridge). hole_fix: the road's trend over it instead. BNE_LIDAR_SMOOTH_HOLE=off: none.
HOLE_ON = _env("BNE_LIDAR_SMOOTH_HOLE", 1.0) > 0
HOLE_CORE_M = 2.0               # m past the way's half width always in the hole (its walls, its stairs' first steps)
HOLE_REACH_M = 25.0             # m either side of the crossing: the farthest a hole reaches ...
HOLE_FIT_M = 15.0               # ... and the road past that fitted with it (the trend)
HOLE_TRIM_M = 0.10              # m the closing lifts a sample of the hole at least
HOLE_SAG_MIN = 0.1              # x k: the least bend of the closing that finds a hole (sag 0: the road's straight line)


def kappa(cls):
    """The class's curvature limit (1 / m), None for a class left as the lidar has it."""
    if cls in UNSMOOTHED:
        return None
    return 1.0 / (MIN_RADIUS_M.get(cls, DEFAULT_RADIUS_M) * RADIUS_SCALE)


def priority(k):
    """A keep's weight against the others where they overlap (a junction), by its curvature limit k: the higher class
    (the larger radius) holds its surface, the lower one comes to it (grading._keep_rank: the same class: the
    longer)."""
    if not k:
        return 1.0
    r = 1.0 / (float(k) * RADIUS_SCALE * DEFAULT_RADIUS_M)
    return r * r * r * r


def _gauss(z, step, sigma):
    """Gaussian smoothing along axis 0 (grading.smooth_profile per column: the ends keep their heights)."""
    z = np.asarray(z, np.float64)
    if sigma <= 0.0 or len(z) < 3:
        return np.array(z, copy=True)
    from grading import smooth_profile
    if z.ndim == 1:
        return smooth_profile(z, step, sigma)
    return np.stack([smooth_profile(np.ascontiguousarray(z[:, j]), step, sigma) for j in range(z.shape[1])], axis=1)


def _pad(z, m):
    """z (n, ...) continued m samples past each end along the straight line fitted to its last END_FREE_M (least
    squares; the whole of a shorter z): the trend goes on, a ledge or a spike at the very end is no anchor."""
    n = len(z)
    e = max(2, min(n, int(END_FREE_M)))
    x = np.arange(e, dtype=np.float64)
    xm = float(x.mean())
    sxx = float(((x - xm) * (x - xm)).sum())
    j = np.arange(1, m + 1, dtype=np.float64)
    parts = []
    for seg, ahead in ((z[:e][::-1], j), (z[-e:], j)):
        ym = seg.mean(axis=0)
        g = ((x - xm).reshape((-1,) + (1,) * (z.ndim - 1)) * (seg - ym)).sum(axis=0) / sxx
        parts.append(ym + g * ((e - 1 - xm) + ahead).reshape((-1,) + (1,) * (z.ndim - 1)))
    return np.concatenate([parts[0][::-1], z, parts[1]])


def _morph(z, k, w, op):
    """Parabolic erosion (op "min": min_j z[i+j] + k j^2 / 2) or dilation ("max": max_j z[i+j] - k j^2 / 2) of z along
    axis 0, |j| <= w."""
    out = z.copy()
    n = len(z)
    for j in range(1, min(w, n - 1) + 1):
        c = k * j * j / 2.0
        if op == "min":
            np.minimum(out[:-j], z[j:] + c, out=out[:-j])
            np.minimum(out[j:], z[:-j] + c, out=out[j:])
        else:
            np.maximum(out[:-j], z[j:] - c, out=out[:-j])
            np.maximum(out[j:], z[:-j] - c, out=out[j:])
    return out


def window(k, step=1.0):
    """Samples the cap looks either side: a parabola of curvature k rises CAP_DEPTH_M over it."""
    return int(min(CAP_WINDOW_MAX, math.ceil(math.sqrt(2.0 * CAP_DEPTH_M / k) / step)))


def cap(z, k, step=1.0, rounds=3):
    """z (n,) or (n, m) every `step` m with its second difference within +-k step^2 (step 1 of the module's docstring):
    the mean of the parabolic opening-then-closing and closing-then-opening, past the ends the trend (_pad)."""
    z = np.asarray(z, np.float64)
    if k is None or len(z) < 3:
        return np.array(z, copy=True)
    kk = k * step * step
    w = window(k, step)
    m = 4 * w
    out = z
    for _ in range(rounds):
        p = _pad(out, m)
        oc = _morph(_morph(_morph(_morph(p, kk, w, "max"), kk, w, "min"), kk, w, "min"), kk, w, "max")
        co = _morph(_morph(_morph(_morph(p, kk, w, "min"), kk, w, "max"), kk, w, "max"), kk, w, "min")
        out = (0.5 * (oc + co))[m:m + len(z)]
        d2 = out[:-2] - 2.0 * out[1:-1] + out[2:]
        if not len(d2) or float(np.max(np.abs(d2))) <= kk * (1.0 + 1e-6):
            break
    return out


def _share(bad, half):
    """The share of True in bad within +-half samples of each (fewer at the ends)."""
    c = np.r_[0.0, np.cumsum(bad.astype(np.float64))]
    n = len(bad)
    i = np.arange(n)
    lo, hi = np.maximum(i - half, 0), np.minimum(i + half + 1, n)
    return (c[hi] - c[lo]) / (hi - lo)


def zone_sigma(z, k, step=1.0):
    """The zone smoothing (sigma m per sample) of profile z: step 2 of the module's docstring."""
    n = len(z)
    if ZONE_SIGMA_M <= 0.0 or n < 3:
        return np.zeros(n)
    kk = k * step * step
    bad = np.zeros(n, bool)
    bad[1:-1] = np.abs(z[:-2] - 2.0 * z[1:-1] + z[2:]) > kk
    share = _share(bad, max(1, int(round(ZONE_HALF_M / step))))
    sig = ZONE_SIGMA_M * np.clip((share - ZONE_FROM) / (ZONE_FULL - ZONE_FROM), 0.0, 1.0)
    if not sig.any():
        return sig
    return np.clip(_gauss(sig, step, ZONE_EASE_M), 0.0, ZONE_SIGMA_M)


def _blend(z, sig, step):
    """z smoothed by sigma sig[i] at sample i: the Gaussians of ZONE_LEVELS_M blended linearly by sigma."""
    levels = [s for s in ZONE_LEVELS_M if s <= ZONE_SIGMA_M + 1e-9]
    if ZONE_SIGMA_M > levels[-1]:
        levels.append(ZONE_SIGMA_M)
    top = float(sig.max())
    use = [s for s in levels if s <= top] + [s for s in levels if s > top][:1]
    bank = np.stack([z if s == 0.0 else _gauss(z, step, s) for s in use])
    f = np.interp(sig, use, np.arange(len(use), dtype=np.float64))
    lo = np.minimum(np.floor(f).astype(np.int64), len(use) - 1)
    hi = np.minimum(lo + 1, len(use) - 1)
    a = f - lo
    i = np.arange(len(z))
    return bank[lo, i] * (1.0 - a) + bank[hi, i] * a


def end_fix(z, kind, at_end, step=1.0):
    """z with the stretch at one end (at_end: its last samples, else its first) that the lidar has no road on
    replaced by the trend of the road before it: kind "bridge" a hole (z under the trend by END_DEV_M or more),
    "tunnel" an embankment (over it). The stretch: from the end inwards while it is that far off (and on to where it
    meets the trend), at most END_MAX_M; the trend: the least-squares line through END_FIT_M from the end (a shorter
    profile: its last 4 m or more past 4 m from the structure), its grade within END_GRADE_MAX. Unchanged when the
    profile is under 8 m or has no such stretch."""
    z = np.array(z, np.float64, copy=True)
    if kind not in ("bridge", "tunnel") or not END_FIX_ON:
        return z
    n = len(z)
    a, b = int(round(END_FIT_M[0] / step)), int(round(END_FIT_M[1] / step))
    a = min(a, n - 4)                       # (a short part, between the structure and a junction: its far end)
    if a < 4:
        return z
    b = min(b, n - 1)
    zz = z[::-1] if at_end else z           # (index 0: the end at the structure)
    s = np.arange(n, dtype=np.float64) * step
    xs, ys = s[a:b + 1], zz[a:b + 1]
    xm, ym = float(xs.mean()), float(ys.mean())
    sxx = float(((xs - xm) * (xs - xm)).sum())
    g = float(((xs - xm) * (ys - ym)).sum()) / sxx if sxx > 0 else 0.0
    g = min(max(g, -END_GRADE_MAX), END_GRADE_MAX)      # (a few metres' fit: no steeper than a road)
    trend = ym + g * (s - xm)
    off = (trend - zz) if kind == "bridge" else (zz - trend)
    lim = min(int(round(END_MAX_M / step)), a)
    run = 0
    while run < lim and off[run] >= END_DEV_M:
        run += 1
    if run == 0:
        return z
    while run < lim and off[run] > 0.0:
        run += 1
    zz = zz.copy()
    zz[:run] = trend[:run]
    return zz[::-1] if at_end else zz


def _fit2(x, y, cmax, cmin=None):
    """(a, b, c) of the least-squares y = a + b x + c x^2 with cmin <= 2 c <= cmax (cmin None: -cmax; both 0: a line),
    by the normal equations in closed form (+ - * / only: the same bits on every platform). None for fewer than 2
    points or no spread."""
    cmin = -cmax if cmin is None else cmin
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    if len(x) < 2:
        return None
    s0 = float(len(x))
    s1, s2 = float(x.sum()), float((x * x).sum())
    t0, t1 = float(y.sum()), float((x * y).sum())
    c = 0.0
    if (cmax > 0 or cmin < 0) and len(x) >= 3:
        s3, s4, t2 = float((x * x * x).sum()), float((x * x * x * x).sum()), float((x * x * y).sum())
        det = s0 * (s2 * s4 - s3 * s3) - s1 * (s1 * s4 - s3 * s2) + s2 * (s1 * s3 - s2 * s2)
        if abs(det) > 1e-9 * max(1.0, s4 * s4):
            c = (s0 * (s2 * t2 - s3 * t1) - s1 * (s1 * t2 - s2 * t1) + t0 * (s1 * s3 - s2 * s2)) / det
            c = min(max(c, 0.5 * cmin), 0.5 * cmax)
        t0, t1 = t0 - c * s2, t1 - c * s3
    den = s0 * s2 - s1 * s1
    if den <= 1e-12:
        return None
    b = (s0 * t1 - s1 * t0) / den
    return (t0 - b * s1) / s0, b, c


def hole_fix(z, at, core, k, step=1.0, sag=1.0):
    """z (n,) sampled every `step` m with the DTM's hole where a way or a stream passes under the road at `at` m
    (core: its half width m) bridged by the road's trend. The hole: the run of samples round the crossing that a
    parabolic closing of bend sag x k (at least HOLE_SAG_MIN x k) lifts by HOLE_TRIM_M or more, if it reaches the
    core (HOLE_CORE_M past the half width), within HOLE_REACH_M of it: a sag the road may have (within sag x k) is no
    hole, a real valley stays. The trend: a parabola (a crest within k, a sag within sag x k: over an underpass the road
    is no lower than its straight line, 0) fitted to HOLE_FIT_M of road either side of the hole (a line, its grade
    within END_GRADE_MAX, where the road goes on to one side only). The hole and on while the road lies under the
    trend take the trend, if it is END_DEV_M deep somewhere; the road is never lowered."""
    z = np.array(z, np.float64, copy=True)
    n = len(z)
    if not HOLE_ON or not k or n < 5 or not np.isfinite(z).all() or not math.isfinite(float(at)):
        return z
    d = np.arange(n, dtype=np.float64) * step - float(at)
    c0 = max(float(core), 0.0) + HOLE_CORE_M
    reach = np.abs(d) <= c0 + HOLE_REACH_M
    near = np.abs(d) <= c0
    if not near.any():
        return z
    w = int(math.ceil((c0 + HOLE_REACH_M + HOLE_FIT_M) / step))
    kh = float(k) * max(float(sag), HOLE_SAG_MIN) * step * step
    p = _pad(z, w)
    closed = _morph(_morph(p, kh, w, "max"), kh, w, "min")[w:w + n]
    lift = (closed - z) >= HOLE_TRIM_M
    i0 = np.flatnonzero(near & lift)
    if not len(i0):
        return z
    lo, hi = int(i0[0]), int(i0[-1])
    while lo > 0 and lift[lo - 1] and reach[lo - 1]:
        lo -= 1
    while hi < n - 1 and lift[hi + 1] and reach[hi + 1]:
        hi += 1
    f = int(round(HOLE_FIT_M / step))
    left = np.arange(max(0, lo - f), lo)
    right = np.arange(hi + 1, min(n, hi + 1 + f))
    both = len(left) >= 3 and len(right) >= 3
    use = np.r_[left, right] if both else (left if len(left) >= len(right) else right)
    if len(use) < 3:
        return z
    got = _fit2(d[use], z[use], float(k) * max(float(sag), 0.0) if both else 0.0, -float(k) if both else 0.0)
    if got is None:
        return z
    a, b, c = got
    if not both:
        b = min(max(b, -END_GRADE_MAX), END_GRADE_MAX)      # (one side's few metres: no steeper than a road)
    tr = a + b * d + c * d * d
    under = tr > z
    while lo > 0 and under[lo - 1] and reach[lo - 1]:
        lo -= 1
    while hi < n - 1 and under[hi + 1] and reach[hi + 1]:
        hi += 1
    run = np.zeros(n, bool)
    run[lo:hi + 1] = True
    run &= under
    if not run.any() or float((tr - z)[run].max()) < END_DEV_M:
        return z
    z[run] = tr[run]
    return z


def profile(z, k, step=1.0, ends=(None, None)):
    """The filtered profile (the module's docstring) of z (n,) sampled every `step` m with the curvature limit k (None:
    z as it is); ends: (kind at the first sample, kind at the last): "bridge", "tunnel" or None."""
    z = np.asarray(z, np.float64)
    if k is None or len(z) < 3 or not np.isfinite(z).all():
        return np.array(z, copy=True)
    for at_end, kind in ((False, ends[0]), (True, ends[1])):
        if kind:
            z = end_fix(z, kind, at_end, step)
    out = cap(z, k, step)
    sig = zone_sigma(z, k, step)
    if sig.any():
        out = cap(_blend(out, sig, step), k, step)
    return out


def surface(raw, offs, k, step=1.0, ends=(None, None), holes=()):
    """The carriageway's surface (n, nt) from the lidar raw (n, nt) at stations every `step` m and offsets offs (nt,)
    across: first each column's structure ends (end_fix) and holes (holes: [(column, station m, half width m, sag)]
    where a way or a stream passes under it, hole_fix) mended on its own (a stream crossing under one side sank that
    side alone; the centre's correction lifted the other side as much); then the centre's height (a trimmed
    least-squares line across each station: a wall or a hedge bleeding into one side does not move it) filtered by
    profile(), its correction added to every column (the lidar across as it is: crossfall, crown; two roads at a
    junction agree where neither is corrected), each column capped again (cap(): every wheel track keeps the limit
    too), then each station across and the columns again (ACROSS_SCALE: no ledge across)."""
    raw = np.asarray(raw, np.float64)
    offs = np.asarray(offs, np.float64)
    if k is None or raw.shape[0] < 3 or not np.isfinite(raw).all():
        return np.array(raw, copy=True)
    if END_FIX_ON and any(ends):
        raw = raw.copy()
        for j in range(raw.shape[1]):
            for at_end, kind in ((False, ends[0]), (True, ends[1])):
                if kind:
                    raw[:, j] = end_fix(raw[:, j], kind, at_end, step)
    if HOLE_ON and len(holes):
        raw = raw.copy()
        for j, at, core, sag in holes:
            if 0 <= int(j) < raw.shape[1]:
                raw[:, int(j)] = hole_fix(raw[:, int(j)], at, core, k, step, sag)
    t = offs[None, :]
    w = np.ones_like(raw)
    for _ in range(2):
        sw = w.sum(axis=1, keepdims=True)
        tm = (w * t).sum(axis=1, keepdims=True) / sw
        zm = (w * raw).sum(axis=1, keepdims=True) / sw
        stt = (w * (t - tm) * (t - tm)).sum(axis=1, keepdims=True)
        b = np.where(stt > 1e-9, (w * (t - tm) * (raw - zm)).sum(axis=1, keepdims=True) / np.maximum(stt, 1e-9), 0.0)
        a = zm - b * tm
        keep = np.abs(raw - (a + b * t)) <= TRIM_M
        keep[keep.sum(axis=1) < 2] = True      # (a station with fewer than two columns left: all of them)
        w = keep.astype(np.float64)
    a = a[:, 0]
    out = cap(raw + (profile(a, k, step, ends) - a)[:, None], k, step, rounds=2)
    if ACROSS_SCALE > 0 and len(offs) >= 3:
        d = float(offs[1] - offs[0])
        for _ in range(ACROSS_ROUNDS):
            out = cap(out.T, k * ACROSS_SCALE, d, rounds=2).T
            out = cap(out, k, step, rounds=1)
    return out


def pin(z, mask, target, k, step=1.0):
    """z (n, m) with its masked samples set to target and the difference eased off along axis 0 past the mask (a
    smoothstep over sqrt(12 |difference| / k) m: its bend at most k / 2; within PIN_EASE_M), between two masked runs
    the mean of both: where a lesser road runs onto a greater one's carriageway it takes that one's surface there and
    meets it without a ledge, its own profile however it was smoothed."""
    z = np.asarray(z, np.float64)
    mask = np.asarray(mask, bool)
    if k is None or not mask.any():
        return np.array(z, copy=True)
    n = z.shape[0]
    d = np.where(mask, np.asarray(target, np.float64) - z, 0.0)
    i = np.arange(n)[:, None] + np.zeros(z.shape[1:], np.int64)
    prev = np.maximum.accumulate(np.where(mask, i, -1), axis=0)
    nxt = np.minimum.accumulate(np.where(mask, i, n)[::-1], axis=0)[::-1]
    num = np.zeros_like(z)
    den = np.zeros_like(z)
    for src, ok in ((prev, prev >= 0), (nxt, nxt < n)):
        at = np.clip(src, 0, n - 1)
        dd = np.take_along_axis(d, at, axis=0)
        ease = np.clip(np.sqrt(12.0 * np.abs(dd) / k), PIN_EASE_M[0], PIN_EASE_M[1])
        t = np.clip(np.abs(i - at) * step / ease, 0.0, 1.0)
        f = np.where(ok, 1.0 - t * t * (3.0 - 2.0 * t), 0.0)
        num += f * dd
        den += f
    return np.where(mask, target, z + num / np.maximum(den, 1.0))
