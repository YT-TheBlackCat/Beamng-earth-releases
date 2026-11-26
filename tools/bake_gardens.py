"""Bake the gardens of Leonberg (or a test spot) once, ahead of the game: download the LGL BW sources and write one
recipe per 1 km square (server/gardenbake.py), which the server reads (server/gardens.py).

    .venv/bin/python tools/bake_gardens.py --spots U8              # the km squares within --radius of U8 Ramtel
    .venv/bin/python tools/bake_gardens.py --area leonberg         # 16 km2 around Leonberg (about 1.6 GB of downloads)
    .venv/bin/python tools/bake_gardens.py --km 502_5404 --force   # one square again

Sources (Datenlizenz Deutschland - Namensnennung 2.0, nothing of them goes into the repository): DOP20 RGBI and nDOM1
as 2 km zips (about 360 MB and 10 MB each) and the ALKIS Shape zips of the Gemarkungen (17 MB for Leonberg), kept
under <cache>/lgl/ (--delete-photos drops the photo zips after their squares are baked). The recipes go to
<cache>/gardens/<version>/<e>_<n>.npz (a few hundred KB each). A game started before a square was baked has its
blocks again (the block keys hold the recipes); its road tiles (hedges) and vegetation come with the next build of
the place (a new code version or a cleared revisit cache).
"""
import argparse
import math
import os
import sys
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "server"))
sys.path.insert(0, HERE)

import gardenbake  # noqa: E402
import gardens  # noqa: E402
import lod2geom  # noqa: E402

AREAS = {"leonberg": (499, 5402, 503, 5406)}      # km: west, south, east, north (exclusive), the research's 4 zips


def download(url, path, tries=4):
    """url -> path (atomic), skipped when path exists. True when the file is there afterwards."""
    if os.path.exists(path):
        return True
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    for attempt in range(tries):
        try:
            t0 = time.time()
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "beamng-earth-bake"}),
                                        timeout=120) as r, open(tmp, "wb") as fh:
                total = int(r.headers.get("Content-Length") or 0)
                got, last = 0, 0.0
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    got += len(chunk)
                    if time.time() - last > 5:
                        last = time.time()
                        print(f"  {os.path.basename(path)}: {got / 1e6:.0f}" + (f" / {total / 1e6:.0f}" if total else "")
                              + " MB", flush=True)
            if total and got != total:
                raise OSError(f"cut short: {got} of {total} bytes")
            os.replace(tmp, path)
            print(f"  {os.path.basename(path)}: {got / 1e6:.0f} MB in {time.time() - t0:.0f} s", flush=True)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"  {url}: not there (404)")
                return False
            print(f"  {url}: {e} (try {attempt + 1} of {tries})")
        except OSError as e:
            print(f"  {url}: {e} (try {attempt + 1} of {tries})")
        time.sleep(3 * (attempt + 1))
    try:
        os.remove(tmp)
    except OSError:
        pass
    return False


def squares_of(args):
    out = set()
    if args.km:
        for k in args.km.split(","):
            e, n = k.split("_")
            out.add((int(e), int(n)))
    if args.area:
        w, s, e, n = AREAS[args.area]
        out |= {(a, b) for a in range(w, e) for b in range(s, n)}
    if args.spots:
        from leonberg_surroundings_report import SPOTS
        spots = {s[0]: s for s in SPOTS}
        for name in args.spots.split(","):
            _, _, lat, lon = spots[name.strip()]
            E, N = lod2geom.utm32_from_wgs(lat, lon)
            r = args.radius
            for a in range(math.floor((float(E) - r) / 1000), math.floor((float(E) + r) / 1000) + 1):
                for b in range(math.floor((float(N) - r) / 1000), math.floor((float(N) + r) / 1000) + 1):
                    out.add((a, b))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spots", help="test spots (tools/leonberg_surroundings_report.SPOTS), e.g. U8,U3")
    ap.add_argument("--radius", type=float, default=300.0, help="m around each spot (default 300)")
    ap.add_argument("--area", choices=sorted(AREAS), help="a whole area")
    ap.add_argument("--km", help="km squares e_n, e.g. 502_5404,501_5404")
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/beamng-earth"))
    ap.add_argument("--force", action="store_true", help="bake squares that have a recipe again")
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--delete-photos", action="store_true", help="remove the DOP20 zips once their squares are baked")
    args = ap.parse_args()
    squares = squares_of(args)
    if not squares:
        ap.error("nothing to bake: --spots, --area or --km")
    src = os.path.join(args.cache, "lgl")
    out_dir = gardens.folder(args.cache)
    todo = [k for k in squares if args.force or not os.path.exists(gardenbake.km_path(out_dir, *k))]
    print(f"{len(squares)} km squares, {len(todo)} to bake -> {out_dir}")
    zips = sorted({gardenbake.zip_of_km(e, n) for e, n in todo})
    ok_zips = set()
    for ze, zn in zips:
        a = download(gardenbake.NDOM_URL.format(e=ze, n=zn), os.path.join(src, "ndom1", f"ndom1_32_{ze}_{zn}_2_bw.zip"))
        b = download(gardenbake.DOP_URL.format(e=ze, n=zn),
                     os.path.join(src, "dop20", f"dop20rgbi_32_{ze}_{zn}_2_bw.zip"))
        if a and b:
            ok_zips.add((ze, zn))
    for e, n in todo:
        for name in gardenbake.alkis_zips_over(e * 1000, n * 1000, (e + 1) * 1000, (n + 1) * 1000):
            download(gardenbake.ALKIS_URL.format(name=urllib.parse.quote(name)), os.path.join(src, "alkis", name))
    if args.download_only:
        return
    baked = 0
    for e, n in todo:
        if gardenbake.zip_of_km(e, n) not in ok_zips:
            print(f"gardens {e}_{n}: sources missing, skipped")
            continue
        if gardenbake.bake_km(e, n, src, gardenbake.km_path(out_dir, e, n)) is not None:
            baked += 1
    if args.delete_photos:
        for ze, zn in ok_zips:
            rest = [k for k in todo if gardenbake.zip_of_km(*k) == (ze, zn)
                    and not os.path.exists(gardenbake.km_path(out_dir, *k))]
            if not rest:
                try:
                    os.remove(os.path.join(src, "dop20", f"dop20rgbi_32_{ze}_{zn}_2_bw.zip"))
                except OSError:
                    pass
    print(f"{baked} squares baked. {gardenbake.CREDIT}")


if __name__ == "__main__":
    main()
