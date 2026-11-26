#!/usr/bin/env python3
"""Prepare the LGL BW LoD2 building geometry of the tiles around a point (or of named tiles) as npz for server/lod2geom.py.

  python3 tools/build_lgl_lod2_geom.py                              Leonberg, 4 km around 48.80 9.01
  python3 tools/build_lgl_lod2_geom.py --lat 48.80 --lon 9.01 --radius-km 4
  python3 tools/build_lgl_lod2_geom.py --tiles 501_5404,499_5406    named tiles (e_km_n_km)

Zips missing from <cache>/lod2geom/zips/ are downloaded; zips already in <cache>/lgl_lod2/ are used in place and never
deleted. Finished npz files are skipped unless --force. Zips downloaded here are deleted after preparing unless
--keep-zips.
"""
import argparse
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))
import lod2geom as lg  # noqa: E402

LEONBERG = (48.80, 9.01)
PAUSES = (10, 20, 40)


def download(e, n, dest):
    """True when the zip is in dest, False for an empty tile (HTTP 404), raises when every attempt failed."""
    url = lg.URL.format(e=e, n=n)
    last = None
    for attempt in range(len(PAUSES) + 1):
        try:
            with urllib.request.urlopen(url, timeout=300) as r, open(dest + ".part", "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(dest + ".part", dest)
            return True
        except urllib.error.HTTPError as err:
            if err.code == 404:
                return False
            last = err
        except OSError as err:
            last = err
        if attempt < len(PAUSES):
            print(f"  download failed ({last}), retry in {PAUSES[attempt]} s")
            time.sleep(PAUSES[attempt])
    raise RuntimeError(f"download of {url} failed: {last}")


def parse_tiles(text):
    out = []
    for item in text.split(","):
        e, n = item.strip().split("_")
        out.append((int(e), int(n)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lat", type=float, default=LEONBERG[0])
    ap.add_argument("--lon", type=float, default=LEONBERG[1])
    ap.add_argument("--radius-km", type=float, default=4.0)
    ap.add_argument("--tiles", help="named tiles, e.g. 501_5404,499_5406 (instead of lat/lon/radius)")
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/beamng-earth"))
    ap.add_argument("--force", action="store_true", help="rebuild npz files that exist")
    ap.add_argument("--keep-zips", action="store_true", help="keep the zips downloaded here")
    args = ap.parse_args()

    if args.tiles:
        tiles = parse_tiles(args.tiles)
    else:
        E, N = lg.utm32_from_wgs(args.lat, args.lon)
        E, N = float(E), float(N)
        r = args.radius_km * 1000
        tiles = lg.zip_tiles_over(E - r, N - r, E + r, N + r)
    print(f"{len(tiles)} tile(s)")

    out_dir = lg.folder(args.cache)
    zip_dir = os.path.join(args.cache, "lod2geom", "zips")
    old_dir = os.path.join(args.cache, "lgl_lod2")
    os.makedirs(zip_dir, exist_ok=True)
    failed = []
    for e, n in tiles:
        name = f"LoD2_32_{e}_{n}_2_bw.zip"
        out = os.path.join(out_dir, f"{e}_{n}.npz")
        if os.path.exists(out) and not args.force:
            print(f"{e}_{n}: exists, skipped")
            continue
        zip_path, ours = os.path.join(old_dir, name), False
        if not os.path.exists(zip_path):
            zip_path, ours = os.path.join(zip_dir, name), True
            if not os.path.exists(zip_path):
                try:
                    if not download(e, n, zip_path):
                        print(f"{e}_{n}: empty tile (404), skipped")
                        continue
                except RuntimeError as err:
                    print(f"{e}_{n}: {err}")
                    failed.append(f"{e}_{n}")
                    continue
        t0 = time.time()
        try:
            units = lg.prepare(zip_path, out)
        except Exception as err:  # keep going with the other tiles
            print(f"{e}_{n}: prepare failed: {err!r}")
            failed.append(f"{e}_{n}")
            continue
        size = os.path.getsize(out) / 1e6
        print(f"{e}_{n}: {units} units, {time.time() - t0:.1f} s, {size:.2f} MB -> {out}")
        if ours and not args.keep_zips:
            os.remove(zip_path)
    if failed:
        print("failed:", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
