#!/usr/bin/env python3
"""Das Fachwerk-Material der Leonberger Altstadt (bne_bld_fachwerk), rein prozedural (kein Bildgenerator, keine
fremden Dateien): ein Gefach von 3 x 3 m (ein Stockwerk, buildings.FACHWERK_UV_M) mit dunklem Eichenholz-Rahmen
(Schwelle unten, Rähm oben, Pfosten alle 1,5 m, Brüstungsriegel auf Brüstungshöhe, Kopfbänder und eine Strebe) und
rauem, cremefarbenem Putz in den Gefachen. Der Putz ist nahezu weiß, damit die Gebäudefarbe (Vertexfarbe) den Ton
gibt; die Balken bleiben dunkel (Vertexfarben tönen beides).

Schreibt bne_bld_fachwerk.color.png / .normal.png / .data.png nach levels/.../shapes/beamngEarth/fachwerk/ und
trägt das Material in buildings.materials.json ein (Text vor der letzten Klammer, das Layout der Datei bleibt).

    python tools/make_fachwerk_texture.py
"""
import json
import os
import uuid

import numpy as np
from PIL import Image, ImageFilter

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SHAPES = os.path.join(ROOT, "mod", "beamng_earth", "levels", "beamng_earth", "art", "shapes", "beamngEarth")
OUT = os.path.join(SHAPES, "fachwerk")
VFS = "/levels/beamng_earth/art/shapes/beamngEarth/fachwerk"
NAME = "bne_bld_fachwerk"
SIZE = 512                  # px pro Wiederholung (3 m: ~170 px/m)
TILE_M = 3.0
BEAM_M = 0.17               # Balkenbreite
PLASTER = (0.93, 0.90, 0.82)
OAK = (0.20, 0.13, 0.085)
NORMAL_STRENGTH = 2.0


def _noise(rng, size, scale, octaves=4):
    """Kachelbares Rauschen (0..1): Summe gleichmäßig verteilter Zufallsgitter, bilinear wrapped hochskaliert."""
    out = np.zeros((size, size))
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        n = max(2, scale * 2 ** o)
        g = rng.random((n, n))
        img = Image.fromarray((g * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)
        out += amp * np.asarray(img, float) / 255
        tot += amp
        amp *= 0.5
    return out / tot


def _segment_mask(size, p0, p1, width):
    """Maske (0/1) eines Balkens von p0 nach p1 (Einheiten: Bruchteil der Kachel), kachelnd (wrap), Breite in Bruchteilen."""
    ys, xs = (np.mgrid[0:size, 0:size] + 0.5) / size
    mask = np.zeros((size, size), bool)
    d = np.array(p1, float) - np.array(p0, float)
    ll = float(d @ d)
    for ox in (-1, 0, 1):
        for oy in (-1, 0, 1):
            px, py = xs + ox - p0[0], ys + oy - p0[1]
            t = np.clip((px * d[0] + py * d[1]) / ll, 0.0, 1.0)
            dist = np.hypot(px - t * d[0], py - t * d[1])
            mask |= dist <= width / 2
    return mask


def beam_mask(size):
    """Alle Balken des Gefachs (v nach unten im Bild: y=0 oben = Rähm, y=1 unten = Schwelle)."""
    w = BEAM_M / TILE_M
    m = np.zeros((size, size), bool)
    for x in (0.0, 0.5):                                        # Pfosten alle 1,5 m
        m |= _segment_mask(size, (x, 0.0), (x, 1.0), w)
    m |= _segment_mask(size, (0.0, 0.0), (1.0, 0.0), w)        # Rähm / Schwelle (wrap: je eine Linie)
    m |= _segment_mask(size, (0.0, 0.72), (1.0, 0.72), w * 0.8)  # Brüstungsriegel
    for x0 in (0.0, 0.5):                                       # Kopfbänder in den oberen Ecken jedes Feldes
        m |= _segment_mask(size, (x0, 0.26), (x0 + 0.13, 0.0), w * 0.8)
        m |= _segment_mask(size, (x0 + 0.5, 0.26), (x0 + 0.37, 0.0), w * 0.8)
    m |= _segment_mask(size, (0.5, 0.72), (1.0, 0.0), w * 0.8)   # eine lange Strebe im zweiten Feld
    return m


def build_arrays(seed=7):
    rng = np.random.default_rng(seed)
    beams = beam_mask(SIZE)
    soft = np.asarray(Image.fromarray((beams * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2)), float) / 255
    plaster_n = _noise(rng, SIZE, 6) * 0.7 + _noise(rng, SIZE, 24, 2) * 0.3
    plaster = np.array(PLASTER) * (0.86 + 0.20 * plaster_n[..., None])
    # Holzmaserung: Streifen längs des Balkens ~ in beiden Richtungen schwach, plus Rauschen
    grain = _noise(rng, SIZE, 40, 2)
    grain_h = np.asarray(Image.fromarray((_noise(rng, SIZE, 8, 3) * 255).astype(np.uint8)).resize((SIZE, SIZE)), float) / 255
    oak = np.array(OAK) * (0.75 + 0.5 * (0.5 * grain + 0.5 * grain_h))[..., None]
    rgb = plaster * (1 - soft[..., None]) + oak * soft[..., None]
    height = 0.55 * plaster_n + soft * 0.8
    return np.clip(rgb * 255, 0, 255).astype(np.uint8), height, soft


def normal_map(height):
    dx = (np.roll(height, -1, 1) - np.roll(height, 1, 1)) * 0.5 * SIZE / 64
    dy = (np.roll(height, -1, 0) - np.roll(height, 1, 0)) * 0.5 * SIZE / 64
    n = np.stack([-dx * NORMAL_STRENGTH, dy * NORMAL_STRENGTH, np.ones_like(height)], 2)
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    return ((n * 0.5 + 0.5) * 255).astype(np.uint8)


def roughness(soft):
    r = 0.92 - 0.08 * soft          # Holz etwas glatter als der Putz
    return np.repeat((r * 255)[:, :, None], 3, 2).astype(np.uint8)


def material_json():
    return {"name": NAME, "mapTo": NAME, "class": "Material", "version": 1.5,
            "persistentId": str(uuid.uuid5(uuid.NAMESPACE_URL, "beamng-earth:" + NAME)),
            "Stages": [{"baseColorMap": f"{VFS}/{NAME}.color.png", "normalMap": f"{VFS}/{NAME}.normal.png",
                        "roughnessMap": f"{VFS}/{NAME}.data.png", "metallicFactor": 0,
                        "baseColorFactor": [1, 1, 1, 1], "vertColor": True},
                       {}, {}, {}], "translucent": False}


def add_material():
    p = os.path.join(SHAPES, "buildings.materials.json")
    s = open(p, encoding="utf-8", newline="").read()
    if f'"{NAME}": {{' in s:
        return
    nl = "\r\n" if "\r\n" in s else "\n"
    block = json.dumps({NAME: material_json()}, indent=2).split("\n")[1:-1]
    end = s.rstrip().rfind("}")
    s = s[:end].rstrip() + "," + nl + nl.join(block) + nl + "}" + nl
    open(p, "w", encoding="utf-8", newline="").write(s)
    json.load(open(p, encoding="utf-8"))


def build():
    os.makedirs(OUT, exist_ok=True)
    rgb, height, soft = build_arrays()
    Image.fromarray(rgb).save(os.path.join(OUT, f"{NAME}.color.png"), optimize=True)
    Image.fromarray(normal_map(height)).save(os.path.join(OUT, f"{NAME}.normal.png"), optimize=True)
    Image.fromarray(roughness(soft)).save(os.path.join(OUT, f"{NAME}.data.png"), optimize=True)
    add_material()
    print("mean colour", (rgb.reshape(-1, 3).mean(0) / 255).round(3))


if __name__ == "__main__":
    build()
