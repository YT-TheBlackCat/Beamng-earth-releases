# Junction fixtures (Leonberg)

Map data round two Leonberg junctions, for the offline junction tests (tests/kreuzfixtures.py, test_kreuz_cases.py).

| file | spot | point (lat, lon) | captured | lane_tags |
|---|---|---|---|---|
| `kreuz_K1.npz` | K1 | 48.798180, 9.014340 | 2026-11-23 (cloud) | true |
| `kreuz_K2.npz` | K2 | 48.801977, 9.016582 | 2026-11-23 (cloud) | true |

Each holds the `roads`, `infrastructure` and `road_routes` layers (mapgeom.Features arrays) of the z14 cell round the
point, only the features within 250 m of it. Overture release 2026-10-21.0. No elevation: the tests run on flat ground.

Both fixtures carry the LaneTags join (`lane_tags` true, re-captured 2026-11-23 once the cloud reached
tecnagamer.github.io): `turn:lanes`, `lanes:forward/backward`, `osm_lane_parts`, `osm_priority_road` are in. A
fixture captured without them shows `lane_tags` false; lane counts and German priority are degraded there.

Recreate one (downloads what the cache lacks):

```text
python tools/leonberg_junction_fixture.py NAME LAT LON [--keep-m 250] [--cache ~/.cache/beamng-earth] [--host cloud|local]
```

Road geometry: © OpenStreetMap contributors, Overture Maps Foundation, ODbL-1.0 (data from TomTom).
