# Notice

**BeamNG.Earth Leonberg** is a modified copy of BeamNG.Earth by TecnaGamer, https://github.com/TecnaGamer/BeamNG.Earth,
released under the MIT licence (see `LICENSE`). The changes for Leonberg are released under the same licence.
This repository is a release build: one commit per tested release, without the development history.

Map, height, building and imagery data are not part of the repository; the server downloads them at run time from open
sources and caches them on the player's disk. Every source with its licence and the attribution it asks for is in
`CREDITS.md` and `SOURCES.md`. For Leonberg these are mainly:

- OpenStreetMap contributors, ODbL-1.0 (roads, places, map features).
- Landesamt fuer Geoinformation und Landentwicklung Baden-Wuerttemberg (LGL), Open GeoData, dl-de/by-2-0:
  LoD2 building models, true orthophotos with infrared (DOP20 RGBI), object heights (nDOM1), cadastral data (ALKIS).
- Mapterhorn and the national elevation sources it combines (see its attribution page); Copernicus DEM GLO-30.

The server reports the number of players anonymously to the original project's service (`earth.tgforever.net`, no
location). Switch it off with `BNE_TELEMETRY=off`; see `TELEMETRY.md`.

Provided "as is", without warranty (MIT).
