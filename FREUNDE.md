# BeamNG.Earth Leonberg: Anleitung für Freunde

Eine Kopie von [BeamNG.Earth](https://github.com/TecnaGamer/BeamNG.Earth) (MIT-Lizenz), bei der **Leonberg und
Umgebung** realistischer gemacht wird: echte Höhen, Gebäude aus Landesdaten, Gärten, Fußwege, Straßenränder.
Du bekommst immer die **neueste stabile, getestete Version** aus dem öffentlichen Release-Repo
(`beamng-earth-releases`) und sie aktualisiert sich bei jedem Start von selbst.

## Du brauchst

- BeamNG.drive (Windows), einmal gestartet.
- [Git für Windows](https://git-scm.com/download/win) (Standardeinstellungen reichen).
- [Python 3.10 oder neuer](https://www.python.org/downloads/) (beim Installieren **"Add python.exe to PATH"** ankreuzen).
- Internet, Platz für ca. 1 GB (Programm 400 MB, dazu der Zwischenspeicher mit Karten- und Höhendaten).
- Ein Rechner, auf dem BeamNG.drive und der Server gleichzeitig laufen (am besten dieselbe Maschine).

## Installieren (einmalig)

1. Ordner aussuchen, in dem du BeamNG.Earth haben willst, im Explorer in die Adresszeile `cmd` tippen und Enter.
2. Diese Zeile einfügen:

   ```
   git clone https://github.com/YT-TheBlackCat/beamng-earth-releases.git BeamNG.Earth
   ```

   Das Release-Repo ist öffentlich, du brauchst keinen GitHub-Login.
3. In den neuen Ordner `BeamNG.Earth` gehen und **`run_server.bat` doppelklicken**. Beim ersten Mal richtet es sich
   selbst ein (Python-Umgebung, Mod in den BeamNG-Ordner kopieren). Das dauert ein paar Minuten.

Wichtig: **nicht** als ZIP herunterladen. Nur ein `git clone` kann sich selbst aktualisieren.

## Spielen

1. `run_server.bat` doppelklicken und das Fenster **offen lassen**, solange du spielst.
2. BeamNG.drive starten, im Freeroam das Level **BeamNG.Earth** wählen.
3. Mit **M** die Weltkarte öffnen, "Leonberg" suchen und hinteleportieren (Suche und Teleport sind in der
   Weltkarte). Gute Stellen zum Ansehen: Marktplatz, Schloss, Stadtkirche, Rathaus, Ramtel.

## Updates

Ganz automatisch: `run_server.bat` holt vor jedem Start die neueste Version von GitHub (`update_release.py`) und
schreibt oben ins Fenster, was neu ist. Es wird nur auf die neueste Version vorgespult (kein Zurücksetzen, kein Überschreiben
deiner Änderungen); Zwischenspeicher, Daten und die Python-Umgebung bleiben unberührt. Der Mod im Spiel und die Python-Pakete
werden dabei vom Start gleich mit aktualisiert.

- Kein Internet oder GitHub nicht erreichbar: es startet die Version, die du hast.
- Hast du selbst Dateien im Ordner geändert und das Update kommt dagegen nicht an: es steht im Fenster, die Version
  startet trotzdem. Dann entweder die Änderung rückgängig machen (`git checkout -- <Datei>`) oder den Ordner neu klonen.
- Update einmal auslassen: in der Konsole `set BNE_NO_UPDATE=1`, dann `run_server.bat`.

## Rasen und Gärten (optional)

Gärten und Rasen aus Luftbildern müssen **einmal** vorberechnet werden (lädt ca. 1,6 GB Daten vom Land
Baden-Württemberg, dauert eine Weile):

```
.venv\Scripts\python tools\bake_gardens.py --area leonberg
```

Ohne das fehlen nur die Gärten, alles andere funktioniert.

## Probleme?

Fenster-Ausgabe oder den Server-Log (Pfad steht beim Start im Fenster) an Lasse schicken.
Der Server meldet anonym die Spielerzahl an den Server-Dienst des Originalprojekts (`earth.tgforever.net`, kein
Standort); abschalten mit `set BNE_TELEMETRY=off` vor dem Start.

## Daten und Lizenzen

Der Code steht unter der MIT-Lizenz (`LICENSE`). Karten-, Höhen- und Gebäudedaten kommen aus offenen Quellen
(u. a. OpenStreetMap, LGL Baden-Württemberg unter dl-de/by-2-0); die vollständige Liste mit den geforderten
Quellenangaben steht in `CREDITS.md`.
