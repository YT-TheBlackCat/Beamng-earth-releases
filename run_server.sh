#!/usr/bin/env bash
# BeamNG.Earth: start the world server with the recommended options.
# Run it, then start BeamNG.drive. Keep this terminal open while you play.
#
# In a git clone of the release repo (see FREUNDE.md) it first fetches the newest stable version
# (update_release.py: fast-forward only, never deletes anything, never stops the start).
# Skip that for one start: BNE_NO_UPDATE=1 ./run_server.sh
cd "$(dirname "$0")" || exit 1
# (the update may replace this very file: bash reads a script as it runs, so everything after this line is one
# block that is parsed completely first)
{
    PYX=python3
    [ -x .venv/bin/python ] && PYX=.venv/bin/python
    if [ -f update_release.py ]; then "$PYX" update_release.py || true; fi
    if [ ! -x .venv/bin/python ]; then
        echo "The server is not installed yet. Running the installer first..."
        python3 install.py || exit 1
    fi
    exec .venv/bin/python start.py "$@"
}
