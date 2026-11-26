@echo off
rem BeamNG.Earth: start the world server with the recommended options.
rem Double-click this file, then start BeamNG.drive. Keep this window open while you play.
rem
rem In a git clone of the release repo (see FREUNDE.md) it first fetches the newest stable version
rem (update_release.py: fast-forward only, never deletes anything, never stops the start).
rem Skip that for one start:  set BNE_NO_UPDATE=1  (in the console, then run this file)
rem
rem The whole file is one block on purpose: cmd reads a block completely before it runs, so the update may
rem replace this very file while it runs without breaking the rest of the start.
setlocal EnableDelayedExpansion
cd /d "%~dp0"
(
set "PYX="
if exist ".venv\Scripts\python.exe" set "PYX=.venv\Scripts\python.exe"
if not defined PYX where py >nul 2>nul && set "PYX=py -3"
if not defined PYX where python >nul 2>nul && set "PYX=python"
if defined PYX if exist update_release.py !PYX! update_release.py
if not exist ".venv\Scripts\python.exe" (
  echo The server is not installed yet. Running the installer first...
  if defined PYX (
    !PYX! install.py
  )
)
if not exist ".venv\Scripts\python.exe" (
  echo.
  echo The installer did not finish. Install Python 3.10+ from https://www.python.org/downloads/
  echo ^(tick "Add python.exe to PATH"^), then double-click run_server.bat again.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" start.py %*
echo.
echo The server has stopped.
pause
exit /b
)
