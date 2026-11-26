@echo off
rem BeamNG.Earth: start the world server with the recommended options.
rem Double-click this file, then start BeamNG.drive. Keep this window open while you play.
rem
rem First start: finds a suitable Python (3.10 to 3.13; 3.12 is what the packages are built for) and installs
rem Python 3.12 for you if there is none (winget, else the python.org installer; no admin rights needed).
rem In a git clone of the release repo (see FREUNDE.md) it then fetches the newest stable version
rem (update_release.py: fast-forward only, never deletes anything, never stops the start).
rem Skip that for one start:  set BNE_NO_UPDATE=1  (in the console, then run this file)
rem
rem The whole file is one block on purpose: cmd reads a block completely before it runs, so the update may
rem replace this very file while it runs without breaking the rest of the start.
setlocal EnableDelayedExpansion
cd /d "%~dp0"
(
set "PYX="
set "PYCHK=import sys;sys.exit(0 if sys.version_info[0]==3 and 10<=sys.version_info[1]<=13 else 1)"
set "PY312=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if exist ".venv\Scripts\python.exe" set "PYX=.venv\Scripts\python.exe"
if not defined PYX for %%V in (3.12 3.13 3.11 3.10) do if not defined PYX (
  py -%%V -c "!PYCHK!" >nul 2>nul && set "PYX=py -%%V"
)
if not defined PYX (
  python -c "!PYCHK!" >nul 2>nul && set "PYX=python"
)
if not defined PYX if exist "!PY312!" set "PYX="!PY312!""
if not defined PYX (
  echo No suitable Python found ^(needed: 3.10 to 3.13; a newer one such as 3.14 has no DuckDB package yet^).
  echo Installing Python 3.12 for you ^(about 25 MB, no admin rights needed^) ...
  where winget >nul 2>nul && winget install -e --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements
  if not exist "!PY312!" (
    echo winget did not work; downloading the installer from python.org ...
    curl.exe -L -o "%TEMP%\python-3.12-installer.exe" https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe
    if exist "%TEMP%\python-3.12-installer.exe" "%TEMP%\python-3.12-installer.exe" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=0 Include_test=0
  )
  if exist "!PY312!" set "PYX="!PY312!""
)
if not defined PYX (
  echo.
  echo Python could not be installed automatically. Please install Python 3.12 by hand:
  echo   https://www.python.org/downloads/release/python-31210/   ^(Windows installer 64-bit^)
  echo   tick "Add python.exe to PATH" at the start, then double-click run_server.bat again.
  echo   ^(If Windows opens the Microsoft Store for "python": Settings, Apps, Advanced app settings,
  echo    App execution aliases, switch off the python.exe entries.^)
  pause
  exit /b 1
)
if exist update_release.py !PYX! update_release.py
if not exist ".venv\Scripts\python.exe" (
  echo The server is not installed yet. Running the installer first...
  !PYX! install.py
)
if not exist ".venv\Scripts\python.exe" (
  echo.
  echo The installer did not finish. Read the messages above; the usual fix is in FREUNDE.md ^(Problems^).
  pause
  exit /b 1
)
".venv\Scripts\python.exe" start.py %*
echo.
echo The server has stopped.
pause
exit /b
)
