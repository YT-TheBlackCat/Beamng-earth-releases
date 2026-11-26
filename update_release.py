#!/usr/bin/env python3
"""Update this folder to the newest stable version (the public release repo) before the server starts.

Called by run_server.bat / run_server.sh. It never stops the start: whatever goes wrong (no git, no network, local
changes, a folder that is not a git clone), it says so in one line and the installed version starts as it is.

What it does, only when this folder is a git clone of a release build (it holds RELEASE.json, which tools/publish_release.py
puts into every release) and sits on the branch `main`:
    git fetch origin main              (60 s limit, no password prompt in the console)
    git merge --ff-only FETCH_HEAD     (only a fast-forward: nothing is overwritten, nothing is deleted)
It does not touch untracked or ignored files (.venv, caches, baked garden data, logs) and never resets, stashes or
force-pulls. A developer checkout (no RELEASE.json, or any other branch) is left alone.

Switch it off for one start with the environment variable BNE_NO_UPDATE=1 (or the option --no-update).
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
BRANCH = "main"
MARKER = "RELEASE.json"      # only in release builds (tools/publish_release.py), never in the development repo
REMOTE = "origin"
FETCH_TIMEOUT_S = 60


def say(msg):
    print("[update] " + msg, flush=True)


def git(*args, timeout=30):
    """Run git in ROOT; (returncode, stdout stripped). Never raises (a missing git or a timeout is returncode -1)."""
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"          # (no password question in a console that nobody can answer)
    env["LC_ALL"] = "C"
    try:
        r = subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True, errors="replace",
                           timeout=timeout, env=env)
    except (OSError, subprocess.SubprocessError):
        return -1, ""
    return r.returncode, (r.stdout or "").strip()


def update():
    """Returns a short status word (for the tests): 'skipped', 'no-git', 'offline', 'current', 'updated', 'diverged',
    'blocked'."""
    if os.environ.get("BNE_NO_UPDATE", "").lower() in ("1", "on", "true", "yes") or "--no-update" in sys.argv[1:]:
        say("update switched off (BNE_NO_UPDATE / --no-update)")
        return "skipped"
    if not os.path.exists(os.path.join(ROOT, ".git")):
        say("this folder is not a git clone: no automatic update (see FREUNDE.md for the clone command)")
        return "no-git"
    rc, branch = git("rev-parse", "--abbrev-ref", "HEAD")
    if rc == -1:
        say("git is not installed (https://git-scm.com/download/win): no automatic update, starting the installed version")
        return "no-git"
    if rc != 0:
        say("git cannot read this folder: no automatic update")
        return "no-git"
    if not os.path.exists(os.path.join(ROOT, MARKER)):
        say("this is a developer checkout (no RELEASE.json): no automatic update")
        return "skipped"
    if branch != BRANCH:
        say(f"branch '{branch}' is not '{BRANCH}': no automatic update")
        return "skipped"

    rc, old = git("rev-parse", "HEAD")
    say(f"looking for a new version (branch {BRANCH}) ...")
    rc, _ = git("fetch", "--quiet", REMOTE, BRANCH, timeout=FETCH_TIMEOUT_S)
    if rc != 0:
        say("no connection to GitHub (or no access): starting the installed version")
        return "offline"
    rc, new = git("rev-parse", "FETCH_HEAD")
    if rc != 0 or not new:
        say("the fetch gave nothing usable: starting the installed version")
        return "offline"
    if new == old:
        say("already the newest version")
        return "current"
    if git("merge-base", "--is-ancestor", new, old)[0] == 0:
        say("this folder is ahead of GitHub: nothing to do")
        return "current"
    if git("merge-base", "--is-ancestor", old, new)[0] != 0:
        say("this folder has its own commits that GitHub does not have: not updating (ask Lasse; "
            "a fresh clone fixes it)")
        return "diverged"
    rc, out = git("merge", "--ff-only", "--quiet", new, timeout=120)
    if rc != 0:
        say("could not update (files in this folder were changed by hand and are in the way?): "
            "starting the installed version. Details: git status")
        return "blocked"
    say(f"updated {old[:7]} -> {new[:7]}. What is new:")
    rc, log = git("log", "--no-merges", "--format=  - %s", f"{old}..{new}")
    if rc == 0 and log:
        lines = log.splitlines()
        for line in lines[:12]:
            say(line)
        if len(lines) > 12:
            say(f"  ... and {len(lines) - 12} more")
    return "updated"


def main():
    try:
        update()
    except Exception as e:                      # (an update must never stop the server from starting)
        say(f"update failed ({type(e).__name__}: {e}): starting the installed version")
    return 0


if __name__ == "__main__":
    sys.exit(main())
