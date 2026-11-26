"""update_release.py: the fast-forward update of a release clone (branch main + RELEASE.json) before the server starts
(real git, temp repos)."""
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import update_release  # noqa: E402


def run(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                   env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
                        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"})


class UpdateReleaseTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        t = self.tmp.name
        self.origin = os.path.join(t, "origin.git")
        self.work = os.path.join(t, "work")       # the maintainer's clone that publishes
        self.friend = os.path.join(t, "friend")   # a friend's clone
        run("init", "--bare", "-b", "main", self.origin, cwd=t)
        run("clone", self.origin, self.work, cwd=t)
        run("checkout", "-b", "main", cwd=self.work)
        Path(self.work, "RELEASE.json").write_text("{}")         # the marker of a release build
        run("add", "RELEASE.json", cwd=self.work)
        self.publish("a.txt", "one", "first")
        run("clone", "--branch", "main", self.origin, self.friend, cwd=t)

    def publish(self, name, text, msg):
        Path(self.work, name).write_text(text)
        run("add", name, cwd=self.work)
        run("commit", "-m", msg, cwd=self.work)
        run("push", "origin", "main", cwd=self.work)

    def update(self, **env):
        out = io.StringIO()
        with mock.patch.object(update_release, "ROOT", self.friend), \
                mock.patch.dict(os.environ, env), \
                mock.patch.object(sys, "argv", ["update_release.py"]), \
                contextlib.redirect_stdout(out):
            return update_release.update(), out.getvalue()

    def test_current_then_updated(self):
        self.assertEqual(self.update()[0], "current")
        self.publish("a.txt", "two", "second change")
        status, out = self.update()
        self.assertEqual(status, "updated")
        self.assertIn("second change", out)
        self.assertEqual(Path(self.friend, "a.txt").read_text(), "two")
        self.assertEqual(self.update()[0], "current")

    def test_untracked_and_ignored_files_survive(self):
        Path(self.friend, "bake.dat").write_text("keep me")
        self.publish("b.txt", "x", "new file")
        self.assertEqual(self.update()[0], "updated")
        self.assertEqual(Path(self.friend, "bake.dat").read_text(), "keep me")

    def test_local_change_in_the_way_is_not_overwritten(self):
        Path(self.friend, "a.txt").write_text("my edit")
        self.publish("a.txt", "two", "upstream edit")
        self.assertEqual(self.update()[0], "blocked")
        self.assertEqual(Path(self.friend, "a.txt").read_text(), "my edit")

    def test_local_change_elsewhere_is_fine(self):
        Path(self.friend, "a.txt").write_text("my edit")
        self.publish("b.txt", "x", "other file")
        self.assertEqual(self.update()[0], "updated")
        self.assertEqual(Path(self.friend, "a.txt").read_text(), "my edit")

    def test_own_commits_are_not_touched(self):
        Path(self.friend, "own.txt").write_text("mine")
        run("add", "own.txt", cwd=self.friend)
        run("commit", "-m", "mine", cwd=self.friend)
        self.assertEqual(self.update()[0], "current")          # ahead of GitHub
        self.publish("b.txt", "x", "upstream")
        self.assertEqual(self.update()[0], "diverged")
        self.assertTrue(Path(self.friend, "own.txt").exists())

    def test_no_network_keeps_going(self):
        run("remote", "set-url", "origin", os.path.join(self.tmp.name, "does-not-exist.git"), cwd=self.friend)
        self.assertEqual(self.update()[0], "offline")

    def test_other_branch_is_left_alone(self):
        run("checkout", "-b", "feature", cwd=self.friend)
        self.publish("b.txt", "x", "upstream")
        self.assertEqual(self.update()[0], "skipped")
        self.assertFalse(Path(self.friend, "b.txt").exists())

    def test_developer_checkout_on_main_is_left_alone(self):
        os.remove(os.path.join(self.friend, "RELEASE.json"))        # (the development repo has no marker)
        run("commit", "-qam", "dev", cwd=self.friend)
        self.publish("b.txt", "x", "upstream")
        self.assertEqual(self.update()[0], "skipped")
        self.assertFalse(Path(self.friend, "b.txt").exists())

    def test_switch_off_and_no_git(self):
        self.publish("b.txt", "x", "upstream")
        self.assertEqual(self.update(BNE_NO_UPDATE="1")[0], "skipped")
        self.assertFalse(Path(self.friend, "b.txt").exists())
        with tempfile.TemporaryDirectory() as plain, mock.patch.object(update_release, "ROOT", plain), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(update_release.update(), "no-git")

    def test_main_never_raises(self):
        with mock.patch.object(update_release, "update", side_effect=RuntimeError("boom")), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(update_release.main(), 0)


if __name__ == "__main__":
    unittest.main()
