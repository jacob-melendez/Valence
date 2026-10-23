"""scripts/hack_log.py: the Hack OHI/O log lists every commit since the baseline, by track, flagging the window."""

import importlib.util
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("hack_log", Path(__file__).resolve().parents[1] / "scripts" / "hack_log.py")
hack_log = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = hack_log  # dataclasses resolve annotations through it
SPEC.loader.exec_module(hack_log)


def _git(repo, *args, date=None):
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@example.com", "HOME": str(repo), "PATH": "/usr/bin:/bin"}
    if date:
        env.update(GIT_AUTHOR_DATE=date, GIT_COMMITTER_DATE=date)
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, env=env)


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / "a.py").write_text("x = 1\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "before the event", date="2026-10-23T20:00:00-04:00")
    _git(tmp_path, "tag", hack_log.BASE_TAG)
    (tmp_path / "web.ts").write_text("export const a = 1;\nexport const b = 2;\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "[web] Port the feature function", date="2026-10-24T13:00:00-04:00")
    (tmp_path / "a.py").write_text("x = 2\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "[emotion] Valence-arousal head", date="2026-10-25T14:00:00-04:00")
    return tmp_path


def test_commits_since_baseline_with_files_and_tracks(repo):
    commits = hack_log.commits_since(repo, hack_log.BASE_TAG)
    assert [c.track for c in commits] == ["web", "emotion"]
    assert commits[0].files == [("web.ts", 2, 0)] and commits[0].title == "Port the feature function"
    assert (commits[1].added, commits[1].deleted) == (1, 1)


def test_render_flags_commits_outside_the_window_and_writes_section(repo, tmp_path):
    log = tmp_path / "LOG.md"
    log.write_text(f"# Log\n\n{hack_log.START_MARK}\nold\n{hack_log.END_MARK}\n\n## Notes\n")
    rc = hack_log.main(["--repo", str(repo), "--log", str(log), "--write",
                        "--start", "2026-10-24T12:00:00-04:00", "--end", "2026-10-25T12:00:00-04:00"])
    text = log.read_text()
    assert rc == 0 and "old" not in text and text.endswith("## Notes\n")
    assert "2 commits since" in text and "1 commit(s) outside the window" in text
    assert "| Valence Web | 1 |" in text and "Valence-arousal head" in text
    body = hack_log.render(hack_log.commits_since(repo, hack_log.BASE_TAG), base="b", base_sha="0" * 40,
                           tz=hack_log.ZoneInfo("America/New_York"), start=datetime.fromisoformat("2026-10-24T12:00:00-04:00"),
                           end=None, shortstat="")
    assert "⚠" not in body  # no end set: nothing is outside


def test_unknown_baseline_exits_2(repo, capsys):
    assert hack_log.main(["--repo", str(repo), "--base", "nope"]) == 2
