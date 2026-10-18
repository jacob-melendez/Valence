"""scripts/audit_verify.py: a genuine audit file verifies; edits, deletions and reordering are caught."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from valence.governance.audit import AuditLog

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_verify.py"
spec = importlib.util.spec_from_file_location("audit_verify", SCRIPT)
av = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = av  # dataclasses resolve annotations through it
spec.loader.exec_module(av)


@pytest.fixture
def log(tmp_path):
    log = AuditLog(tmp_path / "audit", "session-test")
    log.append("operator", {"action": "replay", "scenario": "full_demo", "speed": 20.0})
    for i, label in enumerate(["Focused", "Frustrated_Masked", "Frustrated_Masked", "Collaborative_Happy"]):
        log.append("prediction", {"person": "Person #1", "label": label, "confidence": 0.7 + i / 20,
                                  "mask_override": i == 2, "layers": {"Ambient Physics": 0.1}})
    log.append("operator", {"action": "demo_seed", "simulated_fields": ["hours_since_break"]})
    assert log.verify() == (True, 6)
    return log


def lines(log):
    return log.path.read_text(encoding="utf-8").splitlines()


def write(log, rows):
    log.path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_genuine_file_and_directory_pass(log, capsys):
    v = av.verify_file(log.path)
    assert v.ok and v.entries == 6 and v.head == log.last_hash
    assert av.main([str(log.path)]) == 0
    assert av.main([str(log.path.parent)]) == 0
    out = capsys.readouterr().out
    assert "6 entries checked, chain intact" in out
    assert av.main([str(log.path), "--head", log.last_hash[:16]]) == 0


def edit_field(rows):
    row = json.loads(rows[2])
    row["payload"]["confidence"] = 0.99
    rows[2] = json.dumps(row, sort_keys=True)
    return rows


def edit_and_rehash_alone(rows):
    """Recompute the edited entry's own hash: the next entry's prev no longer matches."""
    from valence.governance.audit import _digest

    row = json.loads(rows[2])
    row["payload"]["label"] = "Focused"
    row["hash"] = _digest(row["prev"], {k: row[k] for k in ("seq", "ts", "kind", "payload")})
    rows[2] = json.dumps(row, sort_keys=True)
    return rows


@pytest.mark.parametrize("tamper, bad_line, reason", [
    (edit_field, 3, "hash mismatch"),
    (edit_and_rehash_alone, 4, "prev does not match"),
    (lambda r: r[:2] + r[3:], 3, "prev does not match"),  # delete a middle line
    (lambda r: r[1:], 1, "prev does not match"),  # delete the first line
    (lambda r: [r[0], r[2], r[1], *r[3:]], 2, "prev does not match"),  # swap two entries
    (lambda r: r[:3] + ["{not json"] + r[3:], 4, "not valid JSON"),
    (lambda r: r[:3] + ['{"seq": 3}'] + r[3:], 4, "missing field"),
])
def test_tampering_is_caught(log, capsys, tamper, bad_line, reason):
    write(log, tamper(lines(log)))
    v = av.verify_file(log.path)
    assert not v.ok and v.line == bad_line and reason in v.reason
    assert v.entries == bad_line - 1
    assert av.main([str(log.path)]) == 1
    assert f"TAMPERED {log.path}: line {bad_line}" in capsys.readouterr().out
    if "JSON" not in reason and "missing" not in reason:
        assert log.verify()[0] is False  # agrees with the module's own verify()


def test_truncated_tail_needs_head(log):
    head = log.last_hash
    write(log, lines(log)[:-2])
    assert av.main([str(log.path)]) == 0  # a shorter chain is still a valid chain
    assert av.main([str(log.path), "--head", head]) == 1
    assert av.main([str(log.path), "--head", head[:16]]) == 1


def test_one_bad_file_in_a_directory_fails(log, tmp_path):
    other = AuditLog(log.path.parent, "session-other")
    other.append("operator", {"action": "live"})
    assert av.main([str(log.path.parent)]) == 0
    write(log, edit_field(lines(log)))
    assert av.main([str(log.path.parent)]) == 1


def test_usage_errors_exit_2(log, tmp_path):
    assert av.main([str(tmp_path / "missing.jsonl")]) == 2
    (tmp_path / "empty").mkdir()
    assert av.main([str(tmp_path / "empty")]) == 2
    assert av.main([str(log.path), "--head", "xyz"]) == 2
    assert av.main([str(log.path), str(log.path), "--head", log.last_hash]) == 2


def test_cli_exit_codes(log):
    def run(*args):
        return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True).returncode

    assert run(str(log.path)) == 0
    write(log, edit_field(lines(log)))
    assert run(str(log.path)) == 1
    assert run(str(log.path.parent / "nope.jsonl")) == 2


def test_verifier_never_modifies_the_audit_dir(log):
    """Unlike AuditLog(...), verifying does not expire old files or create anything."""
    import os
    import time

    old = log.path.parent / "old-session.jsonl"
    old.write_text("")
    past = time.time() - 10 * 86_400
    os.utime(old, (past, past))
    before = sorted(p.name for p in log.path.parent.iterdir())
    av.main([str(log.path.parent)])
    assert sorted(p.name for p in log.path.parent.iterdir()) == before
