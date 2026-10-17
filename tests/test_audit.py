import json
import os
import time

import numpy as np
import pytest

from valence.governance.audit import AuditLog, AuditPayloadError


def test_chain_verifies_and_detects_tampering(tmp_path):
    log = AuditLog(tmp_path, "s1")
    log.append("prediction", {"person": "Person #1", "label": "Focused", "confidence": 0.81})
    log.append("operator", {"action": "demo_seed", "simulated_fields": ["hours_since_break"]})
    assert log.verify() == (True, 2)
    lines = log.path.read_text().splitlines()
    row = json.loads(lines[0])
    row["payload"]["confidence"] = 0.99
    lines[0] = json.dumps(row, sort_keys=True)
    log.path.write_text("\n".join(lines) + "\n")
    ok, _ = log.verify()
    assert not ok


@pytest.mark.parametrize(
    "payload",
    [
        {"frame": np.zeros(3)},
        {"blob": b"\x00"},
        {"transcript": "x" * 500},
        {"v": float("nan")},
    ],
)
def test_raw_or_free_text_payloads_refused(tmp_path, payload):
    log = AuditLog(tmp_path, "s2")
    with pytest.raises(AuditPayloadError):
        log.append("bad", payload)


def test_session_file_deleted_on_close_and_ttl_expiry(tmp_path):
    old = tmp_path / "old-session.jsonl"
    old.write_text("{}\n")
    past = time.time() - 10_000
    os.utime(old, (past, past))
    log = AuditLog(tmp_path, "s3", ttl_s=3600)
    assert not old.exists()
    log.append("prediction", {"label": "Exhausted"})
    assert log.path.exists()
    log.close(delete=True)
    assert not log.path.exists()
