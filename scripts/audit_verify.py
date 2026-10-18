"""Verify the hash chain of Valence audit files offline and report the first tampered entry.

    python scripts/audit_verify.py var/audit/session-20261018-101500.jsonl
    python scripts/audit_verify.py var/audit                 # every *.jsonl in the directory
    python scripts/audit_verify.py FILE --head 9f2c4e1a7b3d5c60   # also catch a truncated tail

With no path, ``$VALENCE_AUDIT_DIR`` (default ``var/audit``) is checked. The chain is the one in
``valence.governance.audit``: each entry's ``hash`` is SHA-256 of ``prev`` + the canonical JSON of
``{seq, ts, kind, payload}``, and ``prev`` is the previous entry's hash (64 zeros first). An edited,
deleted, inserted or reordered entry breaks it. Deleting entries from the *end* leaves a valid shorter
chain, so pass ``--head`` with the latest ``audit_hash`` (the ``governance`` message's 16-character prefix
or the full hash from ``GET /api/governance/stats``) to pin the last entry.

Exit code: 0 every file intact, 1 tampering found, 2 usage error or missing file. Read-only: unlike
``AuditLog(...)``, it never creates the directory or expires old files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))  # `python scripts/audit_verify.py` without an editable install

from valence.governance.audit import GENESIS, _digest  # noqa: E402  the chain's own hash

FIELDS = ("seq", "ts", "kind", "payload", "prev", "hash")


@dataclass
class Verdict:
    path: Path
    ok: bool
    entries: int  # entries verified (all of them when ok)
    line: int | None = None  # 1-based line of the first bad entry
    reason: str = ""
    head: str = GENESIS


def verify_file(path: Path, head: str | None = None) -> Verdict:
    """Recompute the chain over ``path``, like ``AuditLog.verify()``, but say where and why it breaks."""
    prev = GENESIS
    n = 0
    with Path(path).open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue  # AuditLog.entries() skips blank lines too
            try:
                entry = json.loads(line)
            except ValueError:
                return Verdict(path, False, n, lineno, "not valid JSON", prev)
            missing = [k for k in FIELDS if not isinstance(entry, dict) or k not in entry]
            if missing:
                return Verdict(path, False, n, lineno, f"missing field(s): {', '.join(missing)}", prev)
            body = {k: entry[k] for k in ("seq", "ts", "kind", "payload")}
            where = f"seq {entry['seq']}"
            if entry["prev"] != prev:
                return Verdict(path, False, n, lineno, f"{where}: prev does not match the previous entry's hash "
                               "(an entry was deleted, inserted or reordered)", prev)
            if _digest(prev, body) != entry["hash"]:
                return Verdict(path, False, n, lineno, f"{where}: hash mismatch (the entry was edited)", prev)
            if entry["seq"] != n:
                return Verdict(path, False, n, lineno, f"{where}: expected seq {n} (entries missing before it)", prev)
            prev = entry["hash"]
            n += 1
    if head is not None and not prev.startswith(head.lower()):
        return Verdict(path, False, n, None, f"last hash {prev[:16]} does not match --head {head[:16]} "
                       "(entries removed from the end, or a different session)", prev)
    return Verdict(path, True, n, head=prev)


def collect(paths: list[str]) -> tuple[list[Path], list[str]]:
    """Expand directories to their *.jsonl files. Returns (files, errors)."""
    files: list[Path] = []
    errors: list[str] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            found = sorted(p.glob("*.jsonl"))
            if not found:
                errors.append(f"{p}: no audit files (*.jsonl) in this directory")
            files.extend(found)
        elif p.is_file():
            files.append(p)
        else:
            errors.append(f"{p}: no such file or directory")
    return files, errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="*", help="audit file(s) or directories (default: $VALENCE_AUDIT_DIR or var/audit)")
    ap.add_argument("--head", help="expected hash (or 16+ char prefix) of the last entry; needs a single file")
    args = ap.parse_args(argv)
    paths = args.paths or [os.environ.get("VALENCE_AUDIT_DIR", "var/audit")]
    files, errors = collect(paths)
    if args.head is not None:
        head = args.head.strip().lower()
        if len(head) < 16 or any(c not in "0123456789abcdef" for c in head):
            errors.append("--head must be at least 16 hex characters")
        elif len(files) != 1:
            errors.append("--head needs exactly one audit file")
    for e in errors:
        print(f"ERROR {e}", file=sys.stderr)
    if errors:
        return 2

    tampered = 0
    for path in files:
        try:
            v = verify_file(path, args.head)
        except (OSError, UnicodeDecodeError) as exc:
            print(f"ERROR {path}: {exc}", file=sys.stderr)
            return 2
        if v.ok:
            print(f"OK       {path}: {v.entries} entries checked, chain intact (head {v.head[:16]})")
        else:
            tampered += 1
            at = f"line {v.line}" if v.line else "end of file"
            print(f"TAMPERED {path}: {at}: {v.reason}; {v.entries} entries verified before it")
    print(f"{len(files) - tampered}/{len(files)} audit file(s) intact")
    return 1 if tampered else 0


if __name__ == "__main__":
    sys.exit(main())
