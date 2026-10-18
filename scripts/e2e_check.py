"""Pre-demo end-to-end check: replay ``full_demo`` and check the live stream against the contract.

    python scripts/e2e_check.py --start                     # own replay server on a free port
    python scripts/e2e_check.py --api http://127.0.0.1:8000  # a server that is already running

It posts ``/api/replay/full_demo`` (``{"speed": 20, "loop": false}``, docs/API.md §7.4.4), reads
``/ws/events`` until the masked segment has been seen (or ``--seconds`` runs out), and checks:

- every message validates against ``valence.contracts.json_schemas()`` (strict, additionalProperties false);
- predictions arrive;
- every ``explanation`` pairs with a ``prediction`` event_id seen earlier on this connection;
- ``mask_override`` is true at least once (the masked segment of full_demo);
- governance ``violations`` stay 0 and ``purge_gap`` stays <= 2 throughout (and no ``status.purge_violation``);
- ``GET /api/governance/verify`` is ok;
- ``GET /api/health`` is ok and reports ``room_baseline``.

Prints a PASS/FAIL table; exit code 0 when every check passes, 1 otherwise. With ``--start`` the server
runs with its audit dir in a temporary directory that is removed afterwards, and the server is always
terminated. Only message counters are kept; nothing is written to disk.

Note: against a running server (``--api``) this restarts its replay; afterwards the runtime is paused at
the end of full_demo (``loop: false``).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))  # `python scripts/e2e_check.py` without an editable install

from valence.contracts import WS_MESSAGES, json_schemas  # noqa: E402

MAX_PURGE_GAP = 2
MIN_GOVERNANCE = 5  # governance messages (ticks) to see before stopping early
GRACE_S = 3.5  # an explanation this soon after connecting may answer a prediction sent before we connected
SCENARIO_DIR = REPO / "scenarios"
MASKED_STATES = {"Frustrated_Masked", "Tense"}


@dataclass
class Result:
    name: str
    ok: bool
    detail: str


# --- schemas -------------------------------------------------------------------------------------------
Validator = Callable[[dict[str, Any]], "str | None"]


def schema_by_type() -> dict[str, dict[str, Any]]:
    """WebSocket ``type`` value -> its JSON Schema from ``json_schemas()`` (keyed by model class name)."""
    schemas = json_schemas()
    return {m.model_fields["type"].default: schemas[m.__name__] for m in WS_MESSAGES}


def make_validators() -> dict[str, Validator]:
    """One validator per message type: returns None when valid, else a short error string."""
    out: dict[str, Validator] = {}
    try:
        from jsonschema import Draft202012Validator
    except ImportError:  # pydantic models are the source of the schemas, so they validate identically
        for model in WS_MESSAGES:
            def check(msg, model=model):
                try:
                    model.model_validate(msg)
                    return None
                except Exception as exc:  # pydantic.ValidationError
                    return str(exc).splitlines()[0][:160]
            out[model.model_fields["type"].default] = check
        return out
    for kind, schema in schema_by_type().items():
        v = Draft202012Validator(schema)

        def check(msg, v=v):
            err = next(iter(v.iter_errors(msg)), None)
            if err is None:
                return None
            where = "/".join(str(p) for p in err.absolute_path) or "<root>"
            return f"{where}: {err.message}"[:160]
        out[kind] = check
    return out


# --- scenario timing -----------------------------------------------------------------------------------
def masked_window_min(scenario: str = "full_demo", directory: Path = SCENARIO_DIR) -> tuple[float, float] | None:
    """(start, end) in scenario minutes of the first masked segment, or None if the scenario has none."""
    import yaml

    data = yaml.safe_load((directory / f"{scenario}.yaml").read_text(encoding="utf-8"))
    t = 0.0
    for seg in data["segments"]:
        minutes = float(seg["minutes"])
        states = seg.get("states", seg.get("state"))
        states = set(states) if isinstance(states, list) else {states}
        if states & MASKED_STATES or "mask_p" in (seg.get("overrides") or {}):
            return t, t + minutes
        t += minutes
    return None


def default_seconds(scenario: str, speed: float, margin_s: float = 20.0) -> float:
    """Wall seconds needed to get past the masked segment at ``speed``, plus a margin."""
    win = masked_window_min(scenario)
    end_min = win[1] if win else sum(float(s["minutes"]) for s in _segments(scenario))
    return end_min * 60.0 / speed + margin_s


def _segments(scenario: str) -> list[dict[str, Any]]:
    import yaml

    return yaml.safe_load((SCENARIO_DIR / f"{scenario}.yaml").read_text(encoding="utf-8"))["segments"]


# --- the checking logic (pure: fed messages, no I/O) ---------------------------------------------------
@dataclass
class Checker:
    validators: dict[str, Validator]
    grace_s: float = GRACE_S
    masked_window_s: tuple[float, float] | None = None  # expected wall window of the masked segment
    n_messages: int = 0
    by_type: collections.Counter = field(default_factory=collections.Counter)
    schema_errors: int = 0
    first_schema_errors: list[str] = field(default_factory=list)
    first_type: str | None = None
    hello: dict[str, Any] | None = None
    predictions: dict[str, int] = field(default_factory=dict)  # event_id -> person_id
    explanations: int = 0
    paired: int = 0
    ignored_early: int = 0
    orphans: list[str] = field(default_factory=list)
    mask_overrides: int = 0
    first_override_t: float | None = None
    governance: int = 0
    max_violations: int = 0
    max_purge_gap: int = 0
    purge_violation_status: bool = False
    stream_error: str | None = None

    def feed_raw(self, raw: str | bytes, t: float = 0.0) -> None:
        try:
            msg = json.loads(raw)
        except (TypeError, ValueError):
            self._schema_error("<not JSON>", "message is not valid JSON")
            return
        self.feed(msg, t)

    def feed(self, msg: Any, t: float = 0.0) -> None:
        """One message; ``t`` is seconds since the replay started (negative before it)."""
        self.n_messages += 1
        kind = msg.get("type") if isinstance(msg, dict) else None
        if self.first_type is None:
            self.first_type = str(kind)
        self.by_type[str(kind)] += 1
        check = self.validators.get(kind) if isinstance(kind, str) else None
        if check is None:
            self._schema_error(str(kind), "no schema for this message type")
            return
        err = check(msg)
        if err:
            self._schema_error(kind, err)
            return
        if kind == "hello":
            self.hello = msg
        elif kind == "prediction":
            self.predictions[msg["event_id"]] = msg["person_id"]
            if msg["mask_override"]:
                self.mask_overrides += 1
                if self.first_override_t is None:
                    self.first_override_t = t
        elif kind == "explanation":
            self.explanations += 1
            eid = msg["event_id"]
            if eid in self.predictions and self.predictions[eid] == msg["person_id"]:
                self.paired += 1
            elif eid not in self.predictions and t < self.grace_s:
                self.ignored_early += 1  # may answer a prediction published before this client connected
            else:
                why = "unknown event_id" if eid not in self.predictions else "person_id differs"
                self.orphans.append(f"{eid} ({why})")
        elif kind == "governance":
            self.governance += 1
            self.max_violations = max(self.max_violations, msg["violations"])
            self.max_purge_gap = max(self.max_purge_gap, msg.get("purge_gap", 0))
        elif kind == "status":
            self.purge_violation_status |= bool(msg.get("purge_violation"))

    def _schema_error(self, kind: str, err: str) -> None:
        self.schema_errors += 1
        if len(self.first_schema_errors) < 3:
            self.first_schema_errors.append(f"{kind}: {err}")

    def satisfied(self) -> bool:
        """Everything the stream can prove has been seen, so reading can stop early."""
        return bool(self.predictions) and self.mask_overrides > 0 and self.governance >= MIN_GOVERNANCE

    def results(self) -> list[Result]:
        out: list[Result] = []
        if self.stream_error:
            out.append(Result("ws /ws/events", False, self.stream_error))
        elif self.first_type != "hello":
            out.append(Result("ws /ws/events", False, f"first message was {self.first_type!r}, expected 'hello'"))
        else:
            h = self.hello or {}
            out.append(Result("ws /ws/events", True,
                              f"hello first (mode={h.get('mode')}, {len(h.get('states', []))} states)"))
        counts = ", ".join(f"{k} {v}" for k, v in sorted(self.by_type.items()))
        if self.schema_errors:
            out.append(Result("ws schema", False, f"{self.schema_errors}/{self.n_messages} invalid; "
                              + " | ".join(self.first_schema_errors)))
        else:
            out.append(Result("ws schema", self.n_messages > 0, f"{self.n_messages} messages valid ({counts})"))
        out.append(Result("predictions", bool(self.predictions), f"{len(self.predictions)} predictions"))
        if self.orphans:
            out.append(Result("explanation pairing", False, f"{len(self.orphans)} unpaired: "
                              + ", ".join(self.orphans[:3])))
        elif self.explanations:
            extra = f", {self.ignored_early} right after connect ignored" if self.ignored_early else ""
            out.append(Result("explanation pairing", True,
                              f"{self.paired}/{self.explanations} explanations paired{extra}"))
        else:
            out.append(Result("explanation pairing", True, "0 explanations (narrator not answering: template text only)"))
        expect = ""
        if self.masked_window_s:
            expect = f"; masked segment +{self.masked_window_s[0]:.0f}..+{self.masked_window_s[1]:.0f} s"
        if self.mask_overrides:
            out.append(Result("mask_override", True, f"{self.mask_overrides} predictions, first at "
                              f"+{self.first_override_t:.1f} s{expect}"))
        else:
            out.append(Result("mask_override", False, f"never true in {len(self.predictions)} predictions{expect}"))
        gov_ok = (self.governance > 0 and self.max_violations == 0 and self.max_purge_gap <= MAX_PURGE_GAP
                  and not self.purge_violation_status)
        detail = (f"{self.governance} ticks, max violations {self.max_violations}, "
                  f"max purge_gap {self.max_purge_gap} (<= {MAX_PURGE_GAP})")
        if self.purge_violation_status:
            detail += ", status.purge_violation set"
        out.append(Result("governance", gov_ok, detail if self.governance else "no governance messages"))
        return out


def check_replay(code: int, body: Any, scenario: str) -> Result:
    ok = code == 200 and isinstance(body, dict) and body.get("ok") is True and body.get("scenario") == scenario
    return Result(f"POST /api/replay/{scenario}", ok, f"HTTP {code} {json.dumps(body)[:120]}")


def check_verify(code: int, body: Any) -> Result:
    ok = code == 200 and isinstance(body, dict) and body.get("ok") is True
    entries = body.get("entries") if isinstance(body, dict) else None
    detail = f"{entries} entries, chain intact" if ok else f"HTTP {code} {json.dumps(body)[:120]}"
    return Result("GET /api/governance/verify", ok, detail)


def check_health(code: int, body: Any) -> Result:
    if code != 200 or not isinstance(body, dict) or body.get("ok") is not True:
        return Result("GET /api/health", False, f"HTTP {code} {json.dumps(body)[:120]}")
    rb = body.get("room_baseline")
    if not isinstance(rb, dict):
        return Result("GET /api/health", False, "room_baseline missing")
    return Result("GET /api/health", True, f"mode={body.get('mode')} model={body.get('model')} "
                  f"room_baseline={rb.get('mode')}")


def render_table(results: list[Result]) -> str:
    rows = [("CHECK", "RESULT", "DETAIL")] + [(r.name, "PASS" if r.ok else "FAIL", r.detail) for r in results]
    w0 = max(len(r[0]) for r in rows)
    w1 = max(len(r[1]) for r in rows)
    lines = [f"{a:<{w0}}  {b:<{w1}}  {c}" for a, b, c in rows]
    lines.insert(1, f"{'-' * w0}  {'-' * w1}  {'-' * 6}")
    n_ok = sum(r.ok for r in results)
    verdict = "PASS" if results and n_ok == len(results) else "FAIL"
    lines.append(f"\ne2e: {verdict} ({n_ok}/{len(results)} checks passed)")
    return "\n".join(lines)


# --- I/O: server, REST and the stream ------------------------------------------------------------------
def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ReplayServer:
    """``uvicorn valence.api.app:app`` in replay mode on a free port, audit dir in a temp dir."""

    def __init__(self, speed: float) -> None:
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.audit_dir = tempfile.mkdtemp(prefix="valence-e2e-audit-")
        env = {**os.environ, "VALENCE_MODE": "replay", "VALENCE_SPEED": str(speed),
               "VALENCE_AUDIT_DIR": self.audit_dir}
        self.tail: collections.deque[str] = collections.deque(maxlen=40)  # server output stays in RAM
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "valence.api.app:app", "--host", "127.0.0.1",
             "--port", str(self.port), "--log-level", "warning"],
            cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self) -> None:
        for line in self.proc.stdout:  # keep the pipe empty so the server never blocks on output
            self.tail.append(line.rstrip())

    def wait_healthy(self, client, timeout_s: float = 90.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited with code {self.proc.returncode}")
            try:
                if client.get(f"{self.url}/api/health").status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(0.3)
        raise RuntimeError(f"no /api/health within {timeout_s:.0f} s")

    def close(self) -> None:
        try:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=5)
        finally:
            shutil.rmtree(self.audit_dir, ignore_errors=True)


def ws_connect(url: str):
    from websockets.sync.client import connect

    try:
        return connect(url, open_timeout=10, max_size=2**22, proxy=None)  # loopback: never via a proxy
    except TypeError:  # websockets < 15 has no proxy support (and so no proxy argument)
        return connect(url, open_timeout=10, max_size=2**22)


def read_stream(api: str, client, checker: Checker, *, scenario: str, speed: float, seconds: float) -> Result:
    """Connect, start the replay, feed messages to ``checker`` until satisfied or ``seconds`` pass."""
    ws_url = ("wss" + api[5:] if api.startswith("https") else "ws" + api[4:]) + "/ws/events"
    try:
        ws = ws_connect(ws_url)
    except Exception as exc:
        checker.stream_error = f"connect {ws_url}: {type(exc).__name__}: {exc}"
        return Result(f"POST /api/replay/{scenario}", False, "not attempted (no WebSocket)")
    with ws:
        t_connect = time.monotonic()
        r = client.post(f"{api}/api/replay/{scenario}", json={"speed": speed, "loop": False})
        t0 = time.monotonic()
        checker.grace_s = GRACE_S - (t0 - t_connect)  # grace counts from the connection, t from the replay
        try:
            body = r.json()
        except ValueError:
            body = r.text[:120]
        replay = check_replay(r.status_code, body, scenario)
        if not replay.ok:
            return replay
        while (left := seconds - (time.monotonic() - t0)) > 0:
            try:
                raw = ws.recv(timeout=min(left, 10.0))
            except TimeoutError:
                if left > 10.0:
                    checker.stream_error = "no message for 10 s"
                break
            except Exception as exc:  # ConnectionClosed
                checker.stream_error = f"stream closed: {type(exc).__name__}: {exc}"
                break
            checker.feed_raw(raw, time.monotonic() - t0)
            if checker.satisfied():
                break
    return replay


def run(api: str, *, scenario: str, speed: float, seconds: float) -> list[Result]:
    import httpx

    win = masked_window_min(scenario)
    checker = Checker(make_validators(), masked_window_s=(win[0] * 60 / speed, win[1] * 60 / speed) if win else None)
    with httpx.Client(timeout=10.0, trust_env=False) as client:  # loopback: ignore proxy settings
        replay = read_stream(api, client, checker, scenario=scenario, speed=speed, seconds=seconds)
        results = [replay, *checker.results()]
        for path, fn in (("/api/governance/verify", check_verify), ("/api/health", check_health)):
            try:
                r = client.get(f"{api}{path}")
                try:
                    body = r.json()
                except ValueError:
                    body = r.text[:120]
                results.append(fn(r.status_code, body))
            except Exception as exc:
                results.append(Result(f"GET {path}", False, f"{type(exc).__name__}: {exc}"))
    return results


def _sigterm(_signo, _frame):
    raise SystemExit(1)  # run the finally blocks: stop the server, remove the temp audit dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--api", default="http://127.0.0.1:8000", help="base URL of a running API")
    ap.add_argument("--start", action="store_true", help="launch a replay server on a free port for the check")
    ap.add_argument("--scenario", default="full_demo")
    ap.add_argument("--speed", type=float, default=20.0)
    ap.add_argument("--seconds", type=float, default=None,
                    help="max seconds to read the stream (default: past the masked segment + 20 s)")
    args = ap.parse_args(argv)
    seconds = args.seconds if args.seconds is not None else default_seconds(args.scenario, args.speed)
    signal.signal(signal.SIGTERM, _sigterm)

    server = None
    try:
        api = args.api.rstrip("/")
        if args.start:
            import httpx

            server = ReplayServer(args.speed)
            api = server.url
            print(f"started replay server on {api} (audit dir: temporary)")
            with httpx.Client(timeout=2.0, trust_env=False) as c:
                server.wait_healthy(c)
        print(f"replaying {args.scenario} at speed {args.speed:g}, reading /ws/events for up to {seconds:.0f} s ...")
        results = run(api, scenario=args.scenario, speed=args.speed, seconds=seconds)
    except Exception as exc:
        results = [Result("setup", False, f"{type(exc).__name__}: {exc}")]
    finally:
        if server is not None:
            server.close()
    if server is not None and server.tail and not all(r.ok for r in results):
        print("server output (last lines):\n  " + "\n  ".join(server.tail), file=sys.stderr)
    print(render_table(results))
    return 0 if results and all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
