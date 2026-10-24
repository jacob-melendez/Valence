"""Rerun every generator in this folder and compare its output with the literals pasted in
frontend/src/test/engine-runtime.test.ts.

    .venv/bin/python -B scripts/golden_runtime/check.py   # one line per comparison; exit status 1 on any mismatch

extra_runs.py, perturb.py and tracks.py run as subprocesses, each in its own interpreter as when run by hand. The
edge-case specs run in this process through edge_cases.run_spec, which also returns the raw messages that some
assertions look inside. The literals are read from the test file as it stands. Numbers are compared as JavaScript
compares them (1 and 1.0 are equal, -0.0 is not 0.0), except the break payloads, whose int/float types are checked.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from _common import ROOT, canon

HERE = Path(__file__).resolve().parent
TEST = ROOT / "frontend" / "src" / "test" / "engine-runtime.test.ts"


# ---------- a small reader for the TypeScript literals in the test ----------

TOKEN = re.compile(r"""
    (?P<ws>\s+|//[^\n]*|/\*.*?\*/)
  | (?P<str>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')
  | (?P<num>-?\d[\d_]*(?:\.\d+)?(?:[eE][+-]?\d+)?n?)
  | (?P<id>[A-Za-z_$][\w$]*)
  | (?P<punct>[{}\[\](),:+.<>])
  | (?P<other>.)
""", re.X | re.S)
ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "v": "\v", "0": "\0"}


def unescape(s: str) -> str:
    """The value of a quoted TS string literal (\\uXXXX, \\u{X...} and the one-letter escapes)."""
    body, out, i = s[1:-1], [], 0
    while i < len(body):
        c = body[i]
        if c != "\\":
            out.append(c)
            i += 1
        elif body[i + 1] == "u" and body[i + 2] == "{":
            j = body.index("}", i)
            out.append(chr(int(body[i + 3:j], 16)))
            i = j + 1
        elif body[i + 1] == "u":
            out.append(chr(int(body[i + 2:i + 6], 16)))
            i += 6
        else:
            out.append(ESCAPES.get(body[i + 1], body[i + 1]))
            i += 2
    return "".join(out).encode("utf-16", "surrogatepass").decode("utf-16")


class Reader:
    """Reads one TS value: objects, arrays, strings (also 'a' + 'b'), numbers (1n is an int), true / false / null,
    Array<T>(n).fill(x), and names or calls resolved through ``env``."""

    def __init__(self, src: str, env: dict[str, Any]):
        self.src, self.pos, self.env = src, 0, env

    def take(self, want: str | None = None) -> tuple[str, str]:
        while True:
            m = TOKEN.match(self.src, self.pos)
            if m is None:
                raise ValueError(f"cannot read TypeScript at {self.src[self.pos:self.pos + 40]!r}")
            self.pos = m.end()
            if m.lastgroup != "ws":
                break
        if want is not None and m.group() != want:
            raise ValueError(f"expected {want!r}, got {m.group()!r}")
        return m.lastgroup, m.group()

    def peek(self) -> str:
        pos = self.pos
        _, text = self.take()
        self.pos = pos
        return text

    def items(self, close: str, item: Callable[[], Any]) -> list[Any]:
        out = []
        while self.peek() != close:
            out.append(item())
            if self.peek() == ",":
                self.take()
        self.take(close)
        return out

    def pair(self) -> tuple[str, Any]:
        kind, key = self.take()
        self.take(":")
        return (unescape(key) if kind == "str" else key), self.value()

    def value(self) -> Any:
        kind, text = self.take()
        if text == "{":
            return dict(self.items("}", self.pair))
        if text == "[":
            return self.items("]", self.value)
        if kind == "str":
            s = unescape(text)
            while self.peek() == "+":
                self.take()
                s += self.value()
            return s
        if kind == "num":
            t = text.replace("_", "")
            if t.endswith("n"):
                return int(t[:-1])
            return float(t) if any(c in t for c in ".eE") else int(t)
        if text in ("true", "false", "null"):
            return {"true": True, "false": False, "null": None}[text]
        if text == "Array" and self.peek() == "<":
            while self.take()[1] != ">":
                pass
            self.take("(")
            n = self.value()
            self.take(")")
            self.take(".")
            self.take("fill")
            self.take("(")
            x = self.value()
            self.take(")")
            return [x] * n
        if kind == "id":
            v = self.env[text]
            if self.peek() == "(":
                self.take()
                return v(*self.items(")", self.value))
            return v
        raise ValueError(f"unexpected {text!r}")


def literal_after(src: str, anchor: str, env: dict[str, Any] | None = None) -> Any:
    """The TS value that starts where the first match of the regex ``anchor`` ends."""
    m = re.search(anchor, src)
    if m is None:
        raise ValueError(f"anchor not found in the test: {anchor}")
    return Reader(src[m.end():], env or {}).value()


def hello_core(src: str) -> Callable[[str, float, float, bool], str]:
    """The test's HELLO_CORE(mode, w, h, mirrored): its template pieces with the placeholders filled in."""
    m = re.search(r"const HELLO_CORE = \(.*?\): string =>(.*?);\n", src, re.S)
    if m is None:
        raise ValueError("HELLO_CORE not found in the test")
    template = "".join(re.findall(r"`([^`]*)`", m.group(1)))

    def fn(mode: str, w: float, h: float, mirrored: bool) -> str:
        return (template.replace("${mode}", mode).replace("${pyRepr(w)}", repr(float(w)))
                .replace("${pyRepr(h)}", repr(float(h))).replace("${String(mirrored)}", str(mirrored).lower()))
    return fn


# ---------- comparisons ----------

failures: list[str] = []


def js(v: Any) -> Any:
    """A value as JavaScript sees it: every number one double (repr keeps -0.0 apart), dicts unordered."""
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    if isinstance(v, (int, float)):
        return ("number", repr(float(v)))
    if isinstance(v, (list, tuple)):
        return [js(x) for x in v]
    if isinstance(v, dict):
        return {k: js(x) for k, x in v.items()}
    raise TypeError(type(v))


def same(what: str, got: Any, want: Any) -> None:
    ok = js(got) == js(want)
    print(("ok    " if ok else "DIFF  ") + what)
    if not ok:
        failures.append(what)
        print(f"      python: {json.dumps(got)[:500]}\n      test:   {json.dumps(want)[:500]}")


def run(script: str) -> str:
    return subprocess.run([sys.executable, "-B", str(HERE / script)], check=True, capture_output=True, text=True,
                          cwd=ROOT).stdout


def check_extra(src: str) -> None:
    want = literal_after(src, r"const EXTRA\b[^=]*=\s*", {"HELLO_CORE": hello_core(src)})
    runs = json.loads(run("extra_runs.py"))
    same("extra_runs.py  run names", [r["name"] for r in runs], list(want))
    for r in runs:
        got = {"ticks": r["ticks"], "messages": r["messages"], "predictions": r["predictions"], "digest": r["digest"],
               "auditN": r["audit_n"], "auditLast": r["audit_last"],
               "operator": [k for k in r["audit_kinds"] if k.startswith("operator:")], "gapTicks": r["gap_ticks"],
               "latest": r["latest"], "hello": r["hello"]}
        same(f"extra_runs.py  EXTRA.{r['name']}", got, want.get(r["name"]))


def check_edges(src: str) -> None:
    from edge_cases import load_specs, run_spec, summary

    want = literal_after(src, r"const SPEC_EXPECT\b[^=]*=\s*")
    runs = {name: run_spec(spec) for name, spec in load_specs([]).items()}
    same("edge_cases.py  spec names", list(runs), list(want))
    for name, r in runs.items():
        s = summary(name, r)
        same(f"edge_cases.py  SPEC_EXPECT.{name}", {k: s[k] for k in ("digest", "auditN", "auditLast", "state")},
             want.get(name))
        same(f"edge_cases.py  {name}: audit chain verifies", s["verify"], [True, s["auditN"]])

    env = {"C": literal_after(src, r"const C = ")}
    for name in ("sv", "room", "op"):
        same(f"edge_cases.py  {name}: narrator calls", runs[name]["state"]["narrator"],
             literal_after(src, re.escape(f'expect((runs["{name}"] as SpecRun).narrated).toEqual('), env))

    msgs = [m for t in runs["sv"]["messages"] for m in t]
    svs = [m for m in msgs if m["type"] == "state_vector"]
    preds = {m["event_id"]: m for m in msgs if m["type"] == "prediction"}
    same("edge_cases.py  sv: time of day of every state vector", [m["time_of_day"] for m in svs],
         literal_after(src, re.escape(".map((m) => m.time_of_day)).toEqual(")))
    same("edge_cases.py  sv: first state vector, canonical text", canon(svs[0]),
         literal_after(src, re.escape("expect(canon(sv)).toBe(")))
    p2 = preds["p1-000002"]
    same("edge_cases.py  sv: p1-000002 label, display, confidence",
         [p2["label"], p2["label_display"], p2["confidence"]],
         literal_after(src, re.escape("expect([pred?.label, pred?.label_display, pred?.confidence]).toEqual(")))
    same("edge_cases.py  sv: p1-000002 probabilities", p2["probabilities"],
         literal_after(src, re.escape("expect(pred?.probabilities).toEqual(")))
    p1 = preds["p1-000001"]
    same("edge_cases.py  sv: p1-000001 simulated fields", p1["simulated_fields"],
         literal_after(src, re.escape("expect(pred?.simulated_fields).toEqual(")))
    same("edge_cases.py  sv: p1-000001 explanation", p1["explanation"]["text"],
         literal_after(src, re.escape("expect(pred?.explanation.text).toBe(")))
    breaks = [e["payload"]["person_id"] for e in runs["op"]["audit"] if e["payload"].get("action") == "break"]
    want_breaks = literal_after(src, re.escape('expect(breaks.map((e) => e.payload["person_id"])).toEqual('))
    same("edge_cases.py  op: break person ids and their types", [[type(b).__name__, b] for b in breaks],
         [[type(b).__name__, b] for b in want_breaks])


def check_perturb(src: str) -> None:
    got = [json.loads(line) for line in run("perturb.py").splitlines() if line.strip()]
    same("perturb.py     PERTURBED", got, literal_after(src, r"const PERTURBED\b[^=]*=\s*"))


def check_tracks(src: str) -> None:
    out = json.loads(run("tracks.py"))
    got = [[o["fps"], None if o["tracks"] is None else
            [o["tracks"]["t"], o["tracks"]["capture_ts_ms"], o["tracks"]["mirrored"],
             [p["id"] for p in o["tracks"]["persons"]]]] for o in out[:-1]]
    same("tracks.py      fps and tracks after each push", got, literal_after(src, re.escape("expect(got).toEqual(")))
    last = out[-1]
    same("tracks.py      fps, mic, camera after the audio block", [last["fps"], last["mic"], last["camera"]],
         literal_after(src, re.escape("expect([rt.fps, rt.mic, rt.camera]).toEqual(")))


def main() -> None:
    src = TEST.read_text()
    check_extra(src)
    check_edges(src)
    check_perturb(src)
    check_tracks(src)
    print(f"\n{len(failures)} mismatch(es)" if failures else "\nevery generator reproduces the test's literals")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
