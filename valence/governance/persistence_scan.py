"""Static check that raw-media code paths never persist anything.

Scans Python sources with ``ast`` and reports calls that could write pixels, audio or transcripts
to disk:

* ``valence/perception/``: no file writes at all, plus the banned media/serialization calls.
* ``valence/edge/``: may write features-only JSONL in text mode, but no binary writes and none of
  the banned calls.

A write that provably carries no media (e.g. validated calibration scalars) can be allowed with an
explicit comment on the call's line: ``# persistence-ok: <reason>``. Every exception is listed by
``--exceptions`` so reviewers can audit them.

Run ``python -m valence.governance.persistence_scan`` to print violations (exit code 1 if any).
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

BANNED_ATTRS = {
    "imwrite",  # cv2
    "VideoWriter",  # cv2
    "VideoWriter_fourcc",
    "save",  # np.save / PIL Image.save / torch.save
    "savez",
    "savez_compressed",
    "savetxt",
    "tofile",
    "dump",  # pickle / joblib / json.dump of raw data
    "to_pickle",
}
BANNED_MODULE_CALLS = {
    ("wave", "open"),
    ("soundfile", "write"),
    ("sf", "write"),
    ("scipy.io.wavfile", "write"),
    ("wavfile", "write"),
}
MEDIA_MODULES = {"cv2", "numpy", "np", "pickle", "joblib", "torch", "soundfile", "wave", "scipy.io.wavfile", "PIL.Image"}
COPY_CALLS = {("shutil", "copy"), ("shutil", "copy2"), ("shutil", "copyfile"), ("shutil", "copyfileobj"),
              ("shutil", "move")}
PRAGMA = "# persistence-ok:"
STRICT_DIRS = ("valence/perception",)
TEXT_WRITES_OK_DIRS = ("valence/edge",)


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    reason: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.reason}"


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_dotted(node.value)}.{node.attr}"
    return ""


def _open_mode(call: ast.Call) -> str | None:
    mode_node = call.args[1] if len(call.args) > 1 else None
    for kw in call.keywords:
        if kw.arg == "mode":
            mode_node = kw.value
    if mode_node is None:
        return "r"
    if isinstance(mode_node, ast.Constant) and isinstance(mode_node.value, str):
        return mode_node.value
    return None  # dynamic mode: treat as a write


def exceptions_in(source: str, path: str) -> list[Violation]:
    return [Violation(path, i, line.split(PRAGMA, 1)[1].strip() or "(no reason given)")
            for i, line in enumerate(source.splitlines(), start=1) if PRAGMA in line]


def scan_source(source: str, path: str, allow_text_writes: bool) -> list[Violation]:
    out: list[Violation] = []
    allowed = {v.line for v in exceptions_in(source, path) if v.reason != "(no reason given)"}
    tree = ast.parse(source, filename=path)
    # Names imported directly from media/serialization modules ("from cv2 import imwrite as w").
    bare: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in MEDIA_MODULES:
            for alias in node.names:
                if alias.name in BANNED_ATTRS or (node.module == "wave" and alias.name == "open"):
                    bare[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or node.lineno in allowed:
            continue
        func = node.func
        name = _dotted(func)
        if isinstance(func, ast.Name) and func.id in bare:
            out.append(Violation(path, node.lineno, f"banned call {bare[func.id]}() (imported as {func.id})"))
            continue
        if isinstance(func, ast.Attribute) and (_dotted(func.value), func.attr) in COPY_CALLS:
            out.append(Violation(path, node.lineno, f"file copy {name}()"))
            continue
        if not allow_text_writes and name in ("os.write", "os.fdopen", "os.open", "os.writev"):
            out.append(Violation(path, node.lineno, f"low-level file write {name}()"))
            continue
        if isinstance(func, ast.Attribute):
            if func.attr in BANNED_ATTRS:
                out.append(Violation(path, node.lineno, f"banned call {name or func.attr}()"))
                continue
            owner = _dotted(func.value)
            if (owner, func.attr) in BANNED_MODULE_CALLS:
                out.append(Violation(path, node.lineno, f"banned call {name}()"))
                continue
            if func.attr == "write_bytes" or (func.attr == "write_text" and not allow_text_writes):
                out.append(Violation(path, node.lineno, f"file write {name}()"))
                continue
        if name in ("open", "io.open") or (isinstance(func, ast.Attribute) and func.attr == "open" and name != "wave.open"):
            mode = _open_mode(node)
            writes = mode is None or any(c in mode for c in "wax+")
            if not writes:
                continue
            if mode is None or "b" in mode or not allow_text_writes:
                out.append(Violation(path, node.lineno, f"file opened for writing ({mode!r})"))
    return out


def _sources(root: Path):
    for rel_dir, allow_text in [(d, False) for d in STRICT_DIRS] + [(d, True) for d in TEXT_WRITES_OK_DIRS]:
        base = root / rel_dir
        if base.exists():
            for py in sorted(base.rglob("*.py")):
                if ".venv" not in py.parts:
                    yield py, allow_text


def list_exceptions(root: str | Path = ".") -> list[Violation]:
    root = Path(root)
    return [v for py, _ in _sources(root) for v in exceptions_in(py.read_text(encoding="utf-8"), str(py.relative_to(root)))]


def scan_repo(root: str | Path = ".") -> list[Violation]:
    root = Path(root)
    found: list[Violation] = []
    for py, allow_text in _sources(root):
        found.extend(scan_source(py.read_text(encoding="utf-8"), str(py.relative_to(root)), allow_text))
    return found


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    root = args[0] if args else "."
    if "--exceptions" in sys.argv:
        for v in list_exceptions(root):
            print(f"{v.path}:{v.line}: allowed: {v.reason}")
    violations = scan_repo(root)
    for v in violations:
        print(v)
    print(f"{len(violations)} violation(s), {len(list_exceptions(root))} documented exception(s)")
    sys.exit(1 if violations else 0)
