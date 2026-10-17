"""Offline narrator evaluation: faithfulness, fallback and latency over a batch of fact sheets.

    python -m valence.narrator.evaluate --n 100 --seed 0 --mode llm|template \\
        [--adapter PATH] [--model PATH] [--sheets FILE.jsonl] [--out FILE.json]

Prints one JSON report. Rates over raw LLM attempts (hallucinated numbers, false mask language,
banned terms) measure the model itself; ``pass_rate`` measures what users would actually see
after validation, retries and the template fallback. The report holds only metrics and reason
codes, never narration text.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from valence.narrator.engine import Narrator, TemplateOnlyNarrator
from valence.narrator.fake_fact_sheets import make_batch
from valence.narrator.validator import validate

_TOKEN = re.compile(r"[a-z0-9]+(?:[.'%-][a-z0-9]+)*%?")
_COLLAPSE = ("number", "words", "sentences", "drivers", "person")


def load_sheets(path: str | Path) -> list[dict[str, Any]]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile (q in [0, 100])."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return round(ordered[rank - 1], 1)


def distinct_n(texts: Iterable[str], n: int) -> float:
    """Unique n-grams / total n-grams over the whole corpus (lower-cased word tokens)."""
    total, unique = 0, set()
    for text in texts:
        tokens = _TOKEN.findall(text.lower())
        grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        total += len(grams)
        unique.update(grams)
    return round(len(unique) / total, 4) if total else 0.0


def reason_kind(reason: str) -> str:
    """Collapse value-specific codes ("number:57" -> "number"); keep categorical ones whole."""
    kind = reason.partition(":")[0]
    return kind if kind in _COLLAPSE else reason


def _rate(hits: int, total: int) -> float | None:
    return round(hits / total, 4) if total else None


def _case(sheet: Mapping[str, Any]) -> str:
    if sheet.get("mask_override"):
        return "mask"
    return "hidden" if not sheet.get("face_visible", True) else "visible"


def summarize(sheets: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]],
              deadline_s: float | None = None) -> dict[str, Any]:
    """Metrics for paired (sheet, narrate() result) lists."""
    n = len(results)
    sources = Counter(str(r["source"]) for r in results)
    finals = [validate(r["text"], s) for s, r in zip(sheets, results, strict=True)]
    latencies = [float(r["latency_ms"]) for r in results]

    attempts = [(s, reasons) for s, r in zip(sheets, results, strict=True) for reasons in r.get("attempt_reasons", [])]
    unmasked = [(s, reasons) for s, reasons in attempts if not s.get("mask_override")]
    failures = Counter(reason_kind(reason) for _, reasons in attempts for reason in reasons)

    by_case: dict[str, dict[str, Any]] = {}
    for case in ("mask", "hidden", "visible"):
        picked = [r for s, r in zip(sheets, results, strict=True) if _case(s) == case]
        by_case[case] = {"n": len(picked), "first_try": _rate(sum(r["source"] == "llm" for r in picked), len(picked)),
                         "llm_any": _rate(sum(r["source"] != "template" for r in picked), len(picked))}

    report: dict[str, Any] = {
        "n": n,
        "pass_rate": _rate(sum(v.ok for v in finals), n),
        "llm_first_try_pass_rate": _rate(sources["llm"], n),
        "llm_any_pass_rate": _rate(sources["llm"] + sources["llm_retry"], n),
        "sources": {k: sources.get(k, 0) for k in ("llm", "llm_retry", "template")},
        "fallback_rate": _rate(sources["template"], n),
        "salvaged": sum(bool(r.get("salvaged")) for r in results),
        "latency_ms": {"p50": percentile(latencies, 50), "p95": percentile(latencies, 95),
                       "max": max(latencies) if latencies else None},
        "mean_attempts": round(sum(int(r["attempts"]) for r in results) / n, 3) if n else None,
        "distinct_1": distinct_n((r["text"] for r in results), 1),
        "distinct_2": distinct_n((r["text"] for r in results), 2),
        "llm_attempts": len(attempts),
        "hallucinated_number_rate": _rate(sum(any(x.startswith("number:") for x in rs) for _, rs in attempts),
                                          len(attempts)),
        "false_mask_rate": _rate(sum(any(x.startswith("mask_term:") for x in rs) for _, rs in unmasked),
                                 len(unmasked)),
        "banned_term_rate": _rate(sum(any(x.startswith(("banned:", "person:")) for x in rs) for _, rs in attempts),
                                  len(attempts)),
        "final": {
            "hallucinated_number_rate": _rate(sum(any(x.startswith("number:") for x in v.reasons) for v in finals), n),
            "false_mask_rate": _rate(sum(any(x.startswith("mask_term:") for x in v.reasons) for v in finals), n),
            "banned_term_rate": _rate(
                sum(any(x.startswith(("banned:", "person:")) for x in v.reasons) for v in finals), n),
        },
        "final_violations": dict(Counter(reason_kind(x) for v in finals for x in v.reasons).most_common(10)),
        "top_failure_reasons": dict(failures.most_common(12)),
        "by_case": by_case,
        "final_words_mean": round(sum(v.word_count for v in finals) / n, 1) if n else None,
    }
    if deadline_s is not None:
        report["over_deadline"] = sum(ms > deadline_s * 1e3 + 100 for ms in latencies)
    return report


def run(narrator: Narrator, sheets: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [narrator.narrate(sheet) for sheet in sheets]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the narrator on fact sheets.")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=("llm", "template"), default="llm")
    parser.add_argument("--model", default=None)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--sheets", default=None, help="JSONL fact sheets (default: synthetic make_batch)")
    parser.add_argument("--out", default=None, help="also write the JSON report here")
    args = parser.parse_args(argv)

    sheets = load_sheets(args.sheets)[: args.n] if args.sheets else make_batch(args.n, args.seed)
    narrator = TemplateOnlyNarrator() if args.mode == "template" else Narrator(args.model, args.adapter)
    with narrator:
        narrator.load()
        results = run(narrator, sheets)
    report = {
        "mode": args.mode,
        "model": narrator.model_path,
        "adapter": narrator.adapter_path,
        "sheets": args.sheets or f"make_batch(n={args.n}, seed={args.seed})",
        **summarize(sheets, results, narrator.deadline_s),
    }
    text = json.dumps(report, indent=2)
    sys.stdout.write(text + "\n")
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
