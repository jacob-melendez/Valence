from __future__ import annotations

import json

from valence.narrator.evaluate import distinct_n, main, percentile, reason_kind, summarize
from valence.narrator.fake_fact_sheets import make_batch


def test_summarize_reports_contract_metrics():
    sheets = make_batch(6, seed=1, states="all")
    results = [
        {"text": "x", "source": "template", "latency_ms": 10, "attempts": 3,
         "attempt_reasons": [["number:91%", "qualifier:noise:high"], ["mask_term:smile"], ["banned:gendered:she"]]}
        for _ in sheets
    ]
    report = summarize(sheets, results, deadline_s=2.5)
    for key in ("pass_rate", "llm_first_try_pass_rate", "sources", "latency_ms", "distinct_1", "distinct_2",
                "hallucinated_number_rate", "false_mask_rate", "banned_term_rate", "qualifier_mismatch_rate",
                "top_failure_reasons"):
        assert key in report
    assert report["sources"] == {"llm": 0, "llm_retry": 0, "template": 6}
    assert report["hallucinated_number_rate"] == round(1 / 3, 4)
    assert report["qualifier_mismatch_rate"] == round(1 / 3, 4)
    assert reason_kind("qualifier:noise:high") == "qualifier" and reason_kind("state_missing") == "state_missing"
    assert percentile([1, 2, 3, 4], 50) == 2 and distinct_n(["a b", "a c"], 1) == 0.75


def test_cli_template_mode(capsys, tmp_path):
    out = tmp_path / "report.json"
    assert main(["--n", "12", "--mode", "template", "--states", "all", "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["n"] == 12 and report["pass_rate"] == 1.0 and "states='all'" in report["sheets"]
    assert json.loads(capsys.readouterr().out) == report
