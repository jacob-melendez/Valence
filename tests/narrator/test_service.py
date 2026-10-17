from __future__ import annotations

import copy

import pytest
from fastapi.testclient import TestClient

from valence.narrator.engine import Narrator, TemplateOnlyNarrator
from valence.narrator.fake_fact_sheets import example_fact_sheet
from valence.narrator.service import create_app, main
from valence.narrator.templates import render_template
from valence.narrator.validator import validate

GOOD = (
    "Model predicted 'Frustrated (Masked)' with 88% confidence for Person #1. While they show a smiling "
    "expression (facial valence 0.62), the system detected 3 clenched-fist onsets, ambient noise at 72 dB "
    "and 4.2 hours since their last break. Context corrected the facial mask."
)


class FakeNarrator:
    """Minimal stand-in for the engine: counts loads and echoes a fixed answer."""

    def __init__(self) -> None:
        self.loads = 0
        self.closed = False
        self.seen: list[dict] = []

    def load(self) -> None:
        self.loads += 1

    def narrate(self, sheet: dict) -> dict:
        self.seen.append(sheet)
        return {"text": GOOD, "source": "llm", "latency_ms": 12.5, "attempts": 1, "reasons": []}

    def health(self) -> dict:
        return {"ok": True, "model": "fake", "adapter": None, "warm": self.loads > 0,
                "requests": len(self.seen), "by_source": {"llm": len(self.seen), "llm_retry": 0, "template": 0}}

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake():
    return FakeNarrator()


@pytest.fixture
def client(fake):
    with TestClient(create_app(narrator=fake)) as test_client:
        yield test_client


def test_lifespan_loads_once_and_closes(fake):
    with TestClient(create_app(narrator=fake)) as test_client:
        test_client.get("/health")
        test_client.get("/health")
        assert fake.loads == 1
    assert fake.closed


def test_health(client):
    body = client.get("/health").json()
    assert set(body) >= {"ok", "model", "adapter", "warm", "requests", "by_source"}
    assert body["ok"] is True and body["warm"] is True and body["requests"] == 0


def test_narrate_happy_path(client, fake):
    response = client.post("/narrate", json=example_fact_sheet())
    assert response.status_code == 200
    assert response.json() == {"text": GOOD, "source": "llm", "latency_ms": 12.5, "attempts": 1}
    assert fake.seen[0]["event_id"] == "evt_000123"
    assert fake.seen[0]["layers"][0]["drivers"][0]["display"] == "Ambient noise"
    assert client.get("/health").json()["requests"] == 1


def test_narrate_keeps_unknown_core_fields(client, fake):
    sheet = example_fact_sheet()
    sheet["n_people"] = 2
    sheet["layers"][0]["drivers"][0]["level"] = "high"
    assert client.post("/narrate", json=sheet).status_code == 200
    assert fake.seen[0]["n_people"] == 2
    assert fake.seen[0]["layers"][0]["drivers"][0]["level"] == "high"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.pop("label"),
        lambda s: s.update(label="Angry"),
        lambda s: s.update(person="Alice"),
        lambda s: s.update(confidence=1.7),
        lambda s: s.update(layers="none"),
        lambda s: s["layers"][0]["drivers"][0].pop("shap"),
    ],
    ids=["missing-label", "unknown-state", "named-person", "confidence-range", "layers-type", "driver-shap"],
)
def test_narrate_rejects_invalid_sheets(client, fake, mutate):
    sheet = copy.deepcopy(example_fact_sheet())
    mutate(sheet)
    response = client.post("/narrate", json=sheet)
    assert response.status_code == 422
    assert fake.seen == []


def test_narrate_rejects_non_json(client):
    response = client.post("/narrate", content=b"not json", headers={"content-type": "application/json"})
    assert response.status_code == 422


def test_template_only_app_end_to_end():
    with TestClient(create_app(TemplateOnlyNarrator())) as test_client:
        body = test_client.post("/narrate", json=example_fact_sheet()).json()
        health = test_client.get("/health").json()
    assert body["source"] == "template" and body["attempts"] == 0
    assert body["text"] == render_template(example_fact_sheet())
    assert validate(body["text"], example_fact_sheet()).ok
    assert health["by_source"]["template"] == 1 and health["mode"] == "template"


def test_real_engine_with_fake_backend_through_http():
    narrator = Narrator(generate_fn=lambda messages, params, deadline: GOOD)
    with TestClient(create_app(narrator)) as test_client:
        body = test_client.post("/narrate", json=example_fact_sheet()).json()
        health = test_client.get("/health").json()
    assert body["source"] == "llm" and body["text"] == GOOD and body["attempts"] == 1
    assert health["warm"] is True and health["by_source"]["llm"] == 1


def test_model_load_failure_still_serves_templates(tmp_path):
    narrator = Narrator(model_path=str(tmp_path / "missing"))
    with TestClient(create_app(narrator)) as test_client:
        body = test_client.post("/narrate", json=example_fact_sheet()).json()
        health = test_client.get("/health").json()
    assert body["source"] == "template"
    assert health["ok"] is True and health["warm"] is False and health["error"]


def test_cli_template_only_starts_uvicorn_on_loopback(monkeypatch):
    seen = {}

    def fake_run(app, host, port, **kwargs):
        seen.update(host=host, port=port, narrator=app.state.narrator)

    monkeypatch.setattr("uvicorn.run", fake_run)
    assert main(["--port", "8099", "--template-only"]) == 0
    assert seen["host"] == "127.0.0.1" and seen["port"] == 8099
    assert isinstance(seen["narrator"], TemplateOnlyNarrator)
