# Valence — common commands. Python 3.12 venv in .venv (see README).
PY ?= .venv/bin/python
SPEED ?= 1

.PHONY: test scan replay live console train eval facts negative smoke

test:            ## unit + integration tests
	$(PY) -m pytest -q

scan:            ## static check: no persistence in raw-media code paths
	$(PY) -m valence.governance.persistence_scan

replay:          ## API + scripted replay (SCENARIO=full_demo SPEED=1)
	VALENCE_MODE=replay VALENCE_SCENARIO=$(or $(SCENARIO),full_demo) VALENCE_SPEED=$(SPEED) $(PY) -m uvicorn valence.api.app:app --port 8000

live:            ## API + live edge (Mac only; never use --reload with sensors)
	VALENCE_MODE=live $(PY) -m uvicorn valence.api.app:app --port 8000

console:         ## terminal view of the event stream
	$(PY) scripts/console.py

train:           ## simulator corpus -> model ladder -> artifacts/ (STATES=core|all)
	$(PY) -m valence.models.train_ladder --sessions 300 --seed 7 --noise-scale 1.0 --state-set $(or $(STATES),core)

eval:            ## proof charts + metrics -> reports/
	$(PY) -m valence.eval.evaluate --bootstrap $(or $(BOOT),200)

facts:           ## balanced narrator fact sheets -> data/narrator/fact_sheets_sim.jsonl
	$(PY) scripts/export_fact_sheets.py --sessions 25 --per-cell 150

negative:        ## negative control on the live typing capture
	$(PY) -m valence.eval.negative_control data/real/capture_typing*.jsonl --expect Focused

smoke: test scan ## pre-demo check

.PHONY: e2e
e2e:             ## pre-demo end-to-end check: own replay server, or API=http://127.0.0.1:8000 for a running one
	$(PY) scripts/e2e_check.py $(if $(API),--api $(API),--start)
