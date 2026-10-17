# Valence — common commands. Python 3.12 venv in .venv (see README).
PY ?= .venv/bin/python
SPEED ?= 1

.PHONY: test scan replay live console data train eval smoke

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

smoke: test scan ## pre-demo check
