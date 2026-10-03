# make eval reproduces every number from the committed cache (layout is the headline run, plain
# is the ablation). make test runs the unit tests.
PY = .venv/Scripts/python.exe

eval:
	$(PY) evaluate.py --tier1-input layout
	$(PY) evaluate.py --tier1-input plain

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check . && $(PY) -m ruff format --check .
