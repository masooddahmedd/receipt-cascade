# make eval reproduces every number from the committed cache. make test runs the unit tests.
PY = .venv/Scripts/python.exe

eval:
	$(PY) evaluate.py

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check . && $(PY) -m ruff format --check .
