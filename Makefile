PYTHON := code/venv/bin/python
PIP := code/venv/bin/pip
JUPYTEXT := code/venv/bin/jupytext
PRE_COMMIT := code/venv/bin/pre-commit
NOTEBOOK := code/fairness_audit.ipynb

.PHONY: setup notebook-py install-hooks

setup:
	$(PYTHON) -m pip install -r code/requirements.txt

notebook-py:
	$(JUPYTEXT) --from ipynb --to py:percent $(NOTEBOOK)

install-hooks:
	$(PRE_COMMIT) install
