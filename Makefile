ARGS=
PROGRAM= src
TRACE= .install_trace
PYTHON= python

all : run

run : $(TRACE)
	uv run $(PYTHON) -m $(PROGRAM) $(ARGS)

install $(TRACE) : pyproject.toml Makefile
	pip install --upgrade pip ; \
	pip install uv ; \
	uv sync ; \
	touch .install_trace

debug:
	uv run $(PYTHON) -m pdb $(PROGRAM) $(ARGS)

clean:
	$(RM) -r data/output/*
	$(RM) -r __pycache__
	$(RM) -r src/__pycache__
	$(RM) -r .mypy_cache

fclean:
	$(RM) -r data/output/*
	$(RM) -r __pycache__
	$(RM) -r src/__pycache__
	$(RM) -r .mypy_cache
	$(RM) .install_trace
	$(RM) -r .venv
	$(RM) uv.lock
	uv cache prune

lint:
	uv run flake8 . --exclude data,uv_lock,.venv,llm_sdk ; \
	uv run mypy . --exclude data,uv_lock,.venv,llm_sdk --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run  flake8 . --exclude data,uv_lock,.venv,llm_sdk ; \
	uv run  mypy . --exclude data,uv_lock,.venv,llm_sdk --strict

.PHONY : all run install debug clean fclean lint lint-strict
