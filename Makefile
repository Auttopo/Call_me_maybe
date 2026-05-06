ARGS=
PROGRAM= src
TRACE= .install_trace
ENV= .venv
PYTHON= python

all : run

run : $(TRACE)
	uv run $(PYTHON) -m $(PROGRAM) $(ARGS)

install $(TRACE) : pyproject.toml Makefile
	pip install --upgrade pip ; \
	pip install uv ; \
	uv sync ; \
	touch .install_trace

run-trace : $(TRACE)
	uv run $(PYTHON) -m pdb $(PROGRAM) $(ARGS) traceback

debug :
	uv run $(PYTHON) -m pdb $(PROGRAM) $(ARGS)

clean:
	. $(ENV)/bin/activate && \
	$(RM) -r __pycache__
	$(RM) -r .mypy_cache

fclean:
	. $(ENV)/bin/activate && \
	$(RM) -r __pycache__
	$(RM) -r .mypy_cache
	$(RM) .install_trace
	$(RM) -r $(ENV)
	uv cache prune
	$(RM) uv.lock

lint :
	uv run $(PYTHON) -m flake8 --exclude llm_sdk,$(ENV); \
	uv run $(PYTHON) -m mypy . --exclude llm_sdk --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict :
	uv run $(PYTHON) -m flake8 --exclude llm_sdk,$(ENV); \
	uv run $(PYTHON) -m mypy . --exclude llm_sdk --strict

.PHONY : all run install run-trace debug clean fclean lint lint-strict
