# Contributing

Thanks for considering a contribution to this project.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements-dev.txt
pip install -e .
pre-commit install
```

## Before opening a PR

Run the same checks CI runs:

```bash
make check
```

or individually:

```bash
ruff check src tests
black --check src tests
mypy src
pytest
```

All four must pass. Please keep changes focused — small, single-purpose PRs
are easier to review than large ones.

## Guidelines

- Match existing code style (type hints, docstrings, `from __future__ import
  annotations` where already used).
- Add or update tests for any behavior change.
- Update `README.md` if you change user-facing behavior (CLI flags, config
  keys, outputs).
