# Contributing

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pre-commit install
ruff check . && ruff format --check . && mypy && pytest
```

* Commits follow Conventional Commits (`feat(quest): ...`, `fix(hisparse): ...`, `docs: ...`).
* Every user-visible change gets a line under `## [Unreleased]` in `CHANGELOG.md`.
* Anything that changes which positions Quest selects, or attention numerics, must say so in the
  PR and update `docs/quest_walkthrough.md` / `docs/upstream_mapping.md`.
* Versioning, branches, releases and CI: see [docs/versioning.md](docs/versioning.md).
