## What & why

<!-- One or two sentences. Link the issue / upstream SGLang change if any. -->

## Quest / HiSparse semantics

- [ ] No change to what `QuestAlgorithm.retrieve_topk` selects or to the attention it feeds
- [ ] Changes selection or numerics — explained above, tests updated, `docs/` updated

## Checklist

- [ ] `ruff check . && ruff format --check .`
- [ ] `mypy`
- [ ] `pytest` (CPU); GPU-only paths noted if untested
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] If the upstream pin (`UPSTREAM_COMMIT`) moved: `docs/upstream_mapping.md` re-checked
