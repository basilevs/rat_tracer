# Agent guide

Short instructions for AI agents and contributors working in this repository.

## Setup

```sh
make install-dev   # editable install with runtime + dev tooling (ruff, mypy, pytest)
```

## Verify your changes

Run the full local gate before finishing a task:

```sh
make check   # runs lint, typecheck and the offline tests
```

Individual stages:

| Command          | What it does                                          |
| ---------------- | ----------------------------------------------------- |
| `make lint`      | Ruff lint + format check (no changes)                 |
| `make format`    | Ruff auto-format and auto-fix                         |
| `make typecheck` | mypy static type check                                |
| `make test`      | pytest, offline only (`-m "not network"`)             |

## Notes

- **Ruff** is the single linter/formatter; there is no pylint. Run `make format`
  before committing so lint stays clean.
- **mypy** is configured pragmatically: real type errors inside functions are
  caught, but annotations are not yet mandated everywhere. A baseline of
  pre-existing type errors is still being worked down, so `make typecheck` may
  report known failures — do not introduce new ones.
- Tests marked `network` reach Hugging Face and are excluded from `make test`.
  Run them explicitly with `pytest -m network` when you have connectivity.
- The detection model is downloaded from Hugging Face on first use. Set
  `RAT_TRACER_MODEL` to a local `.pt` path to override it.
