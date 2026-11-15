# AGENTS.md

## Project overview
- `nada.py` is a minimal, stdlib-only coding agent driven by a model via
  OpenRouter. It exposes `read_file`, `write_file`, `edit_file`, and `bash`.
- `test_nada.py` holds the unit tests; `README.md` is the human docs.

## Setup commands
- Run the agent: `python3 nada.py` (needs `OPENROUTER_API_KEY`)
- Run tests: `python3 -m unittest -v`

## Code style
- Stdlib only; no new dependencies. Keep the agent in the single `nada.py`
  file unless there is a clear reason not to.
- Follow the existing patterns in `nada.py` (small helpers, docstrings on
  non-obvious functions) and keep behaviour changes covered by tests in
  `test_nada.py`. Update `README.md` when user-visible behaviour changes.

## Testing instructions
- Run `python3 -m unittest -v` before considering any change done; all
  tests must pass.
