# PyPTO Codex Instructions

This repository keeps AI project policy in `.claude/`. Treat `.claude/` as the
authoritative source of truth; this file is only the Codex entrypoint.

## Read First

Before making code changes, reviewing code, committing changes, or working
across layers:

- Read `.claude/CLAUDE.md`
- Read all relevant files in `.claude/rules/`
- Follow `.claude/skills/*/SKILL.md` when the task matches a documented workflow

Task mapping:

- Testing: `.claude/skills/testing/SKILL.md`
- Code review: `.claude/skills/code-review/SKILL.md`
- Commit workflow: `.claude/skills/git-commit/SKILL.md`
- PR workflow: `.claude/skills/github-pr/SKILL.md`
- Issue workflows: `.claude/skills/create-issue/SKILL.md`,
  `.claude/skills/fix-issue/SKILL.md`, `.claude/skills/fix-pr/SKILL.md`
- Branch cleanup: `.claude/skills/clean-branches/SKILL.md`

When a Claude skill or agent refers to `Task`, a subagent, or Claude-only
plugins:

- Execute the workflow directly in Codex
- Use parallel tool calls when safe
- Treat `.claude/agents/*/AGENT.md` as checklists, not as a separate runtime

## Working Agreements

- Use modern language standards: Python 3.10+ and C++17+
- Keep public API changes synchronized across `include/pypto/`, `src/`,
  `python/bindings/`, and `python/pypto/pypto_core/*.pyi`
- Update docs when behavior changes. English docs in `docs/en/dev/` are the
  ground truth; keep `docs/zh-cn/dev/` aligned when English docs change
- Do not create markdown files outside `docs/`, except `KNOWN_ISSUES.md`
- Do not create temporary test scripts or examples outside `tests/` and
  `examples/`
- Build and test from the current worktree. Never copy `.so` files or other
  build artifacts from another checkout
- Treat `.env`, `secrets/`, credentials, and machine-specific absolute paths as
  off-limits unless the user explicitly requires them
- Never add AI co-author lines to commits or PR text

## Preferred Commands

- Install dev dependencies: `pip install -e ".[dev]"`
- Optional test env: `[ -f .claude/skills/testing/testing.env ] && source
  .claude/skills/testing/testing.env`
- Configure build: `[ ! -f build/CMakeCache.txt ] && cmake -B build
  -DCMAKE_BUILD_TYPE=RelWithDebInfo`
- Build: `cmake --build build --parallel`
- Unit tests: `PYTHONPATH="$PWD/python:$PWD/build/python" python -m pytest
  tests/ut -v`
- Single test file: `PYTHONPATH="$PWD/python:$PWD/build/python" python -m
  pytest tests/ut/path/to/test_file.py -v`
- Python lint: `ruff check .`
- Python format check: `ruff format --check .`
- Type check: `pyright`
- Repository hooks: `pre-commit run --all-files`

Run system tests in `tests/st/` only when the task requires them and the
necessary hardware or environment is available.

## Repository Map

- `include/pypto/`: public C++ headers
- `src/`: C++ implementation, passes, codegen, runtime internals
- `python/bindings/`: nanobind extension bindings
- `python/pypto/`: Python API, DSL, pass manager, and type stubs
- `tests/ut/`: unit tests
- `tests/st/`: system and hardware-dependent tests
- `tests/lint/`: repo-specific lint and validation scripts
- `docs/en/dev/`, `docs/zh-cn/dev/`: developer documentation
- `examples/`: user-facing examples only
