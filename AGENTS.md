# PyPTO Codex Instructions

This repository keeps AI project policy in `.claude/`. Treat `.claude/` as the
authoritative source of truth; this file is only the Codex entrypoint.

## Read First

Before making code changes, reviewing code, committing changes, or working
across layers:

- Read `.claude/CLAUDE.md`
- Read all relevant files in `.claude/rules/`
- Follow `.claude/skills/*/SKILL.md` when the task matches a repository-local workflow
- Follow the installed `pypto-developer` or `pypto-user` plugin skill when the
  task matches a shared workflow

Task mapping:

- Testing: `.claude/skills/testing/SKILL.md`
- Code review: `.claude/skills/code-review/SKILL.md`
- Codegen comparison: `.claude/skills/compare-codegen/SKILL.md`
- Shared Git, PR, issue, and branch workflows: `pypto-developer` plugin
- Shared IR trace and in-core profiling workflows: `pypto-user` plugin

## Shared Plugins

Portable skills are published from `hw-native-sys/pypto-skills`; they are not
vendored into this repository. Install the plugins needed for the current
agent and scenario.

Codex:

```bash
codex plugin marketplace add hw-native-sys/pypto-skills
codex plugin add pypto-developer@pypto-skills
codex plugin add pypto-user@pypto-skills
```

Claude Code:

```bash
claude plugin marketplace add hw-native-sys/pypto-skills
claude plugin install pypto-developer@pypto-skills
claude plugin install pypto-user@pypto-skills
```

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
  ground truth; keep `docs/zh/dev/` aligned when English docs change
- Follow `.claude/rules/documentation.md` for markdown file placement; do not
  create markdown files outside `docs/` unless that rule explicitly allows it
- Do not create temporary test scripts or examples outside `tests/` and
  `examples/`
- Build and test from the current worktree. Never copy `.so` files or other
  build artifacts from another checkout
- Treat `.env`, `secrets/`, credentials, and machine-specific absolute paths as
  off-limits unless the user explicitly requires them
- Never add AI co-author lines to commits or PR text

## Preferred Commands

For full testing details, follow `.claude/skills/testing/SKILL.md`.

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Optional test env
[ -f .claude/skills/testing/testing.env ] && . .claude/skills/testing/testing.env

# Configure and build
[ ! -f build/CMakeCache.txt ] && cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --parallel

# Test
export PYTHONPATH=$(pwd)/python:$PYTHONPATH
python -m pytest tests/ut/ -n auto --maxprocesses 8 -v
python -m pytest tests/ut/core/test_error.py -n auto --maxprocesses 8 -v

# Lint and type check
ruff check .
ruff format --check .
pyright
pre-commit run --all-files
```

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
- `docs/en/dev/`, `docs/zh/dev/`: developer documentation
- `examples/`: user-facing examples (kernels/, models/, utils/)
