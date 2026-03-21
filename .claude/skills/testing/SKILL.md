---
name: testing
description: Verify testing coverage and run tests for PyPTO project. Use when running tests, checking test coverage, or when the user asks about testing.
---

# PyPTO Testing Skill

## Overview

Build and test the PyPTO project to verify code changes haven't broken anything.

## How to Use

1. Read agent instructions at [.claude/agents/testing/AGENT.md](.claude/agents/testing/AGENT.md)
2. Invoke Task tool with `subagent_type="testing"` (specialized agent)
3. Agent will build project and run all tests

## Environment Setup

**If `.claude/skills/testing/testing.env` exists**: Source it before testing.

```bash
[ -f .claude/skills/testing/testing.env ] && source .claude/skills/testing/testing.env
```

**If doesn't exist**: Skip and suggest creating one (see `testing.env.example`).

## Testing Workflow

**When working in a git worktree**, always build and test from within the worktree — never copy `.so` files from the main repo. Create a build directory inside the worktree if one doesn't exist.

```bash
# 1. Activate environment (if testing.env exists)
[ -f .claude/skills/testing/testing.env ] && source .claude/skills/testing/testing.env

# 2. Configure CMake if build is not configured (worktree, fresh clone)
[ ! -f build/CMakeCache.txt ] && cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo

# 3. Build project (parallel)
cmake --build build --parallel

# 4. Set PYTHONPATH
export PYTHONPATH=$(pwd)/python:$PYTHONPATH

# 5. Run tests
python -m pytest tests/ut/ -n auto --maxprocesses 8 -v
```

## Test Commands

```bash
# Run all tests
python -m pytest tests/ut/ -n auto --maxprocesses 8 -v

# Run specific test file
python -m pytest tests/ut/core/test_error.py -n auto --maxprocesses 8 -v

# Run specific test
python -m pytest tests/ut/core/test_error.py::TestErrorTypes::test_value_error_type -n auto --maxprocesses 8 -v
```

## Test Structure

```text
tests/ut/
├── core/          # Core functionality
├── ir/            # IR (nodes, expressions, operators, parser)
└── pass/          # Pass manager
```

## Testing Checklist

- [ ] Project builds without errors
- [ ] No new compiler warnings
- [ ] All existing tests pass
- [ ] New features have tests
- [ ] Bug fixes have regression tests
- [ ] Tests in `tests/ut/` (not elsewhere)

## Common Issues

| Issue | Solution |
| ----- | -------- |
| `ImportError: No module named 'pypto_core'` | `export PYTHONPATH=$(pwd)/python:$PYTHONPATH` |
| Tests fail after code changes | `cmake --build build --parallel` then re-run |
| Tests in wrong location | Move to `tests/ut/` |
| Build not configured (worktree/fresh clone) | `cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo` |

## Output Format

```text
## Testing Summary
**Status:** ✅ PASS / ⚠️ WARNINGS / ❌ FAIL

### Build Results
[Compiler output, warnings/errors]

### Test Results
- Total: X | Passed: X | Failed: X | Skipped: X

### Failures
[Failed test details if any]

### Recommendations
[Actions to fix issues]
```

## Decision Criteria

| Status | Criteria |
| ------ | -------- |
| **PASS** | All tests pass, build succeeds, no new warnings |
| **WARNINGS** | Tests pass but new warnings or skipped tests |
| **FAIL** | Build fails or tests fail |

## Important Notes

- Always rebuild before running tests
- Check both build and test output
- Look for new warnings even if tests pass
- Verify new features have corresponding tests

## Related Skills

- **`code-review`** - Code review (runs in parallel with testing)
- **`git-commit`** - Complete commit workflow
