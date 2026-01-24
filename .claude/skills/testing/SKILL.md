---
name: testing
description: Verify testing coverage and run tests for PyPTO project. Use when running tests, checking test coverage, or when the user asks about testing.
---

# PyPTO Testing Skill

## Overview

This skill provides comprehensive testing guidelines for the PyPTO project, including build procedures, test execution, and result analysis.

## How to Use

When you need to run tests:

1. Read the agent instructions at `.claude/agents/testing/AGENT.md`
2. Invoke the Task tool with `subagent_type="generalPurpose"` and include the agent instructions
3. The agent will build the project and run all tests using the guidelines below

## Environment Setup

Before running tests, check for a `testing.env` file in `.claude/skills/testing/`. This file allows you to specify local environment activation commands (e.g., `conda activate pypto`).

**If `testing.env` exists:**
- Source it before running tests: `source .claude/skills/testing/testing.env`
- This ensures your local environment is properly activated

**If `testing.env` doesn't exist:**
- Skip sourcing and proceed with tests
- Show this suggestion to the user:
  ```
  💡 Tip: Create .claude/skills/testing/testing.env to specify your environment activation.
     Example: See .claude/skills/testing/testing.env.example
  ```

## Testing Workflow

1. **Activate environment** (if `testing.env` exists): `source .claude/skills/testing/testing.env`
2. **Build the project**: `cmake --build build`
3. **Set PYTHONPATH**: `export PYTHONPATH=$(pwd)/python:$PYTHONPATH`
4. **Run tests**: `python -m pytest tests/ut/ -v`
5. **Analyze results**: Check for failures, errors, or warnings
6. **Report findings**: Provide clear test results summary

## Test Commands

```bash
# Activate environment (if testing.env exists)
[ -f .claude/skills/testing/testing.env ] && source .claude/skills/testing/testing.env

# Build
cmake --build build

# Set Python path
export PYTHONPATH=$(pwd)/python:$PYTHONPATH

# Run all tests
python -m pytest tests/ut/ -v

# Run specific test file
python -m pytest tests/ut/test_ir_basic.py -v

# Run specific test
python -m pytest tests/ut/test_ir_basic.py::test_tensor_expr_creation -v

# Run with coverage
python -m pytest tests/ut/ --cov=pypto_core --cov-report=html
```

## Test Structure

```
tests/ut/
├── core/              # Core functionality
├── ir/                # IR system
│   ├── core/          # Basic IR nodes
│   ├── expressions/   # Expression tests
│   ├── operators/     # Operator tests
│   └── parser/        # Parser tests
└── pass/              # Pass manager tests
```

## Testing Checklist

- [ ] Project builds without errors
- [ ] No new compiler warnings
- [ ] Python bindings compile successfully
- [ ] All existing tests pass
- [ ] No runtime errors during tests
- [ ] New features have corresponding tests
- [ ] Bug fixes include regression tests
- [ ] Tests are in proper location (`tests/ut/`)
- [ ] No temporary test files outside `tests/`

## Common Issues to Check

**Missing PYTHONPATH:**
```bash
# ❌ ImportError: No module named 'pypto_core'
python -m pytest tests/ut/

# ✅ Set PYTHONPATH first
export PYTHONPATH=$(pwd)/python:$PYTHONPATH
python -m pytest tests/ut/
```

**Stale build:**
```bash
# ✅ Always rebuild after code changes
cmake --build build
python -m pytest tests/ut/
```

**Tests in wrong location:**
```
❌ test_new_feature.py         # Project root
❌ python/test_binding.py      # Alongside code

✅ tests/ut/test_new_feature.py  # Correct location
```

## Output Format

Provide your testing report as:

```
## Testing Summary

**Status:** ✅ PASS / ⚠️ WARNINGS / ❌ FAIL

### Build Results

[Compiler output summary, any warnings or errors]

### Test Results

- Total tests run: X
- Passed: X
- Failed: X
- Skipped: X

### Failures (if any)

[List failed tests with details]

### Coverage (if run)

[Coverage statistics if requested]

### Recommendations

[Actions to fix issues or improve tests]
```

## Decision Criteria

**PASS:** All tests pass, build succeeds with no new warnings
**WARNINGS:** Tests pass but there are new compiler warnings or skipped tests
**FAIL:** Build fails or any tests fail

## Important Notes

- Always rebuild before running tests
- Check both build output and test output
- Look for new warnings even if tests pass
- Verify new features have tests
- Check for regression tests on bug fixes

## Related Skills

- **`code-review`** - Code review (can run in parallel with testing)
- **`git-commit`** - Complete commit workflow
