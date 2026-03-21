# Worktree Build Rules

## Core Principle

**ALWAYS build and test from within the worktree. NEVER copy `.so` files or other build artifacts from the main repo.**

## Why

Copying build artifacts from the main repo is fragile and error-prone:

- The `.so` may be stale (built against different C++ code)
- Path assumptions baked into the build may not match
- It hides real build failures in the worktree's code
- It produces unreliable test results

## How to Build in a Worktree

```bash
# Configure build directory if it doesn't exist
[ ! -f build/CMakeCache.txt ] && cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo

# Build
cmake --build build --parallel

# Set PYTHONPATH to worktree's python/
export PYTHONPATH=$(pwd)/python:$PYTHONPATH

# Run tests
python -m pytest tests/ut/ -n auto --maxprocesses 8 -v
```

## When No C++ Changes Exist

Even if a PR only modifies Python files, still build from the worktree. The build produces the `.so` that Python imports — without it, tests cannot run. Do not shortcut by copying from the main repo.
