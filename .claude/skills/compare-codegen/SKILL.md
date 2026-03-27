---
name: compare-codegen
description: >-
  Compare codegen output (.pto files and pass dumps) between origin/main and
  the current branch for a given test case. Runs the test with --save-kernels
  --dump-passes --codegen-only on both branches via git worktree, then diffs
  the results. Use when the user asks to compare codegen output, diff .pto
  files between branches, or check what changed in generated code.
---

# Compare Codegen Output Between Branches

Compare `.pto` files and pass dump output between `origin/main` and the current
branch for a specific system test case.

## Prerequisites

- Current branch must have a clean or stashed working tree (git worktree
  requires no conflicts with the checked-out branch).
- The project must be buildable (`cmake` + C++ toolchain available).
- `origin/main` must be fetched (`git fetch origin main`).

## Workflow

### Step 0: Validate Inputs

The user provides a **test case specification**, which is a pytest node ID. Examples:

```text
tests/st/runtime/test_matmul.py::TestMatmulOperations::test_matmul_shapes
tests/st/runtime/test_dyn_orch_shape.py -k "test_dyn_orch_paged_attention"
tests/st/codegen/test_paged_attention_multi_config.py::TestPagedAttention::test_basic
```

If the user only gives a short name (e.g., "paged attention"), search under
`tests/st/` to locate the matching test file and confirm with the user.

Optional user inputs:

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| platform | `a2a3sim` | `--platform` value |
| extra pytest args | _(none)_ | Additional pytest flags (e.g. `-k "shape_64"`) |

### Step 1: Prepare Directories

```bash
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
COMPARE_DIR="build_output/compare_$(date +%Y%m%d_%H%M%S)"
MAIN_OUTPUT="$COMPARE_DIR/main"
BRANCH_OUTPUT="$COMPARE_DIR/$CURRENT_BRANCH"
mkdir -p "$MAIN_OUTPUT" "$BRANCH_OUTPUT"
```

### Step 2: Run on Current Branch

Build (if needed) and run the test on the current branch:

```bash
# Source test env if available
[ -f .claude/skills/testing/testing.env ] && source .claude/skills/testing/testing.env

# Build
[ ! -f build/CMakeCache.txt ] && cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --parallel

# Run test with codegen-only + save-kernels + dump-passes
export PYTHONPATH=$(pwd)/python:$PYTHONPATH
python -m pytest "$TEST_SPEC" -v -s --forked \
  --codegen-only --save-kernels --dump-passes \
  --kernels-dir="$BRANCH_OUTPUT" \
  --platform="${PLATFORM:-a2a3sim}" \
  $EXTRA_ARGS
```

### Step 3: Run on origin/main via git worktree

Create a temporary worktree, build from scratch, run the same test:

```bash
git fetch origin main
WORKTREE_DIR=$(mktemp -d -p /tmp pypto-main-XXXXXX)
git worktree add "$WORKTREE_DIR" origin/main

pushd "$WORKTREE_DIR"

# Source test env if available
[ -f .claude/skills/testing/testing.env ] && source .claude/skills/testing/testing.env

# Build inside worktree
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --parallel

# Run test
export PYTHONPATH=$(pwd)/python:$PYTHONPATH
python -m pytest "$TEST_SPEC" -v -s --forked \
  --codegen-only --save-kernels --dump-passes \
  --kernels-dir="$MAIN_OUTPUT_ABSOLUTE" \
  --platform="${PLATFORM:-a2a3sim}" \
  $EXTRA_ARGS

popd

# Cleanup worktree
git worktree remove "$WORKTREE_DIR" --force
```

**Important**: `$MAIN_OUTPUT_ABSOLUTE` must be an absolute path (use `realpath`
on `$MAIN_OUTPUT` before entering the worktree). The `--kernels-dir` in the
worktree must point back to the original repo's compare directory.

### Step 4: Diff the Output

Run the diff script bundled with this skill:

```bash
python .claude/skills/compare-codegen/scripts/diff_pto.py \
  "$MAIN_OUTPUT" "$BRANCH_OUTPUT" \
  --labels main "$CURRENT_BRANCH"
```

The script will:

1. Find all `.pto` files recursively in both directories
2. Match files by relative path
3. Report files only in main, only in branch, and changed files
4. Show unified diffs for changed `.pto` files
5. Optionally diff `passes_dump/` IR files if `--include-passes` is passed

### Step 5: Report Results

Present a summary to the user:

```text
## Codegen Comparison: main vs <branch>
**Test**: <test spec>
**Platform**: <platform>

### .pto File Differences
- Files only in main: N
- Files only in <branch>: M
- Changed files: K
- Unchanged files: J

### Detailed Diffs
[For each changed .pto file, show the unified diff]

### Pass Dump Differences (if --include-passes)
[Summary of IR differences at each pass stage]

### Output Location
- main output: <path>
- branch output: <path>
```

## Cleanup

The worktree is removed immediately after the test run. The output in
`build_output/compare_*/` is kept for the user to inspect. The user can remove
it manually:

```bash
rm -rf build_output/compare_*/
```

## Troubleshooting

| Issue | Solution |
| ----- | -------- |
| "fatal: worktree already exists" | `git worktree prune` then retry |
| Build fails in worktree | Check that `origin/main` is buildable; try `git fetch origin main` first |
| No `.pto` files in output | Verify the test produces codegen output; check `--codegen-only` was passed |
| Permission denied on tmp dir | Use `--compare-dir` to specify a writable directory |
| Test not found | Verify the pytest node ID; use `pytest --collect-only` to list available tests |
