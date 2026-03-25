---
name: git-commit
description: Complete git commit workflow for PyPTO including pre-commit review, staging, message generation, and verification. Use when creating commits, preparing changes for commit, or when the user asks to commit changes.
---

# PyPTO Git Commit Workflow

## Task Tracking

Create tasks and check them off as you complete each step:

```text
- [ ] Step 0: Code simplification decision
- [ ] Step 1: Analyze changes & launch review/test skills
- [ ] Step 2: Address issues, stage & commit
- [ ] Step 3: Post-commit verification
```

## Step 0: Code Simplification Decision (MANDATORY checkpoint)

**You MUST ask the user this question before proceeding. Do NOT skip this step.**

Ask the user:

> Run code simplification before committing? This refines your changed code for clarity, consistency, and maintainability while preserving all functionality. May take extra time. (yes/no)

**If YES** → Run code simplification on changed files, then proceed to Step 1.

**If NO** → Proceed directly to Step 1.

**You MUST wait for the user's answer before continuing.**

## Step 1: Analyze Changes & Launch Review/Test Skills

**Check what changed:**

```bash
git diff --name-only
git diff --cached --name-only
```

**Determine what to run based on changed files:**

| File Types Changed | Run Code Review | Run Testing | Run Clang-Tidy |
| ------------------ | --------------- | ----------- | -------------- |
| C++ (`.cpp`, `.h`) | ✅ Yes | ✅ Yes | ✅ Yes |
| Python (`.py`, bindings, tests) | ✅ Yes | ✅ Yes | ❌ Skip |
| Build system (`.cmake`, `CMakeLists.txt`) | ✅ Yes | ✅ Yes | ✅ Yes |
| Docs only (`.md`, `.rst`, `docs/`) | ✅ Yes | ❌ Skip | ❌ Skip |
| Config only (`.json`, `.yaml`, `.toml`, `.github/`) | ✅ Yes | ❌ Skip | ❌ Skip |
| Mixed (code + docs/config) | ✅ Yes | ✅ Yes | If C++ changed |

**Launch in parallel (each runs in its own forked context):**

- **`code-review`** skill — ALWAYS run for all changes
- **`testing`** skill — ONLY run if code files changed
- **`clang-tidy`** via Bash: `python tests/lint/clang_tidy.py --diff-base HEAD` — ONLY if C++ changed

Wait for all to complete, then address any issues found.

## Step 2: Stage Changes & Commit

**Related changes together**:

```bash
git add path/to/file1.cpp path/to/file2.h
git diff --staged  # Review
```

**Cross-layer pattern** (C++ + Python + Type stubs + Tests):

```bash
git add include/pypto/ir/expr.h python/bindings/ir_binding.cpp \
        python/pypto/pypto_core/__init__.pyi tests/ut/ir/test_expr.py
```

**Never stage**: Build artifacts (`build/`, `*.o`), temp files, IDE configs

## Commit Message Format

**Structure**: `type(scope): description (≤72 chars)`

**Types**: feat, fix, refactor, test, docs, style, chore, perf
**Scope**: Module/component (ir, printer, builder)
**Description**: Present tense, action verb, no period

**Good examples**:

```text
feat(ir): Add unique identifier field to MemRef
fix(printer): Update printer to use yield_ instead of yield
refactor(builder): Simplify tensor construction logic
test(ir): Add edge case coverage for structural comparison
```

**Bad examples** (avoid):

```text
❌ feat(ir): Added feature.  # Past tense, has period
❌ Fix bug                   # Missing type prefix
❌ WIP                       # Not descriptive
```

## Commit

```bash
# Short message
git commit -m "feat(ir): Add tensor rank validation"

# Detailed message (in editor)
git commit
```

**In editor**:

```text
feat(ir): Add tensor rank validation

Validates tensor rank is positive before setting shape.
Raises ValueError for invalid ranks.
Updates tests with edge case coverage.
```

## Co-Author Policy

**❌ NEVER add AI assistants**: No Claude, ChatGPT, Cursor AI, etc.
**✅ Only credit human contributors**: `Co-authored-by: Name <email>`

**Why?** AI tools are not collaborators. Commits reflect human authorship.

## Step 3: Post-Commit Verification

```bash
git show HEAD              # View commit
git log -1                 # Check message
git show HEAD --name-only  # Verify files
```

**Fix issues** (only if not pushed):

```bash
git commit --amend -m "Corrected message"      # Fix message
git add file && git commit --amend --no-edit   # Add forgotten file
```

⚠️ **Only amend unpushed commits!**

## Checklist

- [ ] Code simplification decision obtained from user (Step 0)
- [ ] Changed files analyzed (code vs docs/config only)
- [ ] Code review completed
- [ ] Tests passed (if code changed) or skipped (if docs/config only)
- [ ] Clang-tidy passed (if C++ changed) or skipped (if no C++)
- [ ] Only relevant files staged
- [ ] No build artifacts
- [ ] Message format: `type(scope): description` (≤72 chars, present tense, no period)
- [ ] No AI co-authors

## Remember

A good commit is thoroughly reviewed, groups related changes, has clear "why" message, and attributes only human authors.
