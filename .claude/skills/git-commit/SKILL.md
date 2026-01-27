---
name: git-commit
description: Complete git commit workflow for PyPTO including pre-commit review, staging, message generation, and verification. Use when creating commits, preparing changes for commit, or when the user asks to commit changes.
---

# PyPTO Git Commit Workflow

## Prerequisites

⚠️ **ALWAYS run these agents first (IN PARALLEL):**
1. **`code-reviewer`** - Review code quality and consistency
2. **`testing`** - Verify build and tests pass

**Important:** These agents invoke subagents that can run in parallel. Always launch both agents simultaneously to save time.

## Workflow

```
1. Launch code-review and testing agents in parallel
2. Wait for both agents to complete
3. Address any issues found
4. Stage changes
5. Generate commit message
6. Commit
7. Verify
```

## Stage Changes

**Stage related changes together:**
```bash
git add path/to/file1.cpp path/to/file2.h
git diff --staged  # Review
```

**Cross-layer pattern:**
```bash
git add include/pypto/ir/expr.h              # C++ header
git add src/ir/expr.cpp                      # Implementation
git add python/bindings/ir_binding.cpp       # Binding
git add python/pypto/pypto_core/__init__.pyi # Type stub
git add tests/ut/ir/test_expr.py             # Tests
```

**Never stage:** Build artifacts, temporary files, IDE configs

## Commit Message Format

**Structure:** `type(scope): description (≤72 chars)`

**Format:**
- `type`: feat, fix, refactor, test, docs, style, chore, perf
- `scope`: Optional module/component (e.g., ir, printer, builder)
- `description`: Present tense, action verb, no period

**Rules:**
1. Present tense - "Add feature" not "Added feature"
2. Start with action verb in description
3. No period at end
4. Focus on "why" not just "what"
5. Use lowercase for type and scope

**Common types:**
- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code restructuring
- `test`: Adding/updating tests
- `docs`: Documentation changes
- `style`: Formatting, whitespace
- `chore`: Maintenance tasks
- `perf`: Performance improvements

**Good:**
```
feat(ir): Add unique identifier field to MemRef
fix(printer): Update printer to use yield_ instead of yield to match parser
refactor(builder): Simplify tensor construction logic
test(ir): Add edge case coverage for structural comparison
```

**Bad:**
```
❌ feat(ir): Added feature.     # Past tense, has period
❌ Fix bug                      # Missing type prefix
❌ feat: update files           # Missing scope, lowercase description
❌ WIP                          # Not descriptive
```

## Commit

```bash
# Short message
git commit -m "feat(ir): Add tensor rank validation"

# Detailed message (use editor)
git commit
```

**In editor:**
```
feat(ir): Add tensor rank validation

Validates tensor rank is positive before setting shape.
Raises ValueError for invalid ranks to prevent undefined behavior.

Updates tests/ut/ir/test_tensor.py with edge case coverage.
```

## Co-Author Policy

**❌ NEVER add AI assistants as co-authors:**
```
❌ Co-authored-by: Claude <claude@anthropic.com>
❌ Co-authored-by: ChatGPT <gpt@openai.com>
❌ Co-authored-by: Cursor AI <ai@cursor.com>
```

**✅ Only credit human contributors:**
```
✅ Co-authored-by: John Doe <john@example.com>
```

**Why?** AI tools are not collaborators. Commits must reflect human authorship and responsibility.

## Post-Commit Verification

```bash
# View the commit
git show HEAD

# Check commit message
git log -1

# Verify correct files
git show HEAD --name-only
```

**Fix issues (only if not pushed):**
```bash
# Amend message
git commit --amend -m "Corrected message"

# Add forgotten file
git add forgotten_file.cpp
git commit --amend --no-edit

# Remove wrong file
git reset HEAD^ -- unwanted_file.cpp
git commit --amend --no-edit
```

⚠️ **Only amend commits that haven't been pushed!**

## Quick Checklist

Before committing:
- [ ] Code review completed (use `code-review` skill)
- [ ] Testing completed (use `testing` skill)
- [ ] Only relevant files staged
- [ ] No build artifacts staged
- [ ] Commit message: type(scope): description format, present tense, ≤72 chars, no period
- [ ] No AI co-authors

## Remember

**A good commit:**
- Is thoroughly reviewed and tested
- Groups related changes together
- Has a clear message explaining "why"
- Properly attributes human authors only
- Can be understood in 6 months
