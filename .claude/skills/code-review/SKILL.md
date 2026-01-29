---
name: code-review
description: Review code changes against PyPTO project standards before committing. Use when reviewing code, preparing commits, checking pull requests, or when the user asks for code review.
---

# PyPTO Code Review Skill

## Overview

This skill provides comprehensive code review guidelines for the PyPTO project, checking code quality, documentation alignment, and cross-layer consistency.

## How to Use

When you need to perform a code review:

1. Read the agent instructions at `.claude/agents/code-review/AGENT.md`
2. Invoke the Task tool with `subagent_type="generalPurpose"` and include the agent instructions
3. The agent will analyze changes and provide feedback using the guidelines below

## Review Process

1. **Get changes**: Run `git diff` to see all staged and unstaged changes
2. **Analyze each file** against the review checklist below
3. **Check documentation alignment**: Ensure docs match code changes
4. **Verify cross-layer consistency**: Check C++, Python bindings, and type stubs
5. **Report findings**: Provide clear, actionable feedback

## Review Checklist

### 1. Code Quality

- [ ] Follows project style and conventions
- [ ] No debug code or commented sections
- [ ] No TODOs/FIXMEs unless documented
- [ ] Proper use of `CHECK` vs `INTERNAL_CHECK`
- [ ] PyPTO exceptions (not C++ exceptions)
- [ ] Clear, descriptive names
- [ ] Appropriate comments for complex logic
- [ ] Linter errors fixed (not suppressed)

### 2. Documentation Alignment

- [ ] Documentation reflects code changes
- [ ] Examples in docs still work
- [ ] Documentation files ≤300 lines (split if >600 lines)
- [ ] AI rules/skills/agents ≤150 lines
- [ ] C++ implementation matches Python bindings
- [ ] Type stubs (`.pyi`) match actual API
- [ ] Docstrings complete and accurate
- [ ] Referenced files still exist

**See [documentation-length.md](./.claude/rules/documentation-length.md) for length guidelines**

### 3. Commit Content

- [ ] Only relevant changes included
- [ ] No build artifacts (`build/`, `*.o`, `*.so`)
- [ ] No sensitive information (tokens, keys)
- [ ] No temporary test files
- [ ] Changes are cohesive and related

### 4. Cross-Layer Consistency

When APIs change, all three layers must be updated:
- [ ] C++ headers (`include/pypto/`)
- [ ] Python bindings (`python/bindings/`)
- [ ] Type stubs (`python/pypto/pypto_core/`)

## Common Issues to Flag

**Debug code:**
```cpp
❌ std::cout << "DEBUG: value = " << value;
❌ // TODO: fix this later
```

**Improper error checking:**
```cpp
❌ throw std::runtime_error("Error");  # Use pypto::ValueError
❌ CHECK(internal_invariant);          # Use INTERNAL_CHECK
```

**Outdated docs:** API changed but documentation still shows old usage

**Missing bindings:** Added C++ method but forgot Python binding/stub

**Unrelated changes:** Multiple unrelated edits in one commit

**Build artifacts:** `build/`, `__pycache__/`, `*.pyc`

## Cross-Layer Consistency Example

When adding a method, all three must be synchronized:

**C++ Header:**
```cpp
class TensorExpr : public Expr {
 public:
  void SetRank(int rank);
  int GetRank() const;
};
```

**Python Binding:**
```cpp
nb::class_<TensorExpr>(m, "TensorExpr")
    .def("set_rank", &TensorExpr::SetRank)
    .def("get_rank", &TensorExpr::GetRank);
```

**Type Stub:**
```python
class TensorExpr(Expr):
    def set_rank(self, rank: int) -> None: ...
    def get_rank(self) -> int: ...
```

## Output Format

Provide your review as:

```
## Code Review Summary

**Status:** ✅ PASS / ⚠️ WARNINGS / ❌ FAIL

### Issues Found

[List any issues by category: Code Quality, Documentation, Cross-Layer, etc.]

### Recommendations

[Specific actions to fix issues]

### Approved Items

[List what looks good]
```

## Decision Criteria

**PASS:** No critical issues, minor suggestions only
**WARNINGS:** Non-critical issues that should be addressed but don't block commit
**FAIL:** Critical issues that must be fixed before committing

## Related Skills

- **`testing`** - Build and test verification (can run in parallel with code review)
- **`git-commit`** - Complete commit workflow
