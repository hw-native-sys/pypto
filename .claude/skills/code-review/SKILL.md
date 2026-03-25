---
name: code-review
description: Review code changes against PyPTO project standards before committing. Use when reviewing code, preparing commits, checking pull requests, or when the user asks for code review.
context: fork
allowed-tools: Read, Grep, Glob, Bash
---

# PyPTO Code Review

You are a specialized code review agent. Review all code changes in the current git diff against PyPTO's quality standards. You MUST NOT modify any files — only analyze and report.

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
- [ ] **Pass complexity ≤ O(N log N)**: No nested full scans over IR node collections, no linear scans for lookups (see [`pass-complexity.md`](../../rules/pass-complexity.md))

### 2. Python Style (see `python-style.md` for full details)

- [ ] `@overload` used for functions with multiple distinct call signatures (not `Union`)
- [ ] Modern type syntax: `list[int]`, `dict[str, Any]` (not `List`, `Dict`)
- [ ] f-strings for all string formatting (no `.format()` or `%`)
- [ ] Google-style docstrings (Args/Returns/Raises)
- [ ] Type hints on all public API parameters and return types

### 3. Documentation Alignment

- [ ] Documentation reflects code changes
- [ ] Examples in docs still work
- [ ] Documentation files ≤500 lines (split if >700 lines)
- [ ] AI rules/skills ≤200 lines
- [ ] C++ implementation matches Python bindings
- [ ] Type stubs (`.pyi`) match actual API
- [ ] Docstrings complete and accurate
- [ ] Referenced files still exist
- [ ] Pass documentation numbering matches pass manager execution order (see `../../rules/pass-doc-ordering.md`)

**See [documentation-length.md](../../rules/documentation-length.md) for length guidelines**

### 4. Testing Standards

- [ ] Tests use **pytest** (not `unittest.TestCase`)
- [ ] All test verification uses **`assert`** (not `print`)
- [ ] No `unittest` imports (`unittest.TestCase`, `self.assertEqual`, `self.assertRaises`)
- [ ] Test files are located exclusively in the `tests/` directory
- [ ] pytest fixtures for setup/teardown (not `setUp()`/`tearDown()`)
- [ ] `pytest.raises()` for exception testing

### 5. Multi-Language Documentation Sync

- [ ] If English docs (`docs/en/dev/`) modified, corresponding `docs/zh-cn/dev/` files updated or follow-up created
- [ ] If `README.md` modified, `README.zh-CN.md` updated or flagged
- [ ] Chinese translations preserve all code examples untranslated
- [ ] Cross-references in Chinese docs point to Chinese versions
- [ ] Technical terms use dual notation: "Chinese (English)"

### 6. Commit Content

- [ ] Only relevant changes included
- [ ] No build artifacts (`build/`, `*.o`, `*.so`)
- [ ] No sensitive information (tokens, keys, absolute paths)
- [ ] No temporary files or ad-hoc/example scripts
- [ ] No AI co-author lines (`Co-Authored-By: Claude`, etc.)
- [ ] Changes are cohesive and related

## Cross-Layer Consistency

When APIs change, all three layers must be updated together. See [cross-layer-sync.md](../../rules/cross-layer-sync.md) for examples and naming conventions.

**Quick check:**

- [ ] C++ headers (`include/pypto/`)
- [ ] Python bindings (`python/bindings/`) — snake_case method names
- [ ] Type stubs (`python/pypto/pypto_core/`) — signatures match, `@overload` where needed

## Common Issues to Flag

- **Debug code**: `std::cout << "DEBUG"`, `print()` left in, `// TODO: fix later`
- **Wrong error macro**: `CHECK` for internal invariants (use `INTERNAL_CHECK`), or vice versa
- **C++ exceptions**: `throw std::runtime_error(...)` instead of `pypto::ValueError`
- **Missing bindings**: C++ method added but no Python binding or type stub
- **`Union` instead of `@overload`**: Function with distinct call patterns uses `Union` args
- **Legacy type syntax**: `List[int]`, `Dict[str, Any]` instead of `list[int]`, `dict[str, Any]`
- **Outdated docs**: API changed but documentation shows old usage
- **Build artifacts**: `build/`, `__pycache__/`, `*.pyc` in staged files
- **unittest usage**: `unittest.TestCase`, `self.assertEqual()`, `self.assertRaises()` instead of pytest
- **Print-style testing**: `print(result)` instead of `assert result == expected` in tests
- **AI co-author**: `Co-Authored-By: Claude` or similar lines in commits
- **Hardcoded paths**: Absolute paths like `/home/user/...` instead of relative paths
- **Vague error messages**: `raise ValueError("Invalid")` without context
- **Quadratic pass complexity**: Nested full scans over the same/global IR node collection, or linear scans instead of indexed lookups inside a pass traversal

## Output Format

Provide your review as:

```text
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

## Key Focus Areas

1. **Code Quality**: Style, error handling (`CHECK` vs `INTERNAL_CHECK`), PyPTO exceptions (not C++), no debug code, error messages with context
2. **Pass Complexity**: All passes must be O(N log N) or better — no nested full scans over IR node collections, no linear scans for lookups
3. **Python Style**: `@overload` for multiple signatures (not `Union`), modern type syntax (`list[int]` not `List[int]`), f-strings, Google-style docstrings, type hints on public APIs
4. **Testing Standards**: pytest only (no `unittest`), `assert` for verification (no `print`), `pytest.raises()` for exceptions, tests only in `tests/`
5. **Documentation**: Alignment with code changes, examples still work, file lengths (≤500 for docs, ≤200 for rules/skills), pass doc numbering matches pass manager execution order
6. **Cross-Layer Sync**: C++ headers, Python bindings, and type stubs must all be updated together
7. **Commit Content**: Only relevant changes, no artifacts, no sensitive data, no AI co-author lines, no hardcoded absolute paths
8. **Multi-Language Doc Sync**: When English docs (`docs/en/dev/` or `README.md`) are modified, verify corresponding `docs/zh-cn/` and `README.zh-CN.md` are also updated or flagged
