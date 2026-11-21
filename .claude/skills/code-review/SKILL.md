---
name: code-review
description: Analyze, explain, and review PyPTO pull requests, or review local changes before committing. Verify PR findings on the current head through the normal entry point before assigning severity.
context: fork
allowed-tools: Read, Grep, Glob, Bash
---

# PyPTO Code Review

Review changes against PyPTO's quality standards. Keep the source under review intact. Build and run
reproductions in the review checkout, using a separate checkout when needed to preserve local edits.

## Review Process

1. For PR analysis, explanation, or review, resolve the current PR head to its full commit SHA. Read
   the complete diff for that SHA, the linked issues, the PR discussion, and existing reviews,
   including inline comments. For a local review, inspect both staged and unstaged changes.
2. Analyze the changed files against the checklist below. Trace affected callers, verify
   documentation alignment, and check C++, Python bindings, and type stubs where applicable.
3. Apply the PR evidence requirements before classifying findings or issuing a report. Before
   reporting, recheck the PR head. If it moved, verify the current head and recheck findings against it.

## PR Evidence Requirements

- Build the pinned PR head before assigning severity or publishing findings. Verify the checkout's
  `HEAD` matches the pinned SHA and follow the [testing workflow](../testing/SKILL.md) for the build.
  Use artifacts built in that checkout. Record the actual build result and command.
- Every verified behavior finding must reproduce on the reviewed build from a real input that
  the normal frontend DSL or public API can construct. Supply the runnable input, command, and
  observed symptom through the default lowering pipeline. Use handwritten or patched IR only
  to investigate the cause. For a documentation claim about behavior, quote the claim and run the
  frontend scenario that tests it. For purely static findings, show the relevant source,
  documentation, or configuration entries and the check that demonstrates the discrepancy.
- Classify a suspected defect as **Unverified** if the head build is unavailable or
  unsuccessful, the normal-entry reproduction fails to demonstrate the defect, or the defect can be
  triggered only with handwritten or modified IR. Keep these observations out of severity rankings
  and blocking conclusions. CI results, source inspection, or a build of another revision can provide
  context for an unverified observation.
- Match evidence to the claim. Generated-code string assertions establish what was emitted. Numerical
  checks establish correctness for the tested inputs. Claims about performance or hardware control
  effects require direct evidence, such as measurements or traces showing the claimed behavior under
  the relevant execution conditions. An explanation based on source or compiler artifacts alone is
  an inference.

## PR Walkthrough and Concrete Cases

Explain the PR's intended behavior even when the review finds no defects. Base the walkthrough on
the pinned diff and its callers. Describe the previous limitation, the practical workflow it
affected, what the PR changes for that workflow, and the scope of the new behavior. For internal
changes, name the caller and the observable effect or invariant that motivates the change.

- Use a complete representative before/after DSL or API case for the main behavior change.
  Include the relevant input declarations, values, call, and result or error. Explain why these
  inputs expose the difference and what the reader should look for in the output. For IR-only
  changes, show the relevant before/after IR with enough surrounding context to follow the value.
- Cover materially different entry points, modes, or boundaries with a focused case or a table
  derived from the main case. Keep comparable inputs and settings fixed where practical. Name the
  affected variants and the conditions under which the behavior applies.
- Give the exact runnable command or existing test selector, including relevant parameters,
  target, and toolchain version. Show the observed output for each executed side and label any
  source-derived side as inferred. A call fragment by itself leaves input and output unspecified.
- For each verified finding, cite its own runnable case or the specific walkthrough case that
  demonstrates it. Show the decisive output and cite the contract used to judge it.

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
- [ ] Documentation files ≤1000 lines (split if >1000 lines)
- [ ] `.claude/rules/*.md` ≤200 lines each **and** ≤2500 lines directory-wide
- [ ] Skill `SKILL.md` ≤200 lines; skill reference/support files ≤500 lines
- [ ] C++ implementation matches Python bindings
- [ ] Type stubs (`.pyi`) match actual API
- [ ] Docstrings complete and accurate
- [ ] Referenced files still exist
- [ ] Pass documentation follows the numbering convention in `docs/en/dev/passes/index.md` and preserves relative pipeline order (see `../../rules/pass-doc-ordering.md`)

**See [documentation-length.md](../../rules/documentation-length.md) for length guidelines**

### 4. Testing Standards

- [ ] Tests use **pytest** (not `unittest.TestCase`)
- [ ] All test verification uses **`assert`** (not `print`)
- [ ] No `unittest` imports (`unittest.TestCase`, `self.assertEqual`, `self.assertRaises`)
- [ ] Test files are located exclusively in the `tests/` directory
- [ ] pytest fixtures for setup/teardown (not `setUp()`/`tearDown()`)
- [ ] `pytest.raises()` for exception testing

### 5. Multi-Language Documentation Sync

- [ ] If English docs (`docs/en/dev/`) modified, corresponding `docs/zh/dev/` files updated or follow-up created
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

Use these four sections in order. Follow
[Explaining a Bug, Issue, or Failure](../../rules/explaining-problems.md) when tracing a defect.

1. **PR overview**. Explain the motivation and the previous workflow, what changes for users or
   callers, and the supported scope. Name affected entry points and material constraints.
2. **Concrete cases**. Walk through the before/after behavior using the cases above. Explain what
   each case demonstrates and show the evidence, including a relevant boundary when it clarifies
   the contract. Include these cases even when there are no review findings.
3. **Implementation path**. Explicitly list the passes, IR, and interfaces the PR adds or
   modifies. Name the affected passes, IR nodes, operators, types, attributes, or contracts, and
   frontend or internal interfaces as applicable. Give the concrete change and location for each
   item. Mark categories with no changes and identify existing stages reused by the new path.
   Follow the same representative input from the entry point to the result. Explain what each
   decisive step checks or transforms and why it matters to this case. For a defect, locate the
   first divergence and connect that decision to the symptom.
4. **Review conclusion**. List verified findings from highest to lowest severity. Give each
   finding's location, impact, and concrete correction, and refer to its reproduction above.
   List **Unverified** observations with the attempted command and missing evidence.
   Record the reviewed SHA for a PR, build and test results, and verification limits. Base any
   verdict on the verified findings and state the limits on coverage.

## GitHub Review Comments

Prefix each GitHub review comment that requires author action with the literal Unicode text
**&#x5F85;&#x89E3;&#x51B3;**. Keep the thread unresolved. Resolve it only after verifying the fix on the updated PR head.

## Decision Criteria

- **PASS** applies when the reviewed scope was verified and only minor suggestions remain.
- **WARNINGS** applies when verified non-blocking issues warrant attention or verification coverage is limited.
- **FAIL** requires a verified issue that blocks the change.

## Key Focus Areas

1. **Code Quality**: Style, error handling (`CHECK` vs `INTERNAL_CHECK`), PyPTO exceptions (not C++), no debug code, error messages with context
2. **Pass Complexity**: All passes must be O(N log N) or better — no nested full scans over IR node collections, no linear scans for lookups
3. **Python Style**: `@overload` for multiple signatures (not `Union`), modern type syntax (`list[int]` not `List[int]`), f-strings, Google-style docstrings, type hints on public APIs
4. **Testing Standards**: pytest only (no `unittest`), `assert` for verification (no `print`), `pytest.raises()` for exceptions, tests only in `tests/`
5. **Documentation**: Alignment with code changes, examples still work, file lengths (≤1000 for docs; ≤200 per rule file and ≤2500 across `.claude/rules/`; ≤200 for `SKILL.md` and ≤500 for skill reference files), pass documentation follows the convention in `docs/en/dev/passes/index.md` and preserves relative pipeline order
6. **Cross-Layer Sync**: C++ headers, Python bindings, and type stubs must all be updated together
7. **Commit Content**: Only relevant changes, no artifacts, no sensitive data, no AI co-author lines, no hardcoded absolute paths
8. **Multi-Language Doc Sync**: When English docs (`docs/en/dev/` or `README.md`) are modified, verify corresponding `docs/zh/` and `README.zh-CN.md` are also updated or flagged
