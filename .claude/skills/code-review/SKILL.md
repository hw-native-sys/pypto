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
2. Analyze the changed files against the checklist below. Trace affected callers, evaluate overall
   design soundness and possible alternative designs, verify documentation alignment, and check
   C++, Python bindings, and type stubs where applicable.
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

- Put a complete, runnable before/after DSL or public API case for the main change in the final
  reply as fenced code. Include imports, setup, input declarations and values, and the invocation.
  Show the exact edit or setting for each side and explain why the inputs expose the difference.
  For IR-only changes, show before/after IR directly in the reply with enough context to follow it.
- Cover materially different entry points, modes, or boundaries with a focused case or a table
  derived from the main case. Keep comparable inputs and settings fixed where practical. Name the
  affected variants and the conditions under which the behavior applies.
- Give the exact command or test selector with relevant parameters, target, and toolchain version.
  Show the observed output or error for each executed side and label source-derived behavior as
  inferred. Use links, attachments, and earlier messages as supporting evidence for the inline case.
- For each verified behavior finding, show its runnable case in the reply or reuse one shown there.
  Show the decisive output and cite the contract used to judge it.

## Review Checklist

### 1. Design Soundness

- [ ] Targets the root cause at the right layer/pass rather than patching downstream symptoms
- [ ] Invariants, ownership boundaries, and failure modes are minimal, symmetric, and well-scoped
- [ ] Reuses existing mechanisms without unnecessary cross-module coupling or brittle assumptions
- [ ] Alternative designs evaluated (propose ≥1 viable alternative and compare trade-offs in correctness, complexity, and maintainability)

### 2. Code Quality

- [ ] Follows project style and conventions
- [ ] No debug code or commented sections
- [ ] No TODOs/FIXMEs unless documented
- [ ] Proper use of `CHECK` vs `INTERNAL_CHECK`
- [ ] PyPTO exceptions (not C++ exceptions)
- [ ] Clear, descriptive names
- [ ] Appropriate comments for complex logic
- [ ] Linter errors fixed (not suppressed)
- [ ] **Pass complexity ≤ O(N log N)**: No nested full scans over IR node collections, no linear scans for lookups (see [`pass-complexity.md`](../../rules/pass-complexity.md))

### 3. Python Style (see `python-style.md` for full details)

- [ ] `@overload` used for functions with multiple distinct call signatures (not `Union`)
- [ ] Modern type syntax: `list[int]`, `dict[str, Any]` (not `List`, `Dict`)
- [ ] f-strings for all string formatting (no `.format()` or `%`)
- [ ] Google-style docstrings (Args/Returns/Raises)
- [ ] Type hints on all public API parameters and return types

### 4. Documentation Alignment

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

### 5. Testing Standards

- [ ] Tests use **pytest** (not `unittest.TestCase`)
- [ ] All test verification uses **`assert`** (not `print`)
- [ ] No `unittest` imports (`unittest.TestCase`, `self.assertEqual`, `self.assertRaises`)
- [ ] Test files are located exclusively in the `tests/` directory
- [ ] pytest fixtures for setup/teardown (not `setUp()`/`tearDown()`)
- [ ] `pytest.raises()` for exception testing

### 6. Multi-Language Documentation Sync

- [ ] If English docs (`docs/en/dev/`) modified, corresponding `docs/zh/dev/` files updated or follow-up created
- [ ] If `README.md` modified, `README.zh-CN.md` updated or flagged
- [ ] Chinese translations preserve all code examples untranslated
- [ ] Cross-references in Chinese docs point to Chinese versions
- [ ] Technical terms use dual notation: "Chinese (English)"

### 7. Commit Content

- [ ] Only relevant changes included
- [ ] No build artifacts (`build/`, `*.o`, `*.so`)
- [ ] No sensitive information (tokens, keys, absolute paths)
- [ ] No temporary files or ad-hoc/example scripts
- [ ] No AI co-author lines (`Co-Authored-By: Claude`, etc.)
- [ ] Changes are cohesive and related

## Cross-Layer Consistency

When APIs change, synchronize all three layers ([cross-layer-sync.md](../../rules/cross-layer-sync.md)): C++ headers (`include/pypto/`), Python bindings (`python/bindings/`, snake_case), and type stubs (`python/pypto/pypto_core/`, matching signatures & `@overload`).

## Common Issues to Flag

- **Symptom patching / over-coupling**: Downstream workaround instead of fixing the root layer, asymmetric branch handling, or duplicating existing pass/utility logic
- **Debug code**: `std::cout << "DEBUG"`, `print()` left in, `// TODO: fix later`
- **Wrong error macro**: `CHECK` for internal invariants (use `INTERNAL_CHECK`), or vice versa
- **C++ exceptions**: `throw std::runtime_error(...)` instead of `pypto::ValueError`
- **Missing bindings**: C++ method added but no Python binding or type stub
- **`Union` instead of `@overload`**: Function with distinct call patterns uses `Union` args
- **Legacy type syntax**: `List[int]`, `Dict[str, Any]` instead of `list[int]`, `dict[str, Any]`
- **Outdated docs / build artifacts**: Docs showing old usage; `build/`, `__pycache__/`, `*.pyc` in staged files
- **Non-pytest / print tests**: `unittest.TestCase`, `self.assert*()`, or `print(result)` instead of pytest + `assert`
- **Forbidden content / vague errors**: `Co-Authored-By` AI lines, hardcoded absolute paths, `raise ValueError("Invalid")` without context
- **Quadratic pass complexity**: Nested full scans over the same/global IR node collection, or linear scans instead of indexed lookups inside a pass traversal

## Output Format

Use these four sections in order. Follow
[Explaining a Bug, Issue, or Failure](../../rules/explaining-problems.md) when tracing a defect.

1. **PR overview**. Explain the motivation and the previous workflow, what changes for users or
   callers, and the supported scope. Name affected entry points and material constraints.
2. **Concrete cases**. Show the code, inputs, invocation, and results directly in the reply, even
   when there are no findings. Include a relevant boundary when it clarifies the contract. Before
   sending, check that each case can be reproduced from the reply alone.
3. **Implementation path**. Explicitly list the passes, IR, and interfaces the PR adds or
   modifies. Name the affected passes, IR nodes, operators, types, attributes, or contracts, and
   frontend or internal interfaces as applicable. Give the concrete change and location for each
   item. Mark categories with no changes and identify existing stages reused by the new path.
   Follow the same representative input from the entry point to the result. Explain what each
   decisive step checks or transforms and why it matters to this case. For a defect, locate the
   first divergence and connect that decision to the symptom.
4. **Review conclusion**. List verified findings from highest to lowest severity (location, impact,
   concrete correction, and reference to reproduction above), plus **Unverified** observations with
   missing evidence. Assess overall design soundness (layer placement, invariants, proportionality)
   and propose ≥1 plausible alternative design with trade-offs against the chosen approach.
   Record the reviewed SHA, build and test results, verification limits, and final verdict.

## GitHub Review Comments

Prefix each GitHub review comment that requires author action with the literal Unicode text
**&#x5F85;&#x89E3;&#x51B3;**. Keep the thread unresolved. Resolve it only after verifying the fix on the updated PR head.

## Decision Criteria

- **PASS**: Reviewed scope verified and only minor suggestions remain.
- **WARNINGS**: Verified non-blocking issues warrant attention or verification coverage is limited.
- **FAIL**: Verified issue blocks the change.

## Key Focus Areas

1. **Design Soundness**: Root-cause layer placement, minimal/symmetric invariants, low coupling, and alternative design trade-offs
2. **Code Quality & Pass Complexity**: Style, `CHECK` vs `INTERNAL_CHECK`, PyPTO exceptions, O(N log N) pass complexity
3. **Python & Testing Standards**: Modern typing (`list[int]`, `@overload`), f-strings, Google docstrings, pytest + `assert` in `tests/`
4. **Cross-Layer & Documentation Sync**: C++/bindings/stubs in sync, docs length limits (≤200 `SKILL.md`, ≤1000 docs), EN/ZH alignment
5. **Commit Hygiene**: Cohesive scope, no artifacts/secrets/absolute paths, no AI co-author lines
