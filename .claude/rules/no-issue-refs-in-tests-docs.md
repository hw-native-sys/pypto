# No Issue References in Tests and Docs

## Core Principle

**Test cases and documentation must describe behavior, never reference issue/PR numbers.**

Issue numbers are meaningless to readers without GitHub access and rot as trackers migrate. The
test name and docstring must stand alone: a reader should understand what scenario is covered
and why it matters without opening the issue.

## Rules

1. **Test names describe the scenario**, not the ticket that prompted it:

   ```python
   # ❌ Bad
   def test_issue_1734_repro_compiles(self): ...
   def test_fix_1717(self): ...
   class TestIssue1524: ...

   # ✅ Good
   def test_loop_carried_reassign_with_dim_alias_compiles(self): ...
   def test_hoisted_temp_materialized_in_single_stmt_loop_body(self): ...
   class TestDynamicLocalTensorMetadata: ...
   ```

2. **Test docstrings explain the behavior under test** — actual vs. expected, the failure mode
   being prevented. No "regression for issue #N" phrasing.

3. **User and developer docs (`docs/`)** describe features and behavior — no issue/PR numbers.

4. **Identifiers inside tests** (helper function names, fixtures, module-level vars) follow the
   same rule — name them after the scenario, not the ticket.

## Where Issue References Belong

| Artifact | Issue reference allowed? |
| -------- | ------------------------ |
| Commit message | ✅ Yes — `Fixes #N` is the linkage mechanism |
| PR description | ✅ Yes |
| Test names, docstrings, fixtures | ❌ No |
| `docs/` content | ❌ No |
| Code comments | Avoid — explain the constraint itself; the commit carries the issue link |

## Migration

Existing names referencing issue numbers are renamed when touched — do not mass-rename
otherwise.
