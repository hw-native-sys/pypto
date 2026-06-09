---
name: weekly-changelog
description: Generate a weekly changelog markdown file summarizing external API and feature changes from git commits in a date range. Extracts before/after Python examples per commit, groups by theme (DSL / distributed / runtime / IR deprecations), and attributes each change to its author. Use when the user asks for a weekly report, changelog, commit summary, or interface-change digest.
---

# Weekly Changelog Generator

## Overview

Produces a markdown report of **externally visible** PyPTO changes over a date range (typically one week). Each entry has a one-line summary, before/after Python example, classification (new / replace / deprecate), and the implementer's name so reviewers can find the owner. Internal refactors / chores / CI / internal fixes are excluded by default.

## Step 1: Collect Parameters

Ask the user with `AskUserQuestion`:

| Question | Header | Options |
| -------- | ------ | ------- |
| Date range? | Range | This week / Last week / Custom (YYYY-MM-DD..YYYY-MM-DD) |
| Output path? | Output | `./weekly-<start>-to-<end>.md` (Recommended) / `/tmp/...` / custom |
| Language? | Lang | Chinese / English |
| Scope? | Scope | External APIs only (Recommended) / All commits |

If the user already provided values in their request, skip the corresponding question.

## Step 2: List Commits in Range

```bash
git log --since="<start> 00:00" --until="<end> 23:59" \
        --pretty=format:"%h | %an | %s" --date=short
```

Capture `<hash> | <author> | <subject>` for every commit.

## Step 3: Classify Commits

> **The commit prefix is a hint, NOT the gate.** PyPTO routinely ships
> user-facing DSL/runtime surface behind `fix(codegen)`, `fix(ir)`, and
> `refactor(...)` prefixes. Classifying by prefix alone under-reports the most
> important changes. The **authoritative signal is the diff touching a public
> export surface** — run the surface check over *every* commit in range,
> regardless of prefix.

### 3a. Public-surface pre-pass (authoritative — run on ALL commits)

Before applying any prefix heuristic, scan every commit's diff for additions or
signature changes to these public surfaces. Any hit ⇒ **treat as external**,
even if the prefix is `fix`/`refactor`/`chore`:

- `python/pypto/language/__init__.py` and its `__all__`
- `python/pypto/language/op/*.py` (new/changed `pl.*` ops)
- `python/pypto/runtime/__init__.py` and its `__all__` (exported classes)
- `python/pypto/distributed/__init__.py` (`pld.*` surface)
- `python/pypto/pypto_core/*.pyi` (type stubs — but a `passes.pyi`-only change is usually an internal verifier)
- new bindings in `python/bindings/`
- a `RunConfig` field add/type-change in `python/pypto/runtime/runner.py`

Fast batch scan over the range (adjust `<from>..<to>`):

```bash
for h in $(git log --since=... --until=... --pretty=%h); do
  hit=$(git show --stat $h | grep -E \
    "language/__init__|language/op/.*\.py|runtime/__init__|runtime/runner\.py|distributed/__init__|pypto_core/.*\.pyi|python/bindings/")
  [ -n "$hit" ] && { echo "=== $h ==="; echo "$hit"; }
done
```

For each hit, open the actual diff (`git show <hash> -- <file>`) to confirm it
adds/renames a public symbol (not just an internal edit to that file).

**Known real-world misses (do not repeat):** `fix(codegen): ...selective dump`
introduced `pl.dump_tag` / `dumps=` and leveled `RunConfig.enable_dump_tensor`;
`refactor(runtime): Worker ABC` changed `pypto.runtime` exports
(`ChipWorker` / `RegistrationHandle`); `feat(ir): ...transpose` changed a DSL
builder signature. All three would be skipped by prefix alone.

### 3b. Prefix heuristic (only for commits with NO public-surface hit)

| Prefix / pattern | External? | Action |
| ---------------- | --------- | ------ |
| `feat(language)`, `feat(distributed)`, `feat(runtime)`, `feat(ir)` exposing DSL/IR op | Yes | Include |
| `feat:` with user-visible additions | Yes | Include |
| `fix(runtime)` / `fix(language)` changing public default or signature | Yes | Include |
| `feat`/`fix` strictly inside `src/` or `passes/` with no DSL / bindings / runtime API change | **No** | Skip |
| `refactor`, `chore`, `test`, `docs`, `ci` with no 3a hit | **No** | Skip |

If the diff only touches `src/` or `include/` without altering any 3a surface, treat as internal.

## Step 4: Extract Before/After Per External Commit

For each external commit, in parallel batches of ~5, launch **Explore subagents** to gather:

1. One-sentence summary (Chinese or English per Step 1)
2. **Before** Python snippet (5–10 lines). For pure additions, write `None (new)` or show the prior workaround.
3. **After** Python snippet (5–10 lines), drawn from the PR description (`gh pr view <num>`) or new tests in `tests/ut/`.
4. Classification: new / replace / deprecate. Mark deprecations explicitly when a `DeprecationWarning` is emitted.

**Agent prompt template** (one agent per 3–5 commits):

```text
Investigate the user-facing Python interface changes in the following
commits. For each commit, output:
- One-sentence summary
- Before usage (minimal Python example)
- After usage (minimal Python example)
- Classification (new / replace / deprecate)
Working directory: <project root>
Commands: git show --stat <hash>; gh pr view <pr>; inspect
python/pypto/<area>/ and tests/ut/.
Keep each entry concise (< 120 words).
Commits: <list>
```

## Step 5: Assemble Markdown

Structure of the output file:

```markdown
# PyPTO Weekly: <start> ~ <end> (external features and interface changes)

> Only includes user-visible changes ... internal refactor / chore / ci / internal fix are not listed.

## Overview
| Commit | PR | Author | Topic | Type |

## Owner Index
| Owner | Commit count | Topics covered |

## 1. Python DSL and Operators
### 1.1 <title> (#<pr>)
- **Author**: <author>
- **Type**: new / replace / deprecate
- **Summary**: ...
**Before**: ```python ... ```
**After**: ```python ... ```

## 2. Distributed pld.* API
## 3. Runtime Configuration
## 4. IR Operators / Deprecation Notices
## 5. Migration Guide (deprecations aggregated)
| Old usage | Recommended | Notes |
```

Always include:

- The **Author** line per entry (use `git log --pretty=format:"%an"`).
- An **Owner index** table aggregating commits per author.
- A **Migration guide** table for any deprecation or default-value change.

Theme buckets — pick the four headings that match your commits; common ones:

| Bucket | Typical commits |
| ------ | --------------- |
| Python DSL & operators | `feat(language)`, `feat(ir)` new ops |
| Distributed `pld.*` | `feat(distributed)` |
| Runtime / RunConfig | `feat(runtime)`, `fix(runtime)` user-visible |
| IR & deprecation | RFC-driven type changes, `pl.*` deprecations |

Omit empty buckets.

## Step 6: Save and Report

Write the file to the agreed output path with `Write`. Confirm in chat: line count, commit count covered, deprecation count.

## Conventions

- **Author names** come from `git log` (`%an`), not from `Co-Authored-By` lines.
- **Language**: produce the entire file in the chosen language; do not mix.
- **Examples must be runnable-shaped** — copy-paste from PR descriptions / tests, not invent.
- **Before for pure additions**: write `None (new)` (or the Chinese equivalent when output language is Chinese). Do not fabricate a "before".
- **Mark deprecations explicitly** — the migration table at the end is the deliverable that protects users.

## Important Constraints

- **Never invent commits or PR numbers.** Only use what `git log` and `gh pr view` return.
- **Plan mode**: this workflow is read-only until Step 6. Safe to run during planning.
- **Scope discipline**: when scope is "external only", refusing to include a commit is correct behavior — record the skipped count in the final report.
- **Markdown file location policy**: This skill explicitly creates a markdown file outside `docs/`. Honor the user's chosen path; do not move it to `docs/`.

## Checklist

- [ ] Date range, output path, language, scope captured
- [ ] All commits in range listed with author
- [ ] Public-surface pre-pass (3a) run over EVERY commit, not just `feat` ones
- [ ] Each commit classified external vs internal (prefix never overrides a 3a hit)
- [ ] Before/after example extracted for every external commit
- [ ] Author + theme bucket assigned per entry
- [ ] Overview table + owner index + migration table all present
- [ ] File written to the requested path; summary reported back
