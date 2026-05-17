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

For each commit, classify by subject **prefix and content**:

| Prefix / pattern | External? | Action |
| ---------------- | --------- | ------ |
| `feat(language)`, `feat(distributed)`, `feat(runtime)`, `feat(ir)` exposing DSL/IR op | Yes | Include |
| `feat:` with user-visible additions | Yes | Include |
| `fix(runtime)` / `fix(language)` changing public default or signature | Yes | Include |
| `feat`/`fix` strictly inside `src/` or `passes/` with no DSL / bindings / runtime API change | **No** | Skip |
| `refactor`, `chore`, `test`, `docs`, `ci` | **No** | Skip |

When uncertain, inspect `git show --stat <hash>` and look for changes under:

- `python/pypto/language/`
- `python/pypto/distributed/`
- `python/pypto/runtime/`
- `python/pypto/pypto_core/*.pyi`
- new bindings in `python/bindings/`

If the diff only touches `src/` or `include/` without altering any of the above, treat as internal.

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
- [ ] Each commit classified external vs internal
- [ ] Before/after example extracted for every external commit
- [ ] Author + theme bucket assigned per entry
- [ ] Overview table + owner index + migration table all present
- [ ] File written to the requested path; summary reported back
