# Problem Handling and Known Issues Tracking

## Core Principle

**When encountering technical problems, classify them as blocking or non-blocking and act accordingly.** Never silently work around, ignore, or make assumptions about technical problems.

```text
Technical problem encountered
├─ Does it block the current task?
│  ├─ YES → Stop. Inform the user. Wait for their decision before continuing.
│  └─ NO  → Log to KNOWN_ISSUES.md. Continue with the current task.
```

## Blocking Problems

**A problem is blocking when you cannot make meaningful progress on the current task without resolving it.**

Examples: build failure preventing testing, ambiguous requirements, API behaving differently than documented, test failure that may indicate your change is wrong, missing information needed to complete the task.

**What to do:**

1. **Stop** — do not attempt workarounds or make assumptions
2. **Describe the problem clearly** — what happened, what you expected, and why it blocks progress
3. **Present options** — lay out possible paths forward with trade-offs
4. **Wait for the user's decision** — do not pick an option and continue on your own

**When unsure if blocking:** err on the side of asking — a brief question costs less than a wrong assumption. If the problem might affect correctness, treat it as blocking.

## Non-Blocking Problems (Known Issues)

**A problem is non-blocking when you can complete the current task correctly despite the issue.** Log it to `KNOWN_ISSUES.md` and continue.

**Always write to the main repository's `KNOWN_ISSUES.md`**, even when working in a git worktree. Use `git worktree list` to find the main repo root (the first entry).

### When to Log

- Unexpected behavior, crashes, or errors in the system
- Code defects discovered while reading or modifying code
- Build system quirks or environment issues
- API inconsistencies or missing validation
- Documentation inaccuracies found incidentally

**Do NOT log:** issues you are actively fixing, known limitations already in `docs/`, or user misconfigurations.

### File Format

`KNOWN_ISSUES.md` only contains **unresolved** issues. Resolved issues are removed entirely.

```markdown
# Known Issues

## [Short Title]

- **Date**: YYYY-MM-DD
- **Found during**: [brief context of what task you were working on]
- **Description**: [actual behaviour, expected behaviour, why it matters]
- **Example / Repro**: [smallest artefact that surfaces the issue — see "Entry Quality" below; use `N/A` only for purely descriptive issues]
- **Location**: [file path(s) and line number(s) if applicable]
- **Severity**: low | medium | high

---
```

### Entry Quality

Each entry must be **self-contained** — a future reader (you in two months, or the user filing it as a GitHub issue) should understand the problem without re-deriving it from memory.

- **Description**: name the actual vs. expected behaviour and the consequence. ✅ "ConvertSeq doesn't guard mid-body yields, so malformed SeqStmts survive SSA conversion when verification is off" beats ❌ "ConvertSeq has a bug".
- **Example / Repro**: include the smallest concrete artefact that surfaces the issue. Pick whichever fits:
  - A failing test name + the bottom of the traceback for runtime bugs
  - A short code snippet (DSL / IR / C++) showing the wrong output for codegen / printer / pass issues
  - The exact CLI command + error message for build or tooling issues
  - A grep query + counts for inventory-style observations (e.g. `grep -rE 'INTERNAL_CHECK\(' src/ir | wc -l   # 91`)

**Note on `N/A`:** Mark as `N/A` only when the issue is purely descriptive (doc gap, naming concern) — that signals "considered, not forgotten" rather than "skipped".

If you cannot produce a concrete example, treat that as a signal the issue may not yet be well-understood — flag it to the user before logging.

### How to Log

1. Determine the main repo root (`git worktree list` — first entry)
2. Read `KNOWN_ISSUES.md` (create if it doesn't exist)
3. Check the issue is not already logged (avoid duplicates)
4. Append the new issue using the format above — verify it meets the "Entry Quality" bar before saving
5. Continue with the current task (do not fix the logged issue now)

### Writing from a worktree

While a session is isolated in a worktree, Claude Code **blocks `Edit` / `Write` /
`NotebookEdit` against the main checkout** ([How Claude Code enforces
isolation](https://code.claude.com/docs/en/worktrees#how-claude-code-enforces-isolation)).
Reads still work; only writes are refused, and there is no opt-out. Do **not**
create a worktree-local `KNOWN_ISSUES.md` as a workaround — that fragments the
file, which is exactly what the "main repo root" rule exists to prevent.

Use a Bash command instead, run from the worktree cwd. The isolation checks test
a command's *working directory* and *git redirects* — not file writes by
absolute path — so a plain write to the main repo's `KNOWN_ISSUES.md` passes:

```bash
python3 - <<'PY'
p = '/abs/path/to/main-repo/KNOWN_ISSUES.md'
entry_text = """
## Short Title

- **Date**: YYYY-MM-DD
- **Found during**: [task context]
- **Description**: [actual vs. expected behaviour, why it matters]
- **Example / Repro**: [smallest artefact that surfaces it]
- **Location**: [file path(s) and line number(s)]
- **Severity**: low

---
"""
open(p, 'a').write(entry_text)
PY
```

The heredoc is quoted (`<<'PY'`), so the entry text is passed through verbatim —
define it inside the snippet rather than relying on a shell variable.

Constraints that make this work:

- **Never `cd` into the main checkout**, and never point `git -C` / `--git-dir` /
  `GIT_DIR` / `GIT_WORK_TREE` at it — each is independently blocked.
- **Keep the command simple.** Compound commands (`&&`, `;`-chains, redirects)
  are refused as unverifiable, even when their effect would be legal.
- **Anchor edits on entry heading text, never line numbers — and require exactly
  one match.** The file is shared by every worktree of the repo and concurrent
  sessions may append to it, so headings are *not* guaranteed unique. Count the
  matches first and **abort without modifying the file** when zero or more than
  one heading matches; never rewrite the first match.
- **Back it up first** (`cp` to a scratch dir) before any in-place rewrite, and
  diff against the backup afterward to confirm only the intended hunk changed.
  The diff proves *something* changed, not that the *right* entry changed — it
  supplements the unique-match check above, it does not replace it.

## On Task Completion

**Before finishing any task, revisit `KNOWN_ISSUES.md`:**

1. Read all entries
2. Remove any entries resolved by the current task's changes
3. Present remaining issues to the user as a summary
4. Hint: "You may want to create GitHub issues for these using `/create-issue` and selecting from known issues"

**Do NOT ask the user to fix these issues now** — just inform them.

## Important

- `KNOWN_ISSUES.md` is in `.gitignore` — local-only tracking file
- Each developer's file is independent; it does not get shared via git
- **Never reference `KNOWN_ISSUES.md` or its entries in shared artifacts** — commit messages, PR descriptions, and GitHub issues must not name the file or quote its entries. It is local-only and per-developer, so external readers cannot see it. Describe the actual change, not the local tracking entry it resolves.
- **Always write to the main repo root**, never to a worktree's directory — from a
  worktree this requires Bash, since `Edit` is blocked (see "Writing from a worktree")
- Use `/create-issue` to promote an entry to a proper GitHub issue
