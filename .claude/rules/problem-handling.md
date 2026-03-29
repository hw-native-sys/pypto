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
- **Description**: [clear description of the problem]
- **Location**: [file path(s) and line number(s) if applicable]
- **Severity**: low | medium | high

---
```

### How to Log

1. Determine the main repo root (`git worktree list` — first entry)
2. Read `KNOWN_ISSUES.md` (create if it doesn't exist)
3. Check the issue is not already logged (avoid duplicates)
4. Append the new issue using the format above
5. Continue with the current task (do not fix the logged issue now)

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
- **Always write to the main repo root**, never to a worktree's directory
- Use `/create-issue` to promote an entry to a proper GitHub issue
