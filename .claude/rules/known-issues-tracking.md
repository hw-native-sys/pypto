# Known Issues Tracking

## Overview

During any task, log encountered defects or system problems to `KNOWN_ISSUES.md` in the project root. This captures issues even when they are unrelated to the current task.

## When to Log

**Log an issue when you encounter:**

- Unexpected behavior, crashes, or errors in the system
- Code defects discovered while reading or modifying code
- Build system quirks or environment issues
- API inconsistencies or missing validation
- Documentation inaccuracies found incidentally

**Do NOT log:**

- Issues you are actively fixing as part of the current task
- Known limitations already documented in `docs/`
- User errors or misconfigurations

## File Format

`KNOWN_ISSUES.md` only contains **unresolved** issues. Resolved issues are removed entirely.

```markdown
# Known Issues

## [Short Title 1]

- **Date**: YYYY-MM-DD
- **Found during**: [brief context of what task you were working on]
- **Description**: [clear description of the problem]
- **Location**: [file path(s) and line number(s) if applicable]
- **Severity**: low | medium | high

---

## [Short Title 2]

- **Date**: YYYY-MM-DD
- **Found during**: [brief context of what task you were working on]
- **Description**: [clear description of the problem]
- **Location**: [file path(s) and line number(s) if applicable]
- **Severity**: low | medium | high
```

Each issue is separated by `---`. Keep descriptions concise but include enough context for someone to understand and reproduce the problem.

## How to Log

1. Read `KNOWN_ISSUES.md` first (create if it doesn't exist)
2. Check the issue is not already logged (avoid duplicates)
3. Append the new issue using the format above
4. Continue with the current task (do not fix the logged issue now)

## On Task Completion

**Before finishing any task, revisit `KNOWN_ISSUES.md`:**

1. Read all entries
2. Check if any were resolved by the current task's changes — **remove resolved entries** from the file
3. Present remaining issues to the user as a summary
4. Hint: "You may want to create GitHub issues for these using `/create-issue` and selecting from known issues"

**All entries in `KNOWN_ISSUES.md` are unresolved by definition.** When an issue is resolved, delete its entire section from the file. Do NOT mark as resolved — just remove it.

**Do NOT ask the user to fix these issues now** — just inform them.

## Important

- `KNOWN_ISSUES.md` is in `.gitignore` - it is a local-only tracking file
- Each developer's file is independent; it does not get shared via git
- Use `/create-issue` and select from known issues to promote an entry to a proper GitHub issue
