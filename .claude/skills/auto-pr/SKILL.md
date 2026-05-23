---
name: auto-pr
description: Create a GitHub PR then autonomously loop on CI failures and review comments until the PR is fully green. Combines branch prep, PR creation, and a hands-off fix loop. Use when the user wants to ship a PR end-to-end, auto-fix a PR until green, or create-and-fix a PR in one go.
argument-hint: [pr-number]
---

# Auto PR Workflow

Create a PR (or locate an existing one), then run a **fully automatic** loop that
fixes CI failures and review comments until everything is green.

This skill composes two existing skills — keep them as the source of detail:

- `github-pr` — branch prep, rebase, push, PR creation (Phase A)
- `fix-pr` — issue detection, classification, thread resolution (Phase B)

This skill **overrides their interactive steps** with autonomous behavior — it
never pauses to ask which issues to address.

## Input

| Input | Behavior |
| ----- | -------- |
| no argument | Phase A (create/locate PR) → Phase B (fix loop) |
| PR number (`123`, `#123`) | Skip Phase A, run Phase B on that PR |

## Task Tracking

1. Phase A — create or locate PR
2. Phase B — detect & classify issues (loop)
3. Phase B — auto-fix & push (loop)
4. Phase B — resolve threads (loop)
5. Phase B — wait & re-check (loop)
6. Final report

## Phase A — Create or Locate the PR

Follow `github-pr` Steps 1–6 (prepare branch & commit, check existing PR, fetch
upstream, rebase, push with `--force-with-lease`, create PR), **with one override:**

- `github-pr` Step 2 says "if a PR already exists, display it and exit."
  Here, **do not exit** — record the PR number and proceed to Phase B.

If invoked with a PR number, skip Phase A entirely. Carry the PR number into
Phase B as `<NUMBER>`.

## Phase B — Autonomous Fix Loop

Repeat Steps 1–5 until the **exit condition** is met or the iteration cap (8) is hit.

### Step 1: Detect & Classify Issues

Run `fix-pr` Step 2 (detect) and Step 3 (fetch & classify) verbatim — fetch
review threads, CodeRabbit out-of-diff findings, and CI status. **Observe every
shell pitfall listed in `fix-pr`** (no `gh api graphql | python3`, no `--jq` with
`$`, paginate review threads, etc.).

Classify each item:

| Class | Definition | Loop action |
| ----- | ---------- | ----------- |
| **CI** | Failed/errored GitHub check | Auto-fix |
| **A — Actionable** | Clear bug, missing check, security/correctness issue | Auto-fix |
| **B — Style nit** | Pure formatting/naming/style preference | Auto-skip, cite `.claude/rules/<file>` |
| **C — Informational** | Ack, "optional", non-request | Resolve, no code change |
| **D — Judgement call** | Design/architecture/ambiguous — needs human judgement | **Defer** — do not touch code, do not resolve thread |

Pending checks are NOT clean. Treat bot reviewers (CodeRabbit, Copilot, Gemini)
like humans — classify by content, not author. When unsure between B and D,
choose D (defer rather than guess).

### Step 2: Auto-Fix (no confirmation)

This skill runs **fully automatically** — never pause to ask which issues to address.

- **CI + Class A:** Read affected files, make minimal edits. For CI, analyze logs
  online first; reproduce locally only as a last resort.
- **Class B / C:** No code change.
- **Class D:** Skip — collect into the deferred list for the Final Report.

If an iteration produces zero code changes and issues remain (all D, or a CI
failure with no actionable fix), do not loop pointlessly — go to Final Report.

### Step 3: Commit & Push

Commit via `/git-commit` — let it apply its own review/testing logic based
on the changed file types; do not override.
Message: `fix(pr): resolve issues for #<NUMBER>` + bullet list of fixes.
Then `git push`.

### Step 4: Resolve Threads

For threads addressed **this iteration only**:

- Class A fixed → reply "Fixed in `<commit>` — `<desc>`", then `resolveReviewThread`.
- Class B skipped → reply "Follows `.claude/rules/<file>`", then resolve.
- Class C → reply "Acknowledged!", then resolve.
- Class D deferred → **leave the thread open, do not reply.**

Out-of-diff findings have no thread ID — note the fix in the commit message only;
CodeRabbit re-scans on the next push.

### Step 5: Wait & Re-Check

Wait for all checks to finish using `gh`'s built-in watch — **never a manual
`sleep`/poll loop**:

```bash
gh pr checks <NUMBER> --watch
```

Once it returns, every check is complete. Identify failed runs via
`gh pr checks <NUMBER> --json name,state,link` and extract `<RUN_ID>` from the
link (`/runs/<RUN_ID>/job/...`) — exactly as `fix-pr` Step 3 documents. Then
`gh run view <RUN_ID> --log-failed` is safe to read. For external (non-Actions)
checks there is no run ID; open the `link` URL directly. Then loop back to Step 1.

### Exit Condition

Exit the loop when **any** of:

- All checks green AND no unresolved Class A/B/C comments (Class D may remain open), OR
- Iteration cap (8) reached, OR
- An item is **stuck** — same CI failure or comment unfixed after 2 consecutive
  iterations. Stop retrying it; report it.

## Final Report

Report to the user:

- ✅ / ❌ final CI status per check.
- **Deferred Class D comments** — list each with `path:line`, reviewer, and a
  one-line summary so the user can decide. These threads were intentionally left open.
- **Stuck issues** — anything retried without success; suggest manual follow-up.
- Iteration count and PR URL.

## Safeguards

| Situation | Action |
| --------- | ------ |
| PR already exists (Phase A) | Reuse it — do not exit, proceed to Phase B |
| Same failure 2× in a row | Mark stuck, stop retrying it |
| Iteration cap (8) hit | Stop, report remaining issues |
| Zero-change iteration with issues left | Stop — nothing more to automate |
| Rebase conflict in Phase A | Resolve per `github-pr` Step 4; if stuck, ask user |
| CI run still in progress | `gh pr checks --watch`; never read `--log-failed` early |

## Checklist

- [ ] PR created or located (Phase A — do not exit on existing PR)
- [ ] Issues detected and classified A/B/C/D each iteration
- [ ] CI + Class A auto-fixed; Class D deferred (no code change, thread left open)
- [ ] Fixes committed via `/git-commit` and pushed
- [ ] Resolved threads replied to and closed (except deferred D)
- [ ] Loop exited: green, stuck, or cap reached
- [ ] Final report lists CI status, deferred comments, stuck issues, PR URL
