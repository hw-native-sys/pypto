---
name: fix-pr
description: Fix GitHub PR issues — address review comments and resolve CI failures in a loop until the PR is fully clean. Fetches CI errors online and triages review feedback. Use when fixing PR problems, addressing review comments, or resolving CI failures.
---

# Fix PR Workflow

Fix PR issues (review comments, CI failures) in a loop until the PR is fully clean.

## Input

Accept PR number (`123`, `#123`), branch name (`feature-branch`), or no argument (uses current branch).

## Workflow Overview

```text
Step 1: Match input to PR
        │
        ▼
┌─► Step 2: Detect issues (comments + CI)
│       │
│       ├─ All checks green AND no issues → Done ✓
│       │
│       ▼
│   Step 3: Fetch & classify all issues
│       │
│       ▼
│   Step 4: Get user confirmation
│       │
│       ▼
│   Step 5: Fix selected issues
│       │
│       ▼
│   Step 6: Commit, push, resolve threads
│       │
│       ▼
│   Step 7: Wait for CI + reviews (~10 min)
│       │
└───────┘
```

## Step 1: Match Input to PR

```bash
# PR number
gh pr view <number> --json number,title,headRefName,state

# Branch (default: current branch)
BRANCH=$(git branch --show-current)
gh pr list --head "$BRANCH" --json number,title,state
```

## Step 2: Detect Issues

Run in parallel:

```bash
# Derive repo owner/name dynamically (works across forks)
OWNER=$(gh repo view --json owner -q '.owner.login')
NAME=$(gh repo view --json name -q '.name')

# Check for unresolved review comments (including bot reviewers like CodeRabbit)
# Paginate with endCursor to handle PRs with >100 threads
gh api graphql \
  -F owner="$OWNER" \
  -F name="$NAME" \
  -F number=<NUMBER> \
  -f query='
query($owner: String!, $name: String!, $number: Int!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      reviewThreads(first: 100, after: $cursor) {
        nodes {
          id isResolved
          comments(last: 1) {
            nodes { id databaseId body author { login } path line }
          }
        }
        pageInfo { hasNextPage endCursor }
      }
    }
  }
}'
# If pageInfo.hasNextPage is true, re-run with -F cursor="<endCursor>" until all
# threads are fetched. Accumulate nodes across pages before computing counts.

# Check CI status — includes all checks (CI, review bots, etc.)
gh pr checks <NUMBER>
```

Present a summary: "**Iteration N** — Found X unresolved comments and Y failed/pending checks."

**Exit condition:** All CI/review-bot checks are green AND no unresolved comments → report clean status and exit the loop. Pending checks do NOT count as clean — wait for them to complete.

## Step 3A: Fetch Review Comments

Filter to `isResolved: false` and classify:

| Category | Description | Examples |
| -------- | ----------- | -------- |
| **A: Actionable** | Code changes required | Bugs, missing validation, security issues |
| **B: Discussable** | May skip if follows `.claude/rules/` | Style preferences, premature optimizations |
| **C: Informational** | Resolve without changes | Acknowledgments, "optional" suggestions |

**Note:** Treat bot reviewer comments (e.g., CodeRabbit, Copilot) the same as human comments — classify by content, not author.

## Step 3B: Fetch CI Failures

```bash
# List failed checks
gh pr checks <NUMBER> --json name,state,link | jq '.[] | select(.state == "FAILURE")'

# For GitHub Actions checks, extract the run ID from the link field:
#   link format: https://github.com/<owner>/<repo>/actions/runs/<RUN_ID>/job/<JOB_ID>
RUN_ID=$(echo "$LINK" | grep -oP 'runs/\K[0-9]+')

# Fetch failed run logs online
gh run view "$RUN_ID" --log-failed
```

For each failure, extract:

- **Check name** (e.g., `build`, `lint`, `test`)
- **Error summary** — the relevant error lines from logs
- **Affected file/line** if identifiable from the log output

**Tips for fetching logs:**

- `gh run view <RUN_ID> --log-failed` gives only failed step logs — start here
- If logs are truncated, use `gh run view <RUN_ID> --log` and grep for `error`/`FAILED`
- For large logs, pipe through `tail -100` or grep to extract relevant sections
- **External checks** (non-GitHub Actions): no run ID exists — open the `link` URL directly to view logs from the external provider

## Step 4: Get User Confirmation

Present ALL issues in a numbered list:

```text
Review Comments:
  1. [A] src/foo.cpp:42 — Missing null check (reviewer: alice)
  2. [B] src/bar.py:15 — Style suggestion (reviewer: coderabbitai)
  3. [C] src/baz.h:8 — "Looks good" (reviewer: bob)

CI Failures:
  4. [CI] build — error: 'Foo' is not a member of 'pypto::ir'
  5. [CI] lint — python/pypto/ir/foo.py:10: F401 unused import
  6. [CI] test — FAILED tests/ut/ir/test_foo.py::test_bar - AssertionError
```

Ask user which to address/skip/discuss. Recommend addressing A and CI items.

**First iteration:** Always ask for confirmation.
**Subsequent iterations:** If the user said "address all" previously, apply the same policy to new issues of the same category without re-asking. Only ask again for genuinely new or ambiguous issues.

## Step 5: Fix Issues

### For review comments

1. Read affected files with Read tool
2. Make changes with Edit tool

### For CI failures

1. Analyze the error from fetched logs — do NOT reproduce locally unless online logs are insufficient
2. Read affected files
3. Fix the root cause
4. If a fix might introduce new issues, run relevant tests locally to verify

### Commit

Commit using `/git-commit` skill, skip testing and review for minor fixes.

**Commit message format:**

```text
fix(pr): resolve issues for #<number>

- Fixed null check (review comment #1)
- Fixed missing include (CI build error)
- Removed unused import (CI lint error)
```

### Push

```bash
git push
```

## Step 6: Resolve Comment Threads

Reply using `gh api repos/:owner/:repo/pulls/<number>/comments/<comment_id>/replies -f body="..."` then resolve thread with GraphQL `resolveReviewThread` mutation.

**Response templates:**

- Fixed: "Fixed in `<commit>` - description"
- Skip: "Current code follows `.claude/rules/<file>`"
- Acknowledged: "Acknowledged, thank you!"

## Step 7: Wait and Re-check

After pushing fixes, wait for CI and review bots to process the new commit.

```bash
# Wait ~10 minutes for CI and review bots to run
sleep 600

# Then loop back to Step 2
```

**While waiting:** Inform the user that you're waiting for CI/reviews to complete. Use `gh pr checks <NUMBER>` to poll status before the full wait expires — if all checks finish early, proceed immediately.

**Loop safeguards:**

- **Max iterations: 5** — after 5 fix-wait cycles, stop and report remaining issues to the user
- **Stuck detection** — if the same issue reappears after a fix attempt, flag it to the user rather than retrying blindly
- **User can exit** — the user can interrupt at any time; respect interruptions

## Best Practices

| Area | Guidelines |
| ---- | ---------- |
| **CI errors** | Fetch logs online first; only reproduce locally as last resort |
| **Bot reviews** | Treat CodeRabbit/Copilot comments like human reviews — classify by content |
| **Analysis** | Reference `.claude/rules/`; when unsure → Category B |
| **Changes** | Read full context; minimal edits; follow project conventions |
| **Communication** | Be respectful; explain reasoning; reference rules |

## Error Handling

| Error | Action |
| ----- | ------ |
| PR not found | `gh pr list`; ask user to confirm |
| Not authenticated | "Run: `gh auth login`" |
| Unclear comment | Mark Category B for discussion |
| CI logs too large | Grep for `error`, `FAILED`, `fatal` |
| CI logs unavailable | Fall back to local reproduce |
| Max iterations reached | Stop loop, report remaining issues |
| Same failure persists | Flag to user, do not retry same fix |

## Checklist

- [ ] PR matched and validated
- [ ] Review comments and CI status fetched
- [ ] ALL issues presented to user for selection
- [ ] Code changes made and committed (use `/git-commit`)
- [ ] Changes pushed to remote
- [ ] Review comment threads replied to and resolved
- [ ] Waited for CI/reviews and re-checked
- [ ] Loop exited: all clean OR max iterations reached
