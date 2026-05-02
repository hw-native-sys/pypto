---
name: fix-pr
description: Fix GitHub PR issues — address review comments and resolve CI failures in a loop until the PR is fully clean. Fetches CI errors online and triages review feedback. Use when fixing PR problems, addressing review comments, or resolving CI failures.
argument-hint: [pr-number]
---

# Fix PR Workflow

Fix PR issues (review comments, CI failures) in a loop until the PR is fully clean.

## Task Tracking

Create tasks to track progress through this workflow:

1. Match input to PR
2. Detect & classify issues
3. Get user confirmation
4. Fix issues & push
5. Resolve comment threads
6. Re-check (loop until clean)

## Input

Accept PR number (`123`, `#123`), branch name, or no argument (uses current branch).

## Loop: Steps 1→7, repeat until clean or max 5 iterations

### Step 1: Match Input to PR

```bash
gh pr view <number> --json number,title,headRefName,state
# Or by branch:
BRANCH=$(git branch --show-current)
gh pr list --head "$BRANCH" --json number,title,state
```

### Step 2: Detect Issues (run in parallel)

```bash
OWNER=$(gh repo view --json owner -q '.owner.login')
NAME=$(gh repo view --json name -q '.name')

# Fetch review threads — save to file, then grep (see pitfalls below)
gh api graphql \
  -F owner="$OWNER" -F name="$NAME" -F number=<NUMBER> \
  -f query='
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      reviewThreads(first: 100) {
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
}' > /tmp/threads.json

# Count unresolved threads (use grep -o to count all occurrences, not lines)
grep -o '"isResolved":[[:space:]]*false' /tmp/threads.json | wc -l

# Paginate: if hasNextPage is true, re-run with -F cursor="<endCursor>" until done

# Also fetch CodeRabbit "outside diff range" findings — they live in review BODIES, not threads.
# Use --paginate so PRs with many reviews don't drop the latest CodeRabbit body.
gh api --paginate "repos/$OWNER/$NAME/pulls/<NUMBER>/reviews" > /tmp/reviews.json
# Markers in body: "Outside diff range comments" or "Some comments are outside the diff".
# After stripping the leading "> " (the [!CAUTION] callout makes everything a blockquote),
# the structure is nested <details>:
#   <summary>⚠️ Outside diff range comments (N)</summary><blockquote>
#     <summary>PATH (N)</summary><blockquote>
#       `LINE-RANGE`: severity-tags **Title** ... body ...
# Use Python (re + json) to walk the tree; surface (path, line_range, body) as pseudo-threads.

# Check CI status
gh pr checks <NUMBER>
```

**Shell pitfalls to avoid:**

- Do NOT pipe `gh api graphql` to `python3 -c` with `json.load(sys.stdin)` — `gh` may emit extra metadata that breaks JSON parsing with `JSONDecodeError: Extra data`
- Do NOT use `gh api graphql --jq` with `$` in filter expressions — `gh`'s jq processor interprets `$` as a jq variable sign, causing `Expected VAR_SIGN` errors even when shell quoting is correct
- Use `grep -c` for simple counts; save to a temp file first if complex parsing is needed

Present: "**Iteration N** — Found X unresolved comments (A inline + B outside-diff) and Y failed/pending checks."

**Exit:** All checks green AND no unresolved comments → done. Pending checks do NOT count as clean.

### Step 3: Fetch & Classify Issues

**Review comments** — filter `isResolved: false`, classify:

| Category | Description | Examples |
| -------- | ----------- | -------- |
| **A: Actionable** | Code changes required | Bugs, missing validation, security issues |
| **B: Discussable** | May skip if follows `.claude/rules/` | Style preferences, premature optimizations |
| **C: Informational** | Resolve without changes | Acknowledgments, "optional" suggestions |

Treat bot reviewers (CodeRabbit, Copilot, Gemini) same as human — classify by content.

**Out-of-diff findings** (from `/tmp/reviews.json`) — present alongside inline threads as pseudo-threads (path + line range + body). They have NO thread ID, so Step 6's `resolveReviewThread` mutation cannot apply; address by fixing the code and noting the fix in the next commit message.

**CI failures:**

```bash
# List failed checks to get the link for each failed job
gh pr checks <NUMBER> --json name,state,link

# Extract run ID from a failed check's link
# Link format: https://github.com/<owner>/<repo>/actions/runs/<RUN_ID>/job/<JOB_ID>
RUN_ID=$(echo "$LINK" | sed -En 's|.*/runs/([0-9]+)/.*|\1|p')
gh run view "$RUN_ID" --log-failed
```

**`--log-failed` requires the entire run to be complete** (all jobs, not just the failed one). If any job is still pending, `gh` returns "run is still in progress". Check first: `gh run view <RUN_ID> --json status --jq '.status'` — must return `"completed"`.

For large logs: `gh run view <RUN_ID> --log-failed 2>&1 | grep "error:" | head -20`

**External checks** (non-GitHub Actions): no run ID exists — open the `link` URL directly to view logs from the external provider.

### Step 4: Get User Confirmation

Present ALL issues in a numbered list:

```text
Review Comments:
  1. [A] src/foo.cpp:42 — Missing null check (reviewer: alice)
  2. [B] src/bar.py:15 — Style suggestion (reviewer: coderabbitai)
CI Failures:
  3. [CI] build — error: 'Foo' is not a member of 'pypto::ir'
```

Ask which to address/skip. Recommend A + CI items. On subsequent iterations, reuse prior "address all" policy for same categories. When unsure about a comment's category, default to B.

### Step 5: Fix Issues

1. Read affected files, make changes with Edit tool
2. For CI: analyze logs online first, reproduce locally only as last resort
3. Commit using `/git-commit` skill (skip testing/review for minor fixes)
4. Push: `git push`

**Commit message:** `fix(pr): resolve issues for #<number>` with bullet list of fixes.

### Step 6: Resolve Comment Threads

Reply with `gh api repos/:owner/:repo/pulls/<number>/comments/<comment_id>/replies -f body="..."` then resolve with GraphQL `resolveReviewThread` mutation.

Templates: Fixed → "Fixed in `<commit>` - description" | Skip → "Follows `.claude/rules/<file>`" | Ack → "Acknowledged!"

**Out-of-diff findings** (no thread ID): nothing to resolve via GraphQL — note the fix in the commit message; CodeRabbit re-scans on the next push and won't re-emit fixed findings.

### Step 7: Wait and Re-check

```bash
sleep 600
# Verify run is complete before fetching logs
gh run view <RUN_ID> --json status --jq '.status'  # must be "completed"
# Then loop back to Step 2
```

Poll with `gh pr checks <NUMBER>` — proceed early if all checks finish. **Do not fetch logs until run status is "completed".**

**Loop safeguards:** Max 5 iterations. Flag stuck issues (same failure reappears) to user instead of retrying.

## Reference Tables

| Area | Guidelines |
| ---- | ---------- |
| CI errors | Fetch logs online first; reproduce locally as last resort |
| Bot reviews | Classify by content, not author |
| Changes | Read full context; minimal edits; follow project conventions |

| Error | Action |
| ----- | ------ |
| PR not found | `gh pr list`; ask user |
| CI logs unavailable / run in progress | Wait for run completion; if still unavailable, fall back to local reproduction |
| CI logs too large | `grep -E "error:\|FAILED\|fatal"` |
| Max iterations reached | Stop, report remaining issues |
| Same failure persists | Flag to user, do not retry |

## Checklist

- [ ] PR matched and validated
- [ ] Review comments and CI status fetched
- [ ] ALL issues presented to user for selection
- [ ] Code changes made and committed (use `/git-commit`)
- [ ] Changes pushed to remote
- [ ] Review comment threads replied to and resolved
- [ ] Waited for CI/reviews and re-checked
- [ ] Loop exited: all clean OR max iterations reached
