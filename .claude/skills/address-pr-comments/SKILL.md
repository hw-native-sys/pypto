---
name: address-pr-comments
description: Analyze and address GitHub PR review comments intelligently, distinguishing between actionable feedback requiring code changes and comments that can be resolved without changes. Use when addressing PR feedback or review comments.
---

# Address PR Comments Workflow

Intelligently triage PR review comments, address actionable feedback, and resolve informational comments.

## Input

Accept PR number (`123`, `#123`) or branch name (`feature-branch`).

## Workflow

1. Match input to PR
2. Fetch unresolved comments
3. Classify comments
4. Get user confirmation (Category B)
5. Address comments with code changes
6. Reply and resolve threads

## Step 1: Match Input to PR

```bash
# PR number: gh pr view <number> --json number,title,headRefName,state
# Branch: git branch --show-current && gh pr list --head <branch> --json number,title,state
```

## Step 2: Fetch Unresolved Comments

```bash
gh api graphql -f query='
query {
  repository(owner: "hw-native-sys", name: "pypto") {
    pullRequest(number: <number>) {
      reviewThreads(first: 50) {
        nodes {
          id isResolved
          comments(first: 1) {
            nodes { id databaseId body path line }
          }
        }
      }
    }
  }
}'
```

Filter to `isResolved: false` only.

## Step 3: Classify Comments

| Category             | Description                          | Examples                                   |
|----------------------|--------------------------------------|--------------------------------------------|
| **A: Actionable**    | Code changes required                | Bugs, missing validation, security issues  |
| **B: Discussable**   | May skip if follows `.claude/rules/` | Style preferences, premature optimizations |
| **C: Informational** | Resolve without changes              | Acknowledgments, "optional" suggestions    |

Present summary showing category, file:line, and issue for each comment. For Category B, explain why code may already comply with `.claude/rules/`.

## Step 4: Get User Confirmation

Use `AskUserQuestion` for Category B: Address (make changes) / Skip (resolve as-is) / Discuss (need clarification)

## Step 5: Address Comments

For Category A + approved Category B:

1. Read files with Read tool
2. Make changes with Edit tool
3. Commit using `/commit --skip-testing --skip-review`

**Commit message format:**

```text
chore(pr): resolve review comments for #<number>

- Fixed validation bug (comment #1)
- Added null check (comment #2)
```

**When to skip testing/review:**

- Minor documentation/comment fixes
- Changes already tested in original PR
- Fast iteration needed

## Step 6: Resolve Comments

Reply using `gh api repos/:owner/:repo/pulls/<number>/comments/<comment_id>/replies -f body="..."` then resolve thread with GraphQL `resolveReviewThread` mutation.

**Response templates:**

- Fixed: "Fixed in `<commit>` - description"
- Skip: "Current follows `.claude/rules/<file>`"
- Acknowledged: "Acknowledged, thank you!"

## Best Practices

| Area              | Guidelines                                                   |
|-------------------|--------------------------------------------------------------|
| **Analysis**      | Reference `.claude/rules/`; when unsure → Category B         |
| **Changes**       | Read full context; minimal edits; follow project conventions |
| **Communication** | Be respectful; explain reasoning; reference rules            |

## Error Handling

| Error             | Action                             |
|-------------------|------------------------------------|
| PR not found      | `gh pr list`; ask user to confirm  |
| Not authenticated | "Run: `gh auth login`"             |
| Unclear comment   | Mark Category B for discussion     |

## Checklist

- [ ] PR matched and validated
- [ ] Unresolved comments fetched and classified
- [ ] Category B items reviewed with user
- [ ] Code changes made and committed (use `/commit`)
- [ ] Changes pushed to remote
- [ ] All comments replied to and resolved

## Remember

**Not all comments require code changes.** Evaluate against `.claude/rules/` first. When in doubt, consult user.
