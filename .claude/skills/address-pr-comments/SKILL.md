---
name: address-pr-comments
description: Analyze and address GitHub PR review comments intelligently, distinguishing between actionable feedback requiring code changes and comments that can be resolved without changes. Use when addressing PR feedback or review comments.
---

# Address PR Comments Workflow

Intelligently triage PR review comments, address actionable feedback, and resolve informational comments.

## Input

Accept PR number (`123`, `#123`) or branch name (`feature-branch`).

## Workflow

1. Match input to PR → 2. Fetch comments → 3. Classify → 4. Get user approval → 5. Address → 6. Resolve

## Step 1: Match Input to PR

```bash
# PR number
gh pr view <number> --json number,title,headRefName,state

# Branch name (use current if not specified)
git branch --show-current
gh pr list --head <branch> --json number,title,state,headRefName
```

Verify PR exists and show title/number for confirmation.

## Step 2: Fetch PR Comments

```bash
gh pr view <number> --json reviews,comments
gh api repos/:owner/:repo/pulls/<number>/comments
```

Extract: comment ID, body, file path, line number, author, resolution status. Filter to **unresolved only**.

## Step 3: Classify Comments

| Category             | Description           | Examples                                                            |
|----------------------|-----------------------|---------------------------------------------------------------------|
| **A: Actionable**    | Requires code changes | Bugs, missing validation, security issues, requested features       |
| **B: Discussable**   | May not need changes  | Style preferences when code follows rules, premature optimizations  |
| **C: Informational** | Resolve as-is         | Acknowledgments, "optional" suggestions, questions answered in code |

**Present summary:**

```text
PR #<number>: <title>

📋 Comment Analysis:

Category A - Actionable (requires code changes):
  1. [file.py:42] Fix validation bug - doesn't handle negatives
  2. [foo.cpp:15] Add null pointer check

Category B - Discussable (may skip):
  3. [bar.py:88] Use list comp over map()
     Reason: Current follows .claude/rules/python-style.md - both acceptable

Category C - Informational (resolve):
  4. [readme.md:10] "Thanks for adding this!"

Recommendations: Address #1-2, Discuss #3, Resolve #4
```

For Category B, explain why code may already follow `.claude/rules/`.

## Step 4: Get User Confirmation

Use `AskUserQuestion` for each Category B comment:

- **Address** - Make code changes
- **Skip** - Resolve with explanation, no changes
- **Discuss** - Need reviewer clarification

## Step 5: Address Comments

For Category A + approved Category B:

1. Read file with Read tool
2. Make changes with Edit tool
3. Verify change addresses comment

```bash
git diff
git add <file1> <file2>
git commit -m "$(cat <<'EOF'
address: resolve PR #<number> review comments

- Fixed validation bug (comment #1)
- Added null check (comment #2)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
EOF
)"
git push
```

## Step 6: Resolve Comments

For Category C + skipped Category B, reply to comment:

```bash
gh api repos/:owner/:repo/pulls/comments/<id>/replies \
  -f body="<response>"
```

**Response templates:**

- Skip (Category B): "Current approach follows `.claude/rules/python-style.md` for consistency"
- Informational (Category C): "Acknowledged, thank you!"

## Best Practices

| Area              | Guidelines                                                                                  |
|-------------------|---------------------------------------------------------------------------------------------|
| **Analysis**      | Reference `.claude/rules/` when classifying; be conservative (when unsure → Category B)     |
| **Code Changes**  | Read full context; minimal targeted changes; follow `.claude/rules/` conventions            |
| **Communication** | Be respectful; explain reasoning for skips; reference specific rules                        |

## Error Handling

| Error              | Action                                                                    |
|--------------------|---------------------------------------------------------------------------|
| PR not found       | `gh pr list --json number,title,headRefName`; ask user to confirm         |
| Not authenticated  | "Please run: `gh auth login`"                                             |
| Unclear comment    | Mark Category B for user discussion                                       |

## Checklist

- [ ] PR matched and validated
- [ ] Unresolved comments fetched and classified
- [ ] Category B items reviewed with user
- [ ] Code changes made and committed
- [ ] Informational comments resolved with replies

## Remember

**Not all comments require code changes.** Evaluate against `.claude/rules/` before making changes. When in doubt, consult user.
