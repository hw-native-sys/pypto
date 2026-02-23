---
name: github-pr
description: Create a GitHub pull request after committing, rebasing, and pushing changes. Use when the user asks to create a PR, submit changes for review, or open a pull request.
---

# PyPTO GitHub Pull Request Workflow

## Workflow Steps

1. **Prepare branch and commit** (if on main or uncommitted changes)
2. Check for existing PR (exit if found)
3. Fetch upstream changes
4. Rebase onto upstream/main
5. Resolve conflicts if needed
6. Push to fork with `--force-with-lease`
7. Create PR using gh CLI

## Step 1: Prepare Branch and Commit

**Check current state:**

```bash
BRANCH_NAME=$(git branch --show-current)
git status --porcelain                              # Check for uncommitted changes
git fetch upstream 2>/dev/null || git fetch origin  # Fetch latest
if git rev-parse --verify upstream/main >/dev/null 2>&1; then
  BASE_REF=upstream/main
else
  BASE_REF=origin/main
fi
git rev-list HEAD --not "$BASE_REF" --count         # Commits ahead of base
```

A branch "needs a new branch" when it is effectively on main — either the branch name is `main`/`master`, **or** it has zero commits ahead of upstream/main (e.g., a local branch that was never diverged).

**Decision logic:**

| Needs new branch? | Uncommitted changes? | Action |
| ----------------- | -------------------- | ------ |
| Yes | Yes | Create new branch, then commit via `/git-commit` |
| Yes | No | Error — nothing to PR. Tell user to make changes first |
| No | Yes | Commit on current branch via `/git-commit` |
| No | No | Skip — already committed on a feature branch |

**If a new branch is needed:**

1. Ask the user for a branch name (suggest one based on the changes)
2. Create and switch to the new branch:

```bash
git checkout -b <branch-name>
```

1. Commit via `/git-commit` skill (mandatory — runs code review, testing, linting)

**If on an existing feature branch with uncommitted changes:**

Commit via `/git-commit` skill before proceeding.

## Step 2: Check for Existing PR

```bash
BRANCH_NAME=$(git branch --show-current)
gh pr list --head "$BRANCH_NAME" --state open
```

**If PR exists**: Display with `gh pr view` and exit immediately.

## Step 3: Fetch Upstream

```bash
git remote add upstream https://github.com/hw-native-sys/pypto.git  # If needed
git fetch upstream
```

## Step 4: Rebase

```bash
git rebase upstream/main  # Or user-specified branch
```

**On conflicts**:

```bash
git status                     # View conflicts
# Edit files, remove markers
git add path/to/resolved/file
git rebase --continue
# If stuck: git rebase --abort
```

## Step 5: Push

```bash
# First push
git push --set-upstream origin BRANCH_NAME

# After rebase (use --force-with-lease, NOT --force)
git push --force-with-lease origin BRANCH_NAME
```

⚠️ **Use `--force-with-lease`** - safer than `--force`, fails if remote has unexpected changes.

## Step 6: Create PR

**Check gh CLI**:

```bash
gh auth status
```

**If gh NOT available**: Report to user and provide manual URL: `https://github.com/hw-native-sys/pypto/compare/main...BRANCH_NAME`

**If gh available**:

```bash
gh pr create \
  --title "Brief description of changes" \
  --body "$(cat <<'EOF'
## Summary
- Key change 1
- Key change 2

## Testing
- [ ] All tests pass
- [ ] Code review completed
- [ ] Documentation updated

## Related Issues
Fixes #ISSUE_NUMBER (if applicable)
EOF
)"
```

**PR Title/Body**: Auto-extracted from commit messages since upstream/main.

**Important**:

- ❌ Do NOT add footers like "🤖 Generated with Claude Code" or similar branding
- ✅ Keep PR descriptions professional and focused on technical content only

## Common Issues

| Issue | Solution |
| ----- | -------- |
| PR already exists | `gh pr view` then exit |
| Merge conflicts | Resolve, `git add`, `git rebase --continue` |
| Push rejected | `git push --force-with-lease` |
| gh not authenticated | Tell user to run `gh auth login` |
| Wrong upstream branch | Use `git rebase upstream/BRANCH` |

## Checklist

- [ ] Branch prepared (created from main if needed)
- [ ] Changes committed via `/git-commit` (if uncommitted changes existed)
- [ ] No existing PR for branch (exit if found)
- [ ] Fetched upstream and rebased successfully
- [ ] Conflicts resolved
- [ ] Pushed with `--force-with-lease`
- [ ] PR created with clear title/body

## Remember

- Always rebase before creating PR
- Use `--force-with-lease`, not `--force`
- Don't auto-install gh CLI - let user do it
