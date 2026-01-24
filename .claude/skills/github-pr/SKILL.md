---
name: github-pr
description: Create a GitHub pull request after committing, rebasing, and pushing changes. Use when the user asks to create a PR, submit changes for review, or open a pull request.
---

# PyPTO GitHub Pull Request Workflow

## Overview

Complete workflow for creating a pull request from a feature branch.

## Prerequisites

⚠️ **Run these skills first:**
1. **`git-commit`** - Commit all changes

## Workflow

```
1. Check if branch already has an open PR (exit if yes)
2. Commit changes (use git-commit skill)
3. Fetch upstream changes
4. Rebase onto upstream/main
5. Resolve any conflicts
6. Push to forked repo
7. Create PR using gh CLI
```

## Step 1: Check for Existing PR

**First, check if this branch already has an open PR:**

```bash
# Get current branch name
BRANCH_NAME=$(git branch --show-current)

# Check for existing PR (if gh CLI is available)
if command -v gh &> /dev/null; then
    EXISTING_PR=$(gh pr list --head "$BRANCH_NAME" --state open --json number --jq '.[0].number')

    if [ -n "$EXISTING_PR" ]; then
        echo "✓ PR already exists for branch '$BRANCH_NAME': #$EXISTING_PR"
        gh pr view "$EXISTING_PR"
        exit 0
    fi
fi
```

**If PR already exists:**
- Report the existing PR number and URL to the user
- **Exit immediately** - no further action needed
- Display PR status using `gh pr view`

**If no PR exists:**
- Continue to Step 2

## Step 2: Commit Changes

**Use the `git-commit` skill to commit all changes first.**

If changes aren't committed yet:
```bash
# Apply git-commit skill
# This ensures code review and testing are complete
```

Verify commits are ready:
```bash
git log --oneline -5  # Review recent commits
git status            # Should be clean
```

## Step 3: Fetch Upstream

**Fetch latest changes from upstream repository:**

```bash
# Add upstream if not already added
git remote -v
# If upstream is missing:
git remote add upstream https://github.com/hw-native-sys/pypto.git

# Fetch upstream changes
git fetch upstream

# Verify upstream is up to date
git log --oneline upstream/main -5
```

## Step 4: Rebase onto Upstream

**Rebase your branch onto upstream main (or user-specified branch):**

```bash
# Default: rebase onto upstream/main
git rebase upstream/main

# Or if user specifies different branch:
git rebase upstream/BRANCH_NAME
```

**If conflicts occur:**
```bash
# View conflicting files
git status

# Resolve conflicts in each file
# Edit files, remove conflict markers

# Stage resolved files
git add path/to/resolved/file.cpp

# Continue rebase
git rebase --continue

# If you get stuck, abort and ask user:
git rebase --abort
```

**After successful rebase:**
```bash
# Verify rebase succeeded
git log --oneline -10
git status
```

## Step 5: Push to Forked Repo

**Push your rebased branch to your fork:**

```bash
# First push (new branch)
git push --set-upstream origin BRANCH_NAME

# If branch already exists remotely (after rebase, force push needed)
git push --force-with-lease origin BRANCH_NAME
```

**⚠️ Use `--force-with-lease` not `--force`:**
- `--force-with-lease` is safer - fails if remote has unexpected changes
- `--force` can overwrite others' work

## Step 6: Create PR with gh CLI

**Check if gh CLI is available:**

```bash
# Check if gh is installed
which gh

# Check if authenticated
gh auth status
```

**If gh is NOT installed or NOT authenticated:**
- **DO NOT auto-install or auto-authenticate**
- Report to user: "gh CLI is not installed/authenticated. Please install and configure manually."
- Provide manual PR creation URL:
  ```
  https://github.com/hw-native-sys/pypto/compare/main...BRANCH_NAME
  ```

**If gh is available and authenticated:**

```bash
# Create PR
gh pr create \
  --title "Brief description of changes" \
  --body "$(cat <<'EOF'
## Summary
- Key change 1
- Key change 2
- Key change 3

## Testing
- [ ] All tests pass
- [ ] Code review completed
- [ ] Documentation updated
- [ ] Cross-layer consistency verified

## Related Issues
Fixes #ISSUE_NUMBER (if applicable)
EOF
)"

```

**PR Title and Body:**
- PR title is automatically extracted from the latest commit message (first line)
- PR body contains all commit messages from commits being pushed
- All commits since `upstream/main` will be included in the PR description

## Example Complete Workflow

```bash
# 1. Check for existing PR
BRANCH_NAME=$(git branch --show-current)
gh pr list --head "$BRANCH_NAME" --state open
# If PR exists, exit here

# 2. Commits already done via git-commit skill

# 3. Fetch upstream
git fetch upstream

# 4. Rebase
git rebase upstream/main
# Resolve conflicts if any

# 5. Push
git push --force-with-lease origin feature/my-feature

# 6. Create PR
gh pr create \
  --title "Add support for dynamic tensor shapes" \
  --body "$(cat <<'EOF'
## Summary
- Implement dynamic shape tracking in TensorExpr
- Add validation for shape consistency
- Update Python bindings and type stubs

## Testing
- [x] All tests pass
- [x] Added tests for dynamic shapes
- [x] Cross-layer consistency verified

## Breaking Changes
None

Fixes #123
EOF
)"
```

## Common Issues

**PR already exists for branch:**
```bash
# Check existing PR
gh pr list --head $(git branch --show-current) --state open

# View PR details
gh pr view PR_NUMBER

# Output message and exit
echo "✓ PR #PR_NUMBER already exists for this branch. No action needed."
exit 0
```

**Merge conflicts during rebase:**
```bash
# View conflicts
git status

# Resolve each file, then:
git add path/to/file.cpp
git rebase --continue

# If too complex, abort and ask user:
git rebase --abort
```

**Push rejected (remote has changes):**
```bash
# Fetch and check what changed
git fetch origin
git log origin/BRANCH_NAME

# If safe to overwrite:
git push --force-with-lease origin BRANCH_NAME
```

**gh CLI not authenticated:**
```bash
# Check status
gh auth status

# Report to user - don't auto-authenticate
echo "gh CLI not authenticated. Please run: gh auth login"
```

**Wrong upstream branch:**
```bash
# User might specify different branch
git rebase upstream/dev    # Instead of main
```

## Checklist

Before creating PR:
- [ ] **Verified no existing PR for this branch** (exit if PR exists)
- [ ] All changes committed (via git-commit skill)
- [ ] Code reviewed and tests passed
- [ ] Fetched latest upstream changes
- [ ] Successfully rebased onto upstream/main (or specified branch)
- [ ] Resolved any merge conflicts
- [ ] Pushed to forked repo with `--force-with-lease`
- [ ] gh CLI available and authenticated (or manual URL provided)
- [ ] PR title and body are clear and complete

## Remember

**Always rebase before creating PR** to ensure your changes work with the latest upstream code.

**Use `--force-with-lease`** when pushing after rebase - it's safer than `--force`.

**Don't auto-install gh CLI** - let the user install and configure it themselves.
