---
name: clean-branches
description: Remove stale local and remote git branches that have been merged into main. Detects regular merges and squash merges. Cleans up fork remote branches and prunes tracking refs. Use when the user asks to clean up branches, remove merged branches, or tidy up git.
---

# Clean Merged Git Branches

## Overview

Identifies and removes branches whose work is already in main — both local branches and remote branches on the fork. Detects squash-merged branches that `git branch --merged` cannot detect. Never touches the upstream repo.

## Step 1: Identify Remotes

```bash
git remote -v
```

Determine which remote is the **fork** (typically `origin`) and which is **upstream**. Store the fork remote name as `<fork>` for use in subsequent steps. Only fork remote branches are candidates for deletion.

## Step 2: Gather Branch Information

Run in parallel:

```bash
# Local: regular merges
git branch --merged main | grep -v '^\*' | grep -vx '  main'

# Local: all branches (excluding current and main)
git branch | grep -v '^\*' | grep -vx '  main'

# Remote: all fork branches (exclude main/HEAD)
git fetch <fork>
git branch -r --list '<fork>/*' | grep -vw '<fork>/main' | grep -vw '<fork>/HEAD'

# Stale remote tracking refs
git remote prune <fork> --dry-run
```

**If no local or remote branches exist besides main**: Inform user and exit.

## Step 3: Detect Squash-Merged Branches

For each branch (local or remote-only) NOT in the `--merged` list, check GitHub:

```bash
gh pr list --head "<branch-name>" --state merged --json number,title,headRefOid --limit 1
```

For remote branches, strip the `<fork>/` prefix before querying.

**Branch-reuse safeguard:** If a merged PR is found, compare the branch tip SHA with the PR's `headRefOid`. If they differ, the branch may have new commits after the PR merged — treat it as unfinished, not safe to delete.

**Categorize each branch:**

| Category | Criteria | Safe to delete? |
| -------- | -------- | --------------- |
| Merged (git) | In `git branch --merged main` | Yes |
| Squash-merged | `gh pr list --state merged` returns a PR | Yes |
| No merged PR | No merged PR found | Possibly unfinished work |

## Step 4: Present Summary to User

Display a combined table showing both local and remote status:

```text
**Merged branches (safe to delete):**
| # | Branch | Local | Remote | PR | How merged |
|---|--------|-------|--------|----|------------|
| 1 | feat/foo | yes | yes | #123 | squash-merged |
| 2 | old-branch | yes | no | — | regular merge |
| 3 | docs/old-docs | no | yes | #456 | squash-merged |

**Unfinished branches (no merged PR):**
| # | Branch | Local | Remote | Last Commit |
|---|--------|-------|--------|-------------|
| 4 | wip-thing | yes | no | abc1234 "unfinish" |
```

## Step 5: Ask for Approval

Use `AskUserQuestion` with options:

| Option | Description |
| ------ | ----------- |
| All merged branches (Recommended) | Delete merged branches (local + remote on fork). Keep unfinished. |
| All branches | Delete everything including unfinished work. |
| Let me pick | User specifies which branches to delete. |

**Never delete branches without explicit user approval.**

## Step 6: Delete and Prune

After approval:

```bash
# Delete local branches
git branch -D <branch1> <branch2> ...

# Delete remote branches on fork (only if they exist on the remote)
for b in <branch1> <branch2> ...; do
  if git show-ref --verify --quiet "refs/remotes/<fork>/$b"; then
    git push <fork> --delete "$b"
  fi
done

# Prune stale remote tracking refs
git remote prune <fork>
```

Report results: local branches deleted, remote branches deleted, refs pruned.

## Important Constraints

- **Never delete remote branches on upstream** — only on the fork (`<fork>`)
- **Never delete `main` or `HEAD`** on any remote
- **Current branch**: Warn user if current branch is not main; cannot delete it
- **`gh` unavailable**: Skip squash-merge detection; inform user only `git --merged` is used
- **No remote for a branch**: Skip remote deletion for that branch silently

## Checklist

- [ ] Fork vs upstream remotes identified
- [ ] All local and remote branches categorized
- [ ] Summary table presented (showing local/remote status per branch)
- [ ] User approved deletion list
- [ ] Local branches deleted
- [ ] Remote fork branches deleted
- [ ] Stale refs pruned
- [ ] Results reported
