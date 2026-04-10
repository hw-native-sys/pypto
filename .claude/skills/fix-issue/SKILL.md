---
name: fix-issue
description: Fix a GitHub issue by fetching content, creating a branch, planning the fix, and implementing it. Use when the user asks to fix a specific issue number or work on a GitHub issue.
argument-hint: [issue-number]
---

# PyPTO Issue Fix Workflow

Fetch GitHub issue, create branch, plan, and implement the fix.

## Task Tracking

Create tasks to track progress through this workflow:

1. Fetch issue & create branch
2. Plan the fix
3. Self-assign & set In Progress
4. Implement the fix
5. Run tests
6. Commit changes
7. Create PR (optional)

## Workflow

1. Check gh CLI authentication
2. Fetch issue content
3. Create issue branch
4. Enter plan mode to design fix
5. **Self-assign & set In Progress** (immediately after plan approval)
6. Implement the fix
7. Run tests (use `testing` skill)
8. Commit changes (use `git-commit` skill)
9. Create PR (optional, use `github-pr` skill)

## Step 1: Check gh CLI Authentication

```bash
gh auth status
```

**If not authenticated**, prompt user:

```text
gh CLI is not authenticated. Please run: gh auth login
```

⚠️ **Stop here if not authenticated** - user must login first.

## Step 2: Fetch Issue Content and Check Ownership

```bash
gh issue view ISSUE_NUMBER --json number,title,body,state,labels,assignees
gh issue view ISSUE_NUMBER --comments
```

**Parse**: Issue number, title, description, state (open/closed), labels, assignees, and all comments.

Comments often contain clarifications, reproduction steps, or design decisions that are critical for understanding the full context of the issue.

**If issue is closed**: Ask user if they still want to work on it.

**Check for existing ownership** before proceeding:

1. Check if anyone is already assigned (`assignees` field)
2. Query the project board status:

   ```bash
   gh api graphql -f query='{ repository(owner:"hw-native-sys",name:"pypto") {
     issue(number:ISSUE_NUMBER) { projectItems(first:5) { nodes { id project { number }
       fieldValues(first:10) { nodes {
         ... on ProjectV2ItemFieldSingleSelectValue { field { ... on ProjectV2SingleSelectField { name } } name }
       } } } } } } }'
   ```

   If the project item is not found, skip the board status check (the issue may not be linked to the project yet) and continue.

3. If **assigned to someone** or **Status in project #3 is "In Progress"**: warn the user with `AskUserQuestion` — show who is assigned and/or the current status, and ask whether to proceed anyway or stop.

## Step 3: Create Issue Branch

**Branch naming**: `issue-{number}-{short-description}`

```bash
git checkout main && git pull upstream main
ISSUE_NUM=123
BRANCH_NAME="issue-${ISSUE_NUM}-fix-tensor-validation"
git checkout -b "$BRANCH_NAME"
```

## Step 4: Enter Plan Mode

Use `EnterPlanMode` to design the fix.

**Plan should cover**:

- Root cause analysis (for bugs)
- Files that need changes
- Implementation strategy
- Testing approach
- Documentation updates
- Cross-layer changes (C++, Python, type stubs)

## Step 5: Self-Assign & Set In Progress

**Do this IMMEDIATELY after plan approval, before writing any code.**

```bash
# Self-assign
gh issue edit ISSUE_NUMBER --add-assignee @me
```

Update the project board status using the same GraphQL pattern as `create-issue` Step 7:

1. Get project item ID from the query in Step 2 (already fetched)
2. Fetch field options dynamically (query `organization.projectV2.fields`)
3. Set Status to "In Progress" via `updateProjectV2ItemFieldValue` mutation

If the project item or Status field is not found, skip the board update and notify the user that manual update is needed. Do not block the fix workflow.

## Step 6: Implement the Fix

After plan approval, follow PyPTO conventions:

1. Make code changes following plan
2. Follow `.claude/rules/` conventions
3. Update documentation if needed
4. Add/update tests
5. Maintain cross-layer sync (C++, Python, type stubs)

## Step 7: Run Tests

```text
/testing
```

Fix any failures before committing.

## Step 8: Commit Changes

```text
/git-commit
```

**Commit message format**:

```text
fix(scope): Brief description

Fixes #ISSUE_NUMBER

Detailed explanation of the fix.
```

## Step 9: Create PR (Optional)

```text
/github-pr
```

**PR must reference issue**: "Fixes #ISSUE_NUMBER"

## Common Issue Types

| Type | Approach |
| ---- | -------- |
| Bug fix | Reproduce, root cause, fix, add regression test |
| Feature request | Plan API design, implement, add tests and docs |
| Refactoring | Plan changes, ensure tests pass, maintain API |
| Documentation | Fix/improve docs, verify examples work |

## Checklist

- [ ] gh CLI authenticated
- [ ] Issue content fetched and understood
- [ ] Checked for existing assignees / In Progress status
- [ ] Issue branch created from latest main
- [ ] Plan created and approved
- [ ] Issue self-assigned and set to In Progress on project board
- [ ] Fix implemented following PyPTO rules
- [ ] Tests added/updated and passing
- [ ] Changes committed with issue reference
- [ ] Documentation updated if needed

## Remember

**Reference the issue number** in commit messages and PR description using "Fixes #ISSUE_NUMBER" for auto-linking.
