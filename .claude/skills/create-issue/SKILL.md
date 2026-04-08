---
name: create-issue
description: Create a GitHub issue following the project's issue templates. Classifies the issue type, fills required fields per template, creates it via gh CLI, and sets project board fields (Status, Priority, Effort, Sprint). Use when the user wants to file a bug, request a feature, report a pass bug, or create any GitHub issue.
---

# Create GitHub Issue

Create issues that follow `.github/ISSUE_TEMPLATE/` templates exactly.

## Step 0: Determine Input Source

Check how the issue was triggered:

**A) From `KNOWN_ISSUES.md`** — If user says "create issue from known issues", "/create-issue known", or similar:

1. Read `KNOWN_ISSUES.md` from project root
2. If file doesn't exist or has no entries, tell user "No known issues found" and **stop**
3. List all entries with their title, severity, and brief description
4. Present the list and ask the user which issue they want to file
5. **Verify the selected issue is still real and unresolved:**
   - If `Location` is present and not `N/A`:
     - Read the file(s) mentioned and check if the problem still exists in the current code
     - If **resolved**: remove the entry from `KNOWN_ISSUES.md`, inform user, and **stop**
     - If **still present**: proceed to Step 1 using the issue's description as input
   - If `Location` is missing or `N/A`:
     - Ask the user to confirm whether the issue is still valid based on the description
     - If **no longer valid**: remove the entry from `KNOWN_ISSUES.md`, inform user, and **stop**
     - If **still valid**: proceed to Step 1 using the issue's description as input
6. After the GitHub issue is created, **remove the entry** from `KNOWN_ISSUES.md` (the issue is now tracked on GitHub)

**B) Direct user input** — Normal flow, proceed to Step 1 with user-provided description.

## Step 1: Authenticate

```bash
gh auth status
```

If not authenticated, tell the user to run `gh auth login` and **stop**.

Verify the token has `project` scope (needed for Step 7). If missing, tell the user to run `gh auth refresh -s project` and **stop**.

## Step 2: Check for Existing Issues

**Launch a `general-purpose` agent** (via `Task` tool, **model: haiku**) to perform the dedup check. This keeps the main context clean and fast.

**Agent prompt must include:** the issue summary/keywords, and these exact instructions:

> **IMPORTANT: ONLY use `gh` CLI commands. Do NOT read source code, test files, or explore the repository. Your sole job is to check GitHub issues for duplicates.**
>
> Follow the two-step process below, then return EXACTLY one of: `DUPLICATE #N`, `RELATED #N1 #N2 ...`, or `NO_MATCH`. Keep your response to 1-3 sentences plus the verdict.

### Two-Step Search Process (for the agent)

**Step A — Scan all open issue titles:**

```bash
gh issue list --state open --limit 200 --search "updated:>=YYYY-MM-DD" --json number,title,labels \
  --jq '.[] | "\(.number)\t\(.title)\t\(.labels | map(.name) | join(","))"'
```

Compute the date 6 months ago for the `updated:>=` filter. Scan output for keywords related to the new issue.

**Step B — Deep-read candidates only (max 3):**

For each title that looks related (up to 3), fetch context:

```bash
gh issue view NUMBER
```

Only read body — skip `--comments` unless the body is ambiguous. Determine if it's truly the same issue or just superficially similar.

### Decision rules (agent returns)

- **Exact match** (same root cause/request) → return `DUPLICATE #N`
- **Related but different** → return `RELATED #N1 #N2 ...`
- **No matches** → return `NO_MATCH`

### How to act on the result

- `DUPLICATE #N` → Do NOT create. Tell the user the existing issue. **Stop here.**
- `RELATED #N1 ...` → Proceed, reference in body: `Related: #N1, #N2`
- `NO_MATCH` → Proceed normally.

## Step 3: Classify the Issue

Read `.github/ISSUE_TEMPLATE/` to get the current templates, then match the user's description to the correct template:

| Template | Use When | Labels |
| -------- | -------- | ------ |
| `bug_report.yml` | General bug (parser, printer, bindings, codegen, build) | `bug` |
| `pass_bug.yml` | Bug in an IR pass or transformation | `bug`, `ir-pass` |
| `feature_request.yml` | New feature or enhancement | `enhancement` |
| `new_operation.yml` | New tensor/block-level operation | `enhancement`, `new-operation` |
| `performance_issue.yml` | Performance regression or optimization | `performance` |
| `documentation.yml` | Missing, incorrect, or unclear docs | `documentation` |

**Classification rules:**

- If about a specific pass producing wrong IR → `pass_bug.yml`
- If about a crash/error not in a pass → `bug_report.yml`
- If about slow execution or regression → `performance_issue.yml`
- If requesting a new op (tensor/block) → `new_operation.yml`
- If requesting any other new capability → `feature_request.yml`
- If about docs being wrong/missing → `documentation.yml`

**If ambiguous**, ask the user to clarify using `AskUserQuestion`.

## Step 4: Gather Required Fields

Each template has **required fields** (marked `required: true` in the YAML). You MUST fill every required field.

**Ask the user** for any required information you cannot infer. Use `AskUserQuestion` for dropdown selections (component, NPU kind, host platform, etc.).

**For fields you can auto-fill:**

- **Git Commit ID**: Run `git rev-parse HEAD` to get the current commit
- **Title prefix**: Use the template's title prefix (`[Bug]`, `[Pass Bug]`, etc.)
- **Host Platform**: Run `uname -s -m` to detect OS and arch. Map to: `Linux aarch64` → `Linux (aarch64)`, `Linux x86_64` → `Linux (x86_64)`, `Darwin arm64` → `macOS (aarch64)`, `Darwin x86_64` → `Other (please specify)` with note `macOS (x86_64)`. Fall back to `Other` if unrecognized. For pass bugs (`pass_bug.yml`), default to `N/A (not hardware-specific)` unless the bug is platform-specific.
- **NPU Kind**: Run `npu-smi info 2>/dev/null` to detect NPU. If command not found or no output, default to `N/A (not hardware-specific)` (or `N/A (CPU-only issue)` for `performance_issue.yml`). Parse model name to map to Ascend 910B/910C if present.

## Step 5: Format the Issue Body

Since `gh issue create` uses markdown body (not YAML form fields), format the body to match the template structure using markdown sections:

```markdown
### Field Label

Field content here

### Another Field

More content
```

**For dropdown fields**, state the selected value as plain text.

## Step 6: Create the Issue

```bash
gh issue create \
  --title "[Prefix] Short description" \
  --label "label1" --label "label2" \
  --body "$(cat <<'EOF'
### Field 1
content

### Field 2
content
EOF
)"
```

**After creation**, capture the issue number from the output URL (e.g., `https://github.com/.../issues/123` → `ISSUE_NUMBER=123`) for use in Step 7. Display the issue URL to the user.

## Step 7: Set Project Board Fields

After creation, the "Auto-add to project" workflow adds the issue to **hw-native-sys project #3**. Set Status, Priority, Effort, and Sprint.

### 7a: Retrieve Project Item ID

```bash
gh api graphql -f query='{ repository(owner:"hw-native-sys",name:"pypto") {
  issue(number:ISSUE_NUMBER) { projectItems(first:5) { nodes { id project { number } } } } } }'
```

Extract the item ID where `project.number == 3`. If not found, wait 5s and retry once (auto-add may be delayed).

### 7b: Fetch Field Options

Query current field option IDs dynamically (do NOT hardcode option IDs):

```bash
gh api graphql -f query='{ organization(login:"hw-native-sys") { projectV2(number:3) {
  id
  fields(first:20) { nodes {
    ... on ProjectV2SingleSelectField { name id options { id name } }
    ... on ProjectV2IterationField { name id configuration {
      iterations { id title startDate duration } } } } } } } }'
```

Extract the project ID, and field/option IDs for Status, Priority, Effort, Sprint.

### 7c: Analyze and Suggest Values

Based on the issue type and content, suggest values using this logic:

| Field | Logic |
| ----- | ----- |
| **Status** | Bugs with clear repro → Ready; Features with proposal/design included → Ready; Features needing design → Backlog; External dependency → Blocked |
| **Priority** | Data corruption / security / blocking → Critical; Most issues → Normal; Cosmetic / code-health → Trivial |
| **Effort** | Cross-layer / new pass / major refactor → Large; Single-layer / moderate → Medium; One-file / docs → Small |
| **Sprint** | Critical + Ready → current sprint; Normal planned → next sprint; Backlog → no sprint |

To determine current sprint: find the sprint where `startDate <= today < startDate + duration_days`.

### 7d: Present to User

Use `AskUserQuestion` to present 4 questions (one per field), each with the relevant options. Put the suggested value first with "(Recommended)" suffix. For Sprint, include a "None" option.

### 7e: Apply Values via GraphQL

For each field, run the mutation using the project/field/option IDs from step 7b:

```bash
gh api graphql -f query='mutation { updateProjectV2ItemFieldValue(input: {
  projectId:"PROJECT_ID" itemId:"ITEM_ID" fieldId:"FIELD_ID"
  value:{ singleSelectOptionId:"OPTION_ID" }
}) { projectV2Item { id } } }'
```

For Sprint (iteration field), use `value:{ iterationId:"ITER_ID" }` instead. If user chose "None" for Sprint, skip it.

## Template Field Reference

| Template | Required Fields |
| -------- | --------------- |
| Bug `[Bug]` | Component, Description, Steps to Reproduce, Expected/Actual Behavior, Git Commit ID, NPU Kind, Host Platform |
| Pass Bug `[Pass Bug]` | Pass Name, Description, Git Commit ID, Before IR, Expected IR, Actual IR/Error. Optional: NPU Kind, Host Platform |
| Feature `[Feature]` | Summary, Motivation / Use Case |
| New Op `[New Op]` | Operation Level, Proposed Name & Signature, Semantics, Example Usage, Motivation |
| Performance `[Performance]` | Summary, Git Commit ID, NPU Kind, Host Platform, Reproduction Script, Expected/Actual Performance |
| Docs `[Docs]` | Documentation Location, What's Wrong or Missing? |

## Checklist

- [ ] Input source determined (KNOWN_ISSUES.md or direct)
- [ ] gh CLI authenticated with `project` scope
- [ ] Searched for existing issues (no duplicate)
- [ ] Issue classified to correct template, all required fields filled
- [ ] Issue created with correct prefix, labels, and markdown body
- [ ] If from KNOWN_ISSUES.md: entry removed from file
- [ ] Project board fields set (Status, Priority, Effort, Sprint)
