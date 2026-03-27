# AI Assistant Rules for PyPTO Project

Please follow the following rules when working on the PyPTO project:

## Project Rules

All rules for this project are located in the `.claude/rules/` directory. Please read and follow all applicable rules from that directory when:

- Making code changes
- Reviewing code
- Committing changes
- Working across different layers (C++, Python, etc.)

Refer to the individual rule files in `.claude/rules/` for specific guidance on project conventions, coding standards, and best practices.

## Skills (`.claude/skills/`)

Skills are workflow guides that help the main assistant perform specific tasks:

- **`git-commit`** - Complete commit workflow with review, testing, and optional code simplification
- **`code-review`** - Reviews code changes against project standards (`context: fork` — runs in isolated context)
- **`testing`** - Builds project and runs test suite (`context: fork` — runs in isolated context)
- **`github-pr`** - Creates a GitHub pull request after committing and pushing
- **`create-issue`** - Creates a GitHub issue following project templates
- **`fix-issue`** - Fixes a GitHub issue by fetching, branching, planning, and implementing
- **`fix-pr`** - Fixes PR issues: addresses review comments and resolves CI failures
- **`clean-branches`** - Removes stale local and remote fork branches that have been merged into main
- **`compare-codegen`** - Compares codegen output (.pto files, pass dumps) between origin/main and current branch for a given test case

**Key advantage:** `code-review` and `testing` use `context: fork` to run in isolated subagent contexts. They can run in parallel during commit workflows without polluting the main context window.
