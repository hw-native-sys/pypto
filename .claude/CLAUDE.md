# AI Assistant Rules for PyPTO Project

Please follow the following rules when working on the PyPTO project:

## Project Rules

All rules for this project are located in the `.claude/rules/` directory. Please read and follow all applicable rules from that directory when:

- Making code changes
- Reviewing code
- Committing changes
- Working across different layers (C++, Python, etc.)

Refer to the individual rule files in `.claude/rules/` for specific guidance on project conventions, coding standards, and best practices.

## Skills and Agents

### Skills (`.claude/skills/`)

Skills are workflow guides that help the main assistant perform specific tasks:

- **`git-commit`** - Complete commit workflow with review, testing, and optional code simplification
- **`code-review`** - Invokes code review agent
- **`testing`** - Invokes testing agent
- **`github-pr`** - Creates a GitHub pull request after committing and pushing
- **`create-issue`** - Creates a GitHub issue following project templates
- **`fix-issue`** - Fixes a GitHub issue by fetching, branching, planning, and implementing
- **`address-pr-comments`** - Analyzes and addresses GitHub PR review comments

Skills are activated by the main assistant when the user requests related tasks.

### Agents (`.claude/agents/`)

Agents are specialized subprocesses that execute specific tasks autonomously:

- **`code-reviewer`** - Reviews code changes against project standards
- **`testing`** - Builds project and runs test suite

**Key advantage:** Code review and testing agents can run in parallel during commit workflows, significantly reducing wait time.

Agents are invoked by their corresponding skills using the Task tool.

### Optional Plugins

- **`code-simplifier`** - Marketplace plugin that simplifies and refines changed code for clarity and consistency. Optionally prompted during `git-commit` workflow. Install via `/plugin install code-simplifier` if not already installed.
