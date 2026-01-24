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
- **`git-commit`** - Complete commit workflow with review and testing
- **`code-review`** - Invokes code review agent
- **`testing`** - Invokes testing agent

Skills are activated by the main assistant when the user requests related tasks.

### Agents (`.claude/agents/`)

Agents are specialized subprocesses that execute specific tasks autonomously:
- **`code-review`** - Reviews code changes against project standards
- **`testing`** - Builds project and runs test suite

**Key advantage:** Code review and testing agents can run in parallel during commit workflows, significantly reducing wait time.

Agents are invoked by their corresponding skills using the Task tool.
