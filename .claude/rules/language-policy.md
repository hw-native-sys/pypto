# Language Policy

## Core Principle

**Always use English as the intermediate/working language. Match the user's language only in final outputs.**

## What Must Be in English (Intermediate Work)

- Internal reasoning and analysis
- Code analysis and technical evaluation
- Tool invocations and tool-related descriptions
- **All text between tool calls** — narration, status lines, transitional text (e.g., "Now update the remaining files.", "Let me verify the results.")
- Agent prompts and agent task descriptions
- Code comments and commit messages (default; may be overridden by explicit project/team policy)
- Search queries and grep patterns

## What Must Match the User's Language (Final Outputs)

- **Only the final consolidated response** at the end of a task (summary, conclusion)
- Plans and proposals presented to the user
- Questions asked to the user

**Clarification:** Mid-process narration visible to the user (e.g., "Let me check..." before a tool call) is intermediate work, NOT a "final output." Write it in English.

## Why

- English is the lingua franca of the codebase, tools, and technical context
- Consistent intermediate language avoids mixed-language confusion in logs and tool calls
- User-facing output in their language ensures clear communication
