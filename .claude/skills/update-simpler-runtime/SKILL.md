---
name: update-simpler-runtime
description: Rebase a PyPTO work branch onto upstream and upgrade the Simpler runtime gitlink. Use for Simpler pin bumps and compatibility checks after syncing PyPTO with upstream.
---

# Update the Simpler Runtime

1. Check the current branch, worktree status, and remotes. A Codex worktree may start detached at the fork's `main`. Create a feature branch there when preparing a new PR. Fetch the canonical PyPTO base and rebase the branch before reading its runtime pin. Verify the base is an ancestor of `HEAD`.
2. Read the rebased pin with `git rev-parse HEAD:runtime`. Initialize `runtime/` if needed. Resolve the requested revision against the Simpler URL in `.gitmodules`. For the latest runtime, use the tip of Simpler's upstream default branch. Fetch if the commit is missing. Inspect the range from the rebased pin to the target before checking out the target in `runtime/`.
3. Review the range for changes to Python bindings, callable serialization, orchestration headers, runtime selection, and Simpler's `pto_isa.pin`. Compare changed contracts with the corresponding PyPTO call sites. For an ISA change, check the generated kernels against the target headers and PyPTO's pinned PTOAS. Adapt callers where the contracts changed.
4. Before rebuilding, run `git -C runtime status --porcelain` and require an empty result. If the runtime checkout has uncommitted changes, stop and commit or stash them before treating the pin check as validation. Build PyPTO in the current worktree using the [testing skill](../testing/SKILL.md). Rebuild and install Simpler from this worktree's `runtime/` into the same test environment. Run `pypto-runtime-pin` and confirm its installed build stamp matches the target checkout. An exit status of zero can also mean the comparison was skipped. Before committing, an off-pin warning is expected while the gitlink in `HEAD` still points to the old commit.
5. Run the runtime unit tests and any tests for affected callers. Include focused simulator or device coverage for changed execution paths when available. The runtime unit fixtures can bypass the pin guard and stub Simpler, so also test a real runtime import and exercise the affected binding against the rebuilt extension.
6. Stage `runtime` and any required PyPTO changes. Inspect `git diff --cached --submodule=short` to confirm the old and new gitlinks. After committing, check that `git rev-parse HEAD:runtime`, `git -C runtime rev-parse HEAD`, and the installed build stamp agree. When a PR is requested, use the shared `github-pr` skill and include the Simpler range, compatibility changes, and validation results.

See [The runtime pin](../../../docs/en/dev/00-ecosystem.md#the-runtime-pin) for the source of truth for each revision and how the guard reports a mismatch.
