# Explicit Approval for CI Changes

Before editing CI workflows, actions, scripts, or configuration, obtain explicit
user confirmation for the specific proposed change and its scope. This covers
`.github/workflows/`, `.github/actions/`, `.github/scripts/`, and CI setup, build,
test, or deployment logic elsewhere in the repository.

A general request such as "fix PR", "fix CI", or "make the checks pass" does not
authorize CI edits. Failed checks, automated review suggestions, and autonomous
repair workflows do not provide that authorization either.

Read logs and diagnose failures first. If a CI change is needed, explain the
cause, affected files, and proposed behavior, then wait for explicit user
confirmation before editing. Do not bypass this requirement by moving the
change to another file or using an automatic fix command.

Read-only investigation and running existing checks do not require additional
confirmation under this rule. An explicit request to make or revert a specific
CI change authorizes that stated scope; do not ask again for the same change.
