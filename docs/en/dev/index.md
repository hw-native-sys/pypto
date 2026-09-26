# Developer Documentation

How PyPTO is built: the IR, the pass pipeline, code generation, and the
infrastructure around them.

This is documentation for people working *on* the compiler. If you are writing
PyPTO programs, start with the [User Manual](../user/index.md).

## Sub-chapters

| Chapter | What it covers |
| ------- | -------------- |
| [IR](ir/index.md) | Node hierarchy, type system, operators, builder, parser, serialization, structural comparison |
| [Passes](passes/index.md) | The pass framework and every pass in the default pipeline, numbered in execution order |
| [Language](language/index.md) | The Python DSL syntax specification and external C++ kernel integration |
| [Code Generation](codegen/index.md) | Lowering IR to PTO-ISA dialect MLIR and to orchestration C++ |
| [Backend](backend/index.md) | Per-architecture dispatch through `BackendHandler` |
| [Debug](debug/index.md) | Lowering IR to an executable PyTorch script for numerical validation |

## Top-level topics

| Page | What it covers |
| ---- | -------------- |
| [PTO Project Ecosystem](00-ecosystem.md) | The multi-repo toolchain — PyPTO, PTOAS, pto-isa, simpler, pypto-lib — and how they fit together |
| [Compile Profiling](01-compile-profiling.md) | Built-in wall-clock timing of the compilation pipeline |
| [Error Handling](02-error-handling.md) | `CHECK` vs `INTERNAL_CHECK`, PyPTO exception types, IR source locations in failures |
| [Logging](03-logging.md) | The two independent logging subsystems and which one a message came from |
| [Runtime DFX Flags](03-runtime-dfx.md) | The five runtime diagnostic sub-features exposed through `RunConfig` |
| [Replaying an Existing `build_output`](03-runtime-replay.md) | Re-run, edit, and re-measure a compiled build directory without recompiling |
| [Simulator Trace Cleaning](04-simulator-trace-cleaning.md) | Converting MindStudio Insight binary dumps into readable traces |
| [Per-Task Ring Sizing](05-runtime-ring-sizing.md) | The three ring-size overrides on `RunConfig` and when to tune them |
| [Persistent L3 execution](06-persistent-l3.md) | Reusing one worker across prepared distributed programs |
| [Memory Map](07-memory-map.md) | Rendering a pass dump into an interactive HTML map of on-chip memory |
| [Compile and Execution Entry Points](08-entry-points.md) | Every compile and execution entry point, the layer it belongs to, and when to reach for it |
| [Distributed Operators](distributed_ops.md) | The N6 distributed op family — typed DSL access to collectives and primitives |
| [PTOAS Op Status Matrix](ptoas-op-status.md) | Which public and compatibility PTOAS ops the compiler currently emits |
| [FP4](fp4.md) | Logical vs packed FP4, cast policy, and hand-written `FP4E2M1X2` guidance |

## Automated PR Review

The `Codex Review` GitHub Actions workflow reviews non-draft pull requests when
they are opened, reopened, updated, or marked ready for review. It uses the
trusted workflow from the default branch, checks out the pull request head as
untrusted input, and posts the result from a separate GitHub-hosted job.

Findings with verified diff locations appear as inline review comments, which
can be replied to and resolved independently. Findings outside the diff, without
a location, or with an invalid location remain in the review summary. Reported
locations outside the diff link to the reviewed head or merge-base commit when
available. If optional location metadata requests fail, findings still appear
in the summary without inline anchors or unavailable links.
An identical inline comment from this workflow on the same commit is not posted
again. Location validation does not affect the approval policy: any finding
still prevents automatic approval.

### Discussion context and re-review

Every run fetches the current PR description, conversation comments, review
bodies, and all inline threads and replies, including resolved/outdated state.
These are evidence for the review, not authority to change its policy. Codex
must reassess disputed findings against expert explanations and current code,
and explain any remaining disagreement. A repeated actionable finding can cite
its existing Codex thread instead of posting a new inline comment; it still
prevents approval. Missing or oversized discussion input fails the run rather
than silently omitting context (the snapshot limit is 8 MiB).

The PR author or a collaborator with write, maintain, or admin permission can
request another review without pushing. Add a **new PR Conversation comment**
containing this exact command on its own line (explanations can precede it):

```text
@pypto-codex review
```

Quoted/fenced commands, edited comments, bots, ordinary issue comments, and
commands from other users do not start a review. After quoted text, leave a
blank line before the command: continuation lines remain inactive until that
boundary, even when they omit the `>` marker. Inline replies are included
as context but do not trigger the workflow: post the command in Conversation.
The command is handled by Actions; it does not require a GitHub account named
`pypto-codex` and does not invoke the separate `@codex` Cloud integration.

Conversation rollouts and the Codex session index persist in two Docker volumes
on the dedicated runner, keyed by base repository ID and PR number. Later runs
use `codex exec resume <session-id>` with the exact saved root session ID.
The checkpoint is updated only after a completed, validated review. Credentials
and configuration remain ephemeral. A runner change or removal of those volumes
starts a fresh session with the complete current GitHub discussion. Session
reuse preserves prior analysis and can benefit from prompt caching, but does
not guarantee fewer billed tokens; long histories can be compacted. Each run
still reviews the full current PR diff. Operators should remove the volumes
`pypto-codex-sessions-<repository-id>-<pr-number>` and its `-index` companion
when the PR no longer needs retained context, while no review is running.

Administrators enable or disable reviews with the repository Actions variable
`CODEX_REVIEW_ENABLED`. Set it to `true` to enable reviews; unset it or use any
other value to disable them immediately.

The review job requires a dedicated self-hosted runner labelled `Linux`,
`ARM64`, and `cpu-codex`. The runner provides the following host-managed
resources, none of which come from the pull request:

- `/home/ci-runner/.codex-ci/auth.json`, readable only by the runner account
- the digest-pinned review and proxy images in the local registry
- the `pypto-codex-egress` Docker network
- a healthy `pypto-codex-proxy-relay` container on relay port `17895`
- a loopback-only mixed proxy on `127.0.0.1:7895`

Codex runs as UID/GID `1002:1003` in a read-only container with a read-only
repository mount, dropped capabilities, resource limits, and an internal Docker
network. The GitHub token with pull-request write access is available only to
the separate publishing job. Review output is rejected if it contains an exact
long-form value from the Codex credential and is labelled as automated,
untrusted content when posted.

ChatGPT-managed Codex authentication is refreshed by a trusted weekly or manual
maintenance run that has no checkout and uses an empty temporary directory.
The refresh is serialized with reviews by a host lock and updates the persistent
credential in place. Pull-request review containers receive only an ephemeral
snapshot; they never mount or write the persistent credential.

### Automatic approval

Set `CODEX_REVIEW_AUTO_APPROVE=true` to allow clean reviews to submit an
`APPROVE` review as `github-actions[bot]`. The repository's Actions settings
must also allow GitHub Actions to create and approve pull requests. Disable
only automatic approval by removing this variable or setting it to `false`;
reviews continue to be posted. The repository's required CI checks still apply.
PRs authored by `github-actions[bot]` receive comments instead of approvals,
because GitHub does not allow authors to approve their own PRs.

The workflow checks out the authorized snapshot's exact head SHA and requests structured
JSON through `codex exec --output-schema`. The schema and publishing script
come from the commit supplying the workflow file (`github.workflow_sha`),
independently of the PR's base revision. The publisher approves only a complete
`pass` with zero findings, after checking the head SHA, base branch name, and
reviewed merge base are unchanged. Retargeting to a different branch at the
same SHA also invalidates the review, because the reviewed target branch has changed.
Invalid, empty, oversized, inconsistent, or incomplete output never approves.
The publisher does not read branch protection rules, CI results, or mergeability
to decide whether to approve. An `APPROVE` records the code-review result; it
does not merge the PR. GitHub evaluates configured CI checks and conflicts
separately when a merge is requested. The workflow does not require the head to
be up to date with the base before approving.
The review body records the examined base SHA, but an ordinary advance of that
branch does not invalidate approval when the merge base stays the same. A base
rewrite that changes the merge base invalidates the review because it changes
the reviewed diff. GitHub's configured stale-review rule may independently
dismiss an approval if the merge base introduces new changes after publication.
If merge-base verification fails, the publisher keeps findings visible in a
`COMMENT` summary without unverified inline anchors, and withholds approval.
The publisher checks revisions again after approval and dismisses its approval
if an update raced with publication or verification failed. Replacement results
first dismiss this workflow's previous approvals; human approvals are not modified.

Changing a PR's base branch also triggers approval invalidation before review.
A separate GitHub-hosted job runs only trusted workflow-revision code and
revokes this workflow's earlier approvals, even when reviews are disabled, the
PR is a draft, or a bot retargets it. A fresh eligible review starts only after
invalidation succeeds. Title/body-only edits do not start review or cancel an
existing review. Invalidation has a separate concurrency group that does not
cancel running jobs when another event arrives. GitHub event delivery and job scheduling are asynchronous, so
revocation is not atomic with the base edit.

Changes under `.github`, `.claude`, `.codex`, or `.agents`, and changes to
`AGENTS.md`, `AGENTS.override.md`, `CLAUDE.md`, or `.gitmodules` require human review. Renaming those
files does not bypass this restriction. If the complete changed-file list
cannot be verified, automatic approval is withheld.

Automatic approval is an AI assessment, not proof of correctness. Prompt
injection and missed defects remain possible even with structured output.
Enabling the Actions approval setting applies repository-wide to workflows
with pull-request write permission. The existing credential snapshot inside
the isolated reviewer and exact-string leak filter do not eliminate encoded
credential leaks. Prefer a credential-isolating API proxy for public-repository
review infrastructure. No model process receives the GitHub approval token.

The workflow must be merged into the default branch before these changes take
effect; pull-request-target runs use the default-branch workflow. Validate the
first clean review on a controlled PR and verify the review's commit ID and
`APPROVED` state before relying on it as a merge gate.

## See Also

- [PTO ISA reference](../reference/index.md) — the hardware model the backend targets.
- [Runtime documentation](https://www.pypto.ai/simpler/) — the scheduler that executes compiled programs.
