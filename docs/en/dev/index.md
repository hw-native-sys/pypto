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
| [FP4](fp4.md) | Packed E2M1x2/E1M2x2 types, byte-carrier ABI, and existing E2M1 operator support |

## Comment-triggered full CI

After rollout, PR creation and updates run `changes`, `toolchain`, `pre-commit`
and `clang-tidy`, followed by a lightweight `lint-gate`. The separate `Docs / build` check also runs. Documentation-only
PRs retain the automatic `clang-tidy` exemption. No device tests or wheel builds
start automatically for PRs.

With tiered CI enabled, the automatic PR run loads its controller from the
trusted PR base SHA. Editing `ci_control.py` in the PR therefore does not change
tier selection. Before activation, it uses the workflow revision so this change
can be tested before the helper exists on main.

This is a scheduling convention for the existing `pull_request` flow, not a
complete protection against malicious runner use. A PR can still modify its
workflow, remove a job condition, or add a job that targets an available runner.
The `/ci full` command validates the PR revision, and its result publisher
verifies the trusted dispatch, but neither restricts arbitrary PR-authored workflows.
Enforcing that boundary requires separate GitHub runner access policies and
trusted workflow restrictions. Those administrative settings are unchanged.

Anyone who can comment on a PR, including external contributors without
repository write access, can post exactly this comment in the PR conversation:

```text
/ci full
```

The `CI Command` workflow requires an open PR targeting this repository's `main`
branch and dispatches `CI` from trusted `main`. It fixes the PR head, base and test-merge SHAs and
verifies both GitHub's current test-merge SHA and its parents before any tests start. Fork PRs are supported.
A conflicting PR, or one whose merge is still being computed, must be resolved
or retried. The command's Actions summary records acceptance or rejection.
An identical request already queued/running is not dispatched again. A verified
successful full result for the same head and base also makes the comment a no-op,
so another commenter cannot reset a green gate. Comment
access allows users to request device tests; it does not grant repository write
or merge permission, and duplicate detection is not a general rate limit.
After PR and revision checks, the command writes `ci/full=pending`
before dispatching, clearing any previous success while the publisher is queued.
A read-only admission job validates the PR and pins the head before joining
the publisher's PR/head queue. Inside that queue, the command checks the PR
and its revision again, writes pending, dispatches, and binds the pending status to
the returned run ID. A moved head is rejected rather than written under the wrong lock.
If the initial status write fails, no run is dispatched. If dispatch or run-ID
binding fails, the unbound pending marker is retained and older publishers cannot
clear it. Retry `/ci full`; if a run was accepted despite a lost response, wait for
that run to finish before retrying. Publishers preserve the bound newer run even
when run-history searches have not yet indexed it.

The comment opens the downstream test stage after `pre-commit` and `clang-tidy`
have passed in the latest matching automatic PR run. Matching includes the PR
number, head SHA, base SHA, workflow identity and fork identity. A docs-only PR
may reuse a skipped `clang-tidy` after its changed paths are verified. Missing,
running or failed lint rejects the request; it does not reserve a runner while
waiting. Pushes or base changes require matching lint again.

The downstream run skips both lint jobs. A lightweight gate verifies their
reuse before tests start; revision checks and toolchain preparation still run.
Full includes the reused lint result, wheel warm-up, unit, example, codegen, system, distributed,
pypto-lib and A5/simulator tests. Core tests execute even for documentation-only
PRs. `docs-examples` applies when documentation changes; `serving-guard` applies
when `PTOAS_VERSION` changes. A failed wheel warm-up is advisory because test
jobs build locally if necessary; every applicable test must succeed.

### Results and retries

The PR keeps its original lint checks and displays each downstream job as
`full / <job name>`, including matrix legs and links to job logs. `CI Full Progress`
reads GitHub metadata on a GitHub-hosted runner and refreshes these checks roughly once a minute for one active full run, without executing PR code or giving test jobs write tokens. No
monitor runs while waiting for a comment. Polling slows down with concurrent
monitors and the shared API budget, reserving 200 requests for gate writers and
other automation; transient API errors use bounded backoff. Each live monitor is bounded to five
hours; a separate completion event reconciles final results even after a monitor
failure or timeout. Progress checks are informational: require `ci/full` for
merge, not every `full / ...` check (some tests are conditional or advisory).

`ci/full` is a commit status on the PR head, initialized by `CI Command` and
updated by `CI Full Status`.
It stays pending while full runs and succeeds only after the `full-result`
job confirms all applicable tests passed. Skipped, missing, failed or cancelled
required tests cannot satisfy it. The publisher runs on a fresh GitHub-hosted
runner, reads GitHub metadata, and never executes PR code or downloads artifacts.
Before joining the status writer concurrency group, the router validates the
event's main branch, workflow identity, repository and full-run title. Unrelated
runs cannot displace a pending publisher. Commands and publishers use the same
`queue: max` concurrency group, so older events cannot replace an already queued
completion writer. GitHub permits 100 waiting jobs per group; excess arrivals are
cancelled, and a cancelled command or completion publisher must be retried.
Test jobs do not hold status-write or Actions-write tokens. The entire `CI`
workflow declares `cache-mode: read`: GitHub's cache service prevents cache
writes, including direct API calls from PR code. Hosted wheel sharing uses
artifacts from the same run, with separate names for each producer attempt.
Expired/missing artifacts fall back to building locally. This also makes CI's
main-branch cache access read-only; other workflows and self-hosted disk caches
retain their existing policies.

Use **Re-run failed jobs** on the full run in GitHub Actions. The original test
revision is preserved; the finalizer combines successful dependencies from the
previous attempt with rerun results. If only the finalizer is missing, rerun that
job too. The status publisher handles rerun events and ignores superseded requests.
If publication finds that the PR has moved to another head, closed, or changed
its target, it invalidates the original tested head unless a verified newer request owns its
status. Resetting or reopening the PR cannot revive an old green rerun result.
Only dispatches from `main` can block a
duplicate request or supersede an earlier one; feature-ref dispatches are ignored.
Duplicate detection scans all statuses together from the PR creation time, splitting
creation-time ranges at GitHub’s search limit. Moving between unfinished statuses
cannot hide an existing request from that scan.
A rerun of an old request cannot cancel a
newer request. A new commit needs a new `/ci full` comment. If main advances
before result publication, the old-base result fails. Update the PR branch,
wait for matching lint and post `/ci full` again. This check is made at publication time; an already published success
is not automatically invalidated by a later main push. Requiring an up-to-date
branch at merge time is a separate ruleset policy, unchanged by this PR.
Changed-path classification compares the pinned base and head SHAs, so later
PR pushes cannot change which conditional tests run. If the comparison fails
or reaches GitHub’s 300-file limit, all conditional checks are enabled.

If the live-run lookup fails, the publisher validates the GitHub event’s workflow,
repository, branch, and run identity before invalidating the gate. Event data
can never grant success; a verified newer published request or later attempt of
the same run is still preserved. Later attempts are verified through their exact
attempt metadata, not trusted solely from the status URL.
If a run-history lookup hits GitHub's search limit, or workflow/PR/job metadata cannot
be read, the publisher preserves a verifiably newer published request or attempt; otherwise
it marks `ci/full` failed. An old
green result must not survive an unverifiable rerun. Post `/ci full` again to
create a recent request with a small history window.
The external `pr-rerun` service remains unchanged: it can still rerun eligible
ordinary PR CI, but does not discover/authorize these dispatched full runs.

`main` pushes, manual CI runs without PR inputs, Daily CI and Codex Review keep
their existing policies. Manual PR inputs are intended for the comment handler;
leave all four empty to run CI against main.

### Operator rollout

The repository variable `CI_COMMENT_TIERS_ENABLED` is an activation switch.
Unset or false preserves existing automatic full CI while the required checks
are migrated. No GitHub App secret, public webhook endpoint or service update
is needed.

1. Merge the workflows and helpers onto `main`, leaving the switch unset.
   Update the test PR branch to include this version and let its automatic lint
   finish. Older runs without the PR/head/base lint title cannot be reused.
2. On a test PR, post `/ci full`. Validate fork checkout, an actual failing test,
   **Re-run failed jobs**, and the resulting `ci/full` commit status. Confirm
   external commenters can trigger full, invalid PRs cannot launch it, and
   stale-head results cannot approve a newer revision.
3. Add `ci/full` to the main ruleset's required status checks, selecting the
   GitHub Actions source where available. Keep `pre-commit`, `clang-tidy` and
   existing review requirements. Do not require the dispatch job `full-result`
   itself: it belongs to the main workflow revision, not the PR head.
4. Once `ci/full` is required, remove the former individual heavy-test contexts
   from the required list, including the `unit-tests` matrix context. They may
   be skipped or absent in lint runs; coverage is now enforced by `ci/full`.
5. Set `CI_COMMENT_TIERS_ENABLED=true`. Subsequent PR runs use lint by default.

For rollback, unset the switch to restore automatic tests; keep the `ci/full`
requirement until the previous heavy-test requirements have been restored.
This implementation targets the current PR merge flow, not merge queues;
`merge_group` support is required before enabling a merge queue.

## Automated PR Review

The `Codex Review` GitHub Actions workflow reviews only non-draft pull requests targeting `main` when
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
prevents approval. The model judges semantic equivalence using the discussion;
the publisher accepts only a workflow-bot root comment from this PR whose file
matches the finding or was renamed to the finding's current file according to
the PR file metadata. Matching line numbers or diff sides are not required.
Missing or oversized discussion input fails the run rather
than silently omitting context (the snapshot limit is 8 MiB).

Approval uses the discussion snapshot collected at the start of that review.
Later discussion changes do not invalidate an otherwise valid approval; request
another review after posting a correction when a new assessment is needed.

The PR author or a collaborator with write, maintain, or admin permission can
request another review without pushing for PRs targeting `main`. Add a **new PR Conversation comment**
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

The comment run first looks for a verifiable `pull_request_target` Codex Review
run for this PR's current head and base branch. It reuses a queued/running run,
or reruns **all jobs** of a completed run (including successful runs when a new
assessment is requested). This updates the existing PR Actions checks and
rebuilds both context and result artifacts under the same attempt number.
The comment run itself remains a separate, short routing run; its Actions
summary links to the selected PR run. Requests for the same PR are serialized,
with up to 100 pending preparation jobs retained; further arrivals are canceled
when that queue is full.

Configure the repository secret `CODEX_REVIEW_RERUN_TOKEN` with a dedicated
fine-grained PAT restricted to this repository with **Actions: write**. It is
used only by the trusted preparation step for comment commands, never passed
to the PR checkout or review container. Missing credentials or API failures
fail visibly instead of silently leaving PR checks unchanged. Read-only lookup
uses `GITHUB_TOKEN` with **Actions: read** and **Checks: read**.

Routing starts from the PR number and fully paginates the current commit's
GraphQL `statusCheckRollup`, following Actions check runs to their workflow run
IDs by validating each workflow run's `resourcePath` against the exact repository
path. This avoids depending on a GraphQL integer width. Each page must still
refer to the requested PR head. It verifies the
workflow, event, base repository, and artifact's repository, PR number, head
SHA, and base ref, without relying on workflow-run source metadata. A changed
head stops routing, including standalone fallback.
A small identity artifact is retained for 30 days; discussion content still
expires after one day. Older runs can be verified from an unexpired discussion
snapshot. If no verifiable run exists within GitHub's 30-day rerun window, the
command falls back to standalone review and explicitly reports that existing
PR Actions checks will not be replaced. Rerunning uses the original workflow
revision; it does not upgrade an old run to newly merged workflow code.

Conversation rollouts and the Codex session index persist in two Docker volumes
on the dedicated runner, keyed by base repository ID and PR number. Later runs
use `codex exec resume <session-id>` with the exact saved root session ID.
The checkpoint is updated only after a completed, validated review. Credentials
and configuration remain ephemeral. A runner change or removal of those volumes
starts a fresh session with the complete current GitHub discussion. Session
reuse preserves prior analysis and can benefit from prompt caching, but does
not guarantee fewer billed tokens; long histories can be compacted. Each run
still reviews the full current PR diff.

Closing a PR deletes its `pypto-codex-sessions-<repository-id>-<pr-number>`
volume and `-index` companion, whether merged or unmerged, including draft or
bot-authored PRs and when reviews are disabled. Cleanup and review use the same
runner lock. Before creating or resuming volumes, a trusted host query checks
the live PR state/head and its latest close event. Volume labels record that
close event, so a reopened PR starts fresh and delayed cleanup preserves its
new session while removing older volumes. A queued review of a closed, draft,
or superseded head skips without publishing an artifact. GitHub state can
change after a read; queued review and cleanup jobs recheck under the lock.

Ephemeral review containers and private networks carry repository, run and attempt
labels. Their identities are recorded before startup. Shell traps and a separate
`always()` step remove this attempt's containers, private network and credential
snapshots, including when the review is cancelled. Session volumes and the shared
egress network/relay are preserved.

Before starting another review, trusted host code reconciles leftover resources
under the same runner lock. It queries GitHub with **Actions: read** permission
and removes resources only when their exact owning attempt is `completed` and
its repository and workflow identity match. Discovery uses ownership labels only;
legacy resources without complete labels require a one-time manual cleanup.
Active attempts and other repositories' resources are retained; API failures stop
startup before any stale-resource deletion. A runner outage can prevent the
`always()` step; reconciliation recovers Docker resources on the next review
on that host. It does not recover temporary credential files left by a host outage.

Administrators enable or disable reviews with the repository Actions variable
`CODEX_REVIEW_ENABLED`. Set it to `true` to enable reviews; unset it or use any
other value to disable them immediately. Close-event session cleanup remains enabled.

The review job requires a dedicated self-hosted runner labelled `Linux`,
`ARM64`, and `cpu-codex`. The runner provides the following host-managed
resources, none of which come from the pull request:

- GitHub CLI (`gh`) for trusted host-only state checks; its read token is never
  passed to the model container
- `/home/ci-runner/.codex-ci/auth.json`, readable only by the runner account
- the digest-pinned review and proxy images in the local registry
- the `pypto-codex-egress` Docker network
- a healthy `pypto-codex-proxy-relay` container on relay port `17895`
- a loopback-only mixed proxy on `127.0.0.1:7895`

The standalone CI helpers in `.github/scripts/` support Python 3.8 and newer.
Their type annotations are deferred so importing them does not require Python
3.9 built-in generics or Python 3.10 union operators. This compatibility is
limited to these host-side scripts; PyPTO itself still requires Python 3.10+.
The helper tests run as part of the regular Python 3.10 unit-test job. CI has no
dedicated Python 3.8 helper-test check. The review workflow uses the host
interpreter and does not install Python or depend on the Python version inside
the review container.

Each review checks out the exact PR head into a fresh, full Git repository in a
unique directory, without reusing the runner's cached object store. Before
starting the container, the host fetches the complete base, head, and trusted
workflow history, rejects shallow or partial clones, verifies reachable objects,
and renders the full merge-base-to-head diff, including binary contents. Missing
objects fail the job before review starts; the container does not need Git fetch
access. The temporary checkout is removed after the review.

Codex runs as UID/GID `1002:1003` in a read-only container with a read-only
repository mount, dropped capabilities, resource limits, and an internal Docker
network. The GitHub token with pull-request write access is available only to
the separate publishing job. Review output is rejected if it contains an exact
long-form value from the Codex credential and is labelled as automated,
untrusted content when posted.

Review result uploads have three attempts, each limited to two minutes. Failed
attempts wait 10 and then 30 seconds before retrying, replacing any leftover
artifact with the same run-specific name. Retries stop after a successful upload
or cancellation. If all attempts fail, the review job fails and publishing does
not run; a completed assessment alone does not make the job successful.

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
