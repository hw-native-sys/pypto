---
name: pto-performance-report
description: Report baseline versus candidate operator performance from real Ascend hardware, workflow Worker kernel times and in-core simulation across relevant CI configurations.
---

# Report operator performance

Separate workload benchmark, workflow scheduling, physical Worker execution and
in-core simulation. Use [the report template](references/report-template.md).
Use pto-adaptation-performance-analysis for generated-code/synchronization causes,
pypto-user:critical-path-analysis for dependency-limited paths, and
pypto-user:incore-profiling for simulator collection. Existing evidence may be
analyzed directly; missing measurements remain explicitly unverified.

## Freeze baseline and full related CI coverage

Record operator/callers, purpose, lib/PyPTO/runtime/PTO ISA commits, actual imports,
patches, compiler/PTOAS/CANN versions, build config and kernel/fusion boundary.
Prefer the requested baseline; otherwise isolate the adaptation with unchanged
C++ rebuilt on the same toolchain. Keep independent original artifacts.

Trace workflows, CI scripts, manifests, matrix expansion, defaults and wrapper
overrides to all direct/indirect callers at the selected revision. Distinguish
PR device sweeps from performance manifests; they can use different EP/DP sizes.
Enumerate decode/prefill/full/reuse, dynamic controls, conditional platform paths,
card/rank counts and filtered/unrunnable cases. Do not infer hardware count from
a filename or silently substitute default shapes for explicit CI arguments.

Export ci-case-baseline.json/CSV containing stable case ID, source location,
entry/full effective arguments, platform/cards, call site, actual per-rank
shape/dtype/layout/effective rows, state/slot controls, tiling and fusion boundary.
Keep unknown dynamic values unresolved rather than using allocation limits as
actual values. Associate each case with measured pre-change precision, samples,
benchmark statistics, workflow/Worker times, captures, versions and original paths.
A coverage list is not a measurement. Reuse only version/config/input-compatible
evidence; otherwise collect again. Preserve prior baselines when CI changes.

Compare candidate against the same frozen list; retain the union when CI cases
change and explain additions/removals/parameter changes. Matching shapes do not
merge different workflows. Report total, confirmed callers, paired, passed,
failed, missing and excluded cases separately for precision, benchmark, swimlane
and simulator. Respect user-selected subsets/card restrictions and do not call
subset validation complete CI coverage. Representative shapes support tuning,
not final full-coverage claims.

## Workflow first, then standalone operator

For every included workflow, identify target nodes using orchestration/deps/name
maps and obtain per-rank/capture Worker events before aggregating. A fused target
can only be reported as its comparable fusion group; do not assign its full time
to an unobservable constituent. Missing workflow data cannot be replaced by a
standalone result labeled as CI workflow performance.

Prefer building an independent minimal operator test for every unique actual
configuration, reusing an existing entry if available. Freeze seed AND inputs,
dtype/layout, valid rows, dynamic controls, initial state, outputs, tiling and
fusion. Preserve CI-to-standalone mappings even when configurations can share a
test. Compile and pass correctness before collecting performance. If a standalone
entry cannot be built, explain the blocker. Compare standalone device latency and
kernel statistics with the workflow context; they are different measurement levels.

## Prepare and submit

Compile, initialize inputs, compute golden and verify native artifacts off-device.
Queued CPU preparation uses task-submit --no-device; hardware jobs only load the
prepared artifacts for precision/benchmark/capture. Verify replay compatibility,
reset state, and share frozen inputs/golden between versions without relaxing
thresholds. Never treat identical baseline/candidate failure logs as a precision
pass or proof of full output identity. Record which intermediate outputs passed.

Agents must execute task-submit outside the sandbox, using the tool's escalation
mechanism. Hardware uses --device auto --device-num N and reads TASK_DEVICE inside
the job. Honor pool/blacklist/topology restrictions; never bypass queue locks or
run architecture/device probes outside the allocation. Compare versions within
the same allocation when possible; record actual devices/rank mapping and state.

If outside-sandbox submission fails, preserve error and count; use the user's
retry interval/limit, otherwise wait three minutes and retry at most ten times.
A successfully queued job is not a failed submission: monitor the existing task,
do not resubmit to chase idle cards. Retain task ID, commands, output and exit
status; bounded retries do not authorize administrator policy changes.

## Separate benchmark and swimlane collection

Benchmark with swimlanes disabled. Follow CI sampling; absent an established
contract start with 10 warmups, 100 timed samples and at least three alternating
groups. Verify synchronization and timing boundaries, excluding compilation,
initialization, loading and warmup. Keep raw per-round/per-rank samples and label
host/end-to-end time separately. Record actual rounds, groups and replay mode;
failed-validation timing is diagnostic, not accepted performance.

Collect separate captures with supported enable_chip_swimlane=4 and dependency
export enabled. Verify final effective configuration, including CI wrapper
overrides, without changing workload/shape/path. Preserve raw records, deps.json,
name maps, orchestration/dispatch mapping, merged traces and commands in their
original build_out/build_output locations. Check truncation, dropped events,
missing ranks and graph/trace consistency. Capture process wall time is not a
benchmark and capture events are not independent benchmark rounds.

## Attribute and aggregate Worker events

Read process/thread metadata to identify Worker tracks, then map task IDs,
logical blocks, kernels, core type and rank via the current artifacts. Count
physical executions once, excluding scheduler/orchestrator/dispatch/finish and
duplicate views. Verify units/time domains. Use args["kernel-duration-us"] or
verified raw start/end; Worker span dur can include local_setup_us. Do not infer
pure-kernel intervals from Worker start plus duration alone.

Report per operator/core type/rank: event/core counts, min/mean/median/P95/max,
summed kernel time, interval union and first-to-last span. Union means at least
one execution active; sum double-counts parallel work; span includes gaps.
Unknown kernel intervals make pure union/span N/A. Separate AIC/AIV in MIX and
unaligned rank clocks. Pooling duration samples across cards is allowed only
with explicit sample count/granularity; it is not multi-card workload latency.

Workload statistics describe actual timed rounds. Workflow statistics describe
observable dispatch, setup/wait and completion phases. Worker statistics describe
physical executions in a capture. Compare absolute/relative changes per shape,
state P95 interpolation and uncertainty; use DAG evidence for dependency causes.
Missing evidence is N/A, never zero. Follow repository distributed metric policy
and label fastest-rank versus max-rank/end-to-end conventions explicitly.

## Include simulator results separately

For requested or existing simulator data, verify identical meaningful work:
inputs, controls, slots/state, effective shape, per-core batches and branch paths.
Use a5/dav_3510 for A5. Keep load-image/kernel/VF mapping, all tool versions,
commands/job IDs, original msprof trace.json, instruction CSV, summary and manifest.
Return original trace paths without moving artifacts or inventing pipeline plots.
A failed, empty or unexecuted branch is not zero-time performance.

Per kernel/core/block/subblock report total cycles/time and boundary, MTE2/MTE3/VF
dynamic counts, cumulative duration, active union and span; add CUBE/MTE1 for MIX.
Distinguish static VF functions, dynamic calls and instructions. When observable,
report VLD/EXEC/VST counts, summed time, union and overlap separately. Do not add
parallel durations into kernel wall time. Keep cycles unless frequency conversion
is verified. Explain startup, batch gaps, waits, overlap and drain through PCs,
set/wait/membar and UB producer-consumer ranges, auditing baseline omissions too.

Compare simulator kernel time, real Worker duration and standalone benchmark only
with their different boundaries and workload limitations stated. Simulation is
not hardware precision/performance acceptance and a standalone simulation is not
a complete CI workload simulation. Hardware-only tasks mark simulation uncollected;
simulator-only reports mark hardware unverified. Do not expand execution scope
merely to populate the report.
