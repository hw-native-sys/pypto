# Operator performance report

Lead with per-configuration results and measurement boundaries. Without an agreed
threshold, state data and uncertainty rather than declaring acceptance.

## Scope and provenance

Record target/callers/fusion boundary, baseline/candidate source and imported
versions, patches, compiler/PTOAS/runtime/PTO ISA/CANN, architecture, devices/ranks,
input/golden/state, commands/job IDs, correctness and original evidence paths.

## Complete related CI baseline

| Case/source/arguments | Caller and actual per-rank configuration | Cards/platform | Baseline/candidate precision | Benchmark/capture/simulator status or exclusion |
| --------------------- | ---------------------------------------- | -------------- | ---------------------------- | ----------------------------------------------- |
| N/A | N/A | N/A | N/A | N/A |

Report total callers, paired, passed, failed, missing and excluded cases per
validation type. Keep CI-to-standalone mappings and conditional paths explicit.

## Workload and standalone benchmarks

| CI case / standalone configuration | Rounds/warmups/groups | Baseline median/P95 | Candidate median/P95 | Change and variability | Precision |
| ---------------------------------- | --------------------- | ------------------- | -------------------- | ---------------------- | --------- |
| N/A | N/A | N/A | N/A | N/A | N/A |

Specify units, synchronization boundaries, raw sample paths, replay/reset policy,
per-rank aggregation and timing source. Do not substitute process wall seconds,
swimlane durations or core sums for device latency.

## Workflow and Worker kernels

| Capture/rank/task/kernel/core type | Executions/cores | Baseline min/mean/median/P95/max | Candidate min/mean/median/P95/max | Sum / union / span | Change |
| ---------------------------------- | ---------------- | -------------------------------- | --------------------------------- | ------------------ | ------ |
| N/A | N/A | N/A | N/A | N/A | N/A |

Explain task matching, fusion, metadata/schema/units, deduplication, local setup
exclusion, raw interval joins, capture integrity and unaligned clocks. Mark
unknown pure intervals N/A. Workflow setup/wait/completion stays separate.

## In-core simulator comparison

| Configuration/core/block | Image and raw trace | Baseline/candidate total cycles or verified time | MTE2/MTE3/VF count and cumulative time | Active union/span/overlap | Precision/limitations |
| ------------------------ | ------------------- | ------------------------------------------------ | -------------------------------------- | ------------------------- | --------------------- |
| N/A | N/A | N/A | N/A | N/A | N/A |

For MIX add CUBE/MTE1; for observable VF internals add VLD/EXEC/VST. Separate
static functions, dynamic calls and instructions. Associate startup, gaps, sync
and drain with PC/address/dependency evidence. State simulator versus hardware
boundary differences; uncollected simulation or hardware remains unverified.

## Conclusion and evidence

State supported conclusions, precision failures including baseline failures,
uncertainty, missing coverage and remaining dependencies. Link raw samples,
records, deps/name maps, original traces, commands, versions and logs. Diagnostic
binary patches and failed-validation timings cannot establish accepted speedups.
