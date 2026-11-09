---
name: migrate-pto-operator
description: Use when migrating a user-specified C++ extern operator to PyPTO and validating the result against the current implementation.
---

# Migrate a C++ extern operator to PyPTO

## 0. Resolve the user-selected target

Require an operator name, entry point, source path, or explicit selection from a
candidate list. Reuse a selection already made in the conversation. If only a
model or directory was given, or multiple implementations match, inspect the
call sites and ask the user to specify the target before creating a worktree,
editing code, building, or running a migration. Read-only candidate discovery
may prioritize kernels with VEC_SCOPE. For a batch, migrate only the selected
operators and the helpers needed by their call paths.

## 1. Freeze the baseline

- Work in an isolated checkout of the operator repository and record both the
  operator and compiler revisions, worktree diffs, toolchain versions, and
  actual package import paths. Follow the requested base branch and environment.
- Identify the current C++ extern and its helpers, the real caller, and the
  smallest CI-equivalent case with the same call shape, dtype, layout, control
  inputs, and initial state. Include relevant tail and empty-work cases.
- Preserve a same-toolchain C++ baseline before editing. Keep historical runs
  separately. Save inputs, golden and actual outputs, pass dumps, PTO,
  generated C++, compile commands, final device image, disassembly, and run
  logs. Record the entry-to-image mapping and artifact hashes. Use distinct
  build directories; retain original artifact locations.

## 2. Port the current implementation

- Map algorithm stages, partitioning, rounding, layout, buffer lifetimes, and
  task dependencies to PyPTO. Use historical Python only to understand the
  differences. Use RegScope for register computation in VEC_SCOPE, and choose
  Tile, Cube, or MIX according to the actual computation and schedule.
- If a needed interface is missing, isolate a reproducer and locate the gap in
  the DSL, IR verifier, codegen, PTOAS, or hardware. Fix authorized compiler
  gaps with focused validation. For each change, record the triggering
  operation, reason, semantic impact, affected layers and related operators,
  and evidence. Keep compiler changes distinct from operator changes.
- Leave PyPTO and simpler runtime unchanged unless explicitly authorized.
  Report gaps that need runtime work and continue the independent parts.
- Compile the minimal case offline before submitting any device run. Compare
  actual disassembly by functional stage, preserving operands, branches,
  memory offsets, and synchronization in any normalized comparison.

## 3. Validate correctness and performance

- Use the archived input and existing golden and thresholds for every output
  and InOut tensor. Restore mutable state between runs and verify comparator
  assumptions for stateful writes. Treat NaN/Inf as failures; do not relax
  thresholds to make the migration pass.
- When the project uses task-submit for A5, submit precompiled cases with
  `--device auto` and consume `TASK_DEVICE` inside the job. If device validation
  is unavailable or out of scope, record the limitation.
- Compare device timings on the same card with identical inputs, settings,
  and toolchain. Confirm the harness timing boundary. Use the existing
  benchmark protocol, or start with 10 warm-ups and 100 measured iterations
  per version in at least three alternating groups. Preserve samples and
  report median, P95, variation, and relative change. Report large-operator
  timing separately from the minimal case.
- Capture both chip swimlanes separately from benchmarking, when supported.
  Validate completeness and map Worker View AIC/AIV execution events to
  suboperators using task and kernel metadata. Exclude scheduler spans and
  duplicate views. Report event count, kernel duration distribution, summed
  durations, interval union, and first-to-last span. Label Worker spans
  separately when true kernel intervals are unavailable. Use task dependencies
  before attributing idle gaps to scheduling or data waits.
- For instruction or pipeline diagnosis, use the
  `pto-adaptation-performance-analysis` skill if available. Treat simulator
  measurements separately from device acceptance. If the performance threshold
  is unset, report the measurements and leave acceptance pending.

## 4. Archive one report per operator

Write an `adaptation.log` in the operator's artifact directory. Index the two
builds, exact commands and revisions, input and golden files, compiler change
rationale, compile and precision outcomes, benchmark samples, swimlanes,
suboperator statistics, disassembly, and outstanding gaps. Record task IDs
where available. Audit related operators for the same compiler gap and list
follow-ups outside the selected migration scope.
