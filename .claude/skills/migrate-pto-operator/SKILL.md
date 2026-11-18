---
name: migrate-pto-operator
description: Migrate a user-selected C++ extern operator to PyPTO Tile, Cube, MIX, or RegScope, preserving baseline artifacts and validating numerical correctness and performance.
---

# Migrate a C++ extern operator to PyPTO

## Select the target and freeze the baseline

Require a user-selected operator, unique entry point, source path, or explicit
selection from a candidate list. Reuse a selection already made in the session.
If only a model/directory or ambiguous name is given, inspect callers and list
candidates, prioritizing VEC_SCOPE implementations, then ask for the target.
Do not create a migration worktree, edit, build, or run while selection is pending.
Batch migrations cover only selected operators and helpers needed by their callers.

Create an isolated worktree in the target repository, verify git-common-dir,
and rebase onto the agreed baseline. Record before/after commits and dirty state.
Archive any existing migration patch without silently carrying or discarding it.
Identify the actual extern, headers, helpers, and caller shapes before editing.

Keep two independent baselines: previously passing artifacts and the unchanged
C++ operator rebuilt with the current framework/toolchain. The former does not
replace the latter. Preserve original artifact locations and never overwrite a
baseline to reconstruct it later. Archive:

- Source, caller and helper/header files, workspace diff, lib/PyPTO/runtime/PTO
  ISA commits, actual imported package paths, compiler/PTOAS/CANN versions.
- Inputs, golden, actual outputs, state/slot initialization, commands, config,
  pass dumps, PTO, generated C++, extern/VF helpers, objects, device binaries,
  disassembly, and logs. Explicitly export outputs if the harness omits them.
- A SHA-256 manifest and mapping from each kernel/helper to its executed image,
  including images in compiler caches outside build_output.

Verify the archive is readable before editing. Do not copy native extensions
between checkouts. Runtime and simpler changes are skipped unless explicitly
authorized; record their gaps and continue independent work.

## Map and implement the current C++ behavior

Use the frozen current C++ implementation as the semantic reference. Historical
Python is for comparison only, not a source for restoring old algorithms or
schedules. Map math, tiling, task/block assignment, dtype, rounding, layouts,
mask inactive lanes, synchronization, and dependencies. Translate VEC_SCOPE
register computations to pl.reg_scope and retain surrounding PyPTO operations.
Choose Tile/Cube/MIX when appropriate; do not force RegScope into other work.
Explain and validate any deliberate departure from the C++ order or schedule.

For an unsupported interface, isolate DSL/IR verification, codegen, PTOAS, and
hardware restrictions with a minimal reproducer. Implement authorized PyPTO
support with meaningful tests and synchronized API/docs, rather than silently
retaining the entire extern. Report blockers; do not bypass them with simulation
or relaxed comparisons. Runtime changes still require explicit authorization.

For every framework change, record separately from lib changes:

- The C++ requirement, failing interface/layer, and reproduction evidence.
- Why existing APIs are insufficient, chosen solution, before/after behavior,
  and dtype/mask/rounding/layout/synchronization implications.
- Files/functions/layers, compatibility impact and related OPs, distinguishing
  newly introduced changes from pre-existing branch changes.
- Tests, compile/hardware evidence, artifact paths/task IDs, limitations, and
  unverified coverage. Never equate compilation with hardware validation.

## Validate and compare actual artifacts

First extract the smallest operator test matching real CI/caller shape, dtype,
layout, effective rows, slots, state, fusion, and tiling. Add necessary tail,
empty-task, and branch boundaries. Keep baseline/candidate artifacts independent.
Do not expand to full-model validation without authorization.

Compile and prepare inputs off-device before submitting hardware replay.
Use task-submit --no-device for queued preparation and --device auto for actual
hardware work; read TASK_DEVICE inside the job, without restricting cards to 0-3.
Agents must submit outside the sandbox. Follow pto-performance-report for bounded
queue retries and evidence. Never compile inside a hardware allocation or bypass
queue locks. Record unavailable/maintenance states as blocked validation.

Precision uses identical saved inputs and original golden/thresholds for all
outputs and InOut state. Reset state before each run, check replay semantics and
repeated-slot comparator assumptions, and fail NaN/Inf. A fixed seed alone does
not replace saved inputs. Recompile incompatible replays rather than using old
binary results as new-implementation evidence. Document unfinished subpaths.

Disassemble both executed images using a tool supporting the architecture.
Preserve raw output and normalize only addresses/nonsemantic temporary symbols;
retain instructions, immediates, branches, memory offsets, and barriers. Compare
rounding, loops/carried values, vector/gather/reordering instructions, spills,
UB accesses, and MTE2/V/MTE3 ordering. Explain semantic/performance differences;
PTO/C++ diffs alone do not complete machine-code comparison.

Use pto-performance-report for measurements and full authorized CI coverage:
prefer standalone tests matching caller configurations, separate benchmarks from
level-4 swimlanes, and compare the same devices/toolchain/config/input. Use CI
sampling or, absent an established contract, 10 warmups and 100 samples per group,
with at least three alternating groups. Preserve raw samples; report median,
P95, variability, sample counts, and changes. Device time, workload time, kernel
busy time, and runtime wall seconds are distinct. Missing statistics stay N/A.
Without an agreed performance threshold, report measurements and uncertainty
before asking for acceptance; do not declare all acceptance criteria satisfied.

Map Worker events through metadata, deps, orchestration, dispatch and name maps.
Count physical AIC/AIV executions once, excluding scheduler/dispatch views;
use kernel-duration-us after checking units, not Worker dur containing local
setup. Report per-suboperator counts, cores, min/mean/median/P95/max, summed time,
interval union and span only when true kernel intervals are known. Keep MIX
core types and unaligned rank clocks separate. Inspect dropped/truncated events,
dummy joins, sync_start and dependency edges; static kernel count is not execution
count. Use pto-adaptation-performance-analysis for simulator and synchronization
analysis; simulator results do not replace hardware acceptance.

## Archive the adaptation log

For each completed operator/iteration, save adaptation.log in its allowed local
build/build_output artifact directory, never as a source-controlled machine log.
Index target/callers, baseline/candidate versions, patches, input/golden/output,
framework reasons, generated code/disassembly, compile and numerical outcomes,
benchmark statistics, raw swimlanes/Worker mapping, simulator evidence if used,
commands/task IDs, dependencies and remaining limitations. Separate diagnostic
binary patches from source fixes and hardware from simulator results. Preserve
prior logs and original artifact locations. Finally audit related OPs for the
same missing capability; list follow-ups without silently expanding migration.
