---
name: pto-adaptation-performance-analysis
description: Compare C++ extern and PyPTO adaptations through generated Tile/VF code, final VPTO/LLVM, disassembly, and A5 simulator pipelines to explain and tune performance differences.
---

# Analyze adaptation performance

Use current C++ as the implementation reference; historical Python is analysis
context only. Use migrate-pto-operator for migration and pto-performance-report
for formal hardware measurements. Default to existing artifacts and in-core
simulation; do not initiate hardware runs or modify runtime without authorization.

## Establish evidence and collect intermediate artifacts

Read target repository instructions and setup guidance. Record actual imported
packages, checkout/workspace changes, architecture, toolchain and executed image.
Keep independent baseline/candidate directories; reference original paths without
moving generated artifacts or hardcoding machine/user paths into shared skills.
Resolve pinned dependencies from the selected checkout, not arbitrary newest
versions. Build in that worktree; never transplant another checkout's extension.

Compare Python source first: calls, selected branches, tiling and algorithm.
Locate the actual final host/Tile C++, extern C++ helpers, candidate VF helpers,
and their compile/link commands, rather than presenting pre-PTOAS code as final.
Map each function to the device image actually loaded. For each VF preserve:

1. Original RegScope Python with enclosing control flow and shapes.
2. PyPTO codegen PTO input before PTOAS transformations.
3. Final VPTO/micro MLIR after relevant PTOAS passes, before LLVM lowering.
4. LLVM-dialect MLIR and textual LLVM IR (.ll), labeled separately.
5. C++ baseline VecScope source with invoked helpers/headers.
6. Objects, executed ELF, raw disassembly, pass dumps, commands and tool versions.

Create a local vf-artifacts.json index with original paths and symbol/call-site
mapping. Use compiler-supported pass dumping and retain commands; missing stages
are explicitly unverified, never renamed to masquerade as another IR layer.

## Compare outside and inside VF

Outside RegScope/VecScope compare task assignment, DMA bytes/count/strides,
batching, Tile transformations, UB allocation/reuse, writeback order and tails.
Separate external synchronization from VF-internal barriers. Equal transfer counts
do not prove equal scheduling: a batch can have become per-row transfer/wait pairs.

Inside VF first establish algorithm equivalence: math, reductions, types,
conversions/rounding, inactive mask semantics, tails and intermediate consumers.
Do not count omitted required work as an optimization. Align algorithm stages
across PTO, final VPTO micro instructions, LLVM IR, baseline CCE intrinsics and
actual disassembly. Explain extra load/store/select/reorder/spill/barrier operations
by locating the first introducing stage and following result users.

Inspect symbols, call sites, loop bounds, addresses, control-register setup and
restoration, and vector operations. Distinguish static VF functions, static call
sites and dynamic calls. Check actual fusion rather than inferring it from source
scope grouping; record compiler fusion options. Normalize addresses/temporary
symbols only, preserving immediates, offsets, branches and synchronization.

## Simulate and analyze pipelines step by step

Read pypto-user:incore-profiling for installed tool usage and
[the simulator reference](references/incore-simulator-profiling.md) for workload
checks. List functions before selecting the target. Use a5/dav_3510 for A5 and
verify the actual SoC/image. Do not force a PTO-only exporter onto a C++ extern
without matching PTO. Run a minimal smoke before collecting both variants.

Keep inputs, effective shape, scalar controls and per-core work aligned, recording
block/subblock/loop counts, dynamic allocation bounds and standalone input edits.
Verify nonempty expected pipelines: matmul needs CUBE/MTE1/MTE2, vector work needs
VECTOR. Zero-filled controls or default scalars can create empty runs. Check each
MIX/split-AIV branch. Keep manifest_export.csv, summary.txt, instruction CSV, raw
trace.json, image, commands, raw_simulator and instr_metrics.json when available.
Return original msprof trace paths; do not invent replacement pipeline diagrams.

Follow [pipeline and synchronization evidence](references/pipeline-and-sync.md):

1. Align kernel/core/block/subblock windows and batches at the same entry origin.
   List every dynamic MTE2 transfer, VF call and MTE3 writeback, including init,
   main work and tails. Record issue, execution start/end, relative start,
   duration, bytes/work and gaps; distinguish observable from inferred fields.
2. Per pipe report count, summed execution, active interval union, first-to-last
   span, active fraction, gaps and overlaps. Keep VF-call and internal VLD/EXEC/VST
   levels separate. Late startup, slow execution and batch gaps are different.
3. Prioritize cross-pipe stalls: follow MTE2 -> VF -> MTE3 and reuse dependencies,
   particularly VF -> next MTE2 and MTE3 -> next VF. Trace producer completion,
   set/notify, wait release and consumer issue/start using PC, flags, direction,
   batch, UB address and code. Separate readiness, buffer reuse, broad waits,
   redundant barriers, resource contention and insufficient evidence. Determine
   whether the wait extends kernel completion or is hidden by another pipe.
4. Then inspect each pipe's work. DMA: bytes, addresses/strides, batching, tails,
   execution versus issue gaps. VF: initialization/main function counts, dynamic
   calls, VLD/EXEC/VST overlap, math/gather/reorder, spills and internal membar.
   Trace slower/new work back through codegen, VPTO/LLVM and machine code.
5. Validate attribution with isolated variables and necessary ordering preserved.
   Present first batch, steady state, batch gaps and drain, without adding
   overlapping pipe durations as wall time. Deliver per-task, per-pipe and sync
   evidence tables, retaining unexplained intervals and visibility limitations.

## Audit synchronization and MIX

Use actual read/write ranges, iteration paths and buffer lifetimes to classify
necessary, narrowable, proven redundant, or unresolved synchronization. Audit
C++ for implicit ordering or missing synchronization too. Absence in C++, locally
disjoint addresses or faster execution after removal alone does not prove safety.
Find the first inserting pass/IR and preserve a minimal source/PTO reproducer.

In MIX confirm AIC/AIV producer-consumer ordering through generated and matched
set_intra_block/wait_intra_block or an equivalent mechanism. Distinguish source,
machine-code confirmation and simulator observation. Pure AIV/AIC is not applicable.
Authorized binary/NOP patches are diagnostic variants with separate correctness,
performance and coverage; official fixes start in Python or an authorized compiler
layer. Simulator results never replace end-to-end hardware acceptance.

## Tune by aligning confirmed differences

Only tune when authorized; otherwise deliver evidence and proposed changes.
Maintain a difference ledger: ID, baseline/current code/PC/trace, desired behavior,
affected batches, observed cost, attribution confidence, edit layer and validation.
Fix semantics/workload first, then critical cross-pipe stalls, DMA/tiling/buffer
lifetimes and VF parallelism. Do not copy baseline defects or demand byte-identical
machine code for legitimate alternative implementations.

Change one isolatable factor each iteration, preserving baseline and last passing
version. Explain inseparable groups. Compile off-device, inspect generated code
and machine code, validate identical inputs and boundaries, then resimulate and
compare per-task/pipe/sync evidence against both original and previous versions.
Record cumulative and incremental effects on total completion, not just fewer VF
functions or lower busy time. Keep supported improvements; investigate regressions,
hidden gains or shifted bottlenecks, and revert only task-owned iteration changes.

After combining retained changes, revalidate all authorized caller shapes/tails/
branches and check interactions. Use pto-performance-report for authorized final
hardware coverage. Record dependencies, unresolved intervals and untested scope;
never claim full alignment while blocked. Append/index adaptation.log if present.
For each major difference include paired baseline/current code excerpts, original
paths/lines or PCs, mechanism, measured effect and confidence. Preserve allocation,
reuse loops, set/wait pairs and register restoration context; label pseudocode.
