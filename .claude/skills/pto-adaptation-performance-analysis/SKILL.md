---
name: pto-adaptation-performance-analysis
description: Use when comparing a C++ extern kernel baseline with a PyPTO adaptation to explain generated Tile or RegScope/VF code, A5 simulator pipelines, or synchronization in AIV and MIX kernels.
---

# Analyze operator migration performance

## 0. Compare the Python source and match the workload

Review the operator and its call sites: extern replacement, inputs and outputs,
partitioning, branches, and movement. Use the smallest case with the real
caller's shape and match input data, dtype, layout, control values, and initial
state. Record compiler revisions, options, executed branches, and work counts.

## 1. Trace the actual compiler artifacts

Find the generated outer C++ and the C++ extern or generated VF attached to
it. Verify the compile commands, include/link relationships, symbol map, and
final device image actually used by the simulator. A VF may be inline, in a
header, or in a separate object. Keep original artifacts in place; list paths
and mark any missing evidence.

## 2. Compare the code outside RegScope or VecScope

Compare Tile partitioning, data movement size and count, layout and temporary
buffers, UB addresses and reuse, writeback order, and loop boundaries. Check
whether waits serialize a batch even when the two versions issue the same
number of MTE2 and MTE3 instructions.

## 3. Compare the code inside RegScope or VecScope

1. Map equivalent algorithm stages. Check partitioning, reduction order, dtype,
   conversion and rounding, inactive mask behavior, tails, and intermediate
   values. Flag any unproven equivalence before interpreting speed differences.
2. Follow PyPTO's PTO through PTOAS passes to VPTO micro IR and LLVM IR.
   Compare with the CCE intrinsics in the baseline VecScope and its helpers.
   Align by operation semantics; fusion, unrolling, and address generation may
   differ. For extra memory, select, shuffle, spill, or barrier instructions,
   locate the first stage that introduced them and identify their users.
3. Extract the VFs and their call sites from both actual disassemblies. Distinguish
   static functions, static call sites, and dynamic executions. Verify fusion
   from symbols and calls, and retain offsets, immediates, branches, and
   synchronization in any normalized diff.

## 4. Compare simulator timelines

Use the `pypto-user:incore-profiling` skill. For A5, select `a5` with
`dav_3510` and verify the actual target, toolchain, and loaded image. Check
that both traces executed the intended work. Keep raw trace JSON, the timeline,
the parsing command, and the metrics. Compare MTE2, VF, and MTE3 execution
counts, cumulative duration, active interval union, and span. Attribute gaps
and overlap by batch and report waits separately using the
[analysis reference](references/pipeline-and-sync.md).

## 5. Audit synchronization first

Trace costly waits or membars to instructions, IR, and actual memory
dependencies. Check both whether the C++ baseline has a missing ordering edge
and whether PTOAS inserted a redundant or overly broad one. Use the
[proof criteria](references/pipeline-and-sync.md#synchronization-audit).
Implement production changes at the Python or authorized compiler layer and
regenerate artifacts. If authorized, isolate binary NOP experiments on a
separate copy and report them as diagnostic evidence.

## 6. Check A5 MIX synchronization

For MIX, trace AIC/AIV producer and consumer pairs to actual
`set_intra_block` and `wait_intra_block` operations, or an equivalent
mechanism. Check matching events, directions, work items, and lifetime;
distinguish source, machine-code, and trace evidence. See the
[MIX checklist](references/pipeline-and-sync.md#mix-checklist).

## Deliver

Report baseline and candidate artifact paths, Tile and VF differences,
timeline metrics, and the evidence behind each wait attribution. Label
uncertainty and rank next actions. Keep simulator diagnostics distinct from
device acceptance and index the report from an existing migration log.
