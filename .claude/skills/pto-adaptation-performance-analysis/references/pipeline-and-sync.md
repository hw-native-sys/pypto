# Pipeline and synchronization analysis

## Pipeline metrics

Compare the same core type, time units, and execution window. Count events
dynamically. Report summed duration, union of active intervals, and the span
from the first start to the last completion separately. Sum can exceed wall
time when events overlap; span can include idle gaps. Report synchronization
waits separately and avoid double counting. If an event class is missing, mark
it unobservable instead of reporting zero.

Align corresponding batches and trace MTE2 input, VF execution, and MTE3
writeback. Compare overlap within and across batches. If MTE2 issues the same
number of transfers at similar per-transfer durations but spans more time,
identify the waits around those gaps before attributing them to DMA.

## Synchronization audit

1. Map the wait window to its producer, consumer, PC, disassembly, and IR.
   Distinguish VF membars such as `VV_ALL`, external pipe events, and cross-core
   events. Verify the wait direction from the actual instructions.
2. List the read/write intervals, widths, access types, and execution order.
   Cover read-after-write, write-after-read, and write-after-write hazards within
   and across batches, including tails, aliasing, UB reuse, and DMA completion.
3. Check the ordering guarantees in both versions. If the C++ baseline has no
   explicit wait, identify what guarantees ordering or report a potential
   omission. A wait is redundant only if the relevant paths have no hazard
   or an existing guarantee orders every hazard. Separately classify waits
   whose scope is too broad but whose ordering is required.
4. Find the first pass that inserts the suspect barrier and reduce the case
   if needed. Use matching workloads to isolate changes such as synchronization,
   DMA batching, and their combination. NOP experiments estimate cost; they
   do not establish correctness or justify production deletion.

## MIX checklist

- Trace AIC/AIV task pairing and identify the write, notify, wait, read, and
  buffer reuse for each work item.
- Check generated code and disassembly for the actual event identifier, core
  roles, direction, and scope. Check event reuse and every reachable branch
  for missing or mismatched signal/wait pairs.
- Correlate both sides in the trace when available. A single-side simulator
  run can support static findings but cannot verify cross-core timing.

## Per-task evidence before instruction tuning

For every dynamic MTE2, VF and MTE3 task, record version, core/block/subblock,
batch, initialization/main/tail role, PC/symbol, issue/start/end, duration, bytes
or work, previous gap and predecessor. Mark unobservable issue/ready timestamps
N/A. Keep original timestamps and align entry-relative windows separately.

Per pipe report dynamic count, summed execution duration, interval union,
first-start-to-last-end span, active fraction and cross-pipe overlap. Do not add
VF call duration to its internal VLD/EXEC/VST durations. Compare first-batch
startup, steady state, batch gaps and drain before investigating individual
instruction latency.

For each suspected stall record producer completion, notification/set, wait
release and consumer issue/start, with flag direction, PC, batch, UB read/write
ranges and reuse lifetime. Label readiness, reuse, overly broad wait, proven
redundancy, resource contention or insufficient evidence. Determine whether the
stall changes kernel completion or overlaps independent work. A long MTE span
with unchanged per-transfer execution can be a scheduling gap rather than slow DMA.

Only after cross-pipe dependencies are explained, compare transfer bytes/strides,
batching/tails and VF VLD/EXEC/VST overlap, gather/reorder, spill and membar.
Single-variable experiments need equivalent workload and preserved necessary
ordering; baseline synchronization omissions must be audited as well.

| Difference ID | Baseline/current code, PC, trace | Target and edit layer | Compile/precision/boundaries | Incremental/cumulative time | Keep/revert and remaining evidence |
| ------------- | -------------------------------- | --------------------- | ---------------------------- | --------------------------- | ---------------------------------- |
| N/A | N/A | N/A | N/A | N/A | N/A |
