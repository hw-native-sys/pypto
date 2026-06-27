# SkewCrossCorePipeline Pass

Software-pipelines mixed cube/vector (cross-core) `pl.pipeline` loops so the two cores overlap, replacing the legacy unroll+IO-cluster handling of cross-core loops. Runs immediately before [`LowerPipelineLoops`](28-lower_pipeline_loops.md).

## Overview

On A2/A3 a fused cube+vector kernel (e.g. flash-decode `qk_pv`) round-trips through GM: the cube (AIC) sends scores to the vector (AIV) via `tile.tpush_to_aiv` and gets the softmax result back via `tile.tpop_from_aiv`; the vector mirrors this with `tile.tpop_from_aic` / `tile.tpush_to_aic`. Run naively, each core stalls waiting for the other.

The old approach unrolled these loops (`pl.pipeline(stage=F)`) and let `CanonicalizeIOOrder` cluster the cross-core ops — which produced *back-to-back* `tpop`s that serialised the consumer. `SkewCrossCorePipeline` instead software-pipelines the loop:

- **Single round-trip, producer role** — exactly one `tpush` and one `tpop`, and the `tpush`'s backward slice does not feed the body via an SSA edge (the cube: `QK → tpush`, `tpop → SV`). The two halves are linked only by the in-order cross-core FIFO, so the producer runs **one iteration ahead**: a `produce(start)` prologue, a `ForKind::Sequential` steady loop whose loop var `k` indexes the produce and pairs `produce(k)` with the trailing `consume(k-step)` over `k` in `[start+step, start+trip*step)`, and a `consume(last)` epilogue. The cube issues iteration k's `QK` while the vector runs iteration k-step's softmax.
- **Consumer role, or multi-round-trip** — the lead op feeds the body via SSA (the vector: the popped scores feed softmax), or there is more than one message per FIFO direction. The loop is **demoted to a plain `ForKind::Sequential` loop** (body unchanged). This drops the unroll's back-to-back `tpop` while preserving the in-order FIFO; cross-core overlap then comes from the *peer* core's producer skew putting each tile in the FIFO a step early, so the in-order `tpop` never blocks.

Every **non-cross-core** pipeline loop (same-core GM→L1, L1→L0, nested matmul stage loops — no `tpush`/`tpop`) is left untouched for `LowerPipelineLoops` to replicate.

The output is `ForKind::Sequential` with no `pipeline_stages` marker, so `LowerPipelineLoops` (triggers on `kind == Pipeline`) skips it and `CanonicalizeIOOrder` (scoped to pipeline bodies) does not re-sort the hand-ordered skew.

**Requires**: SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, TileMemoryInferred, NormalizedStmtStructure.

**Pipeline position**: After [`NormalizeReturnOrder`](26-normalize_return_order.md), immediately before [`LowerPipelineLoops`](28-lower_pipeline_loops.md). Nothing runs between them, so a cross-core loop is skewed (→ Sequential) before the unroll pass can replicate it.

## API

```python
from pypto import passes
p = passes.skew_cross_core_pipeline()
```

## Behavior

For a `ForStmt(kind=Pipeline, attrs={"pipeline_stages": F})` with `F > 1`:

1. **Non-cross-core** — body lacks a `tpush`/`tpop` pair → left intact as `ForKind::Pipeline` for `LowerPipelineLoops` to replicate.
2. **Cross-core** — body has both a `tpush` and a `tpop`. Find the lead (first cross-core op in program order), backward-slice it into the producer half, and classify:
   - one `tpush` + one `tpop`, `carried` (lead-defined vars used by the body) empty, statically skewable → **producer skew** (prologue / Sequential steady loop / epilogue).
   - otherwise — `carried` non-empty (consumer role), more than one `tpush`/`tpop` (multi-round-trip; skewing one message would reorder the in-order FIFO — a silent wrong-data bug the verifiers don't catch), dynamic bounds, or trip < 2 → **`DemoteToSequential`**.

Either way a cross-core loop **always** leaves this pass as `ForKind::Sequential` with no `pipeline_stages` marker, so it never reaches `LowerPipelineLoops` or `CanonicalizeIOOrder` as a Pipeline body.

`iter_args` (e.g. flash-attention `mi/li/oi` accumulators) thread through the producer-skew prologue → steady → epilogue; the producer half is iter-arg-transparent.

## Constraints

- Static bounds only for the skew (`start`, `stop`, `step` compile-time). Dynamic-bound cross-core loops bail to the unroll path.
- The producer-skew steady region is **kept as a loop** (not fully unrolled) so the matmul `Acc` double-buffering assigned by `AllocateMemoryAddr` still has a loop to alternate over.
- A true consumer-side prefetch is intentionally **not** done: it would break codegen's `tpop → tfree` FIFO-slot tracking (keyed on SSA var identity, cannot cross an iter_arg) and a blocking `tpop` issued a full iteration early would simply stall.

## Limitations

- **Multi-round-trip loops are not skewed yet (TODO).** A loop with more than one
  message per FIFO direction (e.g. `C→V→C→V` = `tpush, tpop, tpush, tpop`) is
  currently **demoted to `ForKind::Sequential`**, not software-pipelined. Skewing
  only the lead message would reorder the in-order cross-core FIFO and silently
  feed the peer the wrong tile (the property verifiers do not model FIFO order),
  so the conservative demote is correct but leaves overlap on the table. A
  future revision should skew the whole FIFO group together (advance every
  message by one round-trip) so multi-round-trip producers overlap too.

## Examples

```python
# Producer role (single round-trip) — SKEWED
# Before: for i in pl.pipeline(0, 4, 1, stage=2):
#             qk = ...; tpush_to_aiv(qk); p = tpop_from_aiv(); sv = ...(p); store(sv)
# After:  produce(0)                              # prologue
#         for i in pl.range(1, 4, 1):             # steady (Sequential)
#             produce(i); consume(i-1)            # cube QK[i] overlaps vec softmax[i-1]
#         consume(3)                              # epilogue
```

```python
# Consumer role / multi-round-trip — DEMOTED to Sequential
# Before: for i in pl.pipeline(0, 4, 1, stage=2):  (tpop; softmax; tpush; store)
# After:  for i in pl.range(0, 4, 1):              # body unchanged, FIFO order preserved
```

## Related

- [`LowerPipelineLoops`](28-lower_pipeline_loops.md) — replicates the remaining (same-core) pipeline loops for ping-pong.
- [`CanonicalizeIOOrder`](29-canonicalize_io_order.md) — clusters same-core IO within pipeline bodies (cross-core loops no longer reach it; they are Sequential by here).
- [`SplitVectorKernel`](24-split_vector_kernel.md) — `UP_DOWN` vector split, orthogonal to the skew and composable with it.
