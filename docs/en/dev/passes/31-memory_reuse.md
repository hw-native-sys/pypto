# MemoryReuse Pass

Uses dependency analysis to identify memory reuse opportunities and removes redundant alloc operations.

## Overview

This pass analyzes variable lifetimes and dependencies to enable memory sharing. Variables with non-overlapping lifetimes in the same memory space can share MemRef objects, reducing memory footprint.

After applying MemRef sharing, the pass also **removes redundant `tile.alloc` statements** for MemRefs that are no longer referenced by any TileType variable.

**Key insights**:

- Variables that don't overlap in lifetime can reuse memory
- Only variables in the same memory space can share MemRef
- Lifetime is determined by def-use analysis
- After sharing, MemRefs that become unreferenced are cleaned up along with their alloc statements

**When to use**: Run after InitMemRef and before AllocateMemoryAddr. Reduces memory allocation overhead.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::MemoryReuse()` | `passes.memory_reuse()` | Function-level |

**Factory function**:

```cpp
Pass MemoryReuse();
```

**Python usage**:

```python
from pypto.pypto_core import passes

reuse_pass = passes.memory_reuse()
program_optimized = reuse_pass(program)
```

## Algorithm

1. **Lifetime Analysis**: Walk the full IR tree (including nested control flow bodies) to compute variable lifetimes via def-use analysis. Variables defined outside a loop but used inside have their lifetime extended to the end of the loop (loop-aware extension)
2. **Interference Check**: Identify variables with overlapping lifetimes
3. **MemRef Sharing** (global first-fit-decreasing packing, `IdentifyReuseOpportunities`): Within each memory space, intervals are packed **largest-first**; every later interval joins the first existing buffer all of whose members it can share with (non-overlapping lifetime + hazard/no-alias safe — see `can_share`). A buffer's allocated size is fixed by its first (largest) member, so admitting a smaller member afterwards costs nothing — and a *later, larger* interval can now host an *earlier, smaller* one. (The previous definition-order greedy had a one-directional size gate `source.size >= target.size`, so two lifetime-disjoint tiles whose small one was defined first were never coalesced.) The representative each member is rebased onto is the buffer's largest member; its `tile.alloc` dominates the whole function because InitMemRef hoists every alloc to the function-body head, so a representative defined after some of its members is safe. Because the packer no longer runs in program order, every pairwise gate (hazard, no-alias) is checked in both directions.
4. **Loop-carry re-alignment** (`AlignLoopCarriesToInitMutator`): Sharing (step 3) only retypes `AssignStmt`-defined vars (producers/init); loop-carried `iter_arg`/`return_var` nodes are excluded from the lifetime/sharing maps and keep their original MemRef. This step walks `ForStmt`s **top-down** and retypes each loop's `iter_arg`/`return_var` to its (now-reused) `initValue` MemRef, seeding `var_remap_` before recursing so a nested loop observes the corrected outer `iter_arg` as its init. Without it, a reused **nested pipelined `matmul_acc`** accumulator splits across two Acc buffers and step 5 emits invalid `acc→acc tile.move` ops that ptoas rejects on Ascend 910B ([#1352](https://github.com/hw-native-sys/pypto/issues/1352))
5. **Yield fixup**: Fix MemRef mismatches in control flow return variables:
   - **ForStmt**: Ensure all 4 loop-carry variables (initValue, iter_arg, yield value, return_var) share the same MemRef. Inserts `tile.move` before yield if MemRefs differ
   - **IfStmt**: Patch return_vars to match yield value's MemRef
6. **Remove redundant allocs**: Collect all MemRefs still referenced by TileType variables, then remove `tile.alloc` statements whose MemRef is no longer in use

**Reuse conditions**:

- Non-overlapping lifetimes (no interference). Two variables do NOT overlap when `prev.last_use <= curr.def` (i.e., the source's last use can be at the same statement as the target's definition, since inputs are read before outputs are written within a single statement).
- Same memory space
- A buffer is sized to its **largest** member; because packing is largest-first, every member admitted later is no larger than the representative, so no explicit byte-size check is needed (and the reuse direction is no longer constrained to "earlier-and-larger only")
- **No-alias guard** (op-semantic): the op that defines the reusing variable may forbid its output from sharing a buffer with one or more of its input operands, because the hardware reads those inputs *while* writing the output — an in-place write would corrupt the op mid-flight. Three sources feed one per-output forbidden-input set (`ForbidAliasCollector`):
  - `not_inplace_safe()` — the op cannot run with `src == dst`, so its output must not alias **any** input operand.
  - `forbid_output_alias(i)` — the op is in-place-safe w.r.t. its value operands but reads a **specific** operand while writing its output, so the output must not alias that one operand's buffer.
  - **widening `tile.cast`** (handled directly in `ForbidAliasCollector`) — when the output dtype is *wider* than the input, the cast cannot run in place: element `i` is read at `i*in_bytes` but written at `i*out_bytes`, so the write cursor outruns the read cursor and clobbers not-yet-converted input. Narrowing / same-width casts stay in-place-safe (preserving the cross-dtype reuse below).

  MemoryReuse refuses to place the output on a forbidden operand's **physical buffer**, resolving each operand through both reuse-map coalescing *and* VIEW inheritance (a `reshape`/`slice` shares its source's MemRef base) — so a forbidden operand reached indirectly (its owning tile reused onto another buffer, or occupied via a view) is still caught.

  Ops that currently declare a no-alias constraint:

  | Op(s) | Constraint | Why the output cannot alias the input |
  | ----- | ---------- | ------------------------------------- |
  | `tile.recip`, `tile.rsqrt` | `not_inplace_safe` | high-precision path reads the input **and** a tmp scratch while writing the output |
  | `tile.row_sum` / `row_max` / `row_min` | `not_inplace_safe` | `TROW*` reads the full input row + tmp scratch while writing the reduced `[M, 1]` output |
  | `tile.mrgsort_format1` | `not_inplace_safe` | merge-sort intrinsic requires `src != dst` |
  | `tile.fmod`, `tile.fmods` | `not_inplace_safe` | `TFMOD`/`TFMODS` compute `a - trunc(a/b)*b` by overwriting `dst = a/b` first, then re-reading the original `src0` (`a`) for the final subtraction; when `dst == src0` that subtraction sees the already-clobbered quotient and yields `0` for every element |
  | `tile.transpose` | `not_inplace_safe` | `pto.ttrans` is not in-place safe: the a2a3 unaligned scalar path writes `dst` directly from `src` (no tmp staging), so `dst == src` corrupts the data mid-write. The output always gets a fresh buffer (also enforced in InitMemRef, which never inherits the input's buffer for it). |
  | `tile.sel` | `forbid_output_alias(0)` (mask), `(3)` (tmp) | `TSEL` reads the predicate mask + tmp scratch while writing `dst` |
  | `tile.{row,col}_expand{,_mul,_add,_sub,_div}` | `forbid_output_alias(1)` (broadcast vector) | the row/col vector (arg 1) is re-read for **every** output row/col, so an output aliasing it is overwritten after the first row/col |
  | `tile.cast` (widening only) | output ≠ input buffer (conditional, in `ForbidAliasCollector`) | wider output's write cursor outruns the read cursor (see above) |

- **Pipeline-stage guard** (role-aware): `pl.pipeline(stage=F)` replicates a loop body `F` times for ping-pong, and `LowerPipelineLoops` tags each clone's tile-producing `Call` with a `pipeline_membership` `(group, stage)` (see [27-lower_pipeline_loops.md](27-lower_pipeline_loops.md)). The `F` clones run concurrently under the scheduler, so their program-order-disjoint lifetimes are *not* a safe reuse signal. MemoryReuse therefore blocks a cross-stage buffer share (same group, different stage) **when at least one tile is a load** (`tile.load` / `tile.read`) — only load buffers need per-stage privacy. Compute intermediates of different stages may still coalesce, because forbidding *all* cross-stage reuse (depth = `F`) overflows the on-chip budget (e.g. `stage=4` RMSNorm: `4 × 67 KB > 188 KB` UB). The L0 matmul spaces (Left/Right/Acc/Bias) are exempt — their buffering is owned by `AutoTileMatmulL0` / `CanonicalizeIOOrder` extract clustering and is capacity-bound. Separation is applied unconditionally and is **never silently relaxed to fit**: if the separated buffers exceed a space's on-chip budget the overflow surfaces as a hard `AllocateMemoryAddr` error (the kernel must reduce its `stage=` count or tile size), not a quiet fall-back to coalescing. Once the reuse decision is made, MemoryReuse strips the now-consumed `pipeline_membership` attr so it does not ride downstream into later passes or codegen.

**No shape / dtype / TileView compatibility gate**: tiles that share a physical MemRef may carry **different** shapes, dtypes, or `TileView` attributes. PTO codegen binds a per-variable `alloc_tile` to each tile, so each alias declares the shared base with its own static shape / dtype / layout / `valid_shape`. This permits, for example:

- cross-dtype reuse — a BF16 tile reusing a dead FP32 tile's buffer (e.g. across `tile.cast`);
- `tile.fillpad` output reusing its input, and two fillpad outputs with different `pad` sharing one buffer;
- N-D tiles with divergent `valid_shape` sharing a buffer (each keeps its own `valid_shape` on its own `alloc_tile`);
- L0 cube-input `Left` / `Right` sub-tiles of differing shape sharing one slot (e.g. fused-attention QK `Right` `[k, SEQ]` reused by PV `Right` `[k', HEAD]`, halving peak L0B — issue #1595).

  Earlier revisions gated reuse on an `AreTileTypesCompatible` shape / dtype / view match (with a narrow L0 byte-reuse exception); that gate has been removed. Correctness for read-while-write ops is now handled precisely by the no-alias guard above rather than by a coarse whole-tile match.

**Alloc cleanup**:

After MemRef sharing, some MemRef objects become unreferenced (their variables now point to a different shared MemRef). The pass traverses the surrounding `SeqStmts` and removes any `tile.alloc` `AssignStmt` whose LHS MemRef pointer is not in the set of still-used MemRefs.

## Ascend910B load + tpop_from_aic hazard

On Ascend910B AIV functions with a non-`None` `SplitMode`, a writer that consumes **both** a `tile.load` result (or a legal-view descendant of one) **and** a `tile.tpop_from_aic` value must not place its output in the same physical buffer as that load result. Allowing the writer's output to in-place-reuse the load buffer produces silently wrong results on this hardware.

MemoryReuse owns every buffer-coalescing decision, so it prevents the hazardous sharing from ever forming rather than relying on a later split. When the guard is active, the reuse decision is blocked exactly when:

- the writer's defining op consumes a `tile.tpop_from_aic` value, **and**
- the buffer member it would reuse in place (whose last use is the writer's def statement) is load-derived.

The guard is gated by `BackendHandler::RequiresSplitLoadTpopWorkaround()` (true only for Ascend910B) and the function being split-AIV; on every other backend / function kind the inputs are empty and reuse behaviour is unchanged. The writer is still free to reuse any **non**-load buffer — only the load + tpop in-place combination is rejected. (This guard previously lived in a dedicated `LegalizePTOBufferReuse` pass that split the buffer after the fact; it now folds into MemoryReuse.)

## User-managed (pinned) buffers

Automatic reuse is a greedy heuristic and cannot guarantee the optimal layout. For fully-explicit tile-level kernels a user can take over a whole memory space and place buffers by hand. The user-facing API is **`pl.tile.alloc_buffer(tile, addr, size, id=-1)`** — a bare statement right after the tile's defining op that pins that tile's buffer:

```python
q_mat = pl.tile.load(q, [0, 0], [16, 128], target_memory=pl.Mem.Mat)
pl.tile.alloc_buffer(q_mat, addr=0, size=4096)            # pin q_mat's buffer
q_left = pl.tile.move(q_mat, target_memory=pl.Mem.Left)
pl.tile.alloc_buffer(q_left, addr=0, size=4096)
scores = pl.tile.matmul(q_left, k_right)
pl.tile.alloc_buffer(scores, addr=0, size=8192)           # Acc (L0) — pinnable in explicit form
```

Semantics:

- **Buffer identity = `id`.** Tiles sharing a (non-negative) `id` share one physical buffer; `id=-1` (default) gives a unique private buffer. MemoryReuse **never** coalesces a pinned buffer with another — this is exactly how a user prevents the over-reuse that breaks pipelining.
- **Address = user-assigned.** `addr` is honored verbatim by AllocateMemoryAddr (no auto re-layout); `size` is the declared buffer footprint.
- **Whole-space-manual contract.** Once any tile in a pinnable space is pinned, *every* tile in that space must be pinned. A fresh unpinned tile there — e.g. a compiler temporary from composite-op lowering — is a hard error: the feature only supports pure tile-level kernels with no compiler-generated tiles in the managed space.
- **Pinnable spaces.** `Vec`, `Mat`, and the L0 matmul spaces `Left` / `Right` / `Acc` / `Bias`. The L0 spaces are pinnable only when the matmul is written **explicitly** (`load` → `move`-to-`Left`/`Right` → `matmul` → `Acc`): a high-level `pl.matmul` is expanded by `AutoTileMatmulL0` into L0 tiles the user never wrote, which the whole-space-manual contract then rejects. So a matmul/attention kernel can pin every tile only in fully-explicit tile form. `DDR` (tensor params) and the scalar register file are never pinnable.
- **Overlap check.** Two *distinct* pinned buffers whose address ranges **and** lifetimes both overlap is a user error (caught here, where liveness is available). Disjoint-lifetime address overlap is legal manual reuse and allowed.

**Why a marker op, not a type annotation.** The pin must survive the front pipeline to reach InitMemRef. A type-level `pl.Tile[..., pl.MemRef(...)]` annotation is dropped: passes between parse and InitMemRef pervasively rebuild tile ops via op type re-deduction, which produces memref-less types (a pre-InitMemRef memref is, by design, meaningless). `tile.alloc_buffer` instead carries the pin in op **kwargs**, which the rebuilds preserve (`op_registry.Create` is always called with the op's kwargs). So the marker reaches InitMemRef intact. (The `pl.MemRef(...)` annotation form is retained as a low-level mechanism for IR already at tile level — e.g. tests fed straight to the memory passes — but is not the end-to-end user API.)

Implementation: InitMemRef pre-scans for `tile.alloc_buffer` markers, applies each pin to its tile via the shared `HonorUserPin` (interning the base by id, renamed to a `__pinned__<id>` prefix so downstream passes recognise it without an IR-node flag — see [InitMemRef](30-init_memref.md)), and **erases the marker** (it never reaches codegen); MemoryReuse skips pinned bases and runs the overlap check; AllocateMemoryAddr leaves pinned addresses untouched (see [AllocateMemoryAddr](32-allocate_memory_addr.md)).

## Example

### MemRef Sharing with Alloc Cleanup

**Before** (after InitMemRef):

```python
# SeqStmts [
mem_vec_0: MemRefType = tile.alloc(Vec, -1, 16384, 0)
mem_vec_1: MemRefType = tile.alloc(Vec, -1, 16384, 1)
mem_vec_2: MemRefType = tile.alloc(Vec, -1, 16384, 2)
tile_a: Tile[[64, 64], FP32, memref=mem_vec_0] = tile.load(...)
tile_b: Tile[[64, 64], FP32, memref=mem_vec_1] = tile.add(tile_a, ...)
# tile_a last use ↑
tile_c: Tile[[64, 64], FP32, memref=mem_vec_2] = tile.load(...)
# ]
```

**After** (tile_c reuses mem_vec_0 from tile_a, alloc for mem_vec_2 removed):

```python
# SeqStmts [
mem_vec_0: MemRefType = tile.alloc(Vec, -1, 16384, 0)
mem_vec_1: MemRefType = tile.alloc(Vec, -1, 16384, 1)
# mem_vec_2 alloc removed — no longer referenced
tile_a: Tile[[64, 64], FP32, memref=mem_vec_0] = tile.load(...)
tile_b: Tile[[64, 64], FP32, memref=mem_vec_1] = tile.add(tile_a, ...)
tile_c: Tile[[64, 64], FP32, memref=mem_vec_0] = tile.load(...)
# tile_c now shares mem_vec_0 with tile_a
# ]
```

### Producer-Consumer Reuse

When a variable's last use is at the same statement that defines a new variable (producer-consumer relationship), the new variable can reuse the old variable's memory because inputs are read before outputs are written:

```python
# Before:
tile_a: Tile[[64, 64], FP32, memref=mem_vec_0] = tile.create(...)
tile_b: Tile[[64, 64], FP32, memref=mem_vec_1] = tile.muls(tile_a, 0.0)
# tile_a.last_use == tile_b.def → reuse allowed

# After:
tile_a: Tile[[64, 64], FP32, memref=mem_vec_0] = tile.create(...)
tile_b: Tile[[64, 64], FP32, memref=mem_vec_0] = tile.muls(tile_a, 0.0)
# tile_b reuses mem_vec_0
```

### Overlapping Lifetimes (No Reuse)

When a variable is still alive **after** another variable's definition (last_use > def), their lifetimes truly overlap and they cannot share memory:

```python
# SeqStmts [
mem_vec_0: MemRefType = tile.alloc(Vec, -1, 16384, 0)
mem_vec_1: MemRefType = tile.alloc(Vec, -1, 16384, 1)
tile_a: Tile[[64, 64], FP32, memref=mem_vec_0] = tile.load(...)
tile_b: Tile[[64, 64], FP32, memref=mem_vec_1] = tile.load(...)
# tile_a.last_use > tile_b.def → tile_a still live when tile_b is defined
# ]
```

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`

```cpp
Pass MemoryReuse();
```

**Implementation**: `src/ir/transforms/memory_reuse_pass.cpp`

- `LifetimeAnalyzer` walks the full IR tree to compute variable lifetimes (including nested control flow)
- `ComputeLifetimes` builds MemRef sharing groups and lifetime intervals
- `IdentifyReuseOpportunities` finds reuse candidates
- `ApplyMemRefSharing` updates MemRef pointers via `MemRefSharingMutator`
- `YieldFixupMutator` fixes ForStmt/IfStmt yield/return_var MemRef mismatches after reuse (inserts `tile.move` when needed)
- `UsedMemRefCollector` gathers still-referenced MemRef pointers after sharing
- `RemoveUnusedAllocStatements` filters out redundant `tile.alloc` statements from `SeqStmts`

**Python binding**: `python/bindings/modules/passes.cpp`

```cpp
passes.def("memory_reuse", &pass::MemoryReuse, "Memory reuse optimization");
```

**Tests**: `tests/ut/ir/transforms/test_memory_reuse.py`

- Tests non-overlapping lifetime reuse with MemRef sharing
- Tests producer-consumer reuse (last_use == def at same statement)
- Tests overlapping lifetime no-reuse
- Tests memory space separation
- Tests byte-size compatibility
- Tests cross-dtype / cross-`TileView` reuse (now permitted: BF16↔FP32, fillpad output↔input, divergent `valid_shape`)
- Tests the no-alias guard (`TestForbidOutputAlias` + `TestInplaceOps`), one case per constraint above:
  - `tile.recip` / `tile.rsqrt` / `tile.row_sum` — output must not alias input (`not_inplace_safe`)
  - `tile.sel` — output must not alias the mask / tmp (`forbid_output_alias`)
  - `tile.col_expand_mul` — output must not alias the broadcast vector
  - widening `tile.cast` — output must not alias the (narrower) input
  - a forbidden operand reached through a VIEW is still honored (physical-buffer resolution)
- Tests view operation MemRef sharing preservation
- Tests redundant alloc statement removal
- Tests control flow lifetime analysis (nested IfStmt in ForStmt, branch variable sharing)
