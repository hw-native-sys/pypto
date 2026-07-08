# ptoas Multi-Buffer (`use_ptoas_multi_buffer`)

Updated: 2026-07-08

An opt-in switch that lowers a same-core `pl.pipeline(stage=N)` loop's rotating
load to a **ptoas multi-buffer region** instead of pypto's own body-replication
ping-pong. ptoas then delivers the cross-iteration double-buffer overlap, so the
kernel keeps a **single loop body** and a **single N-slot region** (smaller code,
tighter memory) while matching the native pipeline's overlap.

Gated by `PassContext.use_ptoas_multi_buffer` — a no-op by default; the default
pipeline is byte-identical when off.

## How to enable

```python
from pypto.pypto_core import passes

# Explicit PassContext
with passes.PassContext([], use_ptoas_multi_buffer=True):
    ...

# RunConfig / @pl.jit
cfg = RunConfig(platform="a2a3", use_ptoas_multi_buffer=True)

# Env fallback
#   PYPTO_PTOAS_MULTI_BUFFER=1
```

Requires ptoas ≥ 0.48 (`pto.alloc_multi_tile` support): `PTOAS_ROOT=/usr/local/ptoas/0.48`.

## Key design decision: same-slot, not explicit prefetch

ptoas delivers the overlap itself. Given a tile loaded and consumed on the **same
slot `mb[i%N]`, same iteration**, ptoas PlanMemory assigns the N slots concrete
disjoint addresses and its sync pass overlaps **iteration `i`'s load (slot `i%N`)
with iteration `i-1`'s consume (slot `(i-1)%N ≠ i%N`)** via dyn-event
(`set_flag_dyn` / `wait_flag_dyn` + `arith.select`) WAR synchronization.

A manual prologue + prefetch-next / consume-cur split (writing slot `(i+1)%N`
while reading slot `i%N`) instead **fights** that analysis: ptoas pairs static
events in program order, the consume waits on the current iteration's prefetch of
a *different* slot (a false dependency), and the loop serializes. So the pass
emits the simple same-slot form and lets ptoas pipeline it.

## Key constraint: requires ptoas memory planning (level2)

The overlap only materializes at **`--pto-level=level2`**, where ptoas PlanMemory
owns memory and assigns each slot a concrete disjoint address (its `MemAlias`
proves the slots disjoint → dyn-event sync). At `--pto-level=level3` a single
baked base + a *dynamic* slot `i%N` defeats `MemAlias` (it conservatively treats
a dynamic slot as aliasing all slots) → serial.

Therefore `use_ptoas_multi_buffer=True` **auto-forces `memory_planner=PtoAS`**
(in the `PassContext` constructor; `compile.py` reads the effective planner back
so codegen's level matches). If the caller passed a different planner, a warning
notes the override. At level3 the pass still emits valid (but non-overlapping) IR
as a fallback.

## The pass

`ConvertToPtoasMultiBuffer` (`src/ir/transforms/convert_to_ptoas_multi_buffer_pass.cpp`)
runs in `LowerPipelineLoops`' slot. When the switch is on the pass manager also
**drops `LowerPipelineLoops` and `CanonicalizeIOOrder`** (they would replicate /
reorder), so this pass **owns pipeline lowering** and must leave zero
`ForKind::Pipeline` loops behind. For each same-core pipeline loop it either:

- **rewrites** it (exactly one i-dependent Vec/Mat load): hoist
  `region = tile.multi_buffer_alloc(shape; count=N)` before the loop, and replace
  `t = tile.load(args)` **in place** with
  `t = tile.multi_buffer_load_slot(region, i%N, args)` (same tile var → consumers
  need no rewrite), then demote the loop to `Sequential`; or
- **demotes** it to a plain `Sequential` loop (correct, no double-buffer) when it
  is not an eligible shape.

A load is eligible when it is a single-def Vec/Mat `tile.load` whose offset args
reference the loop variable (an i-*independent* load is a loop invariant —
multi-buffering it is pointless).

## New IR ops (pass-synthesized, not DSL-exposed)

Registered under the `tile.` namespace (`src/ir/op/tile_ops/memory.cpp`) so they
reuse the existing printer/parser round-trip; **not** `internal_only` (which would
break reparse):

| Op | Result | Codegen |
| -- | ------ | ------- |
| `tile.multi_buffer_alloc(shape; count=N, dtype, target_memory)` | region (per-slot `TileType`) | `pto.alloc_multi_tile` (addr only at level3) |
| `tile.multi_buffer_load_slot(region, k, tensor, offsets, shapes, valid_shapes)` | filled slot view | `pto.multi_tile_get %mb[k]` + `pto.tload` |
| `tile.multi_buffer_get_slot(region, k)` | consume view | `pto.multi_tile_get %mb[k]` |

`multi_buffer_alloc` reuses `DeduceTileCreateTileType` (`count` is an extra int
kwarg). The dynamic slot `k = i%N` is a normal index SSA operand resolved only at
codegen (`%mb[k]`); it never enters the MemRef — pypto MemRef offsets are static,
which is exactly why the rotation must stay a runtime index and be planned by
ptoas. `multi_buffer_get_slot` is registered but currently unused by the pass
(kept for a future multi-consumer view).

## Memory layer

Because the switch runs under ptoas memory planning (level2), `MemoryReuse` and
`AllocateMemoryAddr` are skipped — **ptoas PlanMemory owns the N-slot region**,
sized from the `multi_tile_buf<..., count=N>` type. There is no pypto-side
reservation; codegen emits an addr-less `pto.alloc_multi_tile`.

- **Region** (`multi_buffer_alloc`) gets a MemRef from `InitMemRef` but no baked
  address.
- **Slot views** (`get_slot` / `load_slot`) are **buffer-less** (`InitMemRef`
  `ProducesBufferLessTile`) so each use gets its own SSA name (no pure-alias
  collapse onto the region); codegen emits their `pto.multi_tile_get` directly.

## Verification

- **Codegen**: the emitted `.pto` shows one hoisted `pto.alloc_multi_tile`
  (addr-less at level2, `count=N`) and one `pto.multi_tile_get %mb[i%N]` per
  iteration (single loop body, no `scf.if` prefetch guard).
- **Overlap (ptoas 0.48, level2)**: the final `.cpp` primes 2 per-slot events and
  uses a **variable event id** (`wait_flag(..., v25)` / `set_flag(..., v26)` — the
  lowered `wait_flag_dyn` / `set_flag_dyn`), so `load[i]` overlaps `consume[i-1]`.
- **On-device numeric parity** (`tests/st/runtime/test_ptoas_multi_buffer_device.py`):
  switch-on == switch-off == torch reference on a2a3.
- **Codegen / round-trip UT**: `tests/st/codegen/dsl/test_ptoas_multi_buffer_codegen.py`.

## Limitations (M1 scope)

- One i-dependent Vec/Mat load per pipeline loop; deeper multi-load bodies and
  `N > 2` beyond the same-slot generalization are future work.
- Dropping `LowerPipelineLoops` is global: under the switch, **all** non-eligible
  pipeline loops (including matmul L0 stage loops) demote to serial (lose
  ping-pong). Acceptable because the switch is opt-in / default-off; a future
  increment can replicate non-eligible loops in-pass.
- Level2-only: the overlap needs ptoas PlanMemory, so the switch forces PtoAS;
  codegen asserts a region never reaches level3.

## File map

| Concern | Path |
| ------- | ---- |
| Pass | `src/ir/transforms/convert_to_ptoas_multi_buffer_pass.cpp` |
| Ops | `src/ir/op/tile_ops/memory.cpp` (registration), `python/pypto/ir/op/tile_ops.py` (builders) |
| Auto-force planner | `src/ir/transforms/pass_context.cpp` (ctor), `python/pypto/ir/compile.py` |
| Pass-manager skip | `python/pypto/ir/pass_manager.py` |
| Codegen | `src/codegen/pto/pto_codegen.cpp` (region alloc + `EmitMultiTileGet`), `src/backend/common/pto_ops_common.cpp` (op emitters) |
| Memory layer | `src/ir/transforms/init_memref.cpp` (buffer-less slot views) |
| ptoas design | `~/PTOAS/docs/designs/ptoas-multi-buffer-explicit-design.md` |
