# LowerTileToBuffer Pass

Converts planned device Tile SSA into explicit Buffer operations at the final
device representation boundary. PTO codegen receives allocation handles and
destination operands directly.

## Placement and API

During migration, construct and run the default pipeline in the same context:

```python
from pypto import passes
from pypto.ir.pass_manager import OptimizationStrategy, PassManager

with passes.PassContext([], enable_buffer_ir=True):
    manager = PassManager(OptimizationStrategy.Default)
    lowered = manager.run_passes(program)
```

`LowerTileToBuffer` runs after `MaterializeValidShapeSymbols`. Storage repair
and `VerifyTileStorage` precede address placement. PYPTO and DSA_RP provide final
effective byte addresses; PTOAS provides allocation identities without addresses.
The pass rechecks storage closure even when automatic verification is disabled.
Addressed planners additionally check effective-range overlap.

Custom pipelines can call `passes.lower_tile_to_buffer()` once the same storage,
SSA, return normalization, and device/orchestration separation invariants hold.
`NoNestedCalls` is also required: run `FlattenCallExpr` before this boundary.
The pass produces `BufferIR` and invalidates properties describing Tile storage.

## Representation example

The following notation abbreviates descriptors and scalar tuples:

```text
# Planned Tile input: lhs, rhs and total carry MemRef storage windows.
lhs = tile.load(A, (0, 0), (16, 32))
rhs = tile.load(B, (0, 0), (16, 32))
total = tile.add(lhs, rhs)
result = tile.store(total, (0, 0), Out)
return result

# Buffer output: allocs occur at the original allocation definitions.
# Each descriptor is Buffer[[16, 32], FP32, Vec].
a_buf = buffer.alloc((), address_a)
b_buf = buffer.alloc((), address_b)
r_buf = buffer.alloc((), address_r)
buffer.load(A, (0, 0), (16, 32), a_buf)
buffer.load(B, (0, 0), (16, 32), b_buf)
buffer.add(a_buf, b_buf, r_buf)
buffer.store(r_buf, (0, 0), (16, 32), Out)
return Out
```

The planner may already reuse an allocation; the example shows three distinct
windows for clarity. PTOAS omits the second `buffer.alloc` operand. Addressed
planners pass the final effective address exactly once. Buffer operations that
write a destination return `VoidType`; allocation returns the handle.

## Conversion contract

An indexed traversal collects storage from Tile variables, whose MemRefs have
been planned. Producer calls retain logical deduced types and do not establish
additional allocation identities. A second traversal replaces each Tile use
with its Buffer handle and rewrites supported calls. The maps remain private to
the pass; the output IR contains no conversion side table.

Existing storage definitions determine allocation placement. Conversion adds
no scratch allocation, address assignment, or implicit transfer. An exact
self-copy is removed. Tensor return aliases are normalized to existing GM
parameters while parameter directions and the orchestration ABI are preserved.

Converted `InCore`, `AIC`, and `AIV` functions receive
`FunctionIRStage.Buffer`; orchestration functions remain unchanged. The pass
verifies its output and is idempotent. Failed conversion leaves the input
program unchanged. No functional Tile pass should run after this boundary.

Synthetic allocations inherit the source location of their indexed Tile handle when the original allocation has no location. This keeps native allocation diagnostics tied to the user source.

## Planned storage views

The `tile.alloc` byte capacity remains authoritative. When a root serves
multiple static descriptors or smaller windows, lowering declares one full-valid
`UINT8[capacity / 32, 32]` Buffer and creates explicit `buffer.subview` and
`buffer.reshape` aliases. Equal descriptor/window pairs share one SSA handle,
including separate Tile variables that name the same view. A single descriptor
covering the full allocation keeps the direct typed-allocation form.

```text
# Planned allocation: capacity 4096, effective base address 8192.
# A dense FP32[16,32] member starts at effective address 8256.
root = buffer.alloc((), 8192) : Buffer[[128,32], UINT8, Vec]
window = buffer.subview(root, (2,0)) : Buffer[[64,32], UINT8, Vec]
value = buffer.reshape(window) : Buffer[[16,32], FP32, Vec]
```

Each alias definition occurs beside the original allocation. It reads descriptor
metadata without reading or copying data. The emitter serializes those exact
native forms and does not recover allocation sizes, choose view instructions,
or add storage. Logical `tile.reshape` disappears after its type and preserved
storage window have been checked; the indexed typed alias supplies its result.

For addressed planners, a member whose MemRef size equals the allocation capacity
must establish the final base address. Interior members alone are rejected:
their minimum address cannot prove where the allocation begins. A later
allocation-fact representation can remove this restriction. PTOAS uses symbolic
origin zero. Every view must fit within the original capacity, with static
32-byte-aligned byte offsets, byte counts and physical rows. Boxed views and
mutable view metadata remain unsupported at the allocation; strided windows
are defined where they are used (below).

## Windows: slice and assemble

`tile.slice` is a window of its source at possibly runtime offsets, keeping the
source's row pitch. Its offsets exist only where the slice is written, so the
window is defined there, not at the allocation, and a relabel of a window is a
`buffer.reshape` of it at the same place. `tile.assemble` writes in place: the
source is copied into a window of the result's storage, after copying the
target into the result when the planner gave them different storage.

```text
# window = tile.slice(whole, [16, 32], [row, 32]); placed = tile.assemble(canvas, doubled, [16, 0])
window = buffer.subview(whole_buffer, (row, 32), (16, 32)) : Buffer[[16,32], FP32, Vec]
buffer.add(window, window, doubled_buffer)
placed_window = buffer.subview(canvas_buffer, (16, 0), (16, 32)) : Buffer[[16,32], FP32, Vec]
buffer.copy(doubled_buffer, placed_window)
```

Every lowered window carries its valid tuple, so PTOAS types each native
dimension from it. PTOAS cannot legalize a subview of a byte-root relabel, so
in addressed storage a tile that windows are cut from gets its own allocation
at its planned address. A source that already is the assembled window was
written in place and needs no copy. Assembles across memory spaces (the
fix-pipe Acc to Mat forms) and rank-reducing slices are rejected until their
recipes exist.

`BufferIR` proves each window disjoint from the destinations written while it
is read. A runtime-offset window can only be placed within its whole source
window. When an addressed planner overlaps part of that range with such a
destination, lowering fails closed instead of guessing that the runtime offset
avoids it.

## Runtime valid extents

A valid extent that is not a constant (or is the lane-1 `[0, 0]` sentinel)
makes both descriptor dimensions dynamic, because the native metadata update
writes both. The allocation starts with its physical extents, and every
operation that defines such a tile first states the extents it produces with
`buffer.set_validshape`; `tile.set_validshape` updates the shared handle
directly. Tiles that name the same storage window and differ only in valid
extents share one dynamic handle. Windows instead keep per-dimension static or
dynamic valid extents in their subview, because a view never takes a metadata
update. A dynamic tile must own its allocation; it cannot share a byte-view
root.

```text
# loaded = tile.load(x, [0, 0], [16, 64], [16, cols]); doubled = tile.add(loaded, loaded)
buffer.set_validshape(loaded_buffer, (16, cols))
buffer.load(x, (0, 0), (16, cols), loaded_buffer)
buffer.set_validshape(doubled_buffer, (16, cols))
buffer.add(loaded_buffer, loaded_buffer, doubled_buffer)
```

## Matrix storage and cube recipes

Mat, Left, Right and Acc tiles keep their resolved fractal layout in the
`BufferType` (`blayout`, `slayout`, `fractal`, `compact`). Their physical
extents are already whole fractal boxes, and a fractal window has no row-major
byte view, so these spaces never use the `UINT8[N,32]` root. Addressed planners
have already fixed every window, so each distinct window becomes its own
`buffer.alloc` at its final effective address; two descriptors at one address
alias exactly as the planner placed them. PTOAS allocations are addressless and
therefore carry one window each; a same-size relabel of that window becomes a
`buffer.reshape` alias.

```text
# x = a @ b; y = x + a @ b; store y  (FP16 operands, FP32 accumulator)
a_mat = buffer.alloc((), 0)      : Buffer[[64,64], FP16, Mat, NZ]
b_mat = buffer.alloc((), 8192)   : Buffer[[64,64], FP16, Mat, NZ]
a_l0  = buffer.alloc((), 0)      : Buffer[[64,64], FP16, Left]
b_l0  = buffer.alloc((), 0)      : Buffer[[64,64], FP16, Right, ZN]
acc   = buffer.alloc((), 0)      : Buffer[[64,64], FP32, Acc, fractal=1024]
buffer.load(A, (0,0), (64,64), a_mat)
buffer.load(B, (0,0), (64,64), b_mat)
buffer.copy(a_mat, a_l0)
buffer.copy(b_mat, b_l0)
buffer.matmul(a_l0, b_l0, acc)
buffer.matmul_acc(a_l0, b_l0, acc)
buffer.store(acc, (0,0), (64,64), Out)
```

`tile.move` from Mat becomes `buffer.copy`, `tile.extract` becomes
`buffer.extract`, and `tile.matmul` becomes `buffer.matmul`. `tile.matmul_acc`
must already accumulate in place (its accumulator and result share storage) and
becomes `buffer.matmul_acc`. Its optional `init_cond` selects the write form: a
literal chooses one call, and a runtime predicate becomes an explicit `if`
whose arms are `buffer.matmul` and `buffer.matmul_acc`. An accumulator whose
valid rectangle is wider than the product is written through a product-shaped
view of the same storage (another addressed allocation, or a PTOAS
`buffer.reshape`), because PTOAS requires a native matmul destination to match
the product. This is the same native write as the legacy emitter; whole
fractal boxes outside the product keep their data, the rest of a partially
covered box does not. `tile.transpose_view`
needs no call: storage indexing has already declared the relabelled window.
Acc stores drain through the fix-pipe with its unscaled conversions only;
`pre_quant`/`pre_relu`, atomic and phased stores, and cache-policy loads still
fail explicitly until their transfer recipes exist.

Pipeline-stage membership attributes only constrain storage planning, which the
MemRefs already record. The pass drops that attribute (address planning strips
it earlier; PTOAS keeps it until here). Any other call attribute still needs an
explicit conversion contract.

## Branches

Storage legalization has already selected one destination window for each Tile
branch result and placed all required transfers in the arms. Final conversion
removes those Tile results and yield operands. It preserves scalar results in
their original relative order, so native `scf.if` carries only real scalar SSA.

```text
# Input: (chosen_tile, selected_offset) = if flag:
#          then yield (product, 16); else yield (input_tile, 0)
# Storage legalization gives chosen_tile a canonical destination.
selected_offset = if flag:
    buffer.mul(a_buf, b_buf, destination)
    yield 16
else:
    buffer.copy(a_buf, destination)
    yield 0
buffer.store(destination, (selected_offset, 0), (16, 32), Out)
```

A branch result that aliases GM is removed when both arms resolve to the same
existing parameter. The result's later uses then name that parameter directly.
Different GM aliases require a separate dynamic-GM recipe and are diagnosed.
Distributed-tensor branch results are explicitly rejected at this boundary, even
when both arms alias the same parameter; they need a separate conversion recipe.
Nested branches use scoped yield contexts; conversion adds no allocation or
copy to repair a region. Branch and yield source comments are preserved.

## Loops

For and While conversion removes Tile initializers, iter_args, results and
backedge yields after verifying that they name the same legalized storage.
Entry copies already run before the loop, so zero iterations preserve the
initial value. Swap and fanout snapshots are ordinary Buffer writes in the
body; final conversion creates no scratch or copy.

Only scalar iter_args remain in native control flow, in their original relative
order. While conditions use the rewritten scalar bindings. GM carries disappear
when their initial value and backedge resolve to the same parameter; a changing
GM selection requires a separate recipe and is diagnosed. Nested loops and
branches use distinct yield contexts, and each initializer is traversed only
at its binding to avoid repeated walks through enclosing carry chains.
Distributed Tensor loop carries are rejected explicitly at this boundary, as
with branch results, until their region-result and device ABI recipe is supported.
Binary round trips restore While carry definitions before decoding their
condition, preserving shared references from both the condition and body.

```text
# Tile carries (left, row, right, column) become two scalar carries.
(row_result, column_result) = for i in range(count), (row=0, column=0):
    buffer.copy(right_buf, scratch_right)
    buffer.copy(left_buf, scratch_left)
    buffer.copy(scratch_right, left_buf)
    buffer.copy(scratch_left, right_buf)
    yield (row + 1, column + 2)
buffer.store(left_buf, (row_result, column_result), (16, 32), Out)
```

The scalar SPMD queries `tile.get_block_idx`, `tile.get_block_num`, and
`tile.get_subblock_idx` become corresponding internal `buffer.*` queries with
an INDEX value result and no memory effects. Native emission reads the existing
runtime-supplied kernel ABI parameters. These queries remain direct SSA
assignments; `FlattenCallExpr` handles nested source expressions beforehand.

## Initial supported recipes

The current recipes support straight-line kernels, branches and loops with
rank-2 dense Vec FP16/BF16/FP32/INT32 tiles with explicit storage views, use-site windows, static or runtime valid extents, ordinary packed ND GM tensors,
and default load/store policies, plus the cube path above: Mat loads, Mat to
Left/Right copies and extracts, matmul/matmul_acc and Acc stores. It converts allocation, create, load, store,
move, already legalized aliases, and the [typed elementwise recipes](../ir/05-operators.md#typed-buffer-elementwise-recipes).
GM load/store preserve matching element types without casts. `add`/`mul` support
FP16/FP32/INT32; BF16 transfer support does not imply arithmetic support.
Scalar recipe inputs are converted explicitly to the destination dtype before
Buffer calls are constructed; fill shape/dtype select the destination descriptor.

Helper calls, alternate layouts, slots, and
other operation recipes are added in subsequent migration slices. Unsupported
forms fail explicitly. The migration option defaults to false until the
complete recipe and runtime acceptance matrix is ready.

Binary serialization preserves the explicit representation and function stage.
The current Python diagnostic printer is not a Buffer DSL parser round trip.

## Tests

`tests/ut/ir/transforms/test_lower_tile_to_buffer.py` exercises the public
frontend through the full pipeline for all three planners, checks explicit
allocations and destination writes, verifies immutable/idempotent conversion
and binary persistence, and compiles the resulting PTO with native PTOAS.

`tests/ut/ir/transforms/test_lower_buffer_views.py` checks authoritative capacity,
nonzero window offsets, repeated view identity, fail-closed placement diagnostics,
binary persistence and native compilation on both targets with all three planners.
It also lowers static and runtime slices, an assemble, and runtime valid extents
from `tile.load` and `tile.set_validshape`, including the fail-closed partial
overlap of a runtime window under the addressed planners.

`tests/ut/ir/transforms/test_lower_buffer_matrix.py` lowers explicit and
auto-tiled cube kernels for FP16, BF16, FP32 and INT8 operands, the three
`init_cond` forms and a transposed operand for all planners, then compiles the
PTO natively on both targets.

For numerical system tests, declare `st.case(..., enable_buffer_ir=True,
memory_planner=...)` on the public `@pl.jit` entry. The harness applies the option
inside both inline and precompile-worker compilations; an outer test-thread
`PassContext` alone does not configure worker threads. Enabled cases have distinct
cache keys. The harness checks the actual final device function stages and saves
the transformed program as `buffer_ir.msgpack` beside the native artifacts.

`tests/st/runtime/ops/test_buffer_ir.py` provides load/add/mul/store numerical
cases with orchestration for all three planners, and exact cube cases:
matmul followed by matmul_acc for FP16/BF16/INT8, a K loop with a runtime
`init_cond`, an auto-tiled BF16 matmul and a `b_trans` matmul. Window and
valid-extent cases cover a column slice assembled into a new tile, a runtime row
slice, and runtime valid columns from `tile.load` and `tile.set_validshape`,
each over four runtime values per artifact. Run only this targeted file:

```bash
source .claude/skills/testing/load-env.sh
python -m pytest tests/st/runtime/ops/test_buffer_ir.py --platform=a2a3 --device=0 \
    --precompile-workers "$PYPTO_TEST_JOBS" --save-kernels -v
```

The precompile mode also checks the saved Buffer program and PTO source from the
executed artifact. `--codegen-only` is useful for compilation checks but does not
provide numerical evidence. Harness tests cover inline and pool-thread option
propagation without requiring a device.

`tests/st/runtime/control_flow/test_buffer_ir.py` adds branch, For, While,
nested-loop and fanout numerical cases for each planner. Each compiled device
kernel receives counts and flags read from an orchestration config tensor, so
one artifact exercises multiple runtime paths. For and While cover counts
0, 1, 2 and 3, both branch arms, odd/even swaps, interleaved scalar offsets and
same-GM carries. Separate output bands retain unwritten sentinel values and
the original input after the loop. An asymmetric final expression detects
swaps that a commutative sum would hide. Nested cases include zero outer or
inner iterations; fanout checks two destinations reading one source.

On hosts with the task-submit device queue, run this bounded matrix with:

```bash
source .claude/skills/testing/load-env.sh
python -m pytest tests/st/runtime/control_flow/test_buffer_ir.py --platform=a2a3 \
    --precompile-workers "$PYPTO_TEST_JOBS" --execute-via-task-submit \
    --execute-batch-size=4 --task-max-time=120 --save-kernels -v
```

The queue chooses an available device. On other hosts, omit the queue options
and select an available device with `--device`. The saved-artifact checks also
require scalar-only native loop results and the expected explicit operations;
native compilation alone does not establish numerical correctness.
