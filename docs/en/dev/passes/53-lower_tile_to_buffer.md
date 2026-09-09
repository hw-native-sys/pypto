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

## Initial supported recipes

The current recipes support straight-line kernels, branches and loops with static
rank-2 dense Vec FP32 tiles with one
descriptor per allocation, static valid extents, ordinary packed ND GM tensors,
and default load/store policies. It converts allocation, create, load, store,
add, multiply, move, and already legalized aliases.

Helper calls, alternate layouts, dynamic metadata, slots, and
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

For numerical system tests, declare `st.case(..., enable_buffer_ir=True,
memory_planner=...)` on the public `@pl.jit` entry. The harness applies the option
inside both inline and precompile-worker compilations; an outer test-thread
`PassContext` alone does not configure worker threads. Enabled cases have distinct
cache keys. The harness checks the actual final device function stages and saves
the transformed program as `buffer_ir.msgpack` beside the native artifacts.

`tests/st/runtime/ops/test_buffer_ir.py` provides load/add/mul/store numerical
cases with orchestration for all three planners. Run only this targeted file:

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
