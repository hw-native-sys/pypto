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

## Initial supported recipes

This slice supports straight-line, static rank-2 dense Vec FP32 tiles with one
descriptor per allocation, static valid extents, ordinary packed ND GM tensors,
and default load/store policies. It converts allocation, create, load, store,
add, multiply, move, and already legalized aliases.

Branches, loops, helper calls, alternate layouts, dynamic metadata, slots, and
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
