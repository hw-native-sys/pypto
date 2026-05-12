# LowerTransposeLoadParamLayout Pass

Lowers ``tile.load(..., transpose=True)`` by emitting an explicit
``tensor.as_layout`` view inside the InCore body (RFC #1300 P6).

## Overview

Before this pass, ``tile.load(transpose=True)`` is the user's way of saying "I
want the column-major view of this source tensor at the load site". After this
pass, that intent is encoded into a body-local ``tensor.as_layout`` view at the
top of the InCore body so codegen, verifier, and downstream passes consume a
single, self-consistent ``(shape, stride, layout)`` triple.

For each InCore parameter ``p`` loaded via ``tile.load(p, ..., transpose=True)``:

- The InCore body is **prepended** with ``p_dn = tensor.as_layout(p, layout=DN)``.
  The new Var ``p_dn`` carries the canonical ``[..., b, a] DN`` view (trailing-pair
  shape swap + DN layout tag with packed canonical strides set by the
  ``tensor.as_layout`` deduce-type).
- Body uses of ``p`` are substituted with ``p_dn``. ``p``'s parameter
  signature is left unchanged — the orch side keeps passing its original
  row-major ND tensor (which matches the runtime torch tensor's layout).
- Every ``tile.load(p, offsets, shapes, valid_shapes, ..., transpose=True)``
  whose source is a promoted parameter is rewritten to ``tile.load(p_dn, ...)``,
  with the three tuples' trailing pair swapped to canonical coords and
  ``transpose=True`` flipped to ``transpose=False``.
  ``DeduceTileLoadType`` reads ``p_dn``'s DN layout to derive the Mat tile-view
  layout that the legacy ``transpose=True`` swap produced — the two signals are
  equivalent (§4.2 canonical pair).

Non-InCore (orch) functions are not touched. The DN reinterpret is a
single-function concern owned by the InCore body that needs it, which keeps the
cross-function boundary trivial: orch always passes a row-major ND tensor.

**Requirements**:

- Input IR must be in SSA form
- InCore functions must already be split out (``SplitIncoreOrch``)
- Tile ops must be present and 2D (``IncoreTileOps``, ``TileOps2D``)
- Promoted parameters must have rank ≥ 2

**When to use**: 18th pass in the ``Default`` strategy, after
``InferTileMemorySpace`` and before ``ResolveBackendOpLayouts``. The 2D shape
produced by ``FlattenTileNdTo2D`` is a precondition.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| ``pass::LowerTransposeLoadParamLayout()`` | ``passes.lower_transpose_load_param_layout()`` | Program-level |

**Python usage**:

```python
from pypto.pypto_core import passes

p = passes.lower_transpose_load_param_layout()
program_canonical = p(program)
```

## Algorithm

```text
For each InCore function f:
  scan body → set P_t  = {param idx with tile.load(p, ..., transpose=True)}
              set P_nt = {param idx with tile.load(p, ..., transpose=False/absent)}
  reject P_t ∩ P_nt  (mixed-use)
  for each idx in P_t:
    let p = f.params[idx]
    skip if p is already DN-tagged (the user-written / pre-canonical case)
    build p_dn := tensor.as_layout(p, layout=DN)  — type derived by op deducer
    prepend (p_dn = ...) AssignStmt to body
    record p → p_dn in substitution map
  substitute body uses of every promoted p with p_dn
  rewrite each tile.load(p_dn, off, shp, vs, transpose=True) in body:
    swap last two dims of off / shp / vs
    drop transpose=True kwarg

(Non-InCore functions are untouched.)
```

**Complexity:** O(N log N) — one body walk per InCore function.

| Behavior | Trigger |
| -------- | ------- |
| Prepend ``p_dn = tensor.as_layout(p, DN)`` and rewrite tile.load | InCore param is source of ``tile.load(..., transpose=True)`` |
| Skip param | Already DN, or no transposed load |
| Skip whole function | Function is Orchestration / Opaque / Group |
| Reject | Mixed transpose=True / transpose=False on same param |
| Reject | DN + explicit physical stride source (would compose as double transpose) |

## Example

**Before**:

```python
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def matmul_incore(
        self,
        a: pl.Tensor[[64, 128], pl.FP32],
        b: pl.Tensor[[32, 128], pl.FP32],
        c: pl.Out[pl.Tensor[[64, 32], pl.FP32]],
    ) -> pl.Tensor[[64, 32], pl.FP32]:
        tile_a = pl.load(a, [0, 0], [64, 128], target_memory=pl.MemorySpace.Mat)
        tile_b = pl.load(b, [0, 0], [32, 128], target_memory=pl.MemorySpace.Mat, transpose=True)
        ...

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(self, a, b):
        c = pl.create_tensor([64, 32], dtype=pl.FP32)
        return self.matmul_incore(a, b, c)
```

**After** (semantic — ``tensor.as_layout`` is an internal IR op, not exposed in pl.*):

```text
@pl.function(type=pl.FunctionType.InCore)
def matmul_incore(
    self,
    a: pl.Tensor[[64, 128], pl.FP32],
    b: pl.Tensor[[32, 128], pl.FP32],            # ← param signature unchanged
    c: pl.Out[pl.Tensor[[64, 32], pl.FP32]],
) -> pl.Tensor[[64, 32], pl.FP32]:
    b_dn = tensor.as_layout(b, layout=DN)         # ← prepended view
                                                   #   type: [128, 32] DN
    tile_a = pl.load(a, [0, 0], [64, 128], target_memory=pl.MemorySpace.Mat)
    tile_b = pl.load(b_dn, [0, 0], [128, 32], target_memory=pl.MemorySpace.Mat)
                                                   # ↑ source switched to b_dn
                                                   # ↑ shapes swapped to canonical coords
                                                   # ↑ no transpose kwarg
    ...

@pl.function(type=pl.FunctionType.Orchestration)
def orchestrator(self, a, b):
    c = pl.create_tensor([64, 32], dtype=pl.FP32)
    return self.matmul_incore(a, b, c)            # ← unchanged
```

``a`` is loaded without transpose, so it is unchanged. ``b``'s param signature
is preserved; the kernel internally derives a DN view via ``tensor.as_layout``
and references that view in its ``tile.load``. The orchestrator is not
touched — it passes its own row-major ``b`` straight through.

## Implementation

**Header**: ``include/pypto/ir/transforms/passes.h``

**Implementation**: ``src/ir/transforms/lower_transpose_load_param_layout_pass.cpp``

**Python binding**: ``python/bindings/modules/passes.cpp``

**Tests**: ``tests/ut/ir/transforms/test_lower_transpose_load_param_layout_pass.py``

## Pass Properties

| Property | Value |
| -------- | ----- |
| Required | SSAForm, IncoreTileOps, SplitIncoreOrch, TileOps2D |
| Produced | SSAForm, IncoreTileOps, SplitIncoreOrch, TileOps2D |
| Invalidated | — |

## Scope

| Function type | Action |
| ------------- | ------ |
| InCore (InCore, AIC, AIV) | Scanned, body prepended with ``tensor.as_layout`` views as needed |
| Orchestration / Group / Opaque | Untouched |

| Parameter state | Action |
| --------------- | ------ |
| Sourced by ``tile.load(..., transpose=True)``, layout != DN, rank ≥ 2 | ``tensor.as_layout`` view prepended; body uses substituted |
| Sourced by ``tile.load(..., transpose=True)``, already DN | Skipped — ``DeduceTileLoadType`` already handles DN-source XOR transpose |
| Mixed transpose=True / transpose=False on same param | ``CHECK`` failure |
| Not sourced by any transposed load | Unchanged |
| Rank < 2 candidate | ``CHECK`` failure |

## Interaction with ``tensor.as_layout`` (P4)

This pass is the first consumer of ``tensor.as_layout`` in the default
pipeline. The bridging op is single-purpose: it flips the layout tag and
derives the new shape from §4.2 canonical pair semantics, then attaches the
packed canonical strides via ``CanonicalizeView``. Codegen lowers
``tensor.as_layout`` to a fresh ``pto.make_tensor_view`` bound to the input
tensor's underlying SSA buffer with the LHS's
``(shape, stride, layout)`` triple — no PTOAS instruction is emitted, the
result is a pure metadata reinterpret.

Per RFC §4.2, the InCore-side reinterpret does not violate the "InCore cannot
create tensors" invariant: ``tensor.as_layout`` allocates nothing, it
re-describes the input's existing physical buffer.

## Interaction with ``tensor.transpose`` at Orchestration

A parameter whose source TensorView carries both ``layout = DN`` *and* an
explicit non-empty ``stride`` is the signature of a ``tensor.transpose`` result.
This pass rejects ``tile.load(transpose=True)`` on such parameters with a
``CHECK`` failure — the two encodings would compose as a double transpose at
codegen time and emit wrong addresses. Slice-derived inputs (explicit strides +
``layout = ND``, attached by ``OptimizeOrchTensors``) are unaffected.

Workaround for the rejected case: drop one of the two transpose layers in the
source program.
