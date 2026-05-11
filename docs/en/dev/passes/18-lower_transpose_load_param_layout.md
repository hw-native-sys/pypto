# LowerTransposeLoadParamLayout Pass

Lowers ``tile.load(..., transpose=True)`` to canonical-form DN parameter layout (RFC #1300 P6).

## Overview

Before this pass, ``tile.load(transpose=True)`` is the user's way of saying "I want
the column-major view of this source tensor at the load site". After this pass, that
intent is encoded into the InCore parameter's TensorType itself — the source/load
combo is rewritten to RFC #1300 §3.3 canonical form so codegen, verifier, and
downstream passes consume a single, self-consistent ``(shape, stride, layout)`` triple.

For each InCore parameter ``p`` loaded via ``tile.load(p, ..., transpose=True)``:

- ``p``'s TensorType is promoted from ``[..., a, b] ND`` to ``[..., b, a] DN`` —
  the trailing-pair shape swap plus the DN layout tag. The new TensorView carries
  an empty stride; ``MaterializeTensorStrides`` (which runs later in the default
  pipeline, after ``CanonicalizeIOOrder``) fills it with the packed canonical
  strides.
- Every ``tile.load(p, offsets, shapes, valid_shapes, ..., transpose=True)`` whose
  source is a promoted parameter is rewritten so the three tuples' trailing pair
  is swapped to canonical coords and the ``transpose=True`` kwarg is dropped.
  ``DeduceTileLoadType`` reads the source's DN layout to derive the Mat tile-view
  layout that the legacy ``transpose=True`` swap produced — the two signals are
  equivalent (§4.2 canonical pair).
- Every non-InCore call site that targets a promoted callee wraps the promoted
  argument in ``tensor.as_layout(arg, DN)`` (RFC #1300 P4). The bridging op is
  pure metadata — it emits no PTOAS instruction; ``make_tensor_view`` consumes
  the new view directly.

**Requirements**:

- Input IR must be in SSA form
- InCore functions must already be split out (``SplitIncoreOrch``)
- Tile ops must be present and 2D (``IncoreTileOps``, ``TileOps2D``)
- Promoted parameters must have rank ≥ 2

**When to use**: 18th pass in the ``Default`` strategy, after
``InferTileMemorySpace`` and before ``ResolveBackendOpLayouts``. The 2D shape
produced by ``FlattenTileNdTo2D`` is a precondition. ``MaterializeTensorStrides``
runs later in the pipeline (after ``CanonicalizeIOOrder``) to materialize the
DN-packed canonical strides on the promoted parameters.

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
    promote f.params[idx].type:  [..., a, b] ND  →  [..., b, a] DN (empty stride)
    substitute old Var → new Var in body
  rewrite each tile.load(promoted_param, off, shp, vs, transpose=True) in body:
    swap last two dims of off / shp / vs
    drop transpose=True kwarg

For each non-InCore function:
  walk body; for every Call whose op is a GlobalVar of a promoted callee:
    wrap each promoted-slot arg with tensor.as_layout(arg, DN)
```

**Complexity:** O(N log N) — one body walk per function plus one program-wide call-site
walk. Map lookups (``promotions_by_callee_name``) are ``log N`` per call.

| Behavior | Trigger |
| -------- | ------- |
| Promote param to ``[..., b, a] DN`` | InCore param is source of ``tile.load(..., transpose=True)`` |
| Skip param | Already DN, or no transposed load |
| Skip whole function | Function is Orchestration / Opaque / Group |
| Wrap call-site arg in ``tensor.as_layout`` | Non-InCore call to a promoted callee |
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
    b: pl.Tensor[[128, 32], pl.FP32, pl.DN],   # ← shape swapped + DN tag
    c: pl.Out[pl.Tensor[[64, 32], pl.FP32]],
) -> pl.Tensor[[64, 32], pl.FP32]:
    tile_a = pl.load(a, [0, 0], [64, 128], target_memory=pl.MemorySpace.Mat)
    tile_b = pl.load(b, [0, 0], [128, 32], target_memory=pl.MemorySpace.Mat)
                                           # ↑ no transpose kwarg
                                           # ↑ shapes swapped to canonical coords
    ...

@pl.function(type=pl.FunctionType.Orchestration)
def orchestrator(self, a, b):
    c = pl.create_tensor([64, 32], dtype=pl.FP32)
    # b is wrapped in tensor.as_layout to bridge ND → DN at the call site:
    bridged_b = tensor.as_layout(b, pl.DN)  # type: [128, 32] DN
    return self.matmul_incore(a, bridged_b, c)
```

``a`` is loaded without transpose, so it is unchanged. ``b`` is promoted in the
InCore signature, all body loads of ``b`` are rewritten to canonical coords with
no transpose, and the orchestrator's call site wraps ``b`` in
``tensor.as_layout`` to bridge ``[32, 128] ND`` → ``[128, 32] DN`` over the same
physical buffer.

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
| InCore (InCore, AIC, AIV) | Scanned, possibly promoted |
| Orchestration / Group / Opaque | Scanned for call sites; promoted-arg wrapped in ``tensor.as_layout`` |

| Parameter state | Action |
| --------------- | ------ |
| Sourced by ``tile.load(..., transpose=True)``, layout != DN, rank ≥ 2 | Promoted (shape swap + DN tag) |
| Sourced by ``tile.load(..., transpose=True)``, already DN | Idempotent — unchanged |
| Mixed transpose=True / transpose=False on same param | ``CHECK`` failure |
| Not sourced by any transposed load | Unchanged |
| Rank < 2 candidate | ``CHECK`` failure |

## Interaction with ``tensor.as_layout`` (P4) and ``MaterializeTensorStrides`` (P3)

This pass is the first real consumer of ``tensor.as_layout`` in the default
pipeline. The bridging op is single-purpose: it flips the layout tag and derives
the new shape from §4.2 canonical pair semantics — callers never write the
target shape, so the call-site rewriter cannot get it wrong.

Downstream, ``MaterializeTensorStrides`` fills the empty stride slot on each
promoted parameter with the packed canonical DN strides (RFC §2.4). The
combination of P6 + P3 is what gives codegen a self-consistent
``(shape, stride, layout)`` triple — no further ``dn_swap`` / ``get_shape_source_idx``
fix-ups are needed in the codegen path for promoted parameters.

## Interaction with ``tensor.transpose`` at Orchestration

A parameter whose source TensorView carries both ``layout = DN`` *and* an
explicit non-empty ``stride`` is the signature of a ``tensor.transpose`` result.
This pass rejects ``tile.load(transpose=True)`` on such parameters with a
``CHECK`` failure — the two encodings would compose as a double transpose at
codegen time and emit wrong addresses. Slice-derived inputs (explicit strides +
``layout = ND``, attached by ``OptimizeOrchTensors``) are unaffected.

Workaround for the rejected case: drop one of the two transpose layers in the
source program.
