# BlockMxScaleTensorViews Pass

## Overview

`BlockMxScaleTensorViews` converts logical MX scale tensor views into the
packed rank-5 SFractal form required by A5, and rewrites every operation whose
coordinates depend on that physical form.

This is a **migration of physical-view lowering** from the backend's former
`EmitMxPhysicalView` helper into an explicit IR pass. After the pass, tensor
types, load windows, aliases, stride materialization, verification, and generic
codegen all see one canonical representation; codegen no longer reconstructs a
different MX view privately.

The pass is independent from `BlockNzTensorViews`. MX and NZ are different
layouts and keep separate transformation and proof implementations.

## Physical form

The DSL exposes two logical rank-2 scale layouts:

```text
MX_A_ZZ [M, G] -> [1, M/16, G/2, 16, 2]
MX_B_NN [G, N] -> [1, N/16, G/2, 16, 2]
```

The trailing `[16, 2]` is one 32-byte FP8E8M0 scale box. Row-major strides over
the blocked shape are the physical GlobalTensor strides, so
`MaterializeTensorStrides` can use its ordinary packed-stride path after this
pass.

## Position in the pipeline

```text
... -> FlattenTileNdTo2D -> BlockNzTensorViews
    -> BlockMxScaleTensorViews -> LegalizeTileCast -> ...
```

The pass runs after `FlattenTileNdTo2D`, when `tile.load` results are logical
2-D tiles, and before all consumers that require a physical MX tensor shape.
`MaterializeTensorStrides` later fills the rank-5 row-major strides.

## Rewrites

For every MX_A_ZZ or MX_B_NN tensor, the pass rewrites:

- `TensorType` slots recursively inside parameters, returns, tuples, variables,
  iteration arguments, Calls, and Submits;
- `tile.load` offsets and shapes into rank-5 coordinates;
- physical `valid_shape` arguments to the complete aligned load box while
  preserving a narrowed logical `TileType.valid_shape` as tile metadata;
- shaped FP8E8M0 `tensor.view` aliases in both ND-to-MX and MX-to-ND directions;
- Submit return types without changing dependencies, keyword arguments,
  attributes, core count, predicate, or synchronization fields.

The destination `TileType` of a load remains logical 2-D. Only the GM source
partition becomes rank-5.

## Offset mapping and proofs

Logical coordinates map as follows:

```text
MX_A_ZZ [m0, g0] -> [0, m0/16, g0/2, 0, 0]
MX_B_NN [g0, n0] -> [0, n0/16, g0/2, 0, 0]
```

Constants must be non-negative and aligned. Symbolic offsets are accepted only
when the pass's private MX proof engine proves both divisibility and
non-negativity. It understands:

- scalar SSA definitions;
- constants, multiplication, addition, and subtraction for divisibility;
- positive power-of-two floor division, including forms such as `k0 // 32`;
- loop variables with constant start and step;
- the non-negative results of `tile.get_block_idx` and `tile.get_block_num`;
- scalar arguments propagated through every Call and Submit site.

Every caller must prove a callee parameter fact. Missing or malformed call
mappings, recursion, and expressions that exceed the bounded 256-step proof
budget are rejected conservatively. The bound keeps the pass O(N) and proof
failure can never turn into assumed alignment.

The quotient is emitted as `FloorDiv(offset, divisor)`, preserving the original
fixed-width expression's evaluation before division.

## Scope and diagnostics

| Condition | Outcome |
| --------- | ------- |
| static, aligned rank-2 MX shape | converted to canonical rank-5 form |
| aligned, provably non-negative symbolic offset | converted to rank-5 coordinates |
| unprovable or negative offset | rejected |
| partial tensor-level `valid_shape` | rejected |
| narrowed load-level `valid_shape` | kept as tile metadata; physical box stays complete |
| `target_memory != Mat` or missing | rejected |
| MX tensor used by an unsupported operator or store | rejected |
| shaped FP8E8M0 ND/MX backing alias | rewritten |
| distributed MX tensor | rejected |

Functions receive the `mx_tensor_views_blocked` attribute after the pass. That
provenance stamp makes a second invocation a no-op without trying to infer pass
state from a shape that might coincidentally look blocked.

## See also

- [BlockNzTensorViews](15-block_nz_tensor_views.md)
- [MaterializeTensorStrides](33-materialize_tensor_strides.md)
- [InsertMxScaleAddr](21-insert_mx_scale_addr.md)
