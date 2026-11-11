# LowerAutoVectorSplit Pass

The public pass name is retained for compatibility. This pass now prepares
**explicit AIV regions** for `ExpandMixedKernel`; it does not implicitly halve
mixed-kernel computation or synthesize shard/gather operations.

## Authoring contract

Keep `pl.split_aiv(2, mode=...)` to declare AIV placement and bind the lane ID.
Write `pl.aiv_shard` at an AIC→AIV boundary and `pl.aic_gather` at an AIV→AIC
boundary. They inherit the region's mode; the high-level calls have no `mode`
parameter of their own.

The region body is authored per lane. Its computation shapes, load/store offsets,
loop-carried values and auxiliary input slices are not implicitly rewritten.
This also applies to regions without a crossing: they need no artificial
shard/gather, but their addresses and shapes remain the author's responsibility.

| Mode | Explicit boundary | Computation body |
| ---- | ----------------- | ---------------- |
| `UP_DOWN` | Shard/gather rows | Authored per lane |
| `LEFT_RIGHT` | Shard/gather columns | Authored per lane |
| `NONE` | Transfer without changing shape | Authored full width; use lane ID for disjoint work |

For example, with a `[128, 128]` accumulator, bias and output:

```python
with pl.at(level=pl.Level.CORE_GROUP):
    acc = pl.matmul(a, b, out_dtype=pl.FP32)
    for lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
        half = pl.aiv_shard(acc)
        row = lane * 64
        out[row:row + 64, :] = pl.add(half, bias[row:row + 64, :])
```

The former `optimizations=[pl.split(pl.SplitMode.UP_DOWN)]` or `LEFT_RIGHT`
AUTO path on a mixed InCore function is rejected with migration guidance.
Removing that annotation alone is not equivalent to explicit splitting.

**Unsplit mixed kernels keep their existing full-width implicit communication.**
This change removes implicit halving, not all automatic cross-core transport.
Standalone AIV/AIC functions retain their separate `SplitVectorKernel` path.

## Lowering

1. Find explicit regions recursively through loops and conditionals.
2. For split modes, retain the existing transpose-hazard validation and localize
   explicit boundary valid extents using the region's authored lane binding.
3. Preserve the body and wrapper; recursively process child regions.
4. Clone the rebuilt function and stamp `split_aiv`. `ExpandMixedKernel` consumes
   the wrappers and lowers explicit boundaries to `tpush`/`tpop`.
5. Reject mixed function-level AUTO splitting. Pass through other functions;
   retain the existing whole-function transpose check where applicable.

No extra lane index is injected into an ordinary region. For a ragged explicit
shard, the per-lane valid extent may still require repair: this is boundary
metadata, not implicit rewriting of the author's compute or addresses. The
existing transport representability restrictions still apply. The removed AUTO
path no longer rebalances a ragged body's partition on the author's behalf.

Non-region scopes must already have been outlined. A region hidden behind such
a scope, or a surviving scope inside a region, produces a diagnostic.

## API and properties

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::LowerAutoVectorSplit()` | `passes.lower_auto_vector_split()` | Program |

The pass runs immediately before `ExpandMixedKernel` in the Default pipeline.

| Property | Contract |
| -------- | -------- |
| Required | `SSAForm`, `IncoreTileOps`, `SplitIncoreOrch`, `TileOps2D`, `TileMemoryInferred`, `NormalizedStmtStructure`, `AivSplitValid` |
| Produced | Same structural properties, plus `AivSplitLoweredValid` |
| Invalidated | `AivSplitValid` |

The source verifier still enforces region placement and explicit crossings.
The lowered verifier admits retained regions and supported flat explicit IR.
`SplitVectorKernel` stamps dispatch metadata for explicit kernels without
halving them again.

## Implementation and tests

- `src/ir/transforms/lower_auto_vector_split_pass.cpp`
- `src/ir/transforms/utils/split_axis_utils.cpp` — shared explicit-boundary and
  standalone-kernel utilities; not all of this library is specific to AUTO.
- `tests/ut/ir/transforms/test_lower_auto_vector_split.py`
- `tests/ut/ir/transforms/test_expand_mixed_kernel_split_aiv.py`
- `tests/st/codegen/torch/test_torch_codegen_cross_core.py`

See [ExpandMixedKernel](26-expand_mixed_kernel.md),
[SplitVectorKernel](28-split_vector_kernel.md), and
[Scopes and Placement](../../user/language/04-scopes.md).
