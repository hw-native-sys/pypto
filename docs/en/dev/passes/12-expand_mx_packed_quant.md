# ExpandMxPackedQuant Pass

Expands packed `tile.tquant_mx` forms into the flat MX quantization operation supported by PTOAS. It preserves the public `MX_A_ZZ` and `MX_B_NN` result layouts while making the required 16×64 box packing explicit in tile IR.

## Overview

`ExpandMxPackedQuant` is a function-level pass over InCore functions. It rewrites only `tile.tquant_mx(..., layout=MX_A_ZZ)` and `tile.tquant_mx(..., layout=MX_B_NN)` calls; flat calls without `layout` are left for [`LowerCompositeOps`](13-lower_composite_ops.md). Functions without packed MX quantization are structural no-ops.

The input must be a static two-dimensional tile whose first dimension is divisible by 16 and whose second dimension is divisible by 64. Each 16×64 box is reshaped to `[32, 32]`, quantized by a flat `tile.tquant_mx`, and reshaped back. Scale groups contain 32 source values.

The two layouts produce:

| Layout | Input | Quant result | Scale result |
| ------ | ----- | ------------ | ------------ |
| `MX_A_ZZ` | `[M, K]` | `[M, K]`, boxes in row-major ZZ order | `[1, M*K/32]`, continuous ZZ order |
| `MX_B_NN` | `[N, K]` | `[K, N]`, produced by an INT8 bit-preserving transpose | `[1, N*K/32]`, continuous NN order |

**Requires**: nothing.

**Produces**: nothing.

**Invalidates**: nothing.

The empty property contract is declared as `kExpandMxPackedQuantProperties` in `include/pypto/ir/transforms/pass_properties.h`.

## When It Runs

This is the first entry of `tile_pto_passes` and the 12th documented pass in the `Default` pipeline. It runs immediately after `OptimizeOrchTensors` and before `LowerCompositeOps`, which must not see a packed `layout` keyword.

## Lowering Paths

The pass first collects packed-quant definitions, tuple projections, and constant-offset stores with one linear IR walk. It follows simple variable aliases to resolve a source `tile.load`.

### Store-fused path

When both results are used exclusively by visible stores to function-parameter destinations, both stores remain in the same straight-line sequence as the quantization, neither destination is accessed between the quantization and its store, and the source resolves to a constant-offset `tile.load`, every box is loaded and quantized independently, then written directly into the destination tensors. The store-only tuple projections and a source load used only by this quantization can then be removed safely. Dynamic load offsets, control-flow-separated stores, intervening destination accesses, later destination SSA values, and additional result consumers all select the assemble fallback.

For `MX_B_NN`, the pass assembles the intermediate `[N, K]` quant data in compiler-owned Vec storage, reinterprets it as `INT8`, transposes it to `[K, N]`, and reinterprets it back to `FP8E4M3FN`. It never borrows an `Out` or `InOut` function parameter as scratch storage.

### Assemble fallback

If the stores are not visible or the source is transformed rather than a resolvable load, the pass slices each box from the input and assembles the packed quant and scale results in Vec tiles. Projection aliases remain live for the original consumers. The scale buffer is reinterpreted with the canonical MX fractal metadata so tuple assignment and IR round-trip type checks remain coherent.

## Temporary Lifetimes

Per-box tiles are drained with `system.bar_all` after every 16 boxes, and after the final partial chunk. This bounds asynchronous Vec lifetimes for large packed inputs. The B transpose input is also kept alive through a drain after the final store.

## API and Implementation

```python
from pypto.pypto_core import passes

packed_quant = passes.expand_mx_packed_quant()
```

- Declaration: `include/pypto/ir/transforms/passes.h`
- Implementation: `src/ir/transforms/expand_mx_packed_quant_pass.cpp`
- Python binding: `python/bindings/modules/passes.cpp`
- Default ordering: `python/pypto/ir/pass_manager.py`

## See Also

- [`LowerCompositeOps`](13-lower_composite_ops.md) — lowers the remaining flat `tile.tquant_mx` call to its raw destination form.
- [Tile Operators](../ir/05-operators.md) — public MX quantization shapes and dtype contract.
- [`InsertMxScaleAddr`](19-insert_mx_scale_addr.md) — materializes scale addresses for later MX matmul consumers.
