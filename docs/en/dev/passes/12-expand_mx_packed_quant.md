# ExpandMxPackedQuant Pass

Early MX legalization: K-split large-K packed quant / `matmul_mx`, then expand packed `tile.tquant_mx` into the flat MX quantization operation supported by PTOAS. It preserves the public `MX_A_ZZ` and `MX_B_NN` result layouts while making the required 16×64 box packing explicit in tile IR.

## Overview

`ExpandMxPackedQuant` is a function-level pass over InCore functions with several phases:

1. **K-split** for static `K>64` with `K%64==0`:
   - **Co-split**: when `matmul_mx*` data/scale are both direct `TupleGetItem` (index 0/1) of the same `tquant_mx(layout)`, rewrite into per-chunk K=64 packed `tquant_mx` + `matmul_mx` / `matmul_mx_acc` and drop the original large-K quant chain. This path consumes scales in **chunk layout** (byte order may differ from full pack). Frontends must not insert a scale reshape in between;
   - **Matmul-only**: otherwise slice data/scale tiles (scales must already be logical 2D; 2 groups per chunk).
2. **Flat→logical 2D reshape**: when a remaining `matmul_mx*` scale is still packed-flat `[1,G]` (lhs `[1,M*(K/32)]` / rhs `[1,N*(K/32)]`), insert `tile.reshape` to `[M,K/32]` / `[K/32,N]` before the matmul. Frontends must not write that reshape.
3. **Expand**: rewrite remaining `tile.tquant_mx(..., layout=MX_A_ZZ|MX_B_NN)` (including isolated large-K). Per-box assemble uses **(mb|nb outer, kb inner)** order matching host `_pack_a_scale` / `_pack_b_scale` full-pack bytes; `K%64==0`. Flat calls without `layout` are left for [`LowerCompositeOps`](13-lower_composite_ops.md).

Each 16×64 box is reshaped to `[32, 32]`, quantized by a flat `tile.tquant_mx`, and reshaped back. Scale groups contain 32 source values.

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

This is the first entry of `tile_pto_passes` and the 12th documented pass in the `Default` pipeline. It runs immediately after `OptimizeOrchTensors` and before `LowerCompositeOps`. Large-K splitting is done here early; later [`InferTileMemorySpace`](18-infer_tile_memory_space.md) / [`InsertMxScaleAddr`](20-insert_mx_scale_addr.md) only see K=64 MX matmuls.

## Lowering Paths

Only the **Vec assemble** path is used: if the source resolves to a constant-offset `tile.load`, each box is reloaded; otherwise each box is taken with `tile.slice` from the aggregate tile. Every box goes through `QuantizeBox` (reshape → flat `tquant_mx` → reshape) and is assembled into quant/scale buffers. For `MX_B_NN`, an INT8 bit-preserving transpose turns assembled `[N,K]` into `[K,N]`.

`system.bar_all` drains temporary tiles after every 16 boxes (and after the final partial chunk), plus once after the B transpose.

## API and Implementation

```python
from pypto.pypto_core import passes

packed_quant = passes.expand_mx_packed_quant()
```

- Declaration: `include/pypto/ir/transforms/passes.h`
- Implementation: `src/ir/transforms/expand_mx_packed_quant_pass.cpp`
- Python binding: `python/bindings/modules/passes.cpp`
- Default order: `python/pypto/ir/pass_manager.py`

## See Also

- [`LowerCompositeOps`](13-lower_composite_ops.md) — lowers remaining flat `tile.tquant_mx` to destination-passing form.
- [Tile operators](../ir/05-operators.md) — public MX quantization shape and dtype contracts.
- [`InsertMxScaleAddr`](20-insert_mx_scale_addr.md) — materializes scale addresses for MX matmul consumers.
- [`ExpandMixedKernel`](23-expand_mixed_kernel.md) — rejects `FP8E8M0` V2C; mixed kernels must stage MX A-scale via GM.
