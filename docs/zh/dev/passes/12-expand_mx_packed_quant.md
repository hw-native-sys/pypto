# ExpandMxPackedQuant Pass

早期 MX 合法化：先按 K=64 切开大 K 的 packed quant / `matmul_mx`，再把紧凑 `tile.tquant_mx` 展开为 PTOAS 支持的平铺 MX 量化。Pass 在 tile IR 中显式化 16×64 分块打包，同时保持公开的 `MX_A_ZZ` 和 `MX_B_NN` 结果布局。

## 概述 (Overview)

`ExpandMxPackedQuant` 是面向 InCore 函数的函数级 Pass，内部多阶段：

1. **K-split**：静态 `K>64` 且 `K%64==0` 时
   - **共切**：若 `matmul_mx*` 的 data/scale 均为同一次 `tquant_mx(layout)` 的直接 `TupleGetItem`（index 0/1），则改为每段 K=64 的 packed `tquant_mx` + `matmul_mx` / `matmul_mx_acc`，并删除原 large-K quant 链。此路径按 **chunk 布局**消费 scale（字节序可不同于 full pack）；前端不要在中间插入 scale reshape；
   - **仅 matmul**：否则只对 data/scale 做 `tile.slice`（scale 须已是逻辑 2D，按每块 2 个 group）。
2. **Flat→逻辑 2D reshape**：若剩余 `matmul_mx*` 的 scale 仍是 packed-flat `[1,G]`（lhs `[1,M*(K/32)]` / rhs `[1,N*(K/32)]`），在 matmul 前插入 `tile.reshape` 到 `[M,K/32]` / `[K/32,N]`。前端不应手写该 reshape。
3. **Expand**：改写剩余的 `tile.tquant_mx(..., layout=MX_A_ZZ|MX_B_NN)`（含孤立 large-K）。逐盒 assemble 的盒序为 **(mb|nb 外层, kb 内层)**，与主机 `_pack_a_scale` / `_pack_b_scale` 的 full pack 字节序一致；`K%64==0`。不带 `layout` 的平铺调用留给 [`LowerCompositeOps`](13-lower_composite_ops.md)。

每个 16×64 分块先 reshape 为 `[32, 32]`，再由平铺 `tile.tquant_mx` 量化，最后 reshape 回原形状。每个 scale group 对应 32 个输入值。

两种布局的结果为：

| 布局 | 输入 | 量化结果 | Scale 结果 |
| ---- | ---- | ---------- | ------------ |
| `MX_A_ZZ` | `[M, K]` | `[M, K]`，分块按行主序 ZZ 顺序排列 | `[1, M*K/32]`，连续 ZZ 顺序 |
| `MX_B_NN` | `[N, K]` | `[K, N]`，通过保持比特的 INT8 转置产生 | `[1, N*K/32]`，连续 NN 顺序 |

**前置条件 (Requires)**：无。

**产生属性 (Produces)**：无。

**失效属性 (Invalidates)**：无。

空属性约定在 `include/pypto/ir/transforms/pass_properties.h` 中声明为 `kExpandMxPackedQuantProperties`。

## 运行时机 (When It Runs)

该 Pass 是 `tile_pto_passes` 的第一项，也是 `Default` 流水线中编号第 12 的 Pass。它紧跟 `OptimizeOrchTensors`，并在 `LowerCompositeOps` 之前运行。大 K 切分也在此 early 完成；后续 [`InferTileMemorySpace`](18-infer_tile_memory_space.md) / [`InsertMxScaleAddr`](20-insert_mx_scale_addr.md) 只看到 K=64 的 MX matmul。

## 下降路径 (Lowering Paths)

统一走 **Vec assemble** 路径：若源可解析为常量偏移 `tile.load`，则逐盒 `tile.load`；否则对聚合 tile 做 `tile.slice`。每个盒经 `QuantizeBox`（reshape → 平铺 `tquant_mx` → reshape）后，assemble 进 quant / scale 缓冲。`MX_B_NN` 在 assemble `[N,K]` 之后再做 INT8 转置得到 `[K,N]`。

每处理 16 个分块（及末尾不满组）插入 `system.bar_all`，限制异步 Vec 生命期；B 转置后再排空一次。

## API 与实现

```python
from pypto.pypto_core import passes

packed_quant = passes.expand_mx_packed_quant()
```

- 声明：`include/pypto/ir/transforms/passes.h`
- 实现：`src/ir/transforms/expand_mx_packed_quant_pass.cpp`
- Python 绑定：`python/bindings/modules/passes.cpp`
- 默认顺序：`python/pypto/ir/pass_manager.py`

## 另请参阅 (See Also)

- [`LowerCompositeOps`](13-lower_composite_ops.md) — 把余下的平铺 `tile.tquant_mx` 下降为原始 destination 形式。
- [Tile 算子](../ir/05-operators.md) — 公开 MX 量化形状与 dtype 约定。
- [`InsertMxScaleAddr`](20-insert_mx_scale_addr.md) — 为后续 MX matmul 消费者物化 scale 地址。
- [`ExpandMixedKernel`](23-expand_mixed_kernel.md) — 拒绝 `FP8E8M0` V2C；mixed kernel 须经 GM 暂存 MX A-scale。
