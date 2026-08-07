# ExpandMxPackedQuant Pass

把紧凑形式的 `tile.tquant_mx` 展开为 PTOAS 支持的平铺 MX 量化算子。Pass 在 tile IR 中显式化 16×64 分块打包，同时保持公开的 `MX_A_ZZ` 和 `MX_B_NN` 结果布局。

## 概述 (Overview)

`ExpandMxPackedQuant` 是面向 InCore 函数的函数级 Pass。它只改写 `tile.tquant_mx(..., layout=MX_A_ZZ)` 和 `tile.tquant_mx(..., layout=MX_B_NN)`；不带 `layout` 的平铺调用留给 [`LowerCompositeOps`](13-lower_composite_ops.md)。不含紧凑 MX 量化的函数在结构上保持不变。

输入必须是静态二维 tile，第一维可被 16 整除，第二维可被 64 整除。每个 16×64 分块先 reshape 为 `[32, 32]`，再由平铺 `tile.tquant_mx` 量化，最后 reshape 回原形状。每个 scale group 对应 32 个输入值。

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

该 Pass 是 `tile_pto_passes` 的第一项，也是 `Default` 流水线中编号第 12 的 Pass。它紧跟 `OptimizeOrchTensors`，并在 `LowerCompositeOps` 之前运行，后者不应看到带紧凑 `layout` 关键字参数的调用。

## 下降路径 (Lowering Paths)

Pass 先通过一次线性 IR 遍历收集紧凑量化定义、元组投影和带常量偏移的 store，并沿简单变量别名解析源 `tile.load`。

### Store 融合路径

当 quant 和 scale 结果都只被可见 store 使用、store 目标是函数参数、两个 store 与量化处于同一直线语句序列、量化到对应 store 之间没有访问该目标，且输入可解析为常量偏移 `tile.load` 时，Pass 会独立加载并量化每个分块，再直接写入目标 tensor。此时才能安全删除仅供 store 使用的元组投影，以及仅供该量化使用的源 load。动态 load 偏移、跨控制流的 store、中间穿插的目标访问、稍后才定义的目标 SSA 值，以及其他结果消费者都会选择 assemble 回退路径。

对于 `MX_B_NN`，Pass 先在编译器自有的 Vec 存储中 assemble 中间 `[N, K]` 量化数据，然后 reinterpret 为 `INT8`，转置为 `[K, N]`，再 reinterpret 回 `FP8E4M3FN`。它不会借用函数的 `Out` 或 `InOut` 参数作为临时缓冲区。

### Assemble 回退路径

当 store 不可见，或输入经过变换而不是可解析的 load 时，Pass 从输入切出每个分块，并在 Vec tile 中 assemble 量化与 scale 结果。元组投影别名会保留给原消费者。Scale 缓冲区会带规范 MX fractal 元数据 reinterpret，以保证元组赋值与 IR round-trip 类型检查一致。

## 临时值生命期 (Temporary Lifetimes)

每处理 16 个分块，以及最后一个不满 16 的分块组后，Pass 都使用 `system.bar_all` 排空临时 tile。这会限制大型紧凑输入的异步 Vec 生命期。B 转置输入也会保持存活，直到最后一次 store 之后的排空点。

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
- [`InsertMxScaleAddr`](19-insert_mx_scale_addr.md) — 为后续 MX matmul 消费者物化 scale 地址。
