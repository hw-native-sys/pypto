# ResolveTransposeLayout Pass

为作为 `tile.load(..., transpose=True)` 源张量的 InCore 函数参数标注 `DN`（列主序）布局。

## 概述

当 `tile.load` 使用 `transpose=True` 发起时，PTO codegen 需要源张量以列主序（`DN`）布局物化 —— 转置通过布局选择来实现，而不是通过对数据进行 reshape。该 Pass 把这一布局需求从 load 站点回传到函数参数类型，让下游 Pass 与 codegen 把参数的 `TensorType` 视为布局的唯一权威。

该 Pass 只标注参数 —— **形状保持不变**。`DN` 是布局/codegen 提示；逻辑张量维度不会被交换。（这正是 #606 的回归测试所保护的不变量：在 `[128, 128]` 上做窗口转置加载时，参数形状必须保持 `[128, 128]`，而不是 load 窗口的形状。）

**前置条件**：

- 输入 IR 必须为 SSA 形式
- InCore 函数已完成拆分（`SplitIncoreOrch`）
- Tile 操作已存在且为 2D（`IncoreTileOps`、`TileOps2D`）
- 待标注的张量参数必须 rank ≥ 2

**使用时机**：在 `Default` 策略中作为第 15 个 Pass 运行，位于 `InferTileMemorySpace` 之后、`ResolveBackendOpLayouts` 之前。`FlattenTileNdTo2D` 产生的 2D 形状是其前置条件。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::ResolveTransposeLayout()` | `passes.resolve_transpose_layout()` | Program 级 |

**Python 用法**：

```python
from pypto.pypto_core import passes

resolve_pass = passes.resolve_transpose_layout()
program_dn = resolve_pass(program)
```

## 算法

对程序中每个函数：

1. **跳过非 InCore 函数**：Orchestration 与 Opaque 函数原样返回。仅处理 InCore 类函数（InCore、AIC、AIV）。
2. **扫描 body 中的转置 load**：遍历函数体，对每个 kwarg `transpose=True` 且第一个参数是该函数某个 parameter 的 `tile.load` 调用，记录该 parameter 的索引。多次出现的同一参数会去重。
3. **重写参数**：对每个被收集到的参数：
   - **若已是 DN 则跳过**：参数的 `TensorType` 已携带 `TensorView{layout=DN}` 时无需重写（幂等）。
   - **要求 rank ≥ 2**：1D 张量谈不上列主序；遇到时通过 `CHECK` 终止。
   - 构造一个新的 `Var`，沿用原有的 `name_hint`、span 与形状，但其 `TensorType` 的 `tensor_view_` 为 `TensorView({}, TensorLayout::DN)`。
4. **替换**：通过 `Substitute` 把函数体内对旧 `Var` 的所有引用替换为新 `Var`，再用 `MutableCopy` 以新参数列表与新 body 重建函数。

不会对 Orchestration 端做任何改写。下游 Pass 与 codegen 把 InCore 签名视为布局的唯一权威。

| 行为 | 触发条件 |
| ---- | -------- |
| 给参数加 `DN` | InCore 函数参数是 `tile.load(..., transpose=True)` 的源 |
| 跳过该参数 | 已是 `DN`，或没有任何转置 load 命中它 |
| 跳过整个函数 | 函数为 Orchestration 或 Opaque |
| `CHECK` 失败 | 待标注参数不是 `TensorType`，或 rank < 2 |

## 示例

**之前**：

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
        tile_a_l0a = pl.move(tile_a, target_memory=pl.MemorySpace.Left)
        tile_b_l0b = pl.move(tile_b, target_memory=pl.MemorySpace.Right)
        tile_c = pl.matmul(tile_a_l0a, tile_b_l0b)
        c_store = pl.store(tile_c, [0, 0], c)
        return c_store

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self, a: pl.Tensor[[64, 128], pl.FP32], b: pl.Tensor[[32, 128], pl.FP32]
    ) -> pl.Tensor[[64, 32], pl.FP32]:
        c: pl.Tensor[[64, 32], pl.FP32] = pl.create_tensor([64, 32], dtype=pl.FP32)
        c_result = self.matmul_incore(a, b, c)
        return c_result
```

**之后**：

```python
@pl.program
class After:
    @pl.function(type=pl.FunctionType.InCore)
    def matmul_incore(
        self,
        a: pl.Tensor[[64, 128], pl.FP32],
        b: pl.Tensor[[32, 128], pl.FP32, pl.DN],
        c: pl.Out[pl.Tensor[[64, 32], pl.FP32]],
    ) -> pl.Tensor[[64, 32], pl.FP32]:
        tile_a = pl.load(a, [0, 0], [64, 128], target_memory=pl.MemorySpace.Mat)
        tile_b = pl.load(b, [0, 0], [32, 128], target_memory=pl.MemorySpace.Mat, transpose=True)
        tile_a_l0a = pl.move(tile_a, target_memory=pl.MemorySpace.Left)
        tile_b_l0b = pl.move(tile_b, target_memory=pl.MemorySpace.Right)
        tile_c = pl.matmul(tile_a_l0a, tile_b_l0b)
        c_store = pl.store(tile_c, [0, 0], c)
        return c_store

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self, a: pl.Tensor[[64, 128], pl.FP32], b: pl.Tensor[[32, 128], pl.FP32]
    ) -> pl.Tensor[[64, 32], pl.FP32]:
        c: pl.Tensor[[64, 32], pl.FP32] = pl.create_tensor([64, 32], dtype=pl.FP32)
        c_result = self.matmul_incore(a, b, c)
        return c_result
```

`b` 是带 `transpose=True` 的 `tile.load` 的源，因此 InCore 参数类型获得 `pl.DN` 布局标注。形状 `[32, 128]` 不变。`a` 没有转置 load，保持原样。Orchestration `orchestrator` 的签名**不会**被改写。

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`

**实现文件**：`src/ir/transforms/resolve_transpose_layout_pass.cpp`

**Python 绑定**：`python/bindings/modules/passes.cpp`

**测试**：`tests/ut/ir/transforms/test_resolve_transpose_layout_pass.py`

## Pass 属性

| 属性 | 值 |
| ---- | -- |
| 所需 | SSAForm, IncoreTileOps, SplitIncoreOrch, TileOps2D |
| 产生 | SSAForm, IncoreTileOps, SplitIncoreOrch, TileOps2D |
| 失效 | — |

该 Pass 保留所有输入属性：仅重写张量参数的类型标注，不改变语句结构或 SSA 形式。

## 作用范围

| 函数类型 | 处理方式 |
| -------- | -------- |
| InCore（InCore、AIC、AIV） | 扫描并可能改写 |
| Orchestration | 原样保留 |
| Opaque | 原样保留 |

| 参数状态 | 处理方式 |
| -------- | -------- |
| 是 `tile.load(..., transpose=True)` 的源、布局非 DN、rank ≥ 2 | 改写并加上 `DN` |
| 是 `tile.load(..., transpose=True)` 的源、布局已是 DN | 原样保留（幂等） |
| 不是任何转置 load 的源 | 原样保留 |
| 候选参数 rank < 2 | `CHECK` 失败 |

如果没有任何 InCore 函数包含以参数为源的 `tile.load(..., transpose=True)`，整个 Pass 是 no-op（由 `TestResolveTransposeLayoutNoOp` 测试类验证）。

## 与 Orchestration 层 `tensor.transpose` 的衔接

Orchestration 层的 `tensor.transpose`（见 issue #1209）在 `DeduceTensorTransposeType` 中根据是否为最后两维互换，把结果类型的 `layout` 在 `ND` 与 `DN` 之间切换。被转置的 Tensor 进入 InCore 参数后，参数自动继承 `DN` 标签，与本 Pass 输出的形式一致 —— 因此 `tile.load(..., target_memory=Mat, transpose=True)`（即 matmul B^T 场景）可以直接消费 Orchestration 层的 transpose，无需用户显式标注 `pl.DN`。

**Vec 目标是已知限制。** 跨布局 `TLOAD(VecTile, GlobalTensor)` 会被 PTO ISA 拒绝（仅支持 `ND→ND` / `DN→DN` / `NZ→NZ`），并且 `tile.load` 进一步限制 `transpose=True` 仅在 `target_memory=Mat` 时合法。因此，`pl.transpose` 在 Orchestration 层之后接 Vec 目标的消费者（例如非 matmul 的 `pl.at(level=CORE_GROUP)` 块内的 `pl.slice`），IR 可正确编译但会在 Ascend 910B 的 PTO ISA 阶段编译失败。变通方法：通过 Mat tile 走 matmul 风格的 load；或在 InCore 中显式插入 `tile.transpose`；或在 Orchestration 层物化一份连续的转置拷贝。
