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

Orchestration 层的 `tensor.transpose`（见 issue #1209）在 `DeduceTensorTransposeType` 中，会在结果类型上同时记录两条信息：

1. **Layout tag。** 对最后两维互换的标准情形，`layout` 在 `ND` 与 `DN` 之间切换。PTOAS 用该 tag 校验 kernel 边界。
2. **显式物理 strides。** 运行时 `Tensor::transpose` 只交换 `shapes` / `offsets` 等元数据，底层 GM 仍按源 tensor 的行主序排布，所以转置后视图的物理 strides 是输入 strides 在 `(axis1, axis2)` 上 swap 后的结果。`DeduceTensorTransposeType` 用 `MakeIndexMul` 按输入 shape 构造行主序 strides（静态 shape 折叠为 ConstInt；动态 shape 得到符号表达式），再在同样的轴上 swap。非末两维 transpose 也由这条路径覆盖 —— 它们保持 `layout = ND`，完全依赖 strides 描述排布。

Codegen 通过检查 `tensor_view_->stride` 是否为空区分 DN tag 的两类来源：

- **来自本 Pass（`tile.load(transpose=True)`）：** `stride` 为空 —— 沿用旧的"末两维隐式 swap"路径，把 IR shape `[M, N]` 发射为 `[N, M]`，strides `[1, M]`。
- **来自 `tensor.transpose`：** `stride` 非空 —— 跳过隐式 swap，直接按 IR shape 发射，strides 用 IR 上记录的。

上述 codegen 规则让 *非转置* 的下游消费者（例如 `tile.load(..., transpose=False)`）能够通过显式 strides 路径正确读取 `tensor.transpose` 的 DN 结果，不会再叠加一次"末两维 swap"造成的双重转置。

**`tile.load(transpose=True)` 直接消费 `tensor.transpose` 的产物 —— 目前不支持。** 这种特定组合（InCore 参数同时携带显式物理 strides 且 `layout = DN`，正是 `tensor.transpose` 结果的独有签名）会被本 Pass 用 `CHECK` 拒绝，因为两套编码会在 codegen 中叠加成双重转置并发出错误地址。源自 slice 的输入（显式 strides + `layout = ND`，由 `OptimizeOrchTensors` 附加）不受影响，仍走标准的"提升 ND → DN，丢弃 strides"路径，paged_attention 的 matmul B^T 等模式继续正常工作。把显式物理 strides 与本 Pass 仅输出 DN 的约定调和需要单独设计 —— 跟踪 issue #1209 的 follow-up。被拒绝场景的变通方法：在 tile 层通过 `tile.load(transpose=True)` 直接对源张量做转置，而不是先在 orchestration 用 `tensor.transpose`。

**Vec 目标仅在 a5 系列可用。** 跨布局 `TLOAD(VecTile_RowMajor, GlobalTensor<DN>)` 在 a2a3 上仍被 PTOAS 拒绝（静态断言 `TLOAD(VecTile, GlobalTensor) only support ND2ND/DN2DN/NZ2NZ`），且 `tile.load` 仍限制 `transpose=True` 仅在 `target_memory=Mat` 时合法。a5（及 `a5sim` 模拟器）解除了该约束，所以 `pl.transpose` 后接 Vec 目标消费者（如非 matmul `pl.at(level=CORE_GROUP)` 块内的 `pl.slice`）在 a5 上可以正确编译并运行。回归测试见 `tests/st/runtime/test_trans.py::test_transpose_slice_assemble[a5sim]`。在 a2a3 上同样的 DSL 现在能产出正确的 IR/.pto，但会在 kernel C++ 阶段失败；变通方法保持不变 —— 通过 Mat tile 走 matmul 风格 load、在 InCore 中显式 `tile.transpose`、或在 Orchestration 层物化一份连续的转置拷贝。
