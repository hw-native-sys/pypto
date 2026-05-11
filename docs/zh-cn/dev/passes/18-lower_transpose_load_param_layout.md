# LowerTransposeLoadParamLayout Pass

将 `tile.load(..., transpose=True)` 下沉为 canonical 形式的 DN 参数布局（RFC #1300 P6）。

## 概述

本 Pass 之前，`tile.load(transpose=True)` 是用户表达"我希望在 load 站点看到源张量的列主序视图"的方式。Pass 之后，这一意图被编码进 InCore 参数的 TensorType 本身 —— 源张量/load 组合被改写为 RFC #1300 §3.3 的 canonical 形式，使 codegen、verifier、下游 Pass 看到一份自洽的 `(shape, stride, layout)` 三元组。

对每个被 `tile.load(p, ..., transpose=True)` 加载的 InCore 参数 `p`：

- `p` 的 TensorType 从 `[..., a, b] ND` 提升为 `[..., b, a] DN` —— 末两维形状互换 + DN 布局标签。新 TensorView 的 stride 为空；`MaterializeTensorStrides`（在默认 pipeline 中位于 `CanonicalizeIOOrder` 之后运行）会把它填为 packed canonical 的 stride。
- 每个 `tile.load(p, offsets, shapes, valid_shapes, ..., transpose=True)`（源是已提升的参数）被改写为：三个 tuple 的末两维互换以匹配 canonical 坐标，丢弃 `transpose=True` kwarg。`DeduceTileLoadType` 通过源张量的 DN 布局推导出 Mat tile-view 的 layout —— 这两种信号在 §4.2 canonical pair 下是等价的。
- 每个目标是已提升 callee 的非 InCore 函数调用站点，会把对应实参用 `tensor.as_layout(arg, DN)` 包一层（RFC #1300 P4）。该桥接 op 是纯元数据 —— 不生成 PTOAS 指令；`make_tensor_view` 直接消费新视图。

**前置条件**：

- 输入 IR 必须为 SSA 形式
- InCore 函数已完成拆分（`SplitIncoreOrch`）
- Tile op 已存在且为 2D（`IncoreTileOps`、`TileOps2D`）
- 被提升的参数 rank ≥ 2

**使用时机**：在 `Default` 策略中作为第 18 个 Pass 运行（文档编号 18 对应于 docs/passes/ 中的执行顺序槽位，与 pass_manager.py 中的相对顺序匹配），位于 `InferTileMemorySpace` 之后、`ResolveBackendOpLayouts` 之前。`FlattenTileNdTo2D` 产生的 2D 形状是前置条件。`MaterializeTensorStrides` 在 pipeline 后段运行（在 `CanonicalizeIOOrder` 之后）以物化 DN-packed canonical stride。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::LowerTransposeLoadParamLayout()` | `passes.lower_transpose_load_param_layout()` | Program 级 |

**Python 用法**：

```python
from pypto.pypto_core import passes

p = passes.lower_transpose_load_param_layout()
program_canonical = p(program)
```

## 算法

```text
对每个 InCore 函数 f：
  扫描 body → 得到 P_t  = {tile.load(p, ..., transpose=True) 命中的 param 索引}
              得到 P_nt = {tile.load(p, ..., transpose=False/缺省) 命中的 param 索引}
  拒绝 P_t ∩ P_nt  （混用）
  对每个 idx in P_t：
    提升 f.params[idx].type：[..., a, b] ND → [..., b, a] DN（stride 留空）
    在 body 中以新 Var 替换旧 Var
  改写 body 中每个 tile.load(promoted_param, off, shp, vs, transpose=True)：
    交换 off / shp / vs 末两维
    丢弃 transpose=True kwarg

对每个非 InCore 函数：
  遍历 body；对每个 op 为已提升 callee 的 GlobalVar 的 Call：
    给每个已提升槽位的实参包一层 tensor.as_layout(arg, DN)
```

**复杂度：** O(N log N) —— 每个函数一次 body 走查，加一次全程序级调用站点走查。Map 查找（`promotions_by_callee_name`）为每次调用 `log N`。

| 行为 | 触发条件 |
| ---- | -------- |
| 提升参数到 `[..., b, a] DN` | InCore 参数是 `tile.load(..., transpose=True)` 的源 |
| 跳过参数 | 已经是 DN，或没有转置 load |
| 整个函数跳过 | 函数为 Orchestration / Opaque / Group |
| 调用站点 wrap `tensor.as_layout` | 非 InCore 函数调用已提升 callee |
| 拒绝 | 同一参数既被 transpose=True 也被 transpose=False 加载 |
| 拒绝 | DN + 显式物理 stride 源（与 tile.load 转置会叠成双重转置） |

## 示例

**前**：

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

**后**（语义层面 —— `tensor.as_layout` 是内部 IR op，不在 pl.* 暴露）：

```text
@pl.function(type=pl.FunctionType.InCore)
def matmul_incore(
    self,
    a: pl.Tensor[[64, 128], pl.FP32],
    b: pl.Tensor[[128, 32], pl.FP32, pl.DN],   # ← 形状互换 + DN 标签
    c: pl.Out[pl.Tensor[[64, 32], pl.FP32]],
) -> pl.Tensor[[64, 32], pl.FP32]:
    tile_a = pl.load(a, [0, 0], [64, 128], target_memory=pl.MemorySpace.Mat)
    tile_b = pl.load(b, [0, 0], [128, 32], target_memory=pl.MemorySpace.Mat)
                                           # ↑ 没有 transpose kwarg
                                           # ↑ shapes 已互换到 canonical 坐标
    ...

@pl.function(type=pl.FunctionType.Orchestration)
def orchestrator(self, a, b):
    c = pl.create_tensor([64, 32], dtype=pl.FP32)
    # b 在调用站点被 tensor.as_layout 包一层做 ND → DN 桥接：
    bridged_b = tensor.as_layout(b, pl.DN)  # type: [128, 32] DN
    return self.matmul_incore(a, bridged_b, c)
```

`a` 不转置加载，原样保留。`b` 在 InCore 签名被提升，body 中所有对 `b` 的加载改写到 canonical 坐标且无转置 kwarg，orchestrator 调用站点把 `b` 用 `tensor.as_layout` 包起来，把 `[32, 128] ND` 桥接到 `[128, 32] DN`（同一片物理内存）。

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`

**实现**：`src/ir/transforms/lower_transpose_load_param_layout_pass.cpp`

**Python 绑定**：`python/bindings/modules/passes.cpp`

**测试**：`tests/ut/ir/transforms/test_lower_transpose_load_param_layout_pass.py`

## Pass 属性

| 属性 | 值 |
| ---- | -- |
| 必需 | SSAForm, IncoreTileOps, SplitIncoreOrch, TileOps2D |
| 产出 | SSAForm, IncoreTileOps, SplitIncoreOrch, TileOps2D |
| 失效 | — |

## 范围

| 函数类型 | 行为 |
| -------- | ---- |
| InCore（InCore、AIC、AIV） | 扫描，可能被提升 |
| Orchestration / Group / Opaque | 扫描调用站点；已提升实参 wrap `tensor.as_layout` |

| 参数状态 | 行为 |
| -------- | ---- |
| 是 `tile.load(..., transpose=True)` 的源，layout != DN，rank ≥ 2 | 提升（形状互换 + DN 标签） |
| 是 `tile.load(..., transpose=True)` 的源，已是 DN | 幂等 —— 保持不变 |
| 同一参数既 transpose=True 又 transpose=False | `CHECK` 失败 |
| 没有转置 load 引用 | 保持不变 |
| Rank < 2 候选 | `CHECK` 失败 |

## 与 `tensor.as_layout`（P4）和 `MaterializeTensorStrides`（P3）的交互

本 Pass 是默认 pipeline 中 `tensor.as_layout` 的第一个真实消费者。该桥接 op 单一职责：翻转 layout 标签，目标 shape 由 §4.2 canonical pair 机械导出 —— 调用方不传 target shape，所以调用站点改写器不会出错。

下游的 `MaterializeTensorStrides` 把每个被提升的参数 TensorView 空 stride 填为 packed canonical DN strides（RFC §2.4）。P6 + P3 的组合让 codegen 看到自洽的 `(shape, stride, layout)` 三元组 —— 对被提升的参数，codegen 路径无需再做 `dn_swap` / `get_shape_source_idx` 修正。

## 与 Orchestration 层 `tensor.transpose` 的交互

源 TensorView 同时携带 `layout = DN` 和非空 `stride` 的参数是 `tensor.transpose` 结果的特征。本 Pass 对这类参数上的 `tile.load(transpose=True)` 直接拒绝（`CHECK` 失败）—— 否则两层转置编码会在 codegen 时叠成双重转置、地址错误。Slice 派生的入参（显式 stride + `layout = ND`，由 `OptimizeOrchTensors` 附加）不受影响。

被拒绝场景的绕过：在源程序中去掉两层转置中的一层。
