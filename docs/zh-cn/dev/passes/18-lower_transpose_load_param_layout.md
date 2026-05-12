# LowerTransposeLoadParamLayout Pass

将 `tile.load(..., transpose=True)` 下沉为 InCore body 内显式的 `tensor.as_layout` 视图（RFC #1300 P6）。

## 概述

本 Pass 之前，`tile.load(transpose=True)` 是用户表达"我希望在 load 站点看到源张量的列主序视图"的方式。Pass 之后，这一意图被编码进 InCore body 顶部的一条 `tensor.as_layout` 视图绑定 —— 使 codegen、verifier、下游 Pass 看到一份自洽的 `(shape, stride, layout)` 三元组。

对每个被 `tile.load(p, ..., transpose=True)` 加载的 InCore 参数 `p`：

- **在 InCore body 顶部插入** `p_dn = tensor.as_layout(p, layout=DN)`。新 Var `p_dn` 携带 canonical `[..., b, a] DN` 视图（末两维 shape 互换 + DN layout 标签 + `tensor.as_layout` 的 deduce-type 填入的 packed canonical strides）。
- body 中对 `p` 的引用被替换为 `p_dn`。`p` 的参数签名保持不变 —— orch 侧继续按原 row-major ND 形式传 tensor（与 runtime 的 torch tensor 一致）。
- body 中每个 `tile.load(p, offsets, shapes, valid_shapes, ..., transpose=True)`（源是已提升参数）被改写为 `tile.load(p_dn, ...)`，三个 tuple 的末两维互换为 canonical 坐标，`transpose=True` 翻为 `transpose=False`。`DeduceTileLoadType` 通过 `p_dn` 的 DN 布局推出 Mat tile-view 的 layout —— 两种信号在 §4.2 canonical pair 下等价。

非 InCore（orch）函数完全不动。DN 重解释是单函数（InCore）内部的关注点，由用到它的 body 自己拥有；跨函数边界保持简单：orch 永远传 row-major ND tensor。

**前置条件**：

- 输入 IR 必须为 SSA 形式
- InCore 函数已完成拆分（`SplitIncoreOrch`）
- Tile op 已存在且为 2D（`IncoreTileOps`、`TileOps2D`）
- 被提升的参数 rank ≥ 2

**使用时机**：在 `Default` 策略中作为第 18 个 Pass 运行（文档编号 18 对应于 docs/passes/ 中的执行顺序槽位，与 pass_manager.py 中的相对顺序匹配），位于 `InferTileMemorySpace` 之后、`ResolveBackendOpLayouts` 之前。`FlattenTileNdTo2D` 产生的 2D 形状是前置条件。

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
    let p = f.params[idx]
    若 p 已经是 DN（用户写的 / 预先 canonical 化的情形）则跳过
    构造 p_dn := tensor.as_layout(p, layout=DN)  —— 类型由 op deduce 推出
    将 (p_dn = ...) AssignStmt 插入 body 顶部
    记录 p → p_dn 的替换映射
  按映射替换 body 中所有对已提升 p 的引用为 p_dn
  改写 body 中每个 tile.load(p_dn, off, shp, vs, transpose=True)：
    交换 off / shp / vs 末两维
    丢弃 transpose=True kwarg

（非 InCore 函数原样保留）
```

**复杂度：** O(N log N) —— 每个 InCore 函数一次 body 走查。

| 行为 | 触发条件 |
| ---- | -------- |
| 插入 `p_dn = tensor.as_layout(p, DN)` 并改写 tile.load | InCore 参数是 `tile.load(..., transpose=True)` 的源 |
| 跳过参数 | 已经是 DN，或没有转置 load |
| 整个函数跳过 | 函数为 Orchestration / Opaque / Group |
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
    b: pl.Tensor[[32, 128], pl.FP32],             # ← 参数签名保持不变
    c: pl.Out[pl.Tensor[[64, 32], pl.FP32]],
) -> pl.Tensor[[64, 32], pl.FP32]:
    b_dn = tensor.as_layout(b, layout=DN)          # ← body 顶部插入的视图
                                                    #   类型：[128, 32] DN
    tile_a = pl.load(a, [0, 0], [64, 128], target_memory=pl.MemorySpace.Mat)
    tile_b = pl.load(b_dn, [0, 0], [128, 32], target_memory=pl.MemorySpace.Mat)
                                                    # ↑ 源切到 b_dn
                                                    # ↑ shapes 已互换到 canonical 坐标
                                                    # ↑ 无 transpose kwarg
    ...

@pl.function(type=pl.FunctionType.Orchestration)
def orchestrator(self, a, b):
    c = pl.create_tensor([64, 32], dtype=pl.FP32)
    return self.matmul_incore(a, b, c)             # ← 保持不变
```

`a` 不转置加载，原样保留。`b` 的参数签名保留；kernel 内部用 `tensor.as_layout` 派生 DN 视图，其 `tile.load` 引用该视图。orchestrator 完全不动 —— 它把自己的 row-major `b` 原样传下去。

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
| InCore（InCore、AIC、AIV） | 扫描，按需在 body 顶部插入 `tensor.as_layout` 视图 |
| Orchestration / Group / Opaque | 不动 |

| 参数状态 | 行为 |
| -------- | ---- |
| 是 `tile.load(..., transpose=True)` 的源，layout != DN，rank ≥ 2 | 插入 `tensor.as_layout` 视图；body 中引用被替换 |
| 是 `tile.load(..., transpose=True)` 的源，已是 DN | 跳过 —— `DeduceTileLoadType` 已经处理 DN-源 XOR transpose |
| 同一参数既 transpose=True 又 transpose=False | `CHECK` 失败 |
| 没有转置 load 引用 | 保持不变 |
| Rank < 2 候选 | `CHECK` 失败 |

## 与 `tensor.as_layout`（P4）的交互

本 Pass 是默认 pipeline 中 `tensor.as_layout` 的第一个消费者。该桥接 op 单一职责：翻转 layout 标签，目标 shape 由 §4.2 canonical pair 机械导出，并通过 `CanonicalizeView` 附加 packed canonical strides。codegen 把 `tensor.as_layout` 下沉为一条新的 `pto.make_tensor_view`，绑定到输入 tensor 的底层 SSA buffer 上，使用 LHS 的 `(shape, stride, layout)` 三元组 —— 不发射任何 PTOAS 指令，结果是纯元数据 reinterpret。

按 RFC §4.2，InCore 侧的 reinterpret 不违反"核内不能创建 tensor"约束：`tensor.as_layout` 不分配任何内存，它只是为输入的现有物理 buffer 换一份描述。

## 与 Orchestration 层 `tensor.transpose` 的交互

源 TensorView 同时携带 `layout = DN` 和非空 `stride` 的参数是 `tensor.transpose` 结果的特征。本 Pass 对这类参数上的 `tile.load(transpose=True)` 直接拒绝（`CHECK` 失败）—— 否则两层转置编码会在 codegen 时叠成双重转置、地址错误。Slice 派生的入参（显式 stride + `layout = ND`，由 `OptimizeOrchTensors` 附加）不受影响。

被拒绝场景的绕过：在源程序中去掉两层转置中的一层。
