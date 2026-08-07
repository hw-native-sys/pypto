# AutoTile

`AutoTile` 将一个显式标记的 tensor 函数转换为一个完整的 Ascend 910B
Vector kernel。用户固定运算图；该 pass 选择核网格、tile 或流式形状、归约阶段、
片上生命周期以及存储方式。

它与图融合（graph fusion）有意区分。`AutoTile` 不会拆分被标记的函数，也不会回退为
多个 kernel。一个被标记的函数要么存在一个精确且容量安全的调度，要么编译失败。
未标记的函数保持不变。

## 位置与 API

默认策略在 [`FlattenCallExpr`](06-flatten_call_expr.md) 之后、层次或 InCore
outline 之前运行 `AutoTile`。此时 call 已经是三地址形式，同时完整 tensor DAG
仍然可见。

用 `auto_tile` 标记函数：

```python
import pypto.language as pl


@pl.program
class Program:
    @pl.function(attrs={"auto_tile": True})
    def kernel(
        self, x: pl.Tensor[[128, 8192], pl.FP32]
    ) -> pl.Tensor[[128, 8192], pl.FP32]:
        shifted: pl.Tensor[[128, 8192], pl.FP32] = pl.add(x, 1.0)
        out: pl.Tensor[[128, 8192], pl.FP32] = pl.mul(shifted, 2.0)
        return out
```

自定义 pipeline 也可以直接调用该 pass。直接调用前必须执行与默认 strategy 相同的
tensor-level 准备前缀：

```python
from pypto import passes

prepared = passes.convert_to_ssa()(program)
prepared = passes.simplify()(prepared)
prepared = passes.normalize_stmt_structure()(prepared)
prepared = passes.flatten_call_expr()(prepared)
result = passes.auto_tile()(prepared)
```

标记缺失或为 false 时该 pass 是 no-op。成功转换会移除标记，因此再次运行也为
no-op。

## 准入契约

初始实现支持以下封闭范围：

- 显式配置 Ascend 910B 后端，并在 scope outline 前标记 tensor-level
  `FunctionType::Opaque` function；
- 具有一个顶层 return、正数静态 rank-2 shape 的直线型、拓扑有序 SSA tensor DAG；
- FP32、FP16 和 BF16 Vector 计算；
- 作为终端且不再被消费的 FP32-to-INT8 cast；
- 逐元素算术、scalar 形式、`part_*`、行/列 broadcast、`exp`、`log`、
  `abs`、`sqrt`、`rsqrt`、`recip` 和 `fmod`；
- `row_sum`、`row_max`、`col_sum` 和 `col_max`；
- 一个归约及其逐元素 producer 或 consumer DAG；
- 规范的五运算 row softmax 图；
- 多个 pointwise 返回值，以及容量允许时物化的 reduction live-out；
- 能由一个非原子 kernel 调度容纳的行、列归约。

统一二元操作在发射前会规范化为显式 row-expand 或 col-expand 操作。含 `[1,1]`
tensor 的歧义 broadcast，以及非交换减法或除法中位于左侧的 broadcast 操作数会被
拒绝。broadcast 除法支持 FP16 和 FP32；其高精度形式不在本契约中。

该 pass 会拒绝动态或非 rank-2 shape、控制流、有副作用的语句、`full`/shape 构造、
minimum/product/argument reduction、matmul 和其他 Cube 工作、mixed kernel、Welford、
不支持的 dtype，以及任何需要多个 kernel 的图。这些是面向用户的准入错误，不是转交
给另一个 planner 的请求。

## 规划模型

planner 首先构造带类型的 Vector 图。Tensor 节点记录静态 shape、dtype、是否为边界
值以及是否必须存活到 return。Operation 节点记录 primitive 与几何类型。原始 SSA
语句顺序就是拓扑顺序；该 pass 不重排用户操作。

随后枚举平衡的二维核网格。每个发射任务使用一个静态最大尺寸 tile。ragged partition
会 clamp 或重叠最后一个 tile，因此重复的边缘工作必须是幂等的，并计入流量模型。
候选任务数围绕硬件的 48 个 Vector core wave 形状选择。

对每个候选，planner 构造显式 `VectorSchedulePlan`，其中包含：

- 行列 partition 和精确 work-unit 数；
- 完整 region 与流式 strip/chunk extent；
- phase 运算列表和边界输入 first/last-use；
- pipeline trip 数与深度；
- 逻辑及 DMA padding 后的 reduction extent；
- 完整及实际发射的 UB 峰值，以及计算/传输周期。

emitter 在生成 IR 前根据源图验证 descriptor。它不会重新推导 tiling、lifetime、split
或 phase。这个单向 plan-to-emission 契约很重要：只有当发射算法执行了 plan 计价的工作
并占有其计价的内存时，cost 才有意义。

在 `INFO` 日志级别，每次成功改写都会输出一行 `AutoTile[name]`，其中包含选中的
schedule、grid、work unit、tile/strip/chunk extent、pipeline depth、UB 峰值、各 phase
流量、modeled cycle，以及 reduction 估计来自实测表还是 fallback。Ascend 910B 系数从
经 silicon 验证的 scheduler model 原样移植而来，没有重新拟合；AutoTile 改变的是规划
与发射的归属，而不是这些测量数据。

## 调度族

### Materialized

完整的每核 region 可放入 UB。每个 phase 的边界 tensor 只 slice 一次，并复用到最后
一次拓扑使用；每个返回值都存活到各自独立的 `tensor.assemble` store。

### Pointwise stream

过大的 pointwise region 沿一个轴切成 strip。发射的 strip loop 是两阶段
`ForKind::Pipeline`；经过 [`LowerPipelineLoops`](29-lower_pipeline_loops.md) 后，一个
strip 的 load/store 可与另一个 strip 的 Vector 工作重叠。所有返回值都由 loop carry
并写回，不引入中间 GM tensor。经过 DMA 对齐的物理 tile 会在每个生成运算后保留精确
的逻辑 valid shape，因此 ragged store 不会写入 padding 的行或列。

### Folded 与 spanning reduction

stats phase 对固定大小 chunk 做归约并携带细长 accumulator。folded 调度在归约后仅运行
一次剩余的细长操作；spanning 调度则对宽输入进行第二次 chunk pass，并应用归约统计量。
完整 chunk 可使用两阶段 pipeline；初始化与 ragged tail 串行执行。

### Online softmax

规范 softmax 调度在 chunk 间携带 running maximum 和修正后的 running sum，然后执行
一次 chunk 化输出 pass。无需在 GM 中物化 exponential 即可保持数值稳定。

### 列归约

列归约沿被归约的行轴流式处理 chunk，同时携带一个细列 accumulator。若 consumer
需要原始宽 tensor，则使用第二个流式 apply phase。AutoTile 不会发射 seed kernel 或
原子 partial store；若列归约图不能由一个容量安全的 kernel 实现，则拒绝整个 marked
function。

## UB 与传输计价

UB 规划感知 dtype 和 lifetime。它计入边界 load、中间值、所有返回 live-out、DMA
padding、两阶段 pipeline 的第二个 bank、tensor-to-tile lowering 插入并按最小宽度
padding 的行 reduction scratch、高精度 `rsqrt` scratch 和细 accumulator。仅修改元数据
的 `set_validshape` alias 不会分配第二个 buffer。列 reduction lowering 不分配 scratch，
模型也不会为它计费。

cost model 组合：

1. primitive 周期估计，以及在实际发射 extent 上插值的 Ascend 910B FP32/FP16 行、列
   归约实测表；
2. online softmax 初始化及更新所生成的精确 primitive 序列；
3. 各 phase 的逻辑 GM-to-UB 输入与 UB-to-GM 输出流量之和；
4. 只有实际发射两阶段 pipeline 的 phase 才使用 `max(compute, transfer)`，否则串行相加；
5. task 与 wave fill 项。

planner 会评估每个容量安全的 reduction chunk，并选择 modeled cost 最小者；不会假定
最大可容纳 chunk 最快。没有实测归约表的 dtype 使用显式保守 fallback，并在选中 plan
日志中报告，而不会伪装成 grounded 数据。

模型有意保留保守的双向 GM 流量和。它不会假设 MTE2/MTE3 相互独立重叠，不会拟合
新的带宽系数，也不会推断 IR 中不存在的隐式 pipeline。

## 输出与调用

对于入口函数，返回 tensor 会变成显式 `Out` 参数，同时保留原始 return tuple 供后续
规范化。对于程序内被调用的 marked helper，输出存储在 helper 内创建，以保持调用签名
有效。多个 live-out 始终拥有不同 store。已有的显式 `Out` 参数按位置复用；直接调用与
`Submit` site 都保持该声明签名。

成功发射包含一个 `pl.spmd` scope 和一个非 split 的 Vector InCore body。因此后续层次
outline 会为该 marked tensor DAG 生成一个 AIV kernel。

## 与其他 tiler 的关系

本 pass 负责从 GM 到 UB 的 tensor 级 Vector 调度。它不替代
[`AutoTileMatmulL0`](16-auto_tile_matmul_l0.md)；后者在更晚阶段处理单个 Cube matmul
的 L0 几何。Cube、mixed-kernel 和图拆分支持有意排除在首个 AutoTile 契约之外。
