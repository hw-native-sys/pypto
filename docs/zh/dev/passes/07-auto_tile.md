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
- FP32 和 FP16 Vector 计算；
- BF16 tensor 存储和原生 cast 链端点；
- 作为终端且不再被消费的 FP32-to-INT8 cast；
- 逐元素算术、scalar 形式、同 shape 的 `part_*`、行/列 broadcast、`exp`、
  `log`、`abs`、`sqrt`、`rsqrt`、`recip`、FP32 同 shape `fmod` 和 FP32
  `fmods`；
- `row_sum`、`row_max`、`col_sum` 和 `col_max`；
- 一个归约及其逐元素 producer 或 consumer DAG；
- 当每个 core 的完整 DAG 可放入 materialized 调度时，支持多个同轴归约；
- 规范的五运算 row softmax 图；
- 多个 pointwise 返回值，以及容量允许时物化的 reduction live-out；
- 能由一个非原子 kernel 调度容纳的行、列归约。

统一二元操作在发射前会规范化为显式 row-expand 或 col-expand 操作。含 `[1,1]`
tensor 的歧义 broadcast，以及非交换减法或除法中位于左侧的 broadcast 操作数会被
拒绝。broadcast 除法支持 FP16 和 FP32；其高精度形式不在本契约中。

Ascend 910B A2/A3 的 `TFMOD` 指令仅接受 FP32 且 shape 相同的两个操作数，
`TFMODS` 也仅接受 FP32。AutoTile 会在准入阶段拒绝 FP16 fmod 和 tensor-fmod
broadcast。`part_*` 指令族同样没有 row/column-expand 形式，因此两个 tensor
操作数必须具有相同 shape。

经过验证的 Ascend 910B A2/A3 AutoTile 算术范围是 FP16/FP32；PTOAS 会在该目标上拒绝
直接 BF16 `TADD`。因此 AutoTile 会在准入阶段、规划或 PTOAS 编译之前拒绝 BF16 算术。
BF16 仍可作为存储 tensor，以及原生 cast 链的源或目标。AutoTile 不会隐式地把 BF16
算术提升为 FP32，因为那会成为另一种算法，并引入额外的 UB 存储、传输和 modeled cost。

原生 cast 链可以消费边界值或 full-frame 逐元素值。若 cast 直接以归约结果为起点，pass
会拒绝该图：归约发射拥有单独 padding 的结果 box，而当前 emitter 尚不能把它扩展到 cast
链要求的公共物理粒度。先把归约结果 apply 或 broadcast 回完整迭代 frame，再执行 cast，
仍属于支持范围。

该 pass 会拒绝动态或非 rank-2 shape、控制流、有副作用的语句、`full`/shape 构造、
minimum/product/argument reduction、matmul 和其他 Cube 工作、mixed kernel、Welford、
不支持的 dtype，以及任何需要多个 kernel 的图。这些是面向用户的准入错误，不是转交
给另一个 planner 的请求。

## 规划模型

planner 首先构造带类型的 Vector 图。Tensor 节点记录静态 shape、dtype、是否为边界
值以及是否必须存活到 return。Operation 节点记录 primitive 与几何类型。原始 SSA
语句顺序就是拓扑顺序；该 pass 不重排用户操作。

通过逐元素运算连接且 shape 相同的值组成一个物理 shape 类。对于原生 cast 链，planner
会对链内各 dtype 的 DMA 元素粒度取最小公倍数，并把同一个“元素个数”粒度赋给整个类，
从而保证每个原生 `TCVT` hop 的物理 shape 完全一致。这**不**意味着所有值都按最宽 dtype
分配：每个 SSA 结果仍按 `physical_elements * sizeof(自身 dtype)` 独立计价并分配。emitter
把原始逻辑 extent 保存在 `valid_shape` 中，因此 padding 不会改变程序的逻辑边界。

随后枚举平衡的二维核网格。每个发射任务使用一个静态最大尺寸 tile。ragged partition
会 clamp 或重叠最后一个 tile，因此重复的边缘工作必须是幂等的，并计入流量模型。
候选任务数围绕硬件的 48 个 Vector core wave 形状选择。

对每个候选，planner 构造显式 `VectorSchedulePlan`，其中包含：

- 行列 partition 和精确 work-unit 数；
- 完整 region 与流式 strip/chunk extent；
- phase 运算列表和边界输入 first/last-use；
- 对 online-softmax update 这类由 planner 合成、而不是直接重放源 op 的 phase，记录显式
  generated-algorithm 标记；
- pipeline trip 数与深度；
- 逻辑及 DMA padding 后的 reduction extent；
- 完整及实际发射的 UB 峰值，以及计算/传输周期。

emitter 在生成 IR 前根据源图验证 descriptor。它不会重新推导 tiling、lifetime、split
或 phase。这个单向 plan-to-emission 契约很重要：只有当发射算法执行了 plan 计价的工作
并占有其计价的内存时，cost 才有意义。

在 `INFO` 日志级别，每次成功改写都会输出一行 `AutoTile[name]`，其中包含选中的
schedule、grid、work unit、tile/strip/chunk extent、pipeline depth、UB 峰值、各 phase
流量、modeled cycle、reduction 估计来自实测表还是 fallback，以及 pointwise 估计是否使用
generic 或 cast proxy。Ascend 910B 系数从
经 silicon 验证的 scheduler model 原样移植而来，没有重新拟合；AutoTile 改变的是规划
与发射的归属，而不是这些测量数据。

## 调度报告

每个成功编译 AutoTile function 的 `ir.compile()` 还会写出两个小型、确定性的 artifact：

```text
<output_dir>/report/auto_tile/<function>.json
<output_dir>/report/auto_tile/<function>.txt
```

JSON 文件是带版本号的 compiler-artifact schema。它记录选中的 grid、平衡 partition、
代表性 region、strip/chunk loop、串行 tail、phase 运算顺序、边界输入 lifetime、逻辑与
物理 tile extent、各 dtype 的 element size、UB 峰值、流量和 modeled cycle。该格式面向
工具使用，目前尚不是稳定的公共 API。

文本文件把同一个结构化 descriptor 渲染成以 tile 为中心的伪代码。它描述一个代表性的
SPMD work unit，而不是绘制所有 core。例如 online-softmax 报告会分别显示串行首 chunk、
两阶段 statistics loop、存在时的串行 ragged tail，以及两阶段 apply/store loop；ping/pong
slot 和持久 running statistic 都会显式显示。`lifetime ends: x(t0)` 这样的行表示逻辑上的
最后使用点，并不意味着 IR 中存在显式 free 指令。

直接调用 `passes.auto_tile()` 时没有编译 artifact 目录，因此仍只输出简洁的 `INFO`
日志。调度报告可与 [IR Lowering Trace](../07-ir-lower-trace.md) 配合检查 transformation，
并与 [Memory Map](../07-memory-map.md) 配合检查最终 UB 地址和物理复用。

## 调度族

### Materialized

完整的每核 region 可放入 UB。每个 phase 的边界 tensor 只 slice 一次，并复用到最后
一次拓扑使用；每个返回值都存活到各自独立的 `tensor.assemble` store。

包含多个归约的一般 DAG（例如 LayerNorm 先计算均值、再计算方差）只支持 materialized
调度。这些归约按源码拓扑顺序执行，普通 lifetime model 会计入中间所有 full-frame 与
thin 值。如果不存在可让完整 live set 放入 UB 的空间 partition，AutoTile 会拒绝该函数，
而不会把单归约 streaming 调度错误地套用到无法表达的依赖链上。

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

### Softmax

规范 softmax 调度在 chunk 间携带 running maximum 和修正后的 running sum，然后执行
一次 chunk 化输出 pass。无需在 GM 中物化 exponential 即可保持数值稳定。

若每个 work unit 的完整 softmax live set 可放入 UB，planner 还会枚举一个单 pass
materialized 候选。该候选只重放一次源 DAG，并将 exponential 与两个 reduction
结果保留在片上直到最终 divide。普通 lifetime model 会计入其完整 UB footprint、
一次输入读取、一次输出写回，以及每个源操作的一次执行。online 候选仍以
statistics 和 apply 两个 pass 独立计价。两者中 modeled cost 较低者胜出；可放入
UB 只是 materialization 的可行性条件，而不是无条件偏好。完整 live set 无法放入的
宽行仍使用 online schedule。

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

普通 cast 是源、目标存储彼此独立的转换；MemoryReuse 不得把 `tile.cast` 变成原地操作。
显式请求等字节数的 `tile.reinterpret_view` 仍是零拷贝 opt-in，并通过 bitcast/view 路径
lower，而不是使用 `TCVT`。

cost model 组合：

1. primitive 周期估计，以及在实际发射 extent 上插值的 Ascend 910B FP32/FP16 行、列
   归约实测表；
2. online softmax 初始化及更新所生成的精确 primitive 序列；
3. 各 phase 的逻辑 GM-to-UB 输入与 UB-to-GM 输出流量之和；
4. 只有实际发射两阶段 pipeline 的 phase 才使用 `max(compute, transfer)`，否则串行相加；
5. task 与 wave fill 项。

planner 会评估有界 reduction 搜索中的每个受支持候选，并选择 modeled cost 最小者；不会
假定最大可容纳 chunk 最快。搜索包含完整 reduction extent，以及不超过 4096 元素且按
16 元素对齐、同时满足 UB 容量的 chunk。当前准入的两种计算 dtype 都使用 grounded
reduction table。实现仍为未来 backend 扩展保留显式保守 fallback，而不会把未 grounded
的估计伪装成实测数据。

大多数 pointwise primitive 使用移植的 910B grounding。被归类为 generic 的运算和原生
cast hop 使用显式保守 proxy 系数：generic proxy 不是针对具体运算的测量值，cast proxy
也不区分具体源/目标 dtype 对。plan 日志和 schedule report 通过 `pointwise_model` 暴露
该来源。

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
