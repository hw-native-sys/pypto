# LegalizeTileCast Pass

把目标 profile（A5 / A2A3）上 `pto.tcvt` **不支持** 的 `tile.cast` `(src, dst)` 对，展开为最短的原生 cast 链，避免静默发出非法 `tcvt`。

## 概述

`LegalizeTileCast` 是函数级 Pass。对每条 `var = tile.cast(...)`：

1. 通过 `GetTcvtAdjacency()` 向当前 `BackendHandler` 取原生转换规则表（转录自 pto-isa `tcvt`
   Supported Conversions）。本 Pass 自身不含任何架构知识，新增后端只需提供自己的表，
   无需改动此处；未配置 backend 时本 Pass 不做任何事。
2. 调用 `backend::IsDirectCast(call, handler.GetTcvtAdjacency())`，检查当前类型、逻辑尺寸及规则中的实现条件；支持则原样保留（含可被 `AutoTileMatmulL0` FIXPIPE-fold 的 `FP32→BF16/FP16` + `rint`）。
3. 否则在邻接图上 BFS 求最短路径。每个候选 hop 均推导自己的类型并调用同一个 `IsDirectCast`，排除几何不合法的边；等长路径优先「同字节转浮点 → 再调宽度」。

典型结果（A5）：

| 用户 Cast | 分解 |
| --------- | ---- |
| INT32→FP16 | INT32→FP32 → FP32→FP16 |
| FP16→BF16 | FP16→FP32 → FP32→BF16 |

搜不到路径则硬失败（带 src/dst/arch）。

**Requires / Produces / Invalidates**：无（空 `PassProperties`）。

## 目标类型饱和模式

`pl.cast`、`pl.tensor.cast` 与 `pl.tile.cast` 接受仅关键字参数 `saturation_mode`，
可写作 `"on"` / `"off"` 或 `1` / `0`：

```python
quantized = pl.cast(rounded_fp16, pl.INT8, mode="trunc", saturation_mode="on")
```

`"on"` 在舍入后把超出目标值域的结果钳制到该值域；`"off"` 使用目标平台的非饱和
转换，其溢出与非有限值行为由架构定义。`Scalar` 输入拒绝该参数。

**目标类型为整数时，默认值是 `"on"`。** 只有这种情况下两种模式才是真正的选择：
没有任何标准规定"转换到整数时溢出"应当产生什么；在两者之间，"意外得到钳制"比
"意外得到回绕"更安全；而且在 A2/A3 上，钳制正是汇编器原生支持的转换，非饱和形式
则要用一段分块向量序列模拟——因此默认值同时也是更快的降级路径。仅当所选目标平台
*已记录的*非饱和行为正是 kernel 所需时，才传入 `"off"`——它**不是**回绕的保证，
溢出时的具体行为由架构定义。

**目标类型为浮点时，保持目标平台自身的行为**，除非作者显式指定。这个问题已经有答案：
IEEE 规定窄化溢出产生无穷，`torch` 与之一致，而
[精度排查流程](../../user/precision/00-workflow.md) 断言 PyPTO 在 `INT32 -> FP16`
上与它们逐位相同。把这类转换也默认成 `"on"` 会让该代码块在 a2a3 模拟器上失败——
65520 被钳制到 65504，而不是溢出为 `inf`——因此默认值刻意只覆盖整数目标类型。

IR 只记录相对适用默认值的*偏离*：想要该默认值的 cast 不携带 `saturation_mode`
kwarg，这与 Pass 合成的 cast 形状一致。正是这一点让打印出的 cast 能重新解析成结构
相同的 IR——若把默认值也写进去，两种语义完全相同的形式反而会不相等。
`include/pypto/ir/cast_saturation.h` 的 `DefaultSaturationModeFor` 与
`python/pypto/ir/utils.py` 的 `default_saturation_mode_for` 是这条规则的两半，必须
保持一致。

**Codegen 始终发射 `satmode`**，整数目标与浮点目标一视同仁，无论 cast 是否指定过。
`GetEmittedSaturationMode` 会把默认值读出来，并把浮点目标那个"交给目标平台"解析成
`OFF`——这正是目标平台的 IEEE 行为。既然该属性从不省略，汇编器对"省略时"的默认值就
无法左右语义——而它确实会变：PTOAS v0.63 把该默认值从 `OFF` 翻成了 `ON`
（`fix(tcvt): default saturation mode to ON`）。若仍沿用"浮点目标不发射"的旧规则，
每个 `INT32 -> FP16` 都会被悄悄变成钳制转换。v0.63 下显式 `satmode = OFF` 汇编出的
结果与 v0.61 下省略该属性逐字节相同。

这个解析刻意放在发射边界，而不是放进 IR 默认值。若把 IR 默认值也放宽成 `OFF`，浮点
目标上显式的 `saturation_mode="off"` 就变成了非偏离，于是不再被记录——而那些询问
"作者是否提出过要求"的 Pass（`CastFoldableToFixpipeMat`）也就看不到它了。A2/A3 上
`INT32` `pto.trem` 之后用于还原 `rhs` 的那条编译器合成 `pto.tcvt` 遵循同一策略。

两种模式仅在目标类型本就能表示的值上一致，所以对整数目标而言，这个默认值是一次行为
选择，而非空操作：依赖目标平台自身非饱和溢出行为的 kernel 现在必须显式写 `"off"`。

**展开链：请求作用于最后一跳。** 饱和描述的是*目标*值域，而只有最后一跳到达目标
dtype；给中间跳打标会钳制到作者从未指定的值域。因此中间跳完全保持原有行为——沿用
原始舍入模式，别无其他。

推迟到最后一跳不会带来损失：上文的 BFS 已经拒绝任何相对目标类型更窄的中间类型，
因此目标类型*能够*表示的值都会原样抵达最后一跳，`"on"` 与 `"off"` 在这些值上依旧
一致。目标类型无法表示的值在链条两端同样越界——中间的浮点类型可能把它溢出为无穷，
但符号保持不变，于是钳制到的端点与假想的单步转换一致。非有限输入仍然不在两种模式
的定义范围内。

同一规则也把显式的 `pl.tile.cast(..., tmp=...)` 临时缓冲区带到最后一跳，即真正做
窄化的那一跳。

**A2/A3 临时缓冲区。** `InitMemRef` 只为*非饱和*的窄化 `pto.tcvt` 生成临时 tile：
PTOAS 对该形式的实现用一段分块向量序列模拟目标平台的溢出行为。饱和转换是原生实现，
不读取临时缓冲区。所有需要该缓冲区的组合目标类型都是整数，因此适用默认值为 `"on"`，
只有显式关闭饱和的 cast 才会分配该 tile。调用方显式提供的 `tmp` 始终保留。

## 目标布局（NZ）

`pl.tile.cast` 与 `pl.cast` 接受可选的仅关键字参数 `blayout` 和 `slayout`，拼写与
[`pl.tile.move`](../ir/05-operators.md) 完全一致：

```python
nz_fp8 = pl.cast(
    packed_fp4,
    pl.FP8E4M3FN,
    blayout=pl.TileLayout.col_major,
    slayout=pl.TileLayout.row_major,
)
```

**两者都省略时继承源 tile 的布局**，即这些属性存在之前每个 cast 的行为；调用上不会记录任何内容，
因此普通 cast 的 IR、打印形式和序列化结果保持不变。

`blayout=col_major` + `slayout=row_major` 搭配默认的 512 字节 fractal 即**标准 NZ**：与
`tile_view_semantics::GetImplicitTileLayout` 为 Mat 或 Left tile 指定的三元组相同，也是矩阵单元
消费的布局。请求该布局后，A5 的打包 FP4 转换会直接把 FP8 结果写入分块网格，从而省去原本在操作数
staging 到 Mat 之前的 Vec 到 Vec 的 ND 转 NZ `pto.tmov`，以及它占用的 Vec 缓冲。

### 为何关键字拼作 `blayout` / `slayout`

tile 的布局本身**就是** `blayout` / `slayout` / `fractal`，即 `TileLayout`，而不是 `pl.NZ` 所指的
GM 侧 `TensorLayout`；并且 `tile.move` 已经以可选属性的形式接受这两个参数，契约同样是"省略即继承源"。
复用它们无需新增枚举，也无需新增打印器、解析器或序列化支持，因为 `TileLayout` kwarg 本就可以往返。
`fractal` 刻意不提供关键字：它由源继承，而 ND 源携带的 512 正是原生 NZ 目标唯一接受的取值。

### cast 会转换与不会转换的布局

cast 执行的**唯一**布局变更是 ND 源到标准 NZ 目标。与继承布局相同的请求按空操作接受；
任何其他变更都会被拒绝并指向 `tile.move`。特别地，ZN（`row_major` / `col_major`）和非 512 的
fractal 都会被拒绝，这与 PTOAS 一致——它对本转换同样不接受这两者。

接受的几何形状，全部在**物理**形状上校验（PTOAS 会把分块目标补齐到块网格，并把更窄的范围
记录为有效区域）：

| 要求 | 原因 |
| ---- | ---- |
| 源为 ND（`row_major` / `none_box`） | 原生转换唯一能读取的源形式；同时避免后端结果修复重新推导该 cast，从而生成退化的 NZ 到 NZ 的 `tile.move` |
| 目标行数为 16 的倍数 | 一个 NZ 块为 16 行；不完整的块行没有可寻址的存储 |
| 目标列数为 `32 / dtype_bytes` 的倍数 | 块的列数；FP8 为 32 |
| 两侧 `compact=null` | 带填充的分块目标不在本次范围内 |
| 源打包列数为 32 的倍数 | 原生转换自身的寄存器对齐要求 |

这两个粒度同时保证 tile 是整数个 512 字节块，因此其字节占用仍是分配器已经计算的
`rows * cols` 乘积——不存在分块填充项，这也是没有任何分配代码变更的原因。

**只有原生就能写出分块目标的 dtype 组合才会真正接受该请求。** PTOAS 的 `tcvt` 校验器*仅*对
`!pto.f4E2M1x2 -> f8E4M3FN` 检查目标布局；对其他任何组合，它都会无提示地接受分块目标，然后按
ISA 自身的方式寻址。无提示通过并不等于具备该能力，因此显式请求、而没有任何原生规则能写出的分块
目标会在此处被拒绝，并报出 dtype 组合与架构名。该检查以*请求*为依据，因此只是顺带带有分块布局的
tile 仍走常规的[布局修复路径](23-resolve_backend_op_layouts.md)。

范围：仅 A5 `E2M1x2 -> FP8E4M3FN`、仅 `compact=null`、且仅 Tile 入口。`tensor.cast` 从不声明这些
属性，因此算子注册表会以自身的 `Unknown kwarg` 错误在那里拒绝它们，不存在 Tensor 层的目标布局路径。

## 原生 cast 与展开链

`pl.cast` 并不总是编译成一条指令。某个 `(src, dst)` 组合究竟是一条硬件
`pto.tcvt`，还是被展开成多跳链，完全取决于目标架构，而两者在性能和数值上都有
差别。下表按架构列出哪些组合是原生的、哪些由本 Pass 展开，`(n)` 为发射的
`tcvt` 指令条数。

**开销**：展开链每一跳都要对整个 tile 发一条 `tcvt`，并额外占用一块中转 tile
（仍受常规的 buffer 复用约束）。因此同样形状下，3 跳链的向量工作量约为原生
cast 的三倍。在向量受限的 kernel 里这是实打实的代价——若热点循环里出现了意料之
外的链，可考虑换一条等价的 dtype 路径避开它。

**数值**：当每个中转类型都能精确表示*落在目标值域内*的源值时，整条链与直接转换
逐位相同，只有最后一跳发生舍入。`INT32 -> FP32 -> FP16` 属于这一类：FP16 在 65504
以上饱和，而该范围内的整数在 FP32 中都是精确的，因此 FP32 这一跳不会舍入。

当中转类型会先舍入时，整条链发生双重舍入，结果可能与直接舍入相差目标类型的
1 个 ULP：

| 链路 | 双重舍入的原因 | 实测 |
| ---- | -------------- | ---- |
| `INT32 -> FP32 -> BF16` | BF16 值域覆盖整个 INT32，但 FP32 只有 24 位有效数字，因此超过 2^24 的输入在第一跳就被舍入 | 20 万均匀 INT32 中有 3 个相差 1 个 BF16 ULP |
| `FP32 -> FP16 -> INT8` | FP16 在取整跳之前先舍入一次 | 仅边界值 |

这并不是相对更优方案的退化：这些组合在 ISA 上压根没有直连转换，展开链是唯一可行
的下降方式。它同样与参考实现一致——`torch` 自身的 `int32 -> bfloat16` 在 200 万个
均匀 INT32 样本上与该链完全相同，因为 torch 也是经 fp32 中转的。

**如何确认自己的 kernel 走了哪条**：每一跳都会在生成的 MLIR 里出现一条 `pto.tcvt`：

```bash
grep -n 'pto.tcvt' build_output/<case>/ptoas/<kernel>.pto
```

### Ascend950 (a5)

| 源类型 | 原生（1 条指令） | 展开链（n 条指令） |
| ------ | ---------------- | ------------------ |
| `bf16` | `fp16`, `fp32`, `fp4`, `int32` | `fp8e4m3`(2), `fp8e5m2`(2), `hf8`(2), `int16`(2), `int8`(2), `uint16`(2), `uint8`(2) |
| `fp16` | `fp32`, `hf8`, `int16`, `int32`, `int8`, `uint8` | `bf16`(2), `fp8e4m3`(2), `fp8e5m2`(2), `uint16`(2), `fp4`(3) |
| `fp32` | `bf16`, `fp16`, `fp8e4m3`, `fp8e5m2`, `hf8`, `int16`, `int32`, `int64` | `fp4`(2), `int8`(2), `uint16`(2), `uint8`(2) |
| `fp4e2m1x2` | `bf16`, `fp8e4m3`（满足 ND 几何约束） | `fp8e4m3`(3，回退), `fp8e5m2`(3), `hf8`(3), `int8`(3), `uint8`(3) |
| `fp4e1m2x2` | `bf16` | `fp8e4m3`(3), `fp8e5m2`(3), `hf8`(3), `int8`(3), `uint8`(3) |
| `fp8e4m3` | `fp32` | `bf16`(2), `fp16`(2), `fp8e5m2`(2), `hf8`(2), `int16`(2), `fp4`(3), `int8`(3), `uint16`(3), `uint8`(3) |
| `fp8e5m2` | `fp32` | `bf16`(2), `fp16`(2), `fp8e4m3`(2), `hf8`(2), `int16`(2), `fp4`(3), `int8`(3), `uint16`(3), `uint8`(3) |
| `hf8` | `fp32` | `bf16`(2), `fp16`(2), `fp8e4m3`(2), `fp8e5m2`(2), `int16`(2), `fp4`(3), `int8`(3), `uint16`(3), `uint8`(3) |
| `int16` | `fp16`, `fp32`, `int32`, `uint32`, `uint8` | `bf16`(2), `fp8e4m3`(2), `fp8e5m2`(2), `hf8`(2), `int8`(2), `uint16`(2), `fp4`(3) |
| `int32` | `fp32`, `int16`, `int64`, `uint16`, `uint8` | `bf16`(2), `fp16`(2), `fp8e4m3`(2), `fp8e5m2`(2), `hf8`(2), `fp4`(3), `int8`(3) |
| `int64` | `fp32`, `int32` | `bf16`(2), `fp16`(2), `fp8e4m3`(2), `fp8e5m2`(2), `hf8`(2), `int16`(2), `uint16`(2), `uint8`(2), `fp4`(3), `int8`(3) |
| `int8` | `fp16`, `int16`, `int32` | `hf8`(2), `uint16`(2), `uint8`(2), `fp8e4m3`(3), `fp8e5m2`(3), `fp4`(4) |
| `uint32` | `int16`, `uint16`, `uint8` | `int8`(3) |
| `uint8` | `fp16`, `uint16` | `hf8`(2), `int8`(2), `fp8e4m3`(3), `fp8e5m2`(3), `fp4`(4) |

其他行中的 `fp4` 代表任一打包格式。E2M1x2→E4M3FN 直接边仅在无分块行主序 ND Vec
二维几何满足约束且有效行列数为静态时选用，
作为中间步骤时同样逐 hop 检查后端规则。条件不满足时保留三步回退链及警告；成功选择直接转换时
不发出慢链警告。详见 [FP4 约束](../fp4.md#数值转换限制)。

### Ascend910 (a2a3)

| 源类型 | 原生（1 条指令） | 展开链（n 条指令） |
| ------ | ---------------- | ------------------ |
| `bf16` | `fp32`, `int32` | `fp16`(2), `int16`(2), `int4`(3), `int8`(3), `uint8`(3) |
| `fp16` | `fp32`, `int16`, `int32`, `int4`, `int8`, `uint8` | `bf16`(2) |
| `fp32` | `bf16`, `fp16`, `int16`, `int32`, `int64` | `int4`(2), `int8`(2), `uint8`(2) |
| `int16` | `fp16`, `fp32` | `bf16`(2), `int4`(2), `int8`(2), `uint8`(2) |
| `int32` | `fp16`, `fp32`, `int16`, `int64` | `bf16`(2), `int4`(2), `int8`(2), `uint8`(2) |
| `int4` | `fp16` | `int8`(2), `uint8`(2) |
| `int64` | `fp32`, `int32` | `bf16`(2), `fp16`(2), `int16`(2), `int4`(3), `int8`(3), `uint8`(3) |
| `int8` | `fp16` | `int4`(2), `uint8`(2) |
| `uint8` | `fp16` | `int4`(2), `int8`(2) |

两列都没有出现的组合会被拒绝：本 Pass 会报出 src、dst 和 arch，而不是生成有损的链。

## 运行时机

Default 流水线：

```text
lower_composite_ops → flatten_tile_nd_to_2d → legalize_tile_cast → auto_tile_matmul_l0
... → split_vector_kernel → legalize_tile_cast → stamp_tfree_split
```

放在 Flatten 之后以覆盖其新插入的 cast；放在 MatmulL0 之前以免拆开本可 fold 的原生降精度 cast。

第二次执行在向量切分后重新检查有条件的原生转换。新增中间 tile 保留已推导的内存空间，
不在该后期阶段重新执行内存放置 pass。空 replay lane 保持为空。

## API

| C++ | Python |
| --- | ------ |
| `pass::LegalizeTileCast()` | `passes.legalize_tile_cast()` |
