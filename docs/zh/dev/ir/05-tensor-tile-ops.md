# Tensor 与 Tile 算子

本文是[算子系统](05-operators.md)参考文档的数据算子部分。

## TensorOp：N 维张量操作

**用途**：支持完整广播的通用 N 维张量
**类型**：`TensorType`（任意维度）
**位置**：`src/ir/op/tensor_ops/`
**Python API**：`from pypto.ir.op import tensor`

**操作：** `tensor.add/sub/mul/div`（逐元素，支持完整 N 维广播），`tensor.maximum/minimum`（逐元素 max/min；rhs 可为 tensor 或 scalar — `ConvertTensorToTileOps` 根据 rhs 类型分发到 `tile.maximum/minimum` 或 `tile.maximums/minimums`），`tensor.set_validshape`（更新 valid_shape 元数据，不搬移数据；也可通过 `pl.set_validshape` 使用），`tensor.sort32` / `tensor.mrgsort_format1` / `tensor.mrgsort_format2`（排序；分别对应 `tile.sort32` / `tile.mrgsort` 的 tensor 层接口，由 `ConvertTensorToTileOps` 转换为 tile 操作），`tensor.gather`（指定 `dim` 时按维索引，省略时按扁平元素索引；见下文），`tensor.gather_mask`（掩码模式选择；对应 `tile.gather_mask`，支持可选同位宽 `output_dtype`；见[掩码模式](#掩码模式)），`tensor.scatter`（按列散布；`tensor.gather` 的按列逆操作，MVP 仅支持 2D 输入 + `dim=-1` —— `out[b, index[b, k]] = src[b, k]`，`index` 与 `src` 同形状 —— 由 `ConvertTensorToTileOps` 下降到 `tile.scatter`），`tensor.scatter_mask`（按掩码模式散布；对应 `tile.scatter_mask`，将紧凑 `input` 按掩码扩展到 `dst` 的对应列 —— 见[掩码模式](#掩码模式)），`tensor.ci` / `tensor.arange`（生成连续整数序列，下层降到 `tile.ci`；同时通过 `pl.arange` 暴露在顶层 namespace），`tensor.and/ands/or/ors/xor/xors/not/shl/shls/shr/shrs`（仅整数的位运算与移位。此处列出的是注册的 *IR* 名称；其中名字本身是 Python 关键字的三个，其 Python 拼写带尾部下划线 —— `tensor.and_`、`tensor.or_`、`tensor.not_` —— printer 也按该形式输出，以保证 IR 能往返为合法 Python；对应同名 `tile.*` 操作。张量-张量形式的两个操作数形状必须相同 —— 硬件没有 `tile.row_expand_and`，因此广播在类型推导阶段即被拒绝，而不是延迟到 pass 中失败。`tensor.not` 仅支持 int16/uint16，与 `tile.not`/TNOT 一致。移位保持 lhs 的元素类型；`and`/`or`/`xor` 要求操作数使用相同的 8/16/32 位 dtype，scalar 形式使用 tile 下沉要求的同位宽 signless `iN` 编码。`ConvertTensorToTileOps` 将其中九个 1:1 下降，并为 `tensor.xor`/`tensor.xors` 合成 `pto.txor` 所需的临时操作数，使 tensor 层调用者无需提供 `tmp`）

`tensor.view` 是只修改元数据的零拷贝 shape/layout 重新解释操作。它注册为 `TensorOp`，并在 `ConvertTensorToTileOps` 中作为 passthrough 处理；PTO in-core codegen 会将其降级为基于原始 base pointer 的 `pto.make_tensor_view`。目标 rank 至少为 1（DN 至少为 2）。编排层通常仅支持 ND shape 重新解释，且不能同时改变 layout；FP8E8M0 dynamic scale storage 还允许在 packed ND 与 `MX_A_ZZ` 或 `MX_B_NN` 之间建立元素数相同的 shaped alias，编排层保留同一个 runtime tensor，不调用 `reshape`。对部分有效的源张量进行 shape 重新解释时，仅支持把 packed ND 的 leading dimensions 折叠为 2D，或把连续前缀线性折叠为 `[1, product(shape)]`；两种形式都必须显式提供目标 `valid_shape`，并会保留源张量类型及其底层元数据。

扁平 gather 复用 `pl.gather(src, index=idx)`（等价于 `pl.gather(src, idx)`），
语义为 `out = src.reshape(-1)[idx]`。索引是二维 INT32 tensor 或 tile，可在内核中
动态计算。结果为 Tensor，shape 和 valid shape 跟随索引，dtype 跟随源
（FP16/FP32/INT16/INT32）。索引必须指向源的有效元素，不支持负索引或越界检查。
仍位于 GM 的连续 ND 源直接下降为 `tile.mgather`，不会把整个源加载进 UB。
GM 源与索引均可使用本地 `DistributedTensor` 窗口。Tile 索引必须位于 Vec，且为
无分形的行主序布局；其他内存空间的索引需要先搬移到 Vec，转置/分形索引布局
会在 codegen 前被拒绝。物理索引列数须为
正的编译期常量：FP16/INT16 源要求为 16 的倍数，FP32/INT32 要求为 8 的倍数，
从而使索引与输出的物理行均按 32 字节对齐。单行同样受此约束，且会在 flat gather
入口检查，而不是延迟到 codegen。调用方应补齐物理索引 tensor，再通过
`pl.set_validshape` 指定较窄的有效区域（valid region）；有效行列数无需对齐。
例如，FP16 的八个有效列应使用物理 shape 为 `[1, 16]` 的索引：

```python
indices = pl.set_validshape(padded_indices, 1, 8)
values = pl.gather(src, index=indices)  # Physical [1, 16], valid [1, 8].
```

片上源下降为 `tile.gather`，scratch 由编译器管理；源必须为静态二维行主序 Vec，
每行按 32 字节对齐（单行除外）。带 stride 的片上窗口先物化为紧凑 tile：
浮点使用 `tile.extract`，INT16/INT32 使用保持数值不变的整数 `tile.adds(..., 0)`。
已证明紧凑的计算结果直接复用，不再复制；存储情况未知时仍进行紧凑物化。
指定 `dim` 时仍按维索引（二维/三维、任意轴），mask/compare
形式保持不变。详见 [gather 下沉](../passes/11-convert_tensor_to_tile_ops.md#扁平-gather-下沉)。

对于普通 `TensorType` 操作数，已支持的 Tensor-scalar 算术算子（`adds`、
`subs`、`muls`、`divs`、`fmods` 以及 scalar `maximum` 或 `minimum`）和
位运算/移位算子（`ands`、`ors`、`shls` 和 `shrs`）会创建新存储，但不能
把 padding 凭空变成有效数据。因此结果保留 Tensor 操作数的 effective
`valid_shape`，同时丢弃源别名、layout、stride 与 padding 元数据。这与已有的
Tile-scalar 规则一致，确保 ragged tail 经 Tensor-to-Tile 下降后仍保持窄有效区。
Scalar 比较与 XOR（`cmp` 和 `xors`）仍不在此规则的支持范围内。

对于普通 Tensor-tensor 算术算子（`add`、`sub`、`mul`、`div`、`fmod`、
`maximum` 和 `minimum`），当两个操作数的物理 shape 相同，且其 effective
`valid_shape` 可证明相等时，结果同样保留该有效区域。`and`、`or`、`shl` 和
`shr` 也采用这条 exact-region 规则。它不需要映射广播轴，并与相应 Tile
结果契约一致；结果仍是新存储，因此不会继承别名、layout、stride 或 padding
元数据。比较、XOR、`part_*`、广播、不同有效区域，以及直接使用 distributed
window 的操作数不在这条规则范围内，因为它们当前的下降或合并契约需要单独处理。

`pl.reinterpret_view(data, dtype, *, shape=None)` 会根据输入分派到等价的 `pl.tensor` 或 `pl.tile` 算子，并保持返回类型种类不变。它是覆盖完全相同字节的零拷贝视图。通用路径支持有/无符号 8/16/32/64 位整数、FP16、BF16 与 FP32；MX 下降额外只允许 INT8↔FP8E4M3FN 和 UINT8↔FP8E8M0 两对等字节 alias。省略 `shape` 时，ND/row-major 缩放最后一轴，DN/col-major 按源/目标字节宽度比例缩放倒数第二轴。显式 shape 必须字节数相等；除非能证明它与自动推导 shape 等价，否则必须完全静态。部分有效的 `valid_shape` 只能使用与自动推导结果等价的 shape。零值/null padding 元数据会保留，依赖 dtype 的 max/min padding 则会清除。初始可执行路径支持 packed ND in-core tensor 及 packed、flat（`none_box`）row/col-major tile；DN tensor 可做类型推导但 Tensor-to-Tile 下降会拒绝，编排层 tensor 暂不支持。

**示例：**

```python
from pypto.ir.op import tensor

ib = IRBuilder()
with ib.function("tensor_example") as f:
    input_a = f.param("input_a", ir.TensorType([128, 64, 32], DataType.FP32))
    input_b = f.param("input_b", ir.TensorType([128, 64, 32], DataType.FP32))
    f.return_type(ir.TensorType([128, 64, 32], DataType.FP32))
    result = ib.let("result", tensor.add(input_a, input_b))
    ib.return_stmt(result)
```

## TileOp：硬件优化 Tile 操作

**用途**：带有显式内存管理的硬件优化 Tile 操作
**类型**：`TileType`（统一缓冲区中的 Tile）
**位置**：`src/ir/op/tile_ops/`
**Python API**：`from pypto.ir.op import tile`

**设计**：使用 `TileType`（而非单独的 `BlockType`）以保持一致性。命名空间 `tile.*` + `TileType` 清楚地表示硬件优化的 Tile 操作。

### 操作列表

| 分类 | 操作 | 描述 |
| ---- | ---- | ---- |
| **内存** | `tile.get_block_idx` | 获取 block 索引（返回 UINT64 标量） |
| - | `tile.load` | TensorType → TileType（DDR 到统一缓冲区） |
| - | `tile.store` | TileType → TensorType（统一缓冲区到 DDR） |
| - | `tile.move` | 在 memory space 之间搬移 tile（`target_memory`）—— 见 [tile.move 的结果 view](#tilemove-的结果-view) |
| **逐元素** | `tile.add/sub/mul/div` | Tile-Tile 操作 |
| - | `tile.adds/subs/muls/divs` | Tile-Scalar 操作。**常量**标量操作数会采用 tile 的元素 dtype（裸整数字面量否则会被解析为 `index`，而任何 `pto.t*s` 算子都不接受它）——但整数 tile 上的浮点字面量仍保持 FP32，以保留类型提升语义。显式的 `pl.const(v, dtype)` 属于用户的有意标注，与任何非常量表达式一样保持不变；非常量的 `index` 标量（循环变量、`pl.dim`）会被拒绝——需用 `pl.cast` 转换。`tensor.*s` 同理。 |
| **一元** | `tile.sqrt` | 逐元素平方根 |
| **量化** | `tile.tquant_mx` / `pl.quant_mx` | 仅 Ascend950 支持的 **MXFP8** block-32 动态量化，返回 `{FP8E4M3FN quant, FP8E8M0 scale}`；`dtype` 必须为 `FP8E4M3FN`。`group_axis` 对齐 PTOAS `grpAxis`（`1` = A 侧 `[M,K]`，`0` = B 侧 `[N,K]` 并转置）。公开 scale shape 为 `[M,K/32]` / `[K/32,N]`；要求完整有效区域和 `K % 64 == 0`（axis1 还要求 `M % 16 == 0`，axis0 还要求 `N % 32 == 0`）。[Pass 13](../passes/13-lower_composite_ops.md) 生成分组 TQUANT 和 X-to-ZZ TMOV。在 mixed task 内，结果可直接经 V2C 供 `matmul_mx` 使用。MXFP4 quant 暂缓。 |
| **变换** | `tile.slice` | 提取子 tile，静态 shape，可选动态 valid_shape |
| - | `tile.extract` | 从 `src` 在 `(index_row, index_col)` 处提取子 tile —— ISA TEXTRACT Variant 1（Mat→Left/Right，Acc→Mat）。结果 layout 取自 `target_memory` 的隐式 view；`Left`/`Right` 例外，使用 TEXTRACT 侧的 L0 格式（与 `tile.move` 的 TMOV 侧不同） |
| - | `tile.reshape` | 重塑 tile 维度（元素总数须一致）。会把源的 `valid_shape` 带到结果上，且绝不扩大 —— 见[reshape 与有效区域（valid region）](#reshape-与有效区域valid-region) |
| - | `tile.reinterpret_view` | 以不同 dtype 对完全相同的字节做零拷贝视图；可选 shape 默认按 layout 推导（仅支持紧密、非分形 tile） |
| - | `tile.transpose` | 交换 tile 的两个轴 |
| - | `tile.set_validshape` | 更新 valid_shape 元数据，不搬移数据 |
| - | `tile.ci` | 生成连续整数序列（升序 start+k 或降序 start-k）；dtype ∈ {INT16, INT32}；最内维 != 1 |
| - | `tile.tri` | 使用 INT32 diagonal offset 生成上三角或下三角 0/1 mask；支持可选的部分 `valid_shape`；映射为 `pto.ttri`。 |
| **规约** | `tile.row_*` / `tile.col_*` | 方向特定的规约（`row_sum`/`row_max`/`row_min`/`row_prod` 折叠最后一轴；`col_*` 折叠第 0 轴）。不存在以 axis 参数化的规约算子 —— ISA 只提供方向特定的指令（`pto.trowsum`、`pto.tcolsum` 等） |
| **聚集** | `tile.gatherb` | 按 32-byte 源块聚集。每个 UINT32 offset 选择一个块；每个 offset 列扩展为 `32 / sizeof(output_dtype)` 个输出元素，valid_shape 同比例扩展。`output_dtype` 默认等于源 dtype，也可选择另一种受支持的字节解释。offset 每行须包含正整数个 8-entry 组。切片源的字节地址必须能被证明为 32-byte 对齐；动态列偏移会被拒绝，而物理行跨度保持对齐时允许动态行偏移。映射为 `pto.tgatherb`。 |
| - | `tile.mgather` | 从 GM tensor 聚集到新 Vec 或 Mat tile。Vec 输出使用 INT32 index tile（`[1,R]`，A5 也支持 `[R,1]`）；Mat 输出使用 ND-layout GM source 与 INT32 index tensor，并采用规范 NZ layout，物理行数按 16 对齐、列数按 `C0 = 32 / sizeof(dtype)` 对齐；可通过较小的二维 `valid_shape` 表达 padding tail。`coalesce="row"` 聚集整行；`"elem"` 按扁平元素索引聚集，且 Mat 输出要求同 dtype、连续 ND、元素数不少于物理输出的 GM `scratch` tensor。`gather_oob` 可选择 `undefined`、`clamp`、`wrap` 或 `zero`。payload dtype 支持 I8/U8/I16/U16/I32/U32/FP16/BF16/FP32，以及仅 A5 支持的 FP8E4M3FN/FP8E5M2/HF8。 |
| **散布** | `tile.scatter` | 按行索引把 `src` 散布到 `dst`（`pto.tscatter` 索引形式；DPS：`dst` 为 in/out，结果别名为 `dst`）。`src` / `dst` dtype ∈ {I8, I16, I32, FP16, FP32, BF16}；`indexes` dtype ∈ {I16, I32}；元素宽度匹配规则：4 字节 dst ↔ INT32，2 字节 dst ↔ INT16，1 字节 dst ↔ INT16。 |
| - | `tile.scatter_mask` | 按掩码模式把 `src` 行写入 `dst` 中由掩码选中的列（DPS：`dst` 为 in/out）。这是 PyPTO codegen 层形式，下降为 `pto.tscatter` 掩码发射 —— **并非**独立的 pto-isa 指令（与 `tile.gather_mask` 不同）。掩码语义见[掩码模式](#掩码模式)。 |

在 Ascend950 上，`quant_mx` 与 `matmul_mx` 可以放在同一个 InCore mixed task
中。编译器会把量化数据与 FP8E8M0 scale 直接经 V2C 传递，同时保留 scale
的逻辑 fractal-32 布局。

`tile.reshape` 保持 dtype、元素总数以及源的有效区域（见下）；`tile.reinterpret_view(data, dtype, *, shape=None)` 改变 dtype，但要求前后总字节数完全相同。省略 `shape` 时，它会根据源/目标 dtype 字节宽度和 tile layout 缩放物理连续轴。在 PTOAS 内存规划下，无论 shape 是否变化，都会下降为保持别名关系的 PTO `treshape` 原语。

### tile.move 的结果 view

推导出的结果 `TileView` 按字段分别取值：

| 字段 | 结果值的来源 |
| ---- | ------------ |
| `blayout` / `slayout` | 凡目标 space 自带 layout（`Mat`、`Acc`、`Left`、`Right`、`LeftScale`、`RightScale`），取**目标**的 implicit layout；扁平 space（`Vec`、`Bias` 等）则沿用源 tile 的 effective layout。两者都可由 `blayout` / `slayout` kwarg 覆盖 |
| `fractal` | **目标** space 的分块（boxing）粒度：`Acc`（L0C，NZ 分形）为 1024，MX scale tile 为 32，其余为 512。窄化例外是承载字节型 MX scale 的 Vec→Vec 重排或 Vec→Mat 跨核暂存 move，此时保留源的 32-byte scale box |
| `valid_shape` / `pad` | 从源带过来 |
| `stride` / `start_offset` | 丢弃 —— 目标是稠密缓冲区 |

layout 来自目标，因为它描述的是目标缓冲区如何分块，由
`tile_view_semantics::GetImplicitTileLayout` 提供。`Right` 仍需就地覆盖：L0B 要求
`blayout=row_major`，而 `[N, 1]` 形状的 implicit `blayout` 是 `col_major`。

`tile.move` 自己把目标 `memory_space` 打到推导出的类型上（参见
[类型](02-types.md#tiletype) 中的 `TileType` 契约），因此当结果 view 与目标 space 的
implicit view 一致时会折叠为 `nullopt` —— 这与
[`InferTileMemorySpace`](../passes/20-infer_tile_memory_space.md) 为重新定型的 tile
刷新的 per-space implicit view 是同一套。

`tile.move` 不支持原地执行：在同一 memory space 内，源和结果必须解析到不同地址。
PyPTO 与 DSA-RP 规划器会落实该约束；如果显式 MemRef 绑定或手工构造的 IR 仍留下
同地址 move，baked-address PTO codegen 会直接报错。

### reshape 与有效区域（valid region）

reshape 是零拷贝视图，无法凭空产生数据：`tensor.reshape` 与 `tile.reshape` 共用
同一条规则，把源的 `valid_shape` 映射到目标 shape，且绝不扩大。有效区域只能表示为
以原点为锚的矩形框，因此并非所有源区域都能在重新切分后保留：

| 源区域 | 结果 |
| ------ | ---- |
| 完全有效 | `new_shape` —— 会被规范化掉，不产生 view，已有程序不受影响 |
| 可证明为空 | 全零矩形框 |
| 仅增删完全有效的单位轴 | 保留的轴按 1:1 映射，可精确保留任意矩形 |
| 目标 shape 以同样方式切分缓冲区 | `new_shape` 中覆盖同一批元素的矩形框（若存在） |
| 其他情况 | **拒绝** —— `valid_shape` 无法描述 reshape 后的区域 |

最后一条规则把有效区域读作它填充的若干**连续段（run）**。相邻的源轴只要满足
“低位轴完全有效”或“高位轴被钉死在单个坐标上”，就属于同一段；否则高位轴的
stride 会残留在区域中并把它切开。每一段都是自身容量的一个扁平前缀，因此当
`new_shape` 把自己的维度分成同样的段、且每段前缀都落在维度边界上时，区域即可
精确映射。对于静态区域这条规则是**精确的**：当且仅当 `new_shape` 下存在某个矩形框
表示完全相同的元素集合时才接受。

因此 `[8, 16]` valid `[5, 16]` 是单段情形（80 个元素的扁平前缀），可映射为
`[16, 8]` valid `[10, 8]` 或 `[128]` valid `[80]`，而 `[4, 32]` 会被拒绝 ——
80 个元素不是整数行（每行 32）。`[2, 2, 2]` valid `[2, 1, 2]` 是两段情形
`2 | 4` —— 扁平元素集合为 `{0, 1, 4, 5}`，根本不是前缀 —— `[2, 4]` 可以把它写成
valid `[2, 2]`，而 `[8]` 没有每 4 个元素一次的维度边界，无法表示。
`[8, 16]` valid `[8, 5]` 切分为 `8 | 16`，`[16, 8]` 无法按同样方式重新分组，因此
被拒绝。`tensor.reshape` 可选的第三个 `valid_shape` 操作数只能*收窄*推导出的区域，
不能声称拥有该区域之外的数据。

符号化 extent 会削弱规则能证明的范围，但本身并不导致拒绝：**任何**一段都可以原样
携带符号化的*有效* extent —— 只要它所在段的目标维度步长恰好等于该段的 trailing
volume。因此 `[4, 2, 8]` valid `[v, 1, 8]` 即便切分为 `4 | 16` 两段，仍可映射为
`[4, 16]` valid `[v, 8]`。必须是静态的是用来度量区域的*物理*几何：目标各维 extent、
每段自由轴以下的各维 extent、符号化路径上自由轴本身（其维度必须可证明足够宽），
以及区域切分为多段时每一段的容量。达不到这些条件时一律拒绝而不做猜测。

**恒等** `tile.reshape`（目标形状与源形状相同）还会保留源的 layout 三元组
（`blayout` / `slayout` / `fractal`）及其已解析的内存空间，而不是按形状重新推导 layout。
重新推导得到的是与空间无关的扁平 layout；`NormalizeImplicitTileView` 只会为可折叠的
view 兜底，而被收窄、带 pad 或声明了 `compact` 的 Acc 盒永远不可折叠——扁平 layout 于是
被固化下来，其读者会把 L0C 当作普通 row-major 缓冲区来遍历（issue #2470）。

**数据流：** `TensorType (DDR) → tile.load → TileType (Unified Buffer) → tile.{ops} → TileType → tile.store → TensorType (DDR)`

### 掩码模式

`*.gather_mask` / `*.scatter_mask` 使用编译期 `MaskPattern`（`pl.tile.MaskPattern`，整数取值 1–7，与硬件 `VREDUCEv2` 的 pattern mode 一致）按行标记列的一个子集（模式名**从右往左**读，最右位对应列 0）。同一标记集合驱动两个算子做**相反方向**的操作。**`gather_mask`** *选择并紧凑*：从宽输入中读取被标记的列，紧凑写入较窄输出的前若干列（`out_cols = cols / stride`）；这是真实的 pto-isa 指令（`pto.tgather` 掩码形式），A2/A3 **与 A5** 均支持。**`scatter_mask`** *放置并扩展*：把紧凑输入写入更宽 `dst` 的被标记列（`dst_cols = cols * stride`），未标记列保留 `dst` 原值（DPS）；这是 **PyPTO codegen 层形式，并非独立的 pto-isa 指令** —— 不存在 `pto.tscatter` 掩码指令（与 gather 不同）—— PyPTO 为 A2/A3 / CPU-sim 类下降路径发射它。例如对 `[a0 a1 a2 a3 a4 a5 a6 a7]`：gather `P0101 → [a0 a2 a4 a6]`；对 `[s0 s1 s2 s3]` 做 scatter `P0101 → [s0 · s1 · s2 · s3 ·]`（`·` 表示保留的 `dst`）。

| 模式 | 整数 | 标记列 `c` 的条件 | 被标记的列 | 步长 |
| ---- | ---- | ----------------- | ---------- | ---- |
| `P0101` | 1 | `c % 2 == 0` | 0, 2, 4, … | 2 |
| `P1010` | 2 | `c % 2 == 1` | 1, 3, 5, … | 2 |
| `P0001` | 3 | `c % 4 == 0` | 0, 4, 8, … | 4 |
| `P0010` | 4 | `c % 4 == 1` | 1, 5, 9, … | 4 |
| `P0100` | 5 | `c % 4 == 2` | 2, 6, 10, … | 4 |
| `P1000` | 6 | `c % 4 == 3` | 3, 7, 11, … | 4 |
| `P1111` | 7 | 全选 | 全部 | 1 |

最后一维须能被步长整除。`gather_mask` 另接受可选的同位宽 `output_dtype`（按位重解释，而非数值转换）。参考：gather 的选择语义见 `pto-isa` 的 `MaskSelect`（`include/pto/cpu/TGather.hpp`）；pypto 类型推导见 `src/ir/op/tile_ops/gather.cpp`（gather）/ `src/ir/op/tile_ops/scatter.cpp`（scatter）。

### 使用示例

```python
from pypto.ir.op import tile

ib = IRBuilder()
with ib.function("tile_computation") as f:
    input_a = f.param("input_a", ir.TensorType([128, 128], DataType.FP32))
    input_b = f.param("input_b", ir.TensorType([128, 128], DataType.FP32))
    output = f.param("output", ir.TensorType([128, 1], DataType.FP32))
    f.return_type(ir.TensorType([128, 1], DataType.FP32))

    # Load, compute, reduce, store
    tile_a = ib.let("tile_a", tile.load(input_a, [0, 0], [32, 128]))
    tile_b = ib.let("tile_b", tile.load(input_b, [0, 0], [32, 128]))
    tile_mul = ib.let("tile_mul", tile.mul(tile_a, tile_b))
    tile_sqrt = ib.let("tile_sqrt", tile.sqrt(tile_mul))
    # row_sum 折叠最后一轴 -> [32, 1]。scratch tile 必须与输入 dtype 和 rank 相同，
    # 且每一维都不小于输入的对应维度。
    tmp_tile = ib.let("tmp_tile", tile.create([32, 128], DataType.FP32))
    tile_sum = ib.let("tile_sum", tile.row_sum(tile_sqrt, tmp_tile))
    result = ib.let("result", tile.store(tile_sum, [0, 0], output))
    ib.return_stmt(result)
```
