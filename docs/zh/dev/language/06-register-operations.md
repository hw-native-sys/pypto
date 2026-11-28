# 寄存器 API 参考

`pl.reg` 提供在 A5（Ascend950）`pl.reg_scope()` 内使用的逻辑向量和掩码操作。
本文完整列出当前支持的 `pl.reg` 操作、参数、返回类型和 dtype 限制。
下表中的所有操作名称均带有 `pl.reg.` 前缀。

编译器在早期将每个寄存器作用域提取为 `FunctionType.VF` 设备辅助函数，
默认 `reg_scope()` 使用逻辑 VMI，`native=True` 选择物理 micro lowering。捕获的 Tile 和 Scalar 成为函数参数，Tile 别名可以
返回，寄存器和局部 Scalar 保留在辅助函数内部。

## 作用域与寄存器值

```python
import pypto.language as pl

@pl.function(type=pl.FunctionType.InCore)
def kernel(
    src: pl.Tile[[128], pl.FP32, pl.MemorySpace.Vec],
    dst: pl.Tile[[128], pl.FP32, pl.MemorySpace.Vec],
):
    with pl.reg_scope():
        x = pl.reg.load(src, lanes=64)
        mask = pl.reg.active_mask(32, lanes=64)
        y = pl.reg.add(x, x, mask=mask)
        out = pl.reg.store(dst, value=y, mask=mask)
    z = pl.tile.add(out, out)
```

`pl.reg_scope(*, auto_sync=True, native=False)` 没有 captures、结果列表或 yield，也不能嵌套。
`auto_sync` 必须是编译期 bool；省略或 `True` 保留 UB 屏障自动插入，
`False` 选择显式同步。
前端允许 InCore、AIV 设备上下文，包括 Opaque、Orchestration、Graph 函数内
在 outlining 前显式声明的 InCore 作用域。`ExpandMixedKernel` 后，寄存器区域必须
位于 AIV 函数内。禁止直接在调度代码中使用，也禁止在 RegScope 内嵌套调度作用域。
显式配置非 A5 目标时拒绝；尚未选定目标时，目标检查延迟到编译阶段。

### 显式 UB 屏障

`pl.reg.mem_bar(kind: str) -> None` 仅允许作为寄存器作用域内的语句，
精确生成 `pto.mem_bar "KIND"`，不能赋值、返回或作为表达式使用。
`kind` 是大小写敏感的编译期字符串，无默认值：

| 家族 | 可用类型 |
| ---- | -------- |
| VV | `VV_ALL`, `VST_VLD`, `VLD_VST`, `VST_VST` |
| VS | `VS_ALL`, `VST_LD`, `VLD_ST`, `VST_ST` |
| SV | `SV_ALL`, `ST_VLD`, `LD_VST`, `ST_VST` |
| SS | `SS_ALL`, `ST_LD`, `LD_ST`, `ST_ST` |

```python
# tile is initialized local Vec storage captured from outside the scope.
with pl.reg_scope(auto_sync=False):
    value = pl.reg.load(tile, lanes=64)
    pl.reg.mem_bar("VLD_VST")
    tile = pl.reg.store(tile, value=value)
    pl.reg.mem_bar("VST_VLD")
    value = pl.reg.load(tile, lanes=64)
```

手动作用域由作者提供所有必要的 RAW/WAR/WAW 和跨迭代屏障；自动作用域
也允许显式屏障。`ALL` 仅覆盖所属家族，`VV_ALL` 不覆盖标量访存。
这些屏障只约束同一 vecscope 内的 UB 访问，不能替代跨作用域、DMA、GM
或跨核同步。参见 [VF 编译契约](../codegen/03-vpto-vf-integration.md)。

内联 helper 可以包含 `reg_scope` 并接收显式标注的 Vec Tile。展开时继承调用方的
执行上下文，最终仍必须位于 AIV。寄存器值不能跨越 helper 边界，应返回由
`reg.store` 写入的 Tile。

`pl.Reg[lanes, dtype]` 表示向量寄存器（VecReg）。
`pl.Reg[lanes, pl.BOOL, pl.RegKind.MaskReg]` 表示谓词掩码（MaskReg）。
第三个可选类型注解参数为寄存器种类，默认为 `pl.RegKind.VecReg`。
通常直接根据操作推导寄存器类型。`lanes` 是静态正整数，表示逻辑元素数量，
不是张量 shape，也不是字节数。

外部 Tile 和只读 Scalar 是隐式输入。寄存器与局部 Scalar 不能通过区域外使用、
函数返回、循环结果、元组或向外部变量赋值逃逸；函数边界递归检查嵌套元组类型。Tile 操作数必须源于区域外；
`store` 和 `scatter` 返回的 Tile 别名可以在区域外使用。寄存器操作的返回值都必须赋给变量，
包括 `store` 和 `scatter`。将 `squeeze_store_step` 返回的状态赋给变量，并在循环内通过
`pl.yield_` 携带该状态。
`pl.hardware_loop()`（`HardwareLoopPolicy.Require`）
只能位于该词法作用域内，零次循环也不例外。循环 API 和 hardware-loop policy 参见
[语句](01-statements.md)。

当前 A5 codegen 边界及后端执行限制见 [VF 执行](../codegen/03-vpto-vf-integration.md#required-hardware-loops)。

## 通用参数与 dtype 记法

| 参数 | 含义 |
| ---- | ---- |
| `value`、`lhs`、`rhs` | 输入 VecReg。二元向量输入的 dtype、signedness 和 lane 数必须相同。 |
| `mask` | 控制操作生效 lane 的 BOOL MaskReg，lane 数与输入向量一致。只有签名带 `mask=None` 时才可省略。 |
| `seed` | 比较操作必需的控制掩码；seed 为 false 的 lane，其比较结果为 false。 |
| `scalar` | Scalar SSA 值或数值字面量。已有 SSA 必须匹配要求的 scalar dtype，不做隐式转换。 |
| `dtype` | 如 `pl.FP32` 的 dtype 常量，指定生成器输出或转换目标类型。 |
| `lanes` | 静态正整数逻辑元素数；load 和生成器通过关键字传入。 |
| `pmode` | 非激活结果 lane 的处理策略。仅接受 `"zero"`，即非激活向量 lane 清零。 |

签名中的 `*` 表示之后的参数只能按关键字传入。省略可选 mask 时所有 lane 参与。
除转换、位重解释、直方图、归约、`expdif` 和字节 `gather` 外，
结果保持输入向量 dtype 和 lane 数。不做隐式向量类型提升或
lane 广播。DSL 数值字面量会归一化为要求的 scalar dtype；整数参数不接受 Python
`bool`。原始 IR 和 Builder 操作数必须已具有要求的 dtype。

下表定义后续 API 表使用的精确 dtype 集合：

| 集合 | Dtype |
| ---- | ----- |
| `F` | FP16、FP32 |
| `I` | INT8、UINT8、INT16、UINT16、INT32、UINT32 |
| `S` | INT8、INT16、INT32 |
| `I16_32` | INT16、UINT16、INT32、UINT32 |
| `T` | FP16、BF16、FP32、INT8、UINT8、INT16、UINT16、INT32、UINT32 |
| `C` | FP16、FP32、INT32、UINT32 |

`+` 表示集合并集。BOOL 仅用于掩码；INDEX 用于 offset 和 active lane 数。
API 表未列出的 dtype 均不代表已支持。

## 内存操作

| API 签名 | 参数与行为 | 返回值 | Dtype |
| -------- | ---------- | ------ | ----- |
| `load(tile, offset=0, *, lanes)` | `tile`：Vec UB 内存中的源 Tile。`offset`：起始元素索引，支持 INDEX Scalar 或整数字面量。读取连续 `lanes` 个元素。 | 与 Tile 同 dtype、包含 `lanes` 个元素的 VecReg | `T` |
| `store(tile, offset=0, value=None, mask=None, *, block_stride=None)` | `tile`：目标 Vec Tile。`offset`：起始元素索引。`value`：必需的 VecReg，虽然 Python 默认值为 None。`mask`：可选写掩码；仅写入激活 lane，保留目标其他元素。 | `tile` 的 Tile 别名 | `T`；value 与 Tile dtype 必须一致 |

`block_stride` 是可选的静态正整数，范围为 `[1, 32767]`，单位为 **32 字节块**；
`offset` 的单位仍是元素。省略时保留连续写入行为。令元素字节数为 `E`、
块宽为 `B = 32 / E`，活动 lane `i` 写入元素
`offset + (i // B) * block_stride * B + i % B`。
非活动 lane 和块间空隙保留原值。静态边界检查包含最后一个跨步块，初始化检查
仅认可实际写入的活动块。动态 offset 的运行时安全由调用者保证。内存效果查询返回
包含空隙的保守包围区间，用于依赖和生命周期分析。
PyPTO 将 stride 生成为 `i16` 操作数，并传入与向量 lane 数相同的掩码。
省略掩码时，PyPTO 生成 `PAT_ALL`。PTOAS 将 BF16 跨步写入降为 `vsstb`。例如
`pl.reg.store(dst, 16, values, mask, block_stride=16)` 将 128 个 BF16 lane
写入八个 NZ 面板，元素偏移为 `16, 272, ..., 1808`。首版不支持动态 stride 标量。
静态初始化分析每次最多记录跨步写入的前 256 个块，使编译工作量不随 lane 数量无限增长。
后续块不会因该次写入而被证明已初始化；若后续寄存器读取需要该区域，请预先初始化目标。
该分析预算不限制写入的 lane 数量，也不改变生成的写入操作。

offset 以**元素**计，不是字节。静态 offset 必须非负，完整的
访问范围必须位于 Tile 容量内（连续写入为 `[offset, offset + lanes)`），
masked store 也不放宽该限制。
静态 load 必须位于已初始化的 `valid_shape` 内，后续 mask 不能保护未初始化读取。
一维 Tile 可直接使用；多维 Tile 需要显式连续行优先 View，拒绝带 padding、boxed
或 compact 的布局。View 起始 offset 计入地址计算和底层分配边界检查。

当静态检查不能证明违规时，动态 offset 和 valid extent 会传递给后端；调用者负责
运行时边界和初始化。store 不扩大 `valid_shape`，不分配新存储，也不创建快照。
其结果与目标 Tile 具有相同类型，并引用同一存储。

## 向量与掩码生成器

| API 签名 | 参数与行为 | 返回值 | Dtype |
| -------- | ---------- | ------ | ----- |
| `broadcast(scalar, *, lanes, dtype)` | 将标量或单 lane VecReg 广播到所有 lane。输入 dtype 必须等于 `dtype`。 | VecReg | `T` |
| `index_gen(base, *, lanes, dtype, order="ascending")` | `base` 为标量起始值；第 `i` 个 lane 为 `base + i`。`order` 仅支持 `"ascending"`。base dtype 必须等于 `dtype`。 | VecReg | `F + I`；类型推导时即拒绝 BF16，不进入 codegen |
| `full_mask(*, lanes)` | 激活所有逻辑 lane。 | BOOL MaskReg | BOOL |
| `active_mask(active_lanes, *, lanes)` | 使用静态或运行时 INDEX 数量激活前缀 `[0, active_lanes)`。常量检查范围 `[0, lanes]`；运行时数量由调用者保证在该范围内。零产生全 false 掩码，输入数量保持不变。 | BOOL MaskReg | BOOL |

## 向量算术

以下每个 API 的签名均为 `(lhs, rhs, mask=None, *, pmode="zero")`，返回 VecReg。
`lhs`、`rhs` 是同类型向量；可选 `mask` 控制参与计算的 lane，非激活结果清零。

| API | 激活 lane 的计算 | Dtype |
| --- | ---------------- | ----- |
| `add` | `lhs + rhs` | `T` |
| `sub` | `lhs - rhs` | `T` |
| `mul` | `lhs * rhs` | FP16、BF16、FP32、INT16、UINT16、INT32、UINT32 |
| `div` | `lhs / rhs` | `F` |
| `minimum` | 逐元素最小值 | `T` |
| `maximum` | 逐元素最大值 | `T` |

## 一元算术

以下每个 API 的签名均为 `(value, mask=None, *, pmode="zero")`，返回 VecReg。
`value` 为输入向量，`mask` 为可选掩码。这些 API 不接受 BF16 输入。

| API | 激活 lane 的计算 | Dtype |
| --- | ---------------- | ----- |
| `abs` | 绝对值 | `F + S` |
| `neg` | `-value` | `F + I` |
| `relu` | `max(value, 0)` | FP16、FP32、INT32 |
| `exp` | 自然指数 `exp(value)` | `F` |
| `log` | 自然对数 `ln(value)` | `F` |
| `sqrt` | 平方根 | `F` |

## 整数位运算与向量移位

二元操作签名为 `(lhs, rhs, mask=None, *, pmode="zero")`。
`bitwise_not` 的签名为 `(value, mask=None, *, pmode="zero")`。
这些操作均返回 VecReg，仅接受 `I`，支持可选的同 lane 掩码。
向量移位的 `rhs` 为每个 lane 提供独立整数移位量，其 dtype 和 lane 数必须与 `lhs` 相同。

| API | 激活 lane 的计算 | Dtype |
| --- | ---------------- | ----- |
| `bitwise_and` | 整数按位 AND | `I` |
| `bitwise_or` | 整数按位 OR | `I` |
| `bitwise_xor` | 整数按位 XOR | `I` |
| `bitwise_not` | 整数按位取反 | `I` |
| `shift_left` | 左移 | `I` |
| `shift_right` | 有符号输入算术右移，无符号输入逻辑右移 | `I` |

上述操作处理向量。布尔谓词掩码请使用下文的 `mask_and`、`mask_or`、`mask_xor`、`mask_not`。

## 向量与标量算术及移位

以下每个 API 的签名均为 `(value, scalar, mask, *, pmode="zero")`，返回 VecReg。
`value` 为向量，`scalar` 应用于每个 lane，`mask` **必需**。
与这些操作不同，`broadcast` 会单独构造一个向量值。

| API | 激活 lane 的计算 | 向量 dtype | Scalar dtype |
| --- | ---------------- | ---------- | ------------ |
| `adds` | `value + scalar` | `F + I` | 与向量相同 |
| `muls` | `value * scalar` | `F + I16_32` | 与向量相同 |
| `maximums` | `max(value, scalar)` | `F + I` | 与向量相同 |
| `minimums` | `min(value, scalar)` | `F + I` | 与向量相同 |
| `shift_lefts` | 每个 lane 左移 `scalar` 位 | `I` | INT16 |
| `shift_rights` | 每个 lane 按 signedness 右移 `scalar` 位 | `I` | INT16 |

scalar 移位量在 PyPTO 中是 INT16，发射为 PTOAS signless `i16`。
已有 INT32 SSA 移位量会被拒绝，不会被静默收窄。这些向量与标量 API 均不接受 BF16 输入。

## 比较、选择与掩码逻辑

| API 签名 | 参数与行为 | 返回值 | Dtype |
| -------- | ---------- | ------ | ----- |
| `compare(lhs, rhs, seed, *, predicate)` | 在必需的 `seed` 控制下比较同类型向量；`predicate` 指定比较关系。 | 与输入同 lane 的 BOOL MaskReg | `C` |
| `compares(value, scalar, seed, *, predicate)` | 在必需的 `seed` 控制下，将每个 lane 与同 dtype 的 `scalar` 比较。 | 与输入同 lane 的 BOOL MaskReg | `C` |
| `select(mask, true_value, false_value)` | `mask[i]` 为 true 时取 `true_value[i]`，否则取 `false_value[i]`。true/false 向量 dtype 和 lane 数必须相同。 | VecReg | `F + I` |
| `gather_lanes(value, indices)` | 第 `i` 个结果为 `value[indices[i]]`。`indices` 必须是与输入等存储位宽、等 lane 的整数向量；运行时索引必须位于源 lane 范围内。 | VecReg | `F + I`，另有下表的 lane 限制 |
| `mask_and(lhs, rhs)` | 同 lane 掩码的布尔 AND | BOOL MaskReg | BOOL |
| `mask_or(lhs, rhs)` | 同 lane 掩码的布尔 OR | BOOL MaskReg | BOOL |
| `mask_xor(lhs, rhs)` | 同 lane 掩码的布尔 XOR | BOOL MaskReg | BOOL |
| `mask_not(value)` | 掩码布尔取反 | BOOL MaskReg | BOOL |

`predicate` 必需，仅接受 `"eq"`（等于）、`"ne"`（不等于）、`"lt"`（小于）、
`"le"`（小于等于）、`"gt"`（大于）、`"ge"`（大于等于）。整数比较遵循向量 dtype
的 signedness。seed 与比较输入、输出具有相同 lane 数。掩码逻辑没有额外的
predication mask 或 `pmode`；`select` 使用两个值操作数，mask 表示选择条件，
不是非激活 lane 处理策略。

`gather_lanes` 对应 PTOAS `vselr`，在寄存器内重排元素，不访问内存。
支持的源向量与索引 lane 数取决于元素存储位宽：

| 元素位宽 | 支持的 lane 数 | 索引 dtype |
| -------- | -------------- | ---------- |
| 8 位 | 64、128、256 | INT8、UINT8 |
| 16 位 | 64、128 | INT16、UINT16 |
| 32 位 | 64 | INT32、UINT32 |

## 归约

| API 签名 | 参数与行为 | 返回值 | Dtype |
| -------- | ---------- | ------ | ----- |
| `reduce_sum(value, mask, *, reassoc=False, group_size=None)` | 对激活 lane 求和。`reassoc=True` 允许硬件树形求和重结合，浮点输入必须显式开启；整数输入可使用任意布尔值。 | 单 lane VecReg，或每组一个 lane | `F + I16_32` |
| `reduce_max(value, mask, *, group_size=None)` | 求激活 lane 的最大值 | 单 lane VecReg，或每组一个 lane | `F + I16_32` |
| `reduce_min(value, mask, *, group_size=None)` | 求激活 lane 的最小值 | 单 lane VecReg，或每组一个 lane | `F + I16_32` |
| `group_broadcast(value, *, lanes)` | 将分组归约结果广播回各自分组 | 含 `lanes` 个 lane、同 dtype 的 VecReg | `F + I16_32` |

`value` 为源向量；必需的 `mask` 与**源向量**同 lane，而非单 lane。
非激活输入对 sum 贡献 0，对 max 贡献类型最小值或负无穷，对 min 贡献类型最大值
或正无穷。不做隐式扩宽。结果仍是局部寄存器，可用 `store` 写出结果元素。

省略 `group_size` 时按整个向量归约为单 lane。给出 `group_size` 时，源向量按该
宽度切分为等长连续分组，每组产生一个 lane 并按组序排列，对应带 group count 的
PTOAS `vcadd`/`vcmax`/`vcmin`。组宽必须整除源 lane 数，且得到的**组数**必须为
1、2、4 或 8：A5 没有其他分组 layout，因此 16 组的请求在 trace 阶段即被拒绝，
而不是留到 layout 分配时失败。64 lane 上的 MX group-32 量化即 `group_size=32`，
得到两组。

`group_broadcast` 正是消费这种结果，把每组的值写回该组的所有 lane，使每组的
scale 能直接除回它来源的激活值。组数取自源 lane 数，不单独传入——单独传入只可能
与操作数矛盾。`lanes` 是广播回的宽度，必须是组数的整数倍。

```python
with pl.reg_scope():
    x = pl.reg.load(src, lanes=64)
    mask = pl.reg.full_mask(lanes=64)
    # 每 32 个元素的 MX 分组求一个 amax，再广播回该组的 32 个 lane。
    amax = pl.reg.reduce_max(pl.reg.abs(x, mask), mask, group_size=32)
    scale = pl.reg.group_broadcast(amax, lanes=64)
    scaled = pl.reg.div(x, scale, mask)
    out = pl.reg.store(dst, value=scaled)
```

```python
with pl.reg_scope():
    x = pl.reg.load(src, lanes=64)
    mask = pl.reg.active_mask(32, lanes=64)
    y = pl.reg.adds(x, 1.0, mask)
    positive = pl.reg.compares(y, 0.0, mask, predicate="gt")
    selected = pl.reg.select(positive, y, x)
    total = pl.reg.reduce_sum(selected, mask, reassoc=True)
    out = pl.reg.store(dst, value=total)
```

## Dtype 转换

`convert(value, *, dtype, rounding=None, saturate=None)` 返回与输入同 lane、
具有目标 `dtype` 的 VecReg。`value` 为源 VecReg。`rounding` 指定受支持收窄转换
的舍入策略，`saturate` 显式指定是否饱和。它们不是通用开关，仅下表中的转换对可接受。

| 源类型 | 目标类型 | `rounding` | `saturate` |
| ------ | -------- | ---------- | ---------- |
| FP16 | FP32 | 必须省略 | 必须省略 |
| BF16 | FP32 | 必须省略 | 必须省略 |
| FP32 | FP16 | R/A/Z；DSL 默认 R | 必需：True 或 False |
| FP32 | BF16 | R/A/Z；DSL 默认 R | 必需：True 或 False |
| FP32 | FP8E4M3FN | R/A/Z；DSL 默认 R | 必需：True 或 False |
| FP32 | INT32 | R/A/F/C/Z；DSL 默认 Z | 必需：True 或 False |
| INT32 | FP32 | 必须省略 | 必须省略 |
| INT8 | FP16 | 必须省略 | 必须省略 |
| INT8 | INT16 | 必须省略 | 必须省略 |
| INT16 | INT32 | 必须省略 | 必须省略 |
| UINT8 | UINT16 | 必须省略 | 必须省略 |
| UINT16 | UINT32 | 必须省略 | 必须省略 |
| INT32 | INT8 | 必须省略 | 必需：False |

R 为最近偶数舍入；A 为远离零舍入；F 为向负无穷舍入；C 为向正无穷舍入；
Z 为向零舍入。FP32→FP16 收窄不支持 H 模式，会在 codegen 前拒绝。
`saturate=True` 表示饱和转换，限制到目标类型范围；`saturate=False` 表示非饱和转换。
同 dtype 的 DSL 转换在两个可选属性均省略时返回原寄存器。
其余 dtype 转换对均拒绝。原始 IR/Builder 调用必须在要求舍入属性
的转换中显式传入归一化后的 rounding；上表默认值由 DSL 封装提供。

FP8E4M3FN 转换保留逻辑 lane 数并产生紧凑的 FP8 值。使用
`interpret_cast(result, dtype=pl.UINT8)` 可通过整数 Tile 写出原始字节。
该转换不开放 FP8 算术、直接寄存器加载/存储或反向数值转换。

## 扩展 VMI 操作

以下 API 位于 `pl.reg`，发射对应的 `pto.vmi.v*` 操作。除 `mula`、`intlv`、`dintlv` 的 mask 可选外，
其余操作均要求执行掩码。算术 `pmode` 为 `"zero"`；进借位操作和 `scatter` 不暴露
`pmode` 参数。除表中特别说明外，向量输入的 dtype、符号性和 lane 数必须一致。
已有 Scalar SSA 不会被隐式转换；`axpy`/`lrelu` 的 Python 标量字面量按向量 dtype 归一化。

| API（位置参数） | VMI op | 结果和约束 |
| --------------- | ------ | ---------- |
| `expdif(x, max_value, mask)` | `vexpdif` | 融合 `exp(x - max_value)`；FP16/FP32 输入，始终返回 FP32 |
| `axpy(x, acc, alpha, mask)` | `vaxpy` | 融合 `alpha * x + acc`；FP16/FP32，alpha 为标量 |
| `mula(acc, lhs, rhs, mask=None)` | `vmula` | 融合 `acc + lhs * rhs`；FP16/BF16/FP32 或有符号/无符号 16/32 位整数 |
| `lrelu(x, slope, mask)` | `vlrelu` | Leaky ReLU；FP16/FP32，slope 为标量 |
| `prelu(x, alpha, mask)` | `vprelu` | PReLU；FP16/FP32，alpha 为逐 lane 向量 |
| `mull(lhs, rhs, mask)` | `vmull` | 乘积的两个 32 位部分 `(low, high)`；INT32/UINT32，64/128/256 lanes |
| `addc(lhs, rhs, mask)` | `vaddc` | `(result, carry)`；INT32/UINT32 |
| `subc(lhs, rhs, mask)` | `vsubc` | `(result, not_borrow)`；INT32/UINT32 |
| `addcs(lhs, rhs, carry_in, mask)` | `vaddcs` | `lhs + rhs + carry_in` 的 `(result, carry_out)`；INT32/UINT32 |
| `subcs(lhs, rhs, carry_in, mask)` | `vsubcs` | `lhs - rhs - (1 - carry_in)` 的 `(result, not_borrow_out)`；INT32/UINT32 |
| `gather(tile, offsets, mask, base_offset=0)` | `vgather` | 带掩码的索引 UB 读取；非活动结果 lane 为零 |
| `scatter(tile, offsets, value, mask, base_offset=0)` | `vscatter` | 带掩码的索引 UB 写入；返回目标 Tile 别名 |
| `intlv(lhs, rhs, mask=None)` | `vintlv` | 交织后的 `(low, high)`；lane 数必须为偶数 |
| `dintlv(lhs, rhs, mask=None)` | `vdintlv` | 解交织后的 `(even, odd)`；lane 数必须为偶数 |

未知索引或动态读取不会关闭同一分配上已知静态读取的初始化检查；调用者仍需保证
未知读取自身的安全。

双结果通过一个返回元组的 IR Call 和两个 SSA 结果表达。元组、其投影和进借位掩码
都只能在定义它们的 RegScope 中使用。索引双结果前必须先绑定调用，例如
`pair = pl.reg.mull(a, b, mask)`，再取 `low = pair[0]`；不支持直接写
`pl.reg.mull(a, b, mask)[0]`。跨 `if` 合流或循环传递时，必须先拆包为独立
寄存器，例如 `low, high = pl.reg.mull(a, b, mask)`；校验器拒绝元组类型的
合流结果和循环携带值。`mull` 的两部分保留输入的符号性。
减法的进位掩码以 **true 表示无借位**，与控制执行的 mask 是两个不同参数。
PyPTO 直接生成带 mask 的 interleave。`intlv`/`dintlv` 两个输出的非活动 lane
均为零，活动 BF16 值保持原始位模式。省略 mask 时，所有 lane 均活动，并保留融合机会。

`tests/st/runtime/vpto/test_reg_interleave_onboard.py` 检查显式和省略 mask 时的两个输出，
包括原始位模式和保护区域。默认运行覆盖不短于单个物理寄存器的向量。
短向量已用 PTOAS v0.77 通过编译检查，A5 数值验证仍待完成。
可用 `--lanes 8` 执行定向硬件回归。先在不占卡时构建，再在 A5 上执行产物：

```bash
task-submit --no-device --run \
  'python tests/st/runtime/vpto/test_reg_interleave_onboard.py --build build/interleave'
task-submit --device auto --run \
  'python tests/st/runtime/vpto/test_reg_interleave_onboard.py --worker build/interleave "$TASK_DEVICE"'
```

```python
# Inside a reg_scope, with matching UINT32 registers lo_a/lo_b/hi_a/hi_b:
lo, carry = pl.reg.addc(lo_a, lo_b, mask)
hi, carry_out = pl.reg.addcs(hi_a, hi_b, carry, mask)
product_lo, product_hi = pl.reg.mull(lo_a, lo_b, mask)
interleaved_lo, interleaved_hi = pl.reg.intlv(lo_a, lo_b, mask)
a, b = pl.reg.dintlv(interleaved_lo, interleaved_hi, mask)
```

索引访存的 `offsets` 为 UINT16/UINT32 VecReg，lane 数必须与值/结果及 mask 一致。
索引按**元素**计数，相对于 Tile view 的起点。8/16 位 Tile 元素使用 UINT16
offsets，32 位 Tile 元素使用 UINT32 offsets，与 PTOAS 的物理索引路径保持一致。
gather 将 INT8/UINT8 源数据分别零扩展为 INT16/UINT16，其余结果保持 Tile dtype。
scatter 的值必须匹配 Tile dtype。已有的连续 Vec Tile 几何约束仍然适用。

调用者负责保证活动索引不越界，且 gather 读取的元素已初始化。活动 scatter 索引
必须互不相同。非活动 lane 不访问内存；scatter 保留所有未写入元素，也不扩大
`valid_shape`。索引访存效果保留分配根和掩码，但区间端点未知；与动态 offset 一样，
不构成静态初始化证明。VF helper 保守捕获整个 Tile：gather 为 `read`，scatter 为
`readwrite`。不同 dtype 的捕获通过同一 UB 基址派生出的对应类型指针访问。

索引 codegen 依赖 PTOAS，对同一 vecscope 内未知的索引访存
范围进行同步，包括 RAW、WAR 和 WAW 依赖。该版本包含
[PTOAS #1730](https://github.com/hw-native-sys/PTOAS/issues/1730) 的修复，
因此 PyPTO 不再为每次索引访问发射 `VV_ALL` 屏障。VF 依赖分析由 PTOAS 负责；
PyPTO 不会在生成的 LLVM 中重新检查屏障覆盖。Caller/helper 边界与 GM 同步
保留已有机制。

```python
with pl.reg_scope():
    offsets = pl.reg.index_gen(0, lanes=64, dtype=pl.UINT32)
    mask = pl.reg.active_mask(32, lanes=64)
    values = pl.reg.gather(src, offsets, mask)
    out = pl.reg.scatter(dst, offsets, values, mask)
```

## IR Builder 与查询

| API | 含义 |
| --- | ---- |
| `with ib.reg_scope(span=None, *, auto_sync=True, native=False):` | 在 Python IRBuilder 中打开寄存器区域。 |
| `ib.emit_reg(name, op_name, args, kwargs=None, span=None)` | 构造命名操作。`op_name` 是如 `"reg.add"` 的注册名称；`args` 为 IR 操作数，`kwargs` 为操作属性。 |
| `ib.emit_reg_mem_bar(kind, span=None)` | 构造无结果的显式 UB 同步屏障。 |
| `ib.emit_reg_store(name, tile, value, offset=0, mask=None, span=None, *, block_stride=None)` | 构造命名 store，返回目标 Tile 别名。 |
| `ir.get_reg_op_spec(name)` | 查询操作的区域要求和 VMI emitter。 |
| `ir.get_reg_memory_effect(call)` | 查询 load/store 的目标 Tile、已知根分配、读写类型、半开元素区间和可选 mask；未知根或区间端点仍表示未知。 |

C++ IRBuilder 提供 `BeginRegScope`、`EmitReg`、`EmitRegStore`、`EndRegScope`。
Builder 与 DSL 共享类型及参数检查。完成函数时检查区域归属；`RegScopeValid`
验证手工 IR，`MixedKernelExpanded` 检查最终 AIV placement。内存效应查询保留
mask，但不据此认定整个缓冲区已初始化或访问区间互不重叠。

打印时在 scope 上保留 `auto_sync=False` 和 `native=True`。其他非空属性或非空
`name_hint` 会导致打印报错；二进制序列化保留这些字段。空区域打印为 `pass`。
以上 API 列表表示当前前端支持范围；PTOAS 中没有对应上述 API 的指令尚未通过
`pl.reg` 暴露。

## 位重解释、直方图与分组掩码

| API | PTOAS 算子 | 契约 |
| --- | ---------- | ---- |
| `interpret_cast(value, *, dtype)` | `vinterpret_cast` | 保留所有位，不做数值转换。源和目标均支持 `T`；结果 lane 数为 `input_lanes * input_bits / output_bits`，必须整除。 |
| `chist(acc, source, mask)` | `vchist` | `result[b] = acc[b] + count(active source <= b)`，按 65536 取模。 |
| `dhist(acc, source, mask)` | `vdhist` | `result[b] = acc[b] + count(active source == b)`，按 65536 取模。 |
| `create_group_mask(active_elems_per_group, *, num_groups, group_size)` | `create_group_mask` | 返回 `num_groups * group_size` lane 的 BOOL MaskReg；当 `i % group_size < active_elems_per_group` 时，第 `i` lane 有效。 |

直方图的 `source` 为 UINT8 VecReg，掩码 lane 数必须与**源**一致。
累加器与结果独立使用 128 或 256 lane 的 UINT16 VecReg，分别覆盖 bin 0–127 或 0–255。
掩码不控制累加器 bin；空掩码保留原累加器值。两个算子均返回新寄存器值，不修改 `acc`。
累积直方图的 bin 128–255 也包含小于 128 的源值计数。结果可通过 `pl.reg.store`
原地写回已读取的累加器区域；PTOAS 根据访存与 SSA 数据依赖确定顺序。
A5 回归覆盖两个直方图算子的 128/256-bin 原地更新，包括累加器溢出回绕。

分组掩码的组数和组大小必须为正静态整数。每组活动数量为 INDEX 标量或整数字面量，
范围为 `[0, group_size]`。支持动态数量，由调用者保证范围；常量数量会在构造时检查。
这四个 API 均无 `pmode` 参数，自身不访问 UB 内存。

PTOAS 0.70 的**动态**分组掩码要求 `group_size` 为 2 的幂，且消费者使用
32 位元素（b32 掩码粒度）。字节和半字消费者（包括直方图）必须使用常量活动数量；
其他动态粒度会被 PTOAS lowering 拒绝。VF helper 支持混合 Tile dtype：
从同一 UB 基址派生对应类型的指针，偏移按各 Tile 自身的元素位宽计算。

可选 `base_offset` 是以源/目标元素为单位的 INDEX 标量偏移，在应用无符号向量索引之前
叠加到 Tile view 原点。它移动 UB 指针，不广播偏移或扩宽索引。
调用方仍须保证动态活动地址的边界及初始化；非活动 lane 不访问内存。
现有调用默认使用零基址偏移。

`intlv` 和 `dintlv` 可省略 mask，表示所有 lane 均活动。
显式 mask 仍保证非活动结果为零；省略 mask 无需额外的结果清零选择指令。
相邻的两次 BF16 load 后接无 mask 的 `dintlv` 因此可降低为 A5 `DINTLV_B16`，
后续 `convert` 自动选择所需 EVEN/ODD 部分。物理转换部分仍由后端决定。

## 保序压缩与连续写出

`squeeze(value, mask)` 将选中的 lane 按原顺序压到寄存器开头。输入必须是一个
完整的 256 字节 UINT16 或 UINT32 寄存器，分别为 128 或 64 个 lane，并提供同
lane 数的 mask。只有选中元素组成的前缀有定义；写回时应使用选中数量对应的
前缀 mask。该操作保持整数位模式，包括大于 `2**24` 的 UINT32。

```python
with pl.reg_scope():
    value = pl.reg.load(src, lanes=64)  # UINT32 Tile
    keep = pl.reg.active_mask(9, lanes=64)
    packed = pl.reg.squeeze(value, keep)
    out = pl.reg.store(dst, value=packed, mask=keep)
```

多批数据需要在 UB 中连续追加时，显式使用
`squeeze_store_init/step/flush`。通过 `pl.range(..., init_values=...)`
携带不透明状态，并在循环体中 yield 更新后的状态：

```python
with pl.reg_scope(native=True):
    init_ureg = pl.reg.squeeze_store_init(dst, offset=7)
    for i, (curr_ureg,) in pl.range(0, 4, init_values=(init_ureg,)):
        value = pl.reg.load(src, offset=i * 64, lanes=64)
        keep = pl.reg.active_mask(9, lanes=64)
        next_ureg = pl.reg.squeeze_store_step(curr_ureg, dst, value, keep)
        final_ureg = pl.yield_(next_ureg)
    written = pl.reg.squeeze_store_flush(final_ureg, dst)
    # dst[7:43] contains four selected prefixes, without gaps.
```

目标参数是 Tile，不是指针。只有 `init` 接收 `offset`，它在初始化时求值一次，
以元素为单位、相对 Tile 连续视图起点计算，并叠加已有的 View `start_offset`。
例如视图起点为 32、`offset=7` 时，从底层分配的第 39 个元素开始写。
`step` 和 `flush` 不接收 offset，直接延续已有状态链。每次 `step` 使用与 `init` 相同的目标，
mask 的 lane 数必须与 VecReg 一致。追加整个寄存器或已知前缀时，分别使用 full
mask 或 prefix mask。squeeze 结果的尾部未定义，因此 mask 只能选择其中已定义的前缀。

`init` 初始化 UReg 追加状态并返回不透明的 `RegKind.UReg` 值；它不初始化目标
Tile。每次 `step` 必须消费当前循环携带的 UReg 值并返回后继值；循环通过 `init_values`
和 `pl.yield_` 携带它。循环外消费 `pl.yield_` 绑定的循环结果（上例为 `final_ureg`），
循环结束后恰好调用一次 `flush`。
后端将状态链降低为一次 alignment 初始化、循环携带的 `!pto.align`、每轮
`vstur "POST_UPDATE"` 和最后一次 `vstar`。`flush` 返回目标 Tile 的 alias，不会
清零未写容量。调用方负责保证输出容量，并只读取已写入的前缀。

输入可能没有任何数据块，例如某个批次没有候选元素。循环执行零次时，初始状态原样
传给循环结果。已经初始化的链仍然需要恰好一次 flush，即使一次 step 都没有执行；
此时追加零个元素，目标内容保持原值。没有 step 的顺序语句也遵循相同规则：

```python
with pl.reg_scope(native=True):
    ureg = pl.reg.squeeze_store_init(dst, offset=7)
    written = pl.reg.squeeze_store_flush(ureg, dst)
```

这样，运行时数据块数量为零或非零都使用同一套 init/step/flush 契约，用户无需另写
条件分支。编译器可以删除已经证明为空的循环，同时转发其初始状态。

PTOAS 0.77 仍将 stored 形式的 VSQZ 标为纯操作，但该指令会推入供 VSTUR
消费的 SQZN 计数。在 PTOAS 完成副作用建模前，验证器拒绝 value 和 mask 都是
循环不变量的写入：至少一个输入必须在当前循环内加载或由循环变量推导。
输入也可以来自循环携带的寄存器，由 `pl.yield_` 用这样的值更新。
同一 RegScope 内多个 step 也不能复用等价的 value/mask 生产者，以免
公共子表达式消除删掉必需的 SQZN 生产者。这包括对同一 value 使用等价的 full mask
或 prefix mask 重复写出，即使这些 mask 分别构造。这是保守的后端限制，追加语义保持不变。

在 A5 上使用 PTOAS 0.77 时，避免在同一 RegScope 内用相同 mask 追加两次从同一未修改地址
加载的值。PTOAS 可能合并这两个 stored VSQZ，导致其中一个 VSTUR 缺少 SQZN 计数。
也应避免在同一 RegScope 内关闭后重新开启流，因为 PTOAS 可能把第二次 AR 重置调度到
第一次 VSTAR flush 之前。受影响用例及后续验证见
[PTOAS #1779](https://github.com/hw-native-sys/PTOAS/issues/1779)。

目标必须是静态一维 Vec Tile，或可以按行优先顺序无复制展平的二维 Vec Tile；
dtype 与输入一致且为 UINT16/UINT32。默认连续布局和显式连续 stride 都可以：
`[2, 64]` 配合 stride `[64, 1]` 合法，stride `[96, 1]` 有行间空隙，会被拒绝。
长度为一的轴不产生间隔，单行、单列视图也支持。首版不支持 boxed、padding、compact
和非退化列优先视图。仅改变 shape 不会让带间隔的视图变连续。
offset 包含 Tile View 起点；入口检查静态容量边界。
与动态 register store 相同，调用方必须保证运行时
`offset + sum(selected_lanes) <= dst_capacity`。接口不会截断或扩容；写出序列
前后的字节保持原值，后续会读取的尾部需要事先初始化。

状态是局限于同一 RegScope 的线性值：每次只消费当前状态，用 step 的结果替换它，
并在退出区域前恰好 flush 一次。状态不能通过函数返回或容器逃逸。活动链期间不得
读写、重绑定目标或其别名。流的开启点只能被顺序 for 循环包围，通过 `init_values` 和
`pl.yield_` 显式携带状态，最多嵌套四层。活动链要求直线控制流或顺序 for 循环，
通过 lane mask 筛选。

当前基于 AR 的 lowering 同时支持一条**活动**追加链。IR 支持 flush 后开启下一条链，
实际执行受上述 PTOAS 0.77 调度限制。Ascend C 分别规定 load 和 store 非对齐寄存器上限为四个。
squeeze-store 还使用共享 AR 状态。支持多条同时活动的 squeeze 链需要独立管理游标与计数，并验证后端；
仅增加资源数量限制不能提供这一能力。参见[寄存器数量限制](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/reg_vector_compute/register_data_types/UnalignRegForLoad-UnalignRegForStore.md)
和 [squeeze-store 的 AR 语义](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta2/API/ascendcopapi/atlasascendc_api_07_0324.html)。

squeeze-store 的对齐状态使用物理 micro 指令，必须在 `pl.reg_scope(native=True)` 中使用。当前支持完整 256 字节 load/store、
标量广播、索引生成、位重解释、比较、选择、mask 生成与逻辑运算及 squeeze；
其他操作和非完整寄存器形状会明确报错。验证器会拒绝含流的 scope 内的 `reg.add`。
默认 scope 继续使用 VMI lowering，纯 `squeeze` 也可走此路径。

### 显式物理寄存器模式（A5）

逐指令移植 native VF kernel 时，使用 `with pl.reg_scope(native=True):`。
整个作用域采用 PTO micro-op ABI，向量寄存器必须为完整的 256 字节，mask
也必须为完整硬件宽度。默认 `native=False` 使用逻辑 VMI；单个算子不会
隐式切换模式。物理专用的 distribution、conversion part 和 strided
post-update store 仅允许在物理作用域内使用。
`native` 与 `auto_sync` 是相互独立的编译期布尔值。

- `load(..., dtype=..., lanes=256, dist="UNPK4")` 读取 64 字节 FP4/FP8
  打包数据，得到 256-lane 寄存器。`BRC_B32` 读取一个 32 位元素，
  `E2B_B16` 读取八个 16 位 group scale，默认的 `NORM` 读取整个寄存器。
- `convert(x, mask, dtype=..., lanes=..., part="PART_P0")` 支持 FP4 到 BF16
  或 FP8 到 FP32。FP32 到 BF16 使用 `PART_EVEN` 或 `PART_ODD`，舍入
  `R` 且关闭饱和，结果保留 128 个物理 lane。
- 两个 BF16 物理寄存器可用 `dintlv` 收集偶数 lane；单个寄存器可用
  `store(..., dist="PK_B32")` 紧凑写出偶数 lane。`ONEPT_B32`
  写出 lane 0。省略 `dist` 时按 mask 正常写回。
  `PK_B32` 使用 64-lane 的 32 位 mask，而非 128-lane mask。
- `reduce_max(x, mask)` 和 `reduce_sum(x, mask, reassoc=True)`
  返回完整的 FP32 寄存器，归约结果位于 lane 0。

偏移量以 backing Tile 元素为单位。边界与初始化检查遵循实际访存范围，
而非未使用的物理 lane 数量。native scope 只支持明确实现的算子，
其余算子会在验证阶段报错，不会静默切换到逻辑 VMI。

`store(..., block_stride=N, post_update=True)` 在物理作用域内选择 native stride store 的
POST_UPDATE 编码，并丢弃更新后的临时指针，不改变 Tile view。
