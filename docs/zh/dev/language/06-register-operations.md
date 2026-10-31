# 寄存器 API 参考

`pl.reg` 提供在 A5（Ascend950）`pl.reg_scope()` 内使用的逻辑向量和掩码操作。
本文完整列出当前支持的 `pl.reg` 操作、参数、返回类型和 dtype 限制。
下表中的所有操作名称均带有 `pl.reg.` 前缀。

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

`pl.reg_scope()` 不接受参数，没有 captures、结果列表或 yield，也不能嵌套。
前端允许 InCore、AIV 设备上下文，包括 Opaque、Orchestration、Graph 函数内
在 outlining 前显式声明的 InCore 作用域。`ExpandMixedKernel` 后，寄存器区域必须
位于 AIV 函数内。禁止直接在调度代码中使用，也禁止在 RegScope 内嵌套调度作用域。
显式配置非 A5 目标时拒绝；尚未选定目标时，目标检查延迟到编译阶段。

`pl.Reg[lanes, dtype]` 表示向量寄存器（VecReg）。
`pl.Reg[lanes, pl.BOOL, pl.RegKind.MaskReg]` 表示谓词掩码（MaskReg）。
第三个可选类型注解参数为寄存器种类，默认为 `pl.RegKind.VecReg`。
通常直接根据操作推导寄存器类型。`lanes` 是静态正整数，表示逻辑元素数量，
不是张量 shape，也不是字节数。

外部 Tile 和只读 Scalar 是隐式输入。寄存器与局部 Scalar 不能通过区域外使用、
函数返回、循环结果、元组或向外部变量赋值逃逸。Tile 操作数必须源于区域外；
`store` 返回的 Tile 别名可以在区域外使用。每个操作的返回值都必须赋给变量，
包括 `store`；裸 store 会被拒绝。循环 API 和 hardware-loop policy 参见
[语句](01-statements.md)。

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
除转换和归约外，结果保持输入向量 dtype 和 lane 数。不做隐式向量类型提升或
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
| `store(tile, offset=0, value=None, mask=None)` | `tile`：目标 Vec Tile。`offset`：起始元素索引。`value`：必需的 VecReg，虽然 Python 默认值为 None。`mask`：可选写掩码；仅写入激活 lane，保留目标其他元素。 | `tile` 的 Tile 别名 | `T`；value 与 Tile dtype 必须一致 |

offset 以**元素**计，不是字节。静态 offset 必须非负，完整的
`[offset, offset + lanes)` 区间必须位于 Tile 容量内，masked store 也不放宽该限制。
静态 load 必须位于已初始化的 `valid_shape` 内，后续 mask 不能保护未初始化读取。
一维 Tile 可直接使用；多维 Tile 需要显式连续行优先 View，拒绝带 padding、boxed
或 compact 的布局。View 起始 offset 计入地址计算和底层分配边界检查。

当静态检查不能证明违规时，动态 offset 和 valid extent 会传递给后端；调用者负责
运行时边界和初始化。store 不扩大 `valid_shape`，不分配新存储，也不创建快照。
其结果与目标 Tile 具有相同类型，并引用同一存储。

## 向量与掩码生成器

| API 签名 | 参数与行为 | 返回值 | Dtype |
| -------- | ---------- | ------ | ----- |
| `broadcast(scalar, *, lanes, dtype)` | 将 `scalar` 广播到所有 lane。scalar dtype 必须等于 `dtype`。 | VecReg | `T` |
| `index_gen(base, *, lanes, dtype, order="ascending")` | `base` 为标量起始值；第 `i` 个 lane 为 `base + i`。`order` 仅支持 `"ascending"`。base dtype 必须等于 `dtype`。 | VecReg | `F + I`；类型推导时即拒绝 BF16，不进入 codegen |
| `full_mask(*, lanes)` | 激活所有逻辑 lane。 | BOOL MaskReg | BOOL |
| `active_mask(active_lanes, *, lanes)` | 激活前缀 `[0, active_lanes)`。`active_lanes` 必须是 `[0, lanes]` 内的静态 INDEX 整数；零产生全 false 掩码，不支持动态数量。 | BOOL MaskReg | BOOL |

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
| `reduce_sum(value, mask, *, reassoc=False)` | 对激活 lane 求和。`reassoc=True` 允许硬件树形求和重结合，浮点输入必须显式开启；整数输入可使用任意布尔值。 | 同 dtype 的单 lane VecReg | `F + I16_32` |
| `reduce_max(value, mask)` | 求激活 lane 的最大值 | 同 dtype 的单 lane VecReg | `F + I16_32` |
| `reduce_min(value, mask)` | 求激活 lane 的最小值 | 同 dtype 的单 lane VecReg | `F + I16_32` |

`value` 为源向量；必需的 `mask` 与**源向量**同 lane，而非单 lane。
非激活输入对 sum 贡献 0，对 max 贡献类型最小值或负无穷，对 min 贡献类型最大值
或正无穷。不做隐式扩宽。结果仍是局部寄存器，可用 `store` 写出单个结果元素。
当前没有分组归约 API。

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
| FP32 | FP16 | R/A/H/Z；DSL 默认 R | 必需：True 或 False |
| FP32 | INT32 | R/A/F/C/Z；DSL 默认 Z | 必需：True 或 False |
| INT32 | FP32 | 必须省略 | 必须省略 |
| INT8 | FP16 | 必须省略 | 必须省略 |
| INT8 | INT16 | 必须省略 | 必须省略 |
| INT16 | INT32 | 必须省略 | 必须省略 |
| UINT8 | UINT16 | 必须省略 | 必须省略 |
| UINT16 | UINT32 | 必须省略 | 必须省略 |
| INT32 | INT8 | 必须省略 | 必需：False |

R 为最近偶数舍入；A 为远离零舍入；F 为向负无穷舍入；C 为向正无穷舍入；
Z 为向零舍入。H 为 PTOAS 的特定 H 舍入模式，用于 FP32→FP16 收窄，原样传给后端。
`saturate=True` 表示饱和转换，限制到目标类型范围；`saturate=False` 表示非饱和转换。
同 dtype 的 DSL 转换在两个可选属性均省略时返回原寄存器。
其余 dtype 转换对均拒绝，包括 BF16 转换。原始 IR/Builder 调用必须在要求舍入属性
的转换中显式传入归一化后的 rounding；上表默认值由 DSL 封装提供。

## IR Builder 与查询

| API | 含义 |
| --- | ---- |
| `with ib.reg_scope(span=None):` | 在 Python IRBuilder 中打开寄存器区域。 |
| `ib.emit_reg(name, op_name, args, kwargs=None, span=None)` | 构造命名操作。`op_name` 是如 `"reg.add"` 的注册名称；`args` 为 IR 操作数，`kwargs` 为操作属性。 |
| `ib.emit_reg_store(name, tile, value, offset=0, mask=None, span=None)` | 构造命名 store，返回目标 Tile 别名。 |
| `ir.get_reg_op_spec(name)` | 查询操作的区域要求和 VMI emitter。 |
| `ir.get_reg_memory_effect(call)` | 查询 load/store 的目标 Tile、已知根分配、读写类型、半开元素区间和可选 mask；未知根或区间端点仍表示未知。 |

C++ IRBuilder 提供 `BeginRegScope`、`EmitReg`、`EmitRegStore`、`EndRegScope`。
Builder 与 DSL 共享类型及参数检查。完成函数时检查区域归属；`RegScopeValid`
验证手工 IR，`MixedKernelExpanded` 检查最终 AIV placement。内存效应查询保留
mask，但不据此认定整个缓冲区已初始化或访问区间互不重叠。

Python 文本语法不接受区域元数据。打印带非空 `name_hint` 或 `attrs` 的手工
RegScope 会报错，避免丢失元数据；二进制序列化保留这些字段。空区域打印为 `pass`。
以上 API 列表表示当前前端支持范围；PTOAS 中没有对应上述 API 的指令尚未通过
`pl.reg` 暴露。
