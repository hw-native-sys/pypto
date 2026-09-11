# PyPTO IR 类型与示例

本文档介绍类型 (Type) 系统并提供实用的使用示例。

## 类型系统

### ScalarType

表示原始标量类型。

```python
from pypto import DataType, ir

int_type = ir.ScalarType(DataType.INT64)
float_type = ir.ScalarType(DataType.FP32)
```

**支持的 DataType：** INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, FP16, FP32, FP64, BOOL, INDEX, TASK_ID

> **注意：** `INDEX` 是用于索引计算（循环变量、维度、偏移量、步长）的独立整数类型。它拥有自己的类型代码和字符串表示（`"index"`）。虽然语义上与 `INT64` 类似，但 `INDEX != INT64` —— 它们是不同的类型。在代码生成中，INDEX 和 INT64 之间的隐式类型转换会被抑制。
>
> **注意：** `TASK_ID` 是一个不透明的 64-bit handle（类型代码 `0x50`），表示 runtime 的 `TaskId`。它**不是**数值类型——上面没有任何算术运算。`Scalar[TASK_ID]` 值由 `with pl.manual_scope():` 内的 `pl.submit(...)` 产生（它返回的二元组第二个元素命名 producer task）。Python 字面量 `None` 是 "暂无 producer" 的哨兵——它用作 TaskId 循环 iter_arg 的种子，也可作为 `deps=[None]` 条目；当 `None` 出现在 TaskId 位置时，会下沉为 [`system.task_invalid`](05-operators.md) builtin → `TaskId::invalid()`。TaskId 值通过 `pl.submit(...)` 的 `deps=[tid1, tid2]` kwarg 传入。codegen 把 `TASK_ID` 下沉为 `TaskId`。

### 内部 Buffer 类型

`BufferType` 描述最终设备 IR 中的可变片上缓冲区，直接继承 `Type`，
不包含 `MemRef`、base pointer、地址或运行时表达式字段。
存储身份由定义该缓冲区的 SSA 值表达，所有权由定义算子声明。

```python
buffer_type = ir.BufferType(
    [32, 64], DataType.FP32, ir.Mem.Vec, valid_shape=[-1, 64]
)
multi_type = ir.MultiBufferType(buffer_type, slot_count=2)
```

物理维度目前必须是静态正整数。`valid_shape` 可以是零到物理维度之间的
静态有效长度，也可以用 `-1` 标记由算子操作数提供的运行时有效长度；
省略时使用完整物理形状。布局、以字节为单位的 fractal 大小、padding
和 compact mode 都是显式描述符字段。`MultiBufferType` 描述一次
多缓冲分配中的相同槽位，槽位数必须为正数。控制流类型检查会在分支结果
和循环携带值之间比较完整描述符，包括槽位数及嵌套 tuple 中的元素。

`VoidType` 表示确定没有 SSA 结果，与 `UnknownType` 不同。
Void call 应放在 `EvalStmt` 中，不能绑定变量、用作操作数、放入 tuple，
也不能作为值 yield 或 return。分配大小表达式（包括 `WindowBuffer.size`）
也必须产生一个值。
`Call` 和 `Submit` 的 attrs、kwargs 中的表达式值同样遵循此规则，
在构造时即进行校验，通过 `ir.set_call_attrs` 附加属性时也会校验。

这些类型支持构造、结构比较和二进制序列化。Buffer 类型 dump 使用原生
`pypto.ir.BufferType(...)` 构造表达式，并保留完整描述符。内部 buffer 算子
使用下文契约，`BufferIR` 属性按下文规则验证表示。PTO 代码生成将单独集成。
自动 tile-to-buffer lowering 尚未启用，公开 Tile DSL 和默认流水线仍使用 `TileType`。
目前不支持通过 DSL parser 重新解析完整的 buffer 程序 dump。

#### Buffer 算子契约

算子注册默认为 `OpIRStage::Functional`。内部 buffer 算子显式选择
`OpIRStage::Buffer` 并调用 `set_internal_only()`。结果数量必须声明：
零结果要求 `VoidType`，单结果使用原生类型，多结果使用数量匹配的
`TupleType`。Functional 注册仍按现有契约要求至少一个结果。

每个 buffer 操作数通过 `set_buffer_arg_effect(i, data, metadata)`
分别声明数据与元数据访问；标量操作数使用 `set_buffer_non_memory_arg(i)`。
两个访问维度都使用 `BufferAccess`（`None`、`Read`、`Write`、`ReadWrite`），
缺少声明不会默认视为读取。`set_buffer_result_behavior(...)` 将结果分类为
分配、别名、借用句柄或原生值，void call 声明 `None`。别名和借用结果
需要指定来源操作数。`Allocate` 声明 root 句柄；指定地址的 root 可以相互
重叠，因此它不证明存储独立或已初始化。描述符及内存空间合法性由各算子的
类型推导或显式结果验证检查。

`buffer.alloc` 使用 `f_validate_explicit_type(...)`，不使用类型推导器：
物理描述符只存在于 `Call.type` 中。两种模式互斥，避免在 kwargs 中重复
存储描述符。私有 IR 构造器在 span 前接收结果类型：

```python
from pypto.pypto_core import ir as _ir

span = ir.Span.unknown()
valid_rows = ir.Var("valid_rows", ir.ScalarType(DataType.INDEX), span)
descriptor = ir.BufferType([32, 64], DataType.FP32, ir.Mem.Vec, valid_shape=[-1, 64])
allocation = _ir._create_internal_op_call(
    "buffer.alloc", [ir.MakeTuple([valid_rows], span)], {}, descriptor, span
)
```

第一个操作数始终是 `MakeTuple`，只包含描述符中 `-1` 维度对应的运行时
valid extent，按维度顺序排列。静态描述符使用空 tuple。可选的第二个操作数
是最终有效字节地址，不再叠加 base 或 offset。省略地址表示请求独立存储；
显式零地址是合法的指定地址分配。负常量地址（包括 `-1`）会被拒绝。
两个操作数都属于非内存值，运行时值必须是整数或 `INDEX` 标量。
常量 valid extent 必须介于零和对应物理维度之间；无法静态检查时，运行时
extent 的边界及地址非负性属于构造调用的前置条件。

`buffer.set_validshape(buffer, valid_extents)` 返回 `VoidType`，只写入元数据。
其 `MakeTuple` 操作数包含**所有**维度。标记为 `-1` 的维度可在物理边界内
变化；静态 valid 维度必须传入与描述符一致的常量。该操作不改变不可变类型
或 buffer 身份。对于句柄生命周期内会变化的 valid 维度，lowering 必须提前
选择动态描述符。

`OpRegistry::ValidateBufferCall` 按创建调用时的同一 schema 检查已有 call，
包括其原始结果类型和 kwargs。`BufferIR` 属性在设备函数中应用此检查，并
拒绝逻辑 tile、隐式 buffer 别名以及携带 buffer 的控制流结果。
它组合 SSA、先定义后使用及赋值类型检查，建立表示层契约；存储生命周期、
重叠和初始化证明属于后续验证。

首批 `buffer.copy(src, dst)` 和 `buffer.mul(lhs, rhs, dst)` 写入显式
destination 并返回 `VoidType`，目前要求所有参数的 Vec buffer 描述符
相同。写效应并不表示所有字节都已初始化。允许输入和 destination 是同一
句柄；构造调用前需要保证运行时 valid extent 一致，并完成部分重叠 view
的合法化。现有 Functional 阶段的
`ArgEffect` 查询会显式拒绝 buffer 算子，buffer 消费方必须使用
`GetBufferArgEffect`。

### TensorType

带可选内存引用 (MemRef) 的多维张量 (Tensor)。

```python
span = ir.Span.unknown()

# Tensor with shape [10, 20]
shape = [ir.ConstInt(10, DataType.INT64, span), ir.ConstInt(20, DataType.INT64, span)]
tensor_type = ir.TensorType(shape, DataType.FP32)

# Tensor with MemRef: base allocation, byte offset within it, size in bytes
memref = ir.MemRef("mem_ddr_0", 0, 800)
tensor_with_memref = ir.TensorType(shape, DataType.FP32, memref)
```

`TensorType.memory_space` 始终是 `ir.Mem.DDR`。`MemRef` 标识一块分配
(`base_`) 以及其中的一段字节区间 (`byte_offset_`、`size_`)；内存空间不再
存储在 `MemRef` 本身上。完整字段列表见
[内存引用 (MemRef)](01-hierarchy.md#memref)。

### DistributedTensorType

`DistributedTensorType` 是 `TensorType` 的精确 `ObjectKind` 子类，作为 chip
orchestrator / InCore 形参的类型注解，用来切片由 `CommDomainScopeStmt` 划分的 HCCL window buffer。
它的存在让跨 rank op 的 verifier（后续 milestone 引入）可以静态拒绝普通的
`Tensor` 实参 —— `As<TensorType>` **不会**匹配 `DistributedTensorType`
（精确 `ObjectKind` 匹配语义，见
[ir-kind-traits.md](../../../../.claude/rules/ir-kind-traits.md)），跨 rank op 用
`As<DistributedTensorType>` 派生。

DSL 形式是 `pld.DistributedTensor[[shape], dtype]`:

```python
import pypto.language.distributed as pld
import pypto.language as pl

@pl.function(type=pl.FunctionType.InCore)
def kernel(self, data: pld.DistributedTensor[[256], pl.FP32]): ...
```

IR 层：

```python
t = ir.DistributedTensorType([64], DataType.FP32)
assert isinstance(t, ir.TensorType)            # C++ 继承关系保留
# As<TensorType>(t) → null；As<DistributedTensorType>(t) → 转型成功
```

分配侧的元数据（每 rank 大小、host staging 标志）挂在 `pld.tensor.alloc_window_buffer`
op 所绑定的 `ir.WindowBuffer`（`Var` 子类）上。通过
`pld.tensor.window(buf, [shape], dtype=...)` 物化的切片在
`DistributedTensorType.window_buffer` 上保留指向源 `WindowBuffer` 的可选反向
引用，从而让两个 shape/dtype 相同但分配来源不同的切片在结构上保持不同。
用户在签名中写的 `pld.DistributedTensor[[shape], dtype]` 不填该字段（为
`None`）。Tile 类型没有 distributed 变体；跨 rank op 始终作用在
`DistributedTensor` 上。

**在 window 上做本地计算。** 在 InCore scope 内，一个 window 切片*就是*本 rank 的
本地 GM，因此普通 tensor op 可以像读写任何 GM tensor 一样读写它。这些 op 用
[`AsTensorTypeLike`](../../../../include/pypto/ir/kind_traits.h)（同时匹配两种
kind）而不是精确匹配的 `As<TensorType>` 来匹配操作数。结果类型取决于该 op 产生的
是 window 的*视图*还是*新数据*：

| 接受 window 的 op | 结果 kind |
| ----------------- | --------- |
| `tensor.slice`、`tensor.assemble`、`tensor.view`、`tensor.write` | `DistributedTensorType` —— 仍然是同一个 comm-group 分配上的视图 |
| 逐元素与一元族、各类 reduction、`tensor.matmul`、`tensor.matmul_acc`（仅 `lhs` / `rhs`） | 普通 `TensorType` —— 结果是新产生的本地数据 |
| `tensor.read` | `ScalarType` —— 单个元素，没有视图 |

两处已记录的拒绝：`tensor.reinterpret_view` 直接拒绝 window；`tensor.matmul_acc`
的 **`acc`** 操作数必须是普通 `TensorType` —— 只有矩阵单元会写 L0C，因此不存在从
window 到 Cube 累加器的数据通路。请先在本地累加，再把结果存回 window。

还有不少读写普通 GM 的 tensor op 目前仍然拒绝 window（全部 broadcast、`reshape`、
`transpose`、`concat`、gather / scatter 族等）。
`tests/ut/ir/operators/test_window_operand_acceptance.py` 保存了逐算子的权威分类，
并负责保证它与实现一致。

### 带 TensorView 的 TensorType

带有布局和步长信息的张量，用于优化内存访问。

```python
# Create tensor with tensor view (stride/valid_shape accept int or Expr)
tensor_view = ir.TensorView(stride=[1, 128], layout=ir.TensorLayout.ND)
tensor_with_view = ir.TensorType([128, 256], DataType.FP32, memref=None, tensor_view=tensor_view)

# With valid_shape
tensor_view = ir.TensorView(stride=[1, 128], layout=ir.TensorLayout.ND, valid_shape=[64, 128])

# With pad mode for out-of-valid-shape accesses (symmetric with TileView)
tensor_view = ir.TensorView(
    stride=[1, 128], layout=ir.TensorLayout.ND, valid_shape=[64, 128], pad=ir.PadValue.zero
)

# Different layouts
nd_view = ir.TensorView(stride=[1, 128], layout=ir.TensorLayout.ND)  # ND layout
dn_view = ir.TensorView(stride=[1, 128], layout=ir.TensorLayout.DN)  # DN layout
nz_view = ir.TensorView(stride=[1, 128], layout=ir.TensorLayout.NZ)  # NZ layout

# Expr values also accepted (e.g., symbolic dimensions)
stride = [ir.ConstInt(1, DataType.INT64, span), ir.ConstInt(128, DataType.INT64, span)]
tensor_view = ir.TensorView(stride=stride, layout=ir.TensorLayout.ND)

# Tensor with both MemRef and TensorView
memref = ir.MemRef("mem_ddr_1", 0, 16384)
tensor_with_both = ir.TensorType([128, 256], DataType.FP16, memref=memref, tensor_view=tensor_view)
```

**TensorLayout 值：**

- `ND`：ND 布局
- `DN`：DN 布局
- `NZ`：NZ 布局

**TensorView 字段：**

- `stride`：每个维度的步长
- `layout`：`TensorLayout.ND` / `DN` / `NZ` / `MX_A_ZZ` / `MX_B_NN`
- `valid_shape`：可选的有效区域维度（为空表示使用完整 shape）
- `pad`：`PadValue.null`（默认）/ `zero` / `max` / `min`，用于访问超出
  `valid_shape` 部分时的填充模式。与 `TileView.pad` 对称；
  `tensor.slice(..., pad_value=PadValue.zero)` 会写入该字段。

#### Canonical TensorView 形式（RFC #1300）

按 RFC #1300 的设计，`(shape, stride, layout)` 三元组在各 pass / verifier /
codegen 之间统一为单一可机械读取的形式：

- `shape` 是**逻辑** shape —— 消费者索引时使用的维度。
- `stride[i]` 是第 *i* 个**逻辑**维递增 1 时的元素步长。
- `layout` 是 `(shape, stride)` 上的派生标签 / 断言，并非独立描述。
  ND / DN 各定义 packed canonical（紧致存储）与 strided 家族（sub-view 继承
  父 stride）两种合法形态。

Packed canonical 公式（`BuildLogicalStridesFromLayout`，见
[`tensor_view_semantics.h`](../../../../include/pypto/ir/transforms/utils/tensor_view_semantics.h)）：

| Layout | Packed canonical |
| ------ | ---------------- |
| `ND` | `stride[n-1] = 1; stride[k] = stride[k+1] * shape[k+1]` |
| `DN`（`n ≥ 2`） | `stride[n-2] = 1`；`stride[n-1] = shape[n-2]`；`stride[n-3] = shape[n-2] * shape[n-1]`；外层按行主序 |
| `NZ` | 对*分块*后的 rank-5 shape `[B, C/c0, R/16, 16, c0]` 求行主序 —— 见 [BlockNzTensorViews](../passes/15-block_nz_tensor_views.md) |

**同一 canonical TensorView 的两种写法**：

- **隐式** —— `view.has_value() && view.stride.empty()`：layout 已设但
  stride 为空，消费者按对应 layout 的 packed canonical 解释。
- **显式** —— 每个维度的 stride 都已写出。

[`MaterializeTensorStrides`](../passes/33-materialize_tensor_strides.md) Pass
将所有隐式形态展开为显式 packed canonical，让 codegen 看到单一契约。
`TensorViewCanonical` IRProperty + verifier 强制此不变量：

- **弱模式**（registry 默认，`passes.PropertyVerifierRegistry.verify`）：
  接受 `stride.empty()` 作为隐式 packed canonical。
- **严格模式**（codegen 入口契约，
  `passes.verify_tensor_view_canonical(program, require_materialized=True)`）：
  必须有非空 `view.stride` 且与 layout 家族一致。

两种模式都拒绝 `TensorType` 上*未分块*的 `NZ` shape，并按
`relaxed_symbolic` 语义接受符号 stride。

### TileType

专用张量类型，带可选内存和视图信息，用于硬件优化操作。

```python
# Basic 16x16 tile
shape = [ir.ConstInt(16, DataType.INT64, span)] * 2
tile_type = ir.TileType(shape, DataType.FP16)

# 3D tile (supported at IR level)
shape_3d = [ir.ConstInt(4, DataType.INT64, span),
            ir.ConstInt(16, DataType.INT64, span),
            ir.ConstInt(16, DataType.INT64, span)]
tile_type_3d = ir.TileType(shape_3d, DataType.FP16)

# Tile with MemRef and TileView. TileView is immutable — every field is passed
# to the constructor; valid_shape / stride / start_offset accept int or Expr.
memref = ir.MemRef("mem_left_0", 0, 512)

tile_view = ir.TileView(valid_shape=[8, 16], stride=[1, 16], start_offset=0)

tile_with_view = ir.TileType(shape, DataType.FP16, memref, tile_view, ir.Mem.Left)
```

`TileType.memory_space` 才是 Tile 放置位置的唯一来源。如果 `TileType`
携带 `MemRef`, 请在 `TileType` 自身上显式提供 tile 内存空间。

上面的 `valid_shape` 是一个真正的子区域（`[16, 16]` Tile 中的 `[8, 16]`）。
若 `valid_shape` 与完整 shape 相同则是冗余的，构造函数会将其清空 ——
参见下文的规范化规则。

对于 Python DSL 类型标注，省略的 `TileView` 语法会被规范化为一个隐式
TileView：它由 tile shape 以及（如果存在）tile memory space 推导得到。
像 `pl.TileView()` 这样的冗余显式默认写法，会与省略写法被视为语义等价，
并且在 printer 输出时可能统一成规范形式。`TileView.compact` 记录部分有效的
boxed tile 是采用 PTO 的有效区域紧凑表示（`CompactMode.normal`），还是普通的
物理 box 表示（默认的 `CompactMode.null`）。它只在 fractal 空间——`Left` / `Right` /
`Acc`——有意义，因为它本身描述的就是 N-fractal pitch；`AccCompactValid` 校验器会拒绝
其它空间上的 compact。编译器会为进入 L0A/L0B 的部分 `tile.extract`、以及行窄化的
matmul 累加器（其 L0C pitch 由 `mad` 按 L0A 操作数的有效行数推导）自动设置该字段，
`AutoTileMatmulL0` 还会通过 `tile.create(..., compact=True)` 在它合成的累加器种子上
声明该字段。普通用户代码无需手动选择。

隐式 view 依赖 memory space，构造函数只会针对传入的 space 把 view 折叠成
`nullopt`。凡能确定结果 space 的 `f_deduce_type`，**都必须把该 space 传进来**：
若先针对 `nullopt` 推导、再由 `OpRegistry::Create` 补盖 space，就会按两套不同的
隐式 layout 规范化两次，结果取决于 view 是否恰好折叠（即 `valid_shape` 与 `pad`）。

### ArrayType

片上定长同构 1-D 数组,存放于标量寄存器堆 / C 栈(memory space `ScalarLocal`)。
区别于 `TensorType`(GM/DDR 指针)和 `TileType`(向量/cube 单元状态)。

```python
arr_type = ir.ArrayType(DataType.INT32, 16)       # 16 个 INT32 元素
# DSL 注解形式:
arr: pl.Array[16, pl.INT32]
```

**v1 约束:**

- 元素 dtype 必须是整型(`INT8/16/32/64`、`UINT8/16/32/64`)或 `BOOL`
- 仅支持 rank-1;extent 必须是编译期 `ConstInt`
- 不携带 `MemRef` —— codegen 直接落到 C 栈数组 `dtype name[N]`(无 STL 依赖)
- 不能跨函数边界(由 `ArrayNotEscaped` 验证器强制)

**操作:**

| Op | 语义 | Orchestration(C++) | InCore（`.pto`） |
| -- | ---- | ------------------ | ---------------- |
| `array.create(N, dtype)` | 分配栈数组 | `dtype arr[N] = {0};` | `pto.declare_local_array -> !pto.local_array<NxT>` |
| `array.get_element(arr, i)` → `Scalar` | 读元素 `i` | `dtype v = arr[i];` | `pto.local_array_get arr[i] : !pto.local_array<NxT> -> T` |
| `array.update_element(arr, i, v)` → `Array` | 函数式更新(SSA-pure) | `arr[i] = v;`(LHS 别名到入参) | `pto.local_array_set arr[i], v : !pto.local_array<NxT>, T` |

`array.update_element` 是 `tensor.assemble` 的 SSA-functional 等价物:返回一个新的
`ArrayType` SSA 值,表示"原数组中第 i 个元素被替换为 v"。两条 codegen 路径都把结果
Var 别名到入参数组的存储,emit 原地写入 —— 不复制。

InCore 路径对齐 PTOAS 的栈数组三件套（`pto.declare_local_array` /
`pto.local_array_get` / `pto.local_array_set`)。下标统一下降为 MLIR `index`（源类型
非 `index` 时插入 `arith.index_cast`)，`set` 的值在与元素 dtype `T` 不一致时也会被
cast（verifier 允许把 `index` 类型的值写入整型数组)。

**DSL 下标糖:**

```python
arr = pl.array.create(8, pl.INT32)
arr[i] = v          # desugar 成: arr = pl.array.update_element(arr, i, v)
x = arr[i]          # desugar 成: x = pl.array.get_element(arr, i)
```

`arr[i] = v` 时 parser 把左边变量重绑定,后续读取看到更新后的数组 —— 与
Tensor/Tile 下标写入糖一致。

### TupleType

异构类型元组。

```python
# Scalar tuple: (int, float)
scalar_tuple = ir.TupleType([
    ir.ScalarType(DataType.INT64),
    ir.ScalarType(DataType.FP32)
])

# Nested tuple
nested = ir.TupleType([
    ir.TupleType([ir.ScalarType(DataType.INT64)]),
    ir.ScalarType(DataType.FP32)
])
```

### PipeType

硬件执行流水线或同步屏障。

```python
pipe_s = ir.PipeType(ir.PipeType.S)    # Scalar pipe
pipe_v = ir.PipeType(ir.PipeType.V)    # Vector pipe
pipe_m = ir.PipeType(ir.PipeType.M)    # Matrix pipe
pipe_all = ir.PipeType(ir.PipeType.ALL) # All pipes
```

### UnknownType

未知或待推断类型的占位符。

```python
unknown = ir.UnknownType()
```

### DSL 中的 MemRef 类型注解

MemRef 可以在 `@pl.program` / `@pl.function` DSL 代码中作为位置参数指定在类型注解中：

```python
import pypto.language as pl

@pl.program
class MyProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[64, 64], pl.FP32]):
        # Tile with MemRef and explicit tile memory space
        tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemRef(0, 16384, 0), pl.Mem.Vec] = pl.tile.load(
            x, offsets=[0, 0], shapes=[64, 64]
        )

        # Tensor with MemRef (3-arg: shape, dtype, memref)
        y: pl.Tensor[[64, 64], pl.FP32, pl.MemRef(0, 16384, 1)] = pl.add(x, 1.0)

        # Tensor with layout and MemRef (4-arg: shape, dtype, layout, memref)
        z: pl.Tensor[[64, 64], pl.FP32, pl.NZ, pl.MemRef(0, 16384, 2)] = pl.add(x, 1.0)
```

**`pl.MemRef(addr, size, id)` 参数：**

| 参数 | 类型 | 说明 |
| ---- | ---- | ---- |
| `addr` | `int` | 基地址偏移 |
| `size` | `int` | 内存分配大小（字节） |
| `id` | `int` | 内存缓冲区标识符 |

`TensorType` 注解默认位于 `DDR`。为了兼容旧代码，解析器仍接受
`pl.MemRef(pl.Mem.DDR, addr, size, id)`，但新代码应优先使用 3 参数形式。

**消歧义（3 参数 Tensor）：** 解析器会自动区分 `pl.MemRef(...)` 和
`pl.NZ`/`pl.DN`/`pl.ND` 布局枚举。

**Tile 规则：** 如果在 `pl.Tile[...]` 注解中使用 `pl.MemRef(...)`，必须再
单独提供一个 `pl.Mem.*` 参数来声明 tile 的内存空间。

### MemorySpace 枚举（别名：`Mem`）

| 值 | 说明 |
| -- | ---- |
| `DDR` | 主存储器（片外） |
| `Vec` | 向量/统一缓冲区（片上） |
| `Mat` | 矩阵/L1 缓冲区 |
| `Left` | 左矩阵操作数缓冲区 |
| `Right` | 右矩阵操作数缓冲区 |
| `Acc` | 累加器缓冲区 |
| `Bias` | Bias 缓冲区 |
| `ScalarLocal` | 片上标量寄存器堆 / C 栈(用于 `ArrayType`) |

## Python 使用示例

### 示例 1：构建表达式

```python
from pypto import DataType, ir

span = ir.Span.unknown()
dtype = DataType.INT64

# Variables and constants
x = ir.Var("x", ir.ScalarType(dtype), span)
y = ir.Var("y", ir.ScalarType(dtype), span)
one = ir.ConstInt(1, dtype, span)
two = ir.ConstInt(2, dtype, span)

# Build: ((x + 1) * (y - 2)) / (x + y)
x_plus_1 = ir.Add(x, one, dtype, span)
y_minus_2 = ir.Sub(y, two, dtype, span)
numerator = ir.Mul(x_plus_1, y_minus_2, dtype, span)
denominator = ir.Add(x, y, dtype, span)
result = ir.FloatDiv(numerator, denominator, dtype, span)
```

### 示例 2：控制流（绝对值）

```python
# if (x >= 0) then { result = x } else { result = -x }
x = ir.Var("x", ir.ScalarType(dtype), span)
result = ir.Var("result", ir.ScalarType(dtype), span)
zero = ir.ConstInt(0, dtype, span)

condition = ir.Ge(x, zero, dtype, span)
then_assign = ir.AssignStmt(result, x, span)
else_assign = ir.AssignStmt(result, ir.Neg(x, dtype, span), span)

abs_stmt = ir.IfStmt(condition, then_assign, else_assign, [result], span)
```

### 示例 3：带累加的循环

```python
# for i, (sum,) in pl.range(n, init_values=(0,)):
#     sum = pl.yield_(sum + i)

n = ir.Var("n", ir.ScalarType(dtype), span)
i = ir.Var("i", ir.ScalarType(dtype), span)
zero = ir.ConstInt(0, dtype, span)
one = ir.ConstInt(1, dtype, span)

sum_iter = ir.IterArg("sum", ir.ScalarType(dtype), zero, span)
add_expr = ir.Add(sum_iter, i, dtype, span)
yield_stmt = ir.YieldStmt([add_expr], span)
sum_final = ir.Var("sum_final", ir.ScalarType(dtype), span)

loop = ir.ForStmt(i, zero, n, one, [sum_iter], yield_stmt, [sum_final], span)
```

### 示例 4：带运算符调用的函数

```python
# def matmul(a, b) -> tensor:
#     result = tensor.matmul(a, b, out_dtype=FP32)

shape_m = ir.ConstInt(128, DataType.INT64, span)
shape_k = ir.ConstInt(64, DataType.INT64, span)
shape_n = ir.ConstInt(256, DataType.INT64, span)

a = ir.Var("a", ir.TensorType([shape_m, shape_k], DataType.FP16), span)
b = ir.Var("b", ir.TensorType([shape_k, shape_n], DataType.FP16), span)

matmul_call = ir.op.tensor.matmul(a, b, out_dtype=DataType.FP32)
result = ir.Var("result", ir.TensorType([shape_m, shape_n], DataType.FP32), span)
body = ir.AssignStmt(result, matmul_call, span)

return_types = [ir.TensorType([shape_m, shape_n], DataType.FP32)]
func = ir.Function("matmul", [a, b], return_types, body, span)
```

### 示例 5：包含多个函数的程序

```python
# Helper: square(x) -> int { return x * x }
x = ir.Var("x", ir.ScalarType(dtype), span)
square_result = ir.Var("result", ir.ScalarType(dtype), span)
square_body = ir.AssignStmt(square_result, ir.Mul(x, x, dtype, span), span)
square_func = ir.Function("square", [x], [ir.ScalarType(dtype)], square_body, span)

# Main: sum_squares(a, b) -> int { return square(a) + square(b) }
a = ir.Var("a", ir.ScalarType(dtype), span)
b = ir.Var("b", ir.ScalarType(dtype), span)

program = ir.Program([square_func], "math", span)
square_gvar = program.get_global_var("square")

call_a = ir.Call(square_gvar, [a], span)
call_b = ir.Call(square_gvar, [b], span)
sum_expr = ir.Add(call_a, call_b, dtype, span)

main_result = ir.Var("result", ir.ScalarType(dtype), span)
main_body = ir.AssignStmt(main_result, sum_expr, span)
main_func = ir.Function("sum_squares", [a, b], [ir.ScalarType(dtype)], main_body, span)

program = ir.Program([square_func, main_func], "math", span)
```

### 示例 6：使用 TileType 的内存布局

```python
# 32x32 tile in Left memory, viewing a 16x32 valid region with custom stride
shape = [ir.ConstInt(32, DataType.INT64, span)] * 2
memref = ir.MemRef("mem_left_0", 0, 2048)

tile_view = ir.TileView(valid_shape=[16, 32], stride=[1, 32], start_offset=0)

tile_type = ir.TileType(shape, DataType.FP16, memref, tile_view, ir.Mem.Left)
```

## 类型系统总结

| 类型 | 维度 | 内存信息 | 使用场景 |
| ---- | ---- | -------- | -------- |
| **ScalarType** | 0 | - | 单个值 |
| **TensorType** | N（任意） | 可选 MemRef | 通用张量 |
| **TileType** | N（任意）* | 可选 MemRef + TileView | 硬件优化 Tile |
| **BufferType** | 静态物理维度 | 定义它的 SSA 句柄 | 显式设备存储 |
| **MultiBufferType** | 元素 BufferType | 原生 slot 组 | 相同描述符的 buffer slots |
| **VoidType** | - | - | 确定不存在 SSA 结果 |
| **TupleType** | - | - | 多返回值 |
| **PipeType** | - | - | 硬件同步 |
| **UnknownType** | - | - | 类型推断占位符 |

## 常用模式

**创建常量：**

```python
i32 = ir.ConstInt(42, DataType.INT32, span)
f32 = ir.ConstFloat(3.14, DataType.FP32, span)
```

**创建运算符：**

```python
# High-level API (recommended)
call = ir.op.tensor.matmul(a, b, out_dtype=DataType.FP32)

# Generic operator with kwargs
call = ir.create_op_call("tensor.matmul", [a, b], {"out_dtype": DataType.FP32}, span)
```

**语句序列：**

```python
seq = ir.SeqStmts([stmt1, stmt2, stmt3], span)
```

## 类型检查与转换

```python
# Check expression types
if isinstance(expr, ir.Var):
    print(expr.name_)

# Check type objects
if isinstance(type_obj, ir.TileType):
    # Access tile-specific properties
    shape = type_obj.shape
```

## 相关文档

- [IR 概述](00-overview.md) - 核心概念与设计原则
- [IR 节点层次结构](01-hierarchy.md) - 完整节点类型参考
- [结构比较](03-structural_comparison.md) - 相等性和哈希工具

## 总结

PyPTO 的类型系统提供：

- **标量类型** 用于原始值
- **张量/Tile 类型** 用于带内存布局的多维数据
- **元组类型** 用于异构集合
- **流水线类型** 用于硬件同步

IR 构建 API 支持：

- 通过共享指针创建不可变节点
- 带编译时检查的类型安全操作
- 通过 MemRef 和 TileView 实现硬件感知的内存管理
- 通过 GlobalVar 实现程序内函数调用
- 通过 IterArg 实现循环携带依赖
