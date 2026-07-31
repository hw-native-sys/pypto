# 类型

PyPTO 程序里每个值都带有类型，说明它住在哪里、元素有多宽。写对注解，就是在告诉编译器该分配什么、允许做什么。

> **前置**：[编程模型 § 内存层次](../03-programming-model.md#内存层次)。

## Concept

一条类型注解里编码了三件事，值得分开看 —— 它们的失败方式不同。

**值住在哪里。** `pl.Tensor` 在 DDR，`pl.Tile` 是片上缓冲区，`pl.Scalar` 是寄存器宽度的值。这不是提示：在执行面上做张量操作、在控制面上做 tile 操作，都会被拒绝，而容器类型就是编译器判断的依据。

**元素有多宽。** dtype 常量（`pl.FP16`、`pl.INT32` …）命名一种硬件元素格式。混用是合法的，但从不隐式：没有类型提升，宽度不同的地方必须写 `pl.cast`。

**调用方可以怎么用它。** 参数方向 —— `In`（默认）、`pl.Out[...]`、`pl.InOut[...]` —— 是签名的一部分，不是约定。编译器从方向推导任务依赖，所以方向声明错了得到的是**一张错的依赖图**，而不是一个编译错误。

Shape 默认是静态的，在解析期检查。`pl.dynamic()` 让某一维退出这种检查，代价是编译器本可以从这个维度推出的一切。

## Quickstart：读懂一个签名

```python
import pypto.language as pl

M = pl.dynamic("M")                       # symbolic dimension, fixed per compilation

@pl.jit.incore
def scale_rows(
    x: pl.Tensor[[M, 128], pl.FP16],                    # In (default): read-only, DDR
    acc: pl.InOut[pl.Tensor[[M, 128], pl.FP32]],        # read-write, DDR
    out: pl.Out[pl.Tensor[[M, 128], pl.FP32]],          # write-only, DDR
    factor: pl.Scalar[pl.FP32],                         # scalar, passed by value
):
    ...
```

| 元素 | 读作 |
| ---- | ---- |
| `pl.Tensor[[M, 128], pl.FP16]` | 二维 DDR 数组，`M` 行（运行期值），128 列，半精度 |
| `pl.InOut[...]` | kernel 既读又写 —— 编译器会把它同时排在此前的写者和读者之后 |
| `pl.Out[...]` | kernel 只写。在写入前读一个 `Out` 参数读到的是未定义内存 |
| `pl.Scalar[pl.FP32]` | 单个值，不是缓冲区 |
| `M = pl.dynamic("M")` | 该维编译期未知，每次启动时绑定 |

## Mechanics

### 数据类型

| 常量 | 位宽 | 说明 |
| ---- | ---- | ---- |
| `pl.BOOL` | 1 | |
| `pl.INT4` / `pl.UINT4` | 4 | |
| `pl.INT8` / `pl.UINT8` | 8 | |
| `pl.INT16` / `pl.UINT16` | 16 | |
| `pl.INT32` / `pl.UINT32` | 32 | |
| `pl.INT64` / `pl.UINT64` | 64 | |
| `pl.FP16` | 16 | IEEE 半精度 |
| `pl.BF16` | 16 | Brain float |
| `pl.FP32` | 32 | IEEE 单精度 |
| `pl.FP4` | 4 | 打包的 MXFP4 E2M1×2 |
| `pl.FP8E4M3FN` / `pl.FP8E5M2` | 8 | MXFP8 数据格式 |
| `pl.FP8E8M0` | 8 | MX 块缩放指数 |
| `pl.HF4` / `pl.HF8` | 4 / 8 | 海思浮点格式 |
| `pl.INDEX` | 64 | 索引运算 —— 循环变量、维度 |
| `pl.TASK_ID` | — | `pl.submit` 返回的生产者句柄，见 [作用域与任务](04-scopes-and-tasks.md) |

`dtype.get_byte()` 返回元素的字节数。只要字节数是算出来的而不是写死的字面量，就用它 —— 把元素个数传到期望字节数的地方是一次**静默的欠分配**。

```python
nbytes = 256 * pl.FP32.get_byte()          # 1024, not 256
```

### 容器类型

| 类型 | 住在 | 写法 |
| ---- | ---- | ---- |
| `pl.Tensor[[shape], dtype]` | DDR | `x: pl.Tensor[[64, 128], pl.FP32]` |
| `pl.Tile[[shape], dtype]` | 片上缓冲区（默认 Vec） | `t: pl.Tile[[64, 64], pl.FP32]` |
| `pl.Scalar[dtype]` | 值，不是缓冲区 | `s: pl.Scalar[pl.FP32]` |
| `pl.Array[extent, dtype]` | 核内数组 | `a: pl.Array[16, pl.INT32]` |
| `pl.Tuple[T1, T2]` | — | 多值返回注解 |

`pl.TaskId` 是 `pl.Scalar[pl.TASK_ID]` 的便捷别名。

`pl.Array` 通常是创建出来的而不是注解出来的 —— 数组不跨函数边界，所以注解形式很少见。见 [编译期指令 § 数组](05-directives.md#数组)。

```python
arr = pl.array.create(16, pl.INT32)
arr[i] = value          # array.update_element — functional, rebinds arr
x = arr[i]              # array.get_element
```

### 布局

**`pl.Tensor` 注解请写运行期行主序 shape，不要写布局标记。** 布局是 IR 内部的事，pass 会从实际产生 / 消费各个视图的算子推导出来。

```python
b: pl.Tensor[[N, K], pl.FP32]              # ✅ source shape, no marker
```

```python
b: pl.Tensor[[K, N], pl.FP32, pl.DN]       # ⚠️ deprecated — DeprecationWarning at parse time
```

DN 简写被弃用，是因为它逼着你同时在脑子里维护两套坐标系：IR 逻辑上的 post-view shape 和运行期行主序 shape。矩阵乘需要转置操作数时，给 `pl.matmul` 传 `a_trans=True` / `b_trans=True`，或者按自然布局 load 之后用 `pl.tile.transpose_view(...)`。对一个产生 DN 的算子做切片，切片会自动继承父布局。

`pl.ND` 是默认的行主序布局，不需要写出来。`pl.NZ` 只用于 tile —— 那是硬件 tile 布局，永远不做 `pl.Tensor` 注解。若确实要在 IR 层构造 DN 张量（测试夹具、printed IR 往返），优先用 `pl.TensorView(stride=[...], layout=pl.TensorLayout.DN)`：它强制 stride 显式化，避开隐式坐标翻转。

### 动态 shape

`pl.dynamic(name)` 创建一个符号维度。同一个 `DynVar` 对象在多处注解中使用时指的是同一维 —— 复用这个对象，不要在表示同一个值时再造一个同名的。

```python
M = pl.dynamic("M")

@pl.jit.incore
def rows(x: pl.Tensor[[M, 64], pl.FP32], out: pl.Out[pl.Tensor[[M, 64], pl.FP32]]):
    ...
```

放弃的东西：编译器本可以从这个尺寸做出的任何决策。分块选择、展开因子、静态边界检查都会因此失去信息。所以只在这一维**确实每次启动都不同**时才让它动态 —— 不要为了少写一个数字而用它。

### 参数方向

| 方向 | 语法 | 编译器据此断定 |
| ---- | ---- | -------------- |
| In（默认） | `x: pl.Tensor[...]` | 只读。排在生产者之后 |
| Out | `x: pl.Out[pl.Tensor[...]]` | 只写不读。排在此前的读者与写者之后 |
| InOut | `x: pl.InOut[pl.Tensor[...]]` | 都有。与一切触碰它的任务之间都有序 |

方向是 `DeriveCallDirections` 传播、`AutoDeriveTaskDependencies` 读取来构建依赖图的东西。把一个 `InOut` 缓冲区声明成 `Out`，等于告诉运行时"在本任务写它之前不需要等任何人" —— 这是一个竞态，不是一条诊断。

若要声明某个实参**不参与**依赖跟踪（例如兄弟任务写的是互不相交的区域），请在调用点用 `pl.no_dep(t)`，而不是弱化方向。见 [作用域与任务 § 退出依赖跟踪](04-scopes-and-tasks.md#退出依赖跟踪)。

## Edge Cases

> **致命陷阱：** 把字节数写成元素个数会静默欠分配。`pld.alloc_window_buffer(256)` 预留的是 256 **字节** —— 只够 64 个 FP32，不是 256 个。任何非字面量尺寸都必须写成 `n * pl.<DTYPE>.get_byte()`。没有任何告警；症状是前 64 个元素之后的数据被破坏。

| 症状 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| **解析期出现布局相关的 `DeprecationWarning`** | 用了 `pl.Tensor[..., pl.DN]` 注解 | 去掉标记，写运行期 shape，给 `pl.matmul` 传 `b_trans=True` |
| **数字看起来对得上却报 shape 不匹配** | DN 注解翻转了坐标系 | 写源 shape；确认消费方要的是不是 `transpose_view` |
| **只有两个任务重叠时结果才出错** | 会被写入的缓冲区声明成了 `In` 或 `Out` 而非 `InOut` | 按 kernel 实际行为声明方向 |
| **读 `Out` 参数读到垃圾** | `Out` 承诺的是先写后读 | 若此前内容有意义，改用 `pl.InOut[...]` |
| **本以为会隐式提升，却要求 `pl.cast`** | 没有隐式提升 | 补上 cast；多跳类型对见 [LegalizeTileCast](../../dev/passes/14-legalize_tile_cast.md) |
| **两个本应相同的维度被当成互相独立** | 调了两次 `pl.dynamic("M")` | 只创建一次 `DynVar` 并复用该对象 |

并非每个 `pl.cast` 都是一条指令。一对 `(src, dst)` 是映射到单条硬件 `pto.tcvt` 还是展开成一条链，取决于目标架构：`INT32 -> FP16` 在 Ascend910B 上是一条指令，在 Ascend950 上会降为 `INT32 -> FP32 -> FP16`。每一跳花费一次 `tcvt`；当中间类型比源类型更窄时，结果可能与直接舍入的转换相差目标类型的 1 ULP。**这是预期行为，不是缺陷** —— 各架构的对照表见 [LegalizeTileCast](../../dev/passes/14-legalize_tile_cast.md)。

## See Also

- [函数与程序](01-functions.md) —— 这些注解出现在哪里，以及签名对调用方意味着什么。
- [内存与数据搬运](03-memory.md) —— 在这些类型所命名的空间之间搬运数据。
- [算子](../ops/index.md) —— 哪些算子接受 `Tensor`、哪些接受 `Tile`。
- [IR 类型](../../dev/ir/02-types.md) —— 这些注解所构建的 IR 层类型系统。
- [LegalizeTileCast](../../dev/passes/14-legalize_tile_cast.md) —— 分架构的 cast 展开及其精度后果。
