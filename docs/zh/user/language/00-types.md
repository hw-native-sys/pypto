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

<!-- doctest: setup -->
```python
import pypto.language as pl
import torch
from pypto.runtime import RunConfig

CFG = RunConfig(platform="__PLATFORM__")
torch.manual_seed(0)
```

<!-- doctest: run -->
```python
M = pl.dynamic("M")                       # an axis whose extent varies at run time


@pl.jit.incore
def scale_rows(
    x: pl.Tensor[[M, 128], pl.FP16],                    # In (default): read-only, DDR
    acc: pl.InOut[pl.Tensor[[M, 128], pl.FP32]],        # read-write, DDR
    out: pl.Out[pl.Tensor[[M, 128], pl.FP32]],          # write-only, DDR
    factor: pl.Scalar[pl.FP32],                         # scalar, passed by value
):
    tx = pl.load(x, [0, 0], [64, 128])
    scaled = pl.mul(pl.cast(tx, pl.FP32), factor)
    acc = pl.store(pl.add(pl.load(acc, [0, 0], [64, 128]), scaled), [0, 0], acc)
    out = pl.store(scaled, [0, 0], out)
    return acc, out


@pl.jit
def apply_scale(
    x: pl.Tensor[[64, 128], pl.FP16],
    acc: pl.InOut[pl.Tensor[[64, 128], pl.FP32]],
    out: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
):
    acc, out = scale_rows(x, acc, out, 2.0)             # the dynamic axis binds to 64 here
    return acc, out


x = torch.randn(64, 128, dtype=torch.float16)
acc = torch.randn(64, 128, dtype=torch.float32)
out = torch.zeros(64, 128, dtype=torch.float32)
acc_before = acc.clone()

apply_scale(x, acc, out, config=CFG)

scaled = x.float() * 2.0
torch.testing.assert_close(out, scaled, rtol=1e-2, atol=1e-2)               # Out was written
torch.testing.assert_close(acc, acc_before + scaled, rtol=1e-2, atol=1e-2)  # InOut was read *and* written
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
| `pl.TASK_ID` | — | 已派发任务的生产者句柄 |

`dtype.get_byte()` 返回向上取整后的可按字节寻址元素大小。对于按字节寻址的 dtype，只要字节数是算出来的而不是写死的字面量，就用它。4-bit buffer 不能使用 `logical_elements * dtype.get_byte()`：PyPTO 会把所有语义 4-bit dtype 按每字节两个逻辑元素打包，物理大小为 `ceil(logical_elements / 2)`。

```python
nbytes = 256 * pl.FP32.get_byte()          # 1024, not 256
```

PyPTO IR 中的 FP4 shape 是以 nibble 计数的逻辑 shape，`valid_shape` 也使用相同单位。Torch/runtime 边界上的 `torch.float4_e2m1fn_x2` 使用物理 x2 carrier shape：末维的每个元素是一字节、承载两个逻辑 FP4。JIT 在入口展开末维，compiled-call metadata 与 orchestration allocation 将末维除以二；`TensorType` 和 `TileType` 不保存单独的 `storage_shape`。Packed FP4 的逻辑末维必须是正偶数，静态 allocation/view shape 同样受此约束，动态宽度会在换算前检查。4-bit slice 起点必须落在字节边界；线性 nibble offset 为奇数时会直接报错。

4-bit 端到端执行按后端做能力检查。Ascend950 支持 `pl.FP4`；`INT4`、`UINT4`、`HF4` 虽使用统一存储计数，但 in-core codegen 会拒绝。Ascend910B/A2A3 会拒绝所有 4-bit in-core dtype，因为它只有孤立的 FP16↔INT4 转换，没有配套的 packed load/store carrier ABI。

### 容器类型

| 类型 | 住在 | 写法 |
| ---- | ---- | ---- |
| `pl.Tensor[[shape], dtype]` | DDR | `x: pl.Tensor[[64, 128], pl.FP32]` |
| `pl.Tile[[shape], dtype]` | 片上缓冲区（默认 Vec） | `t: pl.Tile[[64, 64], pl.FP32]` |
| `pl.Scalar[dtype]` | 值，不是缓冲区 | `s: pl.Scalar[pl.FP32]` |
| `pl.Array[extent, dtype]` | 核内数组 | `a: pl.Array[16, pl.INT32]` |
| `pl.Tuple[T1, T2]` | — | 多值返回注解 |

`pl.TaskId` 是 `pl.Scalar[pl.TASK_ID]` 的便捷别名。

`pl.Array` 通常是创建出来的而不是注解出来的 —— 数组不跨函数边界，所以注解形式很少见。更新是函数式的：它产生一个新的数组值并重绑该名字，因此循环内的数组赋值和其他携带值一样是一个携带值。

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

只写布局的简写 `pl.Tensor[..., pl.DN]` 不被支持：它会抛 `ParserTypeError`。矩阵乘需要转置操作数时，给 `pl.matmul` 传 `a_trans=True` / `b_trans=True`，或在使用处用 `pl.transpose(x, -2, -1)` 导出转置视图。对产生 DN 的算子做切片或 reshape，会自动继承 DN。

`pl.ND` 是默认的行主序布局，不需要写出来。`pl.NZ` 断言该张量在全局内存中的字节**已经**按 PTO 原生 NZ 分形序存放，于是 matmul 权重载入可以跳过在线 ND→NZ 转换。它是对现有字节的断言，不是转换请求：你写的 shape 和切片保持逻辑形式，编译器负责推导分块后的物理描述符。目前要求 dtype 为整字节、张量形状静态且分形对齐（`shape[-2] % 16 == 0`、`shape[-1] % (256 / dtype 位宽) == 0`），并作为 matmul 操作数读取；其余情形一律报错。

当一个张量的行不连续时 —— 大缓冲区里的一个窗口、外部传进来的跨步切片 —— 用 `pl.TensorView` 描述它，把 stride 显式写出来，而不是留给推断：

```python
view = pl.TensorView(stride=[1024, 1], layout=pl.TensorLayout.ND, valid_shape=[16, 64])
```

只要给了 `stride`、`valid_shape`、`pad` 三者之一，`layout=` 就是必填的。`pl.TensorLayout` 是这些布局常量所属的枚举 —— `pl.ND` 就是 `pl.TensorLayout.ND`。

剩下两个布局常量是 `pl.MX_A_ZZ` 与 `pl.MX_B_NN`。它们标注 Ascend950 上 MX（microscaling）操作数的 **GM scale 张量** —— `MX_A_ZZ` 对应左/A 侧 scale pack，`MX_B_NN` 对应右/B 侧 —— 使 Mat 到 scale 的 `pl.move` 能校验源布局，而不是把不兼容的数据按字节拷进 `LeftScale` / `RightScale`。这是唯一一处**要求**在 `pl.Tensor` 注解上写布局标记、而非不建议写的场景。当前限制：MX 的 `pl.load` 必须显式传 `target_memory=pl.Mem.Mat`；MX 子视图（`slice`、`reshape`、`transpose`、`reinterpret_view`、`view`）与 MX `remote_load` 会被拒绝。矩阵乘本身是 `pl.matmul_mx` 及其 `_acc` / `_bias` 变体，每个操作数各接一块 data tile 和一块 scale tile。进入算子的两块 data tile 必须都是 `FP8E4M3FN`。支持的 FP4 输入形式仅为左侧 FP4×右侧 FP8，并且必须在 `matmul_mx` 前显式写 `pl.cast(fp4_tile, pl.FP8E4M3FN)`；A5 的 cast legalization pass 会将其展开为 FP4→BF16→FP32→FP8E4M3FN。原生 FP4×FP4 与反向 FP8×FP4 均不支持，MXFP4 quant 前端本次仍未开放。

### 动态 shape

`pl.dynamic(name)` 用来标注**编译时未知、且每次启动都可能不同**的轴 —— 随请求变化的 batch、解码过程中不断增长的序列长度。该维的尺寸成为运行期的值，因此一份编译产物服务这个轴的所有取值：动态维在 JIT 缓存 key 里折叠为 `None`，尺寸变化不会触发重编译。

<!-- doctest: run -->
```python
N = pl.dynamic("N")
TILE = 32                       # the physical tile the kernel moves per call


@pl.jit.incore
def rows(x: pl.Tensor[[N, 64], pl.FP32], out: pl.Out[pl.Tensor[[N, 64], pl.FP32]]):
    out = pl.store(pl.mul(pl.load(x, [0, 0], [TILE, 64]), 2.0), [0, 0], out)
    return out


@pl.jit
def drive(x: pl.Tensor[[N, 64], pl.FP32], out: pl.Out[pl.Tensor[[N, 64], pl.FP32]]):
    return rows(x, out)         # the entry is dynamic too, so both extents share it


# Two extents through one program: the dynamic dim collapses to None in the JIT
# cache key, so the second call is not a recompile.
for extent in (TILE, 3 * TILE):
    x = torch.randn(extent, 64, dtype=torch.float32)
    out = torch.zeros(extent, 64, dtype=torch.float32)
    drive(x, out, config=CFG)
    torch.testing.assert_close(out[:TILE], x[:TILE] * 2.0, rtol=1e-4, atol=1e-4)
```

入口必须保持 dynamic 才成立。给它一个具体形状会把程序钉死在一个 extent 上；而在 dynamic 形状的张量上做编排级运算（而不是交给 InCore kernel）会更早失败 —— 失败在 `InitMemRef`，它需要常量维。上面这个 kernel 搬的是固定的 `TILE`；要覆盖更大输入的全部，用 [控制流](02-control-flow.md) 里的分块循环。

同一个 `DynVar` 对象在多处注解中使用时指的是同一维 —— 复用这个对象，不要在表示同一个值时再造一个同名的。

确实固定的维度就保持静态。静态尺寸是编译器能据以规划的数字 —— 分块选择、展开因子、静态边界检查 —— 动态维则把这个信息藏了起来。

### 参数方向

| 方向 | 语法 | 编译器据此断定 |
| ---- | ---- | -------------- |
| In（默认） | `x: pl.Tensor[...]` | 只读。排在生产者之后 |
| Out | `x: pl.Out[pl.Tensor[...]]` | 只写不读。排在此前的读者与写者之后 |
| InOut | `x: pl.InOut[pl.Tensor[...]]` | 都有。与一切触碰它的任务之间都有序 |

方向是编译器用来给任务之间定序的依据。把一个 `InOut` 缓冲区声明成 `Out`，等于告诉运行时"在本任务写它之前不需要等任何人" —— 这是一个竞态，不是一条诊断。

## 边界情况

> **致命陷阱：** 把字节数写成元素个数会静默欠分配。`pld.alloc_window_buffer(256)` 预留的是 256 **字节** —— 只够 64 个 FP32，不是 256 个。任何非字面量尺寸都必须写成 `n * pl.<DTYPE>.get_byte()`。没有任何告警；症状是前 64 个元素之后的数据被破坏。

| 症状 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| **报 DN layout-only shorthand 的 `ParserTypeError`** | `pl.Tensor[..., pl.DN]` —— 已移除，它把两套坐标系压进了一条注解 | 写源 shape、不带标记；在使用处用 `pl.transpose(x, -2, -1)` 导出 DN；或让它从产生 DN 的算子经切片/reshape 继承 |
| **只有两个任务重叠时结果才出错** | 读写缓冲区声明成了 `In` 或 `Out` 而非 `InOut` | 按 kernel 实际行为声明方向 |
| **读 `Out` 参数读到垃圾** | `Out` 承诺的是先写后读 | 若此前内容有意义，改用 `pl.InOut[...]` |
| **本以为会隐式提升，却要求 `pl.cast`** | 没有隐式提升 | 补上 cast；多跳类型对见 [LegalizeTileCast](../../dev/passes/15-legalize_tile_cast.md) |
| **两个本应相同的维度被当成互相独立** | 调了两次 `pl.dynamic("M")` | 只创建一次 `DynVar` 并复用该对象 |

并非每个 `pl.cast` 都是一条指令。一对 `(src, dst)` 是映射到单条硬件 `pto.tcvt` 还是展开成一条链，取决于目标架构：`INT32 -> FP16` 在 Ascend910B 上是一条指令，在 Ascend950 上会降为 `INT32 -> FP32 -> FP16`。每一跳花费一次 `tcvt`；当中间类型比源类型更窄时，结果可能与直接舍入的转换相差目标类型的 1 ULP。**这是预期行为，不是缺陷** —— 各架构的对照表见 [LegalizeTileCast](../../dev/passes/15-legalize_tile_cast.md)。

## See Also

- [函数与程序](01-functions.md) —— 这些注解出现在哪里，以及签名对调用方意味着什么。
- [内存与数据搬运](03-memory.md) —— 在这些类型所命名的空间之间搬运数据。
- [算子](../ops/index.md) —— 哪些算子接受 `Tensor`、哪些接受 `Tile`。
- [IR 类型](../../dev/ir/02-types.md) —— 这些注解所构建的 IR 层类型系统。
- [LegalizeTileCast](../../dev/passes/15-legalize_tile_cast.md) —— 分架构的 cast 展开及其精度后果。
