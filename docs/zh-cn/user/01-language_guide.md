# PyPTO 语言指南

`pypto.language`（`pl`）模块的完整参考。

## 类型系统

### 数据类型（DataType）

| 常量 | 位数 | 说明 |
| ---- | ---- | ---- |
| `pl.BOOL` | 1 | 布尔值 |
| `pl.INT4` / `pl.UINT4` | 4 | 有符号 / 无符号 4 位整数 |
| `pl.INT8` / `pl.UINT8` | 8 | 有符号 / 无符号 8 位整数 |
| `pl.INT16` / `pl.UINT16` | 16 | 有符号 / 无符号 16 位整数 |
| `pl.INT32` / `pl.UINT32` | 32 | 有符号 / 无符号 32 位整数 |
| `pl.INT64` / `pl.UINT64` | 64 | 有符号 / 无符号 64 位整数 |
| `pl.FP16` | 16 | IEEE 半精度浮点 |
| `pl.BF16` | 16 | Brain Float 16 |
| `pl.FP32` | 32 | IEEE 单精度浮点 |
| `pl.FP4` | 4 | 4 位浮点 |
| `pl.FP8E4M3FN` | 8 | 8 位浮点（e4m3fn） |
| `pl.FP8E5M2` | 8 | 8 位浮点（e5m2） |
| `pl.HF4` / `pl.HF8` | 4/8 | 昇腾浮点格式 |
| `pl.INDEX` | 64 | 索引类型（INT64 别名）—— 循环变量、维度 |

### 容器类型

**`pl.Tensor[[shape], dtype]`** —— DDR 内存数组（片外全局内存）。

```python
x: pl.Tensor[[64, 128], pl.FP32]        # 二维，64×128，float32
y: pl.Tensor[[256], pl.FP16]            # 一维，256 个元素，float16
z: pl.Tensor[[64, 128], pl.FP16, pl.NZ] # 带 NZ 布局
```

**`pl.Tile[[shape], dtype]`** —— 片上内存缓冲区（默认统一缓冲区）。

```python
t: pl.Tile[[64, 64], pl.FP32]           # 二维 tile，64×64
```

**`pl.Scalar[dtype]`** —— 单个标量值。

```python
s: pl.Scalar[pl.FP32]                   # float32 标量
idx: pl.Scalar[pl.INDEX]                # 索引标量
```

### 张量布局（TensorLayout）

布局控制 Tensor 的物理内存排列：

| 布局 | 说明 |
| ---- | ---- |
| `pl.ND` | N 维（默认，行优先） |
| `pl.DN` | DN 布局 |
| `pl.NZ` | NZ 分形格式（硬件特定分块） |

```python
# 指定布局作为第三个类型参数
a: pl.Tensor[[64, 128], pl.FP16, pl.NZ]
```

### 动态形状（Dynamic Shapes）

使用 `pl.dynamic()` 声明运行时确定的维度：

```python
M = pl.dynamic("M")
N = pl.dynamic("N")

@pl.function
def dynamic_kernel(
    a: pl.Tensor[[M, N], pl.FP32],
) -> pl.Tensor[[M, N], pl.FP32]:
    ...
```

### 参数方向（Parameter Directions）

默认情况下，参数为只读输入。使用包装器声明输出参数：

| 方向 | 语法 | 说明 |
| ---- | ---- | ---- |
| 输入（默认） | `a: pl.Tensor[...]` | 只读 |
| 输出 | `a: pl.Out[pl.Tensor[...]]` | 只写输出 |
| 输入/输出 | `a: pl.InOut[pl.Tensor[...]]` | 读写 |

```python
@pl.function
def kernel(
    input_a: pl.Tensor[[64], pl.FP32],                    # In
    output_b: pl.Out[pl.Tensor[[64], pl.FP32]],            # Out
    accum_c: pl.InOut[pl.Tensor[[64], pl.FP32]],           # InOut
) -> pl.Tensor[[64], pl.FP32]:
    ...
```

## 操作

### 分发模型（Dispatch Model）

PyPTO 操作分为三个层级：

| 命名空间 | 层级 | 说明 |
| -------- | ---- | ---- |
| `pl.*` | 统一 | 根据输入类型（Tensor 或 Tile）自动分发 |
| `pl.tensor.*` | Tensor | DDR 级别的 `Tensor` 操作 |
| `pl.block.*` | Tile | 片上 `Tile` 操作 |

**推荐：** 尽量使用 `pl.*`（统一接口）。分发器会选择正确的实现。

```python
# 统一接口 —— Tensor 和 Tile 都适用
result = pl.add(a, b)       # 分发到 tensor.add 或 block.add
result = pl.mul(a, scalar)   # 分发到 tensor.mul_scalar 或 block.muls

# 显式 tile 级别（需要 tile 特定操作时）
tile = pl.block.load(tensor, [0, 0], [64, 64])
tile = pl.block.adds(tile, 1.0)
```

### Python 运算符

标准 Python 运算符映射到 IR 操作：

| Python | IR 操作 | 示例 |
| ------ | ------- | ---- |
| `a + b` | `add` | `c = a + b` |
| `a - b` | `sub` | `c = a - b` |
| `a * b` | `mul` | `c = a * b` |
| `a / b` | `div` | `c = a / b` |
| `a == b` | `eq`（比较） | `if a == 0:` |
| `a != b` | `ne`（比较） | `if a != 0:` |
| `a < b` | `lt`（比较） | `if a < n:` |
| `a > b` | `gt`（比较） | `if a > 0:` |

### 统一操作（Unified Operations）

适用于 `Tensor` 和 `Tile` 输入：

```python
# 算术（接受 Tensor 或 Tile；标量右操作数自动检测）
c = pl.add(a, b)            # 逐元素加法
c = pl.sub(a, b)            # 逐元素减法
c = pl.mul(a, b)            # 逐元素乘法
c = pl.div(a, b)            # 逐元素除法
c = pl.add(a, 1.0)          # 对所有元素加标量

# 数学
c = pl.exp(a)               # 逐元素指数
c = pl.maximum(a, b)        # 逐元素最大值
c = pl.cast(a, pl.FP16)     # 类型转换

# 形状
c = pl.reshape(a, [16, 8])  # 变形
c = pl.transpose(a, 0, 1)   # 交换轴
c = pl.view(a, [32, 64], [0, 0])  # 视图/切片

# 线性代数
c = pl.matmul(a, b)         # 矩阵乘法
c = pl.matmul(a, b, out_dtype=pl.FP32, b_trans=True)

# 归约
c = pl.row_max(a)            # 沿最后一个轴取最大值
c = pl.row_sum(a)            # 沿最后一个轴求和
```

### 何时使用 `pl.block.*`

在需要以下操作时使用显式 block 操作：

- 内存搬运：`load`、`store`、`move`、`vec_move`
- Tile 创建：`create_tile`、`full`
- 累加操作：`matmul_acc`、`gemv_acc`
- 广播操作：`row_expand`、`col_expand`
- 位运算：`and_`、`or_`、`shl`、`shr`
- Block 索引：`get_block_idx()`

完整列表参见[操作参考](02-operation_reference.md)。

## 控制流

### For 循环 —— `pl.range()`

**简单循环：**

```python
for i in pl.range(10):
    # i = 0, 1, 2, ..., 9
    ...

for i in pl.range(2, 10):
    # i = 2, 3, ..., 9
    ...

for i in pl.range(0, 100, 4):
    # i = 0, 4, 8, ..., 96
    ...
```

**带累加器的循环（`init_values`）：**

累加器在迭代之间传递值。每次迭代接收前一次的值，必须 `yield_` 新值：

```python
@pl.function
def sum_16_elements(data: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[1], pl.FP32]:
    init_sum: pl.Tensor[[1], pl.FP32] = pl.create_tensor([1], dtype=pl.FP32)

    for i, (running_sum,) in pl.range(16, init_values=(init_sum,)):
        chunk: pl.Tensor[[1], pl.FP32] = pl.view(data, [1], [i])
        new_sum: pl.Tensor[[1], pl.FP32] = pl.add(running_sum, chunk)
        sum_out: pl.Tensor[[1], pl.FP32] = pl.yield_(new_sum)

    # 循环结束后 sum_out 保存最终累加值
    return sum_out
```

**多个累加器：**

```python
@pl.function
def find_max_and_sum(
    data: pl.Tensor[[4, 64], pl.FP32],
) -> pl.Tensor[[1, 64], pl.FP32]:
    init_max: pl.Tensor[[1, 64], pl.FP32] = pl.create_tensor([1, 64], dtype=pl.FP32)
    init_sum: pl.Tensor[[1, 64], pl.FP32] = pl.create_tensor([1, 64], dtype=pl.FP32)

    for i, (acc_max, acc_sum) in pl.range(4, init_values=(init_max, init_sum)):
        row: pl.Tensor[[1, 64], pl.FP32] = pl.view(data, [1, 64], [i, 0])
        new_max: pl.Tensor[[1, 64], pl.FP32] = pl.maximum(acc_max, row)
        new_sum: pl.Tensor[[1, 64], pl.FP32] = pl.add(acc_sum, row)
        out_max, out_sum = pl.yield_(new_max, new_sum)

    return out_sum
```

### 并行循环 —— `pl.parallel()`

语法与 `pl.range()` 相同，但迭代可以并行执行：

```python
for i in pl.parallel(0, num_blocks):
    # 迭代相互独立，可以并行运行
    ...
```

### While 循环 —— `pl.while_()`

始终需要 `init_values`。条件通过 `pl.cond()` 作为循环体的**第一条语句**设置：

```python
for (x,) in pl.while_(init_values=(0,)):
    pl.cond(x < 10)          # 当 x < 10 时继续
    new_x = x + 1
    x_out = pl.yield_(new_x)
```

### If/Else 与 `pl.yield_()`

产生值的分支必须 `yield_` 这些值。这会创建 SSA phi 节点 —— 两个分支必须 yield 相同数量和类型的值：

```python
@pl.function
def conditional_update(
    a: pl.Tensor[[64], pl.FP32],
    delta: pl.Tensor[[64], pl.FP32],
) -> pl.Tensor[[64], pl.FP32]:
    init: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)

    for i, (prev,) in pl.range(4, init_values=(init,)):
        if i == 0:
            result: pl.Tensor[[64], pl.FP32] = pl.yield_(a)
        else:
            updated: pl.Tensor[[64], pl.FP32] = pl.add(prev, delta)
            result: pl.Tensor[[64], pl.FP32] = pl.yield_(updated)
        # result 保存执行的那个分支的值
        out: pl.Tensor[[64], pl.FP32] = pl.yield_(result)

    return out
```

**规则：** 如果一个分支 yield，另一个也必须 yield。两个分支 yield 相同数量的值。

## 程序与函数

### `@pl.function`

将 Python 函数解析为 IR：

```python
@pl.function
def my_func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    ...
```

指定函数类型：

```python
@pl.function(type=pl.FunctionType.InCore)
def compute_kernel(...):
    ...

@pl.function(type=pl.FunctionType.Orchestration)
def task_graph(...):
    ...
```

| 函数类型 | 说明 | 典型用途 |
| -------- | ---- | -------- |
| `Opaque` | 未指定上下文（默认） | 独立函数 |
| `InCore` | AICore 计算内核 | Load/compute/store 模式 |
| `Orchestration` | 主机端协调器 | 创建张量、调度 InCore 任务 |

### `@pl.program`

将多个函数组成可编译的程序：

```python
@pl.program
class MyProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, ...):
        ...

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, ...):
        result = self.kernel(...)   # 跨函数调用
        return result
```

**规则：**

- 每个方法必须有 `self` 作为第一个参数（从 IR 中去除）
- 跨函数调用使用 `self.method_name(...)`
- 装饰后的类成为 `ir.Program`，不是 Python 类

### `@pl.inline`

定义一个在每个调用点展开其函数体的函数（程序中不会有单独的函数）：

```python
@pl.inline
def normalize(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    result: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
    return result

@pl.program
class MyProgram:
    @pl.function
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        y: pl.Tensor[[64], pl.FP32] = normalize(x)  # 函数体在此处内联
        return y
```

### 外部函数调用

独立的 `@pl.function` 可以在 `@pl.program` 内被调用。它会作为单独的函数添加到程序中：

```python
@pl.function
def softmax(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    ...

@pl.program
class Model:
    @pl.function
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        y: pl.Tensor[[64], pl.FP32] = softmax(x)  # 调用外部函数
        return y
```

### InCore 作用域

将代码区域标记为 InCore 执行，无需创建单独的函数：

```python
with pl.incore():
    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
```

## 内存与数据搬运

### 内存层次结构

```text
DDR（片外，全局内存）
 │
 ├── Vec（统一缓冲区，片上）         ← pl.load() / pl.store()
 │    └── 计算（向量运算）
 │
 ├── Mat（L1 缓冲区）               ← pl.load(..., target_memory=pl.MemorySpace.Mat)
 │    ├── Left（L0A）               ← pl.move(..., target_memory=pl.MemorySpace.Left)
 │    └── Right（L0B）              ← pl.move(..., target_memory=pl.MemorySpace.Right)
 │         └── Acc（L0C）           ← pl.matmul() 结果
 │              └── DDR             ← pl.store()
```

### 内存空间（MemorySpace）

| 空间 | 枚举 | 说明 |
| ---- | ---- | ---- |
| DDR | `MemorySpace.DDR` | 片外全局内存（Tensor 参数） |
| Vec | `MemorySpace.Vec` | 统一向量缓冲区（`pl.load` 默认目标） |
| Mat | `MemorySpace.Mat` | L1 矩阵缓冲区 |
| Left | `MemorySpace.Left` | L0A —— 矩阵乘法左操作数 |
| Right | `MemorySpace.Right` | L0B —— 矩阵乘法右操作数 |
| Acc | `MemorySpace.Acc` | L0C —— 矩阵乘法累加器 |

### 数据搬运操作

```python
# DDR → Vec（默认）
tile: pl.Tile[[64, 64], pl.FP32] = pl.load(tensor, [0, 0], [64, 64])

# DDR → Mat（L1）
tile_l1: pl.Tile[[32, 32], pl.FP16] = pl.load(
    tensor, [0, 0], [32, 32], target_memory=pl.MemorySpace.Mat
)

# Mat → Left/Right（用于矩阵乘法）
tile_l0a: pl.Tile[[32, 32], pl.FP16] = pl.move(
    tile_l1, target_memory=pl.MemorySpace.Left
)

# Vec → Vec（统一缓冲区内拷贝）
tile_copy: pl.Tile[[64, 64], pl.FP32] = pl.block.vec_move(tile)

# Tile → DDR
out: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile, [0, 0], [64, 64], output)
```

### 模式：向量运算

加载到 Vec，计算，存回：

```python
a_tile: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
b_tile: pl.Tile[[64, 64], pl.FP32] = pl.load(input_b, [0, 0], [64, 64])
result: pl.Tile[[64, 64], pl.FP32] = pl.add(a_tile, b_tile)
out: pl.Tensor[[64, 64], pl.FP32] = pl.store(result, [0, 0], [64, 64], output)
```

### 模式：矩阵乘法

DDR → Mat → Left/Right → Acc → DDR:

```python
# 加载到 L1（Mat）
a_l1: pl.Tile[[32, 32], pl.FP16] = pl.load(
    a, [0, 0], [32, 32], target_memory=pl.MemorySpace.Mat
)
b_l1: pl.Tile[[32, 32], pl.FP16] = pl.load(
    b, [0, 0], [32, 32], target_memory=pl.MemorySpace.Mat
)

# 移动到矩阵乘法输入缓冲区
a_l0a: pl.Tile[[32, 32], pl.FP16] = pl.move(
    a_l1, target_memory=pl.MemorySpace.Left
)
b_l0b: pl.Tile[[32, 32], pl.FP16] = pl.move(
    b_l1, target_memory=pl.MemorySpace.Right
)

# 矩阵乘法（结果进入 Acc）
c_acc: pl.Tile[[32, 32], pl.FP32] = pl.matmul(a_l0a, b_l0b)

# 从 Acc 存到 DDR
out: pl.Tensor[[32, 32], pl.FP32] = pl.store(c_acc, [0, 0], [32, 32], output)
```

## 编译

### `ir.compile()`

```python
from pypto import ir
from pypto.backend import BackendType

output_dir = ir.compile(
    program,
    output_dir=None,                           # 为 None 时自动生成
    strategy=ir.OptimizationStrategy.Default,  # Default 或 PTOAS
    dump_passes=True,                          # 每个 pass 后打印 IR
    backend_type=BackendType.PTO,              # PTO 或 CCE
)
```

| 参数 | 选项 | 说明 |
| ---- | ---- | ---- |
| `strategy` | `Default`、`PTOAS` | `Default` = 完整流水线含同步插入。`PTOAS` = PTO 汇编（无调度） |
| `backend_type` | `PTO`、`CCE` | 代码生成后端 |
| `dump_passes` | `True`/`False` | 每个优化 pass 前后打印 IR |
| `skip_ptoas` | `True`/`False` | 跳过 PTOAS 步骤，输出原始 MLIR 文件（默认 `False`） |
| `output_dir` | 路径或 `None` | 输出目录（`None` 时自动创建） |
| `verification_level` | `NONE`、`BASIC` | IR 校验级别（默认 `BASIC`） |

### 优化流水线

`Default` 策略按顺序运行以下 pass：

1. **ConvertToSSA** —— 转换为静态单赋值形式
2. **FlattenCallExpr** —— 展平嵌套函数调用
3. **RunVerifier** —— 验证 IR 结构完整性
4. **InitMemRef** —— 分配内存空间，插入缓冲区分配
5. **MemoryReuse** —— 共享生命周期不重叠的缓冲区
6. **InsertSync** —— 在流水线阶段之间插入同步屏障
7. **AllocateMemoryAddr** —— 分配具体内存地址

### 调试

使用 `ir.python_print()` 查看函数或程序的 IR。编译时设置 `dump_passes=True` 以查看每个优化阶段的 IR。
