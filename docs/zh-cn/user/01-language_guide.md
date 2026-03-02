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

常用 `pl.*` 操作 —— 完整列表参见[操作参考](02-operation_reference.md)：

```python
c = pl.add(a, b)            # 算术（还有 sub、mul、div）
c = pl.add(a, 1.0)          # 标量右操作数自动检测
c = pl.cast(a, pl.FP16)     # 类型转换
c = pl.reshape(a, [16, 8])  # 形状操作（还有 transpose、view）
c = pl.matmul(a, b)         # 线性代数
c = pl.row_sum(a)            # 归约（还有 row_max）
```

需要 tile 特定操作（内存搬运、广播、位运算等）时使用 `pl.block.*`。

## 变量赋值与 SSA

PyPTO 的 IR 同时支持 **SSA**（静态单赋值，Static Single Assignment）和**非 SSA** 两种形式。在 SSA 形式中，每个变量只被赋值一次；在非 SSA 形式中，可以对同一个变量名多次赋值。

### 编写风格

**非 SSA（默认）** —— 像普通 Python 一样自由重新赋值：

```python
@pl.function
def example(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    result: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
    result: pl.Tensor[[64], pl.FP32] = pl.add(result, 1.0)  # 重新赋值，没问题
    return result
```

**SSA 风格** —— 每个变量只赋值一次，使用不同的名称：

```python
@pl.function
def example(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    result_0: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
    result_1: pl.Tensor[[64], pl.FP32] = pl.add(result_0, 1.0)
    return result_1
```

两种方式都能生成有效的 IR。选择你更习惯的风格即可。

### 自动 SSA 转换

大多数优化 pass 需要 SSA 形式。编译流水线会在早期自动运行 `ConvertToSSA`，因此你无需担心 —— 直接编写非 SSA 代码，编译器会自动处理转换。

### 严格 SSA 模式

传入 `strict_ssa=True` 可在解析阶段强制要求 SSA。如果重新赋值变量，解析器将报错：

```python
@pl.function(strict_ssa=True)
def example(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    result: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
    result: pl.Tensor[[64], pl.FP32] = pl.add(result, 1.0)  # 报错：SSAViolationError
    return result
```

这对于捕获无意的变量覆盖很有用，但完全是可选的。

### 为什么需要 `yield_`

在 SSA 形式中，控制流（循环、if/else）不能简单地重新赋值变量 —— 每次赋值必须是唯一的。`pl.yield_()` 是将值从控制流作用域中传出的机制：

- **循环**：`pl.yield_()` 将更新后的累加器传递给下一次迭代
- **If/else**：两个分支中的 `pl.yield_()` 创建一个合并点（phi 节点），产生一个结果变量

这就是为什么带累加器的循环需要 `init_values` + `yield_`，以及为什么产生值的 if/else 分支必须都使用 `yield_`。

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
tile = pl.load(tensor, [0, 0], [64, 64])                                  # DDR → Vec
tile_l1 = pl.load(tensor, [0, 0], [32, 32], target_memory=pl.MemorySpace.Mat)  # DDR → Mat
tile_l0a = pl.move(tile_l1, target_memory=pl.MemorySpace.Left)            # Mat → Left
out = pl.store(tile, [0, 0], [64, 64], output)                            # Tile → DDR
```

### 模式：矩阵乘法（DDR → Mat → Left/Right → Acc → DDR）

```python
a_l1 = pl.load(a, [0, 0], [32, 32], target_memory=pl.MemorySpace.Mat)
b_l1 = pl.load(b, [0, 0], [32, 32], target_memory=pl.MemorySpace.Mat)
a_l0a = pl.move(a_l1, target_memory=pl.MemorySpace.Left)
b_l0b = pl.move(b_l1, target_memory=pl.MemorySpace.Right)
c_acc = pl.matmul(a_l0a, b_l0b)                     # 结果 → Acc
out = pl.store(c_acc, [0, 0], [32, 32], output)      # Acc → DDR
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
