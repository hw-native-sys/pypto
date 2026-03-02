# PyPTO 入门指南

## 什么是 PyPTO？

PyPTO 是一个基于 Python 的 Ascend NPU 内核编程框架。使用 `pypto.language` 模块编写计算内核，PyPTO 将其编译为优化的设备代码。

```python
import pypto.language as pl
from pypto import ir
```

所有内核代码使用 `pl` 命名空间。`ir` 模块提供编译和 IR 工具。

## Hello World：向量加法（张量级别）

最简单的内核操作 **Tensor（张量）** —— DDR 内存中的高级数组。PyPTO 自动处理数据搬运和内存分配。

```python
import pypto.language as pl
from pypto import ir

@pl.function
def vector_add(
    a: pl.Tensor[[64], pl.FP32],
    b: pl.Tensor[[64], pl.FP32],
) -> pl.Tensor[[64], pl.FP32]:
    result: pl.Tensor[[64], pl.FP32] = pl.add(a, b)
    return result
```

**逐行说明：**

| 行 | 功能 |
| -- | ---- |
| `@pl.function` | 将 Python 函数体解析为 PyPTO IR |
| `a: pl.Tensor[[64], pl.FP32]` | 输入：一维张量，64 个元素，32 位浮点 |
| `pl.add(a, b)` | 逐元素加法（分发到张量加法） |
| `return result` | 函数返回一个张量 |

装饰器执行后，`vector_add` 是一个 `ir.Function` 对象 —— 不是 Python 可调用函数。打印 IR：

```python
print(ir.python_print(vector_add))
```

## Tile 内核：Load-Compute-Store

要进行硬件级控制，使用 **Tile（数据块）** —— 片上内存缓冲区。显式地从 DDR 加载数据、在片上计算、然后将结果存回。

```python
@pl.function
def vector_add_tile(
    a: pl.Tensor[[64], pl.FP32],
    b: pl.Tensor[[64], pl.FP32],
    output: pl.Out[pl.Tensor[[64], pl.FP32]],
) -> pl.Tensor[[64], pl.FP32]:
    # 从 DDR 加载到片上（Vec 内存）
    a_tile: pl.Tile[[64], pl.FP32] = pl.load(a, [0], [64])
    b_tile: pl.Tile[[64], pl.FP32] = pl.load(b, [0], [64])

    # 片上计算
    result: pl.Tile[[64], pl.FP32] = pl.add(a_tile, b_tile)

    # 存回 DDR
    out: pl.Tensor[[64], pl.FP32] = pl.store(result, [0], [64], output)
    return out
```

**与张量版本的关键区别：**

| 概念 | 张量级别 | Tile 级别 |
| ---- | -------- | --------- |
| 数据位置 | DDR（自动） | 显式 load/store |
| 类型 | `pl.Tensor` | `pl.Tile`（片上） |
| 输出参数 | 返回值 | `pl.Out[pl.Tensor[...]]` |
| 内存控制 | 编译器决定 | 用户决定 |

**`pl.load(tensor, offsets, shapes)`** 从 DDR Tensor 拷贝一个区域到片上 Tile。

**`pl.store(tile, offsets, shapes, output_tensor)`** 将 Tile 拷贝回 DDR。

## 循环与累加

使用 `pl.range()` 进行循环。通过 `init_values` 实现循环携带值（累加器）：

```python
@pl.function
def sum_elements(
    a: pl.Tensor[[64], pl.FP32],
) -> pl.Tensor[[1], pl.FP32]:
    zero: pl.Tensor[[1], pl.FP32] = pl.create_tensor([1], dtype=pl.FP32)

    for i, (acc,) in pl.range(64, init_values=(zero,)):
        elem: pl.Tensor[[1], pl.FP32] = pl.view(a, [1], [i])
        new_acc: pl.Tensor[[1], pl.FP32] = pl.add(acc, elem)
        acc_out: pl.Tensor[[1], pl.FP32] = pl.yield_(new_acc)

    return acc_out
```

**`init_values` 工作原理：**

1. `init_values=(zero,)` —— 累加器的初始值
2. `for i, (acc,)` —— `i` 是循环变量，`acc` 是当前累加器
3. `pl.yield_(new_acc)` —— 将 `new_acc` 作为下一次迭代的累加器
4. 循环结束后，`acc_out` 保存最终值

简单循环（无累加器）：

```python
for i in pl.range(10):
    # i 从 0 到 9
    ...

for i in pl.range(0, 100, 2):
    # i 从 0 到 98，步长 2
    ...
```

## 多函数程序

使用 `@pl.program` 将多个相互调用的函数组合在一起：

```python
@pl.program
class VectorAddProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel_add(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        output: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        a_tile: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
        b_tile: pl.Tile[[128, 128], pl.FP32] = pl.load(b, [0, 0], [128, 128])
        result: pl.Tile[[128, 128], pl.FP32] = pl.add(a_tile, b_tile)
        out: pl.Tensor[[128, 128], pl.FP32] = pl.store(
            result, [0, 0], [128, 128], output
        )
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        c: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor(
            [128, 128], dtype=pl.FP32
        )
        c = self.kernel_add(a, b, c)
        return c
```

**关键概念：**

| 概念 | 说明 |
| ---- | ---- |
| `@pl.program` | 装饰类 → 成为 `ir.Program` |
| `self` | 必需的第一个参数；从 IR 中去除 |
| `self.kernel_add(...)` | 程序内跨函数调用 |
| `FunctionType.InCore` | 在 AICore 上运行（计算内核） |
| `FunctionType.Orchestration` | 在主机端运行（任务图协调器） |

**函数类型（FunctionType）：**

- **`Opaque`**（默认）—— 无特定执行上下文
- **`InCore`** —— AICore 计算内核；使用 load/store 进行数据搬运
- **`Orchestration`** —— 主机端函数，创建张量并调度 InCore 任务

## 编译

编译程序以生成设备代码：

```python
from pypto.backend import BackendType

output_dir = ir.compile(
    VectorAddProgram,
    strategy=ir.OptimizationStrategy.Default,
    dump_passes=True,
    backend_type=BackendType.CCE,
)
print(f"Generated code in: {output_dir}")
```

**`ir.compile()` 参数：**

| 参数 | 默认值 | 说明 |
| ---- | ------ | ---- |
| `program` | （必需） | 要编译的 `ir.Program` |
| `strategy` | `Default` | 优化策略（`Default` 或 `PTOAS`） |
| `dump_passes` | `True` | 每个优化 pass 后打印 IR |
| `backend_type` | `PTO` | 代码生成后端（`PTO` 或 `CCE`） |
| `output_dir` | 自动生成 | 输出文件目录 |

**不编译直接查看 IR：**

```python
# 打印单个函数
print(ir.python_print(vector_add))

# 打印整个程序
print(ir.python_print(VectorAddProgram))
```

## 下一步

- **[语言指南](01-language_guide.md)** —— 类型、操作、控制流、内存和编译的完整参考
- **[操作参考](02-operation_reference.md)** —— 所有 `pl.*` 操作的查找表
