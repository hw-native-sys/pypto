# 快速上手

编写、查看并编译你的第一批 PyPTO kernel —— 从一行张量加法到多函数 program。

> **前置**：PyPTO 已安装且可导入 —— 见[安装](01-installation.md)。
> 本页的每个例子在装好 PyPTO 的机器上都能直接跑；它们都不需要 NPU，因为这里没有任何代码
> 会派发到设备。

## Concept

PyPTO 的 kernel 是被**解析**而非被执行的 Python 源码。`@pl.function` 读取被装饰的函数体
并据此构建 PyPTO IR；得到的对象是 `ir.Function`，不是可调用对象。在你编译 IR 并派发它之前，
什么都不会运行。

这也是下面的例子处处都写类型标注的原因。标注不是文档 —— 它们是 parser 用来构建 IR 的形状与
dtype 契约。少一个标注就是少一块程序。

本页出现两个抽象层次。**张量级**下你命名 DDR 中的整个数组，由编译器决定数据放在哪、何时搬运；
**tile 级**下你命名片上缓冲区，自己搬运数据。张量级是起点；当你需要控制"什么东西在片上、
什么时候在"时，就下到 tile 级。

```python
import pypto.language as pl
from pypto import ir
```

`pl` 是语言层 —— 类型、算子、控制流。`ir` 是编译与 IR 工具。下面每个例子都默认有这两行导入。

## Quickstart：张量级向量加法

最小的完整 kernel。它命名两个输入张量、相加、返回结果；数据在哪、怎么搬是编译器的事。

```python
import pypto.language as pl

@pl.function
def vector_add(
    a: pl.Tensor[[64], pl.FP32],
    b: pl.Tensor[[64], pl.FP32],
) -> pl.Tensor[[64], pl.FP32]:
    result: pl.Tensor[[64], pl.FP32] = pl.add(a, b)
    return result

print(vector_add.as_python())
```

| 行 | 作用 |
| -- | ---- |
| `@pl.function` | 把函数体解析成 PyPTO IR。此后 `vector_add` 是一个 `ir.Function` |
| `a: pl.Tensor[[64], pl.FP32]` | 输入：1 维张量、64 个元素、32 位浮点。方向默认为 `In` |
| `result: pl.Tensor[...] = pl.add(a, b)` | 逐元素加。赋值目标上的标注是必需的 —— 它给 IR 绑定定型 |
| `return result` | 函数的返回值，也是返回类型的来源 |

预期输出 —— 注意 `pl.add` 已经解析到了它的张量命名空间形式：

```python
@pl.function
def vector_add(a: pl.Tensor[[64], pl.FP32], b: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    result: pl.Tensor[[64], pl.FP32] = pl.tensor.add(a, b)
    return result
```

`as_python()` 把 IR 重新打印成 DSL 源码。它是确认"parser 到底构建了什么"最快的手段，任何时候
kernel 表现异常都值得先读一遍 —— 你拿到的是编译器看到的那个程序，而不是你以为自己写的那个。

## Mechanics

### Tile 级：load、compute、store

在 tile 级你显式分配片上缓冲区，并自己把数据搬进搬出。

```python
@pl.function
def vector_add_tile(
    a: pl.Tensor[[64], pl.FP32],
    b: pl.Tensor[[64], pl.FP32],
    output: pl.Out[pl.Tensor[[64], pl.FP32]],
) -> pl.Tensor[[64], pl.FP32]:
    # DDR -> 片上
    a_tile: pl.Tile[[64], pl.FP32] = pl.load(a, [0], [64])
    b_tile: pl.Tile[[64], pl.FP32] = pl.load(b, [0], [64])

    # 片上计算
    result: pl.Tile[[64], pl.FP32] = pl.add(a_tile, b_tile)

    # 片上 -> DDR
    out: pl.Tensor[[64], pl.FP32] = pl.store(result, [0], output)
    return out
```

| 概念 | 张量级 | Tile 级 |
| ---- | ------ | ------- |
| 数据在哪 | DDR；编译器决定放置 | 你命名片上缓冲区 |
| 类型 | `pl.Tensor` | `pl.Tile` |
| 数据搬运 | 编译器插入 | 显式 `pl.load` / `pl.store` |
| 结果交付 | 返回值 | `pl.Out[...]` 参数，经 `pl.store` 写入 |

- **`pl.load(tensor, offsets, shapes)`** 从 DDR 张量中拷出一块区域，放进一个新的片上 tile。
  `offsets` 是区域起点，`shapes` 是区域大小 —— 都是逐维给出。
- **`pl.store(tile, offsets, output_tensor)`** 把 tile 拷回 DDR 张量的 `offsets` 处，并返回
  被写入的张量。

`pl.Out[...]` 包装表示**参数方向**，它是承重的而非装饰性的：它告诉编译器这个参数是被写的而不是
被读的，进而决定运行时在调用前是否上传该缓冲区、调用后是否下载。每个张量参数都带方向 ——
默认 `In`，或显式的 `pl.Out[...]` / `pl.InOut[...]`。

### 循环与循环携带值

`pl.range()` 在 IR 里构建循环。配合 `init_values`，循环会把值从一次迭代带到下一次 —— 这就是
PyPTO 里累加器的写法。

```python
@pl.function
def sum_elements(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[1], pl.FP32]:
    zero: pl.Tensor[[1], pl.FP32] = pl.create_tensor([1], dtype=pl.FP32)

    for i, (acc,) in pl.range(64, init_values=(zero,)):
        elem: pl.Tensor[[1], pl.FP32] = pl.slice(a, [1], [i])
        new_acc: pl.Tensor[[1], pl.FP32] = pl.add(acc, elem)
        acc_out: pl.Tensor[[1], pl.FP32] = pl.yield_(new_acc)

    return acc_out
```

1. `init_values=(zero,)` —— 进入第 0 次迭代时的携带值。
2. `for i, (acc,)` —— `i` 是循环下标，`acc` 是本次迭代的携带值。
3. `pl.yield_(new_acc)` —— 把 `new_acc` 作为下一次迭代的 `acc` 交出去。
4. `acc_out` —— 循环结束后，持有最后一次迭代 yield 的值。

`pl.yield_` 正是让这段代码保持 SSA 干净的原因：`acc` 从不被就地修改，每次迭代绑定一个新值
并把它 yield 出去。在循环外读 `acc_out`，就是最终值逃逸出来的方式。

不带携带值的循环既不需要 `init_values` 也不需要 `pl.yield_`：

```python
for i in pl.range(10):        # 0 .. 9
    ...

for i in pl.range(0, 100, 2): # 0, 2, 4, ... 98
    ...
```

### 多函数 program

`@pl.program` 把互相调用的函数组织成一个编译单元。

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
        out: pl.Tensor[[128, 128], pl.FP32] = pl.store(result, [0, 0], output)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        c: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], dtype=pl.FP32)
        c = self.kernel_add(a, b, c)
        return c
```

| 元素 | 含义 |
| ---- | ---- |
| `@pl.program` | 把一个类装饰成 `ir.Program` |
| `self` | 每个方法必需的首参；在 IR 中被剥除 |
| `self.kernel_add(...)` | program 内部的跨函数调用 |
| `type=pl.FunctionType.InCore` | 计算 kernel；运行在 AICore 上 |
| `type=pl.FunctionType.Orchestration` | 主机侧编排者；创建张量并派发 kernel |

这就是**控制面 / 执行面**的划分，也是 PyPTO 中最重要的一条结构性理念：

```text
main            (Orchestration)  —— 主机侧：分配 c，派发 kernel_add
  └── kernel_add (InCore)        —— 设备侧：load tile、计算、store
```

编排函数从不碰 tile 内存；InCore 函数从不分配张量、也不派发任务。把两者混在一起，是初次写
PyPTO 程序时最常见的结构性错误。

早期会遇到的 `FunctionType` 取值：`Opaque`（默认 —— 无特定执行上下文）、`InCore`、
`Orchestration` 和 `Inline`。其余的（`AIC`、`AIV`、`Group`、`Spmd`）由编译器生成，或属于
后续章节。

### 编译

```python
from pypto import ir
from pypto.backend import BackendType

compiled = ir.compile(
    VectorAddProgram,
    strategy=ir.OptimizationStrategy.Default,
    backend_type=BackendType.Ascend910B,
)
print(f"Generated code in: {compiled.output_dir}")
```

`ir.compile()` 返回的是 **`CompiledProgram`**，不是路径 —— 目录在 `compiled.output_dir`。
`CompiledProgram` 同时也是可调用的，这就是拿到 worker 后把它派发到设备的方式。

入门阶段最常用的参数（`ir.compile` 共有 15 个）：

| 参数 | 默认值 | 作用 |
| ---- | ------ | ---- |
| `program` | （必需） | 要编译的 `ir.Program` |
| `output_dir` | `None` → `<base>/<name>_<timestamp>` | codegen、报告与 pass dump 的落盘位置。`<base>` 取 `$PYPTO_PROG_BUILD_DIR`，未设置时为 `build_output` |
| `strategy` | `OptimizationStrategy.Default` | pass 流水线预设。`DebugTileOptimization` 存在但只是调试捷径 —— 优先用 `Default` |
| `dump_passes` | `True` | `bool`，或用 `PassDumpLevel`（`NONE` / `CONCISE` / `EXPLICIT`）做更细的控制。IR 快照写到 `output_dir/passes_dump/` |
| `backend_type` | `BackendType.Ascend910B` | 目标架构 —— `Ascend910B` 或 `Ascend950` |
| `skip_ptoas` | `False` | 只发射 `.pto`（MLIR）后停止，不调用 ptoas。ptoas 工具链不可用时很有用 |

其余九个参数（`verification_level`、`diagnostic_phase`、`disabled_diagnostics`、
`memory_planner`、`enable_pypto_l0c_double_buffer`、`profiling`、`platform`、
`distributed_config`、`analyze_auto_scopes_for_deps`）分别控制校验、诊断、内存规划和分布式
编译，会在对应主题处介绍。

`output_dir` 里会有什么：

```text
kernels/       生成的设备 kernel，每个 InCore 函数一个
orchestration/ 生成的主机侧 C++
report/        编译期报告，含性能提示
debug/         一个可直接运行的 `run.py`
passes_dump/   逐 pass 的 IR 快照（仅在开启 dump_passes 时）
```

### 不编译也能查看 IR

```python
print(vector_add.as_python())                 # 单个函数
print(VectorAddProgram.as_python())           # 整个 program
print(vector_add.as_python(concise=True))     # 去掉中间类型标注
```

## Edge Cases

> **致命陷阱：** `@pl.function` 是**解析**函数体，不是执行它。从普通 Python 里调用
> `vector_add(a, b)` 不会算出任何东西，函数体里的 `print()` 或 `assert` 在运行期也永远不会
> 执行。调试请读 `as_python()`，不要加打印语句。

| Symptom | Likely Cause | Fix |
| ------- | ------------ | --- |
| **被装饰的函数报 `AttributeError`** | 把 `ir.Function` 当成 Python 可调用对象 | 编译 program 后派发 `CompiledProgram`，或改用 `@pl.jit` |
| **解析错误指向某个赋值语句** | 赋值目标缺少类型标注 | 给每个绑定加标注：`x: pl.Tile[[64], pl.FP32] = ...` |
| **输出张量回来时没有变化** | 结果写进了未声明 `pl.Out[...]` 的参数 | 给参数加上方向包装，并通过 `pl.store` 写入 |
| **`compiled.output_dir` 为 `None` 或路径报错** | 把 `ir.compile` 的返回值当作字符串 | `ir.compile` 返回 `CompiledProgram`，读它的 `.output_dir` |
| **没有工具链的机器上 ptoas 失败** | codegen 调用了汇编器 | 传 `skip_ptoas=True`，在 `.pto` 处停下 |

`PYPTO_PROG_BUILD_DIR` 是**运行时环境变量** —— `PYPTO_PROG_BUILD_DIR=/tmp/out python kernel.py`
可以整体改变编译产物位置。请与 `SIMPLER_HOST_STRACE`、`SIMPLER_DFX` 区分开：后两者是运行时的
**编译期宏**（构建时 `-DXXX=1`），在 shell 里设置无效。

## See Also

- [安装](01-installation.md) —— 让这些例子能 import 起来。
- [语言指南](01-language_guide.md) —— 完整的类型系统、控制流、内存模型与作用域。
- [操作参考](02-operation_reference.md) —— `pl.*`、`pl.tensor.*`、`pl.tile.*` 三个命名空间的算子全貌。
- [在设备上运行](00-getting_started.md) —— 常驻设备张量、显式派发、性能基准与分布式执行。
- [Python IR 语法规范](../dev/language/00-python_syntax.md) —— parser 接受的精确语法。
