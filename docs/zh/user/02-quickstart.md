# 快速上手

用 `@pl.jit` 编写、编译并查看你的第一批 PyPTO kernel。

> **前置**：PyPTO 已安装且可导入 —— 见[安装](01-installation.md)。
> 编译与查看类的例子只需要安装本体。唯一真正**运行** kernel 的例子已单独标注，它需要运行时
> 加一块设备或模拟器平台。

## Concept

PyPTO 的 kernel 是被**解析**而非被执行的 Python 源码。`@pl.jit` 读取被装饰函数的函数体，把它
特化成 PyPTO IR；在你编译这份 IR 并派发它之前，什么都不会运行。

`@pl.jit` 标记的是一个**芯片级入口** —— 一个 Orchestration 函数，属于控制面代码。控制面代码
不能碰片上内存，所以 jit 函数体要走到执行面只有两条路：

- `with pl.at(level=pl.Level.CORE_GROUP):` —— 就地开一个片上作用域。这是简短形态，多数单
  kernel 例子都这么写。
- 调用一个 `@pl.jit.incore` 子函数 —— 编译器会从入口体内自动发现它，并把它外提成独立的设备
  kernel。

在 jit 函数体里直接写 `pl.load`、两条路都不走，编译期会失败。这不是怪癖：它就是控制面 / 执行面
的划分以报错形式呈现出来，也是 PyPTO 中最重要的一条结构性理念。

```python
import pypto.language as pl
import torch
```

## Quickstart：逐元素加法

最小的完整 kernel —— 与 `examples/hello_world.py` 是同一个。

```python
import pypto.language as pl
import torch

@pl.jit
def tile_add(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        tile_a = pl.load(a, [0, 0], [128, 128])
        tile_b = pl.load(b, [0, 0], [128, 128])
        tile_c = pl.add(tile_a, tile_b)
        pl.store(tile_c, [0, 0], c)
    return c

a = torch.full((128, 128), 2.0, dtype=torch.float32)
b = torch.full((128, 128), 3.0, dtype=torch.float32)
c = torch.zeros((128, 128), dtype=torch.float32)

# 跑完整条 pass 流水线。不需要设备，也不需要 ptoas。
tile_add.compile_for_test(a, b, c)
print("compiles")
```

| 行 | 作用 |
| -- | ---- |
| `@pl.jit` | 首次编译时把函数体特化成一个 Orchestration 入口 |
| `a: pl.Tensor` | 一个 DDR 张量。没给形状 —— 形状从你传入的 torch 张量读取 |
| `c: pl.Out[pl.Tensor]` | **方向**：这个参数是被写的，不是被读的 |
| `with pl.at(level=pl.Level.CORE_GROUP)` | 开一个片上作用域；tile 操作只在这里面合法 |
| `pl.load(a, [0, 0], [128, 128])` | DDR → 片上 tile。`[0, 0]` 是偏移，`[128, 128]` 是形状 |
| `pl.store(tile_c, [0, 0], c)` | 片上 tile → DDR，写进 `Out` 参数 |
| `return c` | 返回被写入的张量 |

`pl.Out[...]` 是承重的而非装饰性的：它告诉编译器这块缓冲区会被写，进而决定运行时在调用前是否
上传、调用后是否下载。每个张量参数都带方向 —— 默认 `In`，或显式的 `pl.Out[...]` /
`pl.InOut[...]`。

`compile_for_test(...)` 跑完整条 pass 流水线、在代码生成之前停下，因此是检查一个 kernel 是否
形态正确最便宜的手段。它既不需要 ptoas 也不需要设备，这就是本页例子都用它的原因。

### 在硬件上运行

> **需要运行时和一块设备或模拟器平台。** 本行以上的所有内容都不需要。

```python
from pypto.runtime import RunConfig

tile_add(a, b, c, config=RunConfig())          # 编译、缓存、派发
assert torch.allclose(c, a + b, rtol=1e-5, atol=1e-5)
```

直接调用一个 `@pl.jit` 函数会一次做完全部事情：按实参的形状与 dtype 特化、编译、缓存结果、派发。
后续用相同形状调用会复用缓存的编译产物。

## Mechanics

### 形状：来自实参，还是来自签名

`a: pl.Tensor` 把形状交给调用点。改成完整标注，契约就落在签名里 —— 这样连样例张量都不需要：

```python
@pl.jit
def tile_add_128(
    a: pl.Tensor[[128, 128], pl.FP32],
    b: pl.Tensor[[128, 128], pl.FP32],
    c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        pl.store(pl.add(pl.load(a, [0, 0], [128, 128]),
                        pl.load(b, [0, 0], [128, 128])), [0, 0], c)
    return c

compiled = tile_add_128.compile(skip_ptoas=True)   # 不需要 torch.empty(...)
```

探索阶段用裸形态；签名很大的 kernel 用完整标注形态 —— 与其罗列一串用完就丢的
`torch.empty(...)`，不如让签名把契约说清楚一次。

### 片上作用域内的循环

`pl.range()` 在 IR 里构建循环。它放在 `pl.at` **内部** —— 这是设备 kernel 自己跑的循环，不是
主机在迭代：

```python
@pl.jit
def double_thrice(x: pl.Tensor, y: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        acc = pl.load(x, [0, 0], [128, 128])
        for i in pl.range(3):
            acc = pl.add(acc, acc)      # 重复绑定没问题 —— parser 会重命名
        pl.store(acc, [0, 0], y)
    return y
```

重复绑定 `acc` 看起来像就地修改，其实不是：IR 是 SSA 的，parser 给每次迭代的值起独立的名字，
并把它作为携带值穿过循环。循环之后读 `acc`，读到的是最后一次迭代的结果。

循环形态：

```python
for i in pl.range(10):        # 0 .. 9
for i in pl.range(0, 100, 2): # 0, 2, 4, ... 98
```

### 把工作拆到多个函数

只要超出单个 kernel，就把计算放进 `@pl.jit.incore` 子函数，由 `@pl.jit` 入口派发它。入口
不需要 `pl.at` —— 子函数**本身**就是执行面。

```python
@pl.jit.incore
def add_kernel(
    a: pl.Tensor[[128, 128], pl.FP32],
    b: pl.Tensor[[128, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    ta = pl.load(a, [0, 0], [128, 128])
    tb = pl.load(b, [0, 0], [128, 128])
    pl.store(pl.add(ta, tb), [0, 0], out)
    return out

@pl.jit
def add_program(
    a: pl.Tensor[[128, 128], pl.FP32],
    b: pl.Tensor[[128, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    return add_kernel(a, b, out)      # 自动发现 —— 无需注册
```

```text
add_program   (@pl.jit, Orchestration)  —— 控制面：派发
  └── add_kernel (@pl.jit.incore)       —— 执行面：load、计算、store
```

`@pl.jit` 家族，每种 IR 函数类型对应一个装饰器：

| 装饰器 | 特化成 | 用于 |
| ------ | ------ | ---- |
| `@pl.jit` | Orchestration | 芯片级入口 |
| `@pl.jit.incore` | InCore | 设备 kernel，外提成独立文件 |
| `@pl.jit.inline` | Inline | 在每个调用点展开的辅助函数 |
| `@pl.jit.opaque` | Opaque | 独立 IR 函数，可包裹循环与 `pl.at` 作用域 |
| `@pl.jit.host` | `level=HOST, role=Orchestrator` | 分布式（多卡）程序的 HOST 入口 |

子函数从入口体内自动发现，所以你按名字调用即可。有一个刻意的例外：普通 `@pl.jit` 入口**不会**
发现其他 `@pl.jit` 入口 —— 只有 `.host` 会跨越芯片边界，这样可以避免两个不相关的顶层 kernel
被静默折叠进同一个 program。

### 编译，以及读懂产出

```python
compiled = add_program.compile(skip_ptoas=True)
print(f"Generated code in: {compiled.output_dir}")
```

`compile()` 返回的是 **`CompiledProgram`**，不是路径。`compiled.output_dir` 是一个
`pathlib.Path`，里面有：

```text
kernels/       生成的设备 kernel，每个 InCore 函数一个
orchestration/ 生成的主机侧 C++
report/        编译期报告，含性能提示
debug/         一个可直接运行的 `run.py`
passes_dump/   逐 pass 的 IR 快照（仅在开启 dump_passes 时）
```

`skip_ptoas=True` 会在发射 `.pto`（MLIR）之后停下。去掉它才会得到编译好的 C++ kernel
wrapper —— 那一步要调用 **ptoas**，它与 Python 包是分开分发的。`compile()` 接受与
`ir.compile()` 相同的选项（后者共 15 个参数），入门阶段最常用的几个：

| 参数 | 默认值 | 作用 |
| ---- | ------ | ---- |
| `output_dir` | `None` → `<base>/<name>_<timestamp>` | 产物落盘位置。`<base>` 取 `$PYPTO_PROG_BUILD_DIR`，未设置时为 `build_output` |
| `strategy` | `OptimizationStrategy.Default` | pass 流水线预设。`DebugTileOptimization` 只是调试捷径 —— 优先用 `Default` |
| `dump_passes` | `True` | `bool`，或 `PassDumpLevel`（`NONE` / `CONCISE` / `EXPLICIT`） |
| `backend_type` | `BackendType.Ascend910B` | 目标架构 —— `Ascend910B` 或 `Ascend950` |
| `skip_ptoas` | `False` | 停在 `.pto`，不调用 ptoas |

**要看 IR，得经过已编译的 program。** `JITFunction` 没有 `as_python()` —— 它只有 `compile`
与 `compile_for_test` —— 所以 IR 要等其中之一把它产出来之后才可读：

```python
print(compiled.program.as_python())
```

回来的是你那些 jit 函数被特化成的 `@pl.program` 类，这也是看清 `@pl.jit` 到底做了什么最直接的
方式：

```python
@pl.program
class _jit_add_program:
    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def add_kernel(a: pl.Tensor[[128, 128], pl.FP32], ...):
        ta: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec] = pl.tile.load(a, [0, 0], [128, 128], ...)
        ...
```

注意 parser 替你补齐了什么：`pl.load` 解析成了 `pl.tile.load`，tile 带上了 `pl.Mem.Vec`，
子函数被指派了 level 与 role。kernel 行为异常时，读这份输出是确认"编译器究竟构建了什么"最快的
办法。

## Edge Cases

> **致命陷阱：** `@pl.jit` 是**解析**函数体，不是执行它。函数体里的 `print()` 或 `assert`
> 在运行期永远不会执行，用调试器单步进去看到的是解析过程而不是计算过程。调试请读
> `compiled.program.as_python()`。

| Symptom | Likely Cause | Fix |
| ------- | ------------ | --- |
| **编译在编排层 codegen 处报错** | tile 操作直接写在 `@pl.jit` 函数体里 | 用 `with pl.at(level=pl.Level.CORE_GROUP):` 包起来，或移进 `@pl.jit.incore` 子函数 |
| **`missing a required argument`** | 参数是裸 `pl.Tensor`，但 `compile()` / `compile_for_test()` 没给样例 | 传样例张量，或把参数完整标注 |
| **输出张量回来时没有变化** | 结果写进了未声明 `pl.Out[...]` 的参数 | 加上方向，并通过 `pl.store` 写入 |
| **没有工具链的机器上 ptoas 失败** | codegen 调用了汇编器 | 传 `skip_ptoas=True`，或改用 `compile_for_test()` |
| **`AttributeError: as_python`** | 在 jit 函数上调用了 `as_python()` | 它在 IR 上：`compiled.program.as_python()` |

`PYPTO_PROG_BUILD_DIR` 是**运行时环境变量** —— `PYPTO_PROG_BUILD_DIR=/tmp/out python kernel.py`
可以整体改变编译产物位置。请与 `SIMPLER_HOST_STRACE`、`SIMPLER_DFX` 区分开：后两者是运行时的
**编译期宏**（构建时 `-DXXX=1`），在 shell 里设置无效。

## See Also

- [安装](01-installation.md) —— 让这些例子能 import 起来。
- [编程模型](03-programming-model.md) —— `pl.at`、两个面、内存层次背后的抽象。
- [语言指南](01-language_guide.md) —— 完整表面，包括 `@pl.jit` 所特化成的 `@pl.function` / `@pl.program` 类形态。
- [操作参考](02-operation_reference.md) —— `pl.*`、`pl.tensor.*`、`pl.tile.*` 三个命名空间的算子全貌。
- [在设备上运行](00-getting_started.md) —— 常驻设备张量、显式派发、性能基准、分布式执行。
- `examples/kernels/` —— 那里的每个 kernel 都是这套写法。
