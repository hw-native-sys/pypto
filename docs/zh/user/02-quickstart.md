# 快速上手

编写、编译并查看你的第一批 PyPTO kernel —— 在张量级，数据放置由编译器替你决定。

> **前置**：PyPTO 已安装且可导入 —— 见[安装](01-installation.md)。
> 除最后一节外，全部内容在一台仅 `pip install` 过的机器上都能跑 —— 不需要 NPU，也不需要
> ptoas：`@pl.jit` 会自己检测 ptoas 是否存在并相应调整。

## Concept

PyPTO 的 kernel 是被**解析**而非被执行的 Python 源码。`@pl.jit` 读取被装饰函数的函数体，把它
特化成 PyPTO IR；在你编译这份 IR 并派发它之前，什么都不会运行。

本页完全停留在**张量级**：你命名整个数组、对它们施加算子，由编译器决定什么时候把什么放到片上。
下面**没有任何 `pl.load` 或 `pl.store`**。Tile 级写法 —— 自己命名片上缓冲区、自己搬运数据 ——
是另一个主题，它是什么、什么时候才需要它，见[编程模型](03-programming-model.md)。

有两个结构性事实贯穿所有示例：

- `@pl.jit` 标记的是一个**芯片级入口**，属于控制面代码。计算属于执行面，所以算子要写在
  `with pl.at(level=pl.Level.CORE_GROUP):` 里面 —— 这个作用域的含义就是"以下在片上运行"。
  漏掉它会失败并报 *"Misplaced tensor op ... should be inside InCore block"*。
- 输出通过 `pl.Out[...]` 参数写回，而不是作为新数组返回。

```python
import pypto.language as pl
import torch
```

## Quickstart：逐元素加法

```python
import pypto.language as pl

@pl.jit
def add(
    a: pl.Tensor[[128, 128], pl.FP32],
    b: pl.Tensor[[128, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        out = pl.add(a, b)
    return out

compiled = add.compile()
print(f"Generated code in: {compiled.output_dir}")
```

| 行 | 作用 |
| -- | ---- |
| `@pl.jit` | 首次编译时把函数体特化成一个 Orchestration 入口 |
| `a: pl.Tensor[[128, 128], pl.FP32]` | DDR 中一个 128×128 的 FP32 数组。方向默认为 `In` |
| `out: pl.Out[pl.Tensor[...]]` | **方向**：这个参数是被写的，不是被读的 |
| `with pl.at(level=pl.Level.CORE_GROUP)` | 标记片上代码块。算子只在这里面合法 |
| `out = pl.add(a, b)` | 对整个张量做逐元素加。没有偏移、没有形状、没有数据搬运 |
| `return out` | 返回被写入的张量 |
| `add.compile()` | 跑完流水线并返回一个 `CompiledProgram` |

注意这个 kernel 里**没有**什么：没有 tile 类型、没有 `pl.load`、没有 `pl.store`、没有内存空间。
这些全由编译器的 `ConvertTensorToTileOps` pass 插入 —— 结果可以在
`compiled.output_dir/passes_dump/` 里看到。

`pl.Out[...]` 是承重的而非装饰性的：它告诉编译器这块缓冲区会被写，进而决定运行时在调用前是否
上传、调用后是否下载。每个张量参数都带方向 —— 默认 `In`，或显式的 `pl.Out[...]` /
`pl.InOut[...]`。

> **为什么 `pl.at` 不是可选的。** 去掉那一行、其余不变，编译会在编排层 codegen 处失败：
> *"Misplaced tensor op 'tensor.add' in Orchestration function (should be inside InCore
> block)"*。`@pl.jit` 入口是控制面代码，而这个作用域正是把计算搬到执行面的东西。

## Mechanics

### 算子串联

中间值就是普通的 Python 名字。它们不需要标注，也不需要为它们声明缓冲区 —— 这条链需要什么，
编译器就分配什么：

```python
@pl.jit
def add_then_square(
    a: pl.Tensor[[128, 128], pl.FP32],
    b: pl.Tensor[[128, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        s = pl.add(a, b)
        out = pl.mul(s, s)
    return out
```

形状与 dtype 要**内联写在标注里**。模块级别名（`T = pl.Tensor[[128, 128], pl.FP32]`）**不行**：
parser 把标注当源码文本读，解析不了别名，你会得到 *"Parameter 'a' missing type annotation"*。

### 循环

`pl.range()` 在 IR 里构建循环，位置在片上作用域内部。跨迭代携带的值写成普通的重新赋值：

```python
@pl.jit
def accumulate(
    a: pl.Tensor[[128, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        t = pl.add(a, a)
        for i in pl.range(3):
            t = pl.add(t, a)      # 跨迭代携带
        out = pl.mul(t, t)
    return out
```

重复绑定 `t` 看起来像就地修改，其实不是：IR 是 SSA 的，parser 给每次迭代的值起独立的名字，
并把它作为携带值穿过循环。循环之后读 `t`，读到的是最后一次迭代的结果。

循环形态：

```python
for i in pl.range(10):        # 0 .. 9
for i in pl.range(0, 100, 2): # 0, 2, 4, ... 98
```

### 把工作拆到多个函数

超出单个 kernel 时，把计算放进 `@pl.jit.incore` 子函数，由入口派发它。`.incore` 函数本身就在
执行面上，所以它不需要 `pl.at`：

```python
@pl.jit.incore
def add_kernel(
    a: pl.Tensor[[128, 128], pl.FP32],
    b: pl.Tensor[[128, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    out = pl.add(a, b)
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
  └── add_kernel (@pl.jit.incore)       —— 执行面：计算
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

### 编译

```python
compiled = add.compile()
print(f"Generated code in: {compiled.output_dir}")
```

`compile()` 返回的是 **`CompiledProgram`**，不是路径。`compiled.output_dir` 是一个
`pathlib.Path`，里面有：

```text
kernels/       生成的设备 kernel，每个 InCore 函数一个
orchestration/ 生成的主机侧 C++
ptoas/         .pto（MLIR）及其汇编产物 —— 仅当 ptoas 可用时
report/        编译期报告，含性能提示
debug/         一个可直接运行的 `run.py`
passes_dump/   逐 pass 的 IR 快照
```

**`compile()` 不需要任何 ptoas 开关。** `@pl.jit` 会自己查这个二进制 —— `$PTOAS_ROOT/ptoas`
或 `PATH` 上的 `ptoas` —— 不存在时自动跳过汇编步骤。只装了 Python 包的机器照样能拿到 IR 和
生成的 C++。

`compile()` 从哪里取形状：

| 签名写法 | 调用方式 |
| -------- | -------- |
| 完整标注 `pl.Tensor[[...], dtype]` | `kernel.compile()` —— 完全不带参数 |
| 裸 `pl.Tensor` | `kernel.compile(a, b, out)`，传样例张量 |

样例张量只被读取形状与 dtype，内容从不被访问，所以 `torch.empty(...)` 就够了。

> **`compile()` 的参数是 kernel 的，不是编译器的。** `compile(*args, **kwargs)` 绑定的是被装饰
> 函数自己的参数。把 `ir.compile()` 的选项传到这里 —— 例如 `compile(skip_ptoas=True)` ——
> 要么被当作意外的 kernel 参数拒绝，要么被**静默忽略**。编译侧选项要通过
> `config=RunConfig(...)` 传，它的编译类开关会转发给 `ir.compile()`。

只想检查一个 kernel、不产出代码时，`lower()` 会特化 JIT 函数、运行配置对应的 Pass 流水线，
并返回 Pass 后的 `ir.Program`：

```python
import torch

x = torch.zeros((128, 128), dtype=torch.float32)
program = add.lower(x, x, x)
```

它不会执行代码生成，也不会填充编译缓存。这让它很快，但也意味着它**抓不到** codegen 阶段的
错误，比如上面那个 misplaced-tensor-op。需要验证代码生成时请使用 `compile()`。

### 读 IR

`JITFunction` 没有 `as_python()`。可以直接读取 `lower()` 返回的 `ir.Program`，也可以读取
`compile()` 返回的 `CompiledProgram` 中保存的 `program`：

```python
print(program.as_python())
print(compiled.program.as_python())
```

回来的是你那些 jit 函数被特化成的 `@pl.program` 类，这也是看清 `@pl.jit` 到底做了什么最直接的
方式；在张量级，它还让你看清编译器替你补了什么。对比 `ConvertTensorToTileOps` 前后的 pass
dump，就能看到 `pl.tensor.add` 变成 tile load、tile add 和 store 的过程。

## 在硬件上运行

> **需要运行时和一块设备或模拟器平台。** 本节以上的所有内容都不需要。

```python
import torch
from pypto.runtime import RunConfig

a = torch.full((128, 128), 2.0, dtype=torch.float32)
b = torch.full((128, 128), 3.0, dtype=torch.float32)
out = torch.zeros((128, 128), dtype=torch.float32)

add(a, b, out, config=RunConfig())          # 编译、缓存、派发
assert torch.allclose(out, a + b, rtol=1e-5, atol=1e-5)
```

直接调用一个 `@pl.jit` 函数会一次做完全部事情：按实参的形状与 dtype 特化、编译、缓存、派发。
后续用相同形状调用会复用缓存的编译产物。`examples/hello_world.py` 就是这个模式，只是写在
tile 级。

## Edge Cases

> **致命陷阱：** `@pl.jit` 是**解析**函数体，不是执行它。函数体里的 `print()` 或 `assert`
> 在运行期永远不会执行，用调试器单步进去看到的是解析过程而不是计算过程。调试请读
> `compiled.program.as_python()`。

| Symptom | Likely Cause | Fix |
| ------- | ------------ | --- |
| **`Misplaced tensor op ... should be inside InCore block`** | 算子直接写在 `@pl.jit` 函数体里 | 用 `with pl.at(level=pl.Level.CORE_GROUP):` 包起来，或移进 `@pl.jit.incore` 子函数 |
| **`Parameter 'a' missing type annotation`** | 标注是通过模块级别名写的 | 在签名里内联写 `pl.Tensor[[...], dtype]` |
| **`Cannot reassign 'out' with a different type`** | 表达式的 dtype 与声明的 `Out` dtype 不一致 | 让它们一致，或把结果绑到一个新名字上 |
| **`got an unexpected keyword argument 'skip_ptoas'`** | 把 `ir.compile()` 的选项传给了 `compile()` | 编译选项通过 `config=RunConfig(...)` 传 |
| **输出张量回来时没有变化** | 结果写进了未声明 `pl.Out[...]` 的参数 | 加上方向 |
| **`lower()` 成功但 `compile()` 失败** | `lower()` 不执行代码生成 | 预期行为 —— 代码生成检查用 `compile()` |
| **`AttributeError: as_python`** | 在 jit 函数上调用了它 | 它在 IR 上：`compiled.program.as_python()` |

`PYPTO_PROG_BUILD_DIR` 是**运行时环境变量** —— `PYPTO_PROG_BUILD_DIR=/tmp/out python kernel.py`
可以整体改变编译产物位置。请与 `SIMPLER_HOST_STRACE`、`SIMPLER_DFX` 区分开：后两者是运行时的
**编译期宏**（构建时 `-DXXX=1`），在 shell 里设置无效。

## See Also

- [安装](01-installation.md) —— 让这些例子能 import 起来。
- [编程模型](03-programming-model.md) —— 张量 / tile / block 三级、两个面、内存层次与执行模型。
- [语言指南](01-language_guide.md) —— 完整表面：tile 级写法、`pl.load` / `pl.store`、内存空间，以及 `@pl.jit` 所特化成的 `@pl.function` / `@pl.program` 形态。
- [操作参考](02-operation_reference.md) —— `pl.*`、`pl.tensor.*`、`pl.tile.*` 三个命名空间的算子全貌。
- [在设备上运行](00-getting_started.md) —— 常驻设备张量、显式派发、性能基准、分布式执行。
- `examples/kernels/` —— 同一套 `@pl.jit` 写法下的 tile 级 kernel。
