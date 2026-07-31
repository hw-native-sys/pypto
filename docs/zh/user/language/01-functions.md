# 函数与程序

一个 Python 函数如何变成 IR 函数、该用哪个装饰器，以及函数之间如何互相调用。

> **前置**：[类型](00-types.md)。

## Concept

装饰器不是包装你的函数 —— 它**解析你的源码**。函数体从不作为 Python 执行。这一个事实解释了后面大部分内容：闭包变量为什么是那样的行为、`pl.submit` 为什么在被装饰函数之外调用会抛异常、以及 kernel 体里的错误为什么在解析期带行号报出来，而不是调用时给你一条 traceback。

有两种写法，产生同样的 IR。

**`@pl.jit`** 把 kernel 写成普通函数。类型来自首次调用时的实参，函数随之特化，子函数自动被发现。`examples/` 用的是这种写法，本手册除本页外也一律用它。

**`@pl.program` 里的 `@pl.function`** 把一切前置声明清楚：类就是程序，每个方法是一个函数，函数间调用写成 `self.other(...)`。当你希望程序结构被显式写出来，或者某个工具需要在不调用 kernel 的情况下拿到 IR 时，用它。

选哪个不影响编译器看到的东西。`@pl.jit` 特化成的就是 `@pl.program` 源码 —— 你可以把它打印出来。

## Quickstart：同一个程序的两种写法

```python
import pypto.language as pl

# --- jit style -------------------------------------------------------------
@pl.jit.incore
def add_kernel(
    a: pl.Tensor[[128, 128], pl.FP32],
    b: pl.Tensor[[128, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    out = pl.add(a, b)
    return out

@pl.jit
def entry(
    a: pl.Tensor[[128, 128], pl.FP32],
    b: pl.Tensor[[128, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    out = add_kernel(a, b, out)      # sub-function discovered automatically
    return out
```

```python
# --- program style ---------------------------------------------------------
@pl.program
class Adder:
    @pl.function(type=pl.FunctionType.InCore)
    def add_kernel(self, a, b, out): ...

    @pl.function(type=pl.FunctionType.Orchestration)
    def entry(self, a, b, out):
        out = self.add_kernel(a, b, out)     # explicit cross-function call
        return out
```

| | `@pl.jit` | `@pl.program` |
| --- | --------- | ------------- |
| 函数类别 | 由装饰器变体决定 | 由 `type=` 决定 |
| 子函数连接 | 从函数体自动发现 | 写成 `self.method(...)` |
| 类型 | 由首次调用的实参特化 | 在注解里声明 |
| 拿到 IR | `entry.lower(*args)`，或 `entry.compile(*args)` 后取 `compiled.program.as_python()` | `Adder.as_python()` |

## Mechanics

### `@pl.jit` 家族

五个变体，一个对应一种 IR 函数类别，让单个程序可以横跨 host、chip、core 三级：

| 装饰器 | IR 目标 | 用于 |
| ------ | ------- | ---- |
| `@pl.jit` | `Orchestration` | chip 级入口，派发 InCore 工作 |
| `@pl.jit.host` | `level=HOST, role=Orchestrator` | HOST 级入口 —— 分配 window buffer、按 rank 派发 chip 编排器 |
| `@pl.jit.incore` | `InCore` | 设备 kernel（可接受 `level=` 指定层级） |
| `@pl.jit.inline` | `Inline` | 由 `InlineFunctions` 在每个调用点展开的辅助函数 |
| `@pl.jit.opaque` | `Opaque` | 独立 IR 函数，可包含编排循环与 `pl.at` 作用域 |

子函数依赖（`.incore` / `.inline` / `.opaque`）从入口函数体自动发现 —— 按名字调用即可。`@pl.jit.host` 入口还会额外发现 `@pl.jit`（chip 编排）依赖，因此一个完整的分布式程序无需任何 `@pl.program` 类。

下面这段只展示发现结构 —— kernel 体已省略，它用到的分布式类型属于尚未编写的分布式章节：

```python
import pypto.language.distributed as pld

@pl.jit.inline
def reduce_step(local, peer, out): ...

@pl.jit
def chip_orch(inp: pl.Tensor, out: pl.Out[pl.Tensor],
              data: pl.InOut[pld.DistributedTensor], peer: pl.Scalar[pl.INT32]):
    return reduce_step(inp, peer, out)      # auto-discovered sub-function

@pl.jit.host
def host_orch(
    inputs: pl.Tensor[[2, 1, 256], pl.FP32],
    outputs: pl.Out[pl.Tensor[[2, 1, 256], pl.FP32]],
):
    data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
    for r in pl.range(pld.world_size()):
        data = pld.window(data_buf, [1, 256], dtype=pl.FP32)
        chip_orch(inputs[r], outputs[r], data, (r + 1) % pld.world_size(), device=r)
    return outputs
```

普通 `@pl.jit` 入口**不会**发现其他 `@pl.jit` 入口 —— 只有 `.host` 跨越 chip 边界。这防止两个互不相关的顶层 kernel 被静默折叠进同一个程序。

`@pl.jit.host` 拒绝 `level=`（HOST 是隐含的）。

### 决定 jit kernel 能否编译的三条约束

这是新写的 `@pl.jit` 代码会依次撞上的三个失败。

**1. `@pl.jit` 入口体内不能放算子。** 它是 Orchestration 函数 —— 控制面。把算子放进 `with pl.at(level=pl.Level.CORE_GROUP):`，或者移进 `@pl.jit.incore` 子函数。

```python
@pl.jit
def bad(x: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    out = pl.add(x, x)        # ✗ Misplaced tensor op ... should be inside InCore block
    return out

@pl.jit
def good(x: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        out = pl.add(x, x)    # ✓
    return out
```

**2. `JITFunction` 没有 `as_python()`。** 在特化发生之前 IR 并不存在。调用 `lower(*args)` 拿 Pass 后的 `ir.Program`，或调用 `compile(*args)` 后读 `compiled.program.as_python()` 拿特化后、Pass 前的 IR。

**3. `compile()` 收的是 kernel 自己的参数，不是编译选项。** 编译期开关走 `config=RunConfig(...)`；误写的 `compile(skip_ptoas=True)` 会被当成 kernel 实参。`@pl.jit` 会自行检测 ptoas 是否可用，所以你不需要传 `skip_ptoas`。

### `@pl.function` 与 `@pl.program`

`@pl.function` 解析单个函数；`type=` 指明它所在的面：

| 函数类型 | 面 | 典型用途 |
| -------- | -- | -------- |
| `Opaque`（默认） | 尚未确定 | 独立构件；从使用它的位置取得所在的面 |
| `InCore` | 执行面 | load / compute / store kernel |
| `Orchestration` | 控制面 | 创建张量、派发 InCore 任务 |
| `Inline` | 无 | 在每个调用点展开，不留下函数 |

`@pl.program` 把若干函数组成可编译程序。每个方法都要有 `self`（会从 IR 中剥离），跨函数调用写作 `self.method(...)`，被装饰的类会变成 `ir.Program` —— 不再是一个你能实例化的 Python 类。

在 `@pl.program` 内部调用一个独立的 `@pl.function`，它会作为一个独立函数被加入该程序。而 `@pl.inline`（以及 `@pl.jit.inline`）则在调用点展开，不留下函数。

```python
@pl.inline
def normalize(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    return pl.mul(x, 2.0)
```

被装饰的对象是一个 `pl.InlineFunction` —— 供解析器展开的模板，而不是你能从 Python 调用的函数。

### 运行时作用域的放置

默认情况下编译器为你插入 AUTO 运行时作用域（`PTO2_SCOPE`）。传 `auto_scope=False` 可以改用 `with pl.scope():` 手工放置 —— `@pl.jit`、`@pl.jit.host`、`@pl.jit.inline` 接受它，`.incore` / `.opaque` 拒绝（它们会被 outline 成独立 kernel）。inline 体会被拼接进调用方，因此其中手工放置的作用域落在调用方。见 [作用域与任务](04-scopes-and-tasks.md) 与 [MaterializeRuntimeScopes](../../dev/passes/42-materialize_runtime_scopes.md)。

### 把编译与派发拆开

`@pl.jit` kernel 通常把特化 + 编译 + 派发融合进一次 `kernel(*args)` 调用。`JITFunction.compile(*sample_args)` 在编译后停下并交还 `CompiledProgram` —— 用于自行驱动 `ChipWorker`、检查 `compiled.output_dir` 下的产物，或提前做 codegen 校验。

```python
compiled = my_kernel.compile(sample_x, sample_w, sample_out)
print("artifacts in:", compiled.output_dir)
```

返回的对象就是 JIT 缓存持有的那个，因此之后用同一特化 key 再调用会拿到完全相同的实例。

`lower(*sample_args)` 比它早停一站：只跑 Pass 并返回 Pass 后的 `ir.Program`，不做代码生成、不调 `ptoas`、不写产物、不写缓存。要读降级后的 IR 就用它；要检查代码生成本身就用 `compile()`。两者都接受 `config=RunConfig(...)`，但 `lower()` 会忽略其中的运行时与产物字段。编译选项与运行时接口属于执行章节，该章尚未编写 —— 目前见 [编译程序](../01-language_guide.md)。

### 外部 C++ kernel

手写的 C++ kernel 可以像普通函数一样被调用。见 [集成手写 C++ Kernel](../../dev/language/01-external-kernels.md)。

## Edge Cases

> **致命陷阱：** 验证新的 `@pl.jit` 示例要用完整的 `compile()`，不能只用 `lower()`。`lower()` 在 Pass 之后就停下，所以上面那条"Orchestration 体内放算子"的错误根本不会触发 —— kernel 看上去通过了，直到有人真正去跑它才失败。

| 症状 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| **`Misplaced tensor op ... should be inside InCore block`** | 算子直接写在 `@pl.jit` 体内 | 包进 `with pl.at(level=pl.Level.CORE_GROUP):` 或移入 `@pl.jit.incore` |
| **`AttributeError: 'JITFunction' object has no attribute 'as_python'`** | 在 IR 尚不存在时打印它 | 用 `f.lower(*args)`，或 `f.compile(*args)` 后取 `compiled.program.as_python()` |
| **`lower()` 通过但 `compile()` 失败** | `lower()` 不执行代码生成 | 预期行为 —— 代码生成检查用 `compile()` |
| **某个编译选项被静默忽略** | 它作为 kernel 实参传给了 `compile()` | 改传 `config=RunConfig(...)` |
| **程序里少了第二个顶层 kernel** | 普通 `@pl.jit` 不发现其他 `@pl.jit` 入口 | 改用 `@pl.jit.host`，或把被调方改成 `.incore` / `.opaque` |
| **`auto_scope=False` 被拒绝** | 用在了 `.incore` / `.opaque` 上 | 放到入口或 `.inline` 辅助函数上 |
| **`@pl.program` 方法缺 `self`** | 每个方法都需要 | 补上 `self`；它会从 IR 中剥离 |

## See Also

- [控制流](02-control-flow.md) —— 这些函数体里的循环与条件。
- [作用域与任务](04-scopes-and-tasks.md) —— `pl.at`、运行时作用域与任务派发。
- [快速上手](../02-quickstart.md) —— 同样的装饰器在一个完整例子里的用法。
- [InlineFunctions](../../dev/passes/01-inline_functions.md) —— `Inline` 体如何被拼接。
- [集成手写 C++ Kernel](../../dev/language/01-external-kernels.md) —— 调用外部 kernel。
