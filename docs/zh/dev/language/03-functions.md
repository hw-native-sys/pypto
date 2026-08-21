# 函数与程序结构

函数声明形式、参数方向、跨模块复用，以及如何把 IR 打印回 Python 语法。

## 函数

```python
# Single return type
def function_name(param1: pl.INT64, param2: pl.FP32) -> pl.INT64:
    x: pl.INT64 = param1 + 1
    return x

# Multiple return types
def function_name(x: pl.INT64) -> tuple[pl.INT64, pl.INT64]:
    y: pl.INT64 = x + 1
    z: pl.INT64 = x * 2
    return y, z

# No return types
def function_name(x: pl.INT64):
    y: pl.INT64 = x + 1

# With function type
@pl.function(type=pl.FunctionType.Orchestration)
def orchestrator(n: pl.INT64) -> pl.INT64:
    return n + 1

@pl.function(type=pl.FunctionType.InCore)
def aicore_kernel(x: pl.INT64) -> pl.INT64:
    return x * 2
```

### 函数类型

| 类型 | 用途 | 描述 |
| ---- | ---- | ---- |
| `pl.FunctionType.Opaque` | 默认 | 未指定的函数类型 |
| `pl.FunctionType.Orchestration` | Host/AICPU | 控制流和依赖分析 |
| `pl.FunctionType.InCore` | AICore | AICore 子图执行（未特化） |
| `pl.FunctionType.AIC` | Cube 核心 | Cube 核心内核（特化的 InCore） |
| `pl.FunctionType.AIV` | Vector 核心 | Vector 核心内核（特化的 InCore） |
| `pl.FunctionType.Group` | 多核 | AIC + AIV 内核的协调调度组 |
| `pl.FunctionType.Graph` | 主机/AICPU | 可录制的编排片段，由 `host_build_graph` runtime 回放（详见下文） |

未指定类型时, 函数默认为 `Opaque`。

### Graph 片段

`pl.FunctionType.Graph` 把函数标记为可录制的编排片段。在 `host_build_graph`
runtime 下，每个调用点变成一次 task launch：runtime 在第一次调用时录制、之后回放，
于是 N 次调用只付一次建图代价，而不是 N 次：

```python
@pl.program
class Decoder:
    @pl.function(type=pl.FunctionType.Graph)
    def layer(self, cur, normed, next_hidden, wq, layer_base: pl.Scalar[pl.INDEX]):
        ...

    @pl.function
    def decode(self, cur, normed, next_hidden, wq):
        for i in pl.range(40):
            self.layer(cur, normed, next_hidden, wq, i * 5120)
```

一个 Graph 函数就是一份被录制的拓扑：runtime 用生成的 C++ 函数地址来标识这份录制，
因此不存在需要命名、也不需要保证唯一的 cache key。

### 参数方向

参数可以使用包装类型指定 `In` (默认)、`Out` 或 `InOut` 方向:

```python
@pl.function(type=pl.FunctionType.InCore)
def kernel(
    qi: pl.Tensor[[16, 128], pl.BF16],                   # In (default)
    output: pl.InOut[pl.Tensor[[16, 128], pl.FP32]],      # InOut
    result: pl.Out[pl.Tensor[[16, 128], pl.FP32]],        # Out
    scale: pl.Scalar[pl.FP32],                             # In (default)
) -> pl.Tensor[[16, 128], pl.FP32]:
    ...
```

| 方向 | 包装类型 | 描述 |
| ---- | -------- | ---- |
| `In` | 无 (默认) | 只读输入参数 |
| `Out` | `pl.Out[type]` | 只写输出参数 |
| `InOut` | `pl.InOut[type]` | 读写输入/输出参数 |

**约束:** `Scalar` 参数不能使用 `InOut` 方向 (会抛出 `ParserTypeError`)。

#### 写入 `Out` / `InOut` 参数

**裸赋值不会写入参数。** 它只是重新绑定 Python 名字: 参数 Var 指向一个新算出来的
张量，调用方的 buffer 完全没被碰过。程序照样能编译、能运行。调用方拿回什么取决于方向:
`Out` buffer 是新分配且从未初始化的，读出来是垃圾值; `InOut` buffer 里仍是调用方传进来的
输入，结果是悄悄地陈旧。

```python
out = pl.add(a, b)          # ❌ 什么也没写
out[:] = pl.add(a, b)       # ✅ 写入整个张量
```

要真正写入参数，请用下标形式:

| 写法 | 写入内容 | 适用场景 |
| ---- | -------- | -------- |
| `out[:] = <expr>` | 整个张量 | 结果就是整个输出 |
| `out[<slices>] = <expr>` | 该子窗口 | 只写输出的一部分 |
| `out = pl.assemble(out, <expr>, <offset>)` | `<offset>` 处的窗口 | 下标语法糖展开后的显式形式 |
| `out = <expr>` | **什么也不写** | 永远不要这么写——见下面的告警 |

只有第一行和第三行等价，且仅当切片覆盖全部范围、`<offset>` 全为 0 时才等价。

##### `OutParamWriteDropped` 告警

编译器会对丢失写入的裸赋值报告:

```text
[warning] [OutParamWriteDropped] (pipeline_input) Assigning to Out parameter 'out'
in function 'main' rebinds the name only — the caller's buffer is never written.
Use 'out[:] = <expr>' to write the whole tensor, or 'out[<slices>] = <expr>' for
a sub-window. at repro.py:12:9
```

该检查基于数据流而非语法。一个值可以不提参数名就流回该参数——例如经由 loop carry——
那是真正的回写，因此不会告警:

```python
for col, (d,) in pl.range(0, n, chunk, init_values=(data,)):
    d = pl.store(local, [0, col], d)
    staged = pl.yield_(d)
data = pld.tensor.allreduce(staged, signal, ...)   # `staged` 就是 `data`; 不告警
```

该检查刻意保守: 仅仅*读取*参数的值 (例如 `out = pl.add(out, b)`) 同样会丢失写入，
但不会被报告。要区分「读取参数」和「通过参数回写」需要算子注册表并未记录的逐算子写语义，
而对正确代码误报的代价高于漏报。需要全量写入时请写成 `out[:] = pl.add(out, b)`。

如果该检查对你的程序没有价值，可以用 `disabled_diagnostics` 关闭:

```python
disabled = passes.DiagnosticCheckSet()
disabled.insert(passes.DiagnosticCheck.OutParamWriteDropped)
ir.compile(program, disabled_diagnostics=disabled)
```

## `@pl.program` 如何定位类定义

`@pl.program` 是*从源码解析*类体的，因此它必须先找到生成被装饰对象的那条 `class`
语句。仅凭类名无法确定这一点: 同一个函数可以在多个分支里定义同名类，它们的
`__qualname__` 完全相同。

装饰器通过类体中各方法的行号来消歧，因此每个分支都按自己的源码解析:

```python
def make(case):
    if case == "add":
        @pl.program
        class Prog:                                # 解析*这个*类体
            @pl.function
            def main(self, x: pl.Tensor[[8], pl.FP32]) -> pl.Tensor[[8], pl.FP32]:
                return pl.add(x, 1.0)
        return Prog

    @pl.program
    class Prog:                                    # 这个则解析*这个*类体
        @pl.function
        def main(self, x: pl.Tensor[[8], pl.FP32]) -> pl.Tensor[[8], pl.FP32]:
            return pl.mul(x, 3.0)
    return Prog
```

当多个定义确实无法区分时，装饰器会抛出 `ParserSyntaxError` 并列出全部候选行号，
而不是任选其一——选错就会编译出你从未写过的类体。此时请给各个类取不同的名字，
或者只定义一次、用闭包变量参数化:

```python
def make(scale):
    @pl.program
    class Prog:                                    # 单一定义，参数化
        @pl.function
        def main(self, x: pl.Tensor[[8], pl.FP32]) -> pl.Tensor[[8], pl.FP32]:
            return pl.mul(x, scale)
    return Prog
```

## 跨模块函数复用

在 `@pl.program` 类之外定义的函数可通过两种机制复用。

### 外部 `@pl.function` 调用

在 `@pl.program` 内部可按名称调用外部定义的 `@pl.function`。该函数会自动加入 Program，
并生成 `ir.Call(GlobalVar, args)`。

```python
@pl.function
def softmax(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    ...

@pl.program
class MyModel:
    @pl.function
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        y: pl.Tensor[[64], pl.FP32] = softmax(x)   # ir.Call(GlobalVar("softmax"), [x])
        return y
```

**规则：**

- 使用函数的 `.name` 作为 GlobalVar（别名透明）
- 外部与内部函数名不得冲突
- 两个不同的外部函数具有相同 `.name` 是错误
- 同一外部函数从多个 method 调用时只加入一次

### `@pl.inline` 装饰器

`@pl.inline` 捕获函数以便在语句级内联。不会向 Program 添加函数——每次调用点展开函数体。

```python
@pl.inline
def normalize(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    result: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
    return result

@pl.program
class MyModel:
    @pl.function
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        y: pl.Tensor[[64], pl.FP32] = normalize(x)  # statements inlined in-place
        return y
```

**规则：**

- 实参个数必须与形参列表完全一致
- 内联定义处的闭包变量可用
- 内联函数可多次调用（每次展开相互独立）
- 支持嵌套内联调用

## 完整示例

### 张量操作 (带 iter_args 的循环)

```python
# pypto.program: my_program
import pypto.language as pl

def loop_sum(n: pl.INT64) -> pl.INT64:
    sum_init: pl.INT64 = 0
    for i, (sum,) in pl.range(n, init_values=(sum_init,)):
        sum = pl.yield_(sum + i)
    return sum
```

### Tile 操作 (基于 Tile 的计算)

```python
import pypto.language as pl

@pl.program
class BlockExample:
    @pl.function
    def tile_add(
        self,
        input_a: pl.Tensor[[64, 64], pl.FP32],
        input_b: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Tensor[[64, 64], pl.FP32],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
        tile_b: pl.Tile[[64, 64], pl.FP32] = pl.load(input_b, [0, 0], [64, 64])
        tile_c: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_a, tile_b)
        result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_c, [0, 0], output)
        return result
```

## 打印 IR 节点

对任意 IR 节点调用 `as_python()` 获取其 Python 表示：

```python
print(stmt.as_python())          # "x: pl.Scalar[pl.INT64] = a + b"（默认 "pl" 前缀）
print(stmt.as_python("ir"))      # "x: ir.Scalar[ir.INT64] = a + b"（自定义前缀）
```

### 简洁模式 (Concise Mode)

传入 `concise=True` 可省略中间变量的类型标注。函数签名类型（参数和返回值）始终保留：

```python
print(func.as_python())                  # 详细模式（默认）：每个赋值都包含类型
print(func.as_python(concise=True))      # 简洁模式：省略中间类型标注
```

详细输出：

```python
def main(self, x: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP16]:
    y: pl.Tensor[[64, 128], pl.FP32] = pl.some_op(x)
    result: pl.Tensor[[64, 128], pl.FP16] = pl.cast(y, pl.FP16)
    return result
```

简洁输出：

```python
def main(self, x: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP16]:
    y = pl.some_op(x)
    result = pl.cast(y, pl.FP16)
    return result
```

自由函数 `ir.python_print(node)` 同样可用，支持相同的参数。

## 参考资料

- [Python IR 语法规范](00-python_syntax.md) —— 类型与表达式
- [语句与控制流](01-statements.md) —— 函数体内的语句形式
- [集成手写 C++ Kernel](04-external-kernels.md) —— 调用外部 kernel
