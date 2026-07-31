# 编译期指令

这些构造塑造的是编译或观测，而不是计算：编译期语句、dump 标记、数组，以及解析器识别的语法糖。

> **前置**：[函数与程序](01-functions.md)。

## Concept

本页的一切都有一个共同性质：**它们并不在你以为的那个时刻运行。**

`pl.static_print` 与 `pl.static_assert` 在**解析期**运行，之后从 IR 中消失。`pl.dump_tag` 既不在解析期也不在 kernel 执行期运行 —— 它记录一个意图，由运行时在开启 dump 时兑现。`pl.const` 是带类型的字面量，不是一次调用。下标语法糖是改写，不是索引。

搞清楚一个构造属于哪个阶段，决定了它是调试利器还是谜题。一条什么都没打印的 `static_print` 并没有坏 —— 是那个函数根本没被解析过。

## Quickstart：看看解析器看到了什么

```python
import pypto.language as pl

@pl.jit.incore
def probe(x: pl.Tensor[[64, 128], pl.FP32],
          out: pl.Out[pl.Tensor[[64, 128], pl.FP32]]):
    pl.static_print("x =", x)                      # prints at parse time
    pl.static_assert(x.shape[1] == 128, "expected 128 columns")
    out = pl.mul(x, 2.0)
    return out
```

两条语句都会从 IR 中消失。`static_print` 的输出出现在装饰器解析源码之时 —— 对 `@pl.function` 是模块被 import 时，对 `@pl.jit` 是首次触发特化的调用时。

## Mechanics

### 编译期语句

| 构造 | 何时运行 | 失败方式 |
| ---- | -------- | -------- |
| `pl.static_print(*args)` | 解析期 | 无 —— 纯输出 |
| `pl.static_assert(cond, msg)` | 解析期 | 为假时抛 `ParserError` |

`static_assert` 是**仅语句**构造 —— 不能出现在表达式里 —— 且它的 `msg` 在调用点必须是**字符串字面量**。传变量会抛 `ParserSyntaxError`。条件必须是编译期可求值的；它在执行期不做任何检查。

### 带类型的常量

`pl.const(value, dtype)` 构造一个显式指定 dtype 的常量，而不是由字面量推断出的默认类型。它的存在是为了让打印器能往返非默认类型的常量，当字面量的位宽有意义时就该用它：

```python
step = pl.const(1, pl.INT32)
```

### 选择性张量 dump

dump 每一个绑定会在大负载下压垮 host 侧收集器，因此运行时支持带逐张量标记的部分 dump。两种写法喂给同一个 `dump_vars` 集合：

| 形式 | 标记范围 |
| ---- | -------- |
| `pl.dump_tag(t)` 作为独立语句 | 此后**每一个**消费该值的派发 |
| `pl.submit(...)` / `pl.at(...)` 上的 `dumps=[t, ...]` | 仅该次派发 |

当一次声明要对所有后续消费者生效时用 `dump_tag`；当你想在单次启动处显式列出目标时用 `dumps=`。

```python
pl.dump_tag(q)                                    # sticks to later consumers
out, tid = pl.submit(self.attn, q, k, out, dumps=[k])   # this launch only
```

这些标记只在**部分** dump（`RunConfig.enable_dump_args == 1`）下生效。dump 关闭（`0`）时它们是空操作；全量 dump（`2`）下则无关紧要，因为那会不加区分地捕获一切。在大负载上开全量 dump 时，收集器（约 42 MB/s 的排空速率）会跟不上，AICPU 侧最终会触发 STARS 算子超时被杀。部分 dump 加标记才是让 dump 在那里可用的办法。

### 数组

`pl.array` 是一个小的核内数组，用于带索引的标量状态 —— 一张 TaskId 表、一组按 block 的偏移。数组不跨函数边界，所以是创建出来而非注解出来的。

| 调用 | 含义 |
| ---- | ---- |
| `pl.array.create(extent, dtype)` | 分配 |
| `arr[i]` | `array.get_element` |
| `arr[i] = v` | `array.update_element` —— 函数式，会重绑 `arr` |

更新是函数式的：它产生一个新的数组值并重绑该名字。这与 SSA 一致，也意味着循环内的数组赋值和其他携带值一样是一个携带值。在 `pl.parallel` 下，数组携带充当栅栏 —— 见 [作用域与任务](04-scopes-and-tasks.md)。

一个常见用法是收集 TaskId 做扇入：

```python
tids = pl.array.create(4, pl.TASK_ID)
for i in pl.range(4):
    _, tid = pl.submit(self.stage, x, out)
    tids[i] = tid
```

### 下标语法糖

解析器会改写 `Tensor` 与 `Tile` 值上的下标：

| 写法 | 变成 |
| ---- | ---- |
| `A[0:16, :]` | `pl.slice(A, [16, N], [0, 0])` |
| `A[i, j]` | `pl.tensor.read(A, [i, j])` / `pl.tile.read(A, [i, j])` |
| `A[0:16, 0:32]` | `pl.slice(A, [16, 32], [0, 0])` |
| `dst[i:i+16, j:j+32] = src` | `dst = pl.assemble(dst, src, [i, j])` |

写形式会重绑 `dst`，这与严格 SSA 不兼容。在 `@pl.function(strict_ssa=True)` 下 —— 或任何 SSA 之后的上下文 —— 请显式调用 `pl.assemble(...)`。

### Python 运算符

标准运算符在 `Tensor`、`Tile`、`Scalar` 值上映射到 IR 操作：

| Python | 操作 |
| ------ | ---- |
| `a + b` / `a - b` / `a * b` / `a / b` | `add` / `sub` / `mul` / `div` |
| `a == b` / `a != b` | `eq` / `ne` |
| `a < b` / `a > b` | `lt` / `gt` |

任一侧是标量都会被识别并派发到标量操作数形式（`pl.add(a, 1.0)` → `adds`）。

### 闭包捕获

装饰器解析的是源码，所以来自外层 Python 作用域的名字是被解析器解析的，而不是在调用时由闭包捕获的。整数常量和常量算术会折叠进 IR。无法折叠的捕获值在解析期就是错误，而不是运行期的意外。

有一个后果值得知道：像 `pl.system.available_cluster_count()` 这样的表达式应当在调用点**内联**书写，而不要先绑定到一个名字上。绑定之后编译与降级都正确，但被 outline 的包装函数的 printed IR 会引用一个定义在调用方的变量，因而无法被重新解析。

## Edge Cases

> **致命陷阱：** 除非 `RunConfig.enable_dump_args == 1`，否则 `dumps=` 与 `pl.dump_tag` 是静默失效的。dump 关闭时你既拿不到文件也收不到告警；全量 dump 时你会拿到所有绑定并把收集器堵住。在断定标记没生效之前，先确认这个设置。

| 症状 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| **`static_print` 什么都没打印** | 该函数从未被解析 | 对 `@pl.jit`，解析发生在首次触发特化的调用时 |
| **`static_assert` 报 `ParserSyntaxError`** | `msg` 不是字符串字面量，或它被用在表达式里 | 传字面量；作为独立语句使用 |
| **`static_assert` 没抓住某个运行期值** | 它只在解析期 | 运行期值请在 host 代码里校验 |
| **dump 没有产生任何东西** | `enable_dump_args` 是 `0` | 设为 `1` 走部分 dump |
| **dump 把运行搞挂了（STARS 算子超时）** | 全量 dump（`2`）压垮了收集器 | 改用部分 dump 加 `dump_tag` / `dumps=` |
| **数组更新没生效** | `arr[i] = v` 会重绑；旧名字仍持有旧值 | 使用重绑后的名字，或让它穿过循环 |
| **`dst[...] = src` 被拒绝** | `strict_ssa=True` 禁止该重绑 | 改用 `pl.assemble(dst, src, [...])` |
| **printed IR 无法被重新解析** | 设备规模查询在使用前被绑定到了名字上 | 在使用处内联书写该调用 |

## See Also

- [控制流](02-control-flow.md) —— 携带值，数组更新也是其中之一。
- [作用域与任务](04-scopes-and-tasks.md) —— submit 上的 `dumps=`，以及 TaskId 数组。
- [算子](../ops/00-dispatch.md) —— 语法糖背后的算子在哪里。
- [Python IR 语法规范](../../dev/language/00-python_syntax.md) —— 解析器的完整表面。
- [Runtime DFX](../../dev/03-runtime-dfx.md) —— 这些标记所馈入的 dump 管线。
