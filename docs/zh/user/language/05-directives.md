# 编译期指令

在装饰器解析源码时执行、随后从 IR 中消失的语句。

> **前置**：[函数与程序](01-functions.md)。

## Concept

装饰器解析你的函数源码，而不是执行它。有三个构造只活在这一步：

`pl.static_print` 与 `pl.static_assert` 在**解析期**运行，在 IR 里不留任何痕迹。`pl.const` 是带类型的字面量 —— 用来钉住一个常量的 dtype，而不是接受由 Python 字面量推断出的那个。

知道它们属于解析期，决定了它是调试利器还是谜题。一条什么都没打印的 `static_print` 并没有坏 —— 是那个函数根本没被解析过。

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

用 `static_print` 查看解析器推断出的东西 —— 它给某个值定的类型和 shape —— 当你只想确认一个事实时，这比读 printed IR 快得多。

### 带类型的常量

`pl.const(value, dtype)` 构造一个显式指定 dtype 的常量，而不是由字面量推断出的默认类型。它的存在是为了让打印器能往返非默认类型的常量，当字面量的位宽有意义时就该用它：

```python
step = pl.const(1, pl.INT32)
```

## Edge Cases

| 症状 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| **`static_print` 什么都没打印** | 该函数从未被解析 | 对 `@pl.jit`，解析发生在首次触发特化的调用时 |
| **`static_assert` 报 `ParserSyntaxError`** | `msg` 不是字符串字面量，或它被用在表达式里 | 传字面量；作为独立语句使用 |
| **`static_assert` 没抓住某个运行期值** | 它只在解析期 | 运行期值请在 host 代码里校验 |
| **常量的位宽不对** | dtype 是从 Python 字面量推断的 | 用 `pl.const(value, dtype)` 钉住 |

## See Also

- [语言语法](06-syntax.md) —— 另一批解析期行为：下标语法糖、运算符、闭包捕获。
- [函数与程序](01-functions.md) —— 各个装饰器分别在什么时候解析。
- [Python IR 语法规范](../../dev/language/00-python_syntax.md) —— 解析器的完整表面。
