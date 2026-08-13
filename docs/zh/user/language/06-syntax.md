# 语言语法

普通 Python 语法写在 kernel 里是什么意思：下标、运算符，以及来自外层作用域的名字。

> **前置**：[类型](00-types.md)。

## Concept

装饰器解析你的源码，从不把它当 Python 执行。所以 kernel 体里那些眼熟的语法是**被改写成 IR 操作**，而不是被执行：

- 下标不是一次索引操作 —— 它会变成 slice、read 或 assemble。
- 运算符不是 Python 的 `+` —— 它会根据操作数类型变成某个 IR 算子。
- 来自外层作用域的名字由解析器在解析期解析，而不是在调用时由闭包捕获。

实际后果是：这些构造要么在解析期带着行号报错，要么根本编不过，而不会在运行期给你惊喜。

## Quickstart：同一个切片的两种写法

```python
import pypto.language as pl

@pl.jit.incore
def head(x: pl.Tensor[[128, 64], pl.FP32],
         out: pl.Out[pl.Tensor[[16, 64], pl.FP32]]):
    top = x[0:16, :]              # sugar for pl.slice(x, [16, 64], [0, 0])
    out[:] = pl.mul(top, 2.0)        # sugar-free: an explicit operator call
    return out
```

两行都会被解析成 IR 调用。第一行展示了改写本身，第二行就是改写在别处所产生的东西。

## Mechanics

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

## 边界情况

| 症状 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| **`dst[...] = src` 被拒绝** | `strict_ssa=True` 禁止该重绑 | 改用 `pl.assemble(dst, src, [...])` |
| **本想切片，下标却给了一次读** | `A[i, j]` 读单个元素，`A[i:i+1, :]` 才是切片 | 想要一块区域时用切片语法 |
| **printed IR 无法被重新解析** | 设备规模查询在使用前被绑定到了名字上 | 在使用处内联书写该调用 |
| **某个捕获值在解析期被拒绝** | 它无法折叠进 IR | 把它作为参数传入，而不是捕获 |

## See Also

- [类型](00-types.md) —— 这些语法的操作数是什么，包括 `pl.Array`。
- [编译期指令](05-directives.md) —— 另一批解析期构造。
- [算子](../ops/00-dispatch.md) —— 语法糖背后的算子在哪里。
- [Python IR 语法规范](../../dev/language/00-python_syntax.md) —— 解析器的完整表面。
