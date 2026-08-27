# 控制流

循环与条件、它们跨迭代携带的值，以及塑造二者的那条 SSA 规则。

> **前置**：[函数与程序](01-functions.md)。

## Concept

PyPTO 的 IR 是 **SSA**：每个绑定只被写一次。在直线代码里这是不可见的 —— 你重新绑定多少次 `x`，解析器就替你重命名多少次 —— 但在循环边界或分支汇合处它就显形了，因为编译器没有地方安放第二次写入。

`pl.yield_` 就是答案。它命名离开一个作用域的那个值：

- 在循环里，被 yield 的值成为下一次迭代的输入，并在最后一次迭代后成为循环的结果。
- 在 `if` / `else` 里，两个分支都 yield，汇合处产生一个结果变量（phi 节点）。

你并不总需要自己写 `yield_`。在循环里重新绑定一个名字就是常规的累加写法，解析器会替你把它变成携带值 —— `init_values=` 加 `yield_` 是同一件事的显式拼写，且只有 `pl.while_` 强制要求。而产生值的分支则**必须**在两个分支都 yield。本页其余内容都是这些规则的变体。

四种循环构造共用一套语法，区别只在于告诉了编译器什么：

```text
pl.range     sequential — the default
pl.parallel  iterations are independent and may overlap
pl.unroll    fully unrolled at compile time; bounds must be literals
pl.pipeline  body replicated `stage` times for ping-pong buffering
```

## Quickstart：一个累加的循环

```python
import pypto.language as pl

@pl.jit
def accumulate(
    a: pl.Tensor[[128, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        t = pl.add(a, a)
        for i in pl.range(3):
            t = pl.add(t, a)      # carried across iterations
        out[:] = pl.mul(t, t)
    return out
```

| 行 | 作用 |
| -- | ---- |
| `t = pl.add(a, a)` | 在循环之前建立该值 |
| `for i in pl.range(3)` | 三次迭代；`i` 是循环变量 |
| `t = pl.add(t, a)` | 看起来像修改，实为携带值 |
| `out = pl.mul(t, t)` | 循环之后读 `t` 读到的是最后一次迭代的结果 |

在循环里重新绑定 `t` 是携带值的惯用写法。IR 是 SSA 的，解析器给每次迭代的值各自的名字，并把它作为携带值穿过循环 —— 这正是下面的显式形式手工写出来的东西。

## Mechanics

### 四种循环形式

四者都接受 `(stop)`、`(start, stop)` 或 `(start, stop, step)`。除注明外，参数可以是 `int` 字面量或 `pl.Scalar`。

| 形式 | 产生 | 携带值 | 边界 |
| ---- | ---- | ------ | ---- |
| `pl.range(...)` | `ForKind.Sequential` | 支持 | int 或 `Scalar` |
| `pl.parallel(...)` | `ForKind.Parallel` | 支持 | int 或 `Scalar` |
| `pl.unroll(...)` | `ForKind.Unroll` | **不支持** | 仅字面量 |
| `pl.pipeline(..., stage=N)` | Sequential + 软流水 | 支持 | int 或 `Scalar` |

```python
for i in pl.range(10): ...            # 0..9
for i in pl.range(2, 10): ...         # 2..9
for i in pl.range(0, 100, 4): ...     # 0, 4, ..., 96
for i in pl.parallel(0, nblocks): ...
for i in pl.unroll(4): ...            # no init_values here
```

`pl.parallel` 是一个断言而非请求：你在告诉编译器这些迭代彼此独立。如果它们其实不独立，结果就是竞态。

`pl.pipeline(N, stage=F)` 把循环体在每个外层迭代里复制 `F` 份，使缓冲区可以乒乓。外层循环以 `stage * step` 为步长推进，当行程数不能整除时由一个尾部派发覆盖余数。`stage` 是必填的正整数（通常 2–4）。它在 tile 层由 [LowerPipelineLoops](../../dev/passes/29-lower_pipeline_loops.md) 降级；在 `memory_planner=PTOAS` 下则由 [LowerPipelineToSlots](../../dev/passes/28-lower_pipeline_to_slots.md) 接手，改为让单份循环体轮转同一块分配的多个 slot 而不复制。

```python
for i in pl.pipeline(64, stage=4):
    t = pl.load(a, [i * 64, 0], [64, 64])
    pl.store(t, [i * 64, 0], out)
```

### 显式命名携带值

`init_values=` 加 `pl.yield_` 是重新绑定所隐含之事的显式拼写。它对 `pl.while_`（见下）是**必需**的，也是 printed IR 采用的形式 —— 即便你不写，也会读到它。带类型的重载最多支持五个携带值。

下面这段片段假设 `init_max` / `init_sum` 与循环体产出的类型相同：

```python
for i, (acc_max, acc_sum) in pl.range(4, init_values=(init_max, init_sum)):
    out_max, out_sum = pl.yield_(pl.maximum(acc_max, row_i), pl.add(acc_sum, row_i))
```

两条规则约束着它：循环之后要用 `yield_` 绑定的名字去读（不是 `init_values` 里的名字）；并且每个携带值在整个循环中必须保持同一种类型。**混级**正是它咬人的地方 —— 一个由张量级表达式播种、随后在 InCore 作用域里与 tile 级值结合的携带值不会通过类型检查，而报错指向的是算子而不是携带值。

### while 循环

`pl.while_` **总是**要求 `init_values`，条件由 `pl.cond()` 作为函数体的**第一条语句**给出：

```python
for (x,) in pl.while_(init_values=(0,)):
    pl.cond(x < 10)            # continue while true
    x_out = pl.yield_(x + 1)
```

`pl.cond` 纯属语法 —— 解析器把它提升到 `WhileStmt` 上。它不是运行时调用，也不能出现在别处。

### 条件分支

产生值的分支必须在**两个**分支都 yield，且值的数量与类型一致：

```python
for i, (prev,) in pl.range(4, init_values=(init,)):
    if i == 0:
        result = pl.yield_(a)
    else:
        result = pl.yield_(pl.add(prev, delta))
    out = pl.yield_(result)
```

一个分支 yield 了，另一个也必须 yield。没有值要产生的分支则两边都不 yield。

### SSA，以及你何时需要在意它

照常写 Python —— 名字随便重绑。解析器会重命名：

```python
result = pl.mul(x, 2.0)
result = pl.add(result, 1.0)      # fine; the parser produces two bindings
```

`@pl.function(strict_ssa=True)` 会让解析器改为拒绝重绑，偶尔可用于捕捉无意的变量遮蔽。它同时会禁用 `dst[...] = src` 下标写语法糖 —— 那个糖正是靠重绑实现的，见 [编译期指令 § 下标语法糖](06-syntax.md#下标语法糖)。

流水线很早就会跑 [ConvertToSSA](../../dev/passes/04-convert_to_ssa.md)，所以非 SSA 源码是正常输入，不是兼容模式。

## 边界情况

> **致命陷阱：** 循环之后去读 `init_values` 里的那个名字，读到的是**初值**而不是累加结果。得到的往往是一整块初始化时的内容（常常是零），看起来像计算 bug 而不是命名问题。请读 `pl.yield_` 绑定的名字。

| 症状 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| **循环里写过的值出来是空的** | 读了 `init_values` 的名字而不是 `yield_` 的名字 | 改读 `yield_` 的绑定 |
| **`tensor.add requires ... but got TileType`** | 携带值中途换了级别 —— 在 InCore 作用域里 `pl.slice` 产出的是 Tile | 让携带值全程保持同一级别；在作用域内显式使用 `pl.load` / `pl.store` |
| **`while_() requires init_values to be specified`** | `pl.while_()` 没有携带状态 | 补上 `init_values=(...)`，哪怕只是一个计数器 |
| **`pl.cond` 报解析错误** | 它不是 `while_` 体的第一条语句 | 移到第一行 |
| **分支汇合被拒绝** | 只有一个分支 yield，或数量不一致 | 两个分支 yield 相同数量与类型 |
| **`unroll()` 拒绝 `init_values`** | 展开循环不携带状态 | 改用 `pl.range`，或重构成不携带值 |
| **`unroll()` 拒绝某个边界** | 边界必须是编译期字面量 | `Scalar` 边界请用 `pl.range` |
| **`pipeline()` 报 stage 错误** | `stage` 缺失或不是正整数 | 传 `stage=N`，N ≥ 1 |
| **`pl.parallel` 下出现竞态** | 迭代其实并不独立 | 改用 `pl.range`，或消除跨迭代依赖 |
| **`SSAViolationError`** | 在 `strict_ssa=True` 下重绑 | 换用不同名字，或去掉 `strict_ssa` |

## 配套示例

| 示例 | 展示 |
| ---- | ---- |
| `examples/beginner/02_elementwise.py`（`chunked_add`） | 对 tile 分块的朴素 `pl.range` |
| `examples/intermediate/04_matmul_acc.py` | 用 `init_values` 跨迭代携带累加器 |
| `examples/models/03_flash_attention.py` | 循环携带状态 + 嵌套 `if` / `pl.yield_` —— 完整形态 |

## See Also

- [类型](00-types.md) —— 携带值到底是什么。
- [作用域与放置](04-scopes.md) —— 包含这些循环的放置作用域。
- [ConvertToSSA](../../dev/passes/04-convert_to_ssa.md) —— 本页规则的来源。
- [UnrollLoops](../../dev/passes/02-unroll_loops.md) —— `pl.unroll` 变成什么。
- [LowerPipelineToSlots](../../dev/passes/28-lower_pipeline_to_slots.md) —— `memory_planner=PTOAS` 下 `pl.pipeline` 变成什么。
- [LowerPipelineLoops](../../dev/passes/29-lower_pipeline_loops.md) —— 其余情况下 `pl.pipeline` 变成什么。
