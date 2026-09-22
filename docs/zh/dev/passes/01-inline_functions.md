# InlineFunctions Pass

通过将函数体在每个调用点展开来消除 `FunctionType.Inline` 函数。

## 概述

被装饰为 `@pl.function(type=pl.FunctionType.Inline)`(或通过 JIT 端的 `@pl.jit.inline`)的函数是*源级实用工具*:每个调用点展开为一份新的、经过 alpha 重命名的函数体副本,其中形参由实际参数表达式替换。该 pass 运行后,程序中不会再有 `FunctionType.Inline` 函数,也不会有指向它的 `Call` — 后续 pass 把已展开的代码视为如同直接写在调用点处。

作为 `OptimizationStrategy.Default` 中的**第一个** pass 运行,以确保下游 pass(`UnrollLoops`、`OutlineIncoreScopes` 等)永远不会观察到 Inline 函数。

**产生**: `IRProperty.InlineFunctionsEliminated`。

**要求**: 无 — 在新解析的程序上运行。

**何时使用**: 始终作为默认流水线的一部分。当程序中没有 Inline 函数时,该 pass 是空操作。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::InlineFunctions()` | `passes.inline_functions()` | Program 级 |

**Python 用法**:

```python
from pypto.pypto_core import passes

inline_pass = passes.inline_functions()
program_inlined = inline_pass(program)
```

## 算法

1. **收集**所有 `func_type == FunctionType::Inline` 的函数。
2. **环检测** Inline → Inline 调用图;若发现环,抛出 `pypto::ValueError` 并在消息中标明环路径。
3. **迭代到不动点** — 每次迭代遍历所有函数(包括 Inline 函数本身,以便嵌套的 Inline-calls-Inline 也能传递展开):
   - 对函数体中每个顶层 `LHS = inline_call(args)` 或 `EvalStmt(inline_call(args))`:
     - 构建参数替换映射(形参 `Var` → 实参 `Expr`)。该映射同时作用于使用点**和**定义点,因此被调函数中的重绑定 `out = pl.tensor.assemble(out, ...)` 会重绑定调用方的实参 `Var` —— 但仅限于带有原地(in-place)契约的形参(见[重绑定形参](#重绑定形参))。有三类实参会先在展开体之前绑定到新的 `<param>_inline<counter>` `Var`,再用该 `Var` 替换:被重绑定形参的非可赋值 `Var` 实参(切片 `c[r]`、`IterArg`、计算得到的标量);被重绑定的按值传递形参的实参(即既非 `pl.Out` 也非 `pl.InOut` 的普通标量);以及任何计算得到的 tensor / tile 实参(如 `a[r]` 这样的 `Call`),Python 在调用点对其只求值一次。其余实参 —— 绑定到只读形参的 `Var`、标量表达式、常量 —— 仍直接替换,因此读取形参的形状表达式仍可折叠。
     - 对内联体中每个本地绑定的 `Var` 做 alpha 重命名(`<orig>_inline<counter>`,并去掉 `<orig>` 末尾的 `_`),避免多个调用点之间冲突。
     - 在调用点之前插入重命名+替换后的函数体语句。
     - 按调用点形态接线被内联函数的尾部返回值:`LHS = renamed_return`(单返回值赋值;当 `LHS` 与替换后的返回 `Var` 是同一个 `Var` 时省略该赋值,以避免冗余 SSA 拷贝)、逐元素替换 `TupleGetItemExpr` 而不发出 `MakeTuple` 绑定(多返回值赋值)、新的 `ReturnStmt`(`return inline_call(...)`),或者当返回值被丢弃但其求值可观测时发出新的 `EvalStmt`(`EvalStmt` 调用点 — 参见[边界情况](#边界情况))。
4. **删除**所有 Inline 函数。

重命名后缀使用单下划线(`_inline`),因为 `__` 被 IR 自动命名约定保留(参见 `auto_name_utils.h`)。

仅靠单下划线后缀并不足够:`<orig>` 本身可能以 `_` 结尾——`_` 是 Python 的弃值名,也是 `for _ in pl.split_aiv(...)` 文档所载的循环变量——此时直接拼接会把两个各自合法的下划线融合成保留分隔符。因此 `FreshName` 通过 `auto_name::JoinNameSuffix` 拼接,该函数会去掉这段尾巴:`_` 重命名为 `_inline7`,而非 `__inline7`。若 `<orig>` 本身**已经**含有 `__`,则原样透传,使作者手写的 `a__b` 仍然是 `ValidateBaseName` 报告的用户错误,而不会被悄悄规范化。

## 示例

### 单一调用点

**展开前**:

```python
@pl.program
class P:
    @pl.function(type=pl.FunctionType.Inline)
    def helper(self, x):
        y = pl.mul(x, x)
        return y

    @pl.function
    def main(self, a):
        z = self.helper(a)
        return z
```

**展开后**:

```python
@pl.program
class P:
    @pl.function
    def main(self, a):
        y_inline0 = pl.mul(a, a)
        z = y_inline0
        return z
```

### 多个调用点

每个调用点独立 alpha 重命名,本地变量不会冲突:

**展开前**:

```python
@pl.function(type=pl.FunctionType.Inline)
def square(self, x):
    y = pl.mul(x, x)
    return y

@pl.function
def main(self, a, b):
    a2 = self.square(a)
    b2 = self.square(b)
    return pl.add(a2, b2)
```

**展开后**:

```python
@pl.function
def main(self, a, b):
    y_inline0 = pl.mul(a, a)
    a2 = y_inline0
    y_inline1 = pl.mul(b, b)
    b2 = y_inline1
    return pl.add(a2, b2)
```

### 内联体含 `pl.at`

scope 被原样保留,稍后由 `OutlineIncoreScopes` 提取为独立的 InCore 函数,与直接写在调用点处效果一致。

## 重绑定形参

由于替换会作用到定义点,被调函数对自身形参的重绑定有可能落到调用方的 `Var` 上。是否*应该*落上去,取决于该形参的调用约定:

| 形参 | 重绑定落在 | 原因 |
| ---- | ---------- | ---- |
| `pl.Out[...]` / `pl.InOut[...]` | 调用方的 `Var` | 作者显式选择了原地契约。 |
| 任意 tensor / tile / `Array` 形参 | 调用方的 `Var` | 这些形参是*句柄*,其重绑定**就是**那次原地更新:`c[...] = v` 解析为 `c = pl.tensor.assemble(c, ...)`,`a[i] = v` 解析为 `a = pl.array.update_element(a, i, v)`。`@pl.jit.inline` 正因如此会剥掉带形状形参上的 `pl.Out` / `pl.InOut`,所以不能仅凭方向(direction)判断。此处若绑定临时变量,还会产生裸的 `arr_inline0 = arr` 别名,而 orchestration codegen 无法声明它 —— array `Var` 只能来自 `array.create` 或对已有后备数组的别名,绝不能由其类型声明。 |
| 普通标量 | 新的 `<param>_inline<counter>` `Var` | 按值传递,与 Python 一致。 |

最后一行最容易出错。给定:

```python
@pl.function(type=pl.FunctionType.Inline)
def bump(self, n: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
    n = n + 1
    return n

@pl.function
def main(self, k: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
    m: pl.Scalar[pl.INDEX] = self.bump(k)
    return k + m                              # k + (k + 1)
```

调用点会先绑定一个临时变量,因此其后调用方的 `k` 仍可读到原值:

```python
n_inline0: pl.Scalar[pl.INDEX] = k
n_inline0 = n_inline0 + 1
m: pl.Scalar[pl.INDEX] = n_inline0
s: pl.Scalar[pl.INDEX] = k + m                # 原始的 k
```

若改为在定义点替换 `k`,则会把 `k = k + 1` 插入调用方,静默算出 `2k + 2`。注意 JIT specializer 的 alpha 重命名**覆盖不到**这种情况:它只在名字首次绑定的作用域深度上重命名重绑定,因此写在 `pl.range` / `pl.spmd` 内的重绑定会以形参真实定义点的形式到达本 pass。

## 边界情况

| 情况 | 行为 |
| ---- | ---- |
| 无调用点的 Inline 函数 | 静默从程序中移除。 |
| 作为程序入口的 Inline 函数 | 此处不视为错误 — 但因为没有任何 Call 指向它,清理阶段会像任何无调用者函数那样移除。 |
| Inline 调用 Inline(传递) | 迭代到不动点。 |
| 计算得到的 tensor / tile 实参,如 `f(a[r], c[r])`,被调函数读取 `x` 并写入 `c[0:4, j] = v` | 每个实参在调用点只绑定一次 —— `x_inline0 = a[r]`、`c_inline1 = c[r]`,再 `c_inline1 = pl.tensor.assemble(c_inline1, v, ...)` —— 与调用方自行为切片命名时解析器生成的 IR 相同。直接替换 `c[r]` 会把 `Call` 放到重绑定的左值上;直接替换 `a[r]` 会在被调函数的 `pl.spmd` / `pl.pipeline` 体内重复求值,并把它移入外提的 kernel。切片是其源张量的视图,因此写入会到达调用方的 `c`。 |
| 被重绑定的按值传递形参,例如内联被调函数对标量形参执行 `n = n + 1` | 实参在调用点绑定一次(`n_inline0 = k`),重绑定只作用在该临时变量上,因此其后调用方的 `k` 保持不变。见[重绑定形参](#重绑定形参)。 |
| 递归 Inline(自递归或互相调用) | 在任何展开发生之前抛出 `pypto::ValueError`,消息中标明环路径(`a -> b -> a`)。 |
| 多返回值 Inline | **不**发出 `LHS = MakeTuple([rets...])` — 编排层 codegen 无法 lower `MakeTuple`。改为把克隆后的返回值记录在 LHS `Var` 上,并把下游 `TupleGetItemExpr(LHS, i)` 的使用改写为第 `i` 个值,使该 LHS 绑定最终无人引用(参见 `SpliceInlineCallAsTupleSub`)。 |
| 嵌套 Call 到 Inline(如 `pl.add(inline_fn(x), y)`,以及解析器把 `arr[i] = inline_fn(x)` 脱糖成的 `array.update_element(arr, i, inline_fn(x))`) | 先提升为独立的 `AssignStmt`,并在同一轮迭代中展开 — 参见[嵌套调用点](#嵌套调用点)。 |
| 指向**返回元组**的 Inline,或位于 `WhileStmt` 条件、`IterArg` 初值、裸(非 `SeqStmts`)`ForStmt` / `IfStmt` body 中的嵌套 Call | 不提升。`InlineFunctionsEliminated` verifier 会在本 pass 之后立即在其源码行上报告残留的 Call。 |
| `EvalStmt(inline_call(...))` — 返回值被忽略 | 被丢弃的是返回**值**,不是它的**求值**。参见下方[丢弃返回值](#丢弃返回值)。 |

## 嵌套调用点

`HandleTopLevelInlineCall` 只在 `Call` **就是**整个语句值时才识别调用点 —— `LHS = f(...)`、`EvalStmt(f(...))`、`return f(...)`。位于其他位置的 `Call` 会被跳过,而第 5 步仍会删除被调用函数,于是留下一个指向已删除函数的引用,直到 `GenerateOrchestration preconditions` 才以 `references undefined function` 失败。

这类调用来自普通 DSL,有时用户根本没有写嵌套调用。`arr[i] = f(x)` 没有对应的 IR 语句,解析器将其脱糖为函数式更新:

```python
arr[i] = f(x)                              # 用户写的
arr = pl.array.update_element(arr, i, f(x))  # 解析器存下的 —— f 变成了实参
```

因此 `NestedInlineCallHoister` 会在调用点匹配之前遍历每条语句自身的表达式,把每个嵌套的 inline `Call` 提升到该语句之前的新绑定 `t__inline_arg_vN` 上:

```python
# 之前                                      # 提升之后、展开之前
k = self.half(n) + 1                       t__inline_arg_v0 = self.half(n)
                                           k = t__inline_arg_v0 + 1
```

随后 `SpliceHoisted` 立即展开每个提升出来的绑定,使提升与它所触发的展开落在同一轮不动点迭代中,`inline_fns.size() + 1` 的迭代上界仍然成立。

修改这部分时需要记住三点:

- **已处于顶层位置的 Call 保持原样**(`HoistInArgs` 只改写它的实参)。否则每个已有调用点都会多出一次冗余拷贝,并打乱所有 before/after 测试。
- **不触碰 body。** 提升器只改写语句自身的表达式;`InlineCallsMutator` 仍会递归进入循环与分支 body,因此 body 内的提升落在该 body 内。
- **每个被提升的位置都只求值一次**,即提升后语句所在的位置 —— 调用实参、二元操作数、循环边界、`if` 条件、`yield` 值。`WhileStmt` 条件不是如此,因此被排除:提升它会把展开后的 body 变成只求值一次,而非每轮迭代求值。
- **返回元组的被调函数不会被提升。** `SpliceInlineCallAsTupleSub` 刻意不发出 `tmp = ...` 绑定 —— 它把克隆后的返回值记录在 LHS `Var` 上,并改写下游的 `TupleGetItemExpr(tmp, i)`。嵌套消费者持有的是 `tmp` 本身而非 `TupleGetItemExpr`,因此提升会留下未定义的临时变量:`return self.pair(x), y` 会打印出 `t__inline_arg_v0__FREE_VAR`。保持该 `Call` 原样即维持提升前的行为,并由 verifier 指名报告。

这比 `FlattenCallExpr`(pass 06)有意更窄,后者对**所有**调用执行同样的提升。该 pass 声明 `.required = {SSAForm, NormalizedStmtStructure}`,两者都在本 pass 之后才建立,因此它无法直接提前运行。

## 丢弃返回值

`EvalStmt` 调用点(`self.wrapper(x, out)`,没有 LHS)无处安放被调用者的尾部返回值。丢弃那个**值**是对的,丢弃它的**求值**则不对 —— 求值过程可能通过 `Out` / `InOut` 参数写入、启动任务、阻塞等待信号,或完成硬件设置。因此每个被丢弃的值都要分类:

| 被丢弃的值 | 行为 |
| ---------- | ---- |
| `Call` —— 无论被调用者是跨函数还是 builtin | 按 return 顺序重新发出为 `EvalStmt`。若跨函数调用的被调用者同样是 Inline,不动点循环会在下一轮展开它;否则它就保持为一次普通派发,与作者在调用点直接写出来完全一致。 |
| `Submit` | 重新发出为 `EvalStmt`。任务启动本身就是有副作用的,与被调用者做什么无关。 |
| 其它不藏有调用的值 —— `Var`、常量 | 丢弃。 |
| 本身不是 call-like、但**包裹**了调用的值 —— `self.bump(n) + 1` 这类标量算术、`MakeTuple`、`TupleGetItemExpr` | 抛出 `pypto::ValueError`。它无法变成 `EvalStmt`,而删除它会连带删除内层的调用。请直接返回该调用,或在调用点绑定 wrapper 的结果。 |

**为什么是"所有调用",而不只是"会写的调用"。** IR 里没有任何东西能回答"这次调用可以安全删除吗"。最接近的注册表数据 `OpRegistryEntry::WritesAnyArg` 回答的是另一个问题 —— 算子是否**通过参数**写入 —— 以它为删除依据在两个方向上都会出错:

- 大多数算子根本没有分类 —— 撰写时 315 个里有 263 个,其中就包括被 `dce::IsSideEffectOp` 列为副作用算子的 `tile.tpush_to_aiv` 和 `system.aic_initialize_pipe`。`OpRegistryEntry::HasDeclaredArgEffects` 存在的意义正是让分析能区分"已声明不写"和"尚无人查看"。
- **正面**的 `no_arg_writes()` 判定也不等于可删除。`pld.system.wait` 会阻塞直到信号槽满足阈值,`pld.system.defer_wait` 注册任务完成条件,`system.set_ffts` 把 workspace 指针交给 FFTS 单元 —— 三者都声明了 `no_arg_writes()`,却都承担同步或硬件设置语义。

因此该 pass 保留所有调用。被丢弃的真·纯调用会留下一条无用的 `EvalStmt`,流水线可以无害地带着它走完。要收窄这一点,需要一个真正的"可安全删除"算子属性 —— 逐算子声明,而不是从写入行为反推。

**变换前**:

```python
@pl.function(type=pl.FunctionType.Inline)
def writeout(self, t, out: pl.Out[...]):
    return pl.tile.store(t, [0, 0], out)   # 写入本身就是返回表达式

@pl.function(type=pl.FunctionType.InCore)
def kernel(self, a, out: pl.Out[...]):
    t = pl.tile.load(a, [0, 0], [64, 64])
    self.writeout(t, out)                  # 返回值被忽略
    return out
```

**变换后** —— store 被保留:

```python
@pl.function(type=pl.FunctionType.InCore)
def kernel(self, a, out: pl.Out[...]):
    t = pl.tile.load(a, [0, 0], [64, 64])
    pl.tile.store(t, [0, 0], out)
    return out
```

## 验证

`InlineFunctionsEliminated` `PropertyVerifier`(注册到 `IRProperty.InlineFunctionsEliminated`)确认:

1. 不存在 `func_type == FunctionType::Inline` 的 `Function`。
2. 不存在指向 Inline 函数的 `Call`。

## 参见

- `python/pypto/jit/decorator.py` — `@pl.jit.inline` 是用户层入口(`_SubFunctionDecorator("inline", ...)`)。
- [02-unroll_loops](02-unroll_loops.md) — 紧随其后运行。
- [09-outline_incore_scopes](10-outline_incore_scopes.md) — 处理展开后剩余的 `pl.at` scope。
