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
     - 构建参数替换映射(形参 `Var` → 实参 `Expr`)。
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

## 边界情况

| 情况 | 行为 |
| ---- | ---- |
| 无调用点的 Inline 函数 | 静默从程序中移除。 |
| 作为程序入口的 Inline 函数 | 此处不视为错误 — 但因为没有任何 Call 指向它,清理阶段会像任何无调用者函数那样移除。 |
| Inline 调用 Inline(传递) | 迭代到不动点。 |
| 递归 Inline(自递归或互相调用) | 在任何展开发生之前抛出 `pypto::ValueError`,消息中标明环路径(`a -> b -> a`)。 |
| 多返回值 Inline | **不**发出 `LHS = MakeTuple([rets...])` — 编排层 codegen 无法 lower `MakeTuple`。改为把克隆后的返回值记录在 LHS `Var` 上,并把下游 `TupleGetItemExpr(LHS, i)` 的使用改写为第 `i` 个值,使该 LHS 绑定最终无人引用(参见 `SpliceInlineCallAsTupleSub`)。 |
| 嵌套 Call 到 Inline(如 `pl.add(inline_fn(x), y)`) | v1 不处理 — 保持原样。`InlineFunctionsEliminated` verifier 会标记任何残留的 Call。 |
| `EvalStmt(inline_call(...))` — 返回值被忽略 | 被丢弃的是返回**值**,不是它的**求值**。参见下方[丢弃返回值](#丢弃返回值)。 |

## 丢弃返回值

`EvalStmt` 调用点(`self.wrapper(x, out)`,没有 LHS)无处安放被调用者的尾部返回值。丢弃那个**值**是对的,丢弃它的**求值**则不对 —— 求值过程可能通过 `Out` / `InOut` 参数写入。因此每个被丢弃的值都要分类:

| 被丢弃的值 | 行为 |
| ---------- | ---- |
| 算子注册表未正面判定为"不写入"的 `Call` —— 跨函数派发、`tile.store` 这类会写的 builtin,或任何尚无人分类的算子 | 按 return 顺序重新发出为 `EvalStmt`。若跨函数调用的被调用者同样是 Inline,不动点循环会在下一轮展开它;否则它就保持为一次普通派发,与作者在调用点直接写出来完全一致。 |
| `Submit` | 重新发出为 `EvalStmt`。任务启动本身就是有副作用的,与被调用者做什么无关。 |
| 算子已声明"不通过任何参数写入"的 `Call`,或其它不藏有副作用的值(`Var`、常量) | 丢弃。 |
| 本身不是 call-like、但**包裹**了有副作用内容的值 —— `self.bump(n) + 1` 这类标量算术、`MakeTuple`、`TupleGetItemExpr` | 抛出 `pypto::ValueError`。它无法变成 `EvalStmt`,而删除它会连带删除内层调用的写入。请直接返回该调用,或在调用点绑定 wrapper 的结果。 |

注册表刻意区分"已声明不通过任何参数写入"与"尚无人分类此算子"(`OpRegistryEntry::HasDeclaredArgEffects`),而目前大多数算子仍属于后者 —— 其中包括 `tile.tpush_to_aiv`、`system.aic_initialize_pipe` 这些明显有副作用、被 `dce::IsSideEffectOp` 列为副作用算子的项。只有正面判定才允许删除,因此未分类的纯算子会被保守地保留为一条无用但无害的 `EvalStmt`。

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
- [09-outline_incore_scopes](09-outline_incore_scopes.md) — 处理展开后剩余的 `pl.at` scope。
