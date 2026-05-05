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
     - 对内联体中每个本地绑定的 `Var` 做 alpha 重命名(`<orig>_inline<counter>`),避免多个调用点之间冲突。
     - 在调用点之前插入重命名+替换后的函数体语句。
     - 用 `LHS = renamed_return`(单返回值)或 `LHS = MakeTuple([renamed_returns...])`(多返回值)替换调用。当 `LHS` 与替换后的返回 `Var` 是同一个 `Var` 时,赋值被省略以避免冗余 SSA 拷贝。
4. **删除**所有 Inline 函数。

重命名后缀使用单下划线(`_inline`),因为 `__` 被 IR 自动命名约定保留(参见 `auto_name_utils.h`)。

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
| 多返回值 Inline | 在调用点发出 `LHS = MakeTuple([rets...])`。后续 `Simplify` 可能把 `TupleGetItemExpr(MakeTuple(...), i)` 折叠掉。 |
| 嵌套 Call 到 Inline(如 `pl.add(inline_fn(x), y)`) | v1 不处理 — 保持原样。`InlineFunctionsEliminated` verifier 会标记任何残留的 Call。 |

## 验证

`InlineFunctionsEliminated` `PropertyVerifier`(注册到 `IRProperty.InlineFunctionsEliminated`)确认:

1. 不存在 `func_type == FunctionType::Inline` 的 `Function`。
2. 不存在指向 Inline 函数的 `Call`。

## 参见

- `python/pypto/jit/decorator.py` — `@pl.jit.inline` 是用户层入口(`_SubFunctionDecorator("inline", ...)`)。
- [02-unroll_loops](02-unroll_loops.md) — 紧随其后运行。
- [10-outline_incore_scopes](10-outline_incore_scopes.md) — 处理展开后剩余的 `pl.at` scope。
