# MaterializeRuntimeScopes Pass

向 Orchestration 函数中插入显式的 AUTO `RuntimeScopeStmt` 节点，使 PTO
orchestration codegen 直接从 IR 中 1:1 地 emit `PTO2_SCOPE()`，而不再依据
`for` / `if` 语句结构推导 scope。

## 概述

simpler 运行时会把 orchestration 例程的若干区域包裹在 `PTO2_SCOPE()` 块中（通过
OverlapMap 做自动依赖追踪）。过去 orchestration codegen 依据语句结构来决定*在哪里*
emit 这些块：它隐式地把整个函数体、每个 `ForStmt` 体、每个 `IfStmt` 分支体都包进
`PTO2_SCOPE()`——并在 manual scope 内部抑制这种包裹，因为运行时禁止 AUTO 嵌套在
MANUAL 之内。

这把 scope *策略*埋进了 printer。本 pass 把该策略移入 IR。对每个
`FunctionType::Orchestration` 函数，它插入显式的 AUTO `RuntimeScopeStmt`
（`manual_ = false`）节点：

- 包裹整个函数体，以及
- 包裹每个 `ForStmt` 体和每个 `IfStmt` 的 then/else 体，

同时在任何 manual `RuntimeScopeStmt` 内部跳过插入。此后 codegen **只**从
`RuntimeScopeStmt` 节点 emit `PTO2_SCOPE`——与 IR 保持 1:1（参见
[orchestration codegen](../codegen/01-orchestration_codegen.md)）。

插入的 scope 拥有 DSL 表示 `with pl.auto_scope():`，因此该 IR 能像其它构造一样
通过 printer/parser 往返（round-trip）。

**何时使用**：在 `Default` 与 `DebugTileOptimization` 策略中作为最后一个 pass
运行，位于最终的 `Simplify` 之后。放在最末意味着其它任何 transform 都无需处理
被插入的 scope 包裹。

**作用范围**：仅修改 `Orchestration` 函数。InCore / AIC / AIV / Group / Spmd
的函数体从不会被 codegen 包裹 scope，因此原样返回。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::MaterializeRuntimeScopes()` | `passes.materialize_runtime_scopes()` | 函数级 |

```python
from pypto.pypto_core import passes

scoped = passes.materialize_runtime_scopes()(program)
```

## 算法

`InsertAutoScopeMutator` 遍历每个 Orchestration 函数体：

1. 进入 **manual** `RuntimeScopeStmt` 时递增深度计数；当计数非零时抑制 AUTO
   插入（运行时禁止 AUTO 嵌套在 MANUAL 内）。AUTO scope 不抑制嵌套。
2. 对每个 `ForStmt`，若其体尚未被 AUTO 包裹，则替换为
   `RuntimeScopeStmt(manual=false, body)`。
3. 对每个 `IfStmt`，其 then/else 体各自以相同方式包裹。

mutator 运行后，整个函数体再被包进一个最外层 AUTO scope（对应 codegen 过去恒定
emit 的最外层 `PTO2_SCOPE()`）。该包裹是幂等的——已是 AUTO 的函数体保持不变。

| 来源 | 处理 |
| ---- | ---- |
| Orchestration 函数体 | 包进一个 AUTO `RuntimeScopeStmt` |
| `ForStmt` 体（不在 manual scope 内） | 包进 AUTO `RuntimeScopeStmt` |
| `IfStmt` then/else 体（不在 manual scope 内） | 各自包进 AUTO `RuntimeScopeStmt` |
| manual `RuntimeScopeStmt` 内的任何体 | 保持裸露 |
| 非 Orchestration 函数 | 原样返回 |

## 示例

```python
# Before
@pl.function(type=pl.FunctionType.Orchestration)
def orch(self, a, out):
    for i in pl.range(4):
        out = self.kernel(a, out)
    return out
```

```python
# After MaterializeRuntimeScopes
@pl.function(type=pl.FunctionType.Orchestration)
def orch(self, a, out):
    with pl.auto_scope():            # function body
        for i in pl.range(4):
            with pl.auto_scope():    # loop body
                out = self.kernel(a, out)
        return out
```

末尾的 return-var `yield` 保留在 scope 内；printer 会递归穿过 AUTO scope，从而
保留 `var = pl.yield_(...)` 的赋值左值；parser 也会把 `pl.auto_scope()` 内的
yield 视为外层 for/if 的 return-var。

## 验证

**测试**：`tests/ut/ir/transforms/test_materialize_runtime_scopes.py`（函数体 +
for/if 包裹、manual scope 抑制、幂等性、非 Orchestration 不受影响）以及
`tests/ut/language/parser/test_auto_scope_parsing.py`（`pl.auto_scope()` 的
解析 / round-trip / 嵌套限制）。完整的 orchestration codegen 测试套件
（`tests/ut/codegen/test_orchestration_codegen.py`）验证 emit 的 `PTO2_SCOPE`
输出与此前由 codegen 驱动的行为逐字节一致。

## Pass 属性

| 属性 | 取值 |
| ---- | ---- |
| Required | `SplitIncoreOrch`、`CallDirectionsResolved` |
| Produced | `RuntimeScopesMaterialized` |
| Invalidated | — |
