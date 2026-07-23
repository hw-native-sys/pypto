# MaterializeRuntimeScopes Pass

向 Orchestration 函数中插入显式的 AUTO `RuntimeScopeStmt` 节点，使 PTO
orchestration codegen 直接从 IR 中 1:1 地 emit `PTO2_SCOPE()`，而不再依据
`for` / `if` 结构推导 scope —— 除非函数用 `@pl.function(auto_scope=False)`
选择退出，此时由用户手工摆放每个 scope。

## 概述

simpler 运行时把 orchestration 例程的若干区域包进 `PTO2_SCOPE()` 块（通过
OverlapMap 做自动依赖追踪），并提供一个隐式顶层 scope。因此 scope 在编译器侧
是一种**调优 / 放置**手段，从不是正确性要求 —— 一个函数可以一个编译器 scope
都没有。

默认（`auto_scope=True`）由编译器决定 scope 放置：对每个
`FunctionType::Orchestration` 函数，本 pass 插入 AUTO `RuntimeScopeStmt`
（`manual_ = false`），包裹整个函数体以及每个 `ForStmt` 体和 `IfStmt` 的
then/else 体（在 manual scope 内部抑制，因为运行时禁止 AUTO 嵌套在 MANUAL）。
此后 codegen **只**从 `RuntimeScopeStmt` 节点 emit `PTO2_SCOPE`，与 IR 保持
1:1（见 [orchestration codegen](../codegen/01-orchestration_codegen.md)）。

当 `AutoDeriveTaskDependencies(analyze_auto_scopes=True)` 已经证明默认模式下的
函数体或后续 for/if 分区完全由编译器推导出的显式 deps 覆盖时，本 pass 会为该区域
发出编译器拥有的 MANUAL `RuntimeScopeStmt`，而不是匹配任何特定 callee 名称。
call 级 marker 会在这里被消费；只有 scope 级 marker 会保留下来，供 codegen 的
结构性 lookahead 使用。

在 `@pl.function(auto_scope=False)` 下，本 pass **什么都不插**：用户用
`with pl.scope()` / `with pl.scope(mode=pl.ScopeMode.MANUAL)` 摆放 scope，由
parser 直接物化进 IR。这是控制 scope 粒度（ring 隔离）、MANUAL 依赖区域以及
完全接管的旋钮。

默认模式下物化 scope 后，pass 会把函数标记为 `auto_scope=False`（scope 已放置）。
这让 pass 幂等，并让输出能 round-trip：插入的 `with pl.scope()` 块只有在
`auto_scope=False` 下才能解析回来（parser 在默认模式下拒绝手写 AUTO scope，
默认模式由编译器决定放置）。

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

## 行为

| 函数 | for/if + 函数体 | 手写 `with pl.scope()` |
| ---- | --------------- | ---------------------- |
| `auto_scope=True`（默认） | 自动包进 AUTO scope；完整 compiler 覆盖的区域可变成编译器拥有的 MANUAL scope | parser 拒绝（请用 `auto_scope=False`） |
| `auto_scope=False` | 不自动包（pass 为 no-op） | 唯一的 scope；`with pl.scope(mode=MANUAL)` 与 `manual_scope` 别名也允许 |

默认模式下，`InsertAutoScopeMutator` 遍历函数体：

1. 进入 **manual** `RuntimeScopeStmt` 时递增深度计数；计数非零时抑制 AUTO 插入
   （禁止 AUTO-in-MANUAL）。AUTO scope 不抑制嵌套。
2. 每个 `ForStmt` 体若未被 AUTO 包裹则包进 `RuntimeScopeStmt(manual=false)`；
   每个 `IfStmt` 的 then/else 体同理。如果该后续分区内的所有 task call 都带有
   compiler auto-manual marker，则改用编译器拥有的 MANUAL wrapper。
3. 随后整个函数体被包进一个最外层 AUTO scope；如果函数级 marker 存在，则改用
   编译器拥有的 MANUAL scope。最后把函数标记为 `auto_scope=False`。

## 示例

```python
# Before —— 默认 auto_scope=True
@pl.function(type=pl.FunctionType.Orchestration)
def orch(self, a, out):
    for i in pl.range(4):
        out = self.kernel(a, out)
    return out
```

```python
# After MaterializeRuntimeScopes（标记 auto_scope=False；可 round-trip）
@pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
def orch(self, a, out):
    with pl.scope():            # function body
        for i in pl.range(4):
            with pl.scope():    # loop body
                out = self.kernel(a, out)
        return out
```

```python
# 选择退出，自己摆放 scope（此处更粗的粒度：只有一个 scope）
@pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
def orch(self, a, out):
    with pl.scope():
        for i in pl.range(4):
            out = self.kernel(a, out)
        return out
```

末尾的 return-var `yield` 保留在 scope 内；printer 会递归穿过 AUTO scope 以保留
`var = pl.yield_(...)` 的赋值左值；parser 也会把 `pl.scope()` 内的 yield 视为
外层 for/if 的 return-var。

## 验证

**测试**：`tests/ut/ir/transforms/test_materialize_runtime_scopes.py`（auto 模式
包裹、manual scope 抑制、幂等、opt-out no-op / scope 保留、默认模式拒绝 AUTO）
以及 `tests/ut/language/parser/test_scope_parsing.py`（`pl.scope()` 解析 /
round-trip / 模式 / 嵌套 / opt-out 规则）。完整 orchestration codegen 测试套件
（`tests/ut/codegen/test_orchestration_codegen.py`）验证 emit 的 `PTO2_SCOPE`
输出与此前由 codegen 驱动的行为逐字节一致。

## Pass 属性

| 属性 | 取值 |
| ---- | ---- |
| Required | `SplitIncoreOrch`、`CallDirectionsResolved` |
| Produced | `RuntimeScopesMaterialized` |
| Invalidated | — |
