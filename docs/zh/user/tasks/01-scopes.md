# 运行时作用域

依赖推导运行所在的边界，以及关掉它的那个开关。

> **前置**：[依赖模型](00-model.md)。

## Concept

一个**运行时作用域**（`SIMPLER_SCOPE`）同时是两样东西：

- OverlapMap 跟踪依赖的区域，以及
- 一个堆层级，因此嵌套作用域各自独立回收内存。

运行时提供了一个隐式的顶层作用域 —— 这就是为什么你可以从头到尾不提作用域也能写出完整程序。**写作用域是调优与控制手段，从来不是正确性要求。**

默认情况下编译器替你放置 AUTO 作用域：函数体外面一个，每个 `for` 和 `if` 体外面各一个。你接管它，只是为了作用域携带的第二样东西：它的**模式**。

| 模式 | 含义 |
| ---- | ---- |
| `pl.scope()` / `ScopeMode.AUTO` | OverlapMap 自动依赖跟踪开启 |
| `pl.scope(mode=pl.ScopeMode.MANUAL)`，别名 `pl.manual_scope()` | 自动跟踪关闭 —— 每条边由你声明 |

`manual_scope` 是运行时各种退出手段里最粗的一种。动手之前请注意：你通常并不需要它 —— `deps=` 在 auto 作用域里本来就能用，补一条缺失的边不必为此放弃整个区域的推导。更细粒度的退出方式见 [精修依赖图](03-tuning.md)。

## Quickstart：接管一个区域的边

```python
with pl.manual_scope():
    scratch, tid = pl.submit(self.stage1, x, scratch)
    out, _       = pl.submit(self.stage2, scratch, out, deps=[tid])
```

在这个块里，运行时对每一次 submit 都跳过 OverlapMap 的查询与插入，所以本来会被推出的 `scratch` 重叠**不会**被推出 —— 现在 `deps=[tid]` 是唯一给两个阶段定序的东西。去掉它，它们就可能重叠。

> 这是片段：`self.stage1` / `self.stage2` 是外层 `@pl.program` 的方法，这段代码位于一个 Orchestration 函数体内。

## Mechanics

### 作用域可以出现在哪里

| 规则 | 细节 |
| ---- | ---- |
| 只能在 Orchestration | 作用域属于控制面；在 InCore 函数里非法 |
| `mode=AUTO` 需要 `auto_scope=False` | 默认的 `@pl.function(auto_scope=True)` 下，AUTO 的放置归编译器，自己写会被拒绝 |
| `mode=MANUAL` 始终允许 | 它是依赖语义的选择，不是环的调优 |
| AUTO 不得嵌套在 MANUAL 内 | 运行时禁止 |
| `manual_scope` 不得嵌套在 `manual_scope` 内 | 运行时禁止 |

### 哪些装饰器接受 `auto_scope=False`

`@pl.jit`、`@pl.jit.host`、`@pl.jit.inline` 接受。`.incore` 与 `.opaque` 拒绝 —— 它们会被 outline 成独立 kernel，没有可供作用域容身的编排函数体。inline 体会被拼接进调用方，因此写在其中的作用域落在调用方。

### 进入 manual 作用域你放弃了什么

OverlapMap 本来会为该区域内每一次 submit 推出的一切 —— 包括你当时没想到的那些边。这正是这个构造的用意，也是它的风险：manual 作用域里漏写一个 `deps=`，得到的不是一条诊断，而是一个竞态。

当你**确实想拥有整张图**时才用它 —— 比如一条形状你已经了然于胸的手工流水线 —— 而不要把它当作"修正某一条我不认同的推导"的手段。

## 边界情况

> **致命陷阱：** `manual_scope` 不会就你漏掉的边发出任何警告。本来能发现这个遗漏的推导，正是你亲手关掉的那个。

| 症状 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| **kernel 里的作用域被拒绝** | 作用域只属于 Orchestration | 移到派发该 kernel 的编排函数里 |
| **`mode=AUTO` 被拒绝** | 默认的 `auto_scope=True` 把 AUTO 放置留给了编译器 | 设 `@pl.function(auto_scope=False)`，或改用 MANUAL |
| **`auto_scope=False` 被拒绝** | 用在了 `.incore` / `.opaque` 上 | 放到入口或 `.inline` 辅助函数上 |
| **嵌套作用域被拒绝** | AUTO 嵌在 MANUAL 内，或 `manual_scope` 套 `manual_scope` | 拍平 —— 运行时两者都禁止 |
| **加了 `manual_scope` 之后出现竞态** | 原本由推导得出的边现在没有了 | 用 `deps=` 声明它们，或者干脆去掉 manual 作用域、只补你需要的那一条边 |

## See Also

- [依赖模型](00-model.md) —— 跟踪开启时它会推出什么。
- [声明一条边](02-submit.md) —— `deps=`，两种模式下都有效。
- [精修依赖图](03-tuning.md) —— 不必付出整个区域代价的细粒度退出方式。
- [MaterializeRuntimeScopes](../../dev/passes/46-materialize_runtime_scopes.md) —— AUTO 作用域是怎么放置的。
