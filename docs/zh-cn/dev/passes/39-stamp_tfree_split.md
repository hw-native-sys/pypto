# StampTfreeSplit Pass(标记 tfree 的 split）

## 概述

`StampTfreeSplit` 把每个跨核 tpop 的 `split`(以及 pipe `id`)复制到与之配对的 `tfree` 算子上,
使 PTO codegen 能直接从 `tfree` 算子读取这些属性,而不再依赖 codegen 侧的查找表。

`system.tfree_to_aic` / `system.tfree_to_aiv` 自身不携带 split——split 模式存在于发起它的
`tile.tpop_from_aic` / `tile.tpop_from_aiv` 调用上。本 pass 把该信息显式写到 IR 节点上。

## 在流水线中的位置

```text
... -> SplitVectorKernel -> ... -> Simplify(最后一次) -> StampTfreeSplit -> MaterializeRuntimeScopes
```

它运行在 `SplitVectorKernel` 把 tpop 的 `split` 定稿之后、跨核流水 pass 把 tpop/tfree 的位置settle
之后、以及最后一次 `Simplify` 之后,从而保证在 codegen 之前没有 pass 会抹掉所盖的 kwargs。

## 行为

对每个函数(包括 AIC 与 AIV 函数体),pass 先建立一张从每个 tpop 结果 `Var` 到其 `{split, id}` 的
映射,然后对每个 tile 实参是已知 tpop 结果的 `tfree`,把 `split`(以及 tpop 带有 `id` 时的 `id`)
盖到该 `tfree` 调用上:

```text
# 之前
t = tile.tpop_from_aic(split=1, id=2)
...
system.tfree_to_aic(t)

# 之后
t = tile.tpop_from_aic(split=1, id=2)
...
system.tfree_to_aic(t, split=1, id=2)
```

若某个 `tfree` 已携带一个与其 tpop 的 id 不一致的显式 `id`,pass 会报错(用户错误)——这与
codegen 过去执行的一致性校验相同。

## 为什么用一个靠后的统一 pass

有两条写法都会产生 tfree,二者都必须覆盖:

- **混合核**(`pl.at(..., split=...)`):tfree 由 [`ExpandMixedKernel`](21-expand_mixed_kernel.md)
  内的 `FinalizeTpopTfrees` 生成,而后者只处理 InCore 函数。
- **显式** `@pl.function(type=AIC/AIV)`:用户直接写 `pl.tfree_to_aic`,这些完全绕过 finalizer。

在 split 定稿之后用一个 pass 遍历所有函数即可统一覆盖二者,因此 `FinalizeTpopTfrees` 和显式函数的
lowering 都无需各自再实现 split 标记逻辑。

## 消费方

`system.tfree_to_ai{c,v}` 的 PTO codegen 通过 `op->GetKwarg<int>("split", 0)` 从算子读取 `split`
(以及 `id`)。codegen 侧不再有 tpop 跟踪表。
