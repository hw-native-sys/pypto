# StampTfreeSplit Pass(标记 tfree 的 split）

## 概述

`StampTfreeSplit` 把每个跨核 tpop 的 `split`(以及 pipe `id`)复制到与之配对的 `tfree` 算子上,
使 PTO codegen 能直接从 `tfree` 算子读取这些属性,而不再依赖 codegen 侧的查找表。

`system.tfree_to_aic` / `system.tfree_to_aiv` 自身不携带 split——split 模式存在于发起它的
`tile.tpop_from_aic` / `tile.tpop_from_aiv` 调用上。本 pass 把该信息显式写到 IR 节点上。

## 在流水线中的位置

```text
... -> SplitVectorKernel -> StampTfreeSplit -> NormalizeReturnOrder -> SkewCrossCorePipeline -> ...
```

它紧跟在 `SplitVectorKernel` 把 tpop 的 `split` 定稿之后运行,并在 `SkewCrossCorePipeline` 把
tpop/tfree 配对克隆进软件流水的 prologue/epilogue 之前 —— 这样每个克隆都带上已盖好的 `split`,
而且查表时每个 `tfree` 的实参仍是它直接的 tpop 结果 var(还没被流水改成循环 carry 值)。

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

若某个 `tfree` 已携带一个与其 tpop 的 id 不一致的显式 `id`,pass 会报错(用户错误)。它也会拒绝
方向与来源 tpop 不匹配的 `tfree`(例如对 `tpop_from_aic` 的结果用 `tfree_to_aiv`),以及 tile 找
不到来源 tpop 的 `tfree`。这些都是 codegen 过去执行的一致性校验。

## 为什么用一个靠后的统一 pass

有两条写法都会产生 tfree,二者都必须覆盖:

- **混合核**(`pl.at(..., split=...)`):tfree 由 [`ExpandMixedKernel`](19-expand_mixed_kernel.md)
  内的 `FinalizeTpopTfrees` 生成,而后者只处理 InCore 函数。
- **显式** `@pl.function(type=AIC/AIV)`:用户直接写 `pl.tfree_to_aic`,这些完全绕过 finalizer。

在 split 定稿之后用一个 pass 遍历所有函数即可统一覆盖二者,因此 `FinalizeTpopTfrees` 和显式函数的
lowering 都无需各自再实现 split 标记逻辑。

## 消费方

`system.tfree_to_ai{c,v}` 的 PTO codegen 通过 `op->GetKwarg<int>("split", 0)` 从算子读取 `split`
(以及 `id`)。codegen 侧不再有 tpop 跟踪表。
