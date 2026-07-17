# InsertCommFence Pass

## 概述

`InsertCommFence` 实现最新 PTOAS 下放给编译器的 *data-before-signal* 义务 —— 跨 rank
的写必须在释放它的 `pld.system.notify` 之前对 peer 可见 —— 用两类 op、两种粒度完成：

1. `system.cacheinvalid` 是**每地址一条**：**紧跟在每个发布写之后**发射一条，失效该次
   写入区域的缓存行。
2. `system.fence` 是**每 notify 一条**：notify 前的一条 GM 屏障对*之前所有*写做排序，
   因此不管前面有多少个写，共用一条 fence。

```text
remote_store(a); remote_store(b); notify
  -> remote_store(a); cacheinvalid(a); remote_store(b); cacheinvalid(b); fence; notify
```

- `system.cacheinvalid` 降为 `pto.cmo.cacheinvalid … single_cache_line`。
- `system.fence` 降为 `pto.fence.barrier_all #pto.fence_scope<gm>` —— 带 DDR 可见性的
  GM 屏障，强于裸的 `pto.barrier <PIPE_ALL>`。

二者互不相干，因此是两趟**独立遍历**（`CacheInvalidInserter`、`FenceInserter`）——
顺序无所谓（cacheinvalid 对 fence 分析而言是惰性的）。

### cacheinvalid 插在写入点（必在作用域内）

把 cacheinvalid 紧插在其写之后，目标 `Var` 天然在作用域内（写刚用过它）—— 无论它是
window 参数、别名（`dv = pl.tensor.view(win); remote_store(dv)`）、循环携带的 `iter_arg`，
还是分支内定义的值。**不需要任何跨作用域跟踪，也绝不会 silent drop**：每条 cacheinvalid
总是落在它的写旁边，任何嵌套层级皆然。（对比早先设计：把 cacheinvalid 批到 notify 前
插入，就不得不推理哪些目标在那里仍在作用域内。）

cacheinvalid 目前覆盖**整个目标张量**（以全 `0` offset 覆盖完整 shape），复用类型的 dim
表达式。收窄为精确写入子区域（写自身的 `(shapes, offsets)` 参数就在写入点旁边）是后续
升级项。

## 在流水线中的位置

```text
... -> MaterializeRuntimeScopes -> ClassifyIterArgCarry -> InsertCommFence   (最后)
```

它在 Default 流水线中**最后**运行，位于所有会重排语句的 pass
（`SkewCrossCorePipeline`、`LowerPipelineLoops`、`CanonicalizeIOOrder` ...）之后。
`system.fence` 无操作数、无依赖边，若更早插入可能被挪离其 notify；放最后可让 fence
与 notify 在 codegen 前保持相邻。它之前的 `MaterializeRuntimeScopes` /
`ClassifyIterArgCarry` 只改动 Orchestration 函数，因此本 pass 看到的 InCore IR 正是
codegen 最终降级的 IR。

## 什么算发布写

由 `op_predicates::IsPublishingWrite` 判定：

| 情形 | 条件 |
| ---- | ---- |
| `pld.tile.remote_store` / `pld.tile.put` / `pld.tensor.put` | 远程写（始终） |
| `tile.store` | 目标张量（arg 2）为 window-bound 的 `DistributedTensorType`（peer 可 `remote_load`） |
| `pld.tile.get` / `pld.tensor.get` | 本地目标（arg 0）为 window-bound |
| 对未注册 op 名的任意调用 | 保守 —— 未做过程间分析的用户函数 |

`remote_load`（结果是 tile、不写 GM）以及写入普通 `Tensor` 的 `tile.store` **不是**
发布写。

## 算法

两趟独立遍历，各自 `O(N)`。

### 遍历 1 —— `CacheInvalidInserter`（每地址，纯结构）

一趟纯结构改写：在每个 `SeqStmts` 的发布写子节点之后，追加一条覆盖整张量的
`cacheinvalid`（目标取自该写）。若某个 `if`/`for` 的裸单语句体本身就是发布写，则就地
包裹（`body -> { body; cacheinvalid }`）。无控制流分析、无 `pending` 状态。幂等：已经跟
着 cacheinvalid 的写不再重复插入。

### 遍历 2 —— `FenceInserter`（每 notify，控制流敏感）

**趟 1 —— 子树 summary（自底向上 / 后序，记忆化）。** 每条语句记录三个纯结构位：

- `opens_with_notify` —— 存在一条路径在遇到任何写/fence 之前先到达 notify；
- `may_end_with_write` —— 子树出口可能残留未被覆盖的发布写；
- `transparent` —— 存在一条不碰写/fence/notify 的直穿路径。

循环始终 `transparent`（可能零次迭代）；`opens` / `may_end` 取自 body。

**趟 2 —— 前向插入。** 一个 `pending` 布尔按执行序流动。在每个 `SeqStmts` 层，若
`pending` 且下一子节点 `opens_with_notify`，则在其前插入一条 fence（随后清零 `pending`）
—— **但在 `if` 之前不插**：对 `if` 改为递归进入，让每个分支在各自真实的 notify 处落点：

```text
remote_store; cacheinvalid; notify         -> ...; fence; notify           （直线）
remote_store; cacheinvalid; for p: notify  -> ...; fence; for p: notify     （提到循环前）
remote_store; cacheinvalid; if c: notify   -> ...; if c: { fence; notify }  （分支内部）
```

`if` 与循环刻意区别对待。`if` 内的 notify 是*有条件*的，把屏障推入被执行的分支比在
`if` 前无条件地提前一个屏障更精确（分支不走时不发屏障）。body **不以写结尾**的循环
（如 `for p: notify` barrier）*仍会*提前到循环前，用**一条**屏障覆盖所有迭代，而非每
迭代一条。

body **可能以写结尾**的循环以 `pending || may_end_with_write(body)` 进入，因此上一轮
迭代尾部的写会为下一轮迭代头部的 notify 补 fence —— ring-allreduce 的回边（该尾部写
自身的 cacheinvalid 已由遍历 1 插好）。这类循环本就会在自己头部补 fence，因此流入的
pending 写**不再**提前到循环前：迭代 0 的头部 fence 已覆盖它（提前插会是同一 notify
前的第二条冗余屏障）。即 `store; for { notify; store }` 只发一条 fence，在循环头部 ——
而非循环前一条 + 循环内一条：

```text
for s: { for p: notify(...); ...; store(win); cacheinvalid(win) }
     -> for s: { fence; for p: notify(...); ...; store(win); cacheinvalid(win) }
```

已存在的 `system.fence` 会清零 `pending`，因此本 pass **幂等**，并将用户手写的 fence
视为完整屏障（不重复插入 fence）。

## 与 Codegen 的关系

`InsertCommFence` 取代了 `MakePutCodegenPTO` 中原先在每个 TPUT 之后无条件发射的
drain 屏障（PTOAS#872 的 workaround）。该 codegen 屏障被移除：fence 现在仅在确有
notify 跟随时才发射，并提供裸 pipe 屏障无法给出的 DDR 可见性 drain。TPUT/TGET 的
**前置** 屏障与 TGET 的**尾部**屏障与此无关（核内 RAW 排序，而非 data-before-signal），
保留不动。

## 消费者

流水线下游无消费者。PTO codegen 通过既有的 op handler 降级插入的
`system.cacheinvalid` 与 `system.fence`；没有其它 pass 需要理解它们。
