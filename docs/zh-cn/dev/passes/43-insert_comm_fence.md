# InsertCommFence Pass

## 概述

`InsertCommFence` 实现最新 PTOAS 在其 `pto-memory-consistency` pass 中强制、并下放给
编译器的 *data-before-signal* 内存一致性契约。该契约是**双向**的：

- **发布侧。** `pto.comm.tnotify` 要求在其匹配的 `pto.cmo.cacheinvalid` 释放标记之后、
  信号之前，显式插入一条 `pto.fence.barrier_all #pto.fence_scope<gm>` —— 跨 rank 的写
  必须在释放它的 notify 之前对 peer 可见。
- **消费侧。** `pto.comm.twait`（或成功的 `pto.comm.ttest`）之后的可缓存 GM load 之前，
  要求显式插入一条 `pto.cmo.cacheinvalid all #pto.address_space<gm>`，使读方看到 peer
  的最新写。

两处缓存标记都是**同一个** `system.cacheinvalid` op，按参数个数分两种形态：

| 形态 | IR | 降级为 |
| ---- | -- | ------ |
| 区域 | `system.cacheinvalid(tensor, shapes, offsets)` | `pto.cmo.cacheinvalid … single_cache_line` |
| 全 GM | `system.cacheinvalid()`（无参） | `pto.cmo.cacheinvalid all #pto.address_space<gm>` |

`system.fence` 降为 `pto.fence.barrier_all #pto.fence_scope<gm>` —— 带 DDR 可见性的 GM
屏障，强于裸的 `pto.barrier <PIPE_ALL>`。

## 本 pass 插入什么

一趟前向结构遍历（`InsertCommMarkers`）按 op 插入：

- **每个发布写之后** —— 一条覆盖整张量的**区域** `system.cacheinvalid`，**紧跟一条
  `system.fence`**。ptoas 要求 fence 直接跟在释放标记之后，因此两者都落在写入点（每地址
  一条），不推迟到 notify；
- **每个裸 barrier notify 之前** —— 一条**全 GM** `system.cacheinvalid` + `system.fence`。
  *裸 barrier notify* 指没有已 fence 的发布写待处理者（纯 barrier 信号，或其写位于前一个
  循环、`pending` 已被重置）。有已 fence 发布写待处理的 notify 无需任何标记 —— 该写自身的
  `cacheinvalid; fence` 即为释放标记；
- **每个 wait 之后** —— 一条**全 GM** `system.cacheinvalid`（消费侧在下一次可缓存读之前
  的失效）。

```text
remote_store(a); remote_store(b); notify
  -> remote_store(a); cacheinvalid(a); fence; remote_store(b); cacheinvalid(b); fence; notify

for p: (if p != me: notify)                 （裸 barrier notify —— 无写）
  -> for p: (if p != me: cacheinvalid(); fence; notify)

for c: store; for p: notify                 （写与 notify 在不同循环）
  -> for c: (store; cacheinvalid; fence);
     for p: (cacheinvalid(); fence; notify)

wait; read
  -> wait; cacheinvalid(); read
```

fence 为何落在写而非 notify：ptoas 以**词法且局部**的方式把 fence 与释放标记匹配。若某
notify 的数据由*另一个*循环中的 `cacheinvalid` 释放，则放在该 notify 附近的 fence 会被
拒绝 —— fence 必须紧跟那条 `cacheinvalid`。在每个写处一起发射 `cacheinvalid; fence`，
无论其后的 notify 有多远（哪怕跨循环嵌套）都满足要求；而直线 `store; notify` 仍降为规范的
`store; cacheinvalid; fence; notify`。

区域 cacheinvalid 目前覆盖**整个目标张量**（以全 `0` offset 覆盖完整 shape），复用类型的
dim 表达式。收窄为精确写入子区域（写自身的 `(shapes, offsets)` 就在写入点旁边）是后续
升级项。

### 标记落在写 / wait / notify 所在处（必在作用域内）

把区域 cacheinvalid 紧插在其写之后，目标 `Var` 天然在作用域内（写刚用过它）—— 无论它是
window 参数、别名（`dv = pl.tensor.view(win); remote_store(dv)`）、循环携带的 `iter_arg`，
还是分支内定义的值。**不需要任何跨作用域跟踪，也绝不会 silent drop**：每条标记都落在需要
它的 op 旁边，任何嵌套层级皆然。裸单语句的分支/循环体会就地包裹
（`body -> { markers; body; markers }`）；首次运行后该体成为 `SeqStmts`，故本 pass 幂等。

## 在流水线中的位置

```text
... -> MaterializeRuntimeScopes -> ClassifyIterArgCarry -> InsertCommFence   (最后)
```

它在 Default 流水线中**最后**运行，位于所有会重排语句的 pass
（`SkewCrossCorePipeline`、`LowerPipelineLoops`、`CanonicalizeIOOrder` ...）之后。插入的
op 无操作数、无依赖边，若更早插入可能被挪离其 notify/wait；放最后可让它们在 codegen 前
保持相邻。它之前的 pass 只改动 Orchestration 函数，因此本 pass 看到的 InCore IR 正是
codegen 最终降级的 IR。

## 什么算发布写

由 `op_predicates::IsPublishingWrite` 判定：

| 情形 | 条件 |
| ---- | ---- |
| `pld.tile.remote_store` / `pld.tile.put` / `pld.tensor.put` | 远程写（始终） |
| `tile.store` | 目标张量（arg 2）为 window-bound 的 `DistributedTensorType`（peer 可 `remote_load`） |
| `pld.tile.get` / `pld.tensor.get` | 本地目标（arg 0）为 window-bound |
| 对未注册 op 名的任意调用 | 保守 —— 未做过程间分析的用户函数 |
| `Submit`（任务启动） | 保守 —— 排序上视作写 |

`remote_load`（结果是 tile、不写 GM）以及写入普通 `Tensor` 的 `tile.store` **不是**发布
写。普通张量 store 因此不发区域 cacheinvalid，但其后的 notify 仍是*裸 barrier notify*，
会得到全 GM cacheinvalid + fence。

## 算法 —— 一趟前向遍历 + `pending` 布尔

一个 `pending` 布尔按执行序流动，决定 notify 的标记：

- **发布写**置 `pending = true`（它已发射自己的 `cacheinvalid; fence`）；
- **notify** 或 **fence** 清 `pending = false`；
- **wait** / 其它保持不变。

在每个 notify 处：若 `pending` 为真，则已有一条已 fence 的写在其前，不再补任何标记；若
`pending` 为假则该 notify 是*裸 barrier notify*，补一条全 GM `cacheinvalid; fence`。

`pending` 对控制流**保守**，使不确定的路径发射安全的全 GM 标记，而非依赖可能并不先行的
区域标记：

- **`if`** —— 每个分支以流入的 `pending` 标记；`if` 之后 `pending` 取两分支结果的**与**
  （只在一条路径上的写不算证明）。故无已证明写待处理的条件 notify 在其分支内部补全 GM 标记。
- **循环** —— body 以 `pending = false` 进入，循环之后 `pending` 也重置为假。这是**正确性**
  要求而非仅保守：ptoas 以**词法**方式检查释放标记，故循环头部的 notify 不能依赖来自循环前
  （迭代 0）或上一轮迭代尾部写（回边）的 fence —— 二者在 body 中都不词法先行。清零 `pending`
  强制每个循环体内的 notify 各自补标记。

```text
remote_store; notify                 -> remote_store; cacheinvalid; fence; notify
remote_store; for: notify             -> remote_store; cacheinvalid; fence;
                                         for: { cacheinvalid(); fence; notify }
for: { notify; store }                -> for: { cacheinvalid(); fence; notify;
                                                store; cacheinvalid; fence }
```

写之后**紧跟一条 fence** 的已存在区域 `cacheinvalid`、以及紧接 wait 之后已存在的全 GM
cacheinvalid，都会被识别且**不重复插入**，故本 pass 幂等。

## 与 Codegen 的关系

`InsertCommFence` 取代了 `MakePutCodegenPTO` 中原先在每个 TPUT 之后无条件发射的 drain
屏障（PTOAS#872 的 workaround）。该 codegen 屏障被移除：fence 现在仅在确有 notify 跟随时
才发射，并提供裸 pipe 屏障无法给出的 DDR 可见性 drain。TPUT/TGET 的**前置**屏障与 TGET 的
**尾部**屏障与此无关（核内 RAW 排序，而非 data-before-signal），保留不动。

## 消费者

流水线下游无消费者。PTO codegen 通过既有的 op handler 降级插入的 `system.cacheinvalid`
（按参数个数选区域或全 GM）与 `system.fence`；没有其它 pass 需要理解它们。
