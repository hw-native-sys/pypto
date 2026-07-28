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

在 ptoas 0.50 上实测，该契约可归结为**两条纯局部规则** —— *notify* 本身无需任何标记。
一趟结构遍历（`InsertCommMarkers`），不带任何控制流状态，插入：

- **每个本地发布写之后** —— window-bound `tile.store`，或写入本地目标的 `get`：一条覆盖
  整张量的**区域** `system.cacheinvalid`，**紧跟一条 `system.fence`**；
- **每个远端发布写之后** —— `remote_store` / `put`：只插一条 `system.fence`（其 peer 区域
  cacheinvalid 由 codegen 发,见下）；
- **每个不透明发布写之后** —— 一个 `Submit`，或对未注册用户函数（其函数体不在本 pass 内分析,
  没有单一可寻址区域）的调用：保守地插一条**全 GM** `system.cacheinvalid()` + `system.fence`；
- **每个 wait 之后** —— 一条**全 GM** `system.cacheinvalid`（消费侧在下一次可缓存读之前
  的失效）；
- **notify** —— 什么都不插。

区域 `system.cacheinvalid(target)` 寻址的是 `target` 的**本地** base，这对本地窗口写是对的。
但**远端写** `remote_store` / `put` 写到的是 **peer 偏移** GM 地址（`local_ptr +
delems(peer)`），本地 target view 寻址不到。peer 偏移只有在 codegen 里才知道
（`EmitCommRemoteView`）、**目前无法在 IR 上表示**，所以那条 peer 区域
`pto.cmo.cacheinvalid <peer_view> single_cache_line` 是 codegen 发的**规避手段**;而配对的 GM
释放 **fence** 始终由本 pass 插一条显式 `system.fence`（codegen 不得嵌入 fence），让释放排序在
IR 上可见、与本地写路径统一。(后续:给 peer 区域 cacheinvalid 一个 IR 表达,让 pass 拥有完整标记。)

```text
store(win_a); store(win_b); notify        （本地窗口写）
  -> store(win_a); cacheinvalid(win_a); fence; store(win_b); cacheinvalid(win_b); fence; notify

for c: store; for p: notify                 （写与 notify 在不同循环）
  -> for c: (store; cacheinvalid; fence);
     for p: notify

wait; read
  -> wait; cacheinvalid(); read
```

notify 为何无需标记、fence 为何落在写：ptoas 把所需的释放 fence 关联到发布写的
`cacheinvalid`，而非 notify。因此发布「先前（哪怕在*另一个*循环里）写入」数据的 `tnotify`
已由该写的 `cacheinvalid; fence` 满足 —— fence **不必**紧挨 notify。纯 barrier notify
（完全无数据）什么都不需要。（此结论经实测验证：从 ring-allreduce 的 `.pto` 删掉 notify 侧
标记后 ptoas 0.50 仍接受；而删掉 wait 侧的 `cacheinvalid all` 则被拒绝。）

区域 cacheinvalid 目前覆盖**整个目标张量**（以全 `0` offset 覆盖完整 shape），复用类型的
dim 表达式。收窄为精确写入子区域（写自身的 `(shapes, offsets)` 就在写入点旁边）是后续
升级项。

### 标记落在写 / wait / notify 所在处（必在作用域内）

把区域 cacheinvalid 紧插在其写之后，目标 `Var` 天然在作用域内（写刚用过它）—— 无论它是
window 参数、别名（`dv = pl.tensor.view(win); remote_store(dv)`）、循环携带的 `iter_arg`，
还是分支内定义的值。**不需要任何跨作用域跟踪，也绝不会 silent drop**：每条标记都落在需要
它的 op 旁边，任何嵌套层级皆然。裸单语句的分支/循环体会就地包裹
（`body -> { body; markers }`）；首次运行后该体成为 `SeqStmts`，故本 pass 幂等。

## 在流水线中的位置

```text
... -> MaterializeRuntimeScopes -> ClassifyIterArgCarry -> InsertCommFence   (最后)
```

它在 Default 流水线中**最后**运行，位于所有会重排语句的 pass
（`SkewCrossCorePipeline`、`LowerPipelineLoops`、`CanonicalizeIOOrder` ...）之后。插入的
op 无操作数、无依赖边，若更早插入可能被挪离其 notify/wait；放最后可让它们在 codegen 前
保持相邻。它之前的 pass 只改动 Orchestration 函数，因此本 pass 看到的 InCore IR 正是
codegen 最终降级的 IR。

## 本 pass 标记哪些写

| 情形 | cacheinvalid | fence |
| ---- | ------------ | ----- |
| 写入 window-bound `DistributedTensorType` 的 `tile.store`（peer 可 `remote_load`） | pass（本地区域,dst arg 2） | pass |
| `pld.tile.get` / `pld.tensor.get`（读 peer 到本地目标） | pass（本地区域,dst arg 0） | pass |
| `pld.tile.remote_store` / `pld.tile.put` / `pld.tensor.put`（peer 偏移写） | **codegen**（peer 区域,IR 规避） | **pass** |

`remote_load`（结果是 tile、不写 GM）以及写入普通 `Tensor` 的 `tile.store` **不是**发布
写 —— 完全不插标记。

## 算法 —— 一趟结构遍历，无流状态

本 pass **不带**任何控制流状态（无 `pending` 布尔、无 `if`/循环分析、无 notify 分类）：

- 每个**本地发布写**处追加 `region cacheinvalid; fence`；
- 每个**远端发布写**（`remote_store` / `put`）处只追加 `fence`（peer 区域 cacheinvalid 由 codegen 发,见下）；
- 每个 **wait** 处追加 `cacheinvalid()`；
- **notify** 保持不动。

`if`/`for`/`while` 的 body 正常递归访问；唯一的特殊处理是包裹裸单语句 body（作为 `if`/`for`
唯一 body、且无外层 `SeqStmts` 的写/wait），使其标记也能落上。由于两条规则都是局部且只追加，
控制流无关紧要：某个循环内的写会被正确标记，无论其 notify 是否在另一个循环。

```text
store(win); notify                   -> store(win); cacheinvalid(win); fence; notify
store(win); for: notify               -> store(win); cacheinvalid(win); fence; for: notify
for: { notify; store(win) }           -> for: { notify; store(win); cacheinvalid(win); fence }
```

写之后**紧跟一条 fence** 的已存在区域 `cacheinvalid`、以及紧接 wait 之后已存在的全 GM
cacheinvalid，都会被识别且**不重复插入**，故本 pass 幂等。

## 与 Codegen 的关系

`remote_store` 与 `put` 的 codegen 各自在 store 之后发一条 peer 区域
`pto.cmo.cacheinvalid <peer_view>`（peer 偏移只有在这里才知道、且**目前无法在 IR 上表示** ——
一种**规避手段**;配对的 GM fence 由本 pass 以 `system.fence` op 提供,所以这里**不**发 fence）。
`put` 额外在 TPUT 与其 cacheinvalid **之间**保留一条尾部 `pto.barrier <PIPE_ALL>`：TPUT 是 DMA，
GM fence 只排序内存、并不 drain 发起 DMA 的 MTE 管线，缺这条 barrier 时后面的 notify 可能在
（原子）TPUT 真正落到 peer 之前就发出 —— `test_l3_put` 的 atomic-add / 子区域用例在真机上会
flaky；且实测 MTE3 级 barrier 不够(只有 `PIPE_ALL` 稳)。PTOAS#872 的 workaround,待 PTOAS 自行
drain tput 后移除。TPUT/TGET 的**前置**屏障与 TGET 的**尾部**屏障同样保留不动。

**后续:** 远端写的 peer 区域 cacheinvalid 是唯一还留在 codegen(而非 IR op)的标记,因为 peer
地址(`local_ptr + delems(peer)`)暂无 IR 表达。有了一等的 IR 表达后,InsertCommFence 就能像本地写
那样拥有远端写的完整标记(cacheinvalid + fence)。

## 消费者

流水线下游无消费者。PTO codegen 通过既有的 op handler 降级插入的 `system.cacheinvalid`
（按参数个数选区域或全 GM）与 `system.fence`；没有其它 pass 需要理解它们。
