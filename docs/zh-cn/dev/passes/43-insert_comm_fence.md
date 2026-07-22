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
  整张量的**区域** `system.cacheinvalid`，**紧跟一条 `system.fence`**。ptoas 把释放 fence
  关联到该写的 cacheinvalid；任何随后发布该数据的 `tnotify` 都由它满足，因此两者都落在
  写入点（每地址一条）；
- **每个 wait 之后** —— 一条**全 GM** `system.cacheinvalid`（消费侧在下一次可缓存读之前
  的失效）；
- **notify** —— 什么都不插；
- **远端写** `remote_store` / `put` —— 此处不插（见下）。

区域 `system.cacheinvalid(target)` 寻址的是 `target` 的**本地** base，这对本地窗口写是对的。
但**远端写** `remote_store` / `put` 写到的是 **peer 偏移** GM 地址（`local_ptr +
delems(peer)`），本地 target view 寻址不到 —— 本地 target 的 cacheinvalid 会 clean 错的
cache line、漏掉 peer 数据。peer 偏移只有在 codegen 里才知道（`EmitCommRemoteView`），所以
本 pass 不管远端写，由它们的 codegen 在 store 之后自己发一条正确的 peer 区域
`pto.cmo.cacheinvalid <peer_view> single_cache_line` + GM fence。

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

本 pass 只标记**本地**发布写 —— 其写入 GM 地址就是本地 target view，故区域
`cacheinvalid(target)`（寻址本地 base）正确：

| 情形 | 由谁标记 | target arg |
| ---- | -------- | ---------- |
| 写入 window-bound `DistributedTensorType` 的 `tile.store`（peer 可 `remote_load`） | **pass** | dst（arg 2） |
| `pld.tile.get` / `pld.tensor.get`（读 peer 到本地目标） | **pass** | dst（arg 0） |
| `pld.tile.remote_store` / `pld.tile.put` / `pld.tensor.put`（peer 偏移写） | **codegen**（peer 区域 cacheinvalid + fence） | 不适用 |

`remote_load`（结果是 tile、不写 GM）以及写入普通 `Tensor` 的 `tile.store` **不是**发布
写 —— 完全不插标记。

## 算法 —— 一趟结构遍历，无流状态

两条规则都是纯局部的，因此本 pass **不带**任何控制流状态（无 `pending` 布尔、无 `if`/循环
分析、无 notify 分类）：

- 每个**本地发布写**处追加 `region cacheinvalid; fence`；
- 每个 **wait** 处追加 `cacheinvalid()`；
- **notify** 与**远端写**保持不动。

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

`MakePutCodegenPTO` 中原先在每个 TPUT 之后无条件发射的 drain 屏障（PTOAS#872 的
workaround）已移除。取而代之，`remote_store` 与 `put` 的 codegen 各自在 store 之后发一条
自己的 peer 区域 `pto.cmo.cacheinvalid` + GM `pto.fence.barrier_all`（peer 偏移只有在这里
才知道）—— 即远端写的 data-before-signal 释放标记，落在正确的 peer 地址上。TPUT/TGET 的
**前置**屏障与 TGET 的**尾部**屏障与此无关（核内 RAW 排序，而非 data-before-signal），
保留不动。

## 消费者

流水线下游无消费者。PTO codegen 通过既有的 op handler 降级插入的 `system.cacheinvalid`
（按参数个数选区域或全 GM）与 `system.fence`；没有其它 pass 需要理解它们。
