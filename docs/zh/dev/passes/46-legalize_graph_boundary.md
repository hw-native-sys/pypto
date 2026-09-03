# LegalizeGraphBoundary Pass

让每个 `FunctionType::Graph` 函数都能被 `host_build_graph` runtime 合法地录制与
回放：把 Graph 函数体内派生出来的边界标量、取出的边界视图、以及自己分配的中间
张量都外提到调用点，并拒绝那些 runtime 不会缓存的边界形态。

## 概述

`host_build_graph` runtime 在第一次调用时录制 Graph 函数的任务拓扑，之后回放这份
录制。回放只 patch 两样东西：边界张量的地址，以及边界标量的值。其余一切 —— 节点
数、形状、依赖边、block 数 —— 都被烙进录制下来的 Definition。

由此产生四类问题，而且在 runtime 侧都是静默的：

| 问题 | runtime 的行为 | 本 pass 的处理 |
| ---- | -------------- | -------------- |
| 边界标量在区域内被**派生**出来 | 归类为静态数据，把第一次调用的值冻进录制。永远不告警。 | **Step A** —— 把计算外提到调用点；当冻结下来的值可证明就是正确值时，原地保留 |
| 边界张量的视图在区域**内部**取出 | 冻结第一次调用的偏移，只 patch 地址，于是后续调用读到的是第一次调用的窗口 | **Step B** —— 把视图外提到调用点；当其窗口对回放不变时，原地保留 |
| 区域自己分配中间张量 | 能正确录制，但分配在一块运行期间从不回收的堆上，于是活跃集随提交次数增长 | **Step C** —— 把分配外提到调用点，成为 `InOut` 边界张量 |
| 边界本身不可缓存 | 拒绝缓存，静默地按普通任务执行该区域 | **Step D** —— 编译期拒绝 |

前两类产生错误结果。第三类结果正确，但随着层数增长会耗尽内存、或者只是变慢。最
后一类结果正确但完全没有预期的加速 —— 任何数值测试都看不见，这正是这些检查放在
这里、而不是交给一条 runtime 日志的原因。

## Step A —— 派生的边界标量

边界标量是靠**指针身份**追踪的。录制时 runtime 锚定每个 `args.scalar(k)` 槽位的
地址，回放时重新读这些地址。函数体自己算出来的值没有槽位：

```python
@pl.function(type=pl.FunctionType.Graph)
def layer(self, cur, wq, layer_idx: pl.Scalar[pl.INDEX]):
    base = layer_idx * 5120          # <- 派生值：没有实参槽位
    ...                              #    被冻结在第一次调用的取值上
```

Step A 把它改写成以形参形式传入：

```python
# pass 之后，概念上是：
def layer(self, cur, wq, layer_idx, base):   # base 成为真正的边界标量
    ...

# 每个调用点：
self.layer(cur, wq_view(i), i, i * 5120)     # 算术搬到了这里
```

一个值可外提的条件是：它整棵表达式树的叶子只有该 Graph 自己的标量形参和常量 ——
这恰好是调用点能够重算的集合，因为调用点本来就提供这些形参。PyPTO 里的标量算术
是 `BinaryExpr` / `UnaryExpr` 节点而非 `Call`，所以判定沿这两个基类递归，其余节点
一律当作叶子。

新形参是**追加**而不是前置的：`CoreTaskArgs` 要求所有张量实参排在所有标量实参之前。

### 纯重命名会被删除，而不是被接受

`n = batch` 什么也没算，所以没有东西可外提 —— 但把它留在原地，恰恰会把 Step A
本要防的那个 bug 重新引入。codegen 会把幸存的别名发射成一次**值拷贝**：

```cpp
const uint64_t& batch = args.scalar(0);   // 槽位，按引用
int64_t n = batch;                        // 一份拷贝，地址不同
g0_params_t0.add_scalar(n);               // 任务收到的是这份拷贝
```

录制是按*取值来源的地址*给标量分类的，比对的是 `&boundary_args->scalar(i)`。
这份拷贝匹配不上任何槽位，于是被记为 `STATIC_VALUE`，后续每次回放都复用第一次
调用的数值。

因此 Step A 会把这个名字替换掉并删除绑定，让任务读到 `add_scalar(batch)` ——
槽位本身。链式别名一趟就收敛到根（`a = p; b = a;` 把两个读者都指向 `p`），而
指向*已外提*值的别名会落到该值的新形参上。若仍有重命名幸存，Step D 与 verifier
都会拒绝它，而不是放行。

### 可外提（hoistable）与回放不变（replay-invariant）

不可外提并不等于非法。可外提回答的是*「调用点能重算它吗？」*；而 runtime 真正需要的
是*「它每次调用都一样吗？」*，后者严格更弱。把一个值冻进录制，只有在这个值**会**
逐次变化时才是错误答案。

| 性质 | 提出的问题 | 结果 |
| ---- | ---------- | ---- |
| 可外提 | 调用点能重算它吗？ | Step A 把它搬出去，它获得真正的实参槽位 |
| 回放不变 | 它每次调用都一样吗？ | 原地保留、被冻结 —— 而且冻对了 |

`ReplayInvariantSet`（`utils/graph_replay_invariant.h`）划出后一条线。三个种子，
并在标量算术以及「绑定到不变值的名字」上取闭包：

| 种子 | 回放为何能复现它 |
| ---- | ---------------- |
| 字面量 | 平凡成立 |
| **常量迭代次数**循环的归纳变量 | 录制会把循环走一遍，把每次迭代的字面量烙进那次迭代自己的节点；边界是常量，意味着后续每次调用都走出完全相同的序列 |
| 对边界张量形参取的 `tensor.dim`（轴为字面量） | `graph_boundary_matches` 会把每个边界张量的 `ndims`、`shapes`、`strides` 与录制下来的 `GraphBoundarySignature` 比对，任一不符即拒绝该缓存图，所以在同一份录制内边界形状不可能改变 |

**标量形参被刻意排除在不变集合之外。** runtime 每次调用都会 patch 边界标量的槽位 ——
这恰恰是它作为*任务实参*合法、而作为*被冻结的 view 偏移*非法的原因。

正是这一点才让分块 kernel 得以通过。分片偏移 `i * TILE` 无法外提：这个值在调用点
根本不存在。而 decoder layer 里的每个 projection、MLP 和 attention 循环都是这样索引
的，所以拒绝它等于把整类写法排除在外。

`DataType::TASK_ID` 操作数直接跳过，不做分类。task id 从来就不是边界标量 —— 录制
本身捕获的就是依赖结构，而 `graph_boundary_matches` 会拒绝任何携带显式依赖的调用
（`explicit_dep_count() != 0`），所以区域外产生的 id 根本到不了回放。这一点之所以
重要，是因为标量检查会扫描*每一个* Call 的实参：于是 `seeds[0] = seed` —— 一次写入
`TASK_ID` 型 `pl.array` 的 `array.update_element` —— 看起来就像一个任务在消费边界
标量，而同一个 id 直接写进 `deps=[...]` 却一直是被接受的。

如果一个标量流入了任务，却既不可外提、也不是边界形参、也不是回放不变的 —— 因为它
依赖任务输出或张量读取 —— 就会报错，消息里点名该变量。

## Step B —— 边界张量的派生切片

回放 patch 的是边界张量的**地址**。在区域**内部**取的 view 会从录制时冻结下来的
东西重新推导，所以必须改到调用点去取：

```python
wl = pl.tensor.slice(w, [128, 128], [layer_idx * 128, 0])   # 在区域内部
```

Step B 把这个切片搬出去，把结果作为一个新的边界张量传进来。每个切片点各自成为一个
形参、各自带固定形状 —— 这正是 runtime 的 `BOUNDARY_VIEW` 分类所要求的：它按
「同 buffer + 偏移」匹配，形状根本不参与，所以一个形状逐次变化的 view 压根无法被
分类。

外提出来的语句按**先标量、后张量**发射，因为切片的偏移通常就是 Step A 的标量，绑定
必须先于使用。而**形参**顺序恰好相反 —— 张量在前、标量在后 —— 这是 `CoreTaskArgs`
的要求。对区域局部张量取的 view 保持原样。

**对已外提 view 再取的 view 同样会被外提。** `wl` 搬出去之后它就是一个边界形参，于是
`wr = slice(wl, ...)` 所处的位置和当初的 `wl` 完全一样。函数体是按定义顺序的 SSA，所以
一次前向遍历就能走完整条链 —— 一个 view 只能引用在它之前定义的源。把 `wr` 留在原地是
静默的：`graph_rebind_tensor` 会用 `wl` patch buffer 地址，但保留第一次调用时录下的偏移。

**provenance 要穿过原地写回的调用。** `tmp = kernel(a, tmp)` 把同一块 buffer 重新
绑定到一个新的 SSA 名字上，而对*那个*名字取的 view 仍然是对边界张量取的 view。若只
跟踪 `alias = var` 这种裸别名，root 就在这里断了，Step B 会直接跳过该 view —— 既不外
提也不检查 —— 于是一个逐次变化的偏移留在区域内，冻结的是第一次调用的窗口。这里通过
[`ExplicitReturnedParamIndices`](26-normalize_return_order.md) 跟过去，也就是
orchestration codegen 给调用结果做别名时用的同一张「返回位 -> 形参」表，因此
provenance 与 codegen 不会对「这个结果指向哪块 buffer」产生分歧。这一点对边界*形参*
和 Step C 的分配同样适用。

该映射指向的实参由 `CallerSuppliedArg` 解析，它按 `ir/expr.h` 里 `Submit::args_` 契约
划分的三个区域来取。`Submit` 合法地可以省略由 runtime 分配的 `Out` 尾巴，而它调用方
提供的前缀仍然按位置一一对应，所以要求实参与形参数量完全相等，会让每一个这样的 launch
都静默丢掉 provenance。落在 runtime 分配区间的形参则正确地取不到实参：那块 buffer 由
runtime 创建，没有可继承的调用方 root。

**被外提的 view 形状必须是编译期常量。** 回放直接从录制模板里抄 view 的 `shapes` 和
`strides`，只 patch `buffer_addr` 和 `start_offset`，所以从边界标量读出来的 extent 会把
第一次调用的形状套到后续调用的 buffer 上。回放不变的 extent 会被接受，理由和别处一样：
它不可能变化，冻结它什么也改变不了。

**view 有三种结局，而不是两种。** 回放把 `BOUNDARY_VIEW` 还原为
`boundary.start_offset + packed_offset`，其中 `packed_offset` 是**第一次**调用时录下的
增量；被 patch 的只有地址、大小、owner 和 version。也就是说偏移是冻结的 —— 而这只在
偏移会移动时才是错的：

| 除源以外的操作数全都是 | 结局 | 原因 |
| ---------------------- | ---- | ---- |
| 可外提 | 外提到调用点 | 调用方能重建这个 view |
| 仅回放不变 | **原地保留** | 冻结即正确，而且调用点没有名字可指代它 |
| 两者皆非，或二者混合 | 拒绝 | 冻下来的增量会是第一次调用的 |

混合的那种才是关键。`off = layer_idx + i * TILE` 既不能外提（`i` 在调用点不存在），
也不能冻结（`layer_idx` 每次调用都被 patch）；而区域一旦同时用这两者索引权重，产生
的就是这种形状。建立在原地保留 view 之上的任何东西 —— 对它再取的 view，或它的别名
—— 都按同一条规则处理而不外提，因为调用点同样没有名字可指代它们。

**偏移逐次变化是安全的，但这一点并不显然。** codegen 会把运行时 view 钳制成
`min(declared, source.shapes[i] - offset[i])`，所以即使 IR extent 是常量，**实际**形状
也随偏移变化。它传不到 replay，是因为被外提的 view 是作为**自己独立的**边界张量传入的，
而不是在区域内重新推导：`graph_tensor_from_boundary` 会先对所有边界张量试
`BOUNDARY_EXACT`、再试 `BOUNDARY_VIEW`，消费该 view 的节点会命中 view 自身，于是
`graph_rebind_tensor` 整个替换 `GraphTensor`——`shapes`、`strides`、`extent_elem` 全含。
冻结形状是 `BOUNDARY_VIEW` 的行为，也就是本步骤要外提掉的"区域内取 view"那一类。
若要求偏移可证明在界内，则会把主用例（逐层的 `layer_idx * 5120`）一并拒掉。

## Step C —— 区域内的分配

区域顶层的 `pl.create_tensor` 会被**外提到调用点**，成为 `InOut` 边界张量：

```python
# 外提前
@pl.jit.graph
def layer(a: pl.Tensor, acc: pl.InOut[pl.Tensor]):
    tmp = pl.create_tensor([ROWS, COLS], pl.FP32)   # 每次提交都分配一次
    ...
    return acc

# 外提后 —— buffer 归调用点所有
def layer(a, acc, tmp: pl.InOut[pl.Tensor]): ...
tmp__graph_arg0 = pl.tensor.create([ROWS, COLS], pl.FP32)
acc = layer(a, acc, tmp__graph_arg0)
```

原地录制本身是**正确**的 —— codegen 把 create 降级成批量 `alloc_tensors`，runtime
把它记成一个无 kernel 的节点（和 `submit_dummy_task` 记录的形状相同）—— 但 buffer
来自 **graph 堆**，而 `task_allocator.h` 在整轮运行结束前从不回收它（"The whole
graph must fit at once; nothing is reclaimed mid-run"）。于是活跃集随提交次数增长
而不是保持恒定：一个持有 14 个中间张量的 decoder layer，录制 N 层就要占 14 × N 份。
外提之后 buffer 回到普通的可回收堆上，区域内也不再有任何分配节点。

simpler 手写的 `examples/a2a3/host_build_graph/qwen3_14b_decode` 场景就是手工这么
做的，理由也写在它的 README 里 —— "This keeps the temporary live set flat in layer
count and fits the default ring configuration."

形参是 `InOut`，绝不能是 `In`：区域会**写**这块 buffer；声明成 `In` 时 codegen 发射
`add_input`，这次 launch 永远不会被登记为该 buffer 的写者，调用方若把分配提出自己的
循环，前后两次 launch 之间就没有任何定序。`Out` 也不可用 —— 在 Graph 边界上它表示
"由 runtime 分配"，而 `rt_graph_args_cacheable` 会直接拒绝。

被外提的分配**就是**边界张量，所以对它取的视图同样要走上面 Step B 的规则，而不是原
地放过。这不是可选的整理：`GraphBoundaryLegalized` verifier 把每个张量形参都当作
boundary root，因此一个窗口可变、却被留在原地的视图，会让本 pass 产出连自己的
verifier 都拒绝的 IR。

有两类分配是刻意留在原地的：

| 留在原地 | 原因 |
| -------- | ---- |
| `tensor.full` | orchestration codegen 在调用点同样没有它的降级路径，外提只是把失败挪个地方。Step D 直接拒绝它 |
| 循环内的 create | 它是**每次迭代一份**的新 buffer。把 N 份塌缩成一个形参会让各迭代互相别名，而本该重新串行化它们的跨任务依赖边，是更早的 [`AutoDeriveTaskDependencies`](40-auto_derive_task_dependencies.md) 推导出来的 |

无论外提与否，录制都无法复现从边界标量读出来的 **shape**：extent 会被抄进节点、缓冲
区地址由它推出，而回放不会重新执行函数体，所以后续调用即使 extent 更大，拿到的仍是
第一次调用的 buffer —— 这是错误的地址布局，不是 fallback。Step D 在 Step C 之前就拒
绝这种写法，因此 Step C 外提的每个分配的 shape 都是编译期常量。

### 前置条件：Graph 的返回值必须直接指名形参

Step B 和 Step C 都会**追加** `InOut` 形参，而这正是自动化这次外提为什么是一次改造、
而不是接根线的原因。orchestration codegen 通过
`return_lineage::ExplicitReturnedParamIndices`（对被调方 `ReturnStmt` 的指针同一性读
取）把调用结果映射到被调方的某个 `Out`/`InOut` 形参上，只有当该映射给不出结果时，才
退回到"被调方唯一的那个 `Out`/`InOut` 形参"：

```cpp
// GenerateSingleReturnAlias, orchestration_codegen.cpp
INTERNAL_CHECK_SPAN(returned_idx.has_value() || out_indices.size() == 1, call->span_)
```

`OutlineIncoreScopes` 跑完之后，Graph 函数体是 `c_1 = layer_incore_0(a, c); return
c_1` —— 返回的是重绑定而不是形参本身，所以 Graph 一直依赖那条退路；而第二个 `InOut`
形参一出现，这条退路就不存在了。

外提所需要的东西由 [`NormalizeReturnOrder`](26-normalize_return_order.md) 提供：它会
像处理 kernel 与 wrapper 那样规范化 Graph 的张量返回值（**只**规范化，不重排 Graph 的
返回顺序），并且 `IRProperty::ReturnParamsExplicit` 覆盖了 `Graph`，使得从那里到这里的
十九个 pass 无法悄悄破坏它。这部分是在 #2618 中单独落地的。本步骤是**依赖**它而不是
提供它 —— 这也是单元测试里 `_legalize_outlined` 要跑 `NormalizeReturnOrder` 的原因：
不跑的话那张表全是 nullopt，外提会静默地什么都不做。

## Step D —— 边界合法性

| 检查 | 原因 |
| ---- | ---- |
| 编译目标必须是 `host_build_graph` | `GraphTaskArgs` 与 `rt_submit_graph` 只存在于该 runtime 的 orchestration API，而 codegen 无条件发射它们；因此在默认的 `tensormap_and_ringbuffer` 下编译 Graph，产物会引用未声明的符号。在这里报错、指向用户自己写的函数，而不是让它变成生成代码里的 C++ 编译错误 |
| 至少 1 个张量形参 | 空边界的 graph 回放时无处可 patch，runtime 拒绝缓存 |
| 至多 128 个张量形参 | `GRAPH_MAX_TENSOR_ARGS` —— 边界是定长的 `GraphTaskArgs` |
| 至多 64 个标量形参 | `GRAPH_MAX_SCALAR_ARGS`。在 Step A 之后检查：Step A 会**新增**标量形参，所以上提前放得下的签名，上提后可能放不下 |
| 不允许 `Out` 张量形参 | `Out` 意味着 runtime 分配该 buffer；被录制的 graph 其边界张量必须已存在，回放才能 patch 地址 |
| 标量形参必须是 `In` | 边界标量按值传入、由调用点回放 |
| 只能返回自己的形参 | `rt_submit_graph` 只在缓存**命中**时才返回有效 task id，所以任何东西都不能依赖 graph 调用的结果。`return c`（`c` 为 `InOut` 形参）是原地写的写法，可以；返回计算出的新值不行 |
| 被拉起的任务数在 1..1024 之间 | `graph_execution_storage_layout` 既拒绝 0 个节点，也拒绝超过 `GRAPH_MAX_NODES` 的。循环内的 launch 按迭代次数计入而非按词法调用点计 1；`system.task_dummy` 也计入——它 lower 成 `rt_submit_dummy_task`，且 `ExpandManualPhaseFence` 会自动插入。分配同样会记节点，按**上界**计——通过这项检查即意味着 runtime 会接受该 Graph。codegen 会收集一个语句列表里所有符合条件的 create（中间夹着 launch 不会打断批次），再按每次 `alloc_tensors` 最多 `kAllocTensorsArgs`（16）个打包。它的三条不合格规则里有两条在这里不可能触发（shape 读局部变量已被按非常量拒绝；SSA 下不可能出现已声明的 var），这些 create 是**精确**计数的。第三条可能触发——被注入的 GM pipe buffer 在其 `core_num` 读到 body-local 时会离开共享批次，而这只有 emitter 的 use-resolution 才知道——所以这类按最坏情况各计 1 个节点。批量大小和 GM-pipe 判定与 emitter 共用 `utils/alloc_batching.h`，而非各自重述。与本表其他检查不同，这一项在外提步骤**之后**才计数：Step C 会*移除*分配节点，按外提前的函数体计会拒掉本来放得下的 Graph，也会与 verifier 不一致——后者是从改写后的 IR 重新推导同一个计数的 |
| 运行时循环 / 分支内不得有分配 | 每次分配都记一个节点，所以数量随调用变化就是拓扑随调用变化 |
| 分配的 shape 必须是编译期常量 | 录制会把 shape 抄进节点并据此推出缓冲区地址；读取边界标量的 shape 会被冻结在首次调用的值上 |
| 区域内不得有 `tensor.full` | orchestration codegen 没有它的 lowering，会按 misplaced tensor op 拒绝 |
| 运行时循环 / `while` / `if` 内不得有 launch | 录制在首次调用时定死拓扑并原样回放，因此随调用变化的 launch 次数或分支会静默重放第一次的形状 |
| 任务实参里不得内联计算**逐次变化**的标量 | Step A 只上提**具名**的派生值；写在调用处的内联表达式没有名字可提、也没有边界 slot，会被冻结在首次调用的值上。回放不变的表达式会被接受 —— 冻结它是无害的 |
| Graph 不能调用 Graph | runtime 无法在正在录制的 graph 内部再录一个 graph |
| 调用点必须传满全部形参 | `Submit` 通常允许只传前缀、由 runtime 分配尾部 `Out` 形参；Graph 没有这种尾部 |
| launch 上不能带显式依赖 | 显式依赖边会让该次 launch 不可缓存，区域就会静默退化成普通任务 |
| launch 上不能带 dispatch predicate | graph launch 上的 predicate 既不被遵守也不被拒绝 —— runtime 静默清零它，于是区域会无条件执行 |

## 在流水线中的位置

跑在最后一个 `Simplify` 之后、
[`MaterializeRuntimeScopes`](47-materialize_runtime_scopes.md) 之前。

这个位置是两边夹出来的。`DeriveCallDirections` 和 `AutoDeriveTaskDependencies`
必须已经跑完，这样实参方向与跨任务边才是已知的；而 `MaterializeRuntimeScopes`
必须还没跑，这样 Step A 要搬动的语句外面还没有被套上 scope。

## Pass 属性

- **requires**：`SplitIncoreOrch`、`CallDirectionsResolved`
- **produces**：`GraphBoundaryLegalized`、`CallDirectionsResolved`

之所以重新声明 `CallDirectionsResolved`，是因为本 pass 改写了调用实参及其方向
attr；紧随其后的 `MaterializeRuntimeScopes` 要求该属性。

`GraphBoundaryLegalized` verifier 会独立地重新推导拓扑、节点数与外提后置条件，
这样后续 pass 若重新引入一个已被拒绝的状态就能被抓到。它与本 pass 只共享一样
东西：`ReplayInvariantSet`。那不是本 pass 做出的决定、由 verifier 盖章确认 ——
它是对 runtime 自身契约的一次解读，而两份手写副本可能彼此不一致，那会让
verifier 拒绝本 pass 刚刚产出的 IR。

## 尚未处理

外提**循环内**的分配 —— 它是每次迭代一份的新 buffer，调用点得分配一组而不是一个
（见 Step C）。外提 shape 读自边界标量的分配 —— Step D 直接拒绝，而不是在调用点重
建它。以及把超过 128 个张量的边界自动打包进 scratch arena。

也没有任何东西会把调用点的分配提出调用方自己的循环：create 落在 launch 的紧前面，
所以每次调用仍然拿到自己的 buffer。收益在于这块 buffer 现在来自普通的可回收堆而不
是 graph 堆，这与调用方把它放在哪里无关。

## 另见

- [Pass Manager](00-pass_manager.md) —— 完整流水线顺序
- [MaterializeRuntimeScopes](47-materialize_runtime_scopes.md) —— 紧随其后运行
