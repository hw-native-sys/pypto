# 调度与片上内存规划联合优化

## 状态

研究方向。本文记录 DSA adapter 暴露出的优化边界；它不是当前 planner 的实现契约。

## 动机

Buffer 生命周期由合法调度推导。移动 load、compute 或 store 会改变定义点和最后使用点，
因此也可能改变最佳放置。例如，下面两种调度满足相同的数据依赖：

```text
偏重重叠                         偏重内存
load A                           load A
load B                           compute A
compute A                        load B
compute B                        compute B
```

第一种调度暴露 load/compute 重叠，但让 `A` 和 `B` 同时存活；第二种调度可能让它们复用
同一地址范围，却降低重叠。固定生命周期 DSA 只能看到一个选择的结果，而联合 optimizer
可以直接评估这一权衡。

一个有用的形式是：

```text
minimize latency + synchronization cost + spill cost
subject to per-pool capacity, data dependencies, aliasing, and hardware rules
```

Peak memory 也可作为 hard constraint。变量可以包括合法拓扑序、pipeline depth 和
residue mapping、同步以及 buffer offset。

## 当前边界

PyPTO 目前先固定调度，再导出 DSA：

```text
SkewCrossCorePipeline
  -> LowerPipelineLoops
  -> CanonicalizeIOOrder
  -> InitMemRef
  -> MaterializeSemanticAliases
  -> lifetime analysis and DSA export
```

`CanonicalizeIOOrder` 在同核 pipeline loop 内执行保持依赖的优先级拓扑排序；随后
`ComputeLifetimes` 根据 statement position 推导 interval。独立 solver 只为这个固定调度
选择复用关系和 offset。`pipeline_groups`、alias class 和 separation 保留普通 interval
无法表达的事实。

PTOAS 通常消费 frontend 给出的顺序。其 memory planner 将最终 MLIR 线性化，并利用
MLIR liveness 推导 gen/kill point。PTOAS 有一个针对已接受 tile-fusion group 的
block-local `OpScheduling` pass，但目前没有覆盖所有 kernel 的通用 scheduler。自动
sync/event 插入与最终 pipe 行为还会提供更低层的执行事实。

## 可选的归属模型

两个 compiler 都能实现联合优化，但各自暴露的搜索空间不同。

| 归属 | 可改变内容 | 最强信息 | 主要限制 |
| ---- | ---------- | -------- | -------- |
| PyPTO | Tiling、pipeline 构造/depth、cross-core skew、operation order、复用和 offset | 源语义、loop、alias、高层候选 | 必须估计最终 pipe、event 和 instruction cost |
| PTOAS | 降低后的 instruction order、pipe 同步、multi-buffer slot、复用和 offset | 具体 PTO operation、pipe、event、backend legality | 无法恢复 lowering 已删除的高层选择 |
| 跨层 | PyPTO 调度候选加 PTOAS placement/backend cost | 语义结构与 backend truth | 需要稳定协议，并可能需要迭代编译 |

必须遵守一个归属不变量：任何在生命周期推导后改变调度的组件，都必须重新计算或重新验证
placement。PyPTO 不能先分配重叠地址，再允许 PTOAS 自由重排这些 use。由 PyPTO 负责时，
后续 PTOAS transform 必须保持已经证明的生命周期关系；由 PTOAS 负责时，PyPTO 必须跳过
local-address assignment，让 PTOAS 一起完成调度和放置。

### PyPTO 负责

当搜索会改变 loop transform、software-pipeline depth、tiling 或 AIC/AIV 结构时，PyPTO
是自然归属。联合 problem 应在 `CanonicalizeIOOrder` 固定单一顺序前建立。选定调度后，
再物化调度、重新计算生命周期、验证 placement，并通过 DSA 路径发出显式地址。

这种方式不要求修改 PTOAS planner，但需要校准 pipe overlap、同步和 latency 模型。
PTOAS 仍是 legality 与 device-validation 边界。

### PTOAS 负责

PTOAS 能针对固定的 lowered kernel 联合优化调度和内存。它需要一个与 `PlanMemory`
耦合的通用合法 operation scheduler，而不是在单一固定顺序后才推导 liveness。这适合
由 PTOAS 拥有 planner 的模式，此时 PyPTO 不分配 local address。

该边界适合 instruction-level 和 event-aware 决策。但如果 PyPTO 没有把 tile shape、
pipeline 构造或 cross-core decomposition 保留为候选或 metadata，PTOAS 就不能重新考虑
这些选择。

### 跨层联合优化

最完整的设计是分阶段或迭代协议：

1. PyPTO 导出 dependency DAG、合法调度候选、alias、pipeline 结构、pool 和 size，而不只
   导出固定 interval。
2. 联合 solver 提议 schedule 与 placement，或由 PyPTO 枚举一个较小的 Pareto 候选集。
3. PTOAS 评估具体 pipe/event legality 与 cost。
4. Feedback 拒绝候选或重新评分；最终选择经过独立验证和 device 测量。

可移植 benchmark 可以保留固定调度 profile。单独的 PyPTO/PTOAS profile 可以记录更丰富
的实例和 backend feedback，因此独立 benchmark 不必在构建时依赖 PTOAS。

## 推荐研究阶段

1. 保留固定调度 DSA，作为可复现 baseline。
2. 导出调度前 dependency DAG 和足够 metadata，以便重放 `CanonicalizeIOOrder` 候选，但
   暂不改变 solver 行为。
3. 为 `pypto-structured-search` 增加有界 schedule move：ready-node swap、load/store
   motion、pipeline-depth change 和 placement repair。
4. 每次 move 后重新计算生命周期，并独立验证 schedule 与 placement。
5. 将预测的 peak、同步和 latency 与 PTOAS 输出及 device trace 比较。
6. 最后再决定 production heuristic 应位于 PyPTO、PTOAS，还是由二者分担。

可能的 production 划分是分层的：PyPTO 选择高层调度结构，PTOAS 细化低层
instruction/event scheduling，每个 memory planner 在自己的抽象层消费调度。单体联合
solver 对研究有价值，但最终 architecture 不一定需要它。

## 开放问题

- 哪些 operation 可以跨 pipeline stage 或 control-flow boundary 移动？
- Capacity 应作为 hard constraint、latency 作为 objective，还是 benchmark 应暴露
  Pareto frontier？
- PyPTO 能以多高可靠度预测 PTOAS pipe/event 信息？
- 能否增量评估 schedule move，而不重建全部 lifetime 和 conflict？
- 哪些高层候选必须在 lowering 后保留，PTOAS 才能使用？
