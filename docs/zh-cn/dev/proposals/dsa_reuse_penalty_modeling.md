# DSA 复用惩罚建模

## 状态

PyPTO 的复用惩罚 recognizer 仍是默认关闭的实验功能。必须分开理解：

1. 稳定的 DSA-RP 优化问题；
2. PyPTO 当前实现的 recognizer 与 promotion policy；
3. 用于判断哪些 candidate 应获得正权重的实验依据。

现有证据支持继续使用简单的非负优化模型，但不支持在生产环境中启用当前 promotion
policy。

## 稳定优化问题

每个物理内存空间都是独立的标准 DSA arena。容量和 correctness constraint 是硬约束。
稀疏 soft edge `e = (buffer_i, buffer_j, weight_e)` 在两个生命周期可复用 buffer 的
物理 byte range 重叠时激活：

```text
reuse_cost(p) =
    sum(weight_e for active overlapping edges e)
```

planner 在容量内寻找合法 placement，最小化 `reuse_cost`，并可用 peak 作为最终
tie-break。权重保持非负。没有 edge 或权重为零表示 compiler 没有性能损失证据，并不
表示 overlap 一定免费。

优化问题只包含 buffer pair 与 weight。candidate recognition、PTOAS 同步行为和权重
估计是 producer-side 建模问题。pipeline intent 的 hard constraint 及显式 soft
fallback 与本文的 access-hazard recognizer 相互独立。

## 当前 recognizer 实现

`DsaReusePenaltyRecognizer.QUADRATIC` 是唯一启用的研究模式。它默认关闭，并以覆盖率
优先。

### Candidate 生成

recognizer：

1. 把 semantic alias 规范化为物理 allocation identity；
2. 收集执行期读写，包括 tuple result、mutating operation、base allocation 和已知
   byte range；
3. 将 access 映射到抽象 source/destination route 与执行资源；
4. 为每个 allocation 构造 terminal-access 和 initial-write frontier；
5. 扫描一个 address space 内所有 lifetime-compatible buffer pair；
6. 记录 distance-zero 与 distance-one WAR/WAW handoff，并保留 route、range、loop、
   control-flow 和 ordering provenance。

raw record 位于 `metadata.recognized_reuse_candidate_records_v4`。SSA 可达记录带有
确定性的 region/statement `dag_path`，无序记录使用 `dag_path=none`。可用：

```bash
python -m pypto.tools.dsa_reuse_candidates PROBLEM.dsa.json
```

解析这些 policy 过滤前的记录。

### 当前 v5 edge 构造与权重 policy

当前 `cross_resource_completion_pair_v5` policy 在一个 pair 至少存在一条满足以下
条件的记录时构造 pair edge：

- cross-resource；
- full-allocation 且 access 集合完整；
- 不依赖 conservative initial anchor；
- 不是 same-operation alias-contract 问题。

SSA-ordered 与 loop-carried 记录仍然 eligible。设备实验否定了“SSA reachability
能够证明 completion”这一假设，并发现了高代价的 distance-one handoff。
same-resource、partial-view 和不确定记录仍只用于报告。同一 buffer pair 的多条
qualifying record 只构造一条 edge。`unit_v1` 为每条 edge 赋值 `1`。

该 policy 形成 additive、non-negative 的 `cross_pipe` 潜在同步义务模型，但尚不
判断同步义务是否扩展 consumer 的有效 completion frontier，或是否暴露在 critical
path 上。metadata 用
`reuse_penalty_completion_exposure_model=unmodeled_v1` 明确记录此限制。
recognizer 仍为默认关闭的实验功能。

当前实现会在构造 per-allocation access frontier 时使用 same-resource issue order
与 SSA reachability。same-resource ordering 是抽象 completion chain 假设；对于
cross-resource candidate，SSA reachability 仅作为 provenance 导出，不再作为
suppression rule。

## 证据与已否定规则

受控 placement 保持 DSA problem 与生成操作不变，只改变指定物理 overlap。现有结果
表明：

- 大部分合法 reuse 不改变同步；
- synchronization-group count 不能预测 latency；
- route class 不足以分类，同一路由可能 harmful、neutral 或被其他 release 覆盖；
- loop frequency 会放大暴露在 hot path 上的 handoff，但不能让已覆盖 handoff 变贵；
- 同一 consumer 的多个 predecessor 更接近“最新 predecessor”而不是简单求和；
- 尚无实验支持负优化权重；同步减少没有产生可重复的 latency 收益。

最新 exact ordered-pair study 给出了对旧 v4 policy 的两个重要反例：

| Pair 类型 | 结果 |
| --- | --- |
| SSA-ordered `V -> MTE2` WAR | overlap 增加一个匹配的 PTOAS handoff |
| unordered `M -> MTE1` WAR | overlap 删除冗余 handoff，但没有确认 latency 收益 |
| 另外四个匹配 pair | 同步不变 |

因此 v5：

- 将 SSA `dag_path` 保留为 provenance，而不是 suppression predicate；
- 纳入 distance-one candidate，而不是抑制所有 loop-carried handoff；
- 仍把所有已构造 edge 当作尚未校准的 unit obligation。

实验也说明，给所有 cross-resource obligation 相同的正性能权重仍然过于宽泛：

- 不能给所有改变同步的 pair 正权重；
- 非负 pair model 仍然成立：neutral 或表面 beneficial 的 pair 可以不产生正 edge。
- 多个 whole-kernel RP placement 在 UB 和 L1 case 上都能稳定提速，但插入的
  synchronization-group 数量无法对这些收益排序；
- clean pair ablation 可以重现明显 latency 变化，但其 synchronization summary
  delta 方向可能相反，因此 candidate recognition 目前领先于 mechanism attribution
  与 weight calibration；
- pair 的效果可能依赖周围 placement。pair isolation 必须保持 capacity，并统计构造
  过程中改变的每一条 overlap relation。

## Completion-frontier conjecture

必须区分三类 ordering：

| 层次 | 问题 |
| --- | --- |
| Logical ordering | 一个 PyPTO value-producing operation 是否可达另一个？ |
| Completion ordering | overwrite 前，之前的异步 access 是否已经释放复用 byte？ |
| Exposed delay | release point 延后是否改变 initiation interval 或 kernel latency？ |

对于生命周期可复用的不同逻辑 buffer `A` 与 `B`，令 `u` 为 `A` 被复用 subrange 的
最后相关 access，`v` 为 `B` 对同一 subrange 的首次 overwrite。reuse 会产生候选
物理 WAR/WAW handoff `u -> v`。

只有已有 **completion-carrying path** 能保证 `u` 在 `v` 前完成时，下一版 policy
才应 suppress candidate。例如显式 event/barrier/token，或 target 明确保证的 FIFO
completion；普通 SSA reachability 不够。v5 constructor 因此保留这类 candidate，
但其 unit weight 只是实验 surrogate，不是校准后的 latency claim。

只有当 `u` 扩展 `v` 已有的 completion-release frontier 时，才有理由产生正 edge。
定性 cost conjecture 是：

```text
dynamic frequency
    * max(0, completion(u) - latest unavoidable release of v)
```

这只是建模指南，并非 PyPTO 当前可计算的 cycle estimator。当前 v5 unit model
刻意停在这一步之前：它识别稀疏 pair obligation，但不把它们称为 production
performance cost。未来经校准的 producer 应默认使用零权重，并只给具有重复 harmful
证据的机制正权重。对于同一 consumer 的多个 candidate predecessor，应保留有依据的
dominant pair，而不是重复求和。OR group、hyperedge、负权重和全局 event-budget
term 继续推迟。

## 剩余验证

completion-frontier factorial 已经完成。它确认同一 consumer 的多个 active
predecessor 可能合并到一个 release frontier，但隔离出的 frontier extension 是
latency-neutral，因为已有 drain 已使该 resource 静止。因此实验支持
consumer-aware deduplication，但不支持为它赋予正权重。

下一项研究应从具有稳定 RP-versus-compact latency 差异的 kernel 反向分析：

1. 找出 endpoint placement 改变的 candidate pair；
2. 构造 exact-XOR single-pair 与小型 factorial placement，优先采用保持 capacity
   的 address exchange；
3. 在通过 PTOAS 编译前冻结 mechanism 预测；
4. 比较完整 synchronized instruction topology 与 predecessor identity，而不是只看
   group count；
5. 使用真实 kernel input 与 scalar 验证所有 written output；
6. 在两个 device 上测量 kernel-only latency，只有初始 confidence interval 有信息量
   时才增加 sample。

直接目标是已经确认 endpoint speedup、但 pair-level mechanism attribution 仍不完整
的 UB 与 L1 kernel。只有一个 mechanism 能跨 fresh kernel 和 placement background
预测效果方向，才应获得正权重。checked-in v5 unit model 可以生成 algorithm-study
instance，但在完成该 calibration 前仍不支持 production promotion。
