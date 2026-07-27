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

### 当前 v4 promotion 与权重 policy

当前 `cross_resource_pair_v4` policy 在一个 pair 至少存在一条满足以下条件的记录时
构造 pair edge：

- cross-resource；
- full-allocation 且 access 集合完整；
- 不依赖 conservative initial anchor；
- 不是 same-operation alias-contract 问题；
- distance zero；
- recognizer 的 SSA dependency graph 未将其标记为 ordered。

same-resource、loop-carried、partial-view、不确定和 SSA-ordered 记录仍只用于报告。
`unit_v1` 为每条已构造 edge 赋值 `1`，形成 additive、non-negative 的 `cross_pipe`
cost model。

当前实现还会在构造 access frontier 时使用 same-resource issue order 与 SSA
reachability。它们只是 completion ordering 的实验近似，不是硬件保证。

## 证据与已否定规则

受控 placement 保持 DSA problem 与生成操作不变，只改变指定物理 overlap。现有结果
表明：

- 大部分合法 reuse 不改变同步；
- synchronization-group count 不能预测 latency；
- route class 不足以分类，同一路由可能 harmful、neutral 或被其他 release 覆盖；
- loop frequency 会放大暴露在 hot path 上的 handoff，但不能让已覆盖 handoff 变贵；
- 同一 consumer 的多个 predecessor 更接近“最新 predecessor”而不是简单求和；
- 尚无实验支持负优化权重；同步减少没有产生可重复的 latency 收益。

最新 exact ordered-pair study 给出：

| Pair 类型 | 结果 |
| --- | --- |
| SSA-ordered `V -> MTE2` WAR | overlap 增加一个匹配的 PTOAS handoff |
| unordered `M -> MTE1` WAR | overlap 删除冗余 handoff，但没有确认 latency 收益 |
| 另外四个匹配 pair | 同步不变 |

因此：

- SSA `dag_path` 是 provenance，不是安全 suppression predicate；
- promotion 所有 unordered cross-resource candidate 过于宽泛；
- 不能给所有改变同步的 pair 正权重；
- 非负 pair model 仍然成立：neutral 或表面 beneficial 的 pair 可以不产生正 edge。

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
completion；普通 SSA reachability 不够。

只有当 `u` 扩展 `v` 已有的 completion-release frontier 时，才有理由产生正 edge。
定性 cost conjecture 是：

```text
dynamic frequency
    * max(0, completion(u) - latest unavoidable release of v)
```

这只是建模指南，并非 PyPTO 当前可计算的 cycle estimator。验证完成前应保留 raw
candidate、默认权重为零，并只 promotion 具有重复 harmful 证据的机制。对于同一
consumer 的多个 candidate predecessor，应保留有依据的 dominant pair，而不是重复
求和。OR group、hyperedge、负权重和全局 event-budget term 继续推迟。

## 下一步验证

下一项 fixed-placement 实验对同一 consumer 构造 two-edge factorial：

```text
target overlap off/on
covering overlap off/on
```

它需要比较：

- 一个会新增 release 的 uncovered target handoff；
- 已有更晚 completion release 覆盖该 handoff 的情况；
- overlap 删除冗余 release 的 coalesced 情况。

每种 geometry 在两个物理地址重复。运行 PTOAS 前冻结预测：

```text
uncovered target -> synchronization addition
covered target   -> no additional synchronization
coalesced target -> synchronization removal
```

实验记录最终 predecessor identity 和 kernel-only latency，而不是只看 summary count。
所有 endpoint 都必须满足 exact overlap XOR、pre-InsertSync PTO 仅地址变化、输出
bit-identical、使用真实 kernel input/scalar，并对结构上有信息量的 case 在两个设备上
运行 ABBA timing。

只有 completion-frontier rule 能跨多个 kernel 和 memory space 预测 fresh case，且
separation 能消除可重复的 material latency cost 而不引入其他 handoff，才支持
promotion。
