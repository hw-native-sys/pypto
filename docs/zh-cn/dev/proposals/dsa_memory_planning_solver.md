# RFC：可插拔 DSA 内存规划器

## 状态

草案。跟踪 issue #1980 与 #1908 的 fragmentation 问题。

## 问题

PyPTO 目前把内存规划拆成 `MemoryReuse` 与 `AllocateMemoryAddr`。前者合并 allocation
identity，后者为每个 identity 分配一个 slot。这个拆分无法表达普通 freed-region
subdivision。

例如，一个早期 64 KiB buffer 结束后，两个同时存活的 32 KiB buffer 应能分别使用其
两半，总峰值仍为 64 KiB。这是标准 Dynamic Storage Allocation（DSA），不是 PyPTO
特有变体。

## 标准 DSA 路径

物化 semantic alias 后，adapter 导出固定 size、alignment、memory pool 与保守半开
lifetime 的物理 buffer。solver 选择 offset。生命周期重叠或带 hard separation 的
buffer 地址范围必须不相交；其他生命周期不相交的 buffer 可以任意部分复用。

```text
InitMemRef
  -> MaterializeSemanticAliases
  -> 收集未合并的物理 buffer
  -> standalone DSA solver
  -> 独立验证
  -> 把 offset 写回 MemRef
```

adapter 为每个 allocation 导出一个保守 physical-lifetime hull。不能把 SSA member
range 的空洞当作物理 dead time；这种做法曾导致 DeepSeek-v4 loop-carried accumulator
在 device 上被错误覆盖。

## PR #1949 的 pipeline intent

`pl.pipeline(stage=F)` 产生在标量程序序中顺序出现、但设计为在异步硬件单元上重叠的
clone。若并发 stage 共享地址，会形成虚假的 write-after-read dependency，使 ping-pong
pipeline 串行化。

`pipeline_membership=(group,stage)` 因此保留到 DSA 收集阶段。adapter 首先把所有请求
stage 导出为 hard `pipeline_stage` separation。求解策略分两阶段：

1. 在 capacity 与所有 pipeline-stage separation 都为 hard constraint 时求解标准 DSA；
2. 只有该搜索找不到可装入的 placement 时，才移除 `pipeline_stage` reason，保留同一
   pair 上的 target-hazard 与 semantic reason，加入稀疏 `pipeline_serialization`
   reuse cost，并再次求解。

fallback 会发出 `PH-DSA-001`，说明编译通过允许部分 pipeline copy 共享物理范围而成功，
因此 overlap 可能下降。由于当前 solver 是 heuristic，零 cost fallback solution 会先
针对 strict problem 重新验证；如果仍满足全部 hard constraint，则不发 warning。

对 pipeline-intent pair 集合 \(P\)，strict problem 加入：

```text
(i,j) in P  =>  address_range(i) does not intersect address_range(j)
```

fallback 只移除这些额外 edge，并最小化：

```text
lex(capacity_overflow, sum((i,j) in P) w_ij * reuse(i,j),
    total_peak, max_peak)
```

只有生命周期不相交的 buffer 在物理地址上重叠时，`reuse(i,j)` 才为一。lifetime
conflict 与所有非 pipeline separation 仍为 hard constraint。
fallback document 保留请求的 stage/residue mapping 作为 provenance；实际达到的
depth 是 placement measurement，不是 `effective_depth` 字段。

## Research refinement

### 1. Pipeline-overlap-aware placement

已实现的 fallback 在 capacity 后按字典序最小化 cross-stage overlap cost。版本 1 对
每个复用的 stage-member pair 计一个单位 cost。PR #1949 证明了机制，但 pair count
可能让成员更多的 group 权重过高。device A/B 应把它与 group-level lost-depth
objective 比较。

### 2. PTOAS-synchronization-aware placement

其他地址复用也可能让 PTOAS 加入 anti-dependency、event、wait 或 barrier。候选包括
MTE-to-vector/cube reuse，以及使前移 load 失效的 reuse。PyPTO 在导出时不知道最终
hardware-pipe assignment，因此权重必须通过 PTOAS instrumentation 或有界
placement-to-PTOAS feedback 校准。

### 3. Critical-path 与 event-budget-aware placement

同步 cost 不一定可加：已有 dependency 隐含的 reuse edge 可能免费，多个 edge 也可能
形成新的串行 critical path。更强 evaluator 应测量 augmented dependency graph 的
critical-path 增长。event identifier exhaustion 是离散资源限制，可能更适合作为 hard
bound。

Bank cost、multi-interval liveness、flexible pool assignment 与 piecewise size 仍是
hypothesis；没有 export proof 与受控测量时不得进入 required profile。

## 接口与目标

standalone problem 包含 buffer、pool、colocation、separation、reservation、可选固定
offset 与字典序 objective。solver 声明 capability；不支持的 constraint 或 objective
返回 `kUnsupported`，不得静默丢弃。

strict solve 在 capacity 下最小化 peak。显式 fallback 使用：

```text
(capacity overflow, reuse/synchronization cost, total peak, max peak)
```

所有原始 component 都必须报告，不能任意把 byte 换算成 cycle。

## 验证

- host #1908 regression：64 + 32 + 32 KiB 的 peak 为 64 KiB；
- 独立检查 lifetime conflict、separation、capacity、alignment、reservation、alias 与
  writeback；
- PyPTO 与 PyPTO-Lib device numerics，包括 DeepSeek 与 Qwen；
- pipeline test 检查 strict intent 与 `PH-DSA-001` fallback；
- 固定 schedule/tiling 的 A/B placement，记录 PTO、event/wait/barrier、保留 depth、
  latency 与 utilization；以及
- 拟合 cost model 时保留 held-out kernel。

外部 solver dependency 是临时方案。选定并验证 heuristic 后，应把它移植到 PyPTO 并
移除 dependency。

## 参考

Issues/PRs：#1908、#1934、#1949、#1980；PTOAS #913。Baselines：
MiniMalloc、TelaMalloc、TVM USMP 与 OpenXLA heap simulation。
