# 全归约 1：Mesh——每个 rank 读取每个对端

阶梯中的第一个集合通信（collective）。每个 rank 贡献自己的 slice；最终每个
rank 都持有**所有 slice 的元素级求和**。mesh 算法是这一语义最简单的写法：
一个 barrier，然后每个 rank `remote_load` 每个对端的 slice 并本地累加。

> **前置条件：** [12-dynamic_rank_count](12-dynamic_rank_count.md)。建议使用
> 4 个模拟设备——在 2 个 rank 时，步骤 08-10 的三种全归约变体会坍缩为同一次
> 交换，它们的差异只有在 P=4 时才可观测。

**建议阅读顺序（Suggested reading order）：** 01 → 02 → 03 → 04 → 05 → 06 → 07 → **08** — 本页为步骤 08。

## 思路（The idea）

全归约（all-reduce）是所有集合通信对比的起点：你有 `P` 个 rank，各持一个
slice；调用结束后每个 rank 都持有*所有* slice 的归约结果。步骤 11 会揭示
内置原语；这里你先手工构建它，而你建立的成本卡正是内置原语存在的原因以及
它在什么之间做选择。

mesh 是朴素基线：**每个 rank 读取每个对端**。每个 rank 有 O(P) 次远程流量
——简单、round 密集，也是 two-phase 与 ring 步骤的度量基准。

## 运行（Run it）

```bash
# P=4（对比步骤需要）与 P=2：
python examples/distributed/08_allreduce_mesh.py -p a2a3sim -d 0,1,2,3
python examples/distributed/08_allreduce_mesh.py -p a2a3sim -d 0,1
```

预期输出：

```text
OK
```

## 走读（Walkthrough）

这里的 rank 数量始终**不是**编译期常量：它在注解中是
`NR = pl.dynamic("NR")`，在 HOST 编排体内是 `pld.world_size()`。因此一个
模块级 `@pl.program` 即可服务 `-d` 指定的任意 world 大小——不需要 rank
数量工厂。这与本集合通信的系统测试
`tests/st/distributed/collectives/test_l3_allreduce.py` 完全一致。

步骤 01-07 使用 `@pl.jit` 系列；这里改用 class form 是一种呈现上的选择，
而非必需。`signal` 是形状为 `[pld.world_size(), 1]` 的窗口，其行数无法由
任何静态规则折叠。`@pl.jit` 会为这样的维度合成一个动态维，在它生成的程序
里用 `pl.dynamic` 声明，kernel 再从实参的 descriptor 绑定它——这在结构上
就是下面 class form 手写的 `NR`，只是符号名不同。这里采用 class form，是为
了让该形状与它对应的系统测试并排展示得更清楚。

步骤 09 与 10 改用 class form 的理由则**确实是硬性的**：它们的块大小
`SIZE // nr` 是 **tile 形状**，而 tile 形状必须在 kernel 编译时已知，所以
那里确实需要编译期 rank 数量与工厂。这一限制对两种装饰器系列同样成立——
动态维一旦流入 tile 形状，会在下游由 `InitMemRef` 拒绝，而不是在前端。
信号的行数不是 tile 形状，因此这里可以保持动态。

kernel 是每个手工集合通信共有的四阶段：

```python
# Phase 1 — 把本 rank 的 slice 放入自己的窗口槽位。
local = pl.load(x, [0, 0], [1, SIZE])
data = pl.store(local, [0, 0], data)

# Phase 2 — barrier：通知所有对端，等待所有对端槽位。
for peer in pl.range(nranks):
    if peer != my_rank:
        pld.system.notify(signal, peer=peer, offsets=[my_rank, 0],
                          value=1, op=pld.NotifyOp.AtomicAdd)
for src in pl.range(nranks):
    if src != my_rank:
        pld.system.wait(signal, offsets=[src, 0], expected=1, cmp=pld.WaitCmp.Ge)

# Phase 3 — 累加：从自己的 slice 开始，加上每个对端的 slice。
acc = pl.load(data, [0, 0], [1, SIZE])
for peer in pl.range(nranks):
    if peer != my_rank:
        recv = pld.tile.remote_load(data, peer=peer, offsets=[0, 0], shape=[1, SIZE])
        acc = pl.add(acc, recv)

# Phase 4 — 写出：累加结果就是本 rank 的输出。
y = pl.store(acc, [0, 0], y)
```

- **Phase 2 就是步骤 04 的 barrier。** 每个 rank 拥有专属行
  （`offsets=[my_rank, 0]`）；`AtomicAdd`/`Ge(1)` 只在每个对端都完成 staging
  后通过。没有它，Phase 3 可能在某个对端的 store 落地前就 `remote_load`
  该对端的 slice。
- **Phase 3 就是 mesh 本身。** 从自己的 slice 开始，再 `remote_load` 每个
  其他 rank 的 slice 并相加。注意对称性：每个 rank 都这样做，因此每个 rank
  得到相同的和。

**成本卡（每 rank）：** `(P-1) * N` 字节——每个对端一个完整 slice，被每个
rank 读取。round 密集：`P-1` 次远程读取加一个 barrier。正是这种 O(P) 流量
催生了 two-phase 与 ring 变体。

## 边界情况（Edge cases）

> **致命陷阱——缺少 barrier 会让读取与 store 竞争。** 若去掉 Phase 2，某个
> rank 可能在某个对端的 `pl.store` 落地前读取该对端的窗口槽位，把陈旧/零值
> 混入求和。该竞争与时序相关，可能在 P=2 通过而在 P=4 失败。
> **修复：** 任何 `remote_load` 之前必须完成 notify/wait 握手。

| 现象 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| 某些 rank 的和里含零 | barrier 缺失/错误；读取竞争了 store | Phase 3 前做 barrier（通知全部/等待全部） |
| 只在 P=4 出错 | P=2 掩盖竞争（单一对端） | 用 P≥4 运行；检查 barrier 覆盖每个对端 |
| 每个 rank 结果相同但与 torch 和不同 | 归约顺序不同（非 bug） | 用容差比较（示例已如此） |
| `InitMemRef requires static shape ... is dynamic` | 运行期决定的维度流入了 **tile** 形状（例如由 rank 数量推导出的块大小） | 像步骤 09-10 那样用工厂给该 kernel 一个编译期 rank 数量；两种装饰器系列都不允许 tile 形状是运行期尺寸 |
| golden 出现巨大差异 | slice 被求和在错误位置（如自己的 slice 被重复计算） | 只 staging 一次；从自己的 slice 开始再累加对端 |

## 参见（See also）

- [05-tutorials](05-tutorials.md) — 教程索引（本步骤 = 第 08 行）
- [01-collectives](01-collectives.md) §AllReduce — mesh 模式参考
- [09-barrier](09-barrier.md) — 这里复用的 notify/wait barrier（步骤 04）
- [10-remote_load_store](10-remote_load_store.md) — `remote_load`（步骤 05）
- 下一步：[14-allreduce_two_phase](14-allreduce_two_phase.md) — 同样的结果，流量约减半
