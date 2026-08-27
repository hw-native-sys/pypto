# 全归约 2：两阶段（Two-Phase）——先 Reduce-Scatter 再 All-Gather

与步骤 08 相同的结果，远程流量约减半。不再让每个 rank 读取每个对端的*完整*
slice，而是把 slice 切成 `P` 块：先做 **reduce-scatter**，让 rank `r` 持有
归约后的块 `r`；再做 **all-gather**，让每个 rank 收集每一块。

> **前置条件：** [13-allreduce_mesh](13-allreduce_mesh.md)。建议使用 4 个模拟
> 设备（相对 mesh 的流量节省只在 P≥4 可观测）；`SIZE` 必须能被 rank 数量整除。

**建议阅读顺序（Suggested reading order）：** 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → **09** — 本页为步骤 09。

## 思路（The idea）

mesh 每 rank 传输 `(P-1) * N` 字节，因为每个 rank 读取每个对端的*整段*
slice。但求和每 rank 只需要 `N` 个结果值——所以 mesh 移动的大部分是重复
劳动。两阶段把同样的求和重构为两个阶段，每阶段只移动 `N/P` 大小的片段：

1. **Reduce-scatter（RS）：** rank `r` 是块 `r` 的*拥有者*。每个 rank 读取
   每个对端的块 `r`（一个 `N/P` 大小的片段，而非完整 slice）并本地求和。
   之后，只有 rank `r` 持有归约后的块 `r`。
2. **All-gather（AG）：** 每个 rank 读取每个对端归约后的块（每 rank 一块），
   组装出完整结果。

每阶段移动 `(P-1) * N/P` 字节，总流量为 `2 * (P-1) / P * N`——约为 mesh 的
一半，代价是第二个 barrier。

## 运行（Run it）

```bash
# P=4（相对 mesh 的节省在此显现）与 P=2：
python examples/distributed/09_allreduce_two_phase.py -p a2a3sim -d 0,1,2,3
python examples/distributed/09_allreduce_two_phase.py -p a2a3sim -d 0,1
```

预期输出：

```text
OK
```

## 走读（Walkthrough）

与步骤 08 相同的 `@pl.program` class form，但这里多了一个 **rank 数量
工厂**——步骤 08 并不需要它。工厂的作用是把 `nr` 变成编译期常量，而本步骤
是第一个真正需要它的：块大小 `SIZE // nr` 是 **tile 形状**，而 tile 形状
必须在 kernel 编译时已知。此外现在有**两个**窗口（`data` 存 staging 的输入，
`result` 存归约后的块）和一个**两行信号**（`[2, nr]`——每轮 barrier 一行）：

```python
# Phase 1 — 把本 rank 的 slice 放入自己的窗口槽位。
local = pl.load(x, [0, 0], [1, SIZE])
data = pl.store(local, [0, 0], data)

# Barrier A（信号第 0 行）— 所有输入在 RS 读取前完成 staging。
# （同步骤 08：通知全部/等待全部，但用第 0 行）

# Phase 2 — reduce-scatter：rank r 拥有结果的块 r。
acc = pl.load(data, [0, my_rank * chunk], [1, chunk])
for peer in pl.range(nranks):
    if peer != my_rank:
        recv = pld.tile.remote_load(data, peer=peer, offsets=[0, my_rank * chunk],
                                    shape=[1, chunk])
        acc = pl.add(acc, recv)
result = pl.store(acc, [0, my_rank * chunk], result)

# Barrier B（信号第 1 行）— 所有归约块在 AG 读取前完成 staging。

# Phase 3 — all-gather：rank r 读取每个 rank 的归约块。
for c in pl.range(nranks):
    recv = pld.tile.remote_load(result, peer=c, offsets=[0, c * chunk], shape=[1, chunk])
    y = pl.store(recv, [0, c * chunk], y)
```

- **块所有权。** RS 循环在每个对端窗口的 `[0, my_rank * chunk]` 处读取——每个
  rank 从每个对端读取*同一个*块（自己的块）。循环结束后，rank `r` 的
  `result` 持有归约后的块 `r`。
- **AG 每对端读一块**，从对端 `c` 的 `[0, c * chunk]` 处读取——即 RS 中已经
  归约好的片段——并按序写入输出。
- **两个 barrier，两行信号。** `[2, nr]` 信号给每个 barrier 自己的行，于是
  单调计数器（`Ge(1)`）不会在轮次之间泄漏。

**成本卡（每 rank）：** `2 * (P-1) / P * N` 字节——RS 中 `(P-1)` 次 `N/P`
字节读取，AG 中 `(P-1)` 次 `N/P` 字节读取。约为 mesh 的 `(P-1) * N` 的一半，
代价是多一轮 barrier。

## 边界情况（Edge cases）

> **致命陷阱——两个 barrier 复用同一行信号。** 计数器是单调的：barrier A
> 之后该行 `Ge(1)` 已满足，barrier B 会立即返回，AG 读取可能与 RS 的 store
> 竞争。**修复：** 给每轮自己的行（`[2, nr]` 信号）——ring 步骤会把这一
> 纪律推广为每轮一行。

| 现象 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| AG 读到陈旧的块数据 | 两个 barrier 用了同一信号单元 | 每轮一行信号（`[2, nr]`） |
| 只在 P≥4 出错 | 块大小不匹配（`SIZE % P != 0`） | 运行 `SIZE` 能整除的 P（2、4） |
| 结果中块 `r` 位置错误 | AG 组装乱序 | 从对端 `c` 读块 `c`，写到 `[0, c * chunk]` |
| 所有 rank 持有 RS 结果但非总和 | 缺 AG（只跑了 reduce-scatter） | 补 AG 循环：每 rank 读取每个归约块 |
| 每个 rank 结果相同但与 torch 和不同 | 归约顺序不同（非 bug） | 用容差比较 |

## 参见（See also）

- [05-tutorials](05-tutorials.md) — 教程索引（本步骤 = 第 09 行）
- [13-allreduce_mesh](13-allreduce_mesh.md) — 本步骤改进的基线（步骤 08）
- [01-collectives](01-collectives.md) §AllReduce — 参考（Mesh Mode、Ring Mode）
- 下一步：[15-allreduce_ring](15-allreduce_ring.md) — 同样字节数，每步大小恒定
