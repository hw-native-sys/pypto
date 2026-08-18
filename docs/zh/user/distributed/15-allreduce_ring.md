# 全归约 3：Ring——每步大小恒定

步骤 09 的两阶段形态，但每个 rank 只与**左邻居**移动数据。不再每阶段读取
`P-1` 个对端，而是让块绕环旋转：`2 * (P-1)` 步，每步移动一个 `N/P` 大小的
块——在弱扩展下，无论 `P` 多大，每步传输都保持 `N/P`（固定 `N` 时块会缩小）。
同步也是邻居局部的：每个
rank 在 store 之后通知**右**邻居，在读取之前等待**左**邻居——每轮不再有
全 mesh barrier。

> **前置条件：** [14-allreduce_two_phase](14-allreduce_two_phase.md)。
> 建议使用 4 个模拟设备（恒定的每步大小只在 P≥4 可观测）；`SIZE` 必须能被
> rank 数量整除。

**建议阅读顺序（Suggested reading order）：** 01 → … → 09 → **10** — 本页为步骤 10。

## 思路（The idea）

两阶段把 mesh 的流量减半，但仍每阶段读取每个对端——O(P) 个对端，每个
`N/P` 块。ring 去掉"每个对端"这层：把 rank 排成一个环，把块传给**左邻居**。
同样的 reduce-scatter + all-gather 拆分现在各需 `P-1` 步：

- **Reduce-scatter（P-1 步）：** 每步一个块前进一跳并在目的地累加。`P-1`
  步后每个 rank 持有它拥有的归约块。
- **All-gather（P-1 步）：** 归约块继续绕环，每个 rank 在块经过时复制一份。
  再 `P-1` 步后每个 rank 持有完整结果。

总字节数与两阶段相同（`2 * (P-1) / P * N`），但每步只移动 `N/P`——在弱扩展
（workload 随 P 增长）下，随着 `P` 增长每步大小保持恒定，这正是 ring 在大
world 规模下依然高效的原因；固定 `N` 时分块反而会缩小。

## 运行（Run it）

```bash
# P=4（恒定的每步大小在此显现）与 P=2：
python examples/distributed/10_allreduce_ring.py -p a2a3sim -d 0,1,2,3
python examples/distributed/10_allreduce_ring.py -p a2a3sim -d 0,1
```

预期输出：

```text
OK
```

## 走读（Walkthrough）

kernel 是单体的（一个 InCore 函数），信号把两阶段的思路推广为
`[2 * (nr-1), nr]`——**每轮一行**。块索引运算才是调度的核心：

```python
left = (my_rank - 1 + nranks) % nranks      # 永不取负

# Reduce-scatter：(nr-1) 步。
for s in pl.range(nranks - 1):
    step = s + 1
    recv_add_idx = (my_rank - step - 1 + nranks) % nranks
    left_send_idx = (left - step + nranks) % nranks
    # 等待左邻居第 s 轮的块（信号第 s 行），然后：
    pld.system.wait(signal, offsets=[s, left], expected=1, cmp=pld.WaitCmp.Ge)
    recv = pld.tile.remote_load(scratch, peer=left,
                                offsets=[0, left_send_idx * chunk],
                                shape=[1, chunk])
    acc = pl.load(scratch, [0, recv_add_idx * chunk], [1, chunk])
    acc = pl.add(acc, recv)
    scratch = pl.store(acc, [0, recv_add_idx * chunk], scratch)
    # 该 store 为下一轮准备好发送：通知右邻居（第 s+1 行）。
    pld.system.notify(signal, peer=right, offsets=[s + 1, my_rank],
                      value=1, op=pld.NotifyOp.AtomicAdd)

# All-gather：(nr-1) 步（行 nranks-1 .. 2*(nranks-1)-1），把左邻居的发送块
# 复制进本地块。
```

- **`left = (my_rank - 1 + nranks) % nranks`。** `+ nranks` 保证被除数非负——
  裸写 `(my_rank - 1) % nranks` 在 rank 0 处截断取模得 `-1`（步骤 06 的教训，
  这次在索引一侧）。
- **旋转的是块，不是 rank。** 第 `s` 轮里，你从左侧累加的块和你转发的块都
  偏移一位（`- step`），于是每个块在每阶段恰好访问每个 rank 一次。
- **邻居就绪握手，而非 barrier。** 每轮有自己的信号行——store 之后通知**右**
  邻居，`remote_load` 之前等待**左**邻居。单调的 `Ge(1)` 计数器在轮次之间
  永不泄漏（步骤 09 的纪律，现在有 `2*(P-1)` 行），且只有相邻的两个 rank
  参与同步——**每 rank** O(P) 次信号（全系统 O(P²)）——而不是每轮全 mesh
  barrier 的每 rank O(P²)。

**成本卡（每 rank）：** 总计 `2 * (P-1) / P * N`——与两阶段相同——但分在
`2*(P-1)` 步、每步 `N/P` 字节。在弱扩展（workload 随 P 增长）下，每步大小
**随 P 增长保持恒定**——不像 mesh 每步 `(P-1) * N`——这正是 ring 存在的
原因。

## 边界情况（Edge cases）

> **致命陷阱——左邻居索引取负。** `(my_rank - 1) % nranks` 在 rank 0 处截断
> 取模得 `-1`：`remote_load` 会指向一个无效对端。**修复：** 始终写
> `(my_rank - 1 + nranks) % nranks`（在 P=2 时 rank 0 的左邻居是 rank 1；
> `+ nranks` 保证被除数非负）。

| 现象 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| P=2 挂起 | 左邻居索引的被除数为负 | `(my_rank - 1 + nranks) % nranks` |
| 结果中的块错位 | 某轮的块索引运算差一位 | 在纸上为 `s=0` 追踪 `recv_add_idx` / `left_send_idx` |
| 两个握手共用一行信号 | RS 与 AG 复用了行索引 | RS 行 `0..P-2`，AG 行 `P-1..2(P-1)-1` |
| 只在 P=2 正确 | P=2 只有一轮，掩盖旋转 bug | 跑 P=4 并检查每个块位置 |
| 每个 rank 结果相同但与 torch 和不同 | 归约顺序不同（非 bug） | 用容差比较 |

## 参见（See also）

- [05-tutorials](05-tutorials.md) — 教程索引（本步骤 = 第 10 行）
- [14-allreduce_two_phase](14-allreduce_two_phase.md) — 两阶段形态（步骤 09）
- [01-collectives](01-collectives.md) §AllReduce — 参考（Ring Mode、信号形状、`Sum`/`FP32`）
- 下一步：[16-allreduce_reveal](16-allreduce_reveal.md) — 替你选择 mesh 或 ring 的内置原语
