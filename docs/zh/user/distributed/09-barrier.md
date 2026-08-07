# Barrier：仅信号（Signals Only）

用 `notify`/`wait` 为一次汇合构建 N-rank barrier——不移动任何数据——然后揭示
提供同样同步语义的内置原语 `pld.tensor.barrier`。

> **前置条件：** [08-window_buffer](08-window_buffer.md)。两个设备。

**建议阅读顺序（Suggested reading order）：** 01 → 02 → 03 → **04** → 05 → 06 — 本页为步骤 04。

## 思路（The idea）

window buffer 的**信号尾**只有一个职责：跨 rank 同步。两个原语驱动它。
`pld.system.notify(...)` 在对端递增一个信号单元；`pld.system.wait(...)`
阻塞直到信号单元达到阈值。二者共同构成每个集合通信 lower 成的握手。

**Barrier** 是汇合点：每个 rank 等待*所有* rank 到达。不交换任何数据——
barrier 是纯粹的同步。本步骤手工用 `notify`/`wait` 编写 barrier，并让到达
模式*可见*：每个 rank 拥有每个对端信号 window 中的一行，barrier 之后，
rank `r` 自己的行读作 `[1, …, 0, …, 1]`——除自身外每一列都是 `1`，因为
rank 从不通知自己。将该行呈现出来，就是"每个对端都已到达"的证明。

**为什么 `AtomicAdd` + `Ge`。** N 个 rank 写入同一个信号单元，因此贡献必须
累加：`AtomicAdd` 增长计数器，`Ge(1)` 在每个对端都已到达时通过，而 `Set`
会静默覆盖更早的到达。本示例只运行一次汇合——计数器是单调的，因此在同一
window 上复用第二次 barrier 需要重置单元，或按代次提高 `expected` 阈值。

**成本卡片：** 一轮通信，每个 rank 发出 `P-1` 次 notify + `P-1` 次 wait，
零数据字节。这是语言中最便宜的汇合——是每个集合通信的基准下限。

## 运行（Run it）

```bash
# 手工 barrier（默认）：
python examples/distributed/04_barrier.py -p a2a3sim -d 0,1

# 揭示 -- pld.tensor.barrier 为 remote_load 排序：
python examples/distributed/04_barrier.py -p a2a3sim -d 0,1 --use-builtin
```

预期输出：

```text
OK
```

## 走读（Walkthrough）

手工 kernel：

```python
@pl.jit.incore
def barrier_handrolled(
    y: pl.Out[pl.Tensor[[N_RANKS, 1], pl.INT32]],
    signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
):
    ctx = pld.get_comm_ctx(signal)
    my_rank = pld.rank(ctx)

    for peer in pl.range(N_RANKS):
        if peer != my_rank:
            pld.system.notify(
                signal, peer=peer, offsets=[my_rank, 0],
                value=1, op=pld.NotifyOp.AtomicAdd,
            )
    for src in pl.range(N_RANKS):
        if src != my_rank:
            pld.system.wait(
                signal, offsets=[src, 0],
                expected=1, cmp=pld.WaitCmp.Ge,
            )

    for i in pl.range(N_RANKS):
        val = pl.read(signal, [i, 0])
        pl.write(y, [i, 0], val)
    return y
```

- **上下文。** `pld.get_comm_ctx(signal)` 解析 window 所属的通信上下文；
  `pld.rank(ctx)`（以及 `pld.nranks(ctx)`）由此而来。InCore kernel 不把 rank
  作为标量参数接收——它从 window 推导。
- **notify 阶段。** 每个 rank 通知每个*其他* rank，用 `AtomicAdd` 把 `1`
  写入对端信号 window 的第 `my_rank` 行。rank `r` 从不通知自己——这正是
  它自己的行在第 `r` 列以 `0` 结尾的原因。
- **wait 阶段。** 每个 rank 等待*自己* window 中其他每个 rank 的行，
  `expected=1, cmp=Ge`。由于对端使用 `AtomicAdd`，单元只需达到 `1`。
- **可观测结果。** 用 `pl.read`/`pl.write` 逐单元读取信号行，呈现到达模式。
  （对 `[2,1]` 的 `INT32` window 做 *tile* load 会被拒绝：其 8 字节列低于
  ptoas 对列主序 tile 要求的 32 字节对齐——见边界情况。）

内置揭示：

```python
@pl.jit.incore
def barrier_builtin(x, y, data, signal):
    ...
    local = pl.load(x, [0, 0], [1, SIZE])
    data = pl.store(local, [0, 0], data)
    signal = pld.tensor.barrier(signal)
    peer = (my_rank + 1) % nranks
    recv = pld.tile.remote_load(data, peer=peer, offsets=[0, 0], shape=[1, SIZE])
    y = pl.store(recv, [0, 0], y)
    return y
```

`pld.tensor.barrier(signal)` 执行手工循环的相同同步——但它同步时*不在信号
window 中留下计数*，因此揭示改用数据来证明 barrier：每个 rank 先 staging
自己的 slice，再 barrier，然后 remote-load 下一个 rank 的 slice。缺少 barrier
会让 load 与对端的 store 竞争；golden `y[r] = x[(r+1) % N]` 只有靠 barrier
排序才成立。主机侧的 `x`/`signal`/`data` 形态与之前相同，只是一个调用
取代了一个循环。

## 边界情况（Edge cases）

> **致命陷阱——`Set`/`Eq` barrier 永远看不到全部到达。** `NotifyOp.Set` +
> `WaitCmp.Eq` 让 N 个 rank 用普通覆盖写入同一个单元，更早的到达被静默
> 覆盖，barrier 可能在任何对端到达之前就通过。**修复：** 使用 `AtomicAdd`
>
> - `Ge`，让贡献累加。

| 症状 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| 在每个对端到达之前 barrier 就通过 | `Set`/`Eq`——后写覆盖先写 | 使用 `AtomicAdd` + `Ge`，让贡献累加 |
| 第二次 barrier 在对端到达之前就通过 | 复用同一 window——计数器已满足 `Ge(1)` | 重置单元，或跟踪代次并在每次调用时提高 `expected` |
| 自己的信号行在对端到达处显示 `0` | 忘记 rank `r` 在 notify 循环中跳过自己 | 跳过 `peer == my_rank` |
| `pto.alloc_tile` … `32-byte aligned` | tile-load 一个窄的 `INT32` window（如 `[2,1]` = 8 B 列） | 用 `pl.read`/`pl.write` 逐单元读写，或加宽 window |
| 内置揭示输出全零 | 在 `pld.tensor.barrier` 后读取信号计数 | 内置只同步不留下计数——改用数据证明顺序 |
| 某个 rank 永远等待 | notify/wait 目标行不匹配 | notify 写入对端的第 `my_rank` 行；wait 读取自己的第 `src` 行 |

## 参见（See also）

- [05-tutorials](05-tutorials.md) — 教程总览（本步骤 = 第 04 行）
- [02-primitives](../distributed/02-primitives.md) §Notify 与 Wait + §选择
  NotifyOp 与 WaitCmp — 完整信号 API
- [01-collectives](../distributed/01-collectives.md) §Barrier — barrier 在
  集合通信中的位置
- 下一步：[10-remote_load_store](10-remote_load_store.md) — 移动一个 slice
