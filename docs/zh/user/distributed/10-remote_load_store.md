# 远程加载/存储：Tile 级 RMA（Remote Load / Store）

用 `pld.tile.remote_load` 与 `pld.tile.remote_store` 在 rank 之间移动一个
slice——一步环形移位的两个侧面。

> **前置条件：** [09-barrier](09-barrier.md)。两个设备。

**建议阅读顺序（Suggested reading order）：** 01 → 02 → 03 → 04 → **05** → 06 — 本页为步骤 05。

## 思路（The idea）

window 是对称的：你可以到达*任何* rank 的 slice，而不仅是自己的。
**Tile 级 RMA** 直接暴露这一点。`pld.tile.remote_load(...)` 将对端的 slice
拉入本地 tile；`pld.tile.remote_store(...)` 将本地 tile 推入对端的 slice。
这些正是参考章节中手工 all-reduce 所依赖的操作。

示例是在 barrier 之后的一次**一步环形移位**：每个 rank 沿环将自己的数据
移动一位。`--mode load` 展示*拉*侧（每个 rank remote-load 下一个 rank 的
slice → `y[r] = x[(r+1) % N]`）；`--mode store` 展示*推*侧（每个 rank
remote-store 进下一个 rank 的 slice，再读回自己的 → `y[r] = x[(r-1) % N]`）。

**成本卡片：** 一步中每个 rank 向（或从）一个对端移动一个 `N` 字节的 slice，
一轮通信，一次远程读或写。延迟受限：成本在于往返，而非字节数。

## 运行（Run it）

```bash
# 拉侧（默认）：
python examples/distributed/05_remote_load_store.py -p a2a3sim -d 0,1

# 推侧：
python examples/distributed/05_remote_load_store.py -p a2a3sim -d 0,1 --mode store
```

预期输出：

```text
OK
```

## 走读（Walkthrough）

```python
@pl.jit.incore
def shift_by_load(x, y, data, signal):
    ctx = pld.get_comm_ctx(data)
    my_rank = pld.rank(ctx)
    nranks = pld.nranks(ctx)

    local = pl.load(x, [0, 0], [1, SIZE])
    data = pl.store(local, [0, 0], data)

    signal = pld.tensor.barrier(signal)

    peer = (my_rank + 1) % nranks
    recv = pld.tile.remote_load(data, peer=peer, offsets=[0, 0], shape=[1, SIZE])
    y = pl.store(recv, [0, 0], y)
    return y
```

- **Stage 入。** 每个 rank 用普通的 `pl.load`/`pl.store` 把自己的本地 `x`
  slice 复制进自己的 window slice——RMA 读取的是 *window* 内存，因此数据
  必须先进入 window。
- **Barrier。** 步骤 04 的 barrier（此处为已揭示的内置原语）为交换排序：
  在所有 rank 完成 staging 之前，任何 rank 都不会 remote-load。
- **远程加载。** `pld.tile.remote_load(data, peer=peer, offsets=[0, 0],
  shape=[1, SIZE])` 把对端的 window slice 拉入本地 tile，与本地 load 完全
  相同——只是从对端的内存。`peer = (my_rank + 1) % nranks` 是普通的
  `INT32` 标量算术，在 AI 核上合法（不同于 FP32 标量算术——见步骤 01）。

store 侧把移动换成推送：

```python
    local = pl.load(x, [0, 0], [1, SIZE])
    peer = (my_rank + 1) % nranks
    pld.tile.remote_store(local, data, peer=peer, offsets=[0, 0])

    signal = pld.tensor.barrier(signal)

    back = pl.load(data, [0, 0], [1, SIZE])   # rank (r-1) 刚写入我们的 slice
    y = pl.store(back, [0, 0], y)
```

`remote_store` 接收本地 tile、window 与对端——然后推送。barrier 之后，每个
rank 读取*自己*的 window slice，它刚被上一个 rank 写入。为 load 排序的同一
barrier 也为 store 排序。

**`DistributedTensor` 与 `Tensor`。** 只有绑定 window 的 `pld.DistributedTensor`
对其它 rank 可见。`x` 与 `y` 是普通 `pl.Tensor`——任何对端都无法到达的本地
输入/输出。规则是结构性的：任何你想共享的内容都必须流经 window buffer 的
`pld.DistributedTensor` 视图。

## 边界情况（Edge cases）

> **致命陷阱——staging 之前就远程加载。** `remote_load` 读取 *window* 内存。
> 如果对端尚未 staging 自己的 slice，你读到的是陈旧（零填充）数据。
> **修复：** 任何远程 load/store 之前，务必先把数据放入 window *并*跨过
> barrier。

| 症状 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| 远程读取返回零 | 对端 staging 之前就 load / 无 barrier | 先 staging，再 `pld.tensor.barrier`，然后 RMA |
| 移位方向错误 | 混淆 pull 与 push 语义 | `load` 模式：读 `(r+1)`；`store` 模式：写 `(r+1)` 并读自己 |
| `peer` 计算中出现标量 `FP32` 算术错误 | AI 核上的标量浮点运算 | 索引计算保持在 `INT32`（`(r+1) % n`），只为数据运算 cast |
| window 参数类型不匹配 | `DistributedTensor` 被当作 `Tensor` | 共享 buffer 标注为 `pld.DistributedTensor[...]` |
| 某个 rank 读到上一个 rank 的陈旧数据 | 跳过 barrier 或 barrier 位置错误 | barrier 必须位于每个 rank 的 stage 与 RMA 之间 |

## 参见（See also）

- [05-tutorials](05-tutorials.md) — 教程总览（本步骤 = 第 05 行）
- [02-primitives](../distributed/02-primitives.md) §Tile 级 RMA — 完整的
  `pld.tile.*` 表面
- [01-collectives](../distributed/01-collectives.md) — all-reduce 就是这些
  移动加一个 add（步骤 07–09）
- 下一步：[11-put_get](11-put_get.md) — tensor 级 push/pull
