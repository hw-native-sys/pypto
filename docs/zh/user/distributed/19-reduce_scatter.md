# ReduceScatter：全对分块

每个 rank staging 全部分块；每个 rank 最终得到自己索引处的归约分块——两相 all-reduce 的 reduce-scatter 一半——然后内置原语一次调用完成。

> **前置：** [18-allgather](18-allgather.md)。任意 ≥ 2 个设备（示例使用 2 和 4 个 sim 设备）。

**建议阅读顺序：** 01 → … → 13 → **14** —— 本页是步骤 14。

## 思路

Reduce-scatter 是 allgather 的镜像：不是每个 rank 收到*所有* slice，而是每个 rank 收到**一个归约后的分块**——自己索引处的分块，跨所有 rank 归约（这里为求和）。

| 方面 | ReduceScatter |
| ---- | ------------- |
| 输入 | 每个 rank 的完整 `P` 个分块（`N` 个元素） |
| 输出 | rank `r` 得到 `Σ_k chunk_r(inputs[k])`——`N/P` 个元素 |
| 模式 | staging 全部分块 → barrier → 跨对端求和自己的分块 |
| 开销 | 收到 `(P-1)/P · N` 字节——两相的第一半 |

这正是**步骤 09 两相 all-reduce 的第一半**。步骤 13 构建了第二半（allgather）；本步构建第一半。合起来就是你已经作为单个内置原语运行过的两相调度。

## 运行

```bash
# 手工：staging 全部分块、barrier、跨对端求和自己的分块。
python examples/distributed/14_reduce_scatter.py -p a2a3sim -d 0,1

# 揭示：一次调用完成 pld.tensor.reduce_scatter。
python examples/distributed/14_reduce_scatter.py -p a2a3sim -d 0,1 --mode builtin

# 同一源码在 P=4：
python examples/distributed/14_reduce_scatter.py -p a2a3sim -d 0,1,2,3
python examples/distributed/14_reduce_scatter.py -p a2a3sim -d 0,1,2,3 --mode builtin
```

预期输出：

```text
OK
```

golden 是逐 rank 的：`out[r]` 必须等于跨所有 rank 的分块 `r` 的逐元素和——每个 rank 是*不同*的分块，因此归约错分块的 rank 会失败。

## 走读

两种模式共享一个 `[nr, SIZE]` window：每个 rank 把分块 `c` staging 到行 `c`，并归约行 `my_rank`。手工内核：

```python
@pl.function(type=pl.FunctionType.InCore)
def hand_step(self, x, y, data, signal):
    ctx = pld.get_comm_ctx(data)
    my_rank = pld.rank(ctx)

    # Phase 1 — stage every chunk at its row, so each peer can read it.
    for c in pl.range(nr):
        chunk = pl.load(x, [0, c * SIZE], [1, SIZE])
        data = pl.store(chunk, [c, 0], data)

    # Phase 2 — barrier: notify every peer, wait on every peer slot.
    for peer in pl.range(nr):
        if peer != my_rank:
            pld.system.notify(signal, peer=peer, offsets=[my_rank, 0],
                              value=1, op=pld.NotifyOp.AtomicAdd)
    for src in pl.range(nr):
        if src != my_rank:
            pld.system.wait(signal, offsets=[src, 0], expected=1,
                            cmp=pld.WaitCmp.Ge)

    # Phase 3 — reduce: sum row my_rank across every peer.
    acc = pl.load(data, [my_rank, 0], [1, SIZE])
    for peer in pl.range(nr):
        if peer != my_rank:
            recv = pld.tile.remote_load(data, peer=peer, offsets=[my_rank, 0], shape=[1, SIZE])
            acc = pl.add(acc, recv)
    return pl.store(acc, [0, 0], y)
```

- **你 staging 全部分块，而不仅仅是自己的。** 每个 rank 发布整个 `[nr, SIZE]` 矩阵，这样任何对端都能读取它需要的具体分块。这与 allgather 的单 slice staging 相反——你发布的数据是 `P` 个 slice，消费的数据是一个。
- **归约是本地循环。** remote 读取是*加法*而不是存储：`acc` 从每个对端累加分块 `my_rank`。循环顺序在不同 rank 间不同，这正是 golden 使用容差的原因（归约顺序与 torch 不同）。

揭示用一次调用替换阶段 2–3：

```python
    for c in pl.range(nr):
        chunk = pl.load(x, [0, c * SIZE], [1, SIZE])
        data = pl.store(chunk, [c, 0], data)

    data = pld.tensor.reduce_scatter(data, signal, op=pld.ReduceOp.Sum)
    acc = pl.load(data, [my_rank, 0], [1, SIZE])
    return pl.store(acc, [0, 0], y)
```

- **`op=` 选择归约方式**——完整的 `ReduceOp` 家族（`Sum`、`Max`、`Min`、`Prod`）。手工版本硬编码了 `pl.add`；内置原语把归约变成参数。
- **window 的行 `my_rank` 就是你的归约分块**——与手工版本相同的行-分块布局。

### IR 对比（教学工件）

- `--mode hand` lowering 为上述四个阶段：`P` 次存储、barrier、以及用加法累加的 `P-1` 次 remote load。
- `--mode builtin` 展开为相同形状：你的分块被 staging、就绪 barrier、以及你行上的跨 rank 归约。composite 就是把归约参数化的手工调度——没有隐藏内容。

**成本卡（每 rank）：** 你收到 `(P-1)/P · N` 字节，最终得到 `N/P` 个归约元素——两相 all-reduce（步骤 09）的第一半，其第二半（allgather）你已在步骤 13 构建。

## 边界情况

> **致命陷阱——从错误位置归约自己的分块。** 累加必须从*你的* window 行开始（其中包含你自己的贡献），然后加上每个对端的行。如果你只归约 remote 行，就会缺少自己的贡献，golden 会按已知量失败。**修复：** 在对端循环之前用 `pl.load(data, [my_rank, 0], ...)` 初始化 `acc`。

| 症状 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| golden 差自己的分块 | 累加器从零而非自己的行开始 | 对端循环前加载 `[my_rank, 0]` |
| 每个 rank 得到相同分块 | 归约行固定为 `[0, 0]` | 归约行 `my_rank` |
| 分块边界错误 | 分块偏移计算错误 | `x` 中分块 `c` 在 `[0, c*SIZE]`，`data` 中行 `c` |
| 结果与 torch 不同（容差内） | 每 rank 归约顺序不同 | 用容差比较，而非精确相等 |
| 编译期不认识 `op=` | `ReduceOp` 名字错误 | 使用 `pld.ReduceOp.Sum/Max/Min/Prod` |

## 另请参阅

- [05-tutorials](05-tutorials.md) — 教程索引（本步 = 第 14 行）
- [01-collectives](../distributed/01-collectives.md) §ReduceScatter — 完整 API
- [14-allreduce_two_phase](14-allreduce_two_phase.md) — 本步是其第一半的两相 all-reduce
- 下一步：[20-all_to_all](20-all_to_all.md) — 给每个对端一个不同的 slice
