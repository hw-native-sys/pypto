# AllGather：全对全切片

每个 rank 发布自己的 slice，每个 rank 最终得到所有 slice 的按 rank 顺序拼接——两相 all-reduce 的 all-gather 一半——然后内置原语一次调用完成。

> **前置：** [17-broadcast](17-broadcast.md)。任意 ≥ 2 个设备（示例使用 2 和 4 个 sim 设备）。

**建议阅读顺序：** 01 → … → 12 → **13** —— 本页是步骤 13。

## 思路

Allgather 反转了 broadcast 的不对称性：每个 rank 既是生产者也是消费者。每个 rank 贡献一个 slice（`N/P` 个元素），每个 rank 最终得到**按 rank 顺序的拼接** `[x[0], x[1], …, x[P-1]]`。

| 方面 | AllGather |
| ---- | --------- |
| 输入 | 每个 rank 的 slice |
| 输出 | 所有 slice 的拼接，位于**每个** rank |
| 模式 | staging 自己的 slice → barrier → 读取每个对端的 slice |
| 开销 | 每个 rank 向每个对端发送 `N/P`：收到 `(P-1)/P · N` |

你以前见过这个模式：步骤 09 的两相 all-reduce 就是 reduce-scatter **后接 allgather**。本步单独构建 allgather 一半；步骤 14 构建 reduce-scatter 一半。

## 运行

```bash
# 手工：staging、barrier、remote_load 每个对端。
python examples/distributed/13_allgather.py -p a2a3sim -d 0,1

# 揭示：一次调用完成 pld.tensor.allgather。
python examples/distributed/13_allgather.py -p a2a3sim -d 0,1 --mode builtin

# 同一源码在 P=4：
python examples/distributed/13_allgather.py -p a2a3sim -d 0,1,2,3
python examples/distributed/13_allgather.py -p a2a3sim -d 0,1,2,3 --mode builtin
```

预期输出：

```text
OK
```

golden 是按 rank 顺序的拼接——在每个 rank 上完全相同——因此任何产生*错误顺序*（或自己 slice）的 rank 都会失败。

## 走读

两种模式共享一个 `[nr, SIZE]` window：每个 rank 在自己的行 staging，并读回每一行。手工内核：

```python
@pl.function(type=pl.FunctionType.InCore)
def hand_step(self, x, y, data, signal):
    ctx = pld.get_comm_ctx(data)
    my_rank = pld.rank(ctx)

    # Phase 1 — stage this rank's slice into its own row.
    local = pl.load(x, [0, 0], [1, SIZE])
    data = pl.store(local, [my_rank, 0], data)

    # Phase 2 — barrier: notify every peer, wait on every peer slot.
    for peer in pl.range(nr):
        if peer != my_rank:
            pld.system.notify(signal, peer=peer, offsets=[my_rank, 0],
                              value=1, op=pld.NotifyOp.AtomicAdd)
    for src in pl.range(nr):
        if src != my_rank:
            pld.system.wait(signal, offsets=[src, 0], expected=1,
                            cmp=pld.WaitCmp.Ge)

    # Phase 3 — gather: pull every peer's row into the output.
    for peer in pl.range(nr):
        recv = pld.tile.remote_load(data, peer=peer, offsets=[peer, 0], shape=[1, SIZE])
        y = pl.store(recv, [0, peer * SIZE], y)
    return y
```

- **行 `my_rank` 是你的槽位。** 在 `[my_rank, 0]` 而不是 broadcast 的单一根槽位 staging，这正是交换对称的原因：每个 rank 写不同的行，因此没有两个 rank 会冲突。
- **gather 是对对端的循环**——对每个对端 `p` 在行 `p` 处 `remote_load`，存入输出偏移 `p * SIZE`。输出是按 rank 顺序的拼接，这正是循环顺序（和偏移计算）重要的原因：槽位 `p` 必须保存 rank `p` 的 slice。

揭示用一次调用替换阶段 2–3——push 形式：

```python
    data = pld.tensor.allgather(x, data, signal)   # stage + barrier + gather

    for src in pl.range(nr):
        chunk = pl.load(data, [src, 0], [1, SIZE])
        y = pl.store(chunk, [0, src * SIZE], y)
```

- **源是你的本地 `x`（普通 `pl.Tensor`），不是 window。** push 形式的 allgather 替你 staging；目标 window 变成 `[nr, SIZE]` 结果（行 `src` = rank `src` 的 slice）。
- **与手工版本相同的行布局**——内置原语选择相同的调度。

### IR 对比（教学工件）

- `--mode hand` lowering 为上述三个阶段：一次写入你的行、notify/wait barrier、以及 `P` 次 `remote_load`——每个对端一次。
- `--mode builtin` 展开为相同形状：你的 slice 被 staging、`[nr, 1]` 信号上的就绪 barrier、以及逐对端读取。composite 没有增加任何东西——它就是手工调度，由编译器表达。（对更大的 slice，lowering 可能会分块传输，即步骤 11 提到的流水线；这里的小 slice 使用整块形式。）

**成本卡（每 rank）：** 每个 rank 向每个对端发送 `N/P` 字节，因此每个 rank 收到 `(P-1)/P · N` 字节——两相 all-reduce（步骤 09）的 gather 一半，其每一相也都移动 `(P-1)/P · N`。

## 边界情况

> **致命陷阱——把数据 gather 到错误的槽位。** 如果输出偏移与对端 rank 不匹配（从 peer `p` 读出的 `y[p]` 却按 `p+1` 偏移），每个 rank 内部自洽但 golden 仍然失败——顺序就是契约。**修复：** peer `peer` 对应偏移 `peer * SIZE`。

| 症状 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| 行顺序错误 | 输出偏移 ≠ 对端 rank | 把 peer `p` 存入 `[0, p * SIZE]` |
| 每个 rank 显示自己的 slice | 读取自己的 window 而非对端 | 对每个 `peer` 在 `[peer, 0]` 处 `remote_load` |
| `pld.tensor.allgather` 源被拒绝 | 传入的是 tile | 传普通 `pl.Tensor`（或 `DistributedTensor`） |
| 拼接有缺口/重叠 | stage/gather 偏移不匹配 | stage 行 `my_rank`；读行 `peer` |
| P=4 时数据陈旧 | stage 与 gather 之间缺少 barrier | 读取循环前 notify/wait 覆盖全部 `nr` 个对端 |

## 另请参阅

- [05-tutorials](05-tutorials.md) — 教程索引（本步 = 第 13 行）
- [01-collectives](../distributed/01-collectives.md) §AllGather — 完整 API
- [14-allreduce_two_phase](14-allreduce_two_phase.md) — 本步隔离出的两相 all-reduce 的 gather 一半
- 下一步：[19-reduce_scatter](19-reduce_scatter.md) — 归约一半
