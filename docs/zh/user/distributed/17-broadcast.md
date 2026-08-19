# Broadcast：一对多

根 rank 的 slice 到达每个 rank——权重、配置、全局行——然后内置原语一次调用完成。

> **前置：** [16-allreduce_reveal](16-allreduce_reveal.md)。任意 ≥ 2 个设备
> （示例使用 2 和 4 个 sim 设备）。

**建议阅读顺序：** 01 → … → 11 → **12** —— 本页是步骤 12。

## 思路

Broadcast 是最简单的、带有一个特殊 rank 的集合通信：**根（root）** 拥有数据，其他每个 rank 最终都得到它的副本。这就是“加载一次权重、到处共享”的模式——之后每个 rank 的数据都相同，因此无需归约。

| 方面 | Broadcast |
| ---- | --------- |
| 输入 | 仅根的数据（其他 rank 的输入被忽略） |
| 输出 | **每个** rank 都得到根的 slice |
| 模式 | 根 staging → barrier → 每个 rank 读取根 |
| 开销 | 根写入 `N` 字节，每个对端读取：共 `(P-1)·N` 字节，一步完成 |

你已有此所需的一切原语——步骤 04 的 barrier 与步骤 05 的 `remote_load`。唯一的新思想是 *读取目标固定*：所有人都读 rank 0。

## 运行

```bash
# 手工：根 staging、barrier、每个 rank remote_load 根。
python examples/distributed/12_broadcast.py -p a2a3sim -d 0,1

# 揭示：一次调用完成 pld.tensor.broadcast。
python examples/distributed/12_broadcast.py -p a2a3sim -d 0,1 --mode builtin

# 同一源码在 P=4（比较需要 >2 个 rank）：
python examples/distributed/12_broadcast.py -p a2a3sim -d 0,1,2,3
python examples/distributed/12_broadcast.py -p a2a3sim -d 0,1,2,3 --mode builtin
```

预期输出：

```text
OK
```

golden 精确校验契约：每个 rank 的输出等于根的 slice，**并且**任何非根 rank 的输入都没有泄漏到任何输出中。

## 走读

手工内核是来自底层原语的三阶段模式——没有新东西，只是固定了 `peer`：

```python
@pl.function(type=pl.FunctionType.InCore)
def hand_step(self, x, y, data, signal, root):
    ctx = pld.get_comm_ctx(data)
    my_rank = pld.rank(ctx)

    # Phase 1 — stage-in: root only writes its slice into the window.
    if my_rank == root:
        local = pl.load(x, [0, 0], [1, SIZE])
        data = pl.store(local, [0, 0], data)

    # Phase 2 — barrier: notify every peer, wait on every peer slot.
    for peer in pl.range(nr):
        if peer != my_rank:
            pld.system.notify(signal, peer=peer, offsets=[my_rank, 0],
                              value=1, op=pld.NotifyOp.AtomicAdd)
    for src in pl.range(nr):
        if src != my_rank:
            pld.system.wait(signal, offsets=[src, 0], expected=1,
                            cmp=pld.WaitCmp.Ge)

    # Phase 3 — broadcast: pull root's slice into local output.
    recv = pld.tile.remote_load(data, peer=root, offsets=[0, 0], shape=[1, SIZE])
    return pl.store(recv, [0, 0], y)
```

- **staging 是有条件的。** 只有 `my_rank == root` 才写入 window——对运行时 rank 标量的 `if` 是普通控制流分支。其他 rank 保持 window 不变并等待。
- **barrier 与步骤 04 相同**——专用行 `AtomicAdd`/`Ge(1)`。它确保没有任何 rank 在根 staging 之前 `remote_load` 根的 slice。
- **读取是 `remote_load(data, peer=root, ...)`**——步骤 05 的原语，只是把 peer 固定为根。Broadcast 没有任何新原语；它是对你已构建原语的一种*用法*。

揭示用一次调用替换阶段 2–3：

```python
    if my_rank == ROOT_RANK:
        local = pl.load(x, [0, 0], [1, SIZE])
        data = pl.store(local, [0, 0], data)

    data = pld.tensor.broadcast(data, signal, root=ROOT_RANK)
    acc = pl.load(data, [0, 0], [1, SIZE])
    return pl.store(acc, [0, 0], y)
```

- **`root=` 必须是编译期常量**（Python `int`，这里为 `ROOT_RANK = 0`），不能是运行时标量——lowering 需要静态知道根是谁。这也是手工内核把 `root` 作为标量、而内置原语不需要的原因。
- **非根 slot 的输入被忽略**——可以保持未初始化；调用后只读取根的 slot。golden 断言了这一点：非根输入从不泄漏。

### IR 对比（教学工件）

开启 pass dump 编译并对比两种模式的 lowering 后 IR：

- `--mode hand` 恰好 lowering 为上述三个阶段——一次对根的 `remote_load`，由 notify/wait barrier 保护。
- `--mode builtin` 得到相同形状：composite 展开为 `[nr, 1]` 信号上的就绪 barrier 加根读取。差异是“你的调度，由编译器表达”——没有额外内容，没有隐藏流量。

**成本卡（每 rank）：** 根写入 `N` 字节；每个对端读取——共 `(P-1)·N` 字节，一步完成。这是每字节最便宜的集合通信，因为每个 rank 想要的是*相同*的字节。

## 边界情况

> **致命陷阱——非根数据泄漏到输出。** 如果内核从 `my_rank` 自己的 slice 而不是根的 slice 广播，那么每个 rank 最终得到*自己的*数据，而 golden 会静默失败（每个 rank 内部自洽）。**修复：** 读取目标永远是根：`remote_load(data, peer=ROOT_RANK, ...)`。

| 症状 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| 每个 rank 输出自己的 slice | 从 `my_rank` 而非根广播 | 读取 `peer=ROOT_RANK` |
| 某些输出出现非根输入 | 根未 staging → window 陈旧 | 仅在 `if my_rank == root` 下 staging |
| 编译期拒绝 `root` kwarg | 向 `pld.tensor.broadcast` 传了运行时标量 | 传 Python `int` 常量 |
| remote 读取返回零 | 根 staging 之前就 `remote_load` / 无 barrier | load 模式：staging（根）→ barrier → load |
| P=4 时只有一行正确 | 手工读取与 barrier 竞争 | 确认 notify/wait 循环覆盖全部 `nr` 个对端 |

## 另请参阅

- [05-tutorials](05-tutorials.md) — 教程索引（本步 = 第 12 行）
- [01-collectives](../distributed/01-collectives.md) §Broadcast — 完整 API
- [02-primitives](../distributed/02-primitives.md) §Tile-Level RMA — 手工版本所依赖的 `remote_load`
- 下一步：[18-allgather](18-allgather.md) — 全对全切片
