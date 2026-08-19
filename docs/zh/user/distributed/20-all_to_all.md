# AllToAll：给每个对端一个不同的 slice

每个 rank 向每个对端发送一个*不同的* slice，并从每个对端收到一个不同的 slice——点对点模式中最通用的一种——然后内置原语一次调用完成。

> **前置：** [19-reduce_scatter](19-reduce_scatter.md)。任意 ≥ 2 个设备（示例使用 2 和 4 个 sim 设备）。

**建议阅读顺序：** 01 → … → 14 → **15** —— 本页是步骤 15。

## 思路

之前所有集合通信都移动*相同*形状的数据：broadcast 把一个 slice 送到所有地方，allgather 把每个 rank 的 slice 送到所有地方，reduce-scatter 做归约。All-to-all 则不同：rank `r` 向每个对端发送一个**不同的 `N/P` slice**——给目标 `d` 的 slice 不同于给目标 `e` 的。这就是个性化交换。

| 方面 | AllToAll |
| ---- | -------- |
| 输入 | 每个 rank 有 `P` 个不同分块（每个目标一个） |
| 输出 | 每个 rank 有 `P` 个不同分块（每个来源一个） |
| 模式 | push 每个目标分块（`put`）→ barrier → 读回 |
| 开销 | 收到 `(P-1)/P · N` 字节；没有任何两个 rank 想要相同的字节 |

这是真实 dispatch/combine 工作负载背后的模式——分布式 MoE 把每个 token 组发给不同专家的 rank；AllGather-GEMM 流水线按分片路由数据。本页 See-also 按名称链接了这些应用（不重述其内容）。

## 运行

```bash
# 手工：put 每个目标分块、barrier、读回。
python examples/distributed/15_all_to_all.py -p a2a3sim -d 0,1

# 揭示：一次调用完成 pld.tensor.all_to_all。
python examples/distributed/15_all_to_all.py -p a2a3sim -d 0,1 --mode builtin

# 同一源码在 P=4：
python examples/distributed/15_all_to_all.py -p a2a3sim -d 0,1,2,3
python examples/distributed/15_all_to_all.py -p a2a3sim -d 0,1,2,3 --mode builtin
```

预期输出：

```text
OK
```

golden 的构造使每个分块唯一：`input[r, d, j] = r*1000 + d*100 + j`——来源、目标、元素都编码在值中。任何路由错误（错误来源、错误目标）都会显示为错误值，而不是微妙的形状问题。

## 走读

两种模式共享一个 `[nr, SIZE]` window。Rank `r` 在目标 `d` 的 window 的行 `r` 写入其给 `d` 的分块，然后读取自己 window 的行 `src`。手工内核使用步骤 06 的 `put`：

```python
@pl.function(type=pl.FunctionType.InCore)
def hand_step(self, x, y, data, signal):
    ctx = pld.get_comm_ctx(data)
    my_rank = pld.rank(ctx)

    # Phase 1 — push: write chunk-for-dest into dest's window at our row.
    for dest in pl.range(nr):
        pld.tensor.put(data, dest, x, [my_rank, 0], [dest, 0], [1, SIZE])

    # Phase 2 — barrier: notify every peer, wait on every peer slot.
    for peer in pl.range(nr):
        if peer != my_rank:
            pld.system.notify(signal, peer=peer, offsets=[my_rank, 0],
                              value=1, op=pld.NotifyOp.Set)
    for src in pl.range(nr):
        if src != my_rank:
            pld.system.wait(signal, offsets=[src, 0], expected=1,
                            cmp=pld.WaitCmp.Ge)

    # Phase 3 — read-back: row src of our window holds src's chunk for us.
    for src in pl.range(nr):
        chunk = pl.load(data, [src, 0], [1, SIZE])
        y = pl.store(chunk, [src, 0], y)
    return y
```

- **push 是对目标的循环。** `pld.tensor.put(data, dest, x, [my_rank, 0], [dest, 0], [1, SIZE])` 把 `x` 的行 `dest` 写入对端 `dest` 的 window、位于*我们的*行。每次迭代以不同的源行、不同的对端为目标——这就是“个性化”所在。
- **notify 用 `Set` 而非 `AtomicAdd`。** 每个 rank 是其行的唯一写入者（`[my_rank, 0]`），因此这里的 barrier 使用步骤 04 走读中的单写入者形式 `Set`/`Ge(1)`。
- **读回完成交换。** barrier 之后，我们自己 window 的行 `src` 保存着 rank `src` 发给我们的分块。

揭示用一次调用替换阶段 1–3：

```python
    result = pld.tensor.all_to_all(x, data, signal)
    for src in pl.range(nr):
        chunk = pl.load(result, [src, 0], [1, SIZE])
        y = pl.store(chunk, [src, 0], y)
    return y
```

- **源是你的本地 `x`（普通 `pl.Tensor`）**，行 `d` = 给目标 `d` 的分块——与手工循环相同的布局。
- **window 变成结果**（行 `src` = 来自 rank `src` 的分块），且 `input`/`target` 必须是**不同**的缓冲。

### IR 对比（教学工件）

- `--mode hand` lowering 为上述三个阶段：`P` 次 put（每个目标一次）、`Set`/`Ge(1)` barrier、以及读回循环。
- `--mode builtin` 展开为相同形状：你的逐目标分块被 push、就绪 barrier、以及行读回。没有隐藏内容——composite 就是由编译器表达的手工个性化交换（HOST 内置原语会对大传输增加编排级分块）。

**成本卡（每 rank）：** 每个 rank 向每个对端发送一个*不同的* `N/P` slice——收到 `(P-1)/P · N` 字节，且没有两个 rank 收到相同的字节。这正是 all-to-all 是最难调度的点对点模式的原因：每对都交换不同的数据。

## 边界情况

> **致命陷阱——源与结果复用同一个 window。** All-to-all 是就地*交换*，因此 `x` 和 `data` 必须是不同的缓冲。把同一 window 同时作为源与目标，会让 push 覆盖你仍需发送的分块。**修复：** 分配一个独立的目标 window（如示例所做）——内置原语也强制这一点。

| 症状 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| 给目标 `d` 的分块落入错误的对端 | `put` 的对端 ≠ 目标行 | `put(data, dest, x, ..., [dest, 0], ...)` |
| 输出是我自己的分块而非对端的 | 读回自己的行 | barrier 后读行 `src` |
| 源/目标混叠损坏 | 输入与目标用同一缓冲 | 分开 `x` 与 `data` 两个 window |
| 某个 rank 挂起 | `Set` notify/wait 不匹配 | notify 写对端的行 `my_rank`；wait 读自己的行 `src` |
| 仅 P=4 时值错误 | 读回前缺少 barrier | put 循环与读循环之间加 barrier |

## 另请参阅

- [05-tutorials](05-tutorials.md) — 教程索引（本步 = 第 15 行）
- [01-collectives](../distributed/01-collectives.md) §AllToAll — 完整 API
- [03-execution](../distributed/03-execution.md) — 生产级 all-to-all 的 `DistributedWorker` 与设备端 staging
- `examples/runtime/distributed_callback.py` — 围绕 L3 分布式程序的宿主端运行时绑定回调
- [11-put_get](11-put_get.md) — 本步所依赖的 `put`/`get` 底层
- 更高级的应用（此处不重述）：pypto-lib [#869](https://github.com/hw-native-sys/pypto-lib/pull/869)（AllGather-GEMM，来自步骤 13 的 allgather 模式）与 DeepSeek-V4 分布式 MoE dispatch/combine（来自本步的 all-to-all 模式）
- 下一步：[21-putting_it_together](21-putting_it_together.md) — 在一个内核中组合三种集合通信
