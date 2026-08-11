# 组合：Broadcast + AllReduce + AllGather

在一个内核中使用三种集合通信——本教程阶梯的收官之作，也是通往真实模型的桥梁。

> **前置：** [17-broadcast](17-broadcast.md) · [18-allgather](18-allgather.md) · [20-all_to_all](20-all_to_all.md)。任意 ≥ 2 个设备（示例使用 2 和 4 个 sim 设备）。

**建议阅读顺序：** 01 → … → 15 → **16** —— 本页是步骤 16。

## 思路

之前每一步都孤立地教授一个抽象。本步是第一次让一个内核做*不止一种*集合通信——也是第一次字节数不再是重点。真实模型正是这样做的：权重被广播、激活被 allreduce、结果被 allgather。下面的内核就是 picotron `model.py` 思想的缩影。

流水线：

1. **Broadcast**（步骤 12）——根的权重 `w` 到达每个 rank。
2. **Allreduce**（步骤 08–11）——每个 rank 得到 `Σ_k x[k]`。
3. **Allgather**（步骤 13）——每个 rank 得到 `concat(x[0], …, x[P-1])`。
4. **本地计算**——用共享权重 `w` 缩放 gather 后的矩阵（对 gather 到的隐藏状态施加一个学到的逐特征权重）。

## 运行

```bash
# 两个 rank：
python examples/distributed/16_putting_it_together.py -p a2a3sim -d 0,1

# 四个 rank——同一源码，只改 -d：
python examples/distributed/16_putting_it_together.py -p a2a3sim -d 0,1,2,3
```

预期输出：

```text
OK
```

golden 校验**两个**阶段：每个 rank 上 `allred[r] == Σ_k x[k]`，以及 `gathered[r] == concat(x[0], …, x[P-1]) * w`——被广播权重缩放的 allgather 结果，这也证明了权重到达了每个 rank。

## 走读

内核很短——三次内置调用加一次本地乘法——因为阶梯已经完成了工作：

```python
@pl.function(type=pl.FunctionType.InCore)
def compose_step(self, x, w_in, allred, gathered, w_data, ar_data, ag_data,
                 sig_bcast, sig_ar, sig_ag):
    ctx = pld.get_comm_ctx(w_data)
    my_rank = pld.rank(ctx)

    # 1 — Broadcast: root stages its weights, every rank gets them.
    if my_rank == ROOT_RANK:
        local_w = pl.load(w_in, [0, 0], [1, SIZE])
        w_data = pl.store(local_w, [0, 0], w_data)
    w_data = pld.tensor.broadcast(w_data, sig_bcast, root=ROOT_RANK)
    w = pl.load(w_data, [0, 0], [1, SIZE])

    # 2 — Allreduce: every rank ends with the element-wise sum.
    local_x = pl.load(x, [0, 0], [1, SIZE])
    ar_data = pl.store(local_x, [0, 0], ar_data)
    ar_data = pld.tensor.allreduce(ar_data, sig_ar, op=pld.ReduceOp.Sum, mode="mesh")
    total = pl.load(ar_data, [0, 0], [1, SIZE])
    allred = pl.store(total, [0, 0], allred)

    # 3 — Allgather: every rank ends with all ranks' raw slices.
    ag_data = pld.tensor.allgather(x, ag_data, sig_ag)

    # 4 — Local: scale the gathered matrix by the shared weight.
    for src in pl.range(nr):
        chunk = pl.load(ag_data, [src, 0], [1, SIZE])
        chunk = pl.mul(chunk, w)
        gathered = pl.store(chunk, [0, src * SIZE], gathered)
    return gathered
```

- **每种集合通信一个全新的 signal。** 信号计数器是单调且不会自动复位的，因此三个连续集合通信各得一个独立的 `[nr, 1]` window（`sig_bcast`、`sig_ar`、`sig_ag`）。跨调用复用同一个信号会让第二个 wait 提前通过。
- **`mode="mesh"` 显式给出**——步骤 11 的揭示让 mode 成为选择；这里点名写出，让读者看到完整调用。
- **allgather 的源是普通 `x` tensor**（步骤 13 的规则），而 broadcast 与 allreduce 接收 window——这三次调用在一处展示了 `pld.tensor.*` API 的完整面貌。
- **本地步骤是集合通信与数学相遇之处。** `chunk * w` 是 gather 后 tile 上的普通 `pl.mul`——步骤 01 的同一向量操作，如今作用于来自三个不同 rank 的数据。

### IR 对比（教学工件）

Lowering 后的 IR 是你已熟悉的三个手工调度，按顺序：broadcast 的根 staging + barrier + 读取（步骤 12）、allreduce 的 mesh barrier + 累加（步骤 08）、以及 allgather 的 staging + barrier + 逐对端读取（步骤 13）。三种内置原语之所以能组合，是因为每个的 lowering 在其自己的信号 window 上自包含——这正是全新信号规则重要的原因。

**成本卡（每 rank）：** 各组成部分之和——broadcast `(P-1)·N`、mesh allreduce `2·(P-1)/P·N`、allgather `(P-1)/P·N`。第一次，字节数不是重点：重点是三种调度组合进一个内核。

## 边界情况

> **致命陷阱——跨集合通信复用同一个信号。** 信号计数器单调且永不复位，因此第二个集合通信在同一 `[nr, 1]` window 上会看到已满足 `Ge(1)` 的计数器，其 wait 立即通过——一个静默缺失的 barrier。**修复：** 每种集合通信一个全新的信号 window，如内核所示。

| 症状 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| 第二/第三个集合通信提前通过 | 跨调用复用了信号 | 每种集合通信一个 `[nr, 1]` 信号 |
| `allred` 错但 `gathered` 对 | allreduce 源未 staging / op 错误 | 把 `x` staging 进 `ar_data`；`op=Sum`；`mode="mesh"` |
| `gathered` 错但 `allred` 对 | 未应用广播权重，或行错误 | 对每行 `chunk = pl.mul(chunk, w)` |
| `pld.tensor.allgather` 源被拒绝 | 传的是 tile 而非 tensor | 传普通 `x` tensor |
| 非根权重泄漏到输出 | 根未 staging | 仅在 `if my_rank == ROOT_RANK` 下 staging `w_data` |

## 另请参阅

- [05-tutorials](05-tutorials.md) — 教程索引（本步 = 第 16 行）
- [01-collectives](../distributed/01-collectives.md) — 整个集合通信动物园
- [17-broadcast](17-broadcast.md) / [18-allgather](18-allgather.md) / [20-all_to_all](20-all_to_all.md) — 本内核所组合的组件
- 更高级的应用（此处不重述）：pypto-lib [#869](https://github.com/hw-native-sys/pypto-lib/pull/869)（AllGather-GEMM）与 DeepSeek-V4 分布式 MoE dispatch/combine——模型规模下的相同模式
- 这是阶梯的终点——索引按顺序列出了全部内容。
