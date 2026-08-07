# Put 与 Get：Tensor 级 Push 与 Pull

用 `pld.tensor.put` 与 `pld.tensor.get` 在 rank 之间移动整个 window slice——
tensor 级点对点移动，push 与 pull。

> **前置条件：** [10-remote_load_store](10-remote_load_store.md)。两个设备。

**建议阅读顺序（Suggested reading order）：** 01 → 02 → 03 → 04 → 05 → **06** — 本页为步骤 06。

## 思路（The idea）

步骤 05 用 `remote_load`/`remote_store` 一次移动一个 *tile*。tensor 级原语
`pld.tensor.put` 与 `pld.tensor.get` 一次调用即可在 rank 之间移动整段 window
内存 slice，运行时负责 staging，对于大传输还负责分块与流水线化。

区别在于**谁发起**：

| 原语 | 方向 | 发起方 | 结果 |
| ---- | ---- | ------ | ---- |
| `pld.tensor.put(dst, peer, src, atomic=...)` | Push | **发送方** | 发送方的 `src` slice 落在对端的 `dst` |
| `pld.tensor.get(dst, peer, src)` | Pull | **接收方** | 对端的 `src` slice 落在接收方的 `dst` |

示例与步骤 05 是同样的环形移位，一步完成：`--mode put` 推入下一个 rank，
然后读回自己的 `dst` → `y[r] = x[(r-1) % N]`；`--mode get` 拉取下一个 rank
的 `src` → `y[r] = x[(r+1) % N]`。

**成本卡片：** 一步，每个 rank 与一个对端交换一个 slice。小 slice 时为延迟
受限；大 slice 时运行时的分块 + 流水线 staging 会重叠各轮，把延迟受限的
移动变成带宽受限（与步骤 07–09 用于 all-reduce 的技巧相同）。

## 运行（Run it）

```bash
# Push（默认）：
python examples/distributed/06_put_get.py -p a2a3sim -d 0,1

# Pull：
python examples/distributed/06_put_get.py -p a2a3sim -d 0,1 --mode get
```

预期输出：

```text
OK
```

## 走读（Walkthrough）

put 侧：

```python
@pl.jit.incore
def put_step(x, y, src, dst, signal):
    ctx = pld.get_comm_ctx(src)
    my_rank = pld.rank(ctx)
    nranks = pld.nranks(ctx)

    local = pl.load(x, [0, 0], [1, SIZE])
    src = pl.store(local, [0, 0], src)

    peer = (my_rank + 1) % nranks
    pld.tensor.put(dst, peer=peer, src=src, atomic=pld.AtomicType.None_)

    pld.system.notify(signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)
    pld.system.wait(signal, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Ge)

    recv = pl.load(dst, [0, 0], [1, SIZE])
    y = pl.store(recv, [0, 0], y)
    return y
```

- **Stage。** 发送方把本地 slice 复制进自己的 `src` window slice——`put`
  移动的是 *window* 内存到 *window* 内存，因此源必须先进入 window。
- **Put。** `pld.tensor.put(dst, peer=peer, src=src, atomic=pld.AtomicType.None_)`
  把 `src` 推入对端的 `dst`。`atomic` 选择更新模式；`None_` 表示无条件覆盖
  （简单情形）。
- **信号、等待、读取。** put 之后，发送方通知对端，并等待那个以*它*为目标的
  rank——然后读取自己的 `dst`，它刚被上一个 rank 写入。

get 侧是接收方发起的镜像：

```python
    local = pl.load(x, [0, 0], [1, SIZE])
    src = pl.store(local, [0, 0], src)

    peer = (my_rank + 1) % nranks
    pld.system.notify(signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)
    pld.system.wait(signal, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Ge)

    pld.tensor.get(dst, peer=peer, src=src)

    recv = pl.load(dst, [0, 0], [1, SIZE])
    y = pl.store(recv, [0, 0], y)
```

这里每个 rank 仍然 staging 自己的 `src` 并发送信号——但*移动*是
`pld.tensor.get(dst, peer=peer, src=src)`：握手之后，接收方把对端的 `src`
拉入自己的 `dst`。同一个环，发起方相反。

**分块与流水线。** 对大传输，运行时把 slice 分成块并流水线化移动，使下一
块的 staging 与当前块的传输重叠。本示例中的小传输使用整 slice 形式；分块
大小规则见参考章节。

## 边界情况（Edge cases）

> **致命陷阱——不带信号的 put。** 从发送方视角 `put` 是即发即忘；发送方
> 必须 notify，接收方必须在读取目的之前 wait，否则读取会与传输竞争。
> **修复：** 对每个 `put`/`get`，在发送方配一个 notify，并在读取 `dst` 之前
> 在接收方配一个 wait。

| 症状 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| 接收方读到陈旧的 `dst` | put/get 与信号握手竞争 | 移动之后 notify；读取 `dst` 之前 wait |
| 数据落到错误的 rank | `peer` 计算错误 | `put` 写入 `(r+1) % n`；`get` 从 `(r+1) % n` 读取 |
| golden 落后一步 | pull 与 push 混淆 | `put` 模式：`y[r] = x[(r-1)]`；`get` 模式：`y[r] = x[(r+1)]` |
| 大传输停滞或越界 | 忽略分块规则 | 遵循章节中的分块大小与流水线约束 |
| 省略 `atomic` 参数 | 默认不总是覆盖 | 普通覆盖传入 `atomic=pld.AtomicType.None_` |

## 参见（See also）

- [05-tutorials](05-tutorials.md) — 教程总览（本步骤 = 第 06 行）
- [02-primitives](../distributed/02-primitives.md) §Put 与 Get — 分块与流水线
  约束
- [01-collectives](../distributed/01-collectives.md) — 集合通信如何组合这些
  移动（步骤 07–15）
- 下一步：[05-tutorials](05-tutorials.md) — 步骤 07–15（all-reduce）为规划中
