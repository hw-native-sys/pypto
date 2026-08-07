# Window Buffer：内存基座（The Memory Substrate）

分配一个对称的 window buffer，将其视为 `DistributedTensor`，并读写自己的
slice——尚无通信，但后续每一步都通过这个对象移动数据。

> **前置条件：** [07-programming_model](07-programming_model.md)。两个设备。

**建议阅读顺序（Suggested reading order）：** 01 → 02 → **03** → 04 → 05 → 06 — 本页为步骤 03。

## 思路（The idea）

`pld` 中的分布式内存是**对称的**：每个 rank 在相同的虚拟地址分配*相同*的
window buffer，因此"该 buffer"是一个每个 rank 都能到达的对象——本地是
自己的 slice，对端通过 RMA 到达。window buffer 是一个 HCCL buffer，带有一个
**信号尾（signal tail）**，运行时为跨 rank 信号保留（步骤 04–06 会用到）。

两个调用创建它。`pld.alloc_window_buffer(...)` 分配 buffer；
`pld.window(...)` 调用给出它的 `pld.DistributedTensor` 视图——即对端可见的
类型。本步骤对一个 window 做最简单的事：加载自己的 slice，存回自己的
slice，再读一次。目前尚无任何共享，golden 为 `y == x`——但未来每一步
通信所经过的对象已经登场。

## 运行（Run it）

```bash
python examples/distributed/03_window_buffer.py -p a2a3sim -d 0,1
```

预期输出：

```text
OK
```

## 走读（Walkthrough）

```python
SIZE = 256          # 每 rank 1 KiB -- 低于 4 KiB window 下限

@pl.jit.incore
def window_roundtrip(
    x: pl.Tensor[[1, SIZE], pl.FP32],
    y: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    data: pld.DistributedTensor[[1, SIZE], pl.FP32],
):
    local = pl.load(x, [0, 0], [1, SIZE])
    data = pl.store(local, [0, 0], data)      # 写入自身 slice
    back = pl.load(data, [0, 0], [1, SIZE])   # 读回自身 slice
    y = pl.store(back, [0, 0], y)
    return y
```

kernel 的第三个参数是 `pld.DistributedTensor`——绑定 window 的类型。
对它做 `pl.store`/`pl.load` 读写的正是本 rank 在对称 window 中的 slice，
与本地张量无异。kernel 本身没有任何变化表明它是分布式的：是*类型*告诉
编译器这个 buffer 位于共享的 window 内存中。

```python
@pl.jit.host
def window_program(
    x: pl.Tensor[[N_RANKS, 1, SIZE], pl.FP32],
    y: pl.Out[pl.Tensor[[N_RANKS, 1, SIZE], pl.FP32]],
):
    data_buf = pld.alloc_window_buffer([1, SIZE], dtype=pl.FP32)
    for r in pl.range(pld.world_size()):
        data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
        per_rank(x[r], y[r], data, device=r)
```

**主机编排器拥有 window。** `alloc_window_buffer` 在分发循环之前运行一次
——每个 rank 的运行时在相同地址分配相同的 buffer。循环内部，
`pld.window(...)` 产生本 rank 的 `DistributedTensor` 视图，向下传给 kernel。

**4 KiB 下限。** 无论数据大小如何，window buffer 至少会被补齐到 4 KiB。
这里的 `[1, SIZE]` 的 `FP32` 为 `SIZE * 4 = 1 KiB` 每 rank，但 buffer 实际
花费 4 KiB——多出的空间是信号尾加对齐。这是分布式编程的第一个预算约束：
很小的 window 并不按其形状标价。

## 边界情况（Edge cases）

> **致命陷阱——把 window 当作普通张量。** `pl.Tensor` 与
> `pld.DistributedTensor` 是不同类型。在期望本地张量的地方传入
> `DistributedTensor`（或反之）会在编译期失败，而非运行期。**修复：** 任何
> 用 `alloc_window_buffer` 分配的内容都用 `pld.DistributedTensor[...]` 标注，
> 本地输入/输出保留 `pl.Tensor`。

| 症状 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| `alloc_window_buffer` 与本地 `Tensor` 类型混用 | window 视图与本地张量类型混淆 | window 参数标注为 `pld.DistributedTensor[...]` |
| 每 rank 数据 1 KiB 但 buffer 更大 | 4 KiB 下限（信号尾 + 对齐） | 每个 window 至少按 4 KiB 预算，而非 `size * dtype` |
| window 在 rank 循环内分配 | 每次分发都重新分配 | 把 `alloc_window_buffer` 提到循环之上；循环内只调用 `window(...)` |
| 将对端的 slice 当本地 load | 忘记 window 是共享的 | 本地 load 只能看到自己的 slice；对端需要 RMA（步骤 05–06） |

## 参见（See also）

- [05-tutorials](05-tutorials.md) — 教程总览（本步骤 = 第 03 行）
- [02-primitives](../distributed/02-primitives.md) §Window Buffer 管理 — 完整 API
- [00-model](../distributed/00-model.md) §术语表 — window buffer、信号
- 下一步：[09-barrier](09-barrier.md) — 仅信号，无数据
