# 三层编程模型（The Three-Level Programming Model）

标注分布式程序的三个层级——主机编排器、每设备编排、设备 kernel——并看清
每一层由哪个处理器运行。

> **前置条件：** [06-hello_rank](06-hello_rank.md)。两个设备。

**建议阅读顺序（Suggested reading order）：** 01 → **02** → 03 → 04 → 05 → 06 — 本页为步骤 02。

## 思路（The idea）

一个 `pld` 程序不是一个函数，而是三个，模型的关键在于*谁在哪运行*：

| 层级 | 装饰器 | 运行位置 | 职责 |
| ---- | ------ | -------- | ---- |
| 主机编排器 | `@pl.jit.host` | 主机 CPU | 分配 window buffer、遍历 rank、分发 |
| 每设备编排 | `@pl.jit` | AICPU（每个设备） | 每设备一次调用；转发参数、返回结果 |
| 设备 kernel | `@pl.jit.incore` | NPU AI 核 | 实际计算，运行在某个设备上 |

步骤 01 隐式使用了同样的三层；本步骤为它们标注名字，并展示一个值如何沿
链条向下流动。kernel 计算 `y[r] = x[r] * (r+1)`——与之前相同的计算形态，
但现在每一层都被显式标注并在代码中解释。

## 运行（Run it）

```bash
python examples/distributed/02_programming_model.py -p a2a3sim -d 0,1
```

预期输出：

```text
OK
```

## 走读（Walkthrough）

三个函数，自上而下。

```python
@pl.jit.incore
def scale_by_rank(
    x: pl.Tensor[[1, SIZE], pl.FP32],
    y: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    rank: pl.Scalar[pl.INT32],
):
    tile = pl.load(x, [0, 0], [1, SIZE])
    rank_f32 = pl.cast(rank, target_type=pl.FP32)
    scaled = pl.mul(tile, rank_f32)
    result = pl.add(scaled, tile)   # x * rank + x == x * (rank + 1)
    y = pl.store(result, [0, 0], y)
    return y
```

这是**设备 kernel**。它运行在某个设备的 AI 核上，只看到该设备的问题切片
——此处的 `x` 是 `[1, SIZE]`，即 rank-`r` 的切片。注意与步骤 01 相同的标量
纪律：`rank` 是标量 `INT32`，先转成 `FP32` 再折入向量运算。

```python
@pl.jit
def per_rank(x, y, rank):
    return scale_by_rank(x, y, rank)
```

**每设备编排**包装。它在 AICPU 上运行，每设备一份，作用是让主机分发一个
设备级函数，而不是直接触碰 kernel。本示例中它只是透传；后续步骤中它是
每设备 staging 和多调用序列的所在之处。

```python
@pl.jit.host
def scale_program(
    x: pl.Tensor[[N_RANKS, 1, SIZE], pl.FP32],
    y: pl.Out[pl.Tensor[[N_RANKS, 1, SIZE], pl.FP32]],
):
    for r in pl.range(pld.world_size()):
        per_rank(x[r], y[r], r, device=r)
```

**主机编排器**。它在主机 CPU 上运行，持有 world 形状的张量
`[N_RANKS, 1, SIZE]`，是唯一知道所有设备的函数。它遍历 world，为每个 rank
分发一次 `per_rank`。

编译/运行形态与步骤 01 相同：

```python
compiled = scale_program.compile(
    x, y,
    config=RunConfig(
        platform=args.platform,
        distributed_config=DistributedConfig(device_ids=[0, 1], num_sub_workers=0),
    ),
)
compiled(x, y, config=RunConfig(platform=args.platform))
assert torch.allclose(y, x * torch.arange(1, N_RANKS + 1).view(N_RANKS, 1, 1), ...)
```

golden `y == x * (r+1)` 检查每个 rank 用*自己的*索引缩放*自己的*切片。

## 边界情况（Edge cases）

> **致命陷阱——在错误的层级使用标量作为 rank 身份。** 只有主机循环在分发
> 时知道 rank 索引。若在主机函数体内（循环之外）读取 rank，或依赖模块级
> 全局的 per-rank 值，则所有 rank 得到相同的值。**修复：** 像上面那样，把
> `r` 从主机循环经包装传入 kernel。

| 症状 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| 所有 rank 计算同一 slice | 未从主机循环传递 rank 索引 | 通过 `per_rank(..., r, ...)` 向下传 `r` |
| 主机函数没有 rank 参数 | 混淆了编排器与 kernel | 主机*就是*循环；身份来自 `device=r` |
| `@pl.jit.incore` 从未在主机上运行 | 忘记哪个装饰器做什么 | kernel = `@pl.jit.incore`；设备包装 = `@pl.jit`；编排器 = `@pl.jit.host` |
| 每个 rank 计算了两次 | 分发的是 kernel 而非包装 | 主机必须调用 `@pl.jit` 包装，而非直接调用 `@pl.jit.incore` |

## 参见（See also）

- [05-tutorials](05-tutorials.md) — 教程总览（本步骤 = 第 02 行）
- [00-model](../distributed/00-model.md) — 模型词汇，L2 与 L3
- [03-execution](../distributed/03-execution.md) — 每层级的 worker 生命周期
- 下一步：[08-window_buffer](08-window_buffer.md) — 内存基座
