# 你好，Rank（Hello, Rank）

运行你的第一个双 rank 程序：每个 rank 将自己的索引加到输出的对应 slice 上，
golden 证明每个 rank 恰好触碰了自己的那一行。

> **前置条件：** 教程总览见 [05-tutorials](05-tutorials.md)；词汇见
> [分布式编程](../distributed/index.md)章节。需要两个设备（或两个模拟设备）。
> 你的第一个 `pld` 程序不需要任何分布式经验——只需 [快速入门](../00-getting_started.md)
> 的基础。

**建议阅读顺序（Suggested reading order）：** **01** → 02 → 03 → 04 → 05 → 06 — 本页为步骤 01。

## 思路（The idea）

分布式程序在每个参与的设备上运行**相同的源码**，但每个设备需要知道*自己
是哪一个*，才能处理自己那份数据。这个身份就是 **rank**：启动时分配的
唯一索引。

Rank 身份流经三个层级。`@pl.jit.host` 函数是**编排器**——它在主机 CPU 上
运行，是唯一知道*所有*设备的地方。它遍历 world，并用 `device=r` 为每个
rank 分发一次每设备函数。每设备函数（`@pl.jit`）在 AICPU 上运行，并转发给
在 NPU AI 核上运行的 **InCore** kernel（`@pl.jit.incore`）。rank 索引作为
普通参数向下传递。

## 运行（Run it）

```bash
# 模拟器（CI 使用此方式）：
python examples/distributed/01_hello_rank.py -p a2a3sim -d 0,1

# 双卡硬件：
python examples/distributed/01_hello_rank.py -p a2a3 -d 0,1
```

预期输出：

```text
OK
```

`OK` 表示 golden 成立：对每个 rank `r`，`y[r] == x[r] + r`。

## 走读（Walkthrough）

kernel——每次讲解一个概念。

```python
N_RANKS = 2
ROWS = 8
COLS = 8

@pl.jit.incore
def add_rank(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    rank: pl.Scalar[pl.INT32],
):
    tile = pl.load(x, [0, 0], [ROWS, COLS])
    rank_f32 = pl.cast(rank, target_type=pl.FP32)
    tile = pl.add(tile, rank_f32)
    y = pl.store(tile, [0, 0], y)
    return y
```

- **张量在前，标量在后。** 签名是 `(x, y, rank)`——标量 `rank` 位于张量参数
  *之后*。颠倒顺序会在运行时失败：`TaskArgs: cannot add tensor after scalar`。
- **标量活在 AICPU 上。** `rank` 以 `INT32` 标量到达。kernel 将其转换为
  `FP32`，并折入*向量*运算（`x + rank`）。若写成 `rank_f32 + 1.0` 这样的
  标量算术，ptoas 会拒绝（`arith.addf explicitly marked illegal`）。
- **转换参数，而非表达式。** 对 `INT32` 参数使用 `pl.cast(rank, ...)` 是受支持
  的路径；对 index 类型的表达式（如 `rank + 1`）做 cast 则不支持
  （`Cast between float and index types is not supported`）。

每设备包装与编排器：

```python
@pl.jit
def per_rank(x, y, rank):
    return add_rank(x, y, rank)

@pl.jit.host
def hello_rank(
    x: pl.Tensor[[N_RANKS, ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[N_RANKS, ROWS, COLS], pl.FP32]],
):
    for r in pl.range(pld.world_size()):
        per_rank(x[r], y[r], r, device=r)
```

- `x` 和 `y` 携带 **world 形状** `[N_RANKS, ROWS, COLS]`——rank `r` 的 slice 是
  `x[r]` / `y[r]`。
- 主机循环运行 `pld.world_size()` 次——每个 rank 一次——并按 rank 切分
  world 张量。`device=r` 将分发 `r` 固定到设备 `r`。
- 主机函数不接收 `rank` 参数；它*就是*循环。Rank 身份在分发时注入。

测试框架：

```python
compiled = hello_rank.compile(
    x, y,
    config=RunConfig(
        platform=args.platform,
        distributed_config=DistributedConfig(
            device_ids=[0, 1],
            num_sub_workers=0,
        ),
    ),
)
compiled(x, y, config=RunConfig(platform=args.platform))
assert torch.allclose(y, x + torch.arange(N_RANKS).view(N_RANKS, 1, 1), ...)
```

`DistributedConfig(device_ids=[0, 1], num_sub_workers=0)` 声明两个设备且不
启用主机子 worker——最小的多 rank 配置。golden `y == x + r` **带容差**
（`allclose`）校验——计算是逐元素的，容差只是为后端浮点差异预留的余量；
若需要严格保证，可使用精确相等。

## 边界情况（Edge cases）

> **致命陷阱——张量后的标量。** 签名 `fn(x, rank, y)` 能编译，但运行时失败：
> `TaskArgs: cannot add tensor after scalar`。**修复：** 让每个标量参数都位于
> 所有张量参数之后：`fn(x, y, rank)`。

| 症状 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| `TaskArgs: cannot add tensor after scalar` | 子签名中标量参数位于张量参数之前 | 张量全部在前，标量全部在后 |
| `arith.addf explicitly marked illegal` | AI 核上的标量 `FP32` 算术 | 将常量折入向量运算（`x + rank`） |
| `Cast between float and index types is not supported` | 对 index 类型表达式做 `pl.cast` | 先转换 `INT32` 参数，再做向量浮点运算 |
| 只有某个 rank 的行结果错误 | 未使用 rank 索引 / 设备映射错误 | 检查主机循环使用 `device=r` 并切分 `x[r]` |
| 分发时挂起 | 设备 id 不可用 | 确认 `-d 0,1` 存在且空闲（`npu-smi info`） |

## 参见（See also）

- [05-tutorials](05-tutorials.md) — 教程总览（本步骤 = 第 01 行）
- [00-model](../distributed/00-model.md) — 快速入门 + 模型词汇
- [03-execution](../distributed/03-execution.md) — `DistributedConfig` 与
  worker 生命周期
- 下一步：[07-programming_model](07-programming_model.md) — 标注的三层模型
