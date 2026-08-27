# 混合 kernel

cube 与 vector 在同一个作用域里同时工作。

> **前置**：[分块 matmul](02-matmul.md)。
> **配套文件**：`examples/advanced/03_mixed_kernel.py`。

## 你要做的东西

`a @ b + bias` —— 一个 cube 运算接一个 vector 运算 —— 写成让两个单元重叠而非轮流的形式。

## 为什么值得

一个 core group 配一个 cube 单元与若干 vector 单元。按最直观的写法，这条链会一前一后占住它们：

```python
with pl.at(level=pl.Level.CORE_GROUP, name_hint="cube_only"):
    acc = pl.matmul(a, b, out_dtype=pl.FP32)
with pl.at(level=pl.Level.CORE_GROUP, name_hint="vector_only"):
    out[:] = pl.add(acc, bias)
```

两个作用域，两次派发。matmul 作用域没跑完之前 vector 单元无事可做，跑完之后 cube 单元又闲下来。这正是混合 kernel 要取代的写法 —— 也正是那些名为「融合」的 kernel 底下常常仍是的样子：名字上融合，执行上串行。

## 第 1 步：一个作用域，加 split

把两个运算放进同一个作用域，并标记为 split：

```python
import pypto.language as pl
import torch
from pypto.runtime import RunConfig

@pl.jit
def mixed(
    a: pl.Tensor[[128, 256], pl.FP16],
    b: pl.Tensor[[256, 128], pl.FP16],
    bias: pl.Tensor[[128, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    with pl.at(
        level=pl.Level.CORE_GROUP,
        optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
        name_hint="mixed",
    ):
        acc = pl.matmul(a, b, out_dtype=pl.FP32)                 # cube (AIC)
        out[:] = pl.add(acc, bias)        # vector (AIV)
    return out

torch.manual_seed(0)
a = torch.randn(128, 256, dtype=torch.float16)
b = torch.randn(256, 128, dtype=torch.float16)
bias = torch.randn(128, 128, dtype=torch.float32)
out = torch.zeros(128, 128, dtype=torch.float32)
mixed(a, b, bias, out, config=RunConfig(platform="a2a3sim"))
assert torch.allclose(out, a.float() @ b.float() + bias, rtol=1e-2, atol=1e-2)
```

`pl.split(mode)` 把作用域标记为混合，而 mode 指的是 **vector** 子区域沿哪个轴对半。cube 子区域保持全尺寸：split 把 vector 的工作分摊到两个 AIV 通道上，编译器则插入在两个单元间搬运结果的跨核传输（cube→vector 边界上的 `aiv_shard`，回程的 `aic_gather`）。重叠来自 cube 与 vector 并发执行，而不是各拿同一个 tile 的一半。

| 模式 | vector 子区域沿哪个方向对半 |
| ---- | --------------------------- |
| `pl.SplitMode.UP_DOWN` | 行（高度） |
| `pl.SplitMode.LEFT_RIGHT` | 列（宽度） |
| `pl.SplitMode.NONE` | 不 split |

选哪个由 vector 操作数的形状决定：选那个大到能在两个通道间均分的轴。用 `--mode left_right` 跑配套文件可以对比。

## 第 2 步：环会花掉你的 vector 预算

编译器插入的那些传输并非免费。每个跨越边界的 tile 都落在一个环形缓冲里，而这块缓冲是从**消费侧**核的片上内存中划出来的 —— 这里是 UB，因为是 cube 喂给 vector 单元：

| 量 | 值 |
| -- | -- |
| 跨越边界的 tile | `[128, 128]` FP32 = 64 KB |
| 默认环深 | 2 槽 |
| 环的大小 | 2 × 64 KB = **128 KB** |
| vector 预算 | **184 KB** |

环是一个整 tile 的队列，所以它的大小随跨越的 tile 而变，与工作量无关。默认的 2 是仍能双缓冲的最浅深度：cube 填一个槽的同时，vector 抽干另一个。

`pl.cross_core_slot(slot_num=N)` 用来重新调整它。更深的环换来更多重叠 —— 生产者在阻塞前能跑得更靠前 —— 所以当两个单元负载不均衡时可以调高它。但预算很紧：在这个 kernel 上 `slot_num=4` 就已经分配不下了。

```python
with pl.at(
    level=pl.Level.CORE_GROUP,
    optimizations=[pl.split(pl.SplitMode.UP_DOWN), pl.cross_core_slot(slot_num=4)],
    name_hint="mixed",
):
```

```text
Vec buffer usage (294912 bytes) exceeds platform limit (188416 bytes). The first 262144
bytes of that space are reserved by system.reserve_buffer, so tiles are allocated above
them — this is the cross-core pipe ring. Lower its depth with
optimizations=[pl.cross_core_slot(slot_num=N)] on the enclosing pl.at(...), or shrink the
tile that crosses the cube/vector boundary
```

真遇到时有两个杠杆：缩小 tile，或缩短环。**在装得下的前提下选最大的深度**。

## 第 3 步：编译器插进去了什么

`pl.split` 是自动路径。底下，跨核数据流是显式算子，你也可以自己写：

| 算子 | 角色 |
| ---- | ---- |
| `pl.aic_initialize_pipe` / `pl.aiv_initialize_pipe` | 建立管道 |
| `pl.tpush_to_aiv` / `pl.tpush_to_aic` | 把 tile 推给对端核 |
| `pl.tpop_from_aic` / `pl.tpop_from_aiv` | 弹出对端推来的 tile |
| `pl.tfree_to_aic` / `pl.tfree_to_aiv` | 把弹出的槽释放回生产者 |
| `pl.aiv_shard` / `pl.aic_gather` | 在 AIV 通道间分片 / 在 AIC 上聚回 |
| `pl.split_aiv(n, mode=...)` | split 的显式区域形式 |

**每次 push 必须与一次 pop 配对，每次 pop 必须与一次 `tfree` 配对。** 漏掉 `tfree` 不会报错 —— 它泄漏一个环槽，等环满了生产者就卡住。

**显式写法还把跨 lane 的定序也交给了你。** 边界算子只为它所搬运的那一个值定序；没有任何东西会为 cube lane 的写与 vector lane 对**同一块 GM 缓冲区**的读定序。先发布 producer 的写并执行 fence，再在两个阶段之间放置跨核 `pl.system.syncall`，最后在 consumer 读之前使其 cache 失效；barrier 本身只同步到达。可能部分占用时使用 soft 形式，buffer 可能跨多条 cache line 时使用全 GM cache 维护。上面的 `pl.split` 路径不需要这组序列 —— 传输由编译器插入，结果也与 torch 对拍过。规则见 [作用域与放置](../language/04-scopes.md)。

当 `pl.split` 表达不了所需形状时才动用显式形式：逐通道寻址、只有某一个通道算得出的 gather、或者一个混合了 split 与非 split 工作的区域。`tests/st/codegen/dsl/test_split_aiv_gather_row_codegen.py` 是一个实例。其余情况留在 `pl.split` 上 —— 它插入的是同样的算子，而且配对不会错。

机器级契约见 [TPUSH/TPOP](../../reference/pto-isa/01-tpush_tpop.md)；pass 做了什么见 [ExpandMixedKernel](../../dev/passes/22-expand_mixed_kernel.md)。

## 边界情况

> **致命陷阱：** 环是按 cube/vector 边界上的整 tile 计量的。一个变大的 tile 会把本来能跑的 kernel 变成分配不出来的；而报错给的是字节数而非 tile —— 请把它读作「跨越的 tile 太大，或者环太深」。

| 症状 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| **`Vec buffer usage ... exceeds platform limit`** | 环加上 tile 超出片上预算 | 调低 `pl.cross_core_slot(slot_num=N)`，或缩小跨越的 tile |
| **`pl.split` 没带来加速** | 一侧占主导，对半分无从重叠 | 检查这份工作是否真的是 cube 接 vector |
| **生产者跑一阵后卡住** | 弹出的槽从未 `tfree` | 让每次 pop 都配一次 `tfree` |
| **作用域上的 split 被拒** | 区域体混合了 split 与普通全宽 vector 算子 | 改用显式的 `pl.split_aiv` 区域形式 |

## 真实模型里的同一形状

`examples/models/qwen3_jit/` 是一条按模块拆成一文件一模块的 `@pl.jit` decode 路径，其中
`kernels/projection.py` 就是本页这个模式在模型规模上的样子 —— 一个 matmul 和消费它的 vector
工作，放在同一个作用域里。

| 文件 | 模块 |
| ---- | ---- |
| `qwen3_decode.py` | 组合其余部分的 decode 入口 |
| `config.py` | 各 kernel 特化所依据的形状与 dtype |
| `kernels/projection.py` | cube + vector 混合的 projection |
| `kernels/attention.py` | Attention |
| `kernels/mlp.py` | MLP |
| `kernels/rmsnorm.py` | RMSNorm |

## 下一步

[塑形任务图](04-task-graph.md) —— 从一个 kernel 内部，走到 kernel 之间的顺序。
