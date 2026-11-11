# 混合 kernel

> **旧 Python 接口示例：** 本页为兼容现有用法保留 `platform=` 调用，
> 这些调用会发出 `DeprecationWarning`；新代码应使用 `backend_type` 和
> `execution_mode`。参见[目标映射](../../dev/08-entry-points.md#迁移旧-platform-示例)。

cube 与 vector 在同一个作用域内工作，传输由编译器管理。

> **前置**：[分块 matmul](02-matmul.md)。
> **配套文件**：`examples/advanced/03_mixed_kernel.py`。

## 你要做的东西

`a @ b + bias` —— 一个 cube 运算接一个 vector 运算 —— 写成一次混合派发，
而不是两个独立作用域。本示例演示数据流，并不证明性能提升。

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

把 cube 计算和显式分片的 vector 计算放进同一个作用域：

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
        optimizations=[pl.cross_core_slot(slot_num=1)],
        name_hint="mixed",
    ):
        acc = pl.matmul(a, b, out_dtype=pl.FP32)                 # cube (AIC)
        for lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
            half = pl.aiv_shard(acc)
            row = lane * 64
            out[row:row + 64, :] = pl.add(half, bias[row:row + 64, :])
    return out

torch.manual_seed(0)
a = torch.randn(128, 256, dtype=torch.float16)
b = torch.randn(256, 128, dtype=torch.float16)
bias = torch.randn(128, 128, dtype=torch.float32)
out = torch.zeros(128, 128, dtype=torch.float32)
compiled = mixed.compile(a, b, bias, out, config=RunConfig(platform="a2a3sim"))
compiled(a, b, bias, out, config=RunConfig(platform="a2a3sim"))
assert torch.allclose(out, a.float() @ b.float() + bias, rtol=1e-2, atol=1e-2)
```

`pl.split_aiv(mode=...)` 绑定 AIV lane 并提供边界模式，`pl.aiv_shard(acc)` 显式
分片 cube 结果。用户编写每 lane 的 bias 切片和输出偏移；编译器不再拆半 body。
cube 计算保持完整尺寸。mixed 函数级 `pl.split` AUTO 拆半已移除。

| 模式 | 显式边界分片轴 |
| ---- | -------------- |
| `pl.SplitMode.UP_DOWN` | 行（高度） |
| `pl.SplitMode.LEFT_RIGHT` | 列（宽度） |
| `pl.SplitMode.NONE` | 不 split |

选哪个由 vector 操作数的形状决定：选那个大到能在两个通道间均分的轴。用 `--mode left_right` 跑配套文件可以对比。

## 第 2 步：环会花掉你的 vector 预算

编译器插入的那些传输并非免费。每个跨越边界的 tile 都落在一个环形缓冲里，而这块缓冲是从**消费侧**核的片上内存中划出来的 —— 这里是 UB，因为是 cube 喂给 vector 单元：

| 量 | 值 |
| -- | -- |
| 跨越边界的 tile | `[128, 128]` FP32 = 64 KiB |
| 默认环深 | 2 槽 |
| 本示例显式选择的环深 | 1 槽 = **64 KiB** |
| 每个 vector 通道的 bias 与输出 | 各 32 KiB，共 **64 KiB** |
| 单槽时的总占用 | **128 KiB** |
| vector 安全预算 | **184 KiB** |

环是一个整 tile 的队列，所以它的大小随跨越的 tile 而变，与工作量无关。默认的 2 是仍能双缓冲的最浅深度：cube 填一个槽的同时，vector 抽干另一个。

`pl.cross_core_slot(slot_num=N)` 用来重新调整它。在 Ascend910 上，load/TPOP
安全约束要求加载的 bias 与加法输出使用独立缓冲区。因此默认双槽环需要
128 + 32 + 32 = **192 KiB**，超过安全预算。本示例显式选择单槽；它演示混合
数据流，而非生产者/消费者的双缓冲重叠。tile 更小时，剩余空间可能允许更深的环，
让生产者运行得更靠前，但这并不保证加速。

将本示例改回双槽，会在分配阶段失败：

```python
with pl.at(
    level=pl.Level.CORE_GROUP,
    optimizations=[pl.cross_core_slot(slot_num=2)],
    name_hint="mixed",
):
```

```text
Vec buffer usage (196608 bytes) exceeds platform limit (188416 bytes). The first 131072
bytes of that space are reserved by system.reserve_buffer, so tiles are allocated above
them — this is the cross-core pipe ring. Lower its depth with
optimizations=[pl.cross_core_slot(slot_num=N)] on the enclosing pl.at(...), or shrink the
tile that crosses the cube/vector boundary
```

真遇到时有两个杠杆：缩小 tile，或缩短环。**在装得下的前提下选最大的深度**。

## 第 3 步：传输降低

编译器将显式 `aiv_shard`/`aic_gather` 降低为管道 setup、`tpush/tpop` 和对应
`tfree`。region mode 选择分片轴，`NONE` 传输但不改变形状。使用这些显式边界
算子时，用户无需手写管道配对。

手写低层 pipe 时，每次 push 要配对 pop，每次 pop 要配对 free。边界只为它搬运的
值定序；用户管理的 GM 共享和一次性副作用仍需遵守
[作用域与放置](../language/04-scopes.md) 的顺序和 lane 所有权规则。

普通未拆分 mix 保留现有全幅隐式通信；本次删除隐式拆半，不删除这条通信路径。
参见 [TPUSH/TPOP](../../reference/pto-isa/01-tpush_tpop.md) 和
[ExpandMixedKernel](../../dev/passes/26-expand_mixed_kernel.md)。

## 边界情况

> **致命陷阱：** 环是按 cube/vector 边界上的整 tile 计量的。一个变大的 tile 会把本来能跑的 kernel 变成分配不出来的；而报错给的是字节数而非 tile —— 请把它读作「跨越的 tile 太大，或者环太深」。

| 症状 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| **`Vec buffer usage ... exceeds platform limit`** | 环加上 tile 超出片上预算 | 调低 `pl.cross_core_slot(slot_num=N)`，或缩小跨越的 tile |
| **显式分片没有带来加速** | 一侧占主导，对半分无从重叠 | 检查这份工作是否真的是 cube 接 vector |
| **生产者跑一阵后卡住** | 弹出的槽从未 `tfree` | 让每次 pop 都配一次 `tfree` |
| **Implicit AIV splitting is no longer supported** | 混合 kernel 的旧式函数级 AUTO split 已不受支持 | 使用 `pl.split_aiv`、显式 shard/gather，并编写每 lane 的形状和偏移 |

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
