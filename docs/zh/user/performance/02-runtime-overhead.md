# 运行时开销

在不改变任务算什么的前提下，让每个任务花得更少的四种办法。

> **前置**：[任务粒度](01-task-granularity.md)。

## 与上一页的区别

粒度改的是任务**有多少个**。本页改的是**每个任务的代价**：两次派发变一次、N 次派发变一次、让派发更早开始，或者在一个屏障就够的地方干脆不派发。

| 手段 | 消掉什么 |
| ---- | -------- |
| [mix kernel](#构建-mix-kernel) | 两次派发中的一次，以及它们之间的 GM 往返 |
| [SPMD](#使用-spmd) | N 个 block 相同工作的 `N − 1` 次派发 |
| [`allow_early_resolve`](#让消费者提前预置) | 关键路径上的取件延迟 |
| [kernel 内 `syncall`](#在-kernel-内部同步) | 每个同步点一次 AICPU 往返 |

下面这些 kernel 每次 CI 都会被执行，所以它们是真货而不是草图。它们共用这段准备：

<!-- doctest: setup -->
```python
import pypto.language as pl
import torch
from pypto.runtime import RunConfig

BLOCKS, TILE_ROWS, COLS = 4, 64, 128
ROWS = BLOCKS * TILE_ROWS
CFG = RunConfig(platform="__PLATFORM__")

torch.manual_seed(0)
A = torch.randn(ROWS, COLS, dtype=torch.float32)
B = torch.randn(ROWS, COLS, dtype=torch.float32)

def fresh(rows=ROWS):
    return torch.zeros(rows, COLS, dtype=torch.float32)
```

## 构建 mix kernel

**何时适用：** 一个 cube 操作喂给一个 vector 操作。不管的话它们是两个任务：cube 任务跑完写 GM，vector 任务再被派发去把它读回来。

**怎么做：** 一个 `pl.at` 作用域同时承载两者，并用一个 split 模式告诉编译器怎么把 vector 那一半分给共享同一个 cube 的两个 AIV：

```python
with pl.at(
    level=pl.Level.CORE_GROUP,
    optimizations=[pl.split(pl.SplitMode.UP_DOWN), pl.cross_core_slot(slot_num=2)],
):
    acc = pl.matmul(a, b, out_dtype=pl.FP32)   # cube (AIC)
    out[:] = pl.add(acc, bias)                 # vector (AIV)
```

`examples/advanced/03_mixed_kernel.py` 用三种模式跑了这个例子，[教程](../tutorials/03-mixed-kernel.md) 里有逐步讲解。

**代价：** `pl.split` 只把 **vector** 子区域减半；cube 侧仍是全尺寸，所以 vector 缓冲变小而 cube 缓冲不变。另外在两个引擎之间搬运中间结果的那个跨核环是实打实的内存：它默认 **8 个槽**，对一个大中间结果来说远超 vector 预算能让出的量。`slot_num=` 通常不是可选项 —— 默认放不下时编译器会在编译期明说。

**怎么确认：** 看泳道图。两根首尾相接的条应当变成一根、且 cube 与 vector 的 span **重叠**。如果只是变成了一根总宽度不变的条，那说明两个引擎在 kernel 内部仍然是串的，该回头看 split 模式。

## 使用 SPMD

**何时适用：** 同一个 kernel 作用在 N 块互相独立的数据上。逐个派发等于为一件事付 N 次派发。

**怎么做：** 一次派发，由运行时扇出。每个 block 读自己的索引。

下面两种写法算的是同一件事；前者付 `BLOCKS` 次派发，后者只付一次：

<!-- doctest: run -->
```python
@pl.jit
def per_block_tasks(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
    for i in pl.unroll(BLOCKS):                     # BLOCKS dispatches
        with pl.at(level=pl.Level.CORE_GROUP):
            ta = pl.load(a, [i * TILE_ROWS, 0], [TILE_ROWS, COLS])
            tb = pl.load(b, [i * TILE_ROWS, 0], [TILE_ROWS, COLS])
            pl.store(pl.add(ta, tb), [i * TILE_ROWS, 0], c)
    return c

@pl.jit
def spmd_blocks(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
    for i in pl.spmd(BLOCKS):                       # one dispatch, BLOCKS blocks
        ta = pl.load(a, [i * TILE_ROWS, 0], [TILE_ROWS, COLS])
        tb = pl.load(b, [i * TILE_ROWS, 0], [TILE_ROWS, COLS])
        pl.store(pl.add(ta, tb), [i * TILE_ROWS, 0], c)
    return c

for kernel in (per_block_tasks, spmd_blocks):
    c = fresh()
    kernel(A, B, c, config=CFG)
    torch.testing.assert_close(c, A + B, rtol=1e-4, atol=1e-4)
```

能用设备查询就别写字面量：

| 写法 | 适用于 |
| ---- | ------ |
| `pl.system.available_cluster_count()` | mix 或 cube-only kernel |
| `pl.system.available_aiv_count()` | vector-only kernel |

这是**唯一**能跨设备保持满占用的写法 —— 这件事本身就重要，而且是下面那个屏障的硬性前提。

`examples/models/09_paged_attention_spmd.py` 是同一个思路在模型规模上的样子：每个 block 用一个 stride 循环取走一部分 batch，于是 batch 维靠一次派发就并行到了各个硬件 block 上。

**代价：** 每个 block 跑同一个程序。有分支差异的工作需要别的结构；先跑完的 block 会让自己的核空着，直到整个 grid 退休。

**怎么确认：** `deps.json` 里 N 个节点塌缩成一个；泳道图上一个任务同时占据多条核泳道。点击其中一个 block 时插件会高亮同一 SPMD 的全部 block。

## 让消费者提前预置

**何时适用：** 关键路径由许多短任务串成，每个消费者在生产者结束后还要干等自己那份取件延迟。

**怎么做：** 给**生产者**打标记。

<!-- doctest: run -->
```python
@pl.jit
def early_resolve(a: pl.Tensor, b: pl.Tensor, scratch: pl.Out[pl.Tensor], out: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP, allow_early_resolve=True):
        s = pl.add(pl.load(a, [0, 0], [TILE_ROWS, COLS]), pl.load(b, [0, 0], [TILE_ROWS, COLS]))
        pl.store(s, [0, 0], scratch)
    with pl.at(level=pl.Level.CORE_GROUP):
        pl.store(pl.exp(pl.load(scratch, [0, 0], [TILE_ROWS, COLS])), [0, 0], out)
    return scratch, out

scratch, out = fresh(TILE_ROWS), fresh(TILE_ROWS)
early_resolve(A[:TILE_ROWS], B[:TILE_ROWS], scratch, out, config=CFG)
torch.testing.assert_close(out, torch.exp(A[:TILE_ROWS] + B[:TILE_ROWS]), rtol=1e-4, atol=1e-4)
```

调度器于是可以在这个任务**完成之前**就把它的消费者预置到空闲核上，等它一结束就用门铃放行。

`pl.at`、`pl.submit`、`pl.spmd`、`pl.spmd_submit` 上都有，而且它是纯调度提示 —— 不影响结果。

**代价：** 对正确性基本为零，但要注意决定它是否起作用的那条规则：一个消费者只有在它的**所有**生产者都被标记（或已经完成）时才会预置。给一个三生产者的消费者只标一个，什么都买不到。这也是它通常被整条链地施加的原因 —— 比如 `models/qwen3_14b/decode_fwd.py` 里，decode 路径上几乎每个任务都带着它。

**怎么确认：** 关键路径上的 `[dispatch, start]` 间隙变小。任务总数与图的形状不该变 —— 如果变了，说明还改了别的东西。

## 在 kernel 内部同步

**何时适用：** 一次 SPMD 发射的各个 block 需要在某处会合。把它表达成两个任务加一条依赖，等于把同步送到 AICPU 调度器再送回来。

**怎么做：** `pl.system.syncall()` 在 kernel 内部同步参与的各个核。

```python
# 硬屏障（FFTS）：不带操作数，但要求满占用
with pl.spmd(pl.system.available_aiv_count()):
    ...
    pl.system.syncall(core_type="aiv_only")
    ...
```

两种模式，而且这个选择不是风格问题：

| 模式 | 机制 | 占用要求 | 额外参数 |
| ---- | ---- | -------- | -------- |
| `mode="hard"`（默认） | FFTS 屏障 | `core_type` 的**全部**物理核 | 无 |
| `mode="soft"` | GM 轮询计数 | 任意（`used_cores` 个参与者） | `gm_workspace`、`used_cores` |

两种 mode 都只同步到达：它们不会等待前序 `TSTORE`，也不会让业务数据的 cache 保持一致。通过 GM 从 producer 向 consumer 交接可能跨多条 cache line 的数据时，请保守地在 `syncall` 之前使用全 GM `pl.system.cacheinvalid()` + `pl.system.fence()`，然后在 consumer 读之前再次调用 `pl.system.cacheinvalid()`。tensor-region overload 当前只使 view 基地址所在的那一条 cache line 失效。

**跑一下：** `python examples/advanced/05_runtime_overhead.py --mode soft_barrier` —— 它需要 `runtime/pto_isa.pin` 所钉的 pto-isa，因为 cacheinvalid 路径会发出 `cache_line_t::SINGLE_CACHE_LINE`。

**代价，而且很锋利。** 部分发射下的硬 `syncall` 会在设备上**死锁**（错误 507018）。PyPTO 在编译期就拒绝它 —— `HardSyncallOccupancy` 校验器 —— 这正是 grid 必须用 `available_aiv_count()` / `available_cluster_count()` 来定，而不是写一个恰好在今天这台设备上对得上的字面量的原因。如果你无法保证满占用，就用 `mode="soft"`：它轮询一块共享 GM workspace，因此能在部分占用下工作，代价换成了 GM 流量。

```python
# 软屏障：部分占用下也能用
pl.system.syncall(mode="soft", core_type="mix",
                  gm_workspace=ws,     # 独占且零初始化的 16 元素 INT32 GM tensor
                  used_cores=n)
```

**怎么确认：** 泳道图上 AICPU 调度器那条泳道里，原本夹在两半工作之间的那次往返消失了，两个任务变成一个。

## 参见

- [管理任务依赖](03-dependencies.md) —— 当代价不在每个任务上，而在一张被串起来的图上。
- [任务与定序 § 精修依赖图](../tasks/03-tuning.md) —— `allow_early_resolve` 与 `predicate=` 的参考性说明。
