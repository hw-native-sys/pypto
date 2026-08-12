# 规约与 softmax

一个输出元素依赖于整行的场合，tile 词汇多出两种形状。

> **前置**：[第一个算子](00-elementwise.md)。
> **配套文件**：`examples/intermediate/02_softmax.py`。

## 你要做的东西

一个 `[64, 64]` tile 上数值稳定的 softmax。为此需要三样逐元素那条线从没用过的东西：规约用的 scratch tile、一个列向量、以及广播回全宽。

## 形状的来龙去脉

逐元素算子保形，规约不保：

```text
[64, 64]  --row_max-->  [64, 1]  --row_expand_sub-->  [64, 64]
```

本页的一切都是中间那个 `[64, 1]` 带来的后果。它是一个自有形状的真实 tile，你没法用普通的 `pl.sub` 把它从 `[64, 64]` 里减掉 —— 形状对不上。`row_expand_*` 家族就是为了弥合这个落差而存在的。

## 第 1 步：行规约需要一个 scratch tile

```python
max_tmp = pl.create_tile([64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
row_max: pl.Tile[[64, 1], pl.FP32] = pl.row_max(tile_a, max_tmp)
```

`pl.row_max` 要第二个参数：一个与输入同形的**全宽 scratch tile**，供规约当工作空间。它不是输出，你永远不会去读它。`pl.row_sum` 签名相同。

两个值得提前规划的后果：

- scratch 是全宽的，所以一次行规约大约要花掉它所规约 tile 的**两倍**缓冲区。把 kernel 顶出 vector 预算的通常是这个，而不是规约本身。
- 它是用 `target_memory=pl.MemorySpace.Vec` 创建的 —— 规约跑在 vector 单元上，scratch 必须待在它们够得着的地方。

## 第 2 步：广播回去

`[64, 1]` 的结果要作用到每一列上。这就是 `row_expand_*` 家族 —— 每种组合运算一个算子，而不是一条能套用到任意 op 上的广播规则：

| 算子 | 计算 |
| ---- | ---- |
| `pl.row_expand_sub(t, v)` | `t - v`，沿列广播 |
| `pl.row_expand_div(t, v)` | `t / v` |
| `pl.row_expand_expdif(t, v)` | `exp(t - v)`，融合形式 |

`row_expand_expdif` 是下面第 2、3 步的融合版。等未融合版跑通了再换它。

## 第 3 步：完整的 softmax

softmax 是 `exp(x) / sum(exp(x))`，但照字面算会溢出：`exp(88)` 就已经出了 FP32 范围。先减去行最大值在数学上是恒等变换 —— `exp(-max)` 这个因子在分子分母间约掉了 —— 并且能让每个指数都不大于零。

```python
import pypto.language as pl
import torch
from pypto.runtime import RunConfig

@pl.jit
def softmax(
    a: pl.Tensor[[64, 64], pl.FP32],
    out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        tile_a = pl.load(a, [0, 0], [64, 64])

        max_tmp = pl.create_tile([64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        row_max: pl.Tile[[64, 1], pl.FP32] = pl.row_max(tile_a, max_tmp)

        shifted = pl.row_expand_sub(tile_a, row_max)   # x - max(x)
        exp_shifted = pl.exp(shifted)

        sum_tmp = pl.create_tile([64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        row_sum: pl.Tile[[64, 1], pl.FP32] = pl.row_sum(exp_shifted, sum_tmp)

        pl.store(pl.row_expand_div(exp_shifted, row_sum), [0, 0], out)
    return out

torch.manual_seed(0)
a = torch.randn(64, 64)
out = torch.zeros(64, 64)
softmax(a, out, config=RunConfig(platform="a2a3sim"))
assert torch.allclose(out, torch.softmax(a, dim=1), rtol=1e-5, atol=1e-5)
```

两个 scratch tile，因为两次规约的存活期不同，各需自己的工作空间。

跑它：

```bash
python examples/intermediate/02_softmax.py
```

## 第 4 步：不满的行会改变答案

真实输入很少能被 tile 整除。`pl.load` 接受 `valid_shape=` 来声明「只有这个子区域是真数据」，其余部分是 padding。

对逐元素算子来说 padding 无害 —— 垃圾进垃圾出，落在没人读的通道里。**对规约则不然**，因为 padding 会参与运算：

```python
tile = pl.load(a, [0, 0], [64, 64], valid_shape=[64, vlen])
```

若有效列是 40，`row_max` 在这个 tile 上会看到 24 列 padding 里的任何东西。如果那是零而你的数据全为负，每一行的最大值都会回来 `0.0` —— 错了，而且错得悄无声息。

`pl.fillpad` 设定 padding 的内容，取什么值取决于规约：

| 规约 | 填什么 | 为什么 |
| ---- | ------ | ------ |
| `row_max` | `pl.PadValue.min` | 最小可表示值永远赢不了 max |
| `row_sum` | `pl.PadValue.zero` | 零是加法的单位元 |

规则可以推广：**填你即将施加的那个运算的单位元。** `PadValue` 另有 `max`（用于 min 规约）与 `null`（不填充）。

见 [动态有效形状](../language/03-memory.md) 与 `examples/intermediate/06_dyn_valid_shape.py` 里那个遍历末块不满的张量的循环。

## Edge Cases

> **致命陷阱：** 对带 padding 的 tile 做规约会读到 padding。它产出的是一个看起来合理的数，不是报错。只要一个 tile 带了 `valid_shape=` 又遇上 `row_max` / `row_sum`，就必须决定 padding 里放什么。

| 症状 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| **softmax 返回 NaN 或 inf** | 没减行最大值就做了 `exp` | 先减 `row_max` —— 第 3 步 |
| **规约报缺参数** | `row_max` / `row_sum` 需要 scratch tile | 传一个全宽的 `pl.create_tile(...)` |
| **行最大值全是 `0.0`** | padding 参与了规约 | 规约前 `pl.fillpad(..., pl.PadValue.min)` |
| **vector 缓冲区超限** | 每次规约的 scratch 都是全宽 | 缩小 tile，或在生命期不重叠的规约间复用一个 scratch |
| **`row_expand_*` 形状不匹配** | 用普通 `pl.sub` 去作用一个 `[N, 1]` 向量 | 改用该运算对应的 `row_expand_*` 成员 |

## 下一步

[分块 matmul](02-matmul.md) —— 第一个跑在 cube 单元上的算子，也是第一个需要你考虑数据通路的算子。
