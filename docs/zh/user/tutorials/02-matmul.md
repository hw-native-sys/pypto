# 分块 matmul

第一个跑在 cube 单元上的算子，也是第一个 K 轴装不下的算子。

> **前置**：[第一个算子](00-elementwise.md)。
> **配套文件**：`examples/intermediate/04_matmul_acc.py`、`examples/advanced/01_split_k.py`、`examples/advanced/02_auto_tile_matmul.py`。

## 你要做的东西

`C = A @ B`，其中 K 轴大于一个 tile，因此乘积必须跨多步累加。然后是一个用确定性换并行度的变体。

## 第 1 步：装得下的 matmul

```python
import pypto.language as pl
import torch
from pypto.runtime import RunConfig

@pl.jit
def matmul_small(
    a: pl.Tensor[[128, 128], pl.FP16],
    b: pl.Tensor[[128, 128], pl.FP16],
    c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="matmul"):
        c[:] = pl.matmul(a, b, out_dtype=pl.FP32)
    return c

torch.manual_seed(0)
a = torch.randn(128, 128, dtype=torch.float16)
b = torch.randn(128, 128, dtype=torch.float16)
c = torch.zeros(128, 128, dtype=torch.float32)
matmul_small(a, b, c, config=RunConfig(platform="a2a3sim"))
assert torch.allclose(c, a.float() @ b.float(), rtol=1e-2, atol=1e-2)
```

两个不是风格问题的细节：

**FP16 输入配 `out_dtype=pl.FP32`。** cube 单元以输入精度做乘法、以 FP32 做累加。要一个 FP16 累加器是白白丢精度 —— 宽着累加，必要时最后再 cast。

**容差是 `1e-2` 而不是 `1e-5`。** FP16 输入只带约 3 位十进制有效数字。拿 FP16 matmul 去和 FP32 的 torch 参考在 `1e-5` 上比，正确的代码也会挂；让容差匹配输入精度是写测试的一部分。

## 第 2 步：K 轴装不下

`A[128, 512] @ B[512, 128]` 没法把整个 K 一次搬上片。把 K 切块并累加：第一块产生累加器，其余块加进去。

```python
K_CHUNK = 128

@pl.jit
def matmul_k_blocked(
    a: pl.Tensor[[128, 512], pl.FP16],
    b: pl.Tensor[[512, 128], pl.FP16],
    c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="k_blocked"):
        acc = pl.matmul(a[:, 0:K_CHUNK], b[0:K_CHUNK, :], out_dtype=pl.FP32)
        for k in pl.range(1, 512 // K_CHUNK):
            k0 = k * K_CHUNK
            acc = pl.matmul_acc(acc, a[:, k0 : k0 + K_CHUNK], b[k0 : k0 + K_CHUNK, :])
        c[:] = acc
    return c
```

`pl.matmul` **创建**累加器，`pl.matmul_acc` **累加进**一个已有的。这个不对称正是循环从 1 开始的原因 —— 第 0 块还没有可累加的对象。把每一块都写成 `pl.matmul_acc`、另外单独分配累加器也可行，代价是多一次初始化。

累加器在整个循环期间都留在片上，只有最后那次写出才碰 DDR —— 这正是分块 K 而非逐块存回部分积的意义。

## 第 3 步：编译器替你做的部分

`AutoTileMatmulL0` 会对装不进 cube L0 缓冲区的 matmul 重新分块，自行选择 M/N/K 的切法。这就是第 1 步为什么你一个 tile 都没命名却能跑通：形状是整体交给 `pl.matmul` 的，由该 pass 安排搬运。

值得记住的推论：**tensor 级操作数上的 `pl.matmul` 不是一条指令**，而是编译器写出来的一个循环嵌套。当你像第 2 步那样手工分块 K，你是在 K 轴上覆盖了它的选择，M/N 仍留给该 pass。`examples/advanced/02_auto_tile_matmul.py` 走了一遍自动选择与直觉选择不一致的那些情形。

## 第 4 步：split-K 及其代价

分块 K 让一个核忙过每一块。**split-K** 则给每个核自己的一段 K，让它们用原子加累加进同一个输出：

```python
KS = K // SPLITS                       # each core's slice of K

with pl.at(level=pl.Level.CORE_GROUP, name_hint="zero_init"):
    c[:] = pl.full([M, N], dtype=pl.FP32, value=0.0)
for ks in pl.parallel(SPLITS):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="split_k"):
        k0 = ks * KS
        partial = pl.matmul(a[:, k0 : k0 + KS], b[k0 : k0 + KS, :], out_dtype=pl.FP32)
        c = pl.assemble(c, partial, [0, 0], atomic=pl.AtomicType.Add)
```

片段 —— `M`、`N`、`K` 与 `SPLITS` 来自外层函数；可运行版本是 `examples/advanced/01_split_k.py`。注意与第 2 步不同：这里每个核都写**整个** `[M, N]` 输出，被切分的是 K，不是输出。

| 方面 | 代价 |
| ---- | ---- |
| 零初始化 | 输出必须先清零；原子加没有「第一个写者」 |
| 确定性 | **跨核的累加顺序不固定**，重复运行的末位可能不同 |
| 何时划算 | K 很大，而 M/N 小到单靠它们填不满这些核 |

第二行才是要掂量的。如果下游测试按位比较，或者你正在追一个数值差异，split-K 会让答案变成移动靶。当并行度比可复现性更值钱时再用它。

## 边界情况

| 症状 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| **FP16 输入下 `allclose` 在 `1e-5` 失败** | 容差比输入精度还严 | 按 `1e-2` 比；累加器保持 FP32 |
| **相同输入多次运行结果漂移** | split-K 的原子累加顺序不固定 | 属预期 —— 需要确定性就用 K 分块 |
| **split-K 输出大了约一个倍数** | 原子循环前没清零输出 | 先在独立作用域里零初始化 |
| **累加器 dtype 被拒** | `matmul_acc` 要求累加器的 dtype | 用 `pl.matmul(..., out_dtype=pl.FP32)` 创建它 |

## 下一步

[混合 kernel](03-mixed-kernel.md) —— 到目前为止都只用了 cube **或** vector 单元。现在两者同时。
