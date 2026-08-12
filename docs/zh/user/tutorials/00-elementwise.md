# 第一个算子

把 `c = a + b` 写四遍，每一版都更接近真实 kernel 的写法。

> **前置**：[类型](../language/00-types.md) 与 [函数](../language/01-functions.md)。
> **配套文件**：`examples/beginner/02_elementwise.py`。

## 你要做的东西

一个逐元素加法，与 torch 对拍。刻意选了最没意思的算术，因为这里每一步讲的都是**放置** —— 数据在哪、谁有权碰它 —— 算术只会碍事。

四步，每步都可运行：

1. tensor 级 —— 说清算什么，放置交给编译器
2. tile 级 —— 自己放置：load、compute、store
3. 分块 —— 张量大于一个 tile
4. 对拍 —— 与 torch 比较

## 第 1 步：tensor 级

```python
import pypto.language as pl
import torch
from pypto.runtime import RunConfig

@pl.jit
def add_tensor(
    a: pl.Tensor[[128, 128], pl.FP32],
    b: pl.Tensor[[128, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        out = pl.assemble(out, pl.add(a, b), [0, 0])
    return out

a = torch.randn(128, 128)
b = torch.randn(128, 128)
out = torch.zeros(128, 128)
add_tensor(a, b, out, config=RunConfig(platform="a2a3sim"))
assert torch.allclose(out, a + b, rtol=1e-5, atol=1e-5)
```

这里有三件事在起作用。

**`pl.Out[...]` 是方向，不是提示。** 它告诉编译器这个参数被写、不被读。运行时正是据此把本任务与其他任务定序 —— 见 [依赖模型](../tasks/00-model.md)。写错了得到的是竞态，不是报错。

**`with pl.at(level=pl.Level.CORE_GROUP):` 才是放算子的地方。** `@pl.jit` 函数体本身是控制面；它派发工作，但不能容纳算子。把 `out = pl.add(a, b)` 写在块外是错误 —— 解析器会说。

**`pl.assemble` 才是写输出的方式。** 这一步是所有人都会栽的地方：

```python
out = pl.add(a, b)                              # ✗ 编译通过，什么都没写
out = pl.assemble(out, pl.add(a, b), [0, 0])    # ✓
```

第一行只是重绑定了一个局部名字。它编译通过、跑得起来，而输出缓冲区里还是原来的东西 —— 你拿到 NaN，且任何一级都不会给出诊断。`pl.assemble(dst, value, offset)` 才是真正把 `value` 放进 `dst` 的 `offset` 处的那个算子。

> **致命陷阱：** 错的那种写法恰恰读起来最自然。如果一个 kernel 返回垃圾而全程无报错，先检查每一次写入是否都经过 `pl.assemble` 或 `pl.store`。

## 第 2 步：tile 级

第 1 步从没说数据在哪，是编译器选的。自己来则需要三个显式算子：

```python
@pl.jit
def add_tile(
    a: pl.Tensor[[128, 128], pl.FP32],
    b: pl.Tensor[[128, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        tile_a = pl.load(a, [0, 0], [128, 128])
        tile_b = pl.load(b, [0, 0], [128, 128])
        pl.store(pl.add(tile_a, tile_b), [0, 0], out)
    return out
```

| 算子 | 作用 |
| ---- | ---- |
| `pl.load(t, offset, shape)` | 把 DDR 张量的一个窗口拷进片上 tile |
| `pl.add(tile, tile)` | 现在是 *tile* op —— 同名，由操作数类型决定 |
| `pl.store(tile, offset, t)` | 把 tile 拷回去 |

`pl.store` 是 `pl.assemble` 的 tile 级对应物。规则相同：它才是执行写入的算子，所以结果必须流经它。

两版算的是同一件事。第 1 步更短；第 2 步是形状一旦装不下就必须用的写法。

## 第 3 步：大于一个 tile 的张量

tile 是片上内存的固定大小窗口，所以 `[512, 128]` 的张量没法一次载入。对分块做循环，移动的是**偏移** —— shape 参数保持 tile 大小不变：

```python
ROWS, COLS, TILE_ROWS = 512, 128, 128

@pl.jit
def add_chunked(
    a: pl.Tensor[[512, 128], pl.FP32],
    b: pl.Tensor[[512, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        for i in pl.range(ROWS // TILE_ROWS):
            tile_a = pl.load(a, [i * TILE_ROWS, 0], [TILE_ROWS, COLS])
            tile_b = pl.load(b, [i * TILE_ROWS, 0], [TILE_ROWS, COLS])
            pl.store(pl.add(tile_a, tile_b), [i * TILE_ROWS, 0], out)
    return out
```

`pl.range` 是编译期循环：它会被展开进 IR，所以每一步的 `i` 都是常量、偏移都是静态的。这就是每个真实 kernel 的形状 —— 算术坐落在一个逐 tile 走过张量的循环嵌套里。

## 第 4 步：验它

到此为止没有任何东西证明 kernel 是对的。与 torch 比较，并且断言：

```python
torch.manual_seed(0)
a = torch.randn(512, 128)
b = torch.randn(512, 128)
out = torch.zeros(512, 128)
add_chunked(a, b, out, config=RunConfig(platform="a2a3sim"))
assert torch.allclose(out, a + b, rtol=1e-5, atol=1e-5)
```

要断言，不要打印。一个静默什么都没写的 kernel 产出 NaN，而 NaN 与任何值比较都不相等 —— `allclose` 能抓住它，扫一眼打印结果未必。

跑完整文件：

```bash
python examples/beginner/02_elementwise.py
```

## Edge Cases

| 症状 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| **输出是 NaN 且无任何报错** | 结果被赋给了 `Out` 参数而不是写进去 | 让它流经 `pl.assemble` / `pl.store` |
| **`Misplaced tensor op`** | 算子写在 `@pl.jit` 体内、`pl.at` 之外 | 移进 `with pl.at(level=pl.Level.CORE_GROUP):` |
| **tile 形状被拒** | 窗口超出片上内存所能容纳 | 分块 —— 第 3 步 |
| **多次运行结果不同** | 两个任务触碰同一缓冲区却无任何东西定序 | 见 [依赖模型](../tasks/00-model.md) |

## 下一步

[规约与 softmax](01-reduction-softmax.md) —— 一个输出元素依赖于整行的场合，tile 词汇就不够用了。
