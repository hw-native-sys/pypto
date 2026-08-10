# 选择命名空间

三个命名空间暴露了彼此重叠的算子。选择规则很短；它在什么情况下不适用才是有用的部分。

> **前置**：[类型](../language/00-types.md)。

## Concept

同样的算术存在于两个层级 —— 作用于整块 DDR 数组，以及作用于片上缓冲区 —— 因为那是两台不同的机器。`pl.tensor.add` 命名的是一个由编译器替你放置与调度的操作。`pl.tile.add` 命名的是作用在一块**你已经放置好**的缓冲区上的操作。

`pl.*` 两者都不是。它是一个**派发器**：检查实参类型后转发给张量或 tile 实现。它不是第三个层级，也不添加任何自己的语义。

由此得出规则：

```text
要写一个算子？
├─ 存在不带限定的 `pl.` 版本吗？  → 用它
└─ 否则 → 这个算子是分层级的；把层级写出来
```

**默认用 `pl.\*`。** 当 kernel 写在张量级时它让代码保持可读，而当周边代码后来下沉到 tile 级时它依然正确。

## Mechanics

### 三个命名空间

| 命名空间 | 作用于 | 谁决定放置 |
| -------- | ------ | ---------- |
| `pl.*` | `Tensor` 或 `Tile` —— 按实参类型派发 | 取决于它派发到了哪个 |
| `pl.tensor.*` | `Tensor`（DDR） | 编译器 |
| `pl.tile.*` | `Tile`（片上） | 你 |

另有两个命名空间不是层级而是家族：

| 命名空间 | 内容 |
| -------- | ---- |
| `pl.system.*` | 跨核传输、管道、设备规模查询、同步 |
| `pl.array.*` | 核内数组的创建 / 读取 / 更新 |

```python
c = pl.add(a, b)          # dispatches on the type of a and b
c = pl.mul(a, 2.0)        # scalar rhs detected -> the *s form
t = pl.tile.load(x, [0, 0], [64, 64])
t = pl.tile.adds(t, 1.0)  # tile-specific spelling
```

### 路径专属关键字参数会抛异常

有少数算子的某个关键字只有一个层级能兑现。张量没有布局，所以 `matmul` 用 `a_trans=` / `b_trans=` 表达转置；而在 tile 级，转置是一种**类型**属性，用 `pl.tile.transpose_view(...)` 表达。反过来，tile 的 scratch 缓冲区必须由调用方提供，张量路径则由编译器分配。

把这类关键字以非默认值传给错误的路径会抛 `TypeError`，并指明是哪个关键字、以及该层级对应的替代写法 —— 它**不会**被静默丢弃。传文档给出的默认值（比如 `b_trans=False`）是空操作，仍然接受。

### 统一形式不存在的情况

当一个算子只在某一个层级上有意义时，它就没有 `pl.*` 拼法。这些情况下直接使用命名空间是正确做法，而不是退而求其次：

| 仅张量级 | 原因 |
| -------- | ---- |
| `pl.create_tensor`、`pl.full`、`pl.assemble` | 整块数组的创建与放置 |
| `pl.dim` | 张量的运行期维度 |

| 仅 tile 级 | 原因 |
| ---------- | ---- |
| `pl.tile.load` / `store` / `move` | 在内存空间之间搬运 |
| `pl.tile.transpose_view` | 对已放置缓冲区的视图 |
| `pl.tile.get_block_idx` / `get_subblock_idx` | 正在执行的 block 的身份 |

有若干 tile 专属算子为了方便被重新导出到了顶层 —— `pl.load`、`pl.store`、`pl.move`、`pl.create_tile`、`pl.matmul_acc` 等就是它们 `pl.tile.*` 原型的同一个函数。`pl.load` 不是派发器；它就是换了个短名字的 `pl.tile.load`。

### 标量操作数形式

许多算子有一个右侧接标量的伴生形式，名字以 `s` 结尾：`adds`、`muls`、`maximums`、`ands`、`shls`。你很少需要写出它们 —— 给统一形式传一个 Python 数字就会选中它们：

```python
c = pl.add(a, b)        # tensor/tile + tensor/tile
c = pl.add(a, 1.0)      # -> the scalar-operand form
```

当某个算子在派发器无法推断的位置接受标量时，显式的 `pl.tile.*` 拼法就是表达它的方式。

## Edge Cases

| 症状 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| **`pl.<name>` 报 `AttributeError`** | 该算子是分层级的 | 加上限定：`pl.tile.<name>` 或 `pl.tensor.<name>` |
| **tile 算子在 `@pl.jit` 体内被拒绝** | 在控制面上做 tile 工作 | 移进 `pl.at(level=...)` 或 `@pl.jit.incore` 函数 |
| **张量算子在 InCore 函数内被拒绝** | 张量创建是控制面的事 | 在控制面分配；用 `pl.Out[...]` 参数传入 |
| **派发器选了你不想要的层级** | 它按实参类型派发 | 按你想要的层级传操作数，或直接写命名空间 |
| **算子存在但后端拒绝它** | 并非所有算子在每个后端都受支持 | 查 [PTOAS 算子状态](../../dev/ptoas-op-status.md) |

## See Also

- [算子目录](01-catalog.md) —— 各命名空间里都有什么。
- [内存与数据搬运](../language/03-memory.md) —— tile 专属搬运算子为何是那个形态。
- [编程模型](../03-programming-model.md#quickstart一个程序里的三个层次) —— 同一个计算写在两个层级上。
- [ConvertTensorToTileOps](../../dev/passes/11-convert_tensor_to_tile_ops.md) —— 把前者变成后者的 pass。
