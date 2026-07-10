# 统一的 `valid_shape` 语义

`valid_shape` 标记一个 shaped 值的有效子区域。落在物理 `shape_` 之内、
`valid_shape` 之外的访问，读到的是该 view 的 `pad` 值
（`PadValue.null` / `zero` / `max` / `min`）。两个 view 结构体都携带它 ——
`TensorView.valid_shape` 与 `TileView.valid_shape` —— 并共享同一套规则，
从而使 ragged-edge（不整除边界）kernel 可以写成

```text
slice / load  ->  compute  ->  assemble / store
```

而无需把 `min(TILE, N - i*TILE)` 这个 extent 逐个算子地手工传递。

`TensorType` / `TileType` 本身参见 [`02-types.md`](02-types.md)。

## 核心规则

**D1 —— Tile 物理上是二维的。** Tile 的物理 shape 恒为 `ConstInt`。
动态性（dynamism）完全承载在 `valid_shape` 上，其元素可以是 `ConstInt`、`Var`，
或像 `pl.min(...)` 这样的 `Call`。`FlattenTileNdTo2d` 是 ND → 2D 降级的唯一位置，
因此 PTO codegen 只读取 `valid_shape[0]` / `valid_shape[1]` 在构造上是正确的。

**D2 —— 未设置即表示全部有效。** 缺省或为空的 `valid_shape` 表示完整的物理 shape。

**规范编码。** 与物理 shape 相等的 `valid_shape` 不携带任何信息，
在构造时被收敛（collapse），因此每种语义状态在内存中只有唯一一种形式。
`TileType` 重置**整个** view（全有效的 tile 上没有其他有意义的字段）；
`TensorType` 只清空 `valid_shape` **这一个字段**，因为
`stride` / `layout` / `pad` 各自独立且有意义。

```python
from pypto import DataType, ir

span = ir.Span.unknown()
ci = lambda v: ir.ConstInt(v, DataType.INT64, span)

# TILE: explicit valid_shape == shape collapses the whole view to the no-view form.
full = ir.TileType([ci(128), ci(128)], DataType.FP32, None,
                   ir.TileView(valid_shape=[ci(128), ci(128)]), ir.Mem.Vec)
assert full.tile_view is None
ir.assert_structural_equal(
    full, ir.TileType([ci(128), ci(128)], DataType.FP32, None, None, ir.Mem.Vec))

# TENSOR: only the valid_shape field is cleared; the stride/layout view survives.
# A PARTIAL valid_shape (e.g. [32, 64]) is kept verbatim on either view.
tv = ir.TensorView(stride=[ci(128), ci(1)], layout=ir.TensorLayout.ND,
                   valid_shape=[ci(128), ci(128)])
t = ir.TensorType([ci(128), ci(128)], DataType.FP32, tensor_view=tv)
assert len(t.tensor_view.valid_shape) == 0
```

**常驻不变式**（always-on `TypeCheck` verifier）：`rank(valid) == rank(shape)`，
且对每个静态维满足 `0 <= valid[i] <= shape[i]`（符号维推迟判断）。
在 `[128, 128]` 的 tile 上 `pl.load(valid_shapes=[999, 999])` 会被拒绝。

**`GetValidShape()` 是唯一事实来源**
（[`include/pypto/ir/type_inference.h`](../../../../include/pypto/ir/type_inference.h)）：
有 view 时返回其 `valid_shape`，否则返回物理 `shape_`，
因此"未设置"与"显式全有效"对所有消费者不可区分。

## 传播与拒绝

算子类型推导绝不能破坏有效区域。两种失效模式均被禁止：

- **放大（Widening）** —— 把 padding 标记为真实数据，会把垃圾写进 tensor。
- **收窄（Narrowing）** —— 静默丢弃真实数据，导致写出被截断。

当正确的输出区域无法被*证明*时，算子通过 `CHECK_SPAN` **拒绝**，
而不是退化为完整 shape 或 arg0 的区域。共享辅助函数位于
[`src/ir/op/type_inference.cpp`](../../../../src/ir/op/type_inference.cpp)；
`ComputeAssembleUnionValidShape` 是严谨程度的标杆。

| 算子族 | `valid_shape` 规则 | 何时拒绝（`CHECK_SPAN`） |
| ------ | ------------------ | ------------------------ |
| `load(valid_shapes=)` | 设置；与源 tensor 的区域**求交**；继承 `pad` | 负的（非原点锚定）窗口 offset |
| 一元 / cast / 标量二元 / move | 复制输入区域 | —— |
| 逐元素二元 / 多操作数 | 逐维**一致性**：非广播维上 extent 必须相等；`valid_shape` 从不广播 | 可证明的静态 extent 不一致 |
| `part_add` / `part_mul` / `part_max` / `part_min` | 逐维**并集**（任一源有效即有效） | 并集不是原点锚定矩形 |
| reduction | 丢弃归约轴，保留其余 | —— |
| `matmul(A[M,K], B[K,N])` | `[valid(A)[M], valid(B)[N]]`；K 必须一致 | 可证明的静态 K 不匹配 |
| `assemble(target, source, off)` | target 与写入矩形的逐维包围盒 | 并集不是可证明的原点锚定矩形（空隙 / L 形 / 无法证明） |
| `slice(clamp=True)` | 在 `offset` 处把窗口裁剪到源区域（从不放大） | 负 offset |
| `reshape` | 把区域映射到扁平化缓冲区 | 输入区域不是连续的扁平前缀 |
| `transpose` / `extract` / `concat` | 置换 / 切出 / 堆叠区域 | `concat` 的非末位操作数部分有效（L 形） |
| `sort` / `mrgsort` | 输出为完整 shape | 输入部分有效（padding 会进入比较） |
| `store` | 恰好写出 `valid(tile)` | —— |
| tensor→tile `load`；tensor 计算算子 | 继承源区域 / 携带 view | 同其 tile 对应算子 |

### 为什么 `assemble` 会拒绝

`assemble` 对 target 的有效矩形 `[0, valid(target))` 与写入矩形
`[offset, offset + valid(source))` 求并集。而 `(valid_row, valid_col)`
只能描述**一个原点锚定矩形**，因此该并集仅当其本身就是这样一个矩形时才可表示，
此时它等于逐维包围盒：

```text
out_valid[i] = min( shape[i], max( valid(target)[i], offset[i] + valid(source)[i] ) )
```

空隙（在增长维上 `offset > valid(target)`）或 L 形（某一维未被完全覆盖，
同时另一维在增长）会让包围盒把**两个操作数都未写入**的单元标记为有效，
故予以拒绝。该证明是**逐维**进行的：一个符号化的"旁路"维不会让
可证明的静态 L 形停止被拒绝。

该证明同样能判定**符号化的连续追加**（symbolic contiguous append）——
`offset` 与 `valid(target)` 是*同一个表达式*即可证明不存在空隙，
即便二者都不是 `ConstInt`。这正是累加器（accumulator）的惯用写法：

```text
assemble(acc /*valid [v, 128]*/, src /*valid [32, 128]*/, offset=[v, 0])
    -> valid [min(128, max(v, v + 32)), 128]
```

两个*不同*的符号化 extent 仍然无法证明，依旧被拒绝。

### 空累加器

由于"未设置即全有效"，新建的 `pl.create_tile([BM, BD], pl.FP32)` 是全有效的，
并集永远无法收窄。把累加器创建为空，正是让该规则可用的关键：

```python
dst    = pl.create_tile([BM, BD], pl.FP32, valid_shape=[0, 0])
packed = pl.tile.assemble(dst, y, [0, 0])   # -> valid_shape == valid(y)
```

只要每次追加保持可表示（沿单一维连续增长，其余各维完全覆盖），
连续追加会让包围盒单调增长。

## ND → 2D 降级契约

对 tile shape `[d0 ... d_{k-1}]`，flatten 为 `rows = Π(d0 ... d_{k-2})`、
`cols = d_{k-1}`，扁平化后的行下标为

```text
flat_row = i0*(d1*...*d_{k-2}) + i1*(d2*...*d_{k-2}) + ... + i_{k-2}
```

ND 有效区域是**连续的行前缀** —— 这也是 `(valid_row, valid_col)` 唯一能表达的形状 ——
当且仅当只存在一个*自由*行维：把行维按从高位到低位读取，
其之前的每一维都被钉死（`valid[j] == 1`，于是 `i_j` 被强制为 0，不贡献 stride），
其之后的每一维都必须完全有效。

```text
exists f in [0, k-2] such that
    valid[j] == 1           for all j <  f     // pinned: index forced to 0
    valid[j] == shape[j]    for all f < j <= k-2
```

在该前置条件下，乘积折叠 `Π(valid[0..k-2])` 恰好正确。
非单位外层维之下的部分有效中间维会产生*跨步*（strided）区域，予以拒绝。

两个实例。shape `[16,4,8]`、valid `[1,2,8]`：`i0` 被钉死，于是
`flat_row = i1 ∈ {0,1}` —— 连续，折叠得 `1*2 = 2`。✓
shape `[4,8,16]`、valid `[3,1,16]`：自由维为 `0` 但 `valid[1]=1 != 8`，于是
`flat_row = i0*8 ∈ {0,8,16}` —— 跨步，折叠会错误地给出 `3`。✗

"钉死"判据是 `valid[j] == 1` 而非 `shape[j] == 1`：`valid[j]` 为 1
无论物理 extent 多大都会强制 `i_j = 0`。**空**区域（任一维可证明为 `0`）
平凡地是一个前缀，折叠为零行。

参见 [`../passes/13-flatten_tile_nd_to_2d.md`](../passes/13-flatten_tile_nd_to_2d.md)。

## 面向用户的接口

| API | 用途 |
| --- | ---- |
| `pl.load(t, offs, shapes, valid_shapes=...)` | 为加载的 tile 附加有效区域 |
| `pl.slice(x, shape, offset, valid_shape=..., clamp=...)` | 切片；`clamp=True` 推导 ragged-tail extent |
| `pl.create_tile(shape, dtype, valid_shape=...)` | 以显式（可为空）区域创建 tile |
| `pl.valid_dim(t, i)` | 编译期查询第 `i` 轴的有效 extent |
| `pl.fillpad(t, pad_value=...)` | 用 pad 值填充无效区域 |
| `pl.store(t, offs, out, shapes=...)` | 部分写回 |

`pl.set_validshape` 存在，但属于面向编译器的内部 API（仅支持 rank-2）。
