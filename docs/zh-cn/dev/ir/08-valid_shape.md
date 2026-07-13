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

**D1 —— 后端 tile buffer 物理上是二维的。** 在 `FlattenTileNdTo2D` 之前，逻辑
`TileType` 可以是 N 维；该 pass 是阶段边界：之后每个 `TileType` 的 rank 至多为 2，
且物理 extent 均为 `ConstInt`。PTO 的 `tile_buf` 始终严格为二维；逻辑 rank-1 的
`[N]` tile 由 `ExtractTileTypeInfo` 规范化成物理 `[1, N]`。运行时动态性承载在
`valid_shape` 上，其元素可以是 `ConstInt`、`Var`，或像 `pl.min(...)` 这样的 `Call`。
该 rank / 静态 extent 约束是 flatten 之后的不变式，而非全局构造规则。

**D2 —— 未设置即表示全部有效。** 缺省或为空的 `valid_shape` 表示完整的物理 shape。

**规范编码。** 与物理 shape 结构相等的 `valid_shape` 不携带任何信息，
因此构造时会清空该字段。只有其余所有字段都与隐式 view 一致时，才重置整个
optional view。这样，全有效 `TileType` 上独立有意义的 `stride` / `layout` /
`start_offset` / `pad` 会保留；`TensorType` 上的 `stride` / `layout` / `pad` 亦然。

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

# A non-default tile view survives; only its redundant valid_shape is cleared.
strided = ir.TileType(
    [ci(128), ci(128)], DataType.FP32, None,
    ir.TileView(valid_shape=[ci(128), ci(128)], stride=[ci(128), ci(1)]),
    ir.Mem.Vec)
assert len(strided.tile_view.valid_shape) == 0
assert len(strided.tile_view.stride) == 2

# TENSOR: only the valid_shape field is cleared; the stride/layout view survives.
# A PARTIAL valid_shape (e.g. [32, 64]) is kept verbatim on either view.
tv = ir.TensorView(stride=[ci(128), ci(1)], layout=ir.TensorLayout.ND,
                   valid_shape=[ci(128), ci(128)])
t = ir.TensorType([ci(128), ci(128)], DataType.FP32, tensor_view=tv)
assert len(t.tensor_view.valid_shape) == 0
```

**良构不变式**（启用 `TypeCheck` property verifier 时）：
`rank(valid) == rank(shape)`，且 `0 <= valid[i] <= shape[i]`。算术分析器会拒绝
所有可证明为假的关系；确实无法判定的符号化边界则推迟处理。自动 property
verification 可以通过 `VerificationLevel.None` 关闭，因此这并非始终执行的构造器保证。
接受或推导有效区域的算子还会在类型推导边界调用 `ValidateValidShapeBounds`，
即使关闭自动验证，也会拒绝可证明的违规。例如，在 `[128, 128]` 的 tile 上
`pl.load(valid_shapes=[999, 999])` 会被拒绝。

**`GetValidShape()` 是唯一事实来源**
（[`include/pypto/ir/type_inference.h`](../../../../include/pypto/ir/type_inference.h)）：
有 view 时返回其 `valid_shape`，否则返回物理 `shape_`，
因此"未设置"与"显式全有效"对所有消费者不可区分。

**新计算结果不是 view。** 新计算出的值会保留语义上的 effective `valid_shape`，
但不能意外继承属于源 allocation 的元数据。新的 tensor 结果使用默认 ND layout、
空 stride 和 null padding。新的 tile 结果只保留生成结果所需的 block/scatter layout
与 fractal 约束；源的 `stride`、`start_offset`、`memref` 和 `pad` 不传播。
只有 view-producing 或 in-place 算子自身的契约明确要求时，才会保留或重新计算
alias / padding 元数据。

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
| `load(valid_shapes=)` | 设置；与源 tensor 的区域**求交**；继承 `pad` | 窗口与源 rank 不同、请求的有效范围超过 tile 容量、请求的传输越过源分配范围，或 offset 为负 |
| 一元 / cast / 标量二元 / move | 在新结果上保留输入的 effective 区域 | —— |
| 逐元素二元 / 多操作数 | 逐维**一致性**：非广播贡献者必须可证明相等；`valid_shape` 从不广播 | 相等关系为假或**无法证明**；广播单位轴无法证明有效 extent 为 1 |
| `part_add` / `part_mul` / `part_max` / `part_min` | 逐维**并集**（任一源有效即有效） | 并集不是原点锚定矩形 |
| reduction | 丢弃归约轴，保留其余 | —— |
| 二维 `matmul(A[M,K], B[K,N])` | `[valid(A)[M], valid(B)[N]]`；PTO 按 `valid(A)[K]` 收缩 | `valid(A)[K] <= valid(B)[K]` 为假或无法证明 |
| `tile.batch_matmul` | 广播物理 batch shape；传播部分 M/N，并沿用方向性 K 规则 | 任一输入 batch 轴无法证明完全有效（包括符号相等无法证明） |
| `assemble(target, source, off)` | target 与 source-valid 写入矩形的逐维包围盒 | offset 为负；tensor-valid 传输（或 tile 物理 subview）无法放入目标；或并集不是可证明的原点锚定矩形（空隙 / L 形 / 无法证明） |
| `slice(clamp=True)` | 在 `offset` 处把完整 rank 窗口裁剪到源区域（从不放大） | 窗口 / offset / 源 rank 不一致、offset 为负，或有效传输超出源分配范围 |
| `slice(..., drop_dims=...)` | 仅删除静态物理单位轴，且求交后的有效 extent 必须可证明为 1 | 被删除轴为空，或其单位有效性无法证明 |
| `reshape` | 映射连续扁平前缀；插入/删除可证明完全有效的物理单位轴可精确保留任意矩形；空区域仍为空 | 真正重排数据时输入不是连续前缀，或被删除单位轴无法证明有效 |
| `transpose` / `extract` / `concat` | 置换 / 切出 / 堆叠区域 | `concat` 的非末位操作数部分有效（L 形） |
| 间接索引 / gather / scatter / sort 族 | 失败关闭：按完整契约消费的 data、index、workspace、accumulator 或 destination 必须可证明完全有效 | 任一必需操作数部分有效，或完全有效的相等关系无法证明 |
| AIV shard / gather 边界 | 切分轴必须可证明完全有效；非切分轴的有效范围保持不变 | 切分轴部分有效或符号相等性未知 |
| `store` | 精确传输 `valid(tile)`，或 flatten 注入的原 rank 分区 | 传输 rank 与目标不同；offset 可证明为负；或可证明 `offset + transfer > destination` |
| tensor→tile `load`；tensor 计算算子 | 跨边界保留 effective 区域；新计算遵循上述元数据规则 | 同其 tile 对应算子 |

控制流 join 也属于契约。每个 `if` 分支的 yield 与声明的 return variable
必须在 dtype、物理 shape、effective `valid_shape` 和 padding policy 上一致（缺少 view 时
视为 `PadValue.null`）；循环携带值遵循相同不变式。符号化有效 extent 若无法证明相等
会被拒绝，而不会让 join 注解加宽任一路径。

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
其之前的每一维要么由 `valid[j] == 1` 钉死，要么物理上是单位轴
（`shape[j] == 1`，此时符号化 0/1 validity 只是空/非空门控）；其之后的每一维都必须完全有效。

```text
exists f in [0, k-2] such that
    valid[j] == 1 OR shape[j] == 1  for all j < f
    valid[j] == shape[j]            for all f < j <= k-2
```

在该前置条件下，乘积折叠 `Π(valid[0..k-2])` 恰好正确。
非单位外层维之下的部分有效中间维会产生*跨步*（strided）区域，予以拒绝。

两个实例。shape `[16,4,8]`、valid `[1,2,8]`：`i0` 被钉死，于是
`flat_row = i1 ∈ {0,1}` —— 连续，折叠得 `1*2 = 2`。✓
shape `[4,8,16]`、valid `[3,1,16]`：自由维为 `0` 但 `valid[1]=1 != 8`，于是
`flat_row = i0*8 ∈ {0,8,16}` —— 跨步，折叠会错误地给出 `3`。✗

`valid[j] == 1` 无论物理 extent 多大都会钉死下标。物理单位轴上的符号化 validity
同样安全：良构不变式把它限制为 0 或 1，因此只会门控整个扁平前缀。
**空**区域（任一维可证明为 `0`）平凡地是一个前缀，折叠为零行。

参见 [`../passes/13-flatten_tile_nd_to_2d.md`](../passes/13-flatten_tile_nd_to_2d.md)。

## 面向用户的接口

| API | 用途 |
| --- | ---- |
| `pl.load(t, offs, shapes, valid_shapes=...)` | 为加载的 tile 附加有效区域 |
| `pl.slice(x, shape, offset, valid_shape=..., drop_dims=..., clamp=...)` | 完整 rank 切片；`drop_dims` 显式删除有效单位轴，`clamp=True` 推导 ragged-tail extent |
| `pl.create_tile(shape, dtype, valid_shape=...)` | 以显式（可为空）区域创建 tile |
| `pl.valid_dim(t, i)` | 编译期查询第 `i` 轴的有效 extent |
| `pl.fillpad(t, pad_value=...)` | 用 pad 值填充无效区域 |
| `pl.store(t, offs, out, shapes=...)` | 部分写回 |

`pl.set_validshape` 存在，但属于面向编译器的内部 API（仅支持 rank-2）。
