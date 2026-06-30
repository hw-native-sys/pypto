# 操作参考

所有操作通过 `import pypto.language as pl` 访问。

**符号说明：** `T` = `Tensor` 或 `Tile`（统一分发）。`IntLike` = `int | Scalar | Expr`。`Mem` = `pl.Mem`（`pl.MemorySpace` 的简写别名，两者等价）。

## 统一分发（`pl.*`）

根据输入类型自动选择 tensor 或 tile 实现。

| 名称 | 签名 | 说明 |
| ---- | ---- | ---- |
| `add` | `(lhs: T, rhs: T \| int \| float \| Scalar) -> T` | 逐元素加法 |
| `sub` | `(lhs: T, rhs: T \| int \| float \| Scalar) -> T` | 逐元素减法 |
| `mul` | `(lhs: T, rhs: T \| int \| float \| Scalar) -> T` | 逐元素乘法 |
| `div` | `(lhs: T, rhs: T \| int \| float \| Scalar) -> T` | 逐元素除法 |
| `part_add` | `(lhs: T, rhs: T) -> T` | 部分加（仅一侧有效时拷贝该侧） |
| `part_mul` | `(lhs: T, rhs: T) -> T` | 部分乘（仅一侧有效时拷贝该侧） |
| `part_max` | `(lhs: T, rhs: T) -> T` | 部分最大值（仅一侧有效时拷贝该侧） |
| `part_min` | `(lhs: T, rhs: T) -> T` | 部分最小值（仅一侧有效时拷贝该侧） |
| `fmod` | `(lhs: T, rhs: T \| int \| float \| Scalar) -> T` | 浮点取余（`torch.fmod`） |
| `fmods` | `(lhs: T, rhs: int \| float \| Scalar) -> T` | 与标量浮点取余 |
| `maximum` | `(lhs: T, rhs: T) -> T` | 逐元素最大值 |
| `exp` | `(input: T) -> T` | 逐元素指数 |
| `cast` | `(input: T, target_type: int \| DataType, mode="round") -> T` | 类型转换（`mode`：none、rint、round、floor、ceil、trunc、odd） |
| `reshape` | `(input: T, shape: Sequence[IntLike]) -> T` | 变形为新维度 |
| `transpose` | `(input: T, axis1: int, axis2: int) -> T` | 交换两个轴 |
| `slice` | `(input: T, shape: Sequence[IntLike], offset: Sequence[IntLike]) -> T` | 带偏移的切片 |
| `matmul` | `(lhs: T, rhs: T, out_dtype=None, a_trans=False, b_trans=False, c_matrix_nz=False) -> T` | 矩阵乘法 |
| `matmul_acc` | `(acc: T, lhs: T, rhs: T, a_trans=False, b_trans=False) -> T` | 带累加的矩阵乘法：`acc += lhs @ rhs` |
| `row_max` | `(input: T, tmp_tile: Tile \| None = None) -> T` | 行最大值（tile 路径需要 `tmp_tile`） |
| `row_sum` | `(input: T, tmp_tile: Tile \| None = None) -> T` | 行求和（tile 路径需要 `tmp_tile`） |
| `row_prod` | `(input: T, tmp_tile: Tile \| None = None) -> T` | 行乘积（tile 路径需要 `tmp_tile`） |
| `col_sum` | `(input: T, tmp_tile: Tile \| None = None) -> T` | 列求和；Tile 上传入 `tmp_tile` 启用二叉树归约，省略时使用顺序归约；Tensor 输入下沉为顺序归约路径 |
| `col_max` | `(input: T) -> T` | 列最大值 |
| `col_min` | `(input: T) -> T` | 列最小值 |
| `col_prod` | `(input: T) -> T` | 列乘积 |
| `row_argmax` | `(input: T, tmp_tile: Tile \| None = None) -> T` | 行 argmax（每行最大值的列索引，int32 输出；tile 路径需要 `tmp_tile`） |
| `row_argmin` | `(input: T, tmp_tile: Tile \| None = None) -> T` | 行 argmin（每行最小值的列索引，int32 输出；tile 路径需要 `tmp_tile`） |
| `col_argmax` | `(input: T, tmp_tile: Tile \| None = None) -> T` | 列 argmax（每列最大值的行索引，int32 输出；tile 路径需要 `tmp_tile`） |
| `col_argmin` | `(input: T, tmp_tile: Tile \| None = None) -> T` | 列 argmin（每列最小值的行索引，int32 输出；tile 路径需要 `tmp_tile`） |
| `rsqrt` | `(input: T, high_precision: bool = False) -> T` | 倒数平方根；`high_precision=True` 选择高精度路径（仅对 Tensor 输入生效，Tile 路径需要改用 `pl.tile.rsqrt(src, tmp=...)`） |
| `create` / `create_tile` | `(shape: Sequence[IntLike], dtype: DataType, target_memory: Mem) -> Tile` | 在指定内存空间创建 tile（tile-only，对应 `pl.tile.create`） |

## 仅 Tensor（`pl.tensor.*`）

操作 `Tensor` 对象（DDR 内存）。

| 名称 | 签名 | 说明 |
| ---- | ---- | ---- |
| `create` / `create_tensor` | `(shape: Sequence[IntLike], dtype: DataType, layout: TensorLayout = None, init_value: int \| float \| None = None) -> Tensor` | 创建新张量（可选 `layout` 参数，如 `pl.DN`、`pl.NZ`；`init_value` 由 AICPU 预填充缓冲区——`0` 对任意 dtype 清零，非零值需整型或 ≥32 位浮点 dtype） |
| `read` | `(tensor: Tensor, indices: Sequence[IntLike]) -> Scalar` | 读取指定索引的标量。语法糖：`A[i, j]` |
| `dim` | `(tensor: Tensor, axis: int) -> Scalar` | 获取维度大小（支持负索引） |
| `slice` | `(tensor: Tensor, shape: Sequence[IntLike], offset: Sequence[IntLike]) -> Tensor` | 切片。语法糖：`A[0:16, :]` |
| `reshape` | `(tensor: Tensor, shape: Sequence[IntLike]) -> Tensor` | 变形 |
| `transpose` | `(tensor: Tensor, axis1: int, axis2: int) -> Tensor` | 交换两个轴 |
| `assemble` | `(target: Tensor, source: Tensor, offset: Sequence[IntLike], *, atomic: AtomicType = AtomicType.None_) -> Tensor` | 将 source 写入 target 的指定偏移。语法糖（仅 SSA 前）：`target[i:i+H, j:j+W] = source`。`atomic=AtomicType.Add` 改为累加而非覆盖（split-K）——仅当 target 为函数输出（全局内存）时合法；浮点结果不确定，target 需预先清零，支持 dtype fp32/fp16/int32/int16/int8 |
| `scatter_update` | `(input: Tensor, dim: int, index: Tensor, src: Tensor) -> Tensor` | 按 `index` 指定的稀疏行位置，将 `src` 的行数据写入 `input`。`input`/`src`：2D `[rows, d]` 或 4D `[B, S, 1, d]`；`index`：2D `[b, s]` 整型。当前仅支持 `dim=-2` |
| `random` | `(key0, key1, counter0, counter1, counter2, counter3: int \| Scalar, shape: Sequence[IntLike], dtype: DataType = UINT32, rounds: int = 10) -> Tensor` | 基于计数器的（Philox/ChaCha 风格）随机数生成，下沉为 `tile.random`。由 key + counter 种子确定性生成。`dtype` ∈ {INT32, UINT32}；`rounds` ∈ {7, 10}。顶层别名 `pl.random`。**仅 A5** |
| `add` | `(lhs: Tensor, rhs: Tensor \| int \| float \| Scalar) -> Tensor` | 逐元素加法 |
| `sub` | `(lhs: Tensor, rhs: Tensor \| int \| float \| Scalar) -> Tensor` | 逐元素减法 |
| `mul` | `(lhs: Tensor, rhs: Tensor \| int \| float \| Scalar) -> Tensor` | 逐元素乘法 |
| `div` | `(lhs: Tensor, rhs: Tensor \| int \| float \| Scalar) -> Tensor` | 逐元素除法 |
| `adds` | `(lhs: Tensor, rhs: int \| float \| Scalar) -> Tensor` | 加标量 |
| `subs` | `(lhs: Tensor, rhs: int \| float \| Scalar) -> Tensor` | 减标量 |
| `muls` | `(lhs: Tensor, rhs: int \| float \| Scalar) -> Tensor` | 乘标量 |
| `divs` | `(lhs: Tensor, rhs: int \| float \| Scalar) -> Tensor` | 除以标量 |
| `part_add` | `(lhs: Tensor, rhs: Tensor) -> Tensor` | 部分加（仅一侧有效时拷贝该侧） |
| `part_mul` | `(lhs: Tensor, rhs: Tensor) -> Tensor` | 部分乘（仅一侧有效时拷贝该侧） |
| `part_max` | `(lhs: Tensor, rhs: Tensor) -> Tensor` | 部分最大值（仅一侧有效时拷贝该侧） |
| `part_min` | `(lhs: Tensor, rhs: Tensor) -> Tensor` | 部分最小值（仅一侧有效时拷贝该侧） |
| `fmod` | `(lhs: Tensor, rhs: Tensor \| int \| float \| Scalar) -> Tensor` | 浮点取余（`torch.fmod`） |
| `fmods` | `(lhs: Tensor, rhs: int \| float \| Scalar) -> Tensor` | 与标量浮点取余 |
| `maximum` | `(lhs: Tensor, rhs: Tensor) -> Tensor` | 逐元素最大值 |
| `row_max` | `(input: Tensor) -> Tensor` | 行最大值归约 |
| `row_sum` | `(input: Tensor) -> Tensor` | 行求和归约 |
| `row_prod` | `(input: Tensor) -> Tensor` | 行乘积归约 |
| `col_sum` | `(input: Tensor) -> Tensor` | 列求和归约（沿 axis=-2） |
| `col_max` | `(input: Tensor) -> Tensor` | 列最大值归约（沿 axis=-2） |
| `col_min` | `(input: Tensor) -> Tensor` | 列最小值归约（沿 axis=-2） |
| `col_prod` | `(input: Tensor) -> Tensor` | 列乘积归约（沿 axis=-2） |
| `row_argmax` | `(input: Tensor) -> Tensor` | 行 argmax 归约（int32 索引输出） |
| `row_argmin` | `(input: Tensor) -> Tensor` | 行 argmin 归约（int32 索引输出） |
| `col_argmax` | `(input: Tensor) -> Tensor` | 列 argmax 归约（沿 axis=-2，int32 索引输出） |
| `col_argmin` | `(input: Tensor) -> Tensor` | 列 argmin 归约（沿 axis=-2，int32 索引输出） |
| `rsqrt` | `(input: Tensor, high_precision: bool = False) -> Tensor` | 逐元素倒数平方根；`high_precision=True` 时编译器在下沉阶段分配临时 tile，启用高精度 PTO 路径（要求 tile 形状是编译期常量，与 `row_max`/`row_sum` 限制一致） |
| `exp` | `(input: Tensor) -> Tensor` | 逐元素指数 |
| `cast` | `(input: Tensor, target_type: DataType, mode="round") -> Tensor` | 类型转换 |
| `matmul` | `(lhs: Tensor, rhs: Tensor, out_dtype=None, a_trans=False, b_trans=False, c_matrix_nz=False) -> Tensor` | 矩阵乘法 |
| `matmul_acc` | `(acc: Tensor, lhs: Tensor, rhs: Tensor, a_trans=False, b_trans=False) -> Tensor` | 带累加的矩阵乘法：`acc += lhs @ rhs` |

## 数据搬运（`pl.tile.*`）

在内存层次结构之间传输数据。

| 名称 | 签名 | 说明 |
| ---- | ---- | ---- |
| `load` | `(tensor: Tensor, offsets: Sequence[IntLike], shapes: Sequence[IntLike], target_memory: Mem = Mem.Vec) -> Tile` | DDR → 片上 tile。`offsets` 和 `shapes` 均使用源 tensor 的坐标系。转置 matmul 操作数请对 load 结果叠加 `transpose_view`。 |
| `store` | `(tile: Tile, offsets: Sequence[IntLike], output_tensor: Tensor, *, atomic: AtomicType = AtomicType.None_) -> Tensor` | Tile → DDR（pipe 根据源 tile 内存空间自动推断）。`atomic=AtomicType.Add` 将 tile 累加到 DDR 现有内容上（split-K）；浮点结果不确定，目标需预先清零，支持 dtype fp32/fp16/int32/int16/int8 |
| `assemble` | `(target: Tile, source: Tile, offset: Sequence[IntLike]) -> Tile` | 将源 tile 写入目标 tile 的指定偏移处。语法糖（仅 SSA 前）：`target[i:i+H, j:j+W] = source` |
| `scatter_update` | `(input: Tile, dim: int, index: Tile, src: Tile) -> Tile` | 按 `index` tile 指定的稀疏行位置，将 `src` tile 的行数据写入 `input` tile。`input`/`src`：2D `[rows, d]` 或 4D `[B, S, 1, d]`；`index`：2D `[b, s]` 整型。降级为 `tile.scatter`（pto.tscatter，整行 flat 索引）实现。当前仅支持 `dim=-2` |
| `move` | `(tile: Tile, target_memory: Mem) -> Tile` | 在内存层级间移动 tile（包括 Vec→Vec 拷贝） |
| `create` | `(shape: Sequence[IntLike], dtype: DataType, target_memory: Mem = Mem.Vec) -> Tile` | 在指定内存空间创建 tile |
| `full` | `(shape: list[int], dtype: DataType, value: int \| float) -> Tile` | 创建用常量填充的 tile |
| `random` | `(key0, key1, counter0, counter1, counter2, counter3: int \| Scalar, shape: Sequence[int], valid_shape: Sequence[int] \| None = None, dtype: DataType = UINT32, rounds: int = 10) -> Tile` | 用基于计数器的（Philox/ChaCha 风格）伪随机值填充 tile，种子为 64 位 key + 128 位 counter。确定性：相同种子产生相同 tile。可选 `valid_shape`（每维 `<= shape`）只写入有效行/列，其余保持不变。`dtype` ∈ {INT32, UINT32}；`rounds` ∈ {7, 10}。仅支持 2D shape。**仅 A5**（`pto.trandom`） |
| `fillpad` | `(input: Tensor \| Tile, pad_value: PadValue \| int \| float = PadValue.zero) -> Tensor \| Tile` | 按指定 pad 值填充无效视图区域；接受 `PadValue.zero/max/min` 枚举，或字面量 `0`、`0.0`、`math.inf`、`-math.inf`（其他值会报错）；Tensor 输入会在 InCore 代码中下沉为 tile fillpad |
| `fillpad_expand` | `(input: Tensor \| Tile, shape: Sequence[IntLike], pad_value: PadValue \| int \| float = PadValue.zero) -> Tensor \| Tile` | 与 `fillpad` 类似，但目标 `shape` 在任一维度上都可以**大于**源：源的有效区域被拷贝到左上角，其余元素填充 `pad_value`。每个目标维度必须 `>=` 源维度。Tensor 输入会在 InCore 代码中下沉为 tile fillpad_expand |
| `get_block_idx` | `() -> Scalar` | 获取当前 block 索引（UINT64） |

## Tile 算术（`pl.tile.*`）

### 二元运算（Tile × Tile）

| 名称 | 签名 | 说明 |
| ---- | ---- | ---- |
| `add` | `(lhs: Tile, rhs: Tile) -> Tile` | 逐元素加法 |
| `sub` | `(lhs: Tile, rhs: Tile) -> Tile` | 逐元素减法 |
| `mul` | `(lhs: Tile, rhs: Tile) -> Tile` | 逐元素乘法 |
| `div` | `(lhs: Tile, rhs: Tile) -> Tile` | 逐元素除法 |
| `maximum` | `(lhs: Tile, rhs: Tile) -> Tile` | 逐元素最大值 |
| `minimum` | `(lhs: Tile, rhs: Tile) -> Tile` | 逐元素最小值 |

### 二元运算（Tile × 标量）

| 名称 | 签名 | 说明 |
| ---- | ---- | ---- |
| `adds` | `(lhs: Tile, rhs: int \| float \| Scalar) -> Tile` | 加标量 |
| `subs` | `(lhs: Tile, rhs: int \| float \| Scalar) -> Tile` | 减标量 |
| `muls` | `(lhs: Tile, rhs: int \| float \| Scalar) -> Tile` | 乘标量 |
| `divs` | `(lhs: Tile, rhs: int \| float \| Scalar) -> Tile` | 除以标量 |
| `maximums` | `(lhs: Tile, rhs: int \| float \| Scalar) -> Tile` | 与标量取最大值 |
| `minimums` | `(lhs: Tile, rhs: int \| float \| Scalar) -> Tile` | 与标量取最小值 |

### 三输入算术

| 名称 | 签名 | 说明 |
| ---- | ---- | ---- |
| `addc` | `(lhs: Tile, rhs: Tile, rhs2: Tile) -> Tile` | `lhs + rhs + rhs2` |
| `subc` | `(lhs: Tile, rhs: Tile, rhs2: Tile) -> Tile` | `lhs - rhs - rhs2` |
| `addsc` | `(lhs: Tile, rhs: int \| float \| Scalar, rhs2: Tile) -> Tile` | `lhs + 标量 + rhs2` |
| `subsc` | `(lhs: Tile, rhs: int \| float \| Scalar, rhs2: Tile) -> Tile` | `lhs - 标量 - rhs2` |

## Tile 数学（`pl.tile.*`）

| 名称 | 签名 | 说明 |
| ---- | ---- | ---- |
| `neg` | `(tile: Tile) -> Tile` | 取反 |
| `exp` | `(tile: Tile) -> Tile` | 指数 |
| `sqrt` | `(tile: Tile) -> Tile` | 平方根 |
| `rsqrt` | `(tile: Tile, tmp: Tile \| None = None) -> Tile` | 倒数平方根；传入 `tmp`（与 `tile` 同形状同 dtype）时选择高精度 PTO 路径 |
| `recip` | `(tile: Tile) -> Tile` | 倒数（1/x） |
| `log` | `(tile: Tile) -> Tile` | 自然对数 |
| `abs` | `(tile: Tile) -> Tile` | 绝对值 |

## Tile 归约（`pl.tile.*`）

| 名称 | 签名 | 说明 |
| ---- | ---- | ---- |
| `row_max` | `(tile: Tile, tmp_tile: Tile) -> Tile` | 行最大值（需要临时缓冲区） |
| `row_sum` | `(tile: Tile, tmp_tile: Tile) -> Tile` | 行求和（需要临时缓冲区） |
| `row_min` | `(tile: Tile, tmp_tile: Tile) -> Tile` | 行最小值（需要临时缓冲区） |
| `row_prod` | `(tile: Tile, tmp_tile: Tile) -> Tile` | 行乘积（需要临时缓冲区） |
| `col_sum` | `(tile: Tile, tmp_tile: Tile \| None = None) -> Tile` | 列求和；传入 `tmp_tile` 启用二叉树归约，省略时使用顺序归约 |
| `col_max` | `(tile: Tile) -> Tile` | 列最大值 |
| `col_min` | `(tile: Tile) -> Tile` | 列最小值 |
| `col_prod` | `(tile: Tile) -> Tile` | 列乘积 |
| `row_argmax` | `(tile: Tile, tmp_tile: Tile) -> Tile` | 行 argmax，每行最大值的列索引（需要临时缓冲区，int32 输出） |
| `row_argmin` | `(tile: Tile, tmp_tile: Tile) -> Tile` | 行 argmin，每行最小值的列索引（需要临时缓冲区，int32 输出） |
| `col_argmax` | `(tile: Tile, tmp_tile: Tile) -> Tile` | 列 argmax，每列最大值的行索引（需要临时缓冲区，int32 输出） |
| `col_argmin` | `(tile: Tile, tmp_tile: Tile) -> Tile` | 列 argmin，每列最小值的行索引（需要临时缓冲区，int32 输出） |
| `sum` | `(tile: Tile, axis: int, keepdim: bool = False) -> Tile` | 沿轴求和 |
| `max` | `(tile: Tile \| Scalar, axis: int \| Scalar = 0, keepdim: bool = False) -> Tile \| Scalar` | 沿轴取最大值 |
| `min` | `(tile: Tile \| Scalar, axis: int \| Scalar = 0, keepdim: bool = False) -> Tile \| Scalar` | 沿轴取最小值 |

## 线性代数（`pl.tile.*`）

| 名称 | 签名 | 说明 |
| ---- | ---- | ---- |
| `matmul` | `(lhs: Tile, rhs: Tile) -> Tile` | 矩阵乘法：`C = A @ B` |
| `matmul_acc` | `(acc: Tile, lhs: Tile, rhs: Tile) -> Tile` | `acc += A @ B` |
| `matmul_bias` | `(lhs: Tile, rhs: Tile, bias: Tile) -> Tile` | `C = A @ B + bias` |
| `gemv` | `(lhs: Tile, rhs: Tile) -> Tile` | GEMV：`C[1,N] = A[1,K] @ B[K,N]` |
| `gemv_acc` | `(acc: Tile, lhs: Tile, rhs: Tile) -> Tile` | 带累加的 GEMV |
| `gemv_bias` | `(lhs: Tile, rhs: Tile, bias: Tile) -> Tile` | 带偏置的 GEMV |

## 广播/扩展（`pl.tile.*`）

| 名称 | 签名 | 说明 |
| ---- | ---- | ---- |
| `row_expand` | `(target: Tile, row_vec: Tile) -> Tile` | 将 `row_vec[M,1]` 扩展到 `target[M,N]` |
| `row_expand_add` | `(tile: Tile, row_vec: Tile) -> Tile` | `tile + row_vec[M,1]` 广播 |
| `row_expand_sub` | `(tile: Tile, row_vec: Tile) -> Tile` | `tile - row_vec` 广播 |
| `row_expand_mul` | `(tile: Tile, row_vec: Tile) -> Tile` | `tile * row_vec` 广播 |
| `row_expand_div` | `(tile: Tile, row_vec: Tile) -> Tile` | `tile / row_vec` 广播 |
| `row_expand_max` | `(tile: Tile, row_vec: Tile) -> Tile` | `max(tile, row_vec)` 广播 |
| `row_expand_min` | `(tile: Tile, row_vec: Tile) -> Tile` | `min(tile, row_vec)` 广播 |
| `row_expand_expdif` | `(tile: Tile, row_vec: Tile) -> Tile` | `exp(tile - row_vec[M,1])` 广播 |
| `col_expand` | `(target: Tile, col_vec: Tile) -> Tile` | 将 `col_vec[1,N]` 扩展到 `target[M,N]` |
| `col_expand_mul` | `(tile: Tile, col_vec: Tile) -> Tile` | `tile * col_vec` 广播 |
| `col_expand_div` | `(tile: Tile, col_vec: Tile) -> Tile` | `tile / col_vec` 广播 |
| `col_expand_sub` | `(tile: Tile, col_vec: Tile) -> Tile` | `tile - col_vec` 广播 |
| `col_expand_add` | `(tile: Tile, col_vec: Tile) -> Tile` | `tile + col_vec[1,N]` 广播 |
| `col_expand_max` | `(tile: Tile, col_vec: Tile) -> Tile` | `max(tile, col_vec)` 广播 |
| `col_expand_min` | `(tile: Tile, col_vec: Tile) -> Tile` | `min(tile, col_vec)` 广播 |
| `col_expand_expdif` | `(tile: Tile, col_vec: Tile) -> Tile` | `exp(tile - col_vec[1,N])` 广播 |
| `expands` | `(target: Tile, scalar: int \| float \| Scalar) -> Tile` | 将标量扩展到 tile 形状 |

## 比较/选择（`pl.tile.*`）

比较类型：`EQ=0, NE=1, LT=2, LE=3, GT=4, GE=5`。`cmp` 和 `cmps` 返回目标相关的
packed predicate mask；A2/A3 上如需得到数值结果，请配合 `sel` 和显式
`UINT8 [1, 32]` scratch tile 使用。

| 名称 | 签名 | 说明 |
| ---- | ---- | ---- |
| `cmp` | `(lhs: Tile, rhs: Tile, cmp_type: int = 0) -> Tile` | 比较两个 tile |
| `cmps` | `(lhs: Tile, rhs: int \| float \| Scalar, cmp_type: int = 0) -> Tile` | tile 与标量比较 |
| `sel` | `(mask: Tile, lhs: Tile, rhs: Tile, tmp: Tile) -> Tile` | 选择：`mask 为真取 lhs，否则取 rhs`；`tmp` 是 TSEL scratch |
| `sels` | `(lhs: Tile, rhs: Tile, select_mode: int \| float \| Scalar) -> Tile` | 按标量模式选择 |

## 位运算（`pl.tile.*`）

| 名称 | 签名 | 说明 |
| ---- | ---- | ---- |
| `and_` | `(lhs: Tile, rhs: Tile) -> Tile` | 按位与 |
| `ands` | `(lhs: Tile, rhs: int \| Scalar) -> Tile` | 与标量按位与 |
| `or_` | `(lhs: Tile, rhs: Tile) -> Tile` | 按位或 |
| `ors` | `(lhs: Tile, rhs: int \| Scalar) -> Tile` | 与标量按位或 |
| `xor` | `(lhs: Tile, rhs: Tile, tmp: Tile) -> Tile` | 按位异或（需要 tmp） |
| `xors` | `(lhs: Tile, rhs: int \| Scalar, tmp: Tile) -> Tile` | 与标量异或（需要 tmp） |
| `not_` | `(tile: Tile) -> Tile` | 按位取反 |
| `shl` | `(lhs: Tile, rhs: Tile) -> Tile` | 左移 |
| `shls` | `(lhs: Tile, rhs: int \| Scalar) -> Tile` | 左移标量位 |
| `shr` | `(lhs: Tile, rhs: Tile) -> Tile` | 右移 |
| `shrs` | `(lhs: Tile, rhs: int \| Scalar) -> Tile` | 右移标量位 |
| `rem` | `(lhs: Tile, rhs: Tile) -> Tile` | 取余/取模 |
| `rems` | `(lhs: Tile, rhs: int \| float \| Scalar) -> Tile` | 与标量取余 |
| `part_add` | `(lhs: Tile, rhs: Tile) -> Tile` | 部分加（仅一侧有效时拷贝该侧） |
| `part_mul` | `(lhs: Tile, rhs: Tile) -> Tile` | 部分乘（仅一侧有效时拷贝该侧） |
| `part_max` | `(lhs: Tile, rhs: Tile) -> Tile` | 部分最大值（仅一侧有效时拷贝该侧） |
| `part_min` | `(lhs: Tile, rhs: Tile) -> Tile` | 部分最小值（仅一侧有效时拷贝该侧） |
| `fmod` | `(lhs: Tile, rhs: Tile) -> Tile` | 浮点取余（`torch.fmod`） |
| `fmods` | `(lhs: Tile, rhs: int \| float \| Scalar) -> Tile` | 与标量浮点取余 |

## 激活函数（`pl.tile.*`）

| 名称 | 签名 | 说明 |
| ---- | ---- | ---- |
| `relu` | `(tile: Tile) -> Tile` | ReLU：`max(0, x)` |
| `lrelu` | `(tile: Tile, slope: int \| float \| Scalar) -> Tile` | 带标量斜率的 Leaky ReLU |
| `prelu` | `(tile: Tile, slope: Tile, tmp: Tile) -> Tile` | 参数化 ReLU（需要 tmp） |

## 形状操作（`pl.tile.*`）

| 名称 | 签名 | 说明 |
| ---- | ---- | ---- |
| `slice` | `(tile: Tile, shape: Sequence[IntLike], offset: Sequence[IntLike]) -> Tile` | 切片（最多 2D）。语法糖：`A[0:16, :]` |
| `reshape` | `(tile: Tile, shape: Sequence[IntLike]) -> Tile` | 变形（最多 2D） |
| `transpose` | `(tile: Tile, axis1: int, axis2: int) -> Tile` | 交换两个轴 |
| `cast` | `(tile: Tile, target_type: DataType, mode="round") -> Tile` | 类型转换 |

## DSL 辅助函数（`pl.*`）

| 名称 | 签名 | 说明 |
| ---- | ---- | ---- |
| `range` | `(*args: int \| Scalar, init_values: tuple \| None = None) -> RangeIterator` | 顺序 for 循环。参数：`(stop)`、`(start, stop)` 或 `(start, stop, step)` |
| `parallel` | `(*args: int \| Scalar, init_values: tuple \| None = None) -> RangeIterator` | 并行 for 循环（与 range 相同但并行） |
| `while_` | `(*, init_values: tuple) -> WhileIterator` | While 循环（始终需要 init_values） |
| `yield_` | `(*values: Any) -> Any \| tuple[Any, ...]` | 从 for/if 作用域 yield 值 |
| `cond` | `(condition: bool \| Scalar) -> None` | 设置 while 循环条件（必须是第一条语句） |
| `const` | `(value: int \| float, dtype: DataType) -> int \| float` | 类型化常量 |
| `incore` | `() -> IncoreContext` | InCore 作用域的上下文管理器 |
| `dynamic` | `(name: str) -> DynVar` | 创建动态维度变量 |
| `create_tensor` | `(shape: Sequence[IntLike], dtype: DataType, layout: TensorLayout = None, init_value: int \| float \| None = None) -> Tensor` | 创建张量（从 `pl.tensor` 提升；`init_value` 由 AICPU 预填充缓冲区——`0` 对任意 dtype 清零，非零值需整型或 ≥32 位浮点 dtype） |
