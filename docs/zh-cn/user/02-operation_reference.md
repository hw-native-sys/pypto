# 操作参考

所有操作通过 `import pypto.language as pl` 访问。

**符号说明：** `T` = `Tensor` 或 `Tile`（统一分发）。`IntLike` = `int | Scalar | Expr`。

## 统一分发（`pl.*`）

根据输入类型自动选择 tensor 或 tile 实现。

| 名称 | 签名 | 说明 |
| ---- | ---- | ---- |
| `add` | `(lhs: T, rhs: T \| int \| float \| Scalar) -> T` | 逐元素加法 |
| `sub` | `(lhs: T, rhs: T \| int \| float \| Scalar) -> T` | 逐元素减法 |
| `mul` | `(lhs: T, rhs: T \| int \| float \| Scalar) -> T` | 逐元素乘法 |
| `div` | `(lhs: T, rhs: T \| int \| float \| Scalar) -> T` | 逐元素除法 |
| `maximum` | `(lhs: T, rhs: T) -> T` | 逐元素最大值 |
| `exp` | `(input: T) -> T` | 逐元素指数 |
| `cast` | `(input: T, target_type: int \| DataType, mode="round") -> T` | 类型转换（`mode`：none、rint、round、floor、ceil、trunc、odd） |
| `reshape` | `(input: T, shape: Sequence[IntLike]) -> T` | 变形为新维度 |
| `transpose` | `(input: T, axis1: int, axis2: int) -> T` | 交换两个轴 |
| `view` | `(input: T, shape: Sequence[IntLike], offset: Sequence[IntLike]) -> T` | 带偏移的切片/视图 |
| `matmul` | `(lhs: T, rhs: T, out_dtype=None, a_trans=False, b_trans=False, c_matrix_nz=False) -> T` | 矩阵乘法 |
| `row_max` | `(input: T, tmp_tile: Tile \| None = None) -> T` | 行最大值（tile 路径需要 `tmp_tile`） |
| `row_sum` | `(input: T, tmp_tile: Tile \| None = None) -> T` | 行求和（tile 路径需要 `tmp_tile`） |
| `create_tile` | `(shape: Sequence[IntLike], dtype: DataType, target_memory: MemorySpace = MemorySpace.Vec) -> Tile` | 在指定内存空间创建 tile（tile-only，提升自 `pl.block.create_tile`） |

## 仅 Tensor（`pl.tensor.*`）

操作 `Tensor` 对象（DDR 内存）。

| 名称 | 签名 | 说明 |
| ---- | ---- | ---- |
| `create_tensor` | `(shape: Sequence[IntLike], dtype: DataType) -> Tensor` | 创建新张量 |
| `read` | `(tensor: Tensor, indices: Sequence[IntLike]) -> Scalar` | 读取指定索引的标量 |
| `dim` | `(tensor: Tensor, axis: int) -> Scalar` | 获取维度大小（支持负索引） |
| `view` | `(tensor: Tensor, shape: Sequence[IntLike], offset: Sequence[IntLike]) -> Tensor` | 切片/视图 |
| `reshape` | `(tensor: Tensor, shape: Sequence[IntLike]) -> Tensor` | 变形 |
| `transpose` | `(tensor: Tensor, axis1: int, axis2: int) -> Tensor` | 交换两个轴 |
| `assemble` | `(target: Tensor, source: Tensor, offset: Sequence[IntLike]) -> Tensor` | 将 source 写入 target 的指定偏移 |
| `add` | `(lhs: Tensor, rhs: Tensor \| int \| float \| Scalar) -> Tensor` | 逐元素加法 |
| `sub` | `(lhs: Tensor, rhs: Tensor \| int \| float \| Scalar) -> Tensor` | 逐元素减法 |
| `mul` | `(lhs: Tensor, rhs: Tensor \| int \| float \| Scalar) -> Tensor` | 逐元素乘法 |
| `div` | `(lhs: Tensor, rhs: Tensor \| int \| float \| Scalar) -> Tensor` | 逐元素除法 |
| `maximum` | `(lhs: Tensor, rhs: Tensor) -> Tensor` | 逐元素最大值 |
| `row_max` | `(input: Tensor) -> Tensor` | 行最大值归约 |
| `row_sum` | `(input: Tensor) -> Tensor` | 行求和归约 |
| `exp` | `(input: Tensor) -> Tensor` | 逐元素指数 |
| `cast` | `(input: Tensor, target_type: DataType, mode="round") -> Tensor` | 类型转换 |
| `matmul` | `(lhs: Tensor, rhs: Tensor, out_dtype=None, a_trans=False, b_trans=False, c_matrix_nz=False) -> Tensor` | 矩阵乘法 |

## 数据搬运（`pl.block.*`）

在内存层次结构之间传输数据。

| 名称 | 签名 | 说明 |
| ---- | ---- | ---- |
| `load` | `(tensor: Tensor, offsets: Sequence[IntLike], shapes: Sequence[IntLike], target_memory: MemorySpace = MemorySpace.Vec) -> Tile` | DDR → 片上 tile |
| `store` | `(tile: Tile, offsets: Sequence[IntLike], shapes: Sequence[IntLike], output_tensor: Tensor) -> Tensor` | Tile → DDR |
| `l0c_store` | `(tile: Tile, offsets: Sequence[IntLike], shapes: Sequence[IntLike], output_tensor: Tensor) -> Tensor` | Acc tile → DDR |
| `move` | `(tile: Tile, target_memory: MemorySpace, transpose: bool = False) -> Tile` | 在内存层级间移动 tile |
| `vec_move` | `(tile: Tile) -> Tile` | 在 Vec 内存内拷贝 tile |
| `create_tile` | `(shape: Sequence[IntLike], dtype: DataType, target_memory: MemorySpace = MemorySpace.Vec) -> Tile` | 在指定内存空间创建 tile |
| `full` | `(shape: list[int], dtype: DataType, value: int \| float) -> Tile` | 创建用常量填充的 tile |
| `fillpad` | `(tile: Tile) -> Tile` | 用填充值填充 tile |
| `get_block_idx` | `() -> Scalar` | 获取当前 block 索引（UINT64） |

## Tile 算术（`pl.block.*`）

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
| `maxs` | `(lhs: Tile, rhs: int \| float \| Scalar) -> Tile` | 与标量取最大值 |
| `mins` | `(lhs: Tile, rhs: int \| float \| Scalar) -> Tile` | 与标量取最小值 |

### 三输入算术

| 名称 | 签名 | 说明 |
| ---- | ---- | ---- |
| `addc` | `(lhs: Tile, rhs: Tile, rhs2: Tile) -> Tile` | `lhs + rhs + rhs2` |
| `subc` | `(lhs: Tile, rhs: Tile, rhs2: Tile) -> Tile` | `lhs - rhs - rhs2` |
| `addsc` | `(lhs: Tile, rhs: int \| float \| Scalar, rhs2: Tile) -> Tile` | `lhs + 标量 + rhs2` |
| `subsc` | `(lhs: Tile, rhs: int \| float \| Scalar, rhs2: Tile) -> Tile` | `lhs - 标量 - rhs2` |

## Tile 数学（`pl.block.*`）

| 名称 | 签名 | 说明 |
| ---- | ---- | ---- |
| `neg` | `(tile: Tile) -> Tile` | 取反 |
| `exp` | `(tile: Tile) -> Tile` | 指数 |
| `sqrt` | `(tile: Tile) -> Tile` | 平方根 |
| `rsqrt` | `(tile: Tile) -> Tile` | 倒数平方根 |
| `recip` | `(tile: Tile) -> Tile` | 倒数（1/x） |
| `log` | `(tile: Tile) -> Tile` | 自然对数 |
| `abs` | `(tile: Tile) -> Tile` | 绝对值 |

## Tile 归约（`pl.block.*`）

| 名称 | 签名 | 说明 |
| ---- | ---- | ---- |
| `row_max` | `(tile: Tile, tmp_tile: Tile) -> Tile` | 行最大值（需要临时缓冲区） |
| `row_sum` | `(tile: Tile, tmp_tile: Tile) -> Tile` | 行求和（需要临时缓冲区） |
| `row_min` | `(tile: Tile, tmp_tile: Tile) -> Tile` | 行最小值（需要临时缓冲区） |
| `sum` | `(tile: Tile, axis: int, keepdim: bool = False) -> Tile` | 沿轴求和 |
| `max` | `(tile: Tile \| Scalar, axis: int \| Scalar = 0, keepdim: bool = False) -> Tile \| Scalar` | 沿轴取最大值 |
| `min` | `(tile: Tile \| Scalar, axis: int \| Scalar = 0, keepdim: bool = False) -> Tile \| Scalar` | 沿轴取最小值 |

## 线性代数（`pl.block.*`）

| 名称 | 签名 | 说明 |
| ---- | ---- | ---- |
| `matmul` | `(lhs: Tile, rhs: Tile) -> Tile` | 矩阵乘法：`C = A @ B` |
| `matmul_acc` | `(acc: Tile, lhs: Tile, rhs: Tile) -> Tile` | `acc += A @ B` |
| `matmul_bias` | `(lhs: Tile, rhs: Tile, bias: Tile) -> Tile` | `C = A @ B + bias` |
| `gemv` | `(lhs: Tile, rhs: Tile) -> Tile` | GEMV：`C[1,N] = A[1,K] @ B[K,N]` |
| `gemv_acc` | `(acc: Tile, lhs: Tile, rhs: Tile) -> Tile` | 带累加的 GEMV |
| `gemv_bias` | `(lhs: Tile, rhs: Tile, bias: Tile) -> Tile` | 带偏置的 GEMV |

## 广播/扩展（`pl.block.*`）

| 名称 | 签名 | 说明 |
| ---- | ---- | ---- |
| `row_expand` | `(src: Tile) -> Tile` | 将 `src[i,0]` 广播到每行 |
| `row_expand_add` | `(tile: Tile, row_vec: Tile) -> Tile` | `tile + row_vec[M,1]` 广播 |
| `row_expand_sub` | `(tile: Tile, row_vec: Tile) -> Tile` | `tile - row_vec` 广播 |
| `row_expand_mul` | `(tile: Tile, row_vec: Tile) -> Tile` | `tile * row_vec` 广播 |
| `row_expand_div` | `(tile: Tile, row_vec: Tile) -> Tile` | `tile / row_vec` 广播 |
| `col_expand` | `(target: Tile, col_vec: Tile) -> Tile` | 将 `col_vec[1,N]` 扩展到 `target[M,N]` |
| `col_expand_mul` | `(tile: Tile, col_vec: Tile) -> Tile` | `tile * col_vec` 广播 |
| `col_expand_div` | `(tile: Tile, col_vec: Tile) -> Tile` | `tile / col_vec` 广播 |
| `col_expand_sub` | `(tile: Tile, col_vec: Tile) -> Tile` | `tile - col_vec` 广播 |
| `expands` | `(target: Tile, scalar: int \| float \| Scalar) -> Tile` | 将标量扩展到 tile 形状 |

## 比较/选择（`pl.block.*`）

比较类型：`EQ=0, NE=1, LT=2, LE=3, GT=4, GE=5`

| 名称 | 签名 | 说明 |
| ---- | ---- | ---- |
| `cmp` | `(lhs: Tile, rhs: Tile, cmp_type: int = 0) -> Tile` | 比较两个 tile |
| `cmps` | `(lhs: Tile, rhs: int \| float \| Scalar, cmp_type: int = 0) -> Tile` | tile 与标量比较 |
| `sel` | `(mask: Tile, lhs: Tile, rhs: Tile) -> Tile` | 选择：`mask 为真取 lhs，否则取 rhs` |
| `sels` | `(lhs: Tile, rhs: Tile, select_mode: int \| float \| Scalar) -> Tile` | 按标量模式选择 |

## 位运算（`pl.block.*`）

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

## 激活函数（`pl.block.*`）

| 名称 | 签名 | 说明 |
| ---- | ---- | ---- |
| `relu` | `(tile: Tile) -> Tile` | ReLU：`max(0, x)` |
| `lrelu` | `(tile: Tile, slope: int \| float \| Scalar) -> Tile` | 带标量斜率的 Leaky ReLU |
| `prelu` | `(tile: Tile, slope: Tile, tmp: Tile) -> Tile` | 参数化 ReLU（需要 tmp） |

## 形状操作（`pl.block.*`）

| 名称 | 签名 | 说明 |
| ---- | ---- | ---- |
| `view` | `(tile: Tile, shape: Sequence[IntLike], offset: Sequence[IntLike]) -> Tile` | 切片/视图（最多 2D） |
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
| `create_tensor` | `(shape: Sequence[IntLike], dtype: DataType) -> Tensor` | 创建张量（从 `pl.tensor` 提升） |
