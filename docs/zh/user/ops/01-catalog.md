# 算子目录

每个算子家族一行。签名在 docstring 里 —— 本页为何不重复它们，见 [算子](index.md)。

> **怎么读这些表：** **可达**列给出能用的最短拼法。`pl.` 表示该名字不带限定即可访问；`pl.tile.` / `pl.tensor.` 表示该算子是分层级的。标 **(t)** 的名字是为方便而重新导出到顶层的 tile 专属算子 —— `pl.load` **就是** `pl.tile.load`，不是派发器。

## 创建

| 算子 | 可达 | 作用 |
| ---- | ---- | ---- |
| `create_tensor` | `pl.` | 分配 DDR 张量。`manual_dep=True` 让它在整个生命周期退出依赖跟踪 |
| `create_tile` | `pl.` (t) | 分配片上缓冲区 |
| `create_l1` | `pl.` (t) | 显式在 L1 上分配 |
| `full` | `pl.` | 用常量填充的张量 |
| `arange` | `pl.` | 连续整数（`tensor.ci`） |
| `random` | `pl.` | 随机填充的张量 |
| `tri` | `pl.` (t) | 下三角或上三角掩码 tile；`upper=` 选边、`diagonal=` 平移，且只写有效区 |
| `const` | `pl.` | 带显式 dtype 的字面量 —— 见 [编译期指令](../language/05-directives.md#带类型的常量) |
| `array.create` | `pl.array.` | 分配核内数组 |

## 数据搬运

| 算子 | 可达 | 作用 |
| ---- | ---- | ---- |
| `load` | `pl.` (t) | DDR → `Vec`（默认）或 `Mat`，经由 `target_memory=` |
| `store` | `pl.` (t) | 片上 → DDR |
| `move` | `pl.` (t) | 片上 → 片上；进入 `Left` / `Right` / `Bias` 的唯一途径 |
| `reserve_buffer` | `pl.` | 预留跨核缓冲区 |
| `import_peer_buffer` | `pl.` | 引用对端核的缓冲区 |

哪些搬运合法见 [内存与数据搬运](../language/03-memory.md)。

## 逐元素算术

| 算子 | 可达 | 作用 |
| ---- | ---- | ---- |
| `add` `sub` `mul` `div` | `pl.` | 二元算术；右侧给 Python 数字会选中标量操作数形式 |
| `neg` `abs` `recip` | `pl.` | 取负、绝对值、倒数 |
| `rem` `rems` `fmod` `fmods` | `pl.` | 求余与浮点取模，张量与标量形式 |
| `addc` `subc` `addsc` `subsc` | `pl.` (t) | 带进位操作数的三输入加 / 减 |
| `part_add` `part_mul` `part_max` `part_min` | `pl.` | 分段算术 |

## 数学

| 算子 | 可达 | 作用 |
| ---- | ---- | ---- |
| `exp` `log` | `pl.` | 指数、自然对数 |
| `sqrt` `rsqrt` | `pl.` | 平方根；倒数平方根。`high_precision=` 只作用于张量路径，传给 Tile 会**抛异常** —— tile 级的精度由是否传入 scratch tile 决定：`pl.tile.rsqrt(src, tmp)` |
| `sin` `cos` | `pl.` | 三角函数 |

## 比较与选择

| 算子 | 可达 | 作用 |
| ---- | ---- | ---- |
| `cmp` `cmps` | `pl.` | 比较两个操作数 / 操作数与标量 |
| `maximum` `minimum` | `pl.` | 两个操作数的逐元素最大 / 最小 |
| `maximums` `minimums` | `pl.` (t) | 与标量的逐元素最大 / 最小 |
| `max` `min` | `pl.` (t) | 两个标量取最大 / 最小 —— **不是** tile 规约。规约 tile 请用 `row_max` / `col_max`（以及对应的 `min` 形式） |
| `sel` `sels` | `pl.` (t) | 按掩码选择，张量与标量形式 |

## 激活

| 算子 | 可达 | 作用 |
| ---- | ---- | ---- |
| `relu` | `pl.` (t) | 修正线性 |
| `prelu` `lrelu` | `pl.` (t) | 参数化 / 泄漏修正线性 |

## 位运算

| 算子 | 可达 | 作用 |
| ---- | ---- | ---- |
| `and_` `or_` `xor` `not_` | `pl.` | 位逻辑 |
| `ands` `ors` `xors` | `pl.` | 与标量的位逻辑 |
| `shl` `shr` | `pl.` | 左移 / 右移 |
| `shls` `shrs` | `pl.` | 按标量位数移位 |

## 规约

行规约折叠最后一个轴；列规约折叠第一个轴。

| 算子 | 可达 | 作用 |
| ---- | ---- | ---- |
| `row_sum` `row_prod` `row_max` `row_min` | `pl.` | 沿行规约 |
| `col_sum` `col_prod` `col_max` `col_min` | `pl.` | 沿列规约 |
| `row_argmax` `row_argmin` | `pl.` | 行极值的下标 |
| `col_argmax` `col_argmin` | `pl.` | 列极值的下标 |

若干规约接受 `tmp_tile` 参数。传入它会改变规约策略（二叉树 vs 顺序），从而改变浮点结合顺序 —— 结果是在容差内有差异，而不是错误。对部分有效 tile 的规约取决于填充值，见 [内存 § 有效形状](../language/03-memory.md#有效形状与填充)。

## 广播与展开

| 算子 | 可达 | 作用 |
| ---- | ---- | ---- |
| `row_expand` `col_expand` | `pl.` | 把被规约的轴广播回全宽 |
| `row_expand_add` `row_expand_sub` `row_expand_mul` | `pl.` | 广播与算术融合 |
| `row_expand_div` `row_expand_max` `row_expand_min` | `pl.` | 广播与除法、取最大、取最小融合 |
| `col_expand_add` `col_expand_sub` `col_expand_mul` | `pl.` | 按列的对应形式 |
| `col_expand_div` `col_expand_max` `col_expand_min` | `pl.` | 按列的除法、取最大、取最小 |
| `row_expand_expdif` `col_expand_expdif` | `pl.` | 与 `exp(x - m)` 融合的广播 —— softmax 的核心 |
| `expand_clone` `expands` | `pl.` | 把一个值广播到某个形状 |
| `fillpad` `fillpad_expand` | `pl.` | 填充无效区；可在同一步里广播 |

## 形状与布局

| 算子 | 可达 | 作用 |
| ---- | ---- | ---- |
| `reshape` | `pl.` | 重新解释维度 |
| `transpose` | `pl.` | 转置 |
| `slice` | `pl.` | 子区域；也可写作 `A[0:16, :]` |
| `concat` | `pl.` | 沿轴拼接 |
| `assemble` | `pl.` | 把子区域写回；也可写作 `dst[i:i+16] = src` |
| `reinterpret_view` | `pl.` | 不搬数据的重新解释 |
| `set_validshape` | `pl.` | 声明 tile 的有效区域 |
| `cast` | `pl.` | 转换 dtype —— 可能展开成多跳链，见 [LegalizeTileCast](../../dev/passes/15-legalize_tile_cast.md) |
| `dim` | `pl.` | 张量的运行期维度 |
| `read` `write` | `pl.` | 元素访问 |

## 线性代数

| 算子 | 可达 | 作用 |
| ---- | ---- | ---- |
| `matmul` | `pl.` | 矩阵乘；`a_trans=` / `b_trans=` 用来替代 DN 注解转置操作数 |
| `matmul_acc` | `pl.` | 乘累加进已有的 `Acc` tile |
| `matmul_bias` | `pl.` (t) | 带 bias 操作数的乘法 |
| `batch_matmul` | `pl.` (t) | 批量矩阵乘，**只接受 tile 操作数**。张量请调 `pl.matmul` —— rank > 2 会在降级时派发到 `tile.batch_matmul` |
| `gemv` `gemv_acc` `gemv_bias` | `pl.` (t) | 矩阵-向量形式 |
| `matmul_mx` `matmul_mx_acc` `matmul_mx_bias` | `pl.` (t) | MX 块缩放矩阵乘 —— 每个操作数由一块 FP8E4M3FN 数据 tile 和一块 FP8E8M0 scale tile 组成 |

## Gather、Scatter、排序

| 算子 | 可达 | 作用 |
| ---- | ---- | ---- |
| `gather` `gather_row` | `pl.` | 按索引 gather |
| `paged_gather` | `pl.` | 跨分页布局 gather |
| `gatherb` | `pl.` (t) | 按 UINT32 字节偏移 gather 32 字节块；一列偏移展开成 `32 / sizeof(output_dtype)` 个元素 |
| `mgather` | `pl.` (t) | 按索引 tile 从 DDR 张量 gather 行，带 `coalesce=` 与 `gather_oob=` 策略 |
| `scatter` `scatter_update` | `pl.` | 按索引 scatter；原地更新 |
| `mscatter` | `pl.` (t) | 带掩码的 scatter |
| `sort32` `mrgsort` | `pl.` | 对 32 元素组排序；归并已排序段 |

## Block 身份

| 算子 | 可达 | 作用 |
| ---- | ---- | ---- |
| `get_block_idx` | `pl.` | `pl.spmd` 下本 block 的索引 |
| `get_block_num` | `pl.` | block 总数 |
| `get_subblock_idx` | `pl.` | `pl.split_aiv` 下的 AIV lane 索引 |

## 跨核传输

混合 kernel 的接口 —— AIC 与 AIV 在同一个 InCore 函数内协作。

| 算子 | 可达 | 作用 |
| ---- | ---- | ---- |
| `tpush_to_aiv` `tpush_to_aic` | `pl.` | 把一块 tile 推给对端核 |
| `tpop_from_aic` `tpop_from_aiv` | `pl.` | 弹出对端推来的 tile |
| `tfree_to_aic` `tfree_to_aiv` | `pl.` | 把弹出的 slot 释放回生产者 |
| `aic_initialize_pipe` `aiv_initialize_pipe` | `pl.` | 建立跨核管道 |
| `aiv_shard` `aic_gather` | `pl.` | 在 AIV lane 间分片 / 在 AIC 上聚回 |
| `AUTO` | `pl.` | 由编译器选择管道参数的哨兵值 |

push 与 pop 必须**配对**，且每次 pop 都必须有对应的 `tfree`。涵盖这部分的教程尚未编写；机制见 [TPUSH/TPOP](../../reference/pto-isa/01-tpush_tpop.md) 与 [ExpandMixedKernel](../../dev/passes/22-expand_mixed_kernel.md)。

## 任务与依赖

| 算子 | 可达 | 作用 |
| ---- | ---- | ---- |
| `submit` `spmd_submit` | `pl.` | 派发 kernel 并捕获其生产者 TaskId |
| `no_dep` | `pl.` | 让单个任务的单个实参退出依赖跟踪 |
| `dump_tag` | `pl.` | 标记张量做选择性 dump |

见 [作用域与放置](../language/04-scopes.md)。

## 数组

| 算子 | 可达 | 作用 |
| ---- | ---- | ---- |
| `array.create` | `pl.array.` | 分配 |
| `array.get_element` | `pl.array.` | 读取；也可写作 `arr[i]` |
| `array.update_element` | `pl.array.` | 函数式更新；也可写作 `arr[i] = v` |

## 分布式

`pypto.language.distributed`（约定简写 `pld`）承载集合通信与远程原语。完整参考见[分布式编程](../distributed/index.md)；教程见[distributed/00-model.md](../distributed/00-model.md)。

### 功能矩阵

| 操作 | API | 模式 | ReduceOp | Atomic | 支持的 dtype | 说明 |
| ---- | --- | ---- | -------- | ------ | ------------ | ---- |
| AllReduce | `pld.tensor.allreduce` | `mesh`（InCore + HOST），`ring`（InCore + HOST） | `Sum`、`Max`、`Min`、`Prod`（mesh）；仅 `Sum`（HOST ring） | — | FP16、FP32（mesh；编译期硬性检查）；HOST ring：仅 FP32（4 字节） | Mesh: 每步 O(N) 远程流量。Ring: 每步 O(N/P) 远程流量，2(P-1) 步。 |
| AllGather | `pld.tensor.allgather` | — | — | — | 仅 FP32（HOST builtin）；任意 GM dtype（InCore） | 推式。输入和 target 必须是不同的 buffer。 |
| ReduceScatter | `pld.tensor.reduce_scatter` | — | 仅 `Sum` | — | 仅 FP32（HOST builtin）；任意 GM dtype（InCore） | 每个 rank 在调用前将全部 NR 个数据块写入。 |
| Broadcast | `pld.tensor.broadcast` | — | — | — | 仅 FP32（HOST builtin）；任意 GM dtype（InCore） | Root 在调用前将数据写入。 |
| All-to-All | `pld.tensor.all_to_all` | — | — | — | 仅 FP32（HOST builtin）；任意 GM dtype（InCore） | 个性化交换。输入和 target 必须是不同的 buffer。 |
| Barrier | `pld.tensor.barrier` | — | — | — | — | Signal 为 INT32，每次调用单次使用。 |
| Put | `pld.tensor.put` | — | — | `None_` / `Add` | 所有 GM dtype | `dst` 必须是 window-bound。支持分块和流水线 staging。 |
| Get | `pld.tensor.get` | — | — | — | 所有 GM dtype | `src` 必须是 window-bound。支持分块和流水线 staging。 |
| Notify | `pld.system.notify` | `AtomicAdd` / `Set` | — | — | — | 仅副作用的信号投递。 |
| Wait | `pld.system.wait` | `Eq` / `Ge` | — | — | — | 仅副作用的信号阻塞。 |
| Remote Load | `pld.tile.remote_load` | — | — | — | 任意（tile） | Tile 级跨 rank 加载。 |
| Remote Store | `pld.tile.remote_store` | — | — | — | 任意（tile） | Tile 级跨 rank 写入。 |

## See Also

- [选择命名空间](00-dispatch.md) —— 这些拼法该用哪一个。
- [IR 算子](../../dev/ir/05-operators.md) —— 这些名字背后的注册表。
- [PTOAS 算子状态](../../dev/ptoas-op-status.md) —— 各后端的支持情况。
