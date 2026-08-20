# 算子目录

每个算子家族一行。每个名字都链到 [API 参考](../../api/index.md)看签名 —— 本页为何不重复它们，见 [算子](index.md)。

> **怎么读这些表：** **可达**列给出能用的最短拼法。`pl.` 表示该名字不带限定即可访问；`pl.tile.` / `pl.tensor.` 表示该算子是分层级的。标 **(t)** 的名字是为方便而重新导出到顶层的 tile 专属算子 —— `pl.load` **就是** `pl.tile.load`，不是派发器。

## 创建

| 算子 | 可达 | 作用 |
| ---- | ---- | ---- |
| [`create_tensor`][pypto.language.tensor.create] | `pl.` | 分配 DDR 张量。`manual_dep=True` 让它在整个生命周期退出依赖跟踪 |
| [`create_tile`][pypto.language.tile.create] | `pl.` (t) | 分配片上缓冲区 |
| [`create_l1`][pypto.language.tensor.create_l1] | `pl.` (t) | 显式在 L1 上分配 |
| [`full`][pypto.language.tensor.full] | `pl.` | 用常量填充的张量 |
| [`arange`][pypto.language.tensor.ci] | `pl.` | 连续整数（`tensor.ci`） |
| [`random`][pypto.language.tensor.random] | `pl.` | 随机填充的张量 |
| [`tri`][pypto.language.tile.tri] | `pl.` (t) | 下三角或上三角掩码 tile；`upper=` 选边、`diagonal=` 平移，且只写有效区 |
| [`const`][pypto.language.const] | `pl.` | 带显式 dtype 的字面量 —— 见 [编译期指令](../language/05-directives.md#带类型的常量) |
| `array.create` | `pl.array.` | 分配核内数组 |

## 数据搬运

| 算子 | 可达 | 作用 |
| ---- | ---- | ---- |
| [`load`][pypto.language.tile.load] | `pl.` (t) | DDR → `Vec`（默认）或 `Mat`，经由 `target_memory=` |
| [`store`][pypto.language.tile.store] | `pl.` (t) | 片上 → DDR |
| [`move`][pypto.language.tile.move] | `pl.` (t) | 片上 → 片上；进入 `Left` / `Right` / `Bias` 的唯一途径 |
| [`reserve_buffer`][pypto.language.system.reserve_buffer] | `pl.` | 预留跨核缓冲区 |
| [`import_peer_buffer`][pypto.language.system.import_peer_buffer] | `pl.` | 引用对端核的缓冲区 |

哪些搬运合法见 [内存与数据搬运](../language/03-memory.md)。

## 逐元素算术

| 算子 | 可达 | 作用 |
| ---- | ---- | ---- |
| [`add`][pypto.language.add] [`sub`][pypto.language.sub] [`mul`][pypto.language.mul] [`div`][pypto.language.div] | `pl.` | 二元算术；右侧给 Python 数字会选中标量操作数形式 |
| [`neg`][pypto.language.neg] [`abs`][pypto.language.abs] [`recip`][pypto.language.recip] | `pl.` | 取负、绝对值、倒数；FP16/FP32 倒数设置 `high_precision=True` 时会在 A5 上选择速度较慢、精度较高的 PTO 路径 |
| [`rem`][pypto.language.tile.rem] [`rems`][pypto.language.tile.rems] [`fmod`][pypto.language.fmod] [`fmods`][pypto.language.fmods] | `pl.` | 求余与浮点取模，张量与标量形式 |
| [`addc`][pypto.language.tile.addc] [`subc`][pypto.language.tile.subc] [`addsc`][pypto.language.tile.addsc] [`subsc`][pypto.language.tile.subsc] | `pl.` (t) | 带进位操作数的三输入加 / 减 |
| [`part_add`][pypto.language.part_add] [`part_mul`][pypto.language.part_mul] [`part_max`][pypto.language.part_max] [`part_min`][pypto.language.part_min] | `pl.` | 分段算术 |

## 数学

| 算子 | 可达 | 作用 |
| ---- | ---- | ---- |
| [`exp`][pypto.language.exp] [`log`][pypto.language.log] | `pl.` | 指数、自然对数 |
| [`sqrt`][pypto.language.sqrt] [`rsqrt`][pypto.language.rsqrt] | `pl.` | 平方根；倒数平方根。`high_precision=` 只作用于张量路径，传给 Tile 会**抛异常** —— tile 级的精度由是否传入 scratch tile 决定：`pl.tile.rsqrt(src, tmp)` |
| [`sin`][pypto.language.tensor.sin] [`cos`][pypto.language.tensor.cos] | `pl.` | 三角函数 |

## 比较与选择

| 算子 | 可达 | 作用 |
| ---- | ---- | ---- |
| [`cmp`][pypto.language.cmp] [`cmps`][pypto.language.tile.cmps] | `pl.` | 比较两个操作数 / 操作数与标量 |
| [`maximum`][pypto.language.maximum] [`minimum`][pypto.language.minimum] | `pl.` | 两个操作数的逐元素最大 / 最小 |
| [`maximums`][pypto.language.tile.maximums] [`minimums`][pypto.language.tile.minimums] | `pl.` (t) | 与标量的逐元素最大 / 最小 |
| [`max`][pypto.language.tile.max] [`min`][pypto.language.tile.min] | `pl.` (t) | 两个标量取最大 / 最小 —— **不是** tile 规约。规约 tile 请用 `row_max` / `col_max`（以及对应的 `min` 形式） |
| [`sel`][pypto.language.tile.sel] [`sels`][pypto.language.tile.sels] | `pl.` (t) | 按掩码选择，张量与标量形式 |

## 激活

| 算子 | 可达 | 作用 |
| ---- | ---- | ---- |
| [`relu`][pypto.language.tile.relu] | `pl.` (t) | 修正线性 |
| [`prelu`][pypto.language.tile.prelu] [`lrelu`][pypto.language.tile.lrelu] | `pl.` (t) | 参数化 / 泄漏修正线性 |

## 位运算

| 算子 | 可达 | 作用 |
| ---- | ---- | ---- |
| [`and_`][pypto.language.and_] [`or_`][pypto.language.or_] [`xor`][pypto.language.xor] [`not_`][pypto.language.not_] | `pl.` | 位逻辑 |
| [`ands`][pypto.language.ands] [`ors`][pypto.language.ors] [`xors`][pypto.language.xors] | `pl.` | 与标量的位逻辑 |
| [`shl`][pypto.language.shl] [`shr`][pypto.language.shr] | `pl.` | 左移 / 右移 |
| [`shls`][pypto.language.shls] [`shrs`][pypto.language.shrs] | `pl.` | 按标量位数移位 |

## 规约

行规约折叠最后一个轴；列规约折叠第一个轴。

| 算子 | 可达 | 作用 |
| ---- | ---- | ---- |
| [`row_sum`][pypto.language.row_sum] [`row_prod`][pypto.language.row_prod] [`row_max`][pypto.language.row_max] [`row_min`][pypto.language.row_min] | `pl.` | 沿行规约 |
| [`col_sum`][pypto.language.col_sum] [`col_prod`][pypto.language.col_prod] [`col_max`][pypto.language.col_max] [`col_min`][pypto.language.col_min] | `pl.` | 沿列规约 |
| [`row_argmax`][pypto.language.row_argmax] [`row_argmin`][pypto.language.row_argmin] | `pl.` | 行极值的下标 |
| [`col_argmax`][pypto.language.col_argmax] [`col_argmin`][pypto.language.col_argmin] | `pl.` | 列极值的下标 |

若干规约接受 `tmp_tile` 参数。传入它会改变规约策略（二叉树 vs 顺序），从而改变浮点结合顺序 —— 结果是在容差内有差异，而不是错误。对部分有效 tile 的规约取决于填充值，见 [内存 § 有效形状](../language/03-memory.md#有效形状与填充)。

## 广播与展开

| 算子 | 可达 | 作用 |
| ---- | ---- | ---- |
| [`row_expand`][pypto.language.row_expand] [`col_expand`][pypto.language.col_expand] | `pl.` | 把被规约的轴广播回全宽 |
| [`row_expand_add`][pypto.language.row_expand_add] [`row_expand_sub`][pypto.language.row_expand_sub] [`row_expand_mul`][pypto.language.row_expand_mul] | `pl.` | 广播与算术融合 |
| [`row_expand_div`][pypto.language.row_expand_div] [`row_expand_max`][pypto.language.row_expand_max] [`row_expand_min`][pypto.language.row_expand_min] | `pl.` | 广播与除法、取最大、取最小融合 |
| [`col_expand_add`][pypto.language.col_expand_add] [`col_expand_sub`][pypto.language.col_expand_sub] [`col_expand_mul`][pypto.language.col_expand_mul] | `pl.` | 按列的对应形式 |
| [`col_expand_div`][pypto.language.col_expand_div] [`col_expand_max`][pypto.language.col_expand_max] [`col_expand_min`][pypto.language.col_expand_min] | `pl.` | 按列的除法、取最大、取最小 |
| [`row_expand_expdif`][pypto.language.row_expand_expdif] [`col_expand_expdif`][pypto.language.col_expand_expdif] | `pl.` | 与 `exp(x - m)` 融合的广播 —— softmax 的核心 |
| [`expand_clone`][pypto.language.tensor.expand_clone] [`expands`][pypto.language.expands] | `pl.` | 把一个值广播到某个形状 |
| [`fillpad`][pypto.language.fillpad] [`fillpad_expand`][pypto.language.fillpad_expand] | `pl.` | 填充无效区；可在同一步里广播 |

## 形状与布局

| 算子 | 可达 | 作用 |
| ---- | ---- | ---- |
| [`reshape`][pypto.language.reshape] | `pl.` | 重新解释维度 |
| [`transpose`][pypto.language.transpose] | `pl.` | 转置 |
| [`slice`][pypto.language.slice] | `pl.` | 子区域；也可写作 `A[0:16, :]` |
| [`concat`][pypto.language.concat] | `pl.` | 沿轴拼接 |
| [`assemble`][pypto.language.tensor.assemble] | `pl.` | 把子区域写回；也可写作 `dst[i:i+16] = src` |
| [`reinterpret_view`][pypto.language.reinterpret_view] | `pl.` | 不搬数据的重新解释 |
| [`set_validshape`][pypto.language.set_validshape] | `pl.` | 声明 tile 的有效区域 |
| [`cast`][pypto.language.cast] | `pl.` | 转换 dtype —— 可能展开成多跳链，见 [LegalizeTileCast](../../dev/passes/14-legalize_tile_cast.md) |
| [`dim`][pypto.language.tensor.dim] | `pl.` | 张量的运行期维度 |
| [`read`][pypto.language.read] [`write`][pypto.language.write] | `pl.` | 元素访问 |

## 线性代数

| 算子 | 可达 | 作用 |
| ---- | ---- | ---- |
| [`matmul`][pypto.language.matmul] | `pl.` | 矩阵乘；`a_trans=` / `b_trans=` 用来替代 DN 注解转置操作数 |
| [`matmul_acc`][pypto.language.matmul_acc] | `pl.` | 乘累加进已有的 `Acc` tile |
| [`matmul_bias`][pypto.language.tile.matmul_bias] | `pl.` (t) | 带 bias 操作数的乘法 |
| [`batch_matmul`][pypto.language.batch_matmul] | `pl.` (t) | 批量矩阵乘，**只接受 tile 操作数**。张量请调 `pl.matmul` —— rank > 2 会在降级时派发到 `tile.batch_matmul` |
| [`gemv`][pypto.language.tile.gemv] [`gemv_acc`][pypto.language.tile.gemv_acc] [`gemv_bias`][pypto.language.tile.gemv_bias] | `pl.` (t) | 矩阵-向量形式 |
| [`matmul_mx`][pypto.language.tile.matmul_mx] [`matmul_mx_acc`][pypto.language.tile.matmul_mx_acc] [`matmul_mx_bias`][pypto.language.tile.matmul_mx_bias] | `pl.` (t) | A5 MX 块缩放矩阵乘 —— 进入算子的两块 data tile 必须为 FP8E4M3FN；支持的 FP4 输入形式仅为 FP4×FP8，且左侧 FP4 必须先显式 cast 为 FP8；不支持原生 FP4×FP4 |

## Gather、Scatter、排序

| 算子 | 可达 | 作用 |
| ---- | ---- | ---- |
| [`gather`][pypto.language.tensor.gather] [`gather_row`][pypto.language.tensor.gather_row] | `pl.` | 按索引 gather |
| [`paged_gather`][pypto.language.tensor.paged_gather] | `pl.` | 跨分页布局 gather |
| [`gatherb`][pypto.language.tile.gatherb] | `pl.` (t) | 按 UINT32 字节偏移 gather 32 字节块；一列偏移展开成 `32 / sizeof(output_dtype)` 个元素 |
| [`mgather`][pypto.language.tile.mgather] | `pl.` (t) | 按索引 tile 从 DDR 张量 gather 行，带 `coalesce=` 与 `gather_oob=` 策略 |
| [`scatter`][pypto.language.tensor.scatter] [`scatter_update`][pypto.language.tensor.scatter_update] | `pl.` | 按索引 scatter；原地更新 |
| [`mscatter`][pypto.language.tile.mscatter] | `pl.` (t) | 带掩码的 scatter |
| [`sort32`][pypto.language.tensor.sort32] [`mrgsort`][pypto.language.tensor.mrgsort] | `pl.` | 对 32 元素组排序；归并已排序段 |

## Block 身份

| 算子 | 可达 | 作用 |
| ---- | ---- | ---- |
| [`get_block_idx`][pypto.language.tensor.get_block_idx] | `pl.` | `pl.spmd` 下本 block 的索引 |
| [`get_block_num`][pypto.language.tensor.get_block_num] | `pl.` | block 总数 |
| [`get_subblock_idx`][pypto.language.tensor.get_subblock_idx] | `pl.` | `pl.split_aiv` 下的 AIV lane 索引 |

## 跨核传输

混合 kernel 的接口 —— AIC 与 AIV 在同一个 InCore 函数内协作。

| 算子 | 可达 | 作用 |
| ---- | ---- | ---- |
| [`tpush_to_aiv`][pypto.language.system.tpush_to_aiv] [`tpush_to_aic`][pypto.language.system.tpush_to_aic] | `pl.` | 把一块 tile 推给对端核 |
| [`tpop_from_aic`][pypto.language.system.tpop_from_aic] [`tpop_from_aiv`][pypto.language.system.tpop_from_aiv] | `pl.` | 弹出对端推来的 tile |
| [`tfree_to_aic`][pypto.language.system.tfree_to_aic] [`tfree_to_aiv`][pypto.language.system.tfree_to_aiv] | `pl.` | 把弹出的 slot 释放回生产者 |
| [`aic_initialize_pipe`][pypto.language.system.aic_initialize_pipe] [`aiv_initialize_pipe`][pypto.language.system.aiv_initialize_pipe] | `pl.` | 建立跨核管道 |
| [`aiv_shard`][pypto.language.tile.aiv_shard] [`aic_gather`][pypto.language.tile.aic_gather] | `pl.` | 在 AIV lane 间分片 / 在 AIC 上聚回 |
| `AUTO` | `pl.` | 由编译器选择管道参数的哨兵值 |

push 与 pop 必须**配对**，且每次 pop 都必须有对应的 `tfree`。用法见 [混合 kernel 教程](../tutorials/03-mixed-kernel.md)；机制见 [TPUSH/TPOP](../../reference/pto-isa/01-tpush_tpop.md) 与 [ExpandMixedKernel](../../dev/passes/21-expand_mixed_kernel.md)。

## 任务与依赖

| 算子 | 可达 | 作用 |
| ---- | ---- | ---- |
| [`submit`][pypto.language.submit] [`spmd_submit`][pypto.language.spmd_submit] | `pl.` | 派发 kernel 并捕获其生产者 TaskId |
| `deps=` | `pl.at`、内联捕获形式 `pl.spmd` | 添加严格 TaskId 依赖；deferred waiter 使用同一条依赖路径 |
| [`no_dep`][pypto.language.tensor.no_dep] | `pl.` | 让单个任务的单个实参退出依赖跟踪 |
| [`dump_tag`][pypto.language.tensor.dump_tag] | `pl.` | 标记张量做选择性 dump |

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
| Deferred Wait | `pld.system.defer_wait` | 仅 `Ge` | — | — | INT32 signal | 注册单调 counter 条件而不让 AIV 自旋；Simpler 保持普通 waiter TaskId 未完成，后续工作使用普通 `deps=[wait_tid]`。 |
| Remote Load | `pld.tile.remote_load` | — | — | — | 任意（tile） | Tile 级跨 rank 加载。 |
| Remote Store | `pld.tile.remote_store` | — | — | — | 任意（tile） | Tile 级跨 rank 写入。 |

## 配套示例

每类算子一个可运行文件，供表格条目不够用时查阅：

| 类别 | 示例 |
| ---- | ---- |
| 逐元素算术 | `examples/beginner/02_elementwise.py` |
| 标量操作数 | `examples/beginner/03_scalar_ops.py` |
| 激活函数 | `examples/beginner/04_activation.py` |
| Matmul | `examples/beginner/05_matmul.py` |
| 拼接 / assemble | `examples/beginner/06_concat.py`、`examples/intermediate/05_assemble.py` |
| 规约 | `examples/intermediate/02_softmax.py`、`examples/intermediate/03_normalization.py` |
| 跨核搬运 | `examples/advanced/03_mixed_kernel.py` |
| 任务与依赖 | `examples/intermediate/07_task_graph.py` |

## See Also

- [选择命名空间](00-dispatch.md) —— 这些拼法该用哪一个。
- [IR 算子](../../dev/ir/05-operators.md) —— 这些名字背后的注册表。
- [PTOAS 算子状态](../../dev/ptoas-op-status.md) —— 各后端的支持情况。
