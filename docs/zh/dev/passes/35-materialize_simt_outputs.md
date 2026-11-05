# MaterializeSimtOutputs

在 SIMT 部分写入时保留 Tile 值语义 (value semantics)。每个可写 Tile 实参在
launch 前通过 `tile.move` 复制到独立存储。launch 结果使用副本的类型，并在
`InitMemRef` 后共享副本的分配。未写入元素因此保留输入值，旧 Tile 句柄仍可读取。
Tensor 实参保留引用语义 (reference semantics)：GM 更新直接使用原存储，无需
通过 Tile 中转或复制。

| C++ | Python | 层级 |
| --- | ------ | ---- |
| `pass::MaterializeSimtOutputs()` | `passes.materialize_simt_outputs()` | 函数 |

**前置条件：** SSAForm、TileMemoryInferred。
**产生：** SimtOutputsMaterialized。
**失效：** HasMemRefs、AllocatedMemoryAddr。

## 示例

```python
# Before
updated = pl.simt.launch(self.update, old, block_dims=(64, 1, 1), result_param_indices=(0,))

# After (old is a Vec Tile)
seed = pl.tile.move(old, target_memory=pl.Mem.Vec, attrs={"simt_output_seed": True})
updated = pl.simt.launch(self.update, seed, block_dims=(64, 1, 1), result_param_indices=(0,))
```

`simt_output_seed` 是编译器元数据。每个标记副本仅供一个 launch 使用；重复运行
此 pass 不会再次复制。属性验证器检查每个可写 Tile 都使用独立的已初始化副本。

当至少一个实参可写时，此 pass 拒绝已知重叠的实参。它跟踪别名和视图，并比较
已知 MemRef 区间。对于不同的运行时 GM 实参，调用者仍须保证不重叠；此分析不能
证明任意运行时地址无数据竞争。区间排序使 K 个实参的重叠检查保持 O(K log K)。

## 流水线集成

默认流水线在 `MaterializeTensorStrides` 后、`SpecializeSimtHelpers` 和
`InitMemRef` 前运行此 pass。辅助函数借用的参数不单独分配存储。内存规划将每个
launch 结果视为可写副本的定义；循环输入隔离保证 yield 修正把更新写回循环携带
存储时，旧 Tile 值仍被保留。PyPTO、DSA-RP 和 PTOAS 内存规划路径均使用此协议。

SIMT 核函数目前保留在 Functional IR；自动 Buffer IR 选择会对此回退。
显式处于 Buffer 阶段的 SIMT 函数会被拒绝。其余执行限制见
[标量 SIMT IR](../ir/11-simt.md)。
