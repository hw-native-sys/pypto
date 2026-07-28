# 性能调优：切分与并行

> **状态：** 骨架 —— 仅层次 + 旋钮清单；正文/示例待定。
> 单芯片工作分布。（多**卡** → [分布式指南](../../distributed/00-guide.md)。）

## Mixed kernel（cube + vector 切分）

> 仅 tensor 级接口。跨核 push/pop/free 原语由 pass（`ExpandMixedKernel`）插入 ——
> 非用户可见。

需要涵盖的旋钮：

- InCore 作用域上的 `pl.split(mode)`（`pl.at(..., optimizations=[pl.split(...)])`）
- `pl.SplitMode` —— `UP_DOWN` / `LEFT_RIGHT` / `NONE`
- `pl.split_aiv(2, mode=...)`

## 片上多 block 并行（`pl.spmd` 家族）

> 单芯片、多 block（SPMD grid dispatch）—— 非多卡分布式。

需要涵盖的旋钮：

- `pl.spmd` —— `with` / `for i in` / `as tid` 形式
- `pl.cluster`
- `pl.at`
- `pl.spmd_submit`
- `pl.tile.get_block_idx()` / `get_block_num()`
- `pl.system.syncall(core_type=..., mode="hard"|"soft")` —— 跨 SPMD block 的全核
  barrier（`hard` = 满占用 FFTS；`soft` = GM 轮询，支持部分占用）

（`deps=` / `allow_early_resolve=` / `predicate=` → [依赖与分发](03-dependency-dispatch.md)。）

## 验证效果

- Mixed kernel（cube/vector 重叠）→ **in-core profiling** —— mixed kernel 是一个
  task，AIC/AIV 重叠在内部，*不*出现在泳道图上。
- SPMD 多 block（跨核均衡）→ **泳道图** → [DFX › 泳道图](../dfx/01-swimlane.md)
