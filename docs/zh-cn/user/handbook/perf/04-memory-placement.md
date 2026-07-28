# 性能调优：内存放置

> **状态：** 骨架 —— 仅提纲；正文/示例待定。

## Tile 级内存层次与各层搬运引擎

- 回顾层次：DDR → Vec / Mat(L1) → Left/Right(L0A/B) → Acc(L0C) / Bias
  （链接[语言指南 › 内存](../../02-language_guide.md#内存与数据搬运)）。
- 每种搬运由哪个硬件引擎执行（load / move / store 路径，如 MTE 加载、MTE 搬运、
  FIXPIPE 输出）。
- **为什么重要：** 每次搬运在核内泳道图里都是一条泳道 —— 知道"引擎 → 数据搬运"的
  对应关系，才能看懂逐 kernel 的泳道图。

## 编译器 L1(Mat) 复用 —— 及其约束

- 自动且**由编译器负责**（循环不变的 matmul 操作数 Mat 驻留，PR #2080）—— 非用户 API。
- 仅作用于编译器生成的 tensor 级 `pl.matmul` 操作数加载
  （`GM → Mat → Left/Right → matmul`）；用户手写的
  `tile.load(..., target_memory=Mat)` **不会**被提升。
- 触发复用需同时满足的约束：
  - 顺序、静态有界、非空循环；候选无条件执行
  - offset / shape / 依赖均循环不变
  - 无顺序边界（调用、跨核、同步、cache 维护、store、中间控制流）
  - 编译器拥有的存储根；无可写根别名
  - 延长后的生命周期不超过后端 Mat/L0 容量
- 任一约束不满足 → 回退为逐迭代 `GM → Mat` 加载（不复用）。

## L0 分块约束（`AutoTileMatmulL0`）

- 在**单个 matmul 内**选择 L0 分块 + ping/pong 缓冲。
- 自身无法跨外层循环保留操作数（这正是 L1 驻留单独成一个 pass 的原因）。
- 容量 / 分块限制 —— 待补。

## 放置提示：512B 对齐

- L2 读取最小粒度为 **512B**。
- 尽量让连续数据 **512B 对齐**，以获得最佳 L2 读取效率。

## 唯一的用户旋钮：`target_memory`（可选）

- `pl.load(..., target_memory=pl.Mem.*)` / `pl.move(..., target_memory=...)`
- `pl.Mem`：DDR · Vec · Mat · Left · Right · Acc · Bias
- 仅在自动放置不理想时用于覆盖 / 引导推断。

## 验证效果

- in-core profiling / 泳道图 → [测量收益](05-measuring-impact.md)
