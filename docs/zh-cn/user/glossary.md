# 术语表

> **状态：** 草稿骨架。手册各处引用的共享定义。

_TODO —— 每个术语一句精炼定义；各章首次提及处链接回此处。_

| 术语 | 定义 |
| ---- | ---- |
| **Tile（片）** | _TODO —— 在 tile 级操作的定长块。_ |
| **Tensor 级 / Tile 级** | _TODO —— 两级抽象。_ |
| **Scope（作用域）** | _TODO —— hierarchy / InCore / cluster / manual 作用域。_ |
| **Orchestration（编排）** | _TODO —— 多核调度/分发层。_ |
| **InCore（核内）** | _TODO —— 核内计算函数。_ |
| **AIC / AIV** | _TODO —— cube 核 / vector 核。_ |
| **SPMD** | _TODO —— single-program multiple-data，片上多 block。_ |
| **Cluster（簇）** | _TODO —— 芯片上分组的 SPMD dispatch。_ |
| **TaskId** | _TODO —— 依赖句柄（`Scalar[TASK_ID]`）。_ |
| **Early resolve（提前编排）** | _TODO —— 推测式预先编排任务的消费者。_ |
| **Dispatch predicate（分发谓词）** | _TODO —— 跳过任务分发的 gate。_ |
| **DistributedTensor** | _TODO —— 跨多卡的张量。_ |
| **CommCtx** | _TODO —— collective 的通信上下文。_ |
| **DFX** | _TODO —— design-for-X 诊断（dump、依赖图、日志）。_ |
