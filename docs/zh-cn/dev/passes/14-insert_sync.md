# InsertSync Pass

分析数据依赖并插入同步操作，确保多流水线执行的正确性。

## 概述

该 Pass 是 PyPTO 中最复杂的变换 Pass。它分析跨硬件流水线的数据依赖，并插入同步操作（sync_src、sync_dst、bar_v、bar_m）以确保执行正确性。

**核心职责**：

- 分析跨流水线数据依赖
- 插入 sync_src/sync_dst 实现生产者-消费者同步
- 插入屏障（bar_v、bar_m）实现全局同步
- 管理事件 ID 和流水线掩码

**使用时机**：在 InitMemRef 和 BasicMemoryReuse 之后、代码生成 (CodeGen) 之前运行。多流水线硬件正确执行所必需。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::InsertSync()` | `passes.insert_sync()` | 函数级 |

**工厂函数**：

```cpp
Pass InsertSync();
```

**Python 用法**：

```python
from pypto.pypto_core import passes

sync_pass = passes.insert_sync()
program_with_sync = sync_pass(program)
```

## 算法

1. **阶段一 -- 依赖收集**：遍历中间表示 (IR) 树，确定每个操作的流水线分配（使用后端流水线信息），并收集生产者-消费者同步对（包括跨流水线和同流水线）。对于循环，额外展开一次迭代以检测跨迭代依赖。
2. **阶段二 -- 作用域调整**：调整跨越作用域边界（IfStmt/ForStmt）的同步对：
   - 跨迭代（wait <= set 在同一 for 循环体内，包括同流水线屏障）：将 sync_dst/bar 移至迭代末尾
   - 当 wait 在比 set 更深的作用域中时：将 sync_dst 移至 set 所在 OpStmts 的末尾
   - 当 set 在比 wait 更深的作用域中时：将 sync_src 移至 wait 所在 OpStmts 的开头
3. **阶段三 -- 事件 ID 分配**：为同步操作分配唯一的事件 ID，尽可能复用 ID
4. **阶段四 -- AST 构建**：构建包含 sync_src/sync_dst/barrier 插入的最终 IR

**同步模式**：

- **生产者-消费者**：sync_src（生产者） -> sync_dst（消费者）
- **流水线屏障**：bar_v / bar_m
- **If 分支作用域**：当生产者在 if 分支中而消费者在外部时，sync_src 被移至消费者所在 OpStmts 的开头
- **For 循环体到父级**：当生产者在 for 循环体中而消费者在外部时，sync_src 被移至消费者所在 OpStmts 的开头
- **跨迭代**：sync_dst/bar 放置在迭代末尾（yield 之前）
- **OpStmts 合并**：同步操作合并到相邻的 OpStmts 中（不产生独立的同步 OpStmts）

## 示例

### 跨流水线依赖（MTE2 -> V -> MTE3）

加载（MTE2） -> 计算（V） -> 存储（MTE3），在每个流水线边界处插入 sync_src/sync_dst。

**之前**：

```text
tile_a = load(input_a)              # MTE2
tile_b = load(input_b)              # MTE2
tile_c = add(tile_a, tile_b)        # V
store(tile_c, output)               # MTE3
```

**之后**：

```text
tile_a = load(input_a)              # MTE2
tile_b = load(input_b)              # MTE2
sync_src(MTE2 -> V, event=0)
sync_dst(MTE2 -> V, event=0)
tile_c = add(tile_a, tile_b)        # V
sync_src(V -> MTE3, event=0)
sync_dst(V -> MTE3, event=0)
store(tile_c, output)               # MTE3
```

### 同流水线依赖（V -> V）

当连续的 V 流水线操作存在数据依赖时，插入 `bar_v` 屏障而非 sync_src/sync_dst。

**之前**：

```text
t_c = add(t_a, t_b)                # V
t_d = add(t_c, t_a)                # V (depends on t_c)
```

**之后**：

```text
t_c = add(t_a, t_b)                # V
bar_v                               # intra-pipe barrier
t_d = add(t_c, t_a)                # V
```

### CUBE 流水线（MTE2 -> MTE1 -> M -> MTE3）

矩阵乘法需要通过 L1（MTE1）将数据移动到 L0（CUBE/M 流水线），并在每个边界处同步。当同一流水线对有多个独立传输时，使用多个事件 ID。

**之前**：

```text
tile_a = load(input_a)              # MTE2 -> L1
tile_b = load(input_b)              # MTE2 -> L1
tile_a_cube = move(tile_a)          # MTE1 -> Left
tile_b_cube = move(tile_b)          # MTE1 -> Right
tile_c = matmul(tile_a_cube, tile_b_cube)  # CUBE (M pipe)
store(tile_c, output)               # MTE3
```

**之后**：

```text
tile_a = load(input_a)              # MTE2
sync_src(MTE2 -> MTE1, event=0)
tile_b = load(input_b)              # MTE2
sync_src(MTE2 -> MTE1, event=1)
sync_dst(MTE2 -> MTE1, event=0)
tile_a_cube = move(tile_a)          # MTE1
sync_dst(MTE2 -> MTE1, event=1)
tile_b_cube = move(tile_b)          # MTE1
sync_src(MTE1 -> M, event=0)
sync_dst(MTE1 -> M, event=0)
tile_c = matmul(tile_a_cube, tile_b_cube)  # M
sync_src(M -> MTE3, event=0)
sync_dst(M -> MTE3, event=0)
store(tile_c, output)               # MTE3
```

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`

```cpp
Pass InsertSync();
```

**实现文件**：`src/ir/transforms/insert_sync_pass.cpp`

- 使用后端流水线信息（通过全局配置的后端）
- 四阶段流水线：收集 -> 作用域调整 -> 事件ID分配 -> AST构建
- 作用域感知：同步对不会跨越 IfStmt/ForStmt 边界
- 通过循环展开检测跨迭代依赖

**后端集成**：

```cpp
#include "pypto/backend/common/backend_config.h"
// Uses Backend::GetOpInfo() to determine operation pipelines
```

**Python 绑定**：`python/bindings/modules/passes.cpp`

```cpp
passes.def("insert_sync", &pass::InsertSync, "Insert synchronization operations");
```

**测试**：`tests/ut/ir/transforms/test_insert_sync.py`

- 测试同作用域跨流水线依赖（MTE2->V->MTE3）
- 测试同流水线屏障插入（V->V）
- 测试 CUBE 流水线（MTE2->MTE1->M->MTE3）
- 测试 IfStmt 作用域跨越（双分支、单分支、分支合并）
- 测试 ForStmt 作用域跨越（for 之前加载，内部计算）
- 测试跨迭代依赖（V->MTE2、MTE3->MTE2）
- 测试 for+if 组合模式

## 后端依赖

该 Pass 需要配置后端以获取流水线信息：

```python
from pypto import backend

# Set backend before running InsertSync
backend.set_backend(backend.Ascend910B())
program_with_sync = passes.insert_sync()(program)
```
