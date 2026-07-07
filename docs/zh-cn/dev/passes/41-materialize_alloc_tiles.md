# MaterializeAllocTiles Pass

把 PTO tile **handle** 从一个由 codegen 合成的产物提升为显式的 `alloc_tile`
IR op，使 PTO codegen 退化为严格的 1:1 emitter。对函数使用的每个不同 tile
buffer，本 pass 在一个支配（dominate）该 buffer 全部使用的作用域中，恰好插入
一个 `alloc_tile(base, byte_offset, shape)` op（issue #1956）。

## 概述

在本 pass 之前，tile handle（`pto.alloc_tile`）并不是一个 IR 节点 —— PTO
codegen 在每个 tile 变量的定义点合成它。当该定义点位于 `if`/`else` 分支内部
（一个 if/else-yield **phi** 生产者）时，handle *在一个分支内声明、却从另一个
分支读取* —— 这是一种未声明 SSA 的作用域违规。在 `memory_planner=PYPTO` 下，
共享的 `addr` 掩盖了这个问题；而在 `memory_planner=PTOAS` 下（此时 `MemoryReuse`
被跳过）则会错误编译：phi 读取的 buffer 没有任何分支写入过。

把 handle 变成一等 op，从根本上修复了这一点：

- handle **只放置一次**，位于一个支配所有使用的作用域 —— 因此一个跨分支写入的
  buffer 在外层 `if` 之前就被声明，绝不会声明在 `if` 内部。
- codegen 退化为直白的转写：它从这些 op 1:1 emit `pto.alloc_tile`，并把每个
  tile 变量解析到其 buffer 的 handle（见
  [PTO codegen](../codegen/00-pto_codegen.md#allocation-generation)）。任何分支都不再是
  handle 的声明点。

**何时使用**：作为 `Default` 与 `DebugTileOptimization` 策略中的最后一个 pass —
在 MemRef 和地址最终确定之后（`InitMemRef`、`MaterializeSemanticAliases`，以及
PYPTO 下的 `MemoryReuse` + `AllocateMemoryAddr`），并在最终的 `Simplify` 之后，
这样就不会有 DCE 删除那些刻意保持未使用的 handle 变量，也不会有更早的 transform
需要去理解 `alloc_tile` op。**两种** memory planner 下都运行。

**作用范围**：仅处理持有 `TileType` 变量的函数。Orchestration 函数（负责提交
task、从不持有 tile）原样返回。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::MaterializeAllocTiles()` | `passes.materialize_alloc_tiles()` | 函数级 |

```python
from pypto.pypto_core import passes

materialized = passes.materialize_alloc_tiles()(program)
```

## 行为

### 哪些 buffer 会得到 handle

`BufferCollector` 按程序中首次出现的顺序，为每个不同的 tile buffer 记录一个
代表，覆盖 codegen 必须 emit handle 的每种构造：`AssignStmt` 定义的 tile、
`ForStmt` / `WhileStmt` 的 iter-arg 与 return-var，以及 `IfStmt` 的 return-var。
函数参数（绑定到 `%argN`）和无 MemRef 的 tile（活在保留 C2V 槽位中的跨核 `tpop`
结果，以及对它的 view）不会得到 handle。

### handle 粒度取决于 memory planner

分组键（`BufferKey`，与 codegen 的 `BufferHandleKey` 保持一致）根据
`PassContext::GetMemoryPlanner()` 选择：

| Planner | 分组键 | 依据 |
| ------- | ------ | ---- |
| `PYPTO`（默认） | MemRef 身份（base + byte_offset + size）**+ `TileBufSignature`** | handle 携带显式 `addr`，因此多个带类型的 handle 可以别名同一地址。按签名（内存空间、dtype、物理 shape、layout、fractal、**pad**）拆分，可将同一字节槽位上 pad / shape / layout 不同的 handle 保持为独立 handle —— 与 #1956 之前的每变量模型一致。 |
| `PTOAS` | 仅 MemRef 身份 | ptoas 为每个 handle 分配一个 buffer，因此一个字节槽位必须恰好映射到一个 handle（pad 差异被折叠 —— PTOAS 既有特性）。 |

`TileBufSignature` 键刻意**排除**了 valid-shape 范围：一次
`tile.set_validshape` 收窄产生的物理类型不变，范围搭载在 emit 的
`valid_row` / `valid_col` 操作数上 —— 因此这类 tile 共享同一 handle。真正的别名
（loop-carry、if/else-yield phi、原地 op 结果）共享同一签名，因此在两种 planner
下都共享同一 handle。

### 感知依赖的放置

每个 handle 被插入到第一条其子树使用该 buffer 的顶层语句之前。该位置 (a) 支配
所有使用 —— 一个跨分支写入的 buffer 会最先在外层 `IfStmt` 处出现，因此 handle
落在它之前 —— 并且 (b) 位于 handle 的 `TileView` 所引用的任何 body 内定义值
（例如运行时 valid 长度）之后，而盲目地上提到函数头部会跑到它们前面。该扫描是
O(N)：每个顶层子树只访问一次。

## 示例

```python
# Before — phi buffer 的 handle 会被 codegen 合成在分支内
if flag == 0:
    result = partial        # tile
else:
    result = updated        # tile
final = pl.store(result, [0, 0], out)
```

```mlir
; After MaterializeAllocTiles (PTOAS): one handle at the function head, both
; branches and the store share it — no in-branch declaration.
%res__buf1 = pto.alloc_tile valid_row = %c64_index valid_col = %c64_index : !pto.tile_buf<...>
scf.if %flag {
  pto.tmul ... outs(%res__buf1 : !pto.tile_buf<...>)
} else {
  pto.tadd ... outs(%res__buf1 : !pto.tile_buf<...>)
}
pto.tstore ins(%res__buf1 : !pto.tile_buf<...>) outs(...)
```

在 `PYPTO` 下，每个 `alloc_tile` 还会额外烘焙 `addr = <byte_offset>`。

## 验证

**测试**：`tests/ut/ir/operators/test_alloc_tile_op.py`（op 构造 + 类型推导）、
`tests/ut/codegen/test_memory_planner_switch.py`（两种 planner 下的单 handle
别名）、`tests/ut/codegen/test_pto_codegen.py`（fillpad / if-phi 的 handle 粒度），
以及 `tests/ut/ir/transforms/test_verify_alloc_tile_dominates.py`（下述产生的
属性）。

本 pass 产生 `AllocTileDominatesUses` 属性，由一个
[验证器](99-verifier.md)检查，它会标记任何其 buffer 缺少支配 `alloc_tile`
handle 的 tile 使用。

## Pass 属性

| 属性 | 取值 |
| ---- | ---- |
| Required | `SplitIncoreOrch`, `IncoreTileOps`, `HasMemRefs`, `NormalizedStmtStructure` |
| Produced | `NormalizedStmtStructure`, `AllocTileDominatesUses` |
| Invalidated | — |
