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

每个 handle 被放到**支配其 buffer 全部使用、且位于其 `TileView` 操作数依赖之后
的最小作用域**，通过递归遍历语句树确定：

- 跨多条语句使用（或跨 `if`/`else` 分支、或作为循环 carry —— iter-arg / return-var）
  的 buffer 放在当前层、其首个使用之前。因此在两分支都写入的 phi buffer 会声明在
  外层 `IfStmt` 之前，从而支配两分支。
- **完全只在某个嵌套循环/分支体内使用**的 buffer 会下沉进那个作用域，因此一个
  `valid_shape` 引用循环体内标量（例如 `valid = i + 1; t = load(..., valid_shapes=[.., valid])`）
  的 handle 会落在该标量*之后*，而不会被上提到循环之上（那样操作数就越界了）。

每条语句在每个外层被重扫一次，故整体为 O(N × depth)（嵌套深度有界）。

### valid 协调

一个 handle 服务其 buffer 的每个成员，但成员之间可能对 valid 范围有分歧。放置完成
后有一趟遍历语句树、跟踪当前作用域内变量的修复，依据每个成员的*有效 valid*（有
`valid_shape` 则取之，否则取物理形状）协调两种情况：

**1. 动态 valid 被上提出作用域。** 感知依赖的放置能把 handle 保持在*单个*循环体
`valid_shape` 操作数之下，但当内存复用让一个 buffer 被**多个兄弟作用域**共享时
（例如同一个 `mem_vec` 槽被两个相邻循环各自加载，各带自己的循环内 `valid_len`），
就不存在任何一个作用域既支配全部使用又能看见全部操作数 —— 它会被上提到公共祖先，
位于它们全部之上。此时在那里发出 handle 的动态 `valid_col` 会引用越界操作数，ptoas
会报错（`'pto.alloc_tile' op valid_col operand is required because result type
v_col is ?`）。对每个 `valid_shape` 操作数在 handle 位置**不在作用域**内的
`alloc_tile`，让 handle 以**静态** valid（物理形状 —— 自包含、可任意上提的操作数）
声明。

**2. valid 与 handle 声明不同的成员。** 只要一个成员的有效 valid 与 handle 声明的
valid 不同 —— 上面的静态-alloc 情况，*或*一个被复用 buffer 上两个成员携带不同 valid
（例如 SPMD 次子块 replay 的 `valid=[0,0]` 与主块全范围共享同一 MemRef）—— 就在该
生产者*之前*、对 handle 注入一条 `tile.set_validshape`（该处操作数在作用域内）重新
建立成员的真实 valid。必须在生产者**之前**(而非之后)：`tile.load` 的填充/补齐范围
跟随目标 tile 的 valid，所以 load 写入时 buffer 必须已经是正确的 `valid`；放到之后
会让 load 期间 buffer 仍是 handle 声明的 valid，破坏部分-valid 块。

**纯 view 成员被排除。** 由 view op（`tile.slice` / `tile.reshape` /
`tile.transpose` / `tile.assemble` …，以 `set_output_memory_inherit_input` 注册）
产生的成员别名一个已有 tile 的内存，并被 emit 成携带**自己** `valid [...]` 子句的
`pto.subview` —— 它从不从 handle 的 `tile_buf` 类型读取 valid。协调它会 emit 一条多
余的 `set_validshape` 并破坏源的 valid，故 view 成员被跳过。原地写入者
（`set_output_reuses_input`，例如 `fillpad_inplace`）确实在裸 handle 处写入 buffer，
仍参与协调。

```mlir
%h = pto.alloc_tile addr = %c0 valid_row = %c16 valid_col = %c64 : ...   # 静态
scf.for ... {                                     # pass 1
  %valid_len = scf.if ...
  pto.set_validshape %h, %c16, %valid_len : ...   # 注入 —— 真实 valid,在 load 之前
  pto.tload ... outs(%h)
  pto.tfillpad ins(%h) ...                        # 用 min 补 valid_len..64
}
scf.for ... { pto.set_validshape %h, %c16, %valid_len_p2 ... pto.tload ... }   # pass 2
```

这是语义等价的（注入的 `set_validshape` 恢复了下沉的 alloc 本会携带的 valid），并
保持严格的「每 buffer 一个 handle」模型。由于本 pass 是最后一趟，后续没有 DCE 会
删除结果未使用的 `set_validshape` 副作用 op。

## 示例

```python
# Before — phi buffer 的 handle 会被 codegen 合成在分支内
if flag == 0:
    result = partial        # tile
else:
    result = updated        # tile
final = pl.store(result, [0, 0], out)
```

IR op 是 `alloc_tile(base, byte_offset, shape)`；PTO codegen 把它 emit 成
`pto.alloc_tile`，并将 tile 的 `TileView` valid 范围下降为下面的 `valid_row` /
`valid_col` 操作数（类型里是 `v_row=?, v_col=?`）：

```mlir
; After MaterializeAllocTiles (PTOAS): one handle before the if (it dominates
; both branches), which the branches and the store share — no in-branch decl.
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
