# ptoas 多缓冲（`use_ptoas_multi_buffer`）

Updated: 2026-07-08

一个可选开关，把同核 `pl.pipeline(stage=N)` 循环里轮转的 load 降级为 **ptoas 多
缓冲 region**,而非 pypto 自己的循环体复制 ping-pong。跨迭代的双缓冲重叠交给
ptoas 完成,于是 kernel 保持**单体循环 + 单块 N-slot region**(代码更小、内存更
紧凑),同时达到与原生 pipeline 相同的重叠。

由 `PassContext.use_ptoas_multi_buffer` 门控 —— 默认关闭时是 no-op,默认流水字节
不变。

## 如何开启

```python
from pypto.pypto_core import passes

# 显式 PassContext
with passes.PassContext([], use_ptoas_multi_buffer=True):
    ...

# RunConfig / @pl.jit
cfg = RunConfig(platform="a2a3", use_ptoas_multi_buffer=True)

# 环境变量回退
#   PYPTO_PTOAS_MULTI_BUFFER=1
```

需要 ptoas ≥ 0.48(支持 `pto.alloc_multi_tile`):`PTOAS_ROOT=/usr/local/ptoas/0.48`。

## 关键设计:同 slot,而非显式预取

重叠由 ptoas 自己完成。只要一个 tile 在**同一 slot `mb[i%N]`、同一迭代**内被 load
并消费,ptoas PlanMemory 就给 N 个 slot 分配**互不相交的具体地址**,其同步 pass 用
dyn-event(`set_flag_dyn` / `wait_flag_dyn` + `arith.select`)的 WAR 同步,把
**迭代 `i` 的 load(slot `i%N`)与迭代 `i-1` 的消费(slot `(i-1)%N ≠ i%N`)重叠**。

手写「prologue + 预取 next / 消费 cur」的拆分(写 slot `(i+1)%N` 同时读 slot
`i%N`)反而**破坏**这个分析:ptoas 按程序序配静态 event,消费会等到本迭代对**另一
个 slot** 的预取(假依赖),循环退化为串行。因此本 pass 只发同 slot 形态,让 ptoas
去做流水。

## 关键约束:必须配合 ptoas 内存规划(level2)

重叠只在 **`--pto-level=level2`** 下成立 —— 此时 ptoas PlanMemory 拥有内存,给每个
slot 分不相交的具体地址(其 `MemAlias` 判定各 slot disjoint → dyn-event 同步)。在
`--pto-level=level3` 下单一烘死基址 + **动态** slot `i%N` 会击败 `MemAlias`(它保守
地把动态 slot 当作别名所有 slot)→ 串行。

因此 `use_ptoas_multi_buffer=True` 会**自动强制 `memory_planner=PtoAS`**(在
`PassContext` 构造函数里;`compile.py` 把有效 planner 读回,保证 codegen 的 level
一致)。若调用方传了别的 planner,会 warning 说明被覆盖。level3 下本 pass 仍发射有效
(但不重叠)的 IR 作为兜底。

## Pass 本身

`ConvertToPtoasMultiBuffer`(`src/ir/transforms/convert_to_ptoas_multi_buffer_pass.cpp`)
运行在 `LowerPipelineLoops` 的槽位。开关打开时 pass_manager 还会**摘掉
`LowerPipelineLoops` 和 `CanonicalizeIOOrder`**(它们会复制/重排),于是本 pass
**全权接管 pipeline 下降**,必须不留任何 `ForKind::Pipeline` 循环。对每个同核
pipeline 循环,它要么:

- **改写**(恰好一条 i-依赖的 Vec/Mat load):把
  `region = tile.multi_buffer_alloc(shape; count=N)` 提到循环前,并把
  `t = tile.load(args)` **就地**换成
  `t = tile.multi_buffer_load_slot(region, i%N, args)`(同一 tile 变量 → 消费者
  无需改动),然后把循环降为 `Sequential`;要么
- 对不满足条件的循环**降为**普通 `Sequential`(正确、无双缓冲)。

一条 load 满足条件当且仅当它是单-def 的 Vec/Mat `tile.load`,且其 offset 参数引用
了循环变量(i-*无关* 的 load 是循环不变量,多缓冲无意义)。

## 新增 IR op(pass 内部合成,不暴露给 DSL)

注册在 `tile.` 命名空间下(`src/ir/op/tile_ops/memory.cpp`),复用既有的
printer/parser round-trip;**不**标 `internal_only`(否则无法反解析):

| Op | 结果 | Codegen |
| -- | ---- | ------- |
| `tile.multi_buffer_alloc(shape; count=N, dtype, target_memory)` | region(每-slot `TileType`) | `pto.alloc_multi_tile`(仅 level3 带 addr) |
| `tile.multi_buffer_load_slot(region, k, tensor, offsets, shapes, valid_shapes)` | 已填充的 slot 视图 | `pto.multi_tile_get %mb[k]` + `pto.tload` |
| `tile.multi_buffer_get_slot(region, k)` | 消费视图 | `pto.multi_tile_get %mb[k]` |

`multi_buffer_alloc` 复用 `DeduceTileCreateTileType`(`count` 是额外的 int kwarg)。
动态 slot `k = i%N` 是普通 index SSA 操作数,只在 codegen 落地为 `%mb[k]`;它从不
进入 MemRef —— pypto 的 MemRef offset 是静态的,这正是轮转必须留作运行期索引、并交
给 ptoas 规划的原因。`multi_buffer_get_slot` 已注册但当前 pass 未用(留作将来多消费
视图)。

## 内存层

因为开关运行在 ptoas 内存规划(level2)下,`MemoryReuse` 与 `AllocateMemoryAddr`
被跳过 —— **整块 N-slot region 由 ptoas PlanMemory 拥有**,尺寸由
`multi_tile_buf<..., count=N>` 类型给出。pypto 侧不做预留;codegen 发**无 addr** 的
`pto.alloc_multi_tile`。

- **Region**(`multi_buffer_alloc`)从 `InitMemRef` 拿到 MemRef,但不烘死地址。
- **Slot 视图**(`get_slot` / `load_slot`)是 **buffer-less**(`InitMemRef` 的
  `ProducesBufferLessTile`),这样每次使用拿到自己的 SSA 名(不会与 region 纯别名
  塌缩);codegen 直接发它们的 `pto.multi_tile_get`。

## 验证

- **Codegen**:发射的 `.pto` 有一条提前的 `pto.alloc_multi_tile`(level2 无 addr、
  `count=N`)和每迭代一条 `pto.multi_tile_get %mb[i%N]`(单体循环、无 `scf.if` 预取
  guard)。
- **重叠(ptoas 0.48,level2)**:最终 `.cpp` 预置 2 个每-slot event,并使用**变量
  event id**(`wait_flag(..., v25)` / `set_flag(..., v26)` —— 即 lowered 的
  `wait_flag_dyn` / `set_flag_dyn`),于是 `load[i]` 与 `consume[i-1]` 重叠。
- **上板数值 parity**(`tests/st/runtime/test_ptoas_multi_buffer_device.py`):
  开关开 == 开关关 == torch 参考,a2a3 实测通过。
- **Codegen / round-trip 单测**:`tests/st/codegen/dsl/test_ptoas_multi_buffer_codegen.py`。

## 限制(M1 范围)

- 每个 pipeline 循环恰好一条 i-依赖 Vec/Mat load;更深的多 load 循环体、以及超出同
  slot 泛化的 `N > 2` 是后续工作。
- 摘掉 `LowerPipelineLoops` 是全局的:开关下**所有**不满足条件的 pipeline 循环
  (包括 matmul L0 stage 循环)都降为串行(丢 ping-pong)。因为开关是 opt-in / 默认
  关,可接受;后续增量可在 pass 内对非目标循环做复制。
- 仅 level2:重叠需要 ptoas PlanMemory,故开关强制 PtoAS;codegen 断言 region 不会
  走到 level3。

## 文件地图

| 关注点 | 路径 |
| ------ | ---- |
| Pass | `src/ir/transforms/convert_to_ptoas_multi_buffer_pass.cpp` |
| Ops | `src/ir/op/tile_ops/memory.cpp`(注册)、`python/pypto/ir/op/tile_ops.py`(builder) |
| 自动强制 planner | `src/ir/transforms/pass_context.cpp`(构造函数)、`python/pypto/ir/compile.py` |
| pass_manager 摘 pass | `python/pypto/ir/pass_manager.py` |
| Codegen | `src/codegen/pto/pto_codegen.cpp`(region alloc + `EmitMultiTileGet`)、`src/backend/common/pto_ops_common.cpp`(op emitter) |
| 内存层 | `src/ir/transforms/init_memref.cpp`(buffer-less slot 视图) |
| ptoas 设计文档 | `~/PTOAS/docs/designs/ptoas-multi-buffer-explicit-design.md` |
