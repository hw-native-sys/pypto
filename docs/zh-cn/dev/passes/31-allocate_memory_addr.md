# AllocateMemoryAddr Pass

为已有的 alloc 操作分配实际内存地址。

## 概述

该 Pass 是非 DDR 内存引用 (MemRef) 的物理地址边界。它解析
`system.reserve_buffer(base=AUTO)`、选择放置位置，并在 PTO codegen 前更新已有的
`tile.alloc` 语句。它不会创建分配操作：InitMemRef 已经用未分配地址创建了这些操作。

默认的 `MemoryPlanner.PYPTO` 路径在 MemoryReuse 之后保留现有的对齐 bump 放置。
可选的 `MemoryPlanner.DSA` 路径接收尚未机会性合并的分配 identity，导出与 benchmark
框架相同的 structured problem，调用独立 solver，独立验证结果，再把验证过的 offset 写回。

**核心职责**：

- 从 TileType 变量中收集唯一的 MemRef 对象
- 在每个函数中把 `system.reserve_buffer` 的 base 解析成显式地址
- 在每个内存空间内分配顺序的、32 字节对齐的地址
- 或在 DSA 模式下，由独立 solver 联合选择生命周期复用与 offset
- 更新所有变量类型 (Type) 中的 MemRef 地址
- 使用分配的地址更新 `tile.alloc` 语句参数

**使用时机**：在代码生成前运行，作为内存管理的最后一个 Pass。默认流水线在
MemoryReuse 之后运行它。DSA 流水线会刻意跳过 MemoryReuse，但仍先运行
MaterializeSemanticAliases，因此 view、循环 carry 值和原地操作的强制 identity 不变。

## Planner 模式

| 模式 | 本 Pass 的输入 | 放置方式 | 失败行为 |
| ---- | -------------- | -------- | -------- |
| `MemoryPlanner.PYPTO` | MemoryReuse 机会性合并后的 MemRef | 后端策略控制的对齐 bump 分配 | 现有 verifier 报告非法地址或超容量 |
| `MemoryPlanner.DSA` | MaterializeSemanticAliases 后未机会性合并的 MemRef | 独立 schema-v1 DSA solver：first-fit 初始化、受限 structured search，并且仅在严格问题无法装入容量时显式放宽流水线意图 | 非法导出、能力不匹配、不可行或 validator 失败都会终止编译；不会静默回退 |
| `MemoryPlanner.PTOAS` | 无 | 跳过本 Pass；ptoas `PlanMemory` 负责放置 | 交给 ptoas |

DSA 支持是可选的 CMake 依赖。先构建并安装 `dsa-solver` 0.10 package，再让
PyPTO 使用它：

```bash
cmake -S /path/to/dsa-solver -B /path/to/dsa-solver/build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/path/to/dsa-install
cmake --build /path/to/dsa-solver/build --parallel 2
cmake --install /path/to/dsa-solver/build

cmake -B build -DPYPTO_ENABLE_DSA_SOLVER=ON \
  -DCMAKE_PREFIX_PATH=/path/to/dsa-install
cmake --build build --parallel 2
```

默认构建保持 `PYPTO_ENABLE_DSA_SOLVER=OFF`。它仍暴露 planner enum，以便配置保持
一致；若执行时选择 DSA，则会得到明确的重新配置依赖提示。
`passes.is_dsa_solver_available()` 可查询当前构建是否包含 adapter，供可选测试和应用在
选择 DSA 前显式判断。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::AllocateMemoryAddr()` | `passes.allocate_memory_addr()` | 函数级 |

**工厂函数**：

```cpp
Pass AllocateMemoryAddr();
```

**Python 用法**：

```python
from pypto.pypto_core import passes

alloc_pass = passes.allocate_memory_addr()
program_with_addrs = alloc_pass(program)
```

通过 PassContext 配置独立路径，并可选择输出确定性的 corpus 文档：

```python
from pypto.pypto_core import passes

with passes.PassContext(
    [],
    memory_planner=passes.MemoryPlanner.DSA,
    dsa_export_dir="build/dsa-corpus",
    dsa_reuse_penalty_recognizer=passes.DsaReusePenaltyRecognizer.QUADRATIC,
):
    program_with_addrs = passes.allocate_memory_addr()(program)
```

完整编译接受相同的选择：
`ir.compile(..., memory_planner=passes.MemoryPlanner.DSA,
dsa_export_dir="build/dsa-corpus")`。
`RunConfig` 也暴露 `memory_planner` 和 `dsa_export_dir` 字段。设置
`dsa_solution_dir` 可跳过求解并回放已记录的 placement；只有当 fingerprint
与当前重新导出的问题完全匹配且独立验证通过时才会接受。实验性的
`dsa_reuse_penalty_recognizer` 默认关闭。`QUADRATIC` 是当前唯一启用的研究模式，
并优先保证覆盖率：它先根据
已解析的 memory space 推导抽象 source/destination route，再对所有生命周期可复用的
allocation pair 比较每种执行资源上的终止访问 frontier 和初始写
frontier，并覆盖嵌套控制流和跨一次循环回边的交接。逻辑 SSA 可达性只作为证据保留，
不被当作异步访问已经完成的证明。partial view 和同一操作内部的交接会被记录，但不会
定价。实验性的 v3 promotion policy 会把符合条件的 cross-resource 记录聚合成单位
权重 pair edge；same-resource，以及访问范围、操作 alias、结构化控制流或顺序语义
不确定的记录只用于报告。
operation registry 中的 effect 会区分执行期访问、纯声明和仅修改元数据的 view；
会修改数据的 inherit-input 操作以及 tuple 输出仍会进入 access frontier。
权重标定属于独立建模问题。

对于受控 placement 实验，`dsa_reference_placement=COMPACT` 标记正常且已验证的
DSA 结果；`LOOSE` 在不超过容量的前提下贪心减少物理地址复用。
`dsa_reference_target="name"` 只对该精确函数应用 `LOOSE`，同一程序中的其他 kernel
保持 compact。每个端点都在产生它的编译过程中直接构造并独立验证，避免生成函数身份
不稳定时进行跨编译 placement 回放。这些模式仅用于实验测量，不是生产求解策略，
也不能与 `dsa_solution_dir` 同时使用。

system-test harness
支持 `--memory-planner=dsa`、`--dsa-export-dir=...` 以及用于精确 A/B 回放的
`--dsa-solution-dir=...`。使用 `--ptoas-sync-summary-dir=...` 可为每个代码
生成单元保存一份机器可读的 PTOAS InsertSync JSONL 摘要，从而比较两个有效
placement 的下游同步开销。该选项要求 PTOAS 构建包含实验性
`--pto-insert-sync-summary` 参数。

默认导出使用 `pypto_hard_v1`：标准 DSA 几何、固定内存空间、单个保守的分配
生命周期包络、容量/保留区、对齐和带类型的 separation。生命周期不相交的
缓冲可以部分复用已释放区域，包括 #1908 所需的区域细分。若 strict pipeline
intent 无法 fit，adapter 会显式创建 cost-aware `pypto_research_v1` relaxation，
并发出 `PH-DSA-001`。独立工具仍可读取旧的 `pypto_structured` 文档，但 PyPTO
不再生成该 profile。完整问题与 objective 定义由独立 solver 维护，见
[PyPTO 与动态存储分配](https://github.com/tonibohnlein/dsa-solver/blob/main/docs/pypto_dsa.md)。

## 算法

1. **收集 MemRef**：遍历函数体，从 TileType 变量中找到所有唯一的 MemRef 对象
2. **按内存空间分组**：按内存空间（Vec、Mat、Left、Right、Acc）组织 MemRef
3. **解析 reserve_buffer**：在每个函数中扫描 `system.reserve_buffer`，为 AUTO buffer 分配显式 base，并计算每个内存空间的保留区末尾地址
4. **分配地址**：对于每个内存空间，委托给 `MemoryAllocatorPolicy` 进行空间过滤、MemRef 排序和地址对齐。默认策略按 ID 排序、使用 32 字节对齐，并从保留区末尾（或 `0`）开始分配
5. **原地更新**：使用 `MemRefUpdateMutator` 完成以下操作：
   - 将变量类型（TileType/TensorType）中的旧 MemRef 引用替换为包含实际地址的新 MemRef
   - 更新已有的 `tile.alloc` `AssignStmt`：替换左值 MemRef 并更新 Call 表达式 (Expression) 中的 addr 参数
   - 把 `system.reserve_buffer` 的 kwargs 改写为显式 `base`

### 独立 DSA 路径

启用 `MemoryPlanner.DSA` 时，第 4 步替换为下面的受保护路径：

1. 复用 MemoryReuse 中感知 phi/loop 的生命周期分析，但不运行其机会性 coalescer。
2. 每个强制 `MemRef.base_` identity 导出一个 buffer。buffer 大小取成员最大值，因此
   不同大小的值可以在生命周期不同阶段占用该 identity。导出的生命周期采用保守的
   allocation hull：从最早的成员定义一直延伸到最晚的成员使用。单个 SSA 成员之间的
   gap 不会被视为物理内存已经失效，因为 loop carry、view 和原地 alias 可能让值跨越
   该 gap 继续存活。只有单独证明每个 hole 中的物理值确实失效后，才能启用
   multi-interval 复用。
3. 把 PyPTO statement point 转成半开区间的读/写 event。定义从
   `2 * def + 1` 开始，最后一次读在 `2 * last_use + 1` 结束；没有后续读取的值仍占用
   一个写 event。因此，一个输入的最后一次读取可以和同一语句写出的结果共用地址。
4. 导出固定 memory pool、后端容量、前导 reserved range，以及 pipeline clone、后端
   hazard 和算子专用 no-alias 规则产生的 hard separation pair。每个 requested
   pipeline stage 首先获得独立 residue，所有 cross-stage member pair 都保持
   hard-separated；每条 separation 都保留其类型化来源。
5. 保留规范化的 alias class 成员和 pipeline group/stage/residue 数据。provenance
   本身不改变 placement。
6. 显式启用时识别潜在的错误物理依赖。覆盖优先的二次参考模式先把已解析的
   `external`、`UB`、`L1`、`L0` memory class 映射到抽象传输或计算资源，再针对所有
   lifetime 可复用 pair 比较 access frontier，并保留精确 arena、control path、loop 和
   byte range 上下文。末端读写后若出现首次写入，就记录 WAR 或 WAW candidate；SSA
   reachability 只作为 evidence 保留，不能证明异步访问已经完成。实验性 v3 policy
   只把信息完整、flat、cross-resource 的 candidate 转换成单位权重 `cross_pipe`
   schema edge。same-resource、nested、loop-carried、partial-range 和不确定的 candidate
   仅记录而不定价。
7. 验证 strict schema/profile，先尝试 deterministic first-fit，再尝试 bounded
   PyPTO-structured search。若未找到 capacity-fitting placement，则只删除
   `pipeline_stage` reason，保留所有 correctness reason，增加单位
   `pipeline_serialization` penalty，并求解显式 research relaxation。
8. 针对大小、对齐、生命周期、pool、容量、reserved range 和 separation 独立验证
   每个 placement。relaxed solution 还会依据 strict problem 再次验证，避免 relaxed
   search 偶然找到 strict-valid placement 时产生 warning。
9. 写回 placement，同时保留每个 view 的相对 byte offset。仅当最终 placement
   确实放松 pipeline intent 时发出 `PH-DSA-001`。

版本 1 adapter 刻意保持 pool assignment 固定。strict problem 在 capacity 下最小化
peak。显式 fallback 依次优先考虑 capacity、reuse cost、total peak 与 maximum pool
peak。导出 interval 中不可见的 branch exclusivity 仍会保守处理，而不是产生不健全的
复用。buffer 仍是固定大小的分配；所谓 subdivision 是在较早区域失效后联合分配
offset，而不是在 buffer 生命周期中调整其大小。

调度本身也会在本 Pass 前固定，尽管不同的合法调度会产生不同生命周期。有关 PyPTO
负责、PTOAS 负责和跨层联合优化三种方案，请参阅
[调度与片上内存规划联合优化](../proposals/joint_schedule_memory_cooptimization.md)。

设置 `dsa_export_dir` 后，每个 InCore 函数会输出两个文件：

- `pypto_<escaped-function-name>.dsa.json`：确定性问题；
- `pypto_<escaped-function-name>.dsa.solution.json`：所选 placement、问题
  fingerprint 和 solver 元数据。

问题不包含 IR pointer 或机器专用路径，可直接复制到独立 corpus。solution
artifact 是 solver/PTOAS A/B 实验的受控接口：不要编辑 IR 或 problem；使用
`dsa-bench --solution-output` 生成匹配的 solution，再通过 `dsa_solution_dir`
回放。

**地址分配（默认策略）**：

- 每个内存空间有独立的地址空间；如果该空间前面已有 `system.reserve_buffer` 保留窗口，则 tile 会从该窗口之后开始分配
- 地址 32 字节对齐：`next_addr = align32(current_addr + size)`
- MemRef 按 ID 排序以确保确定性的分配顺序
- DDR MemRef 被跳过（地址由外部管理）

**视图 MemRef（切片）共享同一个 slot**：

共享同一 `base_` Ptr 的 MemRef（根分配加上其 `tile.slice` 视图）会被放入同一个 slot，slot 大小取最大成员的大小，因为每个视图在物理上都是父分配的别名。每个成员保留其在 slot 内的相对偏移：`new_addr = slot_base + member.byte_offset`（即 InitMemRef 计算出的相对偏移）。根位于 `slot_base`；第 `k` 行的视图位于 `slot_base + k * row_stride`。这对于那些视图偏移不会在 codegen 阶段重新推导的链尤为重要——例如对 `tile.slice` 做 `tile.reshape` 不会发出 `pto.subview`，其 `pto.alloc_tile addr` 直接从该 MemRef 偏移读取。

后端可以通过 `Backend::CreateMemoryAllocatorPolicy()` 提供自定义 `MemoryAllocatorPolicy` 来覆盖上述默认行为。详见下方[分配策略](#分配策略)章节。

## 示例

### 之前（InitMemRef + MemoryReuse 之后）

```python
# SeqStmts [
mem_vec_0: MemRefType = tile.alloc(Vec, -1, 16384, 0)   # addr=-1 (unallocated)
mem_vec_1: MemRefType = tile.alloc(Vec, -1, 16384, 1)   # addr=-1 (unallocated)
tile_a: Tile[[64, 64], FP32, memref=mem_vec_0] = tile.load(...)
tile_b: Tile[[64, 64], FP32, memref=mem_vec_1] = tile.add(tile_a, ...)
# ]
```

### 之后（地址已分配）

```python
# SeqStmts [
mem_vec_0: MemRefType = tile.alloc(Vec, 0, 16384, 0)      # addr=0
mem_vec_1: MemRefType = tile.alloc(Vec, 16384, 16384, 1)   # addr=16384 (aligned)
tile_a: Tile[[64, 64], FP32, memref=mem_vec_0] = tile.load(...)
tile_b: Tile[[64, 64], FP32, memref=mem_vec_1] = tile.add(tile_a, ...)
# ]
```

### 多内存空间

```python
# Before:
mem_vec_0: MemRefType = tile.alloc(Vec, -1, 2048, 0)
mem_left_1: MemRefType = tile.alloc(Left, -1, 2048, 1)
mem_right_2: MemRefType = tile.alloc(Right, -1, 2048, 2)
mem_acc_3: MemRefType = tile.alloc(Acc, -1, 2048, 3)

# After (each space starts from addr=0):
mem_vec_0: MemRefType = tile.alloc(Vec, 0, 2048, 0)
mem_left_1: MemRefType = tile.alloc(Left, 0, 2048, 1)
mem_right_2: MemRefType = tile.alloc(Right, 0, 2048, 2)
mem_acc_3: MemRefType = tile.alloc(Acc, 0, 2048, 3)
```

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`

```cpp
Pass AllocateMemoryAddr();
```

**实现文件**：`src/ir/transforms/allocate_memory_addr_pass.cpp`

- `memref_collectors::CollectMemRefsWithSpace` 收集唯一的 MemRef 及其内存空间
- `AllocateMemoryAddresses` 使用 `MemoryAllocatorPolicy` 在每个内存空间内分配顺序对齐的地址
- `dsa_adapter::BuildStructuredProblem` 导出与 IR 解耦的 schema-v1 problem
- `dsa_adapter::Solve` 对所选 standalone solver 做 capability matching，并独立验证结果
- DSA 路径首先强制完整请求的 pipeline depth；若找不到可装入的 placement，则显式放松
  仅带 `pipeline_stage` 的 separation，最小化对应 reuse cost，并发出 `PH-DSA-001`
- `dsa_adapter::BuildMemRefReplacements` 完成保留 view offset 的写回
- `MemRefUpdateMutator` 在一次遍历中同时更新变量类型和 `tile.alloc` 语句参数

**Python 绑定**：`python/bindings/modules/passes.cpp`

```cpp
passes.def("allocate_memory_addr", &pass::AllocateMemoryAddr,
           "Allocates real memory addresses for existing alloc operations.");
```

**测试**：`tests/ut/ir/transforms/test_allocate_memory_addr_pass.py`

- 测试 32 字节对齐的地址分配
- 测试多 MemRef 分配
- 测试空函数（无 Tile）
- 测试 alloc 语句被前置到函数体顶层 `SeqStmts`
- 测试 MemRef 去重的原始指针唯一性
- 测试无后端配置时的默认策略行为
- 测试 DSA 读先于写的复用、reserved range、view offset 写回与确定性导出
- 测试 alias class、类型化 separation、strict pipeline intent、显式 reuse-cost fallback
  及其性能提示
- 通过 exporter、独立 solver、validator 和 writeback 重放 #1908 fragmentation 形状

## 分配策略

该 Pass 将放置决策委托给 `MemoryAllocatorPolicy` 接口 (`include/pypto/ir/memory_allocator_policy.h`)，使分配策略可扩展而无需修改 Pass 本身。

### 接口

```cpp
class MemoryAllocatorPolicy {
 public:
  virtual ~MemoryAllocatorPolicy() = default;
  virtual bool ShouldAllocate(MemorySpace space) const = 0;
  virtual uint64_t AlignAddress(uint64_t addr, MemorySpace space) const = 0;
  virtual void OrderMemRefs(std::vector<MemRefPtr>& refs) const = 0;
};
```

| 方法 | 用途 | 默认行为 |
| ---- | ---- | -------- |
| `ShouldAllocate` | 过滤哪些内存空间需要分配地址 | 跳过 DDR；分配所有片上空间 |
| `AlignAddress` | 对给定空间的原始地址进行对齐 | 32 字节对齐 |
| `OrderMemRefs` | 在分配前对空间内的 MemRef 排序 | 按 `MemRef::id_` 升序 |

### 默认策略

`DefaultMemoryAllocatorPolicy` 保留了原始硬编码行为（跳过 DDR、32 字节对齐、按 ID 排序）。

### 后端覆盖

当后端已配置（`BackendConfig::IsConfigured()`）时，Pass 调用 `Backend::CreateMemoryAllocatorPolicy()` 获取策略。默认的 `Backend` 实现返回 `DefaultMemoryAllocatorPolicy`。自定义后端可以覆盖此虚方法以提供不同的对齐规则、排序策略或空间过滤：

```cpp
class MyBackend : public Backend {
 public:
  MemoryAllocatorPolicyPtr CreateMemoryAllocatorPolicy() const override {
    return std::make_unique<MyCustomPolicy>();
  }
};
```

当未配置后端时（例如在单元测试中），Pass 会自动回退到 `DefaultMemoryAllocatorPolicy`。
