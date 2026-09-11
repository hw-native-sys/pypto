# AllocateMemoryAddr Pass

为已有分配所支持的 MemRef 分配实际内存地址。

## 概述

该 Pass 为非 DDR 的 MemRef 分配具体地址。`tile.alloc` 语句声明分配根和大小，
因此其 Ptr 结果与 Call 参数保持不变；带地址的 MemRef 位于 tile 和 tensor 类型上。
它还会在 PTO codegen 之前解析 `system.reserve_buffer(base=AUTO)`。地址放置由
`MemoryPlanner` 选择：

- `PYPTO` 在 `MemoryReuse` 之后使用确定性的顺序分配器。
- `DSA_RP` 在进程内构建并求解带复用惩罚、受容量约束的动态存储分配问题。
- `PTOAS` 跳过本 pass，把地址分配交给 ptoas。

**核心职责**：

- 从 TileType 变量中收集唯一的 MemRef 对象
- 在每个函数中把 `system.reserve_buffer` 的 base 解析成显式地址
- 在每个独立内存空间内分配对齐地址
- `DSA_RP` 在满足容量的放置中保持正确性约束并最小化已识别的高代价复用
- 更新所有变量类型 (Type) 中的 MemRef 地址
- 保留 `tile.alloc` 指针声明，同时更新所有 MemRef 使用点

**使用时机**：代码生成前的最后一个内存管理 pass。`PYPTO` 下它在
`MemoryReuse` 之后运行；`DSA_RP` 下跳过 `MemoryReuse`，本 pass 直接消费
`MaterializeSemanticAliases` 产生的分配身份。

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

`DSA_RP` 是默认 planner。当编译不能继承外层 `PassContext` 的策略时，
可以显式选择它：

```python
from pypto.ir import compile
from pypto.pypto_core import passes

compile(program, memory_planner=passes.MemoryPlanner.DSA_RP)
```

## 算法

1. **收集 MemRef**：遍历函数，收集唯一分配身份及其 view。
2. **解析 reserve_buffer**：为 AUTO 预留区分配显式 base，并把这些范围从普通
   放置中排除。
3. **选择放置**：
   - `PYPTO`：按名称排序 MemRef，并顺序分配对齐地址。
   - `DSA_RP`：构建下述进程内问题并运行 canonical greedy。
4. **原地更新**：使用 `MemRefUpdateMutator` 完成以下操作：
   - 将变量类型（TileType/TensorType）中的旧 MemRef 引用替换为包含实际地址的新 MemRef
   - 把 `system.reserve_buffer` 的 kwargs 改写为显式 `base`

### DSA-RP 策略

每个片上内存空间都是独立的固定容量 arena。强制别名物化后的每个分配身份成为一个
buffer，带有字节大小、对齐和保守的半开生命周期。问题包含：

- 生命周期干涉、预留范围、语义 no-alias、目标 hazard 和请求的流水线 stage
  分离等**硬约束**；作者声明的 `pl.MemRef`
  分配还会与同一内存空间中的其他所有分配建立硬分离。多 slot 声明会作为覆盖完整
  声明范围的单个 buffer 放置，同时每个成员保留其常量或运行时选择的 slot 偏移。
  对于注册为支持函数式原地执行的算子，输入的最后一次读可以与输出写共享算子边界，
  但两个分配必须使用相同基址或完全分离。相同基址允许窄化转换等安全的不同大小操作；
  交错重叠仍被禁止；
- 对生命周期兼容的物理复用，如果内置 recognizer 将其识别为跨 resource WAR 或 WAW
  handoff，则加入**单位权重软边**；
- 硬 arena 容量。容量与正确性绝不会为了降低复用代价而放宽。

### 识别 handoff

只有当较早分配的访问尚未被保证在较晚分配的首次写入之前完成时，复用才值得计费。
因此 recognizer 为每个分配保留在保证执行顺序下的 **maximal access** 以及完整的
**minimal initial-write frontier**，而不是每个 resource 只保留一个访问。把 maximal
access 排在新的首次写入之前，也就排好了其余所有访问；已经排在同一分配的另一个访问
之前的访问不可能产生 handoff，会被丢弃。

当较早分配的某个 maximal access 与较晚分配的某个首次写入使用**两个不同的抽象
resource**，且两个分配的访问集合完整、已分类、覆盖整个分配，并且首次写入已验证时，
该 pair 会被晋升。相同 resource、部分 view、结构有歧义以及访问不完整的情形不加惩罚；
已因正确性或流水线意图分离的 pair 同样不加惩罚。同一算子的两个端点属于该算子契约的
原地别名问题，而不是可选复用，交由上面的 same-base-or-disjoint 关系处理。

访问由与目标无关的 route 标识，即其源和目标 memory class，它指明必须完成的引擎类别。
当前 backend 只判断所选 SoC 能否执行该算子的传输；无法分类的算子会让其分配不加惩罚。
recognizer 不调用也不模拟 ptoas。

顺序来自 chain-cover 可达性索引，而不是逐语句的传递前驱集合。每个抽象 resource 被建模
为一条按完成顺序排列的 issue chain，因此把该 chain 顺序加入语句依赖边后，这些 resource
构成 happens-before 顺序的一个 chain cover，每条语句一个 vector clock 即可通过读取单个
分量回答顺序查询。组合这两类边正是完成模型已有的假设：若两个访问在同一 resource 上有序，
且数据依赖把第二个排在第三个之前，则第一个也排在第三个之前。

显式 pair 模型是 output-sensitive 的：对于一个 InCore 函数内的 `B` 个可复用 buffer，
一个 kernel 最坏可包含 `Theta(B^2)` 个生命周期冲突或候选 penalty pair，任何 pair 枚举
都不可能比它必须报告的 pair 更省。识别本身对 IR 不是二次的：在 resource 数量固定时，
顺序索引耗费 `O(V + E)` 乘以控制嵌套深度（用于按区域构造依赖图），pair 由按 resource
建索引的生命周期 sweep 枚举。该 sweep 只检查那些已经共享 memory space、生命周期兼容且
提供两个不同 resource 的 pair，因此在没有跨 resource 复用的 kernel 上几乎不检查任何 pair，
而 all-pairs 扫描仍需付出 `Theta(B^2)`。被检查的数量不等于被报告的数量：一个被检查的 pair
仍可能因 separation、同一算子或控制路径互斥而被拒绝。每个 allocation 的 completion frontier
按 (resource, 控制路径, 字节范围) 各取一个代表做两两比较。固定数量的 canonical 放置顺序需要
`O(B^2 log B)` 时间和 `O(B^2)` 空间。若所有构造式顺序都失败，一个仅检查可行性的精确
fallback 会在固定的 100,000 个候选放置工作预算内枚举对齐放置。这是默认 `DSA_RP`
planner 对通用 pass 复杂度策略的已记录例外；其范围限于单个函数，使用固定数量的构造式 restart，
并对原本可能呈指数复杂度的 fallback 设置明确上界。显式选择的旧版 `PYPTO` planner 仍使用顺序分配路径。

Canonical greedy 尝试偏移 `0`、预留范围末尾，以及已放置硬/软邻居的对齐顶部。
每个 buffer 先选择增量惩罚最低的候选，再选最低地址。它评估多种确定性顺序，并保留
一个可行的、惩罚盲的 first-fit 放置作为 incumbent。写回前由独立 validator
检查最终放置。若没有构造式顺序找到放置，精确 fallback 会返回可行见证、证明不存在放置，
或报告固定搜索预算已耗尽。预算耗尽会被单独诊断，不会被报告为内存不足。

流水线意图采用先硬后软策略：

1. 先在所有请求的跨 stage 分离均为硬约束时运行 canonical greedy 及其有界精确 fallback。
2. 若严格问题被证明无法放置，或精确 fallback 到达工作预算，则仅放宽唯一硬理由为
   流水线意图的 pair，把它们改为单位复用惩罚后再次搜索。
3. 若最终放置重叠了放宽的 pair，发出 `PH-DSA-001` 性能诊断。所有语义与目标
   hazard 分离始终保持为硬约束。若放宽后的精确搜索证明无解，则报告编译期 OOM/no-fit。
   若它先到达固定工作预算，则报告搜索耗尽，而不声称容量不足。

> **工具链要求：** `DSA_RP` 依赖 ptoas InsertSync 识别不同分配根之间的物理范围
> 重叠。应使用包含 tile-native 内存规划器及其跨根本地范围重叠分析的现代 ptoas
>（PTOAS PR #913 及后续修复）。仅比较分配根身份、而不比较规划后物理范围的旧版本
> 与 DSA-RP 放置不兼容。

模型、recognizer、求解、验证和写回全部在进程内完成。`DSA_RP` 不提供问题导出、
放置 replay、参考放置或 profiling 接口。

### 顺序 `PYPTO` 策略

- 每个内存空间有独立的地址空间；如果该空间前面已有 `system.reserve_buffer` 保留窗口，则 tile 会从该窗口之后开始分配
- 地址 32 字节对齐：`next_addr = align32(current_addr + size)`
- MemRef 按名称排序以确保确定性的分配顺序
- DDR MemRef 被跳过（地址由外部管理）

**视图 MemRef（切片）共享同一个 slot**：

共享同一 `base_` Ptr 的 MemRef（根分配加上其 `tile.slice` 视图）会被放入同一个 slot，slot 大小取最大成员的大小，因为每个视图在物理上都是父分配的别名。每个成员保留其在 slot 内的相对偏移：`new_addr = slot_base + member.byte_offset`（即 InitMemRef 计算出的相对偏移）。根位于 `slot_base`；第 `k` 行的视图位于 `slot_base + k * row_stride`。这对于那些视图偏移不会在 codegen 阶段重新推导的链尤为重要——例如对 `tile.slice` 做 `tile.reshape` 不会发出 `pto.subview`，其 `pto.alloc_tile addr` 直接从该 MemRef 偏移读取。

后端可以通过 `Backend::CreateMemoryAllocatorPolicy()` 提供自定义 `MemoryAllocatorPolicy` 来覆盖上述默认行为。详见下方[分配策略](#分配策略)章节。

## 示例

### 之前（所选复用分析之后）

```python
# SeqStmts [
mem_vec_0: Ptr = tile.alloc(Vec, 16384)
mem_vec_1: Ptr = tile.alloc(Vec, 16384)
tile_a: Tile[[64, 64], FP32, MemRef(mem_vec_0, -1, 16384)] = tile.load(...)
tile_b: Tile[[64, 64], FP32, MemRef(mem_vec_1, -1, 16384)] = tile.add(tile_a, ...)
# ]
```

### 之后（地址已分配）

```python
# SeqStmts [
mem_vec_0: Ptr = tile.alloc(Vec, 16384)  # 声明保持不变
mem_vec_1: Ptr = tile.alloc(Vec, 16384)  # 声明保持不变
tile_a: Tile[[64, 64], FP32, MemRef(mem_vec_0, 0, 16384)] = tile.load(...)
tile_b: Tile[[64, 64], FP32, MemRef(mem_vec_1, 16384, 16384)] = tile.add(tile_a, ...)
# ]
```

### 多内存空间

```python
# Before:
mem_vec_0: Ptr = tile.alloc(Vec, 2048)
mem_left_1: Ptr = tile.alloc(Left, 2048)
mem_right_2: Ptr = tile.alloc(Right, 2048)
mem_acc_3: Ptr = tile.alloc(Acc, 2048)

# After (each space starts from addr=0):
tile_vec: Tile[..., MemRef(mem_vec_0, 0, 2048)] = ...
tile_left: Tile[..., MemRef(mem_left_1, 0, 2048)] = ...
tile_right: Tile[..., MemRef(mem_right_2, 0, 2048)] = ...
tile_acc: Tile[..., MemRef(mem_acc_3, 0, 2048)] = ...
```

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`

```cpp
Pass AllocateMemoryAddr();
```

**实现文件**：`src/ir/transforms/allocate_memory_addr_pass.cpp`

- `MemRefCollectorVisitor` 从 TileType 变量中收集唯一的 MemRef
- `AllocateMemoryAddresses` 使用 `MemoryAllocatorPolicy` 在每个内存空间内分配顺序对齐的地址
- `dsa_adapter::BuildDsaAllocationPlan` 在
  `src/ir/transforms/dsa/allocation_plan.cpp` 中推导保守生命周期和强制 separation
- `dsa_adapter::BuildProblem` 构建精简的进程内 DSA-RP 模型
- `dsa::CanonicalGreedySolver` 搜索满足容量的放置，
  `dsa::ValidateSolution` 独立验证结果
- `MemRefUpdateMutator` 在一次遍历中更新变量与表达式类型中的 MemRef，并改写已解析的
  `system.reserve_buffer` base；`tile.alloc` 保持为指针与大小声明

**Python 绑定**：`python/bindings/modules/passes.cpp`

```cpp
passes.def("allocate_memory_addr", &pass::AllocateMemoryAddr,
           "Allocates real memory addresses for existing alloc operations.");
```

**测试**：
`tests/ut/ir/transforms/test_allocate_memory_addr_pass.py`、
`tests/ut/ir/transforms/test_dsa_reuse_penalty_recognizer.py` 和
`tests/ut/cpp/dsa_reuse_penalty_solver_test.cpp`

- 测试 32 字节对齐的地址分配
- 测试多 MemRef 分配
- 测试空函数（无 Tile）
- 测试 alloc 语句被前置到函数体顶层 `SeqStmts`
- 测试 MemRef 去重的原始指针唯一性
- 测试无后端配置时的默认策略行为
- 测试容量诊断会归因跨核流水 ring 预留的字节数（见下文）
- 测试 DSA-RP 几何、容量、硬约束、惩罚激活、确定性 canonical-greedy 放置、
  以及独立验证
- 直接测试 solver 之前的精确 recognizer edge 集合，并测试最终放置几何
- 测试精确 fallback 的可行见证、已证明 no-fit 和有界搜索耗尽结果
- 测试可选原地执行的同基址或完全分离放置，包括不同大小的窄化转换

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
| `OrderMemRefs` | 在分配前对空间内的 MemRef 排序 | 按 `MemRef::name_hint_` 升序 |

### 默认策略

`DefaultMemoryAllocatorPolicy` 保留了原始硬编码行为（跳过 DDR、32 字节对齐、按名称排序）。

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

## 容量校验

容量检查存在于两处：`AllocateMemoryAddresses` 自身的 in-pass `CHECK`（它掌握唯一精确的 footprint——会计入已声明分配中未绑定的 slot，且不依赖每个 tile 地址都是常量），以及 `AllocatedMemoryAddr` 属性校验器（按内存空间跟踪高水位 `addr + size`）。两者都与 `Backend::GetMemSize(space)` 比较，且都会输出下面这段说明——具体命中哪一个取决于配置，因此该措辞是共享的（`ReservedBytesNote`），而非只写在其中一处。

由于 `system.reserve_buffer` 预留的是一段前导窗口、其后所有 tile 才依次分配，这些字节会计入高水位，但它们**不是** MemRef——在作者能查看的逐 tile 账目中完全不可见。因此当溢出的空间正是为某个 reserve buffer 付费的那个空间时，诊断会显式指出这段窗口。它表述为分配**下界**（`reserved_end_by_space`，即 tile 被放置于其上的对齐后最大末地址），而非“这些 buffer 占用的字节数”：显式指定 base 的 buffer 或对齐空隙都会让下界超过各 buffer 大小之和，而计入本次溢出的正是这个下界。

```text
Function 'qk_pv_aic': Mat buffer usage (1064960 bytes) exceeds platform limit (524288 bytes).
The first 1048576 bytes of that space are reserved by system.reserve_buffer, so tiles
are allocated above them — this is the cross-core pipe ring. Lower its depth with
optimizations=[pl.cross_core_slot(slot_num=N)] on the enclosing pl.at(...), or shrink the
tile that crosses the cube/vector boundary
```

该 ring 大小为 `slot_size x slot_num` 字节，由 `BuildAutomaticPipeSetup`（`src/ir/transforms/utils/cross_core_pipe.cpp`）构建——`slot_size` 是消费方弹出的**完整** tile，`slot_num` 默认取 `kDefaultAutoPipeSlotNum`（每个方向均为 2）。该策略数值刻意不写入诊断信息以免过期；信息中报告的字节数直接取自分配器用作下界的同一个 `ResolveReserveBufferBases` 结果。该 ring 位于**消费侧**核的内存中——V2C（`pl.aic_gather`）为 Mat/L1，C2V（`pl.aiv_shard`）为 Vec/UB——因此该提示只会针对 `GetReserveBufferMemorySpace` 为该函数映射到的空间发出。有两处刻意做了限定：若某函数的 reserve buffer 在 Mat 而溢出发生在 Vec，则只给出基础信息；而 `pl.cross_core_slot` 这条修复建议仅在该 buffer 确实是流水 ring 时才追加。这是**精确**名称匹配而非后缀判断：会用本函数自身的 kernel 名重新调用 `BuildPipeBufferName`（`<kernel>_aic` / `<kernel>_aiv` -> `<kernel>_v2c_slot_buffer`），因为 `pl.reserve_buffer` 接受任意名称，手写的 `scratch_v2c_slot_buffer` 不应被指向一个无法调整它的旋钮。手写的 `pl.reserve_buffer` 仍会得到字节归因，但不会被指向一个无法缩小它的旋钮。

ring 深度刻意**不**自动收缩以适配容量：深度就是跨核流水的深度，静默调小会把一个显式的编译错误变成一次无声的吞吐回退。
