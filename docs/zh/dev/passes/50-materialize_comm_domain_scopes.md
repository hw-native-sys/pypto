# MaterializeCommDomainScopes Pass

## 概览

`MaterializeCommDomainScopes` 扫描每个 host-orchestration 函数，组装出分布式 runtime
为分配 / 填充 per-rank 通信窗口所需要的 host 侧元数据。它与
[`InitMemRef`](38-init_memref.md) 在结构上完全同构：追溯一次分配到所有
消费点，构造反向引用对象，再把该对象挂到 IR 类型上，让下游 codegen 能 O(1)
访问。

| 比较项 | `MemRef` 一侧 | `WindowBuffer` 一侧 |
| ------ | ------------- | ------------------- |
| 分配 op | `tile.alloc(memory_space, size_in_bytes)` | `pld.tensor.alloc_window_buffer(size_in_bytes)` |
| Parse 时赋值语句 LHS | `Var(PtrType)` | `Var(PtrType)`（同一个 singleton） |
| 包装 Var 子类 | `MemRef` | `WindowBuffer` |
| 包装类的 SSA-edge 类型 | `MemRefType`（singleton） | `WindowBufferType`（singleton） |
| 构造者 | `InitMemRef` | **`MaterializeCommDomainScopes`**（本 pass） |
| 回填到 | `TensorType.memref_` | `DistributedTensorType.window_buffer_` |
| Program 级注册表 | `Program.functions_`（alloc 语句） | `CommDomainScopeStmt wrappers in each host_orch body` |

## 异步工作区引擎

`CommDomainScopeStmt.engines` 记录通信域需要的异步工作区引擎 (async workspace engines)。
物化通信域之前，此 Pass 扫描整个 Program 中每个函数的 Call 和 Submit。
只要发现 `pld.tensor.indexed_all_to_all_v` 或其降低后的 builtin，所有物化通信域都设置
`engines=["urma"]`；否则为空列表。

当前采用保守策略，包括未调用函数和无关通信域。精确匹配留待后续实现。
分布式代码生成仅根据该 IR 字段输出 `workspace_engines`，不再分析通信需求。
结构比较、序列化和打印解析往返均保留该字段。

## 流水线位置

```text
... -> ExpandManualPhaseFence -> SynthesizeAllReduceSignals -> MaterializeCommDomainScopes -> LowerHostTensorCollectives -> MaterializeDistTensorCtx -> Simplify（最终）
```

本 pass 跑在默认 pipeline 的末尾阶段，位于
[`LowerHostTensorCollectives`](51-lower_host_tensor_collectives.md) 和最后一次
`Simplify` 之前。从 `InlineFunctions` 到这里之间的所有 pass 都不会触碰
host_orch 的 alloc / window / dispatch 链：host_orch 本身不会被 tile lower，
L2（chip 级）orchestration 也永远不会被 inline 进 L3，所以本 pass 需要的
alloc / view / dispatch 点在此时仍然可见。放在较晚阶段还能让产生 IR 在描述符
分析之前先被充分规范化，最后的 `Simplify` 也能统一对收集到的 size 表达式做常量折叠。

## 算法

对每个 host-orchestration 函数（`Function::level_ == Level::HOST` 且
`Function::role_ == Role::Orchestrator`，不强求 `func_type_`）：

1. **收集 alloc**。找到所有 RHS 是 `pld.tensor.alloc_window_buffer(size, *, name)`
   的 `AssignStmt`。记录 `(ptr_var, size_expr, name, span, call)`。

2. **收集 view**。所有 RHS 是 `pld.tensor.window(ptr_var, [shape], *, dtype)`、且
   引用已记录 `ptr_var` 的 `AssignStmt`，记录 `view_var → alloc` 绑定。

3. **扫描 dispatch**。带着 `ForStmt` 栈遍历函数体。对每个 `op_` 是
   `GlobalVar` 且解析到 chip-level orchestration 的 Call，读 `attrs["device"]`，
   在当前 for 循环上下文中推导 **device 描述符**：

   | `device=` 形态 | 描述符 |
   | -------------- | ------ |
   | `ConstInt(N)` | `subset = {N}` |
   | `for r in pl.range(pld.system.world_size())` 的 IterArg | `kAll` |
   | `for r in pl.range(ConstInt(N))` 的 IterArg | `subset = {0, …, N − 1}` |
   | 其它 | `pypto::ValueError` |

   每个位置参数若是已记录的 view Var，就向对应 alloc 追加该描述符。

4. **合并描述符**。对每个 alloc 折叠记录到的所有描述符：任何 `kAll` ⇒ `kAll`；
   否则取 subset 并集。HOST 级 collective 的 signal（以及
   `pld.tensor.all_to_all_v` 的 `recv_counts`）自身没有 `device=`，因此从
   collective 配对的 data/target alloc 继承设备覆盖范围——一次调用可以注册
   多个共享同一 data alloc 的配对（`all_to_all_v` 会注册两个：一个用于
   `signal`，一个用于 `recv_counts`）。

5. **构造 `WindowBuffer`**。对每个 alloc 构造
   `WindowBuffer(base = ptr_var, size = size_expr, load_from_host = false,
   store_to_host = false)`；`Var::name_hint_` 自动继承自 `ptr_var->name_hint_`。
   （host-staging 标志位是 N4+ 的占位字段。）

6. **改写 view 类型**（仅 host_orch）。对每个 view 绑定，mint 一个同
   `name_hint_` 的新 `Var`，类型为
   `DistributedTensorType(shape, dtype, memref, tensor_view, wb)`；用
   `Substitute` 把所有对旧 view Var 的引用替换为新 Var。`Substitute` 只改写
   *引用*，因此随后还要把定义每个新 Var 的 Call 按同一个带 `window_buffer_`
   的类型重新 mint——否则该赋值语句结束时左值带回指、右值不带，违反
   [`AssignTypeSymmetry`](99-verifier.md#built-in-rules)。重新 mint 只会*补上*
   回指：仅当重建出的类型与 Var 的类型结构相等时才采用，真正的 shape / dtype /
   view 分歧留给验证器报错。同一 alloc 被 N 次
   `pld.tensor.window` 物化的多个 view 共享同一 `shared_ptr<const WindowBuffer>`。
   chip-orch / InCore 形参类型不动。

7. **聚类成 group**。按源代码顺序遍历 alloc 列表，匹配描述符已存在的
   comm-domain scope 则追加 slot，否则新开一个。`CommDomainScopeStmt wrappers in each host_orch body` 最终
   填充该列表。

## Sanity 校验

下列情况抛 `pypto::ValueError`（携带 alloc 的 span）：

- 某 alloc 没有任何 `pld.tensor.window` 物化（dead alloc）。
- 某 alloc 有 view 但没有 chip-orch dispatch 消费它。
- dispatch 的 `device=` 既不是 `ConstInt`、也不是已识别的 `pl.range`
  归纳变量。
- 同一 comm-domain scope 内 `name_hint_` 重名（parser 已在程序范围内做了唯一性
  校验，本 pass 再次断言）。
- `pld.tensor.all_to_all_v` 调用位于 HOST orchestrator 的 `for`/`while`
  循环内——动态重复调用需要循环携带的窗口生命周期管理，本编译器尚未建模
  （信用式信号本身可跨顺序调用复用；与 `LowerCompositeOps` 在 InCore 路径上
  强制的限制相同）。

## 输出不变量

pass 运行之后：

- `CommDomainScopeStmt wrappers in each host_orch body` 已填（程序不分配 window buffer 时为空）。
- 每个 `pld.tensor.window` 结果 Var 的类型是 `DistributedTensorType`，
  `window_buffer_` 字段指向对应的 `WindowBuffer`；定义它的 Call 带同一个类型——
  赋值两侧一致。
- comm-domain 分析和 `LowerHostTensorCollectives` 看到的每个 host-level
  `pld.tensor.allreduce` 都已经有两个位置参数（在
  [`SynthesizeAllReduceSignals`](49-synthesize_allreduce_signals.md) 运行之后）。
  用户省略 signal 时，第二个参数是前置
  `pld.tensor.window` 赋值语句产生的合成 Var。
- 同一 alloc 的多个 view 共享同一 `shared_ptr<const WindowBuffer>`——指针
  相等是下游 codegen 的关键不变量。
- chip-orchestration 与 InCore 的形参类型 `window_buffer_` 仍是 `nullopt`。
  N7 codegen 在 *host_orch* 的 dispatch 处读取反向引用、再为 chip-orch 显式
  下发对应的 `CommContext` 指针。

## 文本 round-trip

用户编写程序时仍然自动推导通信域。打印后的 IR 使用私有 `pl._dump` 语法保存物化状态：

- `comm_domain(name=..., devices=..., slots=[...], attrs={...})` 保存作用域边界、
  设备列表、slot 顺序及属性。
- `window_buffer("buf_window", base="buf", size=16, load_from_host=..., store_to_host=...)` 定义窗口；
  tensor 类型通过 `window_ref("buf_window")` 引用同一个对象。
  名称是文本格式中的引用标识，不是设备 ID、slot 下标或 IR 字段。
  名称由打印后的 base 变量名生成；共享同一 base 的不同窗口通过后缀区分。旧的数字 ID 仍可读取。
- `alloc_window_buffer(size, name=...)` 保存分配的逻辑名称，不受 SSA 变量重命名影响。
- `call(op, args, result_type, kwargs=..., attrs=...)` 直接恢复 Call，接受类型解析器支持的任意结果类型。
  打印器用它保存分布式调用实际记录的类型，包括不存在反向引用的情况；
  pass 改写操作数后，重新推导的类型可能与原类型不同。

解析器先登记窗口描述，再解析 body，在分配和大小表达式可用时绑定引用。
作用域 slot 与 tensor 反向引用共享同一个恢复后的 `WindowBuffer`。
这些写法属于 IR 文本语法，不是在 kernel 中执行的操作。
描述中的 `base` 引用 body 内定义的分配变量；`window_buffer(...)` 不会增加分配语句。
它的 `size` 和分配 Call 的大小参数是 IR 分别保存的字段，因此都会保留。
空作用域打印为 `pass`。`-1` 等属性字面量保留为标量；
`pl.const(-1, pl.INDEX)` 则表示 IR 表达式。

对于自包含的完整程序，`parse(python_print(program))` 必须与原程序结构相等且结构哈希相等。
解析器和调用方均不需要重跑 lowering pass。引用缺失定义的单条语句片段不属于此契约。
旧的 `"window_buffer=<name>"` 提示字符串仍可读取，但不表达真实引用。

## Pass 属性

| 字段 | 取值 |
| ---- | ---- |
| `required` | `{}` |
| `produced` | `{IRProperty::CommDomainScopesMaterialized}` |
| `invalidated` | `{}` |

## 参考

- 实现：[src/ir/transforms/materialize_comm_domain_scopes_pass.cpp](../../../../src/ir/transforms/materialize_comm_domain_scopes_pass.cpp)
- 头文件：[include/pypto/ir/transforms/passes.h](../../../../include/pypto/ir/transforms/passes.h)
- Schema：[include/pypto/ir/program.h](../../../../include/pypto/ir/program.h)
  定义了 `WindowBuffer` 与 comm-domain scope。
- DSL：[`pld.tensor.alloc_window_buffer`](../../../../python/pypto/language/distributed/op/tensor_ops.py)、
  [`pld.tensor.window`](../../../../python/pypto/language/distributed/op/tensor_ops.py)、
  [`pld.system.world_size`](../../../../python/pypto/language/distributed/op/system_ops.py)。

## Host DDR 窗口

`pld.alloc_window_buffer(size, memory_source="host")` 选择 Host DDR 存储；
默认值仍为 `"device"`。shape/dtype 重载也接受此关键字。
`WindowBuffer.memory_source` 在克隆、序列化和私有 `window_buffer` 文本描述符中保留。
通信域按设备覆盖范围和内存来源共同分组：Host 与 Device 分配不共享物理通信域。

Host 代码生成向 Simpler 请求 `AddressSpace.HOST` 和 `workspace_engines=("urma",)`，
并获取带 URMA 工作区的各 rank 通信上下文。这要求 Simpler #2913 或更新版本，以及其支持的
同机 A5 `tensormap_and_ringbuffer` 配置。运行时 API 能力在 worker 初始化加载入口时检查一次，
需要初始化时也检查 `initializer` 参数；每次派发不再重复反射函数签名。平台和运行时组合由
Simpler 校验。当前 runtime 子模块已提供此 API。

所属 rank 可以通过设备映射的 tensor 进行本地访问。Host peer 窗口不是 MTE 映射，
远端访问必须走 URMA。现有 peer-MTE 原语和集合通信在远端操作数来自 Host 窗口时会被拒绝，
包括通过 chip/InCore 参数转发的情况。URMA 异步算子不被归类为 peer MTE。
Host peer URMA 读取的执行接入仍是独立的后续工作。

### 运行时来源初始化

用 `pld.alloc_window_buffer(size, memory_source="host", init="engram_l1")` 声明运行时来源键。
编译阶段仅记录键和逻辑字节数，不打开或读取文件。编译完成后，为每个域内 rank 绑定一个已打开的二进制文件：

```python
from contextlib import ExitStack
from pypto.runtime import DistributedWorker

compiled = program.compile(...)  # 此时无需 shard 文件
with ExitStack() as stack:
    shard_files = [stack.enter_context(open(path, "rb")) for path in shard_paths]
    with DistributedWorker(compiled, persistent=True,
                           window_sources={"engram_l1": shard_files}) as worker:
        worker(...)  # 创建通信域、加载文件，然后提交 chip 任务
        worker(...)  # 复用已加载的 Host 域
```

`init` 必须是非空字符串，且只支持 Host 窗口。`window_sources` 属于 worker 的运行时配置，
也可传给 `compiled.prepare`。文件序列按**域内 rank 顺序**排列，包括只覆盖部分 worker 的域。
缺失或未使用的来源键、错误的文件数量都会报错。

同一映射也接受连续的 CPU `torch.Tensor`，可以混合文件与 tensor：
`window_sources={"engram_l1": [rank0_tensor, rank1_tensor]}`。
Tensor 从视图的存储偏移起按原始字节复制，不转换 dtype、不创建暂存副本，支持 BF16 和 FP8。
非 CPU、非连续、稀疏、量化以及共轭或负值视图会被拒绝。来源至少包含窗口逻辑字节数，
多余尾部字节被忽略。绑定保留 tensor 对象而非内容快照；创建域时才读取数据，因此派发完成前须保持来源稳定。

Codegen 输出 `(buffer_name, init_key, logical_nbytes)` 绑定，并向固定的
`make_window_initializer` helper 传入 worker 的来源映射。Helper 创建内部回调，
由 Simpler 在所有 rank 映射就绪后调用。对每项绑定，将
`window_sources[init_key][domain_rank]` 复制到相应命名 buffer，偏移由
`domain.buffer_range(buffer_name)` 获取。不会根据文件句柄或 tensor 内容动态生成 Python 函数。

PyPTO 接入 Simpler #2939 的创建时 initializer，从文件偏移零开始，分块直接读入对应命名 buffer
的 Host 映射，并恢复文件原先的位置。只复制逻辑字节数，不改动对齐填充，忽略文件尾部多余字节。
文件过短会使域创建失败；异常路径释放所有 CPU memoryview，允许 Simpler 回滚分配。
不会分配另一份完整表大小的暂存空间。文件必须是可读、可定位的二进制流，由调用者管理生命周期，
首次派发完成前须保持打开；初始化过程中不要修改或并发访问这些文件流。

持久域复用不调用 initializer，首次派发成功后即可关闭文件。新建域会重新读取文件。
未启用 `persistent=True` 时，每次派发都会创建并初始化新域，因此文件须保持打开。
worker 创建时固定来源映射与文件列表，修改原来的字典或列表不会重新加载持久表；更新数据需创建新 worker。

持久执行的复用校验包含内存来源、工作区引擎和初始化绑定。Host 窗口始终保留内容，
不受 `reset_persistent_windows` 影响；Device scratch/signal buffer 保持原有重置策略，默认复用前清零。
通信域仍由 Simpler 负责释放。
