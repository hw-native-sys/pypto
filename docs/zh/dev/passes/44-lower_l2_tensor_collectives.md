# LowerL2TensorCollectives Pass

## 概述

`LowerL2TensorCollectives` 是托管集合通信（managed collective）的 CHIP/L2 通路。
它把写在 **CHIP orchestration 函数体**里的 `pld.tensor.*` 集合通信改写成对编译器
合成的 AIV kernel 的调用，使该集合通信成为调用方自身 pipeline 里的一个普通 task。

目前支持 `pld.tensor.all_to_all_v`，且要求 `core_num` 为编译期常量。

HOST 通路（[`LowerHostTensorCollectives`](51-lower_host_tensor_collectives.md)）
在上一层解决同一问题，做法不同：它把集合通信按设备扇出成**每个设备一次**
`builtin.tensor.*` chip dispatch。每次这样的 dispatch 都是一个完整的 L2
orchestration task，而它唯一的工作就是提交一个 AIV kernel，因此
`compute -> collective -> consume` 序列每张卡要花三次 L3 -> L2 往返。在本通路上
只需一次。

```text
HOST 通路                                 CHIP 通路（本 pass）
─────────                                ─────────────────────
L3 -> L2  stage task                     L3 -> L2  chip_pipeline
L3 -> L2  builtin collective dispatch              ├── stage         (AIV task)
          └── rt_submit_aiv_task                   ├── collective    (AIV task)
L3 -> L2  consume task                             └── consume       (AIV task)
```

## 在流水线中的位置

```text
... -> FuseCreateAssembleToSlice -> LowerL2TensorCollectives -> LegalizeSpmdLaunches -> DeriveCallDirections -> AutoDeriveTaskDependencies -> ...
```

这个位置是必要条件而非偏好。改写后的调用必须像任何其他 kernel 调用一样经过
[`DeriveCallDirections`](46-derive_call_directions.md) 和
[`AutoDeriveTaskDependencies`](47-auto_derive_task_dependencies.md)：正是这两个
pass 把合成 kernel 的参数方向转换成排序 `compute -> collective -> consume` 的
TensorMap 依赖边。放在它们之后改写，会让该 collective task 失去顺序约束。

它同样运行在 [`MaterializeDistTensorCtx`](52-materialize_dist_tensor_ctx.md)
之前 —— 后者会补上 kernel 需要的 `CommCtx` 实参（见下文 *ABI*）。

## 行为

对于一个 CHIP orchestration 函数体：

```python
@pl.function(type=pl.FunctionType.Orchestration)
def chip_pipeline(self, inp, out, stage, data, signal, counts, recv):
    stage, counts = self.stage_step(inp, stage, counts)
    data = pld.tensor.all_to_all_v(stage, data, signal, counts, recv, core_num=1)
    return self.consume_step(data, recv, out)
```

该集合通信被改写为：

```python
data = self.__builtin_all_to_all_v__fp32(stage, data, signal, counts, recv)
# INT8（RFC #2521 A1 规范负载）合成为 __builtin_all_to_all_v__int8，
# builtin_template_vars = "dtype_cpp=int8_t"
```

其中 `__builtin_all_to_all_v__{fp32,int8}` 是新增到 program 中的合成
`FunctionType.AIV` 函数：

| 方面 | 取值 |
| ---- | ---- |
| 参数 | `input, target, signal, send_counts, recv_counts` —— 规范化类型，**不是**调用点的类型：`input` / `recv_counts` 声明为普通 `Tensor`；`target`、`signal`、`send_counts` 为 `DistributedTensor` |
| 方向 | `In, InOut, InOut, In, InOut` |
| 函数体 | `return target` —— 单条 `ReturnStmt`，永不参与代码生成 |
| Attrs | `builtin_template_dir`、`builtin_template_vars`、`require_contiguous_operands` |

每个 variant 只合成一个函数，被该 variant 的所有调用点共享。

### 为什么函数体是 `ReturnStmt` 而不是空声明

该 kernel 的实现是手写 builtin 源码，函数体不会被代码生成 —— 对 backend 来说空
声明就够了。之所以写成返回 `target` 参数的真实 `ReturnStmt`，是因为后续仍会读取
该函数的 pass 需要它：`ReturnParamsExplicit` 成立，且 `MaterializeDistTensorCtx`
能把返回的 `DistributedTensor` 解析回它写入的那个参数。返回 `target` 也与 public
op 的 window-as-result 语义一致，使调用点保持为一次普通 rebind。

## Kernel 源码：一份实现，两条通路

合成函数并不像
[external kernel](../language/04-external-kernels.md) 那样直接给出 `.cpp` 路径，
而是指向 builtin **模板包** —— 与 `builtin.tensor.all_to_all_v` 通过
`set_template_dir` 声明的是同一个 —— 外加渲染所需的替换项：

```text
builtin_template_dir  = ":pypto.runtime.builtins.collectives.all_to_all_v"
builtin_template_vars = "dtype_cpp=float"
```

PTO backend 把 `templates/kernel.cpp.in` 渲染到 chip 子构建的
`kernels/aiv/<name>.cpp`，并写入生成的 `kernel_config.py` —— 与 PyPTO 生成的
kernel 走同一条路径，区别只在于文本来自模板而非 ptoas。

`dtype_cpp` 是两条通路**唯一**的替换项，且取值相同，因此两边渲染出的 kernel
**逐字节相同**。ST 通过 diff 两条通路的渲染产物端到端断言这一点。

## ABI

两条通路以**相同**的实参布局到达 kernel：

| 槽位 | HOST 通路 | CHIP 通路（本 pass） |
| ---- | --------- | -------------------- |
| `args[0..4]` | `input, target, signal, send_counts, recv_counts` | 相同 |
| `args[5]` | `CommContext*` | `CommContext*` |
| `args[6..7]` | signal 和 send-count context | 相同 |

两条通路都不传 rank 数标量。kernel 读 `CommContext::rankNum`，它与 HOST dispatch
过去传的 `domain_size` 是同一个数：`comm_derive_context` **按 comm domain** 建
context，其 `rankNum` 就是该域的 rank 数。去掉这个标量的代价是入口处多一次 GM
读取，换来的是一份共享源码。这在 CHIP 通路上也是唯一可行方案 —— 它根本无法计算
rank 数，`pld.system.nranks` 只有 InCore 代码生成，没有 orchestration 代码生成。

`MaterializeDistTensorCtx` 按规范化参数顺序追加每个远端操作数的 context：
target、signal、send counts。本地 input 和 receive counts 在合成签名中始终为
普通 `Tensor`，即使调用者传入窗口 view。每个 context 解析到该操作数自己的域；
不要求同域，参与设备和 rank 顺序的兼容性由调用者保证。

Indexed all-to-all-v 在 `args[5]` 增加本地 index tensor，将三个 context 移至
`args[6..8]`。target 使用 URMA，signal 和 send counts 使用 MTE。

## 启动宽度

`core_num` 是请求的 block 上限 `L`。在本通路上它必须是编译期常量，因此 pass 自己折叠出准入的
block 数 `B = CalAllToAllVBlocks(P, L)`：当 `L < P` 时取 `L`，否则取不超过 `L` 的最大 `P` 的倍数。
`P` 是 `[NR, S]` signal 的静态 `NR`。

- `B == 1` 时 kernel 调用保持不变。
- `B > 1` 时在 kernel 调用上附加 `core_num=B` 与 `sync_start=True`。感知 block 的 barrier 要求所有
  准入的 block 同时驻留，而 runtime 会拒绝超出容量的 sync-start 启动，而不是只启动一部分。

启动规格放在调用的 attrs 上，而 attrs 不是 kernel 参数，所以上文的五操作数 ABI 不变。被校验的是 `B`
而不是 `L`：`signal` 必须满足 `S >= B`，因此 `L=10, P=8`（实际启动 8 个 block）接受 `S=8`。HOST
通路在其生成的 entry 中计算同一个 `B`；两处实现都对照 RFC #2521 的算例表固定。`NR` 必须等于运行期
通信域大小，因为 kernel 从 `CommContext` 读取 rank 数。

## 约束与诊断

| 条件 | 诊断 |
| ---- | ---- |
| `core_num` 不是编译期常量 | 拒绝 —— 本通路暂不支持运行期 `core_num` |
| `core_num` 不在 `[1, INT32_MAX]` 内 | 拒绝 —— 与 HOST 通路 entry 在运行期强制的上界相同 |
| `signal` 步长 `S < B` | 拒绝 —— `B` 是折叠后的 block 数（见 *启动宽度*），不是原始 `core_num` |
| `dtype != FP32 && dtype != INT8` | 拒绝 —— 支持 FP32 与 INT8（与 HOST 通路的 allowlist 一致） |
| 集合通信残留在非 HOST 的 orchestration 函数体中 | 被本 pass 自身的后置条件检查拒绝 |

残留检查覆盖除 HOST orchestrator 之外的每个 orchestration 函数体（HOST 交由自己的
通路处理，在五个 pass 之后）。InCore 函数体不检查：它们归 composite 通路
（[`LowerCompositeOps`](15-lower_composite_ops.md)）所有，且该 pass 早在 26 个
pass 之前就已运行，在这里重复报告会指向错误的 pass。

## 本 pass 不做的事

- 不创建 `CommDomain`，也不分配任何集合通信 staging buffer。L3 仍负责创建域、交换
  window 地址并绑定 window；L2 只消费本地视图和已经建好的 context。
- 不做按设备扇出。`device=` 派发仍留在 HOST orchestrator，每张卡一次
  `chip_pipeline`。
- 不产生嵌套的 L2 -> L2 dispatch。该集合通信是调用方 pipeline 的一个 AIV task，
  不是另一个 chip callable。

## 当前限制

- **运行期 `core_num`**。仅接受编译期 `core_num`。运行期值需要在生成的 orchestration 中计算 `B`，并做
  运行期 `S >= B` 检查，而 orchestration 目前没有抛出该致命错误的途径。HOST 通路因其 entry 运行在
  host 上而接受运行期值。
- **除连续性外，操作数校验是静态的，且别名只覆盖了一部分**。`pld.tensor.all_to_all_v` 的类型
  推导只拒绝仅凭操作数类型即可证明的违规：非 ND 布局、与紧凑步长不符的 stride
  向量、比 shape 更窄的 `valid_shape`，以及 `input` 与 `target` 是**同一个表达式**。
  它**不会**拒绝同一块 allocation 上的两个不同 `pld.window()` 视图 —— 类型推导在
  构造 Call 时就已运行，而 `DistributedTensorType::window_buffer_` 要到
  [`MaterializeCommDomainScopes`](50-materialize_comm_domain_scopes.md)（pass 46）
  才被绑定。整块 allocation 层面的互不相同是 **HOST 通路**的保证：
  `LowerHostTensorCollectives` 能在同一个 `host_orch` 函数体内把每个操作数溯源回
  其 `WindowBuffer`，并对五个操作数运行 `CheckPairwiseDistinctWindows`。本通路
  —— 与 InCore composite 通路一样 —— 看到的操作数是外层 pipeline 的参数，没有这样
  的溯源能力，因此在这里「窗口两两互不相同」是调用方的义务。**连续性在本 pass 中于编译期判定，
  并保留运行期兜底**。类型推导看不到切片的步长（切片类型不携带 stride），而运行期判定是按
  rank 独立做出的：若只抑制某一个 rank 的提交而其对端仍然启动，对端会阻塞在 kernel 的
  `ExchangeBarrier` 上（表现为挂死，而非拒绝）。因此本 pass 沿着所有可能到达的值（赋值、
  循环携带值以及 `if`/`for`/`while` 的结果；循环结果还会追溯其初值，因为零次迭代的循环
  返回的就是初值）追溯调用中按原样书写的每个操作数，并采用
  **白名单**：参数、窗口、`tensor.create` 以及 `pl.submit` 交由运行期分配的 Out 是行主序的根；
  `reshape`、`view`、`set_validshape`、`assemble`（其目标）、collective 的结果（其目标）、
  kernel 或 `pl.submit` 的结果或其元组元素（其返回的实参）以及仅收窄静态形状 tensor 首维的切片
  会被继续追溯到其来源；其余一律拒绝，包括 `tensor.transpose` 及其他任何切片；链上任何非 ND
  布局、或显式 stride 不等于紧凑 stride 的值也会被拒绝（上述各步都假定来源是紧凑行主序的，而切片
  会丢弃其父 tensor 的 stride：DN 或带填充 stride 的 tensor 的首维切片并不连续，而 DN tensor 的
  ND `view` 会降级为运行期转置）。一些安全的程序也会被拒绝 ——
  `tensor.write` 的结果、`[1, W]` tensor 的 `[1, N]` 切片、偏移不是字面元组的切片、被调函数
  ReturnStmt 未直接写为参数的结果 —— 因为漏掉一次接受是安全的，错误的接受则不是。合成的
  kernel 仍带有 `require_contiguous_operands`，orchestration codegen 会在提交之前为每个
  tensor 操作数生成一条 `is_contiguous` 检查，并在第一个违规处锁存
  `SIMPLER_ERROR_INVALID_ARGS`，作为调用方提供的操作数的兜底：参数打包会拒绝非连续的 host
  torch tensor，但预先构建的运行期 `Tensor` 只在这里检查。另一项需要运行期形式的检查
  （signal stride `>= B`）仍由 lowering 阶段静态校验：`S` 与折叠出的 `B` 都是编译期常量。
- **rank 数**。kernel 读取 `CommContext::rankNum`，因此当显式设备子集小于 context
  的 rank 数时，其行为与传入 comm domain `domain_size` 的 HOST 通路不同。
- **域兼容性由调用者保证**。远端操作数使用各自的 context，因此支持不同域；
  参与设备和 rank 顺序必须与 collective 的 target 域兼容。

## 测试

- `tests/ut/ir/transforms/test_lower_l2_tensor_collectives.py` —— 改写后的形态、
  合成签名与方向、模板 attrs、variant 共享、InCore 透传、启动规格附加与 block 数折叠、
  容量检查、运行期 `core_num` 拒绝、INT8 variant。
- `tests/ut/codegen/distributed/test_builtin_collective_kernel_source.py` ——
  HOST 与 CHIP 通路对 FP32 / INT8 渲染逐字节相同的 kernel。
- `tests/ut/ir/transforms/test_lower_composite_ops.py` —— composite 通路把
  CHIP orchestration 中的集合通信交给本 pass，并在 InCore 函数体中拒绝
  `core_num != 1`。
- `tests/ut/ir/transforms/test_lower_host_tensor_collectives.py` —— 把窗口别名
  的拒绝固定为 **HOST 通路**的保证（`CheckPairwiseDistinctWindows`），这正是上文
  *当前限制* 一节所对照的另一面。
- `tests/st/distributed/collectives/test_l2_tensor_all_to_all_v.py` —— P=2/4 的
  硬件正确性；InCore 与 HOST 通路同样会跑的 0 / 1 / 容量 / 超容量 / 负数计数矩阵，
  它把三条通路约束到同一份链路 golden；外加三条结构断言：不产生 builtin chip
  dispatch、builtin kernel 被渲染进 pipeline 自己的子构建、HOST 通路渲染出逐字节
  相同的 kernel 源码。

## A5 URMA 索引交换

`pld.tensor.indexed_all_to_all_v(input, target, signal, send_counts,
recv_counts, index, core_num=L)` 使用托管 L2 任务流程及独立的内核模板。
六个 Tensor 操作数之后，第一个 CommContext 位于槽位 6。input、index 和
recv_counts 可以是本地 Tensor；send_counts 必须是窗口。
包括 `index` 在内的全部六个操作数，与普通算子一样接受连续性检查与兜底（见[当前限制](#当前限制)）。
`L` 为编译期正整数，表示精确启动核数。全部 AIV 在远端提交与 CQ 等待之间
分担本地搬运；超过 P 的 AIV 只参与本地搬运。仅支持 A5 URMA PER_PEER 传输，
每个目标队列只有一个写入者。调用者必须为共享队列的交换建立顺序；即使
payload 窗口不同，复用同一个 InOut signal 也能建立此依赖。
不支持 HOST 和 InCore 调用。参见[用户约定](../../user/distributed/20-all_to_all.md#a5-urma-索引-alltoallv)。

## CCU allgather

`engine="ccu"` 将 Ascend950 CHIP BF16 allgather 降低为 AIV wrapper 和
`FunctionType.CCU` native builtin 声明。支持输入窗口 `[1, N]`、目标 `[NR, N]`
和 INT32 signal `[NR, 1]`，要求完整连续 ND 视图、静态正 shape、2–64 rank，
以及 CHIP 中的直线调用序列。input 和 target 必须不同。

共享 AIV wrapper 的参数为 `(input, target, signal, order)`，三个用户窗口保留
DistributedTensor 类型。函数体以有副作用的语句调用真实 CCU callee，并返回 target 别名。
native 声明通过 `builtin_template_dir/vars` 指定实现，源码按实现和特化去重，与 wrapper
shape 无关。codegen 从 callee 解析 callable 内的定义编号，HOST 的 `ccu_id` 与设备请求
使用同一个编号。不再有调用点编号、slot 形参或 `ccu_slot` attr。

每个 CHIP 共享一个本地 InOut order Tensor，保守串行化 CCU 请求。signal 保留
InOut 调度效果，即使 native allgather 不打包它。方向和依赖先于域物化推导。
`MaterializeDistTensorCtx` 为每个 DistributedTensor 追加 ctx，包括 wrapper 的
三个窗口。编译器追加的 ctx 不进入 native 算法参数包，wrapper 总共有七个参数。

### 域内注册

`MaterializeCommDomainScopes` 保持现有窗口/device 成员推导规则。临时来源分析
沿 SSA 别名、Tensor 视图和函数返回参数别名，在每个 HOST dispatch 解析窗口。
同一次请求的所有窗口（包括 signal）必须同域；一个 CHIP 的不同请求可以使用不同域。

每个被使用的 `(device, domain, definition)` 组合，在代表 allocation 之后、
所属域内插入 HOST 内部操作：

```text
reg0 = pld.system.register_ccu(buffer, callee="ccu_definition", chip_callee="chip") {device=0}
reg1 = pld.system.register_ccu(buffer, callee="ccu_definition", chip_callee="chip") {device=1}
for rank in range(2):
    window = pld.tensor.window(buffer, ...)
    if rank == 0:
        reg = reg0
    else:
        reg = reg1
    chip(window, ..., device=rank, ccu_bindings=[reg])
```

`device` 必填，每次注册返回该 worker 的本地 handle。所有可能首次创建的注册必须先于
本轮第一次任务提交，即使 HOST 有多个 rank 提交循环。窗口创建与 rank 循环保留，
仅在提交处按 rank 选择对应 handle。若 allocation 在首次提交之后才可用，则拒绝新增
CCU 注册。每个 CHIP 调用生成独立的注册 SSA 值，即使调用同一个 callable；仅合并单次
提交内的重复需求，底层注册缓存交给 simpler。handle 不得逃出所属域。

注册具有资源/失败效果，DCE 保留它。handle 是 dispatch attr 中的 SSA 引用，不是 CHIP
算法参数。每个域检查静态参与覆盖与 collective 协议顺序/次数，协议兼容的各 rank 本地
定义符号不必相同。支持最多 64 个 rank 的正次数静态循环及可静态判定的 rank 选择分支；
其他条件/重复提交、动态 rank 选择、嵌套设备调用或控制流内 allocation 仍不支持。

HOST Python 生成 `domain.get_or_register_ccu(callables[name], ccu_id=N, worker=rank)`，
并向同一 rank 的 `_submit_chip` 传递 `ccu_bindings=[local_registration, ...]`。
序列是去重后的启动需求集合，下标和顺序没有身份含义。IR 的 `device` 映射为 runtime `worker`。
register 通过 `chip_callee` 显式指定消费它的 callable。CCU 编号与 AIV `func_id` 一样，
在每个 callable 内按首次使用顺序分配。CHIP codegen 只生成一份注册表，打包与 wrapper
常量使用该表，HOST codegen 直接读取同一结果。相同定义在不同 callable 中可以编号不同，
各自注册。不同提交可在同一域/编号处使用不同定义，单次提交不能在那里绑定冲突注册。
编译器不跨 CHIP 调用共享注册。

### Native 源码契约与验证边界

此次迁移只支持 codegen。产物标记 `CCU_DOMAIN_ABI_VERSION = 2` 和
`CCU_RUNTIME_SUPPORTED = False`。runtime builder 在编译/注册 native module 前
拒绝执行；`_submit_chip` 也拒绝带绑定的提交，等待 PyPTO 集成与端到端验证。
当前 pin 的 simpler runtime 已包含 PR #3093 的域注册与协调 prepare/激活接口，
包括 `ChipCallable.build(..., ccu_modules=...)`。

设备调用采用已核对的 simpler PR #3093 接口：
`submit_ccu_mission(async_ctx, comm_ctx, ccu_id, call_args)`。ctx 选择域，定义编号选择该域
唯一的本地 mission；相同组合的重复请求复用 mission，并通过任务完成依赖串行化。
不再使用独立的 CHIP execution accessor。当前 runtime pin 尚待联调验证，保留 wrapper/native
ABI guard 和运行时执行保护。

请求保留 `9 + 3 * NR` 个 uint64 word，64 字节对齐：input 地址/token、字节数、
来源行字节偏移、segment 数、四个 GoSize 值，再为每个目标保存地址/token 和
域逻辑 rank。peer 使用普通 scalar，删除 `PeerRank` 转换；注册 channel 按域逻辑
rank 索引，包括自 rank 的 LocalCopy 选择。

HBM 算法保留 32 KiB chunk、16 lane、4 GiB 分段、独立 main/remainder LoopGroup、
ready/completion 握手和常驻 trigger 循环。所有请求完成且结果可见后，simpler 必须
停止并确认无剩余访存，再释放本轮资源。不增加静态请求次数或用户 EXIT 请求。

peer 握手统一使用 notify index `0`，ready 使用 mask `1`，completion 使用 mask `2`。
共享同一域 channel 的所有请求必须串行执行，才能安全复用该 index。
callable 局部的 `ccu_id` 仅用于选择 mission，不用于选择 peer notify index，
不同 rank 上的值可以不同。

CPU 测试覆盖完整 lowering/codegen、重叠域、注册复用、定义编号一致性、普通 ctx 转发、
源码打包，以及通过模拟 primitive 执行生成的 wrapper/native 源码。
这些测试不验证硬件调度、资源生命周期或 HCOMM 指令生成。
