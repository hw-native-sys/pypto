# BackendHandler：原则化的后端分发

> 跟踪 issue：[#948](https://github.com/hw-native-sys/pypto/issues/948)

## 背景

早期版本的 PyPTO 在 pass 与 codegen 中直接根据 `backend::BackendType` 做分支：

```cpp
if (backend::GetBackendType() != backend::BackendType::Ascend910) { ... }
```

这种写法导致每新增一种后端，都要去整个代码库里翻这些散落的 `if`，并且为每一处
都补一条 case。`BackendHandler` 把所有这类分支集中到一个虚接口里，使新增后端
变成一次自包含的局部修改。

## 设计

`BackendHandler`（`include/pypto/backend/common/backend_handler.h`）是一个抽象
接口，它把后端之间所有"行为差异"显式命名出来。每个 `Backend` 子类各自持有一个
`BackendHandler` 子类的单例，并通过新引入的纯虚函数 `Backend::GetHandler()`
对外暴露。

```text
                       ┌────────────────────────────┐
        Pass / Codegen │  PassContext::Current()    │
                       │     ->GetBackendHandler()  │
                       └─────────────┬──────────────┘
                                     │
                                     ▼
                       ┌────────────────────────────┐
                       │  BackendConfig::GetBackend │
                       │     ->GetHandler()         │
                       └─────────────┬──────────────┘
                                     │
                                     ▼
                ┌──────────────────────────────────────┐
                │  Backend910B / Backend950 / ...      │
                │     -> Ascend910BHandler::Instance() │
                │     -> Ascend950Handler::Instance()  │
                └──────────────────────────────────────┘
```

`PassContext` 上的便捷访问点满足 `pass-context-config` 规则的要求：pass 通过
当前 `PassContext` 查询后端行为，而不是直接读全局状态。

## 接口

| 方法 | 用途 | Ascend910 | Ascend950 |
| ---- | ---- | --------- | --------- |
| `GetPtoTargetArch()` | `module attributes {pto.target_arch = "..."}` | `"a2a3"` | `"a5"` |
| `GetLaunchSpecCoreCountMethod()` | `launch_spec` 上设置逻辑 block 数量的运行时 API 名 | `"set_block_num"` | `"set_block_num"` |
| `GetDefaultSimPlatform()` | 默认仿真平台名 | `"a2a3sim"` | `"a5sim"` |
| `GetExtraPtoasFlags()` | ptoas 额外参数 | `[]` | `["--pto-arch", "a5"]` |
| `HonorsHighPrecisionAlgorithm()` | PTO-ISA 是否会根据 `high_precision=True` 生成的 `precisionType` 属性选择不同算法；为 false 时两个 PTO 发射路径（Tile 算子回调与 Buffer IR）都会告警该请求被丢弃 | `false` | `true` |
| `RequiresGMPipeBuffer()` | `ExpandMixedKernel` 是否注入 GM 槽位缓冲 | `true` | `false` |
| `SupportsLocalPipeDepth()` | 本地消费者深度是否可独立于跨核传输环深度设置 | `true` | `false` |
| `RequiresSplitLoadTpopWorkaround()` | MemoryReuse 是否做 load + tpop_from_aic 原地复用危害规避 | `true` | `false` |
| `RequiresLevel3TmpScratch()` | InitMemRef 是否物化 level-3 显式 tmp；codegen static-view 桥接 | `true` | `false` |
| `RequiresVtoCFractalAdapt()` | AIV 端 V→C tpush 是否需要 fractal 适配 `tile.move` | `false` | `true` |
| `RequiresRuntimeSubblockBridge()` | 拆分 AIV 包装器是否从 runtime 上下文取 subblock id | `true` | `false` |
| `RequiresNoSplitDualAivDispatch()` | `no_split` 混合 kernel 是否仍需在两个 AIV lane 上同时下发 | `true` | `false` |
| `SplitsCubeToVectorTransportInHardware()` | 切分的 Cube-to-Vector push 是否由硬件把传输的 extent 各分一半给两个 AIV lane（此时该 extent 即 lane 划分） | `false` | `true` |
| `GetL0cMAlignment(dtype)` | L0C 切分、dbC 容量与分配共同使用的物理 M 行对齐 | INT32 为 32；其余为 16 | 16 |
| `BuildCrossCoreTransferView(dest, view)` | 跨核传输边界处的 tile 视图 | Mat/Left/Right 转 NZ；Vec 保持原样 | Mat/Left/Right 转 NZ（a5 硬件要求边界为 fractal）；Vec 保持原样 |

## 内存容量来源

`BackendHandler::GetCapacityBytes(memory_space)` 调用 handler 所属后端的
`Backend::GetMemSize()`。SoC 是容量的唯一来源；handler 不重复保存数值，也不乘以核数
或 die 数。查询返回一个匹配内存组件的字节容量，类型为 `uint64_t`；SoC 未建模该空间
时返回 `0`。若多个核都有该空间，返回第一个匹配组件的容量。未建模不表示硬件没有该内存。

L0A、L0B、L0C、Bias、L1 分别使用 `MemorySpace::Left`、`Right`、`Acc`、`Bias`、`Mat`。
同一个接口也能查询 `Vec` 和 scale 内存。Python 接口为
`handler.get_capacity_bytes(memory_space)`。两种语言原有的五个独立容量方法均由该统一查询
替代。即使尚未配置全局目标，或者当前配置了另一种后端，显式查询该后端的 handler 仍能
返回正确容量。对齐规则、最小 tile 维度和成本模型参数仍与容量分别维护。

## 新增后端流程

1. 在 `src/backend/<arch>/backend_<arch>.cpp` 中实现 `Backend<Arch>` 子类，
   并把源文件加入 `CMakeLists.txt`。
2. 在 `src/backend/<arch>/backend_<arch>_handler.cpp` 中实现
   `Backend<Arch>Handler` 子类。以所属后端引用初始化基类，该后端的生命周期必须长于
   handler。内置 handler 使用后端单例；后端构造期间不能反过来请求 handler，避免单例
   初始化重入。
3. 让 `Backend<Arch>::GetHandler()` 返回你的 handler 单例。
4. 在 `src/backend/common/backend.cpp` 中给 `GetBackendInstance` 和
   `BackendTypeToString` 各加一行，并在 `BackendType` 枚举中添加新值。
   这是整个仓库里唯一按枚举展开的地方。

任何 pass / codegen 文件都不需要修改。可以用
`tests/ut/backend/test_backend_handler.py` 配合 `tests/ut/ir/transforms/` 与
`tests/ut/codegen/` 里的回归测试做验证。

## Python 访问方式

handler 也通过 Python 绑定暴露：

```python
from pypto.pypto_core import backend as _backend_core

# 全局后端类型已配置时：
handler = _backend_core.get_handler()

# 调用方已知道目标后端时：
handler = _backend_core.get_backend_instance(BackendType.Ascend950).get_handler()

handler.get_pto_target_arch()              # "a2a3" 或 "a5"
handler.requires_runtime_subblock_bridge()  # bool
handler.supports_local_pipe_depth()        # A2/A3 为 True，A5 为 False
handler.get_extra_ptoas_flags()            # list[str]
```

运行时相关的 Python 模块（`pypto.runtime.runner`、`pypto.ir.compiled_program`、
`pypto.backend.pto_backend`）现在通过这些访问器来获取后端行为，不再直接判断
`BackendType`，与 C++ 改造保持一致。

## Select 算子规则

`tile.sel` 与 `tile.sels` 在 `BackendOpInfo::scratch_plan_func` 注册 scratch
规划函数；TSELS 还在 `validate_func` 注册原有 dtype 与 MemRef 别名检查。
`PTOCodegen` 在输出 Tile PTO 前调用验证函数。两后端共享 `pto_ops_select.cpp`，
具体规则在注册阶段绑定。原有 `GetTselScratchSpec`、`GetTselsScratchSpec`、
`SupportsTselDataType`、`SupportsTselsDataType` 四个 handler 虚接口已移除。

Pass 经 `PassContext::GetBackend()` 查询算子规则。该入口保留现有全局目标解析语义，
不引入 context 独立持有目标的机制。未配置后端的独立 select lowering 仍使用历史
A5 scratch 几何，并保留原有不按后端限制 dtype 的行为。

```python
from pypto import DataType, backend

target = backend.get_backend_instance(backend.BackendType.Ascend910)
plan = target.plan_op_scratch("tile.sels", DataType.FP16, source_cols=128)
# Required: FP16 [1, 128], one physical source row.
```

`Required` 携带以元素为单位的只读 `TileScratchSpec`；`NotRequired` 明确表示无需
scratch；`Unknown` 表示缺少查询所需信息；`Unsupported` 表示候选不支持。
已注册但没有规划函数的算子返回 `None`，未注册算子报错。查询不读取全局 target、
不创建 IR，也不分配内存。

合成规格与显式操作数保持区分：调用者提供的 scratch 原样传递，包括 A5 接受的其他
dtype 占位空间。本次迁移保留既有编译器检查，不增加 scratch 精确几何校验，不改变
PTOAS 校验。生命周期和复用决策仍由现有内存分配与 IR workspace/alias 元数据处理。
