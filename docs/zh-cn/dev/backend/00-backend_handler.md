# BackendHandler：原则化的后端分发

> 跟踪 issue：[#948](https://github.com/hw-native-sys/pypto/issues/948)

## 背景

早期版本的 PyPTO 在 pass 与 codegen 中直接根据 `backend::BackendType` 做分支：

```cpp
if (backend::GetBackendType() != backend::BackendType::Ascend910B) { ... }
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

| 方法 | 用途 | Ascend910B | Ascend950 | SuperscalarNPU |
| ---- | ---- | ---------- | --------- | -------------- |
| `GetPtoTargetArch()` | `module attributes {pto.target_arch = "..."}` | `"a2a3"` | `"a5"` | `"superscalar"`（占位，无 codegen） |
| `GetLaunchSpecCoreCountMethod()` | `launch_spec` 上设置核数的运行时 API 名 | `"set_block_num"` | `"set_core_num"` | `"set_core_num"` |
| `GetDefaultSimPlatform()` | 默认仿真平台名 | `"a2a3sim"` | `"a5sim"` | `"superscalarsim"` |
| `GetExtraPtoasFlags()` | ptoas 额外参数 | `[]` | `["--pto-arch", "a5"]` | `[]` |
| `RequiresGMPipeBuffer()` | `ExpandMixedKernel` 是否注入 GM 槽位缓冲 | `true` | `false` | `false` |
| `RequiresSplitLoadTpopWorkaround()` | `LegalizePtoBufferReuse` 是否做 split-load tpop 危害规避 | `true` | `false` | `false` |
| `RequiresVtoCFractalAdapt()` | AIV 端 V→C tpush 是否需要 fractal 适配 `tile.move` | `false` | `true` | `false` |
| `RequiresRuntimeSubblockBridge()` | 拆分 AIV 包装器是否从 runtime 上下文取 subblock id | `true` | `false` | `false` |
| `RequiresNoSplitDualAivDispatch()` | `no_split` 混合 kernel 是否仍需在两个 AIV lane 上同时下发 | `true` | `false` | `false` |
| `BuildCrossCoreTransferView(dest, view)` | 跨核传输边界处的 tile 视图 | Mat/Left/Right 转 NZ；Vec 保持原样 | Mat/Left/Right 转 NZ（a5 硬件要求边界为 fractal）；Vec 保持原样 | 不可达（单核，无跨核） |
| `GetDefaultOnChipMemorySpace()` | `InferTileMemorySpace` 中无约束 tile 的默认片上空间 | `Vec` | `Vec` | `TREG` |

### SuperscalarNPU 后端

SuperscalarNPU 是寄存器文件后端：内存空间只有 DDR 与 `TREG`——一个由 256 个固定
4 KB 块（共 1 MB）组成、按块编号寻址的寄存器文件。它没有 cube/vector 核，因此
`SuperscalarNPU` pass 策略（`OptimizationStrategy.SuperscalarNPU`）将 tensor 算子
下降为 tile 算子后，在寄存器重命名式 TREG 分配处停止——省略所有 Ascend 的
cube/vector/跨核 pass，且暂未实现 codegen。

两个机制让共享 pass 无需按 `BackendType` 分支：

- `GetDefaultOnChipMemorySpace()` 让 `InferTileMemorySpace` 为各后端分配各自的
  片上存储（Ascend 为 `Vec`，此处为 `TREG`）。通用的 `Vec` 算子输入约束同样会被
  实现为该默认空间，因此不会插入多余的 `tile.move`。
- `Backend::CreateMemoryAllocatorPolicy()` 返回
  `SuperscalarNPURegisterAllocatorPolicy`，它在 MemRef 中存储**块编号**
  （通过 `MemoryAllocatorPolicy::AddressUnitBytes` = 4096），为大于 4 KB 的 tile
  预留连续块，并在同时存活的块超过 256（`MaxAddressUnits`）时抛出面向用户的
  错误——即寄存器压力。基于生命周期的合并仍由共享的 `MemoryReuse` pass 完成。

## 新增后端流程

1. 在 `src/backend/<arch>/backend_<arch>.cpp` 中实现 `Backend<Arch>` 子类，
   并把源文件加入 `CMakeLists.txt`。
2. 在 `src/backend/<arch>/backend_<arch>_handler.cpp` 中实现
   `Backend<Arch>Handler` 子类。
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
handler.get_extra_ptoas_flags()            # list[str]
```

运行时相关的 Python 模块（`pypto.runtime.runner`、`pypto.ir.compiled_program`、
`pypto.backend.pto_backend`）现在通过这些访问器来获取后端行为，不再直接判断
`BackendType`，与 C++ 改造保持一致。
