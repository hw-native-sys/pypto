<!-- markdownlint-disable MD013 MD060 MD033 -->
# PTOAS Op 状态矩阵

行 = **最新 PTOAS 提供的公共及兼容 op**。接口基线为 Little-oil/PTOAS `main`
`d852dd2dba3e5bf7a69ce8324eb88afc336e8a33`：manual 公共接口 189 个，加当前
`PTOOps.td` 仍保留的 source-only 兼容/tile 接口 15 个，共 204 个。列状态按
**PyPTO 当前源码**核实（最后更新 2026-08-14）。后续每加或修改一个 op，只更新本表对应行。

本表包含公共/兼容接口中 PyPTO 级别为 `internal` 的 op；另有 `PTOOps.td` 中 32 个仅供
lowering/compiler plumbing 使用的额外内部 op 未纳入，也不列 VPTO、VMI、SIMT 等其他 dialect。

## 完成判定原则

**一个 op 是否完成，以最终是否有同名 ST 为准。**

- `pypto 前端✅ + ST❌`：未完成；前端或 codegen 已存在，但没有真机同名覆盖。
- `ST✅`：活跃 `tests/st/` 测试最终生成并执行同名 `pto.*` op。
- `ST✅` 只表示已经具备真机同名执行证据；尚未真机验证的其他架构继续在备注中记录。
- `ST—`：内部原语、编译期辅助或已替代接口，不适合独立 ST；由 codegen/集成测试覆盖。
- 高层测试若最终生成其他 PTO op，不算同名覆盖。例如 collective 被拆成
  `tput/tget/tnotify/twait`，不能计为 `tbroadcast/tgather/tscatter/treduce` 的 ST。

## 内存规划视图编译看护（Planner View Compilation Guard）

`system-tests-a5sim` CI 作业使用固定版本的 PTOAS，在安装设备运行时之前运行
`fillpad_inplace` 的内存规划和视图回归测试：

```bash
python -m pytest tests/ut/codegen/ --require-ptoas -m ptoas_compile -v
```

当 PTOAS 缺失、无法运行或版本低于支持范围时，`--require-ptoas` 会在测试初始化阶段失败，
避免因编译器缺失而静默跳过原生编译。测试覆盖两个后端的 PTOAS 内存规划路径，以及
PYPTO/DSA_RP 对照路径。此看护检查 MLIR 和生成 C++ 的编译过程；不能替代设备数值测试，
也不代表已穷举验证整个内存规划器。

输入由本地分配支撑时，填充结果使用动态有效范围的 reshape 别名，并显式初始化为完整
tile 范围。这保留了后续 `set_validshape` 更新，包括运行时边界以及对原填充句柄的使用，
也覆盖为 `if/else` 结果在函数入口声明的分配。
切片及其他不可修改有效范围的视图输入仍使用静态有效范围，原有的有效范围更新限制保持不变。

公共 `EnsureTileViewSsa` 辅助函数负责创建类型化别名并显式初始化运行时有效范围。
调用方以物理 tile 坐标传入初始范围；选择完整范围仍属于 fillpad 的算子语义。
需要初始化的可修改别名始终获得独立句柄，即使其类型与源相同，因为当前运行时有效范围
可能不同。现有静态视图调用方保留原有行为。Fillpad 仅在 PTOAS 内存规划路径使用此辅助函数；
PYPTO 和 DSA_RP 保留通过显式地址表示别名的生成方式。

扩展看护时，在 `tests/ut/codegen/` 下的回归测试上添加 `@pytest.mark.ptoas_compile`。
测试必须断言相关的生成代码语义，并实际调用 PTOAS 编译；标记只负责选择测试。
CI 会发现该目录下所有带此标记的测试，因此新增算子或测试文件无须修改工作流。
CI 中应保留 `--require-ptoas`，确保编译器不可用时失败。

安装运行时之后，同一 CI 作业还会在两个模拟器上显式使用 PTOAS 内存规划，
运行现有的 `fillpad_inplace` 数值测试：

```bash
python -m pytest tests/st/runtime/ops/test_fillpad_inplace.py --memory-planner=ptoas --platform=a2a3sim,a5sim --forked -v
```

六个基础用例将完整的 `[64, 64]` 输出与参考结果比较：前 48 行保留输入，
后 16 行分别填充零、正无穷或负无穷。另六个用例覆盖填充后静态或动态更新有效范围，
再通过原填充别名执行填充的场景。额外四个用例覆盖 `if/else` 两条路径合并后再填充并缩小有效范围。
另两个用例在缩小有效范围后重复相同填充模式，检查源与目标类型相同时仍保留独立的有效范围。
它们为编译看护补充模拟器数值覆盖，
并不代表编译测试中的所有视图组合都已验证设备执行正确性。

## Cast 临时空间规划

在 A2/A3 上，非饱和的 FP32 -> INT16 和 FP16 -> INT16/INT8/UINT8 转换
需要临时空间（scratch）。PyPTO 和 DSA-RP 在 `InitMemRef` 中为 PTOAS level 3
生成显式临时空间。使用 PTOAS planner 时，由 level 2 负责分配；codegen 提供
具有静态有效范围的输入、输出 reshape 视图，并保留单输入的 `pto.tcvt` 形式。
这些视图共享原有缓冲区，保留实际有效形状，包括部分有效的 tile。
`pto_ops_cast.cpp` 中通过 `f_plan_scratch` 注册的规则统一决定是否需要临时空间、
容量与类型。InitMemRef 和代码生成都通过
`Backend::PlanOpScratch("tile.cast", request)` 查询它。InitMemRef 只根据 `Required`
规格生成临时 tile；不完整的形状返回 `Unknown`，由调用方在原始算子位置报告错误，
不会被当成无需临时空间。
显式传入 `tmp` 时会跳过这条自动生成规则，并保留其原有类型和容量。
显式临时空间仍支持一维输入：代码生成会将形状及静态有效范围从 `[N]`
映射为 `[1, N]`。

A2/A3 窄化转换遵循分配与代码生成共享的安全规则，使用互不重叠的输入、输出存储，
详见 [MemoryReuse](passes/37-memory_reuse.md)。静态有效范围视图保留各自操作数
原有的存储，不会让转换的输入和输出共享同一缓冲区。

只有依赖临时空间的转换（或 A2/A3 上显式传入 tmp 的转换）要求静态有效范围。
原生饱和转换、其他原生转换及 A5 转换保留原有的动态有效范围行为。
A2/A3 上依赖临时空间的转换仍不支持运行时有效范围，会在调用 PTOAS 前给出诊断。

`tests/ut/codegen/test_ptoas_planner_views.py` 覆盖四组类型转换、两种饱和模式、
两种后端、三种 planner，以及完整和部分有效的 tile；安装 PTOAS 时还会实际编译。
CI 在 A2/A3 模拟器上运行 `tests/st/runtime/ops/test_cast_saturation.py`，
对比不同 planner 的精确数值结果，并验证部分写回保留有效矩形之外的输出元素。

## 图例

- **级别**：PyPTO 注册/生成层级（tile / tensor / tile+tensor / comm / internal）。
- **PTOAS接口**：✅ = 最新 PTOAS `main` 提供该 canonical op。
- **pypto-tile / -tensor 前端**：✅ = 已有对应公共前端；❌ = 未添加；`—` = 内部或 comm，不适用。
- **ST测试**：非 comm op 的同名 ST 状态；comm 见下一列。
- **distributed ST测试**：comm op 的同名分布式 ST 状态；非 comm 为 `—`。
- **备注**：只记录当前添加/覆盖事实和直接阻塞，不记录未来实施顺序。

| PTOAS op (pto.*) | pto-isa API | 级别 | PTOAS接口 | pypto-tile前端 | pypto-tensor前端 | ST测试 | distributed ST测试 | 备注 |
|---|---|---|:---:|:---:|:---:|:---:|:---:|---|
| **指针 / View（13）** |  |  |  |  |  |  |  |  |
| pto.ptrtoint | — | internal | ✅ | — | — | — | — | 编译期/分配辅助，不独立建 ST |
| pto.inttoptr | — | internal | ✅ | — | — | — | — | 编译期/分配辅助，不独立建 ST |
| pto.addptr | — | internal | ✅ | — | — | ✅ | — | compiler/system lowering 自动发射 |
| pto.castptr | — | internal | ✅ | — | — | — | — | 编译期/分配辅助，不独立建 ST |
| pto.make_tensor_view | — | tensor | ✅ | ❌ | ✅ | ✅ | — | 由 `tensor.view` 发射 |
| pto.get_tensor_view_dim | — | internal | ✅ | — | — | — | — | 编译期/分配辅助，不独立建 ST |
| pto.get_tensor_view_stride | — | internal | ✅ | — | — | — | — | 编译期/分配辅助，不独立建 ST |
| pto.partition_view | — | internal | ✅ | — | — | ✅ | — | compiler/system lowering 自动发射 |
| pto.alloc_tile | — | internal | ✅ | — | — | ✅ | — | compiler/system lowering 自动发射 |
| pto.alloc_multi_tile | — | internal | ✅ | — | — | — | — | 编译期/分配辅助，不独立建 ST |
| pto.multi_tile_get | — | internal | ✅ | — | — | — | — | 编译期/分配辅助，不独立建 ST |
| pto.subview | — | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.set_validshape | .SetValidShape | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| **Buffer-ID（2）** |  |  |  |  |  |  |  |  |
| pto.get_buf | get_buf | internal | ✅ | — | — | — | — | 编译期/分配辅助，不独立建 ST |
| pto.rls_buf | rls_buf | internal | ✅ | — | — | — | — | 编译期/分配辅助，不独立建 ST |
| **DMA 数据搬运（10）** |  |  |  |  |  |  |  |  |
| pto.tload | TLOAD | tile | ✅ | ✅ | ❌ | ✅ | — |  |
| pto.tprefetch | TPREFETCH | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING：缺完整前端/codegen/ST 链路 |
| pto.tprefetch_async | TPREFETCH_ASYNC | tile | ✅ | ❌ | ❌ | ✅ | — | 由 `pl.prefetch.*`（`make_context` / `async_prefetch` / `session` / `wait`）发射；非 tile/tensor 前端。PTOAS op 可用性不等于 runtime 提供 SDMA workspace；当前由 artifact 自动启用，仅 onboard a2a3 ST 覆盖，无 provider 的平台初始化失败而非 no-op |
| pto.make_prefetch_async_context | pto::PrefetchAsyncContext | internal | ✅ | — | — | — | — | 随 async prefetch 集成验证 |
| pto.get_prefetch_async_session | .session | internal | ✅ | — | — | — | — | 随 async prefetch 集成验证 |
| pto.tstore | TSTORE | tile | ✅ | ✅ | ❌ | ✅ | — |  |
| pto.load | direct pointer load | tensor | ✅ | ❌ | ✅ | ✅ | — | 由 `tensor.read` 发射；PTOAS 0.64 中替代 `pto.load_scalar` |
| pto.store | direct pointer store | tensor | ✅ | ❌ | ✅ | ✅ | — | 由 `tensor.write` 发射；PTOAS 0.64 中替代 `pto.store_scalar` |
| pto.tmov | TMOV / TMOV_FP | tile | ✅ | ✅ | ❌ | ✅ | — |  |
| pto.ttrans | TTRANS | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| **矩阵计算（12）** |  |  |  |  |  |  |  |  |
| pto.tmatmul | TMATMUL | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tmatmul.acc | TMATMUL_ACC | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tmatmul.bias | TMATMUL_BIAS | tile | ✅ | ✅ | ❌ | ✅ | — |  |
| pto.tmatmul.mx | TMATMUL_MX | tile | ✅ | ✅ | ✅ | ✅ | — | A5 MXFP8 host-prequant frontend+codegen+ST；FP4 输入仅支持左侧 FP4×右侧 FP8，且必须显式做 FP4→FP8 cast；原生 FP4×FP4 暂缓；见 [operators MX 约束](ir/05-operators.md#mx--ascend950pto-isa-约束) |
| pto.tmatmul.mx.acc | TMATMUL_MX (overload) | tile | ✅ | ✅ | ✅ | ✅ | — | A5 frontend+codegen+ST（`tile.matmul_mx_acc`） |
| pto.tmatmul.mx.bias | TMATMUL_MX (overload) | tile | ✅ | ✅ | ✅ | ❌ | — | NEW frontend+codegen（`tile.matmul_mx_bias`）；ST 待补 |
| pto.tgemv | TGEMV | tile | ✅ | ✅ | ❌ | ✅ | — | A2/A3 task-submit ST 已通过；A5 真机验证待补 |
| pto.tgemv.acc | TGEMV_ACC | tile | ✅ | ✅ | ❌ | ✅ | — | A2/A3 task-submit ST 已通过；A5 真机验证待补 |
| pto.tgemv.bias | TGEMV_BIAS | tile | ✅ | ✅ | ❌ | ✅ | — | A2/A3 task-submit ST 已通过；A5 真机验证待补 |
| pto.tgemv.mx | TGEMV_MX | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING：缺完整前端/codegen/ST 链路 |
| pto.tgemv.mx.acc | TGEMV_MX (overload) | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING：缺完整前端/codegen/ST 链路 |
| pto.tgemv.mx.bias | TGEMV_MX (overload) | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING：缺完整前端/codegen/ST 链路 |
| **向量算术与数学（42）** |  |  |  |  |  |  |  |  |
| pto.tadd | TADD | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tsub | TSUB | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tmul | TMUL | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tdiv | TDIV | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | A2/A3 真机已验证；A5 真机待验证 |
| pto.tmax | TMAX | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tmin | TMIN | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.trem | TREM | tile | ✅ | ✅ | ❌ | ✅ | — | A2/A3 FP32 及 ISA 定义域 `[-2^24, 2^24]` 内的 INT32 同名真机 ST 已通过，覆盖定义域边界、完整与尾部 valid shape；A5 真机待验证 |
| pto.tpartadd | TPARTADD | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tpartmax | TPARTMAX | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tpartmin | TPARTMIN | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tpartargmax | TPARTARGMAX | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING：缺完整前端/codegen/ST 链路 |
| pto.tpartargmin | TPARTARGMIN | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING：缺完整前端/codegen/ST 链路 |
| pto.tpartmul | TPARTMUL | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tprelu | TPRELU | tile | ✅ | ✅ | ❌ | ✅ | — | 已补齐规范 3 输入链路；A2/A3 真机已验证，A5 真机待验证 |
| pto.tadds | TADDS | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tsubs | TSUBS | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | A2/A3 真机已验证；A5 真机待验证 |
| pto.tmuls | TMULS | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.taxpy | TAXPY | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING：缺完整前端/codegen/ST 链路 |
| pto.tdivs | TDIVS | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tmaxs | TMAXS | tile | ✅ | ✅ | ❌ | ✅ | — |  |
| pto.tmins | TMINS | tile | ✅ | ✅ | ❌ | ✅ | — |  |
| pto.trems | TREMS | tile | ✅ | ✅ | ❌ | ✅ | — | A2/A3 FP32 及 `[-2^24, 2^24]` 内的 INT32 scalar 同名真机 ST 已通过，覆盖 INT32 正负边界、负值与尾部 valid shape；A5 真机待验证 |
| pto.taddc | TADD + TADD | tile | ✅ | ✅ | ❌ | ✅ | — | A2/A3 真机已验证，覆盖溢出、完整/尾部 valid shape 及 result/src0 复用与不复用路径；A5 真机待验证 |
| pto.tsubc | TSUB + TADD | tile | ✅ | ✅ | ❌ | ✅ | — | A2/A3 真机已验证，覆盖借位、完整/尾部 valid shape 及 result/src0 复用与不复用路径；A5 真机待验证 |
| pto.taddsc | TADDS + TADD | tile | ✅ | ✅ | ❌ | ✅ | — | A2/A3 真机的完整、列尾及行列组合尾部场景已通过，覆盖 result/src0 复用与不复用路径；PTO ISA v0.57 的全列行尾快速路径严格预期失败；A5 真机待验证 |
| pto.tsubsc | TSUBS + TADD | tile | ✅ | ✅ | ❌ | ✅ | — | A2/A3 真机的完整、列尾及行列组合尾部场景已通过，覆盖 result/src0 复用与不复用路径；PTO ISA v0.57 的全列行尾快速路径严格预期失败；A5 真机待验证 |
| pto.tabs | TABS | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tneg | TNEG | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.texp | TEXP | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tlog | TLOG | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | A2/A3 真机已验证；A5 真机待验证 |
| pto.tsqrt | TSQRT | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.ttri | TTRI | tile | ✅ | ✅ | ❌ | ✅ | — | 前端 + 精确 codegen + 同名 ST；A2/A3 真机 10/10 通过；A5 真机待验证 |
| pto.trsqrt | TRSQRT | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.trecip | TRECIP | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.trelu | TRELU | tile | ✅ | ✅ | ❌ | ✅ | — |  |
| pto.tlrelu | TLRELU | tile | ✅ | ✅ | ❌ | ✅ | — |  |
| pto.taddrelu | VADDRELU | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING：缺完整前端/codegen/ST 链路 |
| pto.tfmod | TFMOD | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | A2/A3 FP32 tile-tile 同名真机 ST 已通过，覆盖完整与尾部 valid shape；A5 真机待验证 |
| pto.tfmods | TFMODS | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | A2/A3 FP32 scalar 同名真机 ST 已通过，覆盖负值与尾部 valid shape；A5 真机待验证 |
| pto.tpow | TPOW | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING：缺完整前端/codegen/ST 链路 |
| pto.tpows | TPOWS | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING：缺完整前端/codegen/ST 链路 |
| pto.trandom | TRANDOM | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | PTOAS source-only 兼容接口 |
| **归约（13）** |  |  |  |  |  |  |  |  |
| pto.trowsum | TROWSUM | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.trowmax | TROWMAX | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.trowargmax | TROWARGMAX | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.trowmin | TROWMIN | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | A2/A3 真机已验证；A5 真机待验证 |
| pto.trowargmin | TROWARGMIN | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.trowprod | TROWPROD | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.thistogram | THISTOGRAM | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING：缺完整前端/codegen/ST 链路 |
| pto.tcolsum | TCOLSUM | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tcolmax | TCOLMAX | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tcolargmax | TCOLARGMAX | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tcolmin | TCOLMIN | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tcolargmin | TCOLARGMIN | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tcolprod | TCOLPROD | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | PTOAS source-only 兼容接口 |
| **广播（17）** |  |  |  |  |  |  |  |  |
| pto.trowexpand | TROWEXPAND | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tcolexpand | TCOLEXPAND | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tcolexpandmul | TCOLEXPANDMUL | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tcolexpandadd | TCOLEXPANDADD | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tcolexpanddiv | TCOLEXPANDDIV | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tcolexpandsub | TCOLEXPANDSUB | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tcolexpandexpdif | TCOLEXPANDEXPDIF | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tcolexpandmax | TCOLEXPANDMAX | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tcolexpandmin | TCOLEXPANDMIN | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.trowexpandmul | TROWEXPANDMUL | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.trowexpanddiv | TROWEXPANDDIV | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.trowexpandsub | TROWEXPANDSUB | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.trowexpandadd | TROWEXPANDADD | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | A2/A3 真机已验证；A5 真机待验证 |
| pto.trowexpandexpdif | TROWEXPANDEXPDIF | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.trowexpandmax | TROWEXPANDMAX | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.trowexpandmin | TROWEXPANDMIN | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.texpands | TEXPANDS | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| **比较与选择（4）** |  |  |  |  |  |  |  |  |
| pto.tcmp | TCMP | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tcmps | TCMPS | tile | ✅ | ✅ | ❌ | ✅ | — |  |
| pto.tsel | TSEL | tile | ✅ | ✅ | ❌ | ✅ | — | 当某一分支的标量 TSELS 无法承载时，`tile.select` 也降级到这里 |
| pto.tsels | TSELS | tile | ✅ | ✅ | ❌ | ✅ | — | 已补齐规范 4 输入链路；A2/A3 真机已验证，A5 真机待验证。组合算子 `tile.select` 的 `mask ? tile : scalar` 形式降级到这里 —— 它自身没有同名 `pto.*` 算子 |
| **位运算（11）** |  |  |  |  |  |  |  |  |
| pto.tand | TAND | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | A2/A3 真机通过有/无符号 8/16 位 pattern 的完整、行尾、列尾及行列组合尾部场景；A5 真机待验证 |
| pto.tor | TOR | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | A2/A3 真机通过有/无符号 8/16 位 pattern 的完整、行尾、列尾及行列组合尾部场景；A5 真机待验证 |
| pto.txor | TXOR | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | A2/A3 真机通过有/无符号 8/16 位 pattern、显式 tmp 及四类 shape；IR UT 覆盖 alias 拒绝场景；A5 真机待验证 |
| pto.tshl | TSHL | tile+tensor | ✅ | ✅ | ✅ | ❌ | — | 已有链路；历史 ISA/语义问题，需按当前 pin 复验 |
| pto.tshr | TSHR | tile+tensor | ✅ | ✅ | ✅ | ❌ | — | 已有链路；历史 ISA/语义问题，需按当前 pin 复验 |
| pto.tnot | TNOT | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | 已由既有同名 `tile.not` ST 覆盖 |
| pto.tands | TANDS | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | A2/A3 真机通过有/无符号 8/16 位 tile、immediate/SSA scalar 及四类 shape；A5 真机待验证 |
| pto.tors | TORS | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | A2/A3 真机通过有/无符号 8/16 位 tile、immediate/SSA scalar 及四类 shape；A5 真机待验证 |
| pto.txors | TXORS | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | A2/A3 真机通过有/无符号 8/16 位 tile、immediate/SSA scalar、显式 tmp 及四类 shape；IR UT 覆盖 alias 拒绝场景；A5 真机待验证 |
| pto.tshls | TSHLS | tile+tensor | ✅ | ✅ | ✅ | ❌ | — | 已有链路；历史 ISA/语义问题，需按当前 pin 复验 |
| pto.tshrs | TSHRS | tile+tensor | ✅ | ✅ | ✅ | ❌ | — | 已有链路；历史 ISA/语义问题，需按当前 pin 复验 |
| **数据重排（15）** |  |  |  |  |  |  |  |  |
| pto.tconcat | TCONCAT | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tconcatidx | TCONCAT (indexed) | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING：缺完整前端/codegen/ST 链路 |
| pto.tgather | TGATHER | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tgatherb | TGATHERB | tile | ✅ | ✅ | ❌ | ✅ | — | 32-byte 块偏移前端 + 精确 codegen + 同名 ST；A2/A3 真机 8/8 通过；A5 真机待验证 |
| pto.tscatter | TSCATTER | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.mgather | MGATHER | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | 源仍在 GM 时，通过 `pl.gather(src, index=idx)` 提供 tensor 层扁平元素 Vec 接口；tile 层保留规范 Vec/Mat overload 及 row/elem coalesce；扩展后的 Vec/Mat 矩阵待验证 |
| pto.mscatter | MSCATTER | tile | ✅ | ✅ | ❌ | ✅ | — |  |
| pto.treshape | TRESHAPE | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tinsert | TINSERT | tile | ✅ | ❌ | ❌ | ✅ | — | 由 `tile.assemble` / auto matmul lowering 发射 |
| pto.textract | TEXTRACT | tile | ✅ | ✅ | ❌ | ✅ | — |  |
| pto.tfillpad | TFILLPAD | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tfillpad_expand | TFILLPAD_EXPAND | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tfillpad_inplace | TFILLPAD_INPLACE | tile | ✅ | ✅ | ❌ | ❌ | — | 发射 `pto.tfillpad`；PTOAS 内存规划通过 `pto.treshape` 别名保留目标 pad 和完整有效范围；由本地分配支撑的结果保留运行时有效范围更新能力 |
| pto.textract_fp | TEXTRACT_FP / TEXTRACT | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING：缺完整前端/codegen/ST 链路 |
| pto.tinsert_fp | TINSERT_FP / TINSERT | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING：缺完整前端/codegen/ST 链路 |
| **排序（2）** |  |  |  |  |  |  |  |  |
| pto.tsort32 | TSORT32 | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tmrgsort | TMRGSORT | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| **类型转换（1）** |  |  |  |  |  |  |  |  |
| pto.tcvt | TCVT | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| **整数序列（1）** |  |  |  |  |  |  |  |  |
| pto.tci | TCI | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| **标量元素访问（2）** |  |  |  |  |  |  |  |  |
| pto.tgetval | .GetValue | tile | ✅ | ✅ | ❌ | ✅ | — | 由 `tile.read` 发射 |
| pto.tsetval | .SetValue | tile | ✅ | ✅ | ❌ | ✅ | — | 由 `tile.write` 发射 |
| **MX 量化（6）** |  |  |  |  |  |  |  |  |
| pto.tget_scale_addr | GetScaleAddr + TASSIGN | tile | ✅ | ✅ | ✅ | ❌ | — | NEW frontend+codegen；Mat→scale `tmov` 按源序发射，PTOAS `PTOA5NormalizeTMovPass` 重排为 bind-before-fill；见 [operators MX 约束](ir/05-operators.md#mx--ascend950ptoas-约束) |
| pto.tmov.fp | TMOV_FP | tile | ✅ | ❌ | ❌ | ❌ | — | 已有 backend hook，缺 IR/Python 前端与 ST |
| pto.tquant | TQUANT | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING：缺完整前端/codegen/ST 链路 |
| pto.tquant.mx | TQUANT (overload) | tile | ✅ | ✅ | ❌ | ✅ | — | A5 仅 MXFP8 独立量化前端+codegen+真机 ST（`group_axis` A/B）；MXFP4 quant 暂缓 |
| pto.tstore_fp | TSTORE_FP | tile | ✅ | ❌ | ❌ | ❌ | — | 当前 backend 发 `pto.tstore.fp` |
| pto.tdequant | TDEQUANT | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING：缺完整前端/codegen/ST 链路 |
| **同步（8）** |  |  |  |  |  |  |  |  |
| pto.barrier | pipe_barrier / dsb | internal | ✅ | — | — | ✅ | — | compiler/system lowering 自动发射 |
| pto.barrier_sync | barrier lowering | internal | ✅ | — | — | — | — | 同步/调度内部原语，不独立建 ST |
| pto.record_event | set_flag lowering | internal | ✅ | — | — | — | — | 同步/调度内部原语，不独立建 ST |
| pto.wait_event | wait_flag lowering | internal | ✅ | — | — | — | — | 同步/调度内部原语，不独立建 ST |
| pto.syncall | SYNCALL | internal | ✅ | — | — | ✅ | — | compiler/system lowering 自动发射 |
| pto.sync.set | set_intra_block / FFTS | internal | ✅ | — | — | ✅ | — | compiler/system lowering 自动发射 |
| pto.sync.wait | wait_intra_block / wait_flag_dev | internal | ✅ | — | — | ✅ | — | compiler/system lowering 自动发射 |
| pto.tsync | TSYNC | internal | ✅ | — | — | — | — | 同步/调度内部原语，不独立建 ST |
| **Core Virtualization（2）** |  |  |  |  |  |  |  |  |
| pto.section.cube | — | internal | ✅ | — | — | — | — | 同步/调度内部原语，不独立建 ST |
| pto.section.vector | — | internal | ✅ | — | — | — | — | 同步/调度内部原语，不独立建 ST |
| **Frontend Pipe（15）** |  |  |  |  |  |  |  |  |
| pto.reserve_buffer | — | internal | ✅ | — | — | ✅ | — | compiler/system lowering 自动发射 |
| pto.import_reserved_buffer | — | internal | ✅ | — | — | ✅ | — | compiler/system lowering 自动发射 |
| pto.aic_initialize_pipe | TPipe / internal init | internal | ✅ | — | — | ✅ | — | compiler/system lowering 自动发射 |
| pto.aiv_initialize_pipe | TPipe / internal init | internal | ✅ | — | — | ✅ | — | compiler/system lowering 自动发射 |
| pto.talloc_to_aiv | TALLOC | internal | ✅ | — | — | — | — | 由 pipe 生命周期集成覆盖 |
| pto.talloc_to_aic | TALLOC | internal | ✅ | — | — | — | — | 由 pipe 生命周期集成覆盖 |
| pto.tpush_to_aiv | TPUSH | tile | ✅ | ✅ | ❌ | ✅ | — |  |
| pto.tpush_to_aic | TPUSH | tile | ✅ | ✅ | ❌ | ✅ | — |  |
| pto.tpop_from_aic | TPOP | tile | ✅ | ✅ | ❌ | ✅ | — |  |
| pto.tpop_from_aiv | TPOP | tile | ✅ | ✅ | ❌ | ✅ | — |  |
| pto.tfree_from_aic | TFREE | internal | ✅ | — | — | ✅ | — | compiler/system lowering 自动发射 |
| pto.tfree_from_aiv | TFREE | internal | ✅ | — | — | ✅ | — | compiler/system lowering 自动发射 |
| pto.tpush | TPUSH | internal | ✅ | — | — | — | — | legacy；由带方向接口替代 |
| pto.tpop | TPOP | internal | ✅ | — | — | — | — | legacy；由带方向接口替代 |
| pto.tfree | TFREE | internal | ✅ | — | — | — | — | legacy；由带方向接口替代 |
| **Runtime Intrinsics（4）** |  |  |  |  |  |  |  |  |
| pto.get_block_idx | get_block_idx | tile+tensor | ✅ | ✅ | ✅ | — | — | codegen 转为 wrapper 参数，不发同名 PTO op |
| pto.get_subblock_idx | get_subblockid | tile+tensor | ✅ | ✅ | ✅ | — | — | codegen 转为 wrapper 参数，不发同名 PTO op |
| pto.get_block_num | get_block_num | tile+tensor | ✅ | ✅ | ✅ | — | — | codegen 转为 wrapper 参数，不发同名 PTO op |
| pto.get_subblock_num | get_subblockdim | internal | ✅ | — | — | — | — | codegen 转为 wrapper 参数，不发同名 PTO op |
| **调试（3）** |  |  |  |  |  |  |  |  |
| pto.tprint | TPRINT | tile | ✅ | ❌ | ❌ | ❌ | — | 已有 backend hook，缺 IR/Python 前端与 ST |
| pto.print | cce::printf | internal | ✅ | — | — | — | — | 内部/调试辅助，不独立建 ST |
| pto.trap | trap | internal | ✅ | — | — | — | — | 内部/调试辅助，不独立建 ST |
| **通信（14）** |  |  |  |  |  |  |  |  |
| pto.comm.build_async_session | pto::comm::BuildAsyncSession | comm | ✅ | — | — | — | ❌ | 分布式接口未完成同名 ST |
| pto.comm.tput_async | TPUT_ASYNC | comm | ✅ | — | — | — | ❌ | 分布式接口未完成同名 ST |
| pto.comm.tget_async | TGET_ASYNC | comm | ✅ | — | — | — | ❌ | 分布式接口未完成同名 ST |
| pto.comm.wait_async_event | AsyncEvent.Wait | comm | ✅ | — | — | — | ❌ | 分布式接口未完成同名 ST |
| pto.comm.test_async_event | AsyncEvent.Test | comm | ✅ | — | — | — | ❌ | 分布式接口未完成同名 ST |
| pto.comm.tput | TPUT | comm | ✅ | — | — | — | ✅ | test_l3_put / remote_store |
| pto.comm.tget | TGET | comm | ✅ | — | — | — | ✅ | test_l3_get |
| pto.comm.tnotify | TNOTIFY | comm | ✅ | — | — | — | ✅ | test_l3_notify_wait |
| pto.comm.twait | TWAIT | comm | ✅ | — | — | — | ✅ | test_l3_notify_wait |
| pto.comm.ttest | TTEST | comm | ✅ | — | — | — | ❌ | 分布式接口未完成同名 ST |
| pto.comm.tbroadcast | TBROADCAST | comm | ✅ | — | — | — | ❌ | 高层测试会分解，当前无同名 PTO ST |
| pto.comm.tgather | TGATHER | comm | ✅ | — | — | — | ❌ | 高层测试会分解，当前无同名 PTO ST |
| pto.comm.tscatter | TSCATTER | comm | ✅ | — | — | — | ❌ | 高层测试会分解，当前无同名 PTO ST |
| pto.comm.treduce | TREDUCE | comm | ✅ | — | — | — | ❌ | 高层测试会分解，当前无同名 PTO ST |
| **栈局部 Array / Struct（6）** |  |  |  |  |  |  |  |  |
| pto.declare_local_array | — | internal | ✅ | — | — | ✅ | — | compiler/system lowering 自动发射 |
| pto.local_array_get | — | internal | ✅ | — | — | ✅ | — | compiler/system lowering 自动发射 |
| pto.local_array_set | — | internal | ✅ | — | — | ✅ | — | compiler/system lowering 自动发射 |
| pto.declare_struct | — | internal | ✅ | — | — | — | — | 内部/调试辅助，不独立建 ST |
| pto.struct_get | — | internal | ✅ | — | — | — | — | 内部/调试辅助，不独立建 ST |
| pto.struct_set | — | internal | ✅ | — | — | — | — | 内部/调试辅助，不独立建 ST |
| **源码兼容 / 手动模式（1）** |  |  |  |  |  |  |  |  |
| pto.tassign | TASSIGN | internal | ✅ | — | — | — | — | 失活 backend hook，不独立建 ST |

**统计**：共 204 个 PTOAS 公共/兼容 op；pypto tile 前端 113 个，tensor 前端 75 个
（另有 `pl.prefetch.*` 一族 4 个非 tile/tensor op）；同名 ST 覆盖 121 个
（普通 ST 117，distributed ST 4）；无同名 ST 51 个（普通 41，distributed 10）；
其余 32 个 op 不适合独立 ST。
