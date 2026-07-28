<!-- markdownlint-disable MD013 MD060 MD033 -->
# PTOAS Op 状态矩阵

行 = **最新 PTOAS 提供的公共及兼容 op**。接口基线为 Little-oil/PTOAS `main`
`d852dd2dba3e5bf7a69ce8324eb88afc336e8a33`：manual 公共接口 189 个，加当前
`PTOOps.td` 仍保留的 source-only 兼容/tile 接口 15 个，共 204 个。列状态按
**PyPTO 当前源码**核实（最后更新 2026-08-11）。后续每加或修改一个 op，只更新本表对应行。

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
| pto.load_scalar | direct pointer load | tensor | ✅ | ❌ | ✅ | ✅ | — | 由 `tensor.read` 发射 |
| pto.store_scalar | direct pointer store | tensor | ✅ | ❌ | ✅ | ✅ | — | 由 `tensor.write` 发射 |
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
| pto.trem | TREM | tile | ✅ | ✅ | ❌ | ✅ | — | A2/A3 FP32/INT32 同名真机 ST 已通过，覆盖完整与尾部 valid shape；A5 真机待验证 |
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
| pto.trems | TREMS | tile | ✅ | ✅ | ❌ | ✅ | — | A2/A3 FP32/INT32 scalar 同名真机 ST 已通过，覆盖负值与尾部 valid shape；A5 真机待验证 |
| pto.taddc | TADD + TADD | tile | ✅ | ✅ | ❌ | ❌ | — | 已有链路；历史 ISA/语义问题，需按当前 pin 复验 |
| pto.tsubc | TSUB + TADD | tile | ✅ | ✅ | ❌ | ❌ | — | 已有链路；历史 ISA/语义问题，需按当前 pin 复验 |
| pto.taddsc | TADDS + TADD | tile | ✅ | ✅ | ❌ | ❌ | — | 已有链路；历史 ISA/语义问题，需按当前 pin 复验 |
| pto.tsubsc | TSUBS + TADD | tile | ✅ | ✅ | ❌ | ❌ | — | 已有链路；历史 ISA/语义问题，需按当前 pin 复验 |
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
| pto.tsel | TSEL | tile | ✅ | ✅ | ❌ | ✅ | — |  |
| pto.tsels | TSELS | tile | ✅ | ✅ | ❌ | ✅ | — | 已补齐规范 4 输入链路；A2/A3 真机已验证，A5 真机待验证 |
| **位运算（11）** |  |  |  |  |  |  |  |  |
| pto.tand | TAND | tile+tensor | ✅ | ✅ | ✅ | ❌ | — | 已有链路；历史 ISA/语义问题，需按当前 pin 复验 |
| pto.tor | TOR | tile+tensor | ✅ | ✅ | ✅ | ❌ | — | 已有链路；历史 ISA/语义问题，需按当前 pin 复验 |
| pto.txor | TXOR | tile+tensor | ✅ | ✅ | ✅ | ❌ | — | 已有链路；历史 ISA/语义问题，需按当前 pin 复验 |
| pto.tshl | TSHL | tile+tensor | ✅ | ✅ | ✅ | ❌ | — | 已有链路；历史 ISA/语义问题，需按当前 pin 复验 |
| pto.tshr | TSHR | tile+tensor | ✅ | ✅ | ✅ | ❌ | — | 已有链路；历史 ISA/语义问题，需按当前 pin 复验 |
| pto.tnot | TNOT | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tands | TANDS | tile+tensor | ✅ | ✅ | ✅ | ❌ | — | 已有链路；历史 ISA/语义问题，需按当前 pin 复验 |
| pto.tors | TORS | tile+tensor | ✅ | ✅ | ✅ | ❌ | — | 已有链路；历史 ISA/语义问题，需按当前 pin 复验 |
| pto.txors | TXORS | tile+tensor | ✅ | ✅ | ✅ | ❌ | — | 已有链路；历史 ISA/语义问题，需按当前 pin 复验 |
| pto.tshls | TSHLS | tile+tensor | ✅ | ✅ | ✅ | ❌ | — | 已有链路；历史 ISA/语义问题，需按当前 pin 复验 |
| pto.tshrs | TSHRS | tile+tensor | ✅ | ✅ | ✅ | ❌ | — | 已有链路；历史 ISA/语义问题，需按当前 pin 复验 |
| **数据重排（15）** |  |  |  |  |  |  |  |  |
| pto.tconcat | TCONCAT | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tconcatidx | TCONCAT (indexed) | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING：缺完整前端/codegen/ST 链路 |
| pto.tgather | TGATHER | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tgatherb | TGATHERB | tile | ✅ | ✅ | ❌ | ✅ | — | 32-byte 块偏移前端 + 精确 codegen + 同名 ST；A2/A3 真机 8/8 通过；A5 真机待验证 |
| pto.tscatter | TSCATTER | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.mgather | MGATHER | tile | ✅ | ✅ | ❌ | ✅ | — | 规范 Vec/Mat overload，并显式携带 row/elem coalesce；Vec 子集已通过 A2/A3 真机，扩展后的 Vec/Mat 矩阵待验证 |
| pto.mscatter | MSCATTER | tile | ✅ | ✅ | ❌ | ✅ | — |  |
| pto.treshape | TRESHAPE | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tinsert | TINSERT | tile | ✅ | ❌ | ❌ | ✅ | — | 由 `tile.assemble` / auto matmul lowering 发射 |
| pto.textract | TEXTRACT | tile | ✅ | ✅ | ❌ | ✅ | — |  |
| pto.tfillpad | TFILLPAD | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tfillpad_expand | TFILLPAD_EXPAND | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tfillpad_inplace | TFILLPAD_INPLACE | tile | ✅ | ✅ | ❌ | ❌ | — | 当前 codegen 发 `pto.tfillpad` |
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
| pto.tquant.mx | TQUANT (overload) | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING：缺完整前端/codegen/ST 链路 |
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
（另有 `pl.prefetch.*` 一族 4 个非 tile/tensor op）；同名 ST 覆盖 111 个
（普通 ST 107，distributed ST 4）；无同名 ST 61 个（普通 51，distributed 10）；
这 204 个中另有 32 个 op 不适合独立 ST。
