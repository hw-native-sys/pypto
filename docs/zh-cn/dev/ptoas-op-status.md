<!-- markdownlint-disable MD013 MD060 MD033 -->
# PTOAS Op 状态矩阵

行 = **PTOAS 文档提供的全部 op**（去重 148 个唯一 op）。列状态按 **pypto 实际源码**
核实的快照（最后更新 2026-06-23）。**后续每加/改一个 op，请同步更新本表对应行。**

## 完成判定原则

**一个 op 是否"完成"以"是否有 ST 测试"为准。** codegen 即使写错，只要该 op 有 ST，
真机测试时就会暴露；**没有 ST 的 op 一律视为未完成**（无论前端/codegen 是否已写）。
因此本表中：

- `pypto 前端✅ + ST❌` → **未完成**（前端写了但没经过真机验证；可能 codegen 有误，
  也可能是已知的 a2a3 ISA/cube/ptoas 缺陷导致 ST 被下架——两者都按"未完成"对待）。
- `ST✅` → 已被真机 ST 覆盖，视为完成。

> **关于 148 vs 150**：PTOAS 文档"总计 150"是行数，含 2 条重复行——`pto.tstore` 与
> `pto.tmov` 各列两次（基础行 + 方锐的属性建模细节行，备注"已在基础行覆盖"）。本表按
> 唯一 op 去重 = 148，无遗漏。

## 图例

- **级别**：op 在 pypto 注册的层级（tile / tensor / tile+tensor / comm=分布式通讯）
- **PTOAS接口**：✅ = PTOAS"已对应"(pto-isa 有指令)；❌ = PTOAS未实现 / ISA_ONLY
- **pypto-tile / -tensor 前端**：✅ = `REGISTER_OP("tile|tensor.<op>")` 已注册；
  `—` = 不适用（comm 通讯 op 不是 tile/tensor 级，由分布式/system API 提供）
- **ST测试**：✅ = `tests/st/`（不含 `fuzz/` 规格清单）有直接引用该 op 的真实 ST；`—` = comm 走下一列
- **distribute ST测试**：仅对 comm op —— ✅+证据文件 = `tests/st/distributed/` 有覆盖；❌ = 无；非 comm 为 `—`
- **备注**：`NEW`=本轮新增(PR #1824)；`MISSING`=PTOAS有但pypto未写(待补，见 add-op skill)；
  `codegen-incomplete`=IR/转换有但无codegen；`FP variant`/`internal`/`distributed`=变体/内部/分布式

> 维护：本表初版由脚本对照源码生成（PTOAS op 全集 + name-map 内嵌于生成器，临时存于
> `temp/`）。日常以**手工更新对应行**为主：加 op 后把该行的前端/ST 列改为 ✅。

| PTOAS op (pto.*) | pto-isa API | 级别 | PTOAS接口 | pypto-tile前端 | pypto-tensor前端 | ST测试 | distribute ST测试 | 备注 |
|---|---|---|:---:|:---:|:---:|:---:|:---:|---|
| **逐元素 (Tile-Tile)** |  |  |  |  |  |  |  |  |
| pto.tabs | TABS | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.taddc | TADDC | tile | ✅ | ✅ | ❌ | ❌ | — | a2a3 pto.tsubc 误算 a-b-c；PR #1823 ST 暂下架 |
| pto.taddsc | TADDSC | tile | ✅ | ✅ | ❌ | ❌ | — | a2a3 pto.tsubc 误算 a-b-c；PR #1823 ST 暂下架 |
| pto.tadd | TADD | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tand | TAND | tile | ✅ | ✅ | ❌ | ❌ | — | a2a3 ptoas 拒绝该 op；PR #1823 ST 暂下架 |
| pto.tcmp | TCMP | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tcvt | TCVT | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tdiv | TDIV | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | ST: feat-class-a-op-st (a2a3) |
| pto.texp | TEXP | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tlog | TLOG | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | ST: feat-class-a-op-st (a2a3) |
| pto.tmax | TMAX | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | binary max |
| pto.tmin | TMIN | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | binary min |
| pto.tmul | TMUL | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tneg | TNEG | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tnot | TNOT | tile | ✅ | ✅ | ❌ | ✅ | — | ST: PR #1823 |
| pto.tor | TOR | tile | ✅ | ✅ | ❌ | ❌ | — | a2a3 ptoas 拒绝该 op；PR #1823 ST 暂下架 |
| pto.tprelu | TPRELU | tile | ✅ | ✅ | ❌ | ❌ | — | a2a3 3-arg DSL 与 codegen pto.tprelu 不符 |
| pto.trecip | TRECIP | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.trelu | TRELU | tile | ✅ | ✅ | ❌ | ✅ | — | ST: PR #1823 |
| pto.trem | TREM | tile | ✅ | ✅ | ❌ | ❌ | — | a2a3 TREM int32 误算/TREMS alloc 错；PR #1823 修了 tmp 操作数, ST 待 ISA |
| pto.trsqrt | TRSQRT | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tsel | TSEL | tile | ✅ | ✅ | ❌ | ✅ | — |  |
| pto.tshl | TSHL | tile | ✅ | ✅ | ❌ | ❌ | — | a2a3 ptoas 拒绝该 op；PR #1823 ST 暂下架 |
| pto.tshr | TSHR | tile | ✅ | ✅ | ❌ | ❌ | — | a2a3 ptoas 拒绝该 op；PR #1823 ST 暂下架 |
| pto.tsqrt | TSQRT | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | ST: PR #1823 |
| pto.tsubc | TSUBC | tile | ✅ | ✅ | ❌ | ❌ | — | a2a3 误算 a-b-c；PR #1823 ST 暂下架 |
| pto.tsub | TSUB | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tsubsc | TSUBSC | tile | ✅ | ✅ | ❌ | ❌ | — | a2a3 pto.tsubc 误算 a-b-c；PR #1823 ST 暂下架 |
| pto.txor | TXOR | tile | ✅ | ✅ | ❌ | ❌ | — | a2a3 仅 int16/uint16；PR #1823 修了 tmp 操作数 |
| pto.tfmod | TFMOD | tile+tensor | ✅ | ✅ | ✅ | ❌ | — | PR #1837 前端+codegen；a2a3 TFMOD 误算(全0)，ST 暂下架待 ISA |
| pto.pow | TPOW | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING (需 tmp tile) |
| **逐元素与标量** |  |  |  |  |  |  |  |  |
| pto.pows | TPOWS | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING (需 tmp tile) |
| pto.tadds | TADDS | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | ST: feat-class-a-op-st (a2a3) |
| pto.tands | TANDS | tile | ✅ | ✅ | ❌ | ❌ | — | a2a3 ptoas 拒绝该 op；PR #1823 ST 暂下架 |
| pto.tcmps | TCMPS | tile | ✅ | ✅ | ❌ | ✅ | — |  |
| pto.tdivs | TDIVS | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | ST: PR #1823 |
| pto.texpands | TEXPANDS | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | ST: feat-class-a-op-st (a2a3); fixed deducer dtype + scalar-dtype codegen |
| pto.tlrelu | TLRELU | tile | ✅ | ✅ | ❌ | ✅ | — | ST: PR #1823 |
| pto.tmaxs | TMAXS | tile | ✅ | ✅ | ❌ | ✅ | — |  |
| pto.tmins | TMINS | tile | ✅ | ✅ | ❌ | ✅ | — |  |
| pto.tmuls | TMULS | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | ST: PR #1823 |
| pto.tors | TORS | tile | ✅ | ✅ | ❌ | ❌ | — | a2a3 ptoas 拒绝该 op；PR #1823 ST 暂下架 |
| pto.trems | TREMS | tile | ✅ | ✅ | ❌ | ❌ | — | a2a3 TREM int32 误算/TREMS alloc 错；PR #1823 修了 tmp 操作数, ST 待 ISA |
| pto.tsels | TSELS | tile | ✅ | ✅ | ❌ | ✅ | — | ST: feat-class-a-op-st (a2a3, FP32/FP16); redefined to TSELS (mask,src,tmp,scalar); INT32 skipped (integer cmps codegen gap) |
| pto.tshls | TSHLS | tile | ✅ | ✅ | ❌ | ❌ | — | a2a3 ptoas 拒绝该 op；PR #1823 ST 暂下架 |
| pto.tshrs | TSHRS | tile | ✅ | ✅ | ❌ | ❌ | — | a2a3 ptoas 拒绝该 op；PR #1823 ST 暂下架 |
| pto.tsubs | TSUBS | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | ST: feat-class-a-op-st (a2a3) |
| pto.txors | TXORS | tile | ✅ | ✅ | ❌ | ❌ | — | a2a3 仅 int16/uint16；PR #1823 修了 tmp 操作数 |
| pto.tfmods | TFMODS | tile+tensor | ✅ | ✅ | ✅ | ❌ | — | PR #1837 前端+codegen；a2a3 TFMODS 误算(全0)，ST 暂下架待 ISA |
| **按轴逐元素 (reduce/expand)** |  |  |  |  |  |  |  |  |
| pto.tcolexpand | TCOLEXPAND | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tcolmax | TCOLMAX | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tcolmin | TCOLMIN | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tcolsum | TCOLSUM | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.trowexpand | TROWEXPAND | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.trowmax | TROWMAX | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.trowmin | TROWMIN | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | ST: feat-class-a-op-st (a2a3) |
| pto.trowsum | TROWSUM | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tcolprod | TCOLPROD | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | NEW |
| pto.trowprod | TROWPROD | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | NEW |
| pto.trowexpandsub | TROWEXPANDSUB | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.trowexpandmul | TROWEXPANDMUL | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.trowexpanddiv | TROWEXPANDDIV | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.trowexpandadd | TROWEXPANDADD | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | ST: feat-class-a-op-st (a2a3) |
| pto.trowexpandmax | TROWEXPANDMAX | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | NEW |
| pto.trowexpandmin | TROWEXPANDMIN | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | NEW |
| pto.trowexpandexpdif | TROWEXPANDEXPDIF | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | NEW |
| pto.tcolexpandmul | TCOLEXPANDMUL | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tcolexpandadd | TCOLEXPANDADD | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | ST: PR #1823 |
| pto.tcolexpanddiv | TCOLEXPANDDIV | tile+tensor | ✅ | ✅ | ✅ | ❌ | — | codegen-incomplete |
| pto.tcolexpandsub | TCOLEXPANDSUB | tile+tensor | ✅ | ✅ | ✅ | ❌ | — | codegen-incomplete |
| pto.tcolexpandmax | TCOLEXPANDMAX | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | NEW |
| pto.tcolexpandmin | TCOLEXPANDMIN | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | NEW |
| pto.tcolexpandexpdif | TCOLEXPANDEXPDIF | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | NEW |
| **矩阵乘** |  |  |  |  |  |  |  |  |
| pto.tmatmul | TMATMUL | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tmatmul.acc | TMATMUL_ACC | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tmatmul.bias | TMATMUL_BIAS | tile | ✅ | ✅ | ❌ | ✅ | — | ST: PR #1823 |
| pto.tmatmul.mx | TMATMUL_MX | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING |
| pto.tgemv | TGEMV | tile | ✅ | ✅ | ❌ | ❌ | — | a2a3 1-row TExtract dstRow%16；PR #1823 ST 暂下架 |
| pto.tgemv.acc | TGEMV_ACC | tile | ✅ | ✅ | ❌ | ❌ | — | a2a3 acc→acc pto.tmov 未支持；PR #1823 ST 暂下架 |
| pto.tgemv.bias | TGEMV_BIAS | tile | ✅ | ✅ | ❌ | ❌ | — | a2a3 1-row TExtract dstRow%16；PR #1823 ST 暂下架 |
| pto.tgemv.mx | TGEMV_MX | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING |
| **访存** |  |  |  |  |  |  |  |  |
| pto.tload | TLOAD | tile | ✅ | ✅ | ❌ | ✅ | — |  |
| pto.tstore | TSTORE | tile | ✅ | ✅ | ❌ | ✅ | — |  |
| pto.tstore_fp | TSTORE_FP | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING (FP variant) |
| pto.mgather | MGATHER | tile | ✅ | ❌ | ❌ | ❌ | — | issue #1807 (separate worktree) |
| pto.mscatter | MSCATTER | tile | ✅ | ✅ | ❌ | ✅ | — |  |
| **复杂操作** |  |  |  |  |  |  |  |  |
| pto.tci | TCI | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tgather | TGATHER | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tgatherb | TGATHERB | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING |
| pto.tscatter | TSCATTER | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tsort32 | TSORT32 | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tmrgsort | TMRGSORT | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | reg as mrgsort_format1/2; ST: test_sort.py incl. 2/3/4-way (a2a3) |
| pto.tfillpad | TFILLPAD | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tfillpad_inpace | TFILLPAD_INPLACE | tile | ✅ | ✅ | ❌ | ✅ | — |  |
| pto.tfillpad_expand | TFILLPAD_EXPAND | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | NEW tile+tensor 前端+codegen+ST；tile a2a3 CI 通过，tensor 待 CI |
| pto.tpartadd | TPARTADD | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | NEW 前端+codegen+ST；a2a3 真机待 CI（irregular 家族留意 ISA 缺陷） |
| pto.tpartmul | TPARTMUL | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | NEW 前端+codegen+ST；a2a3 真机待 CI（irregular 家族留意 ISA 缺陷） |
| pto.tpartmax | TPARTMAX | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | NEW 前端+codegen+ST；a2a3 真机待 CI（irregular 家族留意 ISA 缺陷） |
| pto.tpartmin | TPARTMIN | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | NEW 前端+codegen+ST；a2a3 真机待 CI（irregular 家族留意 ISA 缺陷） |
| pto.tprint | TPRINT | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING (debug, skip) |
| **量化** |  |  |  |  |  |  |  |  |
| pto.tquant | TQUANT | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING |
| pto.tdequant | TDEQUANT | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING |
| **固定管线** |  |  |  |  |  |  |  |  |
| pto.textract | TEXTRACT | tile | ✅ | ✅ | ❌ | ✅ | — |  |
| pto.textract_fp | TEXTRACT_FP | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING (FP variant) |
| pto.tinsert | TINSERT | tile | ✅ | ❌ | ❌ | ❌ | — | pypto uses assemble |
| pto.tinsert_fp | TINSERT_FP | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING (FP variant) |
| pto.ttrans | TTRANS | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.tmov | TMOV | tile | ✅ | ✅ | ❌ | ✅ | — |  |
| pto.tmov.fp | TMOV_FP | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING (FP variant) |
| **其他** |  |  |  |  |  |  |  |  |
| pto.tconcat | TCONCAT | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.treshape | TRESHAPE | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.set_validshape | SetValidShape | tile+tensor | ✅ | ✅ | ✅ | ✅ | — |  |
| pto.subset | TSUBVIEW | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | pypto slice |
| pto.tpush | TPUSH | tile | ✅ | ✅ | ❌ | ✅ | — | reg as tpush_to_aic/aiv |
| pto.tpop | TPOP | tile | ✅ | ✅ | ❌ | ✅ | — | reg as tpop_from_aic/aiv |
| pto.tfree | TFREE | tile | ✅ | ❌ | ❌ | ❌ | — | internal (skip) |
| pto.tpack | TPACK | tile | ✅ | ❌ | ❌ | ❌ | — | not exposed |
| pto.taxpy | TAXPY | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING (王淼) |
| pto.thistogram | THISTOGRAM | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING (王淼) |
| pto.trandom | TRANDOM | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING (王淼) |
| pto.ttri | TTRI | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING (王淼) |
| pto.tget_scale_addr | TGET_SCALE_ADDR | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING (王淼) |
| pto.tprefetch | TPREFETCH | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING (infra, skip) |
| pto.trowargmax | TROWARGMAX | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | int32 索引输出 + tmp tile；真机 a2a3 已验证 (feat-add-ptoas-argmax) |
| pto.trowargmin | TROWARGMIN | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | int32 索引输出 + tmp tile；真机 a2a3 已验证 (feat-add-ptoas-argmax) |
| pto.tcolargmax | TCOLARGMAX | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | int32 索引输出 + tmp tile（tmp 必须与 src 同形，不可补到 128）；真机 a2a3 已验证 (feat-add-ptoas-argmax) |
| pto.tcolargmin | TCOLARGMIN | tile+tensor | ✅ | ✅ | ✅ | ✅ | — | int32 索引输出 + tmp tile（tmp 必须与 src 同形）；真机 a2a3 已验证 (feat-add-ptoas-argmax) |
| pto.tpartargmax | TPARTARGMAX | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING (方锐) |
| pto.tpartargmin | TPARTARGMIN | tile | ✅ | ❌ | ❌ | ❌ | — | MISSING (方锐) |
| **手动模式** |  |  |  |  |  |  |  |  |
| pto.tsync | TSYNC | tile | ✅ | ❌ | ❌ | ❌ | — | internal sync |
| pto.tassign | TASSIGN | tile | ✅ | ❌ | ❌ | ❌ | — | internal |
| **卷积 (PTOAS未实现)** |  |  |  |  |  |  |  |  |
| — | TIMG2COL | tile | ❌ | ❌ | ❌ | ❌ | — | PTOAS未实现 |
| — | TSETFMATRIX | tile | ❌ | ❌ | ❌ | ❌ | — | PTOAS未实现 |
| — | TSET_IMG2COL_PADDING | tile | ❌ | ❌ | ❌ | ❌ | — | PTOAS未实现 |
| — | TSET_IMG2COL_RPT | tile | ❌ | ❌ | ❌ | ❌ | — | PTOAS未实现 |
| **通讯 (comm — 分布式, 非 tile/tensor)** |  |  |  |  |  |  |  |  |
| pto.comm.tbroadcast | TBROADCAST | comm | ✅ | — | — | — | ✅ test_l3_broadcast | pl.system/distributed |
| pto.comm.tgather | TGATHER | comm | ✅ | — | — | — | ✅ test_l3_allgather | distributed |
| pto.comm.tscatter | TSCATTER | comm | ✅ | — | — | — | ✅ test_l3_reduce_scatter | distributed |
| pto.treduce | TREDUCE | comm | ✅ | — | — | — | ✅ test_l3_allreduce (+ring/host/parallel/intrinsic) | distributed |
| pto.comm.tget | TGET | comm | ✅ | — | — | — | ✅ test_l3_get | distributed |
| pto.comm.tput | TPUT | comm | ✅ | — | — | — | ✅ test_l3_put (+remote_store) | distributed (pld.put) |
| pto.comm.tget_async | TGET_ASYNC | comm | ✅ | — | — | — | ❌ | distributed |
| pto.comm.tput_async | TPUT_ASYNC | comm | ✅ | — | — | — | ❌ | distributed |
| pto.comm.tnotify | TNOTIFY | comm | ✅ | — | — | — | ✅ test_l3_notify_wait | distributed |
| pto.comm.ttest | TTEST | comm | ✅ | — | — | — | ❌ | distributed |
| pto.comm.twait | TWAIT | comm | ✅ | — | — | — | ✅ test_l3_notify_wait | distributed |
| pto.comm.build_async_session | BuildAsyncSession | comm | ✅ | — | — | — | ❌ | distributed |
| pto.tprefetch_async | TPREFETCH_ASYNC | comm | ❌ | — | — | — | — | ptoas未实现 (ISA_ONLY) |

**统计**：共 148 个 PTOAS op 行；PTOAS 提供接口 143；pypto 前端已写好 98；有 ST 测试 67。
