# A5 Mixed Kernel：MX Scale 传输（V2C 失败与 GM 决策）

## 问题

Ascend950 mixed kernel 中，AIV `quant_mx(layout=MX_A_ZZ)` 产出的 A-scale（E8M0）若经 **V2C `tpush`/`tpop`** 送给 AIC `LeftScale`，在**非均匀** scale（各 MX group 指数不同）下数值错误。全 E8M0=127 的烟测不足以证明传输正确。

几何（ST `QuantizedMatmulMxMixedProgram`）：

| 项 | 值 |
| -- | -- |
| `MIX_M` / `MIX_K_CHUNK` / `N` | 32 / 256 / 32 |
| 每 chunk LeftScale | `[32, 8]`（256 B E8M0） |
| `quant_mx` scale 形态 | packed ZZ `[1, 256]`（由 `ExpandMxPackedQuant` 物化） |

## V2C 约束（A5）

- `TPUSH` Vec→Mat 在 L1 落 **ColMajor Mat**（`pushVec2MatFiFo` + `TINSERT`）。
- NoneBox 对齐：RM 要求 `cols*sizeof % 32 == 0`；CM 要求 `rows*sizeof % 32 == 0`。发送/接收两端都会检查，表现为“双维对齐”。
- AIC 上 `treshape` / `reinterpret` **不支持** `f8E8M0` 作为目标类型，无法在 C 侧把 FP32 槽再合法地收成 E8M0。

## 实验结论

| 方案 | 编译 | 非均匀数值 |
| ---- | ---- | ---------- |
| FP32 `[8,8]` 伪装 V2C；C TPOP E8M0 fractal-32 | 通过 | 失败（probe：期望 pitch8，实测 pitch2） |
| ColMajor FP32 `[32,2]` / CM `[8,8]` | 通过 | 更差或非 stride2/8 |
| pad FP32 `[32,8]`（1024 B）+ C TPOP E8M0 | 通过 | 仍失败 |
| C 侧 FP32→E8M0 view/`treshape` | 被拒 | — |
| 手改纯 E8M0 `TPUSH` CM `[32,8]` / `[256,1]` | 通过 | 仍约 120+/1024 错 |

结论：**问题不只是发送端 dtype 对齐**；V2C 路径无法可靠保序地把 ZZ packed scale 交给 LeftScale。同核 / 编排级 **GM `store` → `tensor.view(MX_A_ZZ)` → `load(Mat)` → LeftScale**（如 `quantized_matmul_mx_onboard`）已验证可用。

## 决策

1. Mixed 中 **FP8 data 可继续走 V2C**。
2. **禁止 E8M0 V2C**：[`ExpandMixedKernel`](../passes/23-expand_mixed_kernel.md) 在生成 `tpush_to_aic` 前对 `FP8E8M0` tile 报错；用户须改用 GM + `MX_A_ZZ` load。
3. **MX scale 强制走 GM**，字节形态 = [`ExpandMxPackedQuant`](../passes/12-expand_mx_packed_quant.md) 产出的连续 ZZ packed E8M0（`[1, M*K/32]` 或每 chunk `[1, M*K_chunk/32]`），**不再**做 FP32 伪装 / pad / 复杂 reshape，也不再使用已删除的 `LegalizeMixedMxScaleViaGm` pass。
4. 废弃原 `LegalizeV2CMxScaleTransport`（FP32 双对齐伪装）方案。

## 同步与 ST 形态（v1）

- **Pass 目标形态**：同 chunk 内 AIV `store(scale)` 在 `tpush(data)` 之前；AIC `tpop(data)` 后再 `load(scale)`（依赖 data V2C 握手）。不扩展 `CollectGmCrossLaneSyncs` 的 V2C GM fence。
- **ST 门禁**（`QuantizedMatmulMxMixedProgram`）：编排层先对整块 A 做一次 `quant_mx(MX_A_ZZ)`，把 packed `[1, M*K/32]` 写入 GM（与 onboard 一致）；随后 Group 只走 FP8 data 的 V2C，AIC 按 chunk 从该 GM 做 `MX_A_ZZ` load。  
  **注意**：把各 chunk 的 `[1, M*K_chunk/32]` packed 字节简单首尾拼接**不等于**整块 `MX_A_ZZ` 打包，因此 ST 不用「分块 store 再拼成大 buffer」。

## 另请参阅

- [`ExpandMxPackedQuant`](../passes/12-expand_mx_packed_quant.md)
- [`ExpandMixedKernel`](../passes/23-expand_mixed_kernel.md)（data 边界仍可 V2C；文档编号以 pass 表为准）
