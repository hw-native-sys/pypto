/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#ifndef PYPTO_BACKEND_910B_BACKEND_910B_HANDLER_H_
#define PYPTO_BACKEND_910B_BACKEND_910B_HANDLER_H_

#include <cstdint>
#include <string>
#include <vector>

#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/dtype.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace backend {

/**
 * @brief BackendHandler implementation for Ascend910B (a2a3).
 *
 * Cross-core data is routed via GM, so the AIV side does not need a fractal
 * adapter; the GM-backed slot buffer must be injected by ExpandMixedKernel;
 * the AIV split-load tpop hazard workaround applies on this backend only.
 */
class Ascend910BHandler : public BackendHandler {
 public:
  static const Ascend910BHandler& Instance();

  [[nodiscard]] std::string GetPtoTargetArch() const override { return "a2a3"; }
  [[nodiscard]] std::string GetLaunchSpecCoreCountMethod() const override { return "set_block_num"; }
  [[nodiscard]] std::string GetDefaultSimPlatform() const override { return "a2a3sim"; }
  [[nodiscard]] std::vector<std::string> GetExtraPtoasFlags() const override { return {"--pto-arch", "a3"}; }

  [[nodiscard]] bool RequiresGMPipeBuffer() const override { return true; }
  [[nodiscard]] bool RequiresSplitLoadTpopWorkaround() const override { return true; }

  // A2/A3 has no int32 ordering compare; pto-isa substitutes an equality
  // mask instead of failing (pto-isa issue #321).
  [[nodiscard]] bool SupportsOrderingCompareDataType(const DataType& src_dtype) const override {
    return src_dtype != DataType::INT32;
  }
  [[nodiscard]] bool RequiresLevel3TmpScratch() const override { return true; }

  // A2/A3 TSEL takes a level-3 explicit scratch of UINT32 [1, 16].
  [[nodiscard]] backend::TileScratchSpec GetTselScratchSpec() const override {
    return {DataType::UINT32, 1, 16};
  }

  // PTOAS verifies the A2/A3 TSELS scratch against one complete physical source
  // row, in the source dtype (see tests/st/runtime/ops/test_sels.py).
  [[nodiscard]] backend::TileScratchSpec GetTselsScratchSpec(DataType src_dtype,
                                                             int64_t src_cols) const override {
    return {src_dtype, 1, src_cols};
  }

  // A2/A3 TSELS covers 16/32-bit integers, FP16, and FP32 -- no 8-bit form.
  [[nodiscard]] bool SupportsTselsDataType(const DataType& src_dtype) const override {
    return src_dtype == DataType::INT16 || src_dtype == DataType::UINT16 || src_dtype == DataType::INT32 ||
           src_dtype == DataType::UINT32 || src_dtype == DataType::FP16 || src_dtype == DataType::FP32;
  }

  // A2/A3 TSEL adds bf16 but still stops at 16 bits, as does the texpands behind
  // tile.full -- so an 8-bit select has no lowering at all on this arch.
  [[nodiscard]] bool SupportsTselDataType(const DataType& dtype) const override {
    return dtype == DataType::INT16 || dtype == DataType::UINT16 || dtype == DataType::INT32 ||
           dtype == DataType::UINT32 || dtype == DataType::FP16 || dtype == DataType::BF16 ||
           dtype == DataType::FP32;
  }

  [[nodiscard]] bool RequiresVtoCFractalAdapt() const override { return false; }
  [[nodiscard]] bool RequiresRuntimeSubblockBridge() const override { return true; }
  [[nodiscard]] bool RequiresNoSplitDualAivDispatch() const override { return true; }
  // A2/A3 offset Acc->Mat tinsert requires f32->bf16/f16 (cannot keep f32).
  [[nodiscard]] bool RequiresLowPrecisionMatScratch() const override { return true; }
  // A2/A3 has no packed fp4 load/store ABI; reject the whole FP4 family.
  [[nodiscard]] bool SupportsIncoreDataType(const DataType& dtype) const override {
    return dtype.GetBit() != 4 && !dtype.IsFp4Family();
  }

  // A2/A3 store pipe supports bf16 atomic-add (pto-isa set_atomic_bf16).
  [[nodiscard]] bool SupportsBf16AtomicAdd() const override { return true; }

  // A2/A3 fix-pipe Acc->GM destination whitelist (pto-isa a2a3 CheckAcc2gm,
  // non-quant branch; ptoas rejects anything else at pto.tstore verification).
  [[nodiscard]] bool SupportsAccToGmDtype(const DataType& dtype) const override {
    return dtype == DataType::INT32 || dtype == DataType::FP32 || dtype == DataType::FP16 ||
           dtype == DataType::BF16;
  }

  // A2/A3 scale-bearing fix-pipe writeback. Measured against ptoas v0.61, which
  // verifies these pairs explicitly:
  //   tinsert: "(src=f32,dst=i8) or (src=i32,dst=i8/f16/i16)"
  //   tstore : "i8/ui8"  from an f32 accumulator, "i8/ui8/f16" from an i32 one
  // pto-isa a2a3 `GetScalarPreQuantMode` agrees (QF322B8_PRE / REQ8 / DEQF16).
  // INT16 (SHIFTS322S16) is excluded per the base-class contract: its payload
  // is a shift count, not a scale. Notably there is NO scaled f32 -> f16/bf16
  // here -- that narrowing exists only in the *unscaled* writeback.
  //
  // Acc->Mat is withheld regardless of dtype (PTOAS#1570): ptoas 0.63 assembles
  // the scale onto `pto.tinsert` (it round-trips through `--emit-pto-ir`) but
  // emits a C++ call that pto-isa resolves to the *unscaled* overload, because
  // every scalar operand it emits is `int64_t` and the two wrappers differ only
  // in their parameter types:
  //     TINSERT(Dst&, Src&,           uint16_t row, uint16_t col, WaitEvents&...)
  //     TINSERT(Dst&, Src&, uint64_t, uint16_t row, uint16_t col, WaitEvents&...)
  // For `TINSERT<Mat, Acc, relu>(dst, src, scale_i64, row_i64, col_i64)` the
  // first is the better match -- `col` binds to the event pack by identity
  // while the second needs a conversion there -- so the scale lands in `row`
  // (truncated to 0) and the column index is waited on as an event id. The
  // resulting cast-path `static_assert` in a2a3/common.hpp is the symptom; the
  // silent misbinding is the defect, and no `.pto` spelling avoids it. Acc->GM
  // is unaffected: `TSTORE` has no index operands for the scale to slide into,
  // and it is device-verified in tests/st/runtime/ops/test_fixpipe_epilogue.py.
  [[nodiscard]] bool SupportsFixpipePreQuant(const DataType& src, const DataType& dst,
                                             FixpipeDest dest) const override {
    if (dest == FixpipeDest::kMat) return false;
    const bool dst_is_byte = dst == DataType::INT8 || dst == DataType::UINT8;
    if (src == DataType::INT32) return dst_is_byte || dst == DataType::FP16;
    if (src == DataType::FP32) return dst_is_byte;
    return false;
  }

  [[nodiscard]] ir::TileView BuildCrossCoreTransferView(ir::MemorySpace dest_ms,
                                                        const ir::TileView& original_view) const override;

  /// Native single-instruction `pto.tcvt` pairs for this architecture
  /// (ISA Supported Conversions, pto-isa tcvt docs).
  [[nodiscard]] const TcvtAdjacency& GetTcvtAdjacency() const override;

  [[nodiscard]] uint32_t GetGmAccessGranularityBytes() const override { return 512; }
  [[nodiscard]] uint32_t GetL2CacheLineBytes() const override { return 512; }
  [[nodiscard]] uint32_t GetRecommendedInnermostDimBytes() const override { return 512; }

  // L0 capacity (matches Create910BSoC AIC core memory layout).
  [[nodiscard]] uint32_t GetL0aCapacityBytes() const override { return 64ULL * 1024; }
  [[nodiscard]] uint32_t GetL0bCapacityBytes() const override { return 64ULL * 1024; }
  [[nodiscard]] uint32_t GetL0cCapacityBytes() const override { return 128ULL * 1024; }
  [[nodiscard]] uint32_t GetBiasCapacityBytes() const override { return 1ULL * 1024; }
  [[nodiscard]] bool SupportsMatToBiasMove(const DataType& source_dtype,
                                           const DataType& bias_dtype) const override {
    return (source_dtype == DataType::INT32 && bias_dtype == DataType::INT32) ||
           (bias_dtype == DataType::FP32 &&
            (source_dtype == DataType::FP32 || source_dtype == DataType::FP16));
  }
  [[nodiscard]] uint64_t GetMatCapacityBytes() const override { return 512ULL * 1024; }
  [[nodiscard]] int GetL0cMAlignment(const DataType& accumulator_dtype) const override {
    return accumulator_dtype == DataType::INT32 ? 32 : GetL0FractalAlignment();
  }

 private:
  Ascend910BHandler() = default;
};

}  // namespace backend
}  // namespace pypto

#endif  // PYPTO_BACKEND_910B_BACKEND_910B_HANDLER_H_
