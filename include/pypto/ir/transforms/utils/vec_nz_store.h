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
#ifndef PYPTO_IR_TRANSFORMS_UTILS_VEC_NZ_STORE_H_
#define PYPTO_IR_TRANSFORMS_UTILS_VEC_NZ_STORE_H_

#include <cstddef>
#include <cstdint>

#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/phase.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/type.h"

namespace pypto::ir {

// TStoreVecNZ copies complete 16x32 byte boxes using the source's physical
// panel pitch. Keep the same checks before logical NZ blocking and before
// final backend emission so accepted stores use identical layout contracts.
inline void ValidateVecFp8NzStore(const CallPtr& call, const TileTypePtr& source,
                                  const TensorTypePtr& destination, const backend::BackendHandler* handler) {
  CHECK_SPAN(handler && handler->SupportsVecFp8NzGmStore(), call->span_)
      << "Vec NZ tile.store requires the A5 FP8 NZ store capability";
  CHECK_SPAN(source->GetMemorySpace() == MemorySpace::Vec && source->dtype_ == DataType::FP8E4M3FN &&
                 destination->dtype_ == DataType::FP8E4M3FN,
             call->span_)
      << "Vec NZ tile.store supports only FP8E4M3FN Vec tiles into matching NZ tensors";
  const auto view = tile_view_semantics::GetEffectiveTileView(*source);
  CHECK_SPAN(source->shape_.size() == 2 && view.valid_shape.size() == 2 &&
                 view.blayout == TileLayout::col_major && view.slayout == TileLayout::row_major &&
                 view.fractal == 512 && view.compact == CompactMode::null,
             call->span_)
      << "Vec NZ tile.store requires a standard 512-byte NZ tile with compact=null";
  for (size_t axis = 0; axis < 2; ++axis) {
    const auto physical = As<ConstInt>(source->shape_[axis]);
    const auto valid = As<ConstInt>(view.valid_shape[axis]);
    const int64_t alignment = axis == 0 ? 16 : 32;
    CHECK_SPAN(physical && valid && physical->value_ > 0 && valid->value_ > 0 &&
                   valid->value_ <= physical->value_ && physical->value_ % alignment == 0 &&
                   valid->value_ % alignment == 0,
               call->span_)
        << "Vec NZ tile.store requires static positive complete 16x32 boxes in the source valid_shape";
  }
  CHECK_SPAN(call->GetKwarg<int>("atomic", 0) == static_cast<int>(AtomicType::kNone) &&
                 !GetOptionalDoubleKwarg(call->kwargs_, "pre_quant") &&
                 !call->GetKwarg<bool>("pre_relu", false) &&
                 call->GetKwarg<int>("st_phase", static_cast<int>(STPhase::kUnspecified)) ==
                     static_cast<int>(STPhase::kUnspecified),
             call->span_)
      << "Vec NZ tile.store requires a plain store without atomic, epilogue, or st_phase options";
}

}  // namespace pypto::ir
#endif  // PYPTO_IR_TRANSFORMS_UTILS_VEC_NZ_STORE_H_
