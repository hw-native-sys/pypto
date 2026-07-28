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

/**
 * @file mx.cpp
 * @brief MX (block-scale) tile ops: tget_scale_addr
 */

#include <any>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

namespace {

TypePtr DeduceTileTGetScaleAddrType(const std::vector<ExprPtr>& args,
                                    const std::vector<std::pair<std::string, std::any>>& kwargs,
                                    const std::string& op_name) {
  CHECK(args.size() == 2) << "The operator " << op_name
                          << " requires exactly 2 arguments (dst_scale, src), but got " << args.size();
  auto dst_type = As<TileType>(args[0]->GetType());
  auto src_type = As<TileType>(args[1]->GetType());
  CHECK(dst_type) << "The operator " << op_name << " requires dst_scale to be a TileType";
  CHECK(src_type) << "The operator " << op_name << " requires src to be a TileType";
  CHECK(dst_type->dtype_ == DataType::FP8E8M0 || dst_type->dtype_ == DataType::UINT8)
      << "The operator " << op_name
      << " requires dst_scale dtype FP8E8M0 (or raw UINT8 from host prequant), but got "
      << dst_type->dtype_.ToString();
  // Address-binding op: result reuses dst_scale tile type (same shape/dtype).
  return std::make_shared<TileType>(dst_type->shape_, dst_type->dtype_, /*memref=*/std::nullopt,
                                    dst_type->tile_view_, dst_type->memory_space_);
}

}  // namespace

REGISTER_OP("tile.tget_scale_addr")
    .set_op_category("TileOp")
    .set_description(
        "Bind MX scale-tile address from a Left/Right data tile (A5): "
        "dst_addr = src_addr >> SHIFT_MX_ADDR. Maps to pto.tget_scale_addr.")
    .add_argument("dst_scale", "Destination scale tile (FP8E8M0, LeftScale/RightScale)")
    .add_argument("src", "Source Left/Right data tile whose address is scaled")
    .set_input_memory(0, {MemorySpace::LeftScale, MemorySpace::RightScale, MemorySpace::Mat})
    .set_input_memory(1, {MemorySpace::Left, MemorySpace::Right, MemorySpace::Mat})
    .set_output_memory(MemorySpace::LeftScale)  // overridden by reuse of input 0 when present
    .set_output_reuses_input(0)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileTGetScaleAddrType(args, kwargs, "tile.tget_scale_addr");
    });

}  // namespace ir
}  // namespace pypto
