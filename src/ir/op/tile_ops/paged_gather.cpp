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
 * @file paged_gather.cpp
 * @brief tile.gather_row — load one GM row directly into a sub-region of an
 *        on-chip (L1/Mat or UB/Vec) tile.
 *
 * This is the per-row primitive of the paged-gather lowering. It lowers to
 * ``pto.subview`` (of the destination accumulator) + ``pto.partition_view`` (of
 * the GM source) + ``pto.tload`` writing GM -> the subview directly, with **no
 * pto.tmov**. Filling an L1 (Mat) tile on a2a3 is only possible via GM->Mat
 * ``pto.tload`` (MAT->MAT tmov is unsupported), so the per-row gather must write
 * straight into the accumulator sub-region — mirroring CANN's TGatherInL1.
 *
 * Destination-Preserve-Source (DPS): the op writes into its first argument in
 * place (``set_output_reuses_input(0)``), so a loop-carried accumulator is
 * updated row by row without copying the whole tile.
 */

#include <any>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

static TypePtr DeduceTileGatherRowType(const std::vector<ExprPtr>& args,
                                       const std::vector<std::pair<std::string, std::any>>& /*kwargs*/,
                                       const std::string& op_name) {
  CHECK(args.size() == 5) << "The operator " << op_name
                          << " requires 5 arguments (dst, src, dst_offset, src_offset, shapes), but got "
                          << args.size();

  auto dst_type = As<TileType>(args[0]->GetType());
  CHECK(dst_type) << "The operator " << op_name << " requires dst to be a TileType, but got "
                  << args[0]->GetType()->TypeName();

  auto src_type = As<TensorType>(args[1]->GetType());
  CHECK(src_type) << "The operator " << op_name << " requires src to be a TensorType (GM), but got "
                  << args[1]->GetType()->TypeName();
  CHECK(dst_type->dtype_ == src_type->dtype_)
      << "The operator " << op_name << " requires dst and src to share dtype, but got "
      << dst_type->dtype_.ToString() << " and " << src_type->dtype_.ToString();

  // DPS: the result is the destination accumulator, written in place.
  return dst_type;
}

REGISTER_OP("tile.gather_row")
    .set_op_category("TileOp")
    .set_description(
        "Load one GM row directly into a sub-region of an on-chip tile (Mat/L1 or Vec/UB) via "
        "pto.subview + pto.tload (GM->on-chip, no pto.tmov). DPS: writes into dst in place. Used as "
        "the per-row primitive of the paged-gather lowering (CANN TGatherInL1 equivalent).")
    .add_argument("dst", "Destination accumulator tile (Mat or Vec)")
    .add_argument("src", "Source tensor in GM (TensorType)")
    .add_argument("dst_offset", "Destination [row, col] offset (TupleType of ScalarType)")
    .add_argument("src_offset", "Source [row, col] offset within the GM tensor (TupleType of ScalarType)")
    .add_argument("shapes", "GM row window shape [r, c] (TupleType of ScalarType)")
    .set_attr<bool>("transpose")
    .set_output_reuses_input(0)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileGatherRowType(args, kwargs, "tile.gather_row");
    });

}  // namespace ir
}  // namespace pypto
