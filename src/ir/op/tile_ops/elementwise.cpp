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
 * @file elementwise.cpp
 * @brief Element-wise tile operations (Mul, Add, Div, Sub, and scalar variants)
 *
 * This file implements element-wise tile operations that support
 * 2D tiles (at most 2 dimensions) with 2D broadcasting.
 * Operations are divided into:
 * - Tile-Tile operations (mul, add, div, sub): TileType + TileType
 * - Tile-Scalar operations (muls, adds, divs, subs): TileType + ScalarType
 */

#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

constexpr int64_t kPackedPredicateBitsPerByte = 8;

static ExprPtr MakeIndexConst(int64_t value, const Span& span = Span::unknown()) {
  return std::make_shared<ConstInt>(value, DataType::INDEX, span);
}

static ExprPtr MakeCeilDivIndex(const ExprPtr& value, int64_t divisor) {
  if (auto const_value = As<ConstInt>(value)) {
    return MakeIndexConst((const_value->value_ + divisor - 1) / divisor, value->span_);
  }
  return MakeFloorDiv(MakeAdd(value, MakeIndexConst(divisor - 1, value->span_), value->span_),
                      MakeIndexConst(divisor, value->span_), value->span_);
}

static ExprPtr MakeRoundUpIndex(const ExprPtr& value, int64_t alignment) {
  if (auto const_value = As<ConstInt>(value)) {
    const int64_t rounded = ((const_value->value_ + alignment - 1) / alignment) * alignment;
    return MakeIndexConst(rounded, value->span_);
  }
  return MakeMul(MakeCeilDivIndex(value, alignment), MakeIndexConst(alignment, value->span_), value->span_);
}

static bool IsTDivDataType(DataType dtype) {
  return dtype == DataType::INT16 || dtype == DataType::INT32 || dtype == DataType::FP16 ||
         dtype == DataType::FP32;
}

static bool IsTSubsDataType(DataType dtype) {
  return dtype == DataType::INT8 || dtype == DataType::INT16 || dtype == DataType::INT32 ||
         dtype == DataType::FP16 || dtype == DataType::FP32 || dtype == DataType::BF16;
}

static bool IsTSelsDataType(DataType dtype) {
  return dtype == DataType::INT8 || dtype == DataType::UINT8 || dtype == DataType::INT16 ||
         dtype == DataType::UINT16 || dtype == DataType::INT32 || dtype == DataType::UINT32 ||
         dtype == DataType::FP16 || dtype == DataType::FP32;
}

static DataType GetTSelsScalarDataType(DataType src_dtype) {
  if (src_dtype == DataType::UINT8) return DataType::INT8;
  if (src_dtype == DataType::UINT16) return DataType::INT16;
  if (src_dtype == DataType::UINT32) return DataType::INT32;
  return src_dtype;
}

static bool IsTSelsMaskDataType(DataType dtype) {
  return dtype == DataType::INT8 || dtype == DataType::UINT8 || dtype == DataType::INT16 ||
         dtype == DataType::UINT16 || dtype == DataType::INT32 || dtype == DataType::UINT32;
}

static std::shared_ptr<TileType> MakePackedPredicateTileType(
    const std::vector<ExprPtr>& logical_shape, const std::shared_ptr<const TileType>& source_tile_type) {
  INTERNAL_CHECK(!logical_shape.empty())
      << "tile.cmp/tile.cmps require a non-empty tile shape for packed predicate mask inference";

  constexpr int64_t kA2A3PredicateColAlignment = 32;

  const size_t col_axis = logical_shape.size() - 1;
  std::vector<ExprPtr> mask_shape = logical_shape;
  mask_shape[col_axis] = MakeRoundUpIndex(
      MakeCeilDivIndex(logical_shape[col_axis], kPackedPredicateBitsPerByte), kA2A3PredicateColAlignment);

  auto logical_valid_shape = GetValidShape(source_tile_type);
  TileView tile_view;
  tile_view.valid_shape = logical_valid_shape;
  tile_view.valid_shape[col_axis] =
      MakeCeilDivIndex(logical_valid_shape[col_axis], kPackedPredicateBitsPerByte);
  InheritTileViewLayout(tile_view, source_tile_type);
  return std::make_shared<TileType>(mask_shape, DataType::UINT8, std::nullopt, tile_view);
}

// Forward declarations: trem/trems reuse the tmp-carrying deducers defined
// further down (also used by xor/xors), so they must be visible here.
TypePtr DeduceTileOpTernaryType(const std::vector<ExprPtr>& args,
                                const std::vector<std::pair<std::string, std::any>>& kwargs,
                                const std::string& op_name, bool require_int);
TypePtr DeduceTileOpTileScalarTileType(const std::vector<ExprPtr>& args,
                                       const std::vector<std::pair<std::string, std::any>>& kwargs,
                                       const std::string& op_name);

TypePtr DeduceTileOpElementwiseBinaryType(const std::vector<ExprPtr>& args,
                                          const std::vector<std::pair<std::string, std::any>>& kwargs,
                                          const std::string& op_name, bool require_int = false,
                                          bool require_tdiv_contract = false) {
  CHECK(args.size() == 2) << "The operator " << op_name << " requires exactly 2 arguments, but got "
                          << args.size();

  // Both arguments must be TileType
  auto tile_type1 = As<TileType>(args[0]->GetType());
  auto tile_type2 = As<TileType>(args[1]->GetType());

  CHECK(tile_type1) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                    << args[0]->GetType()->TypeName();
  CHECK(tile_type2) << "The operator " << op_name << " requires second argument to be a TileType, but got "
                    << args[1]->GetType()->TypeName();

  if (require_int) {
    CHECK(tile_type1->dtype_.IsInt())
        << "The operator " << op_name << " requires integer tile dtype, but got "
        << tile_type1->dtype_.ToString();
    CHECK(tile_type2->dtype_.IsInt())
        << "The operator " << op_name << " requires integer tile dtype, but got "
        << tile_type2->dtype_.ToString();
  }

  if (require_tdiv_contract) {
    CHECK(tile_type1->dtype_ == tile_type2->dtype_)
        << "The operator " << op_name << " requires src0, src1, and dst to have the same dtype, but got "
        << tile_type1->dtype_.ToString() << " and " << tile_type2->dtype_.ToString();
    CHECK(IsTDivDataType(tile_type1->dtype_))
        << "The operator " << op_name << " requires dtype in {INT16, INT32, FP16, FP32}, but got "
        << tile_type1->dtype_.ToString();

    CHECK(tile_type1->shape_.size() == tile_type2->shape_.size())
        << "The operator " << op_name
        << " requires src0, src1, and dst to have the same physical shape rank, but got "
        << tile_type1->shape_.size() << " and " << tile_type2->shape_.size();
    for (size_t i = 0; i < tile_type1->shape_.size(); ++i) {
      CHECK(DimensionsEqual(tile_type1->shape_[i], tile_type2->shape_[i]))
          << "The operator " << op_name
          << " requires src0, src1, and dst to have the same physical shape, but dimension " << i
          << " differs; got src0 shape " << FormatShape(tile_type1->shape_) << " and src1 shape "
          << FormatShape(tile_type2->shape_);
    }

    const auto valid_shape1 = GetValidShape(tile_type1);
    const auto valid_shape2 = GetValidShape(tile_type2);
    CHECK(valid_shape1.size() == valid_shape2.size())
        << "The operator " << op_name
        << " requires src0, src1, and dst to have the same valid_shape rank, but got " << valid_shape1.size()
        << " and " << valid_shape2.size();
    for (size_t i = 0; i < valid_shape1.size(); ++i) {
      CHECK(ProveValidExtentEqual(valid_shape1[i], valid_shape2[i]) == ProofResult::kTrue)
          << "The operator " << op_name
          << " requires src0, src1, and dst to have the same valid_shape, but dimension " << i
          << " differs; got src0 valid_shape " << FormatShape(valid_shape1) << " and src1 valid_shape "
          << FormatShape(valid_shape2);
    }

    TileView tile_view;
    tile_view.valid_shape = valid_shape1;
    InheritTileViewLayout(tile_view, tile_type1);
    return std::make_shared<TileType>(tile_type1->shape_, tile_type1->dtype_, std::nullopt, tile_view);
  }

  // Use broadcasting
  auto result_dtype = PromoteDataTypes(tile_type1->dtype_, tile_type2->dtype_);
  CHECK(result_dtype) << "The operator " << op_name << " requires compatible data types, but got "
                      << args[0]->GetType()->TypeName() << " and " << args[1]->GetType()->TypeName();

  auto broadcast_result = BroadcastShapes(tile_type1->shape_, tile_type2->shape_);
  CHECK(broadcast_result.success) << "The operator " << op_name << " requires compatible shapes, but got "
                                  << FormatShape(tile_type1->shape_) << " and "
                                  << FormatShape(tile_type2->shape_);

  // TODO(YunjiQin): assumes both src tiles have the same valid_shape; may need refinement
  // for cases where lhs and rhs have different valid_shape values (e.g. after broadcasting).
  TileView tile_view;
  tile_view.valid_shape = GetValidShape(tile_type1);
  InheritTileViewLayout(tile_view, tile_type1);
  return std::make_shared<TileType>(broadcast_result.shape, *result_dtype, std::nullopt, tile_view);
}

// Tile-tile shift ops (shl, shr): RHS is the shift amount, result type equals LHS tile type,
// consistent with scalar variants (shls/shrs) which preserve the LHS tile dtype.
TypePtr DeduceTileOpShiftBinaryType(const std::vector<ExprPtr>& args,
                                    const std::vector<std::pair<std::string, std::any>>& kwargs,
                                    const std::string& op_name) {
  CHECK(args.size() == 2) << "The operator " << op_name << " requires exactly 2 arguments, but got "
                          << args.size();

  auto tile_type1 = As<TileType>(args[0]->GetType());
  auto tile_type2 = As<TileType>(args[1]->GetType());
  CHECK(tile_type1) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                    << args[0]->GetType()->TypeName();
  CHECK(tile_type2) << "The operator " << op_name << " requires second argument to be a TileType, but got "
                    << args[1]->GetType()->TypeName();
  CHECK(tile_type1->dtype_.IsInt()) << "The operator " << op_name << " requires integer tile dtype, but got "
                                    << tile_type1->dtype_.ToString();
  CHECK(tile_type2->dtype_.IsInt()) << "The operator " << op_name
                                    << " requires integer shift tile dtype, but got "
                                    << tile_type2->dtype_.ToString();

  auto broadcast_result = BroadcastShapes(tile_type1->shape_, tile_type2->shape_);
  CHECK(broadcast_result.success) << "The operator " << op_name << " requires compatible shapes";

  // TODO(YunjiQin): assumes both src tiles have the same valid_shape; may need refinement
  // for cases where lhs and rhs have different valid_shape values (e.g. after broadcasting).
  TileView tile_view;
  tile_view.valid_shape = GetValidShape(tile_type1);
  InheritTileViewLayout(tile_view, tile_type1);
  return std::make_shared<TileType>(broadcast_result.shape, tile_type1->dtype_, std::nullopt, tile_view);
}

TypePtr DeduceTileOpScalarBinaryType(const std::vector<ExprPtr>& args,
                                     const std::vector<std::pair<std::string, std::any>>& kwargs,
                                     const std::string& op_name) {
  CHECK(args.size() == 2) << "The operator " << op_name << " requires exactly 2 arguments, but got "
                          << args.size();

  // First argument must be TileType
  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  // Second argument MUST be ScalarType
  auto scalar_type = As<ScalarType>(args[1]->GetType());
  CHECK(scalar_type) << "The operator " << op_name << " requires second argument to be a ScalarType, but got "
                     << args[1]->GetType()->TypeName();

  // Result preserves the tile's element type. The hardware scalar instructions
  // (e.g. pto.tmuls) require src and dst to share the same element type; the
  // scalar operand is implicitly narrowed to match the tile dtype at runtime.
  TileView tile_view;
  tile_view.valid_shape = GetValidShape(tile_type);
  InheritTileViewLayout(tile_view, tile_type);
  return std::make_shared<TileType>(tile_type->shape_, tile_type->dtype_, std::nullopt, tile_view);
}

TypePtr DeduceTileSubsType(const std::vector<ExprPtr>& args,
                           const std::vector<std::pair<std::string, std::any>>& kwargs,
                           const std::string& op_name) {
  auto result_type = DeduceTileOpScalarBinaryType(args, kwargs, op_name);
  auto tile_type = As<TileType>(args[0]->GetType());
  auto scalar_type = As<ScalarType>(args[1]->GetType());
  CHECK(IsTSubsDataType(tile_type->dtype_))
      << "The operator " << op_name
      << " requires tile dtype in {INT8, INT16, INT32, FP16, FP32, BF16}, but got "
      << tile_type->dtype_.ToString();
  CHECK(IsTSubsDataType(scalar_type->dtype_))
      << "The operator " << op_name
      << " requires scalar dtype in {INT8, INT16, INT32, FP16, FP32, BF16}, but got "
      << scalar_type->dtype_.ToString();
  return result_type;
}

TypePtr DeduceTileOpIntScalarBinaryType(const std::vector<ExprPtr>& args,
                                        const std::vector<std::pair<std::string, std::any>>& kwargs,
                                        const std::string& op_name) {
  CHECK(args.size() == 2) << "The operator " << op_name << " requires exactly 2 arguments, but got "
                          << args.size();

  // First argument must be TileType with integer dtype.
  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();
  CHECK(tile_type->dtype_.IsInt()) << "The operator " << op_name << " requires integer tile dtype, but got "
                                   << tile_type->dtype_.ToString();

  // Second argument must be ScalarType with an integer dtype per ISA spec:
  //   %dst = tshls/tshrs/tands/tors %src, %scalar : !pto.tile<...>, i32
  // The IR allows any integer width (INT8/16/32/64, UINT variants); codegen casts to i32.
  auto scalar_type = As<ScalarType>(args[1]->GetType());
  CHECK(scalar_type) << "The operator " << op_name << " requires second argument to be a ScalarType, but got "
                     << args[1]->GetType()->TypeName();
  CHECK(scalar_type->dtype_.IsInt()) << "The operator " << op_name
                                     << " requires shift/bitwise scalar to be an integer type, but got "
                                     << scalar_type->dtype_.ToString();

  // Result has the same shape and dtype as the input tile; the shift amount does not change element type.
  TileView tile_view;
  tile_view.valid_shape = GetValidShape(tile_type);
  InheritTileViewLayout(tile_view, tile_type);
  return std::make_shared<TileType>(tile_type->shape_, tile_type->dtype_, std::nullopt, tile_view);
}

// ============================================================================
// Op Registration
// ============================================================================

REGISTER_OP("tile.mul")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("Element-wise multiplication of two tiles with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpElementwiseBinaryType(args, kwargs, "tile.mul");
    });

REGISTER_OP("tile.add")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("Element-wise addition of two tiles with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpElementwiseBinaryType(args, kwargs, "tile.add");
    });

REGISTER_OP("tile.div")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("Element-wise division of two tiles with matching physical and valid shapes")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .set_attr<bool>("high_precision")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      auto result_type = DeduceTileOpElementwiseBinaryType(args, kwargs, "tile.div", false, true);
      auto result_tile_type = As<TileType>(result_type);
      CHECK(!GetKwargOr<bool>(kwargs, "high_precision", false) || result_tile_type->dtype_.IsFloat())
          << "The operator tile.div supports high_precision only for FP16 or FP32 because the PTOAS "
             "high-precision template does not implement integer division";
      return result_type;
    });

REGISTER_OP("tile.sub")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("Element-wise subtraction of two tiles with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpElementwiseBinaryType(args, kwargs, "tile.sub");
    });

REGISTER_OP("tile.maximum")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("Element-wise maximum of two tiles with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpElementwiseBinaryType(args, kwargs, "tile.maximum");
    });

REGISTER_OP("tile.minimum")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("Element-wise minimum of two tiles with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpElementwiseBinaryType(args, kwargs, "tile.minimum");
    });

REGISTER_OP("tile.rem")
    .set_op_category("TileOp")
    .set_description("Element-wise remainder (modulo) of two tiles with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .add_argument("tmp", "Temporary tile (TileType) required by the hardware")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_input_memory(2, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpTernaryType(args, kwargs, "tile.rem", false);
    });

// Partial-combine binary ops: elementwise over dst's valid region, but where
// only one source is valid at an element the result copies that source. Shape
// is identical to a plain elementwise binary op; the "partial" behaviour is a
// runtime valid-region effect, so type deduction is the standard binary one.
REGISTER_OP("tile.part_add")
    .set_op_category("TileOp")
    .set_description("Partial element-wise add of two tiles (copies the only valid input)")
    .add_argument("src0", "First source tile (TileType)")
    .add_argument("src1", "Second source tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpElementwiseBinaryType(args, kwargs, "tile.part_add");
    });

REGISTER_OP("tile.part_mul")
    .set_op_category("TileOp")
    .set_description("Partial element-wise multiply of two tiles (copies the only valid input)")
    .add_argument("src0", "First source tile (TileType)")
    .add_argument("src1", "Second source tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpElementwiseBinaryType(args, kwargs, "tile.part_mul");
    });

REGISTER_OP("tile.part_max")
    .set_op_category("TileOp")
    .set_description("Partial element-wise max of two tiles (copies the only valid input)")
    .add_argument("src0", "First source tile (TileType)")
    .add_argument("src1", "Second source tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpElementwiseBinaryType(args, kwargs, "tile.part_max");
    });

REGISTER_OP("tile.part_min")
    .set_op_category("TileOp")
    .set_description("Partial element-wise min of two tiles (copies the only valid input)")
    .add_argument("src0", "First source tile (TileType)")
    .add_argument("src1", "Second source tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpElementwiseBinaryType(args, kwargs, "tile.part_min");
    });

REGISTER_OP("tile.fmod")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("Element-wise floating-point remainder of two tiles with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    // The hardware kernel overwrites src0 mid-computation (dst=src0/src1) but
    // still needs the original src0 for the final subtraction, so dst must not
    // alias any input buffer.
    .not_inplace_safe()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpElementwiseBinaryType(args, kwargs, "tile.fmod");
    });

REGISTER_OP("tile.muls")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("Element-wise multiplication of tile and scalar")
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpScalarBinaryType(args, kwargs, "tile.muls");
    });

REGISTER_OP("tile.adds")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("Element-wise addition of tile and scalar")
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpScalarBinaryType(args, kwargs, "tile.adds");
    });

REGISTER_OP("tile.divs")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("Element-wise division of tile and scalar")
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpScalarBinaryType(args, kwargs, "tile.divs");
    });

REGISTER_OP("tile.subs")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("Element-wise subtraction of tile and scalar")
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileSubsType(args, kwargs, "tile.subs");
    });

REGISTER_OP("tile.rems")
    .set_op_category("TileOp")
    .set_description("Element-wise remainder (modulo) of tile and scalar")
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .add_argument("tmp", "Temporary tile (TileType) required by the hardware")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(2, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpTileScalarTileType(args, kwargs, "tile.rems");
    });

REGISTER_OP("tile.fmods")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("Element-wise floating-point remainder of tile and scalar")
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    // Same hazard as tile.fmod: the kernel clobbers src0 before its final use,
    // so dst must not alias the input buffer.
    .not_inplace_safe()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpScalarBinaryType(args, kwargs, "tile.fmods");
    });

REGISTER_OP("tile.shl")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("Element-wise bitwise left shift of two tiles with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpShiftBinaryType(args, kwargs, "tile.shl");
    });

REGISTER_OP("tile.shls")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("Element-wise bitwise left shift of tile and scalar")
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpIntScalarBinaryType(args, kwargs, "tile.shls");
    });

REGISTER_OP("tile.shr")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("Element-wise bitwise right shift of two tiles with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpShiftBinaryType(args, kwargs, "tile.shr");
    });

REGISTER_OP("tile.shrs")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("Element-wise bitwise right shift of tile and scalar")
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpIntScalarBinaryType(args, kwargs, "tile.shrs");
    });

REGISTER_OP("tile.maximums")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("Element-wise maximum of tile and scalar")
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpScalarBinaryType(args, kwargs, "tile.maximums");
    });

REGISTER_OP("tile.minimums")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("Element-wise minimum of tile and scalar")
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpScalarBinaryType(args, kwargs, "tile.minimums");
    });

REGISTER_OP("tile.and")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("Element-wise bitwise AND of two tiles with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpElementwiseBinaryType(args, kwargs, "tile.and", true);
    });

REGISTER_OP("tile.ands")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("Element-wise bitwise AND of tile and scalar")
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .not_inplace_safe()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpIntScalarBinaryType(args, kwargs, "tile.ands");
    });

REGISTER_OP("tile.or")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("Element-wise bitwise OR of two tiles with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpElementwiseBinaryType(args, kwargs, "tile.or", true);
    });

REGISTER_OP("tile.ors")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("Element-wise bitwise OR of tile and scalar")
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .not_inplace_safe()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpIntScalarBinaryType(args, kwargs, "tile.ors");
    });

// Tile-tile ternary ops with a tmp buffer as the third argument.
// When require_int is true (bitwise ops like xor), both tile dtypes must be integer.
TypePtr DeduceTileOpTernaryType(const std::vector<ExprPtr>& args,
                                const std::vector<std::pair<std::string, std::any>>& kwargs,
                                const std::string& op_name, bool require_int = false) {
  CHECK(args.size() == 3) << "The operator " << op_name << " requires exactly 3 arguments, but got "
                          << args.size();

  auto tile_type1 = As<TileType>(args[0]->GetType());
  auto tile_type2 = As<TileType>(args[1]->GetType());
  CHECK(tile_type1) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                    << args[0]->GetType()->TypeName();
  CHECK(tile_type2) << "The operator " << op_name << " requires second argument to be a TileType, but got "
                    << args[1]->GetType()->TypeName();
  CHECK(As<TileType>(args[2]->GetType()))
      << "The operator " << op_name << " requires third argument (tmp) to be a TileType, but got "
      << args[2]->GetType()->TypeName();

  if (require_int) {
    CHECK(tile_type1->dtype_.IsInt())
        << "The operator " << op_name << " requires integer tile dtype, but got "
        << tile_type1->dtype_.ToString();
    CHECK(tile_type2->dtype_.IsInt())
        << "The operator " << op_name << " requires integer tile dtype, but got "
        << tile_type2->dtype_.ToString();
  }

  auto result_dtype = PromoteDataTypes(tile_type1->dtype_, tile_type2->dtype_);
  CHECK(result_dtype) << "The operator " << op_name << " requires compatible data types";
  auto broadcast_result = BroadcastShapes(tile_type1->shape_, tile_type2->shape_);
  CHECK(broadcast_result.success) << "The operator " << op_name << " requires compatible shapes";

  // TODO(YunjiQin): assumes both src tiles have the same valid_shape; may need refinement
  // for cases where lhs and rhs have different valid_shape values (e.g. after broadcasting).
  TileView tile_view;
  tile_view.valid_shape = GetValidShape(tile_type1);
  InheritTileViewLayout(tile_view, tile_type1);
  return std::make_shared<TileType>(broadcast_result.shape, *result_dtype, std::nullopt, tile_view);
}

// All three tiles are real inputs (addc, subc): promote dtype and broadcast shape across all three.
TypePtr DeduceTileOpTriTileType(const std::vector<ExprPtr>& args,
                                const std::vector<std::pair<std::string, std::any>>& kwargs,
                                const std::string& op_name) {
  CHECK(args.size() == 3) << "The operator " << op_name << " requires exactly 3 arguments, but got "
                          << args.size();

  auto tile_type1 = As<TileType>(args[0]->GetType());
  auto tile_type2 = As<TileType>(args[1]->GetType());
  auto tile_type3 = As<TileType>(args[2]->GetType());
  CHECK(tile_type1) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                    << args[0]->GetType()->TypeName();
  CHECK(tile_type2) << "The operator " << op_name << " requires second argument to be a TileType, but got "
                    << args[1]->GetType()->TypeName();
  CHECK(tile_type3) << "The operator " << op_name << " requires third argument to be a TileType, but got "
                    << args[2]->GetType()->TypeName();

  auto result_dtype12 = PromoteDataTypes(tile_type1->dtype_, tile_type2->dtype_);
  CHECK(result_dtype12) << "The operator " << op_name << " requires compatible data types";
  auto result_dtype = PromoteDataTypes(*result_dtype12, tile_type3->dtype_);
  CHECK(result_dtype) << "The operator " << op_name << " requires compatible data types";

  auto broadcast12 = BroadcastShapes(tile_type1->shape_, tile_type2->shape_);
  CHECK(broadcast12.success) << "The operator " << op_name << " requires compatible shapes";
  auto broadcast_result = BroadcastShapes(broadcast12.shape, tile_type3->shape_);
  CHECK(broadcast_result.success) << "The operator " << op_name << " requires compatible shapes";

  // TODO(YunjiQin): assumes all src tiles have the same valid_shape; may need refinement
  // for cases where tiles have different valid_shape values (e.g. after broadcasting).
  TileView tile_view;
  tile_view.valid_shape = GetValidShape(tile_type1);
  InheritTileViewLayout(tile_view, tile_type1);
  return std::make_shared<TileType>(broadcast_result.shape, *result_dtype, std::nullopt, tile_view);
}

// (Tile, Scalar, Tile) pattern (addsc, subsc): any scalar type, promote output from all three inputs.
TypePtr DeduceTileOpTileScalarTileType(const std::vector<ExprPtr>& args,
                                       const std::vector<std::pair<std::string, std::any>>& kwargs,
                                       const std::string& op_name) {
  CHECK(args.size() == 3) << "The operator " << op_name << " requires exactly 3 arguments, but got "
                          << args.size();

  auto tile_type1 = As<TileType>(args[0]->GetType());
  CHECK(tile_type1) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                    << args[0]->GetType()->TypeName();

  auto scalar_type = As<ScalarType>(args[1]->GetType());
  CHECK(scalar_type) << "The operator " << op_name << " requires second argument to be a ScalarType, but got "
                     << args[1]->GetType()->TypeName();

  auto tile_type2 = As<TileType>(args[2]->GetType());
  CHECK(tile_type2) << "The operator " << op_name << " requires third argument to be a TileType, but got "
                    << args[2]->GetType()->TypeName();

  auto result_dtype12 = PromoteDataTypes(tile_type1->dtype_, scalar_type->dtype_);
  CHECK(result_dtype12) << "The operator " << op_name << " requires compatible data types";
  auto result_dtype = PromoteDataTypes(*result_dtype12, tile_type2->dtype_);
  CHECK(result_dtype) << "The operator " << op_name << " requires compatible data types";

  auto broadcast_result = BroadcastShapes(tile_type1->shape_, tile_type2->shape_);
  CHECK(broadcast_result.success) << "The operator " << op_name << " requires compatible shapes";

  // TODO(YunjiQin): assumes both src tiles have the same valid_shape; may need refinement
  // for cases where lhs and rhs tiles have different valid_shape values (e.g. after broadcasting).
  TileView tile_view;
  tile_view.valid_shape = GetValidShape(tile_type1);
  InheritTileViewLayout(tile_view, tile_type1);
  return std::make_shared<TileType>(broadcast_result.shape, *result_dtype, std::nullopt, tile_view);
}

TypePtr DeduceTileOpXorScalarType(const std::vector<ExprPtr>& args,
                                  const std::vector<std::pair<std::string, std::any>>& kwargs,
                                  const std::string& op_name) {
  CHECK(args.size() == 3) << "The operator " << op_name << " requires exactly 3 arguments, but got "
                          << args.size();

  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();
  CHECK(tile_type->dtype_.IsInt()) << "The operator " << op_name << " requires integer tile dtype, but got "
                                   << tile_type->dtype_.ToString();

  // Second argument must be ScalarType with an integer dtype per ISA spec:
  //   %dst = txors %src, %scalar : !pto.tile<...>, i32
  // The IR allows any integer width (INT8/16/32/64, UINT variants); codegen casts to i32.
  auto scalar_type = As<ScalarType>(args[1]->GetType());
  CHECK(scalar_type) << "The operator " << op_name << " requires second argument to be a ScalarType, but got "
                     << args[1]->GetType()->TypeName();
  CHECK(scalar_type->dtype_.IsInt()) << "The operator " << op_name
                                     << " requires scalar to be an integer type, but got "
                                     << scalar_type->dtype_.ToString();

  CHECK(As<TileType>(args[2]->GetType()))
      << "The operator " << op_name << " requires third argument to be a TileType, but got "
      << args[2]->GetType()->TypeName();

  // Result has the same shape and dtype as the input tile; bitwise ops do not change element type.
  TileView tile_view;
  tile_view.valid_shape = GetValidShape(tile_type);
  InheritTileViewLayout(tile_view, tile_type);
  return std::make_shared<TileType>(tile_type->shape_, tile_type->dtype_, std::nullopt, tile_view);
}

REGISTER_OP("tile.xor")
    .set_op_category("TileOp")
    .set_description("Element-wise bitwise XOR of two tiles with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .add_argument("tmp", "Temporary tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_input_memory(2, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpTernaryType(args, kwargs, "tile.xor", true);
    });

REGISTER_OP("tile.xors")
    .set_op_category("TileOp")
    .set_description("Element-wise bitwise XOR of tile and scalar")
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .add_argument("tmp", "Temporary tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(2, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .not_inplace_safe()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpXorScalarType(args, kwargs, "tile.xors");
    });

// Type deduction for tile.prelu (Src x Slope x Tmp -> Tile).
// TPRELU requires src, slope, and dst to share their dtype, physical shape, and
// valid region. Target-specific tmp and alias rules are checked during codegen.
TypePtr DeduceTilePreluType(const std::vector<ExprPtr>& args,
                            const std::vector<std::pair<std::string, std::any>>& kwargs,
                            const std::string& op_name) {
  CHECK(args.size() == 3) << "The operator " << op_name << " requires exactly 3 arguments, but got "
                          << args.size();

  auto src_type = As<TileType>(args[0]->GetType());
  auto slope_type = As<TileType>(args[1]->GetType());
  auto tmp_type = As<TileType>(args[2]->GetType());
  CHECK_SPAN(src_type, args[0]->span_)
      << "The operator " << op_name << " requires src to be a TileType, but got "
      << args[0]->GetType()->TypeName();
  CHECK_SPAN(slope_type, args[1]->span_)
      << "The operator " << op_name << " requires slope to be a TileType, but got "
      << args[1]->GetType()->TypeName();
  CHECK_SPAN(tmp_type, args[2]->span_)
      << "The operator " << op_name << " requires tmp to be a TileType, but got "
      << args[2]->GetType()->TypeName();

  CHECK_SPAN(src_type->dtype_ == DataType::FP16 || src_type->dtype_ == DataType::FP32, args[0]->span_)
      << "The operator " << op_name << " requires src dtype in {FP16, FP32}, but got "
      << src_type->dtype_.ToString();
  CHECK_SPAN(slope_type->dtype_ == src_type->dtype_, args[1]->span_)
      << "The operator " << op_name << " requires slope dtype to match src dtype, but got "
      << slope_type->dtype_.ToString() << " and " << src_type->dtype_.ToString();
  CHECK_SPAN(src_type->shape_.size() == 2, args[0]->span_)
      << "The operator " << op_name << " requires a rank-2 src tile, but got rank "
      << src_type->shape_.size();
  CHECK_SPAN(slope_type->shape_.size() == src_type->shape_.size(), args[1]->span_)
      << "The operator " << op_name << " requires slope and src to have the same rank, but got "
      << slope_type->shape_.size() << " and " << src_type->shape_.size();
  for (size_t i = 0; i < src_type->shape_.size(); ++i) {
    CHECK_SPAN(DimensionsEqual(src_type->shape_[i], slope_type->shape_[i]), args[1]->span_)
        << "The operator " << op_name
        << " requires slope and src to have the same physical shape, but dimension " << i
        << " differs; got slope shape " << FormatShape(slope_type->shape_) << " and src shape "
        << FormatShape(src_type->shape_);
  }

  const auto src_valid_shape = GetValidShape(src_type);
  const auto slope_valid_shape = GetValidShape(slope_type);
  CHECK_SPAN(slope_valid_shape.size() == src_valid_shape.size(), args[1]->span_)
      << "The operator " << op_name << " requires slope and src to have the same valid_shape rank";
  for (size_t i = 0; i < src_valid_shape.size(); ++i) {
    CHECK_SPAN(ProveValidExtentEqual(src_valid_shape[i], slope_valid_shape[i]) == ProofResult::kTrue,
               args[1]->span_)
        << "The operator " << op_name << " requires slope and src to have the same valid_shape, but "
        << "dimension " << i << " differs; got slope valid_shape " << FormatShape(slope_valid_shape)
        << " and src valid_shape " << FormatShape(src_valid_shape);
  }

  CHECK_SPAN(tmp_type->shape_.size() == 2, args[2]->span_)
      << "The operator " << op_name << " requires a rank-2 tmp tile, but got rank "
      << tmp_type->shape_.size();

  TileView tile_view;
  tile_view.valid_shape = src_valid_shape;
  InheritTileViewLayout(tile_view, src_type);
  return std::make_shared<TileType>(src_type->shape_, src_type->dtype_, std::nullopt, tile_view);
}

REGISTER_OP("tile.prelu")
    .set_op_category("TileOp")
    .set_description("Element-wise parametric ReLU of a tile with slope tile and temporary buffer")
    .add_argument("tile", "Input tile (TileType)")
    .add_argument("slope", "Slope tile (TileType)")
    .add_argument("tmp", "Temporary tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_input_memory(2, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .not_inplace_safe()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTilePreluType(args, kwargs, "tile.prelu");
    });

REGISTER_OP("tile.addc")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("Element-wise addition of three tiles (lhs + rhs + rhs2) with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .add_argument("rhs2", "Third tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_input_memory(2, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpTriTileType(args, kwargs, "tile.addc");
    });

REGISTER_OP("tile.subc")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("Element-wise subtraction of three tiles (lhs - rhs - rhs2) with broadcasting")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .add_argument("rhs2", "Third tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_input_memory(2, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpTriTileType(args, kwargs, "tile.subc");
    });

REGISTER_OP("tile.addsc")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("Element-wise addition of tile, scalar, and tile (lhs + scalar + rhs2)")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .add_argument("rhs2", "Third tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(2, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpTileScalarTileType(args, kwargs, "tile.addsc");
    });

REGISTER_OP("tile.subsc")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("Element-wise subtraction of tile, scalar, and tile (lhs - scalar - rhs2)")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .add_argument("rhs2", "Third tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(2, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpTileScalarTileType(args, kwargs, "tile.subsc");
    });

REGISTER_OP("tile.lrelu")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("Element-wise leaky ReLU of a tile with scalar slope (max(x, slope*x))")
    .add_argument("tile", "Input tile (TileType)")
    .add_argument("slope", "Scalar slope for negative values (ScalarType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileOpScalarBinaryType(args, kwargs, "tile.lrelu");
    });

// Type deduction for tile.sel (MaskTile x Tile x Tile x TmpTile -> Tile)
// The mask tile encodes per-element predicates in a target-defined layout; its dtype/shape
// do not influence the output type.  Output type is derived from lhs and rhs only.
TypePtr DeduceTileSelType(const std::vector<ExprPtr>& args,
                          const std::vector<std::pair<std::string, std::any>>& kwargs,
                          const std::string& op_name) {
  CHECK(args.size() == 4) << "The operator " << op_name << " requires exactly 4 arguments, but got "
                          << args.size();

  CHECK(As<TileType>(args[0]->GetType()))
      << "The operator " << op_name << " requires first argument (mask) to be a TileType, but got "
      << args[0]->GetType()->TypeName();

  auto tile_type1 = As<TileType>(args[1]->GetType());
  auto tile_type2 = As<TileType>(args[2]->GetType());
  CHECK(tile_type1) << "The operator " << op_name
                    << " requires second argument (lhs) to be a TileType, but got "
                    << args[1]->GetType()->TypeName();
  CHECK(tile_type2) << "The operator " << op_name
                    << " requires third argument (rhs) to be a TileType, but got "
                    << args[2]->GetType()->TypeName();
  CHECK(As<TileType>(args[3]->GetType()))
      << "The operator " << op_name << " requires fourth argument (tmp) to be a TileType, but got "
      << args[3]->GetType()->TypeName();

  auto result_dtype = PromoteDataTypes(tile_type1->dtype_, tile_type2->dtype_);
  CHECK(result_dtype) << "The operator " << op_name << " requires compatible data types, but got "
                      << tile_type1->dtype_.ToString() << " and " << tile_type2->dtype_.ToString();

  auto broadcast_result = BroadcastShapes(tile_type1->shape_, tile_type2->shape_);
  CHECK(broadcast_result.success) << "The operator " << op_name << " requires compatible shapes, but got "
                                  << FormatShape(tile_type1->shape_) << " and "
                                  << FormatShape(tile_type2->shape_);

  // TODO(YunjiQin): assumes both src tiles have the same valid_shape; may need refinement
  // for cases where lhs and rhs have different valid_shape values (e.g. after broadcasting).
  TileView tile_view;
  tile_view.valid_shape = GetValidShape(tile_type1);
  InheritTileViewLayout(tile_view, tile_type1);
  return std::make_shared<TileType>(broadcast_result.shape, *result_dtype, std::nullopt, tile_view);
}

REGISTER_OP("tile.sel")
    .set_op_category("TileOp")
    .set_description(
        "Per-element selection between two tiles using a predicate mask tile. "
        "Maps to the TSEL hardware intrinsic.")
    .add_argument("mask", "Predicate mask tile; encoding is target-defined (TileType)")
    .add_argument("lhs", "Source tile 0, selected where mask is true (TileType)")
    .add_argument("rhs", "Source tile 1, selected where mask is false (TileType)")
    .add_argument("tmp", "Scratch tile required by TSEL (TileType UINT32 [1, 16] on A2/A3)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_input_memory(2, MemorySpace::Vec)
    .set_input_memory(3, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    // TSEL reads the predicate mask (arg 0) and tmp scratch (arg 3) while
    // writing dst. Dead lhs/rhs value operands may be reused when lifetimes
    // allow; mask/tmp stay forbidden via registry (see 34-memory_reuse.md).
    .forbid_output_alias(0)
    .forbid_output_alias(3)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileSelType(args, kwargs, "tile.sel");
    });

// Type deduction for tile.sels (Mask x Src x Tmp x Scalar -> Tile).
// dst[i,j] = mask[i,j] ? src[i,j] : scalar; the result mirrors src.
TypePtr DeduceTileSelsType(const std::vector<ExprPtr>& args,
                           const std::vector<std::pair<std::string, std::any>>& kwargs,
                           const std::string& op_name) {
  CHECK(args.size() == 4) << "The operator " << op_name << " requires exactly 4 arguments, but got "
                          << args.size();

  auto mask_type = As<TileType>(args[0]->GetType());
  auto src_type = As<TileType>(args[1]->GetType());
  auto tmp_type = As<TileType>(args[2]->GetType());
  auto scalar_type = As<ScalarType>(args[3]->GetType());
  CHECK_SPAN(mask_type, args[0]->span_)
      << "The operator " << op_name << " requires mask to be a TileType, but got "
      << args[0]->GetType()->TypeName();
  CHECK_SPAN(src_type, args[1]->span_)
      << "The operator " << op_name << " requires src to be a TileType, but got "
      << args[1]->GetType()->TypeName();
  CHECK_SPAN(tmp_type, args[2]->span_)
      << "The operator " << op_name << " requires tmp to be a TileType, but got "
      << args[2]->GetType()->TypeName();
  CHECK_SPAN(scalar_type, args[3]->span_)
      << "The operator " << op_name << " requires scalar to be a ScalarType, but got "
      << args[3]->GetType()->TypeName();

  CHECK_SPAN(mask_type->shape_.size() == 2, args[0]->span_)
      << "The operator " << op_name << " requires a rank-2 mask tile, but got rank "
      << mask_type->shape_.size();
  CHECK_SPAN(IsTSelsMaskDataType(mask_type->dtype_), args[0]->span_)
      << "The operator " << op_name << " requires an 8-, 16-, or 32-bit integer mask, but got "
      << mask_type->dtype_.ToString();
  CHECK_SPAN(src_type->shape_.size() == 2, args[1]->span_)
      << "The operator " << op_name << " requires a rank-2 src tile, but got rank "
      << src_type->shape_.size();
  CHECK_SPAN(IsTSelsDataType(src_type->dtype_), args[1]->span_)
      << "The operator " << op_name
      << " requires src dtype in {INT8, UINT8, INT16, UINT16, INT32, UINT32, FP16, FP32}, but got "
      << src_type->dtype_.ToString();
  CHECK_SPAN(tmp_type->shape_.size() == 2, args[2]->span_)
      << "The operator " << op_name << " requires a rank-2 tmp tile, but got rank "
      << tmp_type->shape_.size();
  const DataType expected_scalar_dtype = GetTSelsScalarDataType(src_type->dtype_);
  CHECK_SPAN(scalar_type->dtype_ == expected_scalar_dtype, args[3]->span_)
      << "The operator " << op_name << " requires scalar dtype " << expected_scalar_dtype.ToString()
      << " for src dtype " << src_type->dtype_.ToString() << ", but got " << scalar_type->dtype_.ToString();

  const auto mask_valid_shape = GetValidShape(mask_type);
  const auto src_valid_shape = GetValidShape(src_type);
  CHECK_SPAN(ProveValidExtentLessEqual(src_valid_shape[0], mask_valid_shape[0]) == ProofResult::kTrue,
             args[0]->span_)
      << "The operator " << op_name
      << " requires mask carrier rows to cover src valid rows, but got mask valid_shape "
      << FormatShape(mask_valid_shape) << " and src valid_shape " << FormatShape(src_valid_shape);
  const auto required_mask_bytes = MakeCeilDivIndex(src_valid_shape[1], kPackedPredicateBitsPerByte);
  const auto mask_row_bytes = MakeMul(
      mask_valid_shape[1], MakeIndexConst(static_cast<int64_t>(mask_type->dtype_.GetByte()), args[0]->span_),
      args[0]->span_);
  CHECK_SPAN(ProveValidExtentLessEqual(required_mask_bytes, mask_row_bytes) == ProofResult::kTrue,
             args[0]->span_)
      << "The operator " << op_name
      << " requires each mask carrier row to hold at least ceil(src valid columns / 8) packed bytes, "
         "but got mask valid_shape "
      << FormatShape(mask_valid_shape) << " with dtype " << mask_type->dtype_.ToString()
      << " and src valid_shape " << FormatShape(src_valid_shape);

  TileView tile_view;
  tile_view.valid_shape = src_valid_shape;
  InheritTileViewLayout(tile_view, src_type);
  return std::make_shared<TileType>(src_type->shape_, src_type->dtype_, std::nullopt, tile_view);
}

REGISTER_OP("tile.sels")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description(
        "Per-element selection between a source tile and a scalar using a predicate mask tile. "
        "dst[i,j] = mask[i,j] ? src[i,j] : scalar. Maps to the TSELS hardware intrinsic.")
    .add_argument("mask", "Predicate mask tile; encoding is target-defined (TileType)")
    .add_argument("src", "Source tile, selected where mask is true (TileType)")
    .add_argument("tmp", "Scratch tile required by TSELS (TileType)")
    .add_argument("scalar", "Scalar value, selected where mask is false (ScalarType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_input_memory(2, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .forbid_output_alias(0)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileSelsType(args, kwargs, "tile.sels");
    });

// Type deduction for tile.cmp and tile.cmps (comparison operations)
TypePtr DeduceTileCmpType(const std::vector<ExprPtr>& args,
                          const std::vector<std::pair<std::string, std::any>>& kwargs,
                          const std::string& op_name, bool is_scalar_rhs = false) {
  CHECK(args.size() == 2) << "The operator " << op_name << " requires exactly 2 arguments, but got "
                          << args.size();

  // Validate cmp_type attribute exists
  bool has_cmp_type = false;
  for (const auto& [key, value] : kwargs) {
    if (key == "cmp_type") {
      has_cmp_type = true;
      break;
    }
  }
  CHECK(has_cmp_type) << "The operator " << op_name << " requires 'cmp_type' attribute";

  // First argument must be TileType
  auto tile_type1 = As<TileType>(args[0]->GetType());
  CHECK(tile_type1) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                    << args[0]->GetType()->TypeName();

  if (is_scalar_rhs) {
    // Second argument MUST be ScalarType
    auto scalar_type = As<ScalarType>(args[1]->GetType());
    CHECK(scalar_type) << "The operator " << op_name
                       << " requires second argument to be a ScalarType, but got "
                       << args[1]->GetType()->TypeName();

    return MakePackedPredicateTileType(tile_type1->shape_, tile_type1);
  } else {
    // Second argument must be TileType
    auto tile_type2 = As<TileType>(args[1]->GetType());
    CHECK(tile_type2) << "The operator " << op_name << " requires second argument to be a TileType, but got "
                      << args[1]->GetType()->TypeName();

    auto broadcast_result = BroadcastShapes(tile_type1->shape_, tile_type2->shape_);
    CHECK(broadcast_result.success) << "The operator " << op_name << " requires compatible shapes, but got "
                                    << FormatShape(tile_type1->shape_) << " and "
                                    << FormatShape(tile_type2->shape_);

    return MakePackedPredicateTileType(broadcast_result.shape, tile_type1);
  }
}

REGISTER_OP("tile.cmp")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("Element-wise comparison of two tiles (returns a packed predicate mask tile)")
    .add_argument("lhs", "Left-hand side tile (TileType)")
    .add_argument("rhs", "Right-hand side tile (TileType)")
    .set_attr<int>("cmp_type")
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileCmpType(args, kwargs, "tile.cmp", false);
    });

REGISTER_OP("tile.cmps")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("Element-wise comparison of tile and scalar (returns a packed predicate mask tile)")
    .add_argument("lhs", "Tile (TileType)")
    .add_argument("rhs", "Scalar (ScalarType)")
    .set_attr<int>("cmp_type")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTileCmpType(args, kwargs, "tile.cmps", true);
    });

REGISTER_OP("tile.fillpad")
    .set_op_category("TileOp")
    .set_description("Fill destination tile with source tile data and pad remaining elements")
    .add_argument("tile", "Input tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .set_attr<PadValue>("pad_value")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      CHECK(args.size() == 1) << "The operator tile.fillpad requires exactly 1 argument, but got "
                              << args.size();

      // Argument must be TileType
      auto tile_type = As<TileType>(args[0]->GetType());
      CHECK(tile_type) << "The operator tile.fillpad requires first argument to be a TileType, but got "
                       << args[0]->GetType()->TypeName();

      // Get pad_value from kwargs, default to PadValue::zero
      PadValue pad_value = PadValue::zero;
      for (const auto& kv : kwargs) {
        if (kv.first == "pad_value") {
          pad_value = std::any_cast<PadValue>(kv.second);
          CHECK(pad_value != PadValue::null)
              << "tile.fillpad requires pad_value to be zero/max/min, not null";
        }
      }

      // Return TileType with pad value set in tile_view
      // After fillpad, the entire tile is valid (padding region is now filled with pad_value)
      TileView tile_view;
      tile_view.valid_shape = tile_type->shape_;  // Expand valid_shape to full shape
      InheritTileViewLayout(tile_view, tile_type);
      tile_view.pad = pad_value;
      return std::make_shared<TileType>(tile_type->shape_, tile_type->dtype_, tile_type->memref_, tile_view,
                                        tile_type->memory_space_);
    });

REGISTER_OP("tile.fillpad_inplace")
    .set_op_category("TileOp")
    .set_description("Fill padding elements of input tile in place with specified pad value")
    .add_argument("tile", "Input tile (TileType)")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .set_output_reuses_input(0)
    // Rewrites only the padding elements; the data region passes through.
    .set_arg_effect(0, ArgEffect::ReadWrite)
    .set_attr<PadValue>("pad_value")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      CHECK(args.size() == 1) << "The operator tile.fillpad_inplace requires exactly 1 argument, but got "
                              << args.size();

      auto tile_type = As<TileType>(args[0]->GetType());
      CHECK(tile_type)
          << "The operator tile.fillpad_inplace requires first argument to be a TileType, but got "
          << args[0]->GetType()->TypeName();

      PadValue pad_value = PadValue::zero;
      for (const auto& kv : kwargs) {
        if (kv.first == "pad_value") {
          pad_value = std::any_cast<PadValue>(kv.second);
          CHECK(pad_value != PadValue::null)
              << "tile.fillpad_inplace requires pad_value to be zero/max/min, not null";
        }
      }

      TileView tile_view;
      tile_view.valid_shape = tile_type->shape_;
      InheritTileViewLayout(tile_view, tile_type);
      tile_view.pad = pad_value;
      return std::make_shared<TileType>(tile_type->shape_, tile_type->dtype_, tile_type->memref_, tile_view,
                                        tile_type->memory_space_);
    });

REGISTER_OP("tile.fillpad_expand")
    .set_op_category("TileOp")
    .functional_execution_memory_access()
    .set_description("Copy a smaller source tile into a larger destination tile, padding the remainder")
    .add_argument("tile", "Source tile (TileType)")
    .add_argument("shape", "Destination shape (Tuple of ConstInt), each dim >= source dim")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    // The destination is a fresh, larger, differently-strided tile (NOT a view of
    // the source like tile.fillpad). The intrinsic reads the full source while
    // writing dst at a wider row stride, so aliasing dst onto src would corrupt
    // the strided copy — same hazard class as the arg-reduction family.
    .not_inplace_safe()
    .set_attr<PadValue>("pad_value")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      // tile.fillpad_expand(src, shape) — TFILLPAD_EXPAND: dst may be larger than
      // src. The valid region of src is copied into the top-left of dst, and all
      // other dst elements are filled with pad_value. The shape tuple is carried
      // for type deduction only (the codegen reads dst extents from the result
      // type, not from this operand).
      CHECK(args.size() == 2)
          << "The operator tile.fillpad_expand requires exactly 2 arguments (tile, shape), but got "
          << args.size();

      auto tile_type = As<TileType>(args[0]->GetType());
      CHECK(tile_type)
          << "The operator tile.fillpad_expand requires first argument to be a TileType, but got "
          << args[0]->GetType()->TypeName();

      auto shape_tuple = As<MakeTuple>(args[1]);
      CHECK(shape_tuple) << "tile.fillpad_expand shape must be a literal tuple of constants, but got "
                         << args[1]->GetType()->TypeName();
      const std::vector<ExprPtr>& new_shape = shape_tuple->elements_;
      CHECK(new_shape.size() == tile_type->shape_.size())
          << "tile.fillpad_expand shape rank (" << new_shape.size() << ") must match source rank ("
          << tile_type->shape_.size() << ")";

      // When both source and destination dims are static, the destination must
      // not be smaller than the source in any dimension (expand-only).
      for (size_t i = 0; i < new_shape.size(); ++i) {
        auto dst_dim = As<ConstInt>(new_shape[i]);
        CHECK(dst_dim) << "tile.fillpad_expand shape dimension " << i << " must be a constant integer";
        CHECK(dst_dim->value_ > 0) << "tile.fillpad_expand shape dimension " << i << " must be positive, got "
                                   << dst_dim->value_;
        if (auto src_dim = As<ConstInt>(tile_type->shape_[i])) {
          CHECK(dst_dim->value_ >= src_dim->value_)
              << "tile.fillpad_expand destination dimension " << i << " (" << dst_dim->value_
              << ") must be >= source dimension (" << src_dim->value_ << ")";
        }
      }

      PadValue pad_value = PadValue::zero;
      for (const auto& kv : kwargs) {
        if (kv.first == "pad_value") {
          pad_value = std::any_cast<PadValue>(kv.second);
          CHECK(pad_value != PadValue::null)
              << "tile.fillpad_expand requires pad_value to be zero/max/min, not null";
        }
      }

      // After expand the entire destination tile is valid (the padding region is
      // filled). Inherit the source layout; only the shape and pad change.
      TileView tile_view;
      tile_view.valid_shape = new_shape;
      InheritTileViewLayout(tile_view, tile_type);
      tile_view.pad = pad_value;
      return std::make_shared<TileType>(new_shape, tile_type->dtype_, tile_type->memref_, tile_view,
                                        tile_type->memory_space_);
    });

}  // namespace ir
}  // namespace pypto
