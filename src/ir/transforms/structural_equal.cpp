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

#include <any>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/reflection/field_visitor.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/printer.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

TileLayout InferCanonicalTileLayoutFromShape(const std::vector<ExprPtr>& shape) {
  if (shape.size() != 2) {
    return TileLayout::row_major;
  }

  auto rows_const = As<ConstInt>(shape[0]);
  auto cols_const = As<ConstInt>(shape[1]);
  if (!rows_const || !cols_const) {
    return TileLayout::row_major;
  }
  return (cols_const->value_ == 1 && rows_const->value_ > 1) ? TileLayout::col_major : TileLayout::row_major;
}

TensorView NormalizeTensorViewForCompare(const std::optional<TensorView>& tensor_view,
                                         const std::vector<ExprPtr>& shape) {
  TensorView normalized = tensor_view.value_or(TensorView{});
  if (normalized.valid_shape.empty()) {
    normalized.valid_shape = shape;
  }
  return normalized;
}

TileView NormalizeTileViewForCompare(const std::optional<TileView>& tile_view,
                                     const std::vector<ExprPtr>& shape,
                                     const std::optional<MemorySpace>& memory_space = std::nullopt) {
  TileView normalized = tile_view.value_or(TileView{});
  if (normalized.valid_shape.empty()) {
    normalized.valid_shape = shape;
  }
  if (!tile_view.has_value()) {
    normalized.blayout = InferCanonicalTileLayoutFromShape(shape);
    if (memory_space.has_value()) {
      switch (*memory_space) {
        case MemorySpace::Left:
          normalized.blayout = TileLayout::col_major;
          normalized.slayout = TileLayout::row_major;
          break;
        case MemorySpace::Right:
          normalized.slayout = TileLayout::col_major;
          break;
        case MemorySpace::Acc:
          normalized.blayout = TileLayout::col_major;
          normalized.slayout = TileLayout::row_major;
          normalized.fractal = 1024;
          break;
        default:
          break;
      }
    }
  }
  if (!normalized.start_offset) {
    normalized.start_offset = std::make_shared<ConstInt>(0, DataType::INDEX, Span::unknown());
  }
  return normalized;
}

bool AreShapeConstIntsCompatible(const ExprPtr& lhs, const ExprPtr& rhs) {
  auto lhs_const = As<ConstInt>(lhs);
  auto rhs_const = As<ConstInt>(rhs);
  if (!lhs_const || !rhs_const) {
    return false;
  }
  if (lhs_const->value_ != rhs_const->value_) {
    return false;
  }
  if (lhs_const->dtype() == rhs_const->dtype()) {
    return true;
  }
  return (lhs_const->dtype() == DataType::INDEX && rhs_const->dtype().IsInt()) ||
         (rhs_const->dtype() == DataType::INDEX && lhs_const->dtype().IsInt());
}

bool AreIndexCompatibleIntegerTypes(const DataType& lhs, const DataType& rhs) {
  if (lhs == rhs) {
    return true;
  }
  return (lhs == DataType::INDEX && rhs.IsInt()) || (rhs == DataType::INDEX && lhs.IsInt());
}

std::string CanonicalizeOpNameForRoundtripCompare(std::string op_name) {
  if (op_name == "tensor.add") return "tensor.adds";
  if (op_name == "tensor.sub") return "tensor.subs";
  if (op_name == "tensor.mul") return "tensor.muls";
  if (op_name == "tensor.div") return "tensor.divs";
  return op_name;
}

bool IsTupleExpr(const ExprPtr& expr) { return static_cast<bool>(As<MakeTuple>(expr)); }

struct CanonicalCallForRoundtripCompare {
  std::string op_name;
  std::vector<ExprPtr> args;
  std::vector<std::pair<std::string, std::any>> kwargs;
};

std::vector<ExprPtr> CanonicalizeTileLoadArgsForCompare(const CallPtr& call) {
  if (call->args_.size() == 3) {
    if (IsTupleExpr(call->args_[1]) && IsTupleExpr(call->args_[2])) {
      auto valid_shapes = std::make_shared<MakeTuple>(As<MakeTuple>(call->args_[2])->elements_, call->span_);
      return {call->args_[0], call->args_[1], call->args_[2], valid_shapes};
    }
    auto offsets = std::make_shared<MakeTuple>(std::vector<ExprPtr>{call->args_[1]}, call->span_);
    auto shapes = std::make_shared<MakeTuple>(std::vector<ExprPtr>{call->args_[2]}, call->span_);
    auto valid_shapes = std::make_shared<MakeTuple>(std::vector<ExprPtr>{call->args_[2]}, call->span_);
    return {call->args_[0], offsets, shapes, valid_shapes};
  }

  if (call->args_.size() == 5) {
    auto offsets =
        std::make_shared<MakeTuple>(std::vector<ExprPtr>{call->args_[1], call->args_[2]}, call->span_);
    auto shapes =
        std::make_shared<MakeTuple>(std::vector<ExprPtr>{call->args_[3], call->args_[4]}, call->span_);
    auto valid_shapes =
        std::make_shared<MakeTuple>(std::vector<ExprPtr>{call->args_[3], call->args_[4]}, call->span_);
    return {call->args_[0], offsets, shapes, valid_shapes};
  }

  return call->args_;
}

std::vector<ExprPtr> CanonicalizeTileStoreArgsForCompare(const CallPtr& call) {
  if (call->args_.size() == 3 && !IsTupleExpr(call->args_[1])) {
    auto offsets = std::make_shared<MakeTuple>(std::vector<ExprPtr>{call->args_[1]}, call->span_);
    return {call->args_[0], offsets, call->args_[2]};
  }

  if (call->args_.size() == 4 && !IsTupleExpr(call->args_[1])) {
    auto offsets =
        std::make_shared<MakeTuple>(std::vector<ExprPtr>{call->args_[1], call->args_[2]}, call->span_);
    return {call->args_[0], offsets, call->args_[3]};
  }

  return call->args_;
}

std::vector<ExprPtr> CanonicalizeCallArgsForRoundtripCompare(const CallPtr& call) {
  if (!call || !call->op_) {
    return call ? call->args_ : std::vector<ExprPtr>{};
  }

  const std::string op_name = CanonicalizeOpNameForRoundtripCompare(call->op_->name_);
  if (op_name == "tile.load") {
    return CanonicalizeTileLoadArgsForCompare(call);
  }
  if (op_name == "tile.store") {
    return CanonicalizeTileStoreArgsForCompare(call);
  }
  return call->args_;
}

std::vector<std::pair<std::string, std::any>> CanonicalizeCallKwargsForRoundtripCompare(const CallPtr& call) {
  if (!call || !call->op_) {
    return call ? call->kwargs_ : std::vector<std::pair<std::string, std::any>>{};
  }

  const std::string op_name = CanonicalizeOpNameForRoundtripCompare(call->op_->name_);
  if (op_name == "tile.load" && call->kwargs_.empty()) {
    return {
        {"target_memory", std::any(MemorySpace::Vec)},
        {"transpose", std::any(false)},
    };
  }

  return call->kwargs_;
}

CanonicalCallForRoundtripCompare CanonicalizeCallForRoundtripCompare(const CallPtr& call) {
  if (!call || !call->op_) {
    return {"", call ? call->args_ : std::vector<ExprPtr>{},
            call ? call->kwargs_ : std::vector<std::pair<std::string, std::any>>{}};
  }

  CanonicalCallForRoundtripCompare canonicalized{
      CanonicalizeOpNameForRoundtripCompare(call->op_->name_),
      call->args_,
      call->kwargs_,
  };

  if (canonicalized.op_name == "tile.create" && call->args_.size() == 1 && !IsTupleExpr(call->args_[0])) {
    if (auto tile_type = As<TileType>(call->GetType())) {
      canonicalized.op_name = "tile.full";
      canonicalized.args = {
          std::make_shared<MakeTuple>(tile_type->shape_, call->span_),
          call->args_[0],
      };
      canonicalized.kwargs = {{"dtype", std::any(tile_type->dtype_)}};
      return canonicalized;
    }
  }

  canonicalized.args = CanonicalizeCallArgsForRoundtripCompare(call);
  canonicalized.kwargs = CanonicalizeCallKwargsForRoundtripCompare(call);
  return canonicalized;
}

TypePtr DeduceCanonicalCallTypeForRoundtripCompare(
    const std::string& op_name, const Span& span, const std::vector<ExprPtr>& args,
    const std::vector<std::pair<std::string, std::any>>& kwargs) {
  if (op_name.empty()) {
    return nullptr;
  }
  try {
    auto deduced = OpRegistry::GetInstance().Create(op_name, args, kwargs, span);
    return deduced->GetType();
  } catch (const Error&) {
    return nullptr;
  }
}

bool IsPredicateBinaryExpr(const BinaryExprPtr& expr) {
  return IsA<Eq>(expr) || IsA<Ne>(expr) || IsA<Lt>(expr) || IsA<Le>(expr) || IsA<Gt>(expr) || IsA<Ge>(expr) ||
         IsA<And>(expr) || IsA<Or>(expr) || IsA<Xor>(expr);
}

bool IsPredicateUnaryExpr(const UnaryExprPtr& expr) { return IsA<Not>(expr); }

bool AreBoolIntegerLegacyPredicateTypesCompatible(const TypePtr& lhs, const TypePtr& rhs) {
  auto lhs_scalar = As<ScalarType>(lhs);
  auto rhs_scalar = As<ScalarType>(rhs);
  if (!lhs_scalar || !rhs_scalar) {
    return false;
  }
  if (lhs_scalar->dtype_ == rhs_scalar->dtype_) {
    return true;
  }
  const bool lhs_bool = lhs_scalar->dtype_ == DataType::BOOL;
  const bool rhs_bool = rhs_scalar->dtype_ == DataType::BOOL;
  const bool lhs_int_like = lhs_scalar->dtype_ == DataType::INDEX || lhs_scalar->dtype_.IsInt();
  const bool rhs_int_like = rhs_scalar->dtype_ == DataType::INDEX || rhs_scalar->dtype_.IsInt();
  return (lhs_bool && rhs_int_like) || (rhs_bool && lhs_int_like);
}

bool AreShapeExprVectorsCompatible(const std::vector<ExprPtr>& lhs, const std::vector<ExprPtr>& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (AreShapeConstIntsCompatible(lhs[i], rhs[i])) {
      continue;
    }
    if (lhs[i] != rhs[i]) {
      return false;
    }
  }
  return true;
}

TypePtr InferAssignedCallTypeForRoundtripCompare(const VarPtr& var, const CallPtr& call) {
  if (!var || !call) {
    return call ? call->GetType() : nullptr;
  }

  auto var_tile = As<TileType>(var->GetType());
  auto call_tile = As<TileType>(call->GetType());
  if (var_tile && call_tile && var_tile->dtype_ == call_tile->dtype_ &&
      AreShapeExprVectorsCompatible(var_tile->shape_, call_tile->shape_)) {
    return var->GetType();
  }

  auto var_tensor = As<TensorType>(var->GetType());
  auto call_tensor = As<TensorType>(call->GetType());
  if (var_tensor && call_tensor && var_tensor->dtype_ == call_tensor->dtype_ &&
      AreShapeExprVectorsCompatible(var_tensor->shape_, call_tensor->shape_)) {
    return var->GetType();
  }

  return call->GetType();
}

template <bool AssertMode>
class TransparentDepthResetGuard {
 public:
  explicit TransparentDepthResetGuard(int& transparent_depth)
      : transparent_depth_(transparent_depth), saved_depth_(transparent_depth) {
    if constexpr (AssertMode) {
      transparent_depth_ = 0;
    }
  }

  ~TransparentDepthResetGuard() {
    if constexpr (AssertMode) {
      transparent_depth_ = saved_depth_;
    }
  }

 private:
  int& transparent_depth_;
  int saved_depth_;
};

}  // namespace

/**
 * @brief Unified structural equality checker for IR nodes
 *
 * Template parameter controls behavior on mismatch:
 * - AssertMode=false: Returns false (for structural_equal)
 * - AssertMode=true: Throws ValueError with detailed error message (for assert_structural_equal)
 *
 * This class is not part of the public API - use structural_equal() or assert_structural_equal().
 *
 * Implements the FieldIterator visitor interface for generic field-based comparison.
 * Uses the dual-node Visit overload which calls visitor methods with two field arguments.
 */
template <bool AssertMode>
class StructuralEqualImpl {
 public:
  using result_type = bool;

  explicit StructuralEqualImpl(bool enable_auto_mapping) : enable_auto_mapping_(enable_auto_mapping) {}

  // Returns bool for structural_equal, throws for assert_structural_equal
  bool operator()(const IRNodePtr& lhs, const IRNodePtr& rhs) {
    if constexpr (AssertMode) {
      Equal(lhs, rhs);
      return true;  // Only reached if no exception thrown
    } else {
      return Equal(lhs, rhs);
    }
  }

  bool operator()(const TypePtr& lhs, const TypePtr& rhs) {
    if constexpr (AssertMode) {
      EqualType(lhs, rhs);
      return true;  // Only reached if no exception thrown
    } else {
      return EqualType(lhs, rhs);
    }
  }

  // FieldIterator visitor interface (dual-node version - methods receive two fields)
  [[nodiscard]] result_type InitResult() const { return true; }

  template <typename IRNodePtrType>
  result_type VisitIRNodeField(const IRNodePtrType& lhs, const IRNodePtrType& rhs) {
    INTERNAL_CHECK(lhs) << "structural_equal encountered null lhs IR node field";
    INTERNAL_CHECK(rhs) << "structural_equal encountered null rhs IR node field";
    return Equal(lhs, rhs);
  }

  // Specialization for std::optional<IRNodePtr>
  template <typename IRNodePtrType>
  result_type VisitIRNodeField(const std::optional<IRNodePtrType>& lhs,
                               const std::optional<IRNodePtrType>& rhs) {
    if (!lhs.has_value() && !rhs.has_value()) {
      return true;
    }
    if (!lhs.has_value() || !rhs.has_value()) {
      if constexpr (AssertMode) {
        ThrowMismatch("Optional field presence mismatch", lhs.has_value() ? *lhs : IRNodePtr(),
                      rhs.has_value() ? *rhs : IRNodePtr(), lhs.has_value() ? "has value" : "nullopt",
                      rhs.has_value() ? "has value" : "nullopt");
      }
      return false;
    }
    if (!*lhs && !*rhs) {
      return true;
    }
    if (!*lhs || !*rhs) {
      if constexpr (AssertMode) {
        ThrowMismatch("Optional field nullptr mismatch", *lhs, *rhs, *lhs ? "has value" : "nullptr",
                      *rhs ? "has value" : "nullptr");
      }
      return false;
    }
    return Equal(*lhs, *rhs);
  }

  template <typename IRNodePtrType>
  result_type VisitIRNodeVectorField(const std::vector<IRNodePtrType>& lhs,
                                     const std::vector<IRNodePtrType>& rhs) {
    if (lhs.size() != rhs.size()) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "Vector size mismatch (" << lhs.size() << " items != " << rhs.size() << " items)";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
      INTERNAL_CHECK(lhs[i]) << "structural_equal encountered null lhs IR node in vector at index " << i;
      INTERNAL_CHECK(rhs[i]) << "structural_equal encountered null rhs IR node in vector at index " << i;

      if constexpr (AssertMode) {
        std::ostringstream index_str;
        index_str << "[" << i << "]";
        path_.emplace_back(index_str.str());
      }

      if (!Equal(lhs[i], rhs[i])) {
        if constexpr (AssertMode) {
          path_.pop_back();
        }
        return false;
      }

      if constexpr (AssertMode) {
        path_.pop_back();
      }
    }
    return true;
  }

  template <typename KeyType, typename ValueType, typename Compare>
  result_type VisitIRNodeMapField(const std::map<KeyType, ValueType, Compare>& lhs,
                                  const std::map<KeyType, ValueType, Compare>& rhs) {
    if (lhs.size() != rhs.size()) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "Map size mismatch (" << lhs.size() << " items != " << rhs.size() << " items)";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    auto lhs_it = lhs.begin();
    auto rhs_it = rhs.begin();
    while (lhs_it != lhs.end()) {
      INTERNAL_CHECK(lhs_it->first) << "structural_equal encountered null lhs key in map";
      INTERNAL_CHECK(lhs_it->second) << "structural_equal encountered null lhs value in map";
      INTERNAL_CHECK(rhs_it->first) << "structural_equal encountered null rhs key in map";
      INTERNAL_CHECK(rhs_it->second) << "structural_equal encountered null rhs value in map";

      if (lhs_it->first->name_ != rhs_it->first->name_) {
        if constexpr (AssertMode) {
          std::ostringstream msg;
          msg << "Map key mismatch ('" << lhs_it->first->name_ << "' != '" << rhs_it->first->name_ << "')";
          ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
        }
        return false;
      }

      if constexpr (AssertMode) {
        std::ostringstream key_str;
        key_str << "['" << lhs_it->first->name_ << "']";
        path_.emplace_back(key_str.str());
      }

      if (!Equal(lhs_it->second, rhs_it->second)) {
        if constexpr (AssertMode) {
          path_.pop_back();
        }
        return false;
      }

      if constexpr (AssertMode) {
        path_.pop_back();
      }
      ++lhs_it;
      ++rhs_it;
    }
    return true;
  }

  // Leaf field comparisons (dual-node version)
  result_type VisitLeafField(const int& lhs, const int& rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "Integer value mismatch (" << lhs << " != " << rhs << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  result_type VisitLeafField(const int64_t& lhs, const int64_t& rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "int64_t value mismatch (" << lhs << " != " << rhs << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  result_type VisitLeafField(const uint64_t& lhs, const uint64_t& rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "uint64_t value mismatch (" << lhs << " != " << rhs << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  result_type VisitLeafField(const double& lhs, const double& rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "double value mismatch (" << lhs << " != " << rhs << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  result_type VisitLeafField(const std::string& lhs, const std::string& rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "String value mismatch (\"" << lhs << "\" != \"" << rhs << "\")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  result_type VisitLeafField(const OpPtr& lhs, const OpPtr& rhs) {
    const std::string lhs_name = CanonicalizeOpNameForRoundtripCompare(lhs->name_);
    const std::string rhs_name = CanonicalizeOpNameForRoundtripCompare(rhs->name_);
    if (lhs_name != rhs_name) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "Operator name mismatch ('" << lhs->name_ << "' != '" << rhs->name_ << "')";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  result_type VisitLeafField(const DataType& lhs, const DataType& rhs) {
    if (!AreIndexCompatibleIntegerTypes(lhs, rhs)) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "DataType mismatch (" << lhs.ToString() << " != " << rhs.ToString() << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  result_type VisitLeafField(const FunctionType& lhs, const FunctionType& rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "FunctionType mismatch (" << FunctionTypeToString(lhs) << " != " << FunctionTypeToString(rhs)
            << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  result_type VisitLeafField(const ForKind& lhs, const ForKind& rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "ForKind mismatch (" << ForKindToString(lhs) << " != " << ForKindToString(rhs) << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  result_type VisitLeafField(const ChunkPolicy& lhs, const ChunkPolicy& rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "ChunkPolicy mismatch (" << ChunkPolicyToString(lhs) << " != " << ChunkPolicyToString(rhs)
            << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  [[nodiscard]] result_type VisitLeafField(const LoopOrigin& lhs, const LoopOrigin& rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "LoopOrigin mismatch (" << LoopOriginToString(lhs) << " != " << LoopOriginToString(rhs) << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  [[nodiscard]] result_type VisitLeafField(const ScopeKind& lhs, const ScopeKind& rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "ScopeKind mismatch (" << ScopeKindToString(lhs) << " != " << ScopeKindToString(rhs) << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  result_type VisitLeafField(const Level& lhs, const Level& rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "Level mismatch (" << LevelToString(lhs) << " != " << LevelToString(rhs) << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  result_type VisitLeafField(const Role& lhs, const Role& rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "Role mismatch (" << RoleToString(lhs) << " != " << RoleToString(rhs) << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  result_type VisitLeafField(const std::optional<Level>& lhs, const std::optional<Level>& rhs) {
    if (lhs.has_value() != rhs.has_value()) {
      if constexpr (AssertMode) {
        ThrowMismatch("Level optional presence mismatch", IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    if (lhs.has_value()) {
      return VisitLeafField(*lhs, *rhs);
    }
    return true;
  }

  result_type VisitLeafField(const std::optional<Role>& lhs, const std::optional<Role>& rhs) {
    if (lhs.has_value() != rhs.has_value()) {
      if constexpr (AssertMode) {
        ThrowMismatch("Role optional presence mismatch", IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    if (lhs.has_value()) {
      return VisitLeafField(*lhs, *rhs);
    }
    return true;
  }

  result_type VisitLeafField(const ParamDirection& lhs, const ParamDirection& rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "ParamDirection mismatch (" << ParamDirectionToString(lhs)
            << " != " << ParamDirectionToString(rhs) << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  result_type VisitLeafField(const std::vector<ParamDirection>& lhs, const std::vector<ParamDirection>& rhs) {
    if (lhs.size() != rhs.size()) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "ParamDirection vector size mismatch (" << lhs.size() << " != " << rhs.size() << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
      if (!VisitLeafField(lhs[i], rhs[i])) {
        return false;
      }
    }
    return true;
  }

  // Compare kwargs (vector of pairs to preserve order)
  result_type VisitLeafField(const std::vector<std::pair<std::string, std::any>>& lhs,
                             const std::vector<std::pair<std::string, std::any>>& rhs) {
    if (lhs.size() != rhs.size()) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "Kwargs size mismatch (" << lhs.size() << " != " << rhs.size() << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
      if (lhs[i].first != rhs[i].first) {
        if constexpr (AssertMode) {
          std::ostringstream msg;
          msg << "Kwargs key mismatch at index " << i << " ('" << lhs[i].first << "' != '" << rhs[i].first
              << "')";
          ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
        }
        return false;
      }
      // Compare std::any values by type and content
      const auto& lhs_val = lhs[i].second;
      const auto& rhs_val = rhs[i].second;
      if (lhs_val.type() != rhs_val.type()) {
        if constexpr (AssertMode) {
          std::ostringstream msg;
          msg << "Kwargs value type mismatch for key '" << lhs[i].first << "'";
          ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
        }
        return false;
      }
      // Type-specific comparison
      bool values_equal = true;
      if (lhs_val.type() == typeid(int)) {
        values_equal = (AnyCast<int>(lhs_val, "comparing kwarg: " + lhs[i].first) ==
                        AnyCast<int>(rhs_val, "comparing kwarg: " + lhs[i].first));
      } else if (lhs_val.type() == typeid(bool)) {
        values_equal = (AnyCast<bool>(lhs_val, "comparing kwarg: " + lhs[i].first) ==
                        AnyCast<bool>(rhs_val, "comparing kwarg: " + lhs[i].first));
      } else if (lhs_val.type() == typeid(std::string)) {
        values_equal = (AnyCast<std::string>(lhs_val, "comparing kwarg: " + lhs[i].first) ==
                        AnyCast<std::string>(rhs_val, "comparing kwarg: " + lhs[i].first));
      } else if (lhs_val.type() == typeid(double)) {
        values_equal = (AnyCast<double>(lhs_val, "comparing kwarg: " + lhs[i].first) ==
                        AnyCast<double>(rhs_val, "comparing kwarg: " + lhs[i].first));
      } else if (lhs_val.type() == typeid(DataType)) {
        values_equal = (AnyCast<DataType>(lhs_val, "comparing kwarg: " + lhs[i].first) ==
                        AnyCast<DataType>(rhs_val, "comparing kwarg: " + lhs[i].first));
      } else if (lhs_val.type() == typeid(MemorySpace)) {
        values_equal = (AnyCast<MemorySpace>(lhs_val, "comparing kwarg: " + lhs[i].first) ==
                        AnyCast<MemorySpace>(rhs_val, "comparing kwarg: " + lhs[i].first));
      }
      if (!values_equal) {
        if constexpr (AssertMode) {
          std::ostringstream msg;
          msg << "Kwargs value mismatch for key '" << lhs[i].first << "'";
          ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
        }
        return false;
      }
    }
    return true;
  }

  result_type VisitLeafField(const MemorySpace& lhs, const MemorySpace& rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "MemorySpace mismatch (" << MemorySpaceToString(lhs) << " != " << MemorySpaceToString(rhs)
            << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  result_type VisitLeafField(const TypePtr& lhs, const TypePtr& rhs) { return EqualType(lhs, rhs); }

  result_type VisitLeafField(const std::vector<TypePtr>& lhs, const std::vector<TypePtr>& rhs) {
    if (lhs.size() != rhs.size()) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "Type vector size mismatch (" << lhs.size() << " types != " << rhs.size() << " types)";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
      INTERNAL_CHECK(lhs[i]) << "structural_equal encountered null lhs TypePtr in vector at index " << i;
      INTERNAL_CHECK(rhs[i]) << "structural_equal encountered null rhs TypePtr in vector at index " << i;
      if (!EqualType(lhs[i], rhs[i])) return false;
    }
    return true;
  }

  [[nodiscard]] result_type VisitLeafField(const Span& lhs, const Span& rhs) const {
    INTERNAL_UNREACHABLE << "structural_equal should not visit Span field";
    return true;  // Never reached
  }

  // Field kind hooks
  template <typename FVisitOp>
  void VisitIgnoreField([[maybe_unused]] FVisitOp&& visit_op) {
    // Ignored fields are always considered equal
  }

  template <typename FVisitOp>
  void VisitDefField(FVisitOp&& visit_op) {
    bool enable_auto_mapping = true;
    std::swap(enable_auto_mapping, enable_auto_mapping_);
    visit_op();
    std::swap(enable_auto_mapping, enable_auto_mapping_);
  }

  template <typename FVisitOp>
  void VisitUsualField(FVisitOp&& visit_op) {
    visit_op();
  }

  // Path tracking hooks called by FieldIterator::VisitFieldImpl for each field.
  // PushFieldName pushes ".name" only when not inside a transparent container.
  // Transparent containers (Program, SeqStmts) suppress their own field
  // names so that their vector/map element accessors ([i] / ['key']) attach directly
  // to the parent field name, producing paths like body[1] instead of body.stmts[1].
  void PushFieldName(const char* name) {
    if constexpr (AssertMode) {
      if (transparent_depth_ == 0) {
        path_.emplace_back(name);  // No dot prefix — ThrowMismatch adds '.' separators
      }
    }
  }

  void PopFieldName() {
    if constexpr (AssertMode) {
      if (transparent_depth_ == 0) {
        path_.pop_back();
      }
    }
  }

  // Combine results (AND logic)
  template <typename Desc>
  void CombineResult(result_type& accumulator, result_type field_result, [[maybe_unused]] const Desc& desc) {
    accumulator = accumulator && field_result;
  }

 private:
  bool Equal(const IRNodePtr& lhs, const IRNodePtr& rhs);
  bool EqualCall(const CallPtr& lhs, const CallPtr& rhs);
  bool EqualAssignStmt(const AssignStmtPtr& lhs, const AssignStmtPtr& rhs);
  bool EqualBinaryExpr(const BinaryExprPtr& lhs, const BinaryExprPtr& rhs);
  bool EqualUnaryExpr(const UnaryExprPtr& lhs, const UnaryExprPtr& rhs);
  bool EqualForStmt(const ForStmtPtr& lhs, const ForStmtPtr& rhs);
  bool EqualVar(const VarPtr& lhs, const VarPtr& rhs);
  bool EqualMemRef(const MemRefPtr& lhs, const MemRefPtr& rhs);
  bool EqualIterArg(const IterArgPtr& lhs, const IterArgPtr& rhs);
  bool EqualType(const TypePtr& lhs, const TypePtr& rhs);

  /**
   * @brief Generic field-based equality check for IR nodes using FieldIterator
   *
   * Uses the dual-node Visit overload which passes two fields to each visitor method.
   *
   * @tparam NodePtr Shared pointer type to the node
   * @param lhs_op Left-hand side node
   * @param rhs_op Right-hand side node
   * @return true if all fields are equal
   */
  template <typename NodePtr>
  bool EqualWithFields(const NodePtr& lhs_op, const NodePtr& rhs_op) {
    using NodeType = typename NodePtr::element_type;
    auto descriptors = NodeType::GetFieldDescriptors();

    return std::apply(
        [&](auto&&... descs) {
          return reflection::FieldIterator<NodeType, StructuralEqualImpl<AssertMode>,
                                           decltype(descs)...>::Visit(*lhs_op, *rhs_op, *this, descs...);
        },
        descriptors);
  }

  // Only used in assert mode for error messages
  void ThrowMismatch(const std::string& reason, const IRNodePtr& lhs, const IRNodePtr& rhs,
                     const std::string& lhs_desc = "", const std::string& rhs_desc = "") {
    if constexpr (AssertMode) {
      std::ostringstream msg;
      msg << "Structural equality assertion failed";

      if (!path_.empty()) {
        msg << " at: ";
        for (size_t i = 0; i < path_.size(); ++i) {
          msg << path_[i];
          if (i < path_.size() - 1 && path_[i + 1][0] != '[') {
            msg << ".";
          }
        }
      }
      msg << "\n\n";

      if (lhs || rhs) {
        msg << "Left-hand side:\n";
        if (lhs) {
          std::string lhs_str = PythonPrint(lhs, "pl");
          std::istringstream iss(lhs_str);
          std::string line;
          while (std::getline(iss, line)) {
            msg << "  " << line << "\n";
          }
        } else {
          msg << "  (null)\n";
        }

        msg << "\nRight-hand side:\n";
        if (rhs) {
          std::string rhs_str = PythonPrint(rhs, "pl");
          std::istringstream iss(rhs_str);
          std::string line;
          while (std::getline(iss, line)) {
            msg << "  " << line << "\n";
          }
        } else {
          msg << "  (null)\n";
        }
        msg << "\n";
      } else if (!lhs_desc.empty() || !rhs_desc.empty()) {
        msg << "Left: " << lhs_desc << "\n";
        msg << "Right: " << rhs_desc << "\n\n";
      }

      msg << "Reason: " << reason;
      throw pypto::ValueError(msg.str());
    }
  }

  bool enable_auto_mapping_;
  std::unordered_map<VarPtr, VarPtr> lhs_to_rhs_var_map_;
  std::unordered_map<VarPtr, VarPtr> rhs_to_lhs_var_map_;
  std::vector<std::string> path_;  // Only used in assert mode
  int transparent_depth_ = 0;      // Depth inside transparent containers (Program/SeqStmts)
};

// Type dispatch macro for generic field-based comparison.
// Saves and resets transparent_depth_ to 0 before entering EqualWithFields so that
// field names of this (non-transparent) node are always pushed into the path, even
// when Equal() is called recursively from within a transparent container's field visit.
#define EQUAL_DISPATCH(Type)                                               \
  if (auto lhs_##Type = As<Type>(lhs)) {                                   \
    auto rhs_##Type = As<Type>(rhs);                                       \
    if constexpr (AssertMode) {                                            \
      int saved_depth = transparent_depth_;                                \
      transparent_depth_ = 0;                                              \
      bool result = rhs_##Type && EqualWithFields(lhs_##Type, rhs_##Type); \
      transparent_depth_ = saved_depth;                                    \
      return result;                                                       \
    } else {                                                               \
      return rhs_##Type && EqualWithFields(lhs_##Type, rhs_##Type);        \
    }                                                                      \
  }

// Dispatch macro for transparent container nodes (Program, SeqStmts).
// Increments transparent_depth_ so that their field names are suppressed in the path,
// allowing vector/map element accessors ([i] / ['key']) to attach directly to the
// parent field name: e.g., body[1] instead of body.stmts[1].
#define EQUAL_DISPATCH_TRANSPARENT(Type)                                 \
  if (auto lhs_##Type = As<Type>(lhs)) {                                 \
    if constexpr (AssertMode) transparent_depth_++;                      \
    auto rhs_##Type = As<Type>(rhs);                                     \
    bool result = rhs_##Type && EqualWithFields(lhs_##Type, rhs_##Type); \
    if constexpr (AssertMode) transparent_depth_--;                      \
    return result;                                                       \
  }

template <bool AssertMode>
bool StructuralEqualImpl<AssertMode>::Equal(const IRNodePtr& lhs, const IRNodePtr& rhs) {
  if (lhs.get() == rhs.get()) return true;

  if (!lhs || !rhs) {
    if constexpr (AssertMode) ThrowMismatch("One node is null, the other is not", lhs, rhs);
    return false;
  }

  if (lhs->TypeName() != rhs->TypeName()) {
    if constexpr (AssertMode) {
      std::ostringstream msg;
      msg << "Node type mismatch (" << lhs->TypeName() << " != " << rhs->TypeName() << ")";
      ThrowMismatch(msg.str(), lhs, rhs);
    }
    return false;
  }

  // Check MemRef before IterArg and Var (MemRef inherits from Var)
  if (auto lhs_memref = As<MemRef>(lhs)) {
    auto rhs_memref = std::static_pointer_cast<const MemRef>(rhs);
    bool result = rhs_memref && EqualMemRef(lhs_memref, rhs_memref);
    return result;
  }

  // Check IterArg before Var (IterArg inherits from Var)
  if (auto lhs_iter = As<IterArg>(lhs)) {
    bool result = EqualIterArg(lhs_iter, std::static_pointer_cast<const IterArg>(rhs));
    return result;
  }

  if (auto lhs_var = As<Var>(lhs)) {
    bool result = EqualVar(lhs_var, std::static_pointer_cast<const Var>(rhs));
    return result;
  }

  // All other types use generic field-based comparison
  EQUAL_DISPATCH(ConstInt)
  EQUAL_DISPATCH(ConstFloat)
  EQUAL_DISPATCH(ConstBool)
  if (auto lhs_call = As<Call>(lhs)) {
    bool result = EqualCall(lhs_call, std::static_pointer_cast<const Call>(rhs));
    return result;
  }
  EQUAL_DISPATCH(MakeTuple)
  EQUAL_DISPATCH(TupleGetItemExpr)

  // BinaryExpr and UnaryExpr are abstract base classes matching multiple kinds
  if (auto lhs_binary = As<BinaryExpr>(lhs)) {
    bool result = EqualBinaryExpr(lhs_binary, std::static_pointer_cast<const BinaryExpr>(rhs));
    return result;
  }
  if (auto lhs_unary = As<UnaryExpr>(lhs)) {
    bool result = EqualUnaryExpr(lhs_unary, std::static_pointer_cast<const UnaryExpr>(rhs));
    return result;
  }

  if (auto lhs_assign = As<AssignStmt>(lhs)) {
    bool result = EqualAssignStmt(lhs_assign, std::static_pointer_cast<const AssignStmt>(rhs));
    return result;
  }
  EQUAL_DISPATCH(IfStmt)
  EQUAL_DISPATCH(YieldStmt)
  EQUAL_DISPATCH(ReturnStmt)
  if (auto lhs_for = As<ForStmt>(lhs)) {
    bool result = EqualForStmt(lhs_for, std::static_pointer_cast<const ForStmt>(rhs));
    return result;
  }
  EQUAL_DISPATCH(WhileStmt)
  EQUAL_DISPATCH(ScopeStmt)
  EQUAL_DISPATCH_TRANSPARENT(SeqStmts)
  EQUAL_DISPATCH(EvalStmt)
  EQUAL_DISPATCH(BreakStmt)
  EQUAL_DISPATCH(ContinueStmt)
  EQUAL_DISPATCH(Function)
  EQUAL_DISPATCH_TRANSPARENT(Program)

  throw pypto::TypeError("Unknown IR node type in StructuralEqualImpl::Equal: " + lhs->TypeName());
}

#undef EQUAL_DISPATCH
#undef EQUAL_DISPATCH_TRANSPARENT

template <bool AssertMode>
bool StructuralEqualImpl<AssertMode>::EqualAssignStmt(const AssignStmtPtr& lhs, const AssignStmtPtr& rhs) {
  TransparentDepthResetGuard<AssertMode> depth_guard(transparent_depth_);
  if (!rhs) {
    if constexpr (AssertMode) {
      ThrowMismatch("Type cast failed for AssignStmt", IRNodePtr(), IRNodePtr(), "", "");
    }
    return false;
  }

  PushFieldName("var");
  bool var_equal = false;
  VisitDefField([&]() { var_equal = Equal(lhs->var_, rhs->var_); });
  PopFieldName();
  if (!var_equal) {
    return false;
  }

  ExprPtr lhs_value = lhs->value_;
  ExprPtr rhs_value = rhs->value_;
  if (auto lhs_call = As<Call>(lhs->value_)) {
    auto lhs_assigned_type = InferAssignedCallTypeForRoundtripCompare(lhs->var_, lhs_call);
    if (lhs_assigned_type && lhs_assigned_type != lhs_call->GetType()) {
      lhs_value = std::make_shared<Call>(lhs_call->op_, lhs_call->args_, lhs_call->kwargs_, lhs_assigned_type,
                                         lhs_call->span_);
    }
  }
  if (auto rhs_call = As<Call>(rhs->value_)) {
    auto rhs_assigned_type = InferAssignedCallTypeForRoundtripCompare(rhs->var_, rhs_call);
    if (rhs_assigned_type && rhs_assigned_type != rhs_call->GetType()) {
      rhs_value = std::make_shared<Call>(rhs_call->op_, rhs_call->args_, rhs_call->kwargs_, rhs_assigned_type,
                                         rhs_call->span_);
    }
  }

  PushFieldName("value");
  bool value_equal = Equal(lhs_value, rhs_value);
  PopFieldName();
  return value_equal;
}

template <bool AssertMode>
bool StructuralEqualImpl<AssertMode>::EqualCall(const CallPtr& lhs, const CallPtr& rhs) {
  TransparentDepthResetGuard<AssertMode> depth_guard(transparent_depth_);
  if (!rhs) {
    if constexpr (AssertMode) {
      ThrowMismatch("Type cast failed for Call", IRNodePtr(), IRNodePtr(), "", "");
    }
    return false;
  }

  const auto lhs_canonical = CanonicalizeCallForRoundtripCompare(lhs);
  const auto rhs_canonical = CanonicalizeCallForRoundtripCompare(rhs);

  PushFieldName("op");
  bool op_equal = lhs_canonical.op_name == rhs_canonical.op_name;
  if constexpr (AssertMode) {
    if (!op_equal) {
      std::ostringstream msg;
      msg << "Operator name mismatch ('" << lhs->op_->name_ << "' != '" << rhs->op_->name_ << "')";
      ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
    }
  }
  PopFieldName();
  if (!op_equal) {
    return false;
  }

  PushFieldName("args");
  bool args_equal = VisitIRNodeVectorField(lhs_canonical.args, rhs_canonical.args);
  PopFieldName();
  if (!args_equal) {
    return false;
  }

  PushFieldName("kwargs");
  bool kwargs_equal = VisitLeafField(lhs_canonical.kwargs, rhs_canonical.kwargs);
  PopFieldName();
  if (!kwargs_equal) {
    return false;
  }

  PushFieldName("type");
  bool type_equal = false;
  auto lhs_unknown = As<UnknownType>(lhs->GetType());
  auto rhs_unknown = As<UnknownType>(rhs->GetType());
  if (lhs_unknown && !rhs_unknown) {
    auto deduced_lhs_type = DeduceCanonicalCallTypeForRoundtripCompare(
        lhs_canonical.op_name, lhs->span_, lhs_canonical.args, lhs_canonical.kwargs);
    type_equal = deduced_lhs_type && EqualType(deduced_lhs_type, rhs->GetType());
  } else if (!lhs_unknown && rhs_unknown) {
    auto deduced_rhs_type = DeduceCanonicalCallTypeForRoundtripCompare(
        rhs_canonical.op_name, rhs->span_, rhs_canonical.args, rhs_canonical.kwargs);
    type_equal = deduced_rhs_type && EqualType(lhs->GetType(), deduced_rhs_type);
  } else {
    type_equal = EqualType(lhs->GetType(), rhs->GetType());
  }
  PopFieldName();
  return type_equal;
}

template <bool AssertMode>
bool StructuralEqualImpl<AssertMode>::EqualBinaryExpr(const BinaryExprPtr& lhs, const BinaryExprPtr& rhs) {
  TransparentDepthResetGuard<AssertMode> depth_guard(transparent_depth_);
  if (!rhs) {
    if constexpr (AssertMode) {
      ThrowMismatch("Type cast failed for BinaryExpr", IRNodePtr(), IRNodePtr(), "", "");
    }
    return false;
  }

  PushFieldName("type");
  bool type_equal = IsPredicateBinaryExpr(lhs)
                        ? AreBoolIntegerLegacyPredicateTypesCompatible(lhs->GetType(), rhs->GetType())
                        : EqualType(lhs->GetType(), rhs->GetType());
  if constexpr (AssertMode) {
    if (!type_equal) {
      std::ostringstream msg;
      msg << "BinaryExpr type mismatch";
      ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
    }
  }
  PopFieldName();
  if (!type_equal) {
    return false;
  }

  PushFieldName("left");
  bool left_equal = Equal(lhs->left_, rhs->left_);
  PopFieldName();
  if (!left_equal) {
    return false;
  }

  PushFieldName("right");
  bool right_equal = Equal(lhs->right_, rhs->right_);
  PopFieldName();
  return right_equal;
}

template <bool AssertMode>
bool StructuralEqualImpl<AssertMode>::EqualUnaryExpr(const UnaryExprPtr& lhs, const UnaryExprPtr& rhs) {
  TransparentDepthResetGuard<AssertMode> depth_guard(transparent_depth_);
  if (!rhs) {
    if constexpr (AssertMode) {
      ThrowMismatch("Type cast failed for UnaryExpr", IRNodePtr(), IRNodePtr(), "", "");
    }
    return false;
  }

  PushFieldName("type");
  bool type_equal = IsPredicateUnaryExpr(lhs)
                        ? AreBoolIntegerLegacyPredicateTypesCompatible(lhs->GetType(), rhs->GetType())
                        : EqualType(lhs->GetType(), rhs->GetType());
  if constexpr (AssertMode) {
    if (!type_equal) {
      std::ostringstream msg;
      msg << "UnaryExpr type mismatch";
      ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
    }
  }
  PopFieldName();
  if (!type_equal) {
    return false;
  }

  PushFieldName("operand");
  bool operand_equal = Equal(lhs->operand_, rhs->operand_);
  PopFieldName();
  return operand_equal;
}

template <bool AssertMode>
bool StructuralEqualImpl<AssertMode>::EqualForStmt(const ForStmtPtr& lhs, const ForStmtPtr& rhs) {
  TransparentDepthResetGuard<AssertMode> depth_guard(transparent_depth_);
  if (!rhs) {
    if constexpr (AssertMode) {
      ThrowMismatch("Type cast failed for ForStmt", IRNodePtr(), IRNodePtr(), "", "");
    }
    return false;
  }

  PushFieldName("loop_var");
  bool loop_var_equal = false;
  VisitDefField([&]() { loop_var_equal = Equal(lhs->loop_var_, rhs->loop_var_); });
  PopFieldName();
  if (!loop_var_equal) {
    return false;
  }

  PushFieldName("start");
  bool start_equal = Equal(lhs->start_, rhs->start_);
  PopFieldName();
  if (!start_equal) {
    return false;
  }

  PushFieldName("stop");
  bool stop_equal = Equal(lhs->stop_, rhs->stop_);
  PopFieldName();
  if (!stop_equal) {
    return false;
  }

  PushFieldName("step");
  bool step_equal = Equal(lhs->step_, rhs->step_);
  PopFieldName();
  if (!step_equal) {
    return false;
  }

  PushFieldName("iter_args");
  bool iter_args_equal = false;
  VisitDefField([&]() { iter_args_equal = VisitIRNodeVectorField(lhs->iter_args_, rhs->iter_args_); });
  PopFieldName();
  if (!iter_args_equal) {
    return false;
  }

  PushFieldName("body");
  bool body_equal = Equal(lhs->body_, rhs->body_);
  PopFieldName();
  if (!body_equal) {
    return false;
  }

  PushFieldName("return_vars");
  bool return_vars_equal = false;
  VisitDefField([&]() { return_vars_equal = VisitIRNodeVectorField(lhs->return_vars_, rhs->return_vars_); });
  PopFieldName();
  if (!return_vars_equal) {
    return false;
  }

  PushFieldName("kind");
  bool kind_equal = lhs->kind_ == rhs->kind_;
  if (!kind_equal) {
    const bool lhs_lowered_unroll =
        lhs->kind_ == ForKind::Unroll && rhs->kind_ == ForKind::Sequential && !lhs->iter_args_.empty();
    const bool rhs_lowered_unroll =
        rhs->kind_ == ForKind::Unroll && lhs->kind_ == ForKind::Sequential && !rhs->iter_args_.empty();
    kind_equal = lhs_lowered_unroll || rhs_lowered_unroll;
  }
  if constexpr (AssertMode) {
    if (!kind_equal) {
      std::ostringstream msg;
      msg << "ForKind mismatch (" << ForKindToString(lhs->kind_) << " != " << ForKindToString(rhs->kind_)
          << ")";
      ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
    }
  }
  PopFieldName();
  if (!kind_equal) {
    return false;
  }

  PushFieldName("chunk_size");
  bool chunk_size_equal = VisitIRNodeField(lhs->chunk_size_, rhs->chunk_size_);
  PopFieldName();
  if (!chunk_size_equal) {
    return false;
  }

  PushFieldName("chunk_policy");
  bool chunk_policy_equal = VisitLeafField(lhs->chunk_policy_, rhs->chunk_policy_);
  PopFieldName();
  return chunk_policy_equal;
}

template <bool AssertMode>
bool StructuralEqualImpl<AssertMode>::EqualType(const TypePtr& lhs, const TypePtr& rhs) {
  if (lhs->TypeName() != rhs->TypeName()) {
    if constexpr (AssertMode) {
      std::ostringstream msg;
      msg << "Type name mismatch (" << lhs->TypeName() << " != " << rhs->TypeName() << ")";
      ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
    }
    return false;
  }

  if (auto lhs_scalar = As<ScalarType>(lhs)) {
    auto rhs_scalar = As<ScalarType>(rhs);
    if (!rhs_scalar) {
      if constexpr (AssertMode) {
        ThrowMismatch("Type cast failed for ScalarType", IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    if (!AreIndexCompatibleIntegerTypes(lhs_scalar->dtype_, rhs_scalar->dtype_)) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "ScalarType dtype mismatch (" << lhs_scalar->dtype_.ToString()
            << " != " << rhs_scalar->dtype_.ToString() << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  } else if (auto lhs_tensor = As<TensorType>(lhs)) {
    auto rhs_tensor = As<TensorType>(rhs);
    if (!rhs_tensor) {
      if constexpr (AssertMode) {
        ThrowMismatch("Type cast failed for TensorType", IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    if (lhs_tensor->dtype_ != rhs_tensor->dtype_) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "TensorType dtype mismatch (" << lhs_tensor->dtype_.ToString()
            << " != " << rhs_tensor->dtype_.ToString() << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    if (lhs_tensor->shape_.size() != rhs_tensor->shape_.size()) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "TensorType shape rank mismatch (" << lhs_tensor->shape_.size()
            << " != " << rhs_tensor->shape_.size() << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    for (size_t i = 0; i < lhs_tensor->shape_.size(); ++i) {
      if (AreShapeConstIntsCompatible(lhs_tensor->shape_[i], rhs_tensor->shape_[i])) continue;
      if (!Equal(lhs_tensor->shape_[i], rhs_tensor->shape_[i])) return false;
    }
    // Compare tensor_view. Missing TensorView is semantically equivalent to an
    // explicit default TensorView() with valid_shape = shape.
    const TensorView lhs_tv = NormalizeTensorViewForCompare(lhs_tensor->tensor_view_, lhs_tensor->shape_);
    const TensorView rhs_tv = NormalizeTensorViewForCompare(rhs_tensor->tensor_view_, rhs_tensor->shape_);
    if (lhs_tv.valid_shape.size() != rhs_tv.valid_shape.size()) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "TensorView valid_shape size mismatch (" << lhs_tv.valid_shape.size()
            << " != " << rhs_tv.valid_shape.size() << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    for (size_t i = 0; i < lhs_tv.valid_shape.size(); ++i) {
      if (AreShapeConstIntsCompatible(lhs_tv.valid_shape[i], rhs_tv.valid_shape[i])) continue;
      if (!Equal(lhs_tv.valid_shape[i], rhs_tv.valid_shape[i])) return false;
    }
    // Compare stride
    if (lhs_tv.stride.size() != rhs_tv.stride.size()) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "TensorView stride size mismatch (" << lhs_tv.stride.size() << " != " << rhs_tv.stride.size()
            << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    for (size_t i = 0; i < lhs_tv.stride.size(); ++i) {
      if (AreShapeConstIntsCompatible(lhs_tv.stride[i], rhs_tv.stride[i])) continue;
      if (!Equal(lhs_tv.stride[i], rhs_tv.stride[i])) return false;
    }
    // Compare layout
    if (lhs_tv.layout != rhs_tv.layout) {
      if constexpr (AssertMode) {
        ThrowMismatch("TensorView layout mismatch", IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  } else if (auto lhs_tile = As<TileType>(lhs)) {
    auto rhs_tile = As<TileType>(rhs);
    if (!rhs_tile) {
      if constexpr (AssertMode) {
        ThrowMismatch("Type cast failed for TileType", IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    // Compare dtype
    if (lhs_tile->dtype_ != rhs_tile->dtype_) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "TileType dtype mismatch (" << lhs_tile->dtype_.ToString()
            << " != " << rhs_tile->dtype_.ToString() << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    // Compare shape size and dimensions
    if (lhs_tile->shape_.size() != rhs_tile->shape_.size()) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "TileType shape rank mismatch (" << lhs_tile->shape_.size()
            << " != " << rhs_tile->shape_.size() << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    for (size_t i = 0; i < lhs_tile->shape_.size(); ++i) {
      if (AreShapeConstIntsCompatible(lhs_tile->shape_[i], rhs_tile->shape_[i])) continue;
      if (!Equal(lhs_tile->shape_[i], rhs_tile->shape_[i])) return false;
    }
    // Compare tile_view. Missing TileView is semantically equivalent to an
    // explicit default TileView() with valid_shape = shape.
    const TileView lhs_tv =
        NormalizeTileViewForCompare(lhs_tile->tile_view_, lhs_tile->shape_, lhs_tile->memory_space_);
    const TileView rhs_tv =
        NormalizeTileViewForCompare(rhs_tile->tile_view_, rhs_tile->shape_, rhs_tile->memory_space_);
    if (lhs_tv.valid_shape.size() != rhs_tv.valid_shape.size()) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "TileView valid_shape size mismatch (" << lhs_tv.valid_shape.size()
            << " != " << rhs_tv.valid_shape.size() << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    for (size_t i = 0; i < lhs_tv.valid_shape.size(); ++i) {
      if (AreShapeConstIntsCompatible(lhs_tv.valid_shape[i], rhs_tv.valid_shape[i])) continue;
      if (!Equal(lhs_tv.valid_shape[i], rhs_tv.valid_shape[i])) return false;
    }
    // Compare stride
    if (lhs_tv.stride.size() != rhs_tv.stride.size()) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "TileView stride size mismatch (" << lhs_tv.stride.size() << " != " << rhs_tv.stride.size()
            << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    for (size_t i = 0; i < lhs_tv.stride.size(); ++i) {
      if (AreShapeConstIntsCompatible(lhs_tv.stride[i], rhs_tv.stride[i])) continue;
      if (!Equal(lhs_tv.stride[i], rhs_tv.stride[i])) return false;
    }
    // Compare start_offset
    if (AreShapeConstIntsCompatible(lhs_tv.start_offset, rhs_tv.start_offset)) {
      // no-op
    } else if (!Equal(lhs_tv.start_offset, rhs_tv.start_offset)) {
      return false;
    }
    // Compare blayout
    if (lhs_tv.blayout != rhs_tv.blayout) {
      if constexpr (AssertMode) {
        ThrowMismatch("TileView blayout mismatch", IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    // Compare slayout
    if (lhs_tv.slayout != rhs_tv.slayout) {
      if constexpr (AssertMode) {
        ThrowMismatch("TileView slayout mismatch", IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    // Compare fractal
    if (lhs_tv.fractal != rhs_tv.fractal) {
      if constexpr (AssertMode) {
        ThrowMismatch("TileView fractal mismatch", IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    // Compare pad
    if (lhs_tv.pad != rhs_tv.pad) {
      if constexpr (AssertMode) {
        ThrowMismatch("TileView pad mismatch", IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    // Compare memory_space
    if (lhs_tile->memory_space_.has_value() != rhs_tile->memory_space_.has_value()) {
      if constexpr (AssertMode) {
        ThrowMismatch("TileType memory_space presence mismatch", IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    if (lhs_tile->memory_space_.has_value() &&
        lhs_tile->memory_space_.value() != rhs_tile->memory_space_.value()) {
      if constexpr (AssertMode) {
        ThrowMismatch("TileType memory_space mismatch", IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  } else if (auto lhs_tuple = As<TupleType>(lhs)) {
    auto rhs_tuple = As<TupleType>(rhs);
    if (!rhs_tuple) {
      if constexpr (AssertMode) {
        ThrowMismatch("Type cast failed for TupleType", IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    if (lhs_tuple->types_.size() != rhs_tuple->types_.size()) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "TupleType size mismatch (" << lhs_tuple->types_.size() << " != " << rhs_tuple->types_.size()
            << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    for (size_t i = 0; i < lhs_tuple->types_.size(); ++i) {
      if (!EqualType(lhs_tuple->types_[i], rhs_tuple->types_[i])) return false;
    }
    return true;
  } else if (IsA<MemRefType>(lhs) || IsA<UnknownType>(lhs)) {
    return true;  // Singleton type, both being MemRefType or UnknownType is sufficient
  }

  INTERNAL_UNREACHABLE << "EqualType encountered unhandled Type: " << lhs->TypeName();
  return false;
}

template <bool AssertMode>
bool StructuralEqualImpl<AssertMode>::EqualVar(const VarPtr& lhs, const VarPtr& rhs) {
  if (!enable_auto_mapping_) {
    auto lhs_it = lhs_to_rhs_var_map_.find(lhs);
    auto rhs_it = rhs_to_lhs_var_map_.find(rhs);
    // Case 1: already mapped to the same variable
    if (lhs_it != lhs_to_rhs_var_map_.end() && rhs_it != rhs_to_lhs_var_map_.end()) {
      if (lhs_it->second != rhs || rhs_it->second != lhs) {
        if constexpr (AssertMode) {
          ThrowMismatch("Variable mapping inconsistent (without auto-mapping)",
                        std::static_pointer_cast<const IRNode>(lhs),
                        std::static_pointer_cast<const IRNode>(rhs), "var " + lhs->name_hint_,
                        "var " + rhs->name_hint_);
        }
        return false;
      }
      return true;
    }
    // Case 2: different variables
    if (lhs.get() != rhs.get()) {
      if constexpr (AssertMode) {
        ThrowMismatch(
            "Variable pointer mismatch (without auto-mapping)", std::static_pointer_cast<const IRNode>(lhs),
            std::static_pointer_cast<const IRNode>(rhs), "var " + lhs->name_hint_, "var " + rhs->name_hint_);
      }
      return false;
    }
    return true;
  }

  if (!EqualType(lhs->GetType(), rhs->GetType())) {
    if constexpr (AssertMode) {
      std::ostringstream msg;
      msg << "Variable type mismatch (" << lhs->GetType()->TypeName() << " != " << rhs->GetType()->TypeName()
          << ")";
      ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
    }
    return false;
  }

  auto it = lhs_to_rhs_var_map_.find(lhs);
  if (it != lhs_to_rhs_var_map_.end()) {
    if (it->second != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "Variable mapping inconsistent ('" << lhs->name_hint_ << "' cannot map to both '"
            << it->second->name_hint_ << "' and '" << rhs->name_hint_ << "')";
        ThrowMismatch(msg.str(), std::static_pointer_cast<const IRNode>(lhs),
                      std::static_pointer_cast<const IRNode>(rhs));
      }
      return false;
    }
    return true;
  }

  auto rhs_it = rhs_to_lhs_var_map_.find(rhs);
  if (rhs_it != rhs_to_lhs_var_map_.end() && rhs_it->second != lhs) {
    if constexpr (AssertMode) {
      std::ostringstream msg;
      msg << "Variable mapping inconsistent ('" << rhs->name_hint_ << "' is already mapped from '"
          << rhs_it->second->name_hint_ << "', cannot map from '" << lhs->name_hint_ << "')";
      ThrowMismatch(msg.str(), std::static_pointer_cast<const IRNode>(lhs),
                    std::static_pointer_cast<const IRNode>(rhs));
    }
    return false;
  }

  lhs_to_rhs_var_map_[lhs] = rhs;
  rhs_to_lhs_var_map_[rhs] = lhs;
  return true;
}

template <bool AssertMode>
bool StructuralEqualImpl<AssertMode>::EqualMemRef(const MemRefPtr& lhs, const MemRefPtr& rhs) {
  // 1. First, compare as Var (handles variable mapping and type comparison)
  if (!EqualVar(lhs, rhs)) {
    return false;
  }

  // 2. Then, compare MemRef-specific fields (except id_ which is a naming counter)
  if (!Equal(lhs->addr_, rhs->addr_)) {
    if constexpr (AssertMode) {
      ThrowMismatch("MemRef addr mismatch", std::static_pointer_cast<const IRNode>(lhs),
                    std::static_pointer_cast<const IRNode>(rhs));
    }
    return false;
  }

  if (lhs->size_ != rhs->size_) {
    if constexpr (AssertMode) {
      std::ostringstream msg;
      msg << "MemRef size mismatch (" << lhs->size_ << " != " << rhs->size_ << ")";
      ThrowMismatch(msg.str(), std::static_pointer_cast<const IRNode>(lhs),
                    std::static_pointer_cast<const IRNode>(rhs));
    }
    return false;
  }

  return true;
}

template <bool AssertMode>
bool StructuralEqualImpl<AssertMode>::EqualIterArg(const IterArgPtr& lhs, const IterArgPtr& rhs) {
  // 1. First, compare as Var (handles variable mapping)
  if (!EqualVar(lhs, rhs)) {
    return false;
  }

  // 2. Then, compare IterArg-specific field: initValue_
  if (!Equal(lhs->initValue_, rhs->initValue_)) {
    if constexpr (AssertMode) {
      ThrowMismatch("IterArg initValue mismatch", std::static_pointer_cast<const IRNode>(lhs),
                    std::static_pointer_cast<const IRNode>(rhs));
    }
    return false;
  }

  return true;
}

// Explicit template instantiations
template class StructuralEqualImpl<false>;  // For structural_equal
template class StructuralEqualImpl<true>;   // For assert_structural_equal

// Type aliases for cleaner code
using StructuralEqual = StructuralEqualImpl<false>;
using StructuralEqualAssert = StructuralEqualImpl<true>;

// Public API implementation
bool structural_equal(const IRNodePtr& lhs, const IRNodePtr& rhs, bool enable_auto_mapping) {
  StructuralEqual checker(enable_auto_mapping);
  return checker(lhs, rhs);
}

bool structural_equal(const TypePtr& lhs, const TypePtr& rhs, bool enable_auto_mapping) {
  StructuralEqual checker(enable_auto_mapping);
  return checker(lhs, rhs);
}

// Public assert API
void assert_structural_equal(const IRNodePtr& lhs, const IRNodePtr& rhs, bool enable_auto_mapping) {
  StructuralEqualAssert checker(enable_auto_mapping);
  checker(lhs, rhs);
}

void assert_structural_equal(const TypePtr& lhs, const TypePtr& rhs, bool enable_auto_mapping) {
  StructuralEqualAssert checker(enable_auto_mapping);
  checker(lhs, rhs);
}

}  // namespace ir
}  // namespace pypto
