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

#include "pypto/ir/transforms/auto_layout/extractor.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/auto_layout/model.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/transforms/utils/packed_fp4_cast.h"
#include "pypto/ir/transforms/utils/reserve_buffer_utils.h"
#include "pypto/ir/transforms/utils/result_alias_utils.h"
#include "pypto/ir/transforms/utils/return_lineage_utils.h"
#include "pypto/ir/transforms/utils/tensor_view_semantics.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace auto_layout {

namespace detail {

std::string CleanName(const std::string& name_hint) { return auto_name::GetBaseName(name_hint); }

std::string TileLayoutPrefix(const ProgramPtr& program, const std::string& function, bool qualify_function,
                             std::optional<size_t> site) {
  const bool plain_name =
      !function.empty() && function.front() != '$' && function.find('.') == std::string::npos;
  if (site.has_value()) {
    const std::string index = std::to_string(*site);
    if (plain_name) {
      const std::string candidate = function + "_" + index;
      if (!program->GetFunction(candidate)) return candidate + ".";
    }
    // A length-prefixed name keeps the reserved namespace unambiguous even
    // for IR function names containing separators.
    return "$site." + std::to_string(function.size()) + "." + function + "." + index + ".";
  }
  if (!qualify_function) return {};
  if (plain_name) return function + ".";
  return "$func." + std::to_string(function.size()) + "." + function + ".";
}

TensorLayout TensorTypeLayout(const TensorTypePtr& tensor_type) {
  if (!tensor_type || !tensor_type->tensor_view_.has_value()) {
    return TensorLayout::ND;
  }
  return tensor_type->tensor_view_->layout;
}

std::vector<StmtPtr> IterChildStmts(const StmtPtr& stmt) {
  if (!stmt) {
    return {};
  }
  if (auto seq = As<SeqStmts>(stmt)) {
    return seq->stmts_;
  }
  if (auto ifs = As<IfStmt>(stmt)) {
    std::vector<StmtPtr> branches{ifs->then_body_};
    if (ifs->else_body_.has_value() && *ifs->else_body_) {
      branches.push_back(*ifs->else_body_);
    }
    return branches;
  }
  if (auto fs = As<ForStmt>(stmt)) {
    return fs->body_ ? std::vector<StmtPtr>{fs->body_} : std::vector<StmtPtr>{};
  }
  if (auto ws = As<WhileStmt>(stmt)) {
    return ws->body_ ? std::vector<StmtPtr>{ws->body_} : std::vector<StmtPtr>{};
  }
  if (auto scope = As<ScopeStmt>(stmt)) {
    return scope->body_ ? std::vector<StmtPtr>{scope->body_} : std::vector<StmtPtr>{};
  }
  return {};
}

std::optional<std::vector<int64_t>> StaticShape(const std::vector<ExprPtr>& exprs) {
  std::vector<int64_t> dims;
  dims.reserve(exprs.size());
  for (const auto& e : exprs) {
    auto ci = As<ConstInt>(e);
    if (!ci) {
      return std::nullopt;
    }
    dims.push_back(ci->value_);
  }
  return dims;
}

std::vector<std::pair<size_t, VarPtr>> TileOperands(const CallPtr& call) {
  std::vector<std::pair<size_t, VarPtr>> operands;
  for (size_t i = 0; i < call->args_.size(); ++i) {
    if (auto var = AsVarLike(call->args_[i]); var && As<TileType>(var->GetType())) {
      operands.emplace_back(i, std::move(var));
    }
  }
  return operands;
}

bool IsTileWorkspace(const CallPtr& call, size_t index) {
  const auto& registry = OpRegistry::GetInstance();
  if (!registry.IsRegistered(call->op_->name_)) return false;
  const auto& op = registry.GetEntry(call->op_->name_);
  return op.IsWorkspaceArg(index) || op.GetLaneInvariantArgKind(index) == LaneInvariantArg::Scratch;
}

bool CanUseVecNzPointwise(const CallPtr& call) {
  // These operations visit corresponding elements in the physically flattened
  // NZ allocation. TSELS reads a packed predicate with its own row geometry.
  static const std::unordered_set<std::string> ops = {
      "tile.add",      "tile.sub",      "tile.mul",      "tile.div",      "tile.maximum",  "tile.minimum",
      "tile.part_add", "tile.part_max", "tile.part_min", "tile.part_mul", "tile.addc",     "tile.subc",
      "tile.selc",     "tile.abs",      "tile.exp",      "tile.neg",      "tile.sqrt",     "tile.rsqrt",
      "tile.log",      "tile.recip",    "tile.not",      "tile.relu",     "tile.lrelu",    "tile.adds",
      "tile.subs",     "tile.muls",     "tile.divs",     "tile.maximums", "tile.minimums", "tile.and",
      "tile.or",       "tile.shl",      "tile.shr",      "tile.ands",     "tile.ors",      "tile.shls",
      "tile.shrs",     "tile.fmod",     "tile.fmods",    "tile.addsc",    "tile.subsc",
  };
  const auto result = As<TileType>(call->GetType());
  if (!result || result->dtype_.IsPackedFp4()) return false;
  const bool cast = IsOp(call, "tile.cast");
  if ((!cast && ops.count(call->op_->name_) == 0) || IsDensePackedFp4DnCast(call)) return false;

  const auto& entry = OpRegistry::GetInstance().GetEntry(call->op_->name_);
  const auto& pointwise = entry.GetPointwiseValidityArgs();
  for (const auto& [index, var] : TileOperands(call)) {
    const auto tile = As<TileType>(var->GetType());
    if (IsTileWorkspace(call, index) || entry.IsLaneInvariantArg(index) ||
        (!pointwise.empty() && pointwise.count(index) == 0) || tile->dtype_.IsPackedFp4()) {
      return false;
    }
    if (cast && DTypeByteWidthSafe(tile->dtype_) != DTypeByteWidthSafe(result->dtype_)) return false;
  }
  return true;
}

bool PreservesTileDescriptors(const CallPtr& call) {
  if (IsOp(call, "tile.reshape") || IsOp(call, "tile.full") || IsDensePackedFp4DnCast(call)) return true;
  if (!IsOp(call, "tile.cast") || call->args_.empty()) return false;
  const auto source = As<TileType>(call->args_[0]->GetType());
  const auto result = As<TileType>(call->GetType());
  return source && result && (source->dtype_.IsPackedFp4() || result->dtype_.IsPackedFp4());
}

bool CanCopySingletonRowSum(const CallPtr& call) {
  if (!IsOp(call, "tile.row_sum") || call->args_.empty()) return false;
  const auto source = As<TileType>(call->args_[0]->GetType());
  const auto result = As<TileType>(call->GetType());
  if (!source || !result || source->dtype_ != result->dtype_ ||
      EffectiveTileLayout(source) != LayoutSpec::ColMajorFlat() ||
      EffectiveTileLayout(result) != LayoutSpec::ColMajorFlat()) {
    return false;
  }
  const auto dtype = source->dtype_;
  if (dtype != DataType::INT16 && dtype != DataType::INT32 && dtype != DataType::FP16 &&
      dtype != DataType::FP32) {
    return false;
  }
  const int bytes = DTypeByteWidthSafe(dtype);
  const auto shape = StaticShape(source->shape_);
  if (!shape || shape->size() != 2 || (*shape)[0] <= 1 || (*shape)[1] != 1 ||
      (*shape)[0] % (32 / bytes) != 0 || StaticShape(result->shape_) != shape) {
    return false;
  }
  const auto view = tile_view_semantics::GetEffectiveTileView(*source);
  if (!view.stride.empty() || view.start_offset || view.compact != CompactMode::null ||
      view.pad != PadValue::null) {
    return false;
  }
  const auto valid = StaticShape(view.valid_shape);
  return valid && valid->size() == 2 && (*valid)[0] > 0 && (*valid)[0] <= (*shape)[0] && (*valid)[1] == 1 &&
         StaticShape(tile_view_semantics::GetEffectiveTileView(*result).valid_shape) == valid;
}

ValueNode VecTransposeGeometry(const TileTypePtr& tile) {
  ValueNode geometry;
  geometry.shape = StaticShape(tile->shape_).value_or(std::vector<int64_t>{});
  geometry.dtype = tile->dtype_;
  geometry.memory_space = tile->GetMemorySpace().value_or(MemorySpace::Vec);
  const auto view = tile_view_semantics::GetEffectiveTileView(*tile);
  const auto offset = As<ConstInt>(view.start_offset);
  geometry.has_dense_vec_span = view.stride.empty() &&
                                (!view.start_offset || (offset && offset->value_ == 0)) &&
                                view.pad == PadValue::null && view.compact == CompactMode::null &&
                                tile_view_semantics::ShapeExprListsEquivalent(view.valid_shape, tile->shape_);
  return geometry;
}

std::optional<size_t> TileBackingOperand(const CallPtr& call) {
  if (IsOp(call, "tile.slice") || IsOp(call, "tile.reshape") || IsOp(call, "tile.reinterpret_view") ||
      IsOp(call, "tile.transpose_view")) {
    return 0;
  }
  return ResultAliasedArgIndex(call);
}

bool AliasesFirstTileOperand(const CallPtr& call) { return TileBackingOperand(call) == 0; }

const std::unordered_set<std::string>& RowReductionOps() {
  static const std::unordered_set<std::string> kOps = {
      "tile.row_sum", "tile.row_max", "tile.row_min", "tile.row_prod", "tile.row_argmax", "tile.row_argmin",
  };
  return kOps;
}

const std::unordered_set<std::string>& ColReductionOps() {
  static const std::unordered_set<std::string> kOps = {
      "tile.col_sum", "tile.col_max", "tile.col_min", "tile.col_prod", "tile.col_argmax", "tile.col_argmin",
  };
  return kOps;
}

const std::unordered_set<std::string>& OtherRowExpandOps() {
  static const std::unordered_set<std::string> kOps = {
      "tile.row_expand",     "tile.row_expand_sub", "tile.row_expand_mul",    "tile.row_expand_div",
      "tile.row_expand_max", "tile.row_expand_min", "tile.row_expand_expdif",
  };
  return kOps;
}

const std::unordered_set<std::string>& ColExpandOps() {
  static const std::unordered_set<std::string> kOps = {
      "tile.col_expand",     "tile.col_expand_mul", "tile.col_expand_add", "tile.col_expand_div",
      "tile.col_expand_sub", "tile.col_expand_max", "tile.col_expand_min", "tile.col_expand_expdif",
  };
  return kOps;
}

bool HasRowMajorInputRequirement(const std::string& op_name, size_t input_index) {
  if (!backend::BackendConfig::IsConfigured()) {
    return false;
  }
  const auto* be = backend::BackendConfig::GetBackend();
  if (be == nullptr) {
    return false;
  }
  const auto* spec = be->GetTileLayoutSpec(op_name);
  return spec != nullptr && input_index < spec->input_layouts.size() &&
         spec->input_layouts[input_index] == TileLayout::row_major;
}

}  // namespace detail

namespace {

using detail::CanUseVecNzPointwise;
using detail::CleanName;
using detail::ColExpandOps;
using detail::ColReductionOps;
using detail::IsTileWorkspace;
using detail::IterChildStmts;
using detail::OtherRowExpandOps;
using detail::RowReductionOps;
using detail::StaticShape;
using detail::TileOperands;

/// Disjoint-set union keyed directly on `const Var*` pointer identity.
class VarUnionFind {
 public:
  const Var* Find(const Var* x) {
    if (x == nullptr) {
      return nullptr;
    }
    auto it = parent_.find(x);
    if (it == parent_.end()) {
      parent_.emplace(x, x);
      return x;
    }
    const Var* root = x;
    while (parent_[root] != root) {
      root = parent_[root];
    }
    const Var* cur = x;
    while (cur != root) {
      const Var* next = parent_[cur];
      parent_[cur] = root;
      cur = next;
    }
    return root;
  }

  const Var* Union(const Var* a, const Var* b) {
    const Var* ra = Find(a);
    const Var* rb = Find(b);
    if (ra == nullptr) return rb;
    if (rb == nullptr) return ra;
    if (ra != rb) {
      parent_[rb] = ra;
    }
    return ra;
  }

 private:
  std::unordered_map<const Var*, const Var*> parent_;
};

const backend::Backend* ResolveBackend(const LayoutCostModel& model) {
  if (model.backend_type.has_value()) {
    return backend::GetBackendInstance(*model.backend_type);
  }
  if (backend::BackendConfig::IsConfigured()) {
    return backend::BackendConfig::GetBackend();
  }
  return nullptr;
}

std::optional<TileLayout> RequiredInputTileLayout(const backend::Backend* be, const std::string& op_name,
                                                  size_t input_index) {
  if (be == nullptr) {
    return std::nullopt;
  }
  const auto* spec = be->GetTileLayoutSpec(op_name);
  if (spec == nullptr || input_index >= spec->input_layouts.size()) {
    return std::nullopt;
  }
  return spec->input_layouts[input_index];
}

std::pair<std::vector<int64_t>, bool> TensorShapeWithDynamicFallback(const std::vector<ExprPtr>& exprs,
                                                                     int64_t default_dyn = 128) {
  std::vector<int64_t> dims;
  dims.reserve(exprs.size());
  bool is_static = true;
  for (const auto& e : exprs) {
    if (auto ci = As<ConstInt>(e)) {
      dims.push_back(ci->value_);
    } else {
      dims.push_back(default_dyn);
      is_static = false;
    }
  }
  return {std::move(dims), is_static};
}

std::optional<int64_t> EvalScalarExpr(const ExprPtr& expr,
                                      const std::unordered_map<const Var*, ExprPtr>& scalar_defs,
                                      std::optional<int64_t> default_dyn = 128, int depth = 0) {
  if (!expr || depth > 64) {
    return std::nullopt;
  }
  if (auto ci = As<ConstInt>(expr)) {
    return ci->value_;
  }
  if (auto v = AsVarLike(expr)) {
    auto it = scalar_defs.find(v.get());
    if (it != scalar_defs.end()) {
      return EvalScalarExpr(it->second, scalar_defs, default_dyn, depth + 1);
    }
    if (default_dyn.has_value()) {
      std::string upper;
      upper.reserve(v->name_hint_.size());
      for (unsigned char c : v->name_hint_) {
        upper.push_back(static_cast<char>(std::toupper(c)));
      }
      if (upper.find("DYN") != std::string::npos || upper.find("TOKEN") != std::string::npos ||
          upper.find("ROW") != std::string::npos || upper.find("BATCH") != std::string::npos ||
          upper.find("SEQ") != std::string::npos) {
        return default_dyn;
      }
    }
    return std::nullopt;
  }
  if (auto add = As<Add>(expr)) {
    auto lhs = EvalScalarExpr(add->left_, scalar_defs, default_dyn, depth + 1);
    auto rhs = EvalScalarExpr(add->right_, scalar_defs, default_dyn, depth + 1);
    return (lhs && rhs) ? std::optional<int64_t>(*lhs + *rhs) : std::nullopt;
  }
  if (auto sub = As<Sub>(expr)) {
    auto lhs = EvalScalarExpr(sub->left_, scalar_defs, default_dyn, depth + 1);
    auto rhs = EvalScalarExpr(sub->right_, scalar_defs, default_dyn, depth + 1);
    if (!lhs || !rhs) return std::nullopt;
    return default_dyn.has_value() ? std::max<int64_t>(0, *lhs - *rhs) : (*lhs - *rhs);
  }
  if (auto mul = As<Mul>(expr)) {
    auto lhs = EvalScalarExpr(mul->left_, scalar_defs, default_dyn, depth + 1);
    auto rhs = EvalScalarExpr(mul->right_, scalar_defs, default_dyn, depth + 1);
    return (lhs && rhs) ? std::optional<int64_t>(*lhs * *rhs) : std::nullopt;
  }
  if (auto fdiv = As<FloorDiv>(expr)) {
    auto lhs = EvalScalarExpr(fdiv->left_, scalar_defs, default_dyn, depth + 1);
    auto rhs = EvalScalarExpr(fdiv->right_, scalar_defs, default_dyn, depth + 1);
    return (lhs && rhs && *rhs > 0) ? std::optional<int64_t>(*lhs / *rhs) : std::nullopt;
  }
  if (auto min_e = As<Min>(expr)) {
    auto lhs = EvalScalarExpr(min_e->left_, scalar_defs, default_dyn, depth + 1);
    auto rhs = EvalScalarExpr(min_e->right_, scalar_defs, default_dyn, depth + 1);
    return (lhs && rhs) ? std::optional<int64_t>(std::min(*lhs, *rhs)) : std::nullopt;
  }
  if (auto max_e = As<Max>(expr)) {
    auto lhs = EvalScalarExpr(max_e->left_, scalar_defs, default_dyn, depth + 1);
    auto rhs = EvalScalarExpr(max_e->right_, scalar_defs, default_dyn, depth + 1);
    return (lhs && rhs) ? std::optional<int64_t>(std::max(*lhs, *rhs)) : std::nullopt;
  }
  return std::nullopt;
}

int64_t RangeTripCount(int64_t start, int64_t stop, int64_t step) {
  if (step == 0 || (step > 0 && start >= stop) || (step < 0 && start <= stop)) return 0;
  const auto start_bits = static_cast<uint64_t>(start);
  const auto stop_bits = static_cast<uint64_t>(stop);
  const auto step_bits = static_cast<uint64_t>(step);
  const auto span = step > 0 ? stop_bits - start_bits : start_bits - stop_bits;
  const auto stride = step > 0 ? step_bits : uint64_t{0} - step_bits;
  return static_cast<int64_t>(
      std::min<uint64_t>(1 + (span - 1) / stride, std::numeric_limits<int64_t>::max()));
}

int64_t ForTripCount(const ForStmtPtr& op, const std::unordered_map<const Var*, ExprPtr>& scalar_defs) {
  auto step_val = EvalScalarExpr(op->step_, scalar_defs);
  if (!step_val || *step_val <= 0) {
    return 1;
  }
  auto stop_val = EvalScalarExpr(op->stop_, scalar_defs);
  if (!stop_val || *stop_val <= 0) {
    return 1;
  }
  auto start_val = EvalScalarExpr(op->start_, scalar_defs);
  if (start_val.has_value()) {
    return std::max<int64_t>(1, RangeTripCount(*start_val, *stop_val, *step_val));
  }
  if (As<Var>(op->start_)) {
    return std::max<int64_t>(1, RangeTripCount(0, *stop_val, *step_val));
  }
  return 1;
}

std::optional<int64_t> StaticInCoreForTripCount(const ForStmtPtr& op,
                                                const std::unordered_map<const Var*, ExprPtr>& scalar_defs) {
  const auto start = EvalScalarExpr(op->start_, scalar_defs, std::nullopt);
  const auto stop = EvalScalarExpr(op->stop_, scalar_defs, std::nullopt);
  const auto step = EvalScalarExpr(op->step_, scalar_defs, std::nullopt);
  if (!start || !stop || !step || *step == 0) return std::nullopt;
  return RangeTripCount(*start, *stop, *step);
}

int64_t InvocationCount(int64_t enclosing, int64_t repetitions) {
  if (enclosing <= 0 || repetitions <= 0) return 0;
  constexpr int64_t max_count = std::numeric_limits<int64_t>::max();
  return enclosing > max_count / repetitions ? max_count : enclosing * repetitions;
}

bool IsTileMatmulOp(const std::string& op_name) {
  return op_name == "tile.matmul" || op_name == "tile.matmul_acc" || op_name == "tile.matmul_bias";
}

bool IsWholeTensorFlatten(const CallPtr& call) {
  if (!IsOp(call, "tensor.reshape") || call->args_.size() < 2) {
    return false;
  }
  auto src_type = AsTensorTypeLike(call->args_[0]->GetType());
  if (!src_type) {
    return false;
  }
  auto src_shape = StaticShape(src_type->shape_);
  if (!src_shape.has_value()) {
    return false;
  }
  auto shape_tuple = As<MakeTuple>(call->args_[1]);
  if (!shape_tuple || shape_tuple->elements_.size() != 1) {
    return false;
  }
  auto flat_ext = As<ConstInt>(shape_tuple->elements_[0]);
  if (!flat_ext) {
    return false;
  }
  int64_t prod = 1;
  for (int64_t d : *src_shape) {
    prod *= d;
  }
  return prod == flat_ext->value_;
}

std::optional<std::vector<int64_t>> IsNzCompatibleTensorType(const TypePtr& type, size_t max_rank = 3) {
  if (IsA<DistributedTensorType>(type)) {
    return std::nullopt;
  }
  auto tensor_type = As<TensorType>(type);
  if (!tensor_type || tensor_view_semantics::IsNzBlocked(*tensor_type)) {
    return std::nullopt;
  }
  if (tensor_type->tensor_view_.has_value() &&
      (!tensor_type->tensor_view_->valid_shape.empty() || !tensor_type->tensor_view_->stride.empty())) {
    return std::nullopt;
  }
  auto shape = StaticShape(tensor_type->shape_);
  if (!shape || shape->size() < 2 || shape->size() > max_rank ||
      std::any_of(shape->begin(), shape->end(), [](int64_t extent) { return extent <= 0; })) {
    return std::nullopt;
  }
  if (!IsNzCompatibleShape({(*shape)[shape->size() - 2], shape->back()}, tensor_type->dtype_)) {
    return std::nullopt;
  }
  return shape;
}

bool HasDenseLogicalInputLayout(const TensorTypePtr& tensor) {
  if (!tensor || IsA<DistributedTensorType>(tensor) || tensor->dtype_.IsPackedFp4()) return false;
  const auto& view = tensor->tensor_view_;
  return !view ||
         ((view->layout == TensorLayout::ND || view->layout == TensorLayout::NZ) && !view->nz_blocked &&
          view->stride.empty() && view->valid_shape.empty() && view->pad == PadValue::null);
}

bool IsNzCompatibleTensorSliceCall(const CallPtr& call, const TypePtr& tensor_type, size_t max_rank = 3) {
  auto shape = IsNzCompatibleTensorType(tensor_type, max_rank);
  auto tt = As<TensorType>(tensor_type);
  if (!shape.has_value() || !tt || call->args_.size() < 3) {
    return false;
  }
  const size_t rank = shape->size();
  auto shapes_arg = As<MakeTuple>(call->args_[1]);
  auto offsets_arg = As<MakeTuple>(call->args_[2]);
  if (!shapes_arg || !offsets_arg || shapes_arg->elements_.size() != rank ||
      offsets_arg->elements_.size() != rank) {
    return false;
  }
  auto shapes_static = StaticShape(shapes_arg->elements_);
  if (!shapes_static || (*shapes_static)[rank - 2] != (*shape)[rank - 2] ||
      (*shapes_static)[rank - 1] != (*shape)[rank - 1]) {
    return false;
  }
  if (!IsConstValue(offsets_arg->elements_[rank - 2], 0) ||
      !IsConstValue(offsets_arg->elements_[rank - 1], 0)) {
    return false;
  }
  // NZ blocking folds the leading axes into one batch coordinate. After any
  // leading axis spans multiple planes, the more minor axes must remain whole
  // for this window to denote a contiguous run of complete matrices.
  bool spans_planes = false;
  for (size_t axis = 0; axis + 2 < rank; ++axis) {
    const int64_t count = (*shapes_static)[axis];
    const int64_t extent = (*shape)[axis];
    if (count <= 0 || count > extent ||
        (spans_planes && (count != extent || !IsConstValue(offsets_arg->elements_[axis], 0)))) {
      return false;
    }
    if (auto offset = As<ConstInt>(offsets_arg->elements_[axis]);
        offset && (offset->value_ < 0 || offset->value_ > extent - count)) {
      return false;
    }
    spans_planes |= count > 1;
  }
  if (call->args_.size() >= 4) {
    if (auto valid_arg = As<MakeTuple>(call->args_[3]); valid_arg && !valid_arg->elements_.empty()) {
      if (valid_arg->elements_.size() != rank || StaticShape(valid_arg->elements_) != shapes_static) {
        return false;
      }
    }
  }
  if (call->args_.size() >= 5) {
    auto drop_arg = As<MakeTuple>(call->args_[4]);
    if (!drop_arg) {
      return false;
    }
    for (const auto& d_expr : drop_arg->elements_) {
      auto ci = As<ConstInt>(d_expr);
      if (!ci || ci->value_ < 0 || ci->value_ + 2 >= static_cast<int64_t>(rank) ||
          (*shapes_static)[ci->value_] != 1) {
        return false;
      }
    }
  }
  return true;
}

/// Owns scalar fact maps and builds `tensor_view_semantics::NzOffsetFacts` cleanly.
struct NzFactContext {
  std::unordered_map<const Var*, ExprPtr> scalar_defs;
  std::unordered_map<const Var*, std::pair<ExprPtr, ExprPtr>> loop_ranges;
  std::unordered_set<const Var*> non_neg_vars;
  VarPtr non_neg_placeholder;

  NzFactContext()
      : non_neg_placeholder(std::make_shared<Var>(
            "_nz_nonneg_param", std::make_shared<ScalarType>(DataType::INDEX), Span::unknown())) {
    non_neg_vars.insert(non_neg_placeholder.get());
  }

  void SeedParamStep(const VarPtr& param, int64_t step) {
    if (step > 1 && (step % 8) == 0) {
      non_neg_vars.insert(param.get());
      scalar_defs[param.get()] =
          MakeMul(non_neg_placeholder, std::make_shared<ConstInt>(step, DataType::INDEX, Span::unknown()),
                  Span::unknown());
    }
  }

  [[nodiscard]] tensor_view_semantics::NzOffsetFacts Facts() const {
    tensor_view_semantics::NzOffsetFacts facts;
    facts.definition = [this](const VarPtr& v) -> ExprPtr {
      auto it = scalar_defs.find(v.get());
      return it == scalar_defs.end() ? nullptr : it->second;
    };
    facts.loop_range = [this](const VarPtr& v) -> std::pair<ExprPtr, ExprPtr> {
      auto it = loop_ranges.find(v.get());
      return it == loop_ranges.end() ? std::pair<ExprPtr, ExprPtr>{nullptr, nullptr} : it->second;
    };
    facts.is_non_negative = [this](const VarPtr& v) { return non_neg_vars.count(v.get()) != 0; };
    return facts;
  }
};

bool CheckNzDivisibleAndNonNeg(const ExprPtr& expr, int64_t divisor,
                               const tensor_view_semantics::NzOffsetFacts& facts) {
  if (auto ci = As<ConstInt>(expr)) {
    return ci->value_ >= 0 && (ci->value_ % divisor) == 0;
  }
  int mul_budget = tensor_view_semantics::kNzDivideStepBudget;
  int sign_budget = tensor_view_semantics::kNzDivideStepBudget;
  return tensor_view_semantics::IsProvableMultipleOf(expr, divisor, facts, &mul_budget) &&
         tensor_view_semantics::IsProvableNonNegative(expr, facts, &sign_budget);
}

bool IsNzCompatibleValidShape(const ExprPtr& valid_arg, size_t rank, int c0,
                              const tensor_view_semantics::NzOffsetFacts& facts) {
  auto vt = As<MakeTuple>(valid_arg);
  if (!vt || vt->elements_.size() != rank) {
    return false;
  }
  for (size_t i = 0; i + 2 < rank; ++i) {
    auto ci = As<ConstInt>(vt->elements_[i]);
    if (!ci || ci->value_ <= 0) {
      return false;
    }
  }
  auto v_cols = As<ConstInt>(vt->elements_[rank - 1]);
  if (!v_cols || v_cols->value_ <= 0 || (v_cols->value_ % c0) != 0) {
    return false;
  }
  const ExprPtr& v_rows = vt->elements_[rank - 2];
  if (auto ci = As<ConstInt>(v_rows)) {
    return ci->value_ > 0 && (ci->value_ % 16) == 0;
  }
  int budget = tensor_view_semantics::kNzDivideStepBudget;
  return tensor_view_semantics::IsProvableMultipleOf(v_rows, 16, facts, &budget);
}

bool IsNzCompatibleTileLoadCall(const CallPtr& call, const TypePtr& tensor_type,
                                const tensor_view_semantics::NzOffsetFacts& facts,
                                const LayoutCostModel& model) {
  auto shape = IsNzCompatibleTensorType(tensor_type);
  auto tt = As<TensorType>(tensor_type);
  if (!shape.has_value() || !tt || call->args_.size() < 3) {
    return false;
  }
  const MemorySpace target_mem = GetMemorySpaceKwarg(call->kwargs_, "target_memory", MemorySpace::Vec);
  if (target_mem != MemorySpace::Mat && target_mem != MemorySpace::Vec) {
    return false;
  }
  const size_t rank = shape->size();
  auto offsets_arg = As<MakeTuple>(call->args_[1]);
  auto shapes_arg = As<MakeTuple>(call->args_[2]);
  if (!offsets_arg || !shapes_arg || offsets_arg->elements_.size() != rank ||
      shapes_arg->elements_.size() != rank) {
    return false;
  }
  auto shapes_static = StaticShape(shapes_arg->elements_);
  auto c0_opt = TryC0Elements(tt->dtype_);
  if (!shapes_static.has_value() || !c0_opt.has_value() || *c0_opt <= 0 ||
      !IsNzCompatibleShape(*shapes_static, tt->dtype_)) {
    return false;
  }
  const int c0 = *c0_opt;
  if (target_mem == MemorySpace::Vec) {
    if (rank == 3 && (*shapes_static)[0] != 1) {
      return false;
    }
    for (size_t i = 0; i + 2 < rank; ++i) {
      int budget = tensor_view_semantics::kNzDivideStepBudget;
      if (!tensor_view_semantics::IsProvableNonNegative(offsets_arg->elements_[i], facts, &budget)) {
        return false;
      }
    }
  }
  if (!CheckNzDivisibleAndNonNeg(offsets_arg->elements_[rank - 2], 16, facts) ||
      !CheckNzDivisibleAndNonNeg(offsets_arg->elements_[rank - 1], c0, facts)) {
    return false;
  }
  MakeTuplePtr part_tuple = shapes_arg;
  if (call->args_.size() >= 4) {
    if (target_mem == MemorySpace::Vec) {
      auto valid_arg = As<MakeTuple>(call->args_[3]);
      if (!valid_arg || StaticShape(valid_arg->elements_) != shapes_static) {
        return false;
      }
      part_tuple = valid_arg;
    } else {
      if (!IsNzCompatibleValidShape(call->args_[3], rank, c0, facts)) {
        return false;
      }
      part_tuple = As<MakeTuple>(call->args_[3]);
    }
  }
  const int64_t col_blocks = As<ConstInt>(part_tuple->elements_[rank - 1])
                                 ? (As<ConstInt>(part_tuple->elements_[rank - 1])->value_ / c0)
                                 : 2;
  const int64_t loaded_rows = As<ConstInt>(part_tuple->elements_[rank - 2])
                                  ? As<ConstInt>(part_tuple->elements_[rank - 2])->value_
                                  : 0;
  return (target_mem == MemorySpace::Mat && model.SupportsWideNzMatLoad()) ||
         !(col_blocks > 1 && ((*shape)[rank - 2] - loaded_rows) > 65535);
}

bool IsNzCompatibleVecTileStoreCall(const CallPtr& call, const TypePtr& tensor_type,
                                    const tensor_view_semantics::NzOffsetFacts& facts) {
  auto shape = IsNzCompatibleTensorType(tensor_type);
  auto tt = As<TensorType>(tensor_type);
  if (!shape.has_value() || !tt || (call->args_.size() != 3 && call->args_.size() != 4)) {
    return false;
  }
  auto tile_type = As<TileType>(call->args_[0]->GetType());
  if (!tile_type || tile_type->GetMemorySpace() != MemorySpace::Vec || tile_type->dtype_ != tt->dtype_) {
    return false;
  }
  auto tile_shape = StaticShape(tile_type->shape_);
  if (!tile_shape.has_value() || tile_shape->size() != 2) {
    return false;
  }
  if (tile_type->tile_view_.has_value() && !tile_type->tile_view_->valid_shape.empty() &&
      StaticShape(tile_type->tile_view_->valid_shape) != tile_shape) {
    return false;
  }
  auto c0_opt = TryC0Elements(tt->dtype_);
  if (!c0_opt.has_value()) {
    return false;
  }
  const int c0 = *c0_opt;
  if ((*tile_shape)[0] <= 0 || ((*tile_shape)[0] % 16) != 0 || (*tile_shape)[1] <= 0 ||
      ((*tile_shape)[1] % c0) != 0) {
    return false;
  }
  if (GetIntKwarg(call->kwargs_, "atomic", 0) != 0 || call->HasKwarg("pre_quant") ||
      call->GetKwarg<bool>("pre_relu", false)) {
    return false;
  }
  const size_t rank = shape->size();
  auto offsets_arg = As<MakeTuple>(call->args_[1]);
  if (!offsets_arg || offsets_arg->elements_.size() != rank) {
    return false;
  }
  if (call->args_.size() == 4) {
    auto win_arg = As<MakeTuple>(call->args_[3]);
    if (!win_arg) {
      return false;
    }
    auto win_shape = StaticShape(win_arg->elements_);
    std::vector<int64_t> expected_win(rank - 2, 1);
    expected_win.push_back((*tile_shape)[0]);
    expected_win.push_back((*tile_shape)[1]);
    if (!win_shape.has_value() || *win_shape != expected_win) {
      return false;
    }
  }
  for (size_t i = 0; i + 2 < rank; ++i) {
    int budget = tensor_view_semantics::kNzDivideStepBudget;
    if (!tensor_view_semantics::IsProvableNonNegative(offsets_arg->elements_[i], facts, &budget)) {
      return false;
    }
  }
  return CheckNzDivisibleAndNonNeg(offsets_arg->elements_[rank - 2], 16, facts) &&
         CheckNzDivisibleAndNonNeg(offsets_arg->elements_[rank - 1], c0, facts);
}

bool IsNzCompatibleTileStoreCall(const CallPtr& call, const TypePtr& tensor_type,
                                 const tensor_view_semantics::NzOffsetFacts& facts,
                                 const LayoutCostModel& model) {
  auto tile_type = !call->args_.empty() ? As<TileType>(call->args_[0]->GetType()) : nullptr;
  if (tile_type && tile_type->GetMemorySpace() == MemorySpace::Vec) {
    return IsNzCompatibleVecTileStoreCall(call, tensor_type, facts);
  }
  auto shape = IsNzCompatibleTensorType(tensor_type);
  auto tt = As<TensorType>(tensor_type);
  if (!shape.has_value() || !tt || (call->args_.size() != 3 && call->args_.size() != 4)) {
    return false;
  }
  if (model.TargetArch() == "a5") {
    return false;
  }
  if (!tile_type || tile_type->dtype_ != DataType::FP32 || tt->dtype_ != DataType::FP32 ||
      tile_type->GetMemorySpace() != MemorySpace::Acc ||
      EffectiveTileLayout(tile_type) != LayoutSpec::FractalAccNZ()) {
    return false;
  }
  auto tile_shape = StaticShape(tile_type->shape_);
  if (!tile_shape.has_value() || tile_shape->size() != 2) {
    return false;
  }
  if (tile_type->tile_view_.has_value() && !tile_type->tile_view_->valid_shape.empty() &&
      StaticShape(tile_type->tile_view_->valid_shape) != tile_shape) {
    return false;
  }
  const size_t rank = shape->size();
  if ((*tile_shape)[0] <= 0 || ((*tile_shape)[0] % 16) != 0 || (*tile_shape)[0] > 65535 ||
      (*tile_shape)[1] <= 0 || ((*tile_shape)[1] % 16) != 0 || (*tile_shape)[0] != (*shape)[rank - 2]) {
    return false;
  }
  if (GetIntKwarg(call->kwargs_, "atomic", 0) != 0 || call->HasKwarg("pre_quant") ||
      call->GetKwarg<bool>("pre_relu", false)) {
    return false;
  }
  auto offsets_arg = As<MakeTuple>(call->args_[1]);
  if (!offsets_arg || offsets_arg->elements_.size() != rank) {
    return false;
  }
  if (call->args_.size() == 4) {
    auto win_arg = As<MakeTuple>(call->args_[3]);
    if (!win_arg) {
      return false;
    }
    auto win_shape = StaticShape(win_arg->elements_);
    std::vector<int64_t> expected_win(rank - 2, 1);
    expected_win.push_back((*tile_shape)[0]);
    expected_win.push_back((*tile_shape)[1]);
    if (!win_shape.has_value() || *win_shape != expected_win) {
      return false;
    }
  }
  for (size_t i = 0; i + 2 < rank; ++i) {
    int budget = tensor_view_semantics::kNzDivideStepBudget;
    if (!tensor_view_semantics::IsProvableNonNegative(offsets_arg->elements_[i], facts, &budget)) {
      return false;
    }
  }
  if (!IsConstValue(offsets_arg->elements_[rank - 2], 0)) {
    return false;
  }
  return CheckNzDivisibleAndNonNeg(offsets_arg->elements_[rank - 1], 16, facts);
}

bool FuncContainsTileExpands(const StmtPtr& stmt) {
  if (!stmt) {
    return false;
  }
  auto check_expr = [](auto& self, const ExprPtr& expr) -> bool {
    if (!expr) return false;
    if (auto call = As<Call>(expr)) {
      if (IsOp(call, "tile.expands")) return true;
      for (const auto& arg : call->args_) {
        if (self(self, arg)) return true;
      }
    } else if (auto sub = As<Submit>(expr)) {
      for (const auto& arg : sub->args_) {
        if (self(self, arg)) return true;
      }
    }
    return false;
  };
  if (auto as = As<AssignStmt>(stmt)) {
    if (check_expr(check_expr, as->value_)) return true;
  } else if (auto es = As<EvalStmt>(stmt)) {
    if (check_expr(check_expr, es->expr_)) return true;
  } else if (auto rs = As<ReturnStmt>(stmt)) {
    for (const auto& val : rs->value_) {
      if (check_expr(check_expr, val)) return true;
    }
  }
  for (const auto& child : IterChildStmts(stmt)) {
    if (FuncContainsTileExpands(child)) return true;
  }
  return false;
}

using GmNames = OrchCallAnalysis::GmNames;

bool HasFixedGmSignature(const FunctionPtr& func) {
  return func->HasAttr(kAttrExternalSource) || func->requires_runtime_binding_;
}

// Assign solver names to IR definitions in program order. Keep the first unqualified
// spelling for existing name-based cost options, then disambiguate unrelated storage.
GmNames MakeGmDefinitionNames(const ProgramPtr& program) {
  std::vector<std::pair<std::string, VarPtr>> definitions;
  auto collect = [&](const FunctionPtr& func, const VarPtr& var) {
    if (AsTensorTypeLike(var->GetType())) {
      definitions.emplace_back(func->name_, var);
    }
  };
  auto walk = [&](auto& self, const FunctionPtr& func, const StmtPtr& stmt) -> void {
    if (auto assign = As<AssignStmt>(stmt)) {
      collect(func, assign->var_);
    } else if (auto loop = As<ForStmt>(stmt)) {
      for (const auto& iter : loop->iter_args_) collect(func, iter);
      for (const auto& result : loop->return_vars_) collect(func, result);
    } else if (auto loop = As<WhileStmt>(stmt)) {
      for (const auto& iter : loop->iter_args_) collect(func, iter);
      for (const auto& result : loop->return_vars_) collect(func, result);
    } else if (auto branch = As<IfStmt>(stmt)) {
      for (const auto& result : branch->return_vars_) collect(func, result);
    }
    for (const auto& child : IterChildStmts(stmt)) self(self, func, child);
  };
  // Preserve the user-facing spellings of orchestration interface and allocation
  // names ahead of internal kernel parameters with common hints like "out".
  for (const auto& [_, func] : program->functions_) {
    if (func->func_type_ == FunctionType::Orchestration) {
      for (const auto& param : func->params_) collect(func, param);
      walk(walk, func, func->body_);
    }
  }
  for (const auto& [_, func] : program->functions_) {
    if (func->func_type_ != FunctionType::Orchestration) {
      for (const auto& param : func->params_) collect(func, param);
      walk(walk, func, func->body_);
    }
  }
  std::unordered_set<std::string> reserved;
  for (const auto& [_, var] : definitions) reserved.insert(CleanName(var->name_hint_));
  std::unordered_set<std::string> used;
  std::unordered_map<std::string, size_t> next_suffix;
  GmNames names;
  for (const auto& [func_name, var] : definitions) {
    if (names.count(var.get()) != 0) continue;
    const std::string base = CleanName(var->name_hint_);
    std::string name = base;
    if (used.count(name) != 0) {
      const std::string qualified = func_name + "." + base;
      name = qualified;
      auto [suffix, _] = next_suffix.try_emplace(qualified, 1);
      while (reserved.count(name) != 0 || used.count(name) != 0) {
        name = qualified + "." + std::to_string(suffix->second++);
      }
    }
    used.insert(name);
    names.emplace(var.get(), std::move(name));
  }
  return names;
}

class GmBindings {
 public:
  GmBindings(const ProgramPtr& program, const FunctionPtr& func, const GmNames& definitions,
             const std::vector<std::optional<std::string>>& overrides, const std::string& site = {})
      : program_(program),
        definitions_(definitions),
        scope_(func->name_ + site),
        is_specialized_(!site.empty()) {
    for (size_t i = 0; i < func->params_.size(); ++i) {
      const auto& param = func->params_[i];
      if (!AsTensorTypeLike(param->GetType())) continue;
      const auto& override_name = i < overrides.size() ? overrides[i] : std::nullopt;
      names_[param.get()] = override_name.value_or(DefinitionName(param.get()));
    }
    Walk(func->body_);
    for (auto& [_, name] : names_) name = Canonical(name);
  }

  const GmNames& Names() const { return names_; }

  std::unordered_set<std::string> Roots(const ExprPtr& expr) {
    std::unordered_set<std::string> roots;
    CollectTensorRoots(expr, expr->GetType(), roots);
    return roots;
  }

  std::optional<std::string> Resolve(const ExprPtr& expr) {
    auto name = Resolve(expr, {});
    return name ? std::optional<std::string>(Canonical(*name)) : std::nullopt;
  }

  void RecordReturnedStorage(const StmtPtr& stmt, std::unordered_set<std::string>& outputs) {
    if (!stmt) return;
    if (auto ret = As<ReturnStmt>(stmt)) {
      for (const auto& value : ret->value_) CollectTensorRoots(value, value->GetType(), outputs);
    }
    for (const auto& child : IterChildStmts(stmt)) RecordReturnedStorage(child, outputs);
  }

  void PinLayouts(OrchCallAnalysis& analysis, bool track_named_inputs = false) const {
    for (const auto& name : pinned_) {
      analysis.nz_ineligible_names.insert(Canonical(name));
      analysis.dn_ineligible_names.insert(Canonical(name));
      if (track_named_inputs) {
        analysis.nz_restrictions.try_emplace(Canonical(name),
                                             "control-flow alternatives bind different storage");
      }
    }
  }

  std::string OmittedArgument(const ExprPtr& call, size_t param) {
    auto [it, _] = call_ids_.try_emplace(call.get(), call_ids_.size());
    return "$" + scope_ + ".call." + std::to_string(it->second) + ".out." + std::to_string(param);
  }

 private:
  std::string DefinitionName(const Var* var) const {
    const auto& name = definitions_.at(var);
    return is_specialized_ ? "$" + scope_ + ".gm." + name : name;
  }

  std::string Canonical(std::string name) const {
    auto it = forwarded_.find(name);
    while (it != forwarded_.end()) {
      name = it->second;
      it = forwarded_.find(name);
    }
    return name;
  }

  std::optional<std::string> ResolveCall(const ExprPtr& expr, const OpPtr& op,
                                         const std::vector<ExprPtr>& args, std::vector<size_t> path) {
    auto global = As<GlobalVar>(op);
    if (!global) {
      auto call = As<Call>(expr);
      if (!call || !path.empty() || IsOp(call, "tensor.reshape") || IsOp(call, "tensor.transpose")) {
        return std::nullopt;
      }
      auto index = ResultAliasedArgIndex(call);
      if (!index && (IsOp(call, "tensor.slice") || IsOp(call, "tensor.view"))) index = 0;
      if (index && *index < args.size()) {
        // A layout-changing view has its own descriptor. Its operand is pinned
        // by the consumer eligibility checks.
        auto src = AsTensorTypeLike(args[*index]->GetType());
        auto dst = AsTensorTypeLike(call->GetType());
        if (src && dst && detail::TensorTypeLayout(src) == detail::TensorTypeLayout(dst)) {
          return Resolve(args[*index], {});
        }
      }
      return std::nullopt;
    }

    const auto callee = program_->GetFunction(global->name_);
    if (!callee) return std::nullopt;
    auto returned = return_lineage::ReturnedParamIndices(callee, program_);
    size_t slot = 0;
    if (callee->return_types_.size() == 1 && As<TupleType>(callee->return_types_.front())) {
      if (As<Submit>(expr)) {
        if (path.empty() || path.back() != 0) return std::nullopt;
        path.pop_back();
      }
      if (path.size() != 1) return std::nullopt;
      returned = return_lineage::ReturnedTupleElementParamIndices(callee, program_);
      slot = path.back();
    } else if (!path.empty()) {
      if (path.size() != 1) return std::nullopt;
      slot = path.back();
    }
    if (slot >= returned.size()) return std::nullopt;
    const auto param_index = returned[slot];
    if (!param_index.has_value()) return std::nullopt;
    const size_t param = *param_index;
    return param < args.size() ? Resolve(args[param], {})
                               : std::optional<std::string>(OmittedArgument(expr, param));
  }

  std::optional<std::string> Resolve(const ExprPtr& expr, std::vector<size_t> path) {
    if (auto var = AsVarLike(expr)) {
      if (path.empty()) {
        auto it = names_.find(var.get());
        if (it != names_.end()) return Canonical(it->second);
      } else if (auto it = tuples_.find(var.get()); it != tuples_.end()) {
        return Resolve(it->second, std::move(path));
      }
    } else if (auto item = As<TupleGetItemExpr>(expr); item && item->index_ >= 0) {
      path.push_back(static_cast<size_t>(item->index_));
      return Resolve(item->tuple_, std::move(path));
    } else if (auto tuple = As<MakeTuple>(expr); tuple && !path.empty()) {
      size_t index = path.back();
      path.pop_back();
      if (index < tuple->elements_.size()) return Resolve(tuple->elements_[index], std::move(path));
    } else if (auto call = As<Call>(expr)) {
      return ResolveCall(call, call->op_, call->args_, std::move(path));
    } else if (auto submit = As<Submit>(expr)) {
      return ResolveCall(submit, submit->op_, submit->args_, std::move(path));
    }
    return std::nullopt;
  }

  using Yields = std::vector<std::vector<ExprPtr>>;

  Yields WalkBody(const StmtPtr& body) {
    yield_frames_.emplace_back();
    Walk(body);
    auto yields = std::move(yield_frames_.back());
    yield_frames_.pop_back();
    return yields;
  }

  // Tuple phis may contain any number of independent tensor descriptors.
  // Keep those alternatives at the declared layout when we cannot forward the
  // tuple as one value. Later projections get their own pinned identities.
  void CollectTensorRoots(const ExprPtr& expr, const TypePtr& type, std::unordered_set<std::string>& roots,
                          std::vector<size_t> path = {}) {
    if (auto tuple = As<TupleType>(type)) {
      for (size_t i = 0; i < tuple->types_.size(); ++i) {
        auto element = path;
        element.insert(element.begin(), i);
        CollectTensorRoots(expr, tuple->types_[i], roots, std::move(element));
      }
    } else if (AsTensorTypeLike(type)) {
      if (auto key = Resolve(expr, std::move(path))) roots.insert(Canonical(*key));
    }
  }

  // The synthetic name of a phi describes a dynamic selection. Only unanimous
  // alternatives may forward to a storage identity, even when name hints match.
  void BindPhi(const VarPtr& result, const Yields& yields, size_t index, const ExprPtr& initial = nullptr,
               const std::optional<std::string>& self = std::nullopt) {
    if (!AsTensorTypeLike(result->GetType())) {
      if (As<TupleType>(result->GetType())) {
        if (initial) CollectTensorRoots(initial, initial->GetType(), pinned_);
        for (const auto& values : yields) {
          if (index < values.size()) CollectTensorRoots(values[index], values[index]->GetType(), pinned_);
        }
      }
      return;
    }
    std::set<std::string> alternatives;
    bool unknown = yields.empty();
    auto add_alternative = [&](const ExprPtr& expr) {
      auto name = Resolve(expr);
      if (!name) {
        unknown = true;
      } else if (name != self) {
        alternatives.insert(*name);
      }
    };
    if (initial) add_alternative(initial);
    for (const auto& values : yields) {
      add_alternative(index < values.size() ? values[index] : nullptr);
    }
    const auto key = DefinitionName(result.get());
    if (!unknown && alternatives.size() == 1) {
      names_[result.get()] = *alternatives.begin();
      if (self) forwarded_.emplace(*self, *alternatives.begin());
      return;
    }
    names_[result.get()] = key;
    pinned_.insert(key);
    pinned_.insert(alternatives.begin(), alternatives.end());
    if (self) pinned_.insert(*self);
  }

  template <typename Loop>
  void WalkLoop(const Loop& loop) {
    for (const auto& iter : loop->iter_args_) {
      if (AsTensorTypeLike(iter->GetType())) names_[iter.get()] = DefinitionName(iter.get());
    }
    auto yields = WalkBody(loop->body_);
    for (size_t i = 0; i < loop->return_vars_.size() && i < loop->iter_args_.size(); ++i) {
      const auto& iter = loop->iter_args_[i];
      std::optional<std::string> self;
      if (AsTensorTypeLike(iter->GetType())) self = DefinitionName(iter.get());
      BindPhi(loop->return_vars_[i], yields, i, iter->initValue_, self);
    }
  }

  void Walk(const StmtPtr& stmt) {
    if (auto assign = As<AssignStmt>(stmt)) {
      const auto& var = assign->var_;
      if (AsTensorTypeLike(var->GetType())) {
        auto name = Resolve(assign->value_);
        if (name) {
          names_[var.get()] = *name;
        } else {
          auto key = DefinitionName(var.get());
          names_[var.get()] = key;
          auto call = As<Call>(assign->value_);
          if (!call || !IsOp(call, "tensor.create")) pinned_.insert(std::move(key));
        }
      } else if (As<TupleType>(var->GetType())) {
        tuples_[var.get()] = assign->value_;
      }
    } else if (auto loop = As<ForStmt>(stmt)) {
      WalkLoop(loop);
      return;
    } else if (auto loop = As<WhileStmt>(stmt)) {
      WalkLoop(loop);
      return;
    } else if (auto branch = As<IfStmt>(stmt)) {
      auto yields = WalkBody(branch->then_body_);
      if (branch->else_body_) {
        auto otherwise = WalkBody(*branch->else_body_);
        yields.insert(yields.end(), otherwise.begin(), otherwise.end());
      }
      for (size_t i = 0; i < branch->return_vars_.size(); ++i) BindPhi(branch->return_vars_[i], yields, i);
      return;
    } else if (auto yield = As<YieldStmt>(stmt); yield && !yield_frames_.empty()) {
      yield_frames_.back().push_back(yield->value_);
    }
    for (const auto& child : IterChildStmts(stmt)) Walk(child);
  }

  const ProgramPtr& program_;
  const GmNames& definitions_;
  std::string scope_;
  bool is_specialized_;
  GmNames names_;
  std::unordered_map<const Var*, ExprPtr> tuples_;
  std::unordered_map<std::string, std::string> forwarded_;
  std::unordered_set<std::string> pinned_;
  std::unordered_map<const Expr*, size_t> call_ids_;
  std::vector<Yields> yield_frames_;
};

std::unordered_set<std::string> CollectFuncExpandsGmNames(const FunctionPtr& func, const GmNames& gm_names) {
  if (!FuncContainsTileExpands(func->body_)) {
    return {};
  }
  std::unordered_set<std::string> names;
  for (const auto& param : func->params_) {
    if (AsTensorTypeLike(param->GetType())) {
      names.insert(gm_names.at(param.get()));
    }
  }
  auto walk = [&](auto& self, const StmtPtr& stmt) -> void {
    if (auto as = As<AssignStmt>(stmt)) {
      if (AsTensorTypeLike(as->var_->GetType())) {
        if (auto call = As<Call>(as->value_); call && IsOp(call, "tensor.create")) {
          names.insert(gm_names.at(as->var_.get()));
        }
      }
    }
    for (const auto& child : IterChildStmts(stmt)) {
      self(self, child);
    }
  };
  walk(walk, func->body_);
  return names;
}

struct FuncGmIneligible {
  std::unordered_set<std::string> nz;
  std::unordered_set<std::string> dn;
  std::unordered_set<std::string> written;
  std::unordered_map<std::string, std::string> nz_restrictions;
};

FuncGmIneligible CollectFuncGmIneligible(const FunctionPtr& func, const LayoutCostModel& model,
                                         const GmNames& gm_names,
                                         const std::map<std::pair<std::string, size_t>, int64_t>& param_steps,
                                         bool track_named_inputs) {
  FuncGmIneligible out;
  NzFactContext fact_ctx;

  for (size_t idx = 0; idx < func->params_.size(); ++idx) {
    const auto& param = func->params_[idx];
    if (auto tt = AsTensorTypeLike(param->GetType())) {
      const auto& name = gm_names.at(param.get());
      if (HasFixedGmSignature(func) || IsA<DistributedTensorType>(param->GetType())) {
        out.nz.insert(name);
        out.dn.insert(name);
      } else {
        if (!IsNzCompatibleTensorType(param->GetType()).has_value()) {
          out.nz.insert(name);
        }
        auto st_shape = StaticShape(tt->shape_);
        if (!st_shape.has_value() || st_shape->size() != 2) {
          out.dn.insert(name);
        }
      }
    } else if (As<ScalarType>(param->GetType())) {
      auto it = param_steps.find({func->name_, idx});
      if (it != param_steps.end()) {
        fact_ctx.SeedParamStep(param, it->second);
      }
    }
  }

  auto collect_facts = [&](auto& self, const StmtPtr& stmt) -> void {
    if (auto as = As<AssignStmt>(stmt)) {
      if (As<ScalarType>(as->var_->GetType())) {
        fact_ctx.scalar_defs[as->var_.get()] = as->value_;
        if (auto call = As<Call>(as->value_);
            call && (IsOp(call, "tile.get_block_idx") || IsOp(call, "tile.get_block_num"))) {
          fact_ctx.non_neg_vars.insert(as->var_.get());
        }
      } else if (auto call = As<Call>(as->value_); call && AsTensorTypeLike(as->var_->GetType())) {
        const auto& name = gm_names.at(as->var_.get());
        if (IsOp(call, "tensor.reshape") || IsOp(call, "tensor.transpose")) {
          out.nz.insert(name);
          out.dn.insert(name);
        } else if (IsOp(call, "tensor.create") &&
                   !IsNzCompatibleTensorType(as->var_->GetType()).has_value()) {
          out.nz.insert(name);
        }
      }
    } else if (auto fs = As<ForStmt>(stmt)) {
      fact_ctx.loop_ranges[fs->loop_var_.get()] = {fs->start_, fs->step_};
    }
    for (const auto& child : IterChildStmts(stmt)) self(self, child);
  };
  collect_facts(collect_facts, func->body_);

  const auto facts = fact_ctx.Facts();
  auto check_expr = [&](auto& self, const ExprPtr& expr) -> void {
    if (!expr) return;
    if (auto sub = As<Submit>(expr)) {
      for (const auto& arg : sub->args_) {
        self(self, arg);
        if (track_named_inputs) {
          const auto var = AsVarLike(arg);
          if (auto root = gm_names.find(var.get()); root != gm_names.end()) {
            out.nz_restrictions.try_emplace(root->second, "unmodeled submit to '" + sub->op_->name_ + "'");
          }
        }
      }
      return;
    }
    auto call = As<Call>(expr);
    if (!call) return;
    for (const auto& arg : call->args_) {
      self(self, arg);
    }
    if (As<GlobalVar>(call->op_)) {
      if (track_named_inputs) {
        for (const auto& arg : call->args_) {
          const auto var = AsVarLike(arg);
          if (auto root = gm_names.find(var.get()); root != gm_names.end()) {
            out.nz_restrictions.try_emplace(root->second,
                                            "unmodeled in-core call to '" + call->op_->name_ + "'");
          }
        }
      }
      return;
    }
    for (size_t i = 0; i < call->args_.size(); ++i) {
      auto arg_v = AsVarLike(call->args_[i]);
      if (!arg_v || !AsTensorTypeLike(arg_v->GetType())) {
        continue;
      }
      auto name = gm_names.find(arg_v.get());
      if (name == gm_names.end()) continue;
      bool ok_nz = false;
      bool ok_dn = false;
      if (IsOp(call, "tensor.create")) {
        ok_nz = IsNzCompatibleTensorType(call->GetType()).has_value();
        ok_dn = true;
      } else if (IsOp(call, "tensor.reshape") && i == 0) {
        ok_nz = IsWholeTensorFlatten(call);
        if (track_named_inputs) {
          out.nz_restrictions.try_emplace(name->second, "tensor.reshape changes the matrix coordinates");
        }
      } else if (IsOp(call, "tensor.slice") && i == 0) {
        ok_nz = IsNzCompatibleTensorSliceCall(call, arg_v->GetType());
      } else if (IsOp(call, "tile.load") && i == 0) {
        ok_nz = IsNzCompatibleTileLoadCall(call, arg_v->GetType(), facts, model);
        const MemorySpace target_mem = GetMemorySpaceKwarg(call->kwargs_, "target_memory", MemorySpace::Vec);
        ok_dn = (target_mem == MemorySpace::Vec);
      } else if (IsOp(call, "tile.store") && i == 2) {
        out.written.insert(name->second);
        ok_nz = IsNzCompatibleTileStoreCall(call, arg_v->GetType(), facts, model);
        auto src_tile = !call->args_.empty() ? As<TileType>(call->args_[0]->GetType()) : nullptr;
        ok_dn = src_tile && src_tile->GetMemorySpace().value_or(MemorySpace::Vec) == MemorySpace::Vec;
      }
      if (!ok_nz) {
        out.nz.insert(name->second);
        if (track_named_inputs) {
          out.nz_restrictions.try_emplace(name->second,
                                          "incompatible '" + call->op_->name_ + "' in '" + func->name_ + "'");
        }
      }
      if (!ok_dn) {
        out.dn.insert(name->second);
      }
    }
  };

  auto check_stmt = [&](auto& self, const StmtPtr& stmt) -> void {
    if (auto as = As<AssignStmt>(stmt)) {
      check_expr(check_expr, as->value_);
    } else if (auto es = As<EvalStmt>(stmt)) {
      check_expr(check_expr, es->expr_);
    } else if (auto rs = As<ReturnStmt>(stmt)) {
      for (const auto& val : rs->value_) {
        check_expr(check_expr, val);
      }
    }
    for (const auto& child : IterChildStmts(stmt)) {
      self(self, child);
    }
  };
  check_stmt(check_stmt, func->body_);
  return out;
}

class OrchAnalyzer {
 public:
  OrchAnalyzer(const ProgramPtr& program, const GmNames& definitions, bool track_named_inputs = false,
               std::optional<int64_t> host_world_size = std::nullopt)
      : program_(program),
        definitions_(definitions),
        track_named_inputs_(track_named_inputs),
        host_world_size_(host_world_size) {}

  void Walk(const FunctionPtr& func) {
    in_host_func_ = func->level_.has_value() && LevelToLinquLevel(*func->level_) >= 3;
    host_whole_matrix_vars_.clear();
    host_rank_stack_params_.clear();
    host_shard_offsets_.clear();
    const auto caller = analysis_.call_arg_names.find(func->name_);
    const std::vector<std::optional<std::string>> empty;
    const auto& overrides = caller != analysis_.call_arg_names.end() ? caller->second : empty;
    GmBindings bindings(program_, func, definitions_, overrides);
    if (caller == analysis_.call_arg_names.end()) {
      bindings.RecordReturnedStorage(func->body_, analysis_.external_outputs);
    }
    bindings_ = &bindings;
    bindings.PinLayouts(analysis_, track_named_inputs_);
    analysis_.gm_names.insert(bindings.Names().begin(), bindings.Names().end());
    for (size_t i = 0; i < func->params_.size(); ++i) {
      const auto& param = func->params_[i];
      auto name = bindings.Names().find(param.get());
      if (name != bindings.Names().end()) {
        if (in_host_func_) {
          host_whole_matrix_vars_.insert(param.get());
          if (caller == analysis_.call_arg_names.end()) {
            host_rank_stack_params_.insert(param.get());
            if (track_named_inputs_) {
              if (auto shape = IsNzCompatibleTensorType(param->GetType(), 4)) {
                host_input_shapes_.emplace(name->second, std::move(*shape));
              }
            }
          }
        }
        if (caller == analysis_.call_arg_names.end() && func->param_directions_[i] != ParamDirection::In) {
          analysis_.external_outputs.insert(name->second);
        }
        if (track_named_inputs_ && func->param_directions_[i] != ParamDirection::In) {
          analysis_.written_storage.insert(name->second);
        }
        if (HasFixedGmSignature(func)) Pin(name->second, "fixed parameter layout in '" + func->name_ + "'");
      }
      auto step_it = analysis_.param_steps.find({func->name_, i});
      if (step_it != analysis_.param_steps.end() && step_it->second > 1) {
        var_steps_[param.get()] = step_it->second;
      }
    }
    auto count = analysis_.call_counts.find(func->name_);
    mult_ = count != analysis_.call_counts.end() ? count->second : 1;
    WalkStmt(func->body_);
    bindings_ = nullptr;
  }

  OrchCallAnalysis Result() && {
    for (const auto& [name, visits] : fixed_dispatches_) {
      analysis_.projected_dispatches[name] += static_cast<double>(visits.size());
    }
    return std::move(analysis_);
  }

 private:
  void Pin(const std::string& name, const std::string& reason = {}) {
    analysis_.nz_ineligible_names.insert(name);
    analysis_.dn_ineligible_names.insert(name);
    if (track_named_inputs_ && !reason.empty()) analysis_.nz_restrictions.try_emplace(name, reason);
  }

  std::optional<int64_t> ArgAlignmentStep(const ExprPtr& arg) {
    if (auto v = AsVarLike(arg)) {
      auto it = var_steps_.find(v.get());
      if (it != var_steps_.end()) {
        return it->second;
      }
    }
    auto c_val = EvalScalarExpr(arg, scalar_defs_, std::nullopt);
    if (c_val.has_value() && *c_val >= 0) {
      if (*c_val == 0) return int64_t{0};
      if ((*c_val % 8) == 0) return c_val;
    }
    return std::nullopt;
  }

  int64_t CallCoreMultiplier(const ExprPtr& expr) {
    if (auto sub = As<Submit>(expr); sub && sub->core_num_.has_value()) {
      auto val = EvalScalarExpr(*sub->core_num_, scalar_defs_);
      return val.has_value() ? std::max<int64_t>(1, *val) : 1;
    }
    if (auto call = As<Call>(expr); call && call->HasAttr(kAttrCoreNum)) {
      auto core_expr = call->GetAttr<ExprPtr>(kAttrCoreNum, nullptr);
      if (core_expr) {
        auto val = EvalScalarExpr(core_expr, scalar_defs_);
        return val.has_value() ? std::max<int64_t>(1, *val) : 1;
      }
    }
    return 1;
  }

  void RecordHostProjection(const ExprPtr& expr, const VarPtr& arg, const TypePtr& formal_type,
                            const std::string& root, int64_t invocations) {
    const auto input = host_input_shapes_.find(root);
    const auto projection = IsNzCompatibleTensorType(arg->GetType());
    const bool is_matrix = input != host_input_shapes_.end() && input->second.size() == 2;
    const auto device_expr = As<Call>(expr) ? As<Call>(expr)->GetAttr<ExprPtr>(kAttrDevice, nullptr)
                                            : As<Submit>(expr)->GetAttr<ExprPtr>(kAttrDevice, nullptr);
    const auto device = EvalScalarExpr(device_expr, scalar_defs_, std::nullopt);
    if (is_matrix) {
      if (projection != input->second) {
        Pin(root, "device dispatch must forward the complete HOST matrix");
        return;
      }
      const auto rank = AsVarLike(ResolveScalarDefinition(device_expr));
      // With a single configured worker, an unpinned dispatch resolves to it
      // and a canonical range(world_size()) loop has only rank zero.
      if (host_world_size_ != 1 ||
          (device_expr && device != 0 && (!rank || !world_size_indices_.count(rank.get())))) {
        Pin(root, "whole-matrix HOST input requires a proven single-device dispatch");
        return;
      }
    } else {
      if (input == host_input_shapes_.end() || !host_shard_offsets_.count(arg.get()) || !projection ||
          projection->size() + 1 != input->second.size() ||
          !std::equal(projection->begin(), projection->end(), input->second.begin() + 1)) {
        analysis_.nz_restrictions.try_emplace(root, "device dispatch must select a complete rank shard");
        return;
      }
      if (host_world_size_ && *host_world_size_ != input->second.front()) {
        analysis_.nz_restrictions.try_emplace(root,
                                              "rank-indexed bank extent must match the HOST world size");
        return;
      }
    }
    const auto layout = detail::TensorTypeLayout(AsTensorTypeLike(formal_type));
    const auto [old, inserted] = host_dispatch_layouts_.try_emplace(root, layout);
    if (!inserted && old->second != layout) {
      analysis_.nz_restrictions.try_emplace(root,
                                            "device dispatches disagree on the declared storage layout");
    }
    if (is_matrix) {
      fixed_dispatches_[root].emplace(0, 0);
      if (!device_expr) analysis_.unpinned_host_matrix_dispatches[root].push_back(expr.get());
      return;
    }
    const auto& shard_expr = host_shard_offsets_.at(arg.get());
    if (auto index = AsVarLike(ResolveScalarDefinition(shard_expr));
        !host_world_size_ && index && world_size_indices_.count(index.get())) {
      analysis_.unknown_rank_counts.insert(root);
    }
    const auto fixed_shard = EvalScalarExpr(shard_expr, scalar_defs_, std::nullopt);
    if (fixed_shard && device) {
      fixed_dispatches_[root].emplace(*device, *fixed_shard);
    } else {
      // Without a fixed projection and device, every dispatch may fetch new
      // bytes. Rereads inside the CHIP can still receive the normal discount.
      analysis_.projected_dispatches[root] += static_cast<double>(invocations);
    }
  }

  void RecordGlobalCall(const ExprPtr& expr, const std::string& name, const std::vector<ExprPtr>& args) {
    const int64_t call_mult = mult_ * CallCoreMultiplier(expr);
    if (analysis_.call_order.count(name) == 0) {
      analysis_.call_order[name] = analysis_.call_order.size();
    }
    analysis_.call_counts[name] += call_mult;

    std::vector<std::optional<std::string>> resolved;
    resolved.reserve(args.size());
    for (const auto& arg : args) {
      resolved.push_back(bindings_->Resolve(arg));
    }
    const auto callee = program_->GetFunction(name);
    if (callee) {
      if (track_named_inputs_ && args.size() != callee->params_.size()) {
        // Before IO canonicalization, an omitted Out can shift the actuals.
        // A positional guess cannot authorize repacking any of them.
        for (const auto& arg : args) {
          for (const auto& root : bindings_->Roots(arg)) {
            Pin(root, "call to '" + name + "' omits a formal argument");
          }
        }
      }
      for (size_t i = 0; track_named_inputs_ && i < std::min(resolved.size(), callee->params_.size()); ++i) {
        if (!resolved[i]) continue;
        if (callee->param_directions_[i] != ParamDirection::In) {
          analysis_.written_storage.insert(*resolved[i]);
        }
        // Both ends of a call describe the same projected matrix. Equal byte
        // sizes alone do not establish identical NZ block boundaries.
        const auto actual = IsNzCompatibleTensorType(args[i]->GetType(), 4);
        const auto formal = IsNzCompatibleTensorType(callee->params_[i]->GetType(), 4);
        const auto actual_tensor = AsTensorTypeLike(args[i]->GetType());
        const auto formal_tensor = AsTensorTypeLike(callee->params_[i]->GetType());
        if (!actual || !formal || actual != formal || !HasDenseLogicalInputLayout(actual_tensor) ||
            !HasDenseLogicalInputLayout(formal_tensor) || actual_tensor->dtype_ != formal_tensor->dtype_) {
          analysis_.nz_restrictions.try_emplace(
              *resolved[i], "incompatible tensor parameter " + std::to_string(i) + " of '" + name + "'");
        }
      }
      for (size_t i = resolved.size(); i < callee->params_.size(); ++i) {
        auto omitted = bindings_->OmittedArgument(expr, i);
        Pin(omitted);
        resolved.emplace_back(std::move(omitted));
      }
    }
    if (!callee || HasFixedGmSignature(callee)) {
      for (const auto& arg : args) {
        if (track_named_inputs_) {
          for (const auto& root : bindings_->Roots(arg)) Pin(root, "opaque consumer '" + name + "'");
        } else if (const auto root = bindings_->Resolve(arg)) {
          Pin(*root);
        }
      }
    }
    const auto [first, inserted_call] = analysis_.call_arg_names.try_emplace(name, resolved);
    if (callee && !IsInCoreType(callee->func_type_) && !inserted_call) {
      // The writer emits a single body for orchestration helpers. If formals
      // bind different storage at different sites, require the declared ABI at
      // those positions before analyzing the shared body under its first binding.
      for (size_t i = 0; i < std::min(first->second.size(), resolved.size()); ++i) {
        const auto& first_binding = first->second[i];
        const auto& current_binding = resolved[i];
        if (first_binding != current_binding) {
          const auto reason = "shared parameter of '" + name + "' binds different storage across calls";
          if (first_binding.has_value()) Pin(*first_binding, reason);
          if (current_binding.has_value()) Pin(*current_binding, reason);
        }
      }
    }

    for (size_t p_idx = 0; p_idx < args.size(); ++p_idx) {
      const auto key = std::make_pair(name, p_idx);
      if (invalid_param_steps_.count(key) != 0) {
        continue;
      }
      auto step = ArgAlignmentStep(args[p_idx]);
      if (!step.has_value()) {
        invalid_param_steps_.insert(key);
        analysis_.param_steps.erase(key);
      } else {
        auto it = analysis_.param_steps.find(key);
        analysis_.param_steps[key] =
            (it == analysis_.param_steps.end()) ? *step : std::gcd(it->second, *step);
      }
    }

    if (in_host_func_ && callee && (!callee->level_.has_value() || LevelToLinquLevel(*callee->level_) < 3)) {
      const bool is_chip_dispatch = callee->level_ == Level::CHIP && callee->role_ == Role::Orchestrator;
      const bool has_device = [&]() {
        if (auto c = As<Call>(expr)) return c->HasAttr(kAttrDevice);
        if (auto s = As<Submit>(expr)) return s->HasAttr(kAttrDevice);
        return false;
      }();
      for (size_t i = 0; i < args.size() && i < resolved.size(); ++i) {
        const auto& name_opt = resolved[i];
        if (!name_opt.has_value()) continue;
        const auto input = host_input_shapes_.find(*name_opt);
        const bool is_matrix = input != host_input_shapes_.end() && input->second.size() == 2;
        const bool is_device_dispatch =
            track_named_inputs_ ? is_chip_dispatch && (has_device || is_matrix) : has_device;
        analysis_.dn_ineligible_names.insert(*name_opt);
        auto arg_v = AsVarLike(args[i]);
        const bool whole_host = arg_v && host_whole_matrix_vars_.count(arg_v.get()) != 0;
        if (!is_device_dispatch || !whole_host) {
          analysis_.nz_ineligible_names.insert(*name_opt);
          if (track_named_inputs_) {
            analysis_.nz_restrictions.try_emplace(
                *name_opt, "dispatch to '" + name + "' does not forward a whole HOST matrix");
          }
        } else if (track_named_inputs_ && i < callee->params_.size()) {
          RecordHostProjection(expr, arg_v, callee->params_[i]->GetType(), *name_opt, call_mult);
        }
      }
    }

    auto& sites = analysis_.call_sites[name];
    auto& idx_map = site_index_[name];
    auto [site_it, inserted] = idx_map.emplace(resolved, sites.size());
    if (inserted) {
      sites.push_back(OrchCallAnalysis::CallSite{std::move(resolved), call_mult, {}, {expr.get()}});
    } else {
      auto& site = sites[site_it->second];
      site.multiplicity += call_mult;
      site.calls.push_back(expr.get());
    }
  }

  ExprPtr DirectHostShardOffset(const CallPtr& call, const VarPtr& source,
                                const std::vector<int64_t>& bank_shape) const {
    // The runtime indexes a StackedDeviceTensor once. It accepts a scalar
    // rank index followed by full slices and returns a DeviceTensor.
    if (bank_shape.size() < 3 || !source || !host_rank_stack_params_.count(source.get()) ||
        call->args_.size() != 5) {
      return nullptr;
    }
    const auto drop = As<MakeTuple>(call->args_[4]);
    if (!drop || drop->elements_.size() != 1 || !IsConstValue(drop->elements_[0], 0)) return nullptr;
    const auto window = As<MakeTuple>(call->args_[1]);
    const auto offsets = As<MakeTuple>(call->args_[2]);
    if (!window || !offsets || offsets->elements_.size() != bank_shape.size()) return nullptr;
    const auto shape = StaticShape(window->elements_);
    if (!shape || shape->size() != bank_shape.size() || shape->front() != 1 ||
        !std::equal(shape->begin() + 1, shape->end(), bank_shape.begin() + 1) ||
        !std::all_of(offsets->elements_.begin() + 1, offsets->elements_.end(),
                     [](const ExprPtr& offset) { return IsConstValue(offset, 0); })) {
      return nullptr;
    }
    return offsets->elements_[0];
  }

  void RecordLocalTensorCall(const CallPtr& call, const VarPtr& assign_var) {
    if (IsOp(call, "tensor.create") && assign_var) {
      const auto& name = bindings_->Names().at(assign_var.get());
      if (in_host_func_) {
        Pin(name);
      } else {
        analysis_.internal_buffers.insert(name);
        if (!IsNzCompatibleTensorType(assign_var->GetType()).has_value()) {
          analysis_.nz_ineligible_names.insert(name);
        }
      }
      return;
    }
    for (size_t i = 0; i < call->args_.size(); ++i) {
      const auto& arg = call->args_[i];
      auto source = bindings_->Resolve(arg);
      if (!source) {
        if (track_named_inputs_) {
          for (const auto& root : bindings_->Roots(arg)) {
            Pin(root, "unmodeled tuple argument to '" + call->op_->name_ + "'");
          }
        }
        continue;
      }
      const bool ok_nz =
          (IsOp(call, "tensor.reshape") && i == 0 && IsWholeTensorFlatten(call)) ||
          (IsOp(call, "tensor.slice") && i == 0 &&
           IsNzCompatibleTensorSliceCall(call, arg->GetType(), in_host_func_ && track_named_inputs_ ? 4 : 3));
      if (track_named_inputs_ && IsOp(call, "tensor.reshape")) {
        analysis_.nz_restrictions.try_emplace(*source, "tensor.reshape changes the matrix coordinates");
      }
      const auto arg_var = AsVarLike(arg);
      if (in_host_func_ && track_named_inputs_ && IsOp(call, "tensor.slice")) {
        const auto input = host_input_shapes_.find(*source);
        if (input != host_input_shapes_.end()) {
          const auto rank =
              ok_nz && assign_var ? DirectHostShardOffset(call, arg_var, input->second) : nullptr;
          if (rank) {
            host_shard_offsets_.emplace(assign_var.get(), rank);
          } else {
            analysis_.nz_restrictions.try_emplace(
                *source, "HOST must index the original rank bank once with a whole-shard selection");
          }
        }
      }
      if (in_host_func_ && assign_var && arg_var && IsOp(call, "tensor.slice") && ok_nz &&
          host_whole_matrix_vars_.count(arg_var.get()) != 0) {
        host_whole_matrix_vars_.insert(assign_var.get());
      }
      analysis_.dn_ineligible_names.insert(*source);
      if (!ok_nz) {
        analysis_.nz_ineligible_names.insert(*source);
        if (track_named_inputs_) {
          analysis_.nz_restrictions.try_emplace(
              *source, IsOp(call, "tensor.slice") ? "tensor.slice does not preserve both matrix axes"
                                                  : "incompatible consumer '" + call->op_->name_ + "'");
        }
        if (!IsOp(call, "tensor.slice")) analysis_.reshaped_names.insert(*source);
      }
      if (assign_var) {
        if (auto target = bindings_->Resolve(assign_var)) {
          analysis_.dn_ineligible_names.insert(*target);
          if (!ok_nz || IsOp(call, "tensor.reshape")) analysis_.nz_ineligible_names.insert(*target);
        }
      }
    }
  }

  void RecordCalls(const ExprPtr& expr, const VarPtr& assign_var = nullptr) {
    if (assign_var && As<ScalarType>(assign_var->GetType())) {
      scalar_defs_[assign_var.get()] = expr;
    }
    if (auto sub = As<Submit>(expr)) {
      if (auto gv = As<GlobalVar>(sub->op_)) {
        RecordGlobalCall(sub, gv->name_, sub->args_);
      } else if (track_named_inputs_) {
        for (const auto& arg : sub->args_) {
          for (const auto& root : bindings_->Roots(arg))
            Pin(root, "unmodeled submit to '" + sub->op_->name_ + "'");
        }
      }
      for (const auto& arg : sub->args_) {
        RecordCalls(arg);
      }
    } else if (auto call = As<Call>(expr)) {
      if (auto gv = As<GlobalVar>(call->op_)) {
        RecordGlobalCall(call, gv->name_, call->args_);
      } else {
        RecordLocalTensorCall(call, assign_var);
      }
      for (const auto& arg : call->args_) {
        RecordCalls(arg);
      }
    }
  }

  void WalkStmt(const StmtPtr& stmt) {
    if (!stmt) return;
    if (auto fs = As<ForStmt>(stmt)) {
      auto start_val = EvalScalarExpr(fs->start_, scalar_defs_, std::nullopt);
      auto step_val = EvalScalarExpr(fs->step_, scalar_defs_, std::nullopt);
      if (start_val && *start_val >= 0 && step_val && *step_val > 0 && (*step_val % 8) == 0 &&
          (*start_val % *step_val) == 0) {
        var_steps_[fs->loop_var_.get()] = *step_val;
      }
      const int64_t prev_mult = mult_;
      const bool world_size_loop =
          in_host_func_ && track_named_inputs_ && IsConstValue(fs->start_, 0) && IsConstValue(fs->step_, 1) &&
          IsOp(As<Call>(ResolveScalarDefinition(fs->stop_)), "pld.system.world_size");
      const int64_t trip_count =
          world_size_loop ? host_world_size_.value_or(1) : ForTripCount(fs, scalar_defs_);
      if (world_size_loop) world_size_indices_.insert(fs->loop_var_.get());
      mult_ = InvocationCount(prev_mult, trip_count);
      WalkStmt(fs->body_);
      mult_ = prev_mult;
      if (world_size_loop) world_size_indices_.erase(fs->loop_var_.get());
      return;
    } else if (auto as = As<AssignStmt>(stmt)) {
      RecordCalls(as->value_, as->var_);
      if (auto source = AsVarLike(as->value_); source && host_whole_matrix_vars_.count(source.get())) {
        host_whole_matrix_vars_.insert(as->var_.get());
        if (host_rank_stack_params_.count(source.get())) host_rank_stack_params_.insert(as->var_.get());
        if (auto shard = host_shard_offsets_.find(source.get()); shard != host_shard_offsets_.end()) {
          host_shard_offsets_.emplace(as->var_.get(), shard->second);
        }
      }
    } else if (auto es = As<EvalStmt>(stmt)) {
      RecordCalls(es->expr_);
    } else if (auto rs = As<ReturnStmt>(stmt)) {
      for (const auto& val : rs->value_) {
        RecordCalls(val);
      }
    }
    for (const auto& child : IterChildStmts(stmt)) {
      WalkStmt(child);
    }
  }

  ExprPtr ResolveScalarDefinition(ExprPtr expr) const {
    // FlattenCallExpr can hoist world_size() out of a bound. Follow SSA
    // bindings without guessing the result of any other dynamic expression.
    for (int depth = 0; depth < 64; ++depth) {
      auto var = AsVarLike(expr);
      auto definition = scalar_defs_.find(var.get());
      if (!var || definition == scalar_defs_.end()) break;
      expr = definition->second;
    }
    return expr;
  }

  const ProgramPtr& program_;
  const GmNames& definitions_;
  bool track_named_inputs_;
  std::optional<int64_t> host_world_size_;
  std::unordered_set<const Var*> world_size_indices_;
  GmBindings* bindings_ = nullptr;
  OrchCallAnalysis analysis_;
  int64_t mult_ = 1;
  std::unordered_map<const Var*, ExprPtr> scalar_defs_;
  std::unordered_map<const Var*, int64_t> var_steps_;
  std::set<std::pair<std::string, size_t>> invalid_param_steps_;
  std::unordered_map<std::string, std::map<std::vector<std::optional<std::string>>, size_t>> site_index_;
  bool in_host_func_ = false;
  std::unordered_set<const Var*> host_whole_matrix_vars_;
  std::unordered_set<const Var*> host_rank_stack_params_;
  std::unordered_map<std::string, std::vector<int64_t>> host_input_shapes_;
  std::unordered_map<std::string, TensorLayout> host_dispatch_layouts_;
  std::unordered_map<const Var*, ExprPtr> host_shard_offsets_;
  std::unordered_map<std::string, std::set<std::pair<int64_t, int64_t>>> fixed_dispatches_;
};

std::vector<LayoutSpec> AdmissibleTileLayouts(const TileTypePtr& tile_type) {
  if (tile_type->dtype_.IsPackedFp4()) return {EffectiveTileLayout(tile_type)};
  const auto space = tile_type->GetMemorySpace();
  if (space == MemorySpace::Mat || space == MemorySpace::Left || space == MemorySpace::Right) {
    const LayoutSpec cur = EffectiveTileLayout(tile_type);
    const LayoutSpec other =
        (cur == LayoutSpec::FractalNZ()) ? LayoutSpec::FractalZN() : LayoutSpec::FractalNZ();
    return {cur, other};
  }
  if (space == MemorySpace::Acc) {
    return {LayoutSpec::FractalAccNZ()};
  }
  return {EffectiveTileLayout(tile_type)};
}

// A copy is inserted after its definition. Operand and result allocations
// overlap during an operation, and its last-use operands can be released before
// the conversion workspace is needed.
struct UnmodeledVecTile {
  size_t definition;
  size_t last;
  int64_t bytes;
  std::optional<std::string> registered_name;
};

void RecordVecMemoryRegion(LayoutProblemInstance& instance, VecMemoryRegion region,
                           const std::unordered_map<std::string, size_t>& definitions,
                           const std::unordered_map<std::string, size_t>& other_uses, size_t event_count,
                           uint64_t reserved_bytes, const std::vector<UnmodeledVecTile>& unmodeled_tiles) {
  const size_t count = region.operation_positions.size();
  if (region.capacity == 0 || count == 0) return;
  std::unordered_map<std::string, size_t> first_uses;
  auto record = [&](const std::string& name, size_t position) {
    const auto* node = instance.FindValue(name);
    if (!node || node->memory_space != MemorySpace::Vec) return;
    auto& first = first_uses.try_emplace(name, position).first->second;
    first = std::min(first, position);
    auto& last = region.last_uses[name];
    last = std::max(last, position);
  };
  for (size_t i = 0; i < count; ++i) {
    const auto& op = instance.operations[region.first_operation + i];
    const size_t position = region.operation_positions[i];
    for (const auto& input : op.inputs) record(input, position);
    // Additional stores have a synthetic Vec result for the solver only.
    if (op.op_name != "tile.store") record(op.output, position);
  }
  for (const auto& [name, position] : other_uses) record(name, position);
  if (first_uses.empty()) return;
  for (size_t i = count; i > 0; --i) {
    const auto& output = instance.operations[region.first_operation + i - 1].output;
    if (const auto it = region.backings.find(output); it != region.backings.end()) {
      const size_t last = region.last_uses.at(output);
      for (const auto& backing : it->second) {
        auto& backing_last = region.last_uses[backing];
        backing_last = std::max(backing_last, last);
      }
    }
  }

  std::vector<int64_t> original_bytes(event_count + 1, 0);
  original_bytes.front() = static_cast<int64_t>(std::min(reserved_bytes, region.capacity));
  for (const auto& tile : unmodeled_tiles) {
    if (tile.registered_name) continue;
    original_bytes[tile.definition] += tile.bytes;
    original_bytes[region.EffectiveLastUse(tile.definition, tile.last) + 1] -= tile.bytes;
  }
  for (auto& [name, last] : region.last_uses) {
    auto it = definitions.find(name);
    const size_t definition = it != definitions.end() ? it->second : first_uses.at(name);
    last = region.EffectiveLastUse(definition, last);
    const size_t copy = definition == 0 ? 0 : definition + 1;
    region.copy_positions.emplace(name, copy);
    if (region.backings.count(name) == 0) {
      const int64_t bytes = instance.values.at(name).ByteSize();
      original_bytes[definition] += bytes;
      original_bytes[last + 1] -= bytes;
    }
  }
  original_bytes.pop_back();
  std::partial_sum(original_bytes.begin(), original_bytes.end(), original_bytes.begin());
  for (const auto& [name, copy] : region.copy_positions) {
    if (copy > region.last_uses.at(name)) continue;
    const auto occupied = static_cast<uint64_t>(original_bytes[copy]);
    instance.values.at(name).vec_conversion_headroom = region.capacity - std::min(region.capacity, occupied);
  }
  region.original_bytes = std::move(original_bytes);
  instance.vec_memory_regions.push_back(std::move(region));
}

class PostInferInstanceExtractor {
 public:
  PostInferInstanceExtractor(const LayoutCostModel& model, const backend::Backend* backend,
                             const std::unordered_set<std::string>& reusable,
                             const std::unordered_set<std::string>& nz_input_roots,
                             const std::unordered_map<std::string, TensorTypePtr>& host_inputs,
                             const std::unordered_map<std::string, double>& projected_dispatches,
                             const std::unordered_set<std::string>& external_out_names,
                             const std::unordered_set<std::string>& reshaped_names,
                             const std::unordered_set<std::string>& nz_ineligible_names,
                             const std::unordered_set<std::string>& dn_ineligible_names,
                             const std::unordered_set<std::string>& internal_buffers,
                             LayoutProblemInstance& instance,
                             std::unordered_set<std::string>& consumed_values, size_t& op_counter,
                             std::string tile_prefix, int64_t call_multiplier, const GmNames& gm_names,
                             const std::map<std::pair<std::string, size_t>, int64_t>& param_steps)
      : model_(model),
        backend_(backend),
        reusable_(reusable),
        nz_input_roots_(nz_input_roots),
        host_inputs_(host_inputs),
        projected_dispatches_(projected_dispatches),
        external_out_names_(external_out_names),
        reshaped_names_(reshaped_names),
        nz_ineligible_names_(nz_ineligible_names),
        dn_ineligible_names_(dn_ineligible_names),
        internal_buffers_(internal_buffers),
        instance_(instance),
        consumed_values_(consumed_values),
        op_counter_(op_counter),
        tile_prefix_(std::move(tile_prefix)),
        loop_multiplier_(std::max<int64_t>(1, call_multiplier)),
        gm_names_(gm_names),
        param_steps_(param_steps) {}

  void Extract(const FunctionPtr& func, LayoutWritebackBindings* bindings) {
    for (size_t i = 0; i < func->params_.size(); ++i) {
      RegisterGmVar(func->params_[i], true, i);
    }
    RegisterGmDefinitions(func->body_);
    AnalyzeVecTiles(func);
    for (const auto& formal : func->params_) {
      if (As<TileType>(formal->GetType())) RegisterTileVar(formal, std::nullopt);
    }
    const size_t begin = instance_.operations.size();
    vec_memory_.first_operation = begin;
    WalkStmt(func->body_);
    if (backend_) {
      vec_memory_.capacity = backend_->GetMemSize(MemorySpace::Vec);
      uint64_t reserved_bytes = 0;
      if (has_reserve_buffer_) {
        const auto policy = backend_->CreateMemoryAllocatorPolicy();
        const auto reserved = ResolveReserveBufferBases(func, *policy);
        if (auto it = reserved.reserved_end_by_space.find(MemorySpace::Vec);
            it != reserved.reserved_end_by_space.end()) {
          reserved_bytes = it->second;
        }
      }
      RecordVecMemoryRegion(instance_, std::move(vec_memory_), vec_definitions_, vec_other_uses_,
                            vec_position_ + 1, reserved_bytes, unmodeled_vec_tiles_);
    }
    if (bindings) {
      auto& site = bindings->sites[tile_prefix_];
      site.value_names = std::move(var_to_name_);
      site.begin = begin;
      site.end = instance_.operations.size();
    }
  }

 private:
  void RegisterGmVar(const VarPtr& param, bool formal = false, size_t param_index = 0) {
    auto tt = AsTensorTypeLike(param->GetType());
    if (!tt) {
      return;
    }
    const bool is_distributed = IsA<DistributedTensorType>(param->GetType());
    auto [shape, is_static] = TensorShapeWithDynamicFallback(tt->shape_);
    if (shape.empty()) {
      return;
    }
    auto name_it = gm_names_.find(param.get());
    if (name_it == gm_names_.end()) return;
    std::string name = name_it->second;
    // A device formal describes one matrix. The GM decision represents the
    // entire public input, including any bank ranks not used by this site.
    bool selectable_host_projection = false;
    const auto declared_layout = LayoutSpec::Tensor(detail::TensorTypeLayout(tt));
    if (auto input = host_inputs_.find(name);
        input != host_inputs_.end() && projected_dispatches_.count(name) &&
        nz_ineligible_names_.count(name) == 0 && reshaped_names_.count(name) == 0 &&
        HasDenseLogicalInputLayout(tt) && tt->dtype_ == input->second->dtype_) {
      const auto input_shape = IsNzCompatibleTensorType(input->second, 4);
      if (input_shape && is_static && shape.size() == 2 &&
          std::equal(shape.begin(), shape.end(), input_shape->end() - 2)) {
        selectable_host_projection = true;
        shape = *input_shape;
      }
    }
    // The supplier starts with logical ND bytes even where a device formal
    // contains an authored NZ preference. A proven projection can reopen it.
    const LayoutSpec cur_layout = selectable_host_projection ? LayoutSpec::ND() : declared_layout;
    const bool is_ext_out = external_out_names_.count(name) != 0;
    const bool is_reusable = reusable_.count(name) != 0 ||
                             (nz_input_roots_.empty() && (reusable_.count(CleanName(param->name_hint_)) ||
                                                          reusable_.count(param->name_hint_)));
    // Reuse affects amortization. Changing a caller-owned buffer also requires
    // explicit permission to change the input or output representation.
    const bool owned = internal_buffers_.count(name) != 0;
    // A packed FP4 layout change retargets the dtype's carrier axis. This
    // pass can keep a declared packed descriptor but cannot reuse its shape
    // to introduce a different GM layout.
    const bool allow_relayout =
        !is_distributed && !tt->dtype_.IsPackedFp4() &&
        (is_ext_out
             ? model_.allow_external_output_relayout
             : (owned || (formal && (model_.allow_external_input_relayout || selectable_host_projection ||
                                     (host_inputs_.count(name) == 0 && nz_input_roots_.count(name))))));
    // The logical NZ selection pipeline reaches this point before blocking. Only
    // private, dense, unaliased-by-opaque-code storage may shed its declared layout.
    const auto& view = tt->tensor_view_;
    const bool compact_dn_stride = cur_layout == LayoutSpec::DN() && is_static && shape.size() == 2 && view &&
                                   StaticShape(view->stride) == std::vector<int64_t>{1, shape[0]};
    const bool private_dense = model_.reselect_internal_layouts && owned && !is_ext_out && allow_relayout &&
                               is_static && view && !view->nz_blocked &&
                               (view->stride.empty() || compact_dn_stride) && view->valid_shape.empty() &&
                               view->pad == PadValue::null && reshaped_names_.count(name) == 0;
    const bool reopen =
        private_dense && ((cur_layout == LayoutSpec::NZ() && nz_ineligible_names_.count(name) == 0) ||
                          (cur_layout == LayoutSpec::DN() && dn_ineligible_names_.count(name) == 0));
    const bool allow_nz = !is_distributed && is_static && (cur_layout == LayoutSpec::ND() || reopen) &&
                          reshaped_names_.count(name) == 0 && nz_ineligible_names_.count(name) == 0 &&
                          (selectable_host_projection || IsNzCompatibleShape(shape, tt->dtype_)) &&
                          allow_relayout;
    // Promoting a host argument to DN changes its physical strides. Only introduce
    // DN for buffers owned by the orchestration, where the pass controls both ends.
    const bool allow_dn = owned && !is_ext_out && !is_distributed && is_static && shape.size() == 2 &&
                          (cur_layout == LayoutSpec::ND() || cur_layout == LayoutSpec::DN() || reopen) &&
                          reshaped_names_.count(name) == 0 && dn_ineligible_names_.count(name) == 0 &&
                          allow_relayout;
    std::vector<LayoutSpec> adm_list{cur_layout};
    if (reopen) adm_list.push_back(LayoutSpec::ND());
    if (allow_dn && cur_layout != LayoutSpec::DN()) {
      adm_list.push_back(LayoutSpec::DN());
    }
    if (allow_nz && cur_layout != LayoutSpec::NZ()) {
      adm_list.push_back(LayoutSpec::NZ());
    }

    if (const ValueNode* prev = instance_.FindValue(name); prev != nullptr) {
      if (!formal) {
        // A local alias shares the registered storage even when its logical
        // shape is smaller than the backing allocation.
        var_to_name_[param.get()] = name;
        return;
      }
      if (prev->current_layout == cur_layout && prev->dtype == tt->dtype_ && prev->shape == shape) {
        var_to_name_[param.get()] = name;
        std::vector<LayoutSpec> common;
        for (const auto& lay : prev->admissible_layouts) {
          if (std::find(adm_list.begin(), adm_list.end(), lay) != adm_list.end()) {
            common.push_back(lay);
          }
        }
        if (common.empty()) {
          common.push_back(prev->current_layout);
        }
        if (common != prev->admissible_layouts) {
          ValueNode updated = *prev;
          updated.admissible_layouts = std::move(common);
          instance_.AddValue(std::move(updated));
        }
        return;
      }
      if (prev->admissible_layouts.size() != 1 || prev->admissible_layouts[0] != prev->current_layout) {
        ValueNode updated = *prev;
        updated.admissible_layouts = {prev->current_layout};
        instance_.AddValue(std::move(updated));
      }
      name = "$" + tile_prefix_ + "gm.formal." + std::to_string(param_index);
      adm_list = {cur_layout};
    }

    var_to_name_[param.get()] = name;
    ValueNode node;
    node.name = name;
    node.shape = std::move(shape);
    node.dtype = tt->dtype_;
    node.memory_space = MemorySpace::DDR;
    node.current_layout = cur_layout;
    node.admissible_layouts = std::move(adm_list);
    node.producer_op_id = std::nullopt;
    node.is_reusable_weight = is_reusable;
    node.is_external_output = is_ext_out;
    node.is_named_nz_input = nz_input_roots_.count(name) != 0;
    instance_.AddValue(std::move(node));
  }

  void RegisterGmDefinitions(const StmtPtr& stmt) {
    if (auto assign = As<AssignStmt>(stmt)) {
      RegisterGmVar(assign->var_);
    } else if (auto loop = As<ForStmt>(stmt)) {
      for (const auto& arg : loop->iter_args_) RegisterGmVar(arg);
      for (const auto& result : loop->return_vars_) RegisterGmVar(result);
    } else if (auto loop = As<WhileStmt>(stmt)) {
      for (const auto& arg : loop->iter_args_) RegisterGmVar(arg);
      for (const auto& result : loop->return_vars_) RegisterGmVar(result);
    } else if (auto branch = As<IfStmt>(stmt)) {
      for (const auto& result : branch->return_vars_) RegisterGmVar(result);
    }
    for (const auto& child : IterChildStmts(stmt)) RegisterGmDefinitions(child);
  }

  static std::optional<std::pair<std::vector<int64_t>, int>> IsVecNzTileShapeEligible(const VarPtr& var) {
    auto tile_type = As<TileType>(var->GetType());
    if (!tile_type) return std::nullopt;
    const MemorySpace space = tile_type->GetMemorySpace().value_or(MemorySpace::Vec);
    if (space != MemorySpace::Vec || EffectiveTileLayout(tile_type) != LayoutSpec::RowMajorFlat()) {
      return std::nullopt;
    }
    auto shape = StaticShape(tile_type->shape_);
    auto c0_opt = TryC0Elements(tile_type->dtype_);
    if (!shape.has_value() || !c0_opt.has_value() || shape->size() != 2 || (*shape)[0] <= 0 ||
        (*shape)[1] <= 0 || *c0_opt <= 0 || ((*shape)[0] % 16) != 0 || ((*shape)[1] % *c0_opt) != 0) {
      return std::nullopt;
    }
    if (tile_type->tile_view_.has_value() && !tile_type->tile_view_->valid_shape.empty() &&
        StaticShape(tile_type->tile_view_->valid_shape) != shape) {
      return std::nullopt;
    }
    return std::make_pair(*shape, *c0_opt);
  }

  void ClassifyVecCallRoles(const CallPtr& call, const VarPtr& lhs) {
    const std::string& op_name = call->op_->name_;
    for (const auto& [index, var] : TileOperands(call)) {
      if (IsTileWorkspace(call, index)) {
        vec_tmp_vars_.insert(var.get());
      } else {
        vec_non_tmp_vars_.insert(var.get());
        if (IsOp(call, "tile.reshape")) vec_non_row_expand_consumer_vars_.insert(var.get());
      }
    }
    if (lhs && As<TileType>(lhs->GetType()) && !IsOp(call, "tile.create")) {
      vec_non_tmp_vars_.insert(lhs.get());
    }
    if (lhs && As<TileType>(lhs->GetType()) && IsOp(call, "tile.create")) {
      vec_non_row_expand_consumer_vars_.insert(lhs.get());
    }
    if (lhs && RowReductionOps().count(op_name) != 0) {
      vec_row_reduction_vars_.insert(lhs.get());
    }
    if (IsOp(call, "tile.row_expand_add") && call->args_.size() >= 2) {
      if (auto v = AsVarLike(call->args_[1])) {
        vec_row_expand_add_vars_.insert(v.get());
      }
    }
  }

  void AnalyzeVecTiles(const FunctionPtr& func) {
    vec_tmp_vars_.clear();
    vec_non_tmp_vars_.clear();
    vec_row_expand_add_vars_.clear();
    vec_non_row_expand_consumer_vars_.clear();
    vec_row_reduction_vars_.clear();
    vec_nz_eligible_vars_.clear();
    nz_fact_ctx_ = NzFactContext{};

    const auto& tensor_vars = var_to_name_;
    std::vector<std::pair<const Var*, const Var*>> assign_aliases;
    VarUnionFind uf;
    std::unordered_set<const Var*> disqualified;
    std::vector<const Var*> disqualified_order;
    auto mark_disqualified = [&](const Var* v) {
      if (v != nullptr && disqualified.insert(v).second) {
        disqualified_order.push_back(v);
      }
    };
    std::unordered_map<const Var*, std::pair<std::vector<int64_t>, int>> comp_sig;

    for (size_t idx = 0; idx < func->params_.size(); ++idx) {
      const auto& param = func->params_[idx];
      if (As<ScalarType>(param->GetType())) {
        auto it = param_steps_.find({func->name_, idx});
        if (it != param_steps_.end()) {
          nz_fact_ctx_.SeedParamStep(param, it->second);
        }
      } else if (As<TileType>(param->GetType())) {
        mark_disqualified(param.get());
        vec_non_row_expand_consumer_vars_.insert(param.get());
      }
    }

    auto scan_facts = [&](auto& self, const StmtPtr& stmt) -> void {
      if (auto as = As<AssignStmt>(stmt)) {
        if (As<ScalarType>(as->var_->GetType())) {
          nz_fact_ctx_.scalar_defs[as->var_.get()] = as->value_;
          if (auto call = As<Call>(as->value_);
              call && (IsOp(call, "tile.get_block_idx") || IsOp(call, "tile.get_block_num"))) {
            nz_fact_ctx_.non_neg_vars.insert(as->var_.get());
          }
        }
      } else if (auto fs = As<ForStmt>(stmt)) {
        nz_fact_ctx_.loop_ranges[fs->loop_var_.get()] = {fs->start_, fs->step_};
        for (const auto& ia : fs->iter_args_) {
          if (As<TileType>(ia->GetType())) {
            mark_disqualified(ia.get());
            vec_non_row_expand_consumer_vars_.insert(ia.get());
            if (auto iv = AsVarLike(ia->initValue_)) {
              mark_disqualified(iv.get());
              vec_non_row_expand_consumer_vars_.insert(iv.get());
            }
          }
        }
        for (size_t i = 0; i < fs->return_vars_.size() && i < fs->iter_args_.size(); ++i) {
          if (As<TileType>(fs->return_vars_[i]->GetType())) {
            mark_disqualified(fs->return_vars_[i].get());
            vec_non_row_expand_consumer_vars_.insert(fs->return_vars_[i].get());
          }
        }
      } else if (auto ws = As<WhileStmt>(stmt)) {
        for (const auto& ia : ws->iter_args_) {
          if (As<TileType>(ia->GetType())) {
            mark_disqualified(ia.get());
            vec_non_row_expand_consumer_vars_.insert(ia.get());
            if (auto iv = AsVarLike(ia->initValue_)) {
              mark_disqualified(iv.get());
              vec_non_row_expand_consumer_vars_.insert(iv.get());
            }
          }
        }
        for (size_t i = 0; i < ws->return_vars_.size() && i < ws->iter_args_.size(); ++i) {
          if (As<TileType>(ws->return_vars_[i]->GetType())) {
            mark_disqualified(ws->return_vars_[i].get());
            vec_non_row_expand_consumer_vars_.insert(ws->return_vars_[i].get());
          }
        }
      } else if (auto ifs = As<IfStmt>(stmt)) {
        for (const auto& rv : ifs->return_vars_) {
          if (As<TileType>(rv->GetType())) {
            mark_disqualified(rv.get());
            vec_non_row_expand_consumer_vars_.insert(rv.get());
          }
        }
      }
      for (const auto& child : IterChildStmts(stmt)) {
        self(self, child);
      }
    };
    scan_facts(scan_facts, func->body_);
    const auto facts = nz_fact_ctx_.Facts();

    auto union_nz = [&](const Var* u, const Var* v) {
      const Var* ru = uf.Find(u);
      const Var* rv = uf.Find(v);
      if (ru == rv) return;
      const Var* r = uf.Union(ru, rv);
      const Var* other = (r == ru) ? rv : ru;
      if (disqualified.count(ru) != 0 || disqualified.count(rv) != 0) {
        mark_disqualified(r);
      }
      auto it_r = comp_sig.find(r);
      auto it_o = comp_sig.find(other);
      if (it_r != comp_sig.end() && it_o != comp_sig.end() && it_r->second != it_o->second) {
        mark_disqualified(r);
      } else if (it_o != comp_sig.end() && it_r == comp_sig.end()) {
        comp_sig.emplace(r, it_o->second);
      }
    };

    auto check_tile_eligible = [&](const VarPtr& var) {
      auto sig = IsVecNzTileShapeEligible(var);
      if (!sig.has_value()) {
        mark_disqualified(var.get());
        return;
      }
      const Var* r = uf.Find(var.get());
      auto it = comp_sig.find(r);
      if (it != comp_sig.end() && it->second != *sig) {
        mark_disqualified(r);
      } else {
        comp_sig.emplace(r, *sig);
      }
    };

    auto check_load_or_store_nz = [&](const CallPtr& call, const VarPtr& tile_var, const ExprPtr& gm_arg,
                                      bool is_store) {
      auto tt = As<TileType>(tile_var->GetType());
      if (!tt || tt->GetMemorySpace().value_or(MemorySpace::Vec) != MemorySpace::Vec) {
        return;
      }
      check_tile_eligible(tile_var);
      auto gm_v = AsVarLike(gm_arg);
      const ValueNode* gm_node = nullptr;
      if (gm_v) {
        auto it = tensor_vars.find(gm_v.get());
        if (it != tensor_vars.end()) {
          gm_node = instance_.FindValue(it->second);
        }
      }
      const bool gm_allows_nz =
          gm_node != nullptr &&
          std::find(gm_node->admissible_layouts.begin(), gm_node->admissible_layouts.end(),
                    LayoutSpec::NZ()) != gm_node->admissible_layouts.end();
      const bool nz_ok = gm_allows_nz && gm_v &&
                         (is_store ? IsNzCompatibleVecTileStoreCall(call, gm_v->GetType(), facts)
                                   : IsNzCompatibleTileLoadCall(call, gm_v->GetType(), facts, model_));
      if (!nz_ok) {
        mark_disqualified(tile_var.get());
      }
    };

    auto visit_call = [&](const CallPtr& call, const VarPtr& lhs) {
      ClassifyVecCallRoles(call, lhs);
      const std::string& op_name = call->op_->name_;
      if (IsOp(call, "tile.load") && lhs && As<TileType>(lhs->GetType())) {
        check_load_or_store_nz(call, lhs, !call->args_.empty() ? call->args_[0] : nullptr, false);
      } else if (IsOp(call, "tile.store") && call->args_.size() >= 3) {
        if (auto tv = AsVarLike(call->args_[0]); tv && As<TileType>(tv->GetType())) {
          check_load_or_store_nz(call, tv, call->args_[2], true);
        }
      } else if ((IsOp(call, "tile.create") || IsOp(call, "tile.full")) && lhs &&
                 As<TileType>(lhs->GetType())) {
        mark_disqualified(lhs.get());
      } else if (op_name.rfind("tile.", 0) == 0) {
        std::vector<VarPtr> data_tiles;
        for (const auto& [index, var] : TileOperands(call)) {
          if (!IsTileWorkspace(call, index)) data_tiles.push_back(var);
        }
        if (lhs && As<TileType>(lhs->GetType())) {
          data_tiles.push_back(lhs);
        }
        for (const auto& tv : data_tiles) {
          check_tile_eligible(tv);
        }
        for (size_t i = 1; i < data_tiles.size(); ++i) {
          union_nz(data_tiles[0].get(), data_tiles[i].get());
        }
        if (!data_tiles.empty() && !CanUseVecNzPointwise(call)) {
          mark_disqualified(data_tiles[0].get());
        }
      }
    };

    auto scan_calls = [&](auto& self, const StmtPtr& stmt) -> void {
      if (auto as = As<AssignStmt>(stmt)) {
        if (auto call = As<Call>(as->value_)) {
          visit_call(call, as->var_);
        } else if (auto rv = AsVarLike(as->value_); rv && As<TileType>(as->var_->GetType())) {
          union_nz(as->var_.get(), rv.get());
          assign_aliases.emplace_back(as->var_.get(), rv.get());
        }
      } else if (auto es = As<EvalStmt>(stmt)) {
        if (auto call = As<Call>(es->expr_)) {
          visit_call(call, nullptr);
        }
      } else if (auto rs = As<ReturnStmt>(stmt)) {
        for (const auto& val : rs->value_) {
          if (auto call = As<Call>(val)) {
            visit_call(call, nullptr);
          } else if (auto tv = AsVarLike(val); tv && As<TileType>(tv->GetType())) {
            vec_non_row_expand_consumer_vars_.insert(tv.get());
          }
        }
      } else if (auto ys = As<YieldStmt>(stmt)) {
        for (const auto& val : ys->value_) {
          if (auto tv = AsVarLike(val); tv && As<TileType>(tv->GetType())) {
            mark_disqualified(tv.get());
            vec_non_row_expand_consumer_vars_.insert(tv.get());
          }
        }
      } else {
        auto mark_loop_inits = [&](const auto& loop) {
          for (const auto& ia : loop->iter_args_) {
            if (auto init = AsVarLike(ia->initValue_); init && As<TileType>(ia->GetType())) {
              mark_disqualified(init.get());
              vec_non_row_expand_consumer_vars_.insert(init.get());
            }
          }
        };
        if (auto fs = As<ForStmt>(stmt)) mark_loop_inits(fs);
        if (auto ws = As<WhileStmt>(stmt)) mark_loop_inits(ws);
      }
      for (const auto& child : IterChildStmts(stmt)) {
        self(self, child);
      }
    };
    scan_calls(scan_calls, func->body_);

    // Propagate non-row-expand consumer usage across Var aliases so aliased row vectors
    // consumed outside `tile.row_expand_add` never receive `ROW_BCAST_FLAT`.
    for (bool changed = true; changed;) {
      changed = false;
      for (const auto& [dst_v, src_v] : assign_aliases) {
        const bool d_in = vec_non_row_expand_consumer_vars_.count(dst_v) != 0;
        const bool s_in = vec_non_row_expand_consumer_vars_.count(src_v) != 0;
        if (d_in && !s_in) {
          vec_non_row_expand_consumer_vars_.insert(src_v);
          changed = true;
        } else if (s_in && !d_in) {
          vec_non_row_expand_consumer_vars_.insert(dst_v);
          changed = true;
        }
      }
    }

    for (const Var* v : vec_tmp_vars_) {
      if (vec_non_tmp_vars_.count(v) != 0) {
        mark_disqualified(v);
      }
    }
    std::unordered_set<const Var*> disq_roots;
    for (const Var* v : disqualified_order) {
      disq_roots.insert(uf.Find(v));
    }
    for (const Var* v : vec_non_tmp_vars_) {
      if (vec_tmp_vars_.count(v) == 0 && disq_roots.count(uf.Find(v)) == 0) {
        vec_nz_eligible_vars_.insert(v);
      }
    }
  }

  std::vector<LayoutSpec> AdmissibleVecTileLayouts(const VarPtr& var) {
    auto tile_type = As<TileType>(var->GetType());
    const LayoutSpec cur_layout = EffectiveTileLayout(tile_type);
    if (tile_type->dtype_.IsPackedFp4() ||
        (vec_tmp_vars_.count(var.get()) != 0 && vec_non_tmp_vars_.count(var.get()) == 0)) {
      return {cur_layout};
    }
    auto shape = StaticShape(tile_type->shape_);
    if (!shape.has_value() || shape->size() != 2) {
      return {cur_layout};
    }
    const int64_t m_dim = (*shape)[0];
    const int64_t n_dim = (*shape)[1];
    const int bpe = DTypeByteWidthSafe(tile_type->dtype_);
    if (n_dim == 1 && (m_dim > 1 || vec_row_reduction_vars_.count(var.get()) != 0)) {
      std::vector<LayoutSpec> candidates{cur_layout};
      const bool dtype_bcast_ok = tile_type->dtype_ == DataType::INT8 ||
                                  tile_type->dtype_ == DataType::INT16 ||
                                  tile_type->dtype_ == DataType::INT32 ||
                                  tile_type->dtype_ == DataType::FP16 || tile_type->dtype_ == DataType::FP32;
      const bool valid_ok = !tile_type->tile_view_.has_value() ||
                            tile_type->tile_view_->valid_shape.empty() ||
                            StaticShape(tile_type->tile_view_->valid_shape) == shape;
      if (cur_layout == LayoutSpec::ColMajorFlat() && vec_row_expand_add_vars_.count(var.get()) != 0 &&
          vec_non_row_expand_consumer_vars_.count(var.get()) == 0 && m_dim > 1 && bpe > 0 &&
          ((m_dim * bpe) % 32) == 0 && dtype_bcast_ok && valid_ok) {
        candidates.push_back(LayoutSpec::RowBcastFlat());
      }
      return candidates;
    }
    if (m_dim == 1 && vec_row_reduction_vars_.count(var.get()) == 0) {
      return {cur_layout};
    }
    std::vector<LayoutSpec> candidates{cur_layout};
    const LayoutSpec other =
        cur_layout == LayoutSpec::RowMajorFlat() ? LayoutSpec::ColMajorFlat() : LayoutSpec::RowMajorFlat();
    if (model_.SupportsVecTileConversion(detail::VecTransposeGeometry(tile_type), cur_layout, other)) {
      candidates.push_back(other);
    }
    if (vec_nz_eligible_vars_.count(var.get()) != 0) {
      candidates.push_back(LayoutSpec::FractalNZ());
    }
    return candidates;
  }

  // Distinct SSA definitions can share a name hint, including a generated suffix.
  // A scratch allocation defines a value without advancing the operation counter.
  std::string FreshValueName(const std::string& hint) {
    if (instance_.FindValue(hint) == nullptr) return hint;
    const std::string base = hint + "_" + std::to_string(op_counter_);
    std::string candidate = base;
    while (instance_.FindValue(candidate) != nullptr) {
      candidate = base + "_" + std::to_string(next_value_suffix_++);
    }
    return candidate;
  }

  const ValueNode* RegisterTileVar(const VarPtr& var, const std::optional<std::string>& producer_op_id,
                                   const std::vector<std::string>& op_inputs = {},
                                   bool is_strided_subview = false, bool is_vec_slice = false) {
    auto tile_type = As<TileType>(var->GetType());
    if (!tile_type) return nullptr;
    auto shape = StaticShape(tile_type->shape_);
    if (!shape.has_value()) return nullptr;
    for (const auto& inp : op_inputs) {
      consumed_values_.insert(inp);
    }
    std::string name = tile_prefix_ + CleanName(var->name_hint_);
    if (instance_.FindValue(name) != nullptr) {
      name = FreshValueName(tile_prefix_ + var->name_hint_);
    }
    var_to_name_[var.get()] = name;
    const MemorySpace space = tile_type->GetMemorySpace().value_or(MemorySpace::Vec);
    const LayoutSpec cur_layout = EffectiveTileLayout(tile_type);
    std::vector<LayoutSpec> admissible;
    if (is_strided_subview && model_.TargetArch() == "a2a3") {
      // A view retains its source layout. Keep consumer configurations from
      // requesting a layout that would require a compact copy of the view.
      admissible = {cur_layout};
    } else if (space == MemorySpace::Vec) {
      admissible = AdmissibleVecTileLayouts(var);
      if (!producer_op_id.has_value()) {
        std::vector<LayoutSpec> filtered;
        for (const auto& lay : admissible) {
          if (lay != LayoutSpec::FractalNZ()) filtered.push_back(lay);
        }
        if (filtered.empty()) filtered.push_back(cur_layout);
        admissible = std::move(filtered);
      }
    } else {
      admissible = AdmissibleTileLayouts(tile_type);
    }
    if (std::find(admissible.begin(), admissible.end(), cur_layout) == admissible.end()) {
      admissible.insert(admissible.begin(), cur_layout);
    }
    ValueNode node;
    node.name = name;
    node.shape = std::move(*shape);
    node.dtype = tile_type->dtype_;
    node.memory_space = space;
    node.current_layout = cur_layout;
    node.admissible_layouts = std::move(admissible);
    node.producer_op_id = producer_op_id;
    node.is_strided_subview = is_strided_subview;
    node.has_dense_vec_span =
        space == MemorySpace::Vec && detail::VecTransposeGeometry(tile_type).has_dense_vec_span;
    node.is_vec_slice = is_vec_slice;
    if (space == MemorySpace::Vec && !producer_op_id) {
      node.conversion_invocations = InvocationCount(model_.expected_uses, loop_multiplier_);
      // A generic consumer can introduce an operand whose producer has no layout factor.
      size_t definition = vec_position_;
      if (auto it = unmodeled_vec_vars_.find(var.get()); it != unmodeled_vec_vars_.end()) {
        auto& tile = unmodeled_vec_tiles_[it->second];
        if (tile.registered_name) {
          var_to_name_[var.get()] = *tile.registered_name;
          return instance_.FindValue(*tile.registered_name);
        }
        definition = tile.definition;
        vec_other_uses_[name] = tile.last;
        tile.registered_name = name;
      }
      vec_definitions_.emplace(name, definition);
    }
    instance_.AddValue(std::move(node));
    return instance_.FindValue(name);
  }

  std::string MakeOpId(const std::string& var_hint, const std::string& tag) {
    return tile_prefix_ + "op_" + std::to_string(op_counter_++) + "_" + var_hint + "_" + tag;
  }

  void RecordVecBacking(const ValueNode& view, const ValueNode& input) {
    if (view.memory_space != MemorySpace::Vec || input.memory_space != MemorySpace::Vec) return;
    auto& backings = vec_memory_.backings[view.name];
    if (std::find(backings.begin(), backings.end(), input.name) == backings.end()) {
      backings.push_back(input.name);
    }
  }

  void AddOperation(OperationNode operation) {
    vec_memory_.operation_positions.push_back(++vec_position_);
    ++vec_position_;  // The writer inserts conversion instructions after the operation.
    instance_.operations.push_back(std::move(operation));
  }

  void ExtractLoad(const AssignStmtPtr& op, const CallPtr& rhs) {
    if (rhs->args_.empty()) return;
    auto src_arg = AsVarLike(rhs->args_[0]);
    if (!src_arg || var_to_name_.count(src_arg.get()) == 0) return;
    const std::string src_name = var_to_name_.at(src_arg.get());
    const std::string op_id = MakeOpId(op->var_->name_hint_, "load");
    const ValueNode* dst_node = RegisterTileVar(op->var_, op_id, {src_name});
    const ValueNode* src_node = instance_.FindValue(src_name);
    if (!src_node || !dst_node) return;

    std::optional<ValueNode> projected_source;
    if (host_inputs_.count(src_name)) {
      if (auto projected_shape = StaticShape(AsTensorTypeLike(src_arg->GetType())->shape_);
          projected_shape && *projected_shape != src_node->shape) {
        projected_source = *src_node;
        projected_source->shape = std::move(*projected_shape);
      }
    }
    const auto& load_source = projected_source ? *projected_source : *src_node;

    const auto facts = nz_fact_ctx_.Facts();
    OperationNode op_node;
    op_node.op_id = op_id;
    op_node.op_name = rhs->op_->name_;
    op_node.inputs = {src_name};
    op_node.output = dst_node->name;
    op_node.invocations = InvocationCount(model_.expected_uses, loop_multiplier_);
    if (projected_source) {
      if (auto visits = projected_dispatches_.find(src_name); visits != projected_dispatches_.end()) {
        // A rank receives its layer bank. Counting each layer as potentially
        // fresh avoids crediting an unrelated layer as an L2 reread.
        const double layers = src_node->shape.size() == 4 ? static_cast<double>(src_node->shape[1]) : 1.0;
        op_node.projected_mte2_source_bytes =
            static_cast<double>(load_source.ByteSize()) * visits->second * layers;
      }
    }

    for (const auto& s_lay : src_node->admissible_layouts) {
      for (const auto& d_lay : dst_node->admissible_layouts) {
        if (!model_.IsValidLoadConfig(load_source, *dst_node, s_lay, d_lay)) {
          continue;
        }
        if (!dst_node->dtype.IsPackedFp4() && dst_node->memory_space == MemorySpace::Vec &&
            s_lay == LayoutSpec::NZ() &&
            !IsNzCompatibleTileLoadCall(rhs, src_arg->GetType(), facts, model_)) {
          continue;
        }
        const double cost = model_.EstimateLoadCost(load_source, *dst_node, s_lay, d_lay);
        std::string kind = "ND2NZ";
        if (s_lay == LayoutSpec::NZ()) {
          kind = "NZ2NZ";
        } else if (s_lay == LayoutSpec::DN()) {
          kind = "DN2CM";
        } else if (dst_node->memory_space == MemorySpace::Vec) {
          kind = "ND2RM";
        }
        op_node.configs.push_back(OperatorConfig{op_id,
                                                 rhs->op_->name_,
                                                 {s_lay},
                                                 d_lay,
                                                 cost,
                                                 "GM(" + s_lay.ShortName() + ")->" +
                                                     MemorySpaceToString(dst_node->memory_space) + " (" +
                                                     kind + ")"});
      }
    }
    AddOperation(std::move(op_node));
  }

  void ExtractTransposeView(const AssignStmtPtr& op, const CallPtr& rhs) {
    if (rhs->args_.empty()) return;
    auto src_arg = AsVarLike(rhs->args_[0]);
    if (!src_arg || var_to_name_.count(src_arg.get()) == 0) return;
    const std::string src_name = var_to_name_.at(src_arg.get());
    const std::string op_id = MakeOpId(op->var_->name_hint_, "tview");
    const auto* src_node = instance_.FindValue(src_name);
    if (!src_node) return;
    const auto* dst_node =
        RegisterTileVar(op->var_, op_id, {src_name}, src_node->is_strided_subview, src_node->is_vec_slice);
    if (!dst_node) return;
    RecordVecBacking(*dst_node, *src_node);

    OperationNode op_node;
    op_node.op_id = op_id;
    op_node.op_name = rhs->op_->name_;
    op_node.inputs = {src_name};
    op_node.output = dst_node->name;
    op_node.invocations = InvocationCount(model_.expected_uses, loop_multiplier_);
    auto has_adm = [](const ValueNode* node, const LayoutSpec& lay) {
      return std::find(node->admissible_layouts.begin(), node->admissible_layouts.end(), lay) !=
             node->admissible_layouts.end();
    };
    const std::pair<LayoutSpec, LayoutSpec> pairs[] = {
        {LayoutSpec::FractalNZ(), LayoutSpec::FractalZN()},
        {LayoutSpec::FractalZN(), LayoutSpec::FractalNZ()},
    };
    for (const auto& [s_lay, d_lay] : pairs) {
      if (has_adm(src_node, s_lay) && has_adm(dst_node, d_lay)) {
        op_node.configs.push_back(OperatorConfig{
            op_id,
            rhs->op_->name_,
            {s_lay},
            d_lay,
            0.0,
            "Zero-copy metadata view (" + s_lay.ShortName() + " -> " + d_lay.ShortName() + ")"});
      }
    }
    if (op_node.configs.empty()) {
      op_node.configs.push_back(OperatorConfig{op_id,
                                               rhs->op_->name_,
                                               {src_node->current_layout},
                                               dst_node->current_layout,
                                               0.0,
                                               "Zero-copy metadata view (" +
                                                   src_node->current_layout.ShortName() + " -> " +
                                                   dst_node->current_layout.ShortName() + ")"});
    }
    AddOperation(std::move(op_node));
  }

  void ExtractMove(const AssignStmtPtr& op, const CallPtr& rhs) {
    if (rhs->args_.empty()) return;
    auto src_arg = AsVarLike(rhs->args_[0]);
    if (!src_arg || var_to_name_.count(src_arg.get()) == 0) return;
    const std::string src_name = var_to_name_.at(src_arg.get());
    const std::string op_id = MakeOpId(op->var_->name_hint_, "move");
    const ValueNode* dst_node = RegisterTileVar(op->var_, op_id, {src_name});
    const ValueNode* src_node = instance_.FindValue(src_name);
    if (!src_node || !dst_node) return;

    OperationNode op_node;
    op_node.op_id = op_id;
    op_node.op_name = rhs->op_->name_;
    op_node.inputs = {src_name};
    op_node.output = dst_node->name;
    op_node.invocations = InvocationCount(model_.expected_uses, loop_multiplier_);
    const bool packed = src_node->dtype.IsPackedFp4() || dst_node->dtype.IsPackedFp4();
    for (const auto& s_lay : src_node->admissible_layouts) {
      if ((s_lay == LayoutSpec::RowBcastFlat() || (!packed && s_lay == LayoutSpec::FractalNZ())) &&
          dst_node->memory_space == MemorySpace::Vec) {
        continue;
      }
      for (const auto& d_lay : dst_node->admissible_layouts) {
        if ((d_lay == LayoutSpec::RowBcastFlat() || (!packed && d_lay == LayoutSpec::FractalNZ())) &&
            dst_node->memory_space == MemorySpace::Vec) {
          continue;
        }
        if (!model_.IsValidMoveConfig(*src_node, *dst_node, s_lay, d_lay)) {
          continue;
        }
        const double cost = model_.EstimateMoveCost(*src_node, *dst_node, s_lay, d_lay);
        op_node.configs.push_back(OperatorConfig{op_id,
                                                 rhs->op_->name_,
                                                 {s_lay},
                                                 d_lay,
                                                 cost,
                                                 "MTE1 " + MemorySpaceToString(src_node->memory_space) +
                                                     "->" + MemorySpaceToString(dst_node->memory_space)});
      }
    }
    AddOperation(std::move(op_node));
  }

  void ExtractMatmul(const AssignStmtPtr& op, const CallPtr& rhs) {
    if (rhs->args_.size() < 2) return;
    const size_t lhs_idx = IsOp(rhs, "tile.matmul_acc") ? 1 : 0;
    const size_t rhs_idx = lhs_idx + 1;
    if (rhs->args_.size() <= rhs_idx) return;
    auto lhs_arg = AsVarLike(rhs->args_[lhs_idx]);
    auto rhs_arg = AsVarLike(rhs->args_[rhs_idx]);
    if (!lhs_arg || !rhs_arg || var_to_name_.count(lhs_arg.get()) == 0 ||
        var_to_name_.count(rhs_arg.get()) == 0) {
      return;
    }
    const std::string lhs_name = var_to_name_.at(lhs_arg.get());
    const std::string rhs_name = var_to_name_.at(rhs_arg.get());
    const std::string op_id = MakeOpId(op->var_->name_hint_, "matmul");
    const ValueNode* dst_node = RegisterTileVar(op->var_, op_id, {lhs_name, rhs_name});
    const ValueNode* lhs_node = instance_.FindValue(lhs_name);
    const ValueNode* rhs_node = instance_.FindValue(rhs_name);
    if (!lhs_node || !rhs_node || !dst_node) return;

    const double cost = model_.EstimateMatmulCost(*lhs_node, *rhs_node, *dst_node);
    const bool lhs_is_fractal = lhs_node->current_layout == LayoutSpec::FractalNZ() ||
                                lhs_node->current_layout == LayoutSpec::FractalZN();
    const bool rhs_is_fractal = rhs_node->current_layout == LayoutSpec::FractalNZ() ||
                                rhs_node->current_layout == LayoutSpec::FractalZN();
    const LayoutSpec req_lhs = lhs_is_fractal ? LayoutSpec::FractalNZ() : lhs_node->current_layout;
    const LayoutSpec req_rhs = rhs_is_fractal ? LayoutSpec::FractalZN() : rhs_node->current_layout;
    const LayoutSpec req_dst = dst_node->current_layout;
    OperationNode op_node;
    op_node.op_id = op_id;
    op_node.op_name = rhs->op_->name_;
    op_node.inputs = {lhs_name, rhs_name};
    op_node.output = dst_node->name;
    op_node.invocations = InvocationCount(model_.expected_uses, loop_multiplier_);
    op_node.configs = {OperatorConfig{op_id,
                                      rhs->op_->name_,
                                      {req_lhs, req_rhs},
                                      req_dst,
                                      cost,
                                      "L0 Cube MAD (Left[" + req_lhs.ShortName() + "] x Right[" +
                                          req_rhs.ShortName() + "] -> Acc[" + req_dst.ShortName() + "])"}};
    AddOperation(std::move(op_node));
  }

  void ExtractStore(const CallPtr& rhs) {
    if (rhs->args_.size() < 3) return;
    auto tile_arg = AsVarLike(rhs->args_[0]);
    auto dst_arg = AsVarLike(rhs->args_[2]);
    if (!tile_arg || !dst_arg || var_to_name_.count(tile_arg.get()) == 0 ||
        var_to_name_.count(dst_arg.get()) == 0) {
      return;
    }
    const std::string tile_name = var_to_name_.at(tile_arg.get());
    const std::string dst_name = var_to_name_.at(dst_arg.get());
    const ValueNode* src_node_ptr = instance_.FindValue(tile_name);
    const ValueNode* dst_node_ptr = instance_.FindValue(dst_name);
    if (!src_node_ptr || !dst_node_ptr) return;
    const ValueNode src_node = *src_node_ptr;
    const ValueNode dst_node = *dst_node_ptr;

    consumed_values_.insert(tile_name);
    const std::string op_id = MakeOpId(dst_name, "store");
    const bool is_additional_store =
        dst_node.producer_op_id.has_value() || consumed_values_.count(dst_name) != 0;
    const bool is_vec = src_node.memory_space == MemorySpace::Vec;
    const auto facts = nz_fact_ctx_.Facts();

    std::vector<OperatorConfig> configs;
    for (const auto& s_lay : src_node.admissible_layouts) {
      for (const auto& d_lay : dst_node.admissible_layouts) {
        if (is_vec) {
          if (!model_.IsValidVecStoreConfig(src_node, dst_node, s_lay, d_lay)) {
            continue;
          }
          if (!src_node.dtype.IsPackedFp4() && d_lay == LayoutSpec::NZ() &&
              !IsNzCompatibleVecTileStoreCall(rhs, dst_arg->GetType(), facts)) {
            continue;
          }
        } else {
          if (d_lay == LayoutSpec::NZ() && !model_.SupportsAccNzStore(src_node, dst_node)) {
            continue;
          }
          if (d_lay == LayoutSpec::DN()) {
            continue;
          }
        }
        const double cost = model_.EstimateStoreCost(src_node, dst_node, d_lay, s_lay);
        const std::string pipe = is_vec ? "MTE3" : "FIXPIPE";
        configs.push_back(OperatorConfig{
            op_id, rhs->op_->name_,
            is_additional_store ? std::vector<LayoutSpec>{s_lay, d_lay} : std::vector<LayoutSpec>{s_lay},
            is_additional_store ? s_lay : d_lay, cost,
            pipe + " " + MemorySpaceToString(src_node.memory_space) + "->GM(" + d_lay.ShortName() + ")"});
      }
    }

    std::vector<LayoutSpec> supported_out_layouts;
    for (const auto& d_lay : dst_node.admissible_layouts) {
      const bool supported = std::any_of(configs.begin(), configs.end(), [&](const OperatorConfig& c) {
        return (is_additional_store ? c.input_layouts[1] : c.output_layout) == d_lay;
      });
      if (supported) {
        supported_out_layouts.push_back(d_lay);
      }
    }

    if (is_additional_store) {
      consumed_values_.insert(dst_name);
      const std::string sink_name = FreshValueName(tile_name + "_stored_" + std::to_string(op_counter_));
      ValueNode sink;
      sink.name = sink_name;
      sink.shape = src_node.shape;
      sink.dtype = src_node.dtype;
      sink.memory_space = src_node.memory_space;
      sink.current_layout = src_node.current_layout;
      sink.admissible_layouts = src_node.admissible_layouts;
      sink.producer_op_id = op_id;
      instance_.AddValue(std::move(sink));
      if (!supported_out_layouts.empty() && !model_.allow_external_output_relayout) {
        ValueNode updated_dst = dst_node;
        updated_dst.admissible_layouts = supported_out_layouts;
        instance_.AddValue(std::move(updated_dst));
      }
      AddOperation(OperationNode{op_id,
                                 rhs->op_->name_,
                                 {tile_name, dst_name},
                                 sink_name,
                                 InvocationCount(model_.expected_uses, loop_multiplier_),
                                 std::move(configs)});
      return;
    }

    ValueNode updated_dst = dst_node;
    if (!supported_out_layouts.empty() && !model_.allow_external_output_relayout) {
      updated_dst.admissible_layouts = std::move(supported_out_layouts);
    }
    updated_dst.producer_op_id = op_id;
    instance_.AddValue(std::move(updated_dst));
    AddOperation(OperationNode{op_id,
                               rhs->op_->name_,
                               {tile_name},
                               dst_name,
                               InvocationCount(model_.expected_uses, loop_multiplier_),
                               std::move(configs)});
  }

  std::vector<OperatorConfig> BuildPointwiseVecConfigs(
      const std::string& op_id, const std::string& op_name, const CallPtr& rhs,
      const std::vector<std::tuple<size_t, std::string, const ValueNode*>>& tile_inputs,
      const ValueNode& dst_node) {
    const bool nz_pointwise_ok = CanUseVecNzPointwise(rhs);

    std::vector<std::vector<LayoutSpec>> allowed_per_input;
    allowed_per_input.reserve(tile_inputs.size());
    bool any_row_major_req = false;
    for (const auto& [idx, _, in_node] : tile_inputs) {
      if (IsTileWorkspace(rhs, idx)) {
        allowed_per_input.push_back({in_node->current_layout});
        continue;
      }
      auto req = RequiredInputTileLayout(backend_, op_name, idx);
      std::vector<LayoutSpec> base_adm;
      for (const auto& lay : in_node->admissible_layouts) {
        if (lay != LayoutSpec::RowBcastFlat() && (lay != LayoutSpec::FractalNZ() || nz_pointwise_ok)) {
          base_adm.push_back(lay);
        }
      }
      if (req.has_value()) {
        if (*req == TileLayout::row_major) {
          any_row_major_req = true;
        }
        std::vector<LayoutSpec> filtered;
        for (const auto& lay : base_adm) {
          if (lay.IsTile() && (lay.blayout == *req || (nz_pointwise_ok && lay == LayoutSpec::FractalNZ()))) {
            filtered.push_back(lay);
          }
        }
        if (filtered.empty()) filtered.push_back(in_node->current_layout);
        allowed_per_input.push_back(std::move(filtered));
      } else if (nz_pointwise_ok &&
                 std::find(base_adm.begin(), base_adm.end(), LayoutSpec::FractalNZ()) != base_adm.end()) {
        allowed_per_input.push_back({in_node->current_layout, LayoutSpec::FractalNZ()});
      } else {
        allowed_per_input.push_back({in_node->current_layout});
      }
    }

    std::vector<LayoutSpec> dst_base_adm;
    for (const auto& lay : dst_node.admissible_layouts) {
      if (lay != LayoutSpec::RowBcastFlat() && (lay != LayoutSpec::FractalNZ() || nz_pointwise_ok)) {
        dst_base_adm.push_back(lay);
      }
    }
    std::vector<LayoutSpec> allowed_outputs;
    if (any_row_major_req) {
      for (const auto& lay : dst_base_adm) {
        if (lay.IsTile() &&
            (lay.blayout == TileLayout::row_major || (nz_pointwise_ok && lay == LayoutSpec::FractalNZ()))) {
          allowed_outputs.push_back(lay);
        }
      }
      if (allowed_outputs.empty()) allowed_outputs.push_back(dst_node.current_layout);
    } else if (nz_pointwise_ok && std::find(dst_base_adm.begin(), dst_base_adm.end(),
                                            LayoutSpec::FractalNZ()) != dst_base_adm.end()) {
      allowed_outputs = {dst_node.current_layout, LayoutSpec::FractalNZ()};
    } else {
      allowed_outputs = {dst_node.current_layout};
    }

    int64_t work_elements = dst_node.NumElements();
    for (const auto& [_, _n, in_node] : tile_inputs) {
      work_elements = std::max(work_elements, in_node->NumElements());
    }
    const double cost = static_cast<double>(work_elements) / model_.vec_elements_per_cycle;

    std::vector<OperatorConfig> configs;
    if (tile_inputs.empty()) {
      for (const auto& out_lay : allowed_outputs) {
        configs.push_back(OperatorConfig{op_id, op_name, {}, out_lay, cost, "Backend-validated " + op_name});
      }
    } else {
      // Uniform layout configurations across inputs and output.
      for (const auto& out_lay : allowed_outputs) {
        std::vector<LayoutSpec> inputs;
        for (size_t i = 0; i < tile_inputs.size(); ++i) {
          const auto& [position, _, node] = tile_inputs[i];
          const auto& required = IsTileWorkspace(rhs, position) ? node->current_layout : out_lay;
          const auto& allowed = allowed_per_input[i];
          if (std::find(allowed.begin(), allowed.end(), required) == allowed.end()) break;
          inputs.push_back(required);
        }
        if (inputs.size() == tile_inputs.size()) {
          configs.push_back(OperatorConfig{op_id, op_name, std::move(inputs), out_lay, cost,
                                           "Backend-validated " + op_name});
        }
      }
    }
    if (configs.empty()) {
      std::vector<LayoutSpec> def_ins;
      def_ins.reserve(tile_inputs.size());
      for (const auto& [_, _n, in_node] : tile_inputs) {
        def_ins.push_back(in_node->current_layout);
      }
      configs.push_back(OperatorConfig{op_id, op_name, std::move(def_ins), dst_node.current_layout, cost,
                                       "Backend-validated " + op_name});
    }
    return configs;
  }

  static bool IsStridedSubview(const VarPtr& result, const CallPtr& call, const ValueNode& source) {
    const auto tile_type = As<TileType>(result->GetType());
    if (tile_type->GetMemorySpace().value_or(MemorySpace::Vec) != MemorySpace::Vec) {
      return false;
    }
    if (!IsOp(call, "tile.slice")) {
      // These operations reuse the first operand's storage. A new logical
      // descriptor alone cannot make a strided window a compact allocation.
      return source.is_strided_subview &&
             (IsOp(call, "tile.reshape") || IsOp(call, "tile.reinterpret_view") ||
              IsOp(call, "tile.set_validshape") || IsOp(call, "tile.assemble"));
    }
    if (source.is_strided_subview) {
      return true;
    }
    const auto shape = StaticShape(tile_type->shape_);
    if (!shape || shape->size() != 2 || source.shape.size() != 2) {
      return true;
    }
    // PTOResolveBufferSelect uses the child physical shape only when the
    // child and parent have identical pointer strides. For flat tiles that
    // means retaining the full width (row-major) or height (column-major).
    if (source.current_layout.slayout == TileLayout::none_box) {
      const size_t pitch_dim = source.current_layout.blayout == TileLayout::col_major ? 0 : 1;
      return (*shape)[pitch_dim] != source.shape[pitch_dim];
    }
    return *shape != source.shape;
  }

  void ExtractGenericTileOp(const AssignStmtPtr& op, const CallPtr& rhs) {
    if (!As<TileType>(op->var_->GetType())) return;
    const std::string& op_name = rhs->op_->name_;

    std::vector<std::tuple<size_t, std::string, const ValueNode*>> tile_inputs;
    for (const auto& [idx, arg] : TileOperands(rhs)) {
      if (var_to_name_.count(arg.get()) == 0) {
        RegisterTileVar(arg, std::nullopt);
      }
      auto it = var_to_name_.find(arg.get());
      if (it != var_to_name_.end()) {
        const ValueNode* in_node = instance_.FindValue(it->second);
        if (in_node != nullptr && in_node->memory_space != MemorySpace::DDR) {
          tile_inputs.emplace_back(idx, it->second, in_node);
        }
      }
    }
    if (tile_inputs.empty() && op_name != "tile.full") {
      return;
    }

    std::string sanitized = op_name;
    std::replace(sanitized.begin(), sanitized.end(), '.', '_');
    const std::string op_id = MakeOpId(op->var_->name_hint_, sanitized);
    std::vector<std::string> inp_names;
    inp_names.reserve(tile_inputs.size());
    for (const auto& [_, name, _node] : tile_inputs) {
      inp_names.push_back(name);
    }
    const auto* first_input = !tile_inputs.empty() && std::get<0>(tile_inputs.front()) == 0
                                  ? std::get<2>(tile_inputs.front())
                                  : nullptr;
    const bool is_strided_subview = first_input && IsStridedSubview(op->var_, rhs, *first_input);
    const bool is_vec_slice = IsOp(rhs, "tile.slice") || (first_input && first_input->is_vec_slice &&
                                                          detail::AliasesFirstTileOperand(rhs));
    const ValueNode* dst_node = RegisterTileVar(op->var_, op_id, inp_names, is_strided_subview, is_vec_slice);
    if (!dst_node) return;
    if (const auto backing = detail::TileBackingOperand(rhs)) {
      for (const auto& [index, _, input] : tile_inputs) {
        if (index == *backing) {
          RecordVecBacking(*dst_node, *input);
          break;
        }
      }
    }

    auto input_layouts = [&](const std::optional<LayoutSpec>& data_layout = std::nullopt) {
      std::vector<LayoutSpec> layouts;
      layouts.reserve(tile_inputs.size());
      for (const auto& [position, _, node] : tile_inputs) {
        layouts.push_back(data_layout && !IsTileWorkspace(rhs, position) ? *data_layout
                                                                         : node->current_layout);
      }
      return layouts;
    };
    std::vector<OperatorConfig> configs;
    if (detail::PreservesTileDescriptors(rhs)) {
      // Reshape pairs define how the output addresses the input's bytes.
      // Preserve those descriptors at every rank and memory space. Packed
      // casts likewise rely on their carrier geometry, and fill retains its
      // original logical dimensions.
      const double cost = IsOp(rhs, "tile.reshape")
                              ? 0.0
                              : static_cast<double>(dst_node->NumElements()) / model_.vec_elements_per_cycle;
      configs.push_back(OperatorConfig{op_id, op_name, input_layouts(), dst_node->current_layout, cost,
                                       "Original descriptors for " + op_name});
    } else if (RowReductionOps().count(op_name) != 0 || ColReductionOps().count(op_name) != 0) {
      const auto* data = std::get<2>(tile_inputs[0]);
      const double cost = detail::CanCopySingletonRowSum(rhs)
                              ? model_.EstimateMoveCost(*data, *dst_node, LayoutSpec::RowMajorFlat(),
                                                        LayoutSpec::RowMajorFlat())
                              : static_cast<double>(data->NumElements()) / model_.vec_elements_per_cycle;
      // A compact [M, 1] already encodes the original reduction axis. Preserve
      // its descriptor instead of requiring an unaligned row-major column.
      const bool singleton = data->shape.size() == 2 && (data->shape[0] == 1 || data->shape[1] == 1);
      auto inputs = singleton ? input_layouts() : input_layouts(LayoutSpec::RowMajorFlat());
      configs.push_back(OperatorConfig{op_id, op_name, std::move(inputs), dst_node->current_layout, cost,
                                       "Axis reduction " + op_name});
    } else if (op_name == "tile.row_expand_add" && tile_inputs.size() >= 2) {
      const ValueNode* tile_mn = std::get<2>(tile_inputs[0]);
      const ValueNode* row_vec = std::get<2>(tile_inputs[1]);
      const double base_cost =
          static_cast<double>(std::max(tile_mn->NumElements(), dst_node->NumElements())) /
          model_.vec_elements_per_cycle;
      auto cm_ins = input_layouts(LayoutSpec::RowMajorFlat());
      cm_ins[1] = LayoutSpec::ColMajorFlat();
      // TROWEXPANDADD consumes the compact column directly. An explicit expansion
      // is charged once by the conversion cost when consumers demand RowBcastFlat.
      configs.push_back(OperatorConfig{op_id, op_name, std::move(cm_ins), LayoutSpec::RowMajorFlat(),
                                       base_cost, "tile.row_expand_add (compact col_major)"});
      if (std::find(row_vec->admissible_layouts.begin(), row_vec->admissible_layouts.end(),
                    LayoutSpec::RowBcastFlat()) != row_vec->admissible_layouts.end()) {
        auto bcast_ins = input_layouts(LayoutSpec::RowMajorFlat());
        bcast_ins[1] = LayoutSpec::RowBcastFlat();
        configs.push_back(OperatorConfig{op_id, op_name, std::move(bcast_ins), LayoutSpec::RowMajorFlat(),
                                         base_cost, "tile.row_expand_add (pre-expanded [M, c0] row_major)"});
      }
    } else if (OtherRowExpandOps().count(op_name) != 0 && tile_inputs.size() >= 2) {
      const ValueNode* tile_mn = std::get<2>(tile_inputs[0]);
      const int64_t m_dim = tile_mn->shape.size() >= 2 ? tile_mn->shape[tile_mn->shape.size() - 2] : 1;
      const double base_cost =
          static_cast<double>(std::max(tile_mn->NumElements(), dst_node->NumElements())) /
          model_.vec_elements_per_cycle;
      auto cm_ins = input_layouts(LayoutSpec::RowMajorFlat());
      cm_ins[1] = LayoutSpec::ColMajorFlat();
      configs.push_back(OperatorConfig{op_id, op_name, std::move(cm_ins), LayoutSpec::RowMajorFlat(),
                                       base_cost + model_.EstimateVbrcbCost(m_dim),
                                       "Row expand " + op_name + " (compact col_major + vbrcb)"});
    } else if (ColExpandOps().count(op_name) != 0 && tile_inputs.size() >= 2) {
      const ValueNode* tile_mn = std::get<2>(tile_inputs[0]);
      const double base_cost =
          static_cast<double>(std::max(tile_mn->NumElements(), dst_node->NumElements())) /
          model_.vec_elements_per_cycle;
      configs.push_back(OperatorConfig{op_id, op_name, input_layouts(LayoutSpec::RowMajorFlat()),
                                       LayoutSpec::RowMajorFlat(), base_cost, "Col expand " + op_name});
    } else {
      configs = BuildPointwiseVecConfigs(op_id, op_name, rhs, tile_inputs, *dst_node);
    }

    AddOperation(OperationNode{op_id, op_name, std::move(inp_names), dst_node->name,
                               InvocationCount(model_.expected_uses, loop_multiplier_), std::move(configs)});
  }

  void WalkAssignStmt(const AssignStmtPtr& op) {
    if (As<ScalarType>(op->var_->GetType())) {
      scalar_defs_[op->var_.get()] = op->value_;
    }
    if (auto rhs = As<Call>(op->value_)) {
      const std::string& op_name = rhs->op_->name_;
      if (IsOp(rhs, "tile.create")) {
        ++vec_position_;
        RegisterTileVar(op->var_, std::nullopt);
        ++vec_position_;
      } else if (IsOp(rhs, "tile.load")) {
        ExtractLoad(op, rhs);
      } else if (IsOp(rhs, "tile.transpose_view")) {
        ExtractTransposeView(op, rhs);
      } else if (IsOp(rhs, "tile.move") ||
                 (IsOp(rhs, "tile.extract") && As<TileType>(op->var_->GetType()) &&
                  (As<TileType>(op->var_->GetType())->GetMemorySpace() == MemorySpace::Left ||
                   As<TileType>(op->var_->GetType())->GetMemorySpace() == MemorySpace::Right))) {
        ExtractMove(op, rhs);
      } else if (IsTileMatmulOp(op_name)) {
        ExtractMatmul(op, rhs);
      } else if (IsOp(rhs, "tile.store")) {
        ExtractStore(rhs);
      } else if (op_name.rfind("tile.", 0) == 0) {
        ExtractGenericTileOp(op, rhs);
      }
    } else if (auto rv = AsVarLike(op->value_)) {
      auto it = var_to_name_.find(rv.get());
      if (!AsTensorTypeLike(op->var_->GetType()) && it != var_to_name_.end()) {
        var_to_name_[op->var_.get()] = it->second;
      } else if (auto allocation = unmodeled_vec_vars_.find(rv.get());
                 allocation != unmodeled_vec_vars_.end()) {
        unmodeled_vec_vars_.emplace(op->var_.get(), allocation->second);
      }
    }
  }

  const ValueNode* LookupTileValue(const Var* var) const {
    auto it = var_to_name_.find(var);
    return it == var_to_name_.end() ? nullptr : instance_.FindValue(it->second);
  }

  void RecordOtherVecUse(const ExprPtr& expression, size_t position) {
    if (const auto var = AsVarLike(expression)) {
      if (const auto* value = LookupTileValue(var.get()); value && value->memory_space == MemorySpace::Vec) {
        vec_other_uses_[value->name] = position;
      } else if (auto it = unmodeled_vec_vars_.find(var.get()); it != unmodeled_vec_vars_.end()) {
        auto& tile = unmodeled_vec_tiles_[it->second];
        tile.last = position;
        if (tile.registered_name) vec_other_uses_[*tile.registered_name] = position;
      }
    } else if (const auto call = As<Call>(expression)) {
      if (IsOp(call, "system.reserve_buffer")) has_reserve_buffer_ = true;
      for (const auto& arg : call->args_) RecordOtherVecUse(arg, position);
    } else if (const auto submit = As<Submit>(expression)) {
      for (const auto& arg : submit->args_) RecordOtherVecUse(arg, position);
    } else if (const auto tuple = As<MakeTuple>(expression)) {
      for (const auto& element : tuple->elements_) RecordOtherVecUse(element, position);
    }
  }

  void WalkEffect(const ExprPtr& expr) {
    const size_t previous = vec_position_;
    if (auto call = As<Call>(expr); call && IsOp(call, "tile.store")) {
      ExtractStore(call);
    }
    if (vec_position_ == previous) RecordOtherVecUse(expr, ++vec_position_);
  }

  const ValueNode* AddVecLoopTransfer(const VarPtr& dst, const ValueNode* src, const ValueNode* initial,
                                      const char* op_name, const LayoutSpec& carried, int64_t use_invocations,
                                      int64_t result_invocations, bool may_execute_body) {
    const std::string op_id = MakeOpId(dst->name_hint_, op_name);
    std::vector<std::string> inputs;
    std::vector<LayoutSpec> layouts;
    if (src) {
      inputs.push_back(src->name);
      layouts.push_back(carried);
    }
    const auto* node = RegisterTileVar(
        dst, op_id, inputs, (src && src->is_strided_subview) || (initial && initial->is_strided_subview),
        (src && src->is_vec_slice) || (initial && initial->is_vec_slice));
    if (!node) return nullptr;
    instance_.values.at(node->name).conversion_invocations = result_invocations;
    if (src && may_execute_body) RecordVecBacking(*node, *src);
    if (initial) RecordVecBacking(*node, *initial);
    AddOperation(OperationNode{op_id,
                               op_name,
                               std::move(inputs),
                               node->name,
                               use_invocations,
                               {OperatorConfig{op_id, op_name, std::move(layouts), carried, 0.0}}});
    return node;
  }

  void WalkLoop(const std::vector<IterArgPtr>& iter_args, const StmtPtr& body,
                const std::vector<VarPtr>& return_vars, int64_t trip_count, bool may_repeat) {
    const int64_t outer_mult = loop_multiplier_;
    const int64_t body_mult = InvocationCount(outer_mult, trip_count);
    const int64_t entry_count = InvocationCount(model_.expected_uses, outer_mult);
    const int64_t body_count = InvocationCount(model_.expected_uses, body_mult);
    // The same declared representation must serve init, every actual backedge,
    // and the zero-iteration exit. Alternate copies have distinct lifetimes
    // and activation counts on the two sides of the loop boundary.
    std::vector<const ValueNode*> vec_iterations(iter_args.size(), nullptr);
    auto backedge = transform_utils::GetLastYieldStmt(body);
    const auto parent_scope = vec_loop_scope_;
    const size_t loop_index = vec_memory_.loops.size();
    if (may_repeat) {
      vec_loop_scope_ = loop_index;
      vec_memory_.loops.push_back({++vec_position_, 0, parent_scope});
    }
    for (size_t i = 0; i < iter_args.size(); ++i) {
      const auto& ia = iter_args[i];
      auto type = As<TileType>(ia->GetType());
      if (!type) continue;
      auto init = AsVarLike(ia->initValue_);
      const auto* init_node = LookupTileValue(init.get());
      const auto result_type = i < return_vars.size() ? As<TileType>(return_vars[i]->GetType()) : nullptr;
      if (type->GetMemorySpace().value_or(MemorySpace::Vec) == MemorySpace::Vec &&
          detail::StaticShape(type->shape_) && result_type &&
          result_type->GetMemorySpace().value_or(MemorySpace::Vec) == MemorySpace::Vec &&
          detail::StaticShape(result_type->shape_) && backedge && i < backedge->value_.size()) {
        vec_iterations[i] =
            AddVecLoopTransfer(ia, init_node, nullptr, kVecLoopInitOp, EffectiveTileLayout(type), entry_count,
                               body_count, /*may_execute_body=*/true);
      } else if (init_node) {
        var_to_name_[ia.get()] = init_node->name;
      }
    }
    yield_stack_.emplace_back();
    loop_multiplier_ = body_mult;
    WalkStmt(body);
    if (may_repeat) {
      vec_memory_.loops[loop_index].end = vec_position_;
      vec_loop_scope_ = parent_scope;
    }
    loop_multiplier_ = outer_mult;
    auto yielded_vars = std::move(yield_stack_.back());
    yield_stack_.pop_back();

    for (size_t i = 0; i < return_vars.size(); ++i) {
      const auto& result = return_vars[i];
      if (!As<TileType>(result->GetType())) continue;
      const auto* yielded = i < yielded_vars.size() ? LookupTileValue(yielded_vars[i]) : nullptr;
      const auto* iter = i < iter_args.size() ? LookupTileValue(iter_args[i].get()) : nullptr;
      if (i < vec_iterations.size() && vec_iterations[i]) {
        const auto* iteration = vec_iterations[i];
        AddVecLoopTransfer(result, yielded, iteration, kVecLoopYieldOp, iteration->current_layout, body_count,
                           entry_count, /*may_execute_body=*/trip_count != 0);
      } else if (const auto* source = iter ? iter : yielded) {
        var_to_name_[result.get()] = source->name;
      }
    }
  }

  void WalkStmt(const StmtPtr& stmt) {
    if (!stmt) return;
    if (auto as = As<AssignStmt>(stmt)) {
      const size_t previous_position = vec_position_;
      WalkAssignStmt(as);
      if (vec_position_ == previous_position && !AsVarLike(as->value_)) {
        RecordOtherVecUse(as->value_, ++vec_position_);
        const auto tile = As<TileType>(as->var_->GetType());
        if (tile && tile->GetMemorySpace().value_or(MemorySpace::Vec) == MemorySpace::Vec &&
            !LookupTileValue(as->var_.get())) {
          // Account for static Vec results even when the operation has no layout factor (e.g. tile.tri).
          if (const auto shape = StaticShape(tile->shape_)) {
            ValueNode node;
            node.shape = *shape;
            node.dtype = tile->dtype_;
            unmodeled_vec_vars_.emplace(as->var_.get(), unmodeled_vec_tiles_.size());
            unmodeled_vec_tiles_.push_back({vec_position_, vec_position_, node.ByteSize(), std::nullopt});
            ++vec_position_;  // A later modeled consumer can request a copy after this definition.
          }
        }
      }
    } else if (auto fs = As<ForStmt>(stmt)) {
      const auto trips = StaticInCoreForTripCount(fs, scalar_defs_);
      WalkLoop(fs->iter_args_, fs->body_, fs->return_vars_, trips ? *trips : ForTripCount(fs, scalar_defs_),
               !trips || *trips > 1);
      return;
    } else if (auto ws = As<WhileStmt>(stmt)) {
      const auto condition = As<ConstBool>(ws->condition_);
      const bool may_execute = !condition || condition->value_;
      WalkLoop(ws->iter_args_, ws->body_, ws->return_vars_, may_execute ? 1 : 0, may_execute);
      return;
    } else if (auto ys = As<YieldStmt>(stmt)) {
      if (!yield_stack_.empty()) {
        std::vector<const Var*> yvars;
        yvars.reserve(ys->value_.size());
        for (const auto& v : ys->value_) {
          auto vl = AsVarLike(v);
          yvars.push_back(vl ? vl.get() : nullptr);
        }
        yield_stack_.back() = std::move(yvars);
      }
    } else if (auto es = As<EvalStmt>(stmt)) {
      WalkEffect(es->expr_);
    } else if (auto rs = As<ReturnStmt>(stmt)) {
      for (const auto& val : rs->value_) WalkEffect(val);
    } else if (auto ifs = As<IfStmt>(stmt)) {
      yield_stack_.emplace_back();
      WalkStmt(ifs->then_body_);
      std::vector<const Var*> then_y = std::move(yield_stack_.back());
      yield_stack_.pop_back();

      std::vector<const Var*> else_y;
      const bool has_else = ifs->else_body_.has_value() && *ifs->else_body_;
      if (has_else) {
        yield_stack_.emplace_back();
        WalkStmt(*ifs->else_body_);
        else_y = std::move(yield_stack_.back());
        yield_stack_.pop_back();
      }
      for (size_t i = 0; i < ifs->return_vars_.size(); ++i) {
        const auto& rv = ifs->return_vars_[i];
        if (AsTensorTypeLike(rv->GetType())) continue;
        const Var* tv = i < then_y.size() ? then_y[i] : nullptr;
        const Var* ev = i < else_y.size() ? else_y[i] : nullptr;
        auto it_t = (tv != nullptr) ? var_to_name_.find(tv) : var_to_name_.end();
        auto it_e = (ev != nullptr) ? var_to_name_.find(ev) : var_to_name_.end();
        const bool has_t = it_t != var_to_name_.end();
        const bool has_e = it_e != var_to_name_.end();
        if (has_t && has_e && it_t->second == it_e->second) {
          var_to_name_[rv.get()] = it_t->second;
          continue;
        }
        if (has_t && has_e) {
          const std::string then_name = it_t->second;
          const std::string else_name = it_e->second;
          const ValueNode* then_node_ptr = instance_.FindValue(then_name);
          const ValueNode* else_node_ptr = instance_.FindValue(else_name);
          if (then_node_ptr == nullptr || else_node_ptr == nullptr) {
            continue;
          }
          const ValueNode then_node = *then_node_ptr;
          const ValueNode else_node = *else_node_ptr;
          const std::string op_id = MakeOpId(rv->name_hint_, "if_join");
          const auto* r_node_ptr = RegisterTileVar(
              rv, op_id, {then_name, else_name}, then_node.is_strided_subview || else_node.is_strided_subview,
              then_node.is_vec_slice || else_node.is_vec_slice);
          if (r_node_ptr == nullptr) {
            continue;
          }
          RecordVecBacking(*r_node_ptr, then_node);
          RecordVecBacking(*r_node_ptr, else_node);
          ValueNode r_node = *r_node_ptr;
          std::vector<LayoutSpec> common;
          for (const auto& lay : then_node.admissible_layouts) {
            if (lay == LayoutSpec::RowBcastFlat()) continue;
            const bool in_else =
                std::find(else_node.admissible_layouts.begin(), else_node.admissible_layouts.end(), lay) !=
                else_node.admissible_layouts.end();
            const bool in_r = std::find(r_node.admissible_layouts.begin(), r_node.admissible_layouts.end(),
                                        lay) != r_node.admissible_layouts.end();
            if (in_else && in_r) {
              common.push_back(lay);
            }
          }
          std::vector<OperatorConfig> join_configs;
          if (!common.empty()) {
            r_node.admissible_layouts = common;
            instance_.AddValue(r_node);
            for (const auto& lay : common) {
              join_configs.push_back(OperatorConfig{op_id,
                                                    "control.if_join",
                                                    {lay, lay},
                                                    lay,
                                                    0.0,
                                                    "Conditional if_join (" + lay.ShortName() + ")"});
            }
          } else {
            ValueNode pinned_t = then_node;
            pinned_t.admissible_layouts = {then_node.current_layout};
            instance_.AddValue(std::move(pinned_t));
            ValueNode pinned_e = else_node;
            pinned_e.admissible_layouts = {else_node.current_layout};
            instance_.AddValue(std::move(pinned_e));
            r_node.admissible_layouts = {r_node.current_layout};
            instance_.AddValue(r_node);
            join_configs.push_back(OperatorConfig{op_id,
                                                  "control.if_join",
                                                  {then_node.current_layout, else_node.current_layout},
                                                  r_node.current_layout,
                                                  0.0,
                                                  "Conditional if_join (pinned baseline)"});
          }
          AddOperation(OperationNode{op_id,
                                     "control.if_join",
                                     {then_name, else_name},
                                     r_node.name,
                                     InvocationCount(model_.expected_uses, loop_multiplier_),
                                     std::move(join_configs)});
        } else if (has_t && !has_else) {
          var_to_name_[rv.get()] = it_t->second;
        } else if (has_t != has_e) {
          const std::string& tracked_name = has_t ? it_t->second : it_e->second;
          if (const ValueNode* tracked = instance_.FindValue(tracked_name)) {
            ValueNode pinned = *tracked;
            pinned.admissible_layouts = {tracked->current_layout};
            instance_.AddValue(std::move(pinned));
          }
        }
      }
      return;
    }
    for (const auto& child : IterChildStmts(stmt)) {
      WalkStmt(child);
    }
  }

  const LayoutCostModel& model_;
  const backend::Backend* backend_;
  const std::unordered_set<std::string>& reusable_;
  const std::unordered_set<std::string>& nz_input_roots_;
  const std::unordered_map<std::string, TensorTypePtr>& host_inputs_;
  const std::unordered_map<std::string, double>& projected_dispatches_;
  const std::unordered_set<std::string>& external_out_names_;
  const std::unordered_set<std::string>& reshaped_names_;
  const std::unordered_set<std::string>& nz_ineligible_names_;
  const std::unordered_set<std::string>& dn_ineligible_names_;
  const std::unordered_set<std::string>& internal_buffers_;
  LayoutProblemInstance& instance_;
  std::unordered_set<std::string>& consumed_values_;
  size_t& op_counter_;
  size_t next_value_suffix_ = 0;
  std::string tile_prefix_;
  int64_t loop_multiplier_;
  const GmNames& gm_names_;
  const std::map<std::pair<std::string, size_t>, int64_t>& param_steps_;

  std::unordered_map<const Var*, std::string> var_to_name_;
  VecMemoryRegion vec_memory_{};
  std::optional<size_t> vec_loop_scope_;
  bool has_reserve_buffer_ = false;
  std::vector<UnmodeledVecTile> unmodeled_vec_tiles_;
  std::unordered_map<const Var*, size_t> unmodeled_vec_vars_;
  std::unordered_map<std::string, size_t> vec_definitions_;
  std::unordered_map<std::string, size_t> vec_other_uses_;
  size_t vec_position_ = 0;
  std::unordered_map<const Var*, ExprPtr> scalar_defs_;
  std::vector<std::vector<const Var*>> yield_stack_;
  std::unordered_set<const Var*> vec_tmp_vars_;
  std::unordered_set<const Var*> vec_non_tmp_vars_;
  std::unordered_set<const Var*> vec_row_expand_add_vars_;
  std::unordered_set<const Var*> vec_non_row_expand_consumer_vars_;
  std::unordered_set<const Var*> vec_row_reduction_vars_;
  std::unordered_set<const Var*> vec_nz_eligible_vars_;
  NzFactContext nz_fact_ctx_;
};

// Bind every caller before its callee so shared helper formals see all incoming
// storage roots and their execution counts before their body is analyzed.
class OrchestrationCallees : public IRVisitor {
 public:
  std::set<std::string> names;

 private:
  void VisitExpr_(const CallPtr& op) override {
    if (auto callee = As<GlobalVar>(op->op_)) names.insert(callee->name_);
    IRVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const SubmitPtr& op) override {
    if (auto callee = As<GlobalVar>(op->op_)) names.insert(callee->name_);
    IRVisitor::VisitExpr_(op);
  }
};

bool IsGmLayoutSupportedByOps(const std::string& val_name, const LayoutSpec& lay,
                              const std::vector<const OperationNode*>& participating) {
  for (const auto* op : participating) {
    for (size_t in_idx = 0; in_idx < op->inputs.size(); ++in_idx) {
      if (op->inputs[in_idx] == val_name) {
        const bool ok = std::any_of(op->configs.begin(), op->configs.end(), [&](const OperatorConfig& cfg) {
          return cfg.input_layouts[in_idx] == lay;
        });
        if (!ok) return false;
      }
    }
    if (op->output == val_name) {
      const bool ok = std::any_of(op->configs.begin(), op->configs.end(),
                                  [&](const OperatorConfig& cfg) { return cfg.output_layout == lay; });
      if (!ok) return false;
    }
  }
  return true;
}

void PruneUnsupportedGmAndFractalLayouts(LayoutProblemInstance& instance,
                                         const std::unordered_set<std::string>& nz_ineligible_names,
                                         const std::unordered_set<std::string>& dn_ineligible_names,
                                         const std::unordered_set<std::string>& reshaped_names) {
  std::unordered_map<std::string, std::vector<size_t>> val_to_op_indices;
  val_to_op_indices.reserve(instance.values.size());
  for (size_t op_idx = 0; op_idx < instance.operations.size(); ++op_idx) {
    const auto& op = instance.operations[op_idx];
    auto add_ref = [&](const std::string& name) {
      auto& vec = val_to_op_indices[name];
      if (vec.empty() || vec.back() != op_idx) {
        vec.push_back(op_idx);
      }
    };
    for (const auto& inp : op.inputs) {
      add_ref(inp);
    }
    add_ref(op.output);
  }

  const size_t max_iters = instance.values.size() + instance.operations.size() + 2;
  std::vector<const OperationNode*> participating;
  for (size_t iter = 0; iter < max_iters; ++iter) {
    bool gm_changed = false;
    for (const auto& val_name : instance.value_order) {
      auto& val_node = instance.values.at(val_name);
      if (val_node.memory_space != MemorySpace::DDR || val_node.admissible_layouts.size() <= 1) {
        continue;
      }
      participating.clear();
      if (auto ref_it = val_to_op_indices.find(val_name); ref_it != val_to_op_indices.end()) {
        participating.reserve(ref_it->second.size());
        for (size_t op_idx : ref_it->second) {
          participating.push_back(&instance.operations[op_idx]);
        }
      }
      std::vector<LayoutSpec> kept;
      for (const auto& lay : val_node.admissible_layouts) {
        bool keep = true;
        if (lay != val_node.current_layout) {
          if (lay == LayoutSpec::NZ()) {
            keep = nz_ineligible_names.count(val_name) == 0 && reshaped_names.count(val_name) == 0 &&
                   IsGmLayoutSupportedByOps(val_name, lay, participating);
          } else if (lay == LayoutSpec::ND()) {
            keep = !participating.empty() && IsGmLayoutSupportedByOps(val_name, lay, participating);
          } else if (lay == LayoutSpec::DN()) {
            keep = dn_ineligible_names.count(val_name) == 0 && reshaped_names.count(val_name) == 0 &&
                   !participating.empty() && IsGmLayoutSupportedByOps(val_name, lay, participating);
          }
        }
        if (keep) {
          kept.push_back(lay);
        }
      }
      if (kept.empty()) kept.push_back(val_node.current_layout);
      if (kept != val_node.admissible_layouts) {
        val_node.admissible_layouts = std::move(kept);
        gm_changed = true;
      }
    }

    for (auto& op : instance.operations) {
      std::vector<OperatorConfig> valid_cfgs;
      for (const auto& cfg : op.configs) {
        const auto* out_node = instance.FindValue(op.output);
        if (out_node == nullptr ||
            std::find(out_node->admissible_layouts.begin(), out_node->admissible_layouts.end(),
                      cfg.output_layout) == out_node->admissible_layouts.end()) {
          continue;
        }
        bool all_in = true;
        for (size_t i = 0; i < op.inputs.size(); ++i) {
          const auto* in_node = instance.FindValue(op.inputs[i]);
          if (in_node == nullptr ||
              std::find(in_node->admissible_layouts.begin(), in_node->admissible_layouts.end(),
                        cfg.input_layouts[i]) == in_node->admissible_layouts.end()) {
            all_in = false;
            break;
          }
        }
        if (all_in) {
          valid_cfgs.push_back(cfg);
        }
      }
      if (!valid_cfgs.empty()) {
        op.configs = std::move(valid_cfgs);
      }
    }

    bool vec_changed = false;
    for (const auto& val_name : instance.value_order) {
      auto& val_node = instance.values.at(val_name);
      if (val_node.memory_space != MemorySpace::Vec ||
          std::find(val_node.admissible_layouts.begin(), val_node.admissible_layouts.end(),
                    LayoutSpec::FractalNZ()) == val_node.admissible_layouts.end()) {
        continue;
      }
      bool nz_viable = true;
      if (auto ref_it = val_to_op_indices.find(val_name); ref_it != val_to_op_indices.end()) {
        for (size_t op_idx : ref_it->second) {
          const auto& op = instance.operations[op_idx];
          if (op.output == val_name) {
            const bool ok = std::any_of(op.configs.begin(), op.configs.end(), [](const OperatorConfig& c) {
              return c.output_layout == LayoutSpec::FractalNZ();
            });
            if (!ok) {
              nz_viable = false;
              break;
            }
          }
          for (size_t in_idx = 0; in_idx < op.inputs.size(); ++in_idx) {
            if (op.inputs[in_idx] == val_name) {
              const bool ok = std::any_of(op.configs.begin(), op.configs.end(), [&](const OperatorConfig& c) {
                return c.input_layouts[in_idx] == LayoutSpec::FractalNZ();
              });
              if (!ok) {
                nz_viable = false;
                break;
              }
            }
          }
          if (!nz_viable) break;
        }
      }
      if (!nz_viable) {
        std::vector<LayoutSpec> kept;
        for (const auto& lay : val_node.admissible_layouts) {
          if (lay != LayoutSpec::FractalNZ()) kept.push_back(lay);
        }
        if (kept.empty()) kept.push_back(val_node.current_layout);
        val_node.admissible_layouts = std::move(kept);
        vec_changed = true;
      }
    }

    if (!gm_changed && !vec_changed) {
      break;
    }
  }
}

void SanitizeInstanceInvariants(LayoutProblemInstance& instance) {
  for (const auto& val_name : instance.value_order) {
    auto& val = instance.values.at(val_name);
    std::vector<LayoutSpec> dedup;
    dedup.reserve(val.admissible_layouts.size() + 1);
    for (const auto& lay : val.admissible_layouts) {
      if (std::find(dedup.begin(), dedup.end(), lay) == dedup.end()) {
        dedup.push_back(lay);
      }
    }
    if (std::find(dedup.begin(), dedup.end(), val.current_layout) == dedup.end()) {
      dedup.insert(dedup.begin(), val.current_layout);
    }
    val.admissible_layouts = std::move(dedup);
  }

  for (auto& op : instance.operations) {
    const ValueNode* out_node = instance.FindValue(op.output);
    if (out_node == nullptr) {
      continue;
    }
    std::vector<OperatorConfig> valid_cfgs;
    valid_cfgs.reserve(op.configs.size());
    double min_cost = std::numeric_limits<double>::infinity();
    for (const auto& cfg : op.configs) {
      min_cost = std::min(min_cost, cfg.op_cost_cycles);
      if (cfg.input_layouts.size() != op.inputs.size()) {
        continue;
      }
      if (std::find(out_node->admissible_layouts.begin(), out_node->admissible_layouts.end(),
                    cfg.output_layout) == out_node->admissible_layouts.end()) {
        continue;
      }
      bool all_in = true;
      for (size_t i = 0; i < op.inputs.size(); ++i) {
        const ValueNode* in_node = instance.FindValue(op.inputs[i]);
        if (in_node == nullptr ||
            std::find(in_node->admissible_layouts.begin(), in_node->admissible_layouts.end(),
                      cfg.input_layouts[i]) == in_node->admissible_layouts.end()) {
          all_in = false;
          break;
        }
      }
      if (all_in) {
        valid_cfgs.push_back(cfg);
      }
    }
    if (valid_cfgs.empty()) {
      std::vector<LayoutSpec> baseline_ins;
      baseline_ins.reserve(op.inputs.size());
      for (const auto& inp_name : op.inputs) {
        const ValueNode* in_node = instance.FindValue(inp_name);
        baseline_ins.push_back(in_node != nullptr ? in_node->current_layout : LayoutSpec::RowMajorFlat());
      }
      valid_cfgs.push_back(OperatorConfig{op.op_id, op.op_name, std::move(baseline_ins),
                                          out_node->current_layout, std::isfinite(min_cost) ? min_cost : 0.0,
                                          "Sanitized baseline (" + op.op_name + ")"});
    }
    op.configs = std::move(valid_cfgs);
  }
}

// Bind loader permissions at the public entry. All downstream decisions use the
// canonical storage keys propagated through call sites and proven aliases.
struct InputLayoutRoots {
  std::unordered_set<std::string> nz;
  std::unordered_set<std::string> reusable;
  std::unordered_map<std::string, TensorTypePtr> host_inputs;
  std::string host_entry;
  std::map<size_t, std::string> host_slots;
};

InputLayoutRoots ResolveInputLayoutRoots(const ProgramPtr& program, const OrchCallAnalysis& orch,
                                         const LayoutCostModel& model,
                                         const std::unordered_set<std::string>& requested,
                                         const std::unordered_set<std::string>& reusable_names) {
  if (requested.empty()) return {{}, reusable_names, {}, {}, {}};
  if (model.allow_external_input_relayout || model.allow_external_output_relayout) {
    throw ValueError("Named NZ inputs cannot be combined with unrestricted external relayout");
  }
  FunctionPtr entry;
  for (const auto& [_, fn] : program->functions_) {
    if (fn->level_ != Level::HOST || fn->role_ != Role::Orchestrator || orch.call_counts.count(fn->name_)) {
      continue;
    }
    if (entry) throw ValueError("Named NZ inputs require a unique HOST entry function");
    entry = fn;
  }
  const bool host_entry = entry != nullptr;
  for (const auto& [_, fn] : program->functions_) {
    if (host_entry) break;
    if (program->functions_.size() == 1 ||
        (fn->func_type_ == FunctionType::Orchestration && !orch.call_counts.count(fn->name_))) {
      if (entry) throw ValueError("Named NZ inputs require a unique entry function");
      entry = fn;
    }
  }
  if (!entry) throw ValueError("Named NZ inputs require an entry function");
  InputLayoutRoots roots;
  if (host_entry) roots.host_entry = entry->name_;
  auto missing = requested;
  for (size_t i = 0; i < entry->params_.size(); ++i) {
    const auto& param = entry->params_[i];
    const auto name = CleanName(param->name_hint_);
    const auto root = orch.gm_names.find(param.get());
    if (root != orch.gm_names.end() && reusable_names.count(name)) roots.reusable.insert(root->second);
    if (!requested.count(name)) continue;
    const auto tensor = AsTensorTypeLike(param->GetType());
    if (host_entry && tensor && tensor->dtype_.IsPackedFp4()) {
      throw ValueError("Named NZ input '" + name + "' cannot select a layout for a packed FP4 bank");
    }
    if (!tensor || entry->param_directions_[i] != ParamDirection::In || root == orch.gm_names.end() ||
        orch.external_outputs.count(root->second) || (host_entry && !HasDenseLogicalInputLayout(tensor)) ||
        (tensor->tensor_view_ &&
         (tensor->tensor_view_->layout != TensorLayout::ND || !tensor->tensor_view_->stride.empty() ||
          !tensor->tensor_view_->valid_shape.empty() || tensor->tensor_view_->pad != PadValue::null))) {
      throw ValueError("Named NZ input '" + name + "' must be a read-only logical ND entry tensor");
    }
    if (!missing.erase(name)) throw ValueError("Ambiguous NZ entry input '" + name + "'");
    roots.nz.insert(root->second);
    if (host_entry) {
      roots.host_inputs.emplace(root->second, tensor);
      roots.host_slots.emplace(i, root->second);
    }
  }
  if (!missing.empty()) throw ValueError("Unknown NZ entry input '" + *missing.begin() + "'");
  return roots;
}

}  // namespace

LayoutSpec EffectiveTileLayout(const TileTypePtr& tile_type) {
  const auto space = tile_type->memory_space_;
  if (tile_type->tile_view_.has_value()) {
    const TileView& view = *tile_type->tile_view_;
    if ((space == MemorySpace::Mat || space == MemorySpace::Left) && view.slayout == TileLayout::none_box) {
      return LayoutSpec::FractalNZ();
    }
    if (space == MemorySpace::Right && view.slayout == TileLayout::none_box) {
      return LayoutSpec::FractalZN();
    }
    return LayoutSpec::Tile(view.blayout, view.slayout, static_cast<int32_t>(view.fractal));
  }
  if (space == MemorySpace::Mat || space == MemorySpace::Left) {
    return LayoutSpec::FractalNZ();
  }
  if (space == MemorySpace::Right) {
    return LayoutSpec::FractalZN();
  }
  if (space == MemorySpace::Acc) {
    return LayoutSpec::FractalAccNZ();
  }
  const TileLayout implicit = tile_view_semantics::InferImplicitTileLayoutFromShape(tile_type->shape_);
  return (implicit == TileLayout::col_major) ? LayoutSpec::ColMajorFlat() : LayoutSpec::RowMajorFlat();
}

namespace {

OrchCallAnalysis AnalyzeOrchestration(const ProgramPtr& program, bool track_named_inputs,
                                      std::optional<int64_t> host_world_size) {
  return_lineage::ScopedTraceCache trace_cache;
  const auto definitions = MakeGmDefinitionNames(program);
  std::map<std::string, FunctionPtr> orchestration;
  std::vector<FunctionPtr> top_orch;
  std::vector<FunctionPtr> sub_orch;
  std::vector<FunctionPtr> incore;
  for (const auto& [_, func] : program->functions_) {
    if (IsInCoreType(func->func_type_)) {
      incore.push_back(func);
    } else if (track_named_inputs) {
      orchestration.emplace(func->name_, func);
    } else if (func->func_type_ == FunctionType::Orchestration) {
      top_orch.push_back(func);
    } else {
      sub_orch.push_back(func);
    }
  }

  std::map<std::string, std::set<std::string>> callees;
  std::unordered_map<std::string, size_t> pending_callers;
  for (const auto& [name, fn] : orchestration) {
    OrchestrationCallees collector;
    if (fn->body_) collector.VisitStmt(fn->body_);
    for (const auto& callee : collector.names) {
      if (!orchestration.count(callee)) continue;
      callees[name].insert(callee);
      ++pending_callers[callee];
    }
  }
  std::vector<std::string> ready;
  for (const auto& [name, _] : orchestration) {
    if (!pending_callers.count(name)) ready.push_back(name);
  }
  OrchAnalyzer analyzer(program, definitions, track_named_inputs, host_world_size);
  for (size_t i = 0; i < ready.size(); ++i) {
    const auto& name = ready[i];
    analyzer.Walk(orchestration.at(name));
    auto children = callees.find(name);
    if (children == callees.end()) continue;
    for (const auto& callee : children->second) {
      if (--pending_callers.at(callee) == 0) ready.push_back(callee);
    }
  }
  INTERNAL_CHECK(ready.size() == orchestration.size()) << "AutoLayout requires acyclic function calls";
  if (!track_named_inputs) {
    // Preserve baseline call-site specialization and operation ordering when
    // the caller has offered no named layout choices.
    OrchAnalyzer top_analyzer(program, definitions);
    for (const auto& fn : top_orch) top_analyzer.Walk(fn);
    const auto top = std::move(top_analyzer).Result();
    auto order = [&](const FunctionPtr& fn) {
      const auto it = top.call_order.find(fn->name_);
      return it != top.call_order.end() ? it->second : std::numeric_limits<size_t>::max();
    };
    std::sort(sub_orch.begin(), sub_orch.end(),
              [&](const auto& a, const auto& b) { return order(a) < order(b); });
    for (const auto& fn : top_orch) analyzer.Walk(fn);
    for (const auto& fn : sub_orch) analyzer.Walk(fn);
  }
  auto analysis = std::move(analyzer).Result();
  const std::vector<std::optional<std::string>> empty;
  auto register_local_buffers = [&](const FunctionPtr& func, const GmNames& names) {
    auto walk = [&](auto& self, const StmtPtr& stmt) -> void {
      if (auto as = As<AssignStmt>(stmt)) {
        if (auto call = As<Call>(as->value_); call && IsOp(call, "tensor.create")) {
          analysis.internal_buffers.insert(names.at(as->var_.get()));
        }
      }
      for (const auto& child : IterChildStmts(stmt)) self(self, child);
    };
    walk(walk, func->body_);
  };
  for (const auto& fn : incore) {
    auto sites = analysis.call_sites.find(fn->name_);
    if (sites != analysis.call_sites.end() && !sites->second.empty()) {
      for (size_t i = 0; i < sites->second.size(); ++i) {
        auto& site = sites->second[i];
        std::string scope = sites->second.size() > 1 ? ".site." + std::to_string(i) : "";
        GmBindings bindings(program, fn, definitions, site.arg_names, scope);
        bindings.PinLayouts(analysis, track_named_inputs);
        site.gm_names = bindings.Names();
        register_local_buffers(fn, site.gm_names);
      }
      const auto& names = sites->second.front().gm_names;
      analysis.gm_names.insert(names.begin(), names.end());
    } else {
      GmBindings bindings(program, fn, definitions, empty);
      bindings.PinLayouts(analysis, track_named_inputs);
      const auto& names = bindings.Names();
      analysis.gm_names.insert(names.begin(), names.end());
      register_local_buffers(fn, names);
      bindings.RecordReturnedStorage(fn->body_, analysis.external_outputs);
      for (size_t i = 0; i < fn->params_.size(); ++i) {
        if (fn->param_directions_[i] != ParamDirection::In) {
          auto key = names.find(fn->params_[i].get());
          if (key != names.end()) analysis.external_outputs.insert(key->second);
        }
      }
    }
  }
  return analysis;
}

}  // namespace

OrchCallAnalysis AnalyzeOrchestration(const ProgramPtr& program) {
  return AnalyzeOrchestration(program, false, std::nullopt);
}

LayoutProblemInstance ExtractLayoutInstance(const ProgramPtr& program, const LayoutCostModel& cost_model,
                                            const std::unordered_set<std::string>& reusable_tensors,
                                            const std::unordered_set<std::string>& nz_inputs) {
  return ExtractLayoutInstance(program, cost_model, reusable_tensors, nz_inputs, nullptr);
}

LayoutProblemInstance ExtractLayoutInstance(const ProgramPtr& input, const LayoutCostModel& cost_model,
                                            const std::unordered_set<std::string>& reusable_tensors,
                                            const std::unordered_set<std::string>& nz_inputs,
                                            LayoutWritebackBindings* bindings) {
  cost_model.ValidateUseHorizon();
  auto program = tensor_view_semantics::RestoreLegacyBlockedNzViews(input);
  if (bindings) {
    bindings->program = program;
    bindings->sites.clear();
    bindings->host_input_roots.clear();
    bindings->host_input_entry.clear();
  }
  LayoutCostModel resolved_model = cost_model;
  if (resolved_model.host_world_size && *resolved_model.host_world_size < 1) {
    throw ValueError("AutoLayout host_world_size must be positive");
  }
  const backend::Backend* be = ResolveBackend(resolved_model);
  if (!resolved_model.backend_type.has_value() && be != nullptr) {
    resolved_model.backend_type = backend::BackendConfig::GetBackendType();
  }

  LayoutProblemInstance instance;
  instance.cost_model = resolved_model;
  instance.forbid_unmaterializable_gm_conversions = true;

  std::vector<FunctionPtr> incore_funcs;
  std::vector<FunctionPtr> non_incore_funcs;
  for (const auto& [gv, func] : program->functions_) {
    if (IsInCoreType(func->func_type_)) {
      incore_funcs.push_back(func);
    } else {
      non_incore_funcs.push_back(func);
    }
  }

  const bool has_host = std::any_of(program->functions_.begin(), program->functions_.end(),
                                    [](const auto& fn) { return fn.second->level_ == Level::HOST; });
  const bool track_host_inputs = has_host && !nz_inputs.empty();
  auto orch = AnalyzeOrchestration(program, track_host_inputs,
                                   track_host_inputs ? cost_model.host_world_size : std::nullopt);
  const auto input_roots =
      ResolveInputLayoutRoots(program, orch, resolved_model, nz_inputs, reusable_tensors);
  if (!non_incore_funcs.empty()) {
    std::stable_sort(
        incore_funcs.begin(), incore_funcs.end(), [&](const FunctionPtr& a, const FunctionPtr& b) {
          auto it_a = orch.call_order.find(a->name_);
          auto it_b = orch.call_order.find(b->name_);
          const size_t oa = it_a != orch.call_order.end() ? it_a->second : std::numeric_limits<size_t>::max();
          const size_t ob = it_b != orch.call_order.end() ? it_b->second : std::numeric_limits<size_t>::max();
          return oa < ob;
        });
  }

  auto merge_eligibility = [&](FuncGmIneligible inelig) {
    orch.nz_ineligible_names.merge(inelig.nz);
    orch.dn_ineligible_names.merge(inelig.dn);
    orch.written_storage.merge(inelig.written);
    orch.nz_restrictions.merge(inelig.nz_restrictions);
  };
  auto check_host_matrix_formals = [&](const FunctionPtr& func, const GmNames& names) {
    if (input_roots.host_inputs.empty()) return;
    for (const auto& param : func->params_) {
      const auto name = names.find(param.get());
      if (name == names.end() || !input_roots.host_inputs.count(name->second)) continue;
      const auto tensor = As<TensorType>(param->GetType());
      if (!tensor || tensor->shape_.size() != 2) {
        orch.nz_restrictions.try_emplace(name->second,
                                         "kernel '" + func->name_ + "' does not receive one complete matrix");
      }
    }
  };
  for (const auto& func : incore_funcs) {
    auto sites_it = orch.call_sites.find(func->name_);
    if (sites_it != orch.call_sites.end() && !sites_it->second.empty()) {
      for (const auto& site : sites_it->second) {
        check_host_matrix_formals(func, site.gm_names);
        merge_eligibility(CollectFuncGmIneligible(func, resolved_model, site.gm_names, orch.param_steps,
                                                  track_host_inputs));
        auto exp_gm = CollectFuncExpandsGmNames(func, site.gm_names);
        orch.nz_ineligible_names.insert(exp_gm.begin(), exp_gm.end());
        orch.reshaped_names.insert(exp_gm.begin(), exp_gm.end());
      }
    } else {
      check_host_matrix_formals(func, orch.gm_names);
      merge_eligibility(
          CollectFuncGmIneligible(func, resolved_model, orch.gm_names, orch.param_steps, track_host_inputs));
      auto exp_gm = CollectFuncExpandsGmNames(func, orch.gm_names);
      orch.nz_ineligible_names.insert(exp_gm.begin(), exp_gm.end());
      orch.reshaped_names.insert(exp_gm.begin(), exp_gm.end());
    }
  }

  for (const auto& name : input_roots.nz) {
    if (orch.written_storage.count(name)) {
      orch.nz_restrictions.try_emplace(name, "storage is written through an alias or mutable parameter");
    }
    if (orch.nz_restrictions.count(name)) {
      orch.nz_ineligible_names.insert(name);
    }
  }

  std::unordered_set<std::string> consumed_values;
  const bool multi_incore = incore_funcs.size() > 1;
  size_t op_counter = 0;

  for (const auto& func : incore_funcs) {
    auto sites_it = orch.call_sites.find(func->name_);
    if (sites_it != orch.call_sites.end() && sites_it->second.size() > 1) {
      for (size_t site_idx = 0; site_idx < sites_it->second.size(); ++site_idx) {
        const auto& site = sites_it->second[site_idx];
        const std::string prefix = detail::TileLayoutPrefix(program, func->name_, multi_incore, site_idx);
        PostInferInstanceExtractor ext(
            resolved_model, be, input_roots.reusable, input_roots.nz, input_roots.host_inputs,
            orch.projected_dispatches, orch.external_outputs, orch.reshaped_names, orch.nz_ineligible_names,
            orch.dn_ineligible_names, orch.internal_buffers, instance, consumed_values, op_counter, prefix,
            site.multiplicity, site.gm_names, orch.param_steps);
        ext.Extract(func, bindings);
      }
      continue;
    }

    const std::string prefix = detail::TileLayoutPrefix(program, func->name_, multi_incore);
    auto cnt_it = orch.call_counts.find(func->name_);
    const int64_t call_mult = (cnt_it != orch.call_counts.end()) ? cnt_it->second : 1;
    PostInferInstanceExtractor ext(resolved_model, be, input_roots.reusable, input_roots.nz,
                                   input_roots.host_inputs, orch.projected_dispatches, orch.external_outputs,
                                   orch.reshaped_names, orch.nz_ineligible_names, orch.dn_ineligible_names,
                                   orch.internal_buffers, instance, consumed_values, op_counter, prefix,
                                   call_mult, orch.gm_names, orch.param_steps);
    ext.Extract(func, bindings);
  }

  for (const auto& [name, _] : input_roots.host_inputs) {
    if (!consumed_values.count(name)) {
      orch.nz_ineligible_names.insert(name);
      orch.nz_restrictions.try_emplace(name, "no modeled device load uses this storage");
    }
  }
  PruneUnsupportedGmAndFractalLayouts(instance, orch.nz_ineligible_names, orch.dn_ineligible_names,
                                      orch.reshaped_names);
  SanitizeInstanceInvariants(instance);
  if (bindings) {
    bindings->host_input_roots = input_roots.host_slots;
    bindings->host_input_entry = input_roots.host_entry;
    std::vector<Diagnostic> hints;
    for (const auto& [name, input] : input_roots.host_inputs) {
      if (orch.unknown_rank_counts.count(name)) {
        hints.emplace_back(
            DiagnosticSeverity::PerfHint, "AutoLayout", 0, "PH-AL-002",
            "Using one modeled HOST rank for input '" + name + "' because no topology was supplied",
            program->span_);
      }
      const auto* value = instance.FindValue(name);
      if (value && value->admissible_layouts.size() == 2) continue;
      const auto restriction = orch.nz_restrictions.find(name);
      const auto shape = IsNzCompatibleTensorType(input, 4);
      std::string reason;
      if (restriction != orch.nz_restrictions.end()) {
        reason = restriction->second;
      } else if (!shape) {
        reason = "entry parameter is not an aligned, static rank-2 matrix or rank-3 or rank-4 matrix bank";
      } else if (orch.reshaped_names.count(name)) {
        reason = "a reshape or expanded tile view changes the matrix geometry";
      } else if (!value) {
        reason = "no modeled device load uses this storage";
      } else {
        reason = "the device loads have no jointly supported ND and NZ configurations";
      }
      hints.emplace_back(DiagnosticSeverity::PerfHint, "AutoLayout", 0, "PH-AL-001",
                         "Keeping the declared representation for input '" + name + "' because " + reason,
                         program->span_);
    }
    if (!hints.empty()) EmitDiagnostics(hints, "AutoLayout");
    bindings->orch = std::move(orch);
  }
  return instance;
}

}  // namespace auto_layout
}  // namespace ir
}  // namespace pypto
