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

#include <algorithm>
#include <any>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/codegen/orchestration/orchestration_analysis.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/tensor_view_semantics.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

using ::pypto::codegen::ComputeGroupEffectiveDirections;
using ::pypto::codegen::IsBuiltinOp;

enum class AccessKind { Read, Write, ReadWrite };

enum class RegionKind { Unknown, Full, Box, Conservative, Dynamic };

constexpr size_t kMaxStaticRootAlternatives = 4;
constexpr size_t kMinWindowedTaskIdTempDeps = 1;
constexpr size_t kInvalidArgIndex = std::numeric_limits<size_t>::max();
constexpr const char* kAttrAutoNoDepCandidateIndices = "__auto_no_dep_candidate_indices";
constexpr const char* kAttrAutoOutputExistingCandidateIndices = "__auto_output_existing_candidate_indices";

struct AccessRegion {
  RegionKind kind = RegionKind::Unknown;
  std::vector<int64_t> offsets;
  std::vector<int64_t> shape;
};

struct StorageAlternative {
  const Var* root = nullptr;
  AccessRegion region;
};

struct StorageLocation {
  std::vector<StorageAlternative> alternatives;
};

enum class LocationStatus { Unknown, Known, Unsupported };

struct ResolvedLocation {
  LocationStatus status = LocationStatus::Unknown;
  StorageLocation location;
};

struct StorageAccess {
  StorageAlternative location;
  AccessKind kind = AccessKind::Read;
  VarPtr task_id_var;
  bool task_id_var_is_direct = false;
  bool dynamic_producer = false;
  size_t innermost_dynamic_loop_depth = 0;
  size_t innermost_dynamic_loop_id = 0;
  size_t arg_index = kInvalidArgIndex;
};

struct AccessSummary {
  std::vector<StorageAccess> accesses;
};

struct AutoNoDepCandidateState {
  bool has_covered_hazard = false;
  bool blocked = false;
  std::string blocked_reason;
};

struct LoopCarryTaskIdMapping {
  VarPtr target;
  bool direct_task_id = false;
};

bool IsTensorType(const TypePtr& type) { return As<TensorType>(type) != nullptr; }

AccessRegion UnknownRegion() { return AccessRegion{RegionKind::Unknown, {}, {}}; }

AccessRegion FullRegion() { return AccessRegion{RegionKind::Full, {}, {}}; }

AccessRegion ConservativeRegion() { return AccessRegion{RegionKind::Conservative, {}, {}}; }

AccessRegion DynamicRegion() { return AccessRegion{RegionKind::Dynamic, {}, {}}; }

StorageLocation UnknownLocation() { return StorageLocation{}; }

StorageLocation SingleLocation(const Var* root, AccessRegion region) {
  if (!root) return UnknownLocation();
  return StorageLocation{{StorageAlternative{root, std::move(region)}}};
}

bool HasLocation(const StorageLocation& location) { return !location.alternatives.empty(); }

bool IsDynamicAccessOp(const OpPtr& op) {
  return IsOp(op, "tensor.gather") || IsOp(op, "tensor.gather_mask") || IsOp(op, "tensor.gather_compare") ||
         IsOp(op, "tensor.scatter_update") || IsOp(op, "tile.gather") || IsOp(op, "tile.gather_mask") ||
         IsOp(op, "tile.gather_compare") || IsOp(op, "tile.scatter_update") || IsOp(op, "tile.mscatter");
}

std::optional<std::vector<int64_t>> ConstIntTupleValues(const ExprPtr& expr) {
  auto tuple = As<MakeTuple>(expr);
  if (!tuple) return std::nullopt;

  std::vector<int64_t> values;
  values.reserve(tuple->elements_.size());
  for (const auto& element : tuple->elements_) {
    auto value = As<ConstInt>(element);
    if (!value) return std::nullopt;
    values.push_back(value->value_);
  }
  return values;
}

std::optional<int64_t> AddInt64(int64_t lhs, int64_t rhs) {
  if ((rhs > 0 && lhs > std::numeric_limits<int64_t>::max() - rhs) ||
      (rhs < 0 && lhs < std::numeric_limits<int64_t>::min() - rhs)) {
    return std::nullopt;
  }
  return lhs + rhs;
}

AccessRegion SliceRegion(const AccessRegion& parent, const ExprPtr& shape_expr, const ExprPtr& offset_expr) {
  auto shape = ConstIntTupleValues(shape_expr);
  auto offsets = ConstIntTupleValues(offset_expr);
  if (!shape.has_value() || !offsets.has_value() || shape->size() != offsets->size()) {
    return parent.kind == RegionKind::Unknown ? UnknownRegion() : DynamicRegion();
  }

  if (parent.kind == RegionKind::Unknown) {
    return UnknownRegion();
  }
  if (parent.kind == RegionKind::Conservative) {
    return ConservativeRegion();
  }
  if (parent.kind == RegionKind::Dynamic) {
    return DynamicRegion();
  }

  std::vector<int64_t> absolute_offsets;
  absolute_offsets.reserve(offsets->size());
  if (parent.kind == RegionKind::Full) {
    absolute_offsets = *offsets;
  } else {
    if (parent.offsets.size() != offsets->size()) return DynamicRegion();
    for (size_t i = 0; i < offsets->size(); ++i) {
      auto absolute = AddInt64(parent.offsets[i], (*offsets)[i]);
      if (!absolute.has_value()) return DynamicRegion();
      absolute_offsets.push_back(*absolute);
    }
  }

  return AccessRegion{RegionKind::Box, std::move(absolute_offsets), std::move(*shape)};
}

bool RegionsMayOverlap(const AccessRegion& lhs, const AccessRegion& rhs) {
  if (lhs.kind == RegionKind::Unknown || rhs.kind == RegionKind::Unknown) return true;
  if (lhs.kind == RegionKind::Conservative || rhs.kind == RegionKind::Conservative) return true;
  if (lhs.kind == RegionKind::Dynamic || rhs.kind == RegionKind::Dynamic) return true;
  if (lhs.kind == RegionKind::Full || rhs.kind == RegionKind::Full) return true;
  if (lhs.offsets.size() != rhs.offsets.size() || lhs.shape.size() != rhs.shape.size() ||
      lhs.offsets.size() != lhs.shape.size()) {
    return true;
  }

  for (size_t i = 0; i < lhs.offsets.size(); ++i) {
    auto lhs_end = AddInt64(lhs.offsets[i], lhs.shape[i]);
    auto rhs_end = AddInt64(rhs.offsets[i], rhs.shape[i]);
    if (!lhs_end.has_value() || !rhs_end.has_value()) return true;
    if (*lhs_end <= rhs.offsets[i] || *rhs_end <= lhs.offsets[i]) return false;
  }
  return true;
}

bool IsPreciseNoDepRegion(const AccessRegion& region) {
  return region.kind == RegionKind::Full || region.kind == RegionKind::Box ||
         region.kind == RegionKind::Conservative;
}

bool SameRegion(const AccessRegion& lhs, const AccessRegion& rhs) {
  return lhs.kind == rhs.kind && lhs.offsets == rhs.offsets && lhs.shape == rhs.shape;
}

void AppendAlternativeUnique(std::vector<StorageAlternative>* alternatives, StorageAlternative candidate) {
  if (!alternatives || !candidate.root) return;
  for (auto& existing : *alternatives) {
    if (existing.root != candidate.root) continue;
    if (!SameRegion(existing.region, candidate.region)) {
      existing.region = UnknownRegion();
    }
    return;
  }
  alternatives->push_back(std::move(candidate));
}

StorageLocation MergeLocations(const StorageLocation& lhs, const StorageLocation& rhs) {
  StorageLocation merged;
  for (const auto& alternative : lhs.alternatives) {
    AppendAlternativeUnique(&merged.alternatives, alternative);
  }
  for (const auto& alternative : rhs.alternatives) {
    AppendAlternativeUnique(&merged.alternatives, alternative);
  }
  return merged;
}

bool ExceedsRootAlternativeLimit(const StorageLocation& location) {
  return location.alternatives.size() > kMaxStaticRootAlternatives;
}

StorageLocation UnknownRegionsFor(const StorageLocation& location) {
  StorageLocation widened;
  widened.alternatives.reserve(location.alternatives.size());
  for (const auto& alternative : location.alternatives) {
    if (alternative.root) {
      widened.alternatives.push_back(StorageAlternative{alternative.root, UnknownRegion()});
    }
  }
  return widened;
}

StorageLocation ConservativeRegionsFor(const StorageLocation& location) {
  StorageLocation widened;
  widened.alternatives.reserve(location.alternatives.size());
  for (const auto& alternative : location.alternatives) {
    if (alternative.root) {
      widened.alternatives.push_back(StorageAlternative{alternative.root, ConservativeRegion()});
    }
  }
  return widened;
}

bool SameStaticConstInt(const ExprPtr& lhs, const ExprPtr& rhs) {
  auto lhs_const = As<ConstInt>(lhs);
  auto rhs_const = As<ConstInt>(rhs);
  return lhs_const && rhs_const && lhs_const->value_ == rhs_const->value_;
}

bool HasPackedNdStrides(const TensorTypePtr& tensor_type, const TensorView& view) {
  if (view.stride.empty()) return true;

  std::vector<ExprPtr> packed_strides;
  try {
    packed_strides =
        tensor_view_semantics::BuildLogicalStridesFromLayout(tensor_type->shape_, TensorLayout::ND);
  } catch (const std::exception&) {
    return false;
  }
  if (view.stride.size() != packed_strides.size()) return false;
  for (size_t i = 0; i < view.stride.size(); ++i) {
    if (!SameStaticConstInt(view.stride[i], packed_strides[i])) return false;
  }
  return true;
}

bool IsPackedNdTensorView(const TensorTypePtr& tensor_type) {
  if (!tensor_type || !tensor_type->tensor_view_.has_value()) return true;

  const auto& view = tensor_type->tensor_view_.value();
  if (view.layout != TensorLayout::ND) return false;
  if (!view.valid_shape.empty()) return false;
  if (view.pad != PadValue::null) return false;
  return HasPackedNdStrides(tensor_type, view);
}

bool IsConservativeNdTensorView(const TensorTypePtr& tensor_type) {
  if (!tensor_type || !tensor_type->tensor_view_.has_value()) return false;

  const auto& view = tensor_type->tensor_view_.value();
  if (view.layout != TensorLayout::ND) return false;
  if (view.valid_shape.empty()) return false;
  if (view.pad != PadValue::null) return false;
  return HasPackedNdStrides(tensor_type, view);
}

StorageLocation MaybeUnknownRegionsForTensorType(const StorageLocation& location, const TypePtr& type) {
  auto tensor_type = As<TensorType>(type);
  if (!tensor_type || IsPackedNdTensorView(tensor_type)) return location;
  if (IsConservativeNdTensorView(tensor_type)) return ConservativeRegionsFor(location);
  return UnknownRegionsFor(location);
}

StorageLocation MaybeConservativeRegionsForTensorType(const StorageLocation& location, const TypePtr& type) {
  auto tensor_type = As<TensorType>(type);
  if (!tensor_type || IsPackedNdTensorView(tensor_type) || IsConservativeNdTensorView(tensor_type)) {
    return ConservativeRegionsFor(location);
  }
  return UnknownRegionsFor(location);
}

StorageLocation SliceLocation(const StorageLocation& parent, const ExprPtr& shape_expr,
                              const ExprPtr& offset_expr) {
  StorageLocation sliced;
  sliced.alternatives.reserve(parent.alternatives.size());
  for (const auto& alternative : parent.alternatives) {
    AppendAlternativeUnique(
        &sliced.alternatives,
        StorageAlternative{alternative.root, SliceRegion(alternative.region, shape_expr, offset_expr)});
  }
  return sliced;
}

bool HasTaskIdTail(const TypePtr& type) {
  auto tuple_ty = As<TupleType>(type);
  if (!tuple_ty || tuple_ty->types_.empty()) return false;
  auto scalar_ty = As<ScalarType>(tuple_ty->types_.back());
  return scalar_ty && scalar_ty->dtype_ == DataType::TASK_ID;
}

bool HasTaskIdTail(const CallPtr& call) { return HasTaskIdTail(call ? call->GetType() : TypePtr{}); }

bool HasTaskIdTail(const SubmitPtr& submit) { return HasTaskIdTail(submit ? submit->GetType() : TypePtr{}); }

std::vector<ParamDirection> ResolveCalleeDirections(const ProgramPtr& program, const CallPtr& call,
                                                    const FunctionPtr& callee) {
  if (!callee) return {};
  if (callee->func_type_ == FunctionType::Group || callee->func_type_ == FunctionType::Spmd) {
    return ComputeGroupEffectiveDirections(callee, program);
  }
  return callee->param_directions_;
}

std::vector<VarPtr> GetDepAttr(const CallPtr& call, const char* key) {
  if (!call) return {};
  for (const auto& [k, v] : call->attrs_) {
    if (k != key) continue;
    if (const auto* edges = std::any_cast<std::vector<VarPtr>>(&v)) {
      return *edges;
    }
    return {};
  }
  return {};
}

std::vector<int32_t> GetIntVectorAttr(const std::vector<std::pair<std::string, std::any>>& attrs,
                                      const char* key) {
  for (const auto& [k, v] : attrs) {
    if (k != key) continue;
    if (const auto* values = std::any_cast<std::vector<int32_t>>(&v)) {
      return *values;
    }
    return {};
  }
  return {};
}

std::vector<std::pair<std::string, std::any>> WithIntVectorAttr(
    std::vector<std::pair<std::string, std::any>> attrs, const char* key, std::vector<int32_t> values) {
  for (auto& [k, v] : attrs) {
    if (k == key) {
      v = std::move(values);
      return attrs;
    }
  }
  attrs.emplace_back(key, std::move(values));
  return attrs;
}

std::vector<std::pair<std::string, std::any>> StripAttr(
    const std::vector<std::pair<std::string, std::any>>& attrs, const char* key) {
  std::vector<std::pair<std::string, std::any>> stripped;
  stripped.reserve(attrs.size());
  for (const auto& attr : attrs) {
    if (attr.first != key) {
      stripped.push_back(attr);
    }
  }
  return stripped;
}

bool HasAttr(const std::vector<std::pair<std::string, std::any>>& attrs, const char* key) {
  for (const auto& [k, v] : attrs) {
    (void)v;
    if (k == key) return true;
  }
  return false;
}

bool ContainsVar(const std::vector<VarPtr>& vars, const VarPtr& candidate) {
  if (!candidate) return false;
  for (const auto& var : vars) {
    if (var && var->UniqueId() == candidate->UniqueId()) return true;
  }
  return false;
}

bool AutoDepsLoopCarryDebugEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("PYPTO_AUTO_DEPS_LOOP_CARRY_DEBUG");
    if (!value) return false;
    std::string text(value);
    return !text.empty() && text != "0" && text != "false" && text != "False";
  }();
  return enabled;
}

void DebugLog(const std::string& message) {
  if (!AutoDepsLoopCarryDebugEnabled()) return;
  std::cerr << "[auto-deps-loop-debug] " << message << '\n';
}

std::string DebugVar(const VarPtr& var) {
  if (!var) return "<null>";
  std::ostringstream oss;
  oss << var->name_hint_ << "#" << var->UniqueId();
  return oss.str();
}

std::string DebugVarList(const std::vector<VarPtr>& vars) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < vars.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << DebugVar(vars[i]);
  }
  oss << "]";
  return oss.str();
}

std::string DebugLocationStatus(LocationStatus status) {
  switch (status) {
    case LocationStatus::Unknown:
      return "Unknown";
    case LocationStatus::Known:
      return "Known";
    case LocationStatus::Unsupported:
      return "Unsupported";
  }
  return "Invalid";
}

std::string DebugAccessKind(AccessKind kind) {
  switch (kind) {
    case AccessKind::Read:
      return "Read";
    case AccessKind::Write:
      return "Write";
    case AccessKind::ReadWrite:
      return "ReadWrite";
  }
  return "Invalid";
}

std::string DebugRegionKind(RegionKind kind) {
  switch (kind) {
    case RegionKind::Unknown:
      return "Unknown";
    case RegionKind::Full:
      return "Full";
    case RegionKind::Box:
      return "Box";
    case RegionKind::Conservative:
      return "Conservative";
    case RegionKind::Dynamic:
      return "Dynamic";
  }
  return "Invalid";
}

void DebugNoDepBucket(const std::string& call_name, size_t arg_index, const std::string& bucket,
                      const std::string& detail = "") {
  if (!AutoDepsLoopCarryDebugEnabled()) return;
  std::ostringstream oss;
  oss << "[auto-no-dep-bucket] call=" << call_name << " arg_index=" << arg_index << " bucket=" << bucket;
  if (!detail.empty()) {
    oss << " " << detail;
  }
  DebugLog(oss.str());
}

YieldStmtPtr GetTrailingYield(const StmtPtr& stmt) {
  if (auto yield = As<YieldStmt>(stmt)) return yield;
  auto seq = As<SeqStmts>(stmt);
  if (!seq || seq->stmts_.empty()) return nullptr;
  return As<YieldStmt>(seq->stmts_.back());
}

bool IsTaskIdVar(const VarPtr& var) {
  auto scalar_ty = As<ScalarType>(var ? var->GetType() : TypePtr{});
  return scalar_ty && scalar_ty->dtype_ == DataType::TASK_ID;
}

bool IsTaskIdArrayType(const TypePtr& type) {
  auto array_ty = As<ArrayType>(type);
  return array_ty && array_ty->dtype_ == DataType::TASK_ID;
}

bool IsTaskIdArrayVar(const VarPtr& var) { return IsTaskIdArrayType(var ? var->GetType() : TypePtr{}); }

std::optional<int64_t> ConstIntValue(const ExprPtr& expr) {
  auto value = As<ConstInt>(expr);
  if (!value) return std::nullopt;
  return value->value_;
}

std::optional<int64_t> TaskIdArrayExtent(const TypePtr& type) {
  auto array_ty = As<ArrayType>(type);
  if (!array_ty || array_ty->dtype_ != DataType::TASK_ID) return std::nullopt;
  return ConstIntValue(array_ty->extent());
}

void AppendUnique(std::vector<VarPtr>* vars, const VarPtr& candidate) {
  if (!vars || !candidate || ContainsVar(*vars, candidate)) return;
  vars->push_back(candidate);
}

void AppendAllUnique(std::vector<VarPtr>* vars, const std::vector<VarPtr>& candidates) {
  if (!vars) return;
  for (const auto& candidate : candidates) {
    AppendUnique(vars, candidate);
  }
}

bool HasCompleteDynamicCoverage(const std::unordered_set<int64_t>& slots, int64_t extent) {
  if (extent <= 0 || static_cast<int64_t>(slots.size()) != extent) return false;
  for (const auto slot : slots) {
    if (slot < 0 || slot >= extent) return false;
  }
  return true;
}

std::vector<std::pair<std::string, std::any>> StripCompilerManualDepEdges(
    const std::vector<std::pair<std::string, std::any>>& attrs) {
  auto stripped = StripAttr(attrs, kAttrCompilerManualDepEdges);
  stripped = StripAttr(stripped, kAttrAutoNoDepCandidateIndices);
  return StripAttr(stripped, kAttrAutoOutputExistingCandidateIndices);
}

bool HasCompilerManualDepEdgesAttr(const std::vector<std::pair<std::string, std::any>>& attrs) {
  return HasAttr(attrs, kAttrCompilerManualDepEdges);
}

bool HasAutoNoDepCandidateAttr(const std::vector<std::pair<std::string, std::any>>& attrs) {
  return HasAttr(attrs, kAttrAutoNoDepCandidateIndices);
}

bool HasAutoOutputExistingCandidateAttr(const std::vector<std::pair<std::string, std::any>>& attrs) {
  return HasAttr(attrs, kAttrAutoOutputExistingCandidateIndices);
}

bool HasHazard(AccessKind current, AccessKind prior) {
  const bool current_writes = current == AccessKind::Write || current == AccessKind::ReadWrite;
  const bool current_reads = current == AccessKind::Read || current == AccessKind::ReadWrite;
  const bool prior_writes = prior == AccessKind::Write || prior == AccessKind::ReadWrite;
  const bool prior_reads = prior == AccessKind::Read || prior == AccessKind::ReadWrite;
  return (current_reads && prior_writes) || (current_writes && prior_reads) ||
         (current_writes && prior_writes);
}

bool IsStaticSingleTripLoop(const ForStmtPtr& op) {
  if (!op) return false;
  auto start = ConstIntValue(op->start_);
  auto stop = ConstIntValue(op->stop_);
  auto step = ConstIntValue(op->step_);
  if (!start.has_value() || !stop.has_value() || !step.has_value() || *step == 0) return false;
  if (*step > 0) {
    return *start < *stop && (*stop - *start + *step - 1) / *step == 1;
  }
  return *start > *stop && (*start - *stop + (-*step) - 1) / (-*step) == 1;
}

std::optional<int64_t> StaticPositiveTripCount(const ForStmtPtr& op) {
  if (!op) return std::nullopt;
  auto start = ConstIntValue(op->start_);
  auto stop = ConstIntValue(op->stop_);
  auto step = ConstIntValue(op->step_);
  if (!start.has_value() || !stop.has_value() || !step.has_value() || *step <= 0) return std::nullopt;
  if (*start >= *stop) return 0;
  return (*stop - *start + *step - 1) / *step;
}

class StorageRootAnalysis : public IRVisitor {
 public:
  explicit StorageRootAnalysis(ProgramPtr program) : program_(std::move(program)) {}

  void Initialize(const std::vector<VarPtr>& params) {
    for (const auto& param : params) {
      if (param && IsTensorType(param->GetType())) {
        RegisterVarLocation(param, MaybeUnknownRegionsForTensorType(SingleLocation(param.get(), FullRegion()),
                                                                    param->GetType()));
      }
    }
  }

  StorageLocation ResolveExpr(const ExprPtr& expr) const {
    auto var = AsVarLike(expr);
    if (!var) return {};
    auto it = locations_.find(var.get());
    return it != locations_.end() ? it->second : StorageLocation{};
  }

  ResolvedLocation ResolveExprStatus(const ExprPtr& expr) const {
    auto var = AsVarLike(expr);
    if (!var) {
      DebugLog("resolve_expr_status expr=<non-var> status=Unknown");
      return ResolvedLocation{LocationStatus::Unknown, {}};
    }
    if (unsupported_locations_.count(var.get()) != 0) {
      DebugLog("resolve_expr_status expr=" + DebugVar(var) + " status=Unsupported");
      return ResolvedLocation{LocationStatus::Unsupported, {}};
    }
    auto location = ResolveExpr(expr);
    if (!HasLocation(location)) {
      DebugLog("resolve_expr_status expr=" + DebugVar(var) + " status=Unknown");
      return ResolvedLocation{LocationStatus::Unknown, {}};
    }
    if (ExceedsRootAlternativeLimit(location)) {
      DebugLog("resolve_expr_status expr=" + DebugVar(var) + " status=Unsupported reason=too_many_roots");
      return ResolvedLocation{LocationStatus::Unsupported, std::move(location)};
    }
    DebugLog("resolve_expr_status expr=" + DebugVar(var) +
             " status=Known roots=" + std::to_string(location.alternatives.size()));
    return ResolvedLocation{LocationStatus::Known, std::move(location)};
  }

  bool MayAlias(const Var* lhs, const Var* rhs) const {
    if (!lhs || !rhs) return false;
    if (lhs == rhs) return true;
    auto lhs_it = root_memrefs_.find(lhs);
    auto rhs_it = root_memrefs_.find(rhs);
    if (lhs_it == root_memrefs_.end() || rhs_it == root_memrefs_.end()) return false;
    return MemRef::MayAlias(lhs_it->second, rhs_it->second);
  }

 protected:
  void VisitStmt_(const IfStmtPtr& op) override {
    IRVisitor::VisitStmt_(op);
    if (!op || op->return_vars_.empty() || !op->else_body_.has_value()) return;

    auto then_yield = GetTrailingYield(op->then_body_);
    auto else_yield = GetTrailingYield(op->else_body_.value());
    if (!then_yield || !else_yield) return;

    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      if (i >= then_yield->value_.size() || i >= else_yield->value_.size()) break;
      auto then_location = ResolveExpr(then_yield->value_[i]);
      auto else_location = ResolveExpr(else_yield->value_[i]);
      if (!HasLocation(then_location) || !HasLocation(else_location)) {
        RegisterUnsupportedLocation(op->return_vars_[i]);
        continue;
      }
      auto merged = MergeLocations(then_location, else_location);
      if (ExceedsRootAlternativeLimit(merged)) {
        RegisterUnsupportedLocation(op->return_vars_[i]);
      } else {
        RegisterVarLocation(op->return_vars_[i], merged);
      }
    }
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    std::vector<StorageLocation> init_locations;
    init_locations.reserve(op->iter_args_.size());
    for (const auto& iter_arg : op->iter_args_) {
      auto location = ResolveExpr(iter_arg->initValue_);
      init_locations.push_back(location);
      if (HasLocation(location)) {
        RegisterVarLocation(iter_arg, location);
      }
    }
    IRVisitor::VisitStmt_(op);

    auto yield = GetTrailingYield(op->body_);
    for (size_t i = 0; i < op->iter_args_.size() && i < op->return_vars_.size(); ++i) {
      auto location = init_locations[i];
      if (yield && i < yield->value_.size()) {
        auto yield_location = ResolveExpr(yield->value_[i]);
        if (!HasLocation(location) || !HasLocation(yield_location)) {
          RegisterUnsupportedLocation(op->return_vars_[i]);
          continue;
        }
        location = MergeLocations(location, yield_location);
      }
      location = MaybeConservativeRegionsForTensorType(location, op->return_vars_[i]->GetType());
      if (ExceedsRootAlternativeLimit(location)) {
        RegisterUnsupportedLocation(op->return_vars_[i]);
      } else if (HasLocation(location)) {
        RegisterVarLocation(op->return_vars_[i], location);
      }
    }
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    std::vector<StorageLocation> init_locations;
    init_locations.reserve(op->iter_args_.size());
    for (const auto& iter_arg : op->iter_args_) {
      auto location = ResolveExpr(iter_arg->initValue_);
      init_locations.push_back(location);
      if (HasLocation(location)) {
        RegisterVarLocation(iter_arg, location);
      }
    }
    IRVisitor::VisitStmt_(op);

    auto yield = GetTrailingYield(op->body_);
    for (size_t i = 0; i < op->iter_args_.size() && i < op->return_vars_.size(); ++i) {
      auto location = init_locations[i];
      if (yield && i < yield->value_.size()) {
        auto yield_location = ResolveExpr(yield->value_[i]);
        if (!HasLocation(location) || !HasLocation(yield_location)) {
          RegisterUnsupportedLocation(op->return_vars_[i]);
          continue;
        }
        location = MergeLocations(location, yield_location);
      }
      location = UnknownRegionsFor(location);
      if (ExceedsRootAlternativeLimit(location)) {
        RegisterUnsupportedLocation(op->return_vars_[i]);
      } else if (HasLocation(location)) {
        RegisterVarLocation(op->return_vars_[i], location);
      }
    }
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (auto call = As<Call>(op->value_)) {
      if (As<TupleType>(call->GetType()) && !IsBuiltinOp(call->op_->name_)) {
        tuple_locations_[op->var_.get()] = CollectCallOutputLocations(call);
        IRVisitor::VisitStmt_(op);
        return;
      }
    } else if (auto submit = As<Submit>(op->value_)) {
      if (As<TupleType>(submit->GetType()) && !IsBuiltinOp(submit->op_->name_)) {
        tuple_locations_[op->var_.get()] = CollectCallOutputLocations(SubmitToCallView(submit));
        IRVisitor::VisitStmt_(op);
        return;
      }
    }

    if (!op->var_ || !IsTensorType(op->var_->GetType())) {
      IRVisitor::VisitStmt_(op);
      return;
    }

    if (auto call = As<Call>(op->value_)) {
      const std::string& op_name = call->op_->name_;
      if (IsOp(call, "tensor.create")) {
        RegisterVarLocation(op->var_, MaybeUnknownRegionsForTensorType(
                                          SingleLocation(op->var_.get(), FullRegion()), op->var_->GetType()));
      } else if (IsOp(call, "tensor.slice")) {
        if (call->args_.size() >= 3) {
          auto parent = ResolveExpr(call->args_[0]);
          if (HasLocation(parent)) {
            RegisterVarLocation(
                op->var_, MaybeUnknownRegionsForTensorType(
                              SliceLocation(parent, call->args_[1], call->args_[2]), op->var_->GetType()));
          }
        }
      } else if (IsOp(call, "tensor.assemble")) {
        if (!call->args_.empty()) {
          auto base = ResolveExpr(call->args_[0]);
          if (HasLocation(base)) {
            RegisterVarLocation(op->var_, MaybeUnknownRegionsForTensorType(base, op->var_->GetType()));
          }
        }
      } else if (IsDynamicAccessOp(call->op_)) {
        RegisterUnsupportedLocation(op->var_);
      } else if (!IsBuiltinOp(op_name)) {
        auto out_locations = CollectCallOutputLocations(call);
        if (As<TupleType>(call->GetType())) {
          tuple_locations_[op->var_.get()] = std::move(out_locations);
        } else if (!out_locations.empty() && HasLocation(out_locations[0])) {
          RegisterVarLocation(op->var_,
                              MaybeUnknownRegionsForTensorType(out_locations[0], op->var_->GetType()));
        }
      }
    } else if (auto submit = As<Submit>(op->value_)) {
      if (!IsBuiltinOp(submit->op_->name_)) {
        auto out_locations = CollectCallOutputLocations(SubmitToCallView(submit));
        if (As<TupleType>(submit->GetType())) {
          tuple_locations_[op->var_.get()] = std::move(out_locations);
        } else if (!out_locations.empty() && HasLocation(out_locations[0])) {
          RegisterVarLocation(op->var_,
                              MaybeUnknownRegionsForTensorType(out_locations[0], op->var_->GetType()));
        }
      }
    } else if (auto tuple_get = As<TupleGetItemExpr>(op->value_)) {
      if (auto tuple_var = AsVarLike(tuple_get->tuple_)) {
        auto it = tuple_locations_.find(tuple_var.get());
        if (it != tuple_locations_.end() && tuple_get->index_ >= 0 &&
            tuple_get->index_ < static_cast<int>(it->second.size()) &&
            HasLocation(it->second[tuple_get->index_])) {
          RegisterVarLocation(
              op->var_, MaybeUnknownRegionsForTensorType(it->second[tuple_get->index_], op->var_->GetType()));
        }
      }
    } else {
      auto location = ResolveExpr(op->value_);
      if (HasLocation(location)) {
        RegisterVarLocation(op->var_, MaybeUnknownRegionsForTensorType(location, op->var_->GetType()));
      }
    }

    IRVisitor::VisitStmt_(op);
  }

 private:
  static YieldStmtPtr GetTrailingYield(const StmtPtr& stmt) {
    if (auto yield = As<YieldStmt>(stmt)) return yield;
    auto seq = As<SeqStmts>(stmt);
    if (!seq || seq->stmts_.empty()) return nullptr;
    return As<YieldStmt>(seq->stmts_.back());
  }

  static MemRefPtr GetShapedMemRef(const TypePtr& type) {
    auto shaped = As<ShapedType>(type);
    if (!shaped || !shaped->memref_.has_value()) return nullptr;
    return shaped->memref_.value();
  }

  void RegisterVarLocation(const VarPtr& var, const StorageLocation& location) {
    if (!var || !HasLocation(location)) return;
    if (ExceedsRootAlternativeLimit(location)) {
      RegisterUnsupportedLocation(var);
      return;
    }
    unsupported_locations_.erase(var.get());
    locations_[var.get()] = location;
    if (const auto memref = GetShapedMemRef(var->GetType())) {
      for (const auto& alternative : location.alternatives) {
        if (alternative.root) {
          root_memrefs_.try_emplace(alternative.root, memref);
        }
      }
    }
  }

  void RegisterUnsupportedLocation(const VarPtr& var) {
    if (!var) return;
    locations_.erase(var.get());
    unsupported_locations_.insert(var.get());
  }

  std::vector<StorageLocation> CollectCallOutputLocations(const CallPtr& call) const {
    auto callee = program_ ? program_->GetFunction(call->op_->name_) : nullptr;
    if (!callee) return {};
    auto dirs = ResolveCalleeDirections(program_, call, callee);
    std::vector<StorageLocation> locations;
    auto returned_locations = CollectReturnedLocations(call, callee);
    if (!returned_locations.empty()) return returned_locations;
    for (size_t i = 0; i < dirs.size() && i < call->args_.size(); ++i) {
      if (dirs[i] != ParamDirection::Out && dirs[i] != ParamDirection::InOut) continue;
      locations.push_back(ResolveExpr(call->args_[i]));
    }
    return locations;
  }

  ProgramPtr program_;
  std::unordered_map<const Var*, StorageLocation> locations_;
  std::unordered_map<const Var*, MemRefPtr> root_memrefs_;
  std::unordered_map<const Var*, std::vector<StorageLocation>> tuple_locations_;
  std::unordered_set<const Var*> unsupported_locations_;

  std::vector<StorageLocation> CollectReturnedLocations(const CallPtr& call,
                                                        const FunctionPtr& callee) const {
    auto ret = GetTrailingReturn(callee ? callee->body_ : StmtPtr{});
    if (!ret) return {};
    std::vector<StorageLocation> locations;
    locations.reserve(ret->value_.size());
    for (const auto& value : ret->value_) {
      locations.push_back(ResolveCalleeReturnExpr(value, call, callee));
    }
    bool has_location = false;
    for (const auto& location : locations) {
      if (HasLocation(location)) {
        has_location = true;
        break;
      }
    }
    if (!has_location) return {};
    return locations;
  }

  StorageLocation ResolveCalleeReturnExpr(const ExprPtr& expr, const CallPtr& call,
                                          const FunctionPtr& callee) const {
    auto var = AsVarLike(expr);
    if (!var || !call || !callee) return {};
    for (size_t i = 0; i < callee->params_.size() && i < call->args_.size(); ++i) {
      if (callee->params_[i] && callee->params_[i]->UniqueId() == var->UniqueId()) {
        return ResolveExpr(call->args_[i]);
      }
    }
    return {};
  }

  static ReturnStmtPtr GetTrailingReturn(const StmtPtr& stmt) {
    if (auto ret = As<ReturnStmt>(stmt)) return ret;
    auto seq = As<SeqStmts>(stmt);
    if (!seq || seq->stmts_.empty()) return nullptr;
    return As<ReturnStmt>(seq->stmts_.back());
  }
};

class SubmitTaskIdCollector : public IRVisitor {
 public:
  void VisitStmt_(const AssignStmtPtr& op) override {
    if (auto tuple_get = As<TupleGetItemExpr>(op->value_)) {
      if (auto tuple_var = AsVarLike(tuple_get->tuple_)) {
        tuple_get_by_tuple_[tuple_var.get()][tuple_get->index_] = op->var_;
        auto expr_it = task_expr_by_tuple_.find(tuple_var.get());
        if (expr_it != task_expr_by_tuple_.end()) {
          auto tuple_ty = As<TupleType>(expr_it->second->GetType());
          const int task_id_index = static_cast<int>(tuple_ty->types_.size()) - 1;
          if (tuple_get->index_ == task_id_index) {
            task_id_by_expr_[expr_it->second.get()] = op->var_;
            for (const auto& [index, var] : tuple_get_by_tuple_[tuple_var.get()]) {
              if (index != task_id_index) {
                task_id_by_var_id_[var->UniqueId()] = op->var_;
              }
            }
          } else {
            auto tid_it = tuple_get_by_tuple_[tuple_var.get()].find(task_id_index);
            if (tid_it != tuple_get_by_tuple_[tuple_var.get()].end()) {
              task_id_by_var_id_[op->var_->UniqueId()] = tid_it->second;
            }
          }
        }
      }
    }

    if (IsTaskIdVar(op->var_)) {
      if (auto source_task_id = CanonicalTaskIdForExpr(op->value_)) {
        if (source_task_id->UniqueId() != op->var_->UniqueId()) {
          task_id_by_var_id_[op->var_->UniqueId()] = source_task_id;
        }
      }
    }

    if (auto call = As<Call>(op->value_)) {
      RecordTaskTupleProducer(op->var_, call);
      RecordTaskIdArrayProducer(op->var_, call);
      RecordTaskIdScalarProducer(op->var_, call);
    } else if (auto submit = As<Submit>(op->value_)) {
      RecordTaskTupleProducer(op->var_, submit);
    }

    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    IRVisitor::VisitStmt_(op);
    PropagateLoopCarriedTaskIds(op->iter_args_, op->return_vars_, op->body_);
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    IRVisitor::VisitStmt_(op);
    PropagateLoopCarriedTaskIds(op->iter_args_, op->return_vars_, op->body_);
  }

  void RecordTaskTupleProducer(const VarPtr& tuple_var, const ExprPtr& expr) {
    if (!tuple_var || !HasTaskIdTail(expr ? expr->GetType() : TypePtr{})) return;
    task_expr_by_tuple_[tuple_var.get()] = expr;
    auto tuple_ty = As<TupleType>(expr->GetType());
    const int task_id_index = static_cast<int>(tuple_ty->types_.size()) - 1;
    auto it = tuple_get_by_tuple_.find(tuple_var.get());
    if (it != tuple_get_by_tuple_.end()) {
      auto elem_it = it->second.find(task_id_index);
      if (elem_it != it->second.end()) {
        task_id_by_expr_[expr.get()] = elem_it->second;
      }
    }
  }

  const std::unordered_map<const Expr*, VarPtr>& task_id_by_expr() const { return task_id_by_expr_; }
  const std::unordered_map<uint64_t, VarPtr>& task_id_by_var_id() const { return task_id_by_var_id_; }
  const std::unordered_map<uint64_t, std::vector<VarPtr>>& task_ids_by_var_id() const {
    return task_ids_by_var_id_;
  }
  const std::unordered_map<uint64_t, std::unordered_map<uint64_t, std::unordered_set<int64_t>>>&
  task_id_dynamic_slots_by_var_id() const {
    return task_id_dynamic_slots_by_var_id_;
  }
  const std::unordered_map<uint64_t, int64_t>& task_id_array_extent_by_var_id() const {
    return task_id_array_extent_by_var_id_;
  }
  const std::unordered_map<uint64_t, std::vector<VarPtr>>& task_ids_by_array_var_id() const {
    return task_ids_by_array_var_id_;
  }
  const std::unordered_set<uint64_t>& complete_task_id_array_var_ids() const {
    return complete_task_id_array_var_ids_;
  }

 private:
  struct TaskIdExprLineage {
    std::vector<VarPtr> task_ids;
    std::unordered_map<uint64_t, std::unordered_set<int64_t>> dynamic_array_slots;
  };

  struct TaskIdArrayLineage {
    std::vector<VarPtr> full_array_task_ids;
    std::unordered_map<int64_t, std::vector<VarPtr>> static_slots;
    std::unordered_map<int64_t, std::unordered_map<uint64_t, std::unordered_set<int64_t>>>
        static_dynamic_slots;
    std::unordered_set<int64_t> unknown_static_slots;
    bool has_unknown_dynamic_update = false;
    std::optional<int64_t> extent;
  };

  VarPtr CanonicalTaskIdForExpr(const ExprPtr& expr) const {
    auto var = AsVarLike(expr);
    if (!var) return nullptr;

    auto it = task_id_by_var_id_.find(var->UniqueId());
    if (it != task_id_by_var_id_.end()) return it->second;

    if (IsTaskIdVar(var)) return var;
    return nullptr;
  }

  TaskIdExprLineage TaskIdsForExpr(const ExprPtr& expr) const {
    TaskIdExprLineage out;
    if (!expr) return out;

    if (auto var = AsVarLike(expr)) {
      auto ids_it = task_ids_by_var_id_.find(var->UniqueId());
      if (ids_it != task_ids_by_var_id_.end()) {
        AppendAllUnique(&out.task_ids, ids_it->second);
      }
      auto slots_it = task_id_dynamic_slots_by_var_id_.find(var->UniqueId());
      if (slots_it != task_id_dynamic_slots_by_var_id_.end()) {
        out.dynamic_array_slots = slots_it->second;
      }
      if (!out.task_ids.empty() || !out.dynamic_array_slots.empty()) return out;
    }

    if (auto task_id = CanonicalTaskIdForExpr(expr)) {
      AppendUnique(&out.task_ids, task_id);
      return out;
    }

    auto call = As<Call>(expr);
    if (!call || !IsOp(call, "array.get_element") || call->args_.size() != 2) return out;

    auto array_var = AsVarLike(call->args_[0]);
    auto index = ConstIntValue(call->args_[1]);
    if (!array_var || !index.has_value()) return out;

    auto lineage_it = array_lineage_by_var_id_.find(array_var->UniqueId());
    if (lineage_it == array_lineage_by_var_id_.end()) return out;

    auto slot_it = lineage_it->second.static_slots.find(*index);
    if (slot_it != lineage_it->second.static_slots.end()) {
      AppendAllUnique(&out.task_ids, slot_it->second);
    }
    const auto expanded_it = task_ids_by_array_var_id_.find(array_var->UniqueId());
    const bool has_dynamic_array_task_ids =
        !lineage_it->second.full_array_task_ids.empty() ||
        (expanded_it != task_ids_by_array_var_id_.end() && !expanded_it->second.empty());
    if (has_dynamic_array_task_ids) {
      out.dynamic_array_slots[array_var->UniqueId()].insert(*index);
    }
    return out;
  }

  std::vector<VarPtr> ExpandTaskIdArray(const VarPtr& array_var) const {
    std::vector<VarPtr> out;
    if (!array_var || !IsTaskIdArrayVar(array_var)) return out;

    auto lineage_it = array_lineage_by_var_id_.find(array_var->UniqueId());
    if (lineage_it == array_lineage_by_var_id_.end()) return out;

    const auto& lineage = lineage_it->second;
    if (!lineage.has_unknown_dynamic_update && lineage.unknown_static_slots.empty()) {
      AppendAllUnique(&out, lineage.full_array_task_ids);
    }
    for (const auto& [_, task_ids] : lineage.static_slots) {
      AppendAllUnique(&out, task_ids);
    }
    std::unordered_map<uint64_t, std::unordered_set<int64_t>> covered_source_slots;
    for (const auto& [_, source_slots] : lineage.static_dynamic_slots) {
      for (const auto& [source_id, slots] : source_slots) {
        for (const auto slot : slots) {
          covered_source_slots[source_id].insert(slot);
        }
      }
    }
    for (const auto& [source_id, slots] : covered_source_slots) {
      auto source_it = array_lineage_by_var_id_.find(source_id);
      if (source_it == array_lineage_by_var_id_.end() || !source_it->second.extent.has_value()) continue;
      if (!HasCompleteDynamicCoverage(slots, source_it->second.extent.value())) continue;
      if (lineage.has_unknown_dynamic_update || !lineage.unknown_static_slots.empty()) continue;
      if (source_it->second.has_unknown_dynamic_update || !source_it->second.unknown_static_slots.empty())
        continue;
      AppendAllUnique(&out, source_it->second.full_array_task_ids);
    }
    return out;
  }

  void PropagateLoopCarriedTaskIds(const std::vector<IterArgPtr>& iter_args,
                                   const std::vector<VarPtr>& return_vars, const StmtPtr& body) {
    auto yield = GetTrailingYield(body);
    if (!yield) return;

    const size_t count = std::min({iter_args.size(), return_vars.size(), yield->value_.size()});
    for (size_t i = 0; i < count; ++i) {
      VarPtr task_id = CanonicalTaskIdForExpr(yield->value_[i]);
      if (!task_id) continue;

      if (iter_args[i]) {
        task_id_by_var_id_[iter_args[i]->UniqueId()] = task_id;
      }
      if (return_vars[i]) {
        task_id_by_var_id_[return_vars[i]->UniqueId()] = task_id;
      }
    }

    PropagateLoopCarriedTaskIdArrays(iter_args, return_vars, yield);
  }

  void PropagateLoopCarriedTaskIdArrays(const std::vector<IterArgPtr>& iter_args,
                                        const std::vector<VarPtr>& return_vars, const YieldStmtPtr& yield) {
    if (!yield) return;

    const size_t count = std::min({iter_args.size(), return_vars.size(), yield->value_.size()});
    for (size_t i = 0; i < count; ++i) {
      auto yielded_array = AsVarLike(yield->value_[i]);
      if (!yielded_array || !IsTaskIdArrayVar(yielded_array)) continue;

      auto lineage_it = array_lineage_by_var_id_.find(yielded_array->UniqueId());
      if (lineage_it == array_lineage_by_var_id_.end()) continue;

      if (iter_args[i] && IsTaskIdArrayVar(iter_args[i])) {
        array_lineage_by_var_id_[iter_args[i]->UniqueId()] = lineage_it->second;
        RecordTaskIdArrayDerivedFacts(iter_args[i]);
      }
      if (return_vars[i] && IsTaskIdArrayVar(return_vars[i])) {
        array_lineage_by_var_id_[return_vars[i]->UniqueId()] = lineage_it->second;
        RecordTaskIdArrayDerivedFacts(return_vars[i]);
      }
    }
  }

  void RecordTaskIdArrayProducer(const VarPtr& var, const CallPtr& call) {
    if (!var || !IsTaskIdArrayVar(var) || !call) return;

    if (IsOp(call, "array.create")) {
      TaskIdArrayLineage lineage;
      lineage.extent = TaskIdArrayExtent(var->GetType());
      array_lineage_by_var_id_[var->UniqueId()] = std::move(lineage);
      RecordTaskIdArrayDerivedFacts(var);
      return;
    }

    if (!IsOp(call, "array.update_element") || call->args_.size() != 3) return;

    TaskIdArrayLineage lineage;
    if (auto base_array = AsVarLike(call->args_[0])) {
      auto base_it = array_lineage_by_var_id_.find(base_array->UniqueId());
      if (base_it != array_lineage_by_var_id_.end()) {
        lineage = base_it->second;
      }
    }
    if (!lineage.extent.has_value()) {
      lineage.extent = TaskIdArrayExtent(var->GetType());
    }

    auto value_lineage = TaskIdsForExpr(call->args_[2]);
    auto index = ConstIntValue(call->args_[1]);
    if (index.has_value()) {
      if (value_lineage.task_ids.empty()) {
        lineage.static_slots.erase(*index);
      } else {
        lineage.static_slots[*index] = value_lineage.task_ids;
      }
      if (value_lineage.dynamic_array_slots.empty()) {
        lineage.static_dynamic_slots.erase(*index);
      } else {
        lineage.static_dynamic_slots[*index] = std::move(value_lineage.dynamic_array_slots);
      }
      if (value_lineage.task_ids.empty() &&
          lineage.static_dynamic_slots.find(*index) == lineage.static_dynamic_slots.end()) {
        lineage.unknown_static_slots.insert(*index);
      } else {
        lineage.unknown_static_slots.erase(*index);
      }
    } else {
      if (!value_lineage.task_ids.empty() && value_lineage.dynamic_array_slots.empty()) {
        AppendAllUnique(&lineage.full_array_task_ids, value_lineage.task_ids);
      } else {
        lineage.has_unknown_dynamic_update = true;
      }
    }

    array_lineage_by_var_id_[var->UniqueId()] = std::move(lineage);
    RecordTaskIdArrayDerivedFacts(var);
  }

  void RecordTaskIdScalarProducer(const VarPtr& var, const CallPtr& call) {
    if (!var || !IsTaskIdVar(var) || !call) return;

    auto lineage = TaskIdsForExpr(call);
    if (!lineage.task_ids.empty()) {
      task_ids_by_var_id_[var->UniqueId()] = lineage.task_ids;
      if (lineage.task_ids.size() == 1) {
        task_id_by_var_id_[var->UniqueId()] = lineage.task_ids.front();
      }
    }
    if (!lineage.dynamic_array_slots.empty()) {
      task_id_dynamic_slots_by_var_id_[var->UniqueId()] = std::move(lineage.dynamic_array_slots);
    }
  }

  void RecordTaskIdArrayExtent(const VarPtr& var) {
    if (!var) return;
    auto lineage_it = array_lineage_by_var_id_.find(var->UniqueId());
    if (lineage_it != array_lineage_by_var_id_.end() && lineage_it->second.extent.has_value()) {
      task_id_array_extent_by_var_id_[var->UniqueId()] = lineage_it->second.extent.value();
    } else {
      task_id_array_extent_by_var_id_.erase(var->UniqueId());
    }
  }

  bool IsCompleteTaskIdArray(const VarPtr& var) const {
    if (!var || !IsTaskIdArrayVar(var)) return false;
    auto lineage_it = array_lineage_by_var_id_.find(var->UniqueId());
    if (lineage_it == array_lineage_by_var_id_.end()) return false;

    const auto& lineage = lineage_it->second;
    if (!lineage.extent.has_value() || *lineage.extent <= 0) return false;
    if (lineage.has_unknown_dynamic_update || !lineage.unknown_static_slots.empty()) return false;

    std::unordered_set<int64_t> covered_slots;
    for (const auto& [slot, task_ids] : lineage.static_slots) {
      if (!task_ids.empty()) covered_slots.insert(slot);
    }
    for (const auto& [slot, source_slots] : lineage.static_dynamic_slots) {
      if (!source_slots.empty()) covered_slots.insert(slot);
    }
    return HasCompleteDynamicCoverage(covered_slots, *lineage.extent);
  }

  void RecordTaskIdArrayDerivedFacts(const VarPtr& var) {
    if (!var) return;
    RecordTaskIdArrayExtent(var);
    if (IsTaskIdArrayVar(var)) {
      task_ids_by_array_var_id_[var->UniqueId()] = ExpandTaskIdArray(var);
    }
    if (IsCompleteTaskIdArray(var)) {
      complete_task_id_array_var_ids_.insert(var->UniqueId());
    } else {
      complete_task_id_array_var_ids_.erase(var->UniqueId());
    }
  }

  std::unordered_map<const Var*, ExprPtr> task_expr_by_tuple_;
  std::unordered_map<const Var*, std::unordered_map<int, VarPtr>> tuple_get_by_tuple_;
  std::unordered_map<const Expr*, VarPtr> task_id_by_expr_;
  std::unordered_map<uint64_t, VarPtr> task_id_by_var_id_;
  std::unordered_map<uint64_t, std::vector<VarPtr>> task_ids_by_var_id_;
  std::unordered_map<uint64_t, std::unordered_map<uint64_t, std::unordered_set<int64_t>>>
      task_id_dynamic_slots_by_var_id_;
  std::unordered_map<uint64_t, int64_t> task_id_array_extent_by_var_id_;
  std::unordered_map<uint64_t, TaskIdArrayLineage> array_lineage_by_var_id_;
  std::unordered_map<uint64_t, std::vector<VarPtr>> task_ids_by_array_var_id_;
  std::unordered_set<uint64_t> complete_task_id_array_var_ids_;
};

class AutoDepMutator : public IRMutator {
 public:
  AutoDepMutator(
      ProgramPtr program, const StorageRootAnalysis* storage,
      const std::unordered_map<const Expr*, VarPtr>* task_id_by_expr,
      const std::unordered_map<uint64_t, VarPtr>* task_id_by_var_id,
      const std::unordered_map<uint64_t, std::vector<VarPtr>>* task_ids_by_var_id,
      const std::unordered_map<uint64_t, std::unordered_map<uint64_t, std::unordered_set<int64_t>>>*
          task_id_dynamic_slots_by_var_id,
      const std::unordered_map<uint64_t, int64_t>* task_id_array_extent_by_var_id,
      const std::unordered_map<uint64_t, std::vector<VarPtr>>* task_ids_by_array_var_id,
      const std::unordered_set<uint64_t>* complete_task_id_array_var_ids, bool analyze_auto_scopes,
      bool analyze_whole_body_as_auto_scope)
      : program_(std::move(program)),
        storage_(storage),
        task_id_by_expr_(task_id_by_expr),
        task_id_by_var_id_(task_id_by_var_id),
        task_ids_by_var_id_(task_ids_by_var_id),
        task_id_dynamic_slots_by_var_id_(task_id_dynamic_slots_by_var_id),
        task_id_array_extent_by_var_id_(task_id_array_extent_by_var_id),
        task_ids_by_array_var_id_(task_ids_by_array_var_id),
        complete_task_id_array_var_ids_(complete_task_id_array_var_ids),
        analyze_auto_scopes_(analyze_auto_scopes),
        analyze_whole_body_as_auto_scope_(analyze_whole_body_as_auto_scope) {
    if (!task_id_by_expr_) return;
    for (const auto& [_, task_id] : *task_id_by_expr_) {
      if (task_id) submit_task_id_var_ids_.insert(task_id->UniqueId());
    }
  }

  StmtPtr AnalyzeBody(const StmtPtr& body) {
    INTERNAL_CHECK(body != nullptr) << "Internal error: AutoDepMutator received null function body";
    if (!analyze_whole_body_as_auto_scope_ || As<RuntimeScopeStmt>(body)) {
      if (analyze_auto_scopes_ && !analyze_whole_body_as_auto_scope_ && !As<RuntimeScopeStmt>(body)) {
        return AnalyzeCrossScopeContainerBody(body);
      }
      return VisitStmt(body);
    }
    return AnalyzeRuntimeScopeBody(body, /*manual=*/false, /*name_hint=*/"", body->span_,
                                   /*is_virtual_whole_body=*/true, /*original_scope=*/nullptr,
                                   std::vector<std::string>{},
                                   std::vector<std::pair<std::string, std::any>>{});
  }

 protected:
  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    if (prior_stack_.empty()) return IRMutator::VisitStmt_(op);
    if (IsStaticSingleTripLoop(op)) {
      auto result = IRMutator::VisitStmt_(op);
      if (auto new_for = As<ForStmt>(result)) {
        RemapSingleTripLoopAccessTaskIds(new_for);
      }
      return result;
    }
    const auto static_trip_count = StaticPositiveTripCount(op);
    const bool dynamic_loop = !static_trip_count.has_value();
    const size_t current_loop_depth = loop_depth_ + 1;
    ++loop_depth_;
    if (dynamic_loop) {
      dynamic_loop_depth_stack_.push_back(current_loop_depth);
      dynamic_loop_id_stack_.push_back(next_dynamic_loop_id_++);
    }
    auto result = IRMutator::VisitStmt_(op);
    if (dynamic_loop) {
      dynamic_loop_id_stack_.pop_back();
      dynamic_loop_depth_stack_.pop_back();
    }
    --loop_depth_;
    if (static_trip_count.has_value() && *static_trip_count > 0) {
      if (auto new_for = As<ForStmt>(result)) {
        RemapFixedTripLoopAccessTaskIds(new_for, current_loop_depth);
      }
      // Only summarize top-level dynamic parallel carriers for now. Nested
      // dynamic producers need extra dominance/hoisting rules; keep their
      // existing conservative fallback behavior.
    } else if (dynamic_loop && op->kind_ == ForKind::Parallel && current_loop_depth == 1) {
      if (auto new_for = As<ForStmt>(result)) {
        RemapDynamicParallelLoopAccessTaskIds(new_for);
      }
    }
    return result;
  }

  StmtPtr VisitStmt_(const WhileStmtPtr& op) override {
    if (prior_stack_.empty()) return IRMutator::VisitStmt_(op);
    const size_t current_loop_depth = loop_depth_ + 1;
    ++loop_depth_;
    dynamic_loop_depth_stack_.push_back(current_loop_depth);
    dynamic_loop_id_stack_.push_back(next_dynamic_loop_id_++);
    auto result = IRMutator::VisitStmt_(op);
    dynamic_loop_id_stack_.pop_back();
    dynamic_loop_depth_stack_.pop_back();
    --loop_depth_;
    return result;
  }

  StmtPtr VisitStmt_(const RuntimeScopeStmtPtr& op) override {
    // Manual scopes still inspect user-covered read-only inputs for NoDep,
    // but never synthesize compiler dependency edges for the scope.
    if (!op->manual_ && !analyze_auto_scopes_) {
      return op;
    }
    return AnalyzeRuntimeScopeBody(op->body_, op->manual_, op->name_hint_, op->span_,
                                   /*is_virtual_whole_body=*/false, op, op->leading_comments_, op->attrs_);
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto saved_lhs = current_assign_lhs_;
    current_assign_lhs_ = op->var_;
    auto result = IRMutator::VisitStmt_(op);
    current_assign_lhs_ = std::move(saved_lhs);
    return result;
  }

  StmtPtr AnalyzeRuntimeScopeBody(const StmtPtr& body, bool manual, const std::string& name_hint,
                                  const Span& span, bool is_virtual_whole_body,
                                  const RuntimeScopeStmtPtr& original_scope,
                                  std::vector<std::string> leading_comments,
                                  std::vector<std::pair<std::string, std::any>> attrs) {
    DebugLog("enter_runtime_scope name_hint=" + name_hint +
             " manual=" + (manual ? std::string("true") : std::string("false")) +
             " virtual_whole_body=" + (is_virtual_whole_body ? std::string("true") : std::string("false")) +
             " loop_depth=" + std::to_string(loop_depth_));
    prior_stack_.emplace_back();
    scope_manual_stack_.push_back(manual);
    fallback_stack_.push_back(false);
    auto_no_dep_candidate_count_stack_.push_back(0);
    INTERNAL_CHECK_SPAN(body, span) << "RuntimeScopeStmt has null body";
    auto new_body = VisitStmt(body);
    INTERNAL_CHECK_SPAN(new_body, span) << "RuntimeScopeStmt body mutated to null";
    const bool fallback = fallback_stack_.back();
    const size_t auto_no_dep_candidate_count = auto_no_dep_candidate_count_stack_.back();
    auto scope_accesses = std::move(prior_stack_.back());
    auto_no_dep_candidate_count_stack_.pop_back();
    fallback_stack_.pop_back();
    scope_manual_stack_.pop_back();
    prior_stack_.pop_back();
    const bool has_parent_runtime_scope = !auto_no_dep_candidate_count_stack_.empty();
    if (!fallback && !prior_stack_.empty()) {
      for (auto& access : scope_accesses) {
        prior_stack_.back().push_back(std::move(access));
      }
    }
    if (fallback) {
      DebugLog("fallback_runtime_scope name_hint=" + name_hint);
      // The virtual whole-body region has no runtime scope boundary to fall back to.
      // Keep representable local edges; the fallback already prevents access export.
      if (is_virtual_whole_body) {
        return auto_no_dep_candidate_count > 0 ? ApplyAutoNoDepCandidates(new_body) : new_body;
      }
      auto stripped_body = StripCompilerDeps(new_body);
      return std::make_shared<const RuntimeScopeStmt>(false, name_hint, std::move(stripped_body), span,
                                                      std::move(leading_comments), std::move(attrs));
    }
    DebugLog("keep_runtime_scope name_hint=" + name_hint +
             " manual=" + (manual ? std::string("true") : std::string("false")));
    if (auto_no_dep_candidate_count > 0 && has_parent_runtime_scope) {
      auto_no_dep_candidate_count_stack_.back() += auto_no_dep_candidate_count;
    } else if (auto_no_dep_candidate_count > 0) {
      new_body = ApplyAutoNoDepCandidates(new_body);
    }
    if (is_virtual_whole_body) return new_body;
    if (new_body.get() != body.get()) {
      return std::make_shared<const RuntimeScopeStmt>(manual, name_hint, std::move(new_body), span,
                                                      std::move(leading_comments), std::move(attrs));
    }
    return original_scope ? original_scope : body;
  }

  StmtPtr AnalyzeCrossScopeContainerBody(const StmtPtr& body) {
    prior_stack_.emplace_back();
    auto_no_dep_candidate_count_stack_.push_back(0);
    auto new_body = VisitStmt(body);
    const size_t auto_no_dep_candidate_count = auto_no_dep_candidate_count_stack_.back();
    auto_no_dep_candidate_count_stack_.pop_back();
    prior_stack_.pop_back();
    if (auto_no_dep_candidate_count > 0) {
      new_body = ApplyAutoNoDepCandidates(new_body);
    }
    return new_body;
  }

  void RemapSingleTripLoopAccessTaskIds(const ForStmtPtr& op) {
    if (prior_stack_.empty() || !op) return;
    auto yield = GetTrailingYield(op->body_);
    if (!yield) return;

    auto remap = BuildLoopCarryTaskIdRemap(op, yield);
    if (remap.empty()) return;

    for (auto& access : prior_stack_.back()) {
      if (!access.task_id_var) continue;
      auto it = remap.find(access.task_id_var->UniqueId());
      if (it != remap.end()) {
        access.task_id_var = it->second.target;
        access.task_id_var_is_direct = it->second.direct_task_id;
      }
    }
  }

  void RemapFixedTripLoopAccessTaskIds(const ForStmtPtr& op, size_t fixed_loop_depth) {
    if (prior_stack_.empty() || !op) return;
    auto yield = GetTrailingYield(op->body_);
    if (!yield) return;

    auto remap = BuildLoopCarryTaskIdRemap(op, yield);
    if (remap.empty()) return;

    for (auto& access : prior_stack_.back()) {
      if (!access.dynamic_producer || !access.task_id_var) continue;
      bool keep_dynamic_producer = false;
      auto it = remap.find(access.task_id_var->UniqueId());
      if (it != remap.end()) {
        access.task_id_var = it->second.target;
        access.task_id_var_is_direct = it->second.direct_task_id;
        keep_dynamic_producer =
            !it->second.direct_task_id && access.innermost_dynamic_loop_depth > fixed_loop_depth;
      } else if (auto return_var = LoopReturnVarForAccess(op, access)) {
        keep_dynamic_producer = access.innermost_dynamic_loop_depth > fixed_loop_depth;
        access.task_id_var = return_var;
        access.task_id_var_is_direct = false;
      } else {
        continue;
      }
      access.dynamic_producer = keep_dynamic_producer;
    }
  }

  void RemapDynamicParallelLoopAccessTaskIds(const ForStmtPtr& op) {
    if (prior_stack_.empty() || !op) return;

    for (auto& access : prior_stack_.back()) {
      if (!access.dynamic_producer || !access.task_id_var) continue;
      auto return_var = LoopReturnVarForAccess(op, access);
      if (!return_var) continue;
      access.task_id_var = return_var;
      access.task_id_var_is_direct = false;
      access.dynamic_producer = false;
      access.innermost_dynamic_loop_depth = 0;
      access.innermost_dynamic_loop_id = 0;
    }
  }

  VarPtr LoopReturnVarForAccess(const ForStmtPtr& op, const StorageAccess& access) const {
    if (!storage_ || !op) return nullptr;
    for (const auto& return_var : op->return_vars_) {
      if (!return_var) continue;
      auto resolved = storage_->ResolveExprStatus(return_var);
      if (resolved.status != LocationStatus::Known) continue;
      for (const auto& alternative : resolved.location.alternatives) {
        if (!storage_->MayAlias(access.location.root, alternative.root)) continue;
        if (!RegionsMayOverlap(access.location.region, alternative.region)) continue;
        return return_var;
      }
    }
    return nullptr;
  }

  std::unordered_map<uint64_t, LoopCarryTaskIdMapping> BuildLoopCarryTaskIdRemap(
      const ForStmtPtr& op, const YieldStmtPtr& yield) const {
    std::unordered_map<uint64_t, LoopCarryTaskIdMapping> remap;
    if (!op || !yield) return remap;
    const bool debug_loop_carry = AutoDepsLoopCarryDebugEnabled();
    if (debug_loop_carry) {
      DebugLog("loop_carry_remap return_vars=" + std::to_string(op->return_vars_.size()) + " yield_values=" +
               std::to_string(yield->value_.size()) + " iter_args=" + std::to_string(op->iter_args_.size()));
    }

    auto add_mapping = [&](const VarPtr& source, const VarPtr& target) {
      if (!source || !target) return;
      const bool direct_task_id = IsTaskIdVar(source) || IsTaskIdVar(target);
      auto insert_mapping = [&](uint64_t id, const VarPtr& mapped_target, bool mapped_direct) {
        auto existing = remap.find(id);
        if (existing != remap.end() && existing->second.direct_task_id && !mapped_direct) return;
        remap[id] = LoopCarryTaskIdMapping{mapped_target, mapped_direct};
      };

      insert_mapping(source->UniqueId(), target, direct_task_id);
      if (debug_loop_carry) {
        DebugLog("loop_carry_map source=" + DebugVar(source) + " target=" + DebugVar(target) +
                 " direct=" + (direct_task_id ? std::string("true") : "false"));
      }
      if (auto canonical = CanonicalTaskId(source)) {
        insert_mapping(canonical->UniqueId(), target, direct_task_id);
        if (debug_loop_carry) {
          DebugLog("loop_carry_map canonical=" + DebugVar(canonical) + " target=" + DebugVar(target) +
                   " direct=" + (direct_task_id ? std::string("true") : "false"));
        }
      }
    };

    const size_t yield_count = std::min(op->return_vars_.size(), yield->value_.size());
    for (size_t i = 0; i < yield_count; ++i) {
      const auto& return_var = op->return_vars_[i];
      if (!return_var) continue;
      add_mapping(AsVarLike(yield->value_[i]), return_var);
    }

    const size_t iter_arg_count = std::min(op->return_vars_.size(), op->iter_args_.size());
    for (size_t i = 0; i < iter_arg_count; ++i) {
      const auto& return_var = op->return_vars_[i];
      if (!return_var) continue;
      add_mapping(op->iter_args_[i], return_var);
    }
    return remap;
  }

  ExprPtr VisitExpr_(const CallPtr& op) override {
    auto base = IRMutator::VisitExpr_(op);
    auto call = As<Call>(base);
    if (!call) return base;
    return AnalyzeCallLike(call, op.get(), call->attrs_);
  }

  ExprPtr VisitExpr_(const SubmitPtr& op) override {
    auto base = IRMutator::VisitExpr_(op);
    auto submit = As<Submit>(base);
    if (!submit) return base;
    auto rewritten = AnalyzeCallLike(SubmitToCallView(submit), op.get(), submit->attrs_);
    auto rewritten_call = As<Call>(rewritten);
    if (!rewritten_call || (!HasCompilerManualDepEdgesAttr(rewritten_call->attrs_) &&
                            !HasAutoNoDepCandidateAttr(rewritten_call->attrs_) &&
                            !HasAutoOutputExistingCandidateAttr(rewritten_call->attrs_))) {
      return submit;
    }
    return std::make_shared<const Submit>(
        submit->op_, submit->args_, submit->deps_, submit->kwargs_, rewritten_call->attrs_, submit->GetType(),
        submit->span_, submit->core_num_, submit->sync_start_, submit->allow_early_resolve_);
  }

  ExprPtr AnalyzeCallLike(const CallPtr& call, const Expr* identity_key,
                          const std::vector<std::pair<std::string, std::any>>& output_attrs) {
    if (prior_stack_.empty() || scope_manual_stack_.empty()) return call;
    if (IsBuiltinOp(call->op_->name_)) return call;

    VarPtr task_id = LookupTaskId(identity_key);
    const bool in_manual_scope = !scope_manual_stack_.empty() && scope_manual_stack_.back();
    if (!task_id) {
      if (!in_manual_scope || IsTaskIdVar(current_assign_lhs_)) {
        task_id = current_assign_lhs_;
      }
    }
    bool needs_fallback = false;
    auto raw_user_edges = GetDepAttr(call, kAttrManualDepEdges);
    auto user_edges = CanonicalizeTaskIds(raw_user_edges);
    const bool debug_loop_carry = AutoDepsLoopCarryDebugEnabled();
    auto summary = SummarizeAccesses(call, user_edges, &needs_fallback);
    if (debug_loop_carry) {
      DebugLog("call=" + call->op_->name_ + " loop_depth=" + std::to_string(loop_depth_) +
               " task_id=" + DebugVar(task_id) + " user_edges=" + DebugVarList(user_edges) +
               " summary_accesses=" + std::to_string(summary.accesses.size()) +
               " needs_fallback_from_summary=" + (needs_fallback ? "true" : "false"));
    }
    if (needs_fallback) {
      if (in_manual_scope) return call;
      if (debug_loop_carry) {
        DebugLog("fallback_reason=summary_unknown_location call=" + call->op_->name_);
      }
      MarkCurrentScopeFallback();
      return call;
    }
    auto& accesses = summary.accesses;
    if (accesses.empty()) return call;

    auto arg_directions = call->GetArgDirections();
    std::unordered_map<size_t, AutoNoDepCandidateState> auto_no_dep_candidates;
    std::unordered_map<size_t, AutoNoDepCandidateState> auto_output_existing_candidates;
    auto can_consider_auto_no_dep = [&](const StorageAccess& access) {
      return access.kind == AccessKind::Read && access.arg_index != kInvalidArgIndex &&
             access.arg_index < arg_directions.size() &&
             arg_directions[access.arg_index] == ArgDirection::Input &&
             IsPreciseNoDepRegion(access.location.region);
    };
    auto can_consider_auto_output_existing = [&](const StorageAccess& access) {
      return !in_manual_scope && access.kind == AccessKind::ReadWrite &&
             access.arg_index != kInvalidArgIndex && access.arg_index < arg_directions.size() &&
             arg_directions[access.arg_index] == ArgDirection::InOut &&
             IsPreciseNoDepRegion(access.location.region);
    };
    auto auto_no_dep_detail = [&](const StorageAccess& access, const VarPtr& prior_edge) {
      std::ostringstream oss;
      oss << "direction=";
      if (access.arg_index < arg_directions.size()) {
        oss << ArgDirectionToString(arg_directions[access.arg_index]);
      } else {
        oss << "Invalid";
      }
      oss << " access=" << DebugAccessKind(access.kind)
          << " region=" << DebugRegionKind(access.location.region.kind)
          << " prior_edge=" << DebugVar(prior_edge) << " current_task_id=" << DebugVar(task_id);
      return oss.str();
    };
    auto log_not_auto_no_dep_candidate = [&](const StorageAccess& access, const VarPtr& prior_edge) {
      if (access.arg_index == kInvalidArgIndex || access.arg_index >= arg_directions.size()) return;
      if (can_consider_auto_output_existing(access)) return;
      if (access.kind != AccessKind::Read || arg_directions[access.arg_index] != ArgDirection::Input) {
        DebugNoDepBucket(call->op_->name_, access.arg_index, "inout_or_output",
                         auto_no_dep_detail(access, prior_edge));
        return;
      }
      if (access.location.region.kind == RegionKind::Unknown) {
        DebugNoDepBucket(call->op_->name_, access.arg_index, "unknown_region",
                         auto_no_dep_detail(access, prior_edge));
        return;
      }
      if (access.location.region.kind == RegionKind::Dynamic) {
        DebugNoDepBucket(call->op_->name_, access.arg_index, "dynamic_region",
                         auto_no_dep_detail(access, prior_edge));
      }
    };
    auto mark_auto_no_dep_blocked = [&](const StorageAccess& access, const std::string& reason,
                                        const VarPtr& prior_edge) {
      if (!can_consider_auto_no_dep(access)) return;
      auto& state = auto_no_dep_candidates[access.arg_index];
      state.blocked = true;
      if (state.blocked_reason.empty()) {
        state.blocked_reason = reason;
      }
      DebugNoDepBucket(call->op_->name_, access.arg_index, reason, auto_no_dep_detail(access, prior_edge));
    };
    auto mark_auto_output_existing_blocked = [&](const StorageAccess& access, const std::string& reason,
                                                 const VarPtr& prior_edge) {
      if (!can_consider_auto_output_existing(access)) return;
      auto& state = auto_output_existing_candidates[access.arg_index];
      state.blocked = true;
      if (state.blocked_reason.empty()) {
        state.blocked_reason = reason;
      }
      DebugNoDepBucket(call->op_->name_, access.arg_index, reason, auto_no_dep_detail(access, prior_edge));
    };
    auto mark_auto_no_dep_covered = [&](const StorageAccess& access) {
      if (!can_consider_auto_no_dep(access)) return;
      auto_no_dep_candidates[access.arg_index].has_covered_hazard = true;
    };
    auto mark_auto_output_existing_covered = [&](const StorageAccess& access) {
      if (!can_consider_auto_output_existing(access)) return;
      auto_output_existing_candidates[access.arg_index].has_covered_hazard = true;
    };

    std::vector<VarPtr> compiler_edges;
    for (const auto& access : accesses) {
      for (auto frame_it = prior_stack_.rbegin(); frame_it != prior_stack_.rend(); ++frame_it) {
        for (const auto& prior : *frame_it) {
          if (!storage_ || !storage_->MayAlias(access.location.root, prior.location.root)) continue;
          if (!RegionsMayOverlap(access.location.region, prior.location.region)) continue;
          if (!HasHazard(access.kind, prior.kind)) continue;
          VarPtr prior_edge =
              prior.task_id_var_is_direct ? prior.task_id_var : CanonicalTaskId(prior.task_id_var);
          if (!prior_edge) prior_edge = prior.task_id_var;
          const bool active_dynamic_prior = prior.dynamic_producer && prior.innermost_dynamic_loop_id != 0 &&
                                            IsActiveDynamicLoopId(prior.innermost_dynamic_loop_id);
          const bool inactive_dynamic_prior =
              prior.innermost_dynamic_loop_id != 0 && !IsActiveDynamicLoopId(prior.innermost_dynamic_loop_id);
          const bool complete_dynamic_user_coverage =
              prior.dynamic_producer && (HasCompleteDynamicTaskIdUserCoverage(raw_user_edges, prior_edge) ||
                                         HasCompleteTaskIdArrayUserCoverage(raw_user_edges, prior_edge));
          const bool covered_by_user_edge =
              (ContainsVar(user_edges, prior_edge) &&
               !(prior.dynamic_producer &&
                 HasIncompleteDynamicTaskIdUserCoverage(raw_user_edges, prior_edge))) ||
              (prior.dynamic_producer &&
               (complete_dynamic_user_coverage || HasWindowedTaskIdTempUserCoverage(user_edges)));
          log_not_auto_no_dep_candidate(access, prior_edge);
          if (prior.dynamic_producer && !active_dynamic_prior && !inactive_dynamic_prior &&
              !complete_dynamic_user_coverage) {
            mark_auto_no_dep_blocked(access, "dynamic_producer_incomplete", prior_edge);
            mark_auto_output_existing_blocked(access, "dynamic_producer_incomplete", prior_edge);
          }
          if (debug_loop_carry) {
            DebugLog("hazard call=" + call->op_->name_ + " prior_task_id=" + DebugVar(prior.task_id_var) +
                     " prior_edge=" + DebugVar(prior_edge) +
                     " covered_by_user_edge=" + (covered_by_user_edge ? std::string("true") : "false") +
                     " complete_dynamic_user_coverage=" +
                     (complete_dynamic_user_coverage ? std::string("true") : "false") +
                     " active_dynamic_prior=" + (active_dynamic_prior ? std::string("true") : "false") +
                     " inactive_dynamic_prior=" + (inactive_dynamic_prior ? std::string("true") : "false") +
                     " dynamic_producer=" + (prior.dynamic_producer ? std::string("true") : "false") +
                     " direct_task_id=" + (prior.task_id_var_is_direct ? std::string("true") : "false") +
                     " current_task_id=" + DebugVar(task_id));
          }
          if (covered_by_user_edge) {
            mark_auto_no_dep_covered(access);
            mark_auto_output_existing_covered(access);
            continue;
          }
          if (in_manual_scope) {
            mark_auto_no_dep_blocked(access, "manual_scope", prior_edge);
            mark_auto_output_existing_blocked(access, "manual_scope", prior_edge);
            continue;
          }
          if (inactive_dynamic_prior) {
            DebugNoDepBucket(
                call->op_->name_, access.arg_index, "fallback",
                auto_no_dep_detail(access, prior_edge) + " fallback_reason=prior_from_inactive_dynamic_loop");
            if (debug_loop_carry) {
              DebugLog("fallback_reason=prior_from_inactive_dynamic_loop call=" + call->op_->name_ +
                       " prior_task_id=" + DebugVar(prior.task_id_var));
            }
            MarkCurrentScopeFallback();
            return call;
          }
          if (prior.dynamic_producer && !active_dynamic_prior) {
            DebugNoDepBucket(call->op_->name_, access.arg_index, "fallback",
                             auto_no_dep_detail(access, prior_edge) +
                                 " fallback_reason=dynamic_prior_producer_requires_scope_lift");
            if (debug_loop_carry) {
              DebugLog("fallback_reason=dynamic_prior_producer_requires_scope_lift call=" + call->op_->name_ +
                       " prior_task_id=" + DebugVar(prior.task_id_var));
            }
            MarkCurrentScopeFallback();
            return call;
          }
          if (!prior.task_id_var) {
            DebugNoDepBucket(
                call->op_->name_, access.arg_index, "fallback",
                auto_no_dep_detail(access, prior_edge) + " fallback_reason=missing_prior_task_id");
            if (debug_loop_carry) {
              DebugLog("fallback_reason=" +
                       std::string(prior.dynamic_producer ? "dynamic_prior_producer_missing_task_id"
                                                          : "missing_prior_task_id") +
                       " call=" + call->op_->name_ + " user_edges=" + DebugVarList(user_edges));
            }
            MarkCurrentScopeFallback();
            return call;
          }
          AppendUnique(&compiler_edges, prior_edge);
          mark_auto_no_dep_covered(access);
          mark_auto_output_existing_covered(access);
        }
      }
    }

    std::vector<int32_t> auto_no_dep_indices;
    for (const auto& [arg_index, state] : auto_no_dep_candidates) {
      if (!state.has_covered_hazard) {
        DebugNoDepBucket(call->op_->name_, arg_index, "no_covered_hazard");
        continue;
      }
      if (state.blocked) {
        DebugNoDepBucket(call->op_->name_, arg_index,
                         state.blocked_reason.empty() ? "blocked" : state.blocked_reason, "final=rejected");
        continue;
      }
      if (arg_index > static_cast<size_t>(std::numeric_limits<int32_t>::max())) continue;
      DebugNoDepBucket(call->op_->name_, arg_index, "applied");
      auto_no_dep_indices.push_back(static_cast<int32_t>(arg_index));
    }
    std::sort(auto_no_dep_indices.begin(), auto_no_dep_indices.end());

    std::vector<int32_t> auto_output_existing_indices;
    for (const auto& [arg_index, state] : auto_output_existing_candidates) {
      if (!state.has_covered_hazard) {
        DebugNoDepBucket(call->op_->name_, arg_index, "no_covered_hazard");
        continue;
      }
      if (state.blocked) {
        DebugNoDepBucket(call->op_->name_, arg_index,
                         state.blocked_reason.empty() ? "blocked" : state.blocked_reason, "final=rejected");
        continue;
      }
      if (arg_index > static_cast<size_t>(std::numeric_limits<int32_t>::max())) continue;
      DebugNoDepBucket(call->op_->name_, arg_index, "applied");
      auto_output_existing_indices.push_back(static_cast<int32_t>(arg_index));
    }
    std::sort(auto_output_existing_indices.begin(), auto_output_existing_indices.end());

    for (auto& access : accesses) {
      access.task_id_var =
          task_id ? task_id : (in_manual_scope ? nullptr : ProducerTaskIdForAccess(call, access));
      access.task_id_var_is_direct = access.task_id_var && IsTaskIdVar(access.task_id_var);
      access.dynamic_producer = loop_depth_ > 0;
      access.innermost_dynamic_loop_depth =
          dynamic_loop_depth_stack_.empty() ? 0 : dynamic_loop_depth_stack_.back();
      access.innermost_dynamic_loop_id = dynamic_loop_id_stack_.empty() ? 0 : dynamic_loop_id_stack_.back();
      prior_stack_.back().push_back(std::move(access));
    }

    if (compiler_edges.empty() && auto_no_dep_indices.empty() && auto_output_existing_indices.empty()) {
      return call;
    }

    auto new_attrs = output_attrs;
    if (!compiler_edges.empty()) {
      new_attrs = WithCompilerManualDepEdgesAttr(std::move(new_attrs), std::move(compiler_edges));
    }
    if (!auto_no_dep_indices.empty()) {
      if (!auto_no_dep_candidate_count_stack_.empty()) {
        auto_no_dep_candidate_count_stack_.back() += auto_no_dep_indices.size();
      }
      new_attrs = WithIntVectorAttr(std::move(new_attrs), kAttrAutoNoDepCandidateIndices,
                                    std::move(auto_no_dep_indices));
    }
    if (!auto_output_existing_indices.empty()) {
      if (!auto_no_dep_candidate_count_stack_.empty()) {
        auto_no_dep_candidate_count_stack_.back() += auto_output_existing_indices.size();
      }
      new_attrs = WithIntVectorAttr(std::move(new_attrs), kAttrAutoOutputExistingCandidateIndices,
                                    std::move(auto_output_existing_indices));
    }
    return std::make_shared<const Call>(call->op_, call->args_, call->kwargs_, std::move(new_attrs),
                                        call->GetType(), call->span_);
  }

 private:
  bool IsActiveDynamicLoopId(size_t id) const {
    return std::find(dynamic_loop_id_stack_.begin(), dynamic_loop_id_stack_.end(), id) !=
           dynamic_loop_id_stack_.end();
  }

  void MarkCurrentScopeFallback() {
    if (fallback_stack_.empty()) return;
    fallback_stack_.back() = true;
  }

  class CompilerDepStripper : public IRMutator {
   protected:
    ExprPtr VisitExpr_(const CallPtr& op) override {
      auto base = IRMutator::VisitExpr_(op);
      auto call = As<Call>(base);
      if (!call) return base;

      auto stripped_attrs = StripCompilerManualDepEdges(call->attrs_);
      if (stripped_attrs.size() == call->attrs_.size()) return call;
      return std::make_shared<const Call>(call->op_, call->args_, call->kwargs_, std::move(stripped_attrs),
                                          call->GetType(), call->span_);
    }

    ExprPtr VisitExpr_(const SubmitPtr& op) override {
      auto base = IRMutator::VisitExpr_(op);
      auto submit = As<Submit>(base);
      if (!submit) return base;

      auto stripped_attrs = StripCompilerManualDepEdges(submit->attrs_);
      if (stripped_attrs.size() == submit->attrs_.size()) return submit;
      return std::make_shared<const Submit>(submit->op_, submit->args_, submit->deps_, submit->kwargs_,
                                            std::move(stripped_attrs), submit->GetType(), submit->span_,
                                            submit->core_num_, submit->sync_start_,
                                            submit->allow_early_resolve_);
    }
  };

  static StmtPtr StripCompilerDeps(const StmtPtr& stmt) {
    CompilerDepStripper stripper;
    return stripper.VisitStmt(stmt);
  }

  class AutoDirectionCandidateApplier : public IRMutator {
   protected:
    ExprPtr VisitExpr_(const CallPtr& op) override {
      auto base = IRMutator::VisitExpr_(op);
      auto call = As<Call>(base);
      if (!call) return base;

      auto attrs = ApplyToAttrs(call->attrs_, call->GetArgDirections());
      if (!attrs.has_value()) return call;
      return std::make_shared<const Call>(call->op_, call->args_, call->kwargs_, std::move(*attrs),
                                          call->GetType(), call->span_);
    }

    ExprPtr VisitExpr_(const SubmitPtr& op) override {
      auto base = IRMutator::VisitExpr_(op);
      auto submit = As<Submit>(base);
      if (!submit) return base;

      auto attrs = ApplyToAttrs(submit->attrs_, submit->GetArgDirections());
      if (!attrs.has_value()) return submit;
      return std::make_shared<const Submit>(
          submit->op_, submit->args_, submit->deps_, submit->kwargs_, std::move(*attrs), submit->GetType(),
          submit->span_, submit->core_num_, submit->sync_start_, submit->allow_early_resolve_);
    }

   private:
    static std::optional<std::vector<std::pair<std::string, std::any>>> ApplyToAttrs(
        const std::vector<std::pair<std::string, std::any>>& attrs, std::vector<ArgDirection> dirs) {
      auto no_dep_indices = GetIntVectorAttr(attrs, kAttrAutoNoDepCandidateIndices);
      auto output_existing_indices = GetIntVectorAttr(attrs, kAttrAutoOutputExistingCandidateIndices);
      if (no_dep_indices.empty() && output_existing_indices.empty() && !HasAutoNoDepCandidateAttr(attrs) &&
          !HasAutoOutputExistingCandidateAttr(attrs)) {
        return std::nullopt;
      }

      for (const int32_t raw_idx : no_dep_indices) {
        if (raw_idx < 0) continue;
        const size_t idx = static_cast<size_t>(raw_idx);
        if (idx >= dirs.size()) continue;
        if (dirs[idx] == ArgDirection::Input) {
          dirs[idx] = ArgDirection::NoDep;
        }
      }
      for (const int32_t raw_idx : output_existing_indices) {
        if (raw_idx < 0) continue;
        const size_t idx = static_cast<size_t>(raw_idx);
        if (idx >= dirs.size()) continue;
        if (dirs[idx] == ArgDirection::InOut) {
          dirs[idx] = ArgDirection::OutputExisting;
        }
      }

      auto stripped = StripAttr(attrs, kAttrAutoNoDepCandidateIndices);
      stripped = StripAttr(stripped, kAttrAutoOutputExistingCandidateIndices);
      return WithArgDirectionsAttr(std::move(stripped), std::move(dirs));
    }
  };

  static StmtPtr ApplyAutoNoDepCandidates(const StmtPtr& stmt) {
    AutoDirectionCandidateApplier applier;
    return applier.VisitStmt(stmt);
  }

  VarPtr LookupTaskId(const Expr* expr) const {
    if (!task_id_by_expr_) return nullptr;
    auto it = task_id_by_expr_->find(expr);
    return it != task_id_by_expr_->end() ? it->second : nullptr;
  }

  VarPtr LookupTaskIdForVar(const ExprPtr& expr) const {
    if (!task_id_by_var_id_) return nullptr;
    auto var = AsVarLike(expr);
    if (!var) return nullptr;
    auto it = task_id_by_var_id_->find(var->UniqueId());
    return it != task_id_by_var_id_->end() ? it->second : nullptr;
  }

  VarPtr CanonicalTaskId(const VarPtr& var) const {
    if (!var) return nullptr;
    if (auto mapped = LookupTaskIdForVar(var)) return mapped;
    return IsTaskIdVar(var) ? var : nullptr;
  }

  std::vector<VarPtr> CanonicalTaskIds(const VarPtr& var) const {
    std::vector<VarPtr> canonical;
    if (!var) return canonical;
    if (task_ids_by_var_id_) {
      auto it = task_ids_by_var_id_->find(var->UniqueId());
      if (it != task_ids_by_var_id_->end()) {
        AppendAllUnique(&canonical, it->second);
        return canonical;
      }
    }
    if (auto scalar = CanonicalTaskId(var)) {
      canonical.push_back(scalar);
      return canonical;
    }
    if (task_ids_by_array_var_id_ && IsTaskIdArrayVar(var)) {
      auto it = task_ids_by_array_var_id_->find(var->UniqueId());
      if (it != task_ids_by_array_var_id_->end()) {
        AppendAllUnique(&canonical, it->second);
      }
    }
    return canonical;
  }

  std::vector<VarPtr> CanonicalizeTaskIds(const std::vector<VarPtr>& vars) const {
    std::vector<VarPtr> canonical;
    canonical.reserve(vars.size());
    std::unordered_map<uint64_t, std::unordered_set<int64_t>> dynamic_array_slots;
    for (const auto& var : vars) {
      bool has_dynamic_slots = false;
      if (task_id_dynamic_slots_by_var_id_ && var) {
        auto slots_it = task_id_dynamic_slots_by_var_id_->find(var->UniqueId());
        if (slots_it != task_id_dynamic_slots_by_var_id_->end()) {
          has_dynamic_slots = true;
          for (const auto& [array_id, slots] : slots_it->second) {
            for (const auto slot : slots) {
              dynamic_array_slots[array_id].insert(slot);
            }
          }
        }
      }
      if (has_dynamic_slots) continue;
      if (IsTaskIdVar(var)) AppendUnique(&canonical, var);
      AppendAllUnique(&canonical, CanonicalTaskIds(var));
    }
    for (const auto& [array_id, slots] : dynamic_array_slots) {
      if (!task_ids_by_array_var_id_ || !task_id_array_extent_by_var_id_) continue;
      auto extent_it = task_id_array_extent_by_var_id_->find(array_id);
      if (extent_it == task_id_array_extent_by_var_id_->end()) continue;
      if (!HasCompleteDynamicCoverage(slots, extent_it->second)) continue;
      auto expanded_it = task_ids_by_array_var_id_->find(array_id);
      if (expanded_it == task_ids_by_array_var_id_->end()) continue;
      AppendAllUnique(&canonical, expanded_it->second);
    }
    return canonical;
  }

  bool HasCompleteDynamicTaskIdUserCoverage(const std::vector<VarPtr>& raw_user_edges,
                                            const VarPtr& prior_edge) const {
    if (!prior_edge || !task_id_dynamic_slots_by_var_id_ || !task_id_array_extent_by_var_id_ ||
        !task_ids_by_array_var_id_) {
      return false;
    }

    std::unordered_map<uint64_t, std::unordered_set<int64_t>> dynamic_array_slots;
    for (const auto& edge : raw_user_edges) {
      if (!edge) continue;
      auto slots_it = task_id_dynamic_slots_by_var_id_->find(edge->UniqueId());
      if (slots_it == task_id_dynamic_slots_by_var_id_->end()) continue;
      for (const auto& [array_id, slots] : slots_it->second) {
        for (const auto slot : slots) {
          dynamic_array_slots[array_id].insert(slot);
        }
      }
    }

    for (const auto& [array_id, slots] : dynamic_array_slots) {
      auto extent_it = task_id_array_extent_by_var_id_->find(array_id);
      if (extent_it == task_id_array_extent_by_var_id_->end()) continue;
      if (!HasCompleteDynamicCoverage(slots, extent_it->second)) continue;

      auto expanded_it = task_ids_by_array_var_id_->find(array_id);
      if (expanded_it == task_ids_by_array_var_id_->end()) continue;
      if (ContainsVar(expanded_it->second, prior_edge)) return true;
    }
    return false;
  }

  bool HasCompleteTaskIdArrayUserCoverage(const std::vector<VarPtr>& raw_user_edges,
                                          const VarPtr& prior_edge) const {
    if (!prior_edge || !task_ids_by_array_var_id_ || !complete_task_id_array_var_ids_) {
      return false;
    }
    for (const auto& edge : raw_user_edges) {
      if (!edge || !IsTaskIdArrayVar(edge)) continue;
      if (complete_task_id_array_var_ids_->find(edge->UniqueId()) == complete_task_id_array_var_ids_->end()) {
        continue;
      }
      auto expanded_it = task_ids_by_array_var_id_->find(edge->UniqueId());
      if (expanded_it == task_ids_by_array_var_id_->end()) continue;
      if (ContainsVar(expanded_it->second, prior_edge)) return true;
    }
    return false;
  }

  bool HasIncompleteDynamicTaskIdUserCoverage(const std::vector<VarPtr>& raw_user_edges,
                                              const VarPtr& prior_edge) const {
    if (!prior_edge || !task_id_dynamic_slots_by_var_id_ || !task_id_array_extent_by_var_id_ ||
        !task_ids_by_array_var_id_) {
      return false;
    }

    std::unordered_map<uint64_t, std::unordered_set<int64_t>> dynamic_array_slots;
    for (const auto& edge : raw_user_edges) {
      if (!edge) continue;
      auto slots_it = task_id_dynamic_slots_by_var_id_->find(edge->UniqueId());
      if (slots_it == task_id_dynamic_slots_by_var_id_->end()) continue;
      for (const auto& [array_id, slots] : slots_it->second) {
        for (const auto slot : slots) {
          dynamic_array_slots[array_id].insert(slot);
        }
      }
    }

    for (const auto& [array_id, slots] : dynamic_array_slots) {
      auto extent_it = task_id_array_extent_by_var_id_->find(array_id);
      if (extent_it == task_id_array_extent_by_var_id_->end()) continue;
      if (HasCompleteDynamicCoverage(slots, extent_it->second)) continue;

      auto expanded_it = task_ids_by_array_var_id_->find(array_id);
      if (expanded_it == task_ids_by_array_var_id_->end()) continue;
      if (ContainsVar(expanded_it->second, prior_edge)) return true;
    }
    return false;
  }

  bool HasWindowedTaskIdTempUserCoverage(const std::vector<VarPtr>& user_edges) const {
    size_t temp_count = 0;
    for (const auto& edge : user_edges) {
      if (!edge || !IsTaskIdVar(edge)) continue;
      if (submit_task_id_var_ids_.find(edge->UniqueId()) != submit_task_id_var_ids_.end()) continue;
      ++temp_count;
      if (temp_count >= kMinWindowedTaskIdTempDeps) return true;
    }
    return false;
  }

  VarPtr ProducerTaskIdForAccess(const CallPtr& call, const StorageAccess& access) const {
    if (access.kind == AccessKind::Read) return nullptr;
    auto dirs = call->GetArgDirections();
    if (dirs.size() != call->args_.size()) return nullptr;
    for (size_t i = 0; i < dirs.size(); ++i) {
      if (dirs[i] != ArgDirection::OutputExisting && dirs[i] != ArgDirection::InOut) continue;
      auto var = AsVarLike(call->args_[i]);
      if (!var) continue;
      auto resolved = storage_ ? storage_->ResolveExprStatus(call->args_[i]) : ResolvedLocation{};
      if (resolved.status != LocationStatus::Known) continue;
      for (const auto& alternative : resolved.location.alternatives) {
        if (!storage_ || !storage_->MayAlias(access.location.root, alternative.root)) continue;
        if (!RegionsMayOverlap(access.location.region, alternative.region)) continue;
        return var;
      }
    }
    return nullptr;
  }

  AccessSummary SummarizeAccesses(const CallPtr& call, const std::vector<VarPtr>& user_edges,
                                  bool* needs_fallback) const {
    AccessSummary out;
    auto dirs = call->GetArgDirections();
    if (dirs.size() != call->args_.size()) return out;

    const bool debug_loop_carry = AutoDepsLoopCarryDebugEnabled();
    if (debug_loop_carry) {
      DebugLog("summarize_accesses_start call=" + call->op_->name_ +
               " args=" + std::to_string(call->args_.size()) + " user_edges=" + DebugVarList(user_edges));
    }

    for (size_t i = 0; i < dirs.size(); ++i) {
      std::optional<AccessKind> kind;
      switch (dirs[i]) {
        case ArgDirection::Input:
          kind = AccessKind::Read;
          break;
        case ArgDirection::Output:
        case ArgDirection::OutputExisting:
          kind = AccessKind::Write;
          break;
        case ArgDirection::InOut:
          kind = AccessKind::ReadWrite;
          break;
        case ArgDirection::NoDep:
        case ArgDirection::Scalar:
          break;
      }
      if (kind.has_value()) {
        auto resolved = storage_ ? storage_->ResolveExprStatus(call->args_[i]) : ResolvedLocation{};
        if (debug_loop_carry) {
          DebugLog("summarize_access_arg call=" + call->op_->name_ + " index=" + std::to_string(i) +
                   " status=" + DebugLocationStatus(resolved.status) +
                   " original_task_id=" + DebugVar(LookupTaskIdForVar(call->args_[i])));
        }
        if (resolved.status != LocationStatus::Known) {
          const VarPtr original_task_id = LookupTaskIdForVar(call->args_[i]);
          const bool covered_by_user_edge =
              kind == AccessKind::Read && ContainsVar(user_edges, original_task_id);
          if (dirs[i] == ArgDirection::Input || dirs[i] == ArgDirection::InOut ||
              dirs[i] == ArgDirection::Output || dirs[i] == ArgDirection::OutputExisting) {
            DebugNoDepBucket(call->op_->name_, i, "fallback",
                             "direction=" + ArgDirectionToString(dirs[i]) + " access=" +
                                 DebugAccessKind(*kind) + " status=" + DebugLocationStatus(resolved.status) +
                                 " original_task_id=" + DebugVar(original_task_id) +
                                 " covered_by_user_edge=" + (covered_by_user_edge ? "true" : "false") +
                                 " fallback_reason=unresolved_location");
          }
          if (debug_loop_carry) {
            DebugLog("summarize_access_unresolved call=" + call->op_->name_ + " index=" + std::to_string(i) +
                     " covered_by_user_edge=" + (covered_by_user_edge ? std::string("true") : "false") +
                     " original_task_id=" + DebugVar(original_task_id));
          }
          if (covered_by_user_edge) {
            continue;
          }
          if (needs_fallback) *needs_fallback = true;
          if (debug_loop_carry) {
            DebugLog("summarize_access_fallback call=" + call->op_->name_ + " index=" + std::to_string(i) +
                     " reason=unresolved_location");
          }
          return {};
        }
        for (const auto& alternative : resolved.location.alternatives) {
          out.accesses.push_back(StorageAccess{alternative, *kind, nullptr, false, false, 0, 0, i});
        }
      }
    }
    return out;
  }

  ProgramPtr program_;
  const StorageRootAnalysis* storage_;
  const std::unordered_map<const Expr*, VarPtr>* task_id_by_expr_;
  const std::unordered_map<uint64_t, VarPtr>* task_id_by_var_id_;
  const std::unordered_map<uint64_t, std::vector<VarPtr>>* task_ids_by_var_id_;
  const std::unordered_map<uint64_t, std::unordered_map<uint64_t, std::unordered_set<int64_t>>>*
      task_id_dynamic_slots_by_var_id_;
  const std::unordered_map<uint64_t, int64_t>* task_id_array_extent_by_var_id_;
  const std::unordered_map<uint64_t, std::vector<VarPtr>>* task_ids_by_array_var_id_;
  const std::unordered_set<uint64_t>* complete_task_id_array_var_ids_;
  std::unordered_set<uint64_t> submit_task_id_var_ids_;
  bool analyze_auto_scopes_ = false;
  bool analyze_whole_body_as_auto_scope_ = false;
  std::vector<std::vector<StorageAccess>> prior_stack_;
  std::vector<bool> scope_manual_stack_;
  std::vector<bool> fallback_stack_;
  std::vector<size_t> auto_no_dep_candidate_count_stack_;
  VarPtr current_assign_lhs_;
  size_t loop_depth_ = 0;
  std::vector<size_t> dynamic_loop_depth_stack_;
  std::vector<size_t> dynamic_loop_id_stack_;
  size_t next_dynamic_loop_id_ = 1;
};

}  // namespace

namespace pass {

Pass AutoDeriveTaskDependencies(bool analyze_auto_scopes) {
  auto pass_func = [analyze_auto_scopes](const ProgramPtr& program) -> ProgramPtr {
    if (!program) return program;

    auto new_functions = program->functions_;
    bool changed = false;

    for (auto& [gvar, func] : new_functions) {
      (void)gvar;
      if (!func || !func->body_) continue;

      StorageRootAnalysis storage(program);
      storage.Initialize(func->params_);
      storage.VisitStmt(func->body_);

      SubmitTaskIdCollector task_ids;
      task_ids.VisitStmt(func->body_);

      const bool analyze_whole_body_as_auto_scope = analyze_auto_scopes &&
                                                    func->func_type_ == FunctionType::Orchestration &&
                                                    func->GetAttr<bool>("auto_scope", true);
      AutoDepMutator mutator(program, &storage, &task_ids.task_id_by_expr(), &task_ids.task_id_by_var_id(),
                             &task_ids.task_ids_by_var_id(), &task_ids.task_id_dynamic_slots_by_var_id(),
                             &task_ids.task_id_array_extent_by_var_id(), &task_ids.task_ids_by_array_var_id(),
                             &task_ids.complete_task_id_array_var_ids(), analyze_auto_scopes,
                             analyze_whole_body_as_auto_scope);
      auto new_body = mutator.AnalyzeBody(func->body_);
      if (new_body.get() == func->body_.get()) continue;

      changed = true;
      func = std::make_shared<Function>(func->name_, func->params_, func->param_directions_,
                                        func->return_types_, new_body, func->span_, func->func_type_,
                                        func->level_, func->role_, func->attrs_);
    }

    if (!changed) return program;
    return std::make_shared<Program>(std::move(new_functions), program->name_, program->span_);
  };

  return CreateProgramPass(pass_func, "AutoDeriveTaskDependencies", kAutoDeriveTaskDependenciesProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
