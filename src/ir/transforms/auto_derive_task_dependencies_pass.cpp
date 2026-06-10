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
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/tensor_view_semantics.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

using ::pypto::codegen::ComputeGroupEffectiveDirections;
using ::pypto::codegen::IsBuiltinOp;

enum class AccessKind { Read, Write, ReadWrite };

enum class RegionKind { Unknown, Full, Box };

constexpr size_t kMaxStaticRootAlternatives = 4;

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
  bool dynamic_producer = false;
};

struct AccessSummary {
  std::vector<StorageAccess> accesses;
};

bool IsTensorType(const TypePtr& type) { return As<TensorType>(type) != nullptr; }

AccessRegion UnknownRegion() { return AccessRegion{RegionKind::Unknown, {}, {}}; }

AccessRegion FullRegion() { return AccessRegion{RegionKind::Full, {}, {}}; }

StorageLocation UnknownLocation() { return StorageLocation{}; }

StorageLocation SingleLocation(const Var* root, AccessRegion region) {
  if (!root) return UnknownLocation();
  return StorageLocation{{StorageAlternative{root, std::move(region)}}};
}

bool HasLocation(const StorageLocation& location) { return !location.alternatives.empty(); }

bool IsDynamicAccessOp(const std::string& op_name) {
  return op_name == "tensor.gather" || op_name == "tensor.gather_mask" ||
         op_name == "tensor.gather_compare" || op_name == "tensor.scatter_update" ||
         op_name == "tile.gather" || op_name == "tile.gather_mask" || op_name == "tile.gather_compare" ||
         op_name == "tile.scatter_update" || op_name == "tile.mscatter";
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
    return UnknownRegion();
  }

  if (parent.kind == RegionKind::Unknown) {
    return UnknownRegion();
  }

  std::vector<int64_t> absolute_offsets;
  absolute_offsets.reserve(offsets->size());
  if (parent.kind == RegionKind::Full) {
    absolute_offsets = *offsets;
  } else {
    if (parent.offsets.size() != offsets->size()) return UnknownRegion();
    for (size_t i = 0; i < offsets->size(); ++i) {
      auto absolute = AddInt64(parent.offsets[i], (*offsets)[i]);
      if (!absolute.has_value()) return UnknownRegion();
      absolute_offsets.push_back(*absolute);
    }
  }

  return AccessRegion{RegionKind::Box, std::move(absolute_offsets), std::move(*shape)};
}

bool RegionsMayOverlap(const AccessRegion& lhs, const AccessRegion& rhs) {
  if (lhs.kind == RegionKind::Unknown || rhs.kind == RegionKind::Unknown) return true;
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

bool SameStaticConstInt(const ExprPtr& lhs, const ExprPtr& rhs) {
  auto lhs_const = As<ConstInt>(lhs);
  auto rhs_const = As<ConstInt>(rhs);
  return lhs_const && rhs_const && lhs_const->value_ == rhs_const->value_;
}

bool IsPackedNdTensorView(const TensorTypePtr& tensor_type) {
  if (!tensor_type || !tensor_type->tensor_view_.has_value()) return true;

  const auto& view = tensor_type->tensor_view_.value();
  if (view.layout != TensorLayout::ND) return false;
  if (!view.valid_shape.empty()) return false;
  if (view.pad != PadValue::null) return false;

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

StorageLocation MaybeUnknownRegionsForTensorType(const StorageLocation& location, const TypePtr& type) {
  auto tensor_type = As<TensorType>(type);
  if (!tensor_type || IsPackedNdTensorView(tensor_type)) return location;
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

bool ContainsVar(const std::vector<VarPtr>& vars, const VarPtr& candidate) {
  if (!candidate) return false;
  for (const auto& var : vars) {
    if (var && var->UniqueId() == candidate->UniqueId()) return true;
  }
  return false;
}

bool AutoDepsLoopCarryDebugEnabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("PYPTO_AUTO_DEPS_LOOP_CARRY_DEBUG");
    if (!value) return false;
    std::string text(value);
    return !text.empty() && text != "0" && text != "false" && text != "False";
  }();
  return enabled;
}

void DebugLog(const std::string& message) {
  if (!AutoDepsLoopCarryDebugEnabled()) return;
  std::cerr << "[auto-deps-loop-debug] " << message << std::endl;
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

void AppendUnique(std::vector<VarPtr>* vars, const VarPtr& candidate) {
  if (!vars || !candidate || ContainsVar(*vars, candidate)) return;
  vars->push_back(candidate);
}

std::vector<std::pair<std::string, std::any>> StripCompilerManualDepEdges(
    const std::vector<std::pair<std::string, std::any>>& attrs) {
  std::vector<std::pair<std::string, std::any>> stripped;
  stripped.reserve(attrs.size());
  for (const auto& attr : attrs) {
    if (attr.first != kAttrCompilerManualDepEdges) {
      stripped.push_back(attr);
    }
  }
  return stripped;
}

bool HasCompilerManualDepEdgesAttr(const std::vector<std::pair<std::string, std::any>>& attrs) {
  for (const auto& [k, v] : attrs) {
    (void)v;
    if (k == kAttrCompilerManualDepEdges) return true;
  }
  return false;
}

bool HasHazard(AccessKind current, AccessKind prior) {
  const bool current_writes = current == AccessKind::Write || current == AccessKind::ReadWrite;
  const bool current_reads = current == AccessKind::Read || current == AccessKind::ReadWrite;
  const bool prior_writes = prior == AccessKind::Write || prior == AccessKind::ReadWrite;
  const bool prior_reads = prior == AccessKind::Read || prior == AccessKind::ReadWrite;
  return (current_reads && prior_writes) || (current_writes && prior_reads) ||
         (current_writes && prior_writes);
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
        RegisterVarLocation(op->return_vars_[i], std::move(merged));
      }
    }
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    std::vector<StorageLocation> init_locations;
    init_locations.reserve(op->iter_args_.size());
    for (size_t i = 0; i < op->iter_args_.size(); ++i) {
      auto location = ResolveExpr(op->iter_args_[i]->initValue_);
      init_locations.push_back(location);
      if (HasLocation(location)) {
        RegisterVarLocation(op->iter_args_[i], std::move(location));
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
        RegisterVarLocation(op->return_vars_[i], std::move(location));
      }
    }
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    std::vector<StorageLocation> init_locations;
    init_locations.reserve(op->iter_args_.size());
    for (size_t i = 0; i < op->iter_args_.size(); ++i) {
      auto location = ResolveExpr(op->iter_args_[i]->initValue_);
      init_locations.push_back(location);
      if (HasLocation(location)) {
        RegisterVarLocation(op->iter_args_[i], std::move(location));
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
        RegisterVarLocation(op->return_vars_[i], std::move(location));
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
      if (op_name == "tensor.create") {
        RegisterVarLocation(op->var_, MaybeUnknownRegionsForTensorType(
                                          SingleLocation(op->var_.get(), FullRegion()), op->var_->GetType()));
      } else if (op_name == "tensor.slice") {
        if (call->args_.size() >= 3) {
          auto parent = ResolveExpr(call->args_[0]);
          if (HasLocation(parent)) {
            RegisterVarLocation(
                op->var_, MaybeUnknownRegionsForTensorType(
                              SliceLocation(parent, call->args_[1], call->args_[2]), op->var_->GetType()));
          }
        }
      } else if (op_name == "tensor.assemble") {
        if (!call->args_.empty()) {
          auto base = ResolveExpr(call->args_[0]);
          if (HasLocation(base)) {
            RegisterVarLocation(op->var_, MaybeUnknownRegionsForTensorType(base, op->var_->GetType()));
          }
        }
      } else if (IsDynamicAccessOp(op_name)) {
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

  void RegisterVarLocation(const VarPtr& var, StorageLocation location) {
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

    if (auto call = As<Call>(op->value_)) {
      RecordTaskTupleProducer(op->var_, call);
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

 private:
  VarPtr CanonicalTaskIdForExpr(const ExprPtr& expr) const {
    auto var = AsVarLike(expr);
    if (!var) return nullptr;

    auto it = task_id_by_var_id_.find(var->UniqueId());
    if (it != task_id_by_var_id_.end()) return it->second;

    if (IsTaskIdVar(var)) return var;
    return nullptr;
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
  }

  std::unordered_map<const Var*, ExprPtr> task_expr_by_tuple_;
  std::unordered_map<const Var*, std::unordered_map<int, VarPtr>> tuple_get_by_tuple_;
  std::unordered_map<const Expr*, VarPtr> task_id_by_expr_;
  std::unordered_map<uint64_t, VarPtr> task_id_by_var_id_;
};

class AutoDepMutator : public IRMutator {
 public:
  AutoDepMutator(ProgramPtr program, const StorageRootAnalysis* storage,
                 const std::unordered_map<const Expr*, VarPtr>* task_id_by_expr,
                 const std::unordered_map<uint64_t, VarPtr>* task_id_by_var_id, bool analyze_auto_scopes,
                 bool analyze_whole_body_as_auto_scope)
      : program_(std::move(program)),
        storage_(storage),
        task_id_by_expr_(task_id_by_expr),
        task_id_by_var_id_(task_id_by_var_id),
        analyze_auto_scopes_(analyze_auto_scopes),
        analyze_whole_body_as_auto_scope_(analyze_whole_body_as_auto_scope) {}

  StmtPtr AnalyzeBody(const StmtPtr& body) {
    INTERNAL_CHECK(body != nullptr) << "Internal error: AutoDepMutator received null function body";
    if (!analyze_whole_body_as_auto_scope_ || As<RuntimeScopeStmt>(body)) {
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
    ++loop_depth_;
    auto result = IRMutator::VisitStmt_(op);
    --loop_depth_;
    return result;
  }

  StmtPtr VisitStmt_(const WhileStmtPtr& op) override {
    if (prior_stack_.empty()) return IRMutator::VisitStmt_(op);
    ++loop_depth_;
    auto result = IRMutator::VisitStmt_(op);
    --loop_depth_;
    return result;
  }

  StmtPtr VisitStmt_(const RuntimeScopeStmtPtr& op) override {
    if (!op->manual_ && !analyze_auto_scopes_) {
      return IRMutator::VisitStmt_(op);
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
    INTERNAL_CHECK_SPAN(body, span) << "RuntimeScopeStmt has null body";
    auto new_body = VisitStmt(body);
    INTERNAL_CHECK_SPAN(new_body, span) << "RuntimeScopeStmt body mutated to null";
    const bool fallback = fallback_stack_.back();
    fallback_stack_.pop_back();
    scope_manual_stack_.pop_back();
    prior_stack_.pop_back();
    if (fallback) {
      DebugLog("fallback_runtime_scope name_hint=" + name_hint);
      auto stripped_body = StripCompilerDeps(new_body);
      if (is_virtual_whole_body) return stripped_body;
      return std::make_shared<const RuntimeScopeStmt>(false, name_hint, std::move(stripped_body), span,
                                                      std::move(leading_comments), std::move(attrs));
    }
    DebugLog("keep_runtime_scope name_hint=" + name_hint +
             " manual=" + (manual ? std::string("true") : std::string("false")));
    if (is_virtual_whole_body) return new_body;
    if (new_body.get() != body.get()) {
      return std::make_shared<const RuntimeScopeStmt>(manual, name_hint, std::move(new_body), span,
                                                      std::move(leading_comments), std::move(attrs));
    }
    return original_scope ? original_scope : body;
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
    if (!rewritten_call || !HasCompilerManualDepEdgesAttr(rewritten_call->attrs_)) return submit;
    return std::make_shared<const Submit>(submit->op_, submit->args_, submit->deps_, submit->kwargs_,
                                          rewritten_call->attrs_, submit->GetType(), submit->span_,
                                          submit->core_num_, submit->sync_start_);
  }

  ExprPtr AnalyzeCallLike(const CallPtr& call, const Expr* identity_key,
                          const std::vector<std::pair<std::string, std::any>>& output_attrs) {
    if (prior_stack_.empty()) return call;
    if (IsBuiltinOp(call->op_->name_)) return call;

    VarPtr task_id = LookupTaskId(identity_key);
    if (!task_id) {
      const bool in_manual_scope = !scope_manual_stack_.empty() && scope_manual_stack_.back();
      if (!in_manual_scope || IsTaskIdVar(current_assign_lhs_)) {
        task_id = current_assign_lhs_;
      }
    }
    bool needs_fallback = false;
    auto user_edges = CanonicalizeTaskIds(GetDepAttr(call, kAttrManualDepEdges));
    const bool debug_loop_carry = AutoDepsLoopCarryDebugEnabled();
    auto summary = SummarizeAccesses(call, user_edges, &needs_fallback);
    if (debug_loop_carry) {
      DebugLog("call=" + call->op_->name_ + " loop_depth=" + std::to_string(loop_depth_) +
               " task_id=" + DebugVar(task_id) + " user_edges=" + DebugVarList(user_edges) +
               " summary_accesses=" + std::to_string(summary.accesses.size()) +
               " needs_fallback_from_summary=" + (needs_fallback ? "true" : "false"));
    }
    if (needs_fallback) {
      if (debug_loop_carry) {
        DebugLog("fallback_reason=summary_unknown_location call=" + call->op_->name_);
      }
      fallback_stack_.back() = true;
      return call;
    }
    auto& accesses = summary.accesses;
    if (accesses.empty()) return call;

    std::vector<VarPtr> compiler_edges;
    for (const auto& access : accesses) {
      for (const auto& prior : prior_stack_.back()) {
        if (!storage_ || !storage_->MayAlias(access.location.root, prior.location.root)) continue;
        if (!RegionsMayOverlap(access.location.region, prior.location.region)) continue;
        if (!HasHazard(access.kind, prior.kind)) continue;
        const bool covered_by_user_edge = ContainsVar(user_edges, prior.task_id_var);
        if (debug_loop_carry) {
          DebugLog("hazard call=" + call->op_->name_ + " prior_task_id=" + DebugVar(prior.task_id_var) +
                   " covered_by_user_edge=" + (covered_by_user_edge ? std::string("true") : "false") +
                   " dynamic_producer=" + (prior.dynamic_producer ? std::string("true") : "false") +
                   " current_task_id=" + DebugVar(task_id));
        }
        if (covered_by_user_edge) continue;
        if (prior.dynamic_producer) {
          const bool in_manual_scope = !scope_manual_stack_.empty() && scope_manual_stack_.back();
          if (!in_manual_scope) {
            if (debug_loop_carry) {
              DebugLog("skip_auto_scope_dynamic_prior_producer call=" + call->op_->name_ +
                       " prior_task_id=" + DebugVar(prior.task_id_var));
            }
            continue;
          }
          if (debug_loop_carry) {
            DebugLog("fallback_reason=dynamic_prior_producer_requires_scope_lift call=" + call->op_->name_ +
                     " prior_task_id=" + DebugVar(prior.task_id_var));
          }
          fallback_stack_.back() = true;
          return call;
        }
        if (!prior.task_id_var) {
          if (debug_loop_carry) {
            DebugLog("fallback_reason=" +
                     std::string(prior.dynamic_producer ? "dynamic_prior_producer_missing_task_id"
                                                        : "missing_prior_task_id") +
                     " call=" + call->op_->name_ + " user_edges=" + DebugVarList(user_edges));
          }
          fallback_stack_.back() = true;
          return call;
        }
        AppendUnique(&compiler_edges, prior.task_id_var);
      }
    }

    for (auto& access : accesses) {
      access.task_id_var = task_id;
      access.dynamic_producer = loop_depth_ > 0;
      prior_stack_.back().push_back(std::move(access));
    }

    if (compiler_edges.empty()) {
      return call;
    }

    auto new_attrs = WithCompilerManualDepEdgesAttr(output_attrs, std::move(compiler_edges));
    return std::make_shared<const Call>(call->op_, call->args_, call->kwargs_, std::move(new_attrs),
                                        call->GetType(), call->span_);
  }

 private:
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
                                            submit->core_num_, submit->sync_start_);
    }
  };

  static StmtPtr StripCompilerDeps(const StmtPtr& stmt) {
    CompilerDepStripper stripper;
    return stripper.VisitStmt(stmt);
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

  std::vector<VarPtr> CanonicalizeTaskIds(const std::vector<VarPtr>& vars) const {
    std::vector<VarPtr> canonical;
    canonical.reserve(vars.size());
    for (const auto& var : vars) {
      AppendUnique(&canonical, CanonicalTaskId(var));
    }
    return canonical;
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
          out.accesses.push_back(StorageAccess{alternative, *kind, nullptr});
        }
      }
    }
    return out;
  }

  ProgramPtr program_;
  const StorageRootAnalysis* storage_;
  const std::unordered_map<const Expr*, VarPtr>* task_id_by_expr_;
  const std::unordered_map<uint64_t, VarPtr>* task_id_by_var_id_;
  bool analyze_auto_scopes_ = false;
  bool analyze_whole_body_as_auto_scope_ = false;
  std::vector<std::vector<StorageAccess>> prior_stack_;
  std::vector<bool> scope_manual_stack_;
  std::vector<bool> fallback_stack_;
  VarPtr current_assign_lhs_;
  size_t loop_depth_ = 0;
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
                             analyze_auto_scopes, analyze_whole_body_as_auto_scope);
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
