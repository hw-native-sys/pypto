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

#include "pypto/ir/transforms/auto_layout/writer.h"

#include <algorithm>
#include <any>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/auto_layout/extractor.h"
#include "pypto/ir/transforms/auto_layout/model.h"
#include "pypto/ir/transforms/auto_layout/solver.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/deep_clone_utils.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/normalize_stmt_structure.h"
#include "pypto/ir/transforms/utils/return_lineage_utils.h"
#include "pypto/ir/transforms/utils/tensor_view_semantics.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace auto_layout {

namespace {

constexpr const char* kOpTensorCreate = "tensor.create";
constexpr const char* kOpTensorView = "tensor.view";
constexpr const char* kOpTileCreate = "tile.create";
constexpr const char* kOpTileLoad = "tile.load";
constexpr const char* kOpTileStore = "tile.store";
constexpr const char* kOpTileSlice = "tile.slice";
constexpr const char* kOpTileMove = "tile.move";
constexpr const char* kOpTileTransposeView = "tile.transpose_view";
constexpr const char* kOpTileTranspose = "tile.transpose";
constexpr const char* kOpTileFull = "tile.full";
constexpr const char* kOpTileReshape = "tile.reshape";
constexpr const char* kOpTileRowExpand = "tile.row_expand";
constexpr const char* kOpTileRowExpandAdd = "tile.row_expand_add";

TensorTypePtr WithTensorLayout(const TensorTypePtr& tensor_type, TensorLayout layout) {
  if (IsA<DistributedTensorType>(tensor_type) || detail::TensorTypeLayout(tensor_type) == layout) {
    return tensor_type;
  }
  const auto& old_view = tensor_type->tensor_view_;
  if (layout == TensorLayout::ND &&
      (!old_view.has_value() ||
       (old_view->stride.empty() && old_view->valid_shape.empty() && old_view->pad == PadValue::null))) {
    return std::make_shared<TensorType>(tensor_type->shape_, tensor_type->dtype_, tensor_type->memref_,
                                        std::nullopt);
  }
  std::vector<ExprPtr> valid_shape = old_view.has_value() ? old_view->valid_shape : std::vector<ExprPtr>{};
  PadValue pad = old_view.has_value() ? old_view->pad : PadValue::null;
  TensorView new_view(std::vector<ExprPtr>{}, layout, std::move(valid_shape), pad);
  return std::make_shared<TensorType>(tensor_type->shape_, tensor_type->dtype_, tensor_type->memref_,
                                      std::move(new_view));
}

TypePtr RemapTypeWithSubst(const TypePtr& type, const std::unordered_map<const Var*, ExprPtr>& subst) {
  if (!type || subst.empty()) {
    return type;
  }
  if (auto tup = As<TupleType>(type)) {
    std::vector<TypePtr> elems;
    elems.reserve(tup->types_.size());
    bool changed = false;
    for (const auto& elem : tup->types_) {
      TypePtr remapped = RemapTypeWithSubst(elem, subst);
      changed |= remapped.get() != elem.get();
      elems.push_back(std::move(remapped));
    }
    return changed ? std::make_shared<TupleType>(std::move(elems)) : type;
  }
  return CloneTypeWithMemRefAndRemapExprs(
      type, GetTypeMemRef(type), [&](const ExprPtr& e) { return transform_utils::Substitute(e, subst); });
}

TypePtr WithVecMemorySpace(const TypePtr& type, const std::optional<TileView>& tile_view = std::nullopt) {
  auto tt = As<TileType>(type);
  if (!tt) {
    return type;
  }
  return std::make_shared<TileType>(tt->shape_, tt->dtype_, tt->memref_,
                                    tile_view.has_value() ? tile_view : tt->tile_view_, MemorySpace::Vec);
}

TileView ColMajorTileView(const TileTypePtr& tile_type) {
  const auto& old = tile_type->tile_view_;
  std::vector<ExprPtr> valid_shape =
      (old.has_value() && !old->valid_shape.empty()) ? old->valid_shape : tile_type->shape_;
  std::vector<ExprPtr> stride = old.has_value() ? old->stride : std::vector<ExprPtr>{};
  ExprPtr start_offset = old.has_value() ? old->start_offset : nullptr;
  PadValue pad = old.has_value() ? old->pad : PadValue::null;
  CompactMode compact = old.has_value() ? old->compact : CompactMode::null;
  return TileView(std::move(valid_shape), std::move(stride), std::move(start_offset), TileLayout::col_major,
                  TileLayout::none_box, 512, pad, compact);
}

CallPtr CreateOpCallWithAttrs(const std::string& op_name, const std::vector<ExprPtr>& args,
                              const std::vector<std::pair<std::string, std::any>>& kwargs,
                              const std::vector<std::pair<std::string, std::any>>& attrs, const Span& span) {
  CallPtr created = OpRegistry::GetInstance().Create(op_name, args, kwargs, span);
  if (attrs.empty()) {
    return created;
  }
  return std::make_shared<Call>(created->op_, created->args_, created->kwargs_, attrs, created->GetType(),
                                created->span_);
}

CallPtr WithCallType(const CallPtr& call, const TypePtr& new_type) {
  if (call->GetType().get() == new_type.get()) {
    return call;
  }
  return std::make_shared<Call>(call->op_, call->args_, call->kwargs_, call->attrs_, new_type, call->span_);
}

std::vector<std::pair<std::string, std::any>> ReplaceOrInsertKwarg(
    std::vector<std::pair<std::string, std::any>> kwargs, const std::string& key, std::any value) {
  for (auto& [k, v] : kwargs) {
    if (k == key) {
      v = std::move(value);
      return kwargs;
    }
  }
  kwargs.emplace_back(key, std::move(value));
  return kwargs;
}

bool UsesBlockedNzCoordinates(const TypePtr& type) {
  auto tensor = AsTensorTypeLike(type);
  return tensor && tensor_view_semantics::IsNzBlocked(*tensor);
}

void CollectExprVars(const ExprPtr& expr, std::unordered_set<const Var*>& seen,
                     std::vector<const Var*>& order) {
  if (!expr) {
    return;
  }
  if (auto v = AsVarLike(expr)) {
    if (seen.insert(v.get()).second) {
      order.push_back(v.get());
    }
    return;
  }
  if (auto tup = As<MakeTuple>(expr)) {
    for (const auto& el : tup->elements_) {
      CollectExprVars(el, seen, order);
    }
    return;
  }
  if (auto cast = As<Cast>(expr)) {
    CollectExprVars(cast->operand_, seen, order);
    return;
  }
  if (auto bin = As<BinaryExpr>(expr)) {
    CollectExprVars(bin->left_, seen, order);
    CollectExprVars(bin->right_, seen, order);
  }
}

std::unordered_set<const Var*> NzOffsetScalarVars(
    const StmtPtr& body, const std::vector<std::pair<VarPtr, ParamDirection>>& new_params,
    const std::vector<VarPtr>& old_params) {
  std::unordered_set<const Var*> nz_tensor_vars;
  for (size_t i = 0; i < new_params.size() && i < old_params.size(); ++i) {
    auto tt = AsTensorTypeLike(new_params[i].first->GetType());
    if (tt && detail::TensorTypeLayout(tt) == TensorLayout::NZ && !tensor_view_semantics::IsNzBlocked(*tt)) {
      nz_tensor_vars.insert(old_params[i].get());
    }
  }
  if (nz_tensor_vars.empty()) {
    return {};
  }

  std::unordered_map<const Var*, ExprPtr> scalar_defs;
  std::unordered_map<const Var*, std::pair<ExprPtr, ExprPtr>> loop_bounds;
  std::unordered_set<const Var*> expanded;
  std::vector<const Var*> worklist;

  auto walk_expr = [&](auto& self, const ExprPtr& expr) -> void {
    auto call = As<Call>(expr);
    if (!call) {
      return;
    }
    for (const auto& arg : call->args_) {
      self(self, arg);
    }
    if (call->op_->name_ == kOpTileLoad && call->args_.size() >= 2) {
      if (auto src = AsVarLike(call->args_[0]); src && nz_tensor_vars.count(src.get()) != 0) {
        CollectExprVars(call->args_[1], expanded, worklist);
        if (call->args_.size() >= 4) {
          CollectExprVars(call->args_[3], expanded, worklist);
        }
      }
    } else if (call->op_->name_ == kOpTileStore && call->args_.size() >= 3) {
      if (auto dst = AsVarLike(call->args_[2]); dst && nz_tensor_vars.count(dst.get()) != 0) {
        CollectExprVars(call->args_[1], expanded, worklist);
      }
    }
  };

  auto walk_stmt = [&](auto& self, const StmtPtr& stmt) -> void {
    if (!stmt) {
      return;
    }
    if (auto asg = As<AssignStmt>(stmt)) {
      if (As<ScalarType>(asg->var_->GetType())) {
        scalar_defs.emplace(asg->var_.get(), asg->value_);
      } else if (AsTensorTypeLike(asg->var_->GetType())) {
        if (auto src = AsVarLike(asg->value_); src && nz_tensor_vars.count(src.get()) != 0) {
          nz_tensor_vars.insert(asg->var_.get());
        }
      }
      walk_expr(walk_expr, asg->value_);
    } else if (auto fs = As<ForStmt>(stmt)) {
      loop_bounds.emplace(fs->loop_var_.get(), std::pair<ExprPtr, ExprPtr>{fs->start_, fs->step_});
    } else if (auto es = As<EvalStmt>(stmt)) {
      walk_expr(walk_expr, es->expr_);
    } else if (auto rs = As<ReturnStmt>(stmt)) {
      for (const auto& v : rs->value_) {
        walk_expr(walk_expr, v);
      }
    }
    for (const auto& child : detail::IterChildStmts(stmt)) {
      self(self, child);
    }
  };
  walk_stmt(walk_stmt, body);

  while (!worklist.empty()) {
    const Var* v = worklist.back();
    worklist.pop_back();
    if (auto def_it = scalar_defs.find(v); def_it != scalar_defs.end()) {
      CollectExprVars(def_it->second, expanded, worklist);
    }
    if (auto lb_it = loop_bounds.find(v); lb_it != loop_bounds.end()) {
      CollectExprVars(lb_it->second.first, expanded, worklist);
      CollectExprVars(lb_it->second.second, expanded, worklist);
    }
  }
  return expanded;
}

std::vector<ExprPtr> NzBackingNdShapeExprs(const std::vector<ExprPtr>& shape, int64_t c0, const Span& span) {
  auto static_dims = detail::StaticShape(shape);
  if (static_dims.has_value() && static_dims->size() >= 2) {
    int64_t batch = 1;
    for (size_t i = 0; i + 2 < static_dims->size(); ++i) {
      batch *= (*static_dims)[i];
    }
    const int64_t rows = (*static_dims)[static_dims->size() - 2];
    const int64_t cols = (*static_dims)[static_dims->size() - 1];
    return {
        MakeIndexConst(batch * (cols / c0), span),
        MakeIndexConst(rows * c0, span),
    };
  }
  ExprPtr c0_expr = MakeIndexConst(c0, span);
  ExprPtr cb_expr = MakeFloorDiv(shape.back(), c0_expr, span);
  ExprPtr row_expr = cb_expr;
  for (size_t i = shape.size() - 2; i > 0; --i) {
    row_expr = MakeMul(shape[i - 1], row_expr, span);
  }
  ExprPtr col_expr = MakeMul(shape[shape.size() - 2], c0_expr, span);
  return {std::move(row_expr), std::move(col_expr)};
}

std::pair<std::vector<ExprPtr>, std::vector<ExprPtr>> NzBackingNdOffsetsAndWindow(
    const std::vector<ExprPtr>& offsets, const std::vector<ExprPtr>& gm_shape, int64_t tr, int64_t tc,
    int64_t c0, const Span& span) {
  INTERNAL_CHECK_SPAN(gm_shape.size() >= 2 && offsets.size() == gm_shape.size(), span)
      << "Internal error: an NZ backing window requires offsets in the logical tensor's coordinates";
  ExprPtr b_idx = nullptr;
  if (gm_shape.size() > 2) {
    ExprPtr stride_expr = nullptr;
    for (size_t i = gm_shape.size() - 2; i > 0; --i) {
      const size_t idx = i - 1;
      const ExprPtr& off = offsets[idx];
      auto off_ci = As<ConstInt>(off);
      if (!(off_ci && off_ci->value_ == 0)) {
        ExprPtr term = stride_expr ? MakeMul(off, stride_expr, span) : off;
        b_idx = b_idx ? MakeAdd(b_idx, term, span) : term;
      }
      const ExprPtr& dim_expr = gm_shape[idx];
      stride_expr = stride_expr ? MakeMul(dim_expr, stride_expr, span) : dim_expr;
    }
  }

  const ExprPtr& off_r = offsets[offsets.size() - 2];
  const ExprPtr& off_c = offsets[offsets.size() - 1];
  ExprPtr c0_expr = MakeIndexConst(c0, span);

  ExprPtr cb_expr;
  if (auto off_c_ci = As<ConstInt>(off_c)) {
    cb_expr = MakeIndexConst(off_c_ci->value_ / c0, span);
  } else {
    cb_expr = MakeFloorDiv(off_c, c0_expr, span);
  }

  ExprPtr nd_row;
  auto b_ci = As<ConstInt>(b_idx);
  if (!b_idx || (b_ci && b_ci->value_ == 0)) {
    nd_row = cb_expr;
  } else {
    ExprPtr c_blocks_expr;
    if (auto last_ci = As<ConstInt>(gm_shape.back())) {
      c_blocks_expr = MakeIndexConst(last_ci->value_ / c0, span);
    } else {
      c_blocks_expr = MakeFloorDiv(gm_shape.back(), c0_expr, span);
    }
    ExprPtr b_scaled = MakeMul(b_idx, c_blocks_expr, span);
    if (auto cb_ci = As<ConstInt>(cb_expr); cb_ci && cb_ci->value_ == 0) {
      nd_row = b_scaled;
    } else {
      nd_row = MakeAdd(b_scaled, cb_expr, span);
    }
  }

  ExprPtr nd_col;
  if (auto off_r_ci = As<ConstInt>(off_r)) {
    nd_col = MakeIndexConst(off_r_ci->value_ * c0, span);
  } else {
    nd_col = MakeMul(off_r, c0_expr, span);
  }

  std::vector<ExprPtr> nd_win = {
      MakeIndexConst(tc / c0, span),
      MakeIndexConst(tr * c0, span),
  };
  return {std::vector<ExprPtr>{std::move(nd_row), std::move(nd_col)}, std::move(nd_win)};
}

void FindReturnStmts(const StmtPtr& stmt, std::vector<ReturnStmtPtr>& out) {
  if (!stmt) {
    return;
  }
  if (auto rs = As<ReturnStmt>(stmt)) {
    out.push_back(rs);
    return;
  }
  for (const auto& child : detail::IterChildStmts(stmt)) {
    FindReturnStmts(child, out);
  }
}

using ParamSig = std::vector<std::optional<TensorLayout>>;

struct ParamSigHash {
  size_t operator()(const std::pair<std::string, ParamSig>& key) const {
    size_t h = std::hash<std::string>{}(key.first);
    for (const auto& opt : key.second) {
      h ^= (opt.has_value() ? static_cast<size_t>(*opt) + 1 : 0) + 0x9e3779b9 + (h << 6) + (h >> 2);
    }
    return h;
  }
};

struct LayoutSpecKeyHash {
  size_t operator()(const std::pair<const Var*, LayoutSpec>& key) const {
    size_t h = std::hash<const void*>{}(key.first);
    const LayoutSpecHash hash;
    h ^= hash(key.second) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};

bool IsVecTileType(const TypePtr& type) {
  auto tt = As<TileType>(type);
  return tt && tt->GetMemorySpace().value_or(MemorySpace::Vec) == MemorySpace::Vec;
}

// Keep logical shapes, symbolic expressions and view metadata from the
// receiving type. Only its physical layout follows the source value.
TypePtr SyncGmLayouts(const TypePtr& target, const TypePtr& source) {
  if (auto target_tensor = As<TensorType>(target)) {
    if (auto source_tensor = AsTensorTypeLike(source)) {
      return WithTensorLayout(target_tensor, detail::TensorTypeLayout(source_tensor));
    }
  } else if (auto target_tuple = As<TupleType>(target)) {
    if (auto source_tuple = As<TupleType>(source);
        source_tuple && source_tuple->types_.size() == target_tuple->types_.size()) {
      auto types = target_tuple->types_;
      bool changed = false;
      for (size_t i = 0; i < types.size(); ++i) {
        auto type = SyncGmLayouts(types[i], source_tuple->types_[i]);
        changed |= type.get() != types[i].get();
        types[i] = std::move(type);
      }
      if (changed) return std::make_shared<TupleType>(std::move(types));
    }
  }
  return target;
}

TypePtr SyncGmResultElements(const TypePtr& target, const std::vector<TypePtr>& sources) {
  auto tuple = As<TupleType>(target);
  if (!tuple) return target;
  auto types = tuple->types_;
  bool changed = false;
  for (size_t i = 0; i < types.size() && i < sources.size(); ++i) {
    auto type = SyncGmLayouts(types[i], sources[i]);
    changed |= type.get() != types[i].get();
    types[i] = std::move(type);
  }
  return changed ? std::make_shared<TupleType>(std::move(types)) : target;
}

// Tell the ordinary HOST-to-CHIP verifier that an NZ argument crosses a
// device boundary. For a whole matrix with one configured worker, explicit
// rank zero has the same placement as the original comm-less dispatch.
class HostMatrixDispatchWriter : public IRMutator {
 public:
  void AddSites(const std::vector<const Expr*>& sites) { sites_.insert(sites.begin(), sites.end()); }

  FunctionPtr VisitFunction(const FunctionPtr& func) override {
    return sites_.empty() ? func : IRMutator::VisitFunction(func);
  }

 protected:
  ExprPtr VisitExpr_(const CallPtr& op) override { return PinToSoleWorker(op, IRMutator::VisitExpr_(op)); }

  ExprPtr VisitExpr_(const SubmitPtr& op) override { return PinToSoleWorker(op, IRMutator::VisitExpr_(op)); }

 private:
  template <typename Dispatch>
  ExprPtr PinToSoleWorker(const std::shared_ptr<const Dispatch>& original, const ExprPtr& rewritten) {
    if (!sites_.count(original.get()) || original->HasAttr(kAttrDevice)) return rewritten;
    const auto pinned = MutableCopy(As<Dispatch>(rewritten));
    ExprPtr rank = std::make_shared<ConstInt>(0, DataType::INDEX, original->span_);
    pinned->attrs_.emplace_back(kAttrDevice, std::move(rank));
    return pinned;
  }

  std::unordered_set<const Expr*> sites_;
};

class LayoutAssignmentWriter : public IRMutator {
 public:
  LayoutAssignmentWriter(const LayoutProblemInstance& instance, LayoutWritebackBindings& bindings,
                         const LayoutAssignment& assignment, LayoutAssignment* inferred = nullptr)
      : op_configs_(assignment.op_configs), instance_(instance), bindings_(bindings), inferred_(inferred) {
    if (inferred_) {
      for (const auto& op : instance_.operations) {
        auto config = op_configs_.find(op.op_id);
        if (config != op_configs_.end() && HasMutableVecBacking(op) &&
            std::find(op.configs.begin(), op.configs.end(), config->second) == op.configs.end()) {
          throw ValueError("Illegal AutoLayout configuration for mutable Vec backing at '" + op.op_id + "'");
        }
      }
    }
    std::unordered_set<std::string> unsupported_host_inputs;
    for (const auto& [_, root] : bindings.host_input_roots) {
      const auto* value = instance.FindValue(root);
      if (!value || value->admissible_layouts.size() != 2) {
        unsupported_host_inputs.insert(root);
        continue;
      }
      const auto selected = assignment.value_layouts.find(root);
      if (selected != assignment.value_layouts.end() && selected->second == LayoutSpec::NZ()) {
        const auto sites = bindings.orch.unpinned_host_matrix_dispatches.find(root);
        if (sites != bindings.orch.unpinned_host_matrix_dispatches.end()) {
          host_dispatch_writer_.AddSites(sites->second);
        }
      }
    }
    for (const auto& [name, spec] : assignment.value_layouts) {
      if (spec.IsTensor()) {
        // An unsupported named HOST root can have different authored device
        // ABIs. Its canonical fallback is only a solver value and cannot
        // override any individual parameter's declared representation.
        if (unsupported_host_inputs.count(name)) continue;
        tensor_layouts_.emplace(name, spec.tensor_layout);
      } else {
        if (const auto* value = instance_.FindValue(name);
            inferred_ && value && value->has_mutable_vec_backing && spec != value->current_layout) {
          throw ValueError("AutoLayout cannot relayout the mutable Vec backing of '" + name + "'");
        }
        tile_layouts_.emplace(name, spec);
      }
    }
  }

  ProgramPtr Rewrite(const ProgramPtr& program) {
    std::unordered_map<std::string, FunctionPtr> func_by_name;
    std::vector<FunctionPtr> incore_funcs;
    for (const auto& [_, fn] : program->functions_) {
      func_by_name.emplace(fn->name_, fn);
      if (IsInCoreType(fn->func_type_)) {
        incore_funcs.push_back(fn);
      }
    }

    auto& orch = bindings_.orch;
    {
      return_lineage::ScopedTraceCache trace_cache;
      for (const auto& name : orch.caller_first_order) {
        const auto& func = func_by_name.at(name);
        if (IsWrapperType(func->func_type_) && !func->return_types_.empty()) {
          wrapper_returns_.emplace(func.get(), return_lineage::ReturnedParamIndices(func, program));
        }
      }
    }

    const bool multi_incore = incore_funcs.size() > 1;
    for (const auto& fn : incore_funcs) {
      tile_prefixes_.emplace(fn->name_, detail::TileLayoutPrefix(program, fn->name_, multi_incore));
    }
    std::vector<FunctionPtr> extra_incore;
    gm_names_ = std::move(orch.gm_names);
    if (!orch.caller_first_order.empty()) {
      param_steps_ = std::move(orch.param_steps);
      extra_incore = RegisterCalleeSpecializations(program, func_by_name, orch);
    }

    std::vector<FunctionPtr> new_funcs;
    new_funcs.reserve(program->functions_.size() + extra_incore.size());
    bool changed = !extra_incore.empty();
    auto rewrite = [&](const FunctionPtr& func) {
      auto rewritten = VisitFunction(func);
      changed |= rewritten.get() != func.get();
      new_funcs.push_back(std::move(rewritten));
    };
    for (const auto& func : incore_funcs) rewrite(func);
    for (const auto& func : extra_incore) rewrite(func);
    // After the InCore signatures and specializations, publish each callee's
    // result layouts before visiting the wrappers and callers that use them.
    for (auto it = orch.caller_first_order.rbegin(); it != orch.caller_first_order.rend(); ++it) {
      rewrite(func_by_name.at(*it));
    }
    if (!changed) return program;
    return std::make_shared<Program>(std::move(new_funcs), program->name_, program->span_);
  }

  FunctionPtr VisitFunction(const FunctionPtr& func) override {
    if (func->HasAttr(kAttrExternalSource) || func->requires_runtime_binding_) {
      func_return_types_[func->name_] = func->return_types_;
      return func;
    }
    if (func->level_.has_value() && LevelToLinquLevel(*func->level_) >= 3) {
      func_return_types_[func->name_] = func->return_types_;
      return host_dispatch_writer_.VisitFunction(func);
    }
    local_var_remap_.clear();
    current_func_name_ = func->name_;
    auto prefix = tile_prefixes_.find(current_func_name_);
    current_tile_site_ = prefix == tile_prefixes_.end() ? nullptr : &bindings_.sites.at(prefix->second);

    auto [new_params, changed, has_nz_param] = RewriteFuncParams(func);

    // Function attributes are in the signature's scope. Save their formal
    // substitutions before offset alignment introduces body-only aliases.
    auto [new_attrs, attrs_changed] = MutateAttrs(func->attrs_);
    changed |= attrs_changed;

    std::vector<StmtPtr> prologue_stmts;
    if (has_nz_param && IsInCoreType(func->func_type_)) {
      prologue_stmts = BuildAlignedParamPrologue(func->name_, new_params, func->params_, func->body_);
    }
    if (IsInCoreType(func->func_type_)) {
      std::vector<StmtPtr> vec_prologue = BuildIncoreVecPrologue(func, new_params);
      prologue_stmts.insert(prologue_stmts.end(), std::make_move_iterator(vec_prologue.begin()),
                            std::make_move_iterator(vec_prologue.end()));
    } else {
      ClearPrescanState();
    }
    if (!prologue_stmts.empty()) {
      changed = true;
    }

    StmtPtr new_body = VisitStmt(func->body_);
    if (!prologue_stmts.empty()) {
      std::vector<StmtPtr> combined = std::move(prologue_stmts);
      if (auto seq = As<SeqStmts>(new_body)) {
        combined.insert(combined.end(), seq->stmts_.begin(), seq->stmts_.end());
      } else {
        combined.push_back(new_body);
      }
      new_body = std::make_shared<SeqStmts>(std::move(combined), new_body->span_);
    }

    auto [new_return_types, ret_changed] = SyncFuncReturnTypes(func, new_params, new_body);
    changed = changed || ret_changed;
    func_return_types_[func->name_] = new_return_types;

    if (!changed && new_body.get() == func->body_.get()) {
      return func;
    }

    std::vector<VarPtr> param_vars;
    std::vector<ParamDirection> param_dirs;
    param_vars.reserve(new_params.size());
    param_dirs.reserve(new_params.size());
    for (auto& [p, d] : new_params) {
      param_vars.push_back(std::move(p));
      param_dirs.push_back(d);
    }

    auto rebuilt = std::make_shared<Function>(
        func->name_, std::move(param_vars), std::move(param_dirs), std::move(new_return_types),
        std::move(new_body), func->span_, func->func_type_, func->level_, func->role_, std::move(new_attrs),
        func->requires_runtime_binding_, func->ir_stage_);
    return NormalizeStmtStructure(rebuilt);
  }

 protected:
  ExprPtr VisitExpr_(const VarPtr& op) override {
    if (auto it = local_var_remap_.find(op.get()); it != local_var_remap_.end()) {
      return it->second;
    }
    TypePtr remapped_type = RemapTypeViaVisitor(op->GetType());
    if (remapped_type.get() != op->GetType().get()) {
      auto fresh = std::make_shared<Var>(op->name_hint_, std::move(remapped_type), op->span_);
      local_var_remap_.emplace(op.get(), fresh);
      return fresh;
    }
    return op;
  }

  ExprPtr VisitExpr_(const IterArgPtr& op) override {
    if (auto it = local_var_remap_.find(op.get()); it != local_var_remap_.end()) {
      return it->second;
    }
    TypePtr remapped_type = RemapTypeViaVisitor(op->GetType());
    if (remapped_type.get() != op->GetType().get()) {
      auto fresh =
          std::make_shared<IterArg>(op->name_hint_, std::move(remapped_type), op->initValue_, op->span_);
      local_var_remap_.emplace(op.get(), fresh);
      return fresh;
    }
    return op;
  }

  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    std::vector<StmtPtr> new_stmts;
    new_stmts.reserve(op->stmts_.size());
    bool changed = false;
    for (const auto& s : op->stmts_) {
      StmtPtr visited = VisitStmt(s);
      if (visited.get() != s.get()) {
        changed = true;
      }
      if (auto vseq = As<SeqStmts>(visited); vseq && !As<SeqStmts>(s)) {
        new_stmts.insert(new_stmts.end(), vseq->stmts_.begin(), vseq->stmts_.end());
      } else {
        new_stmts.push_back(std::move(visited));
      }
    }
    if (!changed) {
      return op;
    }
    return std::make_shared<SeqStmts>(std::move(new_stmts), op->span_);
  }

  struct LoopRewriteResult {
    std::vector<IterArgPtr> new_iter_args;
    StmtPtr new_body;
    std::vector<VarPtr> new_return_vars;
    ExprPtr new_condition;
    bool iter_changed = false;
    bool ret_changed = false;
  };

  LoopRewriteResult RewriteLoopIterArgsAndBody(const std::vector<IterArgPtr>& iter_args, const StmtPtr& body,
                                               const std::vector<VarPtr>& return_vars,
                                               const ExprPtr& condition = nullptr) {
    LoopRewriteResult res;
    res.new_iter_args.reserve(iter_args.size());
    std::vector<std::optional<LayoutSpec>> loop_vec_lays;
    loop_vec_lays.reserve(iter_args.size());
    std::vector<bool> modeled(iter_args.size(), false);
    std::vector<StmtPtr> ia_prologue;

    for (const auto& ia : iter_args) {
      const ExprPtr& old_init = ia->initValue_;
      ExprPtr new_init = old_init ? VisitExpr(old_init) : old_init;
      std::optional<LayoutSpec> ia_lay;
      if (IsVecTileType(ia->GetType())) {
        if (auto it = vec_prod_lay_.find(ia.get()); it != vec_prod_lay_.end()) {
          ia_lay = it->second;
        }
        if (ia_lay.has_value() && vec_loop_inits_.count(ia.get()) != 0) {
          const auto& carried_layout = *ia_lay;
          modeled[res.new_iter_args.size()] = true;
          if (auto init_var = AsVarLike(old_init)) {
            new_init = GetTileVarForLayout(init_var, carried_layout);
          }
          const auto init_type = As<TileType>(new_init->GetType());
          CHECK_SPAN(init_type && EffectiveTileLayout(init_type) == carried_layout, ia->span_)
              << "AutoLayout could not materialize the initializer of a Vec loop carry in "
              << carried_layout.ShortName();
        }
      }
      loop_vec_lays.push_back(ia_lay);

      IterArgPtr new_ia = ia;
      if (new_init &&
          (new_init.get() != old_init.get() || !structural_equal(new_init->GetType(), ia->GetType()))) {
        new_ia = std::make_shared<IterArg>(ia->name_hint_, new_init->GetType(), new_init, ia->span_);
        local_var_remap_[ia.get()] = new_ia;
        res.iter_changed = true;
      }
      res.new_iter_args.push_back(new_ia);

      if (ia_lay.has_value() && vec_demanded_.count(ia.get()) != 0) {
        std::vector<StmtPtr> convs = EmitDemandedConversions(ia.get(), new_ia, *ia_lay, ia->span_);
        ia_prologue.insert(ia_prologue.end(), std::make_move_iterator(convs.begin()),
                           std::make_move_iterator(convs.end()));
      }
    }

    if (condition) {
      res.new_condition = VisitExpr(condition);
    }

    yield_frames_.push_back({std::move(loop_vec_lays), {}, /*is_if_branch=*/false, std::move(modeled)});
    try {
      res.new_body = VisitStmt(body);
    } catch (...) {
      yield_frames_.pop_back();
      throw;
    }
    yield_frames_.pop_back();

    if (!ia_prologue.empty()) {
      if (auto seq = As<SeqStmts>(res.new_body)) {
        ia_prologue.insert(ia_prologue.end(), seq->stmts_.begin(), seq->stmts_.end());
      } else {
        ia_prologue.push_back(res.new_body);
      }
      res.new_body = std::make_shared<SeqStmts>(std::move(ia_prologue), res.new_body->span_);
    }

    res.new_return_vars.reserve(return_vars.size());
    for (size_t idx = 0; idx < return_vars.size(); ++idx) {
      const VarPtr& rv = return_vars[idx];
      TypePtr expected_type =
          idx < res.new_iter_args.size() ? res.new_iter_args[idx]->GetType() : rv->GetType();
      if (!structural_equal(expected_type, rv->GetType())) {
        auto new_rv = std::make_shared<Var>(rv->name_hint_, expected_type, rv->span_);
        local_var_remap_[rv.get()] = new_rv;
        res.new_return_vars.push_back(std::move(new_rv));
        res.ret_changed = true;
      } else {
        res.new_return_vars.push_back(rv);
      }
    }
    return res;
  }

  StmtPtr WrapWithReturnVarConversions(const std::vector<VarPtr>& orig_rvs,
                                       const std::vector<VarPtr>& new_rvs, StmtPtr main_stmt,
                                       const Span& span) {
    std::vector<StmtPtr> convs;
    for (size_t idx = 0; idx < orig_rvs.size() && idx < new_rvs.size(); ++idx) {
      const Var* orig_rv = orig_rvs[idx].get();
      const VarPtr& new_rv = new_rvs[idx];
      if (!IsVecTileType(new_rv->GetType()) || vec_demanded_.count(orig_rv) == 0) {
        continue;
      }
      LayoutSpec prod_lay = EffectiveTileLayout(As<TileType>(new_rv->GetType()));
      if (auto prod_it = vec_prod_lay_.find(orig_rv); prod_it != vec_prod_lay_.end()) {
        prod_lay = prod_it->second;
      }
      std::vector<StmtPtr> extra = EmitDemandedConversions(orig_rv, new_rv, prod_lay, span);
      convs.insert(convs.end(), std::make_move_iterator(extra.begin()), std::make_move_iterator(extra.end()));
    }
    if (convs.empty()) {
      return main_stmt;
    }
    std::vector<StmtPtr> seq;
    seq.reserve(1 + convs.size());
    seq.push_back(std::move(main_stmt));
    seq.insert(seq.end(), std::make_move_iterator(convs.begin()), std::make_move_iterator(convs.end()));
    return std::make_shared<SeqStmts>(std::move(seq), span);
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    ExprPtr new_start = VisitExpr(op->start_);
    ExprPtr new_stop = VisitExpr(op->stop_);
    ExprPtr new_step = VisitExpr(op->step_);
    LoopRewriteResult lr = RewriteLoopIterArgsAndBody(op->iter_args_, op->body_, op->return_vars_);
    auto [new_attrs, attrs_changed] = MutateAttrs(op->attrs_);

    StmtPtr main_stmt = op;
    if (lr.iter_changed || lr.ret_changed || attrs_changed || new_start.get() != op->start_.get() ||
        new_stop.get() != op->stop_.get() || new_step.get() != op->step_.get() ||
        lr.new_body.get() != op->body_.get()) {
      main_stmt = std::make_shared<ForStmt>(
          op->loop_var_, std::move(new_start), std::move(new_stop), std::move(new_step),
          std::move(lr.new_iter_args), std::move(lr.new_body), lr.new_return_vars, op->span_, op->kind_,
          std::move(new_attrs), op->leading_comments_, op->hardware_loop_policy_);
    }
    return WrapWithReturnVarConversions(op->return_vars_, lr.new_return_vars, std::move(main_stmt),
                                        op->span_);
  }

  StmtPtr VisitStmt_(const WhileStmtPtr& op) override {
    LoopRewriteResult lr =
        RewriteLoopIterArgsAndBody(op->iter_args_, op->body_, op->return_vars_, op->condition_);

    StmtPtr main_stmt = op;
    if (lr.iter_changed || lr.ret_changed || lr.new_condition.get() != op->condition_.get() ||
        lr.new_body.get() != op->body_.get()) {
      main_stmt = std::make_shared<WhileStmt>(std::move(lr.new_condition), std::move(lr.new_iter_args),
                                              std::move(lr.new_body), lr.new_return_vars, op->span_,
                                              op->leading_comments_);
    }
    return WrapWithReturnVarConversions(op->return_vars_, lr.new_return_vars, std::move(main_stmt),
                                        op->span_);
  }

  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    ExprPtr new_cond = VisitExpr(op->condition_);

    std::vector<std::optional<LayoutSpec>> then_expected(op->return_vars_.size());
    std::vector<std::optional<LayoutSpec>> else_expected(op->return_vars_.size());
    auto ij_it = if_join_cfgs_.find(op.get());
    for (size_t idx = 0; idx < op->return_vars_.size(); ++idx) {
      const VarPtr& rv = op->return_vars_[idx];
      if (!IsVecTileType(rv->GetType())) {
        continue;
      }
      if (ij_it != if_join_cfgs_.end()) {
        auto cfg_it = ij_it->second.find(idx);
        if (cfg_it != ij_it->second.end()) {
          if (cfg_it->second.input_layouts.size() >= 1 && cfg_it->second.input_layouts[0].IsTile()) {
            then_expected[idx] = cfg_it->second.input_layouts[0];
          }
          if (cfg_it->second.input_layouts.size() >= 2 && cfg_it->second.input_layouts[1].IsTile()) {
            else_expected[idx] = cfg_it->second.input_layouts[1];
          }
          continue;
        }
      }
      if (auto prod_it = vec_prod_lay_.find(rv.get()); prod_it != vec_prod_lay_.end()) {
        then_expected[idx] = prod_it->second;
        else_expected[idx] = prod_it->second;
      }
    }

    yield_frames_.push_back({std::move(then_expected), {}, /*is_if_branch=*/true});
    StmtPtr new_then;
    try {
      new_then = VisitStmt(op->then_body_);
    } catch (...) {
      yield_frames_.pop_back();
      throw;
    }
    std::vector<TypePtr> then_types = std::move(yield_frames_.back().branch_yield_types);
    yield_frames_.pop_back();

    std::optional<StmtPtr> new_else = op->else_body_;
    std::vector<TypePtr> else_types;
    if (op->else_body_.has_value() && *op->else_body_) {
      yield_frames_.push_back({std::move(else_expected), {}, /*is_if_branch=*/true});
      StmtPtr visited_else;
      try {
        visited_else = VisitStmt(*op->else_body_);
      } catch (...) {
        yield_frames_.pop_back();
        throw;
      }
      else_types = std::move(yield_frames_.back().branch_yield_types);
      yield_frames_.pop_back();
      new_else = std::move(visited_else);
    }

    std::vector<VarPtr> new_return_vars;
    new_return_vars.reserve(op->return_vars_.size());
    bool ret_changed = false;
    for (size_t idx = 0; idx < op->return_vars_.size(); ++idx) {
      const VarPtr& rv = op->return_vars_[idx];
      TypePtr target_type = rv->GetType();
      if (idx < then_types.size() && !structural_equal(then_types[idx], rv->GetType())) {
        target_type = then_types[idx];
      } else if (idx < else_types.size() && !structural_equal(else_types[idx], rv->GetType())) {
        target_type = else_types[idx];
      }
      if (!structural_equal(target_type, rv->GetType())) {
        auto new_rv = std::make_shared<Var>(rv->name_hint_, std::move(target_type), rv->span_);
        local_var_remap_[rv.get()] = new_rv;
        new_return_vars.push_back(std::move(new_rv));
        ret_changed = true;
      } else {
        new_return_vars.push_back(rv);
      }
    }

    const bool else_changed = op->else_body_.has_value()
                                  ? (!new_else.has_value() || new_else->get() != (*op->else_body_).get())
                                  : new_else.has_value();
    StmtPtr main_stmt = op;
    if (ret_changed || new_cond.get() != op->condition_.get() || new_then.get() != op->then_body_.get() ||
        else_changed) {
      main_stmt = std::make_shared<IfStmt>(std::move(new_cond), std::move(new_then), std::move(new_else),
                                           new_return_vars, op->span_, op->leading_comments_);
    }
    return WrapWithReturnVarConversions(op->return_vars_, new_return_vars, std::move(main_stmt), op->span_);
  }

  StmtPtr VisitStmt_(const YieldStmtPtr& op) override {
    const std::vector<std::optional<LayoutSpec>> expected_lays =
        yield_frames_.empty() ? std::vector<std::optional<LayoutSpec>>{} : yield_frames_.back().loop_expected;
    const bool is_if_branch = !yield_frames_.empty() && yield_frames_.back().is_if_branch;
    std::vector<ExprPtr> new_vals;
    new_vals.reserve(op->value_.size());
    std::vector<StmtPtr> extra_stmts;
    bool changed = false;
    for (size_t idx = 0; idx < op->value_.size(); ++idx) {
      const ExprPtr& val = op->value_[idx];
      std::optional<LayoutSpec> target_lay = idx < expected_lays.size() ? expected_lays[idx] : std::nullopt;
      if (target_lay.has_value()) {
        if (idx < yield_frames_.back().modeled_loop_carries.size() &&
            yield_frames_.back().modeled_loop_carries[idx]) {
          auto var = AsVarLike(val);
          ExprPtr visited = var ? GetTileVarForLayout(var, *target_lay) : VisitExpr(val);
          auto type = As<TileType>(visited->GetType());
          CHECK_SPAN(type && EffectiveTileLayout(type) == *target_lay, op->span_)
              << "AutoLayout could not materialize the backedge of a Vec loop carry in "
              << target_lay->ShortName();
          changed |= visited.get() != val.get();
          new_vals.push_back(std::move(visited));
          continue;
        }
        if (auto val_var = AsVarLike(val)) {
          const Var* conversion_root = VecAliasRoot(val_var.get());
          if (auto conv_it = vec_conv_vars_.find({conversion_root, *target_lay});
              conv_it != vec_conv_vars_.end()) {
            new_vals.push_back(conv_it->second);
            if (conv_it->second.get() != val_var.get()) {
              changed = true;
            }
            continue;
          }
          auto prod_it = vec_prod_lay_.find(val_var.get());
          auto shape_it = orig_tile_shapes_.find(val_var.get());
          if (prod_it != vec_prod_lay_.end() && prod_it->second != *target_lay &&
              shape_it != orig_tile_shapes_.end()) {
            auto remap_it = local_var_remap_.find(val_var.get());
            VarPtr cur_var = remap_it != local_var_remap_.end() ? remap_it->second : val_var;
            auto [conv_var, conv_stmts] = EmitVecConversion(conversion_root, cur_var, prod_it->second,
                                                            *target_lay, shape_it->second, op->span_);
            if (is_if_branch && !conv_stmts.empty()) {
              // Keep branch-local conversion Var scoped to this branch so a sibling branch cannot
              // accidentally reuse a non-dominating SSA Var from `vec_conv_vars_`.
              vec_conv_vars_.erase({conversion_root, *target_lay});
            }
            extra_stmts.insert(extra_stmts.end(), std::make_move_iterator(conv_stmts.begin()),
                               std::make_move_iterator(conv_stmts.end()));
            new_vals.push_back(std::move(conv_var));
            changed = true;
            continue;
          }
        }
      }
      ExprPtr visited = VisitExpr(val);
      if (visited.get() != val.get()) {
        changed = true;
      }
      new_vals.push_back(std::move(visited));
    }
    if (!yield_frames_.empty() && (is_if_branch || yield_frames_.back().loop_expected.empty())) {
      auto& types = yield_frames_.back().branch_yield_types;
      types.clear();
      types.reserve(new_vals.size());
      for (const auto& nv : new_vals) {
        types.push_back(nv->GetType());
      }
    }
    if (!changed && extra_stmts.empty()) {
      return op;
    }
    auto new_yield = std::make_shared<YieldStmt>(std::move(new_vals), op->span_);
    if (!extra_stmts.empty()) {
      extra_stmts.push_back(std::move(new_yield));
      return std::make_shared<SeqStmts>(std::move(extra_stmts), op->span_);
    }
    return new_yield;
  }

  StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
    if (auto call = As<Call>(op->expr_); call && call->op_->name_ == kOpTileStore) {
      const auto cfg = standalone_store_cfgs_.find(op.get());
      const OperatorConfig* cfg_ptr = cfg == standalone_store_cfgs_.end() ? nullptr : &cfg->second.at(0);
      auto [store_call, pre_stmts, nz_src] = RewriteTileStore(call, cfg_ptr, op->span_);
      (void)nz_src;
      if (store_call.get() == call.get() && pre_stmts.empty()) {
        return op;
      }
      auto new_eval = std::make_shared<EvalStmt>(std::move(store_call), op->span_);
      if (!pre_stmts.empty()) {
        pre_stmts.push_back(std::move(new_eval));
        return std::make_shared<SeqStmts>(std::move(pre_stmts), op->span_);
      }
      return new_eval;
    }
    return IRMutator::VisitStmt_(op);
  }

  StmtPtr VisitStmt_(const ReturnStmtPtr& op) override {
    const auto configs = standalone_store_cfgs_.find(op.get());
    if (configs == standalone_store_cfgs_.end()) return IRMutator::VisitStmt_(op);
    std::vector<ExprPtr> values;
    std::vector<StmtPtr> prefix;
    size_t materialized = 0;
    bool changed = false;
    for (size_t i = 0; i < op->value_.size(); ++i) {
      const auto& old = op->value_[i];
      const auto cfg = configs->second.find(i);
      std::vector<StmtPtr> setup;
      auto value = cfg == configs->second.end()
                       ? VisitExpr(old)
                       : RewriteStoredResult(As<Call>(old), &cfg->second, "returned", op->span_, setup);
      changed |= value != old;
      if (!setup.empty()) {
        // Preserve left-to-right return evaluation before a later NZ store
        // needs statements for its backing view.
        for (; materialized < values.size(); ++materialized) {
          auto& earlier = values[materialized];
          if (AsVarLike(earlier)) continue;
          auto var = std::make_shared<Var>("return_value", earlier->GetType(), op->span_);
          prefix.push_back(std::make_shared<AssignStmt>(var, earlier, op->span_));
          earlier = std::move(var);
        }
        prefix.insert(prefix.end(), setup.begin(), setup.end());
      }
      values.push_back(std::move(value));
    }
    if (!changed && prefix.empty()) return op;
    auto ret = std::make_shared<ReturnStmt>(std::move(values), op->span_, op->leading_comments_);
    if (prefix.empty()) return ret;
    prefix.push_back(std::move(ret));
    return std::make_shared<SeqStmts>(std::move(prefix), op->span_);
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    std::optional<LayoutSpec> target_tile_desc = LookupTileDesc(op->var_);
    const auto target_tensor_layout = LookupTensorLayout(op->var_);

    CallPtr rhs = As<Call>(op->value_);
    if (rhs && rhs->op_->name_ == kOpTensorCreate && target_tensor_layout.has_value() &&
        !IsA<DistributedTensorType>(op->var_->GetType())) {
      if (auto lhs_tt = As<TensorType>(op->var_->GetType());
          lhs_tt && detail::TensorTypeLayout(lhs_tt) != *target_tensor_layout) {
        std::vector<ExprPtr> new_args;
        new_args.reserve(rhs->args_.size());
        for (const auto& a : rhs->args_) {
          new_args.push_back(VisitExpr(a));
        }
        auto new_kwargs = ReplaceOrInsertKwarg(rhs->kwargs_, "layout", *target_tensor_layout);
        CallPtr new_call =
            CreateOpCallWithAttrs(rhs->op_->name_, new_args, new_kwargs, rhs->attrs_, rhs->span_);
        auto new_var = std::make_shared<Var>(op->var_->name_hint_, new_call->GetType(), op->var_->span_);
        local_var_remap_[op->var_.get()] = new_var;
        return std::make_shared<AssignStmt>(std::move(new_var), std::move(new_call), op->span_);
      }
    }

    auto cfg_it = assign_op_cfg_.find(op->var_.get());
    const OperatorConfig* cfg = (cfg_it != assign_op_cfg_.end()) ? &cfg_it->second : nullptr;

    if (rhs && rhs->op_->name_ == kOpTileLoad && IsVecTileType(op->var_->GetType()) && cfg != nullptr &&
        cfg->output_layout.IsTile()) {
      return RewriteVecLoadAssign(op, rhs, *cfg);
    }

    if (rhs && rhs->op_->name_ == kOpTileStore) {
      std::vector<StmtPtr> prefix;
      auto value = RewriteStoredResult(rhs, cfg, detail::CleanName(op->var_->name_hint_), op->span_, prefix);
      auto assign = FinishVecAssign(op->var_, std::move(value), std::nullopt, op->span_, op);
      if (prefix.empty()) return assign;
      prefix.push_back(std::move(assign));
      return std::make_shared<SeqStmts>(std::move(prefix), op->span_);
    }

    if (rhs && rhs->op_->name_ == kOpTileMove && As<TileType>(op->var_->GetType())) {
      if (StmtPtr moved = RewriteTileMoveAssign(op, rhs, cfg, target_tile_desc)) {
        return moved;
      }
    }

    if (rhs && rhs->op_->name_.rfind("tile.", 0) == 0 && rhs->op_->name_ != kOpTileLoad &&
        rhs->op_->name_ != kOpTileStore && IsVecTileType(op->var_->GetType()) && cfg != nullptr &&
        cfg->output_layout.IsTile()) {
      return RewriteVecGenericAssign(op, rhs, *cfg);
    }

    ExprPtr new_value = VisitExpr(op->value_);
    std::optional<LayoutSpec> prod_lay;
    if (IsVecTileType(op->var_->GetType())) {
      if (auto it = vec_prod_lay_.find(op->var_.get()); it != vec_prod_lay_.end()) {
        prod_lay = it->second;
      }
    }
    return FinishVecAssign(op->var_, std::move(new_value), prod_lay, op->span_, op);
  }

  ExprPtr VisitExpr_(const CallPtr& op) override {
    std::vector<ExprPtr> new_args;
    new_args.reserve(op->args_.size());
    bool args_changed = false;
    for (const auto& arg : op->args_) {
      ExprPtr visited = VisitExpr(arg);
      if (visited.get() != arg.get()) {
        args_changed = true;
      }
      new_args.push_back(std::move(visited));
    }
    auto [new_attrs, attrs_changed] = MutateAttrs(op->attrs_);

    if (auto gv = As<GlobalVar>(op->op_)) {
      const std::string target_name = ResolveSpecializedCallee(op.get(), gv->name_, new_args);
      auto ret_it = func_return_types_.find(target_name);
      TypePtr ret_type = (ret_it != func_return_types_.end())
                             ? SyncCallReturnType(op->GetType(), ret_it->second)
                             : RemapTypeViaVisitor(op->GetType());
      const bool type_changed = (ret_type.get() != op->GetType().get());
      if (!args_changed && target_name == gv->name_ && !type_changed && !attrs_changed) {
        return op;
      }
      OpPtr target_gv = (target_name != gv->name_) ? std::make_shared<GlobalVar>(target_name) : op->op_;
      return std::make_shared<Call>(std::move(target_gv), std::move(new_args), op->kwargs_,
                                    std::move(new_attrs), std::move(ret_type), op->span_);
    }

    if (!args_changed && !attrs_changed) {
      return op;
    }
    if (IsOp(op, kOpTileLoad) || detail::PreservesTileDescriptors(op) ||
        UsesBlockedNzCoordinates(op->GetType())) {
      return std::make_shared<Call>(op->op_, std::move(new_args), op->kwargs_, std::move(new_attrs),
                                    RemapTypeViaVisitor(op->GetType()), op->span_);
    }
    return CreateOpCallWithAttrs(op->op_->name_, new_args, op->kwargs_, new_attrs, op->span_);
  }

  ExprPtr VisitExpr_(const SubmitPtr& op) override {
    std::vector<ExprPtr> new_args;
    new_args.reserve(op->args_.size());
    bool args_changed = false;
    for (const auto& arg : op->args_) {
      ExprPtr visited = VisitExpr(arg);
      if (visited.get() != arg.get()) {
        args_changed = true;
      }
      new_args.push_back(std::move(visited));
    }

    std::vector<ExprPtr> new_deps;
    new_deps.reserve(op->deps_.size());
    bool deps_changed = false;
    for (const auto& dep : op->deps_) {
      ExprPtr visited = VisitExpr(dep);
      if (visited.get() != dep.get()) {
        deps_changed = true;
      }
      new_deps.push_back(std::move(visited));
    }

    std::optional<ExprPtr> new_core_num = op->core_num_;
    bool core_num_changed = false;
    if (op->core_num_.has_value() && *op->core_num_) {
      ExprPtr remapped = VisitExpr(*op->core_num_);
      if (remapped.get() != (*op->core_num_).get()) {
        new_core_num = std::move(remapped);
        core_num_changed = true;
      }
    }

    std::optional<ExprPtr> new_predicate = op->predicate_;
    bool predicate_changed = false;
    if (op->predicate_.has_value() && *op->predicate_) {
      ExprPtr remapped = VisitExpr(*op->predicate_);
      if (remapped.get() != (*op->predicate_).get()) {
        new_predicate = std::move(remapped);
        predicate_changed = true;
      }
    }

    auto [new_attrs, attrs_changed] = MutateAttrs(op->attrs_);

    auto gv = As<GlobalVar>(op->op_);
    const std::string target_name = gv ? ResolveSpecializedCallee(op.get(), gv->name_, new_args) : "";
    const bool callee_changed = gv && target_name != gv->name_;
    TypePtr ret_type = op->GetType();
    if (gv) {
      if (auto ret_it = func_return_types_.find(target_name); ret_it != func_return_types_.end()) {
        // Submit adds a task ID after the callee's logical results.
        ret_type = SyncGmResultElements(op->GetType(), ret_it->second);
      } else {
        ret_type = RemapTypeViaVisitor(op->GetType());
      }
    } else {
      ret_type = RemapTypeViaVisitor(op->GetType());
    }
    const bool type_changed = (ret_type.get() != op->GetType().get());

    if (!args_changed && !deps_changed && !core_num_changed && !predicate_changed && !attrs_changed &&
        !callee_changed && !type_changed) {
      return op;
    }

    OpPtr target_op = callee_changed ? std::make_shared<GlobalVar>(target_name) : op->op_;
    return std::make_shared<Submit>(std::move(target_op), std::move(new_args), std::move(new_deps),
                                    op->kwargs_, std::move(new_attrs), std::move(ret_type), op->span_,
                                    std::move(new_core_num), op->sync_start_, op->allow_early_resolve_,
                                    std::move(new_predicate));
  }

 private:
  std::pair<std::vector<std::pair<std::string, std::any>>, bool> MutateAttrs(
      const std::vector<std::pair<std::string, std::any>>& attrs) {
    std::vector<std::pair<std::string, std::any>> out;
    out.reserve(attrs.size());
    bool changed = false;
    for (const auto& [k, v] : attrs) {
      auto remapped = MapAttrExprs(v, [this](const ExprPtr& e) { return VisitExpr(e); });
      if (remapped.has_value()) {
        changed = true;
        out.emplace_back(k, std::move(*remapped));
      } else {
        out.emplace_back(k, v);
      }
    }
    return {std::move(out), changed};
  }

  CallPtr CreateOpCallWithAttrs(const std::string& op_name, const std::vector<ExprPtr>& args,
                                const std::vector<std::pair<std::string, std::any>>& kwargs,
                                const std::vector<std::pair<std::string, std::any>>& attrs,
                                const Span& span) {
    auto [mutated_attrs, attrs_changed] = MutateAttrs(attrs);
    (void)attrs_changed;
    return ::pypto::ir::auto_layout::CreateOpCallWithAttrs(op_name, args, kwargs, mutated_attrs, span);
  }
  std::optional<TensorLayout> LookupTensorLayout(const VarPtr& var) const {
    auto name = gm_names_.find(var.get());
    if (name != gm_names_.end()) {
      if (auto it = tensor_layouts_.find(name->second); it != tensor_layouts_.end()) {
        return it->second;
      }
    }
    return std::nullopt;
  }

  const ValueNode* LookupTileValue(const VarPtr& var) const {
    if (!current_tile_site_) return nullptr;
    const auto& values = current_tile_site_->value_names;
    auto key = values.find(var.get());
    return key == values.end() ? nullptr : instance_.FindValue(key->second);
  }

  std::optional<LayoutSpec> LookupTileDesc(const VarPtr& var) const {
    const auto* node = LookupTileValue(var);
    if (!node) return std::nullopt;
    if (node->has_mutable_vec_backing) return node->current_layout;
    auto layout = tile_layouts_.find(node->name);
    if (layout == tile_layouts_.end()) return std::nullopt;
    return layout->second;
  }

  ParamSig SiteParamSignature(const FunctionPtr& callee,
                              const std::vector<std::optional<std::string>>& site_args) const {
    ParamSig sig;
    sig.reserve(callee->params_.size());
    for (size_t p_idx = 0; p_idx < callee->params_.size(); ++p_idx) {
      const auto& param = callee->params_[p_idx];
      auto tt = AsTensorTypeLike(param->GetType());
      if (!tt) {
        sig.emplace_back(std::nullopt);
        continue;
      }
      const TensorLayout default_lay = detail::TensorTypeLayout(tt);
      const auto& arg_opt = (p_idx < site_args.size()) ? site_args[p_idx] : std::nullopt;
      if (arg_opt.has_value()) {
        auto it = tensor_layouts_.find(*arg_opt);
        sig.emplace_back(it != tensor_layouts_.end() ? it->second : default_lay);
      } else {
        sig.emplace_back(default_lay);
      }
    }
    return sig;
  }

  std::vector<FunctionPtr> RegisterCalleeSpecializations(
      const ProgramPtr& program, const std::unordered_map<std::string, FunctionPtr>& func_by_name,
      const OrchCallAnalysis& orch) {
    std::vector<FunctionPtr> extra_incore;
    std::unordered_set<std::string> used_names;
    for (const auto& [name, fn] : func_by_name) {
      (void)fn;
      used_names.insert(name);
    }

    for (const auto& [callee_name, sites] : orch.call_sites) {
      auto callee_it = func_by_name.find(callee_name);
      if (callee_it == func_by_name.end() || sites.empty()) {
        continue;
      }
      const FunctionPtr& callee = callee_it->second;
      if (!IsInCoreType(callee->func_type_) || callee->HasAttr(kAttrExternalSource) ||
          callee->requires_runtime_binding_ || sites.size() == 1) {
        continue;
      }
      callee_formal_params_.emplace(callee_name, callee->params_);

      // Kernel-local allocations can have different layouts even when the
      // callers selected the same parameter layouts.
      std::vector<const Var*> allocation_vars;
      auto find_allocations = [&](auto& self, const StmtPtr& stmt) -> void {
        if (auto assign = As<AssignStmt>(stmt)) {
          if (auto call = As<Call>(assign->value_); call && call->op_->name_ == kOpTensorCreate) {
            allocation_vars.push_back(assign->var_.get());
          }
        }
        for (const auto& child : detail::IterChildStmts(stmt)) self(self, child);
      };
      find_allocations(find_allocations, callee->body_);
      // Distinct sites can choose different tile layouts even with identical GM interfaces.
      // Compare choices in callee-local coordinates, independent of solver names and modeled cost.
      using TileChoices = std::vector<std::pair<size_t, LayoutSpec>>;
      using OpChoices = std::vector<std::pair<std::string, std::vector<LayoutSpec>>>;
      std::unordered_map<const Var*, size_t> tile_indices;
      std::map<std::tuple<ParamSig, ParamSig, TileChoices, OpChoices>, std::string> variants;
      for (size_t site_idx = 0; site_idx < sites.size(); ++site_idx) {
        const auto& site = sites[site_idx];
        ParamSig sig = SiteParamSignature(callee, site.arg_names);
        ParamSig allocation_layouts;
        for (const auto* allocation : allocation_vars) {
          auto layout = tensor_layouts_.find(site.gm_names.at(allocation));
          allocation_layouts.emplace_back(
              layout == tensor_layouts_.end()
                  ? detail::TensorTypeLayout(AsTensorTypeLike(allocation->GetType()))
                  : layout->second);
        }
        const auto prefix = detail::TileLayoutPrefix(program, callee_name, true, site_idx);
        auto& tile_site = bindings_.sites.at(prefix);
        TileChoices tiles;
        for (const auto& [var, key] : tile_site.value_names) {
          if (auto layout = tile_layouts_.find(key); layout != tile_layouts_.end()) {
            const auto index = tile_indices.try_emplace(var, tile_indices.size()).first->second;
            tiles.emplace_back(index, layout->second);
          }
        }
        std::sort(tiles.begin(), tiles.end());
        OpChoices ops;
        for (size_t i = tile_site.begin; i < tile_site.end; ++i) {
          const auto& id = instance_.operations[i].op_id;
          auto& choice = ops.emplace_back();
          if (auto cfg = op_configs_.find(id); cfg != op_configs_.end()) {
            choice.first = cfg->second.op_name;
            choice.second = cfg->second.input_layouts;
            choice.second.push_back(cfg->second.output_layout);
          }
        }
        auto [variant, new_variant] = variants.try_emplace(
            std::make_tuple(sig, std::move(allocation_layouts), std::move(tiles), std::move(ops)));
        std::string& spec_name = variant->second;
        if (!new_variant) {
          for (const auto* call : site.calls) specialized_calls_.emplace(call, spec_name);
          continue;
        }
        if (site_idx == 0) {
          spec_name = callee_name;
          specialized_callees_.emplace(std::make_pair(callee_name, sig), spec_name);
        } else {
          size_t suffix = site_idx;
          spec_name = callee_name + "_site" + std::to_string(suffix);
          while (used_names.count(spec_name) != 0) {
            ++suffix;
            spec_name = callee_name + "_site" + std::to_string(suffix);
          }
          used_names.insert(spec_name);
          specialized_callees_.emplace(std::make_pair(callee_name, sig), spec_name);
          for (size_t p_idx = 0; p_idx < callee->params_.size(); ++p_idx) {
            if (auto step_it = param_steps_.find({callee_name, p_idx}); step_it != param_steps_.end()) {
              param_steps_[{spec_name, p_idx}] = step_it->second;
            }
          }
          // Clone formal params, their types, return_types_, attrs_, and body defs so every
          // specialized Function maintains disjoint SSA Vars and remapped symbolic expressions.
          std::vector<VarPtr> cloned_params;
          cloned_params.reserve(callee->params_.size());
          std::unordered_map<const Var*, ExprPtr> formal_map;
          for (const auto& p : callee->params_) {
            TypePtr remapped_type = RemapTypeWithSubst(p->GetType(), formal_map);
            auto cp = std::make_shared<Var>(p->name_hint_, std::move(remapped_type), p->span_);
            formal_map.emplace(p.get(), cp);
            cloned_params.push_back(std::move(cp));
          }
          std::vector<TypePtr> cloned_return_types;
          cloned_return_types.reserve(callee->return_types_.size());
          for (const auto& rt : callee->return_types_) {
            cloned_return_types.push_back(RemapTypeWithSubst(rt, formal_map));
          }
          std::vector<std::pair<std::string, std::any>> cloned_attrs = callee->attrs_;
          for (auto& [attr_key, attr_val] : cloned_attrs) {
            (void)attr_key;
            if (auto remapped = MapAttrExprs(
                    attr_val, [&](const ExprPtr& e) { return transform_utils::Substitute(e, formal_map); });
                remapped.has_value()) {
              attr_val = std::move(*remapped);
            }
          }
          DeepCloneResult cloned = DeepClone(callee->body_, formal_map, /*clone_def_vars=*/true);
          const auto& names = sites[site_idx].gm_names;
          auto copy_name = [&](const Var* original, const Var* copy) {
            if (auto it = names.find(original); it != names.end()) {
              gm_names_.emplace(copy, it->second);
            }
            if (auto it = tile_site.value_names.find(original); it != tile_site.value_names.end()) {
              tile_site.value_names.emplace(copy, it->second);
            }
          };
          for (size_t i = 0; i < callee->params_.size(); ++i) {
            copy_name(callee->params_[i].get(), cloned_params[i].get());
          }
          for (const auto& [original, copy] : cloned.var_map) copy_name(original, copy.get());
          extra_incore.push_back(std::make_shared<Function>(
              spec_name, std::move(cloned_params), callee->param_directions_, std::move(cloned_return_types),
              std::move(cloned.cloned_body), callee->span_, callee->func_type_, callee->level_, callee->role_,
              std::move(cloned_attrs), callee->requires_runtime_binding_, callee->ir_stage_));
        }
        tile_prefixes_.insert_or_assign(spec_name, prefix);
        for (const auto* call : site.calls) specialized_calls_.emplace(call, spec_name);
      }
    }
    return extra_incore;
  }

  std::string ResolveSpecializedCallee(const Expr* call, const std::string& callee_name,
                                       const std::vector<ExprPtr>& new_args) const {
    if (auto site = specialized_calls_.find(call); site != specialized_calls_.end()) {
      return site->second;
    }
    if (callee_formal_params_.count(callee_name) == 0) return callee_name;
    ParamSig sig;
    sig.reserve(new_args.size());
    for (const auto& arg : new_args) {
      auto tt = AsTensorTypeLike(arg->GetType());
      sig.push_back(tt ? std::optional<TensorLayout>(detail::TensorTypeLayout(tt)) : std::nullopt);
    }
    if (auto params_it = callee_formal_params_.find(callee_name); params_it != callee_formal_params_.end()) {
      for (size_t idx = sig.size(); idx < params_it->second.size(); ++idx) {
        auto tt = AsTensorTypeLike(params_it->second[idx]->GetType());
        sig.push_back(tt ? std::optional<TensorLayout>(detail::TensorTypeLayout(tt)) : std::nullopt);
      }
    }
    auto spec_it = specialized_callees_.find({callee_name, sig});
    return spec_it != specialized_callees_.end() ? spec_it->second : callee_name;
  }

  void ClearPrescanState() {
    assign_op_cfg_.clear();
    if_join_cfgs_.clear();
    standalone_store_cfgs_.clear();
    vec_prod_lay_.clear();
    vec_alias_roots_.clear();
    vec_loop_inits_.clear();
    vec_slices_.clear();
    vec_demanded_.clear();
    vec_conv_vars_.clear();
    vec_logical_shapes_.clear();
    orig_tile_shapes_.clear();
    nz_nd_needed_vars_.clear();
    nz_nd_backing_vars_.clear();
    yield_frames_.clear();
  }

  std::optional<OperatorConfig> InferLoadConfig(
      const CallPtr& rhs, const VarPtr& lhs_var,
      const std::unordered_map<const Var*, TensorLayout>& var_tensor_lays) const {
    if (!lhs_var || rhs->args_.empty()) {
      return std::nullopt;
    }
    auto src_arg = AsVarLike(rhs->args_[0]);
    if (!src_arg) {
      return std::nullopt;
    }
    std::optional<TensorLayout> gm_lay;
    if (auto it = var_tensor_lays.find(src_arg.get()); it != var_tensor_lays.end()) {
      gm_lay = it->second;
    } else if (auto src_tt = AsTensorTypeLike(src_arg->GetType())) {
      gm_lay = LookupTensorLayout(src_arg);
      if (!gm_lay.has_value()) {
        gm_lay = detail::TensorTypeLayout(src_tt);
      }
    }
    std::optional<LayoutSpec> dst_lay = LookupTileDesc(lhs_var);
    auto lhs_tile = As<TileType>(lhs_var->GetType());
    if (lhs_tile && (!dst_lay.has_value() || lhs_tile->dtype_.IsPackedFp4())) {
      dst_lay = EffectiveTileLayout(lhs_tile);
    }
    if (gm_lay == TensorLayout::DN && lhs_tile && !lhs_tile->dtype_.IsPackedFp4() &&
        lhs_tile->GetMemorySpace().value_or(MemorySpace::Vec) == MemorySpace::Vec) {
      dst_lay = LayoutSpec::ColMajorFlat();
    }
    if (!gm_lay.has_value() || !dst_lay.has_value()) {
      return std::nullopt;
    }
    return OperatorConfig{"fallback", rhs->op_->name_, {LayoutSpec::Tensor(*gm_lay)}, *dst_lay, 0.0};
  }

  std::optional<OperatorConfig> InferStoreConfig(
      const CallPtr& rhs, const std::unordered_map<const Var*, TensorLayout>& var_tensor_lays) const {
    if (rhs->args_.size() < 3) {
      return std::nullopt;
    }
    auto tile_arg = AsVarLike(rhs->args_[0]);
    auto dst_arg = AsVarLike(rhs->args_[2]);
    if (!tile_arg || !dst_arg) {
      return std::nullopt;
    }
    std::optional<TensorLayout> gm_lay;
    if (auto it = var_tensor_lays.find(dst_arg.get()); it != var_tensor_lays.end()) {
      gm_lay = it->second;
    } else if (auto dst_tt = AsTensorTypeLike(dst_arg->GetType())) {
      gm_lay = LookupTensorLayout(dst_arg);
      if (!gm_lay.has_value()) {
        gm_lay = detail::TensorTypeLayout(dst_tt);
      }
    }
    auto tile_t = As<TileType>(tile_arg->GetType());
    if (!tile_t || !gm_lay.has_value()) {
      return std::nullopt;
    }
    LayoutSpec tile_lay = EffectiveTileLayout(tile_t);
    if (tile_t->dtype_.IsPackedFp4()) {
      return OperatorConfig{"fallback", rhs->op_->name_, {tile_lay}, LayoutSpec::Tensor(*gm_lay), 0.0};
    }
    if (auto it = vec_prod_lay_.find(tile_arg.get()); it != vec_prod_lay_.end()) {
      tile_lay = it->second;
    }
    if (tile_t->GetMemorySpace().value_or(MemorySpace::Vec) == MemorySpace::Vec) {
      std::optional<std::vector<int64_t>> t_shape;
      if (auto it = orig_tile_shapes_.find(tile_arg.get()); it != orig_tile_shapes_.end()) {
        t_shape = it->second;
      } else {
        t_shape = detail::StaticShape(tile_t->shape_);
      }
      const bool is_col_vec =
          t_shape.has_value() && t_shape->size() == 2 && (*t_shape)[1] == 1 && (*t_shape)[0] > 1;
      if (*gm_lay == TensorLayout::NZ) {
        tile_lay = LayoutSpec::FractalNZ();
      } else if (*gm_lay == TensorLayout::DN || is_col_vec) {
        tile_lay = LayoutSpec::ColMajorFlat();
      } else {
        tile_lay = LayoutSpec::RowMajorFlat();
      }
    }
    return OperatorConfig{"fallback", rhs->op_->name_, {tile_lay}, LayoutSpec::Tensor(*gm_lay), 0.0};
  }

  std::optional<OperatorConfig> InferOpConfig(
      const CallPtr& rhs, const VarPtr& lhs_var,
      const std::unordered_map<const Var*, TensorLayout>& var_tensor_lays) const {
    const std::string& op_name = rhs->op_->name_;
    if (op_name == kOpTileLoad) {
      return InferLoadConfig(rhs, lhs_var, var_tensor_lays);
    }
    if (op_name == kOpTileStore) {
      return InferStoreConfig(rhs, var_tensor_lays);
    }
    if (op_name == kOpTileMove && lhs_var && !rhs->args_.empty()) {
      auto src_arg = AsVarLike(rhs->args_[0]);
      auto src_t = src_arg ? As<TileType>(src_arg->GetType()) : TileTypePtr(nullptr);
      if (!src_t) {
        return std::nullopt;
      }
      LayoutSpec src_lay = EffectiveTileLayout(src_t);
      if (auto it = vec_prod_lay_.find(src_arg.get()); it != vec_prod_lay_.end()) {
        src_lay = it->second;
      }
      auto lhs_t = As<TileType>(lhs_var->GetType());
      if (lhs_t && (lhs_t->dtype_.IsPackedFp4() || src_t->dtype_.IsPackedFp4())) {
        return OperatorConfig{
            "fallback", op_name, {EffectiveTileLayout(src_t)}, EffectiveTileLayout(lhs_t), 0.0};
      }
      LayoutSpec dst_lay = src_lay;
      if (auto found = LookupTileDesc(lhs_var)) {
        dst_lay = *found;
      } else if (lhs_t) {
        dst_lay = EffectiveTileLayout(lhs_t);
      }
      return OperatorConfig{"fallback", op_name, {src_lay}, dst_lay, 0.0};
    }

    auto lhs_t = lhs_var ? As<TileType>(lhs_var->GetType()) : TileTypePtr(nullptr);
    if (!lhs_t) {
      return std::nullopt;
    }
    const auto tile_args = detail::TileOperands(rhs);
    if (tile_args.empty() && !IsOp(rhs, kOpTileFull)) return std::nullopt;
    auto input_layouts = [&](const std::optional<LayoutSpec>& data_layout = std::nullopt) {
      std::vector<LayoutSpec> layouts;
      layouts.reserve(tile_args.size());
      for (const auto& [index, var] : tile_args) {
        layouts.push_back(data_layout && !detail::IsTileWorkspace(rhs, index)
                              ? *data_layout
                              : EffectiveTileLayout(As<TileType>(var->GetType())));
      }
      return layouts;
    };

    const LayoutSpec original_output = EffectiveTileLayout(lhs_t);
    if (detail::PreservesTileDescriptors(rhs)) {
      return OperatorConfig{"fallback", op_name, input_layouts(), original_output, 0.0};
    }
    if (detail::RowReductionOps().count(op_name) != 0 || detail::ColReductionOps().count(op_name) != 0) {
      const auto input = As<TileType>(tile_args.front().second->GetType());
      auto shape = detail::StaticShape(input->shape_);
      const bool singleton = shape && shape->size() == 2 && ((*shape)[0] == 1 || (*shape)[1] == 1);
      auto inputs = singleton ? input_layouts() : input_layouts(LayoutSpec::RowMajorFlat());
      return OperatorConfig{"fallback", op_name, std::move(inputs), original_output, 0.0};
    }
    if (detail::ColExpandOps().count(op_name) != 0) {
      return OperatorConfig{"fallback", op_name, input_layouts(LayoutSpec::RowMajorFlat()), original_output,
                            0.0};
    }
    if ((detail::OtherRowExpandOps().count(op_name) != 0 || IsOp(rhs, kOpTileRowExpandAdd)) &&
        tile_args.size() >= 2) {
      auto inputs = input_layouts(LayoutSpec::RowMajorFlat());
      const auto& row_arg = tile_args[1].second;
      auto vec_lay = LookupTileDesc(row_arg);
      inputs[1] = IsOp(rhs, kOpTileRowExpandAdd) && vec_lay == LayoutSpec::RowBcastFlat()
                      ? LayoutSpec::RowBcastFlat()
                      : LayoutSpec::ColMajorFlat();
      return OperatorConfig{"fallback", op_name, std::move(inputs), original_output, 0.0};
    }
    LayoutSpec out_lay = LookupTileDesc(lhs_var).value_or(EffectiveTileLayout(lhs_t));
    if (out_lay == LayoutSpec::FractalNZ() && detail::CanUseVecNzPointwise(rhs)) {
      return OperatorConfig{"fallback", op_name, input_layouts(LayoutSpec::FractalNZ()), out_lay, 0.0};
    }
    if (out_lay == LayoutSpec::FractalNZ()) out_lay = original_output;
    auto in_lays = input_layouts();
    for (size_t i = 0; i < tile_args.size(); ++i) {
      const auto& [index, var] = tile_args[i];
      if (detail::IsTileWorkspace(rhs, index)) continue;
      if (detail::HasRowMajorInputRequirement(op_name, index)) {
        const auto tile = As<TileType>(var->GetType());
        auto shape = detail::StaticShape(tile->shape_);
        if (!shape || shape->size() != 2 || ((*shape)[0] != 1 && (*shape)[1] != 1)) {
          in_lays[i] = LayoutSpec::RowMajorFlat();
        }
      } else if (auto it = vec_prod_lay_.find(var.get()); it != vec_prod_lay_.end()) {
        in_lays[i] = it->second;
      }
    }
    return OperatorConfig{"fallback", op_name, std::move(in_lays), out_lay, 0.0};
  }

  std::optional<OperatorConfig> ResolveRequestedVecOutput(
      const OperationNode* modeled, const CallPtr& rhs, const VarPtr& lhs, std::optional<OperatorConfig> cfg,
      const std::unordered_map<const Var*, TensorLayout>& var_tensor_lays) const {
    if (!lhs || !IsVecTileType(lhs->GetType()) || IsOp(rhs, kOpTileLoad) || IsOp(rhs, kOpTileMove)) {
      return cfg;
    }
    auto requested = LookupTileDesc(lhs);
    if (cfg) {
      if (requested && *requested != cfg->output_layout) {
        throw ValueError("Value layout for '" + lhs->name_hint_ + "' contradicts its producer configuration");
      }
      requested = cfg->output_layout;
    }
    if (!requested) return cfg;
    if (!modeled && *requested == EffectiveTileLayout(As<TileType>(lhs->GetType()))) return cfg;

    // A tile's conversion domain can be wider than the layouts its producer can emit.
    if (modeled) {
      const auto preferred = cfg ? cfg : InferOpConfig(rhs, lhs, var_tensor_lays);
      const OperatorConfig* fallback = nullptr;
      for (const auto& candidate : modeled->configs) {
        if (candidate.output_layout != *requested) continue;
        if (preferred && candidate.input_layouts == preferred->input_layouts &&
            candidate.local_vec_conversions == preferred->local_vec_conversions) {
          return candidate;
        }
        // A value-only request leaves the choice of operand layouts to the writer.
        if (!cfg && !fallback) fallback = &candidate;
      }
      if (fallback) return *fallback;
    }
    throw ValueError("AutoLayout cannot produce requested layout " + requested->ShortName() + " for '" +
                     lhs->name_hint_ + "' at " + rhs->op_->name_);
  }

  const Var* VecAliasRoot(const Var* var) const {
    auto it = vec_alias_roots_.find(var);
    return it == vec_alias_roots_.end() ? var : it->second;
  }

  void RecordVecDemandRaw(const Var* tile_var, const LayoutSpec& req_lay) {
    if (tile_var == nullptr || !req_lay.IsTile()) {
      return;
    }
    tile_var = VecAliasRoot(tile_var);
    if (current_tile_site_) {
      auto name = current_tile_site_->value_names.find(tile_var);
      if (name != current_tile_site_->value_names.end() &&
          instance_.values.at(name->second).has_mutable_vec_backing) {
        return;
      }
    }
    auto prod_it = vec_prod_lay_.find(tile_var);
    if (prod_it == vec_prod_lay_.end() || prod_it->second == req_lay ||
        prod_it->second == LayoutSpec::FractalNZ() || req_lay == LayoutSpec::FractalNZ()) {
      return;
    }
    auto& demanded = vec_demanded_[tile_var];
    if (std::find(demanded.begin(), demanded.end(), req_lay) == demanded.end()) {
      demanded.push_back(req_lay);
    }
  }

  void RecordVecDemand(const VarPtr& tile_var, const LayoutSpec& req_lay) {
    if (!IsVecTileType(tile_var->GetType())) {
      return;
    }
    RecordVecDemandRaw(tile_var.get(), req_lay);
  }

  void RecordPrescanCallOutputs(const StmtPtr& stmt, const std::string& op_name, const VarPtr& lhs_var,
                                size_t result_index, const OperatorConfig& cfg,
                                std::unordered_map<const Var*, TensorLayout>& var_tensor_lays) {
    if (lhs_var) {
      assign_op_cfg_.emplace(lhs_var.get(), cfg);
    } else {
      standalone_store_cfgs_[stmt.get()].emplace(result_index, cfg);
    }
    if (!lhs_var) {
      return;
    }
    if (auto lhs_t = As<TileType>(lhs_var->GetType())) {
      if (lhs_t->GetMemorySpace().value_or(MemorySpace::Vec) == MemorySpace::Vec &&
          cfg.output_layout.IsTile()) {
        if (auto shape = detail::StaticShape(lhs_t->shape_)) {
          orig_tile_shapes_[lhs_var.get()] = *shape;
          if (!lhs_t->dtype_.IsPackedFp4() && cfg.output_layout == LayoutSpec::FractalNZ()) {
            vec_logical_shapes_[lhs_var.get()] = *shape;
          }
        }
        vec_prod_lay_.insert_or_assign(lhs_var.get(), cfg.output_layout);
      }
    } else if (op_name == kOpTileStore && AsTensorTypeLike(lhs_var->GetType())) {
      if (cfg.input_layouts.size() >= 2 && cfg.input_layouts[1].IsTensor()) {
        var_tensor_lays[lhs_var.get()] = cfg.input_layouts[1].tensor_layout;
      }
    }
  }

  void RecordPrescanCallDemands(const CallPtr& rhs, const VarPtr& lhs_var, const OperatorConfig& cfg) {
    const std::string& op_name = rhs->op_->name_;
    if (op_name == kOpTileLoad && !rhs->args_.empty()) {
      if (auto gm_var = AsVarLike(rhs->args_[0]);
          gm_var && !UsesBlockedNzCoordinates(gm_var->GetType()) && lhs_var &&
          IsVecTileType(lhs_var->GetType()) && !As<TileType>(lhs_var->GetType())->dtype_.IsPackedFp4() &&
          cfg.output_layout == LayoutSpec::FractalNZ() && !cfg.input_layouts.empty() &&
          cfg.input_layouts[0] == LayoutSpec::Tensor(TensorLayout::NZ)) {
        nz_nd_needed_vars_.insert(gm_var.get());
      }
    } else if (op_name == kOpTileStore && rhs->args_.size() >= 3) {
      auto tile_arg = AsVarLike(rhs->args_[0]);
      auto dst_arg = AsVarLike(rhs->args_[2]);
      const bool tile_is_vec = tile_arg && IsVecTileType(tile_arg->GetType());
      LayoutSpec gm_out_lay = cfg.input_layouts.size() >= 2 ? cfg.input_layouts[1] : cfg.output_layout;
      if (tile_is_vec && !cfg.input_layouts.empty() && cfg.input_layouts[0] != LayoutSpec::FractalNZ()) {
        RecordVecDemand(tile_arg, cfg.input_layouts[0]);
      }
      if (tile_is_vec && dst_arg && !UsesBlockedNzCoordinates(dst_arg->GetType()) &&
          !cfg.input_layouts.empty() && !As<TileType>(tile_arg->GetType())->dtype_.IsPackedFp4() &&
          cfg.input_layouts[0] == LayoutSpec::FractalNZ() &&
          gm_out_lay == LayoutSpec::Tensor(TensorLayout::NZ)) {
        nz_nd_needed_vars_.insert(dst_arg.get());
      }
    } else if (op_name == kOpTileMove && !rhs->args_.empty()) {
      if (auto src_var = AsVarLike(rhs->args_[0]); src_var && !cfg.input_layouts.empty()) {
        RecordVecDemand(src_var, cfg.input_layouts[0]);
      }
    } else {
      std::vector<VarPtr> tile_args;
      for (const auto& a : rhs->args_) {
        if (auto av = AsVarLike(a); av && As<TileType>(av->GetType())) {
          tile_args.push_back(std::move(av));
        }
      }
      for (size_t i = 0; i < tile_args.size() && i < cfg.input_layouts.size(); ++i) {
        RecordVecDemand(tile_args[i], cfg.input_layouts[i]);
      }
    }
  }

  void AliasVarPrescanInfo(const VarPtr& dst_var, const Var* src_var,
                           std::unordered_map<const Var*, TensorLayout>& var_tensor_lays) {
    if (AsTensorTypeLike(dst_var->GetType())) {
      if (auto it = var_tensor_lays.find(src_var); it != var_tensor_lays.end()) {
        var_tensor_lays[dst_var.get()] = it->second;
      }
    } else if (As<TileType>(dst_var->GetType())) {
      if (vec_slices_.count(src_var) != 0) vec_slices_.insert(dst_var.get());
      if (auto it = vec_prod_lay_.find(src_var); it != vec_prod_lay_.end()) {
        vec_prod_lay_.insert_or_assign(dst_var.get(), it->second);
        if (auto sit = orig_tile_shapes_.find(src_var); sit != orig_tile_shapes_.end()) {
          orig_tile_shapes_[dst_var.get()] = sit->second;
        }
        if (auto lit = vec_logical_shapes_.find(src_var); lit != vec_logical_shapes_.end()) {
          vec_logical_shapes_[dst_var.get()] = lit->second;
        }
      }
    }
  }

  void PrescanFunction(const FunctionPtr& func,
                       const std::vector<std::pair<VarPtr, ParamDirection>>& new_params) {
    ClearPrescanState();
    std::unordered_map<const Var*, TensorLayout> var_tensor_lays;
    const auto& site = *current_tile_site_;
    std::unordered_map<std::string, const Var*> vec_definitions;
    auto record_vec_binding = [&](const VarPtr& var) {
      if (!IsVecTileType(var->GetType())) return;
      const auto name = site.value_names.find(var.get());
      if (name == site.value_names.end()) return;
      const auto [original, inserted] = vec_definitions.try_emplace(name->second, var.get());
      if (!inserted && original->second != var.get()) {
        AliasVarPrescanInfo(var, original->second, var_tensor_lays);
        vec_alias_roots_[var.get()] = VecAliasRoot(original->second);
      }
    };
    for (size_t i = 0; i < new_params.size() && i < func->params_.size(); ++i) {
      const VarPtr& new_p = new_params[i].first;
      const VarPtr& old_p = func->params_[i];
      if (auto tt = AsTensorTypeLike(new_p->GetType())) {
        var_tensor_lays[old_p.get()] = detail::TensorTypeLayout(tt);
      } else if (auto tile_t = As<TileType>(new_p->GetType());
                 tile_t && tile_t->GetMemorySpace().value_or(MemorySpace::Vec) == MemorySpace::Vec) {
        auto shape = detail::StaticShape(tile_t->shape_);
        if (shape.has_value()) {
          orig_tile_shapes_[old_p.get()] = *shape;
        }
        LayoutSpec lay = LookupTileDesc(old_p).value_or(EffectiveTileLayout(tile_t));
        vec_prod_lay_.insert_or_assign(old_p.get(), lay);
        if (lay == LayoutSpec::FractalNZ() && shape.has_value()) {
          vec_logical_shapes_[old_p.get()] = *shape;
        }
        record_vec_binding(old_p);
      }
    }

    size_t next = site.begin;
    auto take_op = [&](const std::string& name, const VarPtr& output,
                       const CallPtr& call = nullptr) -> const OperationNode* {
      if (next == site.end) return nullptr;
      const auto& op = instance_.operations[next];
      if (name != op.op_name) return nullptr;
      if (name == kOpTileStore) {
        // A store whose source or destination was outside the model must not
        // consume the configuration of the next modeled store.
        if (!call || call->args_.size() < 3) return nullptr;
        auto src = site.value_names.find(AsVarLike(call->args_[0]).get());
        auto dst = site.value_names.find(AsVarLike(call->args_[2]).get());
        const auto& modeled_dst = op.inputs.size() == 2 ? op.inputs[1] : op.output;
        if (src == site.value_names.end() || src->second != op.inputs.front() ||
            dst == site.value_names.end() || dst->second != modeled_dst) {
          return nullptr;
        }
      } else if (output) {
        auto it = site.value_names.find(output.get());
        if (it == site.value_names.end() || it->second != op.output) return nullptr;
      }
      ++next;
      return &op;
    };
    auto selected = [&](const OperationNode* op) -> std::optional<OperatorConfig> {
      if (op) {
        if (auto it = op_configs_.find(op->op_id); it != op_configs_.end()) {
          return it->second;
        }
      }
      return std::nullopt;
    };
    auto remember = [&](const OperationNode* op, const OperatorConfig& config) {
      if (inferred_ && op) inferred_->op_configs.insert_or_assign(op->op_id, config);
    };
    const auto site_first = instance_.operations.begin() + static_cast<std::ptrdiff_t>(site.begin);
    const auto site_last = instance_.operations.begin() + static_cast<std::ptrdiff_t>(site.end);
    const bool has_op_configs = std::any_of(
        site_first, site_last, [&](const OperationNode& op) { return op_configs_.count(op.op_id) != 0; });

    auto handle_call = [&](const StmtPtr& stmt, const CallPtr& rhs, const VarPtr& lhs_var,
                           size_t result_index = 0) {
      const std::string& op_name = rhs->op_->name_;
      if (lhs_var && IsVecTileType(lhs_var->GetType())) {
        const auto source = rhs->args_.empty() ? nullptr : AsVarLike(rhs->args_[0]);
        if (IsOp(rhs, kOpTileSlice) ||
            (source && vec_slices_.count(source.get()) != 0 && detail::AliasesFirstTileOperand(rhs))) {
          vec_slices_.insert(lhs_var.get());
        }
      }
      if (op_name == kOpTensorCreate && lhs_var) {
        if (auto lhs_tt = AsTensorTypeLike(lhs_var->GetType())) {
          var_tensor_lays[lhs_var.get()] =
              LookupTensorLayout(lhs_var).value_or(detail::TensorTypeLayout(lhs_tt));
          return;
        }
      }
      if (op_name == kOpTileCreate && lhs_var) {
        if (auto lhs_t = As<TileType>(lhs_var->GetType())) {
          if (lhs_t->GetMemorySpace().value_or(MemorySpace::Vec) == MemorySpace::Vec) {
            if (auto shape = detail::StaticShape(lhs_t->shape_)) {
              orig_tile_shapes_[lhs_var.get()] = *shape;
            }
            vec_prod_lay_.insert_or_assign(lhs_var.get(), EffectiveTileLayout(lhs_t));
          }
          return;
        }
      }
      if (op_name.rfind("tile.", 0) != 0) {
        return;
      }
      const auto* modeled = take_op(op_name, lhs_var, rhs);
      auto cfg = selected(modeled);
      if (inferred_) {
        cfg = ResolveRequestedVecOutput(modeled, rhs, lhs_var, std::move(cfg), var_tensor_lays);
      }
      if (!cfg && (!has_op_configs || (inferred_ && modeled && HasMutableVecBacking(*modeled)))) {
        cfg = InferOpConfig(rhs, lhs_var, var_tensor_lays);
        if (cfg && modeled) {
          cfg = ResolveMutableVecConfig(*modeled, std::move(*cfg), rhs->span_);
        }
      }
      if (!cfg.has_value()) {
        if (lhs_var && IsVecTileType(lhs_var->GetType())) {
          auto lhs_t = As<TileType>(lhs_var->GetType());
          if (auto shape = detail::StaticShape(lhs_t->shape_)) {
            orig_tile_shapes_[lhs_var.get()] = *shape;
          }
          vec_prod_lay_.insert_or_assign(lhs_var.get(), EffectiveTileLayout(lhs_t));
        }
        return;
      }
      remember(modeled, *cfg);
      RecordPrescanCallOutputs(stmt, op_name, lhs_var, result_index, *cfg, var_tensor_lays);
      RecordPrescanCallDemands(rhs, lhs_var, *cfg);
    };

    std::vector<std::vector<const Var*>> yield_stacks;
    auto record_loop_binding = [&](const VarPtr& var, const OperatorConfig& cfg, const Var* source) {
      vec_prod_lay_.insert_or_assign(var.get(), cfg.output_layout);
      if (auto shape = detail::StaticShape(As<TileType>(var->GetType())->shape_)) {
        orig_tile_shapes_[var.get()] = *shape;
      }
      if (vec_slices_.count(source) != 0) vec_slices_.insert(var.get());
      if (source && !cfg.input_layouts.empty()) RecordVecDemandRaw(source, cfg.input_layouts[0]);
    };
    auto loop_config = [&](const char* name, const VarPtr& var) -> std::optional<OperatorConfig> {
      const auto* op = take_op(name, var);
      auto config = selected(op);
      // Extraction models each loop transfer with its single carried representation.
      if (op && !config && !has_op_configs && op->configs.size() == 1) {
        config = op->configs.front();
        remember(op, *config);
      }
      return config;
    };
    auto prescan_loop = [&](auto& self, const std::vector<IterArgPtr>& iter_args, const StmtPtr& body,
                            const std::vector<VarPtr>& return_vars) {
      for (const auto& ia : iter_args) {
        auto init_v = AsVarLike(ia->initValue_);
        auto cfg = IsVecTileType(ia->GetType()) ? loop_config(kVecLoopInitOp, ia) : std::nullopt;
        if (cfg) {
          record_loop_binding(ia, *cfg, init_v.get());
          vec_loop_inits_.insert(ia.get());
        } else if (init_v) {
          AliasVarPrescanInfo(ia, init_v.get(), var_tensor_lays);
        }
      }
      yield_stacks.emplace_back();
      self(self, body);
      std::vector<const Var*> yielded = std::move(yield_stacks.back());
      yield_stacks.pop_back();
      for (size_t idx = 0; idx < return_vars.size(); ++idx) {
        const Var* yielded_var = idx < yielded.size() ? yielded[idx] : nullptr;
        const Var* iter = idx < iter_args.size() ? iter_args[idx].get() : nullptr;
        const auto& result = return_vars[idx];
        auto cfg = IsVecTileType(result->GetType()) ? loop_config(kVecLoopYieldOp, result) : std::nullopt;
        if (cfg) {
          record_loop_binding(result, *cfg, yielded_var);
          if (vec_slices_.count(iter) != 0) vec_slices_.insert(result.get());
          continue;
        }
        const Var* src_v = iter ? iter : yielded_var;
        if (src_v != nullptr) {
          AliasVarPrescanInfo(result, src_v, var_tensor_lays);
        }
      }
    };

    auto walk = [&](auto& self, const StmtPtr& stmt) -> void {
      if (!stmt) {
        return;
      }
      if (auto asg = As<AssignStmt>(stmt)) {
        if (auto rhs_call = As<Call>(asg->value_)) {
          handle_call(stmt, rhs_call, asg->var_);
        } else if (auto rhs_var = AsVarLike(asg->value_)) {
          AliasVarPrescanInfo(asg->var_, rhs_var.get(), var_tensor_lays);
          if (IsVecTileType(asg->var_->GetType())) {
            vec_alias_roots_[asg->var_.get()] = VecAliasRoot(rhs_var.get());
          }
        } else if (auto tile = As<TileType>(asg->var_->GetType());
                   tile && IsVecTileType(tile) && As<TupleGetItemExpr>(asg->value_)) {
          vec_prod_lay_.insert_or_assign(asg->var_.get(), EffectiveTileLayout(tile));
        }
        record_vec_binding(asg->var_);
      } else if (auto fs = As<ForStmt>(stmt)) {
        prescan_loop(self, fs->iter_args_, fs->body_, fs->return_vars_);
        return;
      } else if (auto ws = As<WhileStmt>(stmt)) {
        prescan_loop(self, ws->iter_args_, ws->body_, ws->return_vars_);
        return;
      } else if (auto ifs = As<IfStmt>(stmt)) {
        yield_stacks.emplace_back();
        self(self, ifs->then_body_);
        std::vector<const Var*> then_y = std::move(yield_stacks.back());
        yield_stacks.pop_back();

        std::vector<const Var*> else_y;
        if (ifs->else_body_.has_value() && *ifs->else_body_) {
          yield_stacks.emplace_back();
          self(self, *ifs->else_body_);
          else_y = std::move(yield_stacks.back());
          yield_stacks.pop_back();
        }
        for (size_t idx = 0; idx < ifs->return_vars_.size(); ++idx) {
          const VarPtr& rv = ifs->return_vars_[idx];
          const Var* tv = idx < then_y.size() ? then_y[idx] : nullptr;
          const Var* ev = idx < else_y.size() ? else_y[idx] : nullptr;
          if (vec_slices_.count(tv) != 0 || vec_slices_.count(ev) != 0) vec_slices_.insert(rv.get());
          const auto* join = take_op("control.if_join", rv);
          auto join_cfg = selected(join);
          if (join && !join_cfg && !has_op_configs) {
            auto layout = LookupTileDesc(rv);
            auto choice =
                std::find_if(join->configs.begin(), join->configs.end(),
                             [&](const OperatorConfig& cfg) { return cfg.output_layout == layout; });
            if (choice != join->configs.end()) {
              join_cfg = *choice;
              remember(join, *choice);
            }
          }
          if (join_cfg.has_value()) {
            if_join_cfgs_[ifs.get()][idx] = *join_cfg;
            if (IsVecTileType(rv->GetType()) && join_cfg->output_layout.IsTile()) {
              if (auto rt = As<TileType>(rv->GetType())) {
                if (auto shape = detail::StaticShape(rt->shape_)) {
                  orig_tile_shapes_[rv.get()] = *shape;
                }
              }
              vec_prod_lay_.insert_or_assign(rv.get(), join_cfg->output_layout);
              if (tv != nullptr && join_cfg->input_layouts.size() >= 1) {
                RecordVecDemandRaw(tv, join_cfg->input_layouts[0]);
              }
              if (ev != nullptr && join_cfg->input_layouts.size() >= 2) {
                RecordVecDemandRaw(ev, join_cfg->input_layouts[1]);
              }
            } else if (AsTensorTypeLike(rv->GetType()) && join_cfg->output_layout.IsTensor()) {
              var_tensor_lays[rv.get()] = join_cfg->output_layout.tensor_layout;
            }
            continue;
          }
          if (tv != nullptr && ev != nullptr) {
            if (AsTensorTypeLike(rv->GetType())) {
              auto it_t = var_tensor_lays.find(tv);
              auto it_e = var_tensor_lays.find(ev);
              if (it_t != var_tensor_lays.end() && it_e != var_tensor_lays.end() &&
                  it_t->second == it_e->second) {
                var_tensor_lays[rv.get()] = it_t->second;
              }
            } else if (As<TileType>(rv->GetType())) {
              auto it_t = vec_prod_lay_.find(tv);
              auto it_e = vec_prod_lay_.find(ev);
              if (it_t != vec_prod_lay_.end() && it_e != vec_prod_lay_.end() &&
                  it_t->second == it_e->second) {
                AliasVarPrescanInfo(rv, tv, var_tensor_lays);
              } else if (!has_op_configs && IsVecTileType(rv->GetType())) {
                auto rt = As<TileType>(rv->GetType());
                LayoutSpec fallback_lay = LookupTileDesc(rv).value_or(EffectiveTileLayout(rt));
                if (auto shape = detail::StaticShape(rt->shape_)) {
                  orig_tile_shapes_[rv.get()] = *shape;
                }
                vec_prod_lay_.insert_or_assign(rv.get(), fallback_lay);
                RecordVecDemandRaw(tv, fallback_lay);
                RecordVecDemandRaw(ev, fallback_lay);
              }
              // Without a selected join, both yields use the inferred result layout.
              // Record that choice after resolving equal branch layouts so the
              // capacity check sees exactly the conversions requested by writeback.
              if (const auto it = vec_prod_lay_.find(rv.get());
                  join && inferred_ && !has_op_configs && it != vec_prod_lay_.end()) {
                const auto& layout = it->second;
                remember(join, {join->op_id, join->op_name, {layout, layout}, layout});
              }
            }
          } else if (tv != nullptr && else_y.empty()) {
            AliasVarPrescanInfo(rv, tv, var_tensor_lays);
          }
        }
        return;
      } else if (auto ys = As<YieldStmt>(stmt); ys && !yield_stacks.empty()) {
        std::vector<const Var*> vars;
        vars.reserve(ys->value_.size());
        for (const auto& v : ys->value_) {
          auto vv = AsVarLike(v);
          vars.push_back(vv ? vv.get() : nullptr);
        }
        yield_stacks.back() = std::move(vars);
      } else if (auto es = As<EvalStmt>(stmt)) {
        if (auto c = As<Call>(es->expr_); c && c->op_->name_ == kOpTileStore) {
          handle_call(stmt, c, nullptr);
        }
      } else if (auto rs = As<ReturnStmt>(stmt)) {
        for (size_t i = 0; i < rs->value_.size(); ++i) {
          if (auto c = As<Call>(rs->value_[i]); c && c->op_->name_ == kOpTileStore) {
            handle_call(stmt, c, nullptr, i);
          }
        }
      }
      for (const auto& child : detail::IterChildStmts(stmt)) {
        self(self, child);
      }
    };
    walk(walk, func->body_);
  }

  bool HasMutableVecBacking(const OperationNode& op) const {
    auto mutable_backing = [&](const std::string& name) {
      return instance_.values.at(name).has_mutable_vec_backing;
    };
    return mutable_backing(op.output) || std::any_of(op.inputs.begin(), op.inputs.end(), mutable_backing);
  }

  OperatorConfig ResolveMutableVecConfig(const OperationNode& modeled, OperatorConfig inferred,
                                         const Span& span) const {
    if (!HasMutableVecBacking(modeled)) return inferred;
    // Value-only writeback infers layout requirements as seen by the
    // instruction. Recover the catalog's original backing and private copy.
    for (const auto& config : modeled.configs) {
      if (modeled.op_name == kOpTileStore && config.input_layouts.size() == 2) {
        if (inferred.output_layout == config.input_layouts[1] &&
            inferred.input_layouts[0] == config.ExecutionInputLayout(0)) {
          return config;
        }
        continue;
      }
      if (config.local_vec_load_layout.value_or(config.output_layout) != inferred.output_layout ||
          config.input_layouts.size() != inferred.input_layouts.size()) {
        continue;
      }
      size_t operand = 0;
      while (operand < config.input_layouts.size() &&
             config.ExecutionInputLayout(operand) == inferred.input_layouts[operand]) {
        ++operand;
      }
      if (operand == config.input_layouts.size()) return config;
    }
    CHECK_SPAN(false, span) << "AutoLayout cannot infer a legal configuration for mutable Vec backing at '"
                            << modeled.op_id << "'";
    return inferred;
  }

  std::pair<VarPtr, std::vector<StmtPtr>> MaterializeFlatVecTranspose(
      const VarPtr& source, const LayoutSpec& src_layout, const LayoutSpec& dst_layout, const Span& span,
      const std::vector<std::pair<std::string, std::any>>& attrs = {}) {
    const std::string base = detail::CleanName(source->name_hint_);
    std::vector<StmtPtr> stmts;
    auto bind = [&](const CallPtr& call, const std::string& suffix) {
      auto value = WithCallType(call, WithVecMemorySpace(call->GetType()));
      auto var = std::make_shared<Var>(base + suffix, value->GetType(), span);
      stmts.push_back(std::make_shared<AssignStmt>(var, std::move(value), span));
      return var;
    };

    // ND TTRANS writes the element permutation to new storage. Swapping axes
    // with the flat transpose dual on one side restores the logical shape
    // while changing the backing order from ROW to COL, or vice versa.
    VarPtr row_source = source;
    if (src_layout == LayoutSpec::ColMajorFlat()) {
      row_source = bind(CreateOpCallWithAttrs(kOpTileTransposeView, {source}, {}, {}, span), "_row_view");
    }
    const auto row_type = As<TileType>(row_source->GetType());
    auto scratch_shape = std::make_shared<MakeTuple>(row_type->shape_, span);
    auto scratch = bind(
        CreateOpCallWithAttrs(kOpTileCreate, {scratch_shape},
                              {{"dtype", row_type->dtype_}, {"target_memory", MemorySpace::Vec}}, {}, span),
        "_transpose_tmp");
    auto transposed =
        bind(CreateOpCallWithAttrs(kOpTileTranspose,
                                   {row_source, MakeIndexConst(0, span), MakeIndexConst(1, span), scratch},
                                   {}, attrs, span),
             "_transposed");
    if (dst_layout == LayoutSpec::ColMajorFlat()) {
      auto view = CreateOpCallWithAttrs(kOpTileTransposeView, {transposed}, {}, {}, span);
      transposed = bind(view, "_column_view");
    }
    return {std::move(transposed), std::move(stmts)};
  }

  std::pair<VarPtr, std::vector<StmtPtr>> EmitVecConversion(const Var* orig_var, const VarPtr& src_var,
                                                            const LayoutSpec& prod_lay,
                                                            const LayoutSpec& target_lay,
                                                            const std::vector<int64_t>& orig_shape,
                                                            const Span& span) {
    // A null key denotes a private read of mutable storage. Its contents are
    // specific to this use and must never be entered in the SSA copy cache.
    if (orig_var) {
      if (auto it = vec_conv_vars_.find({orig_var, target_lay}); it != vec_conv_vars_.end()) {
        return {it->second, {}};
      }
    }
    if (prod_lay == target_lay) {
      return {src_var, {}};
    }
    auto src_tt = As<TileType>(src_var->GetType());
    if (!src_tt || src_tt->dtype_.IsPackedFp4() || orig_shape.size() != 2) {
      return {src_var, {}};
    }

    std::vector<StmtPtr> stmts;
    const std::string base_name = detail::CleanName(src_var->name_hint_);

    if (target_lay == LayoutSpec::RowBcastFlat()) {
      if (prod_lay != LayoutSpec::ColMajorFlat()) return {src_var, {}};
      const auto& cm_var = src_var;
      const auto& cm_tt = src_tt;
      const int64_t rows = orig_shape[0];
      const int64_t c0 = C0Elements(cm_tt->dtype_);
      auto shape_tuple = std::make_shared<MakeTuple>(
          std::vector<ExprPtr>{MakeIndexConst(rows, span), MakeIndexConst(c0, span)}, span);
      std::vector<std::pair<std::string, std::any>> create_kwargs = {{"dtype", cm_tt->dtype_},
                                                                     {"target_memory", MemorySpace::Vec}};
      CallPtr tgt_call = CreateOpCallWithAttrs(kOpTileCreate, {shape_tuple}, create_kwargs, {}, span);
      tgt_call = WithCallType(tgt_call, WithVecMemorySpace(tgt_call->GetType()));
      auto tgt_var = std::make_shared<Var>(base_name + "_bcast_buf", tgt_call->GetType(), span);
      stmts.push_back(std::make_shared<AssignStmt>(tgt_var, std::move(tgt_call), span));

      CallPtr exp_call = CreateOpCallWithAttrs(kOpTileRowExpand, {tgt_var, cm_var}, {}, {}, span);
      exp_call = WithCallType(exp_call, WithVecMemorySpace(exp_call->GetType()));
      auto exp_var = std::make_shared<Var>(base_name + "_bcast", exp_call->GetType(), span);
      stmts.push_back(std::make_shared<AssignStmt>(exp_var, std::move(exp_call), span));
      if (orig_var) vec_conv_vars_.emplace(std::make_pair(orig_var, target_lay), exp_var);
      return {exp_var, std::move(stmts)};
    }

    if (orig_shape[0] == 1 || orig_shape[1] == 1) return {src_var, {}};

    if (IsFlatVecTransposePair(prod_lay, target_lay)) {
      auto [converted, transpose_stmts] = MaterializeFlatVecTranspose(src_var, prod_lay, target_lay, span);
      if (orig_var) vec_conv_vars_.emplace(std::make_pair(orig_var, target_lay), converted);
      return {std::move(converted), std::move(transpose_stmts)};
    }

    return {src_var, {}};
  }

  std::vector<StmtPtr> EmitDemandedConversions(const Var* orig_var, const VarPtr& prod_var,
                                               const LayoutSpec& prod_lay, const Span& span) {
    auto dem_it = vec_demanded_.find(orig_var);
    if (dem_it == vec_demanded_.end() || !As<TileType>(prod_var->GetType())) {
      return {};
    }
    std::optional<std::vector<int64_t>> orig_shape;
    if (auto it = orig_tile_shapes_.find(orig_var); it != orig_tile_shapes_.end()) {
      orig_shape = it->second;
    } else {
      orig_shape = detail::StaticShape(As<TileType>(prod_var->GetType())->shape_);
    }
    if (!orig_shape.has_value()) {
      return {};
    }
    std::vector<StmtPtr> stmts;
    for (const auto& target_lay : dem_it->second) {
      auto [conv, extra] = EmitVecConversion(orig_var, prod_var, prod_lay, target_lay, *orig_shape, span);
      (void)conv;
      stmts.insert(stmts.end(), std::make_move_iterator(extra.begin()), std::make_move_iterator(extra.end()));
    }
    return stmts;
  }

  VarPtr GetTileVarForLayout(const VarPtr& orig_arg, const LayoutSpec& req_lay) const {
    if (req_lay.IsTile()) {
      if (auto it = vec_conv_vars_.find({VecAliasRoot(orig_arg.get()), req_lay});
          it != vec_conv_vars_.end()) {
        return it->second;
      }
    }
    auto it = local_var_remap_.find(orig_arg.get());
    return it != local_var_remap_.end() ? it->second : orig_arg;
  }

  VarPtr GetVecOperand(const VarPtr& original, const LayoutSpec& backing_layout, const OperatorConfig& cfg,
                       size_t index, const Span& span, std::vector<StmtPtr>& prefix) {
    auto source = GetTileVarForLayout(original, backing_layout);
    if (cfg.local_vec_conversions.empty() || index >= cfg.input_layouts.size()) return source;
    const auto target = cfg.ExecutionInputLayout(index);
    if (target == cfg.input_layouts[index]) return source;
    const auto type = As<TileType>(source->GetType());
    const auto shape = type ? detail::StaticShape(type->shape_) : std::nullopt;
    INTERNAL_CHECK_SPAN(shape && EffectiveTileLayout(type) == backing_layout, span)
        << "Mutable Vec read does not have its modeled backing layout";
    auto [copy, setup] = EmitVecConversion(nullptr, source, backing_layout, target, *shape, span);
    INTERNAL_CHECK_SPAN(!setup.empty(), span) << "Could not materialize the modeled mutable Vec read";
    prefix.insert(prefix.end(), std::make_move_iterator(setup.begin()), std::make_move_iterator(setup.end()));
    return copy;
  }

  static StmtPtr WithVecOperandSetup(std::vector<StmtPtr> setup, StmtPtr statement, const Span& span) {
    if (setup.empty()) return statement;
    setup.push_back(std::move(statement));
    return std::make_shared<SeqStmts>(std::move(setup), span);
  }

  std::pair<VarPtr, std::vector<StmtPtr>> GetNzNdBackingVar(const Var* orig_var, const ExprPtr& remapped_nz,
                                                            const Span& span) {
    if (auto it = nz_nd_backing_vars_.find(orig_var); it != nz_nd_backing_vars_.end()) {
      return {it->second, {}};
    }
    auto nz_tt = As<TensorType>(remapped_nz->GetType());
    if (!nz_tt) {
      throw ValueError("Expected TensorType for NZ backing view");
    }
    INTERNAL_CHECK_SPAN(!tensor_view_semantics::IsNzBlocked(*nz_tt), span)
        << "Internal error: an ND backing alias requires logical NZ coordinates";
    const int64_t c0 = C0Elements(nz_tt->dtype_);
    std::vector<ExprPtr> nd_shape = NzBackingNdShapeExprs(nz_tt->shape_, c0, span);
    auto shape_tuple = std::make_shared<MakeTuple>(std::move(nd_shape), span);
    std::vector<std::pair<std::string, std::any>> view_kwargs = {{"layout", TensorLayout::ND}};
    CallPtr view_call =
        CreateOpCallWithAttrs(kOpTensorView, {remapped_nz, shape_tuple}, view_kwargs, {}, span);
    std::string hint = "nz";
    if (auto rv = AsVarLike(remapped_nz)) {
      hint = detail::CleanName(rv->name_hint_);
    }
    auto nd_var = std::make_shared<Var>(hint + "_nd_view", view_call->GetType(), span);
    auto assign = std::make_shared<AssignStmt>(nd_var, std::move(view_call), span);
    return {nd_var, {std::move(assign)}};
  }

  struct RewriteParamsResult {
    std::vector<std::pair<VarPtr, ParamDirection>> new_params;
    bool changed = false;
    bool has_nz_param = false;
  };

  RewriteParamsResult RewriteFuncParams(const FunctionPtr& func) {
    RewriteParamsResult res;
    res.new_params.reserve(func->params_.size());
    for (size_t i = 0; i < func->params_.size(); ++i) {
      const VarPtr& param = func->params_[i];
      const ParamDirection dir = func->param_directions_[i];
      std::optional<TensorLayout> target_layout = LookupTensorLayout(param);
      auto tt = As<TensorType>(param->GetType());
      if (tt && !IsA<DistributedTensorType>(param->GetType()) && target_layout.has_value() &&
          detail::TensorTypeLayout(tt) != *target_layout && !tensor_view_semantics::IsNzBlocked(*tt) &&
          !(*target_layout == TensorLayout::NZ && tt->shape_.size() < 2)) {
        TensorTypePtr new_type = WithTensorLayout(tt, *target_layout);
        auto new_param = std::make_shared<Var>(param->name_hint_, std::move(new_type), param->span_);
        local_var_remap_[param.get()] = new_param;
        res.new_params.emplace_back(std::move(new_param), dir);
        res.changed = true;
        if (*target_layout == TensorLayout::NZ) {
          res.has_nz_param = true;
        }
      } else {
        res.new_params.emplace_back(param, dir);
        if (tt && detail::TensorTypeLayout(tt) == TensorLayout::NZ &&
            !tensor_view_semantics::IsNzBlocked(*tt)) {
          res.has_nz_param = true;
        }
      }
    }
    return res;
  }

  std::vector<StmtPtr> BuildAlignedParamPrologue(
      const std::string& func_name, const std::vector<std::pair<VarPtr, ParamDirection>>& new_params,
      const std::vector<VarPtr>& old_params, const StmtPtr& body) {
    auto nz_vars = NzOffsetScalarVars(body, new_params, old_params);
    std::vector<StmtPtr> stmts;
    for (size_t p_idx = 0; p_idx < new_params.size(); ++p_idx) {
      const VarPtr& param = new_params[p_idx].first;
      if (nz_vars.count(param.get()) == 0) {
        continue;
      }
      auto step_it = param_steps_.find({func_name, p_idx});
      auto st = As<ScalarType>(param->GetType());
      if (step_it != param_steps_.end() && step_it->second > 1 && step_it->second % 8 == 0 && st) {
        const DataType dt = st->dtype_;
        const Span& span = param->span_;
        auto c0 = std::make_shared<ConstInt>(0, dt, span);
        auto c_step = std::make_shared<ConstInt>(step_it->second, dt, span);
        auto max_expr = std::make_shared<Max>(param, std::move(c0), dt, span);
        auto div_expr = std::make_shared<FloorDiv>(std::move(max_expr), std::move(c_step), dt, span);
        auto c_step2 = std::make_shared<ConstInt>(step_it->second, dt, span);
        auto aligned_expr = std::make_shared<Mul>(std::move(div_expr), std::move(c_step2), dt, span);
        auto aligned_var =
            std::make_shared<Var>(detail::CleanName(param->name_hint_) + "_aligned", param->GetType(), span);
        local_var_remap_[param.get()] = aligned_var;
        stmts.push_back(std::make_shared<AssignStmt>(std::move(aligned_var), std::move(aligned_expr), span));
      }
    }
    return stmts;
  }

  std::vector<StmtPtr> BuildIncoreVecPrologue(
      const FunctionPtr& func, const std::vector<std::pair<VarPtr, ParamDirection>>& new_params) {
    PrescanFunction(func, new_params);
    std::vector<StmtPtr> stmts;
    for (size_t i = 0; i < new_params.size() && i < func->params_.size(); ++i) {
      const VarPtr& new_p = new_params[i].first;
      const VarPtr& old_p = func->params_[i];
      if (nz_nd_needed_vars_.count(old_p.get()) != 0) {
        if (auto tt = As<TensorType>(new_p->GetType());
            tt && detail::TensorTypeLayout(tt) == TensorLayout::NZ) {
          auto [nd_var, extra] = GetNzNdBackingVar(old_p.get(), new_p, new_p->span_);
          nz_nd_backing_vars_[old_p.get()] = std::move(nd_var);
          stmts.insert(stmts.end(), std::make_move_iterator(extra.begin()),
                       std::make_move_iterator(extra.end()));
          continue;
        }
      }
      if (IsVecTileType(new_p->GetType())) {
        if (auto prod_it = vec_prod_lay_.find(old_p.get()); prod_it != vec_prod_lay_.end()) {
          std::vector<StmtPtr> extra =
              EmitDemandedConversions(old_p.get(), new_p, prod_it->second, new_p->span_);
          stmts.insert(stmts.end(), std::make_move_iterator(extra.begin()),
                       std::make_move_iterator(extra.end()));
        }
      }
    }
    return stmts;
  }

  std::pair<std::vector<TypePtr>, bool> SyncFuncReturnTypes(
      const FunctionPtr& func, const std::vector<std::pair<VarPtr, ParamDirection>>& new_params,
      const StmtPtr& new_body) const {
    std::vector<TypePtr> new_return_types = func->return_types_;
    bool changed = false;
    std::vector<ReturnStmtPtr> ret_stmts;
    FindReturnStmts(new_body, ret_stmts);
    if (!ret_stmts.empty()) {
      for (const auto& rs : ret_stmts) {
        std::vector<TypePtr> value_types;
        for (const auto& value : rs->value_) value_types.push_back(value->GetType());
        if (value_types.size() == 1 && new_return_types.size() > 1) {
          if (auto tuple = As<TupleType>(value_types.front())) value_types = tuple->types_;
        }
        for (size_t i = 0; i < value_types.size() && i < new_return_types.size(); ++i) {
          auto type = SyncGmLayouts(new_return_types[i], value_types[i]);
          changed |= type.get() != new_return_types[i].get();
          new_return_types[i] = std::move(type);
        }
      }
      return {std::move(new_return_types), changed};
    }
    // Some wrappers forward the result of their inner call without an explicit
    // ReturnStmt. The validated lineage identifies the forwarded parameter.
    if (auto wrapper = wrapper_returns_.find(func.get()); wrapper != wrapper_returns_.end()) {
      for (size_t i = 0; i < wrapper->second.size() && i < new_return_types.size(); ++i) {
        const auto& param = wrapper->second[i];
        if (param && *param < new_params.size()) {
          auto type = SyncGmLayouts(new_return_types[i], new_params[*param].first->GetType());
          changed |= type.get() != new_return_types[i].get();
          new_return_types[i] = std::move(type);
        }
      }
    }
    return {std::move(new_return_types), changed};
  }

  struct StoreRewriteResult {
    CallPtr store_call;
    std::vector<StmtPtr> pre_stmts;
    ExprPtr nz_logical_shape_src;
  };

  StoreRewriteResult RewriteTileStore(const CallPtr& rhs, const OperatorConfig* cfg, const Span& span) {
    std::optional<LayoutSpec> store_gm_lay;
    if (cfg != nullptr) {
      store_gm_lay = cfg->input_layouts.size() >= 2 ? cfg->input_layouts[1] : cfg->output_layout;
    }
    if (cfg != nullptr && !cfg->input_layouts.empty() && cfg->input_layouts[0] == LayoutSpec::FractalNZ() &&
        store_gm_lay == LayoutSpec::Tensor(TensorLayout::NZ) && rhs->args_.size() >= 3) {
      auto tile_orig = AsVarLike(rhs->args_[0]);
      auto offs_tuple = As<MakeTuple>(rhs->args_[1]);
      auto dst_orig = AsVarLike(rhs->args_[2]);
      if (tile_orig && IsVecTileType(tile_orig->GetType()) &&
          !As<TileType>(tile_orig->GetType())->dtype_.IsPackedFp4() && offs_tuple && dst_orig &&
          !UsesBlockedNzCoordinates(dst_orig->GetType())) {
        auto tile_t = As<TileType>(tile_orig->GetType());
        auto remap_it = local_var_remap_.find(tile_orig.get());
        VarPtr new_tile = remap_it != local_var_remap_.end() ? remap_it->second : tile_orig;
        ExprPtr new_dst = VisitExpr(dst_orig);
        int64_t tr = 0;
        int64_t tc = 0;
        if (auto sit = vec_logical_shapes_.find(tile_orig.get());
            sit != vec_logical_shapes_.end() && sit->second.size() >= 2) {
          tr = sit->second[0];
          tc = sit->second[1];
        } else if (auto st = detail::StaticShape(tile_t->shape_); st.has_value() && st->size() >= 2) {
          tr = (*st)[0];
          tc = (*st)[1];
        }
        auto new_dst_tt = As<TensorType>(new_dst->GetType());
        const int64_t c0 = C0Elements(new_dst_tt->dtype_);
        auto [backing_var, pre_stmts] = GetNzNdBackingVar(dst_orig.get(), new_dst, span);
        std::vector<ExprPtr> visited_offs;
        visited_offs.reserve(offs_tuple->elements_.size());
        for (const auto& el : offs_tuple->elements_) {
          visited_offs.push_back(VisitExpr(el));
        }
        auto [nd_offs, nd_win] =
            NzBackingNdOffsetsAndWindow(visited_offs, new_dst_tt->shape_, tr, tc, c0, rhs->span_);
        std::vector<ExprPtr> new_args = {
            std::move(new_tile),
            std::make_shared<MakeTuple>(std::move(nd_offs), offs_tuple->span_),
            std::move(backing_var),
        };
        if (rhs->args_.size() >= 4) {
          new_args.push_back(std::make_shared<MakeTuple>(std::move(nd_win), rhs->args_[3]->span_));
        }
        CallPtr store_call =
            CreateOpCallWithAttrs(kOpTileStore, new_args, rhs->kwargs_, rhs->attrs_, rhs->span_);
        return {std::move(store_call), std::move(pre_stmts), std::move(new_dst)};
      }
    }

    std::vector<ExprPtr> new_args;
    std::vector<StmtPtr> prefix;
    new_args.reserve(rhs->args_.size());
    bool args_changed = false;
    for (size_t idx = 0; idx < rhs->args_.size(); ++idx) {
      const ExprPtr& arg = rhs->args_[idx];
      if (idx == 0 && cfg != nullptr && !cfg->input_layouts.empty()) {
        if (auto av = AsVarLike(arg); av && IsVecTileType(av->GetType())) {
          VarPtr tv = GetVecOperand(av, cfg->input_layouts[0], *cfg, 0, span, prefix);
          if (tv.get() != arg.get()) {
            args_changed = true;
          }
          new_args.push_back(std::move(tv));
          continue;
        }
      }
      ExprPtr visited = VisitExpr(arg);
      if (visited.get() != arg.get()) {
        args_changed = true;
      }
      new_args.push_back(std::move(visited));
    }
    if (!args_changed) {
      return {rhs, {}, nullptr};
    }
    CallPtr store_call = CreateOpCallWithAttrs(kOpTileStore, new_args, rhs->kwargs_, rhs->attrs_, rhs->span_);
    return {std::move(store_call), std::move(prefix), nullptr};
  }

  // A Vec NZ store writes the flattened ND backing. Restore the logical
  // tensor result for both an assignment and a store returned directly.
  ExprPtr RewriteStoredResult(const CallPtr& rhs, const OperatorConfig* cfg, const std::string& name,
                              const Span& span, std::vector<StmtPtr>& prefix) {
    auto [store, setup, logical] = RewriteTileStore(rhs, cfg, span);
    prefix = std::move(setup);
    if (!logical) return store;
    auto stored = std::make_shared<Var>(name + "_nd_store", store->GetType(), span);
    prefix.push_back(std::make_shared<AssignStmt>(stored, std::move(store), span));
    auto shape = std::make_shared<MakeTuple>(AsTensorTypeLike(logical->GetType())->shape_, span);
    return CreateOpCallWithAttrs(kOpTensorView, {stored, shape}, {{"layout", TensorLayout::NZ}}, {}, span);
  }

  StmtPtr FinishVecAssign(const VarPtr& orig_var, ExprPtr new_value,
                          const std::optional<LayoutSpec>& prod_lay, const Span& span,
                          const AssignStmtPtr& orig_stmt) {
    VarPtr new_var = orig_var;
    const TypePtr& orig_t = orig_var->GetType();
    const TypePtr& new_t = new_value->GetType();
    if (new_value.get() != orig_stmt->value_.get() &&
        (AsTensorTypeLike(orig_t) || As<TileType>(orig_t) || As<TupleType>(orig_t)) &&
        (AsTensorTypeLike(new_t) || As<TileType>(new_t) || As<TupleType>(new_t))) {
      new_var = std::make_shared<Var>(orig_var->name_hint_, new_t, orig_var->span_);
      local_var_remap_[orig_var.get()] = new_var;
    }

    StmtPtr main_stmt;
    if (new_var.get() == orig_var.get() && new_value.get() == orig_stmt->value_.get()) {
      main_stmt = orig_stmt;
    } else {
      main_stmt = std::make_shared<AssignStmt>(new_var, std::move(new_value), span);
    }

    if (prod_lay.has_value() && vec_demanded_.count(orig_var.get()) != 0) {
      std::vector<StmtPtr> conv_stmts = EmitDemandedConversions(orig_var.get(), new_var, *prod_lay, span);
      if (!conv_stmts.empty()) {
        std::vector<StmtPtr> seq;
        seq.reserve(1 + conv_stmts.size());
        seq.push_back(std::move(main_stmt));
        seq.insert(seq.end(), std::make_move_iterator(conv_stmts.begin()),
                   std::make_move_iterator(conv_stmts.end()));
        return std::make_shared<SeqStmts>(std::move(seq), span);
      }
    }
    return main_stmt;
  }

  StmtPtr RewriteVecLoadAssign(const AssignStmtPtr& op, const CallPtr& rhs, const OperatorConfig& cfg) {
    auto lhs_tt = As<TileType>(op->var_->GetType());
    // Packed tiles already carry their native byte geometry. Flattening an
    // explicitly boxed packed load would reinterpret the packed axis.
    if (!lhs_tt->dtype_.IsPackedFp4() && cfg.output_layout == LayoutSpec::FractalNZ() &&
        !cfg.input_layouts.empty() && cfg.input_layouts[0] == LayoutSpec::Tensor(TensorLayout::NZ) &&
        rhs->args_.size() >= 3) {
      auto gm_var = AsVarLike(rhs->args_[0]);
      auto offs_tuple = As<MakeTuple>(rhs->args_[1]);
      if (gm_var && offs_tuple && !UsesBlockedNzCoordinates(gm_var->GetType())) {
        ExprPtr new_gm = VisitExpr(gm_var);
        auto new_gm_tt = As<TensorType>(new_gm->GetType());
        int64_t tr = 0;
        int64_t tc = 0;
        if (auto sit = vec_logical_shapes_.find(op->var_.get());
            sit != vec_logical_shapes_.end() && sit->second.size() >= 2) {
          tr = sit->second[0];
          tc = sit->second[1];
        } else if (auto st = detail::StaticShape(lhs_tt->shape_); st.has_value() && st->size() >= 2) {
          tr = (*st)[0];
          tc = (*st)[1];
        }
        const int64_t c0 = C0Elements(lhs_tt->dtype_);
        auto [backing_var, pre_stmts] = GetNzNdBackingVar(gm_var.get(), new_gm, op->span_);
        std::vector<ExprPtr> visited_offs;
        visited_offs.reserve(offs_tuple->elements_.size());
        for (const auto& el : offs_tuple->elements_) {
          visited_offs.push_back(VisitExpr(el));
        }
        auto [nd_offs, nd_win] =
            NzBackingNdOffsetsAndWindow(visited_offs, new_gm_tt->shape_, tr, tc, c0, rhs->span_);
        std::vector<ExprPtr> new_args = {
            std::move(backing_var),
            std::make_shared<MakeTuple>(std::move(nd_offs), offs_tuple->span_),
            std::make_shared<MakeTuple>(nd_win, rhs->args_[2]->span_),
        };
        if (rhs->args_.size() >= 4) {
          new_args.push_back(std::make_shared<MakeTuple>(std::move(nd_win), rhs->args_[3]->span_));
        }
        CallPtr load_call =
            CreateOpCallWithAttrs(kOpTileLoad, new_args, rhs->kwargs_, rhs->attrs_, rhs->span_);
        load_call = WithCallType(load_call, WithVecMemorySpace(load_call->GetType()));
        auto new_var = std::make_shared<Var>(op->var_->name_hint_, load_call->GetType(), op->var_->span_);
        local_var_remap_[op->var_.get()] = new_var;
        auto assign_load = std::make_shared<AssignStmt>(std::move(new_var), std::move(load_call), op->span_);
        if (!pre_stmts.empty()) {
          pre_stmts.push_back(std::move(assign_load));
          return std::make_shared<SeqStmts>(std::move(pre_stmts), op->span_);
        }
        return assign_load;
      }
    }

    std::vector<ExprPtr> new_args;
    new_args.reserve(rhs->args_.size());
    bool args_changed = false;
    for (const auto& arg : rhs->args_) {
      ExprPtr visited = VisitExpr(arg);
      if (visited.get() != arg.get()) {
        args_changed = true;
      }
      new_args.push_back(std::move(visited));
    }
    const auto load_layout = cfg.local_vec_load_layout.value_or(cfg.output_layout);
    const bool change_flat_layout =
        !lhs_tt->dtype_.IsPackedFp4() && IsFlatVecTransposePair(EffectiveTileLayout(lhs_tt), load_layout);
    ExprPtr new_value = rhs;
    if (args_changed || change_flat_layout) {
      auto load_type = op->var_->GetType();
      if (change_flat_layout) {
        auto view = tile_view_semantics::GetEffectiveTileView(*lhs_tt);
        view.blayout = load_layout.blayout;
        load_type = WithVecMemorySpace(lhs_tt, view);
      }
      new_value = std::make_shared<Call>(rhs->op_, std::move(new_args), rhs->kwargs_, rhs->attrs_,
                                         std::move(load_type), rhs->span_);
    }
    if (cfg.local_vec_load_layout) {
      const auto shape = detail::StaticShape(lhs_tt->shape_);
      INTERNAL_CHECK_SPAN(shape, op->span_) << "Mutable Vec load requires a static shape";
      auto loaded = std::make_shared<Var>(detail::CleanName(op->var_->name_hint_) + "_loaded",
                                          new_value->GetType(), op->span_);
      auto [backing, setup] =
          EmitVecConversion(nullptr, loaded, load_layout, cfg.output_layout, *shape, op->span_);
      INTERNAL_CHECK_SPAN(!setup.empty(), op->span_) << "Could not initialize the modeled mutable Vec load";
      setup.insert(setup.begin(), std::make_shared<AssignStmt>(loaded, std::move(new_value), op->span_));
      auto assignment = FinishVecAssign(op->var_, std::move(backing), cfg.output_layout, op->span_, op);
      return WithVecOperandSetup(std::move(setup), std::move(assignment), op->span_);
    }
    return FinishVecAssign(op->var_, std::move(new_value), cfg.output_layout, op->span_, op);
  }

  StmtPtr RewriteTileMoveAssign(const AssignStmtPtr& op, const CallPtr& rhs, const OperatorConfig* cfg,
                                const std::optional<LayoutSpec>& target_tile_desc) {
    auto lhs_tt = As<TileType>(op->var_->GetType());
    const bool dst_is_vec = lhs_tt->GetMemorySpace() == MemorySpace::Vec;
    const bool packed = lhs_tt->dtype_.IsPackedFp4();
    std::vector<ExprPtr> new_args;
    std::vector<StmtPtr> prefix;
    new_args.reserve(rhs->args_.size());
    bool args_changed = false;
    VarPtr in_arg = !rhs->args_.empty() ? AsVarLike(rhs->args_[0]) : nullptr;
    if (!packed && dst_is_vec && cfg != nullptr && !cfg->input_layouts.empty() && in_arg &&
        IsVecTileType(in_arg->GetType())) {
      VarPtr new_src = GetVecOperand(in_arg, cfg->input_layouts[0], *cfg, 0, op->span_, prefix);
      if (new_src.get() != in_arg.get()) {
        args_changed = true;
      }
      new_args.push_back(std::move(new_src));
      for (size_t i = 1; i < rhs->args_.size(); ++i) {
        ExprPtr v = VisitExpr(rhs->args_[i]);
        if (v.get() != rhs->args_[i].get()) {
          args_changed = true;
        }
        new_args.push_back(std::move(v));
      }
    } else {
      for (const auto& arg : rhs->args_) {
        ExprPtr v = VisitExpr(arg);
        if (v.get() != arg.get()) {
          args_changed = true;
        }
        new_args.push_back(std::move(v));
      }
    }

    std::optional<LayoutSpec> chosen_desc = (cfg != nullptr && cfg->output_layout.IsTile())
                                                ? std::optional<LayoutSpec>(cfg->output_layout)
                                                : target_tile_desc;
    if (packed) chosen_desc = EffectiveTileLayout(lhs_tt);
    if (!packed && dst_is_vec && chosen_desc && !new_args.empty()) {
      auto source = AsVarLike(new_args[0]);
      auto source_type = source ? As<TileType>(source->GetType()) : nullptr;
      if (source_type && IsVecTileType(source_type)) {
        auto source_layout = EffectiveTileLayout(source_type);
        auto geometry = detail::VecTransposeGeometry(source_type);
        // A copy has fresh geometry. For the original backing use the
        // extractor's pitch and slice provenance, including tuple projections.
        const auto remap = local_var_remap_.find(in_arg.get());
        if (source == (remap == local_var_remap_.end() ? in_arg : remap->second)) {
          if (const auto* node = LookupTileValue(in_arg)) {
            geometry.is_vec_slice = node->is_vec_slice;
            geometry.is_strided_subview = node->is_strided_subview;
            geometry.has_dense_vec_span &= node->has_dense_vec_span;
          } else {
            geometry.is_vec_slice = vec_slices_.count(in_arg.get()) != 0;
          }
        }
        if (IsFlatVecTransposePair(source_layout, *chosen_desc) &&
            LayoutCostModel{}.SupportsVecTileConversion(geometry, source_layout, *chosen_desc)) {
          auto [converted, stmts] =
              MaterializeFlatVecTranspose(source, source_layout, *chosen_desc, op->span_, rhs->attrs_);
          prefix.insert(prefix.end(), std::make_move_iterator(stmts.begin()),
                        std::make_move_iterator(stmts.end()));
          auto assignment = FinishVecAssign(op->var_, std::move(converted), chosen_desc, op->span_, op);
          return WithVecOperandSetup(std::move(prefix), std::move(assignment), op->span_);
        }
      }
    }
    if (!chosen_desc.has_value() || (EffectiveTileLayout(lhs_tt) == *chosen_desc && !args_changed)) {
      return nullptr;
    }
    auto new_kwargs = ReplaceOrInsertKwarg(rhs->kwargs_, "blayout", chosen_desc->blayout);
    new_kwargs = ReplaceOrInsertKwarg(std::move(new_kwargs), "slayout", chosen_desc->slayout);
    CallPtr new_call = CreateOpCallWithAttrs(rhs->op_->name_, new_args, new_kwargs, rhs->attrs_, rhs->span_);
    if (dst_is_vec) {
      new_call = WithCallType(new_call, WithVecMemorySpace(new_call->GetType()));
    }
    auto assignment = FinishVecAssign(op->var_, std::move(new_call),
                                      dst_is_vec ? chosen_desc : std::optional<LayoutSpec>{}, op->span_, op);
    return WithVecOperandSetup(std::move(prefix), std::move(assignment), op->span_);
  }

  StmtPtr RewriteSingletonRowSum(const AssignStmtPtr& op, const CallPtr& rhs) {
    const Span& span = rhs->span_;
    const auto type = As<TileType>(op->var_->GetType());
    const auto source = GetTileVarForLayout(AsVarLike(rhs->args_[0]), LayoutSpec::ColMajorFlat());
    const auto row_shape =
        std::make_shared<MakeTuple>(std::vector<ExprPtr>{type->shape_[1], type->shape_[0]}, span);
    const auto col_shape = std::make_shared<MakeTuple>(type->shape_, span);
    const auto base = detail::CleanName(op->var_->name_hint_);

    // TROWSUM requires row-major input. Each original row has one valid cell,
    // so copy the compact run through two storage views and keep the assigned
    // value's [M, 1] axes and valid extent. The copy gives the result its own
    // storage, so writing it later cannot affect another use of the source.
    auto row = CreateOpCallWithAttrs(kOpTileReshape, {source, row_shape}, {}, {}, span);
    row = WithCallType(row, WithVecMemorySpace(row->GetType()));
    auto src_row = std::make_shared<Var>(base + "_input_row", row->GetType(), span);
    auto view_stmt = std::make_shared<AssignStmt>(src_row, std::move(row), span);
    auto move = CreateOpCallWithAttrs(kOpTileMove, {src_row}, {{"target_memory", MemorySpace::Vec}},
                                      rhs->attrs_, span);
    auto dst_row = std::make_shared<Var>(base + "_copied_row", move->GetType(), span);
    auto move_stmt = std::make_shared<AssignStmt>(dst_row, std::move(move), span);
    auto column = CreateOpCallWithAttrs(kOpTileReshape, {dst_row, col_shape}, {}, {}, span);
    column = WithCallType(column, WithVecMemorySpace(column->GetType()));
    auto result = FinishVecAssign(op->var_, std::move(column), LayoutSpec::ColMajorFlat(), op->span_, op);
    return std::make_shared<SeqStmts>(
        std::vector<StmtPtr>{std::move(view_stmt), std::move(move_stmt), std::move(result)}, op->span_);
  }

  StmtPtr RewriteCompactColumnFill(const AssignStmtPtr& op, const CallPtr& rhs) {
    const auto type = As<TileType>(op->var_->GetType());
    const auto shape = detail::StaticShape(type->shape_);
    const int bytes = DTypeByteWidthSafe(type->dtype_);
    if (!shape || shape->size() != 2 || (*shape)[0] <= 1 || (*shape)[1] != 1 ||
        EffectiveTileLayout(type) != LayoutSpec::ColMajorFlat() || type->dtype_.IsPackedFp4() || bytes <= 0 ||
        ((*shape)[0] * bytes) % 32 != 0) {
      return nullptr;
    }

    // TEXPANDS requires a row-major buffer. A dense [1, M] row and an
    // [M, 1] column have the same byte sequence. Bind the original result
    // through a view so every consumer still sees the original logical axes.
    const Span& span = rhs->span_;
    const auto compact_row =
        std::make_shared<MakeTuple>(std::vector<ExprPtr>{type->shape_[1], type->shape_[0]}, span);
    auto fill = CreateOpCallWithAttrs(kOpTileFull, {compact_row, VisitExpr(rhs->args_[1])}, rhs->kwargs_,
                                      rhs->attrs_, span);
    fill = WithCallType(fill, WithVecMemorySpace(fill->GetType()));
    auto compact =
        std::make_shared<Var>(detail::CleanName(op->var_->name_hint_) + "_fill_row", fill->GetType(), span);
    auto fill_stmt = std::make_shared<AssignStmt>(compact, std::move(fill), span);
    auto column = CreateOpCallWithAttrs(kOpTileReshape, {compact, VisitExpr(rhs->args_[0])}, {}, {}, span);
    column = WithCallType(column, WithVecMemorySpace(column->GetType()));
    auto result = FinishVecAssign(op->var_, std::move(column), LayoutSpec::ColMajorFlat(), op->span_, op);
    return std::make_shared<SeqStmts>(std::vector<StmtPtr>{std::move(fill_stmt), std::move(result)},
                                      op->span_);
  }

  StmtPtr RewriteVecGenericAssign(const AssignStmtPtr& op, const CallPtr& rhs, const OperatorConfig& cfg) {
    if (detail::CanCopySingletonRowSum(rhs) && AsVarLike(rhs->args_[0])) {
      return RewriteSingletonRowSum(op, rhs);
    }
    if (IsOp(rhs, kOpTileFull)) {
      if (auto compact = RewriteCompactColumnFill(op, rhs)) return compact;
    }
    std::vector<ExprPtr> new_args;
    std::vector<StmtPtr> prefix;
    new_args.reserve(rhs->args_.size());
    size_t tile_idx = 0;
    const bool preserve_descriptors = detail::PreservesTileDescriptors(rhs);
    for (const auto& arg : rhs->args_) {
      if (auto av = AsVarLike(arg); av && As<TileType>(av->GetType())) {
        const LayoutSpec req =
            preserve_descriptors
                ? EffectiveTileLayout(As<TileType>(av->GetType()))
                : (tile_idx < cfg.input_layouts.size() ? cfg.input_layouts[tile_idx] : cfg.output_layout);
        new_args.push_back(GetVecOperand(av, req, cfg, tile_idx++, op->span_, prefix));
      } else {
        new_args.push_back(VisitExpr(arg));
      }
    }

    bool args_changed = (new_args.size() != rhs->args_.size());
    if (!args_changed) {
      for (size_t i = 0; i < new_args.size(); ++i) {
        if (new_args[i].get() != rhs->args_[i].get()) {
          args_changed = true;
          break;
        }
      }
    }

    ExprPtr new_value = rhs;
    if (args_changed && preserve_descriptors) {
      auto [attrs, _] = MutateAttrs(rhs->attrs_);
      new_value = std::make_shared<Call>(rhs->op_, std::move(new_args), rhs->kwargs_, std::move(attrs),
                                         RemapTypeViaVisitor(rhs->GetType()), rhs->span_);
    } else if (args_changed) {
      CallPtr rebuilt =
          CreateOpCallWithAttrs(rhs->op_->name_, new_args, rhs->kwargs_, rhs->attrs_, rhs->span_);
      std::optional<TileView> tv_override;
      if (cfg.output_layout == LayoutSpec::ColMajorFlat()) {
        if (auto rt = As<TileType>(rebuilt->GetType());
            rt && EffectiveTileLayout(rt) != LayoutSpec::ColMajorFlat()) {
          tv_override = ColMajorTileView(rt);
        }
      }
      new_value = WithCallType(rebuilt, WithVecMemorySpace(rebuilt->GetType(), tv_override));
    }
    const auto output =
        preserve_descriptors ? EffectiveTileLayout(As<TileType>(op->var_->GetType())) : cfg.output_layout;
    auto assignment = FinishVecAssign(op->var_, std::move(new_value), output, op->span_, op);
    return WithVecOperandSetup(std::move(prefix), std::move(assignment), op->span_);
  }

  std::unordered_map<std::string, TensorLayout> tensor_layouts_;
  std::unordered_map<std::string, LayoutSpec> tile_layouts_;
  const std::unordered_map<std::string, OperatorConfig>& op_configs_;

  const LayoutProblemInstance& instance_;
  LayoutWritebackBindings& bindings_;
  HostMatrixDispatchWriter host_dispatch_writer_;
  LayoutAssignment* inferred_;
  const LayoutWritebackBindings::Site* current_tile_site_ = nullptr;
  OrchCallAnalysis::GmNames gm_names_;
  std::unordered_map<std::string, std::string> tile_prefixes_;
  std::unordered_map<std::pair<std::string, ParamSig>, std::string, ParamSigHash> specialized_callees_;
  std::unordered_map<const Expr*, std::string> specialized_calls_;
  std::unordered_map<std::string, std::vector<VarPtr>> callee_formal_params_;
  std::map<std::pair<std::string, size_t>, int64_t> param_steps_;
  std::unordered_map<const Var*, VarPtr> local_var_remap_;
  std::unordered_map<std::string, std::vector<TypePtr>> func_return_types_;
  std::unordered_map<const Function*, std::vector<std::optional<size_t>>> wrapper_returns_;
  std::string current_func_name_;

  // Per-function prescan state.
  std::unordered_map<const Var*, OperatorConfig> assign_op_cfg_;
  std::unordered_map<const IfStmt*, std::unordered_map<size_t, OperatorConfig>> if_join_cfgs_;
  std::unordered_map<const Stmt*, std::unordered_map<size_t, OperatorConfig>> standalone_store_cfgs_;
  std::unordered_map<const Var*, LayoutSpec> vec_prod_lay_;
  std::unordered_map<const Var*, const Var*> vec_alias_roots_;
  std::unordered_set<const Var*> vec_loop_inits_;
  std::unordered_set<const Var*> vec_slices_;
  std::unordered_map<const Var*, std::vector<LayoutSpec>> vec_demanded_;
  std::unordered_map<std::pair<const Var*, LayoutSpec>, VarPtr, LayoutSpecKeyHash> vec_conv_vars_;
  std::unordered_map<const Var*, std::vector<int64_t>> vec_logical_shapes_;
  std::unordered_map<const Var*, std::vector<int64_t>> orig_tile_shapes_;
  std::unordered_set<const Var*> nz_nd_needed_vars_;
  std::unordered_map<const Var*, VarPtr> nz_nd_backing_vars_;
  struct YieldFrame {
    std::vector<std::optional<LayoutSpec>> loop_expected;
    std::vector<TypePtr> branch_yield_types;
    bool is_if_branch = false;
    std::vector<bool> modeled_loop_carries;
  };
  std::vector<YieldFrame> yield_frames_;
};

}  // namespace

TypePtr SyncCallReturnType(const TypePtr& call_type, const std::vector<TypePtr>& ret_types) {
  if (ret_types.empty()) {
    return call_type;
  }
  if (ret_types.size() == 1) return SyncGmLayouts(call_type, ret_types.front());
  return SyncGmResultElements(call_type, ret_types);
}

ProgramPtr ApplyLayoutAssignment(const ProgramPtr& program, const LayoutAssignment& assignment) {
  LayoutWritebackBindings bindings;
  const auto instance = ExtractLayoutInstance(program, {}, {}, {}, &bindings);
  LayoutAssignment effective = assignment;
  LayoutAssignmentWriter writer(instance, bindings, assignment, &effective);
  auto rewritten = writer.Rewrite(bindings.program);
  // Check the configurations that value-only writeback inferred from the actual IR.
  const auto issues = CheckVecConversionCapacity(instance, effective);
  if (!issues.empty()) throw ValueError(issues.front());
  return rewritten;
}

ProgramPtr detail::ApplySolvedLayout(const LayoutProblemInstance& instance, LayoutWritebackBindings bindings,
                                     const LayoutAssignment& assignment) {
  LayoutAssignmentWriter writer(instance, bindings, assignment);
  return writer.Rewrite(bindings.program);
}

}  // namespace auto_layout
}  // namespace ir
}  // namespace pypto
