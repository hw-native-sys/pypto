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
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/deep_clone_utils.h"
#include "pypto/ir/transforms/utils/loop_state_repair.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/split_axis_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

constexpr const char* kDualAivDispatchAttr = "dual_aiv_dispatch";
constexpr const char* kExternalSourceAttr = "external_source";

using split_axis::InjectSubblockIdx;
using split_axis::ProcessStmts;
using split_axis::SplitDimension;
using split_axis::TileInfo;

bool RequiresNoSplitDualAivSync(const FunctionPtr& func) {
  return func != nullptr && func->func_type_ == FunctionType::AIV &&
         pypto::backend::BackendConfig::IsConfigured() &&
         pypto::ir::PassContext::Current()->GetBackendHandler()->RequiresNoSplitDualAivDispatch() &&
         func->HasAttr(kDualAivDispatchAttr) && func->GetAttr<bool>(kDualAivDispatchAttr, false);
}

// ---------------------------------------------------------------------------
// Standalone (non-split_aiv) split lowering.
//
// The auto-split pipeline (pl.split mixed InCore -> LowerAutoVectorSplit ->
// ExpandMixedKernel) reaches this pass with each split function already
// ``split_aiv``-marked and pre-halved (handled by the split_aiv branch below).
// But hand-written *standalone* AIV/AIC kernels — separate functions whose
// cross-core split is signalled either by an explicit function ``split`` attr
// or by a ``split=N`` kwarg on their tpush/tpop pipe ops — never pass through
// LowerAutoVectorSplit (it is InCore-only) and arrive here at full shape. They
// still need the per-lane halving + get_subblock_idx injection. This is the
// dedicated path for them; it reuses the shared ``split_axis`` halving driver
// (the same machinery LowerAutoVectorSplit calls), so the produced per-lane
// body is identical. It runs ONLY for functions that are NOT ``split_aiv``
// marked, so it can never double-halve an already-lowered auto-split function.
// ---------------------------------------------------------------------------

bool IsCrossCoreSplitOp(const OpPtr& op) {
  return IsOp(op, "tile.tpush_to_aiv") || IsOp(op, "tile.tpush_to_aic") || IsOp(op, "tile.tpop_from_aiv") ||
         IsOp(op, "tile.tpop_from_aic");
}

std::optional<SplitMode> SplitModeFromInt(int split) {
  if (split == 0) return std::nullopt;
  if (split == 1) return SplitMode::UpDown;
  if (split == 2) return SplitMode::LeftRight;
  throw pypto::ValueError("SplitVectorKernel found invalid cross-core split attribute: " +
                          std::to_string(split));
}

// Infer the split mode from the function body's cross-core pipe ops. All
// split-carrying pipe ops in one function must agree on the mode.
class CrossCoreSplitCollector : public IRVisitor {
 public:
  [[nodiscard]] std::optional<SplitMode> GetInferredMode() const { return inferred_mode_; }

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    ConsiderCall(As<Call>(op->value_));
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const EvalStmtPtr& op) override {
    ConsiderCall(As<Call>(op->expr_));
    IRVisitor::VisitStmt_(op);
  }

 private:
  std::optional<SplitMode> inferred_mode_;

  void ConsiderCall(const CallPtr& call) {
    if (!call || !call->op_ || !IsCrossCoreSplitOp(call->op_)) return;

    auto mode = SplitModeFromInt(call->GetKwarg<int>("split", 0));
    if (!mode.has_value()) return;

    if (!inferred_mode_.has_value()) {
      inferred_mode_ = mode;
      return;
    }

    if (inferred_mode_.value() != mode.value()) {
      throw pypto::ValueError("SplitVectorKernel found conflicting cross-core split modes in function body");
    }
  }
};

// Resolve the split mode for a standalone function: an explicit function
// ``split`` attr takes precedence (and must not conflict with any pipe-op
// inferred mode); otherwise the pipe-op inferred mode is used.
std::optional<SplitMode> ResolveSplitMode(const FunctionPtr& func) {
  CrossCoreSplitCollector collector;
  if (func->body_) {
    collector.VisitStmt(func->body_);
  }
  auto inferred_mode = collector.GetInferredMode();

  auto func_split_mode = func->GetSplitMode();
  if (func_split_mode.has_value() && func_split_mode.value() != SplitMode::None) {
    if (inferred_mode.has_value() && inferred_mode.value() != func_split_mode.value()) {
      throw pypto::ValueError("SplitVectorKernel found conflicting function split and cross-core op split");
    }
    return func_split_mode;
  }

  return inferred_mode;
}

std::vector<std::pair<std::string, std::any>> WithSplitAttrs(const FunctionPtr& func, SplitMode mode,
                                                             bool is_aiv) {
  auto attrs = func->attrs_;
  attrs.erase(
      std::remove_if(attrs.begin(), attrs.end(),
                     [](const auto& kv) { return kv.first == "split" || kv.first == kDualAivDispatchAttr; }),
      attrs.end());
  if (mode != SplitMode::None) {
    attrs.emplace_back("split", static_cast<int>(mode));
    // AIV functions with a non-None split mode require dual-AIV dispatch at
    // codegen. Stamp the attribute here so codegen reads it directly instead
    // of re-deriving from SplitMode.
    if (is_aiv) {
      attrs.emplace_back(kDualAivDispatchAttr, true);
    }
  }
  return attrs;
}

ExprPtr MakeIndexConst(int64_t value, const Span& span) {
  return std::make_shared<ConstInt>(value, DataType::INDEX, span);
}

TypePtr WithZeroValidShape(const TypePtr& type, const Span& span) {
  auto tt = std::dynamic_pointer_cast<const TileType>(type);
  if (!tt) return type;

  TileView tile_view = tile_view_semantics::GetEffectiveTileView(*tt);
  tile_view.valid_shape.clear();
  tile_view.valid_shape.reserve(tt->shape_.size());
  for (size_t i = 0; i < tt->shape_.size(); ++i) {
    tile_view.valid_shape.push_back(MakeIndexConst(0, span));
  }

  return std::make_shared<TileType>(tt->shape_, tt->dtype_, tt->memref_, std::move(tile_view),
                                    tt->memory_space_);
}

using ExprReplacementMap = std::unordered_map<const Var*, ExprPtr>;

ExprPtr SubstituteExprIfNeeded(const ExprPtr& expr, const ExprReplacementMap& replacements) {
  if (replacements.empty() || !expr) return expr;
  return transform_utils::Substitute(expr, replacements);
}

StmtPtr SubstituteStmtIfNeeded(const StmtPtr& stmt, const ExprReplacementMap& replacements) {
  if (replacements.empty() || !stmt) return stmt;
  return transform_utils::Substitute(stmt, replacements);
}

CallPtr RebuildLane1CallWithZeroValidShape(const CallPtr& call, const ExprReplacementMap& replacements) {
  std::vector<ExprPtr> new_args;
  new_args.reserve(call->args_.size());
  bool args_changed = false;
  for (const auto& arg : call->args_) {
    auto new_arg = SubstituteExprIfNeeded(arg, replacements);
    args_changed = args_changed || (new_arg != arg);
    new_args.push_back(new_arg);
  }

  if (IsOp(call, "tile.load") && new_args.size() >= 3) {
    auto new_type = WithZeroValidShape(call->GetType(), call->span_);
    auto tile_type = std::dynamic_pointer_cast<const TileType>(new_type);
    if (!tile_type) return call;

    std::vector<std::pair<std::string, std::any>> kwargs;
    kwargs.emplace_back("dtype", tile_type->dtype_);
    if (const auto& memory_space = tile_type->memory_space_) {
      kwargs.emplace_back("target_memory", *memory_space);
    }

    auto create_op = OpRegistry::GetInstance().GetOp("tile.create");
    return std::make_shared<Call>(create_op, std::vector<ExprPtr>{new_args[2]}, std::move(kwargs), new_type,
                                  call->span_);
  } else if (IsOp(call, "tile.slice") && new_args.size() >= 3) {
    // tile.slice is a pure view with no cross-core sync side effect, so in the
    // replay lane we only need an empty tile of the slice's result shape for the
    // downstream consumer to run as a no-op. Materialize it as tile.create (like
    // the tile.load case above) rather than a zero-valid subview: forcing the
    // slice's explicit valid_shape to a ConstInt 0 yields a *static* v_row=0,
    // v_col=0 subview. PTOAS accepts that, but pto-isa's Tile::GetValidRow /
    // GetValidCol have overloads only for a static mask > 0 or DYNAMIC — never
    // 0 — so ccec cannot compile the static-zero subview (gh#1649). A
    // tile.create renders with dynamic valid (runtime 0) and compiles cleanly,
    // matching how the sibling non-subview replay tiles behave.
    auto new_type = WithZeroValidShape(call->GetType(), call->span_);
    auto tile_type = std::dynamic_pointer_cast<const TileType>(new_type);
    INTERNAL_CHECK_SPAN(tile_type != nullptr, call->span_)
        << "Internal error: tile.slice must produce a TileType, but got "
        << (new_type ? new_type->TypeName() : "null");

    std::vector<std::pair<std::string, std::any>> kwargs;
    kwargs.emplace_back("dtype", tile_type->dtype_);
    if (const auto& memory_space = tile_type->memory_space_) {
      kwargs.emplace_back("target_memory", *memory_space);
    }

    auto create_op = OpRegistry::GetInstance().GetOp("tile.create");
    return std::make_shared<Call>(create_op, std::vector<ExprPtr>{new_args[1]}, std::move(kwargs), new_type,
                                  call->span_);
  } else if (IsOp(call, "tile.transpose")) {
    // tile.transpose lowers to a pto-isa op that hangs the AICore (507018) when
    // every operand is a zero-valid replay tile — the same static/zero-valid
    // hazard gh#1649 hit for subview slices. The replay result is discarded, so
    // emit an empty tile of the result shape instead of running the transpose.
    auto new_type = WithZeroValidShape(call->GetType(), call->span_);
    auto tile_type = std::dynamic_pointer_cast<const TileType>(new_type);
    // Fail loud (like the tile.slice branch) rather than silently fall through to
    // the generic rebuild below, which would replay the hang-prone transpose.
    INTERNAL_CHECK_SPAN(tile_type != nullptr, call->span_)
        << "Internal error: tile.transpose must produce a TileType, but got "
        << (new_type ? new_type->TypeName() : "null");
    std::vector<ExprPtr> shape_elems(tile_type->shape_.begin(), tile_type->shape_.end());
    auto shape_tuple = std::make_shared<MakeTuple>(std::move(shape_elems), call->span_);
    std::vector<std::pair<std::string, std::any>> kwargs;
    kwargs.emplace_back("dtype", tile_type->dtype_);
    if (const auto& memory_space = tile_type->memory_space_) {
      kwargs.emplace_back("target_memory", *memory_space);
    }
    auto create_op = OpRegistry::GetInstance().GetOp("tile.create");
    return std::make_shared<Call>(create_op, std::vector<ExprPtr>{shape_tuple}, std::move(kwargs), new_type,
                                  call->span_);
  } else if (IsOp(call, "tile.set_validshape") && new_args.size() == 3) {
    new_args[1] = MakeIndexConst(0, call->span_);
    new_args[2] = MakeIndexConst(0, call->span_);
    args_changed = true;
  }

  auto new_type = WithZeroValidShape(call->GetType(), call->span_);
  bool type_changed = new_type != call->GetType();
  if (!args_changed && !type_changed) return call;
  return std::make_shared<Call>(call->op_, std::move(new_args), call->kwargs_, new_type, call->span_);
}

bool IsNoSplitSharedPipeSetupCall(const CallPtr& call) {
  if (!call || !call->op_) return false;
  return IsOp(call, "system.reserve_buffer") || IsOp(call, "system.import_peer_buffer") ||
         IsOp(call, "system.aiv_initialize_pipe") || IsOp(call, "system.aic_initialize_pipe");
}

bool IsNoSplitSharedPipeSetupStmt(const StmtPtr& stmt) {
  CallPtr call;
  if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
    call = std::dynamic_pointer_cast<const Call>(assign->value_);
  } else if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
    call = std::dynamic_pointer_cast<const Call>(eval->expr_);
  }
  return IsNoSplitSharedPipeSetupCall(call);
}

struct NoSplitSharedPrefix {
  std::vector<StmtPtr> shared_setup_stmts;
  std::vector<StmtPtr> branch_stmts;
};

NoSplitSharedPrefix SplitNoSplitSharedPipeSetupPrefix(const std::vector<StmtPtr>& stmts) {
  NoSplitSharedPrefix result;
  size_t prefix_len = 0;
  while (prefix_len < stmts.size() && IsNoSplitSharedPipeSetupStmt(stmts[prefix_len])) {
    result.shared_setup_stmts.push_back(stmts[prefix_len]);
    ++prefix_len;
  }
  result.branch_stmts.insert(result.branch_stmts.end(),
                             stmts.begin() + static_cast<std::ptrdiff_t>(prefix_len), stmts.end());
  return result;
}

TypePtr Lane1LoopVarType(const TypePtr& original_type, const ExprPtr& init_value) {
  if (!std::dynamic_pointer_cast<const TileType>(original_type) || !init_value) {
    return original_type;
  }
  auto init_type = init_value->GetType();
  if (std::dynamic_pointer_cast<const TileType>(init_type)) {
    return init_type;
  }
  return original_type;
}

std::vector<IterArgPtr> RebuildLane1LoopIterArgs(const std::vector<IterArgPtr>& iter_args,
                                                 const ExprReplacementMap& entry_replacements,
                                                 ExprReplacementMap& scoped_replacements) {
  std::vector<IterArgPtr> new_iter_args;
  new_iter_args.reserve(iter_args.size());

  for (const auto& iter_arg : iter_args) {
    auto new_init = SubstituteExprIfNeeded(iter_arg->initValue_, entry_replacements);
    auto new_type = Lane1LoopVarType(iter_arg->GetType(), new_init);
    if (new_init != iter_arg->initValue_ || new_type != iter_arg->GetType()) {
      auto new_iter_arg =
          std::make_shared<IterArg>(iter_arg->name_hint_, new_type, new_init, iter_arg->span_);
      scoped_replacements[iter_arg.get()] = new_iter_arg;
      new_iter_args.push_back(new_iter_arg);
    } else {
      new_iter_args.push_back(iter_arg);
    }
  }

  return new_iter_args;
}

std::vector<VarPtr> RebuildLane1LoopReturnVars(const std::vector<VarPtr>& return_vars,
                                               const std::vector<IterArgPtr>& iter_args,
                                               ExprReplacementMap& replacements) {
  INTERNAL_CHECK(return_vars.size() == iter_args.size())
      << "Internal error: return_vars and iter_args sizes must match";

  std::vector<VarPtr> new_return_vars;
  new_return_vars.reserve(return_vars.size());

  for (size_t i = 0; i < return_vars.size(); ++i) {
    const auto& return_var = return_vars[i];
    auto new_type = return_var->GetType();
    if (i < iter_args.size()) {
      new_type = Lane1LoopVarType(return_var->GetType(), iter_args[i]);
    }
    if (new_type != return_var->GetType()) {
      auto new_return_var = std::make_shared<Var>(return_var->name_hint_, new_type, return_var->span_);
      replacements[return_var.get()] = new_return_var;
      new_return_vars.push_back(new_return_var);
    } else {
      new_return_vars.push_back(return_var);
    }
  }

  return new_return_vars;
}

// Lane 1 still needs to replay cross-core handshakes so Ascend910B GM-backed
// pipes stay balanced. Tile-producing replay work is forced to static
// valid_shape=0, letting PTO ops run as empty tiles while preserving the tpush /
// tpop / tfree synchronization protocol. The only visible side effects we
// intentionally suppress are tile.store writes and their SSA results.
std::vector<StmtPtr> BuildNoSplitLane1ReplayStmts(const std::vector<StmtPtr>& stmts,
                                                  ExprReplacementMap& replacements) {
  std::vector<StmtPtr> result;
  result.reserve(stmts.size());

  for (const auto& stmt : stmts) {
    if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
      auto call = std::dynamic_pointer_cast<const Call>(assign->value_);
      if (!call || !call->op_) {
        result.push_back(SubstituteStmtIfNeeded(stmt, replacements));
        continue;
      }

      if (IsOp(call, "tile.store") && call->args_.size() >= 3) {
        auto passthrough = SubstituteExprIfNeeded(call->args_[2], replacements);
        replacements[assign->var_.get()] = passthrough;
        result.push_back(std::make_shared<AssignStmt>(assign->var_, passthrough, assign->span_));
        continue;
      }

      auto new_value = RebuildLane1CallWithZeroValidShape(call, replacements);
      if (std::dynamic_pointer_cast<const TileType>(new_value->GetType())) {
        auto new_var =
            std::make_shared<Var>(assign->var_->name_hint_, new_value->GetType(), assign->var_->span_);
        replacements[assign->var_.get()] = new_var;
        result.push_back(std::make_shared<AssignStmt>(new_var, new_value, assign->span_));
      } else {
        result.push_back(std::make_shared<AssignStmt>(assign->var_, new_value, assign->span_));
      }
      continue;
    }

    if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
      auto call = std::dynamic_pointer_cast<const Call>(eval->expr_);
      if (!call || !call->op_) {
        result.push_back(SubstituteStmtIfNeeded(stmt, replacements));
        continue;
      }

      if (IsOp(call, "tile.store")) {
        continue;
      }

      auto new_expr = RebuildLane1CallWithZeroValidShape(call, replacements);
      result.push_back(std::make_shared<EvalStmt>(new_expr, eval->span_));
      continue;
    }

    if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      auto loop_replacements = replacements;
      auto new_iter_args = RebuildLane1LoopIterArgs(for_stmt->iter_args_, replacements, loop_replacements);
      auto new_return_vars = RebuildLane1LoopReturnVars(for_stmt->return_vars_, new_iter_args, replacements);
      auto new_body_stmts =
          BuildNoSplitLane1ReplayStmts(transform_utils::FlattenToStmts(for_stmt->body_), loop_replacements);
      auto new_for = MutableCopy(for_stmt);
      new_for->start_ = SubstituteExprIfNeeded(for_stmt->start_, replacements);
      new_for->stop_ = SubstituteExprIfNeeded(for_stmt->stop_, replacements);
      new_for->step_ = SubstituteExprIfNeeded(for_stmt->step_, replacements);
      new_for->iter_args_ = std::move(new_iter_args);
      new_for->body_ = loop_repair::MakeBody(new_body_stmts, for_stmt->span_);
      new_for->return_vars_ = std::move(new_return_vars);
      result.push_back(new_for);
      continue;
    }

    if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
      auto loop_replacements = replacements;
      auto new_iter_args = RebuildLane1LoopIterArgs(while_stmt->iter_args_, replacements, loop_replacements);
      auto new_return_vars =
          RebuildLane1LoopReturnVars(while_stmt->return_vars_, new_iter_args, replacements);
      auto new_body_stmts =
          BuildNoSplitLane1ReplayStmts(transform_utils::FlattenToStmts(while_stmt->body_), loop_replacements);
      auto new_while = MutableCopy(while_stmt);
      new_while->condition_ = SubstituteExprIfNeeded(while_stmt->condition_, loop_replacements);
      new_while->iter_args_ = std::move(new_iter_args);
      new_while->body_ = loop_repair::MakeBody(new_body_stmts, while_stmt->span_);
      new_while->return_vars_ = std::move(new_return_vars);
      result.push_back(new_while);
      continue;
    }

    if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      auto then_replacements = replacements;
      auto new_then_stmts = BuildNoSplitLane1ReplayStmts(transform_utils::FlattenToStmts(if_stmt->then_body_),
                                                         then_replacements);
      std::optional<StmtPtr> new_else;
      if (const auto& else_body = if_stmt->else_body_) {
        auto else_replacements = replacements;
        auto new_else_stmts =
            BuildNoSplitLane1ReplayStmts(transform_utils::FlattenToStmts(*else_body), else_replacements);
        new_else = loop_repair::MakeBody(new_else_stmts, if_stmt->span_);
      }
      auto new_if = MutableCopy(if_stmt);
      new_if->condition_ = SubstituteExprIfNeeded(if_stmt->condition_, replacements);
      new_if->then_body_ = loop_repair::MakeBody(new_then_stmts, if_stmt->span_);
      new_if->else_body_ = new_else;
      result.push_back(new_if);
      continue;
    }

    result.push_back(SubstituteStmtIfNeeded(stmt, replacements));
  }

  return result;
}

// Halve a standalone (non-split_aiv) AIV/AIC function in place along the split
// axis: inject get_subblock_idx (AIV only), route every leaf statement through
// the shared split_axis halving driver, re-localize offsets via the final
// Substitute, and stamp split (+ dual_aiv_dispatch for AIV). Mirrors the
// auto-split LowerFunction path but consumes a body that already carries
// explicit tpush/tpop boundaries (no tile.move -> aiv_shard rewrite needed).
FunctionPtr ProcessStandaloneSplitFunction(const FunctionPtr& func, SplitMode mode) {
  if (mode == SplitMode::None) return func;
  int split_int = static_cast<int>(mode);
  int split_dim = SplitDimension(mode);
  bool is_aiv = (func->func_type_ == FunctionType::AIV);

  std::unordered_map<const Var*, TileInfo> tile_vars;
  std::unordered_map<const Var*, VarPtr> var_replacements;
  std::vector<VarPtr> new_params;
  new_params.reserve(func->params_.size());
  for (const auto& param : func->params_) {
    auto new_param = std::make_shared<Var>(param->name_hint_, param->GetType(), param->span_);
    new_params.push_back(new_param);
    var_replacements[param.get()] = new_param;
  }

  auto injected = InjectSubblockIdx(func, is_aiv);

  auto new_stmts = ProcessStmts(injected.body_stmts, mode, split_int, split_dim, tile_vars, is_aiv,
                                injected.subblock_idx_expr, var_replacements);
  StmtPtr new_body =
      (new_stmts.size() == 1) ? new_stmts[0] : std::make_shared<SeqStmts>(new_stmts, func->span_);
  if (!var_replacements.empty()) {
    new_body = transform_utils::Substitute(new_body, var_replacements);
  }
  auto [cloned_body, clone_map_unused] = DeepClone(new_body);
  (void)clone_map_unused;

  auto new_func = MutableCopy(func);
  new_func->params_ = new_params;
  new_func->body_ = cloned_body;
  new_func->attrs_ = WithSplitAttrs(func, mode, is_aiv);
  return new_func;
}

FunctionPtr ProcessNoSplitDualAivFunction(const FunctionPtr& func) {
  // Plain INTERNAL_CHECK: RequiresNoSplitDualAivSync short-circuits on null
  // func, so func->span_ would dereference null in the failure path.
  INTERNAL_CHECK(RequiresNoSplitDualAivSync(func))
      << "Internal error: ProcessNoSplitDualAivFunction requires dual-dispatch AIV marker";

  std::unordered_map<const Var*, ExprPtr> param_replacements;
  std::vector<VarPtr> new_params;
  new_params.reserve(func->params_.size());
  for (const auto& param : func->params_) {
    auto new_param = std::make_shared<Var>(param->name_hint_, param->GetType(), param->span_);
    new_params.push_back(new_param);
    param_replacements[param.get()] = new_param;
  }

  auto injected = InjectSubblockIdx(func, /*is_aiv=*/true);
  INTERNAL_CHECK(!injected.body_stmts.empty())
      << "Internal error: dual-dispatch no-split AIV body must contain injected subblock_idx";
  std::vector<StmtPtr> guarded_stmts(injected.body_stmts.begin() + 1, injected.body_stmts.end());
  auto hoisted_prefix = SplitNoSplitSharedPipeSetupPrefix(guarded_stmts);
  auto lane0_body = loop_repair::MakeBody(hoisted_prefix.branch_stmts, func->span_);

  ExprReplacementMap lane1_replacements = param_replacements;
  auto lane1_stmts = BuildNoSplitLane1ReplayStmts(hoisted_prefix.branch_stmts, lane1_replacements);
  auto lane1_body = loop_repair::MakeBody(lane1_stmts, func->span_);
  auto [lane1_cloned_body, lane1_clone_map_unused] = DeepClone(lane1_body);
  (void)lane1_clone_map_unused;
  lane1_body = lane1_cloned_body;

  auto zero = std::make_shared<ConstInt>(0, DataType::INDEX, func->span_);
  auto lane0_cond = MakeEq(injected.subblock_idx_expr, zero, func->span_);
  auto branch_stmt = std::make_shared<IfStmt>(lane0_cond, lane0_body, std::make_optional(lane1_body),
                                              std::vector<VarPtr>{}, func->span_);

  std::vector<StmtPtr> new_body_stmts{injected.body_stmts.front()};
  new_body_stmts.insert(new_body_stmts.end(), hoisted_prefix.shared_setup_stmts.begin(),
                        hoisted_prefix.shared_setup_stmts.end());
  new_body_stmts.push_back(branch_stmt);
  StmtPtr new_body = loop_repair::MakeBody(new_body_stmts, func->span_);
  if (!param_replacements.empty()) {
    new_body = transform_utils::Substitute(new_body, param_replacements);
  }
  auto [cloned_body, clone_map_unused] = DeepClone(new_body);
  (void)clone_map_unused;

  auto new_func = MutableCopy(func);
  new_func->params_ = new_params;
  new_func->body_ = cloned_body;
  return new_func;
}

}  // namespace

namespace pass {

Pass SplitVectorKernel() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    std::vector<FunctionPtr> new_functions;
    bool changed = false;

    for (const auto& [gvar, func] : program->functions_) {
      // External kernels are signature-only declarations. Their hand-written
      // source owns sub-lane partitioning, so preserve launch attrs but never
      // synthesize a DSL body (which would violate the external-source contract).
      if (func->HasAttr(kExternalSourceAttr)) {
        if (func->func_type_ == FunctionType::AIV) {
          auto explicit_mode = func->GetSplitMode();
          bool split_aiv = func->HasAttr("split_aiv") && func->GetAttr<bool>("split_aiv", false);
          if (explicit_mode.has_value() && explicit_mode.value() != SplitMode::None) {
            auto external_func = MutableCopy(func);
            external_func->attrs_ = WithSplitAttrs(func, explicit_mode.value(), /*is_aiv=*/true);
            new_functions.push_back(external_func);
            changed = true;
            continue;
          } else if (split_aiv && !func->GetAttr<bool>(kDualAivDispatchAttr, false)) {
            auto external_func = MutableCopy(func);
            auto attrs = external_func->attrs_;
            attrs.erase(std::remove_if(attrs.begin(), attrs.end(),
                                       [](const auto& kv) { return kv.first == kDualAivDispatchAttr; }),
                        attrs.end());
            attrs.emplace_back(kDualAivDispatchAttr, true);
            external_func->attrs_ = std::move(attrs);
            new_functions.push_back(external_func);
            changed = true;
            continue;
          }
        }
        new_functions.push_back(func);
        continue;
      }

      // split_aiv kernels arrive here already in the explicit form: either
      // hand-written, or produced by LowerAutoVectorSplit from an AUTO pl.split
      // mixed InCore function. Their tile.aiv_shard / tile.aic_gather have been
      // folded into split-stamped tpush/tpop pairs (via ExpandMixedKernel's
      // boundary machinery) and they carry already-halved compute tiles. This is
      // the SOLE split path through SplitVectorKernel: just stamp split +
      // dual_aiv_dispatch and pass the body through unchanged. The former per-op
      // halving driver was deleted — after LowerAutoVectorSplit runs, every split
      // function reaches here split_aiv-marked, so re-halving here would
      // double-halve the (already-half) body.
      if ((func->func_type_ == FunctionType::AIV || func->func_type_ == FunctionType::AIC) &&
          func->HasAttr("split_aiv") && func->GetAttr<bool>("split_aiv", false)) {
        auto explicit_mode = func->GetSplitMode();
        auto new_func = MutableCopy(func);
        if (explicit_mode.has_value() && explicit_mode.value() != SplitMode::None) {
          // Single-mode split_aiv: a function-level "split" attr survives (a
          // hand-written kernel, an AUTO function converged by LowerAutoVectorSplit,
          // or a single-mode explicit region). Stamp split + dual_aiv_dispatch.
          new_func->attrs_ =
              WithSplitAttrs(func, explicit_mode.value(), func->func_type_ == FunctionType::AIV);
        } else {
          // Multi-mode explicit split_aiv: the per-region modes were lowered and
          // erased by LowerAutoVectorSplit (pass 21); no single function-level mode
          // survives. The authoritative per-op "split" ints already sit on the
          // tpop/tpush pairs, so only the mode-agnostic dual_aiv_dispatch bool needs
          // stamping here (all RequiresDualAivDispatch consults).
          auto attrs = func->attrs_;
          attrs.erase(std::remove_if(attrs.begin(), attrs.end(),
                                     [](const auto& kv) { return kv.first == kDualAivDispatchAttr; }),
                      attrs.end());
          if (func->func_type_ == FunctionType::AIV) {
            attrs.emplace_back(kDualAivDispatchAttr, true);
          }
          new_func->attrs_ = std::move(attrs);
        }
        new_functions.push_back(new_func);
        changed = true;
        continue;
      }

      // Standalone (non-split_aiv) split kernels: hand-written separate AIV/AIC
      // functions whose cross-core split is signalled by an explicit function
      // "split" attr or by a split=N kwarg on their pipe ops. These never pass
      // through LowerAutoVectorSplit (InCore-only) and arrive un-halved, so they
      // need the per-lane halving here. Guarded by NOT split_aiv (above), so a
      // pre-halved auto-split function can never reach this branch.
      if (func->func_type_ == FunctionType::AIV || func->func_type_ == FunctionType::AIC) {
        auto standalone_mode = ResolveSplitMode(func);
        if (standalone_mode.has_value() && standalone_mode.value() != SplitMode::None) {
          new_functions.push_back(ProcessStandaloneSplitFunction(func, standalone_mode.value()));
          changed = true;
          continue;
        }
      }

      // No-split dual-AIV dispatch: the orthogonal Ascend910B path that replays
      // cross-core handshakes on lane 1 of a non-split AIV function. Untouched by
      // the convergence refactor.
      if (RequiresNoSplitDualAivSync(func)) {
        new_functions.push_back(ProcessNoSplitDualAivFunction(func));
        changed = true;
      } else {
        new_functions.push_back(func);
      }
    }

    if (!changed) return program;
    return std::make_shared<Program>(new_functions, program->name_, program->span_);
  };

  return CreateProgramPass(pass_func, "SplitVectorKernel", kSplitVectorKernelProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
