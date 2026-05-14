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

#include "pypto/codegen/orchestration/orchestration_codegen.h"

#include <algorithm>
#include <any>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/codegen/codegen_base.h"
#include "pypto/codegen/orchestration/orchestration_analysis.h"
#include "pypto/codegen/orchestration_op_registry.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/pipe.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/transforms/utils/core_affinity.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/transforms/utils/wrapper_call_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace codegen {

using namespace pypto::ir;  // NOLINT(build/namespaces)

CoreType InferFunctionCoreType(const FunctionPtr& func) {
  // After ExpandMixedKernel runs (part of every Default / DebugTileOptimization
  // pipeline), every InCore function reaching codegen has been split into AIC,
  // AIV, or Group / Spmd wrappers. The two callers of this function
  // (GenerateFunctionCallCode and GenerateSpmdCallCode) both filter Spmd /
  // Group out before invoking it. Tests that bypass the pipeline must declare
  // their kernels with the appropriate AIC / AIV type explicitly so codegen
  // sees the concrete core type without re-deriving from body memory spaces.
  switch (func->func_type_) {
    case FunctionType::AIC:
      return CoreType::CUBE;
    case FunctionType::AIV:
      return CoreType::VECTOR;
    default:
      INTERNAL_UNREACHABLE_SPAN(func->span_)
          << "InferFunctionCoreType expects AIC or AIV (Spmd/Group are filtered upstream); got "
          << FunctionTypeToString(func->func_type_) << " on function '" << func->name_
          << "'. Either run ExpandMixedKernel before codegen or declare the function "
          << "with @pl.function(type=pl.FunctionType.AIC|AIV) directly.";
  }
}

namespace {

constexpr const char* kDualAivDispatchAttr = "dual_aiv_dispatch";

// The runtime primitive ``Arg::set_dependencies(ptr, count)`` has no upper
// bound on the explicit dep count, and codegen emits ``ArgWithDeps<N>`` with
// N sized to each call's exact dep count, so there is no codegen-time cap on
// per-task explicit dependencies.

int GetGMPipeSlotCount(int dir_mask) {
  const int bidirectional = core_affinity::kDirMaskC2V | core_affinity::kDirMaskV2C;
  if (dir_mask == bidirectional) {
    return 4;
  }
  if (dir_mask == core_affinity::kDirMaskC2V || dir_mask == core_affinity::kDirMaskV2C) {
    return 8;
  }
  return 0;
}

int64_t ComputeGMPipeWorkspaceElements(const ProgramPtr& program, const FunctionPtr& root_func) {
  std::map<std::pair<int, int>, int> slot_size_by_pipe;

  std::unordered_set<std::string> visited_funcs;
  std::function<void(const std::vector<StmtPtr>&)> scan_stmts;
  std::function<void(const FunctionPtr&)> scan_func;
  scan_stmts = [&](const std::vector<StmtPtr>& stmts) {
    for (const auto& stmt : stmts) {
      auto call = transform_utils::GetCallFromStmt(stmt);
      if (op_predicates::IsInitializePipe(call)) {
        const int pipe_id = call->GetKwarg<int>("id", 0);
        const int dir_mask = call->GetKwarg<int>("dir_mask", 0);
        const int slot_size = call->GetKwarg<int>("slot_size", 0);
        if (dir_mask > 0 && slot_size > 0) {
          const auto key = std::make_pair(pipe_id, dir_mask);
          auto [it, inserted] = slot_size_by_pipe.emplace(key, slot_size);
          CHECK(inserted || it->second == slot_size)
              << "initialize_pipe for frontend pipe id " << pipe_id << " and dir_mask " << dir_mask
              << " uses inconsistent slot_size values: " << it->second << " and " << slot_size;
        }
      } else if (call) {
        auto gv = As<GlobalVar>(call->op_);
        if (gv) {
          scan_func(program->GetFunction(gv->name_));
        }
      }

      if (auto for_stmt = As<ForStmt>(stmt)) {
        scan_stmts(transform_utils::FlattenToStmts(for_stmt->body_));
      } else if (auto if_stmt = As<IfStmt>(stmt)) {
        scan_stmts(transform_utils::FlattenToStmts(if_stmt->then_body_));
        const auto& else_body = if_stmt->else_body_;
        if (else_body) {
          scan_stmts(transform_utils::FlattenToStmts(*else_body));
        }
      } else if (auto while_stmt = As<WhileStmt>(stmt)) {
        scan_stmts(transform_utils::FlattenToStmts(while_stmt->body_));
      }
    }
  };

  scan_func = [&](const FunctionPtr& func) {
    if (!func || !visited_funcs.insert(func->name_).second) {
      return;
    }
    if (func->body_) {
      scan_stmts(transform_utils::FlattenToStmts(func->body_));
    }
  };

  scan_func(root_func);

  int64_t total_bytes = 0;
  for (const auto& [key, slot_size] : slot_size_by_pipe) {
    const int dir_mask = key.second;
    const int slot_count = GetGMPipeSlotCount(dir_mask);
    CHECK(slot_count > 0) << "initialize_pipe has invalid dir_mask for GM slot buffer: " << dir_mask;
    CHECK(total_bytes <= std::numeric_limits<int64_t>::max() -
                             static_cast<int64_t>(slot_count) * static_cast<int64_t>(slot_size))
        << "GM slot buffer size overflow while sizing frontend pipe id " << key.first;
    total_bytes += static_cast<int64_t>(slot_count) * static_cast<int64_t>(slot_size);
  }

  if (total_bytes == 0) {
    return 0;
  }
  return (total_bytes + static_cast<int64_t>(sizeof(float)) - 1) / static_cast<int64_t>(sizeof(float));
}

// ---------------------------------------------------------------------------
// Template / boilerplate generation helpers
// ---------------------------------------------------------------------------

std::string GenerateIncludes(bool include_optional) {
  std::ostringstream oss;
  oss << "#include <stddef.h>\n";
  oss << "#include <stdint.h>\n";
  oss << "#include <stdio.h>\n";
  if (include_optional) {
    oss << "#include <optional>\n";
  }
  oss << "\n";
  oss << "#include \"pto_orchestration_api.h\"\n\n";
  return oss.str();
}

std::string GenerateScalarUnpack(const std::string& var_name, int scalar_index,
                                 const ScalarTypePtr& scalar_type) {
  std::ostringstream oss;
  std::string cpp_type = scalar_type->dtype_.ToCTypeString();
  oss << "    " << cpp_type << " " << var_name << " = from_u64<" << cpp_type << ">(orch_args.scalar("
      << scalar_index << "));\n";
  return oss.str();
}

std::string GenerateConfigFunction(int expected_arg_count) {
  std::ostringstream oss;
  oss << "__attribute__((visibility(\"default\")))\n";
  oss << "PTO2OrchestrationConfig aicpu_orchestration_config(const ChipStorageTaskArgs& orch_args) {\n";
  oss << "    (void)orch_args;\n";
  oss << "    return PTO2OrchestrationConfig{\n";
  oss << "        .expected_arg_count = " << expected_arg_count << ",\n";
  oss << "    };\n";
  oss << "}\n\n";
  return oss.str();
}

// AIV functions whose body has been split across two vector cores carry the
// `dual_aiv_dispatch` attribute. SplitVectorKernel is the single source of
// truth: any non-None SplitMode on an AIV function ends up reflected in this
// attribute on pass exit, so codegen reads the attribute directly without
// re-deriving from SplitMode.
bool RequiresDualAivDispatch(const FunctionPtr& aiv_func) {
  if (aiv_func == nullptr) return false;
  return aiv_func->HasAttr(kDualAivDispatchAttr) && aiv_func->GetAttr<bool>(kDualAivDispatchAttr, false);
}

// Returns the opening of a rt_submit_{aic,aiv}_task call.
// Caller appends: func_id << ", " << params << ");".
std::string CoreTypeToSubmitPrefix(CoreType core_type) {
  std::string func = core_type == CoreType::CUBE ? "rt_submit_aic_task" : "rt_submit_aiv_task";
  return func + "(";
}

std::string GenerateMakeTensorExternal(const std::string& var_name, int orch_index,
                                       const TensorTypePtr& tensor_type, const CodegenBase& codegen) {
  std::ostringstream oss;
  oss << "    Tensor ext_" << var_name << " = from_tensor_arg(orch_args.tensor(" << orch_index << "));\n";
  return oss.str();
}

class CodegenEffectiveUseCollector : public var_collectors::VarDefUseCollector {
 protected:
  void VisitStmt_(const ReturnStmtPtr&) override {}
};

}  // namespace

// Statement code generator for orchestration
class OrchestrationStmtCodegen : public CodegenBase {
 public:
  explicit OrchestrationStmtCodegen(const ProgramPtr& prog, std::map<std::string, int>* func_ids,
                                    std::map<std::string, CoreType>* core_types, int* next_id,
                                    std::unordered_map<const Var*, std::string> param_to_emit_name,
                                    std::set<std::string> param_name_set,
                                    std::map<std::string, int> param_name_to_orch_index)
      : program_(prog),
        func_name_to_id_(func_ids),
        func_name_to_core_type_(core_types),
        next_func_id_(next_id),
        emit_name_map_(std::move(param_to_emit_name)),
        param_name_set_(std::move(param_name_set)),
        param_name_to_orch_index_(std::move(param_name_to_orch_index)) {
    declared_var_names_ = param_name_set_;
  }

  void SetCallTupleElements(const std::map<std::string, std::vector<TupleElement>>& elements) {
    tuple_var_to_elements_ = elements;
    for (auto& [key, vec] : tuple_var_to_elements_) {
      std::sort(vec.begin(), vec.end(),
                [](const TupleElement& a, const TupleElement& b) { return a.index < b.index; });
    }
  }

  void SetCallToTupleKey(const std::map<const Call*, std::string>& mapping) { call_to_tuple_key_ = mapping; }

  void SetInitialIndent(int indent) { indent_ = indent; }

  void SetEffectiveUses(std::unordered_set<const Var*> uses) { effective_uses_ = std::move(uses); }

  std::string GetGeneratedCode() const { return code_.str(); }
  // --- CodegenBase pure virtual implementations ---
  [[nodiscard]] std::string GetCurrentResultTarget() const override { return current_result_var_; }
  void Emit(const std::string& line) override { code_ << line; }
  std::string GetExprAsCode(const ExprPtr& expr) override { return GenerateExprString(expr); }
  [[nodiscard]] std::string GetTypeString(const DataType& dtype) const override {
    return dtype.ToCTypeString();
  }
  int64_t GetConstIntValue(const ExprPtr& expr) const override {
    auto ci = As<ConstInt>(expr);
    INTERNAL_CHECK_SPAN(ci, expr->span_) << "Internal error: expected ConstInt expression";
    return ci->value_;
  }
  std::string GetVarName(const VarPtr& var) const override {
    auto it = emit_name_map_.find(var.get());
    if (it != emit_name_map_.end()) {
      return it->second;
    }
    return GetSSABaseName(var->name_hint_);
  }
  [[nodiscard]] std::string TryGetVarName(const ir::ExprPtr& expr) const override {
    if (auto var = AsVarLike(expr)) {
      return GetVarName(var);
    }
    return CodegenBase::TryGetVarName(expr);
  }
  [[nodiscard]] std::string GetTensorDataPtr(const std::string& name) const override {
    auto it = param_name_to_orch_index_.find(name);
    if (it != param_name_to_orch_index_.end()) {
      return "orch_args.tensor(" + std::to_string(it->second) + ").data_as<void>()";
    }
    return name + ".data";
  }

  [[nodiscard]] std::string GetTensorShapeDim(const std::string& name, int64_t axis) const override {
    auto it = param_name_to_orch_index_.find(name);
    if (it != param_name_to_orch_index_.end()) {
      return "(int64_t)orch_args.tensor(" + std::to_string(it->second) + ").shapes[" + std::to_string(axis) +
             "]";
    }
    return "(int64_t)" + name + ".shapes[" + std::to_string(axis) + "]";
  }

  [[nodiscard]] std::string GetTensorCreateSizeExpr(const std::string& result_var,
                                                    const std::string& default_dim_expr) const override {
    auto size_it = tensor_create_size_expr_by_emit_name_.find(result_var);
    if (size_it != tensor_create_size_expr_by_emit_name_.end()) {
      return size_it->second;
    }
    return CodegenBase::GetTensorCreateSizeExpr(result_var, default_dim_expr);
  }

  void VisitStmt_(const ForStmtPtr& for_stmt) override {
    if (for_stmt->kind_ == ForKind::Unroll) {
      LOG_WARN << "ForKind::Unroll loop was not expanded before codegen; "
                  "generating sequential loop as fallback";
    } else if (for_stmt->kind_ == ForKind::Pipeline) {
      LOG_WARN << "ForKind::Pipeline loop reached codegen; CanonicalizeIOOrder "
                  "should have demoted it to Sequential. Generating sequential loop as fallback.";
    }

    std::string loop_var = GetVarName(for_stmt->loop_var_);
    // Guard against an empty emit name (e.g. python `for _ in pl.range(...)`
    // turns into an SSA name that GetSSABaseName strips to ""). Without this
    // we would emit `for (int64_t  = 0; ...)`, which compiles only by accident.
    if (loop_var.empty()) {
      loop_var = ReserveSyntheticEmitName("i");
      emit_name_map_[for_stmt->loop_var_.get()] = loop_var;
    }
    std::string start_expr = GenerateExprString(for_stmt->start_);
    std::string stop_expr = GenerateExprString(for_stmt->stop_);
    std::string step_expr = GenerateExprString(for_stmt->step_);

    INTERNAL_CHECK_SPAN(for_stmt->iter_args_.size() == for_stmt->return_vars_.size(), for_stmt->span_)
        << "Internal error: ForStmt iter_args/return_vars size mismatch";

    // Classify each iter_arg by its yield: trivial (yields back the iter_arg
    // itself, no body-level rebind) vs rebind (yields a different value, e.g.
    // a freshly-created tensor). The two need different lowerings:
    //
    //   trivial: alias iter_arg/return_var to the init's emit name. The runtime
    //     dependency tracker keys off Tensor* identity, and OUTPUT_EXISTING /
    //     INOUT params record the address of the Tensor lvalue passed in. If we
    //     materialised a fresh `Tensor` value for the carry, the kernel reads
    //     and writes would see a different `&tensor` than the producer that
    //     materialised the buffer, breaking dep chains. So for the trivial case
    //     we keep the legacy aliasing behaviour.
    //
    //   rebind: predeclare a mutable carry variable initialised from the init,
    //     and route YieldStmt to assign back to it. Without this, a python
    //     rebind like `current = next` inside the loop body would never
    //     propagate to the next iteration or to code following the loop. See
    //     issue #1286.
    //
    // We need to know which case applies before visiting the body, so we look
    // up the yield once here.
    auto yield = transform_utils::GetLastYieldStmt(for_stmt->body_);
    std::vector<bool> is_rebind(for_stmt->iter_args_.size(), false);
    if (yield) {
      INTERNAL_CHECK_SPAN(yield->value_.size() == for_stmt->iter_args_.size(), for_stmt->span_)
          << "Internal error: ForStmt yield/iter_args size mismatch";
      // Build the alias-equivalence set for each iter_arg. A Var is in
      // `iter_arg`'s class if it IS the iter_arg, or it was assigned the
      // result of `tensor.assemble(<member>, ...)` — assemble writes in place
      // to its first arg so the result Var is just another name for the same
      // backing buffer. The transitive closure is computed by repeatedly
      // walking AssignStmts in the body until no new members are added.
      // (This mirrors HandleTensorAssembleAssign at codegen-emit time, but
      // we need it pre-body so we can decide on the carry lowering.)
      std::vector<std::unordered_set<const Var*>> aliases(for_stmt->iter_args_.size());
      for (size_t i = 0; i < for_stmt->iter_args_.size(); ++i) {
        aliases[i].insert(for_stmt->iter_args_[i].get());
      }
      // Collect: (a) AssignStmts producing tensor.assemble aliases, and (b)
      // nested ForStmts (so we can map their iter_args -> their return_vars,
      // which are how a parent's carry "comes out of" an inner loop).
      class AliasingNodeCollector : public IRVisitor {
       public:
        std::vector<AssignStmtPtr> assigns_;
        std::vector<ForStmtPtr> nested_fors_;
        void VisitStmt_(const AssignStmtPtr& a) override {
          assigns_.push_back(a);
          IRVisitor::VisitStmt_(a);
        }
        void VisitStmt_(const ForStmtPtr& f) override {
          nested_fors_.push_back(f);
          IRVisitor::VisitStmt_(f);
        }
      };
      AliasingNodeCollector collector;
      collector.VisitStmt(for_stmt->body_);
      // Index assignments by produced var so rule (d) can climb tuple chains.
      std::unordered_map<const Var*, AssignStmtPtr> var_to_assign;
      for (const auto& a : collector.assigns_) {
        var_to_assign[a->var_.get()] = a;
      }

      bool changed = true;
      while (changed) {
        changed = false;
        for (const auto& assign : collector.assigns_) {
          // (d) TupleGetItemExpr: climb to the tuple-producing call and resolve
          // the corresponding output arg. Multi-output InCore kernels return
          // tuples; each `var = ret_tuple[i]` extract should alias the i-th
          // output-side arg of the call (using the codegen's own indexing).
          if (auto tge = As<TupleGetItemExpr>(assign->value_)) {
            auto tuple_var = AsVarLike(tge->tuple_);
            if (tuple_var) {
              auto it = var_to_assign.find(tuple_var.get());
              if (it != var_to_assign.end()) {
                auto tcall = As<Call>(it->second->value_);
                if (tcall) {
                  auto tdirs = tcall->GetArgDirections();
                  if (tdirs.size() == tcall->args_.size()) {
                    int64_t out_seen = 0;
                    int64_t target_idx = static_cast<int64_t>(tge->index_);
                    for (size_t a = 0; a < tdirs.size(); ++a) {
                      if (tdirs[a] != ArgDirection::OutputExisting && tdirs[a] != ArgDirection::InOut &&
                          tdirs[a] != ArgDirection::Output) {
                        continue;
                      }
                      if (out_seen == target_idx) {
                        auto out_arg = AsVarLike(tcall->args_[a]);
                        if (out_arg) {
                          for (auto& cls : aliases) {
                            if (cls.count(out_arg.get()) && !cls.count(assign->var_.get())) {
                              cls.insert(assign->var_.get());
                              changed = true;
                            }
                          }
                        }
                        break;
                      }
                      ++out_seen;
                    }
                  }
                }
              }
            }
            continue;
          }
          auto call = As<Call>(assign->value_);
          if (!call) continue;
          // (a) tensor.assemble: result var aliases its first arg (the target).
          if (call->op_->name_ == "tensor.assemble" && !call->args_.empty()) {
            auto first_arg = AsVarLike(call->args_[0]);
            if (first_arg) {
              for (auto& cls : aliases) {
                if (cls.count(first_arg.get()) && !cls.count(assign->var_.get())) {
                  cls.insert(assign->var_.get());
                  changed = true;
                }
              }
            }
          }
          // (c) Calls with output_existing/inout args (e.g. InCore kernels):
          // the result aliases the FIRST output-side arg, mirroring the codegen
          // alias `const Tensor& result = args[out_idx];` emitted later by
          // GenerateSingleReturnAlias / GenerateTupleReturnAliases. If that arg
          // is in an iter_arg's class, the result is in the class too.
          auto call_dirs = call->GetArgDirections();
          if (call_dirs.size() == call->args_.size()) {
            for (size_t a = 0; a < call_dirs.size(); ++a) {
              if (call_dirs[a] != ArgDirection::OutputExisting && call_dirs[a] != ArgDirection::InOut) {
                continue;
              }
              auto out_arg = AsVarLike(call->args_[a]);
              if (!out_arg) continue;
              for (auto& cls : aliases) {
                if (cls.count(out_arg.get()) && !cls.count(assign->var_.get())) {
                  cls.insert(assign->var_.get());
                  changed = true;
                }
              }
              break;  // only the first output-side arg is the alias target
            }
          }
        }
        // (b) Nested ForStmts: the parent's carry threaded through a nested
        // loop comes out via the nested loop's return_var. Specifically, for
        // each (nested iter_arg, nested return_var) pair, if nested iter_arg's
        // init value is in the parent class, then nested return_var is too.
        //
        // ArrayType iter_args are EXCLUDED from this propagation: unlike
        // TensorType (a pointer-to-buffer alias), an ArrayType iter_arg owns a
        // *fresh* C-stack array at each level. Treating the inner rv as an
        // alias of the outer iter_arg would mis-mark the outer slot as
        // ``is_rebind=false`` (silently dropping the outer's yield-back copy,
        // which is the very mechanism that propagates state across phases in
        // a SEQ x PARALLEL phase fence). The outer carry must be a distinct
        // backing array and the outer yield must emit an explicit array-array
        // copy back into it (see VisitStmt_(YieldStmtPtr)).
        for (const auto& nf : collector.nested_fors_) {
          for (size_t k = 0; k < nf->iter_args_.size(); ++k) {
            if (As<ArrayType>(nf->iter_args_[k]->GetType())) continue;
            auto init_var = AsVarLike(nf->iter_args_[k]->initValue_);
            if (!init_var) continue;
            const auto* rv = nf->return_vars_[k].get();
            for (auto& cls : aliases) {
              if (cls.count(init_var.get()) && !cls.count(rv)) {
                cls.insert(rv);
                changed = true;
              }
            }
          }
        }
      }
      for (size_t i = 0; i < for_stmt->iter_args_.size(); ++i) {
        auto yield_var = AsVarLike(yield->value_[i]);
        // True rebind iff yield value is not in the iter_arg's alias class
        // (i.e. not the iter_arg itself nor any tensor.assemble alias of it).
        is_rebind[i] = !yield_var || !aliases[i].count(yield_var.get());
        // TaskId iter_args are always rebind: the alias-closure logic above is
        // about Tensor buffer identity (OUTPUT_EXISTING aliases), which doesn't
        // apply to PTO2TaskId values — the runtime task_id at iter N+1 is
        // genuinely different from iter N even when the IR Var aliases.
        if (auto sty = As<ScalarType>(for_stmt->iter_args_[i]->GetType())) {
          if (sty->dtype_ == DataType::TASK_ID) is_rebind[i] = true;
        }
      }
    }

    // For TaskId iter_args inside a manual scope we always lower via array
    // carry: a Parallel ForStmt produces ``PTO2TaskId arr[N]`` with yields
    // writing per-slot, and downstream ``add_dep`` iterates every slot. A
    // Sequential ForStmt whose yield value is an inner Parallel rv is also
    // array-carry of the same size. Scalar (single-name) carry would only
    // fence the last-dispatched task — which is unsafe under MANUAL scope.

    // Per-iter-arg array-carry size (0 means non-TaskId scalar carry).
    std::vector<int64_t> array_sizes(for_stmt->iter_args_.size(), 0);
    if (in_manual_scope_depth_ > 0) {
      for (size_t i = 0; i < for_stmt->iter_args_.size(); ++i) {
        if (is_rebind[i]) {
          array_sizes[i] = ResolveArrayCarrySize(for_stmt, i);
        }
      }
    }

    // A Parallel ForStmt with a TaskId iter_arg REQUIRES a const trip count
    // so we can allocate a ``PTO2TaskId[N]`` backing store. Without it we
    // would silently fall back to last-dispatched fence semantics — wrong
    // under MANUAL scope. Surface this as a clear user-facing error instead
    // of emitting incorrect code.
    if (for_stmt->kind_ == ForKind::Parallel && in_manual_scope_depth_ > 0) {
      for (size_t i = 0; i < for_stmt->iter_args_.size(); ++i) {
        if (!is_rebind[i]) continue;
        auto sty = As<ScalarType>(for_stmt->iter_args_[i]->GetType());
        if (!sty || sty->dtype_ != DataType::TASK_ID) continue;
        CHECK(array_sizes[i] > 0) << "manual_scope: pl.parallel loops carrying a manual_scope dep "
                                  << "(via ``deps=[...]``) must have a statically-known trip count. "
                                  << "The runtime fence requires a PTO2TaskId[N] array of fixed N. "
                                  << "Either make the parallel loop's trip count a Python int "
                                  << "(e.g. ``pl.parallel(4)``) or restructure to put the parallel "
                                  << "loop inside a const-bounded scope.";
      }
    }

    for (size_t i = 0; i < for_stmt->iter_args_.size(); ++i) {
      const auto& iter_arg = for_stmt->iter_args_[i];
      const auto& return_var = for_stmt->return_vars_[i];
      std::string init_var_name = TryGetVarName(iter_arg->initValue_);
      INTERNAL_CHECK_SPAN(!init_var_name.empty(), for_stmt->span_)
          << "Internal error: ForStmt iter_arg initValue must be a variable, got non-variable expr";
      // Function tensor params get rewritten to `ext_<name>` in the emitted C++,
      // so the bare emit name is not a valid identifier when the init value is
      // a param. Apply the same translation as everything else that names a
      // tensor in the emitted code.
      init_var_name = GetExternalTensorName(init_var_name);

      if (array_sizes[i] > 0) {
        // ARRAY CARRY PATH — allocate ``PTO2TaskId <name>[N]`` and init it.
        // Initialisation rule: if the iter_arg's init value is itself an
        // array-carry Var, copy slot-by-slot; otherwise broadcast the scalar
        // init expression to every slot.
        const int64_t N = array_sizes[i];
        std::string rv_array_name = ReserveSyntheticEmitName(return_var->name_hint_);
        code_ << Indent() << "PTO2TaskId " << rv_array_name << "[" << N << "];\n";

        auto outer_init_var = AsVarLike(iter_arg->initValue_);
        const ArrayCarryEntry* outer_init_arr = nullptr;
        if (outer_init_var) {
          auto outer_it = array_carry_vars_.find(outer_init_var.get());
          if (outer_it != array_carry_vars_.end()) outer_init_arr = &outer_it->second;
        }
        if (outer_init_arr && outer_init_arr->size == N) {
          code_ << Indent() << "for (int64_t __init_i = 0; __init_i < " << N << "; ++__init_i) "
                << rv_array_name << "[__init_i] = " << outer_init_arr->array_name << "[__init_i];\n";
        } else {
          code_ << Indent() << "for (int64_t __init_i = 0; __init_i < " << N << "; ++__init_i) "
                << rv_array_name << "[__init_i] = " << init_var_name << ";\n";
        }
        // Register rv as array-carry (yields target into this array).
        RegisterArrayCarry(return_var.get(), rv_array_name, N);
        if (for_stmt->kind_ == ForKind::Parallel) {
          // Parallel iter_arg: per-iter "value" is the init (same for all iters)
          // — used as the deps source. If init is an array, alias to it; if
          // init is scalar, register the iter_arg's slot map to that scalar
          // broadcast (so EmitManualDeps emits add_dep on the same scalar).
          if (outer_init_arr && outer_init_arr->size == N) {
            RegisterArrayCarry(iter_arg.get(), outer_init_arr->array_name, N);
          } else {
            manual_task_id_map_[iter_arg.get()] = init_var_name;
            // Also forward the iter_arg's emit name to the scalar init so that
            // nested ForStmts whose iter_args use this iter_arg as their own
            // ``initValue_`` resolve via ``TryGetVarName`` to the right scalar
            // TaskId variable (not the function param's bare name).
            emit_name_map_[iter_arg.get()] = init_var_name;
          }
        } else {
          // Sequential: iter_arg and rv share the same array (in-place
          // updates via yield).
          RegisterArrayCarry(iter_arg.get(), rv_array_name, N);
        }
      } else if (auto array_ty = As<ArrayType>(iter_arg->GetType()); array_ty && is_rebind[i]) {
        // ArrayType iter_arg — declare a fresh C-stack array and route both
        // the iter_arg and the return_var emit names through it. The loop
        // body's ``array.update_element`` calls already alias their LHS to
        // the input array's emit name (see HandleArrayUpdateElementAssign),
        // so writes land in-place on the carry array. The matching YieldStmt
        // is skipped via name-equality below — no value copy is needed.
        //
        // Distinct from the TaskId array-carry path above: that path is
        // driven by the *trip count* of the surrounding ForStmt and uses
        // ``array_carry_vars_`` to fan out add_dep across slots. Here the
        // size comes from the iter_arg's own ArrayType extent.
        auto extent_const = As<ConstInt>(array_ty->extent());
        INTERNAL_CHECK_SPAN(extent_const, for_stmt->span_)
            << "Internal error: ArrayType iter_arg extent must be ConstInt, got "
            << array_ty->extent()->TypeName();
        const int64_t N = extent_const->value_;
        const std::string cpp_dtype =
            array_ty->dtype_ == DataType::TASK_ID ? "PTO2TaskId" : array_ty->dtype_.ToCTypeString();
        std::string carry_name = ReserveSyntheticEmitName(return_var->name_hint_);
        code_ << Indent() << cpp_dtype << " " << carry_name << "[" << N << "] = {0};\n";
        // Init by slot-by-slot copy from the init array.
        code_ << Indent() << "for (int64_t __init_i = 0; __init_i < " << N << "; ++__init_i) " << carry_name
              << "[__init_i] = " << init_var_name << "[__init_i];\n";
        emit_name_map_[iter_arg.get()] = carry_name;
        emit_name_map_[return_var.get()] = carry_name;
      } else if (is_rebind[i]) {
        // Scalar (single-name) carry path — only fires for non-TaskId
        // iter_args (e.g. Tensor) or Sequential TaskId iter_args whose
        // yield value isn't an inner array. Parallel TaskId iter_args are
        // guaranteed to use the array-carry path above (the const-trip-count
        // CHECK above would have fired otherwise).
        //
        // Pick a fresh, distinct name for the carry. We can't reuse
        // ReserveVarEmitName(return_var) because the lineage analyzer has
        // already populated emit_name_map_[return_var] with the init's param
        // name (when the iter_arg's init value is a function parameter), which
        // would emit the self-referential `auto x = x;`. Use the return_var's
        // raw name_hint (e.g. `current_hidden__rv_v3`) — uniqued against all
        // declared names.
        std::string carry_name = ReserveSyntheticEmitName(return_var->name_hint_);
        code_ << Indent() << GetCppType(return_var->GetType()) << " " << carry_name << " = " << init_var_name
              << ";\n";
        emit_name_map_[return_var.get()] = carry_name;
        emit_name_map_[iter_arg.get()] = carry_name;
        // Sequential TaskId carry: register both endpoints in the task-id
        // map so EmitManualDeps and yield writes can find the carry name.
        auto sty = As<ScalarType>(iter_arg->GetType());
        if (in_manual_scope_depth_ > 0 && sty && sty->dtype_ == DataType::TASK_ID) {
          manual_task_id_map_[iter_arg.get()] = carry_name;
          manual_task_id_map_[return_var.get()] = carry_name;
        }
      } else {
        // Trivial yield: preserve the legacy aliasing — both names route to
        // the init's emit name. YieldStmt for this slot will be a self-assign
        // and skipped (see VisitStmt_(YieldStmtPtr) for the equality guard).
        emit_name_map_[iter_arg.get()] = init_var_name;
        emit_name_map_[return_var.get()] = init_var_name;
      }
    }

    code_ << Indent() << "for (int64_t " << loop_var << " = " << start_expr << "; " << loop_var << " < "
          << stop_expr << "; " << loop_var << " += " << step_expr << ") {\n";
    indent_ += 4;

    // Inside a MANUAL scope, the runtime forbids nested AUTO scope, so we
    // omit the implicit ``PTO2_SCOPE()`` wrapper around the loop body.
    bool emit_implicit_scope = (in_manual_scope_depth_ == 0);
    if (emit_implicit_scope) {
      code_ << Indent() << "PTO2_SCOPE() {\n";
      indent_ += 4;
    }

    auto saved = current_return_vars_;
    // Only register return_vars whose yield is a true rebind. Trivial slots
    // are aliased to the init name and don't need a yield-time assignment;
    // emitting one would just produce `init = init;`.
    current_return_vars_.clear();
    for (size_t i = 0; i < for_stmt->return_vars_.size(); ++i) {
      if (is_rebind[i]) {
        current_return_vars_.push_back(for_stmt->return_vars_[i]);
      } else {
        current_return_vars_.push_back(nullptr);
      }
    }
    // Array-carry slot index = (loop_var - start) / step (0-based ordinal).
    // For the common ``pl.parallel(N)`` case (start=0, step=1) this reduces
    // to ``loop_var`` itself; peephole the expression so generated code stays
    // readable. For non-trivial ranges like ``pl.parallel(2, 10, 2)`` the
    // raw loop_var (2,4,6,8) would index out of the ``arr[N=4]`` allocation —
    // see CodeRabbit thread on PR #1330.
    std::string slot_expr = loop_var;
    auto start_ci = As<ConstInt>(for_stmt->start_);
    auto step_ci = As<ConstInt>(for_stmt->step_);
    bool trivial_range = start_ci && step_ci && start_ci->value_ == 0 && step_ci->value_ == 1;
    if (!trivial_range) {
      slot_expr = "((" + loop_var + " - " + start_expr + ") / " + step_expr + ")";
    }
    current_loop_slot_exprs_.push_back(slot_expr);
    VisitStmt(for_stmt->body_);
    current_loop_slot_exprs_.pop_back();
    current_return_vars_ = saved;

    if (emit_implicit_scope) {
      indent_ -= 4;
      code_ << Indent() << "}\n";
    }
    indent_ -= 4;
    code_ << Indent() << "}\n";
  }

  void VisitStmt_(const RuntimeScopeStmtPtr& scope) override {
    code_ << Indent() << "PTO2_SCOPE(" << (scope->manual_ ? "PTO2ScopeMode::MANUAL" : "") << ") {\n";
    indent_ += 4;
    decltype(manual_task_id_map_) saved_map;
    decltype(array_carry_vars_) saved_array_carry;
    if (scope->manual_) {
      ++in_manual_scope_depth_;
      saved_map = std::move(manual_task_id_map_);
      saved_array_carry = std::move(array_carry_vars_);
      manual_task_id_map_.clear();
      array_carry_vars_.clear();
    }
    VisitStmt(scope->body_);
    if (scope->manual_) {
      --in_manual_scope_depth_;
      manual_task_id_map_ = std::move(saved_map);
      array_carry_vars_ = std::move(saved_array_carry);
    }
    indent_ -= 4;
    code_ << Indent() << "}\n";
  }

  void VisitStmt_(const IfStmtPtr& if_stmt) override {
    std::string cond_expr = GenerateExprString(if_stmt->condition_);

    // ``Tensor`` has no public default ctor, so ``Tensor x;`` won't compile.
    // The phi declaration for a Tensor return_var must be initialised with a
    // valid Tensor that's already in scope. Try sources in order:
    //   1. The first tensor function parameter (selected by orch arg index,
    //      not lexical name — picking from ``param_name_set_`` could yield a
    //      scalar param and emit ``Tensor x = <scalar>;`` which won't compile).
    //   2. A Var yielded by either branch that's already declared at
    //      if-entry (an outer iter_arg, or a prior in-body assignment). This
    //      handles parameterless functions whose branches yield a name
    //      computed before the if.
    std::string tensor_phi_init;
    if (!param_name_to_orch_index_.empty()) {
      // param_name_to_orch_index_ is keyed by tensor param name only (scalar
      // params are excluded from this map). Pick the entry with the smallest
      // orch index to keep the choice deterministic across parameter order.
      const std::string* first_tensor_name = nullptr;
      int min_idx = std::numeric_limits<int>::max();
      for (const auto& [name, idx] : param_name_to_orch_index_) {
        if (idx < min_idx) {
          min_idx = idx;
          first_tensor_name = &name;
        }
      }
      if (first_tensor_name) {
        tensor_phi_init = GetExternalTensorName(*first_tensor_name);
      }
    }
    if (tensor_phi_init.empty()) {
      auto find_in_scope_tensor_var = [&](const StmtPtr& body) -> std::string {
        auto y = transform_utils::GetLastYieldStmt(body);
        if (!y) return {};
        for (const auto& v : y->value_) {
          auto var = AsVarLike(v);
          if (!var) continue;
          if (!As<TensorType>(var->GetType())) continue;
          auto it = emit_name_map_.find(var.get());
          if (it == emit_name_map_.end()) continue;
          return GetExternalTensorName(it->second);
        }
        return {};
      };
      tensor_phi_init = find_in_scope_tensor_var(if_stmt->then_body_);
      if (tensor_phi_init.empty() && if_stmt->else_body_.has_value()) {
        tensor_phi_init = find_in_scope_tensor_var(*if_stmt->else_body_);
      }
    }

    for (const auto& rv : if_stmt->return_vars_) {
      const std::string emit_name = ReserveVarEmitName(rv.get());
      const std::string cpp_type = GetCppType(rv->GetType());
      if (cpp_type == "Tensor") {
        INTERNAL_CHECK_SPAN(!tensor_phi_init.empty(), if_stmt->span_)
            << "Internal error: IfStmt return_var '" << rv->name_hint_
            << "' is a Tensor but no in-scope Tensor was found to use as the "
            << "phi placeholder (``Tensor`` has a private default ctor). "
            << "Expected either a function parameter or a branch yield value "
            << "to resolve to a Var already declared at if-entry.";
        // Phi placeholder init — overwritten by branch yields.
        code_ << Indent() << cpp_type << " " << emit_name << " = " << tensor_phi_init << ";\n";
      } else {
        code_ << Indent() << cpp_type << " " << emit_name << ";\n";
      }
    }

    code_ << Indent() << "if (" << cond_expr << ") {\n";
    VisitScopedBranchBody(if_stmt->then_body_, if_stmt->return_vars_);

    const auto& else_body = if_stmt->else_body_;
    if (else_body.has_value()) {
      code_ << Indent() << "} else {\n";
      VisitScopedBranchBody(*else_body, if_stmt->return_vars_);
    }

    code_ << Indent() << "}\n";
  }

  void VisitStmt_(const AssignStmtPtr& assign) override {
    std::string var_name = ReserveVarEmitName(assign->var_.get());

    if (auto call = As<Call>(assign->value_)) {
      const std::string& op_name = call->op_->name_;

      // Special-case TaskId synthesis ops (synthesized by LowerManualDepsToTaskId).
      if (op_name == "system.task_invalid") {
        code_ << Indent() << "PTO2TaskId " << var_name << " = PTO2TaskId::invalid();\n";
        return;
      }
      if (op_name == "system.task_id_of") {
        // Argument is the producer Var; emit
        //   ``PTO2TaskId <lhs> = task_<n>_outs.task_id();``
        // and register the LHS as a string-variant entry so downstream
        // ``add_dep`` consumers reference the LHS variable name directly
        // (no need for a ``is_valid()`` guard here — task_id_of always
        // produces a valid id).
        INTERNAL_CHECK_SPAN(call->args_.size() == 1, assign->span_)
            << "Internal error: system.task_id_of expects exactly 1 argument";
        auto producer = AsVarLike(call->args_[0]);
        INTERNAL_CHECK_SPAN(producer, assign->span_)
            << "Internal error: system.task_id_of argument must be a Var";
        auto it = manual_task_id_map_.find(producer.get());
        INTERNAL_CHECK_SPAN(it != manual_task_id_map_.end(), assign->span_)
            << "Internal error: system.task_id_of producer '" << producer->name_hint_
            << "' has no task counter in current manual scope";
        const int* idx = std::get_if<int>(&it->second);
        INTERNAL_CHECK_SPAN(idx, assign->span_)
            << "Internal error: system.task_id_of producer '" << producer->name_hint_
            << "' is not a kernel-Call LHS (variant holds non-int)";
        code_ << Indent() << "PTO2TaskId " << var_name << " = task_" << *idx << "_outs.task_id();\n";
        // Register so downstream EmitManualDeps emits ``add_dep(<var_name>);``
        // (string variant — but no is_valid guard since this id is known
        // valid; the EmitManualDeps string branch unconditionally guards,
        // which is harmless for a known-valid id).
        manual_task_id_map_[assign->var_.get()] = var_name;
        return;
      }
      if (op_name == "system.task_is_valid") {
        // ``b = task_is_valid(t)`` lowers to ``bool b = <t>.is_valid();``.
        // The argument is any Scalar[TASK_ID] expression — a Var holding a
        // companion id, or an ``array.get_element`` call on a TASK_ID array
        // carry. ``GenerateExprString`` handles both.
        INTERNAL_CHECK_SPAN(call->args_.size() == 1, assign->span_)
            << "Internal error: system.task_is_valid expects exactly 1 argument";
        std::string arg_expr = GenerateExprString(call->args_[0]);
        code_ << Indent() << "bool " << var_name << " = " << arg_expr << ".is_valid();\n";
        return;
      }

      if (IsTensorOp(op_name)) {
        if (op_name == "tensor.assemble") {
          HandleTensorAssembleAssign(assign, call);
        } else {
          GenerateTensorOpCode(call, var_name, assign->var_);
        }
      } else if (IsArrayOp(op_name)) {
        // ArrayType ops emit C-stack array operations. ``array.update_element``
        // is SSA-functional: the LHS Var refers to the post-update array. At
        // codegen time we alias the LHS to the input array's emit name so the
        // emitted write lands on the same storage.
        if (op_name == "array.update_element") {
          HandleArrayUpdateElementAssign(assign, call);
        }
        GenerateTensorOpCode(call, var_name, assign->var_);
      } else if (!IsBuiltinOp(op_name)) {
        std::string result_key;
        if (As<TupleType>(call->GetType())) {
          auto it = call_to_tuple_key_.find(call.get());
          result_key = (it != call_to_tuple_key_.end()) ? it->second : var_name;
        } else {
          result_key = var_name;
        }
        int task_idx_before = task_counter_;
        GenerateFunctionCallCode(call, result_key);

        if (in_manual_scope_depth_ > 0 && task_counter_ > task_idx_before) {
          // Bind the LHS Var to the just-emitted ``task_<n>`` so a downstream
          // sibling call inside the same manual scope can ``add_dep`` on it.
          manual_task_id_map_[assign->var_.get()] = task_idx_before;
        }

        if (!As<TupleType>(call->GetType())) {
          if (effective_uses_.count(assign->var_.get())) {
            GenerateSingleReturnAlias(call, var_name);
          }
        } else {
          GenerateTupleReturnAliases(call);
        }
      } else {
        INTERNAL_CHECK_SPAN(false, assign->span_)
            << "Misplaced builtin op '" << op_name
            << "' in Orchestration function (should be inside InCore block)";
      }
    } else if (As<TupleGetItemExpr>(assign->value_)) {
      // No-op: tuple elements handled via tuple_var_to_elements_
    } else {
      std::string value_expr = GenerateExprString(assign->value_);
      // Drop a no-op `X = X;` that arises when VarLineageCollector has
      // collapsed both LHS and RHS onto the same param-rooted emit name
      // (both Vars alias the same buffer). Without this guard the catch-all
      // emit path produces `auto X = X;`, which gcc rejects with
      // "use of 'X' before deduction of 'auto'".
      //
      // FIXME(#1281): this is a codegen-layer band-aid. The cleaner home is
      // an IR-level copy-propagation pass that drops lineage-redundant
      // Var-RHS AssignStmts so downstream analyses don't have to reason
      // about no-op aliases at all (today's Simplify only does scalar
      // constant propagation, not tensor-Var copy prop). Once such a pass
      // exists, this guard can go.
      if (AsVarLike(assign->value_) && value_expr == var_name) {
        return;
      }
      code_ << Indent() << GetCppType(assign->var_->GetType()) << " " << var_name << " = " << value_expr
            << ";\n";
    }
  }

  void VisitStmt_(const ReturnStmtPtr& ret) override {
    // No-op: return tensors are already make_tensor_external
  }

  void VisitStmt_(const SeqStmtsPtr& seq) override {
    EmitBatchedAllocTensors(seq->stmts_);
    for (const auto& stmt : seq->stmts_) {
      if (batched_create_stmts_.count(stmt.get())) continue;
      VisitStmt(stmt);
    }
  }

  void VisitStmt_(const YieldStmtPtr& yield_stmt) override {
    for (size_t i = 0; i < yield_stmt->value_.size(); ++i) {
      if (i >= current_return_vars_.size() || !current_return_vars_[i]) {
        // Null slots in current_return_vars_ are placeholders for trivial
        // yields (where the body does not rebind the iter_arg) — skip those.
        continue;
      }
      const auto& rv = current_return_vars_[i];
      // Array-carry rv: route yield writes into the underlying ``arr[N]``.
      auto rv_arr_it = array_carry_vars_.find(rv.get());
      if (rv_arr_it != array_carry_vars_.end()) {
        const auto& rv_arr = rv_arr_it->second;
        auto yield_var = AsVarLike(yield_stmt->value_[i]);
        INTERNAL_CHECK_SPAN(yield_var, yield_stmt->span_)
            << "Internal error: array-carry yield expects a Var value";
        auto inner_it = array_carry_vars_.find(yield_var.get());
        if (inner_it != array_carry_vars_.end()) {
          // Array → array slot-by-slot copy (Sequential outer receiving an
          // inner Parallel's array).
          INTERNAL_CHECK_SPAN(inner_it->second.size == rv_arr.size, yield_stmt->span_)
              << "Internal error: array-carry yield size mismatch (rv=" << rv_arr.size
              << ", yield=" << inner_it->second.size << ")";
          code_ << Indent() << "for (int64_t __yield_i = 0; __yield_i < " << rv_arr.size << "; ++__yield_i) "
                << rv_arr.array_name << "[__yield_i] = " << inner_it->second.array_name << "[__yield_i];\n";
        } else {
          // Scalar → array slot write (Parallel inner yielding a fresh task id
          // into its slot). The slot index is the enclosing loop's 0-based
          // ordinal, so non-trivial parallel ranges still index ``arr[0..N-1]``.
          auto map_it = manual_task_id_map_.find(yield_var.get());
          INTERNAL_CHECK_SPAN(map_it != manual_task_id_map_.end(), yield_stmt->span_)
              << "Internal error: scalar yield to array carry must resolve to a "
              << "TaskId variable registered in manual_task_id_map_";
          auto* scalar_name = std::get_if<std::string>(&map_it->second);
          INTERNAL_CHECK_SPAN(scalar_name, yield_stmt->span_)
              << "Internal error: scalar yield to array carry expects string-variant entry";
          INTERNAL_CHECK_SPAN(!current_loop_slot_exprs_.empty(), yield_stmt->span_)
              << "Internal error: scalar yield to array carry requires an enclosing loop var";
          code_ << Indent() << rv_arr.array_name << "[" << current_loop_slot_exprs_.back()
                << "] = " << *scalar_name << ";\n";
        }
        continue;
      }
      // Scalar rv: existing path.
      std::string value_expr = GenerateExprString(yield_stmt->value_[i]);
      // Function tensor params are renamed to ``ext_<name>`` in the emitted
      // C++; if the yield value is a Var aliased to a param's bare name,
      // translate it here. Safe for non-params (returns input unchanged).
      value_expr = GetExternalTensorName(value_expr);
      auto yield_var = AsVarLike(yield_stmt->value_[i]);
      std::string lhs_name = GetVarName(rv);
      // Skip self-assigns. Pointer identity catches the trivial-yield case;
      // the name-equality check catches ArrayType iter_args where the body's
      // ``array.update_element`` aliased its LHS back to the iter_arg's emit
      // name — the carry already holds the post-update value, so emitting
      // ``carry = carry;`` would be both useless and invalid C for arrays.
      if (rv.get() == yield_var.get() || lhs_name == value_expr) {
        continue;
      }
      // ArrayType rv: raw C arrays are not directly assignable. Emit an
      // explicit slot-by-slot copy. This fires at the outer-ForStmt yield in
      // a SEQ x PARALLEL phase fence where the inner ArrayType rv must be
      // copied back into the outer ArrayType carry so state propagates
      // across phases.
      if (auto rv_array_ty = As<ArrayType>(rv->GetType())) {
        auto extent_const = As<ConstInt>(rv_array_ty->extent());
        INTERNAL_CHECK_SPAN(extent_const, yield_stmt->span_)
            << "Internal error: ArrayType rv extent must be ConstInt for slot-by-slot yield copy";
        const int64_t N = extent_const->value_;
        code_ << Indent() << "for (int64_t __yield_i = 0; __yield_i < " << N << "; ++__yield_i) " << lhs_name
              << "[__yield_i] = " << value_expr << "[__yield_i];\n";
        continue;
      }
      code_ << Indent() << lhs_name << " = " << value_expr << ";\n";
    }
  }

  void VisitStmt_(const EvalStmtPtr& eval) override {
    if (auto call = As<Call>(eval->expr_)) {
      const std::string& op_name = call->op_->name_;
      if (IsTensorOp(op_name) || IsArrayOp(op_name)) {
        GenerateTensorOpCode(call, "", nullptr);
      } else if (!IsBuiltinOp(op_name)) {
        GenerateFunctionCallCode(call, "");
      } else {
        INTERNAL_CHECK_SPAN(false, eval->span_)
            << "Misplaced builtin op '" << op_name
            << "' in Orchestration function (should be inside InCore block)";
      }
    }
  }

 private:
  std::string Indent() const { return std::string(indent_, ' '); }

  std::string GetCppType(const TypePtr& type) {
    if (auto scalar_type = As<ScalarType>(type)) {
      // TaskId is an opaque 64-bit handle, not numeric — emit as PTO2TaskId.
      if (scalar_type->dtype_ == DataType::TASK_ID) return "PTO2TaskId";
      return scalar_type->dtype_.ToCTypeString();
    }
    // TensorType: use ``Tensor`` so default-constructible declarations are
    // legal (C++ rejects ``auto x;`` without init). Yield/Assign downstream
    // will rebind it.
    if (As<TensorType>(type)) return "Tensor";
    // ArrayType has split declaration syntax (``dtype name[N]``) — there's no
    // single "type expression" that names a C array. Callers that need to
    // emit a Var of ArrayType always go through array.create's op codegen,
    // which emits the declaration directly. If this branch ever fires, the
    // catch-all ``auto X = Y;`` path would produce invalid C — treat it as a
    // missed dispatch and surface it loudly.
    INTERNAL_CHECK(!As<ArrayType>(type))
        << "GetCppType called for ArrayType — array vars must be declared via "
           "array.create's op codegen, not the catch-all AssignStmt path";
    return "auto";
  }

  // Encode a scalar variable for the orchestration API.
  // float variables must be bit-cast via to_u64(); other types pass through as-is.
  static std::string EncodeScalarVar(const std::string& var_name, const std::string& cpp_type) {
    return cpp_type == "float" ? "to_u64(" + var_name + ")" : var_name;
  }

  // Encode a scalar constant expression for the orchestration API.
  // float literals need to_u64() and an "f" suffix; other types need an explicit (uint64_t) cast.
  static std::string EncodeScalarConst(const std::string& value, const std::string& cpp_type) {
    return cpp_type == "float" ? "to_u64(" + value + "f)" : "(uint64_t)" + value;
  }

  [[nodiscard]] std::string GetExternalTensorName(const std::string& name) const override {
    if (param_name_set_.count(name)) {
      return "ext_" + name;
    }
    return name;
  }

  // Map an IR ArgDirection directly to the runtime Arg::add_* method name.
  // ArgDirection is the single source of truth for codegen.
  static const char* ArgDirectionToMethodName(ArgDirection dir) {
    switch (dir) {
      case ArgDirection::Input:
        return "add_input";
      case ArgDirection::Output:
      case ArgDirection::OutputExisting:
        // The runtime overloads add_output on the argument type:
        //   add_output(TensorCreateInfo&)  -> OUTPUT       (runtime allocates)
        //   add_output(Tensor&)            -> OUTPUT_EXISTING (write-only existing tensor)
        // The codegen pre-allocates via tensor.create + alloc_tensors, so the emitted
        // call site always passes a Tensor& and the OUTPUT_EXISTING overload is selected.
        // We still distinguish the two ArgDirections in the IR to let downstream phases
        // switch to the runtime-allocated form without an IR change.
        return "add_output";
      case ArgDirection::InOut:
        return "add_inout";
      case ArgDirection::NoDep:
        return "add_no_dep";
      case ArgDirection::Scalar:
        return "add_scalar";
    }
    INTERNAL_CHECK(false) << "Internal error: unexpected ArgDirection value";
    return "";
  }

  struct ParamEntry {
    ArgDirection direction;
    std::string value;
  };

  std::vector<ParamEntry> BuildTaskParams(const CallPtr& call, const FunctionPtr& callee_func) {
    std::vector<ParamEntry> params;
    const std::string& callee_name = callee_func->name_;

    // Phase-5 invariant: every Call entering codegen must carry an explicit
    // ArgDirection vector (populated by the DeriveCallDirections IR pass).
    // The legacy ParamDirection fallback has been removed.
    auto call_arg_directions = call->GetArgDirections();
    INTERNAL_CHECK_SPAN(call_arg_directions.size() == call->args_.size(), call->span_)
        << "Call to '" << callee_name << "' has arg_directions size " << call_arg_directions.size()
        << " but args size " << call->args_.size()
        << ". DeriveCallDirections must run before orchestration codegen.";

    for (size_t arg_idx = 0; arg_idx < call->args_.size(); ++arg_idx) {
      const auto& arg = call->args_[arg_idx];
      std::string var_name = TryGetVarName(arg);
      if (!var_name.empty()) {
        if (auto scalar_type = As<ScalarType>(arg->GetType())) {
          std::string cpp_type = scalar_type->dtype_.ToCTypeString();
          params.push_back({ArgDirection::Scalar, EncodeScalarVar(var_name, cpp_type)});
          continue;
        }

        std::string ext_name = GetExternalTensorName(var_name);
        params.push_back({call_arg_directions[arg_idx], ext_name});
      } else if (auto const_int = As<ConstInt>(arg)) {
        std::string cpp_type = const_int->dtype().ToCTypeString();
        std::string value = FormatConstIntValue(const_int, cpp_type);
        params.push_back({ArgDirection::Scalar, "(uint64_t)" + value});
      } else if (auto const_float = As<ConstFloat>(arg)) {
        std::string cpp_type = const_float->dtype().ToCTypeString();
        std::string value = FormatConstFloatValue(const_float, cpp_type);
        params.push_back({ArgDirection::Scalar, EncodeScalarConst(value, cpp_type)});
      } else if (auto const_bool = As<ConstBool>(arg)) {
        params.push_back({ArgDirection::Scalar, const_bool->value_ ? "(uint64_t)1" : "(uint64_t)0"});
      } else {
        INTERNAL_CHECK_SPAN(false, call->span_) << "Call to '" << callee_name << "' arg " << arg_idx
                                                << " is neither a variable nor a recognized constant literal "
                                                << "(unsupported expression kind for orchestration codegen).";
      }
    }

    INTERNAL_CHECK_SPAN(params.size() == call->args_.size(), call->span_)
        << "Call to '" << callee_name << "' built " << params.size() << " params for " << call->args_.size()
        << " call args (1:1 invariant violated).";

    // New PTOParam API: tensors must precede scalars (see check_add_tensor_valid() in pto_types.h)
    std::stable_partition(params.begin(), params.end(),
                          [](const ParamEntry& p) { return p.direction != ArgDirection::Scalar; });

    return params;
  }

  void GenerateTensorOpCode(const CallPtr& call, const std::string& result_var, const VarPtr& assign_var) {
    const std::string& op_name = call->op_->name_;

    auto& registry = OrchestrationOpRegistry::GetInstance();
    auto codegen_func = registry.Get(op_name);
    INTERNAL_CHECK_SPAN(codegen_func.has_value(), call->span_)
        << "Misplaced tensor op '" << op_name
        << "' in Orchestration function (should be inside InCore block)";

    if (op_name == "tensor.create" && assign_var &&
        (declared_var_ptrs_.count(assign_var.get()) || param_name_set_.count(GetVarName(assign_var)))) {
      return;
    }

    std::string emit_var = result_var;
    if (op_name == "tensor.create" && assign_var) {
      declared_var_ptrs_.insert(assign_var.get());
      emit_var = ReserveVarEmitName(assign_var.get());
    }

    current_result_var_ = emit_var;

    std::string gen_code = (*codegen_func)(call, *this);

    std::istringstream iss(gen_code);
    std::string line;
    while (std::getline(iss, line)) {
      if (!line.empty()) {
        code_ << Indent() << line << "\n";
      }
    }

    if (op_name == "tensor.create") {
      EmitAllocBatch({emit_var});
    }
  }

  struct GroupCalleeInfo {
    std::string aic_name;
    std::string aiv_name;
    CallPtr inner_call;        // The call to the InCore kernel inside the Group body
    FunctionPtr inner_callee;  // The InCore function being called
  };

  struct WrapperCallInfo {
    CallPtr inner_call;
    FunctionPtr inner_callee;
  };

  WrapperCallInfo FindWrapperInnerCall(const FunctionPtr& wrapper_func) {
    auto info = ir::FindFirstInnerCall(wrapper_func, program_);
    return {std::move(info.inner_call), std::move(info.inner_callee)};
  }

  /// Walk the Group function body to find the AIC and AIV callee names
  /// and the inner InCore call (needed for param reordering).
  GroupCalleeInfo FindGroupCallees(const FunctionPtr& group_func) {
    auto info = ir::FindGroupCallees(group_func, program_);
    return {std::move(info.aic_name), std::move(info.aiv_name), std::move(info.inner_call),
            std::move(info.inner_callee)};
  }

  /// Build task params for a wrapper function call, reordered to match the
  /// inner callee's parameter order.
  ///
  /// Wrapper functions may omit constants or otherwise expose a different
  /// parameter order than the callee binary expects. Submit args using the
  /// inner callee's order, not the wrapper's order.
  std::vector<ParamEntry> BuildWrapperReorderedParams(const CallPtr& outer_call,
                                                      const FunctionPtr& wrapper_func,
                                                      const CallPtr& inner_call,
                                                      const FunctionPtr& inner_callee) {
    std::unordered_map<const Var*, size_t> wrapper_param_to_outer_idx;
    for (size_t i = 0; i < wrapper_func->params_.size(); ++i) {
      wrapper_param_to_outer_idx[wrapper_func->params_[i].get()] = i;
    }

    // Phase-5 invariant: the outer Call must carry explicit arg_directions
    // (populated by DeriveCallDirections). The legacy ParamDirection fallback
    // has been removed from codegen.
    auto outer_arg_directions = outer_call->GetArgDirections();
    INTERNAL_CHECK_SPAN(outer_arg_directions.size() == outer_call->args_.size(), outer_call->span_)
        << "Outer call to wrapper '" << wrapper_func->name_ << "' has arg_directions size "
        << outer_arg_directions.size() << " but args size " << outer_call->args_.size()
        << ". DeriveCallDirections must run before orchestration codegen.";

    std::vector<ParamEntry> params;
    for (size_t inner_idx = 0; inner_idx < inner_call->args_.size(); ++inner_idx) {
      const auto& inner_arg = inner_call->args_[inner_idx];
      auto inner_arg_var = AsVarLike(inner_arg);

      // Constant args are inlined in the wrapper body — emit them directly.
      if (!inner_arg_var) {
        if (auto const_int = As<ConstInt>(inner_arg)) {
          std::string cpp_type = const_int->dtype().ToCTypeString();
          std::string value = FormatConstIntValue(const_int, cpp_type);
          params.push_back({ArgDirection::Scalar, "(uint64_t)" + value});
        } else if (auto const_float = As<ConstFloat>(inner_arg)) {
          std::string cpp_type = const_float->dtype().ToCTypeString();
          std::string value = FormatConstFloatValue(const_float, cpp_type);
          params.push_back({ArgDirection::Scalar, EncodeScalarConst(value, cpp_type)});
        } else if (auto const_bool = As<ConstBool>(inner_arg)) {
          params.push_back({ArgDirection::Scalar, const_bool->value_ ? "(uint64_t)1" : "(uint64_t)0"});
        } else {
          INTERNAL_CHECK_SPAN(false, inner_call->span_) << "Internal error: inner call arg " << inner_idx
                                                        << " is neither a variable nor a recognized constant";
        }
        continue;
      }

      auto it = wrapper_param_to_outer_idx.find(inner_arg_var.get());
      if (it == wrapper_param_to_outer_idx.end()) {
        // Some wrapper-expansion paths can leave inner-call scalar ivs that are
        // not part of the user-visible wrapper signature. They should not be
        // forwarded as orchestration args.
        if (As<ScalarType>(inner_arg->GetType()) != nullptr) {
          continue;
        }
        INTERNAL_CHECK_SPAN(false, inner_call->span_)
            << "Internal error: inner call arg " << inner_idx << " does not map to any wrapper parameter";
      }

      size_t outer_idx = it->second;
      const auto& outer_arg = outer_call->args_[outer_idx];
      std::string var_name = TryGetVarName(outer_arg);

      if (!var_name.empty()) {
        if (auto scalar_type = As<ScalarType>(outer_arg->GetType())) {
          std::string cpp_type = scalar_type->dtype_.ToCTypeString();
          params.push_back({ArgDirection::Scalar, EncodeScalarVar(var_name, cpp_type)});
          continue;
        }

        std::string ext_name = GetExternalTensorName(var_name);
        params.push_back({outer_arg_directions[outer_idx], ext_name});
      } else if (auto const_int = As<ConstInt>(outer_arg)) {
        std::string cpp_type = const_int->dtype().ToCTypeString();
        std::string value = FormatConstIntValue(const_int, cpp_type);
        params.push_back({ArgDirection::Scalar, "(uint64_t)" + value});
      } else if (auto const_float = As<ConstFloat>(outer_arg)) {
        std::string cpp_type = const_float->dtype().ToCTypeString();
        std::string value = FormatConstFloatValue(const_float, cpp_type);
        params.push_back({ArgDirection::Scalar, EncodeScalarConst(value, cpp_type)});
      } else if (auto const_bool = As<ConstBool>(outer_arg)) {
        params.push_back({ArgDirection::Scalar, const_bool->value_ ? "(uint64_t)1" : "(uint64_t)0"});
      } else {
        INTERNAL_CHECK_SPAN(false, outer_call->span_)
            << "Outer call to wrapper '" << wrapper_func->name_ << "' arg " << outer_idx
            << " is neither a variable nor a recognized constant literal "
            << "(unsupported expression kind for orchestration codegen).";
      }
    }

    INTERNAL_CHECK_SPAN(params.size() == inner_call->args_.size(), inner_call->span_)
        << "Wrapper '" << wrapper_func->name_ << "' built " << params.size() << " params for "
        << inner_call->args_.size() << " inner-call args (1:1 invariant violated).";

    // Tensors must precede scalars
    std::stable_partition(params.begin(), params.end(),
                          [](const ParamEntry& p) { return p.direction != ArgDirection::Scalar; });

    return params;
  }

  // Render the launched function's core_num attribute as a C++ scalar expression.
  // Accepts a ConstInt literal or a Var resolving to an orchestration-scope scalar
  // variable (routed through TryGetVarName so SSA/emit-name mapping is respected).
  [[nodiscard]] std::string RenderLaunchCoreNum(const ExprPtr& expr) const {
    if (auto ci = As<ConstInt>(expr)) {
      return std::to_string(ci->value_);
    }
    if (As<Var>(expr) != nullptr) {
      return TryGetVarName(expr);
    }
    INTERNAL_CHECK_SPAN(false, expr->span_)
        << "Unsupported core_num expression kind for orchestration codegen: "
        << "expected ConstInt or Var, got kind=" << static_cast<int>(expr->GetKind());
    return "";
  }

  void EmitLaunchSpec(const std::string& ind, const std::string& task_var, const FunctionPtr& launch_func) {
    auto core_num_expr = launch_func->GetAttr<ExprPtr>("core_num", nullptr);
    bool sync_start = launch_func->GetAttr<bool>("sync_start", false);
    if (core_num_expr) {
      const std::string method = pypto::backend::GetBackend()->GetHandler()->GetLaunchSpecCoreCountMethod();
      code_ << ind << task_var << ".launch_spec." << method << "(" << RenderLaunchCoreNum(core_num_expr)
            << ");\n";
    }
    if (sync_start) {
      code_ << ind << task_var << ".launch_spec.set_require_sync_start(true);\n";
    }
  }

  void EmitTaskSubmitAndBind(const std::string& submit_expr) {
    if (in_manual_scope_depth_ > 0) {
      // Manual scope: capture the submit's outputs so later code can call
      // ``.task_id()`` on it. We do NOT pre-emit a ``PTO2TaskId task_<n>``
      // variable — instead, downstream consumers (system.task_id_of in IR,
      // EmitManualDeps for direct deps) emit ``task_<n>_outs.task_id()``
      // expressions directly. This avoids a redundant variable in the common
      // case where the caller threads the task id through a TaskId iter_arg
      // carry rather than referring to the kernel's task counter by name.
      const std::string outs_var = "task_" + std::to_string(task_counter_) + "_outs";
      code_ << Indent() << "TaskOutputTensors " << outs_var << " = " << submit_expr << ";\n";
    } else {
      code_ << Indent() << submit_expr << ";\n";
    }
    task_counter_++;
  }

  /// Upper bound on the number of ``add_dep`` entries that ``EmitManualDeps``
  /// would emit for ``call``. Used to size the per-task ``ArgWithDeps<N>``
  /// wrapper. Returns 0 outside a manual scope (auto scope derives deps from
  /// data flow at runtime, so codegen emits no ``add_dep`` calls).
  size_t CountManualDeps(const CallPtr& call) const {
    if (in_manual_scope_depth_ == 0) return 0;
    for (const auto& [k, v] : call->attrs_) {
      if (k != kAttrManualDepEdges) continue;
      const auto* edges = std::any_cast<std::vector<VarPtr>>(&v);
      if (edges == nullptr) return 0;
      size_t total = 0;
      for (const auto& edge : *edges) {
        if (!edge) continue;
        auto it = manual_task_id_map_.find(edge.get());
        if (it == manual_task_id_map_.end()) continue;
        if (std::get_if<int>(&it->second)) continue;
        if (auto* names = std::get_if<std::vector<std::string>>(&it->second)) {
          total += names->size();
        } else {
          total += 1;
        }
      }
      return total;
    }
    return 0;
  }

  /// Emit the per-task ``Arg`` (or ``ArgWithDeps<N>``) declaration. Sizes
  /// ``N`` to the exact dep-edge count when in a manual scope; emits plain
  /// ``Arg`` outside manual scope or when no edges are present.
  void EmitTaskParamsDecl(const std::string& ind, const std::string& task_var, const CallPtr& call) {
    const size_t dep_count = CountManualDeps(call);
    if (dep_count == 0) {
      code_ << ind << "Arg " << task_var << ";\n";
    } else {
      code_ << ind << "ArgWithDeps<" << dep_count << "> " << task_var << ";\n";
    }
  }

  /// Emit ``params_t<n>.add_dep(task_<m>);`` for every entry resolved by the
  /// manual-scope lowering phase of ``DeriveCallDirections``. No-op outside a
  /// manual scope (auto scope derives deps from data flow at runtime).
  void EmitManualDeps(const CallPtr& call, const std::string& task_var) {
    if (in_manual_scope_depth_ == 0) return;
    for (const auto& [k, v] : call->attrs_) {
      if (k != kAttrManualDepEdges) continue;
      const auto* edges = std::any_cast<std::vector<VarPtr>>(&v);
      INTERNAL_CHECK_SPAN(edges != nullptr, call->span_)
          << "Internal error: " << kAttrManualDepEdges << " attr must hold std::vector<VarPtr>";
      for (const auto& edge : *edges) {
        if (!edge) continue;
        auto it = manual_task_id_map_.find(edge.get());
        INTERNAL_CHECK_SPAN(it != manual_task_id_map_.end(), call->span_)
            << "Internal error: manual_dep_edge var '" << edge->name_hint_
            << "' has no producer task in current manual scope";
        if (std::get_if<int>(&it->second)) {
          // Invariant: a ``manual_dep_edges`` entry should never resolve
          // directly to a kernel-Call LHS (int-variant entry).
          // ``LowerManualDepsToTaskId`` rewrites every dep edge Var to its
          // ``system.task_id_of`` LHS (TaskIdType, string-variant entry).
          // Hitting this branch means the pass was bypassed or the IR
          // was hand-built without going through it.
          INTERNAL_CHECK_SPAN(false, call->span_)
              << "Internal error: manual_dep_edge var '" << edge->name_hint_
              << "' resolves to a kernel-Call LHS (int variant). Expected "
              << "LowerManualDepsToTaskId to have rewritten this edge to a "
              << "system.task_id_of LHS (TaskIdType, string variant).";
        } else if (auto* names = std::get_if<std::vector<std::string>>(&it->second)) {
          // Array-carry iter_arg: depend on every slot, each guarded by
          // ``is_valid()`` (early-phase slots may still be invalid).
          for (const auto& name : *names) {
            code_ << Indent() << "if (" << name << ".is_valid()) {\n";
            code_ << Indent() << "    " << task_var << ".add_dep(" << name << ");\n";
            code_ << Indent() << "}\n";
          }
        } else {
          const auto& name = std::get<std::string>(it->second);
          // Iter-arg TaskId carries can be ``PTO2TaskId::invalid()`` on the
          // first loop iteration; guard with is_valid(). For non-iter-arg
          // string entries (e.g. task_id_of LHS) the value is always valid,
          // so emit unguarded.
          if (edge->GetKind() == ObjectKind::IterArg) {
            code_ << Indent() << "if (" << name << ".is_valid()) {\n";
            code_ << Indent() << "    " << task_var << ".add_dep(" << name << ");\n";
            code_ << Indent() << "}\n";
          } else {
            code_ << Indent() << task_var << ".add_dep(" << name << ");\n";
          }
        }
      }
      break;
    }
  }

  static constexpr size_t kMaxAllocTensorsArgs = 16;

  static bool IsInjectedGMPipeCreateVar(const VarPtr& var) {
    return var && var->name_hint_.rfind("gm_pipe_buffer_", 0) == 0;
  }

  int64_t GetGMPipeWorkspaceElements(const FunctionPtr& root_func) {
    INTERNAL_CHECK(root_func != nullptr)
        << "Internal error: GM pipe workspace root function must not be null";
    auto it = gm_pipe_workspace_elements_by_callee_.find(root_func->name_);
    if (it == gm_pipe_workspace_elements_by_callee_.end()) {
      const int64_t elements = ComputeGMPipeWorkspaceElements(program_, root_func);
      it = gm_pipe_workspace_elements_by_callee_.emplace(root_func->name_, elements).first;
    }
    return it->second;
  }

  struct GMPipeCreateUse {
    FunctionPtr callee;
    std::string core_num_expr;
  };

  [[nodiscard]] std::optional<GMPipeCreateUse> ResolveGMPipeCreateUse(const std::vector<StmtPtr>& stmts,
                                                                      size_t create_stmt_idx,
                                                                      const VarPtr& create_var) const {
    if (!create_var) return std::nullopt;

    for (size_t i = create_stmt_idx + 1; i < stmts.size(); ++i) {
      auto assign = As<AssignStmt>(stmts[i]);
      if (assign && assign->var_ && assign->var_.get() == create_var.get()) {
        break;
      }

      CallPtr call;
      if (assign) {
        call = As<Call>(assign->value_);
      } else if (auto eval = As<EvalStmt>(stmts[i])) {
        call = As<Call>(eval->expr_);
      }
      if (!call) continue;

      bool uses_create_var = false;
      for (const auto& arg : call->args_) {
        auto arg_var = AsVarLike(arg);
        if (arg_var && arg_var.get() == create_var.get()) {
          uses_create_var = true;
          break;
        }
      }
      if (!uses_create_var) continue;

      auto gv = As<GlobalVar>(call->op_);
      if (!gv) return std::nullopt;
      auto callee_func = program_->GetFunction(gv->name_);
      if (!callee_func) return std::nullopt;

      auto core_num_expr = callee_func->GetAttr<ExprPtr>("core_num", nullptr);
      std::string rendered_core_num;
      if ((callee_func->func_type_ == FunctionType::Spmd || callee_func->func_type_ == FunctionType::Group) &&
          core_num_expr) {
        rendered_core_num = RenderLaunchCoreNum(core_num_expr);
      }
      return GMPipeCreateUse{callee_func, rendered_core_num};
    }
    return std::nullopt;
  }

  static bool ExprRefsAnyOf(const ExprPtr& expr, const std::unordered_set<const Var*>& vars) {
    if (!expr) {
      return false;
    }
    if (auto var = As<Var>(expr)) {
      return vars.count(var.get()) > 0;
    }
    if (auto bin = As<BinaryExpr>(expr)) {
      return ExprRefsAnyOf(bin->left_, vars) || ExprRefsAnyOf(bin->right_, vars);
    }
    if (auto un = As<UnaryExpr>(expr)) {
      return ExprRefsAnyOf(un->operand_, vars);
    }
    if (auto cast_expr = As<Cast>(expr)) {
      return ExprRefsAnyOf(cast_expr->operand_, vars);
    }
    return false;
  }

  bool ShapeDependsOnLocalVars(const CallPtr& call,
                               const std::unordered_set<const Var*>& locally_defined) const {
    auto result_type = As<TensorType>(call->GetType());
    if (!result_type) {
      return false;
    }
    for (const auto& dim : result_type->shape_) {
      if (ExprRefsAnyOf(dim, locally_defined)) {
        return true;
      }
    }
    return false;
  }

  void EmitAllocBatch(const std::vector<std::string>& emit_names) {
    std::string alloc_var = "alloc_" + std::to_string(alloc_counter_++);
    code_ << Indent() << "TaskOutputTensors " << alloc_var << " = alloc_tensors(";
    for (size_t i = 0; i < emit_names.size(); i++) {
      if (i > 0) {
        code_ << ", ";
      }
      code_ << emit_names[i] << "_ci";
    }
    code_ << ");\n";

    for (size_t i = 0; i < emit_names.size(); i++) {
      code_ << Indent() << "const Tensor& " << emit_names[i] << " = " << alloc_var << ".get_ref(" << i
            << ");\n";
    }
  }

  void EmitBatchedAllocTensors(const std::vector<StmtPtr>& stmts) {
    struct PendingCreate {
      std::string emit_name;
      CallPtr call;
    };
    std::vector<PendingCreate> creates;

    std::unordered_set<const Var*> locally_defined;

    for (size_t stmt_idx = 0; stmt_idx < stmts.size(); ++stmt_idx) {
      const auto& stmt = stmts[stmt_idx];
      auto assign = As<AssignStmt>(stmt);
      if (!assign) {
        continue;
      }
      auto call = As<Call>(assign->value_);
      if (!call || call->op_->name_ != "tensor.create") {
        locally_defined.insert(assign->var_.get());
        continue;
      }
      if (declared_var_ptrs_.count(assign->var_.get()) || param_name_set_.count(GetVarName(assign->var_))) {
        continue;
      }
      if (ShapeDependsOnLocalVars(call, locally_defined)) {
        locally_defined.insert(assign->var_.get());
        continue;
      }

      declared_var_ptrs_.insert(assign->var_.get());
      std::string emit_var = ReserveVarEmitName(assign->var_.get());
      if (IsInjectedGMPipeCreateVar(assign->var_)) {
        auto create_use = ResolveGMPipeCreateUse(stmts, stmt_idx, assign->var_);
        CHECK(create_use.has_value())
            << "Internal error: injected gm_pipe_buffer tensor.create is not passed to a callee";
        int64_t workspace_elems = GetGMPipeWorkspaceElements(create_use->callee);
        CHECK(workspace_elems > 0)
            << "Internal error: injected gm_pipe_buffer tensor.create found without initialize_pipe ops";
        std::string size_expr = std::to_string(workspace_elems);
        const std::string& core_num_expr = create_use->core_num_expr;
        if (!core_num_expr.empty()) {
          size_expr = "static_cast<uint32_t>((" + size_expr + ") * (" + core_num_expr + "))";
        }
        tensor_create_size_expr_by_emit_name_[emit_var] = size_expr;
      }
      creates.push_back({emit_var, call});
      batched_create_stmts_.insert(stmt.get());
    }
    if (creates.empty()) {
      return;
    }

    auto& registry = OrchestrationOpRegistry::GetInstance();
    auto handler = registry.Get("tensor.create");
    if (!handler.has_value()) {
      INTERNAL_CHECK_SPAN(false, creates[0].call->span_)
          << "Internal error: tensor.create handler not registered";
    }
    const auto& tensor_create_handler = *handler;

    for (size_t batch_start = 0; batch_start < creates.size(); batch_start += kMaxAllocTensorsArgs) {
      size_t batch_end = std::min(batch_start + kMaxAllocTensorsArgs, creates.size());

      for (size_t i = batch_start; i < batch_end; i++) {
        current_result_var_ = creates[i].emit_name;
        std::string gen_code = tensor_create_handler(creates[i].call, *this);
        std::istringstream iss(gen_code);
        std::string line;
        while (std::getline(iss, line)) {
          if (!line.empty()) {
            code_ << Indent() << line << "\n";
          }
        }
      }

      std::vector<std::string> batch_names;
      for (size_t i = batch_start; i < batch_end; i++) {
        batch_names.push_back(creates[i].emit_name);
      }
      EmitAllocBatch(batch_names);
    }
  }

  void GenerateFunctionCallCode(const CallPtr& call, const std::string& result_var) {
    const std::string& callee_name = call->op_->name_;

    FunctionPtr callee_func = program_->GetFunction(callee_name);
    INTERNAL_CHECK_SPAN(callee_func != nullptr, call->span_)
        << "Internal error: function '" << callee_name << "' not found after validation.";

    if (callee_func->func_type_ == FunctionType::Spmd) {
      GenerateSpmdCallCode(call, callee_func);
      return;
    }

    if (callee_func->func_type_ == FunctionType::Group) {
      GenerateGroupCallCode(call, callee_func, callee_func);
      return;
    }

    CoreType core_type = InferFunctionCoreType(callee_func);
    (*func_name_to_core_type_)[callee_name] = core_type;

    int func_id = GetOrCreateFuncId(callee_name, func_name_to_id_, next_func_id_);

    auto params = BuildTaskParams(call, callee_func);

    std::string ind = Indent();
    std::string task_var = "params_t" + std::to_string(task_counter_);
    code_ << "\n";
    code_ << ind << "// Task " << task_counter_ << ": " << callee_name << "\n";
    EmitTaskParamsDecl(ind, task_var, call);
    for (const auto& p : params) {
      code_ << ind << task_var << "." << ArgDirectionToMethodName(p.direction) << "(" << p.value << ");\n";
    }
    EmitManualDeps(call, task_var);

    std::string submit_expr =
        CoreTypeToSubmitPrefix(core_type) + std::to_string(func_id) + ", " + task_var + ")";
    EmitTaskSubmitAndBind(submit_expr);
  }

  void GenerateSpmdCallCode(const CallPtr& call, const FunctionPtr& spmd_func) {
    auto info = FindWrapperInnerCall(spmd_func);
    INTERNAL_CHECK(info.inner_call != nullptr && info.inner_callee != nullptr)
        << "Internal error: no inner call found in Spmd function '" << spmd_func->name_ << "'";

    if (info.inner_callee->func_type_ == FunctionType::Group) {
      GenerateGroupCallCode(call, info.inner_callee, spmd_func);
      return;
    }

    const std::string& callee_name = info.inner_callee->name_;
    CoreType core_type = InferFunctionCoreType(info.inner_callee);
    (*func_name_to_core_type_)[callee_name] = core_type;

    int func_id = GetOrCreateFuncId(callee_name, func_name_to_id_, next_func_id_);
    auto params = BuildWrapperReorderedParams(call, spmd_func, info.inner_call, info.inner_callee);

    std::string ind = Indent();
    std::string task_var = "params_t" + std::to_string(task_counter_);
    code_ << "\n";
    code_ << ind << "// Spmd " << spmd_func->name_ << ": " << callee_name << "\n";
    EmitTaskParamsDecl(ind, task_var, call);
    for (const auto& p : params) {
      code_ << ind << task_var << "." << ArgDirectionToMethodName(p.direction) << "(" << p.value << ");\n";
    }
    EmitLaunchSpec(ind, task_var, spmd_func);
    EmitManualDeps(call, task_var);

    std::string submit_expr =
        CoreTypeToSubmitPrefix(core_type) + std::to_string(func_id) + ", " + task_var + ")";
    EmitTaskSubmitAndBind(submit_expr);
  }

  void GenerateGroupCallCode(const CallPtr& call, const FunctionPtr& group_func,
                             const FunctionPtr& launch_func) {
    std::string group_name = group_func->name_;

    auto info = FindGroupCallees(group_func);

    // AIV-only Group: pure vector SPMD kernel (no AIC callee).
    // Dispatch as a single AIV task with core_num/sync_start from the Group.
    // Use rt_submit_aiv_task which dispatches across independent AIV cores,
    // unlike rt_submit_task (MixedKernels) which dispatches full clusters.
    if (info.aic_name.empty() && !info.aiv_name.empty()) {
      FunctionPtr aiv_func = program_->GetFunction(info.aiv_name);
      INTERNAL_CHECK(aiv_func != nullptr) << "Internal error: AIV function '" << info.aiv_name
                                          << "' not found for Group '" << group_name << "'";

      (*func_name_to_core_type_)[info.aiv_name] = CoreType::VECTOR;
      int aiv_id = GetOrCreateFuncId(info.aiv_name, func_name_to_id_, next_func_id_);

      // Reorder params from wrapper param order to inner kernel arg order.
      INTERNAL_CHECK(info.inner_call != nullptr && info.inner_callee != nullptr)
          << "Internal error: no inner call found in AIV-only Group '" << group_name << "'";
      auto params = BuildWrapperReorderedParams(call, group_func, info.inner_call, info.inner_callee);

      std::string ind = Indent();
      std::string task_var = "params_t" + std::to_string(task_counter_);
      code_ << "\n";
      code_ << ind << "// Group " << group_name << ": AIV-only SPMD\n";
      EmitTaskParamsDecl(ind, task_var, call);
      for (const auto& p : params) {
        code_ << ind << task_var << "." << ArgDirectionToMethodName(p.direction) << "(" << p.value << ");\n";
      }

      EmitLaunchSpec(ind, task_var, launch_func);
      EmitManualDeps(call, task_var);

      std::string submit_expr =
          CoreTypeToSubmitPrefix(CoreType::VECTOR) + std::to_string(aiv_id) + ", " + task_var + ")";
      EmitTaskSubmitAndBind(submit_expr);
      return;
    }

    INTERNAL_CHECK_SPAN(!info.aic_name.empty(), call->span_)
        << "Internal error: no AIC callee found in Group '" << group_name << "' body";
    INTERNAL_CHECK_SPAN(!info.aiv_name.empty(), call->span_)
        << "Internal error: no AIV callee found in Group '" << group_name << "' body";

    FunctionPtr aic_func = program_->GetFunction(info.aic_name);
    FunctionPtr aiv_func = program_->GetFunction(info.aiv_name);
    INTERNAL_CHECK_SPAN(aic_func != nullptr, call->span_) << "Internal error: AIC function '" << info.aic_name
                                                          << "' not found for Group '" << group_name << "'";
    INTERNAL_CHECK_SPAN(aiv_func != nullptr, call->span_) << "Internal error: AIV function '" << info.aiv_name
                                                          << "' not found for Group '" << group_name << "'";

    (*func_name_to_core_type_)[info.aic_name] = CoreType::CUBE;
    (*func_name_to_core_type_)[info.aiv_name] = CoreType::VECTOR;

    // Reorder params from wrapper param order to inner kernel arg order.
    INTERNAL_CHECK(info.inner_call != nullptr && info.inner_callee != nullptr)
        << "Internal error: no inner call found in MixedKernels Group '" << group_name << "'";
    auto params = BuildWrapperReorderedParams(call, group_func, info.inner_call, info.inner_callee);

    int aic_id = GetOrCreateFuncId(info.aic_name, func_name_to_id_, next_func_id_);
    int aiv_id = GetOrCreateFuncId(info.aiv_name, func_name_to_id_, next_func_id_);

    std::string ind = Indent();
    std::string task_var = "params_t" + std::to_string(task_counter_);

    code_ << "\n";
    code_ << ind << "// Group " << group_name << ": MixedKernels (AIC + AIV lanes)\n";
    EmitTaskParamsDecl(ind, task_var, call);
    for (const auto& p : params) {
      code_ << ind << task_var << "." << ArgDirectionToMethodName(p.direction) << "(" << p.value << ");\n";
    }
    // Split AIV groups dispatch the same kernel on both vector lanes. The
    // kernel body uses tile.get_subblock_idx() to select its lane-local slice.
    std::string third_id = RequiresDualAivDispatch(aiv_func) ? std::to_string(aiv_id) : "INVALID_KERNEL_ID";
    code_ << ind << "MixedKernels mixed_" << task_counter_ << " = {" << aic_id << ", " << aiv_id << ", "
          << third_id << "};\n";

    EmitLaunchSpec(ind, task_var, launch_func);
    EmitManualDeps(call, task_var);

    std::string submit_expr = "rt_submit_task(mixed_" + std::to_string(task_counter_) + ", " + task_var + ")";
    EmitTaskSubmitAndBind(submit_expr);
  }

  // --- Alias generation helpers ---

  std::vector<ParamDirection> GetEffectiveDirections(const FunctionPtr& callee) {
    if (callee->func_type_ == FunctionType::Group || callee->func_type_ == FunctionType::Spmd) {
      return ComputeGroupEffectiveDirections(callee, program_);
    }
    return callee->param_directions_;
  }

  std::vector<size_t> CollectOutIndices(const FunctionPtr& callee) {
    const auto dirs = GetEffectiveDirections(callee);
    std::vector<size_t> out_indices;
    for (size_t i = 0; i < dirs.size(); ++i) {
      if (dirs[i] == ParamDirection::Out || dirs[i] == ParamDirection::InOut) {
        out_indices.push_back(i);
      }
    }
    return out_indices;
  }

  void EmitTensorAlias(const std::string& alias_name, const CallPtr& call, size_t arg_idx) {
    std::string out_arg = TryGetVarName(call->args_[arg_idx]);
    if (!out_arg.empty() && alias_name != out_arg) {
      code_ << Indent() << "const Tensor& " << alias_name << " = " << GetExternalTensorName(out_arg) << ";\n";
    }
  }

  void GenerateSingleReturnAlias(const CallPtr& call, const std::string& var_name) {
    FunctionPtr callee = program_->GetFunction(call->op_->name_);
    if (!callee) return;
    auto out_indices = CollectOutIndices(callee);
    if (!out_indices.empty()) {
      EmitTensorAlias(var_name, call, out_indices[0]);
    }
  }

  void GenerateTupleReturnAliases(const CallPtr& call) {
    auto tuple_key_it = call_to_tuple_key_.find(call.get());
    if (tuple_key_it == call_to_tuple_key_.end()) return;
    auto elements_it = tuple_var_to_elements_.find(tuple_key_it->second);
    if (elements_it == tuple_var_to_elements_.end()) return;
    FunctionPtr callee = program_->GetFunction(call->op_->name_);
    if (!callee) return;

    auto call_arg_directions = call->GetArgDirections();
    bool has_call_dirs = (call_arg_directions.size() == call->args_.size());
    std::vector<size_t> out_indices;
    if (has_call_dirs) {
      for (size_t i = 0; i < call_arg_directions.size(); ++i) {
        ArgDirection d = call_arg_directions[i];
        if (d == ArgDirection::Output || d == ArgDirection::InOut || d == ArgDirection::OutputExisting) {
          out_indices.push_back(i);
        }
      }
    } else {
      auto effective_dirs = GetEffectiveDirections(callee);
      for (size_t i = 0; i < effective_dirs.size(); ++i) {
        if (effective_dirs[i] == ParamDirection::Out || effective_dirs[i] == ParamDirection::InOut) {
          out_indices.push_back(i);
        }
      }
    }

    int max_tuple_index = -1;
    for (const auto& elem : elements_it->second) {
      max_tuple_index = std::max(max_tuple_index, elem.index);
    }
    size_t tuple_arity = max_tuple_index >= 0 ? static_cast<size_t>(max_tuple_index + 1) : 0;
    size_t tuple_out_base = tuple_arity >= out_indices.size() ? (tuple_arity - out_indices.size()) : 0;

    for (const auto& elem : elements_it->second) {
      // Some wrappers (notably SPMD helpers) return auxiliary scalars before
      // Out/InOut tensors, e.g. (idx, out_tensor). Map tuple tail elements to
      // Out/InOut params and ignore leading non-output tuple elements.
      if (elem.index < 0) {
        continue;
      }
      size_t elem_pos = static_cast<size_t>(elem.index);
      if (elem_pos < tuple_out_base) {
        // Leading tuple elements are auxiliary values (e.g. loop iv from
        // SPMD wrappers). They are not returned by runtime task submission.
        // If such scalar is referenced later, materialize a safe default to
        // keep generated orchestration compilable.
        if (effective_uses_.count(elem.var)) {
          std::string elem_name = ReserveVarEmitName(elem.var);
          if (auto st = As<ScalarType>(elem.var->GetType())) {
            code_ << Indent() << st->dtype_.ToCTypeString() << " " << elem_name << " = 0;\n";
          }
        }
        continue;
      }
      size_t out_pos = elem_pos - tuple_out_base;
      if (out_pos >= out_indices.size()) {
        continue;
      }
      size_t param_idx = out_indices[out_pos];
      INTERNAL_CHECK_SPAN(param_idx < call->args_.size(), call->span_)
          << "Internal error: resolved param_idx " << param_idx << " out of range for " << call->op_->name_
          << " (has " << call->args_.size() << " args)";

      if (!effective_uses_.count(elem.var)) {
        continue;
      }
      std::string elem_name = ReserveVarEmitName(elem.var);
      EmitTensorAlias(elem_name, call, param_idx);
    }
  }

  void VisitScopedBranchBody(const StmtPtr& body, const std::vector<VarPtr>& return_vars) {
    indent_ += 4;
    bool emit_implicit_scope = (in_manual_scope_depth_ == 0);
    if (emit_implicit_scope) {
      code_ << Indent() << "PTO2_SCOPE() {\n";
      indent_ += 4;
    }

    auto saved = current_return_vars_;
    current_return_vars_.assign(return_vars.begin(), return_vars.end());
    VisitStmt(body);
    current_return_vars_ = saved;

    if (emit_implicit_scope) {
      indent_ -= 4;
      code_ << Indent() << "}\n";
    }
    indent_ -= 4;
  }

  void HandleTensorAssembleAssign(const AssignStmtPtr& assign, const CallPtr& call) {
    INTERNAL_CHECK_SPAN(call->args_.size() == 3, call->span_)
        << "Internal error: tensor.assemble expects 3 arguments";

    std::string target_name = GenerateExprString(call->args_[0]);
    target_name = GetExternalTensorName(target_name);
    emit_name_map_[assign->var_.get()] = target_name;
  }

  void HandleArrayUpdateElementAssign(const AssignStmtPtr& assign, const CallPtr& call) {
    // array.update_element(array, index, value) -> ArrayType.
    // The SSA-functional return value shares storage with the first arg; alias
    // the LHS so subsequent references resolve to the same C variable.
    INTERNAL_CHECK_SPAN(call->args_.size() == 3, call->span_)
        << "Internal error: array.update_element expects 3 arguments";
    std::string array_name = GenerateExprString(call->args_[0]);
    emit_name_map_[assign->var_.get()] = array_name;
  }

  std::string ReserveVarEmitName(const Var* var) {
    auto it = emit_name_map_.find(var);
    if (it != emit_name_map_.end()) {
      return it->second;
    }

    auto parsed = auto_name::Parse(var->name_hint_);
    bool preserve_raw_name = parsed.role.has_value() && *parsed.role == "out";
    std::string base_name = GetSSABaseName(var->name_hint_);
    if (preserve_raw_name || declared_var_names_.count(base_name)) {
      base_name = var->name_hint_;
    }

    std::string emit_name = auto_name::ReserveUniqueName(base_name, declared_var_names_);
    emit_name_map_[var] = emit_name;
    return emit_name;
  }

  std::string ReserveSyntheticEmitName(const std::string& base_name) {
    return auto_name::ReserveUniqueName(base_name, declared_var_names_);
  }

  /// Register ``var`` as backed by ``array_name[size]``; also populates the
  /// ``manual_task_id_map_`` with the per-slot expressions so EmitManualDeps
  /// emits one ``add_dep`` per slot when this Var appears as a deps source.
  void RegisterArrayCarry(const Var* var, const std::string& array_name, int64_t size) {
    array_carry_vars_[var] = ArrayCarryEntry{array_name, size};
    std::vector<std::string> slot_names;
    slot_names.reserve(static_cast<size_t>(size));
    for (int64_t i = 0; i < size; ++i) {
      slot_names.push_back(array_name + "[" + std::to_string(i) + "]");
    }
    manual_task_id_map_[var] = std::move(slot_names);
  }

  /// Constant-evaluate ``expr`` if it is a ConstInt; returns ``nullopt``
  /// otherwise. Used to size TaskId carry arrays at codegen time.
  static std::optional<int64_t> EvalConstInt(const ExprPtr& expr) {
    if (auto ci = As<ConstInt>(expr)) return ci->value_;
    return std::nullopt;
  }

  /// Return the const trip count of ``for_stmt`` if start/stop/step are all
  /// ConstInts and step is positive; 0 otherwise. We only support array carry
  /// for Parallel loops with statically-known trip counts.
  static int64_t EvalConstTripCount(const ForStmtPtr& for_stmt) {
    auto start = EvalConstInt(for_stmt->start_);
    auto stop = EvalConstInt(for_stmt->stop_);
    auto step = EvalConstInt(for_stmt->step_);
    if (!start || !stop || !step || *step <= 0) return 0;
    int64_t trip = (*stop - *start + *step - 1) / *step;
    return trip > 0 ? trip : 0;
  }

  /// Find a ForStmt within ``body`` whose ``return_vars_`` contains a Var
  /// equal to ``target``. Returns nullptr if none. Used by
  /// ``ResolveArrayCarrySize`` to chase Sequential→Parallel array threading.
  static ForStmtPtr FindForStmtByReturnVar(const StmtPtr& body, const Var* target) {
    class Finder : public IRVisitor {
     public:
      ForStmtPtr result;
      const Var* target = nullptr;
      void VisitStmt_(const ForStmtPtr& f) override {
        if (result) return;
        for (const auto& rv : f->return_vars_) {
          if (rv.get() == target) {
            result = f;
            return;
          }
        }
        IRVisitor::VisitStmt_(f);
      }
    };
    Finder finder;
    finder.target = target;
    finder.VisitStmt(body);
    return finder.result;
  }

  /// Determine the carry size for a TaskId iter_arg of ``for_stmt`` at slot
  /// ``idx``. Returns 0 when the iter_arg is scalar-carry.
  ///   * Parallel ForStmt with const trip count: carry size = trip count.
  ///   * Sequential ForStmt whose yield value at ``idx`` is the rv of an
  ///     inner array-carry ForStmt: carry size = inner's size (recurses).
  ///   * Anything else: 0 (scalar carry).
  int64_t ResolveArrayCarrySize(const ForStmtPtr& for_stmt, size_t idx) const {
    if (idx >= for_stmt->iter_args_.size()) return 0;
    const auto& iter_arg = for_stmt->iter_args_[idx];
    auto sty = As<ScalarType>(iter_arg->GetType());
    if (!sty || sty->dtype_ != DataType::TASK_ID) return 0;
    if (for_stmt->kind_ == ForKind::Parallel) {
      return EvalConstTripCount(for_stmt);
    }
    if (for_stmt->kind_ != ForKind::Sequential) return 0;
    auto yield = transform_utils::GetLastYieldStmt(for_stmt->body_);
    if (!yield || idx >= yield->value_.size()) return 0;
    auto yield_var = AsVarLike(yield->value_[idx]);
    if (!yield_var) return 0;
    auto inner = FindForStmtByReturnVar(for_stmt->body_, yield_var.get());
    if (!inner) return 0;
    size_t inner_idx = SIZE_MAX;
    for (size_t j = 0; j < inner->return_vars_.size(); ++j) {
      if (inner->return_vars_[j].get() == yield_var.get()) {
        inner_idx = j;
        break;
      }
    }
    if (inner_idx == SIZE_MAX) return 0;
    return ResolveArrayCarrySize(inner, inner_idx);
  }

  const ProgramPtr& program_;
  std::map<std::string, int>* func_name_to_id_;
  std::map<std::string, CoreType>* func_name_to_core_type_;
  int* next_func_id_;
  std::unordered_map<const Var*, std::string> emit_name_map_;
  std::set<std::string> declared_var_names_;
  std::set<std::string> param_name_set_;
  std::map<std::string, int> param_name_to_orch_index_;
  std::ostringstream code_;
  int indent_ = 4;
  std::string current_result_var_;
  std::vector<VarPtr> current_return_vars_;
  int task_counter_ = 0;
  int alloc_counter_ = 0;
  /// Depth of nested ``RuntimeScopeStmt(manual=true)``. While > 0, the codegen
  /// suppresses the implicit ``PTO2_SCOPE()`` wrapper around ForStmt/IfStmt
  /// bodies (the runtime forbids nesting AUTO scope inside MANUAL).
  int in_manual_scope_depth_ = 0;
  /// Map from a producer ``Var`` to the task identity to use when it appears
  /// as a ``manual_dep_edge``. Three cases:
  ///   * ``int`` value: kernel-Call LHS, the int is the task counter assigned
  ///     by ``EmitTaskSubmitAndBind``. Invariant-only: should be rewritten by
  ///     ``LowerManualDepsToTaskId`` before reaching dep emission.
  ///   * ``std::string`` value: a ``ForStmt`` TaskId iter_arg's emit name OR a
  ///     ``system.task_id_of`` LHS emit name (single PTO2TaskId variable).
  ///   * ``std::vector<std::string>`` value: a ``ForStmt`` TaskId iter_arg that
  ///     carries an *array* of task ids (Parallel ForStmt with const trip
  ///     count, or Sequential ForStmt propagating an inner Parallel array
  ///     across iterations). Each element of the vector is the C++ expression
  ///     naming one slot of the array (e.g. ``out_arr[0]``, ``out_arr[1]``).
  ///     EmitManualDeps emits ``add_dep`` for each slot with is_valid() guard.
  /// Cleared on entry to each manual scope.
  std::unordered_map<const Var*, std::variant<int, std::string, std::vector<std::string>>>
      manual_task_id_map_;
  /// Records the C++ array allocation backing a TaskId carry that holds an
  /// array of task ids (not a scalar). Used by ``YieldStmt`` to decide how
  /// to write into the carry:
  ///   * Parallel ForStmt rv: yield writes one slot ``arr[<loop_var>] = value``
  ///   * Sequential ForStmt rv whose body yields an array: yield copies
  ///     slot-by-slot from the inner array
  /// The key is either a ``ForStmt`` iter_arg (when used as a deps source) or
  /// a ``ForStmt`` return_var (when used as a yield target). For each key the
  /// recorded ``array_name`` is the C++ identifier of the underlying
  /// ``PTO2TaskId[N]`` array and ``size`` is the slot count ``N``.
  struct ArrayCarryEntry {
    std::string array_name;
    int64_t size;
  };
  std::unordered_map<const Var*, ArrayCarryEntry> array_carry_vars_;
  /// Stack of 0-based slot expressions for the enclosing ForStmts. Pushed
  /// when entering a ForStmt body and popped on exit. Used by ``YieldStmt``
  /// to emit ``arr[<slot>] = value`` for Parallel inner array writes. The
  /// expression is ``(loop_var - start) / step``, peephole-simplified to
  /// just ``loop_var`` when start=0 and step=1.
  std::vector<std::string> current_loop_slot_exprs_;
  std::map<std::string, std::vector<TupleElement>> tuple_var_to_elements_;
  std::map<const Call*, std::string> call_to_tuple_key_;
  std::unordered_set<const Var*> declared_var_ptrs_;
  std::unordered_set<const Stmt*> batched_create_stmts_;
  std::unordered_set<const Var*> effective_uses_;
  std::unordered_map<std::string, int64_t> gm_pipe_workspace_elements_by_callee_;
  std::unordered_map<std::string, std::string> tensor_create_size_expr_by_emit_name_;
};

OrchestrationResult GenerateOrchestration(const ir::ProgramPtr& program, const ir::FunctionPtr& func) {
  CHECK(program != nullptr) << "Cannot generate orchestration for null program";
  CHECK(func != nullptr) << "Cannot generate orchestration for null function";

  // OrchestrationReferencesResolved is verified by the pass pipeline (registered
  // as a property produced by OutlineHierarchyScopes). Codegen assumes well-formed IR.

  std::map<std::string, int> func_name_to_id;
  std::map<std::string, CoreType> func_name_to_core_type;
  int next_func_id = 0;

  OrchestrationInfoCollector info_collector;
  info_collector.VisitStmt(func->body_);

  VarLineageCollector lineage(program);
  lineage.Initialize(func->params_);
  lineage.VisitStmt(func->body_);

  BufferRootCollector root_collector(program);
  root_collector.Initialize(func->params_);
  root_collector.VisitStmt(func->body_);

  CodegenEffectiveUseCollector use_collector;
  use_collector.VisitStmt(func->body_);

  std::unordered_map<const Var*, std::string> emit_name_map;
  std::set<std::string> param_name_set;
  std::map<std::string, int> param_name_to_orch_index;
  int tensor_param_count = 0;
  struct ScalarParamInfo {
    std::string emit_name;
    ScalarTypePtr scalar_type;
  };
  std::vector<ScalarParamInfo> scalar_params;
  for (const auto& var : func->params_) {
    std::string emit_name = GetSSABaseName(var->name_hint_);
    emit_name_map[var.get()] = emit_name;
    param_name_set.insert(emit_name);
    if (As<TensorType>(var->GetType())) {
      param_name_to_orch_index[emit_name] = tensor_param_count;
      tensor_param_count++;
    } else if (auto stype = As<ScalarType>(var->GetType())) {
      scalar_params.push_back({emit_name, stype});
    }
  }

  for (const auto& [body_var, param_var] : lineage.var_to_param) {
    if (emit_name_map.count(body_var) == 0) {
      auto it = emit_name_map.find(param_var);
      if (it != emit_name_map.end()) {
        emit_name_map[body_var] = it->second;
      }
    }
  }

  int expected_arg_count = tensor_param_count + static_cast<int>(scalar_params.size());

  std::ostringstream oss;

  OrchestrationStmtCodegen stmt_codegen(program, &func_name_to_id, &func_name_to_core_type, &next_func_id,
                                        std::move(emit_name_map), std::move(param_name_set),
                                        std::move(param_name_to_orch_index));
  stmt_codegen.SetCallTupleElements(info_collector.call_tuple_elements);
  stmt_codegen.SetCallToTupleKey(info_collector.call_to_tuple_key);
  stmt_codegen.SetEffectiveUses(std::move(use_collector.var_uses));
  stmt_codegen.SetInitialIndent(8);
  stmt_codegen.VisitStmt(func->body_);

  oss << GenerateIncludes(false);

  oss << "extern \"C\" {\n\n";

  oss << GenerateConfigFunction(expected_arg_count);

  oss << "__attribute__((visibility(\"default\")))\n";
  oss << "void aicpu_orchestration_entry(const ChipStorageTaskArgs& orch_args) {\n";

  oss << "    // External tensors\n";
  int orch_idx = 0;
  for (const auto& var : func->params_) {
    auto tensor_type = As<TensorType>(var->GetType());
    if (tensor_type) {
      std::string name = auto_name::GetCompatibleBaseName(var->name_hint_);
      oss << GenerateMakeTensorExternal(name, orch_idx, tensor_type, stmt_codegen);
      orch_idx++;
    }
  }

  if (!scalar_params.empty()) {
    oss << "\n    // Scalar params\n";
    for (size_t i = 0; i < scalar_params.size(); ++i) {
      oss << GenerateScalarUnpack(scalar_params[i].emit_name, static_cast<int>(i),
                                  scalar_params[i].scalar_type);
    }
  }

  oss << "\n    PTO2_SCOPE() {\n";
  oss << stmt_codegen.GetGeneratedCode();
  oss << "    }\n";

  oss << "}\n\n";
  oss << "}  // extern \"C\"\n";

  return OrchestrationResult{oss.str(), std::move(func_name_to_id), std::move(func_name_to_core_type)};
}

}  // namespace codegen
}  // namespace pypto
