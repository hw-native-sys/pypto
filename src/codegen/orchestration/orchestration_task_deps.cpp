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
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/codegen/orchestration/orchestration_analysis.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/type.h"
#include "src/codegen/orchestration/orchestration_stmt_codegen.h"

namespace pypto {
namespace codegen {

using namespace pypto::ir;  // NOLINT(build/namespaces)

void OrchestrationStmtCodegen::PrepareCrossScopeTaskIdHoists(const StmtPtr& body) {
  struct ScopeInfo {
    std::vector<const RuntimeScopeStmt*> ref_scopes;
  };

  class Collector : public IRVisitor {
   public:
    Collector(const std::map<const Var*, std::string>* tuple_var_to_key,
              const std::map<std::string, std::vector<TupleElement>>* tuple_var_to_elements)
        : tuple_var_to_key_(tuple_var_to_key), tuple_var_to_elements_(tuple_var_to_elements) {}

    std::unordered_map<const RuntimeScopeStmt*, const RuntimeScopeStmt*> parent_scope;
    std::unordered_map<std::string, const RuntimeScopeStmt*> defining_scope;
    std::unordered_map<std::string, const Var*> ref_var_by_key;
    std::unordered_map<std::string, ScopeInfo> refs;

   protected:
    void VisitStmt_(const RuntimeScopeStmtPtr& scope) override {
      const RuntimeScopeStmt* parent = CurrentScope();
      parent_scope[scope.get()] = parent;
      scope_stack_.push_back(scope.get());
      IRVisitor::VisitStmt_(scope);
      scope_stack_.pop_back();
    }

    void VisitStmt_(const AssignStmtPtr& assign) override {
      RecordDefinition(assign->var_.get());
      auto tuple_key_it = tuple_var_to_key_->find(assign->var_.get());
      if (tuple_key_it != tuple_var_to_key_->end()) {
        auto elements_it = tuple_var_to_elements_->find(tuple_key_it->second);
        if (elements_it != tuple_var_to_elements_->end()) {
          for (const auto& elem : elements_it->second) {
            RecordDefinition(elem.var);
          }
        }
      }
      IRVisitor::VisitStmt_(assign);
    }

    void VisitExpr_(const CallPtr& call) override {
      RecordCompilerDepRefs(call->attrs_);
      IRVisitor::VisitExpr_(call);
    }

    void VisitExpr_(const SubmitPtr& submit) override {
      RecordCompilerDepRefs(submit->attrs_);
      IRVisitor::VisitExpr_(submit);
    }

   private:
    const RuntimeScopeStmt* CurrentScope() const {
      return scope_stack_.empty() ? nullptr : scope_stack_.back();
    }

    void RecordDefinition(const Var* var) {
      if (!var) return;
      defining_scope.emplace(VarKey(var), CurrentScope());
    }

    void RecordCompilerDepRefs(const std::vector<std::pair<std::string, std::any>>& attrs) {
      for (const auto& [key, value] : attrs) {
        if (key != kAttrCompilerManualDepEdges) continue;
        const auto* edges = std::any_cast<std::vector<VarPtr>>(&value);
        if (!edges) continue;
        for (const auto& edge : *edges) {
          if (!edge) continue;
          const std::string var_key = VarKey(edge.get());
          ref_var_by_key.emplace(var_key, edge.get());
          refs[var_key].ref_scopes.push_back(CurrentScope());
        }
      }
    }

    static std::string VarKey(const Var* var) {
      if (!var->name_hint_.empty()) return var->name_hint_;
      return std::to_string(var->UniqueId());
    }

    const std::map<const Var*, std::string>* tuple_var_to_key_;
    const std::map<std::string, std::vector<TupleElement>>* tuple_var_to_elements_;
    std::vector<const RuntimeScopeStmt*> scope_stack_;
  };

  Collector collector(&tuple_var_to_key_, &tuple_var_to_elements_);
  collector.VisitStmt(body);

  auto parent_of = [&](const RuntimeScopeStmt* scope) -> const RuntimeScopeStmt* {
    auto parent_it = collector.parent_scope.find(scope);
    return parent_it != collector.parent_scope.end() ? parent_it->second : nullptr;
  };

  auto depth_of = [&](const RuntimeScopeStmt* scope) {
    size_t depth = 0;
    for (const RuntimeScopeStmt* cur = scope; cur != nullptr; cur = parent_of(cur)) {
      ++depth;
    }
    return depth;
  };

  auto child_under_lca = [&](const RuntimeScopeStmt* descendant, const RuntimeScopeStmt* lca) {
    const RuntimeScopeStmt* child = descendant;
    for (const RuntimeScopeStmt* parent = parent_of(child); child != nullptr && parent != lca;
         child = parent, parent = parent_of(child)) {
    }
    return child;
  };

  auto hoist_target_for_ref = [&](const RuntimeScopeStmt* defining_scope, const RuntimeScopeStmt* ref_scope) {
    const RuntimeScopeStmt* def_cursor = defining_scope;
    const RuntimeScopeStmt* ref_cursor = ref_scope;
    size_t def_depth = depth_of(def_cursor);
    size_t ref_depth = depth_of(ref_cursor);

    while (def_depth > ref_depth) {
      def_cursor = parent_of(def_cursor);
      --def_depth;
    }
    while (ref_depth > def_depth) {
      ref_cursor = parent_of(ref_cursor);
      --ref_depth;
    }
    while (def_cursor != ref_cursor) {
      def_cursor = parent_of(def_cursor);
      ref_cursor = parent_of(ref_cursor);
    }

    const RuntimeScopeStmt* lca = def_cursor;
    if (lca == defining_scope) return static_cast<const RuntimeScopeStmt*>(nullptr);
    return child_under_lca(defining_scope, lca);
  };

  for (const auto& [var_key, info] : collector.refs) {
    auto def_it = collector.defining_scope.find(var_key);
    if (def_it == collector.defining_scope.end()) continue;
    const RuntimeScopeStmt* defining_scope = def_it->second;
    if (!defining_scope) continue;
    auto ref_var_it = collector.ref_var_by_key.find(var_key);
    if (ref_var_it == collector.ref_var_by_key.end()) continue;
    const RuntimeScopeStmt* hoist_scope = defining_scope;
    size_t hoist_depth = depth_of(hoist_scope);
    for (const RuntimeScopeStmt* ref_scope : info.ref_scopes) {
      const RuntimeScopeStmt* target_scope = hoist_target_for_ref(defining_scope, ref_scope);
      if (!target_scope) continue;
      size_t target_depth = depth_of(target_scope);
      if (target_depth < hoist_depth) {
        hoist_scope = target_scope;
        hoist_depth = target_depth;
      }
    }
    if (hoist_scope != defining_scope || hoist_depth != depth_of(defining_scope)) {
      hoisted_task_id_vars_by_scope_[hoist_scope].push_back(ref_var_it->second);
    }
  }
}

std::vector<VarPtr> OrchestrationStmtCodegen::GetCompilerDependencyEdges(const CallPtr& call) const {
  std::vector<VarPtr> edges;
  if (!call) return edges;
  std::unordered_set<uint64_t> seen;
  for (const auto& [key, value] : call->attrs_) {
    if (key != kAttrCompilerManualDepEdges) continue;
    const auto* attr_edges = std::any_cast<std::vector<VarPtr>>(&value);
    if (!attr_edges) return edges;
    for (const auto& edge : *attr_edges) {
      if (!edge || !seen.insert(edge->UniqueId()).second) continue;
      edges.push_back(edge);
    }
    return edges;
  }
  return edges;
}

std::vector<const Var*> OrchestrationStmtCodegen::CollectCompilerDepArrayBarrierEdges(
    const ForStmtPtr& for_stmt) const {
  class Collector : public IRVisitor {
   public:
    explicit Collector(const OrchestrationStmtCodegen* codegen) : codegen_(codegen) {}

    void VisitStmt_(const AssignStmtPtr& assign) override {
      if (assign->var_) body_defined_vars.insert(assign->var_.get());
      IRVisitor::VisitStmt_(assign);
    }

    void VisitStmt_(const ForStmtPtr& nested_for) override {
      for (const auto& iter_arg : nested_for->iter_args_) {
        if (iter_arg) body_defined_vars.insert(iter_arg.get());
      }
      for (const auto& return_var : nested_for->return_vars_) {
        if (return_var) body_defined_vars.insert(return_var.get());
      }
      IRVisitor::VisitStmt_(nested_for);
    }

    void VisitExpr_(const CallPtr& call) override {
      RecordCompilerDeps(call);
      IRVisitor::VisitExpr_(call);
    }

    void VisitExpr_(const SubmitPtr& submit) override {
      RecordCompilerDeps(SubmitToCallView(submit));
      IRVisitor::VisitExpr_(submit);
    }

    std::unordered_map<const Var*, int64_t> consumer_counts;
    std::unordered_set<const Var*> body_defined_vars;

   private:
    void RecordCompilerDeps(const CallPtr& call) {
      for (const auto& edge : codegen_->GetCompilerDependencyEdges(call)) {
        const auto* binding = codegen_->ResolveManualTaskIdBinding(edge.get());
        if (!binding || !std::holds_alternative<std::vector<std::string>>(*binding)) continue;
        consumer_counts[edge.get()] += 1;
      }
    }

    const OrchestrationStmtCodegen* codegen_;
  };

  const int64_t trip_count = EvalConstTripCount(for_stmt);
  if (trip_count <= 1) return {};

  Collector collector(this);
  for (const auto& iter_arg : for_stmt->iter_args_) {
    if (iter_arg) collector.body_defined_vars.insert(iter_arg.get());
  }
  for (const auto& return_var : for_stmt->return_vars_) {
    if (return_var) collector.body_defined_vars.insert(return_var.get());
  }
  collector.VisitStmt(for_stmt->body_);

  std::vector<const Var*> edges;
  for (const auto& [edge, consumers_per_iter] : collector.consumer_counts) {
    if (!edge || collector.body_defined_vars.count(edge) != 0) continue;
    const auto* binding = ResolveManualTaskIdBinding(edge);
    if (!binding) continue;
    const auto* names = std::get_if<std::vector<std::string>>(binding);
    if (!names || names->empty()) continue;
    const int64_t producer_count = static_cast<int64_t>(names->size());
    const int64_t consumer_count = consumers_per_iter * trip_count;
    const int64_t estimated_saving = producer_count * consumer_count - (producer_count + consumer_count);
    if (estimated_saving <= 0) continue;
    edges.push_back(edge);
  }
  std::sort(edges.begin(), edges.end(),
            [](const Var* lhs, const Var* rhs) { return lhs->UniqueId() < rhs->UniqueId(); });
  return edges;
}

std::string OrchestrationStmtCodegen::EmitCompilerDepArrayBarrier(const std::vector<std::string>& names) {
  const int barrier_idx = phase_fence_barrier_counter_++;
  const std::string task_var = "params_phase_fence_barrier_" + std::to_string(barrier_idx);
  const std::string deps_arr = task_var + "_deps";
  const std::string deps_cnt = task_var + "_deps_count";
  const std::string outs_var = "phase_fence_barrier_" + std::to_string(barrier_idx) + "_outs";
  const std::string tid_name = "phase_fence_barrier_" + std::to_string(barrier_idx) + "_tid";
  EmitBlankLine();
  EmitIndentedLine("// Compiler-dependency barrier " + std::to_string(barrier_idx) +
                   ": compressed loop fan-in");
  EmitTaskParamsDecl(task_var);
  EmitIndentedLine("TaskId " + deps_arr + "[" + std::to_string(names.size()) + "];");
  EmitIndentedLine("uint32_t " + deps_cnt + " = 0;");
  for (const auto& name : names) {
    EmitDepArrayInsert(name, deps_arr, deps_cnt);
  }
  EmitIndentedLine(task_var + ".set_dependencies(" + deps_arr + ", " + deps_cnt + ");");
  EmitIndentedLine("TaskId " + tid_name + " = TaskId::invalid();");
  EmitIndentedLine("if (" + deps_cnt + " > 0) {");
  {
    IndentGuard guard(Active());
    EmitIndentedLine("TaskOutputTensors " + outs_var + " = rt_submit_dummy_task(" + task_var + ");");
    EmitIndentedLine(tid_name + " = " + outs_var + ".task_id();");
  }
  EmitIndentedLine("}");
  return tid_name;
}

std::string OrchestrationStmtCodegen::EmitDynamicCompilerDepBarrier(
    const DynamicTaskIdCollection& collection) {
  const int barrier_idx = phase_fence_barrier_counter_++;
  const std::string task_var = "params_phase_fence_barrier_" + std::to_string(barrier_idx);
  const std::string outs_var = "phase_fence_barrier_" + std::to_string(barrier_idx) + "_outs";
  const std::string tid_name = "phase_fence_barrier_" + std::to_string(barrier_idx) + "_tid";
  EmitBlankLine();
  EmitIndentedLine("// Dynamic compiler-dependency barrier " + std::to_string(barrier_idx) +
                   ": compressed loop fan-in");
  EmitTaskParamsDecl(task_var);
  EmitIndentedLine("TaskId " + tid_name + " = TaskId::invalid();");
  EmitIndentedLine("if (" + collection.count_name + " > 0) {");
  {
    IndentGuard guard(Active());
    EmitIndentedLine(task_var + ".set_dependencies(" + collection.data_name + ".data(), " +
                     collection.count_name + ");");
    EmitIndentedLine("TaskOutputTensors " + outs_var + " = rt_submit_dummy_task(" + task_var + ");");
    EmitIndentedLine(tid_name + " = " + outs_var + ".task_id();");
  }
  EmitIndentedLine("}");
  return tid_name;
}

/// Upper bound on the number of dependency entries ``EmitManualDeps`` could
/// fill for ``call``. Used to size the per-task ``TaskId <task>_deps[K]``
/// stack array. ``manual_dep_edges`` is orthogonal to scope mode: the
/// runtime adds these on top of any auto-tracked deps in auto scope (final
/// fanin = auto ∪ explicit), so this count fires whenever the parser
/// attached ``deps=[...]`` to the Call.
std::vector<OrchestrationStmtCodegen::DepEdge> OrchestrationStmtCodegen::GetDependencyEdges(
    const CallPtr& call) const {
  std::vector<DepEdge> merged;
  std::unordered_set<uint64_t> seen;
  auto append_edges = [&](const char* key, bool user_written) {
    for (const auto& [k, v] : call->attrs_) {
      if (k != key) continue;
      const auto* edges = std::any_cast<std::vector<VarPtr>>(&v);
      if (edges == nullptr) return;
      for (const auto& edge : *edges) {
        if (!edge) continue;
        if (!seen.insert(edge->UniqueId()).second) continue;
        merged.push_back(DepEdge{edge, user_written});
      }
      return;
    }
  };
  // Every ``manual_dep_edges`` carrier is treated as user-authored. The two
  // shapes that reach here are a ``pl.submit(..., deps=[...])`` (via
  // ``SubmitToCallView``) and a ``system.task_dummy`` barrier — and the
  // barrier is NOT reliably compiler-authored: the parser stamps the same
  // ``dummy_task`` attr on a user-written ``pl.system.task_dummy(deps=[...])``,
  // whose edges must be enforced like any other user edge.
  //
  // ``ExpandManualPhaseFence`` also synthesises barriers under this key, but
  // it only ever names a TaskId live in the same manual scope it rewrites, so
  // those always resolve. Were one not to, raising is still the right answer —
  // a dropped fanin there is a genuinely lost ordering edge.
  append_edges(kAttrManualDepEdges, /*user_written=*/true);
  append_edges(kAttrCompilerManualDepEdges, /*user_written=*/false);
  return merged;
}

/// Resolve ``edge`` to the live ``TaskId`` binding of its producer, or
/// ``nullptr`` when no binding is visible here — the producer sits in a
/// scope that has already closed, so its C++ local is gone and the edge is
/// dropped from the emitted ``set_dependencies`` call.
///
/// Dropping is benign for a *compiler-derived* edge: it is a best-effort
/// hazard patch and may legitimately name a TaskId produced inside a closed
/// scope. For a *user-written* ``deps=[...]`` edge it is not — the consumer
/// would be left unordered against its producer, which surfaces at runtime
/// as a silent stale read — so fail loudly instead of emitting wrong code.
///
/// ``CountManualDeps`` and ``EmitManualDeps`` both resolve through here, so
/// the dep-array sizing and the dep-array fill never disagree on which edges
/// survive.
const OrchestrationStmtCodegen::ManualTaskIdBinding* OrchestrationStmtCodegen::ResolveDepEdgeBinding(
    const DepEdge& edge, const CallPtr& call) const {
  if (!edge.var) return nullptr;
  const auto* binding = ResolveManualTaskIdBinding(edge.var.get());
  if (binding == nullptr) {
    if (edge.user_written) {
      // ``GetSSABaseName`` can itself strip to "" (see the loop-var note
      // below), so fall back on the stripped result, not the raw hint.
      std::string name = GetSSABaseName(edge.var->name_hint_);
      if (name.empty()) name = "<anonymous>";
      CHECK_SPAN(false, call->span_)
          << "Task dependency deps=[" << name << "] cannot be honored: its TaskId is produced inside a "
          << "scope that has already closed at this point, so the ordering edge would be lost and the "
          << "consumer could read stale data. Note that pl.range / pl.parallel loop bodies and if/else "
          << "branches each open their own scope, so this fires even with no pl.scope() in the source. "
          << "Either consume the TaskId inside the scope that produces it (for example, keep producer "
          << "and consumer in one pl.manual_scope()), or hoist the producer out of the inner scope. For "
          << "a TaskId produced in a loop body, consume it in that same body or accumulate the ids into "
          << "an Array[N, TASK_ID] and depend on that after the loop.";
    }
    return nullptr;
  }
  // Invariant: a dep edge never resolves directly to a kernel-Call LHS
  // (int-variant entry). The parser enforces that ``deps=[...]`` only
  // accepts ``Scalar[TASK_ID]`` Vars, so an edge always resolves to a TaskId
  // binding (string variant) or a TaskId iter_arg array (vector variant).
  INTERNAL_CHECK_SPAN(std::get_if<int>(binding) == nullptr, call->span_)
      << "Internal error: manual_dep_edge var '" << edge.var->name_hint_
      << "' resolves to a kernel-Call LHS (int variant). Expected "
      << "a Scalar[TASK_ID] Var (string variant).";
  return binding;
}

void OrchestrationStmtCodegen::CollectCompilerDepTaskIds(const ProgramPtr& program) {
  class Collector : public IRVisitor {
   public:
    std::unordered_set<const Var*> vars;
    std::unordered_set<std::string> loop_return_task_id_keys;

   protected:
    void VisitStmt_(const ForStmtPtr& for_stmt) override {
      for (const auto& return_var : for_stmt->return_vars_) {
        if (return_var) {
          loop_return_task_id_keys.insert(TaskIdKey(return_var.get()));
        }
      }
      IRVisitor::VisitStmt_(for_stmt);
    }

    void VisitExpr_(const CallPtr& call) override {
      CollectFromAttrs(call->attrs_);
      IRVisitor::VisitExpr_(call);
    }

    void VisitExpr_(const SubmitPtr& submit) override {
      CollectFromAttrs(submit->attrs_);
      IRVisitor::VisitExpr_(submit);
    }

   private:
    void CollectFromAttrs(const std::vector<std::pair<std::string, std::any>>& attrs) {
      for (const auto& [key, value] : attrs) {
        if (key != kAttrCompilerManualDepEdges) continue;
        const auto* edges = std::any_cast<std::vector<VarPtr>>(&value);
        if (!edges) continue;
        for (const auto& edge : *edges) {
          if (edge) vars.insert(edge.get());
        }
      }
    }

    static std::string TaskIdKey(const Var* var) {
      if (!var->name_hint_.empty()) return GetSSABaseName(var->name_hint_);
      return std::to_string(var->UniqueId());
    }
  };

  Collector collector;
  collector.VisitProgram(program);
  compiler_dep_task_id_vars_ = std::move(collector.vars);
  compiler_dep_loop_return_task_id_keys_ = std::move(collector.loop_return_task_id_keys);
  compiler_dep_task_id_keys_.clear();
  for (const Var* var : compiler_dep_task_id_vars_) {
    compiler_dep_task_id_keys_.insert(CompilerDepAliasKey(var));
  }
}

bool OrchestrationStmtCodegen::ShouldCaptureTaskOutputs(const CallPtr& call, bool capture_plain_task_id) {
  return IsSubmitCall(call) || capture_plain_task_id;
}

bool OrchestrationStmtCodegen::NeedsCompilerDepTaskId(const Var* var) const {
  if (!var) return false;
  if (compiler_dep_task_id_vars_.count(var) > 0) return true;
  return compiler_dep_task_id_keys_.count(CompilerDepAliasKey(var)) > 0;
}

std::vector<const Var*> OrchestrationStmtCodegen::CompilerDepOutputArgs(const CallPtr& call) const {
  std::vector<const Var*> vars;
  auto dirs = call->GetArgDirections();
  if (dirs.size() != call->args_.size()) return vars;
  std::unordered_set<uint64_t> seen;
  for (size_t i = 0; i < dirs.size(); ++i) {
    if (dirs[i] != ArgDirection::OutputExisting && dirs[i] != ArgDirection::InOut) continue;
    auto var = AsVarLike(call->args_[i]);
    if (!var || !NeedsCompilerDepTaskId(var.get())) continue;
    if (!seen.insert(TaskIdHoistKey(var.get())).second) continue;
    vars.push_back(var.get());
  }
  return vars;
}

size_t OrchestrationStmtCodegen::CountManualDeps(const std::vector<DepEdge>& edges,
                                                 const CallPtr& call) const {
  size_t total = 0;
  std::unordered_set<std::string> seen_names;
  for (const auto& edge : edges) {
    const auto* binding = ResolveDepEdgeBinding(edge, call);
    if (!binding) continue;
    if (auto* names = std::get_if<std::vector<std::string>>(binding)) {
      for (const auto& name : *names) {
        if (seen_names.insert(name).second) {
          total += 1;
        }
      }
    } else {
      const auto& name = std::get<std::string>(*binding);
      if (seen_names.insert(name).second) {
        total += 1;
      }
    }
  }
  return total;
}

/// Emit explicit dependency wiring for a kernel ``Call``: a fixed-size
/// ``TaskId <task>_deps[K]`` stack array filled with the valid producer
/// TaskIds from ``Call.attrs["manual_dep_edges"]``, followed by a single
/// ``<task>.set_dependencies(<task>_deps, <task>_deps_count)``.
///
/// The edge list is written directly by the parser from a ``pl.submit(...)``
/// ``deps=[tid1, tid2]`` kwarg — each entry a ``Scalar[TASK_ID]`` Var, or an
/// ``Array[N, TASK_ID]`` that contributes one slot each. Each entry is
/// emitted via ``EmitDepArrayInsert``: an entry that may hold the
/// ``TaskId::invalid()`` sentinel — a ``None`` loop-carry seed, an early
/// loop iteration's iter_arg carry, an unwritten array slot, or a hoisted /
/// dummy / barrier tid — is guarded by ``is_valid()`` so an invalid id never
/// reaches ``set_dependencies``. A fresh direct-producer id, proven always
/// valid, skips the guard.
///
/// Orthogonal to scope mode: ``set_dependencies`` adds explicit edges on top
/// of any auto-tracked deps the runtime infers from OverlapMap (final
/// fanin = auto ∪ explicit), so this fires in both auto and manual scopes.
/// No-op when there are no edges attached.
void OrchestrationStmtCodegen::EmitDepArrayInsert(const std::string& name, const std::string& deps_arr,
                                                  const std::string& deps_cnt) {
  const std::string assign = deps_arr + "[" + deps_cnt + "++] = " + name + ";";
  // A fresh direct-producer TaskId is statically valid (see
  // guaranteed_valid_task_ids_) so its insert skips the runtime is_valid()
  // guard; every other id keeps the guard because it may hold the
  // ``TaskId::invalid()`` sentinel.
  if (guaranteed_valid_task_ids_.count(name)) {
    EmitIndentedLine(assign);
  } else {
    EmitIndentedLine("if (" + name + ".is_valid()) " + assign);
  }
}

// The runtime primitive ``Arg::set_dependencies(ptr, count)`` has no upper
// bound on the explicit dep count, and codegen sizes each call's
// ``TaskId <task>_deps[K]`` stack array to its exact edge count, so there
// is no codegen-time cap on per-task explicit dependencies.

void OrchestrationStmtCodegen::EmitManualDeps(const CallPtr& call, const std::string& task_var) {
  const auto edges = GetDependencyEdges(call);
  const size_t dep_capacity = CountManualDeps(edges, call);
  if (dep_capacity == 0) return;
  const std::string deps_arr = task_var + "_deps";
  const std::string deps_cnt = task_var + "_deps_count";
  EmitIndentedLine("TaskId " + deps_arr + "[" + std::to_string(dep_capacity) + "];");
  EmitIndentedLine("uint32_t " + deps_cnt + " = 0;");
  std::unordered_set<std::string> emitted_names;
  auto emit_one_dep = [&](const std::string& name) {
    if (!emitted_names.insert(name).second) return;
    EmitDepArrayInsert(name, deps_arr, deps_cnt);
  };
  for (const auto& edge : edges) {
    const auto* binding = ResolveDepEdgeBinding(edge, call);
    if (!binding) {
      // A compiler-derived edge may name a TaskId produced inside a closed
      // ``pl.scope()`` that is no longer visible here. ``CountManualDeps``
      // already skips it when sizing the array, so emit must be consistent
      // and silently drop it too. (A *user* edge in this state threw above.)
      continue;
    }
    if (auto* names = std::get_if<std::vector<std::string>>(binding)) {
      // Array-carry iter_arg: include every valid slot.
      for (const auto& name : *names) {
        emit_one_dep(name);
      }
    } else {
      const auto& name = std::get<std::string>(*binding);
      // A scalar TaskId may hold the ``TaskId::invalid()`` sentinel — an
      // iter_arg carry on the first loop iteration, or a ``None``
      // (``system.task_invalid``) loop-carry seed. ``EmitDepArrayInsert``
      // guards those with ``is_valid()``, and elides the guard only for a
      // fresh direct-producer id proven always-valid.
      emit_one_dep(name);
    }
  }
  EmitIndentedLine(task_var + ".set_dependencies(" + deps_arr + ", " + deps_cnt + ");");
}

void OrchestrationStmtCodegen::EmitDummyTask(const CallPtr& call, const std::string& tid_name) {
  const int barrier_idx = phase_fence_barrier_counter_++;
  const std::string task_var = "params_phase_fence_barrier_" + std::to_string(barrier_idx);
  const std::string deps_cnt = task_var + "_deps_count";
  const std::string outs_var = "phase_fence_barrier_" + std::to_string(barrier_idx) + "_outs";
  const size_t dep_capacity = CountManualDeps(GetDependencyEdges(call), call);
  EmitBlankLine();
  EmitIndentedLine("// Phase-fence barrier " + std::to_string(barrier_idx) + ": dependency-only dummy task");

  EmitTaskParamsDecl(task_var);
  if (dep_capacity > 0) {
    // Dep-carrying barrier: deps are appended under per-edge is_valid()
    // guards, so the effective count is only known at runtime. Skip the
    // submit only when the barrier ends up fencing nothing (every dep
    // resolved to an invalid sentinel).
    EmitManualDeps(call, task_var);
    EmitIndentedLine("TaskId " + tid_name + " = TaskId::invalid();");
    EmitIndentedLine("if (" + deps_cnt + " > 0) {");
    {
      IndentGuard guard(Active());
      EmitIndentedLine("TaskOutputTensors " + outs_var + " = rt_submit_dummy_task(" + task_var + ");");
      EmitIndentedLine(tid_name + " = " + outs_var + ".task_id();");
    }
    EmitIndentedLine("}");
  } else {
    // Statically empty-deps barrier: a user-written task_dummy(deps=[]).
    // (ExpandManualPhaseFence only inserts barriers when profitable, never
    // empty, so this path is exclusively user-originated.) Such a dummy has
    // no producers and is ready immediately, but it must still materialize
    // as a real task with a valid id that is added to each consumer's fanin.
    // Having no predecessors affects neither its submission nor the edges to
    // its successors, so submit it unconditionally — eliding it would
    // silently drop those edges.
    EmitIndentedLine("TaskOutputTensors " + outs_var + " = rt_submit_dummy_task(" + task_var + ");");
    EmitIndentedLine("TaskId " + tid_name + " = " + outs_var + ".task_id();");
  }
}

void OrchestrationStmtCodegen::DeclareHoistedTaskIdsForScope(const RuntimeScopeStmt* scope) {
  auto it = hoisted_task_id_vars_by_scope_.find(scope);
  if (it == hoisted_task_id_vars_by_scope_.end()) return;
  for (const Var* var : it->second) {
    DeclareHoistedTaskId(var);
  }
}

void OrchestrationStmtCodegen::DeclareCompilerDepTaskIdsAtFirstScope() {
  if (declared_compiler_dep_task_ids_) return;
  declared_compiler_dep_task_ids_ = true;
  for (const Var* var : compiler_dep_task_id_vars_) {
    if (compiler_dep_loop_return_task_id_keys_.count(CompilerDepAliasKey(var)) > 0) continue;
    DeclareHoistedTaskId(var);
  }
}

void OrchestrationStmtCodegen::DeclareHoistedTaskId(const Var* var) {
  if (!var || hoisted_task_id_emit_names_by_key_.count(TaskIdHoistKey(var))) return;
  std::string name = ReserveSyntheticEmitName(GetSSABaseName(var->name_hint_) + "_tid");
  EmitIndentedLine("TaskId " + name + " = TaskId::invalid();");

  hoisted_task_id_emit_names_by_key_[TaskIdHoistKey(var)] = name;
  hoisted_task_id_emit_names_[var->UniqueId()] = name;
  manual_task_id_map_[var] = name;
  manual_task_id_map_by_key_[TaskIdHoistKey(var)] = name;
}

uint64_t OrchestrationStmtCodegen::TaskIdHoistKey(const Var* var) { return var->UniqueId(); }

std::string OrchestrationStmtCodegen::CompilerDepAliasKey(const Var* var) {
  if (!var->name_hint_.empty()) return GetSSABaseName(var->name_hint_);
  return std::to_string(var->UniqueId());
}

const OrchestrationStmtCodegen::ManualTaskIdBinding* OrchestrationStmtCodegen::ResolveManualTaskIdBinding(
    const Var* var) const {
  if (!var) return nullptr;
  auto direct = manual_task_id_map_.find(var);
  if (direct != manual_task_id_map_.end()) return &direct->second;
  auto by_key = manual_task_id_map_by_key_.find(TaskIdHoistKey(var));
  if (by_key == manual_task_id_map_by_key_.end()) return nullptr;
  return &by_key->second;
}

std::string OrchestrationStmtCodegen::BindSyntheticTaskId(const Var* var, const std::string& base_name,
                                                          const std::string& value_expr) {
  auto hoisted_by_key = var ? hoisted_task_id_emit_names_by_key_.find(TaskIdHoistKey(var))
                            : hoisted_task_id_emit_names_by_key_.end();
  if (hoisted_by_key != hoisted_task_id_emit_names_by_key_.end()) {
    EmitIndentedLine(hoisted_by_key->second + " = " + value_expr + ";");

    return hoisted_by_key->second;
  }
  auto hoisted_it =
      var ? hoisted_task_id_emit_names_.find(var->UniqueId()) : hoisted_task_id_emit_names_.end();
  if (hoisted_it != hoisted_task_id_emit_names_.end()) {
    EmitIndentedLine(hoisted_it->second + " = " + value_expr + ";");

    return hoisted_it->second;
  }
  std::string tid_name = ReserveSyntheticEmitName(base_name);
  EmitIndentedLine("TaskId " + tid_name + " = " + value_expr + ";");
  // Fresh producer local: every caller binds a ``task_<n>_outs.task_id()``
  // producer id, which the runtime guarantees valid. Record so its dep-array
  // insert can skip the redundant is_valid() guard. The hoisted branches
  // above deliberately do NOT record: those names are pre-declared
  // ``= TaskId::invalid()`` loop carries that may be invalid at a read.
  guaranteed_valid_task_ids_.insert(tid_name);

  return tid_name;
}

std::string OrchestrationStmtCodegen::BindVarTaskId(const Var* var, const std::string& value_expr) {
  auto hoisted_by_key = var ? hoisted_task_id_emit_names_by_key_.find(TaskIdHoistKey(var))
                            : hoisted_task_id_emit_names_by_key_.end();
  if (hoisted_by_key != hoisted_task_id_emit_names_by_key_.end()) {
    EmitIndentedLine(hoisted_by_key->second + " = " + value_expr + ";");

    emit_name_map_[var] = hoisted_by_key->second;
    return hoisted_by_key->second;
  }
  auto hoisted_it =
      var ? hoisted_task_id_emit_names_.find(var->UniqueId()) : hoisted_task_id_emit_names_.end();
  if (hoisted_it != hoisted_task_id_emit_names_.end()) {
    EmitIndentedLine(hoisted_it->second + " = " + value_expr + ";");

    emit_name_map_[var] = hoisted_it->second;
    return hoisted_it->second;
  }
  std::string tid_name = ReserveVarEmitName(var);
  EmitIndentedLine("TaskId " + tid_name + " = " + value_expr + ";");
  // Fresh producer local (see BindSyntheticTaskId): the sole caller binds a
  // ``task_<n>_outs.task_id()`` producer id, always valid. The hoisted
  // branches above are left unrecorded for the same reason.
  guaranteed_valid_task_ids_.insert(tid_name);

  return tid_name;
}

/// Return the TaskId producer local if ``value`` names one from a closed
/// scope. Both array stores and yields must reject these reads before the
/// host compiler encounters an out-of-scope C++ identifier.
///
/// A stale value is identified positively: ``emit_name_map_`` still holds the
/// name it was bound to (never scope-restored), while ``manual_task_id_map_``
/// does not (restored at each closing brace — see
/// ``VisitStmt_(RuntimeScopeStmtPtr)``). Function ``Scalar[TASK_ID]``
/// parameters are registered at construction (``RegisterFunctionTaskIdParams``)
/// so yielding / storing them is not mistaken for a closed-scope producer. A
/// TaskId this codegen never bound to a local has no entry in either map and
/// is left alone.
VarPtr OrchestrationStmtCodegen::FindClosedScopeTaskId(const ExprPtr& value) const {
  auto value_var = AsVarLike(value);
  if (!value_var) return nullptr;
  auto scalar_ty = As<ScalarType>(value_var->GetType());
  if (!scalar_ty || scalar_ty->dtype_ != DataType::TASK_ID) return nullptr;
  // Resolve through pointer and UniqueId maps (both scope-restored together).
  // A live sequential / phi carry or function parameter stays registered; a
  // producer local from a closed scope does not (issue #2677 / #2659).
  if (ResolveManualTaskIdBinding(value_var.get())) return nullptr;
  auto emit_it = emit_name_map_.find(value_var.get());
  if (emit_it == emit_name_map_.end()) return nullptr;
  // Emit-name alias of a still-registered live carry (SSA rename after inline).
  if (live_manual_task_id_emit_names_.count(emit_it->second)) return nullptr;
  return value_var;
}

void OrchestrationStmtCodegen::CheckTaskIdSlotValueInScope(const CallPtr& call,
                                                           const std::string& array_name) {
  auto value_var = FindClosedScopeTaskId(call->args_[2]);
  if (!value_var) return;
  CHECK_SPAN(false, call->span_)
      << "Task id '" << value_var->name_hint_ << "' is stored into array '" << array_name
      << "' after the `pl.scope()` that produced it has closed, so the store has no runtime "
         "value to name. Move the store inside that `pl.scope()`, where the task id is still "
         "live.";
}

void OrchestrationStmtCodegen::HandleArrayUpdateElementAssign(const AssignStmtPtr& assign,
                                                              const CallPtr& call) {
  // array.update_element(array, index, value) -> ArrayType.
  // The SSA-functional return value shares storage with the first arg; alias
  // the LHS so subsequent references resolve to the same C variable.
  INTERNAL_CHECK_SPAN(call->args_.size() == 3, call->span_)
      << "Internal error: array.update_element expects 3 arguments";
  std::string array_name = GenerateExprString(call->args_[0]);
  CheckTaskIdSlotValueInScope(call, array_name);
  emit_name_map_[assign->var_.get()] = array_name;
  // Propagate ``array_carry_vars_`` and ``manual_task_id_map_`` from the
  // input array to the LHS: they share storage, so a downstream
  // ``deps=[lhs]`` or array-yield treats them identically. Without this,
  // a parallel iter_arg yielding the post-update Array would fall through
  // ``array_carry_vars_.find(yield_var)`` and trip the scalar-yield branch.
  if (auto input_var = AsVarLike(call->args_[0])) {
    auto carry_it = array_carry_vars_.find(input_var.get());
    if (carry_it != array_carry_vars_.end()) {
      array_carry_vars_[assign->var_.get()] = carry_it->second;
    }
    auto tid_it = manual_task_id_map_.find(input_var.get());
    if (tid_it != manual_task_id_map_.end()) {
      manual_task_id_map_[assign->var_.get()] = tid_it->second;
    }
  }
}

/// Rebuild the O(1) reverse index of live scalar TaskId emit names from the
/// current pointer / UniqueId maps (call after every scope restore).
void OrchestrationStmtCodegen::RebuildLiveManualTaskIdEmitNames() {
  live_manual_task_id_emit_names_.clear();
  auto note = [&](const ManualTaskIdBinding& binding) {
    if (const auto* name = std::get_if<std::string>(&binding)) {
      live_manual_task_id_emit_names_.insert(*name);
    }
  };
  for (const auto& [_, binding] : manual_task_id_map_) note(binding);
  for (const auto& [_, binding] : manual_task_id_map_by_key_) note(binding);
}

/// Note a scalar TaskId emit name as live (kept in sync with map inserts).
void OrchestrationStmtCodegen::NoteLiveManualTaskIdEmitName(const std::string& name) {
  live_manual_task_id_emit_names_.insert(name);
}

/// Bind every ``Scalar[TASK_ID]`` already present in ``emit_name_map_`` (function
/// parameters and their lineage aliases seeded before construction) as a live
/// TaskId for the whole orchestration / Graph body.
void OrchestrationStmtCodegen::RegisterFunctionTaskIdParams() {
  for (const auto& [var, emit_name] : emit_name_map_) {
    if (!var) continue;
    auto scalar_ty = As<ScalarType>(var->GetType());
    if (!scalar_ty || scalar_ty->dtype_ != DataType::TASK_ID) continue;
    manual_task_id_map_[var] = emit_name;
    manual_task_id_map_by_key_[TaskIdHoistKey(var)] = emit_name;
    NoteLiveManualTaskIdEmitName(emit_name);
  }
}

/// Register ``var`` as backed by ``array_name[size]``; also populates the
/// ``manual_task_id_map_`` with the per-slot expressions so EmitManualDeps
/// emits one ``add_dep`` per slot when this Var appears as a deps source.
void OrchestrationStmtCodegen::RegisterArrayCarry(const Var* var, const std::string& array_name,
                                                  int64_t size) {
  array_carry_vars_[var] = ArrayCarryEntry{array_name, size};
  std::vector<std::string> slot_names;
  slot_names.reserve(static_cast<size_t>(size));
  for (int64_t i = 0; i < size; ++i) {
    slot_names.push_back(array_name + "[" + std::to_string(i) + "]");
  }
  manual_task_id_map_[var] = std::move(slot_names);
}

}  // namespace codegen
}  // namespace pypto
