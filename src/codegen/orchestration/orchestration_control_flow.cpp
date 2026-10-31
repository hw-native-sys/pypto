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

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "pypto/codegen/code_emitter.h"
#include "pypto/codegen/orchestration/orchestration_analysis.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"
#include "src/codegen/orchestration/orchestration_stmt_codegen.h"

namespace pypto {
namespace codegen {

using namespace pypto::ir;  // NOLINT(build/namespaces)

namespace {

/// Per-iter_arg carry lowering plan consumed by the ForStmt emitter.
///
/// ``is_rebind`` / ``array_size`` are read straight off ``ForStmt::attrs_``
/// (stamped by the ``ClassifyIterArgCarry`` pass). The two compiler-dep flags
/// are a codegen-local overlay: they depend on the program-wide compiler-derived
/// dependency edges collected by ``CollectCompilerDepTaskIds``, not on the loop's
/// own structure.
struct IterArgCarryPlan {
  /// True when the carry needs a materialised mutable variable (vs. a trivial
  /// alias to the init value's emit name).
  bool is_rebind = false;
  /// TaskId manual-scope array-carry extent; 0 means scalar/tensor/ArrayType path.
  int64_t array_size = 0;
  /// True when this iter_arg collects compiler-derived task dependencies
  /// (NeedsCompilerDepTaskId). The carry is initialised with
  /// TaskId::invalid() and filled by yielded producer TaskIds.
  bool compiler_dep_collection = false;
  /// True when compiler-dep collection needs a dynamic (vector) backing store
  /// because the ForStmt trip count is not a compile-time constant.
  bool dynamic_compiler_dep_collection = false;
};

}  // namespace

void OrchestrationStmtCodegen::VisitStmt_(const ForStmtPtr& for_stmt) {
  INTERNAL_CHECK_SPAN(for_stmt->kind_ != ForKind::Unroll, for_stmt->span_)
      << "Internal error: ForKind::Unroll reached codegen — UnrollLoops "
      << "should have resolved it. The pipeline is incomplete.";
  INTERNAL_CHECK_SPAN(for_stmt->kind_ != ForKind::Pipeline, for_stmt->span_)
      << "Internal error: ForKind::Pipeline reached codegen — LowerPipelineLoops "
      << "and CanonicalizeIOOrder should have demoted it to Sequential. "
      << "The pipeline is incomplete.";

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

  // ForStmt visitor pipeline: read the iter-arg carry plan stamped by
  // ClassifyIterArgCarry -> post-process compiler-derived deps -> emit carry
  // declarations -> emit loop body.
  std::vector<IterArgCarryPlan> carry_plans(for_stmt->iter_args_.size());
  for (size_t i = 0; i < carry_plans.size(); ++i) {
    carry_plans[i].is_rebind = transform_utils::IterArgIsRebind(for_stmt, i);
    carry_plans[i].array_size = transform_utils::IterArgArraySize(for_stmt, i);
  }

  // Post-process: compiler-derived dep collections use NeedsCompilerDepTaskId,
  // which lives on the codegen class (the pass classifies carries from the
  // loop's own structure, not from program-wide compiler-dep edges). Mark
  // these iter_args as needing compiler-dep collection and set the array-carry
  // size from the const trip count. Carries without a const trip count on
  // Parallel loops defer to a dynamic (vector) collection.
  for (size_t i = 0; i < carry_plans.size(); ++i) {
    if (i < for_stmt->return_vars_.size() && NeedsCompilerDepTaskId(for_stmt->return_vars_[i].get())) {
      // Always size compiler-dep carries from the outer loop's const trip
      // count.  ResolveArrayCarrySize may have already set array_size to an
      // inner Parallel loop's trip count (a Sequential outer wrapping a
      // Parallel inner that also carries task ids), which would mis-size the
      // carry array when the two trip counts differ.  The outer loop's trip
      // count is authoritative for the fan-in array that collects all
      // producer TaskIds across iterations (YunjiQin review, PR #1813).
      carry_plans[i].array_size = EvalConstTripCount(for_stmt);
      carry_plans[i].compiler_dep_collection = true;
      // compiler_dep_collection is functionally only consumed inside the
      // array_size > 0 branch below (for TaskId::invalid() init).
      // When array_size == 0 the dynamic path handles collection instead.
      if (carry_plans[i].array_size <= 0 && for_stmt->kind_ == ForKind::Parallel) {
        carry_plans[i].dynamic_compiler_dep_collection = true;
      }
      // Override is_rebind: ClassifyIterArgCarry only sets it for TASK_ID
      // iter_args, but Tensor-typed iter_args with compiler-dep edges also
      // need true so the yield handler emits dynamic-collection writes.
      carry_plans[i].is_rebind = true;
    }
  }

  // Count dynamic compiler-dep slots per iteration (for vector sizing).
  size_t dynamic_compiler_dep_slots_per_iter = 0;
  for (const auto& plan : carry_plans) {
    if (plan.dynamic_compiler_dep_collection) ++dynamic_compiler_dep_slots_per_iter;
  }
  std::optional<DynamicTaskIdCollection> dynamic_compiler_dep_collection_info;

  struct TensorCarryValues {
    VarPtr input;
    std::string name;
    size_t init;
    size_t entry;
    size_t exit;
  };
  std::vector<TensorCarryValues> tensor_carries;

  // Emit carry declarations for each iter_arg. Three lowering paths:
  //   - array_size > 0  -> TaskId array-carry (TaskId arr[N])
  //   - ArrayType carry  -> C-stack array with in-place-update semantics
  //   - is_rebind        -> scalar/Tensor mutable carry variable
  //   - else             -> trivial alias to init's emit name
  for (size_t i = 0; i < for_stmt->iter_args_.size(); ++i) {
    const auto& iter_arg = for_stmt->iter_args_[i];
    const auto& return_var = for_stmt->return_vars_[i];
    const bool is_rebind = carry_plans[i].is_rebind;
    const int64_t array_size = carry_plans[i].array_size;
    // An init value is usually an SSA Var, but a scalar carry seeded by a
    // constant reaches codegen as a bare expression: Simplify propagates
    // ``acc: pl.Scalar[pl.INT64] = 0`` into the iter_arg, and
    // ``pl.range(..., init_values=(0,))`` writes the literal directly. Both
    // are legal IR, so emit the init expression instead of demanding an
    // identifier. Only the ArrayType copy-in below genuinely needs a name.
    ValidateValueUse(iter_arg->initValue_, for_stmt->span_);
    const std::string init_emit_name = TryGetVarName(iter_arg->initValue_);
    const bool init_is_var = !init_emit_name.empty();
    // Function tensor params get rewritten to `ext_<name>` in the emitted C++,
    // so the bare emit name is not a valid identifier when the init value is
    // a param. Apply the same translation as everything else that names a
    // tensor in the emitted code.
    const std::string init_var_name = init_is_var && !As<ScalarType>(iter_arg->initValue_->GetType())
                                          ? GetExternalTensorName(init_emit_name)
                                          : GenerateExprString(iter_arg->initValue_);

    if (array_size > 0) {
      // ARRAY CARRY PATH — allocate ``TaskId <name>[N]`` and init it.
      // Initialisation rule: if the iter_arg's init value is itself an
      // array-carry Var, copy slot-by-slot; otherwise broadcast the scalar
      // init expression to every slot.
      const int64_t N = array_size;
      std::string rv_array_name = ReserveSyntheticEmitName(return_var->name_hint_);
      EmitIndentedLine("TaskId " + rv_array_name + "[" + std::to_string(N) + "];");

      auto outer_init_var = AsVarLike(iter_arg->initValue_);
      const ArrayCarryEntry* outer_init_arr = nullptr;
      if (outer_init_var) {
        auto outer_it = array_carry_vars_.find(outer_init_var.get());
        if (outer_it != array_carry_vars_.end()) outer_init_arr = &outer_it->second;
      }
      if (outer_init_arr && outer_init_arr->size == N) {
        EmitArrayCopyLoop(N, rv_array_name, outer_init_arr->array_name, "__init_i");
      } else if (carry_plans[i].compiler_dep_collection) {
        EmitArrayFillLoop(N, rv_array_name, "TaskId::invalid()", "__init_i");
      } else {
        EmitArrayFillLoop(N, rv_array_name, init_var_name, "__init_i");
      }
      // Register rv as array-carry (yields target into this array).
      RegisterArrayCarry(return_var.get(), rv_array_name, N);
      if (for_stmt->kind_ == ForKind::Parallel) {
        // Parallel iter_arg: per-iter "value" is the init (same for all iters)
        // — used as the deps source. If init is an array, alias to it; if
        // init is scalar, register the iter_arg's slot map to that scalar
        // broadcast (so EmitManualDeps emits add_dep on the same scalar).
        if (carry_plans[i].compiler_dep_collection) {
          // Compiler-derived fan-in collections collect producer TaskIds
          // yielded by the loop body. The tensor or None init value is not
          // itself a dependency source for this collection.
        } else if (outer_init_arr && outer_init_arr->size == N) {
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
    } else if (auto array_ty = As<ArrayType>(iter_arg->GetType()); array_ty && is_rebind) {
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
      const bool is_task_id_array = (array_ty->dtype_ == DataType::TASK_ID);
      const std::string cpp_dtype = is_task_id_array ? "TaskId" : array_ty->dtype_.ToCTypeString();

      // An ArrayType carry is in-place-update semantics: the body's
      // ``array.update_element`` aliases its LHS back to the carry's emit
      // name, so every SSA rename of the same logical array can share one
      // C-stack array. When the iter_arg's init value is itself a backing
      // array (an ``array.create`` result or an enclosing loop's ArrayType
      // carry), reuse it directly — no fresh declaration and no
      // slot-by-slot copy-in are needed, and the matching yield self-copy
      // is skipped in VisitStmt_(YieldStmtPtr). Otherwise (e.g. an
      // ArrayType function parameter) fall back to a fresh backing array
      // initialised by a slot-by-slot copy.
      std::string carry_name;
      auto init_var = AsVarLike(iter_arg->initValue_);
      const ArrayCarryEntry* init_carry = nullptr;
      if (init_var) {
        auto it = array_carry_vars_.find(init_var.get());
        if (it != array_carry_vars_.end() && it->second.size == N) init_carry = &it->second;
      }
      if (init_carry) {
        carry_name = init_carry->array_name;
      } else {
        // The copy-in indexes the init slot-by-slot (``src[i]``), so this is
        // the one carry path that needs the init to name a backing array
        // rather than be an arbitrary expression.
        INTERNAL_CHECK_SPAN(init_is_var, for_stmt->span_)
            << "Internal error: ArrayType iter_arg init value must be a variable naming a backing "
            << "array, got " << iter_arg->initValue_->TypeName();
        carry_name = ReserveSyntheticEmitName(return_var->name_hint_);
        EmitIndentedLine(cpp_dtype + " " + carry_name + "[" + std::to_string(N) + "];");

        EmitArrayCopyLoop(N, carry_name, init_var_name, "__init_i");
      }
      emit_name_map_[iter_arg.get()] = carry_name;
      emit_name_map_[return_var.get()] = carry_name;
      // Register the iter_arg / return_var as array carries so a nested
      // ForStmt seeded by either can reuse this same backing array, and a
      // downstream ``deps=[arr]`` (TaskId arrays) expands into N guarded
      // per-slot dependency fills.
      RegisterArrayCarry(iter_arg.get(), carry_name, N);
      RegisterArrayCarry(return_var.get(), carry_name, N);
    } else if (is_rebind) {
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
      const std::string cpp_type = GetCppType(return_var->GetType());
      // A Tensor loop carry directly in a ``pl.manual_scope`` body is hoisted to
      // the enclosing scope so a task / method-receiver placed AFTER the block
      // resolves it (issue #1713; see EmitMutableTensorCarryDecl). Non-Tensor
      // (e.g. Sequential TaskId scalar) carries keep their in-block decl.
      if (cpp_type == "Tensor") {
        EmitMutableTensorCarryDecl(carry_name, init_var_name);
        // The C++ carry has one mutable name, but its iteration input and
        // loop result are distinct values. Copies in the body capture the
        // input; a later yield must not change those copies retroactively.
        const size_t entry = GetTensorValue(carry_name);
        const size_t exit = NewTensorValue();
        tensor_yield_values_[return_var.get()] = exit;
        if (auto_scope_carry_values_.count(entry)) auto_scope_carry_values_.insert(exit);
        tensor_carries.push_back({iter_arg, carry_name, GetTensorValue(init_var_name), entry, exit});
      } else {
        EmitIndentedLine(cpp_type + " " + carry_name + " = " + init_var_name + ";");

        RegisterMutableTensorName(cpp_type, carry_name);
      }
      emit_name_map_[return_var.get()] = carry_name;
      emit_name_map_[iter_arg.get()] = carry_name;
      // Sequential TaskId carry: register both endpoints in the task-id
      // map so EmitManualDeps / yield writes / CheckTaskIdSlotValueInScope
      // can find the live carry name. Must not gate on manual_scope depth:
      // AUTO-scope loops (and inlined callees that re-publish the incoming
      // carry into a pl.array) still need the binding, otherwise the
      // closed-scope store diagnostic false-positives on a legal loop-carried
      // TaskId (issue #2677).
      auto sty = As<ScalarType>(iter_arg->GetType());
      if (sty && sty->dtype_ == DataType::TASK_ID) {
        manual_task_id_map_[iter_arg.get()] = carry_name;
        manual_task_id_map_[return_var.get()] = carry_name;
        manual_task_id_map_by_key_[TaskIdHoistKey(iter_arg.get())] = carry_name;
        manual_task_id_map_by_key_[TaskIdHoistKey(return_var.get())] = carry_name;
        NoteLiveManualTaskIdEmitName(carry_name);
      }
    } else {
      // Trivial yield: preserve the legacy aliasing — both names route to
      // the init's emit name. YieldStmt for this slot will be a self-assign
      // and skipped (see VisitStmt_(YieldStmtPtr) for the equality guard).
      emit_name_map_[iter_arg.get()] = init_var_name;
      emit_name_map_[return_var.get()] = init_var_name;
    }
    if (carry_plans[i].dynamic_compiler_dep_collection) {
      if (!dynamic_compiler_dep_collection_info) {
        DynamicTaskIdCollection collection;
        collection.data_name = ReserveSyntheticEmitName("dynamic_compiler_dep_tids");
        collection.count_name = ReserveSyntheticEmitName(collection.data_name + "_count");
        const std::string capacity_name = ReserveSyntheticEmitName(collection.data_name + "_capacity");
        // Bind the bounds to locals first: each appears several times in the
        // trip-count expression, and the loop may descend, so the emitted
        // formula needs both branches. Mirrors
        // ``transform_utils::ComputeStaticTripCount`` — keep the two in sync.
        const std::string cap_start = ReserveSyntheticEmitName(capacity_name + "_start");
        const std::string cap_stop = ReserveSyntheticEmitName(capacity_name + "_stop");
        const std::string cap_step = ReserveSyntheticEmitName(capacity_name + "_step");
        EmitIndentedLine("const int64_t " + cap_start + " = " + start_expr + ";");
        EmitIndentedLine("const int64_t " + cap_stop + " = " + stop_expr + ";");
        EmitIndentedLine("const int64_t " + cap_step + " = " + step_expr + ";");
        EmitIndentedLine("const int64_t " + capacity_name + " = ((" + cap_step + " > 0 && " + cap_start +
                         " < " + cap_stop + ") ? ((" + cap_stop + " - " + cap_start + " + " + cap_step +
                         " - 1) / " + cap_step + ") : (" + cap_step + " < 0 && " + cap_start + " > " +
                         cap_stop + ") ? ((" + cap_start + " - " + cap_stop + " + (-" + cap_step +
                         ") - 1) / (-" + cap_step + ")) : 0) * " +
                         std::to_string(dynamic_compiler_dep_slots_per_iter) + ";");
        const std::string profile_start_name =
            ReserveSyntheticEmitName(collection.data_name + "_profile_start");
        EmitIndentedLine("#if SIMPLER_ORCH_PROFILING");
        EmitIndentedLine("uint64_t " + profile_start_name + " = rt_orch_profile_now();");
        EmitIndentedLine("#endif");
        EmitIndentedLine("std::vector<TaskId> " + collection.data_name + "(static_cast<size_t>(" +
                         capacity_name + "));");
        EmitIndentedLine("uint32_t " + collection.count_name + " = 0;");
        EmitIndentedLine("#if SIMPLER_ORCH_PROFILING");
        EmitIndentedLine("rt_orch_profile_add_dynamic_dep_vector(rt_orch_profile_now() - " +
                         profile_start_name + ", 0);");
        EmitIndentedLine("#endif");
        needs_vector_include_ = true;
        dynamic_compiler_dep_collection_info = collection;
      }
      dynamic_task_id_collections_[return_var.get()] = *dynamic_compiler_dep_collection_info;
    }
  }

  for (const Var* edge : CollectCompilerDepArrayBarrierEdges(for_stmt)) {
    const auto* binding = ResolveManualTaskIdBinding(edge);
    if (!binding) continue;
    const auto* names = std::get_if<std::vector<std::string>>(binding);
    if (!names || names->empty()) continue;
    const std::string barrier_tid = EmitCompilerDepArrayBarrier(*names);
    manual_task_id_map_[edge] = barrier_tid;
    manual_task_id_map_by_key_[TaskIdHoistKey(edge)] = barrier_tid;
  }

  EmitForLoopHeader(loop_var, start_expr, stop_expr, step_expr, for_stmt->step_);
  {
    IndentGuard indent_guard(Active());
    PushCppScope();

    // The implicit ``SIMPLER_SCOPE()`` wrapper around the loop body is now an
    // explicit AUTO RuntimeScopeStmt inserted by MaterializeRuntimeScopes
    // (suppressed inside a manual scope); visiting the body emits it 1:1.

    auto saved = current_return_vars_;
    // Only register return_vars whose yield is a true rebind. Trivial slots
    // are aliased to the init name and don't need a yield-time assignment;
    // emitting one would just produce `init = init;`.
    current_return_vars_.clear();
    for (size_t i = 0; i < for_stmt->return_vars_.size(); ++i) {
      if (carry_plans[i].is_rebind) {
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

    PopCppScope();
  }
  EmitIndentedLine("}");

  const auto trip_count = transform_utils::EvalConstTripCount(for_stmt);
  for (const auto& carry : tensor_carries) {
    if (!trip_count || *trip_count > 1) RecordTensorSource(carry.entry, carry.exit);
    // A zero-trip loop returns its initializer. Unknown trip counts merge
    // that path with the yielded result; a known nonempty loop does not.
    if (!trip_count) RecordTensorSource(carry.exit, carry.init);
  }
  for (const auto& carry : tensor_carries) {
    // Backedges can expose a closed allocation to the next iteration even
    // when the first iteration's read was valid. Recheck only inputs that
    // the body actually read, before switching names to their exit values.
    if ((!trip_count || *trip_count > 1) && tensor_read_values_.count(carry.entry) &&
        auto_scope_carry_values_.count(carry.entry)) {
      ValidateTensorStorage(carry.entry, carry.input, for_stmt->span_);
    }
    tensor_values_[carry.name] = trip_count && *trip_count == 0 ? carry.init : carry.exit;
  }

  std::unordered_map<std::string, std::string> dynamic_barrier_tids;
  for (size_t i = 0; i < for_stmt->return_vars_.size() && i < carry_plans.size(); ++i) {
    if (!carry_plans[i].dynamic_compiler_dep_collection) continue;
    const auto& return_var = for_stmt->return_vars_[i];
    if (!return_var) continue;
    auto vec_it = dynamic_task_id_collections_.find(return_var.get());
    if (vec_it == dynamic_task_id_collections_.end()) continue;
    auto [barrier_it, inserted] = dynamic_barrier_tids.emplace(vec_it->second.data_name, std::string());
    if (inserted) {
      barrier_it->second = EmitDynamicCompilerDepBarrier(vec_it->second);
    }
    const std::string& barrier_tid = barrier_it->second;
    manual_task_id_map_[return_var.get()] = barrier_tid;
    manual_task_id_map_by_key_[TaskIdHoistKey(return_var.get())] = barrier_tid;
  }
}

void OrchestrationStmtCodegen::VisitStmt_(const RuntimeScopeStmtPtr& scope) {
  DeclareCompilerDepTaskIdsAtFirstScope();
  DeclareHoistedTaskIdsForScope(scope.get());
  // Snapshot the TaskId / array-carry bindings on entry to EVERY generated
  // SIMPLER_SCOPE (AUTO or MANUAL); restore on exit. A binding added inside the
  // block names a C++ local (e.g. ``TaskId tid = ...``, ``TaskId prev
  // = arr[k]``) declared inside the generated ``{ }`` — it dies at the closing
  // brace, so it must not leak to the enclosing scope where a later
  // ``set_dependencies`` / array-yield would reference a now-out-of-scope
  // identifier (issue #1577). The snapshot is by COPY so outer entries stay
  // visible inside the block: a task in the scope can still fence on TaskIds
  // produced by tasks emitted in the enclosing (auto or outer) scope. Loop /
  // branch carries are registered *before* their body's SIMPLER_SCOPE (outside
  // this frame), so they correctly survive the block.
  //
  // ``manual_task_id_map_by_key_`` is snapshotted with the pointer map: UniqueId
  // entries added inside the block must die with it, or
  // ``ResolveManualTaskIdBinding`` / ``CheckTaskIdSlotValueInScope`` would treat
  // a dead loop-carry local as still live after the closing brace.
  auto saved_map = manual_task_id_map_;
  auto saved_map_by_key = manual_task_id_map_by_key_;
  auto saved_array_carry = array_carry_vars_;

  // Runtime scopes are transparent to Tensor SSA values, but introduce a
  // C++ block. Hoist mutable Tensor carries/phis whose initializers are
  // enclosing-scope-valid in both AUTO and MANUAL scopes. In particular, an
  // explicit AUTO scope can be followed by its enclosing loop's yield.
  // Buffer allocations retain AUTO scope placement; only MANUAL scopes move
  // allocations as well as handles to their enclosing scheduling region.
  const int parent_indent_level = Active().GetIndentLevel();
  const std::string parent_indent = IndentAtLevel(parent_indent_level);
  std::vector<std::string>* saved_sink = scope_hoist_sink_;
  const int saved_hoist_indent_level = scope_hoist_indent_level_;
  const bool saved_hoist_allocations = scope_hoist_allocations_;
  std::set<std::string>* saved_local_names = scope_local_names_;
  std::set<std::string>* saved_enclosing_local_names = enclosing_scope_local_names_;
  std::vector<std::string> hoisted;
  std::set<std::string> local_names;
  scope_hoist_sink_ = &hoisted;
  scope_hoist_allocations_ = scope->manual_;
  scope_hoist_indent_level_ = parent_indent_level;
  // Record hoisted names in the enclosing runtime scope: moving a declaration
  // out of this block does not make it visible outside its parent block.
  enclosing_scope_local_names_ = scope_local_names_;
  scope_local_names_ = &local_names;
  tensor_hoist_frames_.push_back({scope.get(),
                                  &hoisted,
                                  &local_names,
                                  enclosing_scope_local_names_,
                                  parent_indent_level,
                                  mutable_tensor_name_scopes_.size() - 1,
                                  {}});

  CodeEmitter body_emitter;
  body_emitter.SetIndentLevel(parent_indent_level);
  CodeEmitter* saved_active = active_emitter_;
  active_emitter_ = &body_emitter;
  IndentGuard body_indent(Active());
  PushCppScope();
  if (scope->manual_) ++in_manual_scope_depth_;
  VisitStmt(scope->body_);
  if (scope->manual_) --in_manual_scope_depth_;
  for (const auto value : tensor_hoist_frames_.back().allocations) {
    PropagateTensorFlag(value, &invalid_tensor_storage_);
  }
  tensor_hoist_frames_.pop_back();
  PopCppScope();
  active_emitter_ = saved_active;

  scope_hoist_sink_ = saved_sink;
  scope_hoist_allocations_ = saved_hoist_allocations;
  scope_hoist_indent_level_ = saved_hoist_indent_level;
  scope_local_names_ = saved_local_names;
  enclosing_scope_local_names_ = saved_enclosing_local_names;

  for (const auto& line : hoisted) {
    Active().AppendRaw(line);
  }
  Active().AppendRaw(parent_indent +
                     (scope->manual_ ? "SIMPLER_SCOPE(ScopeMode::MANUAL) {\n" : "SIMPLER_SCOPE() {\n"));
  Active().AppendRaw(body_emitter.GetCode());
  Active().AppendRaw(parent_indent + "}\n");

  if (!scope->manual_) closed_auto_scope_names_.insert(local_names.begin(), local_names.end());

  // Restore the outer scheduling bindings. A binding minted inside the block
  // that names a scope-local C++ identifier (e.g. ``TaskId prev =
  // arr[k];``) dies at the closing brace and must not leak (issue #1577).
  // BUT an array carry registered inside the scope can reuse a backing array
  // declared in the ENCLOSING scope (issue #1811) — see
  // ``PreserveEnclosingArrayCarries``. Both scope modes track local names;
  // declarations hoisted out of a MANUAL scope are removed from its set.
  PreserveEnclosingArrayCarries(&saved_array_carry, &saved_map, &saved_map_by_key,
                                [&](const std::string& name) { return local_names.count(name) != 0; });

  manual_task_id_map_ = std::move(saved_map);
  manual_task_id_map_by_key_ = std::move(saved_map_by_key);
  array_carry_vars_ = std::move(saved_array_carry);
  RebuildLiveManualTaskIdEmitNames();
}

/// Mark each ArrayType ``IfStmt`` return_var (a phi) as bound-by-yield, so no
/// declaration is emitted for it.
///
/// An ArrayType SSA value is a *reference* to one backing C-stack array, not a
/// copyable value: ``array.update_element`` is SSA-functional in the IR but
/// in-place in the emitted C, and every result aliases its input's emit name
/// (see ``HandleArrayUpdateElementAssign``). Both branches therefore mutate the
/// same storage and the merge is a no-op — there is nothing to declare, and a
/// raw C array could not be declared from its type or assigned anyway. The PTO
/// backend keeps in-place (array / tensor) return_vars out of the ``scf.if``
/// results for the same reason (``pto_control_flow_codegen.cpp``).
///
/// Each branch's YieldStmt records the backing array it resolves to (see
/// ``RecordArrayPhiYield``); ``InstallArrayPhiBindings`` then binds the phi
/// after both branches are emitted. Resolving at yield time — rather than by
/// pre-scanning the branch — is what lets an array updated under *nested*
/// control flow work: a nested ``if`` / ``for`` has already bound its own
/// return_var by the time the enclosing branch yields it.
void OrchestrationStmtCodegen::MarkArrayReturnVars(const IfStmtPtr& if_stmt) {
  for (const auto& rv : if_stmt->return_vars_) {
    if (!As<ArrayType>(rv->GetType())) continue;
    INTERNAL_CHECK_SPAN(if_stmt->else_body_.has_value(), if_stmt->span_)
        << "Internal error: IfStmt with ArrayType return_vars requires an else_body to name the "
           "backing array its branches mutate";
    pending_array_phis_.emplace(rv.get(), ArrayPhiBinding{});
  }
}

/// Record the backing array one branch yields for an ArrayType phi. Called
/// from the branch's YieldStmt, where the yielded Var is already bound —
/// including when it is a nested ``if`` / ``for`` return_var.
void OrchestrationStmtCodegen::RecordArrayPhiYield(const VarPtr& rv, const ExprPtr& yield_value,
                                                   const Span& span) {
  auto yield_var = AsVarLike(yield_value);
  INTERNAL_CHECK_SPAN(yield_var, span) << "Internal error: ArrayType IfStmt phi '" << rv->name_hint_
                                       << "' expects a Var yield value, got " << yield_value->TypeName();
  auto name_it = emit_name_map_.find(yield_var.get());
  INTERNAL_CHECK_SPAN(name_it != emit_name_map_.end(), span)
      << "Internal error: ArrayType IfStmt phi '" << rv->name_hint_ << "' yield value '"
      << yield_var->name_hint_ << "' has no backing array emit name";

  auto& binding = pending_array_phis_.at(rv.get());
  if (binding.resolved) {
    // The other branch got here first — both must name the same storage.
    // Creating a separate array per branch has no single backing to bind to.
    CHECK_SPAN(binding.array_name == name_it->second, span)
        << "Assigning a different array in each branch of an `if` is not supported: '" << rv->name_hint_
        << "' would name one array on one path (" << binding.array_name << ") and another on the other ("
        << name_it->second
        << "). Create the array before the `if` and write its elements inside the branches "
           "instead.";
    return;
  }
  binding.resolved = true;
  binding.array_name = name_it->second;
  // Carry the backing array's registrations so a downstream ``deps=[arr[i]]``
  // or an enclosing ForStmt carry seeded by the phi resolves to the same
  // storage. Captured here because a generated ``SIMPLER_SCOPE`` restores
  // ``array_carry_vars_`` / ``manual_task_id_map_`` at its closing brace.
  auto carry_it = array_carry_vars_.find(yield_var.get());
  if (carry_it != array_carry_vars_.end()) binding.carry = carry_it->second;
  auto tid_it = manual_task_id_map_.find(yield_var.get());
  if (tid_it != manual_task_id_map_.end()) binding.task_id = tid_it->second;
}

/// Install the array-phi bindings captured from the branches. Runs after both
/// branches are emitted, at the enclosing level, because every generated
/// ``SIMPLER_SCOPE`` snapshots and restores ``array_carry_vars_`` — registering
/// inside a branch body would be discarded at its closing brace.
void OrchestrationStmtCodegen::InstallArrayPhiBindings(const IfStmtPtr& if_stmt) {
  for (const auto& rv : if_stmt->return_vars_) {
    if (!As<ArrayType>(rv->GetType())) continue;
    auto it = pending_array_phis_.find(rv.get());
    if (it == pending_array_phis_.end()) continue;
    INTERNAL_CHECK_SPAN(it->second.resolved, if_stmt->span_)
        << "Internal error: IfStmt ArrayType return_var '" << rv->name_hint_
        << "' was not resolved to a backing array by either branch yield";
    emit_name_map_[rv.get()] = it->second.array_name;
    if (it->second.carry.has_value()) array_carry_vars_[rv.get()] = *it->second.carry;
    if (it->second.task_id.has_value()) manual_task_id_map_[rv.get()] = *it->second.task_id;
    pending_array_phis_.erase(it);
  }
}

void OrchestrationStmtCodegen::VisitStmt_(const IfStmtPtr& if_stmt) {
  std::string cond_expr = GenerateExprString(if_stmt->condition_);

  // A default-constructed ``Tensor`` is deliberately uninitialised. Seed
  // the phi declaration for a Tensor return_var from a valid Tensor
  // that's already in scope instead. Try sources in order:
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
      // The branch body may be wrapped in an AUTO scope by
      // MaterializeRuntimeScopes; peek through it to reach the trailing yield.
      auto y = transform_utils::GetLastYieldStmt(UnwrapAutoScope(body));
      if (!y) return {};
      for (const auto& v : y->value_) {
        auto var = AsVarLike(v);
        if (!var) continue;
        if (!AsTensorTypeLike(var->GetType())) continue;
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

  MarkArrayReturnVars(if_stmt);

  for (const auto& rv : if_stmt->return_vars_) {
    // ArrayType return_vars are bound to their backing array by the branch
    // yields and need no declaration — see MarkArrayReturnVars.
    if (As<ArrayType>(rv->GetType())) continue;
    const std::string emit_name = ReserveVarEmitName(rv.get());
    const std::string cpp_type = GetCppType(rv->GetType());
    if (cpp_type == "Tensor") {
      INTERNAL_CHECK_SPAN(!tensor_phi_init.empty(), if_stmt->span_)
          << "Internal error: IfStmt return_var '" << rv->name_hint_
          << "' is a Tensor but no in-scope Tensor was found to use as the "
          << "valid phi placeholder (a default-constructed ``Tensor`` is uninitialised). "
          << "Expected either a function parameter or a branch yield value "
          << "to resolve to a Var already declared at if-entry.";
      // Phi placeholder init — overwritten by branch yields. When the IfStmt is
      // directly in a ``pl.manual_scope`` body, the decl is hoisted to the
      // enclosing scope so a reader placed AFTER the block resolves the phi
      // (issue #1713 — same shape as a loop carry; the branch ``phi = ...;``
      // merges stay in-block). The phi init (a param or a pre-if Var) must be
      // enclosing-scope-valid, or the decl stays in place (EmitMutableTensorCarryDecl).
      EmitMutableTensorCarryDecl(emit_name, tensor_phi_init);
    } else if (auto sty = As<ScalarType>(rv->GetType()); sty && sty->dtype_ == DataType::TASK_ID) {
      // A TaskId phi is declared HERE, outside the branches, and each arm's
      // yield assigns into it — so unlike a branch-local producer id it is
      // still live after the ``if`` closes. Seed it with the sentinel (a
      // default-constructed ``TaskId`` is uninitialised, and an arm may leave
      // it unassigned) and register it at this enclosing level, mirroring
      // ``InstallArrayPhiBindings`` below: a registration made inside a branch
      // body would be dropped by that branch's scope restore. Without it a
      // downstream ``deps=[phi]`` cannot resolve and an ``arr[i] = phi``
      // publish looks like a store of a closed-scope local.
      EmitIndentedLine(cpp_type + " " + emit_name + " = TaskId::invalid();");
      manual_task_id_map_[rv.get()] = emit_name;
      manual_task_id_map_by_key_[TaskIdHoistKey(rv.get())] = emit_name;
      NoteLiveManualTaskIdEmitName(emit_name);
    } else {
      EmitIndentedLine(cpp_type + " " + emit_name + ";");
    }
  }

  EmitIndentedLine("if (" + cond_expr + ") {");

  VisitScopedBranchBody(if_stmt->then_body_, if_stmt->return_vars_);

  const auto& else_body = if_stmt->else_body_;
  if (else_body.has_value()) {
    EmitIndentedLine("} else {");
    VisitScopedBranchBody(*else_body, if_stmt->return_vars_);
  }

  EmitIndentedLine("}");

  // Bind array phis here, at the enclosing level: a registration made inside a
  // branch body would be dropped by that branch's scope restore.
  InstallArrayPhiBindings(if_stmt);
}

void OrchestrationStmtCodegen::VisitStmt_(const YieldStmtPtr& yield_stmt) {
  std::unordered_set<std::string> emitted_dynamic_dep_yields;
  for (size_t i = 0; i < yield_stmt->value_.size(); ++i) {
    if (i >= current_return_vars_.size() || !current_return_vars_[i]) {
      // Null slots in current_return_vars_ are placeholders for trivial
      // yields (where the body does not rebind the iter_arg) — skip those.
      continue;
    }
    const auto& rv = current_return_vars_[i];
    // A carry may be declared outside the loop/branch while its yielded
    // producer was local to a nested scope that has already closed.
    if (auto stale_tid = FindClosedScopeTaskId(yield_stmt->value_[i])) {
      CHECK_SPAN(false, yield_stmt->span_)
          << "Task id '" << stale_tid->name_hint_
          << "' is yielded after the `pl.scope()` that produced it has closed, so the yield "
             "has no runtime value to name. Publish the task id into an array declared outside "
             "that scope before it closes, then yield a read of that array element.";
    }
    // ArrayType IfStmt phi: record the backing array this branch yields; the
    // enclosing IfStmt binds it once both branches are emitted. Nothing is
    // emitted — the branch already mutated that array in place.
    if (pending_array_phis_.count(rv.get())) {
      RecordArrayPhiYield(rv, yield_stmt->value_[i], yield_stmt->span_);
      continue;
    }
    auto dyn_it = dynamic_task_id_collections_.find(rv.get());
    if (dyn_it != dynamic_task_id_collections_.end()) {
      auto yield_var = AsVarLike(yield_stmt->value_[i]);
      INTERNAL_CHECK_SPAN(yield_var, yield_stmt->span_)
          << "Internal error: dynamic compiler-dep yield expects a Var value";
      const ManualTaskIdBinding* tid_binding = ResolveManualTaskIdBinding(yield_var.get());
      INTERNAL_CHECK_SPAN(tid_binding != nullptr, yield_stmt->span_)
          << "Internal error: dynamic compiler-dep yield must resolve to a TaskId";
      auto* scalar_name = std::get_if<std::string>(tid_binding);
      INTERNAL_CHECK_SPAN(scalar_name, yield_stmt->span_)
          << "Internal error: dynamic compiler-dep yield expects string-variant TaskId";
      const std::string dedup_key = dyn_it->second.data_name + "\n" + *scalar_name;
      if (!emitted_dynamic_dep_yields.insert(dedup_key).second) {
        continue;
      }
      const std::string profile_start_name =
          ReserveSyntheticEmitName(dyn_it->second.data_name + "_write_profile_start");
      EmitIndentedLine("#if SIMPLER_ORCH_PROFILING");
      EmitIndentedLine("uint64_t " + profile_start_name + " = rt_orch_profile_now();");
      EmitIndentedLine("#endif");
      // A fresh direct-producer yield is statically valid, so skip the
      // is_valid() guard block and append unconditionally; any other id
      // (loop-carried / None-seed) keeps the guard.
      const bool needs_guard = guaranteed_valid_task_ids_.count(*scalar_name) == 0;
      if (needs_guard) {
        EmitIndentedLine("if (" + *scalar_name + ".is_valid()) {");
        Active().IncreaseIndent();
      }
      EmitIndentedLine(dyn_it->second.data_name + "[" + dyn_it->second.count_name + "++] = " + *scalar_name +
                       ";");
      EmitIndentedLine("#if SIMPLER_ORCH_PROFILING");
      EmitIndentedLine("rt_orch_profile_add_dynamic_dep_vector(rt_orch_profile_now() - " +
                       profile_start_name + ", 1);");
      EmitIndentedLine("#endif");
      if (needs_guard) {
        Active().DecreaseIndent();
        EmitIndentedLine("}");
      }
    }
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
        // inner Parallel's array). When the yield value and the carry
        // resolve to the *same* backing array, the body's
        // ``array.update_element`` already wrote it in place — a
        // slot-by-slot self-copy would be a useless no-op, so skip it.
        if (inner_it->second.array_name == rv_arr.array_name) {
          continue;
        }
        INTERNAL_CHECK_SPAN(inner_it->second.size == rv_arr.size, yield_stmt->span_)
            << "Internal error: array-carry yield size mismatch (rv=" << rv_arr.size
            << ", yield=" << inner_it->second.size << ")";
        EmitArrayCopyLoop(rv_arr.size, rv_arr.array_name, inner_it->second.array_name, "__yield_i");
      } else {
        // Scalar → array slot write (Parallel inner yielding a fresh task id
        // into its slot). The slot index is the enclosing loop's 0-based
        // ordinal, so non-trivial parallel ranges still index ``arr[0..N-1]``.
        const ManualTaskIdBinding* tid_binding = ResolveManualTaskIdBinding(yield_var.get());
        INTERNAL_CHECK_SPAN(tid_binding != nullptr, yield_stmt->span_)
            << "Internal error: scalar yield to array carry must resolve to a "
            << "TaskId variable registered in manual_task_id_map_";
        auto* scalar_name = std::get_if<std::string>(tid_binding);
        INTERNAL_CHECK_SPAN(scalar_name, yield_stmt->span_)
            << "Internal error: scalar yield to array carry expects string-variant entry";
        INTERNAL_CHECK_SPAN(!current_loop_slot_exprs_.empty(), yield_stmt->span_)
            << "Internal error: scalar yield to array carry requires an enclosing loop var";
        EmitIndentedLine(rv_arr.array_name + "[" + current_loop_slot_exprs_.back() + "] = " + *scalar_name +
                         ";");
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
    // The destination must be in C++ scope, but this write does not read its
    // old backing storage (including values recorded from the other branch).
    ValidateValueUse(rv, yield_stmt->span_, /*read_storage=*/false);
    std::string lhs_name = GetVarName(rv);
    if (AsTensorTypeLike(rv->GetType())) {
      const auto destination = tensor_yield_values_.find(rv.get());
      const size_t result =
          destination == tensor_yield_values_.end() ? GetTensorValue(lhs_name) : destination->second;
      RecordTensorSource(result, GetTensorValue(value_expr));
    }
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
      EmitArrayCopyLoop(N, lhs_name, value_expr, "__yield_i");
      continue;
    }
    EmitIndentedLine(lhs_name + " = " + value_expr + ";");
  }
}

/// Emit the C++ ``for`` header for a ForStmt.
///
/// The continuation test depends on the sign of the step: ``ForStmt::step_``
/// carries no sign restriction and ``pl.range(64, 0, -1)`` is valid DSL, so a
/// hard-coded ``i < stop`` silently compiles a descending loop into zero
/// iterations. When the step is a compile-time constant its sign picks the
/// operator directly, which keeps the overwhelmingly common ascending case
/// emitting exactly the same text as before.
///
/// A zero step must yield zero iterations, matching
/// ``transform_utils::ComputeStaticTripCount``. Letting it pick either
/// comparison emits ``i += 0`` under a condition that can hold, i.e. a loop
/// that never terminates. So the constant case tests the sign three ways, and
/// the runtime case requires a non-zero step in *both* direction branches
/// rather than treating "not positive" as "negative".
void OrchestrationStmtCodegen::EmitForLoopHeader(const std::string& loop_var, const std::string& start_expr,
                                                 const std::string& stop_expr, const std::string& step_expr,
                                                 const ExprPtr& step) {
  std::string cond;
  if (auto const_step = transform_utils::EvalConstInt(step)) {
    if (*const_step > 0) {
      cond = loop_var + " < " + stop_expr;
    } else if (*const_step < 0) {
      cond = loop_var + " > " + stop_expr;
    } else {
      cond = "false";
    }
  } else {
    cond = "(((" + step_expr + ") > 0 && " + loop_var + " < " + stop_expr + ") || ((" + step_expr +
           ") < 0 && " + loop_var + " > " + stop_expr + "))";
  }
  EmitIndentedLine("for (int64_t " + loop_var + " = " + start_expr + "; " + cond + "; " + loop_var +
                   " += " + step_expr + ") {");
}

void OrchestrationStmtCodegen::EmitArrayCopyLoop(int64_t extent, const std::string& dst_array,
                                                 const std::string& src_array, const std::string& index_var) {
  const std::string n = std::to_string(extent);
  EmitIndentedLine("for (int64_t " + index_var + " = 0; " + index_var + " < " + n + "; ++" + index_var +
                   ") " + dst_array + "[" + index_var + "] = " + src_array + "[" + index_var + "];");
}

void OrchestrationStmtCodegen::EmitArrayFillLoop(int64_t extent, const std::string& dst_array,
                                                 const std::string& value_expr,
                                                 const std::string& index_var) {
  const std::string n = std::to_string(extent);
  EmitIndentedLine("for (int64_t " + index_var + " = 0; " + index_var + " < " + n + "; ++" + index_var +
                   ") " + dst_array + "[" + index_var + "] = " + value_expr + ";");
}

void OrchestrationStmtCodegen::VisitScopedBranchBody(const StmtPtr& body,
                                                     const std::vector<VarPtr>& return_vars) {
  IndentGuard indent_guard(Active());
  PushCppScope();
  // The implicit ``SIMPLER_SCOPE()`` wrapper around the branch body is now an
  // explicit AUTO RuntimeScopeStmt inserted by MaterializeRuntimeScopes
  // (suppressed inside a manual scope); visiting the body emits it 1:1.
  auto saved = current_return_vars_;
  current_return_vars_.assign(return_vars.begin(), return_vars.end());
  VisitStmt(body);
  current_return_vars_ = saved;

  PopCppScope();
}

}  // namespace codegen
}  // namespace pypto
