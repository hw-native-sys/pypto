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

#include <exception>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

/// Walks every AssignStmt reachable from a function body and asserts that the
/// left-hand-side Var's type is structurally equal to the right-hand-side
/// value's type. ``structural_equal`` (the IR's own type-equality contract)
/// compares, per type kind:
///   - TileType:   dtype, shape, tile_view, and memory_space
///   - TensorType: dtype, shape, tensor_view (DistributedTensorType also
///                 compares the window_buffer back-reference)
///   - TupleType:  every element recursively (so Submit's TASK_ID-augmented
///                 return tuples are compared element-wise)
/// so a single check covers the "pass mutated one side of an assignment" bug
/// class (e.g. #1262 TileType memory_space, #1278 tile_view). ``memref_``
/// (inherited from ShapedType) is intentionally NOT compared by
/// ``structural_equal`` — it is an allocation detail bound to the Var, governed
/// by ``HasMemRefs`` / ``AllocatedMemoryAddr`` — so MemRef asymmetry,
/// legitimate after ``InitMemRef``, is out of scope here. Note ``memory_space``
/// exists only on TileType, not TensorType.
class AssignTypeSymmetryVisitor : public IRVisitor {
 public:
  AssignTypeSymmetryVisitor(std::vector<Diagnostic>& diagnostics, std::string func_name)
      : diagnostics_(diagnostics), func_name_(std::move(func_name)) {}

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    if (op && op->var_ && op->value_) CheckAssign(op);
    IRVisitor::VisitStmt_(op);
  }

 private:
  void CheckAssign(const AssignStmtPtr& op) {
    const auto& var_type = op->var_->GetType();
    const auto& rhs_type = op->value_->GetType();
    // Only compare when both sides carry a type. A null type means the value
    // has not been typed yet (legal in pre-type-check IR); flagging that here
    // would be a TypeCheck concern, not an asymmetry concern.
    if (!var_type || !rhs_type || structural_equal(var_type, rhs_type)) return;

    std::ostringstream msg;
    msg << "AssignStmt LHS/RHS type mismatch in function '" << func_name_ << "': var '"
        << op->var_->name_hint_ << "' type is not structurally equal to its RHS type. ";
    // assert_structural_equal throws with the precise mismatch path / reason
    // (e.g. "TileType memory_space mismatch"). Forward that into the message
    // so the diagnostic pinpoints which field diverged.
    try {
      assert_structural_equal(var_type, rhs_type);
    } catch (const std::exception& e) {
      msg << e.what();
    }
    diagnostics_.emplace_back(DiagnosticSeverity::Error, "AssignTypeSymmetry", /*error_code=*/1, msg.str(),
                              op->span_);
  }

  std::vector<Diagnostic>& diagnostics_;
  std::string func_name_;
};

}  // namespace

class AssignTypeSymmetryPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "AssignTypeSymmetry"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [global_var, func] : program->functions_) {
      if (!func || !func->body_) continue;
      AssignTypeSymmetryVisitor visitor(diagnostics, func->name_);
      visitor.VisitStmt(func->body_);
    }
  }
};

PropertyVerifierPtr CreateAssignTypeSymmetryPropertyVerifier() {
  return std::make_shared<AssignTypeSymmetryPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
