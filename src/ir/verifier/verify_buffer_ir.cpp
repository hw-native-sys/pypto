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
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/buffer_elementwise_recipes.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/ir/arith/const_fold.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

// Representation validation is a single walk, with memoized summaries for
// shared tuple types. The three composed structural checks each add one fixed
// walk; no storage handle triggers a rescan of the function.
class BufferIRVisitor : public IRVisitor {
 public:
  BufferIRVisitor(std::vector<Diagnostic>& diagnostics, std::string function_name)
      : diagnostics_(diagnostics), function_name_(std::move(function_name)) {}

  [[nodiscard]] bool HasMalformedStructure() const { return malformed_structure_; }

  void VisitFunction(const FunctionPtr& function) override {
    for (const auto& type : function->return_types_) {
      CheckType(type, function->span_);
      if (ContainsBuffer(type)) {
        Error("Buffer handles cannot escape through function return types", function->span_);
      }
    }
    // Legacy SSA counts assignments without counting parameter bindings. The
    // final buffer boundary additionally preserves each incoming handle's
    // identity, including parameters containing handles inside tuples.
    for (const auto& param : function->params_) {
      if (param && ContainsBuffer(param->GetType())) buffer_parameters_.insert(param.get());
    }
    VisitValues(function->params_);
    CheckAttrs(function->attrs_, "Function attribute");
    if (function->body_) VisitStmt(function->body_);
  }

  void VisitStmt(const StmtPtr& stmt) override {
    if (auto scope = As<ScopeStmt>(stmt)) CheckAttrs(scope->attrs_, "Scope attribute");
    if (auto spmd = As<SpmdScopeStmt>(stmt)) CheckNoBufferValue(spmd->core_num_, "SPMD core count");
    IRVisitor::VisitStmt(stmt);
  }

  void VisitExpr(const ExprPtr& expr) override {
    if (!expr) return;
    CheckType(expr->GetType(), expr->span_);
    if (!attribute_context_.empty()) CheckNoBufferValue(expr, attribute_context_);
    if (ContainsBuffer(expr->GetType()) && !AsVarLike(expr) && !As<Call>(expr) &&
        !As<TupleGetItemExpr>(expr)) {
      Error("Buffer handles must be parameters or direct results of registered buffer operations",
            expr->span_);
    }
    IRVisitor::VisitExpr(expr);
  }

 protected:
  // CheckType visits all expression-valued metadata once per shared type,
  // including distributed tensor views. Avoid repeating the base shape walk.
  void VisitVarLike_(const VarPtr&) override {}

  // CheckAttrs already visits every scope attribute, including scope kinds
  // whose default visitor omits attrs. Do not traverse those references twice.
  [[nodiscard]] bool ShouldVisitScopeAttr(const std::string&) const override { return false; }

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (!op->var_ || !op->value_) {
      Error("AssignStmt requires a variable and a value", op->span_);
      malformed_structure_ = true;
      if (op->value_) VisitExpr(op->value_);
      return;
    }
    if (buffer_parameters_.count(op->var_.get())) {
      Error("Incoming buffer parameter '" + op->var_->name_hint_ + "' cannot be redefined", op->span_);
    }
    if (ContainsBuffer(op->var_->GetType()) && !As<Call>(op->value_) && !As<TupleGetItemExpr>(op->value_)) {
      Error(
          "A buffer definition requires an explicit allocation, alias, or borrow operation; "
          "assigning an existing handle creates an implicit alias",
          op->span_);
    }
    statement_call_ = As<Call>(op->value_).get();
    statement_assigns_result_ = true;
    IRVisitor::VisitStmt_(op);
    statement_call_ = nullptr;
    if (auto call = As<Call>(op->value_); IsOp(call, "buffer.alloc") && valid_calls_.count(call.get())) {
      allocations_[op->var_.get()] = Allocation{
          call->args_.size() == 1, call->args_.size() == 2 ? ConstantAddress(call->args_[1]) : nullptr};
    }
    if (As<ScalarType>(op->var_->GetType())) constants_[op->var_.get()] = ConstantAddress(op->value_);
  }

  void VisitStmt_(const EvalStmtPtr& op) override {
    statement_call_ = As<Call>(op->expr_).get();
    statement_assigns_result_ = false;
    IRVisitor::VisitStmt_(op);
    statement_call_ = nullptr;
  }

  void VisitExpr_(const CallPtr& op) override {
    if (!op->op_) {
      Error("Call has no callee", op->span_);
      malformed_structure_ = true;
      VisitCallOperands(op);
      return;
    }
    const auto& name = op->op_->name_;
    auto& registry = OpRegistry::GetInstance();
    const bool registered = !As<GlobalVar>(op->op_) && registry.IsRegistered(name);
    const bool buffer_stage = registered && registry.GetEntry(name).GetIRStage() == OpIRStage::Buffer;
    // These are family checks, including unregistered/deserialized tile ops.
    if (name.rfind("tile.", 0) == 0 || name.rfind("pld.tile.", 0) == 0) {
      Error("Logical tile operation '" + name + "' remains in buffer IR", op->span_);
    } else if (buffer_stage) {
      const bool produces_result = registry.GetEntry(name).GetOutputArity() != 0;
      if (statement_call_ != op.get() || statement_assigns_result_ != produces_result) {
        Error(
            "Buffer operations with SSA results require a direct AssignStmt; "
            "zero-result buffer operations require a direct EvalStmt",
            op->span_);
      }
      try {
        // Validate the original result and operands, not a newly deduced Call:
        // recreating it would hide malformed result types from the verifier.
        registry.ValidateBufferCall(op);
        valid_calls_.insert(op.get());
        if (const auto* recipe = backend::FindBufferElementwiseRecipe(name)) CheckRecipeWindows(op, *recipe);
      } catch (const pypto::Error& error) {
        Error("Invalid buffer call '" + name + "': " + error.what(), op->span_);
      }
    } else {
      if (name.rfind("buffer.", 0) == 0) {
        Error("Operation '" + name + "' has no registered buffer-stage contract", op->span_);
      }
      if (ContainsBuffer(op->GetType())) {
        Error("Non-buffer operation '" + name + "' cannot produce a buffer handle", op->span_);
      }
      CheckNoBufferValues(op->args_, "Non-buffer operation '" + name + "' arguments");
    }
    VisitCallOperands(op);
  }

  void VisitExpr_(const SubmitPtr& op) override {
    // Submit launches a runtime task; it is never an internal buffer operation.
    if (ContainsBuffer(op->GetType())) {
      Error("Submit cannot produce a buffer handle", op->span_);
    }
    CheckNoBufferValues(op->args_, "Submit arguments");
    CheckNoBufferValues(op->deps_, "Submit dependencies");
    if (op->core_num_) CheckNoBufferValue(*op->core_num_, "Submit core count");
    if (op->predicate_) CheckNoBufferValue(*op->predicate_, "Submit predicate");
    VisitValues(op->args_);
    VisitValues(op->deps_);
    if (op->core_num_) VisitExpr(*op->core_num_);
    if (op->predicate_) VisitExpr(*op->predicate_);
    CheckAttrs(op->attrs_, "Submit attribute");
    CheckAttrs(op->kwargs_, "Submit keyword");
  }

  void VisitBinaryExpr_(const BinaryExprPtr& op) override {
    CheckNoBufferValue(op->left_, "Scalar expression operand");
    CheckNoBufferValue(op->right_, "Scalar expression operand");
    IRVisitor::VisitBinaryExpr_(op);
  }

  void VisitUnaryExpr_(const UnaryExprPtr& op) override {
    CheckNoBufferValue(op->operand_, "Scalar expression operand");
    IRVisitor::VisitUnaryExpr_(op);
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    CheckNoBufferValues(op->return_vars_, "If return_vars");
    CheckNoBufferValue(op->condition_, "If condition");
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    CheckNoBufferValues(op->iter_args_, "For iter_args");
    CheckNoBufferValues(op->return_vars_, "For return_vars");
    CheckNoBufferValue(op->loop_var_, "For loop variable");
    CheckNoBufferValue(op->start_, "For start");
    CheckNoBufferValue(op->stop_, "For stop");
    CheckNoBufferValue(op->step_, "For step");
    CheckAttrs(op->attrs_, "For attribute");
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    CheckNoBufferValues(op->iter_args_, "While iter_args");
    CheckNoBufferValues(op->return_vars_, "While return_vars");
    CheckNoBufferValue(op->condition_, "While condition");
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const YieldStmtPtr& op) override {
    CheckNoBufferValues(op->value_, "Yield values");
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ReturnStmtPtr& op) override {
    CheckNoBufferValues(op->value_, "Function return values");
    IRVisitor::VisitStmt_(op);
  }

 private:
  struct Allocation {
    bool symbolic;
    ConstIntPtr address;  // Null with !symbolic means placement is unproven.
  };

  // Follow scalar SSA definitions once, without expanding IterArg initializers.
  // Existing checked arithmetic folds Add/Sub/Mul; a result outside the actual
  // integer width is deliberately unproven rather than treated as an address.
  ConstIntPtr ConstantAddress(const ExprPtr& expr) {
    if (!expr) return nullptr;
    if (auto cached = constants_.find(expr.get()); cached != constants_.end()) {
      return cached->second;
    }
    constants_[expr.get()] = nullptr;
    const auto scalar = As<ScalarType>(expr->GetType());
    if (!scalar) return nullptr;
    const auto dtype = scalar->dtype_;
    if (dtype != DataType::INDEX && !dtype.IsInt()) return nullptr;
    auto result = As<ConstInt>(expr);
    if (auto binary = As<BinaryExpr>(expr); binary && (As<Add>(expr) || As<Sub>(expr) || As<Mul>(expr))) {
      auto lhs = ConstantAddress(binary->left_);
      auto rhs = ConstantAddress(binary->right_);
      if (lhs && rhs) result = As<ConstInt>(arith::TryConstFoldBinary(expr->GetKind(), lhs, rhs));
    } else if (auto cast = As<Cast>(expr)) {
      result = ConstantAddress(cast->operand_);
    }
    if (result) {
      const auto bits = dtype.GetBit();
      if ((dtype.IsUnsignedInt() && result->value_ < 0) ||
          (bits < 64 && (result->value_ < (dtype.IsUnsignedInt() ? 0 : -(int64_t{1} << (bits - 1))) ||
                         result->value_ > ((int64_t{1} << (bits - (dtype.IsUnsignedInt() ? 0 : 1))) - 1)))) {
        result = nullptr;
      }
    }
    constants_[expr.get()] = result;
    return result;
  }

  std::optional<uint64_t> DenseBytes(const BufferTypePtr& type) {
    if (auto cached = byte_sizes_.find(type.get()); cached != byte_sizes_.end()) {
      return cached->second;
    }
    byte_sizes_[type.get()] = std::nullopt;
    if ((type->shape_.size() != 1 && type->shape_.size() != 2) || type->blayout_ != TileLayout::row_major ||
        type->slayout_ != TileLayout::none_box || type->fractal_ != 512 || type->pad_ != PadValue::null ||
        type->compact_ != CompactMode::null || type->dtype_.GetBit() % 8 != 0)
      return std::nullopt;
    uint64_t bytes = type->dtype_.GetBit() / 8;
    for (const auto extent : type->shape_) {
      if (static_cast<uint64_t>(extent) > std::numeric_limits<uint64_t>::max() / bytes) return std::nullopt;
      bytes *= static_cast<uint64_t>(extent);
    }
    byte_sizes_[type.get()] = bytes;
    return bytes;
  }

  void CheckRecipeWindows(const CallPtr& call, const backend::BufferElementwiseRecipe& recipe) {
    const auto destination = AsVarLike(call->args_.back());
    const auto dst = allocations_.find(destination.get());
    const bool exact_allowed =
        recipe.destination_alias == backend::BufferDestinationAliasPolicy::ExactOrDisjoint;
    for (size_t i = 0; i < recipe.input_count; ++i) {
      const auto source = AsVarLike(call->args_[i]);
      if (source && source == destination && exact_allowed) continue;
      const auto src = allocations_.find(source.get());
      if (src == allocations_.end() || dst == allocations_.end()) {
        Error(std::string(recipe.buffer_op) + " requires proven allocation provenance for distinct operands",
              call->span_);
        continue;
      }
      if (src->second.symbolic && dst->second.symbolic && source != destination) continue;
      const auto& src_address = src->second.address;
      const auto& dst_address = dst->second.address;
      if (!src_address || !dst_address) {
        Error(std::string(recipe.buffer_op) +
                  " requires provably disjoint constant addresses or addressless allocations",
              call->span_);
        continue;
      }
      if (src_address->value_ < 0 || dst_address->value_ < 0) {
        Error(std::string(recipe.buffer_op) + " requires nonnegative placed addresses", call->span_);
        continue;
      }
      if (src_address->value_ == dst_address->value_ && exact_allowed) continue;
      const auto bytes = DenseBytes(As<BufferType>(destination->GetType()));
      if (!bytes) {
        Error(std::string(recipe.buffer_op) + " requires a supported dense byte-window contract",
              call->span_);
        continue;
      }
      const auto source_end = static_cast<__int128>(src_address->value_) + *bytes;
      const auto destination_end = static_cast<__int128>(dst_address->value_) + *bytes;
      if (source_end > dst_address->value_ && destination_end > src_address->value_) {
        Error(std::string(recipe.buffer_op) +
                  (exact_allowed ? " rejects partially overlapping placed source and destination ranges"
                                 : " requires disjoint placed source and destination ranges"),
              call->span_);
      }
    }
  }

  enum TypeFlags : uint8_t { kNone = 0, kTile = 1, kRawStorage = 2, kBuffer = 4 };

  template <typename T>
  void VisitValues(const std::vector<T>& values) {
    for (const auto& value : values) VisitExpr(value);
  }

  void VisitCallOperands(const CallPtr& op) {
    VisitValues(op->args_);
    CheckAttrs(op->attrs_, "Call attribute");
    CheckAttrs(op->kwargs_, "Call keyword");
  }

  // Storage references in attributes have no ordinary-operand effect contract.
  // Keep the context across nested expressions, not just their result type, so
  // a scalar wrapper cannot hide a buffer dependency. Each attr edge is walked
  // once, even when an attribute expression contains another Call with attrs.
  void CheckAttrs(const std::vector<std::pair<std::string, std::any>>& attrs, const std::string& owner) {
    for (const auto& [key, value] : attrs) {
      auto previous_context = std::exchange(attribute_context_, owner + " '" + key + "'");
      ForEachAttrExpr(value, [this](const ExprPtr& expr) { VisitExpr(expr); });
      attribute_context_ = std::move(previous_context);
    }
  }

  uint8_t SummarizeType(const TypePtr& type) {
    if (!type) return kNone;
    auto found = type_flags_.find(type.get());
    if (found != type_flags_.end()) return found->second;
    uint8_t flags = kNone;
    if (As<TileType>(type)) {
      flags = kTile;
    } else if (As<MemRefType>(type) || As<PtrType>(type)) {
      flags = kRawStorage;
    } else if (As<BufferType>(type) || As<MultiBufferType>(type)) {
      flags = kBuffer;
    } else if (auto tuple = As<TupleType>(type)) {
      for (const auto& element : tuple->types_) flags |= SummarizeType(element);
    }
    // TensorType's GM MemRef is an allocation detail, not a device buffer SSA
    // operand. Deliberately do not descend into its storage metadata.
    type_flags_.emplace(type.get(), flags);
    return flags;
  }

  bool ContainsBuffer(const TypePtr& type) { return (SummarizeType(type) & kBuffer) != 0; }

  void CheckTypeMetadata(const TypePtr& type) {
    if (!type || !checked_metadata_types_.insert(type.get()).second) return;
    // Traverse tuple elements through this memoized entry point: a shared
    // nested tuple must not repeatedly expand all of its metadata fields.
    if (auto tuple = As<TupleType>(type)) {
      for (const auto& element : tuple->types_) CheckTypeMetadata(element);
      return;
    }
    auto previous_context = std::exchange(attribute_context_, "Type metadata");
    // The utility visits shape/view and MemRef slot expressions, not the GM
    // MemRef carrier itself. Scalar metadata remains legal; handles must be
    // ordinary buffer operands, never implicit type-dynamic definitions.
    var_collectors::VisitTypeExprFields(*this, type);
    if (auto tensor = AsTensorTypeLike(type); tensor && tensor->memref_ && *tensor->memref_) {
      const auto& memref = *tensor->memref_;
      if (checked_gm_memrefs_.insert(memref.get()).second) {
        // Exempt only the GM allocation's own Ptr carrier. Its child
        // expressions, and every offset expression, still use the ordinary
        // representation checks and cannot hide buffer references.
        if (memref->base_ && As<PtrType>(memref->base_->GetType())) {
          IRVisitor::VisitExpr(memref->base_);
        } else {
          VisitExpr(memref->base_);
        }
        VisitExpr(memref->byte_offset_);
      }
    }
    attribute_context_ = std::move(previous_context);
  }

  void CheckType(const TypePtr& type, const Span& span) {
    const auto flags = SummarizeType(type);
    if ((flags & kTile) != 0) Error("Logical TileType remains in buffer IR", span);
    if ((flags & kRawStorage) != 0) {
      Error("Standalone MemRef/Ptr values are not allowed in buffer IR", span);
    }
    CheckTypeMetadata(type);
  }

  void CheckNoBufferValue(const ExprPtr& expr, const std::string& context) {
    if (expr && ContainsBuffer(expr->GetType())) {
      Error(context + " cannot carry buffer handles; use explicit buffer operands and writes", expr->span_);
    }
  }

  template <typename T>
  void CheckNoBufferValues(const std::vector<T>& values, const std::string& context) {
    for (const auto& value : values) CheckNoBufferValue(value, context);
  }

  void Error(const std::string& message, const Span& span) {
    diagnostics_.emplace_back(DiagnosticSeverity::Error, "BufferIR", /*error_code=*/1,
                              message + " in function '" + function_name_ + "'", span);
  }

  std::vector<Diagnostic>& diagnostics_;
  std::string function_name_;
  std::unordered_map<const Type*, uint8_t> type_flags_;
  std::unordered_set<const Type*> checked_metadata_types_;
  std::unordered_set<const MemRef*> checked_gm_memrefs_;
  std::unordered_set<const Var*> buffer_parameters_;
  std::unordered_map<const Var*, Allocation> allocations_;
  std::unordered_map<const Expr*, ConstIntPtr> constants_;
  std::unordered_map<const BufferType*, std::optional<uint64_t>> byte_sizes_;
  std::unordered_set<const Call*> valid_calls_;
  const Call* statement_call_ = nullptr;
  bool statement_assigns_result_ = false;
  std::string attribute_context_;
  bool malformed_structure_ = false;
};

class BufferIRPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "BufferIR"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    std::vector<FunctionPtr> functions;
    for (const auto& [global_var, function] : program->functions_) {
      if (!function || !IsInCoreType(function->func_type_)) continue;
      BufferIRVisitor visitor(diagnostics, function->name_);
      visitor.VisitFunction(function);
      // Other verifiers assume intact structural nodes. The diagnostics above
      // already reject missing callees or assignment fields; do not send those
      // malformed functions into verifiers with stronger input assumptions.
      if (!visitor.HasMalformedStructure()) functions.push_back(function);
    }
    if (functions.empty()) return;
    // Existing verifiers own binding and dominance algorithms. Restrict their
    // input to device functions so this property leaves orchestration alone.
    auto device_program = std::make_shared<Program>(functions, program->name_, program->span_);
    CreateSSAPropertyVerifier()->Verify(device_program, diagnostics);
    CreateLexicalUseAfterDefPropertyVerifier()->Verify(device_program, diagnostics);
    CreateAssignTypeSymmetryPropertyVerifier()->Verify(device_program, diagnostics);
  }
};

}  // namespace

PropertyVerifierPtr CreateBufferIRPropertyVerifier() {
  return std::make_shared<BufferIRPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
