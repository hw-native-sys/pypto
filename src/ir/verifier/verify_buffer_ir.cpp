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
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/buffer_elementwise_recipes.h"
#include "pypto/backend/common/buffer_view_semantics.h"
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
      if (param && As<BufferType>(param->GetType())) RegisterWindow(param, param.get(), 0);
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

  void VisitExpr_(const WindowBufferPtr& op) override { CheckWindowBuffer(op); }

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
      RegisterWindow(op->var_, op->var_.get(), 0);
    } else if (call && valid_calls_.count(call.get()) &&
               (IsOp(call, "buffer.subview") || IsOp(call, "buffer.reshape"))) {
      const auto source = AsVarLike(call->args_[0]);
      const auto window = windows_.find(source.get());
      view_handles_.insert(op->var_.get());
      if (window == windows_.end()) {
        Error("Buffer view requires proven source-window provenance", call->span_);
      } else {
        uint64_t offset = window->second.offset;
        if (IsOp(call, "buffer.subview")) {
          const auto offsets = As<MakeTuple>(call->args_[1]);
          offset += static_cast<uint64_t>(As<ConstInt>(offsets->elements_[0])->value_) * 32;
        }
        RegisterWindow(op->var_, window->second.root, offset);
      }
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
        if (IsOp(op, "buffer.copy") && (view_handles_.count(AsVarLike(op->args_[0]).get()) ||
                                        view_handles_.count(AsVarLike(op->args_[1]).get()))) {
          CheckWindowPair(op, op->args_[0], op->args_[1], true);
        }
        if (IsOp(op, "buffer.extract")) CheckWindowPair(op, op->args_[0], op->args_[3], false);
        if (IsOp(op, "buffer.set_validshape") && view_handles_.count(AsVarLike(op->args_[0]).get())) {
          Error("buffer.set_validshape cannot mutate static view metadata", op->span_);
        }
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
      if (bits == 0 || (dtype.IsUnsignedInt() && result->value_ < 0) ||
          (bits < 64 && (result->value_ < (dtype.IsUnsignedInt() ? 0 : -(int64_t{1} << (bits - 1))) ||
                         result->value_ > ((int64_t{1} << (bits - (dtype.IsUnsignedInt() ? 0 : 1))) - 1)))) {
        result = nullptr;
      }
    }
    constants_[expr.get()] = result;
    return result;
  }

  struct Window {
    const Var* root;
    uint64_t offset;
    uint64_t bytes;
  };

  std::optional<uint64_t> PhysicalBytes(const BufferTypePtr& type) {
    const auto [entry, inserted] = byte_sizes_.try_emplace(type.get(), std::nullopt);
    if (inserted) entry->second = backend::PhysicalBufferBytes(type);
    return entry->second;
  }

  void RegisterWindow(const VarPtr& variable, const Var* root, uint64_t offset) {
    if (const auto bytes = PhysicalBytes(As<BufferType>(variable->GetType()))) {
      windows_[variable.get()] = Window{root, offset, *bytes};
    }
  }

  // This pairwise proof accepts different extents. Same-root views use relative
  // windows; separately placed roots require effective addresses, not distinct
  // SSA pointers. Later instruction recipes can reuse this storage proof.
  void CheckWindowPair(const CallPtr& call, const ExprPtr& source_expr, const ExprPtr& destination_expr,
                       bool exact_allowed) {
    const auto source = AsVarLike(source_expr);
    const auto destination = AsVarLike(destination_expr);
    if (source && source == destination && exact_allowed) return;
    // Placed addresses are per memory space: distinct spaces never overlap.
    const auto source_type = source ? As<BufferType>(source->GetType()) : nullptr;
    const auto destination_type = destination ? As<BufferType>(destination->GetType()) : nullptr;
    if (source_type && destination_type && source_type->memory_space_ != destination_type->memory_space_) {
      return;
    }
    const auto src = windows_.find(source.get());
    const auto dst = windows_.find(destination.get());
    const std::string name = call->op_->name_;
    if (src == windows_.end() || dst == windows_.end()) {
      Error(name + " requires proven allocation provenance for distinct operands", call->span_);
      return;
    }
    __int128 source_start = src->second.offset;
    __int128 destination_start = dst->second.offset;
    if (src->second.root != dst->second.root) {
      const auto src_root = allocations_.find(src->second.root);
      const auto dst_root = allocations_.find(dst->second.root);
      if (src_root == allocations_.end() || dst_root == allocations_.end()) {
        Error(name + " requires proven allocation provenance for distinct operands", call->span_);
        return;
      }
      if (src_root->second.symbolic && dst_root->second.symbolic) return;
      const auto& src_address = src_root->second.address;
      const auto& dst_address = dst_root->second.address;
      if (!src_address || !dst_address) {
        Error(name + " requires provably disjoint constant addresses or addressless allocations",
              call->span_);
        return;
      }
      if (src_address->value_ < 0 || dst_address->value_ < 0) {
        Error(name + " requires nonnegative placed addresses", call->span_);
        return;
      }
      source_start += src_address->value_;
      destination_start += dst_address->value_;
    }
    if (source_start == destination_start && src->second.bytes == dst->second.bytes && exact_allowed) return;
    if (source_start + src->second.bytes > destination_start &&
        destination_start + dst->second.bytes > source_start) {
      Error(name + (exact_allowed ? " rejects partially overlapping placed source and destination ranges"
                                  : " requires disjoint placed source and destination ranges"),
            call->span_);
    }
  }

  void CheckRecipeWindows(const CallPtr& call, const backend::BufferElementwiseRecipe& recipe) {
    const bool exact_allowed =
        recipe.destination_alias == backend::BufferDestinationAliasPolicy::ExactOrDisjoint;
    for (size_t i = 0; i < recipe.inputs.size(); ++i) {
      if (recipe.inputs[i].kind == backend::BufferElementwiseOperandKind::Buffer) {
        CheckWindowPair(call, call->args_[i], call->args_.back(), exact_allowed);
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

  // Exempt only the allocation's own Ptr carrier. Child expressions still
  // undergo representation checks and cannot hide buffer references.
  void VisitStorageBase(const ExprPtr& base) {
    if (base && As<PtrType>(base->GetType())) {
      IRVisitor::VisitExpr(base);
    } else {
      VisitExpr(base);
    }
  }

  void CheckWindowBuffer(const WindowBufferPtr& window) {
    if (!window || !checked_window_buffers_.insert(window.get()).second) return;
    // The base visitor treats windows as leaves. Check shared back-references
    // once, whether reached through tensor metadata or as ordinary expressions.
    auto previous_context = std::exchange(attribute_context_, "WindowBuffer metadata");
    VisitStorageBase(window->base_);
    VisitExpr(window->size_);
    attribute_context_ = std::move(previous_context);
  }

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
        VisitStorageBase(memref->base_);
        VisitExpr(memref->byte_offset_);
      }
    }
    if (auto tensor = As<DistributedTensorType>(type);
        tensor && tensor->window_buffer_ && *tensor->window_buffer_) {
      CheckWindowBuffer(*tensor->window_buffer_);
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
  std::unordered_set<const WindowBuffer*> checked_window_buffers_;
  std::unordered_set<const Var*> buffer_parameters_;
  std::unordered_map<const Var*, Allocation> allocations_;
  std::unordered_map<const Var*, Window> windows_;
  std::unordered_set<const Var*> view_handles_;
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
    // Existing verifiers own binding and lexical definition checks. Restrict their
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
