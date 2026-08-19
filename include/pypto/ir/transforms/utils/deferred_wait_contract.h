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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_DEFERRED_WAIT_CONTRACT_H_
#define PYPTO_IR_TRANSFORMS_UTILS_DEFERRED_WAIT_CONTRACT_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/transform_utils.h"

namespace pypto {
namespace ir {
namespace outline_utils {

/**
 * @brief Return whether @p body contains a deferred-completion registration.
 *
 * Keep the visitor private to this small shared query so scope outlining and
 * late validation cannot drift into separate operator-detection rules.
 */
[[nodiscard]] inline bool ContainsDeferredWait(const StmtPtr& body) {
  class Finder : public IRVisitor {
   public:
    bool found = false;

   protected:
    void VisitExpr_(const CallPtr& call) override {
      if (IsOp(call, "pld.system.defer_wait")) found = true;
      IRVisitor::VisitExpr_(call);
    }
  };

  Finder finder;
  finder.VisitStmt(body);
  return finder.found;
}

/**
 * @brief Per-waiter completion-condition budget.
 *
 * Must match the runtime's ``MAX_COMPLETIONS_PER_TASK``
 * (``runtime/src/{arch}/runtime/{rt}/runtime/aicore_completion_mailbox_types.h``).
 * PyPTO's C++ never includes a runtime header, so the two cannot be linked
 * directly; the generated adapter in ``pto_backend.py`` carries a
 * ``static_assert`` against the real constant to catch drift at kernel-compile
 * time.
 *
 * Exceeding the budget on device is not a hang: the AIV writes
 * ``ASYNC_WAIT_OVERFLOW`` into the completion slab and the scheduler surfaces it
 * as ``sched_error_code=102`` while aborting the run. This compile-time bound
 * exists for attribution — a span-anchored error naming the offending
 * ``pl.at`` scope — not to avoid a worse device-side failure.
 */
inline constexpr int64_t kMaxDeferredConditionsPerWaiter = 64;

/**
 * @brief Validate the dedicated deferred-waiter kernel body contract.
 *
 * A waiter may use scalar expressions and sequential scalar control flow to
 * compute/register conditions, but must not perform payload, cache, or other
 * communication work.  Once a defer_wait registration has executed on a
 * path, only more registrations or scalar bookkeeping may follow.  The
 * registration upper bound is proved statically. Conditional registration
 * (the common ``if peer != rank`` form) may execute zero times; Simpler accepts
 * that as an already-complete waiter. Loops with an unknown trip count are
 * rejected because they cannot prove the per-waiter condition budget.
 */
class DeferredWaitContractValidator {
 public:
  struct Result {
    bool has_deferred_wait = false;
    int64_t static_max_count = 0;
    // True when this subtree is safe to execute after an earlier condition was
    // registered. Control flow, pure scalar bookkeeping, and further
    // registrations are safe; tensor reads and continuation work are not.
    bool safe_after_registration = true;
  };

  static Result Validate(const StmtPtr& body, const Span& span) {
    // Most InCore scopes are ordinary payload kernels. Keep the strict waiter
    // validator off that hot path: first perform a side-effect-free recursive
    // probe, then validate the complete scope only when it really contains a
    // deferred registration.
    if (!ContainsDeferredWait(body)) return {};

    DeferredWaitContractValidator validator(span);
    auto result = validator.ValidateStmt(body);
    INTERNAL_CHECK_SPAN(result.has_deferred_wait, span)
        << "Internal error: deferred-wait finder/validator disagreement";
    CHECK_SPAN(result.static_max_count <= kMaxDeferredConditionsPerWaiter, span)
        << "deferred waiter supports at most " << kMaxDeferredConditionsPerWaiter
        << " conditions, but may statically register " << result.static_max_count;
    return result;
  }

 private:
  explicit DeferredWaitContractValidator(Span span) : span_(std::move(span)) {}

  [[nodiscard]] Result ValidateStmt(const StmtPtr& stmt) const {
    if (auto seq = As<SeqStmts>(stmt)) {
      Result total;
      bool registration_started = false;
      for (const auto& child : seq->stmts_) {
        auto child_result = ValidateStmt(child);
        CHECK_SPAN(!registration_started || child_result.safe_after_registration, child->span_)
            << "deferred waiter cannot execute tensor reads, payload, cache, communication, or "
               "continuation work after pld.system.defer_wait registration begins";
        registration_started = registration_started || child_result.has_deferred_wait;
        Merge(&total, child_result);
      }
      return total;
    }
    if (auto eval = As<EvalStmt>(stmt)) {
      return ValidateCall(As<Call>(eval->expr_), stmt->span_);
    }
    if (auto assign = As<AssignStmt>(stmt)) {
      auto call = As<Call>(assign->value_);
      if (call && IsOp(call, "pld.system.defer_wait")) {
        CHECK_SPAN(false, stmt->span_)
            << "pld.system.defer_wait is side-effect-only and must be used as a standalone statement";
      }
      CHECK_SPAN(!assign->var_ || !AsTensorTypeLike(assign->var_->GetType()), stmt->span_)
          << "deferred waiter is registration-only and cannot create or update payload tensors";
      auto effect = ValidateScalarExpr(assign->value_, stmt->span_, /*allow_tensor_read=*/true);
      return Result{false, 0, effect == ScalarEffect::kPure};
    }
    if (auto ret = As<ReturnStmt>(stmt)) {
      CHECK_SPAN(ret->value_.empty(), stmt->span_)
          << "deferred waiter is registration-only and cannot return a value";
      // Scope outlining appends an empty function return. It is structural,
      // not continuation work, and remains safe after registrations.
      return Result{false, 0, true};
    }
    if (auto loop = As<ForStmt>(stmt)) {
      CHECK_SPAN(loop->kind_ == ForKind::Sequential || loop->kind_ == ForKind::Unroll, stmt->span_)
          << "deferred waiter supports only sequential/unrolled scalar registration loops";
      static_cast<void>(ValidateScalarExpr(loop->start_, stmt->span_, /*allow_tensor_read=*/false));
      static_cast<void>(ValidateScalarExpr(loop->stop_, stmt->span_, /*allow_tensor_read=*/false));
      static_cast<void>(ValidateScalarExpr(loop->step_, stmt->span_, /*allow_tensor_read=*/false));
      auto body_result = ValidateStmt(loop->body_);
      // A loop that registers nothing contributes nothing to the budget, so it
      // needs no statically known trip count. Returning before the arithmetic
      // below also keeps `trip_count` out of the divisor when it is unknown.
      if (!body_result.has_deferred_wait) return body_result;
      auto trip_count = transform_utils::EvalConstTripCount(loop).value_or(0);
      CHECK_SPAN(trip_count > 0, stmt->span_)
          << "deferred waiter registration loop must have a statically known positive trip count "
             "so the per-waiter condition budget can be proved";
      CHECK_SPAN(trip_count == 1 || body_result.safe_after_registration, stmt->span_)
          << "deferred waiter loop may execute tensor reads or continuation work after registering its "
             "first condition; loop bodies may contain only pure scalar bookkeeping, control flow, "
             "and defer_wait registrations";
      // Saturate just above the supported limit rather than multiplying
      // unchecked: enormous constant bounds must fail the budget contract, not
      // wrap int64_t and accidentally pass it.
      body_result.static_max_count =
          body_result.static_max_count > kMaxDeferredConditionsPerWaiter / trip_count
              ? kMaxDeferredConditionsPerWaiter + 1
              : body_result.static_max_count * trip_count;
      return body_result;
    }
    if (auto branch = As<IfStmt>(stmt)) {
      static_cast<void>(ValidateScalarExpr(branch->condition_, stmt->span_, /*allow_tensor_read=*/false));
      auto then_result = ValidateStmt(branch->then_body_);
      auto else_result = branch->else_body_.has_value() ? ValidateStmt(*branch->else_body_) : Result{};
      Result result;
      result.has_deferred_wait = then_result.has_deferred_wait || else_result.has_deferred_wait;
      result.static_max_count = std::max(then_result.static_max_count, else_result.static_max_count);
      result.safe_after_registration =
          then_result.safe_after_registration && else_result.safe_after_registration;
      return result;
    }
    if (auto yield = As<YieldStmt>(stmt)) {
      // SSA carries, not user-written statements: ConvertToSSA emits a yield
      // for a scalar that crosses a branch merge (phi), a loop iteration
      // (iter_arg), or that escapes the loop defining it. They are bookkeeping,
      // so they neither register a condition nor count against the budget.
      // ``allow_tensor_read=false`` keeps a carry from smuggling in a memory
      // read, and ValidateScalarExpr's tensor-type guard rejects a tile carry
      // with its own message.
      for (const auto& value : yield->value_) {
        static_cast<void>(ValidateScalarExpr(value, stmt->span_, /*allow_tensor_read=*/false));
      }
      // Every carry that survives the loop above is pure scalar, so it remains
      // safe to execute after an earlier registration.
      return Result{false, 0, true};
    }
    CHECK_SPAN(false, stmt ? stmt->span_ : span_)
        << "deferred waiter body supports only scalar assignments/control flow and "
           "pld.system.defer_wait registrations";
    return {};
  }

  [[nodiscard]] Result ValidateCall(const CallPtr& call, const Span& span) const {
    CHECK_SPAN(call, span) << "deferred waiter contains a non-call side-effect statement";
    if (IsOp(call, "pld.system.defer_wait")) {
      INTERNAL_CHECK_SPAN(call->args_.size() == 3, span)
          << "Internal error: pld.system.defer_wait argument count was not verified";
      auto offsets = As<MakeTuple>(call->args_[1]);
      INTERNAL_CHECK_SPAN(offsets, span) << "Internal error: pld.system.defer_wait offsets were not verified";
      for (const auto& offset : offsets->elements_) {
        static_cast<void>(ValidateScalarExpr(offset, span, /*allow_tensor_read=*/false));
      }
      static_cast<void>(ValidateScalarExpr(call->args_[2], span, /*allow_tensor_read=*/false));
      return Result{true, 1, true};
    }
    CHECK_SPAN(false, span) << "deferred waiter is registration-only; unexpected operation '"
                            << call->op_->name_ << "'";
    return {};
  }

  enum class ScalarEffect {
    kPure,
    kTensorRead,
  };

  static ScalarEffect MergeScalarEffects(ScalarEffect lhs, ScalarEffect rhs) {
    return lhs == ScalarEffect::kTensorRead || rhs == ScalarEffect::kTensorRead ? ScalarEffect::kTensorRead
                                                                                : ScalarEffect::kPure;
  }

  [[nodiscard]] ScalarEffect ValidateScalarExpr(const ExprPtr& expr, const Span& span,
                                                bool allow_tensor_read) const {
    if (!expr) return ScalarEffect::kPure;
    CHECK_SPAN(!AsTensorTypeLike(expr->GetType()), span)
        << "deferred waiter scalar bookkeeping cannot produce a tensor value";
    if (auto call = As<Call>(expr)) {
      const bool permitted_anchor = allow_tensor_read && IsOp(call, "tensor.read");
      // Fail closed: a scalar-returning GlobalVar or newly added op may still
      // hide communication, a blocking wait, or payload effects. V1 needs only
      // an explicit pre-registration tensor.read anchor; scalar arithmetic and
      // casts are represented by their own IR expression nodes.
      CHECK_SPAN(permitted_anchor, span)
          << "deferred waiter scalar bookkeeping supports only a pre-registration tensor.read call; "
             "unexpected operation '"
          << call->op_->name_ << "'";
      // The first tensor.read argument is the tensor source. Skip that
      // position, not every ExprPtr that happens to alias it.
      for (size_t i = 1; i < call->args_.size(); ++i) {
        static_cast<void>(ValidateScalarExpr(call->args_[i], span, /*allow_tensor_read=*/false));
      }
      return ScalarEffect::kTensorRead;
    }
    if (auto binary = As<BinaryExpr>(expr)) {
      auto lhs = ValidateScalarExpr(binary->left_, span, allow_tensor_read);
      auto rhs = ValidateScalarExpr(binary->right_, span, allow_tensor_read);
      return MergeScalarEffects(lhs, rhs);
    }
    if (auto unary = As<UnaryExpr>(expr)) {
      return ValidateScalarExpr(unary->operand_, span, allow_tensor_read);
    }
    if (auto tuple = As<MakeTuple>(expr)) {
      auto effect = ScalarEffect::kPure;
      for (const auto& element : tuple->elements_) {
        effect = MergeScalarEffects(effect, ValidateScalarExpr(element, span, allow_tensor_read));
      }
      return effect;
    }
    if (auto get = As<TupleGetItemExpr>(expr)) {
      return ValidateScalarExpr(get->tuple_, span, allow_tensor_read);
    }
    if (AsVarLike(expr) || As<ConstInt>(expr) || As<ConstFloat>(expr) || As<ConstBool>(expr)) {
      return ScalarEffect::kPure;
    }
    CHECK_SPAN(false, span) << "deferred waiter scalar bookkeeping contains unsupported expression '"
                            << expr->TypeName() << "'";
    return ScalarEffect::kPure;
  }

  static void Merge(Result* dst, const Result& src) {
    dst->has_deferred_wait = dst->has_deferred_wait || src.has_deferred_wait;
    dst->static_max_count += src.static_max_count;
    dst->safe_after_registration = dst->safe_after_registration && src.safe_after_registration;
  }

  Span span_;
};

}  // namespace outline_utils
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_DEFERRED_WAIT_CONTRACT_H_
