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
#include "pypto/ir/cast_saturation.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {
namespace pass {

namespace {

constexpr const char* kPassName = "FoldFixpipeAccEpilogue";

/// Where the ReLU sits relative to the multiply in the matched chain.
///
/// The cube's fix-pipe computes `clamp(ReLU(acc) * scale)` -- the activation is
/// a *pre*-quant stage reading the raw accumulator, with the destination clamp
/// last. That was measured on a2a3 by `acc_to_gm_negative_scale` in
/// tests/st/runtime/ops/test_fixpipe_epilogue.py; pto-isa's CPU reference model
/// (`quantize_element`) applies it on the other side of the multiply, so do not
/// "correct" this against that model.
///
/// Two DSL spellings reach the same instruction:
///
///   A. `muls(maximums(acc, 0), s)` -- identical to the hardware for every `s`
///   B. `maximums(muls(acc, s), 0)` -- identical only for `s >= 0`
///
/// B is what users write (dequantize, then activate), which is why a sign guard
/// exists rather than a blanket rejection.
enum class ReluPosition { kNone, kBeforeScale, kAfterScale };

struct AccEpilogue {
  std::vector<AssignStmtPtr> chain;  ///< cast / muls / maximums statements to delete
  AssignStmtPtr store;               ///< the `tile.store` to rewrite in place
  VarPtr acc;                        ///< the accumulator the rewritten store reads
  std::optional<double> pre_quant;
  bool pre_relu = false;
};

/// Counts every *read* of every Var in a function body. The fold requires each
/// link of the chain to be consumed exactly once: a value read twice cannot be
/// deleted, because the second reader would lose its operand.
class UseCounter : public IRVisitor {
 public:
  std::unordered_map<const Var*, int> counts;

 protected:
  void VisitExpr_(const VarPtr& op) override { counts[op.get()] += 1; }
  void VisitExpr_(const IterArgPtr& op) override { counts[op.get()] += 1; }

  void VisitStmt_(const AssignStmtPtr& op) override {
    // The LHS is a definition, not a use -- visit only the value.
    if (op->value_) VisitExpr(op->value_);
  }
};

/// `true` for the cube matmul family, whose result is an Acc tile by
/// construction. Memory spaces are still unresolved here (`InferTileMemorySpace`
/// runs immediately after this pass), so operator identity -- not
/// `memory_space_` -- is what establishes Acc residency.
bool IsCubeMatmul(const CallPtr& call) {
  return IsOp(call, "tile.matmul") || IsOp(call, "tile.matmul_acc") || IsOp(call, "tile.matmul_bias");
}

/// A cast the fix-pipe can absorb, on the same terms as the existing unscaled
/// fold (`CastFoldableToFixpipeMat`, auto_tile_matmul_l0_pass.cpp) -- the two
/// must agree, since they compete for the same IR.
///
/// FIXPIPE's narrowing is round-half-to-**even**, i.e. `RINT`. The frontend
/// default is `ROUND` (round-half-*away*, pto-isa `CAST_ROUND`), which breaks
/// ties the other way, so a default-written cast is deliberately NOT foldable:
/// only `pto.tcvt` honors the requested mode. `pto.tstore` likewise carries no
/// `satmode`, so a cast that asked for a specific destination saturation cannot
/// be reproduced -- an *absent* mode is the "did not ask" case and stays
/// foldable.
bool IsFoldableCast(const CallPtr& call) {
  constexpr int kModeNone = 0;
  constexpr int kModeRint = 1;
  constexpr int kModeRound = 2;
  const int mode = GetIntKwarg(call->kwargs_, "mode", kModeRound);
  if (mode != kModeNone && mode != kModeRint) return false;
  if (GetSaturationMode(call).has_value()) return false;
  // A second operand is the A2/A3 narrowing scratch, i.e. a multi-instruction
  // lowering rather than the one conversion the writeback performs.
  return call->args_.size() == 1;
}

/// `true` when a cast is foldable in every respect *except* its rounding mode.
/// That is the one rejection a caller can act on, so it earns a PerfHint.
bool IsCastBlockedOnlyByRoundMode(const CallPtr& call) {
  constexpr int kModeRint = 1;
  constexpr int kModeRound = 2;
  if (GetIntKwarg(call->kwargs_, "mode", kModeRound) == kModeRint) return false;
  return !GetSaturationMode(call).has_value() && call->args_.size() == 1;
}

/// The constant behind `tile.muls(x, <const>)` / `tile.maximums(x, <const>)`.
std::optional<double> ConstScalarOperand(const ExprPtr& expr) {
  if (auto c = As<ConstFloat>(expr)) return c->value_;
  if (auto c = As<ConstInt>(expr)) return static_cast<double>(c->value_);
  return std::nullopt;
}

/// A literal zero -- the only threshold that makes `tile.maximums` a ReLU.
bool IsZeroThreshold(const ExprPtr& expr) {
  auto value = ConstScalarOperand(expr);
  return value.has_value() && *value == 0.0;
}

/// The dtype a tile-producing assignment yields.
std::optional<DataType> AssignedTileDtype(const AssignStmtPtr& assign) {
  if (!assign || !assign->var_) return std::nullopt;
  auto tile = As<TileType>(assign->var_->GetType());
  if (!tile) return std::nullopt;
  return tile->dtype_;
}

/// Walks the single-use chain hanging off an accumulator and reports the
/// epilogue when the whole shape is foldable.
///
/// The walk is deliberately shallow: every link must be the *sole* consumer of
/// the previous one, so folding can never delete a value someone else reads.
class EpilogueMatcher {
 public:
  EpilogueMatcher(const std::unordered_map<const Var*, int>& use_counts,
                  const std::unordered_map<const Var*, AssignStmtPtr>& consumer,
                  const backend::BackendHandler& handler, std::vector<Diagnostic>* hints)
      : use_counts_(use_counts), consumer_(consumer), handler_(handler), hints_(hints) {}

  [[nodiscard]] std::optional<AccEpilogue> Match(const AssignStmtPtr& matmul) const {
    auto acc_dtype = AssignedTileDtype(matmul);
    if (!acc_dtype.has_value()) return std::nullopt;

    AccEpilogue result;
    result.acc = matmul->var_;

    const Var* cursor = matmul->var_.get();
    auto relu_pos = ReluPosition::kNone;

    // Form A's leading ReLU: exactly what the hardware performs.
    if (auto relu = MatchOp(cursor, "tile.maximums")) {
      auto call = As<Call>(relu->value_);
      if (call->args_.size() == 2 && IsZeroThreshold(call->args_[1])) {
        result.pre_relu = true;
        relu_pos = ReluPosition::kBeforeScale;
        result.chain.push_back(relu);
        cursor = relu->var_.get();
      }
    }

    // A widening cast to FP32 before the multiply: the fix-pipe multiplies in
    // FP32 regardless, so this is absorbed rather than reproduced.
    if (auto cast = MatchOp(cursor, "tile.cast")) {
      auto call = As<Call>(cast->value_);
      auto produced = AssignedTileDtype(cast);
      if (produced.has_value() && *produced == DataType::FP32 && IsFoldableCast(call)) {
        result.chain.push_back(cast);
        cursor = cast->var_.get();
      }
    }

    if (auto muls = MatchOp(cursor, "tile.muls")) {
      auto call = As<Call>(muls->value_);
      if (call->args_.size() != 2) return std::nullopt;
      auto scale = ConstScalarOperand(call->args_[1]);
      if (!scale.has_value()) {
        Hint("PH-FE-001", muls->span_,
             "fix-pipe epilogue not folded: the scale is not a compile-time constant, so it cannot be "
             "packed into the writeback's configuration word. Pass a Python float, or keep the "
             "multiply in the vector unit.");
        return std::nullopt;
      }
      result.pre_quant = scale;
      result.chain.push_back(muls);
      cursor = muls->var_.get();
    }

    // Form B: the ReLU sits after the multiply.
    if (relu_pos == ReluPosition::kNone) {
      if (auto relu = MatchOp(cursor, "tile.maximums")) {
        auto call = As<Call>(relu->value_);
        if (call->args_.size() == 2 && IsZeroThreshold(call->args_[1])) {
          result.pre_relu = true;
          relu_pos = ReluPosition::kAfterScale;
          result.chain.push_back(relu);
          cursor = relu->var_.get();
        }
      }
    }

    // The narrowing cast to the destination dtype -- the fix-pipe's own
    // conversion.
    if (auto cast = MatchOp(cursor, "tile.cast")) {
      auto call = As<Call>(cast->value_);
      if (!IsFoldableCast(call)) {
        if (IsCastBlockedOnlyByRoundMode(call)) {
          Hint("PH-FE-004", cast->span_,
               "fix-pipe epilogue not folded: the cube's writeback rounds half-to-even, but this "
               "cast uses the frontend default (round half away from zero), so folding it would "
               "change results at ties. Pass mode=\"rint\" on the cast to let the whole epilogue "
               "ride the writeback.");
        }
        return std::nullopt;
      }
      result.chain.push_back(cast);
      cursor = cast->var_.get();
    }

    if (result.chain.empty()) return std::nullopt;  // nothing to fold

    // `maximum(acc * s, 0)` matches the hardware's `maximum(acc, 0) * s` only
    // for a non-negative scale; for `s < 0` they disagree at every element.
    if (relu_pos == ReluPosition::kAfterScale && result.pre_quant.has_value() && *result.pre_quant < 0.0) {
      Hint("PH-FE-002", result.chain.back()->span_,
           "fix-pipe epilogue not folded: the cube applies ReLU to the accumulator *before* the "
           "scale, so maximum(tile * s, 0) with a negative s is not what the writeback would "
           "compute. Write maximum(tile, 0) * s if that is the intent.");
      return std::nullopt;
    }

    auto store = MatchOp(cursor, "tile.store");
    if (!store) return std::nullopt;
    auto store_call = As<Call>(store->value_);
    if (!store_call || store_call->args_.size() < 3) return std::nullopt;
    // An epilogue already on the store means the user wrote one by hand; the
    // two would have to be composed, which is not this pass's job.
    if (GetOptionalDoubleKwarg(store_call->kwargs_, "pre_quant").has_value()) return std::nullopt;
    if (store_call->GetKwarg<bool>("pre_relu", false)) return std::nullopt;

    auto out_type = AsTensorTypeLike(store_call->args_[2]->GetType());
    if (!out_type) return std::nullopt;

    if (!IsLegalWriteback(*acc_dtype, out_type->dtype_, result, store->span_)) return std::nullopt;

    result.store = store;
    return result;
  }

 private:
  /// The sole consumer of `value` when it is a top-level `lhs = <op_name>(...)`
  /// reading `value` as its first operand, else null.
  [[nodiscard]] AssignStmtPtr MatchOp(const Var* value, const std::string& op_name) const {
    if (value == nullptr) return nullptr;
    auto uses = use_counts_.find(value);
    if (uses == use_counts_.end() || uses->second != 1) return nullptr;
    auto it = consumer_.find(value);
    if (it == consumer_.end() || !it->second) return nullptr;
    auto call = As<Call>(it->second->value_);
    if (!call || !call->op_ || !IsOp(call, op_name)) return nullptr;
    // Position matters: `muls(x, s)` folds, `muls(s, x)` is a different
    // expression, and a store that merely *writes into* x is not a writeback.
    if (call->args_.empty() || AsVarLike(call->args_[0]).get() != value) return nullptr;
    return it->second;
  }

  /// The handler tables decide which `(accumulator, destination)` pairs the
  /// fix-pipe can perform. Folding a pair they reject would hand
  /// `AccToGmStoreValid` an IR it refuses, turning a program that compiles
  /// today into a compile error -- which an optimization must never do.
  [[nodiscard]] bool IsLegalWriteback(DataType acc_dtype, DataType out_dtype, const AccEpilogue& epi,
                                      const Span& span) const {
    if (epi.pre_quant.has_value()) {
      if (handler_.SupportsFixpipePreQuant(acc_dtype, out_dtype, backend::BackendHandler::FixpipeDest::kGm)) {
        return true;
      }
      Hint("PH-FE-003", span,
           "fix-pipe epilogue not folded: the '" + handler_.GetPtoTargetArch() +
               "' backend has no scale-bearing " + acc_dtype.ToString() + " -> " + out_dtype.ToString() +
               " writeback, so the multiply stays in the vector unit (per-backend table: "
               "docs/en/dev/passes/99-verifier.md).");
      return false;
    }
    // Without a scale the writeback performs exactly one conversion, and the
    // destination must be one the store pipe can reach at all.
    if (!handler_.SupportsAccToGmDtype(out_dtype)) return false;
    return out_dtype == acc_dtype || CubeWritebackSupportsDataType(acc_dtype, out_dtype);
  }

  void Hint(const std::string& code, const Span& span, const std::string& message) const {
    if (hints_ == nullptr) return;
    hints_->emplace_back(DiagnosticSeverity::PerfHint, kPassName, 0, code, message, span);
  }

  const std::unordered_map<const Var*, int>& use_counts_;
  const std::unordered_map<const Var*, AssignStmtPtr>& consumer_;
  const backend::BackendHandler& handler_;
  std::vector<Diagnostic>* hints_;
};

/// Rebuilds `tile.store(tail, offs, out, ...)` as a writeback reading the
/// accumulator directly and carrying the epilogue as kwargs. Every kwarg the
/// original store had (`atomic`, `st_phase`, ...) is preserved.
CallPtr BuildFoldedStore(const CallPtr& original, const AccEpilogue& epi) {
  auto args = original->args_;
  args[0] = epi.acc;
  auto kwargs = original->kwargs_;
  if (epi.pre_quant.has_value()) kwargs.emplace_back("pre_quant", std::any(*epi.pre_quant));
  if (epi.pre_relu) kwargs.emplace_back("pre_relu", std::any(true));
  // The result type is the destination tensor and does not change -- only the
  // source operand and the epilogue kwargs do. `attrs_` carries compiler
  // metadata (arg directions) that must survive the rewrite.
  return std::make_shared<Call>(original->op_, args, kwargs, original->attrs_, original->GetType(),
                                original->span_);
}

class FoldMutator : public IRMutator {
 public:
  FoldMutator(std::unordered_map<const AssignStmt*, AccEpilogue> folds,
              std::unordered_set<const AssignStmt*> dropped)
      : folds_(std::move(folds)), dropped_(std::move(dropped)) {}

 protected:
  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    auto visited = IRMutator::VisitStmt_(op);
    auto seq = As<SeqStmts>(visited);
    if (!seq) return visited;
    std::vector<StmtPtr> kept;
    kept.reserve(seq->stmts_.size());
    bool changed = false;
    for (const auto& stmt : seq->stmts_) {
      auto assign = As<AssignStmt>(stmt);
      if (assign && dropped_.count(assign.get()) != 0) {
        changed = true;
        continue;
      }
      kept.push_back(stmt);
    }
    if (!changed) return visited;
    return std::make_shared<SeqStmts>(std::move(kept), seq->span_);
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto it = folds_.find(op.get());
    if (it == folds_.end()) return IRMutator::VisitStmt_(op);
    auto call = As<Call>(op->value_);
    if (!call) return IRMutator::VisitStmt_(op);
    return std::make_shared<AssignStmt>(op->var_, BuildFoldedStore(call, it->second), op->span_);
  }

 private:
  std::unordered_map<const AssignStmt*, AccEpilogue> folds_;
  std::unordered_set<const AssignStmt*> dropped_;
};

/// Indexes, per Var, the top-level AssignStmt that reads it, plus every cube
/// matmul in the body. A Var read by several statements keeps only the last one
/// here; the matcher's use-count check rejects it before that matters.
class ConsumerIndex : public IRVisitor {
 public:
  std::unordered_map<const Var*, AssignStmtPtr> consumer;
  std::vector<AssignStmtPtr> matmuls;

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    auto call = As<Call>(op->value_);
    if (call && call->op_) {
      if (IsCubeMatmul(call)) matmuls.push_back(op);
      for (const auto& arg : call->args_) {
        if (auto var = AsVarLike(arg)) consumer[var.get()] = op;
      }
    }
    IRVisitor::VisitStmt_(op);
  }
};

FunctionPtr TransformFunction(const FunctionPtr& func, std::vector<Diagnostic>* hints) {
  if (!func || !func->body_) return func;
  if (!IsInCoreType(func->func_type_)) return func;

  // Which `(accumulator, destination)` pairs the fix-pipe can perform is a
  // backend fact. Several codegen tests drive passes with no backend configured;
  // `GetBackendHandler()` *throws* in that state rather than returning null, and
  // guessing a profile could fold a pair the real target cannot perform.
  if (!backend::BackendConfig::IsConfigured()) return func;
  const auto* ctx = PassContext::Current();
  const auto* handler =
      ctx != nullptr ? ctx->GetBackendHandler() : backend::BackendConfig::GetBackend()->GetHandler();
  if (handler == nullptr) return func;

  UseCounter uses;
  uses.VisitStmt(func->body_);
  ConsumerIndex index;
  index.VisitStmt(func->body_);

  EpilogueMatcher matcher(uses.counts, index.consumer, *handler, hints);
  std::unordered_map<const AssignStmt*, AccEpilogue> folds;
  std::unordered_set<const AssignStmt*> dropped;
  for (const auto& matmul : index.matmuls) {
    auto epi = matcher.Match(matmul);
    if (!epi.has_value()) continue;
    folds.emplace(epi->store.get(), *epi);
    for (const auto& link : epi->chain) dropped.insert(link.get());
  }
  if (folds.empty()) return func;

  FoldMutator mutator(std::move(folds), std::move(dropped));
  auto new_body = mutator.VisitStmt(func->body_);
  if (new_body.get() == func->body_.get()) return func;
  return std::make_shared<Function>(func->name_, func->params_, func->param_directions_, func->return_types_,
                                    new_body, func->span_, func->func_type_, func->level_, func->role_,
                                    func->attrs_, func->requires_runtime_binding_, func->ir_stage_);
}

}  // namespace

Pass FoldFixpipeAccEpilogue() {
  auto pass_func = [](const FunctionPtr& func) -> FunctionPtr {
    std::vector<Diagnostic> hints;
    auto result = TransformFunction(func, &hints);
    if (!hints.empty()) EmitDiagnostics(hints, kPassName);
    return result;
  };
  return CreateFunctionPass(pass_func, kPassName, kFoldFixpipeAccEpilogueProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
