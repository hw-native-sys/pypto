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

/**
 * @file insert_mx_scale_addr_pass.cpp
 * @brief Insert tile.tget_scale_addr bindings before MX matmul consumers.
 *
 * Runs after InferTileMemorySpace so Left/LeftScale and Right/RightScale pairs
 * are already resolved. Because tget_scale_addr mutates shared physical scale
 * buffers in place, bindings are never reused across MX matmul consumers.
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {
namespace {

// Facts are collected once before rewriting, so exclusive staging and the
// immediate four-move bundle can be checked without rescanning whole functions.
// Requiring the bundle is important: an explicit earlier move is a snapshot,
// and delaying its reads past an unrelated effect could change that snapshot.
class MxL0Facts : public IRVisitor {
 public:
  std::unordered_map<const Var*, AssignStmtPtr> defs;
  std::unordered_map<const Var*, size_t> uses;
  std::unordered_set<const Var*> l0_vars;
  std::unordered_map<const AssignStmt*, std::vector<StmtPtr>> predecessors;
  std::vector<AssignStmtPtr> matmuls;

 protected:
  void VisitVarLike_(const VarPtr& var) override {
    ++uses[var.get()];
    RecordL0(var);
  }
  void VisitStmt_(const AssignStmtPtr& stmt) override {
    defs.emplace(stmt->var_.get(), stmt);
    RecordL0(stmt->var_);
    if (auto call = As<Call>(stmt->value_);
        call && (IsOp(call, "tile.matmul_mx") || IsOp(call, "tile.matmul_mx_acc"))) {
      matmuls.push_back(stmt);
    }
    VisitExpr(stmt->value_);  // The LHS is a definition, not a read.
  }
  void VisitStmt_(const SeqStmtsPtr& seq) override {
    for (size_t i = 0; i < seq->stmts_.size(); ++i) {
      if (auto assign = As<AssignStmt>(seq->stmts_[i]); assign && i >= 4) {
        predecessors.emplace(assign.get(),
                             std::vector<StmtPtr>(seq->stmts_.begin() + i - 4, seq->stmts_.begin() + i));
      }
      VisitStmt(seq->stmts_[i]);
    }
  }

 private:
  void RecordL0(const VarPtr& var) {
    auto tile = As<TileType>(var->GetType());
    if (!tile) return;
    auto space = tile->GetMemorySpace();
    if (space == MemorySpace::Left || space == MemorySpace::Right || space == MemorySpace::LeftScale ||
        space == MemorySpace::RightScale) {
      l0_vars.insert(var.get());
    }
  }
};

struct MxL0Plan {
  std::array<VarPtr, 4> sources;  // A, A-scale, B, B-scale, all unchanged Mat tiles.
  std::array<AssignStmtPtr, 4> staging;
  int64_t rows;
};

bool IsFullMxTile(const ExprPtr& value, int64_t rows, int64_t cols, DataType dtype) {
  auto tile = As<TileType>(value->GetType());
  if (!tile || tile->dtype_ != dtype || tile->shape_.size() != 2) return false;
  auto valid = GetValidShape(tile);
  for (size_t axis = 0; axis < 2; ++axis) {
    auto physical_dim = As<ConstInt>(tile->shape_[axis]);
    auto valid_dim = As<ConstInt>(valid[axis]);
    const int64_t expected = axis == 0 ? rows : cols;
    if (!physical_dim || !valid_dim || physical_dim->value_ != expected || valid_dim->value_ != expected) {
      return false;
    }
  }
  return true;
}

std::optional<MxL0Plan> AnalyzeMxL0Plan(const AssignStmtPtr& assign, const MxL0Facts& facts) {
  auto call = As<Call>(assign->value_);
  const size_t first = IsOp(call, "tile.matmul_mx_acc") ? 1 : 0;
  if (call->args_.size() != first + 4) return std::nullopt;
  auto lhs = As<TileType>(call->args_[first]->GetType());
  auto m = lhs && lhs->shape_.size() == 2 ? As<ConstInt>(lhs->shape_[0]) : nullptr;
  // This opt-in first implementation deliberately keeps the proven full-tile
  // geometry. Dynamic/tail K, different N, explicit L0 operands and bias keep
  // the original lowering and its original numerical behavior.
  if (!m || (m->value_ != 48 && m->value_ != 64)) return std::nullopt;
  const std::array<int64_t, 4> rows = {m->value_, m->value_, 256, 8};
  const std::array<int64_t, 4> cols = {256, 8, 256, 256};
  const std::array<MemorySpace, 4> spaces = {MemorySpace::Left, MemorySpace::LeftScale, MemorySpace::Right,
                                             MemorySpace::RightScale};
  MxL0Plan plan{{}, {}, m->value_};
  auto preceding = facts.predecessors.find(assign.get());
  if (preceding == facts.predecessors.end()) return std::nullopt;
  std::unordered_set<const Stmt*> bundle;
  for (const auto& stmt : preceding->second) bundle.insert(stmt.get());
  for (size_t i = 0; i < 4; ++i) {
    auto operand = AsVarLike(call->args_[first + i]);
    const DataType dtype = i % 2 == 0 ? DataType::FP8E4M3FN : DataType::FP8E8M0;
    if (!operand || !IsFullMxTile(operand, rows[i], cols[i], dtype)) return std::nullopt;
    auto def = facts.defs.find(operand.get());
    auto uses = facts.uses.find(operand.get());
    if (def == facts.defs.end() || uses == facts.uses.end() || uses->second != 1 ||
        !bundle.count(def->second.get()))
      return std::nullopt;
    auto move = As<Call>(def->second->value_);
    if (!move || !IsOp(move, "tile.move") || move->args_.size() != 1 ||
        move->GetKwarg<MemorySpace>("target_memory") != spaces[i])
      return std::nullopt;
    auto source = AsVarLike(move->args_[0]);
    auto source_type = source ? As<TileType>(source->GetType()) : nullptr;
    if (!source_type || source_type->GetMemorySpace() != MemorySpace::Mat ||
        !IsFullMxTile(source, rows[i], cols[i], dtype))
      return std::nullopt;
    // An author-bound destination is an explicit storage contract, not a
    // compiler staging temporary we can replace with our own slot ownership.
    auto operand_type = As<TileType>(operand->GetType());
    if (operand_type->memref_.has_value()) return std::nullopt;
    plan.sources[i] = source;
    plan.staging[i] = def->second;
  }
  return plan;
}

class MxL0PingPongLowerer : public IRMutator {
 public:
  explicit MxL0PingPongLowerer(std::unordered_map<const AssignStmt*, MxL0Plan> plans)
      : plans_(std::move(plans)) {
    for (const auto& [assign, plan] : plans_) {
      (void)assign;
      for (const auto& staging : plan.staging) dropped_.insert(staging.get());
    }
  }

  StmtPtr Rewrite(const StmtPtr& body) { return VisitStmt(body); }

 protected:
  StmtPtr VisitStmt_(const SeqStmtsPtr& seq) override {
    std::vector<StmtPtr> stmts;
    for (const auto& stmt : seq->stmts_) {
      if (!dropped_.count(stmt.get())) stmts.push_back(VisitStmt(stmt));
    }
    return SeqStmts::Flatten(std::move(stmts), seq->span_);
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& assign) override {
    auto found = plans_.find(assign.get());
    if (found == plans_.end()) return IRMutator::VisitStmt_(assign);
    const auto& plan = found->second;
    auto call = As<Call>(assign->value_);
    auto& registry = OpRegistry::GetInstance();
    const bool accumulate = IsOp(call, "tile.matmul_mx_acc");
    std::vector<StmtPtr> stmts;
    ExprPtr accumulator = accumulate ? call->args_[0] : nullptr;
    // Preserve increasing K and the original accumulator chain: the first
    // output uses exactly the original fresh/accumulate operation, and the
    // second half always accumulates into that FP32 result.
    for (size_t slot = 0; slot < 2; ++slot) {
      std::vector<ExprPtr> args;
      if (accumulator) args.push_back(accumulator);
      const std::array<int64_t, 4> rows = {plan.rows, plan.rows, 128, 4};
      const std::array<int64_t, 4> cols = {128, 4, 256, 256};
      const std::array<MemorySpace, 4> spaces = {MemorySpace::Left, MemorySpace::LeftScale,
                                                 MemorySpace::Right, MemorySpace::RightScale};
      for (size_t i = 0; i < 4; ++i) {
        const int64_t offset = static_cast<int64_t>(slot) * (i % 2 == 0 ? 128 : 4);
        const int64_t row = i >= 2 ? offset : 0;
        const int64_t col = i < 2 ? offset : 0;
        auto extract = registry.Create("tile.extract",
                                       {plan.sources[i], Index(row, assign->span_), Index(col, assign->span_),
                                        Shape(rows[i], cols[i], assign->span_)},
                                       {{"target_memory", spaces[i]}}, assign->span_);
        auto tile_type = As<TileType>(extract->GetType());
        if (i == 0) {
          // tile.extract's generic deducer currently stamps A2/A3 L0A RR.
          // This rewrite is A5-only, whose TEXTRACT verifier/ISA require CR.
          // Correct both the call result and its bound destination annotation.
          auto view = tile_type->tile_view_.value_or(TileView{});
          view.blayout = TileLayout::col_major;
          view.slayout = TileLayout::row_major;
          tile_type =
              std::make_shared<TileType>(tile_type->shape_, tile_type->dtype_, std::nullopt, view, spaces[i]);
          auto extract_call = As<Call>(extract);
          extract = std::make_shared<Call>(extract_call->op_, extract_call->args_, extract_call->kwargs_,
                                           extract_call->attrs_, tile_type, extract_call->span_);
        }
        // Pipeline clones deliberately reuse these physical slots. Keep their
        // extracts in compute-stage order: clustering stage1's L0 copies before
        // stage0's MMAD would overwrite stage0's still-live pinned buffers.
        // Ordinary L1 loads retain the pipeline's prefetch scheduling.
        auto extract_call = As<Call>(extract);
        auto extract_attrs = extract_call->attrs_;
        extract_attrs.emplace_back("mx_l0_ping_pong", true);
        extract =
            std::make_shared<Call>(extract_call->op_, extract_call->args_, extract_call->kwargs_,
                                   std::move(extract_attrs), extract_call->GetType(), extract_call->span_);
        auto base = SlotBase(i, slot, assign->span_);
        auto binding = std::make_shared<MemRef>(base, 0, 0, assign->span_, /*is_pinned=*/true);
        auto bound_type = std::make_shared<TileType>(tile_type->shape_, tile_type->dtype_, binding,
                                                     tile_type->tile_view_, spaces[i]);
        auto var = std::make_shared<Var>(
            assign->var_->name_hint_ + "_mx_half" + std::to_string(slot) + "_operand" + std::to_string(i),
            bound_type, assign->span_);
        stmts.push_back(std::make_shared<AssignStmt>(var, extract, assign->span_));
        args.push_back(var);
      }
      const std::string op = accumulator ? "tile.matmul_mx_acc" : "tile.matmul_mx";
      auto half = registry.Create(op, args, call->kwargs_, call->span_);
      auto typed_half = As<Call>(half);
      auto attributed_half = std::make_shared<Call>(typed_half->op_, typed_half->args_, typed_half->kwargs_,
                                                    call->attrs_, typed_half->GetType(), call->span_);
      VarPtr result = slot == 1 ? assign->var_
                                : std::make_shared<Var>(assign->var_->name_hint_ + "_mx_first_half",
                                                        half->GetType(), assign->span_);
      stmts.push_back(std::make_shared<AssignStmt>(result, attributed_half, assign->span_));
      accumulator = result;
    }
    // These two extracts remain reads of the SAME K256 Mat/FIFO source. Mixed
    // kernel finalization therefore places its one TFREE after the second B
    // extract; no FIFO pop/free protocol is synthesized by this lowering.
    return SeqStmts::Flatten(std::move(stmts), assign->span_);
  }

 private:
  static ExprPtr Index(int64_t value, const Span& span) {
    return std::make_shared<ConstInt>(value, DataType::INDEX, span);
  }
  static ExprPtr Shape(int64_t rows, int64_t cols, const Span& span) {
    return std::make_shared<MakeTuple>(std::vector<ExprPtr>{Index(rows, span), Index(cols, span)}, span);
  }
  VarPtr SlotBase(size_t operand, size_t slot, const Span& span) {
    auto& base = bases_[operand][slot];
    if (base) return base;
    base = std::make_shared<Var>("mx_l0_operand" + std::to_string(operand) + "_slot" + std::to_string(slot),
                                 GetPtrType(), span);
    // Each live extract declares its own physical capacity: M*128 bytes for
    // A and 32768 for B. Scale bindings derive from the allocated data address,
    // so A does not need an otherwise unused oversized tile.create reservation.
    return base;
  }

  std::unordered_map<const AssignStmt*, MxL0Plan> plans_;
  std::unordered_set<const Stmt*> dropped_;
  std::array<std::array<VarPtr, 2>, 4> bases_{};
};

StmtPtr LowerMxL0PingPong(const FunctionPtr& function) {
  const auto* context = PassContext::Current();
  if (!context || !context->GetEnableMxL0PingPong()) return function->body_;
  // The fixed-slot proof relies on PyPTO/DSA-RP preserving these independent
  // pinned allocations. PTOAS currently rejects single-slot declarations and
  // has no low-precision Left/Right/Scaling multi-tile region representation.
  // Keep its existing MX lowering instead of generating unsupported storage.
  if (context->GetMemoryPlanner() == MemoryPlanner::PtoAS) return function->body_;
  const auto* handler = context->GetBackendHandler();
  if (!handler || handler->GetPtoTargetArch() != "a5") return function->body_;
  MxL0Facts facts;
  facts.VisitStmt(function->body_);
  std::unordered_map<const AssignStmt*, MxL0Plan> plans;
  std::unordered_set<const Var*> staged;
  for (const auto& assign : facts.matmuls) {
    if (auto plan = AnalyzeMxL0Plan(assign, facts)) {
      for (const auto& move : plan->staging) staged.insert(move->var_.get());
      plans.emplace(assign.get(), std::move(*plan));
    }
  }
  if (plans.empty()) return function->body_;
  // Two B halves fill L0B. A function with other L0 consumers must
  // keep its old allocation plan, including explicit user-bound operands.
  for (const auto* var : facts.l0_vars) {
    if (!staged.count(var)) return function->body_;
  }
  for (const auto& param : function->params_) {
    auto tile = As<TileType>(param->GetType());
    if (tile && (tile->GetMemorySpace() == MemorySpace::Left || tile->GetMemorySpace() == MemorySpace::Right))
      return function->body_;
  }
  return MxL0PingPongLowerer(std::move(plans)).Rewrite(function->body_);
}

bool IsResolvedMxScaleDataPair(const VarPtr& scale, const VarPtr& data) {
  auto scale_type = As<TileType>(scale->GetType());
  auto data_type = As<TileType>(data->GetType());
  if (!scale_type || !data_type) return false;
  auto scale_space = scale_type->GetMemorySpace();
  auto data_space = data_type->GetMemorySpace();
  if (!scale_space.has_value() || !data_space.has_value()) return false;
  const bool is_left = *scale_space == MemorySpace::LeftScale && *data_space == MemorySpace::Left;
  const bool is_right = *scale_space == MemorySpace::RightScale && *data_space == MemorySpace::Right;
  return is_left || is_right;
}

class MxScaleAddrInserter : public IRMutator {
 public:
  // Entry / control-flow body helper: NormalizedStmtStructure unwraps single-child
  // SeqStmts, so if/for/while (and even the function body) may be a bare AssignStmt.
  // SeqStmts bodies use VisitStmt_(SeqStmts); bare MX matmul assigns are wrapped
  // into a SeqStmts with the inserted bindings (same pattern as InsertCommFence).
  StmtPtr RewriteBody(const StmtPtr& body) {
    if (As<SeqStmts>(body)) return VisitStmt(body);

    auto visited = VisitStmt(body);
    auto assign = As<AssignStmt>(visited);
    auto call = assign ? As<Call>(assign->value_) : nullptr;
    if (!assign || !call || GetDataScalePairs(call).empty()) return visited;

    bool changed = false;
    std::vector<StmtPtr> new_stmts;
    auto rewritten = InsertBindingsForMxMatmul(new_stmts, assign, call, changed);
    new_stmts.push_back(rewritten);
    return SeqStmts::Flatten(std::move(new_stmts), body->span_);
  }

 protected:
  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    bool changed = false;
    std::vector<StmtPtr> new_stmts;
    new_stmts.reserve(op->stmts_.size());

    for (const auto& stmt : op->stmts_) {
      auto rewritten = IRMutator::VisitStmt(stmt);
      if (rewritten.get() != stmt.get()) changed = true;

      auto assign = As<AssignStmt>(rewritten);
      auto call = assign ? As<Call>(assign->value_) : nullptr;
      if (assign && call) {
        rewritten = InsertBindingsForMxMatmul(new_stmts, assign, call, changed);
      }
      new_stmts.push_back(rewritten);
    }

    if (!changed) return op;
    return SeqStmts::Flatten(std::move(new_stmts), op->span_);
  }

  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    auto new_then = RewriteBody(op->then_body_);
    std::optional<StmtPtr> new_else = op->else_body_;
    if (op->else_body_.has_value()) new_else = RewriteBody(op->else_body_.value());
    const bool then_changed = new_then.get() != op->then_body_.get();
    const bool else_changed = op->else_body_.has_value() && new_else->get() != op->else_body_->get();
    if (!then_changed && !else_changed) return op;
    auto result = MutableCopy(op);
    result->then_body_ = std::move(new_then);
    result->else_body_ = std::move(new_else);
    return result;
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override { return VisitLoop(op, op->body_); }
  StmtPtr VisitStmt_(const WhileStmtPtr& op) override { return VisitLoop(op, op->body_); }

 private:
  static std::vector<std::pair<size_t, size_t>> GetDataScalePairs(const CallPtr& call) {
    if (IsOp(call, "tile.matmul_mx") || IsOp(call, "tile.matmul_mx_bias")) {
      return {{0, 1}, {2, 3}};
    }
    if (IsOp(call, "tile.matmul_mx_acc")) {
      return {{1, 2}, {3, 4}};
    }
    return {};
  }

  template <typename LoopPtr>
  StmtPtr VisitLoop(const LoopPtr& op, const StmtPtr& body) {
    auto new_body = RewriteBody(body);
    if (new_body.get() == body.get()) return op;
    auto result = MutableCopy(op);
    result->body_ = std::move(new_body);
    return result;
  }

  StmtPtr InsertBindingsForMxMatmul(std::vector<StmtPtr>& new_stmts, const AssignStmtPtr& assign,
                                    const CallPtr& call, bool& changed) {
    const auto pairs = GetDataScalePairs(call);
    if (pairs.empty()) return assign;

    if (backend::BackendConfig::IsConfigured()) {
      const auto* pass_context = PassContext::Current();
      const auto* handler = pass_context ? pass_context->GetBackendHandler()
                                         : backend::BackendConfig::GetBackend()->GetHandler();
      CHECK_SPAN(handler->GetPtoTargetArch() == "a5", call->span_)
          << call->op_->name_ << " is only supported on the Ascend950 ('a5') backend, but got '"
          << handler->GetPtoTargetArch() << "'";
    }

    std::vector<ExprPtr> new_args = call->args_;
    auto& registry = OpRegistry::GetInstance();
    bool call_changed = false;

    for (const auto& [data_index, scale_index] : pairs) {
      INTERNAL_CHECK_SPAN(data_index < new_args.size() && scale_index < new_args.size(), call->span_)
          << "Internal error: malformed " << call->op_->name_ << " operand list";
      auto data = AsVarLike(new_args[data_index]);
      auto scale = AsVarLike(new_args[scale_index]);
      INTERNAL_CHECK_SPAN(data && scale, call->span_)
          << "Internal error: " << call->op_->name_
          << " MX data/scale operands must be Var-like (Var or IterArg)";
      INTERNAL_CHECK_SPAN(IsResolvedMxScaleDataPair(scale, data), call->span_)
          << "Internal error: InsertMxScaleAddr requires resolved LeftScale↔Left or "
             "RightScale↔Right pairing before inserting tile.tget_scale_addr (run "
             "InferTileMemorySpace first)";

      auto binding_call = registry.Create("tile.tget_scale_addr", {scale, data}, {}, call->span_);
      auto bound_scale =
          std::make_shared<Var>(scale->name_hint_ + "_bound", binding_call->GetType(), call->span_);
      new_stmts.push_back(std::make_shared<AssignStmt>(bound_scale, binding_call, call->span_));
      new_args[scale_index] = bound_scale;
      call_changed = true;
      changed = true;
    }

    if (!call_changed) return assign;
    auto deduced = registry.Create(call->op_->name_, new_args, call->kwargs_, call->span_);
    auto rebound_call = std::make_shared<Call>(deduced->op_, deduced->args_, deduced->kwargs_, call->attrs_,
                                               deduced->GetType(), deduced->span_);
    return std::make_shared<AssignStmt>(assign->var_, rebound_call, assign->span_);
  }
};

FunctionPtr TransformInsertMxScaleAddr(const FunctionPtr& func) {
  MxScaleAddrInserter inserter;
  auto new_body = inserter.RewriteBody(LowerMxL0PingPong(func));
  if (new_body.get() == func->body_.get()) return func;
  auto updated = MutableCopy(func);
  updated->body_ = new_body;
  return updated;
}

}  // namespace

namespace pass {

Pass InsertMxScaleAddr() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> new_functions;
    for (const auto& [gvar, func] : program->functions_) {
      if (IsInCoreType(func->func_type_)) {
        new_functions[gvar] = TransformInsertMxScaleAddr(func);
      } else {
        new_functions[gvar] = func;
      }
    }
    return std::make_shared<Program>(std::move(new_functions), program->name_, program->span_);
  };
  return CreateProgramPass(pass_func, "InsertMxScaleAddr", kInsertMxScaleAddrProperties);
}

}  // namespace pass

}  // namespace ir
}  // namespace pypto
