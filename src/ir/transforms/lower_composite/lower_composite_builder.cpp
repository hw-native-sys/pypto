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

#include "src/ir/transforms/lower_composite/lower_composite_builder.h"

#include <any>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/comm.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/transforms/utils/tile_conversion_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace lower_composite {

namespace {

/// Safe negation: folds ConstInt(-value) directly when possible so the
/// PyPTO printer->parser roundtrip (which folds ``Neg(ConstInt)`` into
/// ``ConstInt(-value)``) produces structurally equal IR. Runtime
/// expressions are still wrapped with ``Neg``. File-local: the only callers
/// are EmitEpilogueReset overloads below.
ExprPtr MakeNegation(const ExprPtr& value) {
  if (auto c = As<ConstInt>(value)) {
    return std::make_shared<ConstInt>(-c->value_, GetScalarDtype(value), value->span_);
  }
  return MakeNeg(value, value->span_);
}

}  // namespace

LoweringBuilder::LoweringBuilder(std::string base_name, std::size_t& temp_counter)
    : base_name_(std::move(base_name)), temp_counter_(temp_counter) {}

LoweringBuilder::LoweringBuilder(std::string base_name, std::size_t& temp_counter, bool nested)
    : base_name_(std::move(base_name)), temp_counter_(temp_counter), nested_(nested) {}

ExprPtr LoweringBuilder::Bind(const std::string& qualifier, const ExprPtr& expr, const Span& span) {
  auto var = std::make_shared<Var>(MakeTempName(qualifier), expr->GetType(), span);
  stmts_.push_back(std::make_shared<AssignStmt>(var, expr, span));
  return var;
}

void LoweringBuilder::EmitEval(const ExprPtr& expr, const Span& span) {
  stmts_.push_back(std::make_shared<EvalStmt>(expr, span));
}

ExprPtr LoweringBuilder::Muls(const ExprPtr& x, float c, const Span& span) {
  auto tile_type = As<TileType>(x->GetType());
  INTERNAL_CHECK_SPAN(tile_type, span) << "tile.muls input must be TileType";
  auto scalar = std::make_shared<ConstFloat>(static_cast<double>(c), tile_type->dtype_, span);
  return OpRegistry::GetInstance().Create("tile.muls", {x, scalar}, {}, span);
}

ExprPtr LoweringBuilder::Adds(const ExprPtr& x, float c, const Span& span) {
  auto tile_type = As<TileType>(x->GetType());
  INTERNAL_CHECK_SPAN(tile_type, span) << "tile.adds input must be TileType";
  auto scalar = std::make_shared<ConstFloat>(static_cast<double>(c), tile_type->dtype_, span);
  return OpRegistry::GetInstance().Create("tile.adds", {x, scalar}, {}, span);
}

ExprPtr LoweringBuilder::Add(const ExprPtr& a, const ExprPtr& b, const Span& span) {
  return OpRegistry::GetInstance().Create("tile.add", {a, b}, {}, span);
}

ExprPtr LoweringBuilder::Sub(const ExprPtr& a, const ExprPtr& b, const Span& span) {
  return OpRegistry::GetInstance().Create("tile.sub", {a, b}, {}, span);
}

ExprPtr LoweringBuilder::Mul(const ExprPtr& a, const ExprPtr& b, const Span& span) {
  return OpRegistry::GetInstance().Create("tile.mul", {a, b}, {}, span);
}

ExprPtr LoweringBuilder::Reduce(ReduceOp op, const ExprPtr& a, const ExprPtr& b, const Span& span) {
  const char* op_name;
  switch (op) {
    case ReduceOp::kSum:
      op_name = "tile.add";
      break;
    case ReduceOp::kMax:
      op_name = "tile.maximum";
      break;
    case ReduceOp::kMin:
      op_name = "tile.minimum";
      break;
    case ReduceOp::kProd:
      op_name = "tile.mul";
      break;
    default:
      INTERNAL_CHECK_SPAN(false, span)
          << "pld.tensor.allreduce lowering received unknown ReduceOp " << static_cast<int>(op);
  }
  return OpRegistry::GetInstance().Create(op_name, {a, b}, {}, span);
}

ExprPtr LoweringBuilder::Cast(const ExprPtr& x, DataType to, int mode, const Span& span) {
  std::vector<std::pair<std::string, std::any>> kw = {{"target_type", to}, {"mode", mode}};
  return OpRegistry::GetInstance().Create("tile.cast", {x}, kw, span);
}

ExprPtr LoweringBuilder::NotEq(const ExprPtr& left, const ExprPtr& right, const Span& span) {
  return MakeNe(left, right, span);
}

ExprPtr LoweringBuilder::Gt(const ExprPtr& left, const ExprPtr& right, const Span& span) {
  return MakeGt(left, right, span);
}

CommSetup LoweringBuilder::EmitCommSetup(const ExprPtr& comm_target, const Span& span) {
  auto& reg = OpRegistry::GetInstance();
  CommSetup s;
  s.ctx = Bind("ctx", reg.Create("pld.system.get_comm_ctx", {comm_target}, {}, span), span);
  s.nranks_i32 = Bind("nranks", reg.Create("pld.system.nranks", {s.ctx}, {}, span), span);
  s.nranks_idx = Bind("nranks_idx", std::make_shared<ir::Cast>(s.nranks_i32, DataType::INDEX, span), span);
  s.my_rank = Bind("my_rank", reg.Create("pld.system.rank", {s.ctx}, {}, span), span);
  return s;
}

void LoweringBuilder::EmitNotifyAll(const ExprPtr& signal, const ExprPtr& nranks_idx, const ExprPtr& my_rank,
                                    NotifyOp notify_op, const ExprPtr& value, const std::string& suffix,
                                    const Span& span) {
  auto zero_idx = std::make_shared<ConstInt>(0, DataType::INDEX, span);
  auto one_idx = std::make_shared<ConstInt>(1, DataType::INDEX, span);
  auto my_offsets = tile_conversion_utils::MakeSignalOffsets(my_rank, span);

  EmitFor(
      "peer" + suffix, zero_idx, nranks_idx, one_idx,
      [&](LoweringBuilder& body, const VarPtr& peer) {
        body.EmitIf(
            body.NotEq(peer, my_rank, span),
            [&](LoweringBuilder& then_body) {
              auto call =
                  OpRegistry::GetInstance().Create("pld.system.notify", {signal, peer, my_offsets, value},
                                                   {{"op", static_cast<int>(notify_op)}}, span);
              then_body.Bind("notify" + suffix + "_ret", call, span);
            },
            /*else_fn=*/nullptr, span);
      },
      span);
}

void LoweringBuilder::EmitNotifyAll(const ExprPtr& signal, const ExprPtr& nranks_idx, const ExprPtr& my_rank,
                                    const ExprPtr& row_offset, NotifyOp notify_op, const ExprPtr& value,
                                    const std::string& suffix, const Span& span) {
  auto zero_idx = std::make_shared<ConstInt>(0, DataType::INDEX, span);
  auto one_idx = std::make_shared<ConstInt>(1, DataType::INDEX, span);
  auto my_offsets = tile_conversion_utils::MakeSignalOffsets(my_rank, row_offset, span);

  EmitFor(
      "peer" + suffix, zero_idx, nranks_idx, one_idx,
      [&](LoweringBuilder& body, const VarPtr& peer) {
        body.EmitIf(
            body.NotEq(peer, my_rank, span),
            [&](LoweringBuilder& then_body) {
              auto call =
                  OpRegistry::GetInstance().Create("pld.system.notify", {signal, peer, my_offsets, value},
                                                   {{"op", static_cast<int>(notify_op)}}, span);
              then_body.Bind("notify" + suffix + "_ret", call, span);
            },
            /*else_fn=*/nullptr, span);
      },
      span);
}

void LoweringBuilder::EmitWaitAll(const ExprPtr& signal, const ExprPtr& nranks_idx, const ExprPtr& my_rank,
                                  const ExprPtr& expected, const std::string& suffix, const Span& span) {
  auto zero_idx = std::make_shared<ConstInt>(0, DataType::INDEX, span);
  auto one_idx = std::make_shared<ConstInt>(1, DataType::INDEX, span);

  EmitFor(
      "src" + suffix, zero_idx, nranks_idx, one_idx,
      [&](LoweringBuilder& body, const VarPtr& src) {
        auto src_offsets = tile_conversion_utils::MakeSignalOffsets(src, span);
        body.EmitIf(
            body.NotEq(src, my_rank, span),
            [&](LoweringBuilder& then_body) {
              auto call = OpRegistry::GetInstance().Create("pld.system.wait", {signal, src_offsets, expected},
                                                           {{"cmp", static_cast<int>(WaitCmp::kGe)}}, span);
              then_body.Bind("wait" + suffix + "_ret", call, span);
            },
            /*else_fn=*/nullptr, span);
      },
      span);
}

void LoweringBuilder::EmitWaitAll(const ExprPtr& signal, const ExprPtr& nranks_idx, const ExprPtr& my_rank,
                                  const ExprPtr& row_offset, const ExprPtr& expected,
                                  const std::string& suffix, const Span& span) {
  auto zero_idx = std::make_shared<ConstInt>(0, DataType::INDEX, span);
  auto one_idx = std::make_shared<ConstInt>(1, DataType::INDEX, span);

  EmitFor(
      "src" + suffix, zero_idx, nranks_idx, one_idx,
      [&](LoweringBuilder& body, const VarPtr& src) {
        auto src_offsets = tile_conversion_utils::MakeSignalOffsets(src, row_offset, span);
        body.EmitIf(
            body.NotEq(src, my_rank, span),
            [&](LoweringBuilder& then_body) {
              auto call = OpRegistry::GetInstance().Create("pld.system.wait", {signal, src_offsets, expected},
                                                           {{"cmp", static_cast<int>(WaitCmp::kGe)}}, span);
              then_body.Bind("wait" + suffix + "_ret", call, span);
            },
            /*else_fn=*/nullptr, span);
      },
      span);
}

int64_t LoweringBuilder::EmitBarrier(const ExprPtr& signal, const CommSetup& comm, const std::string& suffix,
                                     const Span& span) {
  INTERNAL_CHECK_SPAN(!nested_, span)
      << "Internal error: EmitBarrier must only be called from a top-level lowering rule, not from inside "
      << "EmitFor / EmitIf / EmitIfExpr bodies. Loop- or condition-resident barriers must "
      << "emit notify/wait by hand with a call-local expected value.";
  const int64_t generation = ++barrier_count_;
  auto one_i32 = std::make_shared<ConstInt>(1, DataType::INT32, span);
  auto expected_i32 = std::make_shared<ConstInt>(generation, DataType::INT32, span);
  EmitNotifyAll(signal, comm.nranks_idx, comm.my_rank, NotifyOp::kAtomicAdd, one_i32, suffix, span);
  EmitWaitAll(signal, comm.nranks_idx, comm.my_rank, expected_i32, suffix, span);
  return generation;
}

void LoweringBuilder::EmitEpilogueReset(const ExprPtr& signal, const CommSetup& comm, const ExprPtr& total,
                                        const Span& span) {
  INTERNAL_CHECK_SPAN(!nested_, span)
      << "EmitEpilogueReset must only be called from a top-level lowering rule, exactly once "
         "per call, after every EmitBarrier / hand-rolled notify-wait pair the rule issues.";
  auto neg_total = MakeNegation(total);
  auto zero_idx = std::make_shared<ConstInt>(0, DataType::INDEX, span);
  auto one_idx = std::make_shared<ConstInt>(1, DataType::INDEX, span);
  EmitFor(
      "reset_src", zero_idx, comm.nranks_idx, one_idx,
      [&](LoweringBuilder& body, const VarPtr& src) {
        body.EmitIf(
            body.NotEq(src, comm.my_rank, span),
            [&](LoweringBuilder& then_body) {
              auto src_offsets = tile_conversion_utils::MakeSignalOffsets(src, span);
              auto call = OpRegistry::GetInstance().Create(
                  "pld.system.notify", {signal, comm.my_rank, src_offsets, neg_total},
                  {{"op", static_cast<int>(NotifyOp::kAtomicAdd)}}, span);
              then_body.Bind("epilogue_reset_ret", call, span);
            },
            /*else_fn=*/nullptr, span);
      },
      span);
}

void LoweringBuilder::EmitEpilogueReset(const ExprPtr& signal, const CommSetup& comm, const ExprPtr& num_rows,
                                        const ExprPtr& total_per_row, const Span& span) {
  INTERNAL_CHECK_SPAN(!nested_, span)
      << "EmitEpilogueReset must only be called from a top-level lowering rule, exactly once per call.";
  auto neg_total = MakeNegation(total_per_row);
  auto zero_idx = std::make_shared<ConstInt>(0, DataType::INDEX, span);
  auto one_idx = std::make_shared<ConstInt>(1, DataType::INDEX, span);
  EmitFor(
      "reset_row", zero_idx, num_rows, one_idx,
      [&](LoweringBuilder& row_body, const VarPtr& row) {
        row_body.EmitFor(
            "reset_src", zero_idx, comm.nranks_idx, one_idx,
            [&](LoweringBuilder& body, const VarPtr& src) {
              body.EmitIf(
                  body.NotEq(src, comm.my_rank, span),
                  [&](LoweringBuilder& then_body) {
                    auto src_offsets = tile_conversion_utils::MakeSignalOffsets(src, row, span);
                    auto call = OpRegistry::GetInstance().Create(
                        "pld.system.notify", {signal, comm.my_rank, src_offsets, neg_total},
                        {{"op", static_cast<int>(NotifyOp::kAtomicAdd)}}, span);
                    then_body.Bind("epilogue_reset_ret", call, span);
                  },
                  /*else_fn=*/nullptr, span);
            },
            span);
      },
      span);
}

void LoweringBuilder::EmitFor(const std::string& loop_var_name, const ExprPtr& start, const ExprPtr& stop,
                              const ExprPtr& step,
                              const std::function<void(LoweringBuilder&, const VarPtr&)>& body_fn,
                              const Span& span) {
  auto loop_var = std::make_shared<Var>(MakeTempName(loop_var_name), start->GetType(), span);
  LoweringBuilder body_builder(base_name_, temp_counter_, /*nested=*/true);
  body_fn(body_builder, loop_var);
  auto body_stmt = WrapBodyStmts(body_builder.TakeStmts(), span);
  stmts_.push_back(std::make_shared<ForStmt>(loop_var, start, stop, step, std::vector<IterArgPtr>{},
                                             body_stmt, std::vector<VarPtr>{}, span));
}

ExprPtr LoweringBuilder::EmitForReduce(
    const std::string& loop_var_name, const ExprPtr& start, const ExprPtr& stop, const ExprPtr& step,
    const ExprPtr& init_value,
    const std::function<ExprPtr(LoweringBuilder&, const VarPtr&, const VarPtr&)>& body_fn, const Span& span) {
  auto loop_var = std::make_shared<Var>(MakeTempName(loop_var_name), start->GetType(), span);
  auto iter_arg = std::make_shared<IterArg>(MakeTempName(loop_var_name + "_acc"), init_value->GetType(),
                                            init_value, span);
  LoweringBuilder body_builder(base_name_, temp_counter_, /*nested=*/true);
  ExprPtr yield_val = body_fn(body_builder, loop_var, iter_arg);
  INTERNAL_CHECK_SPAN(yield_val, span)
      << "EmitForReduce body_fn must return the next iteration's accumulator value";
  body_builder.stmts_.push_back(std::make_shared<YieldStmt>(std::vector<ExprPtr>{yield_val}, span));
  auto body_stmt = WrapBodyStmts(body_builder.TakeStmts(), span);
  auto return_var =
      std::make_shared<Var>(MakeTempName(loop_var_name + "_final"), init_value->GetType(), span);
  stmts_.push_back(std::make_shared<ForStmt>(loop_var, start, stop, step, std::vector<IterArgPtr>{iter_arg},
                                             body_stmt, std::vector<VarPtr>{return_var}, span));
  return return_var;
}

void LoweringBuilder::EmitIf(const ExprPtr& cond, const std::function<void(LoweringBuilder&)>& then_fn,
                             const std::function<void(LoweringBuilder&)>& else_fn, const Span& span) {
  LoweringBuilder then_builder(base_name_, temp_counter_, /*nested=*/true);
  then_fn(then_builder);
  auto then_body = WrapBodyStmts(then_builder.TakeStmts(), span);

  std::optional<StmtPtr> else_body = std::nullopt;
  if (else_fn) {
    LoweringBuilder else_builder(base_name_, temp_counter_, /*nested=*/true);
    else_fn(else_builder);
    else_body = WrapBodyStmts(else_builder.TakeStmts(), span);
  }
  stmts_.push_back(std::make_shared<IfStmt>(cond, then_body, else_body, std::vector<VarPtr>{}, span));
}

ExprPtr LoweringBuilder::EmitIfExpr(const ExprPtr& cond,
                                    const std::function<ExprPtr(LoweringBuilder&)>& then_fn,
                                    const std::function<ExprPtr(LoweringBuilder&)>& else_fn,
                                    const Span& span) {
  INTERNAL_CHECK_SPAN(then_fn && else_fn, span)
      << "EmitIfExpr requires both then_fn and else_fn (the if must yield a value on every path)";
  LoweringBuilder then_builder(base_name_, temp_counter_, /*nested=*/true);
  ExprPtr then_val = then_fn(then_builder);
  INTERNAL_CHECK_SPAN(then_val, span) << "EmitIfExpr then_fn must return the yielded value";
  then_builder.stmts_.push_back(std::make_shared<YieldStmt>(std::vector<ExprPtr>{then_val}, span));
  auto then_body = WrapBodyStmts(then_builder.TakeStmts(), span);

  LoweringBuilder else_builder(base_name_, temp_counter_, /*nested=*/true);
  ExprPtr else_val = else_fn(else_builder);
  INTERNAL_CHECK_SPAN(else_val, span) << "EmitIfExpr else_fn must return the yielded value";
  else_builder.stmts_.push_back(std::make_shared<YieldStmt>(std::vector<ExprPtr>{else_val}, span));
  auto else_body = WrapBodyStmts(else_builder.TakeStmts(), span);

  auto return_var = std::make_shared<Var>(MakeTempName("if_res"), then_val->GetType(), span);
  stmts_.push_back(std::make_shared<IfStmt>(cond, then_body, std::optional<StmtPtr>(else_body),
                                            std::vector<VarPtr>{return_var}, span));
  return return_var;
}

std::vector<StmtPtr> LoweringBuilder::TakeStmts() { return std::move(stmts_); }

std::string LoweringBuilder::MakeTempName(const std::string& qualifier) {
  return auto_name::BuildName(auto_name::GetBaseName(base_name_), qualifier, "tmp",
                              static_cast<int>(temp_counter_++));
}

StmtPtr LoweringBuilder::WrapBodyStmts(std::vector<StmtPtr> body_stmts, const Span& span) {
  if (body_stmts.empty()) return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, span);
  if (body_stmts.size() == 1) return body_stmts.front();
  return std::make_shared<SeqStmts>(std::move(body_stmts), span);
}

}  // namespace lower_composite
}  // namespace ir
}  // namespace pypto
