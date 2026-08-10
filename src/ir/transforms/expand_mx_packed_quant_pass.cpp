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
 * @file expand_mx_packed_quant_pass.cpp
 * @brief Early MX legalization: K-split large packed quant + matmul_mx, then
 *        expand ``tile.tquant_mx(..., layout=MX_A_ZZ|MX_B_NN)`` into per-box flat
 *        quant + Vec assemble (B also INT8-transposes to [K,N]).
 *
 * Phase 1: co-split packed quant↔matmul or matmul-only slice (K>64, %64==0)
 *         — matmul path may rewrite to chunk layout (byte order differs from full pack).
 * Phase 2: reshape remaining packed-flat matmul scales ``[1,G]`` → ``[M,G]`` / ``[G,N]``.
 * Phase 3: expand packed quant via per-box assemble; isolated large-K keeps host full-pack
 *         order (mb/nb outer, kb inner), matching ``_pack_a_scale`` / ``_pack_b_scale``.
 */

#include <any>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace pass {
namespace {

constexpr int64_t kMxPackTileM = 16;
constexpr int64_t kMxPackTileK = 64;
constexpr int64_t kMxPackGroup = 32;
constexpr int64_t kMxChunkGroups = kMxPackTileK / kMxPackGroup;  // 2
constexpr int64_t kMxPackBoxRows = kMxPackTileM * kMxChunkGroups;  // 32
constexpr int64_t kMxPackReuseChunkBoxes = 16;

using DefMap = std::unordered_map<const Var*, ExprPtr>;

ExprPtr MakeIndex(int64_t value, const Span& span) {
  return std::make_shared<ConstInt>(value, DataType::INDEX, span);
}

ExprPtr MakeShape2(int64_t d0, int64_t d1, const Span& span) {
  return std::make_shared<MakeTuple>(std::vector<ExprPtr>{MakeIndex(d0, span), MakeIndex(d1, span)}, span);
}

const Var* GetVarIdentity(const ExprPtr& expr) {
  if (auto var = As<Var>(expr)) return var.get();
  if (auto iter_arg = As<IterArg>(expr)) return iter_arg.get();
  return nullptr;
}

bool ConstInt2(const ExprPtr& tup_expr, int64_t* a, int64_t* b) {
  auto tup = As<MakeTuple>(tup_expr);
  if (!tup || tup->elements_.size() != 2) return false;
  auto x = As<ConstInt>(tup->elements_[0]);
  auto y = As<ConstInt>(tup->elements_[1]);
  if (!x || !y) return false;
  *a = x->value_;
  *b = y->value_;
  return true;
}

ExprPtr FollowDefs(ExprPtr expr, const DefMap& defs, int max_depth = 8) {
  for (int d = 0; d < max_depth && expr; ++d) {
    const Var* v = GetVarIdentity(expr);
    if (!v) return expr;
    auto it = defs.find(v);
    if (it == defs.end()) return expr;
    expr = it->second;
  }
  return expr;
}

std::optional<TensorLayout> GetMxPackLayout(const CallPtr& call) {
  if (!call || !IsOp(call, "tile.tquant_mx")) return std::nullopt;
  for (const auto& [key, value] : call->kwargs_) {
    if (key != "layout") continue;
    auto layout = AnyCast<TensorLayout>(value, "kwarg key: layout");
    CHECK(layout == TensorLayout::MX_A_ZZ || layout == TensorLayout::MX_B_NN)
        << "tile.tquant_mx layout must be MX_A_ZZ or MX_B_NN (ND/None are not allowed), got "
        << TensorLayoutToString(layout);
    return layout;
  }
  return std::nullopt;
}

struct ResolvedTileLoad {
  ExprPtr tensor;
  int64_t row0 = 0;
  int64_t col0 = 0;
  std::vector<std::pair<std::string, std::any>> kwargs;
};

std::optional<ResolvedTileLoad> ResolveTileLoad(ExprPtr expr, const DefMap& defs) {
  for (int d = 0; d < 8 && expr; ++d) {
    if (auto call = As<Call>(expr); call && IsOp(call, "tile.load")) {
      CHECK(call->args_.size() >= 4) << "tile.load expects tensor, offsets, shapes, valid_shape";
      int64_t r = 0;
      int64_t c = 0;
      if (!ConstInt2(call->args_[1], &r, &c)) return std::nullopt;
      return ResolvedTileLoad{call->args_[0], r, c, call->kwargs_};
    }
    const Var* v = GetVarIdentity(expr);
    if (!v) return std::nullopt;
    auto it = defs.find(v);
    if (it == defs.end()) return std::nullopt;
    expr = it->second;
  }
  return std::nullopt;
}

AssignStmtPtr Bind(const std::string& name, const ExprPtr& expr, const Span& span) {
  return std::make_shared<AssignStmt>(std::make_shared<Var>(name, expr->GetType(), span), expr, span);
}

FunctionPtr WithBody(const FunctionPtr& func, const StmtPtr& body) {
  if (body.get() == func->body_.get()) return func;
  return std::make_shared<Function>(func->name_, func->params_, func->param_directions_,
                                    func->return_types_, body, func->span_, func->func_type_,
                                    func->level_, func->role_, func->attrs_);
}

void CollectDefs(const StmtPtr& body, DefMap* defs) {
  class C : public IRVisitor {
   public:
    DefMap* defs;
    void VisitStmt_(const AssignStmtPtr& op) override {
      (*defs)[op->var_.get()] = op->value_;
      IRVisitor::VisitStmt_(op);
    }
  } c;
  c.defs = defs;
  c.VisitStmt(body);
}

// -----------------------------------------------------------------------------
// Phase 1: K-split helpers + flat-scale legalize
// -----------------------------------------------------------------------------

enum class MxMatmulKind { kBase, kAcc, kBias };

std::optional<MxMatmulKind> ClassifyMxMatmul(const CallPtr& call) {
  if (IsOp(call, "tile.matmul_mx")) return MxMatmulKind::kBase;
  if (IsOp(call, "tile.matmul_mx_acc")) return MxMatmulKind::kAcc;
  if (IsOp(call, "tile.matmul_mx_bias")) return MxMatmulKind::kBias;
  return std::nullopt;
}

struct MxOperands {
  ExprPtr acc;
  ExprPtr lhs;
  ExprPtr lhs_scale;
  ExprPtr rhs;
  ExprPtr rhs_scale;
  ExprPtr bias;
  int64_t m = 0;
  int64_t k = 0;
  int64_t n = 0;
};

std::optional<MxOperands> ParseMxOperands(const CallPtr& call, MxMatmulKind kind) {
  MxOperands ops;
  size_t lhs_i = 0;
  if (kind == MxMatmulKind::kAcc) {
    if (call->args_.size() != 5) return std::nullopt;
    ops.acc = call->args_[0];
    lhs_i = 1;
  } else if (kind == MxMatmulKind::kBias) {
    if (call->args_.size() != 5) return std::nullopt;
    ops.bias = call->args_[4];
  } else if (call->args_.size() != 4) {
    return std::nullopt;
  }
  ops.lhs = call->args_[lhs_i];
  ops.lhs_scale = call->args_[lhs_i + 1];
  ops.rhs = call->args_[lhs_i + 2];
  ops.rhs_scale = call->args_[lhs_i + 3];
  auto lhs_ty = As<TileType>(ops.lhs->GetType());
  auto rhs_ty = As<TileType>(ops.rhs->GetType());
  if (!lhs_ty || !rhs_ty || lhs_ty->shape_.size() != 2 || rhs_ty->shape_.size() != 2) return std::nullopt;
  auto m = As<ConstInt>(lhs_ty->shape_[0]);
  auto k = As<ConstInt>(lhs_ty->shape_[1]);
  auto n = As<ConstInt>(rhs_ty->shape_[1]);
  if (!m || !k || !n) return std::nullopt;
  ops.m = m->value_;
  ops.k = k->value_;
  ops.n = n->value_;
  return ops;
}

/// ``[1, rows*cols]`` packed flat from ``quant_mx(layout)`` (else nullopt).
std::optional<std::pair<int64_t, int64_t>> PackedFlatLogicalShape(const TileTypePtr& scale, int64_t rows,
                                                                 int64_t cols) {
  if (!scale || scale->shape_.size() != 2) return std::nullopt;
  auto r = As<ConstInt>(scale->shape_[0]);
  auto c = As<ConstInt>(scale->shape_[1]);
  if (!r || !c || r->value_ != 1 || c->value_ != rows * cols) return std::nullopt;
  return std::make_pair(rows, cols);
}

class LegalizeFlatMxScaleMutator : public IRMutator {
 public:
  StmtPtr RewriteBody(const StmtPtr& body) {
    if (As<SeqStmts>(body)) return VisitStmt(body);
    auto visited = VisitStmt(body);
    auto assign = As<AssignStmt>(visited);
    auto call = assign ? As<Call>(assign->value_) : nullptr;
    if (!assign || !call) return visited;
    std::vector<StmtPtr> stmts;
    if (!TryRewrite(assign, call, stmts)) return visited;
    return SeqStmts::Flatten(std::move(stmts), body->span_);
  }

 protected:
  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    bool changed = false;
    std::vector<StmtPtr> new_stmts;
    new_stmts.reserve(op->stmts_.size());
    for (const auto& stmt : op->stmts_) {
      auto rewritten = IRMutator::VisitStmt(stmt);
      auto assign = As<AssignStmt>(rewritten);
      auto call = assign ? As<Call>(assign->value_) : nullptr;
      if (assign && call) {
        std::vector<StmtPtr> expansion;
        if (TryRewrite(assign, call, expansion)) {
          changed = true;
          new_stmts.insert(new_stmts.end(), expansion.begin(), expansion.end());
          continue;
        }
      }
      changed = changed || rewritten.get() != stmt.get();
      new_stmts.push_back(rewritten);
    }
    return changed ? SeqStmts::Flatten(std::move(new_stmts), op->span_) : op;
  }

  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    auto then_b = RewriteBody(op->then_body_);
    std::optional<StmtPtr> else_b = op->else_body_;
    if (op->else_body_) else_b = RewriteBody(*op->else_body_);
    if (then_b.get() == op->then_body_.get() &&
        (!op->else_body_ || else_b->get() == op->else_body_->get())) {
      return op;
    }
    auto result = MutableCopy(op);
    result->then_body_ = std::move(then_b);
    result->else_body_ = std::move(else_b);
    return result;
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override { return VisitLoop(op); }
  StmtPtr VisitStmt_(const WhileStmtPtr& op) override { return VisitLoop(op); }

 private:
  template <typename LoopPtr>
  StmtPtr VisitLoop(const LoopPtr& op) {
    auto body = RewriteBody(op->body_);
    if (body.get() == op->body_.get()) return op;
    auto result = MutableCopy(op);
    result->body_ = std::move(body);
    return result;
  }

  bool TryRewrite(const AssignStmtPtr& assign, const CallPtr& call, std::vector<StmtPtr>& out) {
    auto kind = ClassifyMxMatmul(call);
    if (!kind) return false;
    auto ops = ParseMxOperands(call, *kind);
    if (!ops) return false;

    auto lhs_ty = As<TileType>(ops->lhs_scale->GetType());
    auto rhs_ty = As<TileType>(ops->rhs_scale->GetType());
    const int64_t g = ops->k / kMxPackGroup;
    auto lhs_flat = PackedFlatLogicalShape(lhs_ty, ops->m, g);
    auto rhs_flat = PackedFlatLogicalShape(rhs_ty, g, ops->n);
    if (!lhs_flat && !rhs_flat) return false;

    const Span& span = call->span_;
    const std::string base = assign->var_->name_hint_;
    auto& reg = OpRegistry::GetInstance();
    ExprPtr lhs_s = ops->lhs_scale;
    ExprPtr rhs_s = ops->rhs_scale;
    if (lhs_flat) {
      auto reshaped =
          reg.Create("tile.reshape", {lhs_s, MakeShape2(lhs_flat->first, lhs_flat->second, span)}, {}, span);
      auto as = Bind(base + "_ls_2d", reshaped, span);
      out.push_back(as);
      lhs_s = as->var_;
    }
    if (rhs_flat) {
      auto reshaped =
          reg.Create("tile.reshape", {rhs_s, MakeShape2(rhs_flat->first, rhs_flat->second, span)}, {}, span);
      auto as = Bind(base + "_rs_2d", reshaped, span);
      out.push_back(as);
      rhs_s = as->var_;
    }

    CallPtr mx_call;
    if (*kind == MxMatmulKind::kBase) {
      mx_call = reg.Create("tile.matmul_mx", {ops->lhs, lhs_s, ops->rhs, rhs_s}, {}, span);
    } else if (*kind == MxMatmulKind::kBias) {
      mx_call = reg.Create("tile.matmul_mx_bias", {ops->lhs, lhs_s, ops->rhs, rhs_s, ops->bias}, {}, span);
    } else {
      mx_call = reg.Create("tile.matmul_mx_acc", {ops->acc, ops->lhs, lhs_s, ops->rhs, rhs_s}, {}, span);
    }
    out.push_back(std::make_shared<AssignStmt>(assign->var_, mx_call, span));
    return true;
  }
};

struct PackedQuantSite {
  CallPtr quant_call;
  TensorLayout layout = TensorLayout::ND;
  ExprPtr src;
  int64_t d0 = 0;
  int64_t k = 0;
  std::unordered_set<const Var*> chain_vars;
};

const Var* PeelTupleGet(ExprPtr expr, int index, const DefMap& defs) {
  expr = FollowDefs(expr, defs);
  auto get = As<TupleGetItemExpr>(expr);
  return (get && get->index_ == index) ? GetVarIdentity(get->tuple_) : nullptr;
}

/// Co-split requires data/scale to be direct ``TupleGetItem(quant, 0/1)`` (no user reshape).
std::optional<PackedQuantSite> ResolvePackedQuantSite(const ExprPtr& data, const ExprPtr& scale,
                                                      TensorLayout expect, const DefMap& defs) {
  const Var* data_t = PeelTupleGet(data, 0, defs);
  const Var* scale_t = PeelTupleGet(scale, 1, defs);
  if (!data_t || data_t != scale_t) return std::nullopt;
  auto q_it = defs.find(data_t);
  if (q_it == defs.end()) return std::nullopt;
  auto q_call = As<Call>(q_it->second);
  auto layout = GetMxPackLayout(q_call);
  if (!layout || *layout != expect || q_call->args_.empty()) return std::nullopt;

  auto src_ty = As<TileType>(q_call->args_[0]->GetType());
  if (!src_ty || src_ty->shape_.size() != 2) return std::nullopt;
  auto d0 = As<ConstInt>(src_ty->shape_[0]);
  auto k = As<ConstInt>(src_ty->shape_[1]);
  if (!d0 || !k || k->value_ <= kMxPackTileK || (k->value_ % kMxPackTileK) != 0) return std::nullopt;

  PackedQuantSite site;
  site.quant_call = q_call;
  site.layout = expect;
  site.src = q_call->args_[0];
  site.d0 = d0->value_;
  site.k = k->value_;
  site.chain_vars.insert(data_t);
  if (const Var* v = GetVarIdentity(data)) site.chain_vars.insert(v);
  if (const Var* v = GetVarIdentity(scale)) site.chain_vars.insert(v);
  for (const auto& [var, val] : defs) {
    if (auto get = As<TupleGetItemExpr>(val); get && GetVarIdentity(get->tuple_) == data_t) {
      site.chain_vars.insert(var);
    }
  }
  return site;
}

bool ChainOnlyUsedByMatmul(const PackedQuantSite& site, const ExprPtr& data_op, const ExprPtr& scale_op,
                           const Call* matmul_call, const StmtPtr& body) {
  std::unordered_set<const Var*> ops;
  if (auto* x = GetVarIdentity(data_op)) ops.insert(x);
  if (auto* x = GetVarIdentity(scale_op)) ops.insert(x);

  class OpOnly : public IRVisitor {
   public:
    std::unordered_set<const Var*>* ops;
    std::unordered_map<const Var*, int> uses;
    void VisitStmt_(const AssignStmtPtr& op) override { VisitExpr(op->value_); }
    void VisitExpr_(const VarPtr& op) override {
      if (ops->count(op.get())) uses[op.get()]++;
    }
    void VisitExpr_(const IterArgPtr& op) override {
      if (ops->count(op.get())) uses[op.get()]++;
    }
  } oc;
  oc.ops = &ops;
  oc.VisitStmt(body);
  for (const Var* x : ops) {
    if (oc.uses[x] != 1) return false;
  }

  class Ext : public IRVisitor {
   public:
    const std::unordered_set<const Var*>* chain;
    const Call* matmul = nullptr;
    bool bad = false;
    void VisitStmt_(const AssignStmtPtr& op) override {
      if (auto c = As<Call>(op->value_); c && c.get() == matmul) return;
      if (chain->count(op->var_.get())) return;
      VisitExpr(op->value_);
    }
    void VisitExpr_(const VarPtr& op) override {
      if (chain->count(op.get())) bad = true;
    }
    void VisitExpr_(const IterArgPtr& op) override {
      if (chain->count(op.get())) bad = true;
    }
  } ext;
  ext.chain = &site.chain_vars;
  ext.matmul = matmul_call;
  ext.VisitStmt(body);
  return !ext.bad;
}

std::pair<ExprPtr, ExprPtr> EmitPackedQuantChunk(const PackedQuantSite& site, int64_t k0,
                                                 const std::string& prefix, const DefMap& defs,
                                                 const Span& span, std::vector<StmtPtr>& out) {
  auto& reg = OpRegistry::GetInstance();
  auto push = [&](const std::string& n, const ExprPtr& e) {
    auto as = Bind(prefix + n, e, span);
    out.push_back(as);
    return as->var_;
  };

  ExprPtr src_chunk;
  if (auto ld = ResolveTileLoad(site.src, defs)) {
    auto shape = MakeShape2(site.d0, kMxPackTileK, span);
    src_chunk = reg.Create("tile.load",
                           {ld->tensor, MakeShape2(ld->row0, ld->col0 + k0, span), shape, shape},
                           ld->kwargs, span);
  } else {
    src_chunk = reg.Create(
        "tile.slice", {site.src, MakeShape2(site.d0, kMxPackTileK, span), MakeShape2(0, k0, span)}, {},
        span);
  }
  auto src = push("_src", src_chunk);
  auto pair = push("_tq", reg.Create("tile.tquant_mx", {src}, site.quant_call->kwargs_, span));
  auto q = push("_q", std::make_shared<TupleGetItemExpr>(pair, 0, span));
  auto s = push("_s", std::make_shared<TupleGetItemExpr>(pair, 1, span));
  auto s2_shape = site.layout == TensorLayout::MX_A_ZZ ? MakeShape2(site.d0, kMxChunkGroups, span)
                                                       : MakeShape2(kMxChunkGroups, site.d0, span);
  auto s2 = push("_s2", reg.Create("tile.reshape", {s, s2_shape}, {}, span));

  // Vec→Mat staging so Infer can insert Mat→Left / Mat→LeftScale.
  auto q_mat = push("_q_mat", reg.Create("tile.move", {q}, {{"target_memory", MemorySpace::Mat}}, span));
  const TileLayout sl =
      site.layout == TensorLayout::MX_A_ZZ ? TileLayout::row_major : TileLayout::col_major;
  auto s_mat = push("_s_mat", reg.Create("tile.move", {s2},
                                         {{"target_memory", MemorySpace::Mat},
                                          {"blayout", sl},
                                          {"slayout", sl}},
                                         span));
  return {q_mat, s_mat};
}

class KSplitMxMutator : public IRMutator {
 public:
  explicit KSplitMxMutator(const StmtPtr& whole_body) : whole_body_(whole_body) {
    CollectDefs(whole_body_, &defs_);
  }

  StmtPtr RewriteBody(const StmtPtr& body) {
    if (As<SeqStmts>(body)) return VisitStmt(body);
    auto visited = VisitStmt(body);
    auto assign = As<AssignStmt>(visited);
    auto call = assign ? As<Call>(assign->value_) : nullptr;
    if (!assign || !call) return visited;
    std::vector<StmtPtr> stmts;
    std::unordered_set<const Var*> dead;
    if (!TryRewrite(assign, call, stmts, dead)) return visited;
    return SeqStmts::Flatten(std::move(stmts), body->span_);
  }

 protected:
  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    struct Plan {
      size_t index;
      std::vector<StmtPtr> expansion;
      std::unordered_set<const Var*> dead;
    };
    std::vector<Plan> plans;
    std::unordered_set<const Var*> all_dead;
    for (size_t i = 0; i < op->stmts_.size(); ++i) {
      auto assign = As<AssignStmt>(op->stmts_[i]);
      auto call = assign ? As<Call>(assign->value_) : nullptr;
      if (!assign || !call) continue;
      std::vector<StmtPtr> expansion;
      std::unordered_set<const Var*> dead;
      if (TryRewrite(assign, call, expansion, dead)) {
        all_dead.insert(dead.begin(), dead.end());
        plans.push_back(Plan{i, std::move(expansion), std::move(dead)});
      }
    }
    bool changed = !plans.empty();
    std::vector<StmtPtr> new_stmts;
    new_stmts.reserve(op->stmts_.size());
    size_t pi = 0;
    for (size_t i = 0; i < op->stmts_.size(); ++i) {
      if (auto assign = As<AssignStmt>(op->stmts_[i]); assign && all_dead.count(assign->var_.get())) {
        changed = true;
        continue;
      }
      if (pi < plans.size() && plans[pi].index == i) {
        new_stmts.insert(new_stmts.end(), plans[pi].expansion.begin(), plans[pi].expansion.end());
        ++pi;
        changed = true;
        continue;
      }
      auto rewritten = IRMutator::VisitStmt(op->stmts_[i]);
      changed = changed || rewritten.get() != op->stmts_[i].get();
      new_stmts.push_back(rewritten);
    }
    return changed ? SeqStmts::Flatten(std::move(new_stmts), op->span_) : op;
  }

  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    auto then_b = RewriteBody(op->then_body_);
    std::optional<StmtPtr> else_b = op->else_body_;
    if (op->else_body_) else_b = RewriteBody(*op->else_body_);
    if (then_b.get() == op->then_body_.get() &&
        (!op->else_body_ || else_b->get() == op->else_body_->get())) {
      return op;
    }
    auto result = MutableCopy(op);
    result->then_body_ = std::move(then_b);
    result->else_body_ = std::move(else_b);
    return result;
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override { return VisitLoop(op); }
  StmtPtr VisitStmt_(const WhileStmtPtr& op) override { return VisitLoop(op); }

 private:
  template <typename LoopPtr>
  StmtPtr VisitLoop(const LoopPtr& op) {
    auto body = RewriteBody(op->body_);
    if (body.get() == op->body_.get()) return op;
    auto result = MutableCopy(op);
    result->body_ = std::move(body);
    return result;
  }

  bool TryRewrite(const AssignStmtPtr& assign, const CallPtr& call, std::vector<StmtPtr>& out,
                  std::unordered_set<const Var*>& dead) {
    auto kind = ClassifyMxMatmul(call);
    if (!kind) return false;
    auto ops = ParseMxOperands(call, *kind);
    if (!ops || ops->k <= kMxPackTileK || (ops->k % kMxPackTileK) != 0) return false;

    auto take_site = [&](auto site_opt, const ExprPtr& d, const ExprPtr& s) {
      if (site_opt && ChainOnlyUsedByMatmul(*site_opt, d, s, call.get(), whole_body_)) return site_opt;
      return std::optional<PackedQuantSite>{};
    };
    auto lhs_site = take_site(ResolvePackedQuantSite(ops->lhs, ops->lhs_scale, TensorLayout::MX_A_ZZ, defs_),
                              ops->lhs, ops->lhs_scale);
    auto rhs_site = take_site(ResolvePackedQuantSite(ops->rhs, ops->rhs_scale, TensorLayout::MX_B_NN, defs_),
                              ops->rhs, ops->rhs_scale);
    if (lhs_site) dead.insert(lhs_site->chain_vars.begin(), lhs_site->chain_vars.end());
    if (rhs_site) dead.insert(rhs_site->chain_vars.begin(), rhs_site->chain_vars.end());

    const Span& span = call->span_;
    const std::string base = assign->var_->name_hint_;
    auto& reg = OpRegistry::GetInstance();
    const int64_t chunks = ops->k / kMxPackTileK;
    ExprPtr acc = ops->acc;

    auto push_slice = [&](const std::string& n, const ExprPtr& src, int64_t d0, int64_t d1, int64_t o0,
                          int64_t o1) {
      auto as = Bind(n, reg.Create("tile.slice", {src, MakeShape2(d0, d1, span), MakeShape2(o0, o1, span)},
                                   {}, span),
                     span);
      out.push_back(as);
      return as->var_;
    };

    for (int64_t ci = 0; ci < chunks; ++ci) {
      const int64_t k0 = ci * kMxPackTileK;
      const int64_t g0 = ci * kMxChunkGroups;
      const std::string p = base + "_k" + std::to_string(ci);
      ExprPtr lhs;
      ExprPtr lhs_s;
      ExprPtr rhs;
      ExprPtr rhs_s;
      if (lhs_site) {
        std::tie(lhs, lhs_s) = EmitPackedQuantChunk(*lhs_site, k0, p + "_a", defs_, span, out);
      } else {
        lhs = push_slice(p + "_lhs", ops->lhs, ops->m, kMxPackTileK, 0, k0);
        lhs_s = push_slice(p + "_ls", ops->lhs_scale, ops->m, kMxChunkGroups, 0, g0);
      }
      if (rhs_site) {
        std::tie(rhs, rhs_s) = EmitPackedQuantChunk(*rhs_site, k0, p + "_b", defs_, span, out);
      } else {
        rhs = push_slice(p + "_rhs", ops->rhs, kMxPackTileK, ops->n, k0, 0);
        rhs_s = push_slice(p + "_rs", ops->rhs_scale, kMxChunkGroups, ops->n, g0, 0);
      }

      CallPtr mx_call;
      if (ci == 0 && *kind == MxMatmulKind::kBase) {
        mx_call = reg.Create("tile.matmul_mx", {lhs, lhs_s, rhs, rhs_s}, {}, span);
      } else if (ci == 0 && *kind == MxMatmulKind::kBias) {
        mx_call = reg.Create("tile.matmul_mx_bias", {lhs, lhs_s, rhs, rhs_s, ops->bias}, {}, span);
      } else {
        INTERNAL_CHECK_SPAN(acc, span) << "Internal error: MX K-split missing accumulator";
        mx_call = reg.Create("tile.matmul_mx_acc", {acc, lhs, lhs_s, rhs, rhs_s}, {}, span);
      }
      VarPtr out_var = (ci + 1 == chunks)
                           ? assign->var_
                           : std::make_shared<Var>(p + "_acc", mx_call->GetType(), span);
      out.push_back(std::make_shared<AssignStmt>(out_var, mx_call, span));
      acc = out_var;
    }
    return true;
  }

  StmtPtr whole_body_;
  DefMap defs_;
};

// -----------------------------------------------------------------------------
// Phase 2: Expand packed quant (kb==1)
// -----------------------------------------------------------------------------

class ExpandBuilder {
 public:
  ExpandBuilder(std::string base_name, std::size_t& temp_counter)
      : base_name_(std::move(base_name)), temp_counter_(temp_counter) {}

  ExprPtr Bind(const std::string& qualifier, const ExprPtr& expr, const Span& span) {
    auto var = std::make_shared<Var>(
        auto_name::BuildName(auto_name::GetBaseName(base_name_), qualifier, "tmp",
                             static_cast<int>(temp_counter_++)),
        expr->GetType(), span);
    stmts_.push_back(std::make_shared<AssignStmt>(var, expr, span));
    return var;
  }

  void DrainChunk(const std::vector<ExprPtr>& tiles, const Span& span,
                  const std::string& qualifier = "chunk_keep") {
    for (const auto& tile : tiles) Bind(qualifier, tile, span);
    stmts_.push_back(
        std::make_shared<EvalStmt>(OpRegistry::GetInstance().Create("system.bar_all", {}, span), span));
  }

  std::vector<StmtPtr> TakeStmts() { return std::move(stmts_); }

 private:
  std::string base_name_;
  std::size_t& temp_counter_;
  std::vector<StmtPtr> stmts_;
};

ExprPtr LoadBox(const ResolvedTileLoad& ld, int64_t row, int64_t col, ExpandBuilder& b, const Span& span) {
  auto shape = MakeShape2(kMxPackTileM, kMxPackTileK, span);
  return b.Bind("box",
                OpRegistry::GetInstance().Create(
                    "tile.load", {ld.tensor, MakeShape2(ld.row0 + row, ld.col0 + col, span), shape, shape},
                    ld.kwargs, span),
                span);
}

ExprPtr SliceBox(const ExprPtr& src, int64_t row, int64_t col, ExpandBuilder& b, const Span& span) {
  return b.Bind("box",
                OpRegistry::GetInstance().Create(
                    "tile.slice",
                    {src, MakeShape2(kMxPackTileM, kMxPackTileK, span), MakeShape2(row, col, span)}, {},
                    span),
                span);
}

std::pair<ExprPtr, ExprPtr> QuantizeBox(const ExprPtr& box, ExpandBuilder& b, const Span& span) {
  auto& reg = OpRegistry::GetInstance();
  auto box32 =
      b.Bind("box32", reg.Create("tile.reshape", {box, MakeShape2(kMxPackBoxRows, kMxPackGroup, span)},
                                 {}, span),
             span);
  auto pair = b.Bind("tq", reg.Create("tile.tquant_mx", {box32}, {{"mode", std::string("mxfp8_e4m3")}}, span),
                     span);
  auto q_box = b.Bind("qb", std::make_shared<TupleGetItemExpr>(pair, 0, span), span);
  auto s_box = b.Bind("sb", std::make_shared<TupleGetItemExpr>(pair, 1, span), span);
  auto q_mk = b.Bind(
      "qmk", reg.Create("tile.reshape", {q_box, MakeShape2(kMxPackTileM, kMxPackTileK, span)}, {}, span),
      span);
  return {q_mk, s_box};
}

ExprPtr CreateTypedU8(ExpandBuilder& b, int64_t rows, int64_t cols, bool mx_scale, const std::string& name,
                      const Span& span) {
  auto& reg = OpRegistry::GetInstance();
  auto r = MakeIndex(rows, span);
  auto c = MakeIndex(cols, span);
  TileView view;
  view.valid_shape = {r, c};
  view.blayout = TileLayout::row_major;
  view.slayout = TileLayout::none_box;
  if (mx_scale) view.fractal = tile_view_semantics::kMXScaleFractal;
  auto type =
      std::make_shared<TileType>(std::vector<ExprPtr>{r, c}, DataType::UINT8, std::nullopt, view,
                                 MemorySpace::Vec);
  auto raw = As<Call>(reg.Create("tile.create", {MakeShape2(rows, cols, span)},
                                 {{"dtype", DataType::UINT8}, {"target_memory", MemorySpace::Vec}}, span));
  INTERNAL_CHECK_SPAN(raw, span) << "Internal error: tile.create did not produce a Call";
  return b.Bind(name, std::make_shared<Call>(raw->op_, raw->args_, raw->kwargs_, raw->attrs_, type, span),
                span);
}

ExprPtr ReinterpretMxScaleBuffer(const ExprPtr& scale_u8, int64_t groups, ExpandBuilder& b,
                                 const Span& span) {
  auto& reg = OpRegistry::GetInstance();
  auto raw = As<Call>(reg.Create("tile.reinterpret_view", {scale_u8}, {{"dtype", DataType::FP8E8M0}}, span));
  INTERNAL_CHECK_SPAN(raw, span) << "Internal error: MX scale reinterpret_view did not produce a Call";
  auto one = MakeIndex(1, span);
  auto groups_dim = MakeIndex(groups, span);
  TileView scale_view;
  scale_view.valid_shape = {one, groups_dim};
  scale_view.blayout = TileLayout::row_major;
  scale_view.slayout = TileLayout::none_box;
  scale_view.fractal = tile_view_semantics::kMXScaleFractal;
  auto scale_type = std::make_shared<TileType>(std::vector<ExprPtr>{one, groups_dim}, DataType::FP8E8M0,
                                               std::nullopt, scale_view, MemorySpace::Vec);
  return b.Bind("s", std::make_shared<Call>(raw->op_, raw->args_, raw->kwargs_, raw->attrs_, scale_type, span),
                span);
}

void CheckPackShape(TensorLayout layout, int64_t d0, int64_t d1, const Span& span) {
  const char* name = layout == TensorLayout::MX_A_ZZ ? "MX_A_ZZ" : "MX_B_NN";
  CHECK_SPAN(d0 % kMxPackTileM == 0 && d1 % kMxPackTileK == 0, span)
      << "tile.tquant_mx(layout=" << name << ") requires dim0%" << kMxPackTileM << "==0 and dim1%"
      << kMxPackTileK << "==0, got [" << d0 << ", " << d1 << "]";
}

std::pair<ExprPtr, ExprPtr> ExpandMxPackedAssemble(TensorLayout layout, const ExprPtr& src,
                                                   const std::optional<ResolvedTileLoad>& ld, int64_t d0,
                                                   int64_t k, ExpandBuilder& b, const Span& span) {
  auto& reg = OpRegistry::GetInstance();
  INTERNAL_CHECK_SPAN(k > 0 && (k % kMxPackTileK) == 0, span)
      << "Internal error: MX packed assemble requires K%" << kMxPackTileK << "==0, got K=" << k;
  // Full-pack order matches host _pack_a_scale / _pack_b_scale: (mb|nb outer, kb inner).
  // Co-split (Phase 1) uses chunk-cat order instead when quant feeds matmul_mx.
  const int64_t mboxes = d0 / kMxPackTileM;
  const int64_t kboxes = k / kMxPackTileK;
  const int64_t total_boxes = mboxes * kboxes;
  const int64_t groups = d0 * k / kMxPackGroup;
  const bool is_a = layout == TensorLayout::MX_A_ZZ;

  auto q_u8 = CreateTypedU8(b, d0, k, false, is_a ? "q_u8" : "nk_u8", span);
  auto s_u8 = CreateTypedU8(b, 1, groups, true, "s_u8", span);

  std::vector<ExprPtr> chunk_tiles;
  int64_t box_i = 0;
  for (int64_t mb = 0; mb < mboxes; ++mb) {
    for (int64_t kb = 0; kb < kboxes; ++kb) {
      const int64_t row = mb * kMxPackTileM;
      const int64_t col = kb * kMxPackTileK;
      const int64_t box_id = mb * kboxes + kb;
      ExprPtr box = ld ? LoadBox(*ld, row, col, b, span) : SliceBox(src, row, col, b, span);
      auto [q_mk, s_box] = QuantizeBox(box, b, span);
      chunk_tiles.insert(chunk_tiles.end(), {box, q_mk, s_box});

      auto q_box_u8 =
          b.Bind(is_a ? "qmk_u8" : "qnk_u8",
                 reg.Create("tile.reinterpret_view", {q_mk}, {{"dtype", DataType::UINT8}}, span), span);
      q_u8 = b.Bind(is_a ? "q_u8" : "nk_u8",
                    reg.Create("tile.assemble", {q_u8, q_box_u8, MakeShape2(row, col, span)}, {}, span),
                    span);
      auto s_box_u8 =
          b.Bind("sb_u8", reg.Create("tile.reinterpret_view", {s_box}, {{"dtype", DataType::UINT8}}, span),
                 span);
      s_u8 = b.Bind("s_u8",
                    reg.Create("tile.assemble",
                               {s_u8, s_box_u8, MakeShape2(0, box_id * kMxPackBoxRows, span)}, {}, span),
                    span);
      ++box_i;
      if (box_i % kMxPackReuseChunkBoxes == 0 || box_i == total_boxes) {
        b.DrainChunk(chunk_tiles, span);
        chunk_tiles.clear();
      }
    }
  }

  auto s_acc = ReinterpretMxScaleBuffer(s_u8, groups, b, span);
  if (is_a) {
    return {b.Bind("q", reg.Create("tile.reinterpret_view", {q_u8}, {{"dtype", DataType::FP8E4M3FN}}, span),
                   span),
            s_acc};
  }
  auto nk_q =
      b.Bind("nk_q", reg.Create("tile.reinterpret_view", {q_u8}, {{"dtype", DataType::FP8E4M3FN}}, span),
             span);
  auto nk_i8 =
      b.Bind("nk_i8", reg.Create("tile.reinterpret_view", {nk_q}, {{"dtype", DataType::INT8}}, span), span);
  auto kn_i8 = b.Bind(
      "kn_i8", reg.Create("tile.transpose", {nk_i8, MakeIndex(0, span), MakeIndex(1, span)}, {}, span), span);
  auto kn_q = b.Bind(
      "kn_q", reg.Create("tile.reinterpret_view", {kn_i8}, {{"dtype", DataType::FP8E4M3FN}}, span), span);
  b.DrainChunk({nk_q, kn_q}, span, "transpose_keep");
  return {kn_q, s_acc};
}

class ExpandMxPackedQuantMutator : public IRMutator {
 public:
  ExprPtr VisitExpr_(const TupleGetItemExprPtr& op) override {
    if (const Var* tuple_var = GetVarIdentity(op->tuple_)) {
      if (auto it = tuple_outputs_.find(tuple_var); it != tuple_outputs_.end()) {
        INTERNAL_CHECK_SPAN(op->index_ >= 0 && op->index_ < 2, op->span_)
            << "Internal error: expanded MX packed quant tuple index out of range: " << op->index_;
        return it->second[static_cast<size_t>(op->index_)];
      }
    }
    return IRMutator::VisitExpr_(op);
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto call = As<Call>(op->value_);
    auto layout = GetMxPackLayout(call);
    if (!layout) {
      if (!call) {
        if (const Var* source = GetVarIdentity(op->value_)) {
          if (auto it = tuple_outputs_.find(source); it != tuple_outputs_.end()) {
            tuple_outputs_[op->var_.get()] = it->second;
            auto tuple_value =
                std::make_shared<MakeTuple>(std::vector<ExprPtr>{it->second[0], it->second[1]}, op->span_);
            def_map_[op->var_.get()] = tuple_value;
            return std::make_shared<AssignStmt>(op->var_, tuple_value, op->span_);
          }
        }
      }
      auto new_stmt = IRMutator::VisitStmt_(op);
      if (auto as = As<AssignStmt>(new_stmt)) def_map_[as->var_.get()] = as->value_;
      return new_stmt;
    }

    const Span& span = op->span_;
    auto src = call->args_[0];
    auto src_type = As<TileType>(src->GetType());
    INTERNAL_CHECK_SPAN(src_type && src_type->shape_.size() == 2, span)
        << "Internal error: MX packed quant expand requires 2D src";
    auto d0 = As<ConstInt>(src_type->shape_[0]);
    auto d1 = As<ConstInt>(src_type->shape_[1]);
    INTERNAL_CHECK_SPAN(d0 && d1, span) << "Internal error: MX packed quant expand requires static shape";
    CheckPackShape(*layout, d0->value_, d1->value_, span);

    ExpandBuilder builder(op->var_->name_hint_, temp_counter_);
    auto [quant, scale] =
        ExpandMxPackedAssemble(*layout, src, ResolveTileLoad(src, def_map_), d0->value_, d1->value_,
                               builder, span);
    auto stmts = builder.TakeStmts();
    tuple_outputs_[op->var_.get()] = {quant, scale};
    auto tuple_val = std::make_shared<MakeTuple>(std::vector<ExprPtr>{quant, scale}, span);
    stmts.push_back(std::make_shared<AssignStmt>(op->var_, tuple_val, span));
    def_map_[op->var_.get()] = tuple_val;
    return std::make_shared<SeqStmts>(std::move(stmts), span);
  }

 private:
  std::size_t temp_counter_ = 0;
  std::unordered_map<const Var*, std::array<ExprPtr, 2>> tuple_outputs_;
  DefMap def_map_;
};

}  // namespace

Pass ExpandMxPackedQuant() {
  auto pass_func = [](const FunctionPtr& func) -> FunctionPtr {
    if (!func || !func->body_ || !IsInCoreType(func->func_type_)) return func;
    // K-split first so co-split sees direct TupleGetItem(quant, 1) flat scales.
    KSplitMxMutator split(func->body_);
    auto after_split = WithBody(func, split.RewriteBody(func->body_));
    LegalizeFlatMxScaleMutator legalize;
    auto after_legalize = WithBody(after_split, legalize.RewriteBody(after_split->body_));
    ExpandMxPackedQuantMutator expand;
    return WithBody(after_legalize, expand.VisitStmt(after_legalize->body_));
  };
  return CreateFunctionPass(pass_func, "ExpandMxPackedQuant", kExpandMxPackedQuantProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
