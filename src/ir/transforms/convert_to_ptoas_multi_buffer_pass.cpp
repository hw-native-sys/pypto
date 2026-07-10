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

// ConvertToPtoasMultiBuffer — ptoas multi-buffer (same-slot) lowering.
//
// Gated by `PassContext::UsePtoasMultiBuffer()`. When the switch is on, the pass
// manager also drops `LowerPipelineLoops` and `CanonicalizeIOOrder`, so THIS pass
// owns pipeline lowering: it must leave zero `ForKind::Pipeline` loops behind. For
// each same-core pipeline loop it either
//   (a) rewrites the i-dependent load into a ptoas multi-buffer region access
//       (load slot `i%N`, consumed same-slot same-iteration), or
//   (b) demotes it to a plain Sequential loop (correct, no double-buffer) when it
//       is not an eligible multi-buffer shape.
//
// Why same-slot, not an explicit prologue/prefetch split: ptoas delivers the
// cross-iteration double-buffer OVERLAP itself. Given `t = mb[i%N]` loaded and
// consumed each iteration, ptoas PlanMemory assigns the N slots disjoint
// addresses and its sync pass overlaps iteration i's load (slot i%N) with
// iteration i-1's consume (slot (i-1)%N ≠ i%N) via dyn-event (`set_flag_dyn` /
// `wait_flag_dyn`) WAR sync. A manual prefetch-next/consume-cur split instead
// fights that analysis and serializes. Verified: same-slot overlaps, explicit
// prefetch does not (see docs/en/dev/passes/ptoas-multi-buffer.md).
//
// **Level2-only (`memory_planner=PtoAS`, --pto-level=level2).** The overlap only
// materializes when ptoas PlanMemory assigns the slots concrete disjoint
// addresses; at level3 a baked base + dynamic slot defeats ptoas MemAlias. So
// `use_ptoas_multi_buffer` forces PtoAS (PassContext ctor), MemoryReuse /
// AllocateMemoryAddr are skipped, and ptoas owns the N-slot region; codegen
// asserts a region never reaches level3 (emit_tile_addr_).
//
// Emits two pass-internal ops (see src/ir/op/tile_ops/memory.cpp):
//   tile.multi_buffer_alloc(shape; count=N, ...)          -> region (N-slot)
//   tile.multi_buffer_load_slot(region, k, tensor, ...)   -> load slot k
// The dynamic slot `k = i%N` is a normal index SSA operand resolved only at
// codegen (`pto.multi_tile_get %mb[k]`); it never enters the MemRef.
//
// M1 scope: exactly one i-dependent Vec/Mat load per pipeline loop. Anything else
// falls through to the Sequential demotion.

#include <any>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/attrs.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace pass {

namespace {

// True when `tile_type` lives in vec/mat local memory — the only spaces ptoas
// multi-buffer supports.
bool IsMultiBufferEligibleSpace(const std::shared_ptr<const TileType>& tile_type) {
  if (!tile_type || !tile_type->memory_space_.has_value()) return false;
  MemorySpace space = *tile_type->memory_space_;
  return space == MemorySpace::Vec || space == MemorySpace::Mat;
}

// Whether `expr` references `target` anywhere in its subtree.
class VarRefFinder : public IRVisitor {
 public:
  explicit VarRefFinder(const Var* target) : target_(target) {}
  bool found() const { return found_; }

 protected:
  void VisitVarLike_(const VarPtr& op) override {
    if (op.get() == target_) found_ = true;
    IRVisitor::VisitVarLike_(op);
  }

 private:
  const Var* target_;
  bool found_ = false;
};

bool ExprReferencesVar(const ExprPtr& expr, const Var* target) {
  if (!expr || !target) return false;
  VarRefFinder finder(target);
  finder.VisitExpr(expr);
  return finder.found();
}

// A single eligible i-dependent load inside a pipeline loop body.
struct EligibleLoad {
  size_t index = 0;      // position in the top-level body SeqStmts
  AssignStmtPtr assign;  // T = tile.load(...)
  CallPtr load;          // the tile.load Call
  VarPtr tile_var;       // T
  std::shared_ptr<const TileType> tile_type;
};

// Return the loop body's top-level statement list (SeqStmts unwrapped).
std::vector<StmtPtr> BodyStmts(const StmtPtr& body) {
  if (auto seq = As<SeqStmts>(body)) return seq->stmts_;
  return {body};
}

// Find the sole i-dependent Vec/Mat `tile.load` at the top level of `body`.
// Returns nullopt unless EXACTLY one exists (M1 restriction).
std::optional<EligibleLoad> FindEligibleLoad(const StmtPtr& body, const VarPtr& loop_var) {
  auto stmts = BodyStmts(body);
  std::optional<EligibleLoad> found;
  for (size_t i = 0; i < stmts.size(); ++i) {
    auto assign = As<AssignStmt>(stmts[i]);
    if (!assign) continue;
    auto call = As<Call>(assign->value_);
    if (!call || !IsOp(call, "tile.load")) continue;
    auto tile_type = As<TileType>(assign->var_->GetType());
    if (!IsMultiBufferEligibleSpace(tile_type)) continue;
    // i-dependence: the load offsets (arg 1) must reference the loop var, else
    // the load is loop-invariant and multi-buffering it is pointless.
    if (call->args_.size() < 2 || !ExprReferencesVar(call->args_[1], loop_var.get())) continue;
    if (found.has_value()) return std::nullopt;  // more than one — M1 bails.
    found = EligibleLoad{i, assign, call, assign->var_, tile_type};
  }
  return found;
}

// Strip `pipeline_stages` and demote to Sequential, keeping the (kind, attr)
// invariant `kind == Pipeline <=> pipeline_stages` whole (PipelineLoopValid).
StmtPtr DemoteToSequential(const ForStmtPtr& loop) {
  auto cleaned = MutableCopy(loop);
  cleaned->kind_ = ForKind::Sequential;
  cleaned->attrs_ = StripAttr(loop->attrs_, kPipelineStagesAttr);
  return cleaned;
}

ExprPtr MakeIndexConst(int64_t value, const Span& span) {
  return std::make_shared<ConstInt>(value, DataType::INDEX, span);
}

// Build `tile.multi_buffer_load_slot(region, slot, tensor, offsets, shapes,
// valid_shapes)` from the original tile.load's arg list.
CallPtr MakeLoadSlot(const VarPtr& region, const ExprPtr& slot, const std::vector<ExprPtr>& load_args,
                     const Span& span) {
  INTERNAL_CHECK_SPAN(load_args.size() >= 3, span)
      << "Internal error: tile.load must have >=3 args (tensor, offsets, shapes)";
  ExprPtr tensor = load_args[0];
  ExprPtr offsets = load_args[1];
  ExprPtr shapes = load_args[2];
  ExprPtr valid_shapes = load_args.size() >= 4 ? load_args[3] : load_args[2];
  std::vector<ExprPtr> args{region, slot, tensor, offsets, shapes, valid_shapes};
  return OpRegistry::GetInstance().Create("tile.multi_buffer_load_slot", args, span);
}

// Same-slot multi-buffer rewrite: hoist the region alloc, then replace the
// i-dependent load with a `load_slot(region, i%N, <load args>)` on the SAME tile
// var, so its consumers need no rewrite. ptoas delivers the cross-iteration
// overlap; no prologue / prefetch / guard is emitted here.
StmtPtr BuildMultiBuffer(const ForStmtPtr& loop, const EligibleLoad& e, int64_t n) {
  const Span& span = loop->span_;
  const VarPtr& iv = loop->loop_var_;
  ExprPtr n_const = MakeIndexConst(n, span);

  // region alloc (before the loop): multi_buffer_alloc(shape; dtype, mem, count=N).
  // Reuses DeduceTileCreateTileType, so shape is a MakeTuple of per-slot ConstInts.
  auto shape_tuple = std::make_shared<MakeTuple>(e.tile_type->shape_, span);
  std::vector<std::pair<std::string, std::any>> alloc_kwargs{{"dtype", e.tile_type->dtype_},
                                                             {"target_memory", *e.tile_type->memory_space_},
                                                             {"count", static_cast<int>(n)}};
  auto region_call =
      OpRegistry::GetInstance().Create("tile.multi_buffer_alloc", {shape_tuple}, alloc_kwargs, span);
  auto region_var = std::make_shared<Var>(e.tile_var->name_hint_ + "_mb", region_call->GetType(), span);
  auto region_assign = std::make_shared<AssignStmt>(region_var, region_call, span);

  // Replace the load in place: same tile var, slot `i % N`, original load args.
  ExprPtr slot = MakeFloorMod(iv, n_const, span);
  auto load_slot = MakeLoadSlot(region_var, slot, e.load->args_, span);
  auto new_load_assign = std::make_shared<AssignStmt>(e.tile_var, load_slot, span);

  std::vector<StmtPtr> body_stmts = BodyStmts(loop->body_);
  body_stmts[e.index] = new_load_assign;

  auto new_loop = MutableCopy(loop);
  new_loop->kind_ = ForKind::Sequential;
  new_loop->attrs_ = StripAttr(loop->attrs_, kPipelineStagesAttr);
  new_loop->body_ = SeqStmts::Flatten(std::move(body_stmts), span);

  std::vector<StmtPtr> result{region_assign, new_loop};
  return SeqStmts::Flatten(std::move(result), span);
}

class ConvertMutator : public IRMutator {
 protected:
  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    // Recurse first so nested pipeline loops convert/demote independently.
    auto visited = IRMutator::VisitStmt_(op);
    auto loop = As<ForStmt>(visited);
    if (!loop || loop->kind_ != ForKind::Pipeline || !loop->HasAttr(kPipelineStagesAttr)) return visited;

    int n = loop->GetAttr<int>(kPipelineStagesAttr, 0);
    // A single-i-dependent-load stage>=2 loop becomes a same-slot multi-buffer
    // access (slot i%N); everything else (stage==1 markers, matmul stage loops,
    // non-eligible shapes) demotes to a plain Sequential loop so no Pipeline loop
    // survives.
    if (n >= 2) {
      auto eligible = FindEligibleLoad(loop->body_, loop->loop_var_);
      if (eligible.has_value()) return BuildMultiBuffer(loop, *eligible, n);
    }
    return DemoteToSequential(loop);
  }
};

}  // namespace

Pass ConvertToPtoasMultiBuffer() {
  auto pass_func = [](const FunctionPtr& func) -> FunctionPtr {
    if (!func || !func->body_) return func;
    // Self-gated: no-op unless the active PassContext enables ptoas multi-buffer.
    auto* ctx = PassContext::Current();
    if (ctx == nullptr || !ctx->UsePtoasMultiBuffer()) return func;

    ConvertMutator mutator;
    auto new_body = mutator.VisitStmt(func->body_);
    if (new_body.get() == func->body_.get()) return func;
    return std::make_shared<Function>(func->name_, func->params_, func->param_directions_,
                                      func->return_types_, new_body, func->span_, func->func_type_,
                                      func->level_, func->role_, func->attrs_);
  };
  return CreateFunctionPass(pass_func, "ConvertToPtoasMultiBuffer", kConvertToPtoasMultiBufferProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
