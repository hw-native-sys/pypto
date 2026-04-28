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

/// AutoTileMatmulL0
/// ----------------
/// For each ``tile.matmul`` whose operands live in ``MemorySpace::Mat`` with
/// static 2D shape, picks an L0 tile shape ``(m, n, k)`` from the active
/// ``BackendHandler``'s L0 capacities (via ``utils::ChooseL0Tile``) and rewrites
/// the call into a K-loop that branches on the loop index: the first iteration
/// uses ``tile.matmul`` (fresh accumulator) and subsequent iterations use
/// ``tile.matmul_acc`` (accumulating into the iter-arg).  The loop is marked
/// ``ForKind::Pipeline`` with ``pipeline_stages=2`` whenever it has at least
/// two iterations so the downstream ``LowerPipelineLoops`` pass produces a
/// 2-deep ping-pong on the auto-inserted Mat→Left/Right moves.
///
/// Layout (for K = ceil(K/k) * k):
///   c_init = tile.create([m, n], dtype=FP32, target_memory=Vec)  // placeholder
///   for ko in <Pipeline?>(0, K, k) iter_args=[c_iter=c_init] return_vars=[out]:
///     sa = tile.slice(x_mat, [m, k], [0, ko])
///     sb = tile.slice(y_mat, [k, n], [ko, 0])
///     if ko == 0:
///       c1 = tile.matmul(sa, sb)             // fresh Acc
///       c_phi = pl.yield_(c1)                // if's return_var
///     else:
///       c2 = tile.matmul_acc(c_iter, sa, sb) // accumulate
///       c_phi = pl.yield_(c2)
///     yield c_phi                            // for-loop's iter-arg next
///
/// The ``out`` variable in the original ``out = tile.matmul(...)`` is reused
/// as the for-loop's return_var so downstream uses keep referencing the same
/// SSA value.  ``InferTileMemorySpace`` (the next pass) inserts moves for
/// Mat→Left/Right on the slices and a Vec/Acc bridge for the iter-arg.
///
/// V1 scope:
///   * Only ``tile.matmul`` (no ``tile.matmul_acc`` / ``tile.matmul_bias``).
///   * Only K tiling (``m == M`` and ``n == N``).  M/N tiling is handled in a
///     follow-up.
///   * Requires ``K % k == 0``.  K-boundary handling is deferred.
///
/// Cases outside this scope emit a ``DiagnosticSeverity::PerfHint`` describing
/// why no rewrite happened and leave the matmul untouched.

#include <any>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/attrs.h"
#include "pypto/ir/transforms/utils/l0_tile_chooser.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

constexpr const char* kPassName = "AutoTileMatmulL0";

ExprPtr MakeIndex(int64_t v, const Span& span) {
  return std::make_shared<ConstInt>(v, DataType::INDEX, span);
}

ExprPtr MakeIndexTuple(const std::vector<int64_t>& values, const Span& span) {
  std::vector<ExprPtr> elements;
  elements.reserve(values.size());
  for (auto v : values) elements.push_back(MakeIndex(v, span));
  return std::make_shared<MakeTuple>(std::move(elements), span);
}

/// True if `tile`'s 2D shape is static and `tile.memory_space_ == Mat`.
bool IsMatResidentStatic2D(const TileTypePtr& tile, int64_t& out_d0, int64_t& out_d1) {
  if (!tile || tile->shape_.size() != 2) return false;
  auto mem = tile->GetMemorySpace();
  if (!mem.has_value() || *mem != MemorySpace::Mat) return false;
  auto a = As<ConstInt>(tile->shape_[0]);
  auto b = As<ConstInt>(tile->shape_[1]);
  if (!a || !b) return false;
  out_d0 = a->value_;
  out_d1 = b->value_;
  return true;
}

/// Element width in bytes for a tile dtype.  Returns 0 for sub-byte types
/// (INT4, FP4 et al.) which the cube path does not support.
uint32_t DTypeBytes(const DataType& dt) {
  size_t bits = dt.GetBit();
  if (bits == 0 || bits % 8 != 0) return 0;
  return static_cast<uint32_t>(bits / 8);
}

/// Build a ``tile.slice(source, [shape], [offset])`` AssignStmt.  ``offset``
/// may include a runtime ``ko`` Var; the call accepts any 2-element tuple of
/// index-typed exprs.
AssignStmtPtr BuildSlice(const VarPtr& source, const std::vector<int64_t>& shape,
                         const std::vector<ExprPtr>& offset, const std::string& name_hint, const Span& span) {
  auto& reg = OpRegistry::GetInstance();
  auto offset_tuple = std::make_shared<MakeTuple>(offset, span);
  std::vector<ExprPtr> args = {source, MakeIndexTuple(shape, span), offset_tuple};
  auto call = reg.Create("tile.slice", args, span);
  auto var = std::make_shared<Var>(name_hint, call->GetType(), span);
  return std::make_shared<AssignStmt>(var, call, span);
}

/// Build the ``tile.create([m, n], dtype, target_memory=Vec)`` placeholder
/// that initializes the iter-arg.  Vec is used because:
///   * The actual accumulator buffer is materialized inside the loop (the
///     first matmul produces a fresh Acc tile and subsequent matmul_accs
///     accumulate into it).  The Vec init is a typed dummy whose payload is
///     never read.
///   * Vec is the natural ``tile.create`` default and survives print → parse
///     roundtrip without TileView annotation drift.
///   * ``InferTileMemorySpace`` (the next pass) inserts the Vec→Acc bridge
///     for the iter-arg's use sites, making the runtime IR type-correct.
AssignStmtPtr BuildAccInit(int64_t m, int64_t n, const DataType& dtype, const std::string& name_hint,
                           const Span& span) {
  auto& reg = OpRegistry::GetInstance();
  std::vector<std::pair<std::string, std::any>> kwargs = {{"dtype", dtype},
                                                          {"target_memory", MemorySpace::Vec}};
  auto call = reg.Create("tile.create", {MakeIndexTuple({m, n}, span)}, kwargs, span);
  auto var = std::make_shared<Var>(name_hint, call->GetType(), span);
  return std::make_shared<AssignStmt>(var, call, span);
}

struct KLoopRewrite {
  AssignStmtPtr original;
  VarPtr lhs_mat;  // x_mat: TileType([M, K], FP*, Mat)
  VarPtr rhs_mat;  // y_mat: TileType([K, N], FP*, Mat)
  int64_t M = 0;
  int64_t N = 0;
  int64_t K = 0;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
};

/// Pipeline marker only helps when the loop has at least two iterations to
/// overlap (one prefetch buffer in flight, one consumed) — i.e. ``K/k >= 2``.
bool ShouldMarkPipeline(int64_t K, int64_t k) { return (K / k) >= 2; }

/// Build the replacement statements for one Mat-resident matmul.  See the
/// file-level comment for the emitted shape.
std::vector<StmtPtr> BuildKLoopRewrite(const KLoopRewrite& r) {
  const Span sp = r.original->span_;
  const std::string base = r.original->var_->name_hint_;
  auto& reg = OpRegistry::GetInstance();

  std::vector<StmtPtr> out;
  out.reserve(2);

  // Accumulator init: a Vec-resident placeholder of shape [m, n].  The real
  // accumulator buffer is materialized by the first matmul each iteration;
  // this is just a typed dummy for the for-loop's iter-arg slot.
  auto acc_dtype = As<TileType>(r.original->var_->GetType())->dtype_;
  auto c_init = BuildAccInit(r.m, r.n, acc_dtype, base + "_l0_init", sp);
  out.push_back(c_init);

  // Loop variable.
  auto ko_var = std::make_shared<Var>(base + "_l0_ko", std::make_shared<ScalarType>(DataType::INDEX), sp);

  // Iter-arg is typed from the Vec init.  The yields inside the if-else are
  // Acc-typed (matmul / matmul_acc results); ``InferTileMemorySpace`` (the
  // next pass) inserts the Acc↔Vec bridges so the runtime IR is type-correct.
  auto c_iter = std::make_shared<IterArg>(base + "_l0_c", c_init->var_->GetType(), c_init->var_, sp);

  // Slices use the loop var as the K-axis offset.
  auto sa = BuildSlice(r.lhs_mat, {r.M, r.k}, {MakeIndex(0, sp), ko_var}, base + "_l0_a", sp);
  auto sb = BuildSlice(r.rhs_mat, {r.k, r.N}, {ko_var, MakeIndex(0, sp)}, base + "_l0_b", sp);

  // Then-branch: tile.matmul (no acc input — produces a fresh Acc tile).
  auto c_then_call = reg.Create("tile.matmul", {sa->var_, sb->var_}, sp);
  auto c_then_var = std::make_shared<Var>(base + "_l0_c_first", c_then_call->GetType(), sp);
  auto c_then_assign = std::make_shared<AssignStmt>(c_then_var, c_then_call, sp);
  auto then_yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{c_then_var}, sp);
  StmtPtr then_body = SeqStmts::Flatten(std::vector<StmtPtr>{c_then_assign, then_yield}, sp);

  // Else-branch: tile.matmul_acc (accumulates into the iter-arg).
  auto c_else_call = reg.Create("tile.matmul_acc", {ExprPtr(c_iter), sa->var_, sb->var_}, sp);
  auto c_else_var = std::make_shared<Var>(base + "_l0_c_acc", c_else_call->GetType(), sp);
  auto c_else_assign = std::make_shared<AssignStmt>(c_else_var, c_else_call, sp);
  auto else_yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{c_else_var}, sp);
  StmtPtr else_body = SeqStmts::Flatten(std::vector<StmtPtr>{c_else_assign, else_yield}, sp);

  // IfStmt return_var captures the chosen branch's yield (Acc-typed).
  auto c_phi = std::make_shared<Var>(base + "_l0_c_phi", c_then_call->GetType(), sp);
  auto cond = MakeEq(ko_var, MakeIndex(0, sp), sp);
  auto if_stmt = std::make_shared<IfStmt>(cond, then_body, std::optional<StmtPtr>(else_body),
                                          std::vector<VarPtr>{c_phi}, sp);

  // Outer yield carries c_phi back to the iter-arg for the next iteration.
  auto outer_yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{c_phi}, sp);

  std::vector<StmtPtr> body_stmts = {sa, sb, if_stmt, outer_yield};
  auto body = SeqStmts::Flatten(std::move(body_stmts), sp);

  // ForKind / attrs.
  ForKind kind = ForKind::Sequential;
  std::vector<std::pair<std::string, std::any>> attrs;
  if (ShouldMarkPipeline(r.K, r.k)) {
    kind = ForKind::Pipeline;
    attrs.emplace_back(kPipelineStagesAttr, /*pipeline_stages=*/2);
  }

  // Build a fresh return_var typed identically to the iter-arg.  Reusing the
  // original matmul's Var (Acc-typed) here would create an iter_arg/return_var
  // type mismatch that the post-pass TypeCheck flags.  The caller substitutes
  // uses of the original Var with this new one in subsequent statements of
  // the enclosing SeqStmts.
  auto rv = std::make_shared<Var>(base, c_iter->GetType(), r.original->var_->span_);

  auto for_stmt =
      std::make_shared<ForStmt>(ko_var, MakeIndex(0, sp), MakeIndex(r.K, sp), MakeIndex(r.k, sp),
                                std::vector<IterArgPtr>{c_iter}, body, std::vector<VarPtr>{rv}, sp, kind,
                                /*chunk_config=*/std::nullopt, std::move(attrs));
  out.push_back(for_stmt);
  return out;
}

/// Get the new return-var Var from a freshly-built rewrite.  The rewrite is
/// always a ``tile.create`` (init) followed by a ``ForStmt``; the ForStmt's
/// single return_var is what downstream uses of the original matmul Var
/// should be redirected to.
VarPtr GetReturnVar(const std::vector<StmtPtr>& rewrite_stmts) {
  INTERNAL_CHECK(rewrite_stmts.size() >= 2)
      << "Internal error: rewrite must contain at least an init and a ForStmt";
  auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(rewrite_stmts.back());
  INTERNAL_CHECK(for_stmt) << "Internal error: last rewrite stmt is not a ForStmt";
  INTERNAL_CHECK(for_stmt->return_vars_.size() == 1)
      << "Internal error: rewrite ForStmt must have exactly one return_var";
  return for_stmt->return_vars_[0];
}

/// Decide whether `assign` is a Mat-resident matmul that we know how to tile.
/// Returns the rewrite statements on success; otherwise nullopt and (when
/// useful) appends a PerfHint diagnostic.
std::optional<std::vector<StmtPtr>> MaybeRewriteMatmul(const AssignStmtPtr& assign,
                                                       std::vector<Diagnostic>& hints) {
  auto call = As<Call>(assign->value_);
  if (!call || !call->op_) return std::nullopt;

  const std::string& op_name = call->op_->name_;
  // V1 scope: only plain ``tile.matmul``.  ``tile.matmul_acc`` (caller-
  // provided accumulator) and ``tile.matmul_bias`` are out of scope and left
  // untouched.  Skipping silently keeps the pass idempotent — the rewritten
  // K-loop body itself contains a ``tile.matmul_acc`` which we don't want to
  // flag on a second run.
  if (op_name != "tile.matmul") return std::nullopt;

  if (call->args_.size() != 2) return std::nullopt;
  auto lhs = As<Var>(call->args_[0]);
  auto rhs = As<Var>(call->args_[1]);
  if (!lhs || !rhs) return std::nullopt;
  auto lhs_tile = As<TileType>(lhs->GetType());
  auto rhs_tile = As<TileType>(rhs->GetType());
  if (!lhs_tile || !rhs_tile) return std::nullopt;

  // Both operands must be Mat-resident with static 2D shape.  Other cases
  // (Vec/Acc operands, dynamic shapes) are out of scope; return silently.
  int64_t M = 0, K_lhs = 0, K_rhs = 0, N = 0;
  if (!IsMatResidentStatic2D(lhs_tile, M, K_lhs) || !IsMatResidentStatic2D(rhs_tile, K_rhs, N)) {
    return std::nullopt;
  }
  if (K_lhs != K_rhs) {
    hints.emplace_back(DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-002",
                       "tile.matmul: K dimensions don't match (lhs K=" + std::to_string(K_lhs) +
                           ", rhs K=" + std::to_string(K_rhs) + ") — left untouched",
                       assign->span_);
    return std::nullopt;
  }
  const int64_t K = K_lhs;

  uint32_t bytes_a = DTypeBytes(lhs_tile->dtype_);
  uint32_t bytes_b = DTypeBytes(rhs_tile->dtype_);
  constexpr uint32_t kBytesC = 4;  // matmul output is FP32 / INT32
  if (bytes_a == 0 || bytes_b == 0) {
    hints.emplace_back(DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-003",
                       "tile.matmul: unsupported operand dtype (zero element width) — left untouched",
                       assign->span_);
    return std::nullopt;
  }

  auto* ctx = PassContext::Current();
  if (!ctx) {
    hints.emplace_back(DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-004",
                       "tile.matmul: no PassContext active; cannot read L0 capacities — left untouched",
                       assign->span_);
    return std::nullopt;
  }
  const auto* handler = ctx->GetBackendHandler();
  INTERNAL_CHECK(handler) << "Internal error: PassContext returned a null BackendHandler";

  utils::L0TileConfig cfg;
  cfg.M = static_cast<int>(M);
  cfg.N = static_cast<int>(N);
  cfg.K = static_cast<int>(K);
  cfg.l0a_bytes = handler->GetL0aCapacityBytes();
  cfg.l0b_bytes = handler->GetL0bCapacityBytes();
  cfg.l0c_bytes = handler->GetL0cCapacityBytes();
  cfg.bytes_a = bytes_a;
  cfg.bytes_b = bytes_b;
  cfg.bytes_c = kBytesC;
  cfg.align_m = handler->GetL0FractalAlignment();
  cfg.align_n = handler->GetL0FractalAlignment();
  cfg.align_k = handler->GetL0FractalAlignment();
  cfg.min_m = handler->GetMinL0TileDim();
  cfg.min_n = handler->GetMinL0TileDim();
  cfg.min_k = handler->GetMinL0TileDim();
  cfg.double_buffer_a = true;
  cfg.double_buffer_b = true;
  cfg.double_buffer_c = false;
  cfg.c_read = false;
  cfg.allow_padding = false;

  utils::L0TileResult res;
  try {
    res = utils::ChooseL0Tile(cfg);
  } catch (const pypto::ValueError& e) {
    hints.emplace_back(
        DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-005",
        std::string("tile.matmul: ChooseL0Tile rejected configuration — left untouched. ") + e.what(),
        assign->span_);
    return std::nullopt;
  }

  // Already L0-sized — nothing to do.
  if (res.m == M && res.n == N && res.k == K) return std::nullopt;

  // V1 scope: K-tiling only.  M/N tiling is deferred.
  if (res.m != M || res.n != N) {
    hints.emplace_back(DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-006",
                       "tile.matmul: chooser picked m=" + std::to_string(res.m) + ", n=" +
                           std::to_string(res.n) + " (M=" + std::to_string(M) + ", N=" + std::to_string(N) +
                           "); M/N tiling not yet supported in this pass — left untouched",
                       assign->span_);
    return std::nullopt;
  }

  // V1 scope: require K % k == 0.  Boundary handling is deferred.
  if (K % res.k != 0) {
    hints.emplace_back(DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-007",
                       "tile.matmul: chooser picked k=" + std::to_string(res.k) + " not dividing K=" +
                           std::to_string(K) + "; K-boundary handling not yet supported — left untouched",
                       assign->span_);
    return std::nullopt;
  }

  INTERNAL_CHECK(K / res.k >= 2) << "Internal error: K=" << K << " not properly tiled by k=" << res.k;

  if (!res.perf_hint.empty()) {
    hints.emplace_back(DiagnosticSeverity::PerfHint, kPassName, 0, "PH-AT-008",
                       "tile.matmul: ChooseL0Tile fallback. " + res.perf_hint, assign->span_);
  }

  KLoopRewrite r;
  r.original = assign;
  r.lhs_mat = lhs;
  r.rhs_mat = rhs;
  r.M = M;
  r.N = N;
  r.K = K;
  r.m = res.m;
  r.n = res.n;
  r.k = res.k;
  return BuildKLoopRewrite(r);
}

class AutoTileMutator : public IRMutator {
 public:
  std::vector<Diagnostic> hints;

  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    // Per-SeqStmts substitution map: when we rewrite ``c = tile.matmul(...)``
    // into a ForStmt with a fresh return_var, subsequent statements in the
    // same SeqStmts that referenced ``c`` need to be redirected to that
    // return_var.  Scoped to this SeqStmts so substitutions don't leak into
    // sibling regions.
    std::unordered_map<const Var*, VarPtr> remap;
    std::vector<StmtPtr> out;
    out.reserve(op->stmts_.size());
    bool changed = false;
    for (const auto& child : op->stmts_) {
      // Apply the running remap to redirect prior rewrites' downstream uses.
      StmtPtr current = remap.empty() ? child : transform_utils::Substitute(child, remap);

      // Check if this is a matmul we rewrite *at this SeqStmts level*.  We
      // try this before recursive visitation so the rewrite — which produces
      // a sequence of stmts — lands in this enclosing SeqStmts.  Recursive
      // visitation happens after rewrite-rejection so nested matmuls inside
      // ForStmt bodies still get rewritten by the recursive visit.
      if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(current)) {
        if (auto rewrite = MaybeRewriteMatmul(assign, hints)) {
          remap[assign->var_.get()] = GetReturnVar(*rewrite);
          for (auto& s : *rewrite) out.push_back(std::move(s));
          changed = true;
          continue;
        }
      }
      auto visited = VisitStmt(current);
      if (visited.get() != child.get()) changed = true;
      out.push_back(visited);
    }
    if (!changed) return op;
    return SeqStmts::Flatten(std::move(out), op->span_);
  }
};

FunctionPtr TransformFunction(const FunctionPtr& func, std::vector<Diagnostic>& hints) {
  if (!func || !func->body_) return func;
  if (!IsInCoreType(func->func_type_)) return func;
  AutoTileMutator mutator;
  auto new_body = mutator.VisitStmt(func->body_);
  for (auto& d : mutator.hints) hints.push_back(std::move(d));
  if (new_body == func->body_) return func;
  auto new_func = MutableCopy(func);
  new_func->body_ = new_body;
  return new_func;
}

}  // namespace

namespace pass {

Pass AutoTileMatmulL0() {
  auto run = [](const ProgramPtr& program) -> ProgramPtr {
    if (!program) return program;
    std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> new_functions;
    bool any_change = false;
    std::vector<Diagnostic> hints;
    for (const auto& [gvar, func] : program->functions_) {
      auto new_func = TransformFunction(func, hints);
      if (new_func != func) any_change = true;
      new_functions.emplace(gvar, new_func);
    }
    if (!hints.empty()) EmitDiagnostics(hints, kPassName);
    if (!any_change) return program;
    auto new_program = MutableCopy(program);
    new_program->functions_ = std::move(new_functions);
    return new_program;
  };
  return CreateProgramPass(run, kPassName, kAutoTileMatmulL0Properties);
}

}  // namespace pass

}  // namespace ir
}  // namespace pypto
