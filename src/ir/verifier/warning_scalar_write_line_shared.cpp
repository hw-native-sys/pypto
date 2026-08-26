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
 * @file warning_scalar_write_line_shared.cpp
 * @brief Warn when concurrent task instances may `pl.write` into one 64-byte line.
 *
 * `tensor.write` lowers to `pto.store_scalar`, which PTOAS emits as a plain
 * cacheable store through a `__gm__` pointer plus ONE whole-cache
 * `dcci(0, ENTIRE_DATA_CACHE)` at the function tail. The store lands in the
 * issuing core's data cache and reaches GM only at kernel exit, written back a
 * whole 64-byte line at a time. Nothing keeps different cores' caches coherent.
 *
 * A core storing one element of a line first *fills* that line from GM, so its
 * copy holds one fresh element and 15 read at fill time. Two cores writing
 * DIFFERENT elements of one line therefore clobber each other: the later
 * write-back carries 60 bytes of stale snapshot over the other's store. Nothing
 * reports it — the tensor keeps its old value at most indices, and *which*
 * indices survive changes run to run.
 *
 * ## What is and is not checked
 *
 * The hazard needs two writers running at once, so the check is driven by
 * **instance multiplicity**, not by any single construct:
 *
 *   - Writes reached by more than one concurrent task instance are analysed.
 *     Two constructs multiply instances, and both are plain IR:
 *     `SpmdScopeStmt` with `core_num_ != 1`, and `ForStmt` with
 *     `ForKind::Parallel` (`pl.parallel`). Multiplicity propagates through
 *     calls, so a kernel dispatched from such a context is analysed too.
 *   - Writes inside a single instance are skipped. One instance executes its
 *     body sequentially on one core, so its stores land in one cache in program
 *     order and no line is contended.
 *
 * ## The model
 *
 * With `i_d` the index of instance dimension `d` and `E` the element size, the
 * flat byte address of a write is modelled as
 *
 *     addr = sum_d(coef_d * E * i_d) + r,   r in [lo*E, hi*E)
 *
 * and the write is disjoint when every `coef_d * E` is a nonzero multiple of 64,
 * the instance-independent span `(hi-lo)*E` fits inside the smallest of them,
 * and `lo*E` is 64-aligned. Then distinct instance tuples own distinct whole
 * lines. Everything else is reported, in two flavours the message distinguishes:
 *
 *   - Interleaved: some `coef_d` is known and `coef_d * E` is not a line
 *     multiple. The grid-stride `for i in pl.range(blk, N, BLOCKS)` fill is this
 *     case, and so is any per-instance counter.
 *   - Indeterminate: the address depends on a value read from memory, or on
 *     arithmetic outside the model, so nothing can be proven either way.
 *
 * Indeterminate is the common outcome — a plan index read out of another tensor
 * is unanalysable by construction — and it is reported rather than assumed safe,
 * because the accident that makes such code correct (a group stride that happens
 * to be 64 bytes) is invisible at the write site and silently lost when a
 * neighbouring constant changes.
 *
 * ## Known gaps
 *
 * Deliberately out of scope, each because closing it needs information that does
 * not exist at `PrePipeline`:
 *
 *   - Two *distinct* tasks writing one tensor. Whether they overlap in time
 *     depends on the dependency graph, which `AutoDeriveTaskDependencies`
 *     (pass 38) builds much later; without it every producer/consumer pair would
 *     report.
 *   - A write guarded by a predicate that pins the instance index
 *     (`if blk == 0: pl.write(...)`), which is a single writer wearing
 *     multi-instance clothing. Reported, conservatively.
 *   - Two instance dimensions that alias to the same address (`coef` equal
 *     across dims). That is a same-address write-write race, a different and
 *     more visible bug, and calling it line sharing would mislead.
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/diagnostic_check_registry.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

/// Warning error code (1000+ range for warnings; see warning_unused_var.cpp).
constexpr int kScalarWriteLineSharedCode = 1004;

/// Write-back granularity of the AI core's scalar data cache, and the only
/// granularity at which a scalar store reaches GM.
constexpr int64_t kCacheLineBytes = 64;

/// Instance dimensions tracked at once (nested `pl.parallel` / `pl.spmd`).
/// Deeper nesting than this abandons the model rather than dropping a term,
/// which would understate the sharing.
constexpr size_t kMaxInstanceDims = 4;

/**
 * @brief A scalar value as `sum_d(coef[d] * i_d) + r`, with `r` in `[lo, hi)`.
 *
 * `known == false` means the model lost track (a value read from memory, a
 * non-affine operation, an unbound Var); nothing may be read from the other
 * fields. A constant `c` is `coef == {0...}` with `r` in `[c, c+1)`.
 */
struct InstAffine {
  bool known = false;
  std::array<int64_t, kMaxInstanceDims> coef{};
  int64_t lo = 0;
  int64_t hi = 0;

  static InstAffine Unknown() { return {}; }

  static InstAffine Const(int64_t c) {
    InstAffine v;
    v.known = true;
    v.lo = c;
    v.hi = c + 1;
    return v;
  }

  /// The index of instance dimension `dim` itself: coefficient 1, no remainder.
  static InstAffine Instance(size_t dim) {
    InstAffine v;
    v.known = true;
    if (dim >= kMaxInstanceDims) return InstAffine::Unknown();
    v.coef[dim] = 1;
    v.hi = 1;
    return v;
  }

  [[nodiscard]] bool HasInstanceTerm() const {
    return std::any_of(coef.begin(), coef.end(), [](int64_t c) { return c != 0; });
  }
};

bool AddOverflows(int64_t a, int64_t b, int64_t* out) { return __builtin_add_overflow(a, b, out); }
bool SubOverflows(int64_t a, int64_t b, int64_t* out) { return __builtin_sub_overflow(a, b, out); }
bool MulOverflows(int64_t a, int64_t b, int64_t* out) { return __builtin_mul_overflow(a, b, out); }

InstAffine AffineAdd(const InstAffine& a, const InstAffine& b) {
  if (!a.known || !b.known) return InstAffine::Unknown();
  InstAffine r;
  // `hi` is exclusive, so the inclusive tops are `hi - 1`; adding the exclusive
  // tops directly would overcount by one.
  if (AddOverflows(a.lo, b.lo, &r.lo) || AddOverflows(a.hi - 1, b.hi - 1, &r.hi) ||
      AddOverflows(r.hi, 1, &r.hi)) {
    return InstAffine::Unknown();
  }
  for (size_t d = 0; d < kMaxInstanceDims; ++d) {
    if (AddOverflows(a.coef[d], b.coef[d], &r.coef[d])) return InstAffine::Unknown();
  }
  r.known = true;
  return r;
}

InstAffine AffineSub(const InstAffine& a, const InstAffine& b) {
  if (!a.known || !b.known) return InstAffine::Unknown();
  InstAffine r;
  // Subtracting an interval flips it: [a.lo - (b.hi-1), (a.hi-1) - b.lo].
  if (SubOverflows(a.lo, b.hi - 1, &r.lo) || SubOverflows(a.hi - 1, b.lo, &r.hi) ||
      AddOverflows(r.hi, 1, &r.hi)) {
    return InstAffine::Unknown();
  }
  for (size_t d = 0; d < kMaxInstanceDims; ++d) {
    if (SubOverflows(a.coef[d], b.coef[d], &r.coef[d])) return InstAffine::Unknown();
  }
  r.known = true;
  return r;
}

/// True when the value is a single constant, i.e. a legal multiplier.
bool IsConstAffine(const InstAffine& v) { return v.known && !v.HasInstanceTerm() && v.lo + 1 == v.hi; }

/// Only const*affine stays affine. A product of two instance-dependent values
/// is not, and neither is a product of two ranges.
InstAffine AffineMul(const InstAffine& a, const InstAffine& b) {
  if (!a.known || !b.known) return InstAffine::Unknown();
  const InstAffine* var = &a;
  const InstAffine* cst = &b;
  if (IsConstAffine(a)) {
    var = &b;
    cst = &a;
  } else if (!IsConstAffine(b)) {
    return InstAffine::Unknown();
  }
  const int64_t k = cst->lo;
  InstAffine r;
  // Multiply the *inclusive* endpoints, then re-derive the exclusive top. A
  // negative factor flips which endpoint is the low one, so the orientation must
  // be chosen before the `+1` — swapping the exclusive bounds afterwards moves
  // both ends inward and understates the span by two.
  int64_t end_lo = 0;
  int64_t end_hi = 0;
  if (MulOverflows(var->lo, k, &end_lo) || MulOverflows(var->hi - 1, k, &end_hi)) {
    return InstAffine::Unknown();
  }
  if (end_lo > end_hi) std::swap(end_lo, end_hi);
  r.lo = end_lo;
  if (AddOverflows(end_hi, 1, &r.hi)) return InstAffine::Unknown();
  for (size_t d = 0; d < kMaxInstanceDims; ++d) {
    if (MulOverflows(var->coef[d], k, &r.coef[d])) return InstAffine::Unknown();
  }
  r.known = true;
  return r;
}

/// True when `expr` reads the SPMD block index. Both `pl.spmd` forms bind it the
/// same way — the loop form emits `i = tile.get_block_idx()` as the first
/// statement of the outlined body — so matching the call covers both.
bool IsBlockIndexCall(const ExprPtr& expr) {
  auto call = As<Call>(expr);
  return call && IsOp(call, "tile.get_block_idx");
}

/// What the analysis concluded about one write.
enum class LineVerdict {
  Disjoint,       ///< Proven: each instance owns whole, private 64-byte lines.
  Interleaved,    ///< Proven otherwise: instances land inside a shared line.
  Indeterminate,  ///< Not provable either way.
};

/**
 * @brief Classify a write from its flat element-offset model and element size.
 *
 * Instance tuple `i` covers bytes `[sum_d(coef_d*E*i_d) + lo*E, ... + hi*E)`.
 * Those are whole, instance-private lines exactly when every per-dimension
 * stride is a nonzero multiple of the line size, the span fits inside the
 * smallest stride, and the first byte is line-aligned.
 *
 * `stride_out` receives the smallest per-dimension stride in bytes, for the
 * message; it is 0 when nothing could be measured.
 */
LineVerdict Classify(const InstAffine& offset, int64_t elem_bytes, int64_t* stride_out) {
  *stride_out = 0;
  if (!offset.known || elem_bytes <= 0) return LineVerdict::Indeterminate;

  int64_t base = 0;
  int64_t span = 0;
  if (MulOverflows(offset.lo, elem_bytes, &base) || SubOverflows(offset.hi, offset.lo, &span) ||
      MulOverflows(span, elem_bytes, &span)) {
    return LineVerdict::Indeterminate;
  }

  int64_t min_stride = 0;
  bool every_stride_aligned = true;
  for (size_t d = 0; d < kMaxInstanceDims; ++d) {
    if (offset.coef[d] == 0) continue;
    int64_t stride = 0;
    if (MulOverflows(offset.coef[d], elem_bytes, &stride)) return LineVerdict::Indeterminate;
    if (stride < 0) stride = -stride;
    // EVERY dimension must step by whole lines. Checking only the smallest
    // would call `g*16 + h*17` (INT32: strides 64 and 68) disjoint, even though
    // (g=1,h=0) at byte 64 and (g=0,h=1) at byte 68 land in the same line.
    if (stride % kCacheLineBytes != 0) every_stride_aligned = false;
    if (min_stride == 0 || stride < min_stride) min_stride = stride;
  }

  // No instance term: every instance writes the SAME bytes. That is a plain
  // write-write race the SPMD contract already forbids, not a line-sharing
  // question, and the model cannot tell a genuine one from a guard this walk
  // did not read (see "Known gaps").
  if (min_stride == 0) return LineVerdict::Indeterminate;

  *stride_out = min_stride;
  if (!every_stride_aligned) return LineVerdict::Interleaved;
  if (span > min_stride) return LineVerdict::Interleaved;
  if (base % kCacheLineBytes != 0) return LineVerdict::Interleaved;
  return LineVerdict::Disjoint;
}

/**
 * @brief Set of functions that may execute as more than one concurrent instance.
 *
 * Seeded with every callee invoked inside a multi-instance construct, then
 * closed transitively over the call graph: a helper called from a multi-instance
 * function is itself multi-instance. Lets a kernel dispatched by
 * `with pl.spmd(n): self.kernel(...)` be analysed in its own function body,
 * where the writes actually live.
 */
class MultiInstanceFunctions {
 public:
  static std::unordered_set<std::string> Compute(const ProgramPtr& program) {
    Collector collector;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      collector.current_function_ = func->name_;
      collector.VisitStmt(func->body_);
    }
    // Transitive closure over the call graph, seeded by the direct hits.
    std::unordered_set<std::string> result;
    std::vector<std::string> worklist(collector.seeds_.begin(), collector.seeds_.end());
    while (!worklist.empty()) {
      const std::string name = std::move(worklist.back());
      worklist.pop_back();
      if (!result.insert(name).second) continue;
      auto it = collector.edges_.find(name);
      if (it == collector.edges_.end()) continue;
      for (const auto& callee : it->second) {
        if (result.find(callee) == result.end()) worklist.push_back(callee);
      }
    }
    return result;
  }

 private:
  class Collector : public IRVisitor {
   public:
    std::string current_function_;
    std::unordered_set<std::string> seeds_;
    std::unordered_map<std::string, std::unordered_set<std::string>> edges_;

   protected:
    void VisitStmt_(const SpmdScopeStmtPtr& op) override {
      auto core_num = As<ConstInt>(op->core_num_);
      const bool multi = !(core_num && core_num->value_ == 1);
      depth_ += multi ? 1 : 0;
      IRVisitor::VisitStmt_(op);
      depth_ -= multi ? 1 : 0;
    }

    void VisitStmt_(const ForStmtPtr& op) override {
      const bool multi = op->kind_ == ForKind::Parallel;
      depth_ += multi ? 1 : 0;
      IRVisitor::VisitStmt_(op);
      depth_ -= multi ? 1 : 0;
    }

    void VisitExpr_(const CallPtr& op) override {
      IRVisitor::VisitExpr_(op);
      NoteCallee(op->op_);
    }

    void VisitExpr_(const SubmitPtr& op) override {
      // Submit is a task launch: its callee is by definition another instance.
      IRVisitor::VisitExpr_(op);
      NoteCallee(op->op_);
    }

   private:
    void NoteCallee(const OpPtr& callee) {
      if (!callee) return;
      const std::string& name = callee->name_;
      edges_[current_function_].insert(name);
      if (depth_ > 0) seeds_.insert(name);
    }

    int depth_ = 0;
  };
};

struct Report {
  LineVerdict verdict;
  std::string tensor_name;
  std::string origin;         ///< Preformatted "N concurrent blocks ('fill')".
  std::string instance_word;  ///< "blocks" or "task instances".
  int64_t elem_bytes;
  std::string dtype_name;
  int64_t stride_bytes;  ///< Smallest per-dimension stride; 0 if unmeasured.
  Span span;
};

/**
 * @brief Walks one function, tracking instance dimensions and Var bindings.
 *
 * O(N) over the function body: one traversal, a bindings map keyed by Var
 * pointer, and a scope-local Var stack unwound on scope exit.
 */
class ScalarWriteVisitor : public IRVisitor {
 public:
  explicit ScalarWriteVisitor(bool entered_multi_instance) {
    if (entered_multi_instance) {
      // The caller's dispatch supplies instance dimension 0; inside this body
      // `tile.get_block_idx()` is that dimension's index.
      instance_dims_ = 1;
      spmd_dim_stack_.push_back(0);
      contexts_.push_back(
          {"multiple concurrent task instances (dispatched from a "
           "multi-instance scope)",
           "task instances", /*dim_allocated=*/true});
    }
  }

  std::vector<Report> reports;

 protected:
  void VisitStmt_(const SpmdScopeStmtPtr& op) override {
    auto core_num = As<ConstInt>(op->core_num_);
    const bool multi = !(core_num && core_num->value_ == 1);
    const size_t mark = scope_local_stack_.size();
    if (multi) {
      const std::string count = core_num ? std::to_string(core_num->value_) : "a runtime number of";
      PushInstanceDim(
          /*is_spmd=*/true,
          count + " concurrent blocks ('" + (op->name_hint_.empty() ? "<unnamed>" : op->name_hint_) + "')",
          "blocks");
    }
    IRVisitor::VisitStmt_(op);
    TrimScopeLocals(mark);
    if (multi) PopInstanceDim(/*is_spmd=*/true);
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    const bool multi = op->kind_ == ForKind::Parallel;
    const size_t mark = scope_local_stack_.size();
    if (multi) {
      auto stop = As<ConstInt>(op->stop_);
      const std::string count = stop ? std::to_string(stop->value_) : "a runtime number of";
      PushInstanceDim(/*is_spmd=*/false, count + " concurrent task instances (pl.parallel)",
                      "task instances");
    }
    if (op->loop_var_) {
      // A parallel loop variable IS the instance index of the dimension just
      // pushed; a sequential one inherits its range from start/stop.
      bindings_[op->loop_var_.get()] =
          multi ? InstAffine::Instance(instance_dims_ - 1) : SequentialLoopVarModel(op);
    }
    IRVisitor::VisitStmt_(op);
    TrimScopeLocals(mark);
    if (multi) PopInstanceDim(/*is_spmd=*/false);
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    IRVisitor::VisitStmt_(op);
    if (!op->var_) return;
    // The value is modelled BEFORE the LHS is bound: an assignment's RHS cannot
    // read the name it defines (the IR is use-after-def even pre-SSA).
    bindings_[op->var_.get()] = Eval(op->value_);
    // Only an allocation made inside the scope is instance-private. Binding the
    // definition site instead would let an ordinary alias of an external tensor
    // (`alias = out`, or the `out = pl.write(out, ...)` rebind form) suppress
    // the warning for the very corruption this check exists to find.
    if (IsInstancePrivateAllocation(op->value_)) NoteScopeLocal(op->var_.get());
  }

  void VisitExpr_(const CallPtr& op) override {
    IRVisitor::VisitExpr_(op);
    if (contexts_.empty() || !IsOp(op, "tensor.write") || op->args_.size() != 3) return;

    auto dst = AsVarLike(op->args_[0]);
    if (!dst || IsScopeLocal(dst.get())) return;  // instance-private buffer

    auto tensor_type = AsTensorTypeLike(op->args_[0]->GetType());
    if (!tensor_type) return;
    const int64_t elem_bytes = static_cast<int64_t>(tensor_type->dtype_.GetByte());

    // A scope past the tracking limit contributes a dimension the model cannot
    // represent, so nothing below it can be proven — report rather than guess.
    const InstAffine offset = unmodelled_dims_ > 0
                                  ? InstAffine::Unknown()
                                  : FlatOffset(As<MakeTuple>(op->args_[1]), tensor_type->shape_);
    int64_t stride_bytes = 0;
    const LineVerdict verdict = Classify(offset, elem_bytes, &stride_bytes);
    if (verdict == LineVerdict::Disjoint) return;

    const Context& ctx = contexts_.back();
    // Aggregate-initialised in one expression: Span holds const members, so it
    // is copy-constructible but not assignable.
    reports.push_back(Report{verdict, dst->name_hint_, ctx.origin, ctx.instance_word, elem_bytes,
                             DataTypeToString(tensor_type->dtype_), stride_bytes, op->span_});
  }

 private:
  struct Context {
    std::string origin;
    std::string instance_word;
    bool dim_allocated;  ///< False once nesting exceeds kMaxInstanceDims.
  };

  [[nodiscard]] InstAffine Eval(const ExprPtr& expr) const {
    if (!expr) return InstAffine::Unknown();
    if (auto c = As<ConstInt>(expr)) return InstAffine::Const(c->value_);
    if (IsBlockIndexCall(expr)) {
      // Resolves to the innermost enclosing SPMD dimension, or to the caller's
      // dispatch dimension when this function was entered multi-instance.
      if (spmd_dim_stack_.empty()) return InstAffine::Unknown();
      return InstAffine::Instance(spmd_dim_stack_.back());
    }
    if (auto v = AsVarLike(expr)) {
      auto it = bindings_.find(v.get());
      return it == bindings_.end() ? InstAffine::Unknown() : it->second;
    }
    // Index casts (INT32 <-> INDEX) preserve the value in every shape the DSL
    // produces here.
    if (auto cast = As<Cast>(expr)) return Eval(cast->operand_);
    if (auto add = As<Add>(expr)) return AffineAdd(Eval(add->left_), Eval(add->right_));
    if (auto sub = As<Sub>(expr)) return AffineSub(Eval(sub->left_), Eval(sub->right_));
    if (auto mul = As<Mul>(expr)) return AffineMul(Eval(mul->left_), Eval(mul->right_));
    return InstAffine::Unknown();
  }

  /// Model for a sequential `for v in range(start, stop, step)`.
  [[nodiscard]] InstAffine SequentialLoopVarModel(const ForStmtPtr& op) const {
    const InstAffine start = Eval(op->start_);
    const InstAffine stop = Eval(op->stop_);
    const InstAffine step = Eval(op->step_);
    if (!start.known || !stop.known || !step.known) return InstAffine::Unknown();

    // Only a forward walk is modelled. A negative or instance-dependent step
    // makes the value set run the other way (or vary per instance), which the
    // `sum(coef*i) + [lo, hi)` shape cannot express.
    if (!IsConstAffine(step) || step.lo <= 0) return InstAffine::Unknown();

    // `v` inherits `start`'s instance coefficients; the remainder
    // `r = v - sum(coef*i)` is bounded below by `start.lo` and above by `stop`'s
    // top. That upper bound needs every `stop.coef[d] <= start.coef[d]`: with
    // `i_d >= 0`,
    //     v < stop <= sum(stop.coef*i) + (stop.hi - 1) <= sum(coef*i) + (stop.hi - 1).
    // The grid-strided `pl.range(blk, N, BLOCKS)` is exactly the
    // `stop.coef == 0 < start.coef == 1` case, so requiring equality here would
    // lose the very shape this check exists to catch.
    for (size_t d = 0; d < kMaxInstanceDims; ++d) {
      if (stop.coef[d] > start.coef[d]) return InstAffine::Unknown();
    }
    InstAffine v = start;
    v.lo = start.lo;
    v.hi = stop.hi - 1;
    if (v.hi <= v.lo) return InstAffine::Unknown();  // empty or malformed range
    return v;
  }

  /// Flat element offset of `indices` into a tensor of `shape`, row-major.
  /// Unknown if any extent is dynamic — a symbolic extent makes the stride
  /// itself unknown, so nothing downstream can be proven.
  [[nodiscard]] InstAffine FlatOffset(const MakeTuplePtr& indices, const std::vector<ExprPtr>& shape) const {
    if (!indices || indices->elements_.size() != shape.size()) return InstAffine::Unknown();

    InstAffine acc = InstAffine::Const(0);
    for (size_t i = 0; i < indices->elements_.size(); ++i) {
      // Row-major stride of dimension i is the product of all later extents.
      int64_t stride = 1;
      for (size_t d = i + 1; d < shape.size(); ++d) {
        auto dim = As<ConstInt>(shape[d]);
        if (!dim) return InstAffine::Unknown();
        if (MulOverflows(stride, dim->value_, &stride)) return InstAffine::Unknown();
      }
      acc = AffineAdd(acc, AffineMul(Eval(indices->elements_[i]), InstAffine::Const(stride)));
      if (!acc.known) return InstAffine::Unknown();
    }
    return acc;
  }

  /// Pop must mirror push exactly. A push past `kMaxInstanceDims` allocates no
  /// dimension, so popping it unconditionally would drop an *outer* dimension's
  /// entry and leave every enclosing scope mis-modelled. The per-context flag
  /// records what the push actually did.
  void PushInstanceDim(bool is_spmd, std::string origin, std::string instance_word) {
    const bool dim_allocated = instance_dims_ < kMaxInstanceDims;
    if (dim_allocated) {
      if (is_spmd) spmd_dim_stack_.push_back(instance_dims_);
      ++instance_dims_;
    } else {
      ++unmodelled_dims_;
    }
    contexts_.push_back({std::move(origin), std::move(instance_word), dim_allocated});
  }

  void PopInstanceDim(bool is_spmd) {
    const bool dim_allocated = contexts_.back().dim_allocated;
    if (dim_allocated) {
      if (is_spmd && !spmd_dim_stack_.empty()) spmd_dim_stack_.pop_back();
      if (instance_dims_ > 0) --instance_dims_;
    } else if (unmodelled_dims_ > 0) {
      --unmodelled_dims_;
    }
    contexts_.pop_back();
  }

  /// True for a tensor allocated inside the current scope: each instance runs
  /// its own copy, so no other instance can observe writes to it.
  [[nodiscard]] static bool IsInstancePrivateAllocation(const ExprPtr& value) {
    auto call = As<Call>(value);
    return call && (IsOp(call, "tensor.create") || IsOp(call, "tensor.full"));
  }

  void NoteScopeLocal(const Var* v) {
    if (contexts_.empty()) return;
    scope_local_stack_.push_back(v);
    scope_local_set_.insert(v);
  }

  [[nodiscard]] bool IsScopeLocal(const Var* v) const {
    // O(1). A linear scan here would make the whole check O(N^2) on a scope
    // holding many allocations (see .claude/rules/pass-complexity.md).
    return scope_local_set_.find(v) != scope_local_set_.end();
  }

  void TrimScopeLocals(size_t mark) {
    while (scope_local_stack_.size() > mark) {
      scope_local_set_.erase(scope_local_stack_.back());
      scope_local_stack_.pop_back();
    }
  }

  size_t instance_dims_ = 0;
  size_t unmodelled_dims_ = 0;  ///< Instance scopes past the tracking limit.
  std::vector<size_t> spmd_dim_stack_;
  std::vector<Context> contexts_;
  std::vector<const Var*> scope_local_stack_;
  std::unordered_set<const Var*> scope_local_set_;
  std::unordered_map<const Var*, InstAffine> bindings_;
};

std::string BuildMessage(const Report& r, const std::string& func_name) {
  const int64_t per_line = r.elem_bytes > 0 ? kCacheLineBytes / r.elem_bytes : 0;
  std::ostringstream msg;
  msg << "pl.write into '" << r.tensor_name << "' from " << r.origin << " in function '" << func_name
      << "': ";

  if (r.verdict == LineVerdict::Interleaved) {
    msg << "consecutive " << r.instance_word << " write " << r.stride_bytes << " bytes apart, so "
        << (r.stride_bytes > 0 ? kCacheLineBytes / r.stride_bytes : 0) << " of them share each "
        << kCacheLineBytes << "-byte cache line and their stores overwrite one another. ";
  } else {
    msg << "the index is computed at runtime, so the compiler cannot tell whether two " << r.instance_word
        << " share a " << kCacheLineBytes << "-byte cache line. ";
  }

  msg << "A scalar write reaches DDR a whole " << kCacheLineBytes
      << "-byte line at a time, carrying the neighbouring elements as the writing core last saw "
         "them, so two "
      << r.instance_word << " sharing a line silently lose each other's stores. Give each one whole "
      << kCacheLineBytes << "-byte lines (" << per_line << " x " << r.dtype_name
      << "), or issue the writes from a single instance (pl.spmd(1)).";
  return msg.str();
}

}  // namespace

class ScalarWriteLineSharedVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "ScalarWriteLineShared"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    const std::unordered_set<std::string> multi_instance = MultiInstanceFunctions::Compute(program);

    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      const bool entered_multi = multi_instance.find(func->name_) != multi_instance.end();
      ScalarWriteVisitor visitor(entered_multi);
      visitor.VisitStmt(func->body_);
      for (const auto& report : visitor.reports) {
        diagnostics.emplace_back(DiagnosticSeverity::Warning, "ScalarWriteLineShared",
                                 kScalarWriteLineSharedCode, BuildMessage(report, func->name_), report.span);
      }
    }
  }
};

PropertyVerifierPtr CreateScalarWriteLineSharedWarningVerifier() {
  return std::make_shared<ScalarWriteLineSharedVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
