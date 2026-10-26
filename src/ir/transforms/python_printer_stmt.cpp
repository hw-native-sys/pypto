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
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/attrs.h"
#include "pypto/ir/type.h"
#include "src/ir/transforms/python_printer.h"

namespace pypto {
namespace ir {

/// Convert SplitMode to Python enum member name (UP_DOWN, LEFT_RIGHT).
std::string IRPythonPrinter::SplitModeToPythonString(SplitMode mode) {
  switch (mode) {
    case SplitMode::None:
      return "NONE";
    case SplitMode::UpDown:
      return "UP_DOWN";
    case SplitMode::LeftRight:
      return "LEFT_RIGHT";
  }
  throw pypto::TypeError("Unknown SplitMode");
}

namespace {

/// Detects a ``tile.get_block_idx`` read anywhere in a statement subtree.
class BlockIdxReadDetector : public IRVisitor {
 public:
  void VisitExpr_(const CallPtr& op) override {
    if (IsOp(op, "tile.get_block_idx")) found_ = true;
    IRVisitor::VisitExpr_(op);
  }

  [[nodiscard]] bool found() const { return found_; }

 private:
  bool found_ = false;
};

/// Whether the flattened SPMD spelling can preserve the carrier's attributes.
bool SpmdInlineBodyPreservesAttrs(const InCoreScopeStmtPtr& incore) {
  if (!incore) return false;
  // Only these attributes are reconstructed by the flattened SPMD spelling.
  // Keep the explicit InCore header for scoped metadata such as dump_vars.
  for (const auto& [key, value] : incore->attrs_) {
    if (key != "slot_num" && key != "local_slot_num" && key != kAttrCachePolicyVars) return false;
  }
  return true;
}

/// True when re-parsing ``with pl.spmd(...):`` with ``incore``'s statements printed
/// inline would rebuild the InCore carrier.
///
/// The with-form parser synthesises a carrier only for a body that carries an
/// ``optimizations=`` entry (a concrete split mode or a ``slot_num`` slot count —
/// the same pair ``PrintScopeOptimizations`` emits) or reads the per-block index.
/// For any other body the sugar is lossy: the carrier must be spelled out as a
/// nested ``pl.at(level=pl.Level.CORE_GROUP)`` instead, or the round-trip silently
/// drops it.
bool SpmdInlineBodyRebuildsCarrier(const InCoreScopeStmtPtr& incore) {
  if (!SpmdInlineBodyPreservesAttrs(incore)) return false;
  const bool has_mode = incore->split_ != SplitMode::None;
  if (has_mode || incore->HasAttr("slot_num")) return true;
  BlockIdxReadDetector detector;
  detector.VisitStmt(incore->body_);
  return detector.found();
}

}  // namespace

// Statement visitors with proper Python syntax
void IRPythonPrinter::VisitStmt_(const AssignStmtPtr& op) {
  // Print with type annotation: var: type = value
  // In concise mode, omit the type annotation: var = value
  VisitExpr(op->var_);
  if (!concise_) {
    stream_ << ": " << Print(op->var_->GetType());
  }
  stream_ << " = ";
  force_recorded_call_ = As<Call>(op->value_) && HasStorageType(op->var_->GetType());
  VisitExpr(op->value_);
  force_recorded_call_ = false;
}

void IRPythonPrinter::VisitStmt_(const IfStmtPtr& op) {
  // SSA-style if with pl.yield_()
  stream_ << "if ";
  VisitExpr(op->condition_);
  stream_ << ":\n";

  IncreaseIndent();
  VisitStmtBody(op->then_body_, op->return_vars_);
  DecreaseIndent();

  if (op->else_body_.has_value()) {
    stream_ << "\n" << GetIndent() << "else:\n";
    IncreaseIndent();
    VisitStmtBody(*op->else_body_, op->return_vars_);
    DecreaseIndent();
  }
}

void IRPythonPrinter::VisitStmt_(const YieldStmtPtr& op) {
  // Note: In function context, this will be changed to "return" by VisitFunction
  stream_ << prefix_ << ".yield_(";
  for (size_t i = 0; i < op->value_.size(); ++i) {
    if (i > 0) stream_ << ", ";
    VisitExpr(op->value_[i]);
  }
  stream_ << ")";
}

void IRPythonPrinter::VisitStmt_(const ReturnStmtPtr& op) {
  stream_ << "return";
  if (!op->value_.empty()) {
    stream_ << " ";
    for (size_t i = 0; i < op->value_.size(); ++i) {
      if (i > 0) stream_ << ", ";
      VisitExpr(op->value_[i]);
    }
  }
}

void IRPythonPrinter::VisitStmt_(const ForStmtPtr& op) {
  CHECK_SPAN(op->hardware_loop_policy_ == HardwareLoopPolicy::Auto, op->span_)
      << "HardwareLoopPolicy.Require Python DSL printing is not supported yet; use serialization";
  // SSA-style for with pl.range() or pl.parallel() - no inline type annotations in unpacking
  stream_ << "for " << GetVarName(op->loop_var_.get());

  // If we have iter_args, add tuple unpacking without type annotations
  if (!op->iter_args_.empty()) {
    stream_ << ", (";
    for (size_t i = 0; i < op->iter_args_.size(); ++i) {
      if (i > 0) stream_ << ", ";
      stream_ << GetVarName(op->iter_args_[i].get());
    }
    // Add trailing comma for single-element tuples to distinguish from parenthesized expression
    if (op->iter_args_.size() == 1) {
      stream_ << ",";
    }
    stream_ << ")";
  }

  // Kind drives syntax unconditionally (see ir-print-form-follows-kind). The
  // structural invariant `kind == Pipeline ⇔ pipeline_stages attr present`
  // (PipelineLoopValid, always-on) lets us read `stage=` from the attr below
  // without a dual-mode branch.
  const char* range_func = ".range(";
  switch (op->kind_) {
    case ForKind::Unroll:
      range_func = ".unroll(";
      break;
    case ForKind::Parallel:
      range_func = ".parallel(";
      break;
    case ForKind::Pipeline:
      range_func = ".pipeline(";
      break;
    case ForKind::Sequential:
      break;
    default:
      INTERNAL_CHECK_SPAN(false, op->span_)
          << "Unknown ForKind in python_printer: " << static_cast<int>(op->kind_);
      break;
  }
  stream_ << " in " << prefix_ << range_func;

  // Use concise range form like Python: range(stop) when start==0 and step==1,
  // range(start, stop) when step==1, range(start, stop, step) otherwise.
  auto is_const_int = [](const ExprPtr& expr, int64_t value) -> bool {
    if (auto ci = As<ConstInt>(expr)) {
      // Elide start/step only when the omitted literal round-trips identically.
      // An elided bound reparses as ConstInt(INDEX) (the parser's bare-literal
      // dtype), so elision is sound only for INDEX-typed bounds. INT64 and any
      // other dtype must print explicitly via pl.const(...) to preserve fidelity.
      return ci->value_ == value && ci->dtype() == DataType::INDEX;
    }
    return false;
  };

  bool start_is_zero = is_const_int(op->start_, 0);
  bool step_is_one = is_const_int(op->step_, 1);

  if (start_is_zero && step_is_one) {
    // range(stop)
    VisitExpr(op->stop_);
  } else if (step_is_one) {
    // range(start, stop)
    VisitExpr(op->start_);
    stream_ << ", ";
    VisitExpr(op->stop_);
  } else {
    // range(start, stop, step)
    VisitExpr(op->start_);
    stream_ << ", ";
    VisitExpr(op->stop_);
    stream_ << ", ";
    VisitExpr(op->step_);
  }

  // When rendering as `.pipeline(...)`, surface `pipeline_stages` as the required
  // `stage=` kwarg. The attr itself is stripped from the printed attrs={...} dict
  // below so it never leaks as storage detail. PipelineLoopValid guarantees the
  // attr is present whenever the kind is Pipeline; assert it here so a malformed
  // loop reaching the printer (e.g. with verification disabled) fails loudly
  // instead of emitting `stage=0`, which the parser rejects.
  if (op->kind_ == ForKind::Pipeline) {
    INTERNAL_CHECK_SPAN(op->HasAttr(kPipelineStagesAttr), op->span_)
        << "ForKind::Pipeline loop missing attrs[\"" << kPipelineStagesAttr << "\"]";
    stream_ << ", stage=" << op->GetAttr<int>(kPipelineStagesAttr, 0);
  }
  if (!op->iter_args_.empty()) {
    stream_ << ", init_values=(";
    for (size_t i = 0; i < op->iter_args_.size(); ++i) {
      if (i > 0) stream_ << ", ";
      VisitExpr(op->iter_args_[i]->initValue_);
    }
    // Add trailing comma for single-element tuple
    if (op->iter_args_.size() == 1) stream_ << ",";
    stream_ << ")";
  }

  // Add attrs kwargs. When the loop prints as `.pipeline(...)`, `pipeline_stages`
  // is already surfaced as `stage=` above and must be stripped from the visible
  // attrs dict — the kind drives the textual form, not the storage mechanism.
  // Emit the `, attrs={` header lazily so we can skip it entirely when every
  // attr is filtered out.
  bool header_emitted = false;
  for (const auto& [key, value] : op->attrs_) {
    if (op->kind_ == ForKind::Pipeline && key == kPipelineStagesAttr) continue;
    stream_ << (header_emitted ? ", " : ", attrs={");
    header_emitted = true;
    stream_ << std::quoted(key) << ": ";
    if (value.type() == typeid(int)) {
      stream_ << AnyCast<int>(value, key);
    } else if (value.type() == typeid(double)) {
      stream_ << FormatFloatLiteral(AnyCast<double>(value, key));
    } else if (value.type() == typeid(float)) {
      stream_ << FormatFloatLiteral(static_cast<double>(AnyCast<float>(value, key)));
    } else if (value.type() == typeid(bool)) {
      stream_ << (AnyCast<bool>(value, key) ? "True" : "False");
    } else if (value.type() == typeid(std::string)) {
      stream_ << std::quoted(AnyCast<std::string>(value, key));
    } else {
      INTERNAL_CHECK(false) << "Unsupported attrs value type for key '" << key
                            << "': " << DemangleTypeName(value.type().name());
    }
  }
  if (header_emitted) stream_ << "}";

  stream_ << "):\n";

  IncreaseIndent();
  VisitStmtBody(op->body_, op->return_vars_);
  DecreaseIndent();
}

void IRPythonPrinter::VisitStmt_(const WhileStmtPtr& op) {
  // Check if this is SSA-style (with iter_args) or natural style
  if (op->iter_args_.empty()) {
    // Natural while loop without iter_args
    stream_ << "while ";
    VisitExpr(op->condition_);
    stream_ << ":\n";

    IncreaseIndent();
    VisitStmtBody(op->body_, op->return_vars_);
    DecreaseIndent();
  } else {
    // SSA-style while with iter_args - print as explicit DSL syntax
    stream_ << "for (";
    for (size_t i = 0; i < op->iter_args_.size(); ++i) {
      if (i > 0) stream_ << ", ";
      stream_ << GetVarName(op->iter_args_[i].get());
    }
    // Add trailing comma for single-element tuples
    if (op->iter_args_.size() == 1) {
      stream_ << ",";
    }
    stream_ << ") in " << prefix_ << ".while_(init_values=(";

    // Add init_values for iter_args
    for (size_t i = 0; i < op->iter_args_.size(); ++i) {
      if (i > 0) stream_ << ", ";
      VisitExpr(op->iter_args_[i]->initValue_);
    }
    // Add trailing comma for single-element tuple
    if (op->iter_args_.size() == 1) stream_ << ",";
    stream_ << ")):\n";

    IncreaseIndent();

    // Print condition as pl.cond() call as first body statement
    stream_ << GetIndent() << prefix_ << ".cond(";
    VisitExpr(op->condition_);
    stream_ << ")\n";

    VisitStmtBody(op->body_, op->return_vars_);
    DecreaseIndent();
  }
}

// Surface ``ScopeStmt.attrs[arg_direction_overrides_vars]`` (set by
// ``pl.at(no_dep_args=[t1, t2])``) as a trailing ``no_dep_args=[...]`` kwarg
// so the roundtrip parser can recover it. Returns true when something was
// printed.
// Emit a comma-prefixed kwarg ``, <name>=[v1, v2, ...]`` from the list-of-VarPtr
// attr ``key`` on ``op``. Null entries are skipped so the rendered Python is
// always syntactically valid (e.g. ``deps=[t1, t3]`` rather than ``deps=[t1, , t3]``)
// — some transforms intentionally leave null slots in dep-edge lists.
// Returns false (no output) when the attr is missing or every entry is null.
bool IRPythonPrinter::PrintScopeVarListKwarg(const ScopeStmtPtr& op, const char* attr_key,
                                             const char* kwarg_name, bool leading_comma) {
  for (const auto& [k, v] : op->attrs_) {
    if (k != attr_key) continue;
    const auto* vars = std::any_cast<std::vector<VarPtr>>(&v);
    if (!vars) continue;
    std::vector<const Var*> non_null;
    non_null.reserve(vars->size());
    for (const auto& var : *vars) {
      if (var) non_null.push_back(var.get());
    }
    if (non_null.empty()) continue;
    stream_ << (leading_comma ? ", " : "") << kwarg_name << "=[";
    for (size_t i = 0; i < non_null.size(); ++i) {
      if (i > 0) stream_ << ", ";
      stream_ << GetVarName(non_null[i]);
    }
    stream_ << "]";
    return true;
  }
  return false;
}

bool IRPythonPrinter::PrintScopeNoDepsAttr(const ScopeStmtPtr& op) {
  return PrintScopeVarListKwarg(op, kAttrArgDirOverrideVars, "no_dep_args");
}

bool IRPythonPrinter::PrintScopeDepsAttr(const ScopeStmtPtr& op) {
  return PrintScopeVarListKwarg(op, kAttrManualDepEdges, "deps");
}

// One ``<indent>pl.set_cache_policy(<tensor>, pl.CachePolicy.<NAME>)`` line per
// declaration. Callers invoke this after ``IncreaseIndent()`` and before the
// body block, so the markers lead the scope body exactly as the parser expects
// to find them. Null Vars are skipped for the same reason
// ``PrintScopeVarListKwarg`` skips them: the rendered Python must stay valid.
bool IRPythonPrinter::PrintScopeCachePolicyStmts(const ScopeStmtPtr& op) {
  if (!op) return false;
  bool printed = false;
  for (const auto& [k, v] : op->attrs_) {
    if (k != kAttrCachePolicyVars) continue;
    const auto* entries = std::any_cast<std::vector<std::pair<VarPtr, int>>>(&v);
    if (!entries) continue;
    for (const auto& [var, policy] : *entries) {
      if (!var) continue;
      stream_ << GetIndent() << prefix_ << ".set_cache_policy(" << GetVarName(var.get()) << ", " << prefix_
              << ".CachePolicy."
              << AttrValueToEnumMember(CachePolicyToString(static_cast<CachePolicy>(policy))) << ")\n";
      printed = true;
    }
  }
  return printed;
}

void IRPythonPrinter::PrintScopeOptimizations(SplitMode split, const ScopeStmtPtr& slot_num_holder) {
  const bool has_mode = split != SplitMode::None;
  const bool has_slot_num = slot_num_holder && slot_num_holder->HasAttr("slot_num");
  if (!has_mode && !has_slot_num) return;
  stream_ << ", optimizations=[";
  if (has_mode) {
    stream_ << prefix_ << ".split(" << prefix_ << ".SplitMode." << SplitModeToPythonString(split) << ")";
  }
  if (has_slot_num) {
    if (has_mode) stream_ << ", ";
    stream_ << prefix_ << ".cross_core_slot(slot_num=" << slot_num_holder->GetAttr<int>("slot_num", 0);
    if (slot_num_holder->HasAttr("local_slot_num")) {
      stream_ << ", local_slot_num=" << slot_num_holder->GetAttr<int>("local_slot_num", 0);
    }
    stream_ << ")";
  }
  stream_ << "]";
}

bool IRPythonPrinter::PrintScopeDumpAttr(const ScopeStmtPtr& op) {
  return PrintScopeVarListKwarg(op, kAttrDumpVars, "dumps");
}

bool IRPythonPrinter::PrintScopeAllowEarlyResolveAttr(const ScopeStmtPtr& op) {
  if (!op->GetAttr<bool>("allow_early_resolve", false)) return false;
  stream_ << ", allow_early_resolve=True";
  return true;
}

bool IRPythonPrinter::PrintScopePredicateAttr(const ScopeStmtPtr& op) {
  auto pred = op->GetAttr<ExprPtr>(kAttrPredicate, nullptr);
  if (!pred) return false;
  stream_ << ", predicate=(";
  VisitExpr(pred);
  stream_ << ")";
  return true;
}

bool IRPythonPrinter::PrintScopeWindowizeAttr(const ScopeStmtPtr& op) {
  if (!op->GetAttr<bool>("windowize", false)) return false;
  stream_ << ", windowize=True";
  return true;
}

bool IRPythonPrinter::PrintScopeTaskIdVarSuffix(const ScopeStmtPtr& op) {
  for (const auto& [k, v] : op->attrs_) {
    if (k != kAttrTaskIdVar) continue;
    const auto* tid = std::any_cast<VarPtr>(&v);
    if (!tid || !*tid) continue;
    stream_ << " as " << GetVarName((*tid).get());
    return true;
  }
  return false;
}

VarPtr IRPythonPrinter::GetScopeTaskIdVar(const StmtPtr& stmt) const {
  // ``As<ScopeStmt>`` is the polymorphic form — KindTrait<ScopeStmt> matches
  // every concrete scope subclass (see include/pypto/ir/kind_traits.h:152).
  // InCore / Hierarchy scopes carry ``kAttrTaskIdVar`` via
  // ``with pl.at(...) as tid:``; Spmd scopes carry it via
  // ``with pl.spmd(...) as tid:``. Cluster / Runtime scopes never do, but
  // ``GetAttr<VarPtr>`` returns null for them, so the broader cast is harmless.
  if (auto s = As<ScopeStmt>(stmt)) return s->GetAttr<VarPtr>(kAttrTaskIdVar);
  return nullptr;
}

bool IRPythonPrinter::IsTaskInvalidPlaceholderFor(const StmtPtr& candidate, const VarPtr& tid_var) const {
  if (!tid_var) return false;
  auto assign = As<AssignStmt>(candidate);
  if (!assign) return false;
  // Preserve any leading comments the user (or another transform) attached to
  // the placeholder itself — only the synthetic, comment-less placeholder is
  // safe to drop.
  if (!assign->leading_comments_.empty()) return false;
  if (assign->var_.get() != tid_var.get()) return false;
  auto call = As<Call>(assign->value_);
  if (!call || !call->op_) return false;
  return IsOp(call, "system.task_invalid");
}

bool IRPythonPrinter::ShouldSuppressPlaceholder(const std::vector<StmtPtr>& stmts, size_t i) const {
  if (i + 1 >= stmts.size()) return false;
  auto next_tid = GetScopeTaskIdVar(stmts[i + 1]);
  return IsTaskInvalidPlaceholderFor(stmts[i], next_tid);
}

void IRPythonPrinter::VisitStmt_(const HierarchyScopeStmtPtr& op) {
  // Print as: with pl.at(level=pl.Level.X, role=pl.Role.Y, [name_hint="..."]) [as <tid>]:
  stream_ << "with " << prefix_ << ".at(level=" << prefix_ << ".Level." << LevelToString(op->level_);
  if (op->role_.has_value()) {
    stream_ << ", role=" << prefix_ << ".Role." << RoleToString(*op->role_);
  }
  if (!op->name_hint_.empty()) {
    stream_ << ", name_hint=\"" << op->name_hint_ << "\"";
  }
  PrintScopeDepsAttr(op);
  PrintScopeNoDepsAttr(op);
  PrintScopeDumpAttr(op);
  PrintScopeAllowEarlyResolveAttr(op);
  // The parser rejects ``pl.at(level != CORE_GROUP, predicate=...)`` and
  // ScopeOutliner asserts the same for hand-built IR, so a Hierarchy scope
  // should never carry one. Print it anyway rather than silently dropping
  // state from a mid-pipeline dump of such IR.
  PrintScopePredicateAttr(op);
  PrintScopeWindowizeAttr(op);
  stream_ << ")";
  PrintScopeTaskIdVarSuffix(op);
  stream_ << ":\n";
  IncreaseIndent();
  PrintScopeCachePolicyStmts(op);
  PrintStmtBlock(op->body_);
  DecreaseIndent();
}

void IRPythonPrinter::VisitStmt_(const InCoreScopeStmtPtr& op) {
  stream_ << "with " << prefix_ << ".at(level=" << prefix_ << ".Level.CORE_GROUP";
  // Emit ``optimizations=[...]`` for a non-None split mode and/or a slot_num
  // (a NONE mixed kernel still drives a cross-core pipe and carries a slot
  // count). A plain InCore with neither prints as ``pl.at(level=...)`` with no
  // optimizations list.
  PrintScopeOptimizations(op->split_, op);
  if (!op->name_hint_.empty()) {
    stream_ << ", name_hint=\"" << op->name_hint_ << "\"";
  }
  PrintScopeDepsAttr(op);
  PrintScopeNoDepsAttr(op);
  PrintScopeDumpAttr(op);
  PrintScopeAllowEarlyResolveAttr(op);
  PrintScopePredicateAttr(op);
  PrintScopeWindowizeAttr(op);
  stream_ << ")";
  PrintScopeTaskIdVarSuffix(op);
  stream_ << ":\n";
  IncreaseIndent();
  PrintScopeCachePolicyStmts(op);
  PrintStmtBlock(op->body_);
  DecreaseIndent();
}

void IRPythonPrinter::VisitStmt_(const ClusterScopeStmtPtr& op) {
  stream_ << "with " << prefix_ << ".cluster(";
  const bool has_name_hint = !op->name_hint_.empty();
  if (has_name_hint) {
    stream_ << "name_hint=\"" << op->name_hint_ << "\"";
  }
  // ``dumps=`` may be the first kwarg here, so it leads with a comma only after
  // ``name_hint``.
  PrintScopeVarListKwarg(op, kAttrDumpVars, "dumps", /*leading_comma=*/has_name_hint);
  stream_ << "):\n";
  IncreaseIndent();
  PrintStmtBlock(op->body_);
  DecreaseIndent();
}

void IRPythonPrinter::VisitStmt_(const RegScopeStmtPtr& op) {
  // PR1 has a binary IR fixture, but no scope DSL parser yet. Omitting this
  // override would silently print the body without its register boundary.
  CHECK_SPAN(false, op->span_) << "RegScopeStmt Python DSL printing is not supported yet; use serialization";
}

void IRPythonPrinter::VisitStmt_(const GraphScopeStmtPtr& op) {
  // ``name_hint_`` is the region name the user wrote; the parser requires it,
  // so it is printed positionally rather than as an optional keyword.
  stream_ << "with " << prefix_ << ".graph(\"" << op->name_hint_ << "\"";
  PrintScopeDumpAttr(op);
  stream_ << "):\n";
  IncreaseIndent();
  PrintStmtBlock(op->body_);
  DecreaseIndent();
}

void IRPythonPrinter::VisitStmt_(const SpmdScopeStmtPtr& op) {
  // Detect the ``for i in pl.spmd(...):`` desugaring emitted by the parser:
  // SpmdScopeStmt(body=InCoreScopeStmt(body=<AssignStmt(i, Call(tile.get_block_idx)), ...>)).
  // Printing it back as a for-loop keeps round-trips stable: the for-form binds the
  // loop variable to the leading get_block_idx read, which the with-form does not.
  auto incore = As<InCoreScopeStmt>(op->body_);
  auto incore_seq = incore ? As<SeqStmts>(incore->body_) : nullptr;

  // ``with pl.spmd(...) [deps=] as tid:`` — the scope carries kAttrTaskIdVar
  // (the grid-dispatch producer TaskId, mirroring ``pl.at ... as tid``). Checked
  // BEFORE the for-form sniff because an inline ``as tid`` body's first statement
  // may itself be a user ``x = pl.tile.get_block_idx()`` call, which would
  // otherwise be mistaken for the synthesised for-loop variable. The body is the
  // inner InCore (auto-outlined kernel); print its statements directly (no
  // synthesised loop-var to skip), reconstructing optimizations / deps / `as tid`.
  if (op->GetAttr<VarPtr>(kAttrTaskIdVar)) {
    const bool inlines_incore_body = SpmdInlineBodyRebuildsCarrier(incore);
    stream_ << "with " << prefix_ << ".spmd(";
    VisitExpr(op->core_num_);
    if (op->sync_start_) {
      stream_ << ", sync_start=True";
    }
    if (!op->name_hint_.empty()) {
      stream_ << ", name_hint=\"" << op->name_hint_ << "\"";
    }
    if (inlines_incore_body) {
      PrintScopeOptimizations(incore->split_, incore);
    }
    PrintScopeDepsAttr(op);
    PrintScopeDumpAttr(op);
    PrintScopeAllowEarlyResolveAttr(op);
    PrintScopePredicateAttr(op);
    stream_ << ")";
    PrintScopeTaskIdVarSuffix(op);
    stream_ << ":\n";
    IncreaseIndent();
    // The inlining branches below drop the nested ``pl.at(level=CORE_GROUP)``
    // header, and a ``pl.set_cache_policy`` written inside the Spmd body
    // attaches to that inner InCore scope — so its declarations must be printed
    // here, from ``incore``, or they are lost. When the carrier is spelled out
    // instead, the nested InCore prints its own markers and re-printing them
    // here would duplicate them.
    PrintScopeCachePolicyStmts(op);
    if (inlines_incore_body) {
      PrintScopeCachePolicyStmts(incore);
    }
    if (!inlines_incore_body) {
      // Print the body as-is. Either inlining would lose the carrier — the parser
      // re-synthesises an InCore only for a body that carries a split or reads the
      // per-block index, and even then cannot recover a header-only kwarg such as
      // ``dumps=`` — so the carrier must be spelled out as a nested
      // ``pl.at(level=pl.Level.CORE_GROUP, ...)``, the same shape the plain
      // with-form prints; or (defensively) there is no InCore wrapper at all,
      // which an `as tid` Spmd scope should never be missing.
      PrintStmtBlock(op->body_);
    } else if (incore_seq && !incore_seq->stmts_.empty()) {
      for (size_t i = 0; i < incore_seq->stmts_.size(); ++i) {
        if (ShouldSuppressPlaceholder(incore_seq->stmts_, i)) continue;
        PrintStmtBlock(incore_seq->stmts_[i]);
        if (i + 1 < incore_seq->stmts_.size()) stream_ << "\n";
      }
    } else {
      // Non-null: SpmdInlineBodyRebuildsCarrier only returns true for a real InCore.
      PrintStmtBlock(incore->body_);
    }
    DecreaseIndent();
    return;
  }

  auto first_assign = incore_seq
                          ? (incore_seq->stmts_.empty() ? nullptr : As<AssignStmt>(incore_seq->stmts_[0]))
                          : (incore ? As<AssignStmt>(incore->body_) : nullptr);
  auto first_call = first_assign ? As<Call>(first_assign->value_) : nullptr;
  auto first_op = first_call ? As<Op>(first_call->op_) : nullptr;
  if (first_op && IsOp(first_op, "tile.get_block_idx") && SpmdInlineBodyPreservesAttrs(incore)) {
    stream_ << "for " << GetVarName(first_assign->var_.get()) << " in " << prefix_ << ".spmd(";
    VisitExpr(op->core_num_);
    if (op->sync_start_) {
      stream_ << ", sync_start=True";
    }
    if (!op->name_hint_.empty()) {
      stream_ << ", name_hint=\"" << op->name_hint_ << "\"";
    }
    if (incore) {
      PrintScopeOptimizations(incore->split_, incore);
    }
    // ``deps=`` needs no ``as tid``: the Spmd outliner synthesises the TaskId Var
    // it needs to emit a Submit, so the for-form carries edges too and must print
    // them or the round-trip drops the dependency.
    PrintScopeDepsAttr(op);
    PrintScopeDumpAttr(op);
    PrintScopeAllowEarlyResolveAttr(op);
    PrintScopePredicateAttr(op);
    stream_ << "):\n";
    IncreaseIndent();
    // The for-form drops the nested ``pl.at(level=CORE_GROUP)`` header, so the
    // inner InCore's declarations are printed here or they are lost.
    const bool printed_cache_policy = PrintScopeCachePolicyStmts(incore);
    // Emit the InCore body skipping the get_block_idx binding we just
    // materialized as the loop variable. ShouldSuppressPlaceholder keeps the
    // ``with pl.at(...) as tid:`` round-trip working if such a scope ever
    // appears nested inside the SPMD body (the lookahead operates on the
    // already-trimmed subrange via the same SeqStmts contract).
    if (incore_seq && incore_seq->stmts_.size() > 1) {
      for (size_t i = 1; i < incore_seq->stmts_.size(); ++i) {
        if (ShouldSuppressPlaceholder(incore_seq->stmts_, i)) continue;
        PrintStmtBlock(incore_seq->stmts_[i]);
        if (i + 1 < incore_seq->stmts_.size()) stream_ << "\n";
      }
    } else if (!printed_cache_policy) {
      // Only a body with nothing left after the loop-var binding needs the
      // ``pass`` filler — the marker statements above already count as body
      // content, and printing ``pass`` after them is both redundant and (were
      // the markers ever dropped) a silent loss of the declaration.
      stream_ << GetIndent() << "pass\n";
    }
    DecreaseIndent();
    return;
  }

  stream_ << "with " << prefix_ << ".spmd(";
  VisitExpr(op->core_num_);
  if (op->sync_start_) {
    stream_ << ", sync_start=True";
  }
  if (!op->name_hint_.empty()) {
    stream_ << ", name_hint=\"" << op->name_hint_ << "\"";
  }
  // Same as the for-form above: a plain ``with pl.spmd(...):`` may carry
  // ``manual_dep_edges`` without a captured TaskId.
  PrintScopeDepsAttr(op);
  PrintScopeDumpAttr(op);
  PrintScopeAllowEarlyResolveAttr(op);
  PrintScopePredicateAttr(op);
  stream_ << "):\n";
  IncreaseIndent();
  PrintStmtBlock(op->body_);
  DecreaseIndent();
}

void IRPythonPrinter::VisitStmt_(const SplitAivScopeStmtPtr& op) { PrintSplitAivScope(op); }

void IRPythonPrinter::PrintSplitAivScope(const SplitAivScopeStmtPtr& op,
                                         const std::vector<VarPtr>& return_vars) {
  // Round-trips as `for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.X): <body>`.
  // The region body begins with `aiv_id = tile.get_subblock_idx()`; that binding
  // is materialized as the loop variable and skipped in the body.
  auto body_seq = As<SeqStmts>(op->body_);
  auto first_assign = body_seq ? (body_seq->stmts_.empty() ? nullptr : As<AssignStmt>(body_seq->stmts_[0]))
                               : As<AssignStmt>(op->body_);
  auto first_call = first_assign ? As<Call>(first_assign->value_) : nullptr;
  auto first_op = first_call ? As<Op>(first_call->op_) : nullptr;
  const bool has_idx_binding = first_op && IsOp(first_op, "tile.get_subblock_idx");
  // DCE-safe: if the `aiv_id` binding was stripped, synthesize a name and print
  // the full body so the round-trip stays valid.
  const std::string var_name = has_idx_binding ? GetVarName(first_assign->var_.get()) : "aiv_id";
  stream_ << "for " << var_name << " in " << prefix_ << ".split_aiv(" << op->count_ << ", mode=" << prefix_
          << ".SplitMode." << SplitModeToPythonString(op->split_) << "):\n";
  IncreaseIndent();
  ++split_aiv_scope_depth_;
  const size_t start = has_idx_binding ? 1 : 0;
  if (body_seq && body_seq->stmts_.size() > start) {
    auto first_stmt = body_seq->stmts_.begin();
    if (has_idx_binding) ++first_stmt;
    std::vector<StmtPtr> stmts(first_stmt, body_seq->stmts_.end());
    VisitStmtBody(SeqStmts::Flatten(std::move(stmts), body_seq->span_), return_vars);
  } else if (!body_seq && !has_idx_binding) {
    VisitStmtBody(op->body_, return_vars);
  } else {
    stream_ << GetIndent() << "pass\n";
  }
  --split_aiv_scope_depth_;
  DecreaseIndent();
}

void IRPythonPrinter::VisitStmt_(const SeqStmtsPtr& op) {
  for (size_t i = 0; i < op->stmts_.size(); ++i) {
    if (ShouldSuppressPlaceholder(op->stmts_, i)) continue;
    PrintStmtBlock(op->stmts_[i]);
    if (i < op->stmts_.size() - 1) {
      stream_ << "\n";
    }
  }
}

void IRPythonPrinter::PrintLeadingComments(const StmtPtr& stmt) {
  for (const auto& line : stmt->leading_comments_) {
    stream_ << "# " << line << "\n" << GetIndent();
  }
}

void IRPythonPrinter::PrintStmtBlock(const StmtPtr& stmt) {
  if (auto seq = As<SeqStmts>(stmt)) {
    INTERNAL_CHECK(seq->leading_comments_.empty()) << "SeqStmts should not carry leading comments directly";
    for (size_t i = 0; i < seq->stmts_.size(); ++i) {
      if (ShouldSuppressPlaceholder(seq->stmts_, i)) continue;
      PrintStmtBlock(seq->stmts_[i]);
      if (i < seq->stmts_.size() - 1) stream_ << "\n";
    }
  } else {
    stream_ << GetIndent();
    PrintLeadingComments(stmt);
    VisitStmt(stmt);
  }
}

void IRPythonPrinter::VisitStmt_(const RuntimeScopeStmtPtr& op) {
  // Both AUTO (manual=false) and MANUAL (manual=true) runtime scopes have a DSL
  // surface and preserve their structural name_hint through `pl.scope(...)`.
  PrintRuntimeScopeHeader(op->manual_, op->name_hint_);
  IncreaseIndent();
  VisitStmtBody(op->body_);
  DecreaseIndent();
}

std::string IRPythonPrinter::GetWindowBufferName(const WindowBufferPtr& window) {
  auto it = window_buffer_names_.find(window.get());
  if (it != window_buffer_names_.end()) return it->second;
  // The label identifies a WindowBuffer object, independently of its base Var.
  // A suffix distinguishes different windows backed by the same allocation.
  const auto stem = GetVarName(window->base_.get()) + "_window";
  auto& count = window_buffer_name_counts_[stem];
  const auto name = count == 0 ? stem : stem + "_" + std::to_string(count);
  ++count;
  window_buffer_names_.emplace(window.get(), name);
  return name;
}

std::string IRPythonPrinter::PrintWindowBuffer(const WindowBufferPtr& window) {
  std::ostringstream oss;
  oss << prefix_ << "._dump.window_buffer(" << std::quoted(GetWindowBufferName(window))
      << ", base=" << std::quoted(GetVarName(window->base_.get())) << ", size=" << PrintSubExpr(window->size_)
      << ", load_from_host=" << (window->load_from_host_ ? "True" : "False")
      << ", store_to_host=" << (window->store_to_host_ ? "True" : "False") << ")";
  return oss.str();
}

void IRPythonPrinter::VisitStmt_(const CommDomainScopeStmtPtr& op) {
  ++comm_domain_depth_;
  stream_ << "with " << prefix_ << "._dump.comm_domain(name=" << std::quoted(op->name_hint_) << ", devices=[";
  for (size_t i = 0; i < op->devices_.size(); ++i) {
    if (i > 0) stream_ << ", ";
    stream_ << op->devices_[i];
  }
  stream_ << "], slots=[";
  for (size_t i = 0; i < op->slots_.size(); ++i) {
    if (i > 0) stream_ << ", ";
    INTERNAL_CHECK_SPAN(op->slots_[i], op->span_) << "CommDomainScopeStmt has null slot at index " << i;
    stream_ << PrintWindowBuffer(op->slots_[i]);
  }
  stream_ << "]";
  if (!op->attrs_.empty()) {
    stream_ << ", attrs={";
    for (size_t i = 0; i < op->attrs_.size(); ++i) {
      if (i > 0) stream_ << ", ";
      stream_ << std::quoted(op->attrs_[i].first) << ": ";
      PrintAttrValue(op->attrs_[i].second, op->span_);
    }
    stream_ << "}";
  }
  stream_ << "):\n";
  IncreaseIndent();
  if (auto seq = As<SeqStmts>(op->body_); seq && seq->stmts_.empty()) {
    stream_ << GetIndent() << "pass\n";
  } else {
    PrintStmtBlock(op->body_);
  }
  DecreaseIndent();
  --comm_domain_depth_;
}

void IRPythonPrinter::VisitStmt_(const EvalStmtPtr& op) {
  // Print expression statement: expr
  VisitExpr(op->expr_);
}

void IRPythonPrinter::VisitStmt_(const BreakStmtPtr& op) { stream_ << "break"; }

void IRPythonPrinter::VisitStmt_(const ContinueStmtPtr& op) { stream_ << "continue"; }

void IRPythonPrinter::VisitStmt_(const InlineStmtPtr& op) {
  // Emit the verbatim body. The caller already emitted the leading indent for
  // the first line; subsequent non-empty lines are re-indented to match.
  const std::string& body = op->body_;
  const std::string indent = GetIndent();
  for (std::string::size_type pos = 0; pos <= body.size();) {
    auto eol = body.find('\n', pos);
    std::string line = body.substr(pos, eol == std::string::npos ? std::string::npos : eol - pos);
    if (pos > 0) {
      stream_ << "\n";
      if (!line.empty()) stream_ << indent;
    }
    stream_ << line;
    if (eol == std::string::npos) break;
    pos = eol + 1;
  }
}

void IRPythonPrinter::VisitStmt_(const StmtPtr& op) { stream_ << op->TypeName(); }

void IRPythonPrinter::PrintYieldAssignmentVars(const std::vector<VarPtr>& return_vars) {
  if (return_vars.size() == 1) {
    stream_ << GetVarName(return_vars[0].get());
    if (!concise_) {
      stream_ << ": " << Print(return_vars[0]->GetType());
    }
  } else {
    for (size_t i = 0; i < return_vars.size(); ++i) {
      if (i > 0) stream_ << ", ";
      stream_ << GetVarName(return_vars[i].get());
    }
  }
}

void IRPythonPrinter::PrintYieldWithResults(const YieldStmtPtr& yield_stmt,
                                            const std::vector<VarPtr>& return_vars) {
  stream_ << GetIndent();
  PrintLeadingComments(yield_stmt);
  PrintYieldAssignmentVars(return_vars);
  stream_ << " = " << prefix_ << (return_vars.size() > 1 ? "._dump.yield_(" : ".yield_(");
  for (size_t i = 0; i < yield_stmt->value_.size(); ++i) {
    if (i > 0) stream_ << ", ";
    VisitExpr(yield_stmt->value_[i]);
  }
  if (return_vars.size() > 1) {
    stream_ << ", types=[";
    for (size_t k = 0; k < return_vars.size(); ++k) {
      if (k > 0) stream_ << ", ";
      stream_ << Print(return_vars[k]->GetType());
    }
    stream_ << "]";
  }
  stream_ << ")";
}

void IRPythonPrinter::VisitStmtBody(const StmtPtr& body, const std::vector<VarPtr>& return_vars) {
  // A single-statement SeqStmts is a transparent wrapper (e.g. a user-written
  // `with pl.auto_scope():` body before NormalizeStmtStructure collapses it).
  // Unwrap it first so a transparent scope reaches its branch below with
  // return_vars intact — otherwise a trailing return-var yield inside the scope
  // would lose its `var = pl.yield_(...)` assignment LHS.
  if (auto seq = As<SeqStmts>(body); seq && seq->stmts_.size() == 1) {
    VisitStmtBody(seq->stmts_[0], return_vars);
    return;
  }
  // Helper to visit statement body and wrap YieldStmt with assignment if needed
  if (auto yield_stmt = As<YieldStmt>(body)) {
    // If parent has return_vars, wrap yield as assignment
    if (!yield_stmt->value_.empty() && !return_vars.empty()) {
      PrintYieldWithResults(yield_stmt, return_vars);
    } else {
      stream_ << GetIndent();
      PrintLeadingComments(yield_stmt);
      VisitStmt(yield_stmt);
    }
  } else if (auto seq_stmts = As<SeqStmts>(body)) {
    // Process each statement in sequence
    if (seq_stmts->stmts_.empty()) {
      stream_ << GetIndent() << "pass";
      return;
    }
    for (size_t i = 0; i < seq_stmts->stmts_.size(); ++i) {
      if (ShouldSuppressPlaceholder(seq_stmts->stmts_, i)) continue;
      auto stmt = seq_stmts->stmts_[i];

      // Only the trailing spine can yield this owner's results. Forward them
      // through a trailing sequence/runtime/split wrapper as well as a yield.
      bool is_last = (i == seq_stmts->stmts_.size() - 1);
      if (is_last) {
        VisitStmtBody(stmt, return_vars);
      } else {
        PrintStmtBlock(stmt);
      }

      if (i < seq_stmts->stmts_.size() - 1) {
        stream_ << "\n";
      }
    }
  } else if (auto rscope = As<RuntimeScopeStmt>(body)) {
    // Both runtime modes share the enclosing control flow's result bindings.
    stream_ << GetIndent();
    PrintLeadingComments(rscope);
    PrintRuntimeScopeHeader(rscope->manual_, rscope->name_hint_);
    IncreaseIndent();
    VisitStmtBody(rscope->body_, return_vars);
    DecreaseIndent();
  } else if (auto split = As<SplitAivScopeStmt>(body)) {
    stream_ << GetIndent();
    PrintLeadingComments(split);
    PrintSplitAivScope(split, return_vars);
  } else {
    PrintStmtBlock(body);
  }
}

}  // namespace ir
}  // namespace pypto
