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

#ifndef SRC_IR_TRANSFORMS_PYTHON_PRINTER_H_
#define SRC_IR_TRANSFORMS_PYTHON_PRINTER_H_

#include <any>
#include <cstddef>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

/**
 * @brief Python-style IR printer
 *
 * Prints IR nodes in Python syntax with type annotations and SSA-style control flow.
 * This is the recommended printer for new code that outputs valid Python syntax.
 *
 * Key features:
 * - Type annotations (e.g., x: pl.INT64, a: pl.Tensor[[4, 8], pl.FP32])
 * - SSA-style if/for with pl.yield_() and pl.range()
 * - Op attributes as keyword arguments
 * - Program headers with # pypto.program: name
 */
// Private implementation shared by the printer translation units. All visitors
// operate on one instance: naming maps, signature context, scope depth, and the
// output stream must stay synchronized across expression/type/statement visits.
class IRPythonPrinter : public IRVisitor {
 public:
  explicit IRPythonPrinter(std::string prefix = "pl", bool concise = false, bool explicit_layout = false)
      : prefix_(std::move(prefix)), concise_(concise), explicit_layout_(explicit_layout) {}
  ~IRPythonPrinter() override = default;

  /**
   * @brief Print an IR node to a string in Python IR syntax
   *
   * @param node IR node to print (can be Expr, Stmt, Function, or Program)
   * @return Python-style string representation
   */
  std::string Print(const IRNodePtr& node);
  std::string Print(const TypePtr& type);

 protected:
  // Expression visitors
  void VisitExpr_(const VarPtr& op) override;
  void VisitExpr_(const IterArgPtr& op) override;
  void VisitExpr_(const MemRefPtr& op) override;
  void VisitExpr_(const WindowBufferPtr& op) override;
  void VisitExpr_(const ConstIntPtr& op) override;
  void VisitExpr_(const ConstFloatPtr& op) override;
  void VisitExpr_(const ConstBoolPtr& op) override;
  void VisitExpr_(const CallPtr& op) override;
  void VisitExpr_(const SubmitPtr& op) override;
  void VisitExpr_(const MakeTuplePtr& op) override;
  void VisitExpr_(const TupleGetItemExprPtr& op) override;

  // Binary operations
  void VisitExpr_(const AddPtr& op) override;
  void VisitExpr_(const SubPtr& op) override;
  void VisitExpr_(const MulPtr& op) override;
  void VisitExpr_(const FloorDivPtr& op) override;
  void VisitExpr_(const FloorModPtr& op) override;
  void VisitExpr_(const FloatDivPtr& op) override;
  void VisitExpr_(const MinPtr& op) override;
  void VisitExpr_(const MaxPtr& op) override;
  void VisitExpr_(const PowPtr& op) override;
  void VisitExpr_(const EqPtr& op) override;
  void VisitExpr_(const NePtr& op) override;
  void VisitExpr_(const LtPtr& op) override;
  void VisitExpr_(const LePtr& op) override;
  void VisitExpr_(const GtPtr& op) override;
  void VisitExpr_(const GePtr& op) override;
  void VisitExpr_(const AndPtr& op) override;
  void VisitExpr_(const OrPtr& op) override;
  void VisitExpr_(const XorPtr& op) override;
  void VisitExpr_(const BitAndPtr& op) override;
  void VisitExpr_(const BitOrPtr& op) override;
  void VisitExpr_(const BitXorPtr& op) override;
  void VisitExpr_(const BitShiftLeftPtr& op) override;
  void VisitExpr_(const BitShiftRightPtr& op) override;

  // Unary operations
  void VisitExpr_(const AbsPtr& op) override;
  void VisitExpr_(const NegPtr& op) override;
  void VisitExpr_(const NotPtr& op) override;
  void VisitExpr_(const BitNotPtr& op) override;
  void VisitExpr_(const CastPtr& op) override;

  // Statement visitors
  void VisitStmt_(const AssignStmtPtr& op) override;
  void VisitStmt_(const IfStmtPtr& op) override;
  void VisitStmt_(const YieldStmtPtr& op) override;
  void VisitStmt_(const ReturnStmtPtr& op) override;
  void VisitStmt_(const ForStmtPtr& op) override;
  void VisitStmt_(const WhileStmtPtr& op) override;
  void VisitStmt_(const InCoreScopeStmtPtr& op) override;
  void VisitStmt_(const ClusterScopeStmtPtr& op) override;
  void VisitStmt_(const HierarchyScopeStmtPtr& op) override;
  void VisitStmt_(const SpmdScopeStmtPtr& op) override;
  void VisitStmt_(const SplitAivScopeStmtPtr& op) override;
  void VisitStmt_(const RuntimeScopeStmtPtr& op) override;
  void VisitStmt_(const CommDomainScopeStmtPtr& op) override;
  void VisitStmt_(const GraphScopeStmtPtr& op) override;
  void VisitStmt_(const RegScopeStmtPtr& op) override;
  void VisitStmt_(const SeqStmtsPtr& op) override;
  void VisitStmt_(const EvalStmtPtr& op) override;
  void VisitStmt_(const BreakStmtPtr& op) override;
  void VisitStmt_(const ContinueStmtPtr& op) override;
  void VisitStmt_(const InlineStmtPtr& op) override;
  void VisitStmt_(const StmtPtr& op) override;

  // Function and program visitors
  void VisitFunction(const FunctionPtr& func) override;
  void VisitProgram(const ProgramPtr& program) override;

 private:
  std::ostringstream stream_;
  int indent_level_ = 0;
  // Nesting depth inside a printed SplitAivScopeStmt region. While > 0, the
  // redundant op-level ``split=`` kwarg on aiv_shard/aic_gather is suppressed
  // (the region's mode is the authoritative carrier; the parser re-stamps it).
  int split_aiv_scope_depth_ = 0;
  std::string prefix_;  // Prefix for type names (e.g., "pl" or "ir")
  bool concise_;        // When true, omit intermediate type annotations
  // When true, print every tile's fully-resolved blayout/slayout/fractal from
  // GetEffectiveTileView — including tiles whose canonical tile_view_ is nullopt
  // (whose layout would otherwise be silently implicit). Makes a dump
  // self-describing for tile layouts (opt-in debugging aid, issue #2088); the
  // concise canonical form stays the default.
  bool explicit_layout_;
  ProgramPtr current_program_ = nullptr;  // Track when printing within Program (for self.method() calls)
  std::unordered_map<const WindowBuffer*, std::string> window_buffer_names_;
  std::unordered_map<std::string, size_t> window_buffer_name_counts_;
  size_t comm_domain_depth_ = 0;

  // Per-function rename map: Var pointer → unique printed name.
  // Built by BuildVarRenameMap() at the start of each function to handle SSA name shadowing.
  std::unordered_map<const Var*, std::string> var_rename_map_;

  // Vars defined in the current function body (for PrintMemRef formatting).
  std::unordered_set<const Var*> body_defined_vars_;

  // True while printing a function signature (parameter annotations and return
  // types). Body definitions are not in scope there, so a MemRef whose base Ptr
  // is allocated in the body must print as a string literal rather than a bare
  // name — see PrintMemRef.
  bool printing_signature_ = false;

  // Free variables of the current function: Vars used in the body that are
  // neither a parameter nor a body-local definition. A well-formed function is
  // a closed scope, so a non-empty set marks malformed IR (e.g. a transform
  // that leaked another function's Var across the function boundary). These
  // get a visible suffix in GetVarName so the dump is unmistakably invalid.
  // Populated only for full functions, never for bare-stmt roots (a statement
  // fragment legitimately references vars defined by its absent context).
  std::unordered_set<const Var*> free_body_vars_;

  // Program-level dyn var rename map: Var pointer → disambiguated printed name.
  // Built once by VisitProgram for dynamic dimension variables used in type annotations.
  // When two distinct Var* share the same name_hint_, they get unique suffixed names.
  std::unordered_map<const Var*, std::string> dyn_var_rename_map_;

  // When true, INDEX-typed ConstInt leaves print as pl.const(v, pl.INDEX)
  // instead of bare literals. Enabled inside composite type-context dims
  // (shape / valid_shape) so the parser rebuilds them verbatim instead of
  // Python-eval folding `32 + 32` into `64`. Bare literals stay the default
  // everywhere else.
  bool typed_index_consts_ = false;
  bool force_recorded_call_ = false;
  bool orchestration_function_ = false;
  std::unordered_map<const Type*, bool> storage_type_cache_;

  // Formatting shared by expression attributes, loops, and function/scope headers.
  static std::string SplitModeToPythonString(SplitMode mode);
  static std::string AttrValueToEnumMember(std::string value);
  static std::string FormatFloatLiteral(double value);
  static std::string FormatFloatValue(double value, const std::string& prefix);

  // Helper methods
  std::string GetIndent() const;
  void IncreaseIndent();
  void DecreaseIndent();

  // Return the printed name for a Var, using rename map if SSA name shadowing occurred.
  std::string GetVarName(const Var* var) const;

  // Print an expression to a string without disturbing the main stream.
  //
  // Nested inside a type annotation (a MemRef byte offset, a slot index), an
  // expression cannot go through the main stream, so it needs its own printer.
  // That printer must still inherit the naming state: an offset or slot index may
  // name a Var whose printed name was disambiguated (two loops both spelled `i`
  // print as `i` and `i_1`), and a fresh printer would emit the bare name_hint_
  // and rebind it to the wrong Var on reparse.
  std::string PrintSubExpr(const ExprPtr& expr, bool typed_index = false) const;

  // Build var_rename_map_ from a body stmt (and optional params).
  // Assigns unique suffixed names (e.g., "i", "i_1") when two distinct Vars share a name.
  // Called from VisitFunction (with params) and from Print() for bare Stmt roots
  // (no params) so that standalone stmt.as_python() also gets disambiguation.
  void BuildVarRenameMap(const std::vector<VarPtr>& params, const StmtPtr& body, bool is_function = false);
  void BuildVarRenameMap(const FunctionPtr& func) {
    BuildVarRenameMap(func->params_, func->body_, /*is_function=*/true);
  }

  // Print a statement block at current indent level.
  // SeqStmts is a transparent container - recursed into without extra indent.
  void PrintStmtBlock(const StmtPtr& stmt);

  // Emit the `with pl.scope(...):` header for a RuntimeScopeStmt. AUTO prints as
  // the bare `pl.scope()`; MANUAL as `pl.scope(mode=pl.ScopeMode.MANUAL)`.
  // Callers emit the leading indent themselves (it differs by call site).
  void PrintRuntimeScopeHeader(bool manual, const std::string& name_hint = "") {
    stream_ << "with " << prefix_ << ".scope(";
    if (manual) stream_ << "mode=" << prefix_ << ".ScopeMode.MANUAL";
    if (!name_hint.empty()) {
      if (manual) stream_ << ", ";
      stream_ << "name_hint=\"";
      // Runtime labels do not become identifiers. Preserve arbitrary strings,
      // including control characters that std::quoted alone does not escape.
      constexpr char kHex[] = "0123456789abcdef";
      for (unsigned char ch : name_hint) {
        if (ch == '"' || ch == '\\') {
          stream_ << '\\' << ch;
        } else if (ch < 0x20 || ch == 0x7f) {
          stream_ << "\\x" << kHex[ch >> 4] << kHex[ch & 0xf];
        } else {
          stream_ << ch;
        }
      }
      stream_ << '"';
    }
    stream_ << "):\n";
  }

  // Emit a comma-prefixed kwarg ``, <kwarg_name>=[v1, v2, ...]`` from the
  // list-of-VarPtr attr keyed by ``attr_key`` on ``op``. Skips null entries so
  // the rendered Python stays syntactically valid. Returns false when the attr
  // is absent or every entry is null. Shared by the two scope-attr printers
  // below so the null-filter rule lives in one place. ``leading_comma=false``
  // is for a call whose first kwarg this may be (``pl.cluster(...)``).
  bool PrintScopeVarListKwarg(const ScopeStmtPtr& op, const char* attr_key, const char* kwarg_name,
                              bool leading_comma = true);

  // Emit ``no_dep_args=[t1, t2]`` if the scope carries ``kAttrArgDirOverrideVars``;
  // returns true when something was printed. Common to InCore/Hierarchy scope
  // printers so the parser can recover the marker after a print/reparse roundtrip.
  bool PrintScopeNoDepsAttr(const ScopeStmtPtr& op);

  // Surface ``ScopeStmt.attrs[kAttrCachePolicyVars]`` as leading marker
  // STATEMENTS in the scope body — ``pl.set_cache_policy(t, pl.CachePolicy.X)``
  // — and NOT as a header kwarg the way ``no_dep_args=`` / ``dumps=`` do,
  // because a statement is the DSL surface the parser accepts. The parser
  // hoists the markers back onto the scope attr wherever they appear in the
  // body, so the roundtrip is position-normalising: however the user ordered
  // them, they print first. Emitted from whichever scope object carries the
  // attr — in the Spmd inlining branches that is the nested InCore, not ``op``.
  // Returns true when something was printed, so a caller whose body may be
  // empty can tell whether a ``pass`` filler is still needed.
  bool PrintScopeCachePolicyStmts(const ScopeStmtPtr& op);

  // Emit ``deps=[t1, t2]`` if the scope carries ``kAttrManualDepEdges``; returns
  // true when something was printed. Mirrors PrintScopeNoDepsAttr — the parser
  // recovers ``deps=`` on ``pl.at(...)`` via _parse_at_meta.
  bool PrintScopeDepsAttr(const ScopeStmtPtr& op);

  // Emit ``dumps=[t1, t2]`` if the scope carries ``kAttrDumpVars``; returns
  // true when something was printed. Mirrors PrintScopeNoDepsAttr — the scope
  // dump list is the carrier from ``pl.dump_tag`` / an explicit ``dumps=`` to
  // the outliner (which translates it into the synthesised dispatch's
  // ``kAttrDumpVars``), so it must survive a print/reparse roundtrip while the
  // scope still exists.
  bool PrintScopeDumpAttr(const ScopeStmtPtr& op);

  // Emit ``allow_early_resolve=True`` if the scope carries the
  // ``allow_early_resolve`` attr; returns true when printed. The outliner reads
  // this bool off the scope and threads it onto the synthesised ``Submit`` —
  // it must survive a print/reparse roundtrip while the scope still exists.
  bool PrintScopeAllowEarlyResolveAttr(const ScopeStmtPtr& op);

  // Emit ``predicate=(<expr>)`` if the scope carries ``kAttrPredicate``;
  // returns true when printed. Same contract as
  // PrintScopeAllowEarlyResolveAttr — the outliner reads the predicate off the
  // scope and threads it onto the synthesised ``Submit``, so it must survive a
  // print/reparse roundtrip while the scope still exists. The comparison Expr
  // prints itself, so there is no bespoke syntax. Shared by the ``pl.spmd``
  // scope printers and the ``pl.at`` (InCore / Hierarchy) ones.
  bool PrintScopePredicateAttr(const ScopeStmtPtr& op);

  // Emit ``windowize=True`` for an explicitly opted-in InCore scope.
  bool PrintScopeWindowizeAttr(const ScopeStmtPtr& op);

  // Emit ``, optimizations=[...]`` for a scope carrying a concrete split mode
  // (UP_DOWN / LEFT_RIGHT) and/or a ``slot_num`` cross-core slot count, read
  // from ``slot_num_holder``'s attrs (the scope carrying it). The two are
  // independent entries — ``pl.split(pl.SplitMode.X)`` and
  // ``pl.cross_core_slot(slot_num=N)`` — so a bare slot count never fabricates
  // a ``pl.SplitMode.NONE`` (which OutlineIncoreScopes rejects on a scope
  // holding pl.split_aiv regions). Emits nothing when the scope carries
  // neither. Used by the flattened spmd with-tid / for-loop forms; the
  // nested-scope forms round-trip via the InCoreScopeStmt printer.
  void PrintScopeOptimizations(SplitMode split, const ScopeStmtPtr& slot_num_holder);

  // Emit `` as <tid>`` if the scope carries ``kAttrTaskIdVar``. The caller is
  // responsible for placing the ``)`` before and the ``:\n`` after this call.
  bool PrintScopeTaskIdVarSuffix(const ScopeStmtPtr& op);

  // Return the ``task_id_var`` VarPtr from any ``ScopeStmt`` subclass; nullptr
  // for any other stmt or when the attr is absent. Used by SeqStmts printing
  // to detect when an immediately-preceding ``AssignStmt(tid, system.task_invalid())``
  // placeholder should be suppressed (the printer re-creates it from the
  // ``as <tid>`` clause, so emitting the placeholder would double up after
  // reparse).
  VarPtr GetScopeTaskIdVar(const StmtPtr& stmt) const;
  bool IsTaskInvalidPlaceholderFor(const StmtPtr& candidate, const VarPtr& tid_var) const;

  // Return true when ``stmts[i]`` is the synthetic ``system.task_invalid()``
  // placeholder for the next-sibling scope's ``task_id_var`` and should be
  // skipped by the printer. Called from every site that iterates a SeqStmts —
  // keeping the lookahead in one helper avoids the 4-way drift hazard that
  // an inline check at each site would create.
  bool ShouldSuppressPlaceholder(const std::vector<StmtPtr>& stmts, size_t i) const;

  // Emit each leading comment line of `stmt` as `# <text>` above the stmt itself.
  // Assumes the current indent has already been written to the stream.
  void PrintLeadingComments(const StmtPtr& stmt);

  // Statement body visitor with SSA-style handling
  void VisitStmtBody(const StmtPtr& body, const std::vector<VarPtr>& return_vars = {});
  void PrintSplitAivScope(const SplitAivScopeStmtPtr& op, const std::vector<VarPtr>& return_vars = {});
  void PrintYieldAssignmentVars(const std::vector<VarPtr>& return_vars);
  void PrintYieldWithResults(const YieldStmtPtr& yield, const std::vector<VarPtr>& return_vars);
  void PrintLoopInitializer(const IterArgPtr& argument);

  // Binary/unary operator helpers (reuse precedence logic)
  void PrintBinaryOp(const BinaryExprPtr& op, const char* op_symbol);
  void PrintFunctionBinaryOp(const BinaryExprPtr& op, const char* func_name);
  void PrintChild(const ExprPtr& parent, const ExprPtr& child, bool is_left);
  bool NeedsParens(const ExprPtr& parent, const ExprPtr& child, bool is_left);

  // Shape printing helper
  void PrintShapeDims(std::ostringstream& oss, const std::vector<ExprPtr>& shape);

  // Print an expression for use in type annotations (shapes, views).
  // Uses GetVarName for Var nodes to pick up dyn_var_rename_map_ disambiguation.
  std::string PrintExprForType(const ExprPtr& expr);

  // MemRef and TileView printing helpers
  std::string PrintMemRef(const MemRef& memref);
  std::string GetWindowBufferName(const WindowBufferPtr& window);
  std::string PrintWindowBuffer(const WindowBufferPtr& window);
  void PrintCall(const CallPtr& op);
  bool HasStorageType(const TypePtr& type);
  std::string PrintTileView(const TileView& tile_view, const std::vector<ExprPtr>& tile_shape,
                            const std::optional<MemorySpace>& memory_space = std::nullopt);
  std::string PrintTensorView(const TensorView& tensor_view, const std::vector<ExprPtr>& tensor_shape);

  // Render one ``attrs={...}`` value as a parseable Python DSL expression. This
  // is the generic round-trip "safety net": every Call/Submit attr that has no
  // bespoke kwarg surface (deps=/device=/dumps=) and is not emitted by a
  // dedicated emitter (dump_vars/arg_directions) is rendered here, so print ->
  // parse never silently drops an attr. Dispatches on the ``std::any`` value
  // type and FAILS LOUD (INTERNAL_CHECK_SPAN) on a type it cannot represent in
  // the DSL — surfacing the gap rather than dropping the attr. The matching
  // reader is ``ast_parser._parse_attr_value``. Keep the two in sync.
  void PrintAttrValue(const std::any& value, const Span& span);
};

}  // namespace ir
}  // namespace pypto

#endif  // SRC_IR_TRANSFORMS_PYTHON_PRINTER_H_
