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

#include <algorithm>
#include <any>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/comm.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/pipe.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/printer.h"
#include "pypto/ir/transforms/utils/attrs.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

/// Convert SplitMode to Python enum member name (UP_DOWN, LEFT_RIGHT).
std::string SplitModeToPythonString(SplitMode mode) {
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
  if (!incore) return false;
  const bool has_mode = incore->split_ != SplitMode::None;
  if (has_mode || incore->HasAttr("slot_num")) return true;
  BlockIdxReadDetector detector;
  detector.VisitStmt(incore->body_);
  return detector.found();
}

/// Convert cast round mode integer to its string name for printing.
/// Inverse of the CAST_MODE_NAMES mapping in python/pypto/ir/utils.py.
std::string CastModeToString(int mode) {
  switch (mode) {
    case 0:
      return "none";
    case 1:
      return "rint";
    case 2:
      return "round";
    case 3:
      return "floor";
    case 4:
      return "ceil";
    case 5:
      return "trunc";
    case 6:
      return "odd";
    default:
      throw ValueError("Cast round mode must be in range [0, 6], got " + std::to_string(mode));
  }
}

/// Spell a lowercase IR attr value as its Python enum member name.
/// ``"soft"`` -> ``"SOFT"``, for the enums whose members are named after the
/// attr string they lower to.
std::string AttrValueToEnumMember(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return value;
}

/// Spell a cross-core `core_type` attr as its `pl.KernelType` member.
///
/// Both sync families name the same kernel, in their own vocabulary:
/// `system.syncall` writes a participant set (`"aic_only"` / `"aiv_only"` /
/// `"mix"`), the event ops pin one kernel (`"aic"` / `"aiv"`). Returns nullptr
/// for a spelling the op cannot carry, which is a malformed attr.
const char* CoreTypeToKernelMember(const std::string& value, bool is_syncall) {
  if (is_syncall) {
    if (value == "aic_only") return "AIC";
    if (value == "aiv_only") return "AIV";
    if (value == "mix") return "MIX";
    return nullptr;
  }
  if (value == "aic") return "AIC";
  if (value == "aiv") return "AIV";
  return nullptr;
}

/// Whether an op's printed DSL spelling needs a trailing underscore.
///
/// PyPTO spells wrappers whose name is a Python keyword with a trailing underscore
/// (``pypto/language/op/{tile,tensor}_ops.py``: ``and_``/``or_``/``not_``) while the IR
/// keeps the bare operator name. Printing the bare name yields ``pl.tile.and(a, b)``,
/// which is a syntax error — the round-trip parser cannot even compile it.
///
/// Keyed off the full Python keyword set rather than the three leaves that exist
/// today, so a future ``REGISTER_OP`` with a keyword leaf cannot silently
/// reintroduce unparseable output.
///
/// A predicate over ``string_view``, not a name-rewriting function: this runs on every
/// printed ``Call``, and roundtrip verification re-prints the whole program after every
/// pass. Returning a rewritten name would allocate on every call to serve the three ops
/// that actually need the suffix. The keys are string literals with static storage
/// duration, so the ``string_view`` set is safe.
bool NeedsDslUnderscore(std::string_view op_name) {
  static const std::unordered_set<std::string_view> kPythonKeywords = {
      "False", "None",     "True",  "and",    "as",   "assert", "async",  "await",    "break",
      "class", "continue", "def",   "del",    "elif", "else",   "except", "finally",  "for",
      "from",  "global",   "if",    "import", "in",   "is",     "lambda", "nonlocal", "not",
      "or",    "pass",     "raise", "return", "try",  "while",  "with",   "yield"};
  const size_t dot = op_name.rfind('.');
  if (dot == std::string_view::npos) return false;
  return kPythonKeywords.count(op_name.substr(dot + 1)) > 0;
}

}  // namespace

// Precedence mapping for each expression type
Precedence GetPrecedence(const ExprPtr& expr) {
  // Using a static map is more efficient and maintainable than a long chain of dynamic_casts.
  static const std::unordered_map<std::type_index, Precedence> kPrecedenceMap = {
      // Logical operators≥
      {std::type_index(typeid(Or)), Precedence::kOr},
      {std::type_index(typeid(Xor)), Precedence::kXor},
      {std::type_index(typeid(And)), Precedence::kAnd},
      {std::type_index(typeid(Not)), Precedence::kNot},

      // Comparison operators
      {std::type_index(typeid(Eq)), Precedence::kComparison},
      {std::type_index(typeid(Ne)), Precedence::kComparison},
      {std::type_index(typeid(Lt)), Precedence::kComparison},
      {std::type_index(typeid(Le)), Precedence::kComparison},
      {std::type_index(typeid(Gt)), Precedence::kComparison},
      {std::type_index(typeid(Ge)), Precedence::kComparison},

      // Bitwise operators
      {std::type_index(typeid(BitOr)), Precedence::kBitOr},
      {std::type_index(typeid(BitXor)), Precedence::kBitXor},
      {std::type_index(typeid(BitAnd)), Precedence::kBitAnd},
      {std::type_index(typeid(BitShiftLeft)), Precedence::kBitShift},
      {std::type_index(typeid(BitShiftRight)), Precedence::kBitShift},

      // Arithmetic operators
      {std::type_index(typeid(Add)), Precedence::kAddSub},
      {std::type_index(typeid(Sub)), Precedence::kAddSub},
      {std::type_index(typeid(Mul)), Precedence::kMulDivMod},
      {std::type_index(typeid(FloorDiv)), Precedence::kMulDivMod},
      {std::type_index(typeid(FloatDiv)), Precedence::kMulDivMod},
      {std::type_index(typeid(FloorMod)), Precedence::kMulDivMod},
      {std::type_index(typeid(Pow)), Precedence::kPow},

      // Unary operators
      {std::type_index(typeid(Neg)), Precedence::kUnary},
      {std::type_index(typeid(BitNot)), Precedence::kUnary},

      // Function-like operators and atoms
      {std::type_index(typeid(Abs)), Precedence::kCall},
      {std::type_index(typeid(Cast)), Precedence::kCall},
      {std::type_index(typeid(Min)), Precedence::kCall},
      {std::type_index(typeid(Max)), Precedence::kCall},
      {std::type_index(typeid(Call)), Precedence::kCall},
      {std::type_index(typeid(Var)), Precedence::kAtom},
      {std::type_index(typeid(IterArg)), Precedence::kAtom},
      {std::type_index(typeid(ConstInt)), Precedence::kAtom},
      {std::type_index(typeid(ConstFloat)), Precedence::kAtom},
      {std::type_index(typeid(ConstBool)), Precedence::kAtom},
      {std::type_index(typeid(TupleGetItemExpr)), Precedence::kAtom},
  };

  INTERNAL_CHECK(expr) << "Expression is null";
  const Expr& expr_ref = *expr;
  const auto it = kPrecedenceMap.find(std::type_index(typeid(expr_ref)));
  if (it != kPrecedenceMap.end()) {
    return it->second;
  }

  // Default for any other expression types.
  return Precedence::kAtom;
}

bool IsRightAssociative(const ExprPtr& expr) {
  // Only ** (power) is right-associative in Python
  return IsA<Pow>(expr);
}

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
  std::string PrintSubExpr(const ExprPtr& expr) const;

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
  void PrintRuntimeScopeHeader(bool manual) {
    stream_ << "with " << prefix_
            << (manual ? ".scope(mode=" + prefix_ + ".ScopeMode.MANUAL):\n" : ".scope():\n");
  }

  // Emit a comma-prefixed kwarg ``, <kwarg_name>=[v1, v2, ...]`` from the
  // list-of-VarPtr attr keyed by ``attr_key`` on ``op``. Skips null entries so
  // the rendered Python stays syntactically valid. Returns false when the attr
  // is absent or every entry is null. Shared by the two scope-attr printers
  // below so the null-filter rule lives in one place.
  bool PrintScopeVarListKwarg(const ScopeStmtPtr& op, const char* attr_key, const char* kwarg_name);

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
  // prints itself, so there is no bespoke syntax.
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
  void PrintYieldAssignmentVars(const std::vector<VarPtr>& return_vars);

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

// Helper function to format a float literal so it re-parses as a ``ConstFloat``.
//
// Emits enough significant digits to parse back to the exact same ``double``.
// This makes ``ConstFloat`` (and float kwargs / attrs) round-trip bit-exactly
// through print → parse: the parser stores ``ConstFloat.value_`` as the FP64
// value it reads, so the printed text must encode the full ``double`` regardless
// of the constant's ``dtype_``. Use iostreams rather than floating-point
// ``std::to_chars`` because GCC 10's libstdc++ does not provide that overload.
// Output may use exponent notation (``1e+16``, ``1e-07``) — still a valid Python
// float literal.
//
// The default float format emits a bare integer (``"4"``) for integer-valued
// doubles without an exponent, which would re-parse as ``ConstInt``; append
// ``.0`` in that case. The append is skipped for exponent forms (``"1e+16"``
// already parses as a Python float) and for non-finite values (``"nan"`` /
// ``"inf"`` have no DSL syntax — emitted verbatim, matching the prior behavior;
// no IR construction path produces them).
std::string FormatFloatLiteral(double value) {
  std::ostringstream os;
  os.imbue(std::locale::classic());
  os << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
  std::string text = os.str();
  if (std::isfinite(value) && text.find_first_of(".eE") == std::string::npos) {
    text += ".0";
  }
  return text;
}

// DataTypeToPythonString removed — now uses DataTypeToString from dtype.h

// IRPythonPrinter implementation
std::string IRPythonPrinter::Print(const IRNodePtr& node) {
  stream_.str("");
  stream_.clear();
  indent_level_ = 0;

  // Try each type in order
  if (auto program = As<Program>(node)) {
    VisitProgram(program);
  } else if (auto func = As<Function>(node)) {
    VisitFunction(func);
  } else if (auto stmt = As<Stmt>(node)) {
    // Bare-stmt roots (e.g. for_stmt.as_python(), seq.as_python()) need the
    // same disambiguation a function body gets — without this, two distinct
    // Var*/IterArg* sharing a name_hint collapse to one printed identifier.
    BuildVarRenameMap({}, stmt);
    VisitStmt(stmt);
  } else if (auto expr = As<Expr>(node)) {
    VisitExpr(expr);
  } else {
    // Unsupported node type
    stream_ << "<unsupported IRNode type>";
  }

  return stream_.str();
}

std::string IRPythonPrinter::Print(const TypePtr& type) {
  if (auto scalar_type = As<ScalarType>(type)) {
    // Print as pl.Scalar[pl.INT64] for proper round-trip support
    return prefix_ + ".Scalar[" + prefix_ + "." + DataTypeToString(scalar_type->dtype_) + "]";
  }

  // Tensor / DistributedTensor share the same rendering surface — only the
  // subscript head differs (``pl.Tensor`` vs ``pld.DistributedTensor``). Note
  // ``As<TensorType>`` is precise-match and would not fire for the subclass,
  // so dispatch on DistributedTensorType first and pass it through the
  // TensorType base for shared field access.
  auto dt_tensor = As<DistributedTensorType>(type);
  TensorTypePtr tensor_type;
  std::string tensor_head;
  if (dt_tensor) {
    tensor_type = dt_tensor;
    tensor_head = "pld.DistributedTensor";
  } else if (auto plain_tensor = As<TensorType>(type)) {
    tensor_type = plain_tensor;
    tensor_head = prefix_ + ".Tensor";
  }
  if (tensor_type) {
    std::ostringstream oss;
    oss << tensor_head << "[[";
    PrintShapeDims(oss, tensor_type->shape_);
    oss << "], " << prefix_ << "." << DataTypeToString(tensor_type->dtype_);

    // Add optional tensor_view parameter if present.
    // Always emit something when tensor_view is set so that print->parse roundtrip
    // preserves presence: structural equality distinguishes present vs absent.
    if (tensor_type->tensor_view_.has_value()) {
      auto tv_str = PrintTensorView(tensor_type->tensor_view_.value(), tensor_type->shape_);
      if (!tv_str.empty()) {
        oss << ", " << tv_str;
      } else {
        oss << ", " << prefix_ << ".TensorView()";
      }
    }

    // Add optional memref as positional arg
    if (tensor_type->memref_.has_value()) {
      oss << ", " << PrintMemRef(*tensor_type->memref_.value());
    }

    // Fully-resolved dump (issue #2088): a DistributedTensorType carries a
    // WindowBuffer back-reference that the concise form drops, so two same
    // shape/dtype distributed tensors viewing *different* window buffers print
    // identically. Surface the buffer name so a dump distinguishes them. This is
    // an informational marker (a quoted string): the subscript DSL has no
    // window_buffer slot and Python forbids keyword subscripts, so it is emitted
    // as a trailing string element. The parser strips it (type_resolver +
    // DistributedTensorMeta) and re-derives the real reference from
    // pld.tensor.window, so EXPLICIT dumps still reparse to identical IR — which
    // validate_ir relies on (it reloads every dump). Emitted only under
    // explicit_layout_.
    if (explicit_layout_ && dt_tensor && dt_tensor->window_buffer_.has_value()) {
      oss << ", \"window_buffer=" << dt_tensor->window_buffer_.value()->name_hint_ << "\"";
    }

    oss << "]";
    return oss.str();
  }

  if (auto array_type = As<ArrayType>(type)) {
    std::ostringstream oss;
    // Subscript-style: pl.Array[N, dtype]. Use PrintShapeDims (which routes through
    // PrintExprForType into a temp stream) so the recursive printing doesn't reset
    // the main stream_ buffer.
    oss << prefix_ << ".Array[";
    PrintShapeDims(oss, array_type->shape_);
    oss << ", " << prefix_ << "." << DataTypeToString(array_type->dtype_);
    oss << "]";
    return oss.str();
  }

  if (auto tile_type = As<TileType>(type)) {
    std::ostringstream oss;
    // Subscript-style: pl.Tile[[shape], dtype]
    oss << prefix_ << ".Tile[[";
    PrintShapeDims(oss, tile_type->shape_);
    oss << "], " << prefix_ << "." << DataTypeToString(tile_type->dtype_);

    // Add optional memref as positional arg
    if (tile_type->memref_.has_value()) {
      oss << ", " << PrintMemRef(*tile_type->memref_.value());
    }

    if (tile_type->memory_space_.has_value()) {
      auto mem_str = MemorySpaceToString(tile_type->memory_space_.value());
      oss << ", " << prefix_ << ".Mem." << mem_str;
    }

    // Pick the view to render, then print once. explicit_layout_ resolves an
    // absent canonical view (tile_view_ == nullopt) to its implicit form via
    // GetEffectiveTileView so the annotation states its real blayout/slayout/
    // fractal (issue #2088) — PrintTileView forces those fields in that mode, so
    // it never returns "" for the explicit view. The concise branch only renders
    // a present view; when a present view happens to match the implicit one
    // (possible only on incoherent IR — canonical IR stores that as nullopt), the
    // empty-string fallback keeps the output parseable so TileTypeCoherence can
    // flag the real bug.
    std::optional<TileView> view;
    if (explicit_layout_) {
      view = tile_view_semantics::GetEffectiveTileView(*tile_type);
    } else if (tile_type->tile_view_.has_value()) {
      view = tile_type->tile_view_.value();
    }
    if (view.has_value()) {
      auto view_str = PrintTileView(*view, tile_type->shape_, tile_type->memory_space_);
      oss << ", " << (view_str.empty() ? prefix_ + ".TileView()" : view_str);
    }

    oss << "]";
    return oss.str();
  }

  if (auto tuple_type = As<TupleType>(type)) {
    std::ostringstream oss;
    if (tuple_type->types_.empty()) {
      oss << prefix_ << ".Tuple[()]";
    } else {
      oss << prefix_ << ".Tuple[";
      for (size_t i = 0; i < tuple_type->types_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << Print(tuple_type->types_[i]);
      }
      oss << "]";
    }
    return oss.str();
  }

  if (auto memref_type = As<MemRefType>(type)) {
    return prefix_ + ".MemRefType";
  }

  if (auto ptr_type = As<PtrType>(type)) {
    return prefix_ + ".Ptr";
  }

  if (As<WindowBufferType>(type)) {
    // Singleton marker — no per-instance fields. Render as a bare attribute
    // so it round-trips through the parser via the same path as ``pld.``
    // namespace lookups.
    return "pld.WindowBufferType";
  }

  if (As<CommCtxType>(type)) {
    // Singleton marker — printed with the public DSL wrapper name, matching the
    // ``pl.AsyncEvent`` family below. ``pld`` is hardcoded rather than derived
    // from ``prefix_`` because that prefix aliases only ``pypto.language``; the
    // text parser injects ``pld`` into the exec namespace itself.
    return "pld.CommCtx";
  }

  // Async-prefetch handle markers — fieldless singletons, rendered as bare
  // public wrapper attributes on the main DSL namespace so the output can be
  // evaluated by normal Python as well as round-tripped through the parser.
  if (As<PrefetchAsyncContextType>(type)) {
    return prefix_ + ".PrefetchAsyncContext";
  }

  if (As<AsyncEventType>(type)) {
    return prefix_ + ".AsyncEvent";
  }

  if (As<AsyncSessionType>(type)) {
    return prefix_ + ".AsyncSession";
  }

  return prefix_ + ".UnknownType";
}

std::string IRPythonPrinter::GetIndent() const {
  return std::string(static_cast<size_t>(indent_level_ * 4), ' ');
}

void IRPythonPrinter::IncreaseIndent() { indent_level_++; }

void IRPythonPrinter::DecreaseIndent() {
  if (indent_level_ > 0) {
    indent_level_--;
  }
}

// Expression visitors - reuse precedence logic from base printer
void IRPythonPrinter::VisitExpr_(const VarPtr& op) { stream_ << GetVarName(op.get()); }

void IRPythonPrinter::VisitExpr_(const IterArgPtr& op) { stream_ << GetVarName(op.get()); }

void IRPythonPrinter::VisitExpr_(const MemRefPtr& op) { stream_ << op->name_hint_; }

void IRPythonPrinter::VisitExpr_(const WindowBufferPtr& op) { stream_ << op->name_hint_; }

void IRPythonPrinter::VisitExpr_(const ConstIntPtr& op) {
  // A bare integer literal in the DSL canonically denotes INDEX -- that is what
  // the parser (`ast_parser.parse_constant`) produces. Only INDEX may print
  // bare; every other integer dtype (INT64 included) must carry an explicit
  // `pl.const(value, pl.<DTYPE>)` annotation so print -> reparse round-trips.
  if (op->dtype() == DataType::INDEX && !typed_index_consts_) {
    stream_ << op->value_;
  } else {
    stream_ << prefix_ << ".const(" << op->value_ << ", " << prefix_ << "." << DataTypeToString(op->dtype())
            << ")";
  }
}

void IRPythonPrinter::VisitExpr_(const ConstFloatPtr& op) {
  if (op->dtype() != DataType::DEFAULT_CONST_FLOAT) {
    stream_ << prefix_ << ".const(" << FormatFloatLiteral(op->value_) << ", " << prefix_ << "."
            << DataTypeToString(op->dtype()) << ")";
  } else {
    stream_ << FormatFloatLiteral(op->value_);
  }
}

void IRPythonPrinter::VisitExpr_(const ConstBoolPtr& op) { stream_ << (op->value_ ? "True" : "False"); }

// Lowercase-snake-case names for ArgDirection used by the DSL helpers in
// ``pypto.language.arg_direction`` (``pl.adir.<name>``). Kept in sync with
// ``python/pypto/language/arg_direction.py::NAME_TO_DIRECTION``.
static const char* ArgDirectionToDslName(ArgDirection dir) {
  switch (dir) {
    case ArgDirection::Input:
      return "input";
    case ArgDirection::Output:
      return "output";
    case ArgDirection::OutputExisting:
      return "output_existing";
    case ArgDirection::InOut:
      return "inout";
    case ArgDirection::NoDep:
      return "no_dep";
    case ArgDirection::Scalar:
      return "scalar";
  }
  throw pypto::TypeError("Unknown ArgDirection in printer");
}

void IRPythonPrinter::PrintAttrValue(const std::any& value, const Span& span) {
  const std::type_info& t = value.type();
  if (t == typeid(int)) {
    stream_ << std::any_cast<int>(value);
  } else if (t == typeid(bool)) {
    stream_ << (std::any_cast<bool>(value) ? "True" : "False");
  } else if (t == typeid(std::string)) {
    stream_ << std::quoted(std::any_cast<std::string>(value));
  } else if (t == typeid(double)) {
    stream_ << FormatFloatLiteral(std::any_cast<double>(value));
  } else if (t == typeid(float)) {
    stream_ << FormatFloatLiteral(static_cast<double>(std::any_cast<float>(value)));
  } else if (t == typeid(DataType)) {
    // ``LowerHostTensorCollectives`` stamps a DataType attr on every
    // ``builtin.tensor.<collective>`` call. Printed in the ``pl.<DTYPE>`` DSL
    // form the dtype resolver reads back (ast_parser._parse_attr_value).
    stream_ << prefix_ << "." << DataTypeToString(std::any_cast<DataType>(value));
  } else if (t == typeid(MemorySpace)) {
    stream_ << prefix_ << ".Mem." << MemorySpaceToString(std::any_cast<MemorySpace>(value));
  } else if (t == typeid(TensorLayout)) {
    stream_ << prefix_ << ".TensorLayout." << TensorLayoutToString(std::any_cast<TensorLayout>(value));
  } else if (t == typeid(TileLayout)) {
    stream_ << prefix_ << ".TileLayout." << TileLayoutToString(std::any_cast<TileLayout>(value));
  } else if (t == typeid(PadValue)) {
    stream_ << prefix_ << ".PadValue.";
    switch (std::any_cast<PadValue>(value)) {
      case PadValue::null:
        stream_ << "null";
        break;
      case PadValue::zero:
        stream_ << "zero";
        break;
      case PadValue::max:
        stream_ << "max";
        break;
      case PadValue::min:
        stream_ << "min";
        break;
    }
  } else if (t == typeid(ArgDirection)) {
    stream_ << prefix_ << ".adir." << ArgDirectionToDslName(std::any_cast<ArgDirection>(value));
  } else if (t == typeid(std::vector<ArgDirection>)) {
    const auto& dirs = std::any_cast<std::vector<ArgDirection>>(value);
    stream_ << "[";
    for (size_t i = 0; i < dirs.size(); ++i) {
      if (i > 0) stream_ << ", ";
      stream_ << prefix_ << ".adir." << ArgDirectionToDslName(dirs[i]);
    }
    stream_ << "]";
  } else if (t == typeid(std::vector<int32_t>)) {
    const auto& idxs = std::any_cast<std::vector<int32_t>>(value);
    stream_ << "[";
    for (size_t i = 0; i < idxs.size(); ++i) {
      if (i > 0) stream_ << ", ";
      stream_ << idxs[i];
    }
    stream_ << "]";
  } else if (t == typeid(std::vector<VarPtr>)) {
    const auto& vars = std::any_cast<std::vector<VarPtr>>(value);
    stream_ << "[";
    for (size_t i = 0; i < vars.size(); ++i) {
      if (i > 0) stream_ << ", ";
      INTERNAL_CHECK_SPAN(vars[i], span)
          << "Internal error: null Var in attr list; the DSL has no null-Var syntax to round-trip";
      stream_ << GetVarName(vars[i].get());
    }
    stream_ << "]";
  } else if (t == typeid(std::vector<std::pair<int32_t, int>>)) {
    // ``kAttrCachePolicyParams`` on an outlined Function: (param index, policy)
    // pairs. Printed as a list of tuples so a pass dump taken between
    // OutlineIncoreScopes and ConvertTensorToTileOps — the only window where
    // this attr exists — round-trips instead of aborting.
    const auto& pairs = std::any_cast<std::vector<std::pair<int32_t, int>>>(value);
    stream_ << "[";
    for (size_t i = 0; i < pairs.size(); ++i) {
      if (i > 0) stream_ << ", ";
      stream_ << "(" << pairs[i].first << ", " << pairs[i].second << ")";
    }
    stream_ << "]";
  } else if (t == typeid(std::vector<std::pair<VarPtr, int>>)) {
    // ``kAttrCachePolicyVars``. On a ScopeStmt this attr is surfaced as
    // ``pl.set_cache_policy(...)`` marker statements instead (see
    // PrintScopeCachePolicyStmts); this arm is the generic-attr fallback, so a
    // dump never aborts on a carrier the scope printer did not claim.
    const auto& pairs = std::any_cast<std::vector<std::pair<VarPtr, int>>>(value);
    stream_ << "[";
    for (size_t i = 0; i < pairs.size(); ++i) {
      if (i > 0) stream_ << ", ";
      INTERNAL_CHECK_SPAN(pairs[i].first, span)
          << "Internal error: null Var in attr list; the DSL has no null-Var syntax to round-trip";
      stream_ << "(" << GetVarName(pairs[i].first.get()) << ", " << pairs[i].second << ")";
    }
    stream_ << "]";
  } else if (t == typeid(VarPtr)) {
    const auto& var = std::any_cast<VarPtr>(value);
    INTERNAL_CHECK_SPAN(var, span) << "Internal error: null Var attr; no DSL syntax to round-trip";
    stream_ << GetVarName(var.get());
  } else if (t == typeid(ExprPtr)) {
    const auto& expr = std::any_cast<ExprPtr>(value);
    INTERNAL_CHECK_SPAN(expr, span) << "Internal error: null Expr attr; no DSL syntax to round-trip";
    VisitExpr(expr);
  } else {
    // No silent drop: if a new attr value type reaches the printer without a
    // codec arm here (and a matching arm in ast_parser._parse_attr_value), it
    // is a compiler bug — surface it loudly so the round-trip gap is fixed at
    // the source rather than masked by a dropped attr.
    INTERNAL_CHECK_SPAN(false, span)
        << "Internal error: no DSL attr-value codec for type '" << DemangleTypeName(t.name())
        << "'. The python printer round-trips scalar, supported enum, vector, VarPtr, and ExprPtr "
           "attrs; add a PrintAttrValue arm, a "
           "matching _parse_attr_value case, and ConvertKwargsDict support for this type instead "
           "of dropping it.";
  }
}

void IRPythonPrinter::VisitExpr_(const CallPtr& op) {
  INTERNAL_CHECK_SPAN(op->op_, op->span_) << "Call has null op";
  // Check if this is a GlobalVar call within a Program context

  if (auto gvar = As<GlobalVar>(op->op_)) {
    if (current_program_) {
      // This is a cross-function call - print as self.method_name()
      stream_ << "self." << gvar->name_ << "(";

      // Selective dump (``kAttrDumpVars``) is surfaced below inside the
      // machine-only ``attrs={...}`` dict, NOT by wrapping args — there is no
      // call-arg wrapper. Collect the marked Vars now so the round-trip
      // recovers the same attr by VarPtr identity.
      std::set<const Var*> dump_set;
      for (const auto& [k, v] : op->attrs_) {
        if (k != kAttrDumpVars) continue;
        if (const auto* vars = std::any_cast<std::vector<VarPtr>>(&v)) {
          for (const auto& var : *vars) {
            if (var) dump_set.insert(var.get());
          }
        }
      }

      for (size_t i = 0; i < op->args_.size(); ++i) {
        if (i > 0) stream_ << ", ";
        VisitExpr(op->args_[i]);
      }

      // Manual dependency edges live on ``Submit::deps_``, never on a plain
      // cross-function Call (ManualDepsOnSubmitOnly invariant). There is no
      // ``deps=`` surface for plain calls — the parser rejects it — so a Call
      // carrying the attr is a compiler bug; fail loudly instead of printing
      // unparseable text.
      INTERNAL_CHECK_SPAN(!op->HasAttr(kAttrManualDepEdges), op->span_)
          << "Internal error: plain cross-function Call to '" << gvar->name_
          << "' carries attrs[\"manual_dep_edges\"]; manual deps must be on Submit::deps_";
      // Zero-arg self.fn() needs no leading comma before the first kwarg, so
      // gate every kwarg separator on a shared flag rather than hard-coding
      // ", " — otherwise self.fn(, device=...) breaks the round-trip.
      bool need_kwarg_comma = !op->args_.empty();

      // Surface ``attrs["device"]`` (set by N3 parser on host_orch → chip_orch
      // dispatches) as a ``device=<expr>`` kwarg so it round-trips through
      // reparse and remains observable by the MaterializeCommDomainScopes pass.
      for (const auto& [k, v] : op->attrs_) {
        if (k != kAttrDevice) continue;
        if (const auto* p = std::any_cast<ExprPtr>(&v)) {
          if (*p) {
            stream_ << (need_kwarg_comma ? ", " : "") << "device=";
            VisitExpr(*p);
            need_kwarg_comma = true;
          }
        }
        break;
      }

      // Machine-only ``attrs={...}`` dict for IR metadata that has no user-facing
      // call kwarg. Two keys keep a bespoke emitter for clarity / arg-order
      // stability: ``dump_vars`` (selective dump, seeded by ``pl.dump_tag`` /
      // ``dumps=``) and ``arg_directions`` (post DeriveCallDirections). EVERY
      // other attr — ``arg_direction_overrides``, ``dummy_task``, any future
      // key — is rendered generically by ``PrintAttrValue`` into the same dict
      // (the round-trip safety net), so nothing is silently dropped. The sugar
      // key already surfaced above (``device`` -> ``device=``) is skipped
      // here. The parser recovers the bespoke keys via dedicated extractors
      // and the rest via ``_parse_attr_value``.
      auto call_arg_directions = op->GetArgDirections();
      const bool has_dirs = !call_arg_directions.empty();
      const bool has_dump = !dump_set.empty();
      std::vector<const std::pair<std::string, std::any>*> generic_attrs;
      for (const auto& kv : op->attrs_) {
        const std::string& k = kv.first;
        if (k == kAttrDevice) continue;                               // emitted as device=
        if (k == kAttrDumpVars || k == kAttrArgDirections) continue;  // bespoke emitters below
        generic_attrs.push_back(&kv);
      }
      if (has_dirs || has_dump || !generic_attrs.empty()) {
        stream_ << (need_kwarg_comma ? ", " : "") << "attrs={";
        bool first_key = true;
        if (has_dump) {
          stream_ << "\"dump_vars\": [";
          bool first = true;
          for (const auto& arg : op->args_) {
            auto var = AsVarLike(arg);
            if (!var || dump_set.count(var.get()) == 0) continue;
            if (!first) stream_ << ", ";
            first = false;
            stream_ << GetVarName(var.get());
          }
          stream_ << "]";
          first_key = false;
        }
        if (has_dirs) {
          INTERNAL_CHECK_SPAN(call_arg_directions.size() == op->args_.size(), op->span_)
              << "Call arg_directions size (" << call_arg_directions.size() << ") must match args size ("
              << op->args_.size() << ")";
          stream_ << (first_key ? "" : ", ") << "\"arg_directions\": [";
          for (size_t i = 0; i < call_arg_directions.size(); ++i) {
            if (i > 0) stream_ << ", ";
            stream_ << prefix_ << ".adir." << ArgDirectionToDslName(call_arg_directions[i]);
          }
          stream_ << "]";
          first_key = false;
        }
        for (const auto* kv : generic_attrs) {
          stream_ << (first_key ? "" : ", ");
          first_key = false;
          stream_ << std::quoted(kv->first) << ": ";
          PrintAttrValue(kv->second, op->span_);
        }
        stream_ << "}";
      }

      stream_ << ")";
      return;
    }
  }

  // Format operation name for printing
  // Operations are stored with internal names like "tensor.adds" or "tile.matmul"
  // and are printed in parseable format like "pl.tensor.adds"
  std::string op_name = op->op_->name_;
  const bool has_physical_remote_tail =
      IsOp(op, "pld.tile.remote_load") && op->GetKwarg<bool>("allow_physical_tail_padding", false);
  if (has_physical_remote_tail) {
    // The aligned FP16 tail is a compiler-only lowering detail. Print a private
    // parser sentinel that reconstructs the marker instead of exposing the
    // internal kwarg on the public pld.tile.remote_load API.
    op_name = "pld.tile._remote_load_with_physical_tail_padding";
  }

  // Normalize tensor.add with scalar rhs to tensor.adds (matches Python API dispatch)
  if (IsOp(op, "tensor.add") && op->args_.size() == 2) {
    if (std::dynamic_pointer_cast<const ConstFloat>(op->args_[1]) ||
        std::dynamic_pointer_cast<const ConstInt>(op->args_[1])) {
      op_name = "tensor.adds";
    }
  }

  // Check if this is a registered operation (contains a dot)
  if (op_name.find('.') != std::string::npos) {
    // ``pld.*`` ops live in the ``pypto.language.distributed`` namespace (aliased
    // to ``pld`` by the parser). Print them bare so the roundtrip parser
    // resolves them correctly via the same ``pld`` import path.
    if (op_name.rfind("pld.", 0) == 0) {
      stream_ << op_name;
    } else {
      // Print with pl. prefix for the standard pl namespace.
      stream_ << prefix_ << "." << op_name;
    }
    // ``tile.and`` / ``tensor.not`` and friends must print as the DSL's
    // underscore-suffixed spelling to stay valid Python.
    if (NeedsDslUnderscore(op_name)) stream_ << "_";
    stream_ << "(";
  } else {
    // Not a registered operation, print as-is
    stream_ << op_name << "(";
  }

  // Serialize op-call attrs into a trailing machine-only ``attrs={...}`` dict
  // via an open-world DENYLIST: EVERY key round-trips unless it already owns a
  // bespoke surface. This mirrors the GlobalVar-call branch above and the
  // Submit branch below, so the three call-like paths share one policy — a new
  // attr on a builtin-op call can never be silently dropped on print -> parse.
  // A value type without a codec fails loudly inside ``PrintAttrValue``
  // (surfacing the gap) rather than vanishing. The matching reader is
  // ``ast_parser._parse_op_attrs`` -> ``_parse_attr_value``. Keep this helper
  // available to the special call forms below so an early return cannot skip
  // the dict.
  const bool is_task_dummy = IsOp(op, "system.task_dummy");
  auto print_serialized_attrs = [&](bool need_comma) {
    std::vector<const std::pair<std::string, std::any>*> serialized_attrs;
    for (const auto& kv : op->attrs_) {
      if (kv.first == kAttrManualDepEdges || kv.first == kAttrDummyTask) {
        // Bespoke surfaces, but ONLY on ``system.task_dummy``: there
        // ``manual_dep_edges`` prints as ``deps=[...]`` and ``dummy_task`` is
        // re-derived by the parser from the op itself. On any other op neither
        // surface exists, so skipping the key would be precisely the silent
        // drop this denylist removes — fail loud instead. Manual dependency
        // edges belong on ``Submit::deps_`` (ManualDepsOnSubmitOnly).
        INTERNAL_CHECK_SPAN(is_task_dummy, op->span_)
            << "Internal error: call to '" << op->op_->name_ << "' carries attrs[\"" << kv.first
            << "\"], which has a DSL surface only on system.task_dummy; manual dependency edges "
               "belong on Submit::deps_";
        continue;
      }
      serialized_attrs.push_back(&kv);
    }
    if (serialized_attrs.empty()) return;

    stream_ << (need_comma ? ", " : "") << "attrs={";
    bool first_key = true;
    for (const auto* kv : serialized_attrs) {
      stream_ << (first_key ? "" : ", ");
      first_key = false;
      stream_ << std::quoted(kv->first) << ": ";
      PrintAttrValue(kv->second, op->span_);
    }
    stream_ << "}";
  };

  // reinterpret_view keeps dtype as the second public-API input and shape as a
  // keyword-only optional input, while the IR stores shape positionally and
  // dtype in kwargs. Print the public ordering explicitly so both tensor and
  // tile forms round-trip through their DSL wrappers.
  if ((IsOp(op, "tile.reinterpret_view") || IsOp(op, "tensor.reinterpret_view")) &&
      (op->args_.size() == 1 || op->args_.size() == 2)) {
    VisitExpr(op->args_[0]);
    bool found_dtype = false;
    for (const auto& [key, value] : op->kwargs_) {
      if (key != "dtype") continue;
      stream_ << ", dtype=" << prefix_ << "."
              << DataTypeToString(AnyCast<DataType>(value, op->op_->name_ + " dtype"));
      found_dtype = true;
      break;
    }
    INTERNAL_CHECK_SPAN(found_dtype, op->span_)
        << "Internal error: " << op->op_->name_ << " is missing its required dtype kwarg";
    if (op->args_.size() == 2) {
      stream_ << ", shape=";
      VisitExpr(op->args_[1]);
    }
    print_serialized_attrs(/*need_comma=*/true);
    stream_ << ")";
    return;
  }

  // Special handling for tile.full / tensor.full: print as keyword args to match Python API
  // IR stores: args_=[shape, value_expr], kwargs_={"dtype": dtype}
  // Python API: full(shape, dtype, value) — print as full(shape, dtype=.., value=..)
  // because pl.FP32 as positional is rejected by the parser (standalone attribute access)
  if ((IsOp(op, "tile.full") || IsOp(op, "tensor.full")) && op->args_.size() >= 2) {
    VisitExpr(op->args_[0]);  // shape (positional)
    for (const auto& [key, val] : op->kwargs_) {
      if (key == "dtype") {
        stream_ << ", dtype=" << prefix_ << "."
                << DataTypeToString(AnyCast<DataType>(val, op->op_->name_ + " dtype"));
        break;
      }
    }
    stream_ << ", value=";
    // Print value as a bare numeric literal (dtype is already captured in dtype=...).
    // Using VisitExpr would emit pl.const(v, pl.BF16) which the Python API cannot accept
    // as the `value: int | float` parameter.
    const auto& val_expr = op->args_[1];
    if (auto cf = As<ConstFloat>(val_expr)) {
      stream_ << FormatFloatLiteral(cf->value_);
    } else if (auto ci = As<ConstInt>(val_expr)) {
      stream_ << ci->value_;
    } else {
      VisitExpr(val_expr);
    }
    print_serialized_attrs(/*need_comma=*/true);
    stream_ << ")";
    return;
  }

  // The current PTO-ISA gives every system.syncall soft form the same operands:
  // [gm_workspace] or [gm_workspace, used_cores]. Print them through the
  // high-level keyword-only DSL surface so the program round-trips.
  if (IsOp(op, "system.syncall") && (op->args_.size() == 1 || op->args_.size() == 2)) {
    std::string core_type = "mix";
    std::string mode = "hard";
    for (const auto& [key, val] : op->kwargs_) {
      if (key == "core_type") {
        core_type = AnyCast<std::string>(val, "syncall core_type");
      } else if (key == "mode") {
        mode = AnyCast<std::string>(val, "syncall mode");
      }
    }
    if (mode == "soft") {
      const char* kernel = CoreTypeToKernelMember(core_type, /*is_syncall=*/true);
      INTERNAL_CHECK_SPAN(kernel != nullptr, op->span_)
          << "Internal error: system.syncall core_type must be aic_only|aiv_only|mix, got " << core_type;
      stream_ << "mode=" << prefix_ << ".SyncAllMode.SOFT, core_type=" << prefix_ << ".KernelType." << kernel
              << ", gm_workspace=";
      VisitExpr(op->args_[0]);
      if (op->args_.size() == 2) {
        stream_ << ", used_cores=";
        if (auto ci = As<ConstInt>(op->args_[1])) {
          stream_ << ci->value_;
        } else {
          VisitExpr(op->args_[1]);
        }
      } else {
        // The high-level API requires an explicit participant-count choice so
        // users do not accidentally select PTO-ISA's runtime-sensitive auto
        // path. An explicit zero reconstructs this one-operand IR form.
        stream_ << ", used_cores=0";
      }
      print_serialized_attrs(/*need_comma=*/true);
      stream_ << ")";
      return;
    }
  }

  // gather_row's optional 6th operand is keyword-only in the DSL: `transpose`
  // already owned the 6th positional slot before valid_shape existed, and taking
  // it would silently reinterpret an existing `gather_row(..., shapes, True)` as
  // a shape. Print it as a kwarg so the round-trip matches the Python signature.
  const bool gather_row_kw_valid =
      (IsOp(op, "tile.gather_row") || IsOp(op, "tensor.gather_row")) && op->args_.size() == 6;
  // The optional `init_cond` operand is keyword-only in two DSL signatures,
  // because another parameter already owns positional slot 4:
  //   ``tensor.matmul_acc(acc, lhs, rhs, a_trans, b_trans, init_cond)`` — the
  //     predicate printed positionally re-parses as a transpose flag, and then
  //     collides with the printed ``a_trans=`` kwarg.
  //   ``tile.gemv_acc(acc, lhs, rhs, acc_phase, *, init_cond)`` — it would bind
  //     to ``acc_phase``, which is likewise also printed as a kwarg.
  // ``tile.matmul_acc`` takes the predicate in positional slot 4 and needs no
  // such fixup.
  const bool acc_kw_init_cond =
      (IsOp(op, "tensor.matmul_acc") || IsOp(op, "tile.gemv_acc")) && op->args_.size() == 4;
  // tile.ci keeps dtype/descending in their established positional slots; its
  // compiler-generated third IR operand therefore prints as keyword-only tmp.
  const bool ci_kw_tmp = IsOp(op, "tile.ci") && op->args_.size() == 3;
  // tile.cast keeps (tile, target_type, mode) as the public signature; its
  // compiler-generated scratch operand likewise prints as keyword-only tmp.
  const bool cast_kw_tmp = IsOp(op, "tile.cast") && op->args_.size() == 2;
  const bool sort32_kw_tmp = IsOp(op, "tile.sort32") && op->args_.size() == 3;
  const bool mgather = IsOp(op, "tile.mgather");
  const int mgather_coalesce = mgather ? op->GetKwarg<int>("coalesce", 0) : 0;
  const bool mgather_kw_scratch = mgather && mgather_coalesce == 1 && op->args_.size() >= 3;
  const bool mgather_kw_valid = mgather && ((mgather_coalesce == 0 && op->args_.size() == 3) ||
                                            (mgather_coalesce == 1 && op->args_.size() == 4));

  // Print positional arguments
  for (size_t i = 0; i < op->args_.size(); ++i) {
    if (gather_row_kw_valid && i == 5) continue;
    if (acc_kw_init_cond && i == 3) continue;
    if (ci_kw_tmp && i == 2) continue;
    if (cast_kw_tmp && i == 1) continue;
    if (sort32_kw_tmp && i == 2) continue;
    if (mgather && i >= 2) continue;
    if (i > 0) stream_ << ", ";

    // Special handling for tile.alloc/tensor.alloc first argument (memory_space)
    if ((IsOp(op, "tile.alloc") || IsOp(op, "tensor.alloc")) && i == 0) {
      // Try to extract the integer value and convert it to MemorySpace enum
      if (auto const_int = std::dynamic_pointer_cast<const ConstInt>(op->args_[i])) {
        int space_value = static_cast<int>(const_int->value_);
        stream_ << prefix_ << ".Mem." << MemorySpaceToString(static_cast<MemorySpace>(space_value));
      } else {
        VisitExpr(op->args_[i]);
      }
    } else {
      VisitExpr(op->args_[i]);
    }
  }

  // Print kwargs as keyword arguments
  bool need_comma = !op->args_.empty();
  if (gather_row_kw_valid) {
    stream_ << ", valid_shape=";
    VisitExpr(op->args_[5]);
    need_comma = true;
  }
  if (acc_kw_init_cond) {
    stream_ << ", init_cond=";
    VisitExpr(op->args_[3]);
    need_comma = true;
  }
  if (ci_kw_tmp) {
    stream_ << ", tmp=";
    VisitExpr(op->args_[2]);
    need_comma = true;
  }
  if (cast_kw_tmp) {
    stream_ << ", tmp=";
    VisitExpr(op->args_[1]);
    need_comma = true;
  }
  if (sort32_kw_tmp) {
    stream_ << ", tmp=";
    VisitExpr(op->args_[2]);
    need_comma = true;
  }
  if (mgather_kw_scratch) {
    stream_ << ", scratch=";
    VisitExpr(op->args_[2]);
    need_comma = true;
  }
  if (mgather_kw_valid) {
    stream_ << ", valid_shape=";
    VisitExpr(op->args_[mgather_coalesce == 1 ? 3 : 2]);
    need_comma = true;
  }
  if (IsOp(op, "system.task_dummy")) {
    const std::vector<VarPtr>* deps_to_print = nullptr;
    for (const auto& [k, v] : op->attrs_) {
      if (k != kAttrManualDepEdges) continue;
      deps_to_print = std::any_cast<std::vector<VarPtr>>(&v);
      break;
    }
    stream_ << (need_comma ? ", " : "") << "deps=[";
    if (deps_to_print) {
      bool printed_dep = false;
      for (const auto& dep : *deps_to_print) {
        // Invalid/null dependency slots do not contribute user-visible edges.
        if (!dep) continue;
        if (printed_dep) stream_ << ", ";
        stream_ << GetVarName(dep.get());
        printed_dep = true;
      }
    }
    stream_ << "]";
    need_comma = true;
  }
  for (const auto& [key, value] : op->kwargs_) {
    if (has_physical_remote_tail && key == "allow_physical_tail_padding") continue;
    // ``pld.tensor.alloc_window_buffer`` injects its ``name`` kwarg from the LHS
    // at parse time and explicitly rejects a user-written ``name=`` kwarg. Skip
    // it on print so the round-trip parser can re-derive the name from the
    // assignment LHS without tripping the no-user-kwargs check.
    if (IsOp(op, "pld.tensor.alloc_window_buffer") && key == "name") continue;
    // Inside a live SplitAivScopeStmt region the per-op ``split=`` int on
    // aiv_shard/aic_gather is redundant: the region node's mode is the
    // authoritative carrier and the parser re-stamps it from the enclosing
    // ``pl.split_aiv(..., mode=...)``. Suppress it so the printed body reparses
    // (the parser rejects an explicit ``split=`` on these ops). This covers both
    // the lowered tile form (tile.aiv_shard / tile.aic_gather) and the pre-pass
    // high-level tensor form (tensor.aiv_shard / tensor.aic_gather emitted by the
    // @pl.jit / pl.spmd surface before ConvertTensorToTileOps). Outside a region
    // (lowered form) it prints as usual.
    if (split_aiv_scope_depth_ > 0 && key == "split" &&
        (IsOp(op, "tile.aiv_shard") || IsOp(op, "tile.aic_gather") || IsOp(op, "tensor.aiv_shard") ||
         IsOp(op, "tensor.aic_gather"))) {
      continue;
    }
    if (need_comma) {
      stream_ << ", ";
    }
    need_comma = true;
    stream_ << key << "=";

    // Print value based on type
    if (value.type() == typeid(int)) {
      int int_val = AnyCast<int>(value, "printing kwarg: " + key);
      // Print pipe kwargs as PipeType enum names for readability
      if (key == "set_pipe" || key == "wait_pipe" ||
          (key == "pipe" && (IsOp(op, "system.sync_set") || IsOp(op, "system.sync_wait")))) {
        stream_ << prefix_ << ".PipeType." << PipeTypeToString(static_cast<PipeType>(int_val));
      } else if (key == "mode") {
        stream_ << "'" << CastModeToString(int_val) << "'";
      } else if (key == "atomic") {
        // Stored as int (the DSL casts AtomicType -> int before stashing on
        // kwargs_; nb::isinstance<AtomicType> in bindings does the same). The
        // public DSL signature is `atomic: AtomicType`, so restore the enum
        // form on print to keep the output type-correct for static checkers
        // and round-trippable through the parser (pl.AtomicType is exposed).
        stream_ << prefix_ << ".AtomicType." << AtomicTypeToString(static_cast<AtomicType>(int_val));
      } else {
        stream_ << int_val;
      }
    } else if (value.type() == typeid(bool)) {
      stream_ << (AnyCast<bool>(value, "printing kwarg: " + key) ? "True" : "False");
    } else if (value.type() == typeid(std::string)) {
      const auto str_val = AnyCast<std::string>(value, "printing kwarg: " + key);
      // The cross-core sync ops take enum-typed DSL kwargs but store the PTO-ISA
      // spelling as a string attr. Restore the enum form so the printed call is
      // type-correct for static checkers and re-parses (the DSL rejects strings).
      const bool is_syncall = IsOp(op, "system.syncall");
      const bool is_sync_event = IsOp(op, "system.sync_set") || IsOp(op, "system.sync_wait");
      if (key == "core_type" && (is_syncall || is_sync_event)) {
        const char* kernel = CoreTypeToKernelMember(str_val, is_syncall);
        INTERNAL_CHECK_SPAN(kernel != nullptr, op->span_)
            << "Internal error: " << op->op_->name_ << " carries an unknown core_type " << str_val;
        stream_ << prefix_ << ".KernelType." << kernel;
      } else if (is_syncall && key == "mode") {
        stream_ << prefix_ << ".SyncAllMode." << AttrValueToEnumMember(str_val);
      } else {
        stream_ << "'" << str_val << "'";
      }
    } else if (value.type() == typeid(double)) {
      stream_ << FormatFloatLiteral(AnyCast<double>(value, "printing kwarg: " + key));
    } else if (value.type() == typeid(float)) {
      stream_ << FormatFloatLiteral(static_cast<double>(AnyCast<float>(value, "printing kwarg: " + key)));
    } else if (value.type() == typeid(DataType)) {
      stream_ << prefix_ << "." << DataTypeToString(AnyCast<DataType>(value, "printing kwarg: " + key));
    } else if (value.type() == typeid(MemorySpace)) {
      stream_ << prefix_ << ".Mem."
              << MemorySpaceToString(AnyCast<MemorySpace>(value, "printing kwarg: " + key));
    } else if (value.type() == typeid(TensorLayout)) {
      stream_ << prefix_ << ".TensorLayout."
              << TensorLayoutToString(AnyCast<TensorLayout>(value, "printing kwarg: " + key));
    } else if (value.type() == typeid(TileLayout)) {
      stream_ << prefix_ << ".TileLayout."
              << TileLayoutToString(AnyCast<TileLayout>(value, "printing kwarg: " + key));
    } else if (value.type() == typeid(PadValue)) {
      auto pad = AnyCast<PadValue>(value, "printing kwarg: " + key);
      stream_ << prefix_ << ".PadValue.";
      switch (pad) {
        case PadValue::null:
          stream_ << "null";
          break;
        case PadValue::zero:
          stream_ << "zero";
          break;
        case PadValue::max:
          stream_ << "max";
          break;
        case PadValue::min:
          stream_ << "min";
          break;
      }
    } else {
      throw TypeError("Invalid kwarg type for key: " + key +
                      ", expected int, bool, std::string, double, float, DataType, MemorySpace, "
                      "TensorLayout, or PadValue, but got " +
                      DemangleTypeName(value.type().name()));
    }
  }

  print_serialized_attrs(need_comma);

  stream_ << ")";
}

void IRPythonPrinter::VisitExpr_(const SubmitPtr& op) {
  INTERNAL_CHECK_SPAN(op->op_, op->span_) << "Submit has null op";
  // Submit normally appears inside a Program (manual_scope body of a
  // function), so the callee is emitted as ``self.<kernel_name>``. When a
  // Submit is printed standalone (debugging, unit tests that build IR by
  // hand without a Program), fall back to naming the op directly so the
  // printer never crashes — matches the contract in the comment above.
  // ``pl.spmd_submit(...)`` when this Submit carries an SPMD launch spec
  // (``core_num_`` present); plain ``pl.submit(...)`` otherwise. The surface
  // form is kind-driven — it follows the presence of the launch spec — so
  // print → parse round-trips (.claude/rules ir-print-form-follows-kind).
  stream_ << prefix_ << (op->core_num_.has_value() ? ".spmd_submit(" : ".submit(");
  if (auto gvar = As<GlobalVar>(op->op_)) {
    // ``self.<name>`` inside a Program (the canonical case); bare ``<name>``
    // when the printer is invoked standalone (debugging / unit tests).
    if (current_program_) {
      stream_ << "self.";
    }
    stream_ << gvar->name_;
  } else {
    // Non-GlobalVar callee (Op or other) — fall back to the raw op name so
    // the printer never crashes on a hand-built Submit.
    stream_ << op->op_->name_;
  }
  // Selective dump (``kAttrDumpVars``) is surfaced below as a ``dumps=[...]``
  // kwarg (symmetric with ``deps=``) on the submit surface. Collect the marked
  // Vars now; emit the kwarg after ``deps=`` so the round-trip recovers the
  // same attr (VarPtr identity).
  std::set<const Var*> dump_set;
  for (const auto& [k, v] : op->attrs_) {
    if (k != kAttrDumpVars) continue;
    if (const auto* vars = std::any_cast<std::vector<VarPtr>>(&v)) {
      for (const auto& var : *vars) {
        if (var) dump_set.insert(var.get());
      }
    }
  }
  for (const auto& arg : op->args_) {
    stream_ << ", ";
    VisitExpr(arg);
  }

  if (!op->deps_.empty()) {
    stream_ << ", deps=[";
    for (size_t i = 0; i < op->deps_.size(); ++i) {
      if (i > 0) stream_ << ", ";
      INTERNAL_CHECK_SPAN(op->deps_[i], op->span_) << "Submit dep at index " << i << " is null";
      VisitExpr(op->deps_[i]);
    }
    stream_ << "]";
  }

  // ``dumps=[...]`` — emitted in arg order so the round-trip recovers the same
  // dump_vars vector (the parser builds it in arg order too). Only emitted when
  // non-empty, mirroring ``deps=``. Strict parse validation guarantees every
  // dump_var is a positional arg, so arg-order emission covers them all.
  if (!dump_set.empty()) {
    stream_ << ", dumps=[";
    bool first = true;
    for (const auto& arg : op->args_) {
      auto var = AsVarLike(arg);
      if (!var || dump_set.count(var.get()) == 0) continue;
      if (!first) stream_ << ", ";
      first = false;
      VisitExpr(arg);
    }
    stream_ << "]";
  }

  // SPMD launch spec — ``core_num=`` (required on pl.spmd_submit) and the
  // optional ``sync_start=True``. Emitted as kwargs so the parser recovers
  // them via the spmd_submit kwargs path.
  if (op->core_num_.has_value()) {
    INTERNAL_CHECK_SPAN(*op->core_num_, op->span_) << "Submit core_num is null";
    stream_ << ", core_num=";
    VisitExpr(*op->core_num_);
    if (op->sync_start_) {
      stream_ << ", sync_start=True";
    }
  }

  // Speculative early-dispatch opt-in — emitted as a kwarg so the parser
  // recovers it via the submit/spmd_submit kwargs path. Independent of the
  // launch spec, so emit it for a plain pl.submit too (not nested above).
  if (op->allow_early_resolve_) {
    stream_ << ", allow_early_resolve=True";
  }

  // Dispatch predicate — emitted as ``predicate=(<expr>)``; the comparison Expr
  // prints itself, and the parser recovers it by parsing the kwarg as an
  // ordinary expression, so the round trip needs no bespoke syntax.
  if (op->predicate_.has_value()) {
    INTERNAL_CHECK_SPAN(*op->predicate_, op->span_) << "Submit predicate is null";
    stream_ << ", predicate=(";
    VisitExpr(*op->predicate_);
    stream_ << ")";
  }

  // Surface the machine-only ``attrs={...}`` dict the same way Call does:
  // ``arg_directions`` keeps a bespoke emitter; every other attr (e.g.
  // ``arg_direction_overrides``) is rendered generically by ``PrintAttrValue``
  // so nothing is silently dropped. ``dump_vars`` is skipped here — it is
  // surfaced above as the ``dumps=`` kwarg on the submit surface.
  auto submit_arg_directions = op->GetArgDirections();
  const bool has_dirs = !submit_arg_directions.empty();
  std::vector<const std::pair<std::string, std::any>*> generic_attrs;
  for (const auto& kv : op->attrs_) {
    const std::string& k = kv.first;
    if (k == kAttrDumpVars) continue;       // emitted as dumps=
    if (k == kAttrArgDirections) continue;  // bespoke emitter below
    generic_attrs.push_back(&kv);
  }
  if (has_dirs || !generic_attrs.empty()) {
    stream_ << ", attrs={";
    bool first_key = true;
    if (has_dirs) {
      INTERNAL_CHECK_SPAN(submit_arg_directions.size() == op->args_.size(), op->span_)
          << "Submit arg_directions size (" << submit_arg_directions.size() << ") must match args size ("
          << op->args_.size() << ")";
      stream_ << "\"arg_directions\": [";
      for (size_t i = 0; i < submit_arg_directions.size(); ++i) {
        if (i > 0) stream_ << ", ";
        stream_ << prefix_ << ".adir." << ArgDirectionToDslName(submit_arg_directions[i]);
      }
      stream_ << "]";
      first_key = false;
    }
    for (const auto* kv : generic_attrs) {
      stream_ << (first_key ? "" : ", ");
      first_key = false;
      stream_ << std::quoted(kv->first) << ": ";
      PrintAttrValue(kv->second, op->span_);
    }
    stream_ << "}";
  }
  stream_ << ")";
}

void IRPythonPrinter::VisitExpr_(const MakeTuplePtr& op) {
  stream_ << "[";
  for (size_t i = 0; i < op->elements_.size(); ++i) {
    if (i > 0) stream_ << ", ";
    VisitExpr(op->elements_[i]);
  }
  stream_ << "]";
}

void IRPythonPrinter::VisitExpr_(const TupleGetItemExprPtr& op) {
  VisitExpr(op->tuple_);
  stream_ << "[" << op->index_ << "]";
}

// Binary and unary operators - reuse from base printer logic
void IRPythonPrinter::PrintChild(const ExprPtr& parent, const ExprPtr& child, bool is_left) {
  bool needs_parens = NeedsParens(parent, child, is_left);

  if (needs_parens) {
    stream_ << "(";
  }

  VisitExpr(child);

  if (needs_parens) {
    stream_ << ")";
  }
}

bool IRPythonPrinter::NeedsParens(const ExprPtr& parent, const ExprPtr& child, bool is_left) {
  Precedence parent_prec = GetPrecedence(parent);
  Precedence child_prec = GetPrecedence(child);

  if (child_prec < parent_prec) {
    return true;
  }

  if (child_prec == parent_prec) {
    if (IsRightAssociative(parent)) {
      return is_left;
    } else {
      return !is_left;
    }
  }

  return false;
}

namespace {
// A binary tree whose leaves are all ConstInt would print as bare Python
// arithmetic (e.g. `32 + 32`) and fold to one int on reparse. Such trees
// must print typed const leaves so the parser rebuilds them verbatim;
// folding is the Simplify pass's job, not the printer/parser's.
bool IsPureConstIntTree(const ExprPtr& expr) {
  if (As<ConstInt>(expr)) return true;
  if (auto bin = As<BinaryExpr>(expr)) {
    return IsPureConstIntTree(bin->left_) && IsPureConstIntTree(bin->right_);
  }
  return false;
}
}  // namespace

void IRPythonPrinter::PrintBinaryOp(const BinaryExprPtr& op, const char* op_symbol) {
  const bool saved_typed = typed_index_consts_;
  if (!typed_index_consts_ && IsPureConstIntTree(op)) {
    typed_index_consts_ = true;
  }
  PrintChild(op, op->left_, true);
  stream_ << " " << op_symbol << " ";
  PrintChild(op, op->right_, false);
  typed_index_consts_ = saved_typed;
}

void IRPythonPrinter::PrintFunctionBinaryOp(const BinaryExprPtr& op, const char* func_name) {
  stream_ << prefix_ << "." << func_name << "(";
  VisitExpr(op->left_);
  stream_ << ", ";
  VisitExpr(op->right_);
  stream_ << ")";
}

// Arithmetic binary operators
void IRPythonPrinter::VisitExpr_(const AddPtr& op) { PrintBinaryOp(op, "+"); }
void IRPythonPrinter::VisitExpr_(const SubPtr& op) { PrintBinaryOp(op, "-"); }
void IRPythonPrinter::VisitExpr_(const MulPtr& op) { PrintBinaryOp(op, "*"); }
void IRPythonPrinter::VisitExpr_(const FloorDivPtr& op) { PrintBinaryOp(op, "//"); }
void IRPythonPrinter::VisitExpr_(const FloorModPtr& op) { PrintBinaryOp(op, "%"); }
void IRPythonPrinter::VisitExpr_(const FloatDivPtr& op) { PrintBinaryOp(op, "/"); }
void IRPythonPrinter::VisitExpr_(const PowPtr& op) { PrintBinaryOp(op, "**"); }

// Function-style binary operators
void IRPythonPrinter::VisitExpr_(const MinPtr& op) { PrintFunctionBinaryOp(op, "min"); }
void IRPythonPrinter::VisitExpr_(const MaxPtr& op) { PrintFunctionBinaryOp(op, "max"); }

// Comparison operators
void IRPythonPrinter::VisitExpr_(const EqPtr& op) { PrintBinaryOp(op, "=="); }
void IRPythonPrinter::VisitExpr_(const NePtr& op) { PrintBinaryOp(op, "!="); }
void IRPythonPrinter::VisitExpr_(const LtPtr& op) { PrintBinaryOp(op, "<"); }
void IRPythonPrinter::VisitExpr_(const LePtr& op) { PrintBinaryOp(op, "<="); }
void IRPythonPrinter::VisitExpr_(const GtPtr& op) { PrintBinaryOp(op, ">"); }
void IRPythonPrinter::VisitExpr_(const GePtr& op) { PrintBinaryOp(op, ">="); }

// Logical operators
void IRPythonPrinter::VisitExpr_(const AndPtr& op) { PrintBinaryOp(op, "and"); }
void IRPythonPrinter::VisitExpr_(const OrPtr& op) { PrintBinaryOp(op, "or"); }
void IRPythonPrinter::VisitExpr_(const XorPtr& op) { PrintBinaryOp(op, "xor"); }

// Bitwise operators
void IRPythonPrinter::VisitExpr_(const BitAndPtr& op) { PrintBinaryOp(op, "&"); }
void IRPythonPrinter::VisitExpr_(const BitOrPtr& op) { PrintBinaryOp(op, "|"); }
void IRPythonPrinter::VisitExpr_(const BitXorPtr& op) { PrintBinaryOp(op, "^"); }
void IRPythonPrinter::VisitExpr_(const BitShiftLeftPtr& op) { PrintBinaryOp(op, "<<"); }
void IRPythonPrinter::VisitExpr_(const BitShiftRightPtr& op) { PrintBinaryOp(op, ">>"); }

// Unary operators
void IRPythonPrinter::VisitExpr_(const NegPtr& op) {
  stream_ << "-";
  Precedence operand_prec = GetPrecedence(op->operand_);
  if (operand_prec < Precedence::kUnary) {
    stream_ << "(";
    VisitExpr(op->operand_);
    stream_ << ")";
  } else {
    VisitExpr(op->operand_);
  }
}

void IRPythonPrinter::VisitExpr_(const AbsPtr& op) {
  stream_ << "abs(";
  VisitExpr(op->operand_);
  stream_ << ")";
}

void IRPythonPrinter::VisitExpr_(const CastPtr& op) {
  auto scalar_type = As<ScalarType>(op->GetType());
  INTERNAL_CHECK_SPAN(scalar_type, op->span_) << "Cast has non-scalar type";
  stream_ << prefix_ << ".cast(";
  VisitExpr(op->operand_);
  stream_ << ", " << prefix_ << "." << DataTypeToString(scalar_type->dtype_) << ")";
}

void IRPythonPrinter::VisitExpr_(const NotPtr& op) {
  stream_ << "not ";
  Precedence operand_prec = GetPrecedence(op->operand_);
  if (operand_prec < Precedence::kNot) {
    stream_ << "(";
    VisitExpr(op->operand_);
    stream_ << ")";
  } else {
    VisitExpr(op->operand_);
  }
}

void IRPythonPrinter::VisitExpr_(const BitNotPtr& op) {
  stream_ << "~";
  Precedence operand_prec = GetPrecedence(op->operand_);
  if (operand_prec < Precedence::kUnary) {
    stream_ << "(";
    VisitExpr(op->operand_);
    stream_ << ")";
  } else {
    VisitExpr(op->operand_);
  }
}

// Statement visitors with proper Python syntax
void IRPythonPrinter::VisitStmt_(const AssignStmtPtr& op) {
  // Print with type annotation: var: type = value
  // In concise mode, omit the type annotation: var = value
  VisitExpr(op->var_);
  if (!concise_) {
    stream_ << ": " << Print(op->var_->GetType());
  }
  stream_ << " = ";
  VisitExpr(op->value_);
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

  // Unroll loops cannot have iter_args. The DSL parser forbids init_values for
  // pl.unroll().
  if (op->kind_ == ForKind::Unroll && !op->iter_args_.empty()) {
    INTERNAL_CHECK_SPAN(false, op->span_) << "ForKind::Unroll does not support iter_args/init_values";
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
                                             const char* kwarg_name) {
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
    stream_ << ", " << kwarg_name << "=[";
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
    stream_ << prefix_ << ".cross_core_slot(slot_num=" << slot_num_holder->GetAttr<int>("slot_num", 0) << ")";
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
  if (!op->name_hint_.empty()) {
    stream_ << "name_hint=\"" << op->name_hint_ << "\"";
  }
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
    stream_ << "with " << prefix_ << ".spmd(";
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
    PrintScopeDepsAttr(op);
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
    const bool inlines_incore_body = SpmdInlineBodyRebuildsCarrier(incore);
    PrintScopeCachePolicyStmts(op);
    if (inlines_incore_body) {
      PrintScopeCachePolicyStmts(incore);
    }
    if (!inlines_incore_body) {
      // Print the body as-is. Either inlining would drop the carrier — the parser
      // re-synthesises an InCore only for a body that carries a split or reads the
      // per-block index, so the carrier must be spelled out as a nested
      // ``pl.at(level=pl.Level.CORE_GROUP)``, the same shape the plain with-form
      // prints — or (defensively) there is no InCore wrapper at all, which an
      // `as tid` Spmd scope should never be missing.
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
  if (first_op && IsOp(first_op, "tile.get_block_idx")) {
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
  PrintScopeAllowEarlyResolveAttr(op);
  PrintScopePredicateAttr(op);
  stream_ << "):\n";
  IncreaseIndent();
  PrintStmtBlock(op->body_);
  DecreaseIndent();
}

void IRPythonPrinter::VisitStmt_(const SplitAivScopeStmtPtr& op) {
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
    for (size_t i = start; i < body_seq->stmts_.size(); ++i) {
      if (ShouldSuppressPlaceholder(body_seq->stmts_, i)) continue;
      PrintStmtBlock(body_seq->stmts_[i]);
      if (i + 1 < body_seq->stmts_.size()) stream_ << "\n";
    }
  } else if (!body_seq && !has_idx_binding) {
    PrintStmtBlock(op->body_);
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
  // surface and round-trip: `with pl.auto_scope():` / `with pl.manual_scope():`.
  PrintRuntimeScopeHeader(op->manual_);
  IncreaseIndent();
  PrintStmtBlock(op->body_);
  DecreaseIndent();
}

void IRPythonPrinter::VisitStmt_(const CommDomainScopeStmtPtr& op) {
  // CommDomainScopeStmt is synthesized by MaterializeCommDomainScopes — it
  // has no user DSL surface, so the printer does NOT emit a `with` wrapper
  // (a wrapper would have no matching parser path). Reparse of the printed
  // text produces pre-pass IR; re-running the pass pipeline reconstructs
  // the scope.
  //
  // To keep dumps informative (since the Python printer is the only
  // round-trip surface for IR), we emit a single leading comment that
  // surfaces the scope's identity — covered devices and slot names — then
  // descend into the body at the SAME indent (no IncreaseIndent: the body
  // is structurally at the same level the comment occupies in the parent
  // statement stream).
  stream_ << "# pld.comm_domain: devices=";
  if (op->devices_.empty()) {
    stream_ << "all";
  } else {
    stream_ << "[";
    for (size_t i = 0; i < op->devices_.size(); ++i) {
      if (i > 0) stream_ << ", ";
      stream_ << op->devices_[i];
    }
    stream_ << "]";
  }
  stream_ << ", slots=[";
  for (size_t i = 0; i < op->slots_.size(); ++i) {
    if (i > 0) stream_ << ", ";
    INTERNAL_CHECK_SPAN(op->slots_[i], op->span_) << "CommDomainScopeStmt has null slot at index " << i;
    stream_ << op->slots_[i]->name_hint_;
  }
  stream_ << "]\n";
  PrintStmtBlock(op->body_);
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

void IRPythonPrinter::VisitStmtBody(const StmtPtr& body, const std::vector<VarPtr>& return_vars) {
  // A single-statement SeqStmts is a transparent wrapper (e.g. a user-written
  // `with pl.auto_scope():` body before NormalizeStmtStructure collapses it).
  // Unwrap it first so an AUTO scope reaches the rscope branch below with
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
      stream_ << GetIndent();
      PrintLeadingComments(yield_stmt);
      PrintYieldAssignmentVars(return_vars);
      stream_ << " = " << prefix_ << ".yield_(";
      for (size_t i = 0; i < yield_stmt->value_.size(); ++i) {
        if (i > 0) stream_ << ", ";
        VisitExpr(yield_stmt->value_[i]);
      }
      stream_ << ")";
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

      // Check if this is the last statement and it's a YieldStmt
      bool is_last = (i == seq_stmts->stmts_.size() - 1);
      if (auto yield_stmt = As<YieldStmt>(stmt)) {
        if (is_last && !yield_stmt->value_.empty() && !return_vars.empty()) {
          // Wrap as assignment
          stream_ << GetIndent();
          PrintLeadingComments(yield_stmt);
          PrintYieldAssignmentVars(return_vars);
          stream_ << " = " << prefix_ << ".yield_(";
          for (size_t j = 0; j < yield_stmt->value_.size(); ++j) {
            if (j > 0) stream_ << ", ";
            VisitExpr(yield_stmt->value_[j]);
          }
          stream_ << ")";
        } else {
          stream_ << GetIndent();
          PrintLeadingComments(yield_stmt);
          VisitStmt(stmt);
        }
      } else {
        PrintStmtBlock(stmt);
      }

      if (i < seq_stmts->stmts_.size() - 1) {
        stream_ << "\n";
      }
    }
  } else if (auto rscope = As<RuntimeScopeStmt>(body); rscope && !rscope->manual_) {
    // An AUTO RuntimeScopeStmt wrapping a for/if body (inserted by
    // MaterializeRuntimeScopes): emit the `with pl.auto_scope():` header and
    // recurse with return_vars so a trailing return-var yield *inside* the
    // scope still prints its `var = pl.yield_(...)` assignment LHS.
    stream_ << GetIndent();
    PrintRuntimeScopeHeader(/*manual=*/false);
    IncreaseIndent();
    VisitStmtBody(rscope->body_, return_vars);
    DecreaseIndent();
  } else {
    PrintStmtBlock(body);
  }
}

std::string IRPythonPrinter::GetVarName(const Var* var) const {
  // Free variables print with a visible suffix so a dump of malformed IR is
  // unmistakably invalid instead of silently reusing an ordinary-looking name
  // (the marked use-site pinpoints the SSAVerify "used outside its defining
  // scope" diagnostic). dyn_var_rename_map_ entries are never flagged free.
  std::string suffix = free_body_vars_.count(var) ? "__FREE_VAR" : "";
  auto it = var_rename_map_.find(var);
  if (it != var_rename_map_.end()) return it->second + suffix;
  auto dyn_it = dyn_var_rename_map_.find(var);
  if (dyn_it != dyn_var_rename_map_.end()) return dyn_it->second;
  return var->name_hint_ + suffix;
}

std::string IRPythonPrinter::PrintSubExpr(const ExprPtr& expr) const {
  IRPythonPrinter sub_printer(prefix_);
  sub_printer.var_rename_map_ = var_rename_map_;
  sub_printer.dyn_var_rename_map_ = dyn_var_rename_map_;
  sub_printer.free_body_vars_ = free_body_vars_;
  return sub_printer.Print(expr);
}

void IRPythonPrinter::BuildVarRenameMap(const std::vector<VarPtr>& params, const StmtPtr& body,
                                        bool is_function) {
  // Collect Var/IterArg pointers in DFS pre-order: params, then body defs,
  // then body uses. Defs precede uses so a pointer appearing as both keeps
  // its def-site canonical name; pointers appearing only as uses (e.g. a
  // dangling reference from a buggy transform, or a sibling-scope iter_arg
  // sharing a name_hint) still get their own suffix. (#1244)
  std::vector<const Var*> ordered_refs;
  ordered_refs.reserve(params.size());
  std::unordered_set<const Var*> param_ptrs;
  for (const auto& p : params) {
    ordered_refs.push_back(p.get());
    param_ptrs.insert(p.get());
  }
  body_defined_vars_.clear();
  free_body_vars_.clear();
  if (body) {
    var_collectors::VarDefUseCollector body_collector;
    body_collector.VisitStmt(body);
    ordered_refs.insert(ordered_refs.end(), body_collector.var_defs_ordered.begin(),
                        body_collector.var_defs_ordered.end());
    ordered_refs.insert(ordered_refs.end(), body_collector.var_uses_ordered.begin(),
                        body_collector.var_uses_ordered.end());
    // A function is a closed scope: every body use must resolve to a parameter
    // or a body-local definition. A use that resolves to neither is a free
    // variable — malformed IR — so flag it (see free_body_vars_). Skipped for
    // bare-stmt roots, whose absent context legitimately supplies such vars.
    if (is_function) {
      for (const Var* use : body_collector.var_uses_ordered) {
        if (!param_ptrs.count(use) && !body_collector.var_defs.count(use)) {
          free_body_vars_.insert(use);
        }
      }
    }
    body_defined_vars_ = std::move(body_collector.var_defs);
  }
  // Drop program-level dynamic dim vars: they are already disambiguated
  // globally in dyn_var_rename_map_, and re-registering them here would
  // shadow that map in GetVarName's two-tier lookup. They are program-scoped,
  // so they are also legitimately "free" in a function body — drop them from
  // free_body_vars_ too to avoid false positives.
  if (!dyn_var_rename_map_.empty()) {
    ordered_refs.erase(std::remove_if(ordered_refs.begin(), ordered_refs.end(),
                                      [this](const Var* v) { return dyn_var_rename_map_.count(v) > 0; }),
                       ordered_refs.end());
    for (auto it = free_body_vars_.begin(); it != free_body_vars_.end();) {
      if (dyn_var_rename_map_.count(*it)) {
        it = free_body_vars_.erase(it);
      } else {
        ++it;
      }
    }
  }
  auto_name::BuildRenameMapForDefs(ordered_refs, var_rename_map_);
}

void IRPythonPrinter::VisitFunction(const FunctionPtr& func) {
  // Build rename map for this function to handle SSA name shadowing.
  BuildVarRenameMap(func);

  // A ``split`` of ``SplitMode::None`` is filtered because it is a non-canonical
  // spelling of "no split" (``Function::GetSplitMode`` maps a stored 0 to
  // ``nullopt``, exactly as an absent key does), and the parser drops it — so
  // emitting it would make print -> parse lossy. No pass produces it; this only
  // normalizes IR that bypassed them (a ``.pto`` blob written before
  // OutlineIncoreScopes stopped stamping it, a programmatically built Function).
  // ``auto_scope`` rides in attrs_ but prints as a dedicated kwarg.
  // The pointer form of ``any_cast`` keeps a mistyped attr failing at the
  // existing print site rather than in this predicate, which also runs for the
  // ``has_attrs`` tests.
  // Every function attr prints as a ``pl.func_attr({...})`` body prologue except
  // the two the parser must read BEFORE it can walk the body — a body-position
  // declaration would arrive too late to take effect. Those two print as their
  // own ``@pl.function(...)`` keyword (``auto_scope=`` / ``external_source=``),
  // which is also the user-facing spelling, so reparsing never goes through the
  // deprecated ``attrs=`` keyword and the compiler never warns on its own
  // output. Kept in sync with the parser's ``_DECORATOR_ONLY_FUNC_ATTRS``.
  auto is_filtered_attr = [](const std::string& k, const std::any& v) {
    if (k == kAttrAutoScope || k == kAttrExternalSource) return true;
    if (k != "split") return false;
    const int* split_value = std::any_cast<int>(&v);
    return split_value != nullptr && *split_value == static_cast<int>(SplitMode::None);
  };
  auto print_func_attr_value = [&](const std::string& key, const std::any& value) {
    if (key == "split") {
      int split_value = AnyCast<int>(value, "func attr key: " + key);
      auto split_mode = static_cast<SplitMode>(split_value);
      stream_ << prefix_ << ".SplitMode." << SplitModeToPythonString(split_mode);
    } else {
      PrintAttrValue(value, func->span_);
    }
  };

  // Print decorator
  stream_ << GetIndent() << "@" << prefix_ << ".function";
  {
    bool has_type = func->func_type_ != FunctionType::Opaque;
    bool has_level = func->level_.has_value();
    bool has_role = func->role_.has_value();
    // Absent ⇒ default True ⇒ not printed.
    bool auto_scope_off = !func->GetAttr<bool>(kAttrAutoScope, true);
    std::string external_source = func->GetAttr<std::string>(kAttrExternalSource, "");
    if (has_type || has_level || has_role || auto_scope_off || !external_source.empty()) {
      stream_ << "(";
      bool first = true;
      if (has_type) {
        stream_ << "type=" << prefix_ << ".FunctionType." << FunctionTypeToString(func->func_type_);
        first = false;
      }
      if (has_level) {
        if (!first) stream_ << ", ";
        stream_ << "level=" << prefix_ << ".Level." << LevelToString(*func->level_);
        first = false;
      }
      if (has_role) {
        if (!first) stream_ << ", ";
        stream_ << "role=" << prefix_ << ".Role." << RoleToString(*func->role_);
        first = false;
      }
      if (auto_scope_off) {
        if (!first) stream_ << ", ";
        stream_ << "auto_scope=False";
        first = false;
      }
      // Last keyword: nothing reads `first` after this, so it is not updated.
      if (!external_source.empty()) {
        if (!first) stream_ << ", ";
        stream_ << "external_source=" << std::quoted(external_source);
      }
      stream_ << ")";
    }
  }
  stream_ << "\n";

  // Print function signature
  stream_ << GetIndent() << "def " << func->name_ << "(";

  // Add 'self' as first parameter when inside @pl.program — except for
  // SubWorker functions, which are declared self-contained.
  bool is_sub_worker = func->role_.has_value() && *func->role_ == Role::SubWorker;
  bool emit_self = current_program_ && !is_sub_worker;
  if (emit_self) {
    stream_ << "self";
  }

  // Everything from here to the closing ":" is the signature, which Python
  // evaluates before the body binds any name. Flag it so a MemRef annotation
  // referring to a body-allocated base Ptr prints its name as a string instead
  // of an unbound forward reference (see PrintMemRef).
  printing_signature_ = true;

  // Print parameters with type annotations and direction wrappers
  for (size_t i = 0; i < func->params_.size(); ++i) {
    if (i > 0 || emit_self) stream_ << ", ";
    const auto& var = func->params_[i];
    const auto& dir = func->param_directions_[i];
    stream_ << GetVarName(var.get()) << ": ";
    if (dir == ParamDirection::InOut) {
      stream_ << prefix_ << ".InOut[" << Print(var->GetType()) << "]";
    } else if (dir == ParamDirection::Out) {
      stream_ << prefix_ << ".Out[" << Print(var->GetType()) << "]";
    } else {
      stream_ << Print(var->GetType());
    }
  }

  stream_ << ")";

  // Print return type annotation
  if (!func->return_types_.empty()) {
    stream_ << " -> ";
    if (func->return_types_.size() == 1) {
      stream_ << Print(func->return_types_[0]);
    } else {
      stream_ << "tuple[";
      for (size_t i = 0; i < func->return_types_.size(); ++i) {
        if (i > 0) stream_ << ", ";
        stream_ << Print(func->return_types_[i]);
      }
      stream_ << "]";
    }
  }

  printing_signature_ = false;

  stream_ << ":\n";

  // Print body - convert yield to return in function context
  IncreaseIndent();

  // ``pl.func_attr({...})`` prologue: every attr that is not decorator-only.
  // Emitted before the body statements so it reparses as a prologue. Skipped
  // for bodyless functions (abstract SubWorker / external kernel), which must
  // round-trip as a bare ``...`` — those carry only decorator-only attrs.
  std::vector<const std::pair<std::string, std::any>*> prologue_attrs;
  for (const auto& kv : func->attrs_) {
    if (is_filtered_attr(kv.first, kv.second)) continue;
    prologue_attrs.push_back(&kv);
  }
  // An abstract SubWorker prints its prologue too, before the bare ``...``.
  // ``_is_abstract_subworker_body`` skips the directive exactly as it skips a
  // docstring, so the body still reads as abstract on reparse and the attrs
  // survive — they would otherwise have no printable position at all.
  if (!prologue_attrs.empty()) {
    stream_ << GetIndent() << prefix_ << ".func_attr({";
    for (size_t i = 0; i < prologue_attrs.size(); ++i) {
      if (i > 0) stream_ << ", ";
      stream_ << std::quoted(prologue_attrs[i]->first) << ": ";
      print_func_attr_value(prologue_attrs[i]->first, prologue_attrs[i]->second);
    }
    stream_ << "})\n";
  }

  if (func->requires_runtime_binding_) {
    // Abstract SubWorker: runtime-bound callback. Round-trips as `...`, which
    // the parser re-detects (see `_is_abstract_subworker_body`).
    stream_ << GetIndent() << "...";
  } else if (func->body_) {
    if (auto seq_stmts = As<SeqStmts>(func->body_)) {
      if (seq_stmts->stmts_.empty()) {
        stream_ << GetIndent() << "pass";
      } else {
        for (size_t i = 0; i < seq_stmts->stmts_.size(); ++i) {
          if (ShouldSuppressPlaceholder(seq_stmts->stmts_, i)) continue;
          // Convert yield to return in function context
          if (auto yield_stmt = As<YieldStmt>(seq_stmts->stmts_[i])) {
            stream_ << GetIndent();
            PrintLeadingComments(yield_stmt);
            stream_ << "return";
            if (!yield_stmt->value_.empty()) {
              stream_ << " ";
              for (size_t j = 0; j < yield_stmt->value_.size(); ++j) {
                if (j > 0) stream_ << ", ";
                VisitExpr(yield_stmt->value_[j]);
              }
            }
          } else {
            PrintStmtBlock(seq_stmts->stmts_[i]);
          }
          if (i < seq_stmts->stmts_.size() - 1) {
            stream_ << "\n";
          }
        }
      }
    } else if (auto yield_stmt = As<YieldStmt>(func->body_)) {
      stream_ << GetIndent();
      PrintLeadingComments(yield_stmt);
      stream_ << "return";
      if (!yield_stmt->value_.empty()) {
        stream_ << " ";
        for (size_t i = 0; i < yield_stmt->value_.size(); ++i) {
          if (i > 0) stream_ << ", ";
          VisitExpr(yield_stmt->value_[i]);
        }
      }
    } else {
      PrintStmtBlock(func->body_);
    }
  }
  DecreaseIndent();
}

// Helper class to collect GlobalVar references from a function's body
class GlobalVarCollector : public IRVisitor {
 public:
  std::set<GlobalVarPtr, GlobalVarPtrLess> collected_gvars;

  void VisitExpr_(const CallPtr& op) override {
    // Visit the op field (which may be a GlobalVar for cross-function calls)
    INTERNAL_CHECK_SPAN(op->op_, op->span_) << "Call has null op";
    if (auto gvar = As<GlobalVar>(op->op_)) {
      collected_gvars.insert(gvar);
    }
    // Visit arguments
    IRVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const SubmitPtr& op) override {
    // Submit's callee is a GlobalVar (the kernel being launched). Treat the
    // same as Call so cross-function dependencies through pl.submit are
    // captured by the topological sort.
    INTERNAL_CHECK_SPAN(op->op_, op->span_) << "Submit has null op";
    if (auto gvar = As<GlobalVar>(op->op_)) {
      collected_gvars.insert(gvar);
    }
    IRVisitor::VisitExpr_(op);
  }
};

// Topologically sort functions so called functions come before callers
// This ensures that when reparsing, function return types are known when needed
static std::vector<std::pair<GlobalVarPtr, FunctionPtr>> TopologicalSortFunctions(
    const std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess>& functions) {
  // Build dependency graph: function -> set of functions it calls
  std::map<GlobalVarPtr, std::set<GlobalVarPtr, GlobalVarPtrLess>, GlobalVarPtrLess> dependencies;
  std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> gvar_to_func;

  for (const auto& [gvar, func] : functions) {
    gvar_to_func[gvar] = func;
    // Collect all GlobalVars referenced in the function body
    GlobalVarCollector collector;
    if (func->body_) {
      collector.VisitStmt(func->body_);
    }
    // Only keep GlobalVars that are actually functions in this program
    for (const auto& called_gvar : collector.collected_gvars) {
      if (functions.count(called_gvar) > 0) {
        dependencies[gvar].insert(called_gvar);
      }
    }
  }

  // Topological sort using DFS
  std::vector<std::pair<GlobalVarPtr, FunctionPtr>> sorted;
  std::set<GlobalVarPtr, GlobalVarPtrLess> visited;
  std::set<GlobalVarPtr, GlobalVarPtrLess> in_progress;  // For cycle detection

  std::function<bool(const GlobalVarPtr&)> dfs = [&](const GlobalVarPtr& gvar) -> bool {
    if (visited.count(gvar)) return true;
    if (in_progress.count(gvar)) return false;  // Cycle detected

    in_progress.insert(gvar);

    // Visit dependencies first (dependencies = functions this function calls)
    if (dependencies.count(gvar)) {
      for (const auto& dep : dependencies[gvar]) {
        if (!dfs(dep)) return false;  // Cycle detected
      }
    }

    in_progress.erase(gvar);
    visited.insert(gvar);
    // Add to sorted AFTER visiting dependencies, so dependencies come first
    sorted.emplace_back(gvar, gvar_to_func[gvar]);
    return true;
  };

  // Visit all functions
  for (const auto& [gvar, func] : functions) {
    if (!dfs(gvar)) {
      // Cycle detected, fall back to original order
      sorted.clear();
      for (const auto& pair : functions) {
        sorted.emplace_back(pair);
      }
      return sorted;
    }
  }

  return sorted;
}

static std::unordered_map<const Var*, std::string> CollectDynVarMapping(const ProgramPtr& program) {
  // Phase 1: Collect all distinct Var* pointers used as dynamic dims,
  // preserving insertion order for deterministic output.
  std::vector<const Var*> dyn_var_ptrs;
  std::unordered_set<const Var*> seen_ptrs;

  auto try_insert = [&](const Var* var) {
    if (seen_ptrs.insert(var).second) {
      dyn_var_ptrs.push_back(var);
    }
  };

  // Walk an expression tree to find all Var nodes (handles both bare Var dims
  // and complex expressions like M + 1 where Var is a sub-expression).
  std::function<void(const ExprPtr&)> collect_vars_from_expr = [&](const ExprPtr& expr) {
    if (!expr) return;
    if (auto var = As<Var>(expr)) {
      try_insert(var.get());
    } else if (auto bin = As<BinaryExpr>(expr)) {
      collect_vars_from_expr(bin->left_);
      collect_vars_from_expr(bin->right_);
    } else if (auto unary = As<UnaryExpr>(expr)) {
      collect_vars_from_expr(unary->operand_);
    }
  };

  std::function<void(const TypePtr&)> collect_from_type = [&](const TypePtr& type) {
    if (auto tensor_type = As<TensorType>(type)) {
      for (const auto& dim : tensor_type->shape_) {
        collect_vars_from_expr(dim);
      }
      if (tensor_type->tensor_view_.has_value()) {
        for (const auto& dim : tensor_type->tensor_view_->valid_shape) {
          collect_vars_from_expr(dim);
        }
        for (const auto& dim : tensor_type->tensor_view_->stride) {
          collect_vars_from_expr(dim);
        }
      }
    } else if (auto tile_type = As<TileType>(type)) {
      for (const auto& dim : tile_type->shape_) {
        collect_vars_from_expr(dim);
      }
      if (tile_type->tile_view_.has_value()) {
        for (const auto& dim : tile_type->tile_view_->valid_shape) {
          collect_vars_from_expr(dim);
        }
        for (const auto& dim : tile_type->tile_view_->stride) {
          collect_vars_from_expr(dim);
        }
        collect_vars_from_expr(tile_type->tile_view_->start_offset);
      }
    } else if (auto tuple_type = As<TupleType>(type)) {
      for (const auto& elem_type : tuple_type->types_) {
        collect_from_type(elem_type);
      }
    }
  };
  // Use a full IRVisitor so that dynamic-dimension Var names are found in every
  // expression context: loop bounds (ForStmt start/stop/step),
  // if/while conditions, EvalStmt expressions, AssignStmt values, etc.
  // The ad-hoc collect_from_stmt only inspected AssignStmt variable types and
  // missed all those other locations.
  class DynVarCollector : public IRVisitor {
   public:
    explicit DynVarCollector(const std::function<void(const TypePtr&)>& collect_from_type)
        : collect_from_type_(collect_from_type) {}

    void VisitExpr_(const VarPtr& op) override {
      // Collect dynamic dimension names from the variable's type annotation.
      // Handles TensorType/TileType shapes and their tensor_view_/tile_view_ fields.
      collect_from_type_(op->GetType());
    }

   private:
    const std::function<void(const TypePtr&)>& collect_from_type_;
  };

  // Collect dynamic dim vars from type annotations, and simultaneously track
  // all defined vars (params + body defs) so we can filter them out below.
  DynVarCollector collector(collect_from_type);
  std::unordered_set<const Var*> defined_vars;
  for (const auto& [gvar, func] : program->functions_) {
    for (const auto& param : func->params_) {
      collector.VisitExpr(param);
      defined_vars.insert(param.get());
    }
    for (const auto& ret_type : func->return_types_) {
      collect_from_type(ret_type);
    }
    if (func->body_) {
      collector.VisitStmt(func->body_);
      var_collectors::VarDefUseCollector body_def_collector;
      body_def_collector.VisitStmt(func->body_);
      defined_vars.insert(body_def_collector.var_defs.begin(), body_def_collector.var_defs.end());
    }
  }

  // A parameter whose name is ALSO read inside a parameter *type annotation*
  // still needs its pl.dynamic() declaration: Python evaluates annotations in the
  // enclosing scope, before any parameter exists, so without the declaration the
  // printed signature raises NameError on re-parse. MaterializeValidShapeSymbols
  // produces exactly this shape — the valid_shape symbol becomes a scalar
  // parameter while the tensor parameter's annotation keeps naming it — and the
  // parser re-unifies the two (see ast_parser's DynVar re-point on f.param).
  // Collected separately from the pass above: that one dedups through seen_ptrs,
  // which would swallow every repeat occurrence.
  // Deliberately narrow (issue #854 keeps the other two cases undeclared):
  //   * a body-local var read in a valid_shape stays undeclared — it is defined
  //     in the body, so the annotation resolves against the local;
  //   * a param read as a *physical* dim stays undeclared — orchestration codegen
  //     defines those symbols from the runtime tensor's shapes[].
  // Only a param read in a valid_shape has neither escape, which is exactly what
  // MaterializeValidShapeSymbols produces.
  std::unordered_set<const Var*> read_in_param_types;
  {
    std::unordered_set<const Var*> param_vars;
    for (const auto& [gvar, func] : program->functions_) {
      for (const auto& param : func->params_) param_vars.insert(param.get());
    }
    std::function<void(const ExprPtr&)> collect_reads = [&](const ExprPtr& expr) {
      if (!expr) return;
      if (auto var = As<Var>(expr)) {
        if (param_vars.count(var.get()) > 0) read_in_param_types.insert(var.get());
      } else if (auto bin = As<BinaryExpr>(expr)) {
        collect_reads(bin->left_);
        collect_reads(bin->right_);
      } else if (auto unary = As<UnaryExpr>(expr)) {
        collect_reads(unary->operand_);
      }
    };
    for (const auto& [gvar, func] : program->functions_) {
      for (const auto& param : func->params_) {
        // AsTensorTypeLike, not As<TensorType>: DistributedTensorType has its own
        // ObjectKind and carries valid_shape symbols the same way.
        auto tensor_type = AsTensorTypeLike(param->GetType());
        if (!tensor_type || !tensor_type->tensor_view_.has_value()) continue;
        for (const auto& dim : tensor_type->tensor_view_->valid_shape) collect_reads(dim);
      }
    }
  }

  // Filter out locally-defined vars and function params: they should not get
  // pl.dynamic() declarations — only truly free dimension variables should, plus
  // the params that a parameter annotation reads (see read_in_param_types above).
  dyn_var_ptrs.erase(std::remove_if(dyn_var_ptrs.begin(), dyn_var_ptrs.end(),
                                    [&defined_vars, &read_in_param_types](const Var* v) {
                                      return defined_vars.count(v) > 0 && read_in_param_types.count(v) == 0;
                                    }),
                     dyn_var_ptrs.end());

  // Phase 2: Assign unique printed names, disambiguating collisions.
  // Reuses the same rename logic as BuildVarRenameMap.
  // include_unique_names=true so VisitProgram can iterate the full map for
  // pl.dynamic() declarations.
  std::unordered_map<const Var*, std::string> result;
  auto_name::BuildRenameMapForDefs(dyn_var_ptrs, result, /*include_unique_names=*/true);
  return result;
}

void IRPythonPrinter::VisitProgram(const ProgramPtr& program) {
  CHECK(prefix_ != "pld")
      << "Python printer prefix 'pld' is reserved for pypto.language.distributed; choose another prefix";

  // Render everything below the imports into a scratch buffer first. Whether the
  // program needs ``import pypto.language.distributed as pld`` is only knowable
  // once the body exists: every distributed spelling is hardcoded with the
  // ``pld.`` prefix (types, ops, and the comm-domain comment alike), so a plain
  // substring search over the rendered body is an exact test. Without the import
  // the output references an unbound name and is not valid standalone Python --
  // it only re-parses because the text parser injects ``pld`` into the exec
  // namespace itself.
  std::ostringstream body;
  stream_.swap(body);

  // Emit pl.dynamic() declarations for dynamic shape variables used in function signatures.
  // Uses pointer-identity-aware collection so distinct Var* with the same name_hint_
  // get disambiguated printed names (issue #618).
  dyn_var_rename_map_ = CollectDynVarMapping(program);
  if (!dyn_var_rename_map_.empty()) {
    // Sort by disambiguated name for deterministic output
    std::vector<std::pair<const Var*, std::string>> sorted_dyn_vars(dyn_var_rename_map_.begin(),
                                                                    dyn_var_rename_map_.end());
    std::sort(sorted_dyn_vars.begin(), sorted_dyn_vars.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });
    for (const auto& [var, name] : sorted_dyn_vars) {
      stream_ << name << " = " << prefix_ << ".dynamic(\"" << name << "\")\n";
    }
    stream_ << "\n";
  }

  // Print as @pl.program class with @pl.function methods
  stream_ << "@" << prefix_ << ".program\n";
  stream_ << "class " << (program->name_.empty() ? "Program" : program->name_) << ":\n";

  IncreaseIndent();

  // Sort functions in dependency order (called functions before callers)
  auto sorted_functions = TopologicalSortFunctions(program->functions_);

  // Print each function as a method, delegating to VisitFunction
  // Setting current_program_ enables self parameter and self.method() call printing
  auto prev_program = current_program_;
  current_program_ = program;

  bool first = true;
  for (const auto& [gvar, func] : sorted_functions) {
    if (!first) {
      stream_ << "\n";  // Blank line between functions
    }
    first = false;

    VisitFunction(func);
  }

  current_program_ = prev_program;
  DecreaseIndent();

  // Restore the caller's stream and prepend the header + imports the body needs.
  const std::string body_str = stream_.str();
  stream_.swap(body);

  stream_ << "# pypto.program: " << (program->name_.empty() ? "Program" : program->name_) << "\n";
  if (prefix_ == "pl") {
    stream_ << "import pypto.language as pl\n";
  } else {
    stream_ << "from pypto import language as " << prefix_ << "\n";
  }
  if (body_str.find("pld.") != std::string::npos) {
    stream_ << "import pypto.language.distributed as pld\n";
  }
  stream_ << "\n" << body_str;
}

std::string IRPythonPrinter::PrintExprForType(const ExprPtr& expr) {
  // Top-level static dims always print bare regardless of dtype: dim dtype is
  // insignificant for static shapes (structural equality compares ConstInt by
  // value) and bare literals reparse to the canonical INDEX form. Only leaves
  // inside composite trees need pl.const typing — see typed_index_consts_.
  if (auto const_int = As<ConstInt>(expr)) {
    return std::to_string(const_int->value_);
  }
  if (auto var = As<Var>(expr)) {
    return GetVarName(var.get());
  }
  IRPythonPrinter temp_printer(prefix_);
  temp_printer.dyn_var_rename_map_ = dyn_var_rename_map_;
  // Composite dims must reparse to the same tree, not Python-eval to a folded
  // int. Typed const leaves (pl.const(v, pl.INDEX)) make them self-describing;
  // simplification stays the Simplify pass's job.
  temp_printer.typed_index_consts_ = true;
  return temp_printer.Print(expr);
}

void IRPythonPrinter::PrintShapeDims(std::ostringstream& oss, const std::vector<ExprPtr>& shape) {
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << PrintExprForType(shape[i]);
  }
}

// Helper methods for MemRef and TileView printing
std::string IRPythonPrinter::PrintMemRef(const MemRef& memref) {
  std::ostringstream oss;

  // An author-declared allocation prints in the one-argument form: it carries no
  // size or address to print, since InitMemRef derives both. Printing it with the
  // invented pair would make it indistinguishable from a compiler allocation on
  // reparse.
  if (memref.is_pinned_) {
    oss << prefix_ << ".MemRef(\"" << GetVarName(memref.base_.get()) << "\"";
    // `slots=` is what makes the subscript below legal on reparse, so it has to
    // ride along; omitted at 1 so an unsubscripted declaration prints as before.
    if (memref.slot_count_ > 1) oss << ", slots=" << memref.slot_count_;
    oss << ")";
    if (memref.slot_index_.has_value() && *memref.slot_index_) {
      // The index may be a runtime expression, so print it through the expression
      // printer rather than assuming a constant.
      oss << "[" << PrintSubExpr(*memref.slot_index_) << "]";
    }
    return oss.str();
  }

  oss << prefix_ << ".MemRef(";

  // Base Ptrs defined in the function body (by alloc statements) are printed as
  // bare variable references; everything else uses a string literal (forward
  // references in parameter annotations, standalone type printing, etc.).
  //
  // A signature annotation is exactly such a forward reference: it is emitted
  // before the body, so even a base Ptr that *is* body-defined is not yet bound
  // there and must take the string form, or the reparse fails with NameError.
  if (!printing_signature_ && body_defined_vars_.count(memref.base_.get())) {
    oss << GetVarName(memref.base_.get());
  } else {
    oss << "\"" << GetVarName(memref.base_.get()) << "\"";
  }

  // Print byte offset off the main stream, but with this printer's naming state:
  // a dynamic offset (a tile.slice view) names loop vars, which may have been
  // disambiguated.
  oss << ", " << PrintSubExpr(memref.byte_offset_);

  // Print size
  oss << ", " << memref.size_;

  // Slot geometry survives InitMemRef (it says which slot of what this MemRef is,
  // not merely how to compute its offset), so a resolved slot has to print it too
  // or the reparsed MemRef silently becomes an ordinary one at the same offset —
  // and PTO codegen would emit N unrelated allocs instead of one multi-buffer.
  if (memref.slot_count_ > 1) oss << ", slots=" << memref.slot_count_;
  oss << ")";
  if (memref.slot_index_.has_value() && *memref.slot_index_) {
    // The index may be a runtime expression, so print it through the expression
    // printer rather than assuming a constant.
    oss << "[" << PrintSubExpr(*memref.slot_index_) << "]";
  }
  return oss.str();
}

std::string IRPythonPrinter::PrintTileView(const TileView& tile_view, const std::vector<ExprPtr>& tile_shape,
                                           const std::optional<MemorySpace>& memory_space) {
  // Caller already gated on has_value() (or passed the resolved effective view in
  // explicit_layout_ mode); a present view is non-implicit by the TileType
  // canonical-encoding invariant, so always render the explicit form.
  std::ostringstream oss;
  oss << prefix_ << ".TileView(";

  bool first = true;
  auto maybe_comma = [&]() {
    if (!first) oss << ", ";
    first = false;
  };

  // Compute the implicit view so we can elide layout fields that match it — but
  // only in concise mode. Under explicit_layout_ the blayout/slayout/fractal
  // guards short-circuit on `explicit_layout_ ||` and never read it, so skipping
  // it avoids a wasted GetImplicitTileView (already folded into the effective
  // view the caller resolved for absent-view tiles).
  TileView implicit_view;
  if (!explicit_layout_) {
    implicit_view = tile_view_semantics::GetImplicitTileView(tile_shape, memory_space);
  }

  // valid_shape — omit if it matches the parent tile's shape
  bool valid_shape_matches = tile_view.valid_shape.empty() ||
                             tile_view_semantics::ShapeExprListsEquivalent(tile_view.valid_shape, tile_shape);
  if (!valid_shape_matches) {
    maybe_comma();
    oss << "valid_shape=[";
    for (size_t i = 0; i < tile_view.valid_shape.size(); ++i) {
      if (i > 0) oss << ", ";
      oss << PrintExprForType(tile_view.valid_shape[i]);
    }
    oss << "]";
  }

  // stride — omit if empty
  if (!tile_view.stride.empty()) {
    maybe_comma();
    oss << "stride=[";
    for (size_t i = 0; i < tile_view.stride.size(); ++i) {
      if (i > 0) oss << ", ";
      oss << PrintExprForType(tile_view.stride[i]);
    }
    oss << "]";
  }

  // start_offset — omit if null
  if (tile_view.start_offset) {
    maybe_comma();
    oss << "start_offset=";
    oss << PrintExprForType(tile_view.start_offset);
  }

  // blayout — omit if matches the implicit view for this shape+memory_space
  // (kept in explicit_layout_ mode so every tile self-describes its layout).
  if (explicit_layout_ || tile_view.blayout != implicit_view.blayout) {
    maybe_comma();
    oss << "blayout=" << prefix_ << ".TileLayout.";
    switch (tile_view.blayout) {
      case TileLayout::none_box:
        oss << "none_box";
        break;
      case TileLayout::row_major:
        oss << "row_major";
        break;
      case TileLayout::col_major:
        oss << "col_major";
        break;
    }
  }

  // slayout — omit if matches the implicit view for this shape+memory_space
  // (kept in explicit_layout_ mode so every tile self-describes its layout).
  if (explicit_layout_ || tile_view.slayout != implicit_view.slayout) {
    maybe_comma();
    oss << "slayout=" << prefix_ << ".TileLayout.";
    switch (tile_view.slayout) {
      case TileLayout::none_box:
        oss << "none_box";
        break;
      case TileLayout::row_major:
        oss << "row_major";
        break;
      case TileLayout::col_major:
        oss << "col_major";
        break;
    }
  }

  // fractal — omit if matches the implicit view for this shape+memory_space
  // (kept in explicit_layout_ mode so every tile self-describes its layout).
  if (explicit_layout_ || tile_view.fractal != implicit_view.fractal) {
    maybe_comma();
    oss << "fractal=" << tile_view.fractal;
  }

  // pad — omit if null (default)
  if (tile_view.pad != PadValue::null) {
    maybe_comma();
    oss << "pad=" << prefix_ << ".PadValue.";
    switch (tile_view.pad) {
      case PadValue::null:
        oss << "null";
        break;
      case PadValue::zero:
        oss << "zero";
        break;
      case PadValue::max:
        oss << "max";
        break;
      case PadValue::min:
        oss << "min";
        break;
    }
  }

  // compact — omit if null (default)
  if (tile_view.compact != CompactMode::null) {
    maybe_comma();
    oss << "compact=" << prefix_ << ".CompactMode.";
    switch (tile_view.compact) {
      case CompactMode::null:
        oss << "null";
        break;
      case CompactMode::normal:
        oss << "normal";
        break;
    }
  }

  // If all fields were at defaults, return empty string to skip tile_view entirely
  if (first) return "";

  oss << ")";
  return oss.str();
}

std::string IRPythonPrinter::PrintTensorView(const TensorView& tensor_view,
                                             const std::vector<ExprPtr>& tensor_shape) {
  std::ostringstream oss;
  oss << prefix_ << ".TensorView(";

  bool first = true;
  auto maybe_comma = [&]() {
    if (!first) oss << ", ";
    first = false;
  };

  // valid_shape — omit if it matches the parent tensor's shape
  bool valid_shape_matches = (tensor_view.valid_shape.size() == tensor_shape.size());
  if (valid_shape_matches) {
    for (size_t i = 0; i < tensor_shape.size(); ++i) {
      const auto& vs_expr = tensor_view.valid_shape[i];
      const auto& ts_expr = tensor_shape[i];
      if (vs_expr == ts_expr) continue;
      auto vs = As<ConstInt>(vs_expr);
      auto ts = As<ConstInt>(ts_expr);
      if (!vs || !ts || vs->value_ != ts->value_) {
        valid_shape_matches = false;
        break;
      }
    }
  }
  if (!valid_shape_matches && !tensor_view.valid_shape.empty()) {
    maybe_comma();
    oss << "valid_shape=[";
    for (size_t i = 0; i < tensor_view.valid_shape.size(); ++i) {
      if (i > 0) oss << ", ";
      oss << PrintExprForType(tensor_view.valid_shape[i]);
    }
    oss << "]";
  }

  bool has_stride = !tensor_view.stride.empty();
  bool has_non_default_layout = (tensor_view.layout != TensorLayout::ND);
  bool has_non_default_pad = (tensor_view.pad != PadValue::null);

  // If all fields are at defaults, skip TensorView entirely
  if (first && !has_stride && !has_non_default_layout && !has_non_default_pad) return "";

  // When TensorView is non-trivial, always emit both stride and layout to satisfy
  // the C++ constructor signature TensorView(stride, layout, valid_shape=[], pad=null).
  // Omitting either required arg causes TypeError when Python eagerly evaluates
  // function parameter annotations during exec() in the text parser.
  maybe_comma();
  oss << "stride=[";
  for (size_t i = 0; i < tensor_view.stride.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << PrintExprForType(tensor_view.stride[i]);
  }
  oss << "]";

  maybe_comma();
  oss << "layout=" << prefix_ << ".TensorLayout." << TensorLayoutToString(tensor_view.layout);

  // pad — omit if null (default)
  if (has_non_default_pad) {
    maybe_comma();
    oss << "pad=" << prefix_ << ".PadValue.";
    switch (tensor_view.pad) {
      case PadValue::null:
        oss << "null";
        break;
      case PadValue::zero:
        oss << "zero";
        break;
      case PadValue::max:
        oss << "max";
        break;
      case PadValue::min:
        oss << "min";
        break;
    }
  }

  oss << ")";
  return oss.str();
}

// ================================
// Public API
// ================================
std::string PythonPrint(const IRNodePtr& node, const std::string& prefix, bool concise,
                        bool explicit_layout) {
  IRPythonPrinter printer(prefix, concise, explicit_layout);
  return printer.Print(node);
}

std::string PythonPrint(const TypePtr& type, const std::string& prefix, bool explicit_layout) {
  IRPythonPrinter printer(prefix, /*concise=*/false, explicit_layout);
  return printer.Print(type);
}

// ================================
// Format Callback
// ================================
namespace {
FormatCallback g_format_callback;  // set once at import time, read-only after
}  // namespace

void RegisterFormatCallback(FormatCallback callback) { g_format_callback = std::move(callback); }

std::string ApplyFormatCallback(const std::string& code) {
  if (!g_format_callback) {
    return code;
  }
  try {
    return g_format_callback(code);
  } catch (...) {
    // Best-effort: return raw output on any failure (e.g., Python exception in ruff)
    return code;
  }
}

}  // namespace ir
}  // namespace pypto
