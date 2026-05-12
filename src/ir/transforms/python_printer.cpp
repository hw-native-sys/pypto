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
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/pipe.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
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
  explicit IRPythonPrinter(std::string prefix = "pl", bool concise = false)
      : prefix_(std::move(prefix)), concise_(concise) {}
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
  void VisitExpr_(const ConstIntPtr& op) override;
  void VisitExpr_(const ConstFloatPtr& op) override;
  void VisitExpr_(const ConstBoolPtr& op) override;
  void VisitExpr_(const CallPtr& op) override;
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
  void VisitStmt_(const AutoInCoreScopeStmtPtr& op) override;
  void VisitStmt_(const ClusterScopeStmtPtr& op) override;
  void VisitStmt_(const HierarchyScopeStmtPtr& op) override;
  void VisitStmt_(const SpmdScopeStmtPtr& op) override;
  void VisitStmt_(const RuntimeScopeStmtPtr& op) override;
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
  std::string prefix_;                    // Prefix for type names (e.g., "pl" or "ir")
  bool concise_;                          // When true, omit intermediate type annotations
  ProgramPtr current_program_ = nullptr;  // Track when printing within Program (for self.method() calls)

  // Per-function rename map: Var pointer → unique printed name.
  // Built by BuildVarRenameMap() at the start of each function to handle SSA name shadowing.
  std::unordered_map<const Var*, std::string> var_rename_map_;

  // Vars defined in the current function body (for PrintMemRef formatting).
  std::unordered_set<const Var*> body_defined_vars_;

  // Program-level dyn var rename map: Var pointer → disambiguated printed name.
  // Built once by VisitProgram for dynamic dimension variables used in type annotations.
  // When two distinct Var* share the same name_hint_, they get unique suffixed names.
  std::unordered_map<const Var*, std::string> dyn_var_rename_map_;

  // Helper methods
  std::string GetIndent() const;
  void IncreaseIndent();
  void DecreaseIndent();

  // Return the printed name for a Var, using rename map if SSA name shadowing occurred.
  std::string GetVarName(const Var* var) const;

  // Build var_rename_map_ from a body stmt (and optional params).
  // Assigns unique suffixed names (e.g., "i", "i_1") when two distinct Vars share a name.
  // Called from VisitFunction (with params) and from Print() for bare Stmt roots
  // (no params) so that standalone stmt.as_python() also gets disambiguation.
  void BuildVarRenameMap(const std::vector<VarPtr>& params, const StmtPtr& body);
  void BuildVarRenameMap(const FunctionPtr& func) { BuildVarRenameMap(func->params_, func->body_); }

  // Print a statement block at current indent level.
  // SeqStmts is a transparent container - recursed into without extra indent.
  void PrintStmtBlock(const StmtPtr& stmt);

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
  TensorTypePtr tensor_type;
  std::string tensor_head;
  if (auto dt_tensor = As<DistributedTensorType>(type)) {
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

    if (tile_type->tile_view_.has_value()) {
      // PrintTileView elides every default field and returns "" when the explicit
      // view happens to match the implicit one — possible only on incoherent IR
      // (canonical IR stores that as nullopt and never reaches this branch).
      // Fall back to an empty TileView() literal so the output stays parseable
      // and TileTypeCoherence can flag the real bug.
      auto view_str =
          PrintTileView(tile_type->tile_view_.value(), tile_type->shape_, tile_type->memory_space_);
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

void IRPythonPrinter::VisitExpr_(const ConstIntPtr& op) {
  // DEFAULT_CONST_INT (= INT64) and INDEX both represent 64-bit integer constants
  // in the Python DSL, so they print as bare integers. Other integer types (INT8,
  // INT32, etc.) need explicit dtype annotation.
  if (op->dtype() == DataType::DEFAULT_CONST_INT || op->dtype() == DataType::INDEX) {
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

void IRPythonPrinter::VisitExpr_(const CallPtr& op) {
  INTERNAL_CHECK_SPAN(op->op_, op->span_) << "Call has null op";
  // Check if this is a GlobalVar call within a Program context

  if (auto gvar = As<GlobalVar>(op->op_)) {
    if (current_program_) {
      // This is a cross-function call - print as self.method_name()
      stream_ << "self." << gvar->name_ << "(";

      for (size_t i = 0; i < op->args_.size(); ++i) {
        if (i > 0) stream_ << ", ";
        VisitExpr(op->args_[i]);
      }

      // Surface manual_scope dep edges so they show up in IR dumps. The pass
      // ``LowerManualDepsToTaskId`` rewrites the user-facing
      // ``kAttrUserManualDepEdges`` (Tensor Vars) into ``kAttrManualDepEdges``
      // (TaskId Vars). Print whichever is present; both pre- and post-lower
      // forms round-trip via the parser-recognised ``deps=[...]`` kwarg.
      const std::vector<VarPtr>* deps_to_print = nullptr;
      for (const auto& [k, v] : op->attrs_) {
        if (k != kAttrManualDepEdges && k != kAttrUserManualDepEdges) continue;
        const auto* edges = std::any_cast<std::vector<VarPtr>>(&v);
        if (!edges || edges->empty()) continue;
        deps_to_print = edges;
        if (k == kAttrManualDepEdges) break;  // prefer resolved over user-supplied
      }
      // Zero-arg self.fn() needs no leading comma before the first kwarg, so
      // gate every kwarg separator on a shared flag rather than hard-coding
      // ", " — otherwise self.fn(, deps=[...]) breaks the round-trip.
      bool need_kwarg_comma = !op->args_.empty();
      if (deps_to_print) {
        stream_ << (need_kwarg_comma ? ", " : "") << "deps=[";
        for (size_t i = 0; i < deps_to_print->size(); ++i) {
          if (i > 0) stream_ << ", ";
          if ((*deps_to_print)[i]) {
            stream_ << GetVarName((*deps_to_print)[i].get());
          }
        }
        stream_ << "]";
        need_kwarg_comma = true;
      }

      // When ``attrs_["arg_directions"]`` is populated (post DeriveCallDirections),
      // surface the direction vector as a trailing ``attrs={"arg_directions": [...]}``
      // keyword so the parser can recover it on the round-trip. When empty
      // (legacy / pre-derive) keep the call bare for back-compatibility.
      // A non-empty vector with a mismatched size is invalid IR — fail loudly
      // instead of silently dropping the metadata.
      auto call_arg_directions = op->GetArgDirections();
      if (!call_arg_directions.empty()) {
        INTERNAL_CHECK_SPAN(call_arg_directions.size() == op->args_.size(), op->span_)
            << "Call arg_directions size (" << call_arg_directions.size() << ") must match args size ("
            << op->args_.size() << ")";
        stream_ << (need_kwarg_comma ? ", " : "") << "attrs={\"arg_directions\": [";
        for (size_t i = 0; i < call_arg_directions.size(); ++i) {
          if (i > 0) stream_ << ", ";
          stream_ << prefix_ << ".adir." << ArgDirectionToDslName(call_arg_directions[i]);
        }
        stream_ << "]}";
      }

      stream_ << ")";
      return;
    }
  }

  // Format operation name for printing
  // Operations are stored with internal names like "tensor.adds" or "tile.matmul"
  // and are printed in parseable format like "pl.tensor.adds"
  std::string op_name = op->op_->name_;

  // Normalize tensor.add with scalar rhs to tensor.adds (matches Python API dispatch)
  if (op_name == "tensor.add" && op->args_.size() == 2) {
    if (std::dynamic_pointer_cast<const ConstFloat>(op->args_[1]) ||
        std::dynamic_pointer_cast<const ConstInt>(op->args_[1])) {
      op_name = "tensor.adds";
    }
  }

  // Check if this is a registered operation (contains a dot)
  if (op_name.find('.') != std::string::npos) {
    // Print with pl. prefix
    stream_ << prefix_ << "." << op_name << "(";
  } else {
    // Not a registered operation, print as-is
    stream_ << op_name << "(";
  }

  // Special handling for tile.full / tensor.full: print as keyword args to match Python API
  // IR stores: args_=[shape, value_expr], kwargs_={"dtype": dtype}
  // Python API: full(shape, dtype, value) — print as full(shape, dtype=.., value=..)
  // because pl.FP32 as positional is rejected by the parser (standalone attribute access)
  if ((op->op_->name_ == "tile.full" || op->op_->name_ == "tensor.full") && op->args_.size() >= 2) {
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
    stream_ << ")";
    return;
  }

  // Print positional arguments
  for (size_t i = 0; i < op->args_.size(); ++i) {
    if (i > 0) stream_ << ", ";

    // Special handling for tile.alloc/tensor.alloc first argument (memory_space)
    if ((op->op_->name_ == "tile.alloc" || op->op_->name_ == "tensor.alloc") && i == 0) {
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
  for (const auto& [key, value] : op->kwargs_) {
    if (need_comma) {
      stream_ << ", ";
    }
    need_comma = true;
    stream_ << key << "=";

    // Print value based on type
    if (value.type() == typeid(int)) {
      int int_val = AnyCast<int>(value, "printing kwarg: " + key);
      // Print pipe kwargs as PipeType enum names for readability
      if (key == "set_pipe" || key == "wait_pipe") {
        stream_ << prefix_ << ".PipeType." << PipeTypeToString(static_cast<PipeType>(int_val));
      } else if (key == "mode") {
        stream_ << "'" << CastModeToString(int_val) << "'";
      } else {
        stream_ << int_val;
      }
    } else if (value.type() == typeid(bool)) {
      stream_ << (AnyCast<bool>(value, "printing kwarg: " + key) ? "True" : "False");
    } else if (value.type() == typeid(std::string)) {
      stream_ << "'" << AnyCast<std::string>(value, "printing kwarg: " + key) << "'";
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

void IRPythonPrinter::PrintBinaryOp(const BinaryExprPtr& op, const char* op_symbol) {
  PrintChild(op, op->left_, true);
  stream_ << " " << op_symbol << " ";
  PrintChild(op, op->right_, false);
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
      // Only elide for integer types that round-trip as the same value.
      // INDEX and INT64 are structurally equivalent (structural_equal.cpp).
      // Non-standard dtypes (e.g. INT32) are preserved to maintain fidelity.
      return ci->value_ == value && (ci->dtype() == DataType::DEFAULT_CONST_INT ||
                                     ci->dtype() == DataType::INDEX || ci->dtype() == DataType::INT64);
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
  // pl.unroll(), and SplitChunkedLoops preserves this: chunk-split unroll loops
  // always take the simple (no iter_args) path.
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

  // Add chunk kwargs
  if (op->chunk_config_.has_value()) {
    stream_ << ", chunk=";
    VisitExpr(op->chunk_config_->size);
    if (op->chunk_config_->policy != ChunkPolicy::Guarded) {
      // Emit lowercase policy string to match DSL/parser convention.
      stream_ << ", chunk_policy=\"";
      switch (op->chunk_config_->policy) {
        case ChunkPolicy::LeadingFull:
          stream_ << "leading_full";
          break;
        case ChunkPolicy::Guarded:
          stream_ << "guarded";
          break;
      }
      stream_ << "\"";
    }
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
    if (value.type() == typeid(LoopOrigin)) {
      stream_ << prefix_ << ".LoopOrigin." << LoopOriginToString(AnyCast<LoopOrigin>(value, key));
    } else if (value.type() == typeid(int)) {
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

void IRPythonPrinter::VisitStmt_(const HierarchyScopeStmtPtr& op) {
  // Print as: with pl.at(level=pl.Level.X, role=pl.Role.Y, [name_hint="..."]):
  stream_ << "with " << prefix_ << ".at(level=" << prefix_ << ".Level." << LevelToString(op->level_);
  if (op->role_.has_value()) {
    stream_ << ", role=" << prefix_ << ".Role." << RoleToString(*op->role_);
  }
  if (!op->name_hint_.empty()) {
    stream_ << ", name_hint=\"" << op->name_hint_ << "\"";
  }
  stream_ << "):\n";
  IncreaseIndent();
  PrintStmtBlock(op->body_);
  DecreaseIndent();
}

void IRPythonPrinter::VisitStmt_(const InCoreScopeStmtPtr& op) {
  stream_ << "with " << prefix_ << ".at(level=" << prefix_ << ".Level.CORE_GROUP";
  if (op->split_.has_value() && op->split_.value() != SplitMode::None) {
    stream_ << ", split=" << prefix_ << ".SplitMode." << SplitModeToPythonString(op->split_.value());
  }
  if (!op->name_hint_.empty()) {
    stream_ << ", name_hint=\"" << op->name_hint_ << "\"";
  }
  stream_ << "):\n";
  IncreaseIndent();
  PrintStmtBlock(op->body_);
  DecreaseIndent();
}

void IRPythonPrinter::VisitStmt_(const AutoInCoreScopeStmtPtr& op) {
  stream_ << "with " << prefix_ << ".at(level=" << prefix_ << ".Level.CORE_GROUP, optimization=";
  if (op->split_.has_value() && op->split_.value() != SplitMode::None) {
    stream_ << prefix_ << ".chunked_loop_optimizer(split=" << prefix_ << ".SplitMode."
            << SplitModeToPythonString(op->split_.value()) << ")";
  } else {
    stream_ << prefix_ << ".chunked_loop_optimizer";
  }
  if (!op->name_hint_.empty()) {
    stream_ << ", name_hint=\"" << op->name_hint_ << "\"";
  }
  stream_ << "):\n";
  IncreaseIndent();
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
  // Printing it back as a for-loop keeps round-trips stable (the
  // with-form parser enforces a single kernel call, so printing a
  // multi-statement InCore-wrapped body as `with pl.spmd():` would fail
  // to reparse).
  auto incore = As<InCoreScopeStmt>(op->body_);
  auto incore_seq = incore ? As<SeqStmts>(incore->body_) : nullptr;
  auto first_assign = incore_seq
                          ? (incore_seq->stmts_.empty() ? nullptr : As<AssignStmt>(incore_seq->stmts_[0]))
                          : (incore ? As<AssignStmt>(incore->body_) : nullptr);
  auto first_call = first_assign ? As<Call>(first_assign->value_) : nullptr;
  auto first_op = first_call ? As<Op>(first_call->op_) : nullptr;
  if (first_op && first_op->name_ == "tile.get_block_idx") {
    stream_ << "for " << GetVarName(first_assign->var_.get()) << " in " << prefix_ << ".spmd(";
    VisitExpr(op->core_num_);
    if (op->sync_start_) {
      stream_ << ", sync_start=True";
    }
    if (!op->name_hint_.empty()) {
      stream_ << ", name_hint=\"" << op->name_hint_ << "\"";
    }
    stream_ << "):\n";
    IncreaseIndent();
    // Emit the InCore body skipping the get_block_idx binding we just
    // materialized as the loop variable.
    if (incore_seq && incore_seq->stmts_.size() > 1) {
      for (size_t i = 1; i < incore_seq->stmts_.size(); ++i) {
        PrintStmtBlock(incore_seq->stmts_[i]);
        if (i + 1 < incore_seq->stmts_.size()) stream_ << "\n";
      }
    } else {
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
  stream_ << "):\n";
  IncreaseIndent();
  PrintStmtBlock(op->body_);
  DecreaseIndent();
}

void IRPythonPrinter::VisitStmt_(const SeqStmtsPtr& op) {
  for (size_t i = 0; i < op->stmts_.size(); ++i) {
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
  // Only the manual=true form has a DSL surface today.
  // Auto scope (manual=false) is reserved; printer falls back to a
  // transparent body emit so that round-trip never fails on legacy IR.
  if (op->manual_) {
    stream_ << "with " << prefix_ << ".manual_scope():\n";
    IncreaseIndent();
    PrintStmtBlock(op->body_);
    DecreaseIndent();
  } else {
    PrintStmtBlock(op->body_);
  }
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
  } else {
    PrintStmtBlock(body);
  }
}

std::string IRPythonPrinter::GetVarName(const Var* var) const {
  auto it = var_rename_map_.find(var);
  if (it != var_rename_map_.end()) return it->second;
  auto dyn_it = dyn_var_rename_map_.find(var);
  if (dyn_it != dyn_var_rename_map_.end()) return dyn_it->second;
  return var->name_hint_;
}

void IRPythonPrinter::BuildVarRenameMap(const std::vector<VarPtr>& params, const StmtPtr& body) {
  // Collect Var/IterArg pointers in DFS pre-order: params, then body defs,
  // then body uses. Defs precede uses so a pointer appearing as both keeps
  // its def-site canonical name; pointers appearing only as uses (e.g. a
  // dangling reference from a buggy transform, or a sibling-scope iter_arg
  // sharing a name_hint) still get their own suffix. (#1244)
  std::vector<const Var*> ordered_refs;
  ordered_refs.reserve(params.size());
  for (const auto& p : params) ordered_refs.push_back(p.get());
  body_defined_vars_.clear();
  if (body) {
    var_collectors::VarDefUseCollector body_collector;
    body_collector.VisitStmt(body);
    ordered_refs.insert(ordered_refs.end(), body_collector.var_defs_ordered.begin(),
                        body_collector.var_defs_ordered.end());
    ordered_refs.insert(ordered_refs.end(), body_collector.var_uses_ordered.begin(),
                        body_collector.var_uses_ordered.end());
    body_defined_vars_ = std::move(body_collector.var_defs);
  }
  // Drop program-level dynamic dim vars: they are already disambiguated
  // globally in dyn_var_rename_map_, and re-registering them here would
  // shadow that map in GetVarName's two-tier lookup.
  if (!dyn_var_rename_map_.empty()) {
    ordered_refs.erase(std::remove_if(ordered_refs.begin(), ordered_refs.end(),
                                      [this](const Var* v) { return dyn_var_rename_map_.count(v) > 0; }),
                       ordered_refs.end());
  }
  auto_name::BuildRenameMapForDefs(ordered_refs, var_rename_map_);
}

void IRPythonPrinter::VisitFunction(const FunctionPtr& func) {
  // Build rename map for this function to handle SSA name shadowing.
  BuildVarRenameMap(func);

  // Print decorator
  stream_ << GetIndent() << "@" << prefix_ << ".function";
  {
    bool has_type = func->func_type_ != FunctionType::Opaque;
    bool has_level = func->level_.has_value();
    bool has_role = func->role_.has_value();
    bool has_attrs = !func->attrs_.empty();
    auto print_func_attr_value = [&](const std::string& key, const std::any& value) {
      if (key == "split") {
        int split_value = AnyCast<int>(value, "func attr key: " + key);
        auto split_mode = static_cast<SplitMode>(split_value);
        stream_ << prefix_ << ".SplitMode." << SplitModeToPythonString(split_mode);
      } else if (value.type() == typeid(int)) {
        stream_ << AnyCast<int>(value, "func attr key: " + key);
      } else if (value.type() == typeid(double)) {
        stream_ << FormatFloatLiteral(AnyCast<double>(value, "func attr key: " + key));
      } else if (value.type() == typeid(float)) {
        stream_ << FormatFloatLiteral(static_cast<double>(AnyCast<float>(value, "func attr key: " + key)));
      } else if (value.type() == typeid(bool)) {
        stream_ << (AnyCast<bool>(value, "func attr key: " + key) ? "True" : "False");
      } else if (value.type() == typeid(std::string)) {
        stream_ << std::quoted(AnyCast<std::string>(value, "func attr key: " + key));
      } else if (value.type() == typeid(ExprPtr)) {
        VisitExpr(AnyCast<ExprPtr>(value, "func attr key: " + key));
      } else {
        INTERNAL_CHECK(false) << "Unsupported function attrs value type for key '" << key
                              << "': " << DemangleTypeName(value.type().name());
      }
    };
    if (has_type || has_level || has_role || has_attrs) {
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
      if (has_attrs) {
        if (!first) stream_ << ", ";
        stream_ << "attrs={";
        bool first_attr = true;
        for (const auto& [key, value] : func->attrs_) {
          if (!first_attr) stream_ << ", ";
          stream_ << std::quoted(key) << ": ";
          print_func_attr_value(key, value);
          first_attr = false;
        }
        stream_ << "}";
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

  stream_ << ":\n";

  // Print body - convert yield to return in function context
  IncreaseIndent();
  if (func->body_) {
    if (auto seq_stmts = As<SeqStmts>(func->body_)) {
      if (seq_stmts->stmts_.empty()) {
        stream_ << GetIndent() << "pass";
      } else {
        for (size_t i = 0; i < seq_stmts->stmts_.size(); ++i) {
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
  // expression context: loop bounds (ForStmt start/stop/step/chunk_size),
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

  // Filter out locally-defined vars and function params: they should not get
  // pl.dynamic() declarations — only truly free dimension variables should.
  dyn_var_ptrs.erase(std::remove_if(dyn_var_ptrs.begin(), dyn_var_ptrs.end(),
                                    [&defined_vars](const Var* v) { return defined_vars.count(v) > 0; }),
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
  // Print program header comment
  stream_ << "# pypto.program: " << (program->name_.empty() ? "Program" : program->name_) << "\n";

  // Print import statement based on prefix
  if (prefix_ == "pl") {
    stream_ << "import pypto.language as pl\n\n";
  } else {
    stream_ << "from pypto import language as " << prefix_ << "\n\n";
  }

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
}

std::string IRPythonPrinter::PrintExprForType(const ExprPtr& expr) {
  if (auto const_int = As<ConstInt>(expr)) {
    return std::to_string(const_int->value_);
  }
  if (auto var = As<Var>(expr)) {
    return GetVarName(var.get());
  }
  IRPythonPrinter temp_printer(prefix_);
  temp_printer.dyn_var_rename_map_ = dyn_var_rename_map_;
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
  oss << prefix_ << ".MemRef(";

  // Base Ptrs defined in the function body (by alloc statements) are printed as
  // bare variable references; everything else uses a string literal (forward
  // references in parameter annotations, standalone type printing, etc.).
  if (body_defined_vars_.count(memref.base_.get())) {
    oss << GetVarName(memref.base_.get());
  } else {
    oss << "\"" << GetVarName(memref.base_.get()) << "\"";
  }

  // Print byte offset using a temp printer to avoid corrupting the main stream.
  // The temp printer has its own stream_ but shares no rename maps — that's fine
  // because byte_offset expressions are ConstInt or arithmetic trees of loop vars
  // which print by name_hint_ and don't need SSA renaming.
  oss << ", ";
  IRPythonPrinter temp_printer(prefix_);
  oss << temp_printer.Print(memref.byte_offset_);

  // Print size
  oss << ", " << memref.size_ << ")";
  return oss.str();
}

std::string IRPythonPrinter::PrintTileView(const TileView& tile_view, const std::vector<ExprPtr>& tile_shape,
                                           const std::optional<MemorySpace>& memory_space) {
  // Caller already gated on has_value(); a present view is non-implicit by the
  // TileType canonical-encoding invariant, so always render the explicit form.
  std::ostringstream oss;
  oss << prefix_ << ".TileView(";

  bool first = true;
  auto maybe_comma = [&]() {
    if (!first) oss << ", ";
    first = false;
  };

  // Compute the implicit view so we can elide fields that match it.
  TileView implicit_view = tile_view_semantics::GetImplicitTileView(tile_shape, memory_space);

  // valid_shape — omit if it matches the parent tile's shape
  bool valid_shape_matches = tile_view_semantics::ShapeExprListsEquivalent(tile_view.valid_shape, tile_shape);
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
  if (tile_view.blayout != implicit_view.blayout) {
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
  if (tile_view.slayout != implicit_view.slayout) {
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
  if (tile_view.fractal != implicit_view.fractal) {
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
std::string PythonPrint(const IRNodePtr& node, const std::string& prefix, bool concise) {
  IRPythonPrinter printer(prefix, concise);
  return printer.Print(node);
}

std::string PythonPrint(const TypePtr& type, const std::string& prefix) {
  IRPythonPrinter printer(prefix);
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
