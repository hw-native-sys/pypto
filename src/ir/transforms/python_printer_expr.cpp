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
#include <cstring>
#include <iomanip>
#include <ios>
#include <limits>
#include <locale>
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
#include "pypto/ir/cast_saturation.h"
#include "pypto/ir/comm.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/phase.h"
#include "pypto/ir/pipe.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/transforms/printer.h"
#include "pypto/ir/type.h"
#include "src/ir/transforms/python_printer.h"

namespace pypto {
namespace ir {

namespace {

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

/// Spell a lowercase IR attr value as its Python enum member name.
/// ``"soft"`` -> ``"SOFT"``, for the enums whose members are named after the
/// attr string they lower to.
std::string IRPythonPrinter::AttrValueToEnumMember(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return value;
}

// Precedence mapping for each expression type
Precedence GetPrecedence(const ExprPtr& expr) {
  // Using a static map is more efficient and maintainable than a long chain of dynamic_casts.
  static const std::unordered_map<std::type_index, Precedence> kPrecedenceMap = {
      // Logical operators≥
      {std::type_index(typeid(Or)), Precedence::kOr},
      {std::type_index(typeid(Xor)), Precedence::kAtom},
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
// already parses as a Python float). ConstFloat's visitor handles non-finite
// values separately using an exact binary64 dump spelling.
std::string IRPythonPrinter::FormatFloatLiteral(double value) {
  std::ostringstream os;
  os.imbue(std::locale::classic());
  os << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
  std::string text = os.str();
  if (std::isfinite(value) && text.find_first_of(".eE") == std::string::npos) {
    text += ".0";
  }
  return text;
}

// ConstFloat values may also be emitted by special call printers such as full.
std::string IRPythonPrinter::FormatFloatValue(double value, const std::string& prefix) {
  if (std::isfinite(value)) return FormatFloatLiteral(value);
  uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  std::ostringstream encoded;
  encoded << prefix << "._dump.float64(\"" << std::hex << std::setfill('0') << std::setw(16) << bits << "\")";
  return encoded.str();
}

// Expression visitors - reuse precedence logic from base printer
void IRPythonPrinter::VisitExpr_(const VarPtr& op) { stream_ << GetVarName(op.get()); }

void IRPythonPrinter::VisitExpr_(const IterArgPtr& op) { stream_ << GetVarName(op.get()); }

void IRPythonPrinter::VisitExpr_(const MemRefPtr& op) { stream_ << op->name_hint_; }

void IRPythonPrinter::VisitExpr_(const WindowBufferPtr& op) {
  stream_ << prefix_ << "._dump.window_ref(" << std::quoted(GetWindowBufferName(op)) << ")";
}

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
    stream_ << prefix_ << ".const(" << FormatFloatValue(op->value_, prefix_) << ", " << prefix_ << "."
            << DataTypeToString(op->dtype()) << ")";
  } else {
    stream_ << FormatFloatValue(op->value_, prefix_);
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
    if (auto integer = As<ConstInt>(expr)) {
      // A bare literal denotes a Python integer attribute, not an IR Expr.
      stream_ << prefix_ << ".const(" << integer->value_ << ", " << prefix_ << "."
              << DataTypeToString(integer->dtype()) << ")";
    } else {
      VisitExpr(expr);
    }
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

bool IRPythonPrinter::HasStorageType(const TypePtr& type) {
  if (!type) return false;
  if (auto found = storage_type_cache_.find(type.get()); found != storage_type_cache_.end()) {
    return found->second;
  }
  bool result = false;
  if (auto shaped = As<ShapedType>(type)) {
    result = shaped->memref_.has_value();
  } else if (auto tuple = As<TupleType>(type)) {
    result = std::any_of(tuple->types_.begin(), tuple->types_.end(),
                         [&](const TypePtr& element) { return HasStorageType(element); });
  }
  storage_type_cache_.emplace(type.get(), result);
  return result;
}

void IRPythonPrinter::VisitExpr_(const CallPtr& op) {
  INTERNAL_CHECK_SPAN(op->op_, op->span_) << "Call has null op";
  auto distributed_type = As<DistributedTensorType>(op->GetType());
  // Type deduction may propagate an operand's allocation into the result,
  // while the recorded call retains an allocation-free value type. Preserve
  // the recorded type using the existing lossless dump form.
  const bool storage_call = std::exchange(force_recorded_call_, false) || HasStorageType(op->GetType()) ||
                            std::any_of(op->args_.begin(), op->args_.end(),
                                        [&](const ExprPtr& arg) { return HasStorageType(arg->GetType()); });
  // Simplify can fold a dynamic flag operand to ConstInt without changing the
  // operation to its static form. The public API treats constants as static
  // event_id attributes, so use the lossless dump to preserve this operand.
  const bool folded_flag = (IsOp(op, "system.set_flag") || IsOp(op, "system.wait_flag")) &&
                           op->args_.size() == 1 && As<ConstInt>(op->args_[0]);
  // Inlining can retain dimension reads in orchestration expressions, including
  // nested arithmetic. Its authoring parser folds those reads, so preserve the
  // recorded Call. Type expressions use a separate printer and keep their DSL
  // spelling, as do Inline/InCore functions whose parser does not fold reads.
  const bool dimension_call = orchestration_function_ && IsOp(op, "tensor.dim");
  if (!storage_call && !folded_flag && !dimension_call &&
      (!distributed_type || (comm_domain_depth_ == 0 && !distributed_type->window_buffer_.has_value()))) {
    PrintCall(op);
    return;
  }
  // Preserve the recorded result type and operand layout directly. Re-running
  // an operator's type deducer can change metadata or reject an intermediate IR
  // that a downstream verifier is responsible for diagnosing.
  stream_ << prefix_ << "._dump.call(";
  if (auto callee = As<GlobalVar>(op->op_)) {
    stream_ << "self." << callee->name_;
  } else {
    stream_ << std::quoted(op->op_->name_);
  }
  stream_ << ", [";
  for (size_t i = 0; i < op->args_.size(); ++i) {
    if (i > 0) stream_ << ", ";
    VisitExpr(op->args_[i]);
  }
  stream_ << "], " << Print(op->GetType());
  auto print_metadata = [&](const char* name, const auto& fields) {
    if (fields.empty()) return;
    stream_ << ", " << name << "={";
    for (size_t i = 0; i < fields.size(); ++i) {
      if (i > 0) stream_ << ", ";
      stream_ << std::quoted(fields[i].first) << ": ";
      PrintAttrValue(fields[i].second, op->span_);
    }
    stream_ << "}";
  };
  print_metadata("kwargs", op->kwargs_);
  print_metadata("attrs", op->attrs_);
  stream_ << ")";
}

void IRPythonPrinter::PrintCall(const CallPtr& op) {
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
  // Keep the existing barrier IR while presenting its parameterized public
  // API. The native ops have no operands or kwargs; retain the legacy call
  // spelling for noncanonical calls rather than silently dropping metadata.
  const char* barrier_pipe = nullptr;
  if (op->args_.empty() && op->kwargs_.empty()) {
    if (IsOp(op, "system.bar_v")) {
      barrier_pipe = "V";
    } else if (IsOp(op, "system.bar_m")) {
      barrier_pipe = "M";
    } else if (IsOp(op, "system.bar_all")) {
      barrier_pipe = "ALL";
    }
  }
  if (barrier_pipe) op_name = "system.barrier";
  if (IsOp(op, "pld.tensor.alloc_window_buffer")) {
    // SSA renames the bound Var without changing the allocation's logical name.
    op_name = "_dump.alloc_window_buffer";
  }
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

  if (barrier_pipe) {
    stream_ << "pipe=" << prefix_ << ".PipeType." << barrier_pipe;
    print_serialized_attrs(/*need_comma=*/true);
    stream_ << ")";
    return;
  }

  // Flag IR keeps the event as either an attribute or the sole dynamic
  // operand. Print the public src/dst/event order explicitly; generic Call
  // printing would put the dynamic operand before src_pipe and dst_pipe.
  if (IsOp(op, "system.set_flag") || IsOp(op, "system.wait_flag")) {
    const auto src = static_cast<PipeType>(op->GetKwarg<int>("src_pipe"));
    const auto dst = static_cast<PipeType>(op->GetKwarg<int>("dst_pipe"));
    stream_ << "src_pipe=" << prefix_ << ".PipeType." << PipeTypeToString(src) << ", dst_pipe=" << prefix_
            << ".PipeType." << PipeTypeToString(dst) << ", event_id=";
    if (op->args_.empty()) {
      stream_ << op->GetKwarg<int>("event_id");
    } else {
      INTERNAL_CHECK_SPAN(op->args_.size() == 1, op->span_)
          << "Internal error: flag has multiple event operands";
      VisitExpr(op->args_[0]);
    }
    if (op->HasKwarg("core_type")) {
      const auto core = op->GetKwarg<std::string>("core_type");
      const char* member = CoreTypeToKernelMember(core, /*is_syncall=*/false);
      INTERNAL_CHECK_SPAN(member, op->span_) << "Internal error: flag has invalid core_type " << core;
      stream_ << ", core_type=" << prefix_ << ".KernelType." << member;
    }
    print_serialized_attrs(true);
    stream_ << ")";
    return;
  }

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
    // tile.full preserves the fill's own dtype, which may differ from the tile dtype.
    // Even FP32 needs explicit typing here: a bare keyword literal is retyped by full.
    // tensor.full still requires a bare numeric value at its Python API boundary.
    const auto& val_expr = op->args_[1];
    const auto cf = As<ConstFloat>(val_expr);
    const auto ci = As<ConstInt>(val_expr);
    const bool typed_fill = IsOp(op, "tile.full") && (cf || ci);
    if (typed_fill) stream_ << prefix_ << ".const(";
    if (cf) {
      stream_ << FormatFloatValue(cf->value_, prefix_);
    } else if (ci) {
      stream_ << ci->value_;
    } else {
      VisitExpr(val_expr);
    }
    if (typed_fill) {
      stream_ << ", " << prefix_ << "." << DataTypeToString(cf ? cf->dtype() : ci->dtype()) << ")";
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
  // pld.tensor.all_to_all_v carries its requested block limit as a keyword-only
  // 6th operand; printing it positionally would re-parse as too many arguments.
  const bool a2av_kw_core_num = IsOp(op, "pld.tensor.all_to_all_v") && op->args_.size() == 6;
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
    if (a2av_kw_core_num && i == 5) continue;
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
  if (a2av_kw_core_num) {
    stream_ << ", core_num=";
    VisitExpr(op->args_[5]);
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
      } else if (key == "saturation_mode") {
        // Print the DSL spelling ('off'/'on') rather than the raw code, the
        // same way `mode` is restored, so a printed cast reparses as the call
        // the author would have written.
        stream_ << "'" << SaturationModeToName(int_val) << "'";
      } else if (key == "atomic") {
        // Stored as int (the DSL casts AtomicType -> int before stashing on
        // kwargs_; nb::isinstance<AtomicType> in bindings does the same). The
        // public DSL signature is `atomic: AtomicType`, so restore the enum
        // form on print to keep the output type-correct for static checkers
        // and round-trippable through the parser (pl.AtomicType is exposed).
        stream_ << prefix_ << ".AtomicType." << AtomicTypeToString(static_cast<AtomicType>(int_val));
      } else if (key == "acc_phase") {
        stream_ << prefix_ << ".AccPhase." << AccPhaseToString(static_cast<AccPhase>(int_val));
      } else if (key == "st_phase") {
        stream_ << prefix_ << ".STPhase." << STPhaseToString(static_cast<STPhase>(int_val));
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
  const bool recorded = HasStorageType(op->GetType()) ||
                        std::any_of(op->args_.begin(), op->args_.end(),
                                    [&](const ExprPtr& arg) { return HasStorageType(arg->GetType()); });
  if (recorded) stream_ << prefix_ << "._dump.submit(";
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
  if (recorded) stream_ << ", " << Print(op->GetType()) << ")";
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
void IRPythonPrinter::VisitExpr_(const XorPtr& op) {
  stream_ << prefix_ << "._dump.logical_xor(";
  VisitExpr(op->left_);
  stream_ << ", ";
  VisitExpr(op->right_);
  stream_ << ")";
}

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

}  // namespace ir
}  // namespace pypto
