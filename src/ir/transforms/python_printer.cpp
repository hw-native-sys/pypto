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

#include <cmath>
#include <functional>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/printer.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

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
  explicit IRPythonPrinter(std::string prefix = "pl") : prefix_(std::move(prefix)) {}
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
  void VisitStmt_(const SeqStmtsPtr& op) override;
  void VisitStmt_(const OpStmtsPtr& op) override;
  void VisitStmt_(const EvalStmtPtr& op) override;
  void VisitStmt_(const StmtPtr& op) override;

  // Function and program visitors
  void VisitFunction(const FunctionPtr& func);
  void VisitProgram(const ProgramPtr& program);

 private:
  std::ostringstream stream_;
  int indent_level_ = 0;
  std::string prefix_;                    // Prefix for type names (e.g., "pl" or "ir")
  ProgramPtr current_program_ = nullptr;  // Track when printing within Program (for self.method() calls)

  // Helper methods
  std::string GetIndent() const;
  void IncreaseIndent();
  void DecreaseIndent();

  // Statement body visitor with SSA-style handling
  void VisitStmtBody(const StmtPtr& body, const std::vector<VarPtr>& return_vars = {});

  // Statement body visitor in program context (for self.method() call printing)
  void VisitStmtInProgramContext(const StmtPtr& stmt, const ProgramPtr& program);

  // Binary/unary operator helpers (reuse precedence logic)
  void PrintBinaryOp(const BinaryExprPtr& op, const char* op_symbol);
  void PrintFunctionBinaryOp(const BinaryExprPtr& op, const char* func_name);
  void PrintChild(const ExprPtr& parent, const ExprPtr& child, bool is_left);
  bool NeedsParens(const ExprPtr& parent, const ExprPtr& child, bool is_left);

  // MemRef and TileView printing helpers
  std::string PrintMemRef(const MemRef& memref);
  std::string PrintTileView(const TileView& tile_view);
};

// Helper function to format float literals with decimal point
std::string FormatFloatLiteral(double value) {
  // Check if the value is an integer (no fractional part)
  if (value == std::floor(value)) {
    // For integer values, format as X.0
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << value;
    return oss.str();
  } else {
    // For non-integer values, use default formatting with enough precision
    std::ostringstream oss;
    oss << value;
    return oss.str();
  }
}

// Helper function to convert DataType to Python IR string
std::string DataTypeToPythonString(DataType dtype, const std::string& prefix) {
  std::string p = prefix + ".";
  if (dtype == DataType::INT4) return p + "INT4";
  if (dtype == DataType::INT8) return p + "INT8";
  if (dtype == DataType::INT16) return p + "INT16";
  if (dtype == DataType::INT32) return p + "INT32";
  if (dtype == DataType::INT64) return p + "INT64";
  if (dtype == DataType::UINT4) return p + "UINT4";
  if (dtype == DataType::UINT8) return p + "UINT8";
  if (dtype == DataType::UINT16) return p + "UINT16";
  if (dtype == DataType::UINT32) return p + "UINT32";
  if (dtype == DataType::UINT64) return p + "UINT64";
  if (dtype == DataType::FP4) return p + "FP4";
  if (dtype == DataType::FP8E4M3FN) return p + "FP8E4M3FN";
  if (dtype == DataType::FP8E5M2) return p + "FP8E5M2";
  if (dtype == DataType::FP16) return p + "FP16";
  if (dtype == DataType::FP32) return p + "FP32";
  if (dtype == DataType::BF16) return p + "BFLOAT16";
  if (dtype == DataType::HF4) return p + "HF4";
  if (dtype == DataType::HF8) return p + "HF8";
  if (dtype == DataType::BOOL) return p + "BOOL";
  return p + "UnknownType";
}

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
    return prefix_ + ".Scalar[" + DataTypeToPythonString(scalar_type->dtype_, prefix_) + "]";
  }

  if (auto tensor_type = As<TensorType>(type)) {
    std::ostringstream oss;
    // Subscript-style: pl.Tensor[[shape], dtype]
    oss << prefix_ << ".Tensor[[";
    for (size_t i = 0; i < tensor_type->shape_.size(); ++i) {
      if (i > 0) oss << ", ";
      // Use a temporary printer with same prefix for dimension expressions
      IRPythonPrinter temp_printer(prefix_);
      oss << temp_printer.Print(tensor_type->shape_[i]);
    }
    oss << "], " << DataTypeToPythonString(tensor_type->dtype_, prefix_);

    // Add optional memref parameter if present
    if (tensor_type->memref_.has_value()) {
      oss << ", memref=" << PrintMemRef(*tensor_type->memref_.value());
    }
    oss << "]";
    return oss.str();
  }

  if (auto tile_type = As<TileType>(type)) {
    std::ostringstream oss;
    // Subscript-style: pl.Tile[[shape], dtype]
    oss << prefix_ << ".Tile[[";
    for (size_t i = 0; i < tile_type->shape_.size(); ++i) {
      if (i > 0) oss << ", ";
      // Use a temporary printer with same prefix for dimension expressions
      IRPythonPrinter temp_printer(prefix_);
      oss << temp_printer.Print(tile_type->shape_[i]);
    }
    oss << "], " << DataTypeToPythonString(tile_type->dtype_, prefix_);

    // Add optional memref parameter if present
    if (tile_type->memref_.has_value()) {
      oss << ", memref=" << tile_type->memref_.value()->name_;
    }

    // Add optional tile_view parameter if present
    if (tile_type->tile_view_.has_value()) {
      oss << ", tile_view=" << PrintTileView(tile_type->tile_view_.value());
    }
    oss << "]";
    return oss.str();
  }

  if (auto tuple_type = As<TupleType>(type)) {
    std::ostringstream oss;
    oss << prefix_ << ".Tuple([";
    for (size_t i = 0; i < tuple_type->types_.size(); ++i) {
      if (i > 0) oss << ", ";
      oss << Print(tuple_type->types_[i]);
    }
    oss << "])";
    return oss.str();
  }

  if (auto memref_type = As<MemRefType>(type)) {
    return prefix_ + ".MemRefType";
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
void IRPythonPrinter::VisitExpr_(const VarPtr& op) { stream_ << op->name_; }

void IRPythonPrinter::VisitExpr_(const IterArgPtr& op) { stream_ << op->name_; }

void IRPythonPrinter::VisitExpr_(const MemRefPtr& op) { stream_ << op->name_; }

void IRPythonPrinter::VisitExpr_(const ConstIntPtr& op) { stream_ << op->value_; }

void IRPythonPrinter::VisitExpr_(const ConstFloatPtr& op) { stream_ << FormatFloatLiteral(op->value_); }

void IRPythonPrinter::VisitExpr_(const ConstBoolPtr& op) { stream_ << (op->value_ ? "True" : "False"); }

void IRPythonPrinter::VisitExpr_(const CallPtr& op) {
  INTERNAL_CHECK(op->op_) << "Call has null op";
  // Check if this is a GlobalVar call within a Program context

  if (auto gvar = As<GlobalVar>(op->op_)) {
    if (current_program_) {
      // This is a cross-function call - print as self.method_name()
      stream_ << "self." << gvar->name_ << "(";

      // Print positional arguments
      for (size_t i = 0; i < op->args_.size(); ++i) {
        if (i > 0) stream_ << ", ";
        VisitExpr(op->args_[i]);
      }

      stream_ << ")";
      return;
    }
  }

  // Format operation name for printing
  // Operations are stored with internal names like "tensor.add_scalar"
  // but need to be printed in parseable format like "pl.tensor.add"
  std::string op_name = op->op_->name_;

  // Check if this is a registered operation (contains a dot)
  if (op_name.find('.') != std::string::npos) {
    // This is an operation like "tensor.add_scalar" or "block.matmul"
    // Convert internal operation names to high-level API format
    // Remove "_scalar" suffix if present (e.g., "tensor.add_scalar" -> "tensor.add")
    size_t scalar_pos = op_name.find("_scalar");
    if (scalar_pos != std::string::npos) {
      op_name = op_name.substr(0, scalar_pos);
    }

    // Print with pl. prefix
    stream_ << prefix_ << "." << op_name << "(";
  } else {
    // Not a registered operation, print as-is
    stream_ << op_name << "(";
  }

  // Print positional arguments
  for (size_t i = 0; i < op->args_.size(); ++i) {
    if (i > 0) stream_ << ", ";

    // Special handling for block.alloc's first argument (memory_space)
    if (op->op_->name_ == "block.alloc" && i == 0) {
      // Try to extract the integer value and convert it to MemorySpace enum
      if (auto const_int = std::dynamic_pointer_cast<const ConstInt>(op->args_[i])) {
        int space_value = static_cast<int>(const_int->value_);
        stream_ << prefix_ << ".MemorySpace." << MemorySpaceToString(static_cast<MemorySpace>(space_value));
      } else {
        VisitExpr(op->args_[i]);
      }
    } else {
      VisitExpr(op->args_[i]);
    }
  }

  // Print kwargs as keyword arguments
  for (const auto& [key, value] : op->kwargs_) {
    stream_ << ", " << key << "=";

    // Print value based on type
    if (value.type() == typeid(int)) {
      stream_ << AnyCast<int>(value, "printing kwarg: " + key);
    } else if (value.type() == typeid(bool)) {
      stream_ << (AnyCast<bool>(value, "printing kwarg: " + key) ? "True" : "False");
    } else if (value.type() == typeid(std::string)) {
      stream_ << "'" << AnyCast<std::string>(value, "printing kwarg: " + key) << "'";
    } else if (value.type() == typeid(double)) {
      stream_ << FormatFloatLiteral(AnyCast<double>(value, "printing kwarg: " + key));
    } else if (value.type() == typeid(float)) {
      stream_ << FormatFloatLiteral(static_cast<double>(AnyCast<float>(value, "printing kwarg: " + key)));
    } else if (value.type() == typeid(DataType)) {
      stream_ << DataTypeToPythonString(AnyCast<DataType>(value, "printing kwarg: " + key), prefix_);
    } else {
      throw TypeError("Invalid kwarg type for key: " + key +
                      ", expected int, bool, std::string, double, float, or DataType, but got " +
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
  stream_ << func_name << "(";
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
  INTERNAL_CHECK(scalar_type) << "Cast has non-scalar type";
  stream_ << prefix_ << ".cast(";
  VisitExpr(op->operand_);
  stream_ << ", " << DataTypeToPythonString(scalar_type->dtype_, prefix_) << ")";
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
  // First print variable name
  VisitExpr(op->var_);
  stream_ << ": " << Print(op->var_->GetType()) << " = ";
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
  stream_ << "for " << op->loop_var_->name_;

  // Determine range keyword based on kind
  bool is_parallel = (op->kind_ == ForKind::Parallel);

  // If we have iter_args, add tuple unpacking without type annotations
  if (!op->iter_args_.empty()) {
    stream_ << ", (";
    for (size_t i = 0; i < op->iter_args_.size(); ++i) {
      if (i > 0) stream_ << ", ";
      stream_ << op->iter_args_[i]->name_;
    }
    // Add trailing comma for single-element tuples to distinguish from parenthesized expression
    if (op->iter_args_.size() == 1) {
      stream_ << ",";
    }
    stream_ << ")";
  }
  stream_ << " in " << prefix_ << (is_parallel ? ".parallel(" : ".range(");

  VisitExpr(op->start_);
  stream_ << ", ";
  VisitExpr(op->stop_);
  stream_ << ", ";
  VisitExpr(op->step_);

  // Add init_values for iter_args
  if (!op->iter_args_.empty()) {
    stream_ << ", init_values=[";
    for (size_t i = 0; i < op->iter_args_.size(); ++i) {
      if (i > 0) stream_ << ", ";
      VisitExpr(op->iter_args_[i]->initValue_);
    }
    stream_ << "]";
  }

  stream_ << "):\n";

  IncreaseIndent();
  VisitStmtBody(op->body_, op->return_vars_);
  DecreaseIndent();
}

void IRPythonPrinter::VisitStmt_(const SeqStmtsPtr& op) {
  for (size_t i = 0; i < op->stmts_.size(); ++i) {
    stream_ << GetIndent();
    VisitStmt(op->stmts_[i]);
    if (i < op->stmts_.size() - 1) {
      stream_ << "\n";
    }
  }
}

void IRPythonPrinter::VisitStmt_(const OpStmtsPtr& op) {
  for (size_t i = 0; i < op->stmts_.size(); ++i) {
    stream_ << GetIndent();
    VisitStmt(op->stmts_[i]);
    if (i < op->stmts_.size() - 1) {
      stream_ << "\n";
    }
  }
}

void IRPythonPrinter::VisitStmt_(const EvalStmtPtr& op) {
  // Print expression statement: expr
  VisitExpr(op->expr_);
}

void IRPythonPrinter::VisitStmt_(const StmtPtr& op) { stream_ << op->TypeName(); }

void IRPythonPrinter::VisitStmtBody(const StmtPtr& body, const std::vector<VarPtr>& return_vars) {
  // Helper to visit statement body and wrap YieldStmt with assignment if needed
  if (auto yield_stmt = As<YieldStmt>(body)) {
    // If parent has return_vars, wrap yield as assignment (no inline type annotations)
    if (!yield_stmt->value_.empty() && !return_vars.empty()) {
      stream_ << GetIndent();
      // Print variable names without type annotations (not valid in tuple unpacking)
      for (size_t i = 0; i < return_vars.size(); ++i) {
        if (i > 0) stream_ << ", ";
        stream_ << return_vars[i]->name_;
      }
      stream_ << " = " << prefix_ << ".yield_(";
      for (size_t i = 0; i < yield_stmt->value_.size(); ++i) {
        if (i > 0) stream_ << ", ";
        VisitExpr(yield_stmt->value_[i]);
      }
      stream_ << ")";
    } else {
      stream_ << GetIndent();
      VisitStmt(yield_stmt);
    }
  } else if (auto seq_stmts = As<SeqStmts>(body)) {
    // Process each statement in sequence
    for (size_t i = 0; i < seq_stmts->stmts_.size(); ++i) {
      auto stmt = seq_stmts->stmts_[i];

      // Check if this is the last statement and it's a YieldStmt
      bool is_last = (i == seq_stmts->stmts_.size() - 1);
      if (auto yield_stmt = As<YieldStmt>(stmt)) {
        if (is_last && !yield_stmt->value_.empty() && !return_vars.empty()) {
          // Wrap as assignment without inline type annotations
          stream_ << GetIndent();
          for (size_t j = 0; j < return_vars.size(); ++j) {
            if (j > 0) stream_ << ", ";
            stream_ << return_vars[j]->name_;
          }
          stream_ << " = " << prefix_ << ".yield_(";
          for (size_t j = 0; j < yield_stmt->value_.size(); ++j) {
            if (j > 0) stream_ << ", ";
            VisitExpr(yield_stmt->value_[j]);
          }
          stream_ << ")";
        } else {
          stream_ << GetIndent();
          VisitStmt(stmt);
        }
      } else {
        stream_ << GetIndent();
        VisitStmt(stmt);
      }

      if (i < seq_stmts->stmts_.size() - 1) {
        stream_ << "\n";
      }
    }
  } else {
    stream_ << GetIndent();
    VisitStmt(body);
  }
}

void IRPythonPrinter::VisitFunction(const FunctionPtr& func) {
  // Print decorator with type parameter if not opaque
  stream_ << "@" << prefix_ << ".function";
  if (func->func_type_ != FunctionType::Opaque) {
    stream_ << "(type=" << prefix_ << ".FunctionType." << FunctionTypeToString(func->func_type_) << ")";
  }
  stream_ << "\n";
  stream_ << "def " << func->name_ << "(";

  // Print parameters with type annotations
  for (size_t i = 0; i < func->params_.size(); ++i) {
    if (i > 0) stream_ << ", ";
    stream_ << func->params_[i]->name_ << ": " << Print(func->params_[i]->GetType());
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
      for (size_t i = 0; i < seq_stmts->stmts_.size(); ++i) {
        stream_ << GetIndent();
        // Convert yield to return in function context
        if (auto yield_stmt = As<YieldStmt>(seq_stmts->stmts_[i])) {
          stream_ << "return";
          if (!yield_stmt->value_.empty()) {
            stream_ << " ";
            for (size_t j = 0; j < yield_stmt->value_.size(); ++j) {
              if (j > 0) stream_ << ", ";
              VisitExpr(yield_stmt->value_[j]);
            }
          }
        } else {
          VisitStmt(seq_stmts->stmts_[i]);
        }
        if (i < seq_stmts->stmts_.size() - 1) {
          stream_ << "\n";
        }
      }
    } else if (auto yield_stmt = As<YieldStmt>(func->body_)) {
      stream_ << GetIndent() << "return";
      if (!yield_stmt->value_.empty()) {
        stream_ << " ";
        for (size_t i = 0; i < yield_stmt->value_.size(); ++i) {
          if (i > 0) stream_ << ", ";
          VisitExpr(yield_stmt->value_[i]);
        }
      }
    } else {
      stream_ << GetIndent();
      VisitStmt(func->body_);
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
    INTERNAL_CHECK(op->op_) << "Call has null op";
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

void IRPythonPrinter::VisitProgram(const ProgramPtr& program) {
  // Print program header comment
  stream_ << "# pypto.program: " << (program->name_.empty() ? "Program" : program->name_) << "\n";

  // Print import statement based on prefix
  if (prefix_ == "pl") {
    stream_ << "import pypto.language as pl\n\n";
  } else {
    stream_ << "from pypto import language as " << prefix_ << "\n\n";
  }

  // Print as @pl.program class with @pl.function methods
  stream_ << "@" << prefix_ << ".program\n";
  stream_ << "class " << (program->name_.empty() ? "Program" : program->name_) << ":\n";

  IncreaseIndent();

  // Sort functions in dependency order (called functions before callers)
  auto sorted_functions = TopologicalSortFunctions(program->functions_);

  // Print each function as a method
  bool first = true;
  for (const auto& [gvar, func] : sorted_functions) {
    if (!first) {
      stream_ << "\n";  // Blank line between functions
    }
    first = false;

    stream_ << GetIndent() << "@" << prefix_ << ".function\n";
    stream_ << GetIndent() << "def " << func->name_ << "(";

    // IMPORTANT: Add 'self' as first parameter for methods in @pl.program class
    stream_ << "self";

    // Print remaining parameters with type annotations
    for (const auto& param : func->params_) {
      stream_ << ", ";  // Always add comma since self comes first
      stream_ << param->name_ << ": " << Print(param->GetType());
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

    // Print body - Call expressions with GlobalVar should print as self.method_name()
    IncreaseIndent();
    VisitStmtInProgramContext(func->body_, program);
    DecreaseIndent();
  }

  DecreaseIndent();
}

// Helper to visit statements in program context (for self.method() printing)
void IRPythonPrinter::VisitStmtInProgramContext(const StmtPtr& stmt, const ProgramPtr& program) {
  // Save current program context
  auto prev_program = current_program_;
  current_program_ = program;

  // Visit statement (will affect how Call expressions are printed)
  if (stmt) {
    if (auto seq_stmts = As<SeqStmts>(stmt)) {
      for (size_t i = 0; i < seq_stmts->stmts_.size(); ++i) {
        stream_ << GetIndent();
        // Convert yield to return in function context
        if (auto yield_stmt = As<YieldStmt>(seq_stmts->stmts_[i])) {
          stream_ << "return";
          if (!yield_stmt->value_.empty()) {
            stream_ << " ";
            for (size_t j = 0; j < yield_stmt->value_.size(); ++j) {
              if (j > 0) stream_ << ", ";
              VisitExpr(yield_stmt->value_[j]);
            }
          }
        } else {
          VisitStmt(seq_stmts->stmts_[i]);
        }
        if (i < seq_stmts->stmts_.size() - 1) {
          stream_ << "\n";
        }
      }
    } else if (auto yield_stmt = As<YieldStmt>(stmt)) {
      stream_ << GetIndent() << "return";
      if (!yield_stmt->value_.empty()) {
        stream_ << " ";
        for (size_t i = 0; i < yield_stmt->value_.size(); ++i) {
          if (i > 0) stream_ << ", ";
          VisitExpr(yield_stmt->value_[i]);
        }
      }
    } else {
      stream_ << GetIndent();
      VisitStmt(stmt);
    }
  }

  // Restore previous context
  current_program_ = prev_program;
}

// Helper methods for MemRef and TileView printing
std::string IRPythonPrinter::PrintMemRef(const MemRef& memref) {
  std::ostringstream oss;
  oss << prefix_ << ".MemRef(" << prefix_ << ".MemorySpace." << MemorySpaceToString(memref.memory_space_)
      << ", ";

  // Print address expression
  IRPythonPrinter temp_printer(prefix_);
  oss << temp_printer.Print(memref.addr_);

  // Print size and id
  oss << ", " << memref.size_ << ", " << memref.id_ << ")";
  return oss.str();
}

std::string IRPythonPrinter::PrintTileView(const TileView& tile_view) {
  std::ostringstream oss;
  oss << prefix_ << ".TileView(valid_shape=[";

  // Print valid_shape
  for (size_t i = 0; i < tile_view.valid_shape.size(); ++i) {
    if (i > 0) oss << ", ";
    IRPythonPrinter temp_printer(prefix_);
    oss << temp_printer.Print(tile_view.valid_shape[i]);
  }

  oss << "], stride=[";

  // Print stride
  for (size_t i = 0; i < tile_view.stride.size(); ++i) {
    if (i > 0) oss << ", ";
    IRPythonPrinter temp_printer(prefix_);
    oss << temp_printer.Print(tile_view.stride[i]);
  }

  oss << "], start_offset=";

  // Print start_offset
  IRPythonPrinter temp_printer(prefix_);
  oss << temp_printer.Print(tile_view.start_offset);

  oss << ")";
  return oss.str();
}

// ================================
// Public API
// ================================
std::string PythonPrint(const IRNodePtr& node, const std::string& prefix) {
  IRPythonPrinter printer(prefix);
  return printer.Print(node);
}

std::string PythonPrint(const TypePtr& type, const std::string& prefix) {
  IRPythonPrinter printer(prefix);
  return printer.Print(type);
}

}  // namespace ir
}  // namespace pypto
