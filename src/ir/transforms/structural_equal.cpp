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
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
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
#include "pypto/ir/memref.h"
#include "pypto/ir/program.h"
#include "pypto/ir/reflection/field_visitor.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/printer.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

/**
 * @brief Unified structural equality checker for IR nodes
 *
 * Template parameter controls behavior on mismatch:
 * - AssertMode=false: Returns false (for structural_equal)
 * - AssertMode=true: Throws ValueError with detailed error message (for assert_structural_equal)
 *
 * This class is not part of the public API - use structural_equal() or assert_structural_equal().
 *
 * Implements the FieldIterator visitor interface for generic field-based comparison.
 * Uses the dual-node Visit overload which calls visitor methods with two field arguments.
 */
template <bool AssertMode>
class StructuralEqualImpl {
 public:
  using result_type = bool;

  explicit StructuralEqualImpl(bool enable_auto_mapping) : enable_auto_mapping_(enable_auto_mapping) {}

  // Returns bool for structural_equal, throws for assert_structural_equal
  bool operator()(const IRNodePtr& lhs, const IRNodePtr& rhs) {
    if constexpr (AssertMode) {
      Equal(lhs, rhs);
      return true;  // Only reached if no exception thrown
    } else {
      return Equal(lhs, rhs);
    }
  }

  bool operator()(const TypePtr& lhs, const TypePtr& rhs) {
    if constexpr (AssertMode) {
      EqualType(lhs, rhs);
      return true;  // Only reached if no exception thrown
    } else {
      return EqualType(lhs, rhs);
    }
  }

  // FieldIterator visitor interface (dual-node version - methods receive two fields)
  [[nodiscard]] result_type InitResult() const { return true; }

  template <typename IRNodePtrType>
  result_type VisitIRNodeField(const IRNodePtrType& lhs, const IRNodePtrType& rhs) {
    INTERNAL_CHECK(lhs) << "structural_equal encountered null lhs IR node field";
    INTERNAL_CHECK(rhs) << "structural_equal encountered null rhs IR node field";
    return Equal(lhs, rhs);
  }

  // Specialization for std::optional<IRNodePtr>
  template <typename IRNodePtrType>
  result_type VisitIRNodeField(const std::optional<IRNodePtrType>& lhs,
                               const std::optional<IRNodePtrType>& rhs) {
    if (!lhs.has_value() && !rhs.has_value()) {
      return true;
    }
    if (!lhs.has_value() || !rhs.has_value()) {
      if constexpr (AssertMode) {
        ThrowMismatch("Optional field presence mismatch", lhs.has_value() ? *lhs : IRNodePtr(),
                      rhs.has_value() ? *rhs : IRNodePtr(), lhs.has_value() ? "has value" : "nullopt",
                      rhs.has_value() ? "has value" : "nullopt");
      }
      return false;
    }
    if (!*lhs && !*rhs) {
      return true;
    }
    if (!*lhs || !*rhs) {
      if constexpr (AssertMode) {
        ThrowMismatch("Optional field nullptr mismatch", *lhs, *rhs, *lhs ? "has value" : "nullptr",
                      *rhs ? "has value" : "nullptr");
      }
      return false;
    }
    return Equal(*lhs, *rhs);
  }

  template <typename IRNodePtrType>
  result_type VisitIRNodeVectorField(const std::vector<IRNodePtrType>& lhs,
                                     const std::vector<IRNodePtrType>& rhs) {
    if (lhs.size() != rhs.size()) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "Vector size mismatch (" << lhs.size() << " items != " << rhs.size() << " items)";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
      INTERNAL_CHECK(lhs[i]) << "structural_equal encountered null lhs IR node in vector at index " << i;
      INTERNAL_CHECK(rhs[i]) << "structural_equal encountered null rhs IR node in vector at index " << i;

      if constexpr (AssertMode) {
        std::ostringstream index_str;
        index_str << "[" << i << "]";
        path_.push_back(index_str.str());
      }

      if (!Equal(lhs[i], rhs[i])) {
        if constexpr (AssertMode) {
          path_.pop_back();
        }
        return false;
      }

      if constexpr (AssertMode) {
        path_.pop_back();
      }
    }
    return true;
  }

  template <typename KeyType, typename ValueType, typename Compare>
  result_type VisitIRNodeMapField(const std::map<KeyType, ValueType, Compare>& lhs,
                                  const std::map<KeyType, ValueType, Compare>& rhs) {
    if (lhs.size() != rhs.size()) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "Map size mismatch (" << lhs.size() << " items != " << rhs.size() << " items)";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    auto lhs_it = lhs.begin();
    auto rhs_it = rhs.begin();
    while (lhs_it != lhs.end()) {
      INTERNAL_CHECK(lhs_it->first) << "structural_equal encountered null lhs key in map";
      INTERNAL_CHECK(lhs_it->second) << "structural_equal encountered null lhs value in map";
      INTERNAL_CHECK(rhs_it->first) << "structural_equal encountered null rhs key in map";
      INTERNAL_CHECK(rhs_it->second) << "structural_equal encountered null rhs value in map";

      if (lhs_it->first->name_ != rhs_it->first->name_) {
        if constexpr (AssertMode) {
          std::ostringstream msg;
          msg << "Map key mismatch ('" << lhs_it->first->name_ << "' != '" << rhs_it->first->name_ << "')";
          ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
        }
        return false;
      }

      if constexpr (AssertMode) {
        std::ostringstream key_str;
        key_str << "['" << lhs_it->first->name_ << "']";
        path_.push_back(key_str.str());
      }

      if (!Equal(lhs_it->second, rhs_it->second)) {
        if constexpr (AssertMode) {
          path_.pop_back();
        }
        return false;
      }

      if constexpr (AssertMode) {
        path_.pop_back();
      }
      ++lhs_it;
      ++rhs_it;
    }
    return true;
  }

  // Leaf field comparisons (dual-node version)
  result_type VisitLeafField(const int& lhs, const int& rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "Integer value mismatch (" << lhs << " != " << rhs << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  result_type VisitLeafField(const int64_t& lhs, const int64_t& rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "int64_t value mismatch (" << lhs << " != " << rhs << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  result_type VisitLeafField(const uint64_t& lhs, const uint64_t& rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "uint64_t value mismatch (" << lhs << " != " << rhs << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  result_type VisitLeafField(const double& lhs, const double& rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "double value mismatch (" << lhs << " != " << rhs << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  result_type VisitLeafField(const std::string& lhs, const std::string& rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "String value mismatch (\"" << lhs << "\" != \"" << rhs << "\")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  result_type VisitLeafField(const OpPtr& lhs, const OpPtr& rhs) {
    if (lhs->name_ != rhs->name_) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "Operator name mismatch ('" << lhs->name_ << "' != '" << rhs->name_ << "')";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  result_type VisitLeafField(const DataType& lhs, const DataType& rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "DataType mismatch (" << lhs.ToString() << " != " << rhs.ToString() << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  result_type VisitLeafField(const FunctionType& lhs, const FunctionType& rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "FunctionType mismatch (" << FunctionTypeToString(lhs) << " != " << FunctionTypeToString(rhs)
            << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  result_type VisitLeafField(const ForKind& lhs, const ForKind& rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "ForKind mismatch (" << ForKindToString(lhs) << " != " << ForKindToString(rhs) << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  [[nodiscard]] result_type VisitLeafField(const ScopeKind& lhs, const ScopeKind& rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "ScopeKind mismatch (" << ScopeKindToString(lhs) << " != " << ScopeKindToString(rhs) << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  result_type VisitLeafField(const ParamDirection& lhs, const ParamDirection& rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "ParamDirection mismatch (" << ParamDirectionToString(lhs)
            << " != " << ParamDirectionToString(rhs) << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  result_type VisitLeafField(const std::vector<ParamDirection>& lhs, const std::vector<ParamDirection>& rhs) {
    if (lhs.size() != rhs.size()) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "ParamDirection vector size mismatch (" << lhs.size() << " != " << rhs.size() << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
      if (!VisitLeafField(lhs[i], rhs[i])) {
        return false;
      }
    }
    return true;
  }

  // Compare kwargs (vector of pairs to preserve order)
  result_type VisitLeafField(const std::vector<std::pair<std::string, std::any>>& lhs,
                             const std::vector<std::pair<std::string, std::any>>& rhs) {
    if (lhs.size() != rhs.size()) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "Kwargs size mismatch (" << lhs.size() << " != " << rhs.size() << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
      if (lhs[i].first != rhs[i].first) {
        if constexpr (AssertMode) {
          std::ostringstream msg;
          msg << "Kwargs key mismatch at index " << i << " ('" << lhs[i].first << "' != '" << rhs[i].first
              << "')";
          ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
        }
        return false;
      }
      // Compare std::any values by type and content
      const auto& lhs_val = lhs[i].second;
      const auto& rhs_val = rhs[i].second;
      if (lhs_val.type() != rhs_val.type()) {
        if constexpr (AssertMode) {
          std::ostringstream msg;
          msg << "Kwargs value type mismatch for key '" << lhs[i].first << "'";
          ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
        }
        return false;
      }
      // Type-specific comparison
      bool values_equal = true;
      if (lhs_val.type() == typeid(int)) {
        values_equal = (AnyCast<int>(lhs_val, "comparing kwarg: " + lhs[i].first) ==
                        AnyCast<int>(rhs_val, "comparing kwarg: " + lhs[i].first));
      } else if (lhs_val.type() == typeid(bool)) {
        values_equal = (AnyCast<bool>(lhs_val, "comparing kwarg: " + lhs[i].first) ==
                        AnyCast<bool>(rhs_val, "comparing kwarg: " + lhs[i].first));
      } else if (lhs_val.type() == typeid(std::string)) {
        values_equal = (AnyCast<std::string>(lhs_val, "comparing kwarg: " + lhs[i].first) ==
                        AnyCast<std::string>(rhs_val, "comparing kwarg: " + lhs[i].first));
      } else if (lhs_val.type() == typeid(double)) {
        values_equal = (AnyCast<double>(lhs_val, "comparing kwarg: " + lhs[i].first) ==
                        AnyCast<double>(rhs_val, "comparing kwarg: " + lhs[i].first));
      } else if (lhs_val.type() == typeid(DataType)) {
        values_equal = (AnyCast<DataType>(lhs_val, "comparing kwarg: " + lhs[i].first) ==
                        AnyCast<DataType>(rhs_val, "comparing kwarg: " + lhs[i].first));
      }
      if (!values_equal) {
        if constexpr (AssertMode) {
          std::ostringstream msg;
          msg << "Kwargs value mismatch for key '" << lhs[i].first << "'";
          ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
        }
        return false;
      }
    }
    return true;
  }

  result_type VisitLeafField(const MemorySpace& lhs, const MemorySpace& rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "MemorySpace mismatch (" << MemorySpaceToString(lhs) << " != " << MemorySpaceToString(rhs)
            << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  result_type VisitLeafField(const TypePtr& lhs, const TypePtr& rhs) { return EqualType(lhs, rhs); }

  result_type VisitLeafField(const std::vector<TypePtr>& lhs, const std::vector<TypePtr>& rhs) {
    if (lhs.size() != rhs.size()) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "Type vector size mismatch (" << lhs.size() << " types != " << rhs.size() << " types)";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
      INTERNAL_CHECK(lhs[i]) << "structural_equal encountered null lhs TypePtr in vector at index " << i;
      INTERNAL_CHECK(rhs[i]) << "structural_equal encountered null rhs TypePtr in vector at index " << i;
      if (!EqualType(lhs[i], rhs[i])) return false;
    }
    return true;
  }

  [[nodiscard]] result_type VisitLeafField(const Span& lhs, const Span& rhs) const {
    INTERNAL_UNREACHABLE << "structural_equal should not visit Span field";
    return true;  // Never reached
  }

  // Field kind hooks
  template <typename FVisitOp>
  void VisitIgnoreField([[maybe_unused]] FVisitOp&& visit_op) {
    // Ignored fields are always considered equal
  }

  template <typename FVisitOp>
  void VisitDefField(FVisitOp&& visit_op) {
    bool enable_auto_mapping = true;
    std::swap(enable_auto_mapping, enable_auto_mapping_);
    visit_op();
    std::swap(enable_auto_mapping, enable_auto_mapping_);
  }

  template <typename FVisitOp>
  void VisitUsualField(FVisitOp&& visit_op) {
    visit_op();
  }

  // Combine results (AND logic)
  template <typename Desc>
  void CombineResult(result_type& accumulator, result_type field_result, [[maybe_unused]] const Desc& desc) {
    accumulator = accumulator && field_result;
  }

 private:
  bool Equal(const IRNodePtr& lhs, const IRNodePtr& rhs);
  bool EqualVar(const VarPtr& lhs, const VarPtr& rhs);
  bool EqualMemRef(const MemRefPtr& lhs, const MemRefPtr& rhs);
  bool EqualIterArg(const IterArgPtr& lhs, const IterArgPtr& rhs);
  bool EqualType(const TypePtr& lhs, const TypePtr& rhs);

  /**
   * @brief Generic field-based equality check for IR nodes using FieldIterator
   *
   * Uses the dual-node Visit overload which passes two fields to each visitor method.
   *
   * @tparam NodePtr Shared pointer type to the node
   * @param lhs_op Left-hand side node
   * @param rhs_op Right-hand side node
   * @return true if all fields are equal
   */
  template <typename NodePtr>
  bool EqualWithFields(const NodePtr& lhs_op, const NodePtr& rhs_op) {
    using NodeType = typename NodePtr::element_type;
    auto descriptors = NodeType::GetFieldDescriptors();

    return std::apply(
        [&](auto&&... descs) {
          return reflection::FieldIterator<NodeType, StructuralEqualImpl<AssertMode>,
                                           decltype(descs)...>::Visit(*lhs_op, *rhs_op, *this, descs...);
        },
        descriptors);
  }

  // Only used in assert mode for error messages
  void ThrowMismatch(const std::string& reason, const IRNodePtr& lhs, const IRNodePtr& rhs,
                     const std::string& lhs_desc = "", const std::string& rhs_desc = "") {
    if constexpr (AssertMode) {
      std::ostringstream msg;
      msg << "Structural equality assertion failed";

      if (!path_.empty()) {
        msg << " at: ";
        for (size_t i = 0; i < path_.size(); ++i) {
          msg << path_[i];
          if (i < path_.size() - 1 && path_[i + 1][0] != '[') {
            msg << ".";
          }
        }
      }
      msg << "\n\n";

      if (lhs || rhs) {
        msg << "Left-hand side:\n";
        if (lhs) {
          std::string lhs_str = PythonPrint(lhs, "pl");
          std::istringstream iss(lhs_str);
          std::string line;
          while (std::getline(iss, line)) {
            msg << "  " << line << "\n";
          }
        } else {
          msg << "  (null)\n";
        }

        msg << "\nRight-hand side:\n";
        if (rhs) {
          std::string rhs_str = PythonPrint(rhs, "pl");
          std::istringstream iss(rhs_str);
          std::string line;
          while (std::getline(iss, line)) {
            msg << "  " << line << "\n";
          }
        } else {
          msg << "  (null)\n";
        }
        msg << "\n";
      } else if (!lhs_desc.empty() || !rhs_desc.empty()) {
        msg << "Left: " << lhs_desc << "\n";
        msg << "Right: " << rhs_desc << "\n\n";
      }

      msg << "Reason: " << reason;
      throw pypto::ValueError(msg.str());
    }
  }

  bool enable_auto_mapping_;
  std::unordered_map<VarPtr, VarPtr> lhs_to_rhs_var_map_;
  std::unordered_map<VarPtr, VarPtr> rhs_to_lhs_var_map_;
  std::vector<std::string> path_;  // Only used in assert mode
};

// Type dispatch macro for generic field-based comparison
#define EQUAL_DISPATCH(Type)                                             \
  if (auto lhs_##Type = As<Type>(lhs)) {                                 \
    if constexpr (AssertMode) path_.emplace_back(#Type);                 \
    auto rhs_##Type = As<Type>(rhs);                                     \
    bool result = rhs_##Type && EqualWithFields(lhs_##Type, rhs_##Type); \
    if constexpr (AssertMode) path_.pop_back();                          \
    return result;                                                       \
  }

// Dispatch macro for abstract base classes
#define EQUAL_DISPATCH_BASE(Type)                                        \
  if (auto lhs_##Type = As<Type>(lhs)) {                                 \
    if constexpr (AssertMode) path_.emplace_back(#Type);                 \
    auto rhs_##Type = As<Type>(rhs);                                     \
    bool result = rhs_##Type && EqualWithFields(lhs_##Type, rhs_##Type); \
    if constexpr (AssertMode) path_.pop_back();                          \
    return result;                                                       \
  }

template <bool AssertMode>
bool StructuralEqualImpl<AssertMode>::Equal(const IRNodePtr& lhs, const IRNodePtr& rhs) {
  if (lhs.get() == rhs.get()) return true;

  if (!lhs || !rhs) {
    if constexpr (AssertMode) ThrowMismatch("One node is null, the other is not", lhs, rhs);
    return false;
  }

  if (lhs->TypeName() != rhs->TypeName()) {
    if constexpr (AssertMode) {
      std::ostringstream msg;
      msg << "Node type mismatch (" << lhs->TypeName() << " != " << rhs->TypeName() << ")";
      ThrowMismatch(msg.str(), lhs, rhs);
    }
    return false;
  }

  // Check MemRef before IterArg and Var (MemRef inherits from Var)
  if (auto lhs_memref = As<MemRef>(lhs)) {
    if constexpr (AssertMode) path_.emplace_back("MemRef");
    auto rhs_memref = std::static_pointer_cast<const MemRef>(rhs);
    bool result = rhs_memref && EqualMemRef(lhs_memref, rhs_memref);
    if constexpr (AssertMode) path_.pop_back();
    return result;
  }

  // Check IterArg before Var (IterArg inherits from Var)
  if (auto lhs_iter = As<IterArg>(lhs)) {
    if constexpr (AssertMode) path_.emplace_back("IterArg");
    bool result = EqualIterArg(lhs_iter, std::static_pointer_cast<const IterArg>(rhs));
    if constexpr (AssertMode) path_.pop_back();
    return result;
  }

  if (auto lhs_var = As<Var>(lhs)) {
    if constexpr (AssertMode) path_.emplace_back("Var");
    bool result = EqualVar(lhs_var, std::static_pointer_cast<const Var>(rhs));
    if constexpr (AssertMode) path_.pop_back();
    return result;
  }

  // All other types use generic field-based comparison
  EQUAL_DISPATCH(ConstInt)
  EQUAL_DISPATCH(ConstFloat)
  EQUAL_DISPATCH(ConstBool)
  EQUAL_DISPATCH(Call)
  EQUAL_DISPATCH(MakeTuple)
  EQUAL_DISPATCH(TupleGetItemExpr)

  // BinaryExpr and UnaryExpr are abstract base classes, use dynamic_pointer_cast
  EQUAL_DISPATCH_BASE(BinaryExpr)
  EQUAL_DISPATCH_BASE(UnaryExpr)

  EQUAL_DISPATCH(AssignStmt)
  EQUAL_DISPATCH(IfStmt)
  EQUAL_DISPATCH(YieldStmt)
  EQUAL_DISPATCH(ReturnStmt)
  EQUAL_DISPATCH(ForStmt)
  EQUAL_DISPATCH(WhileStmt)
  EQUAL_DISPATCH(ScopeStmt)
  EQUAL_DISPATCH(SeqStmts)
  EQUAL_DISPATCH(OpStmts)
  EQUAL_DISPATCH(EvalStmt)
  EQUAL_DISPATCH(BreakStmt)
  EQUAL_DISPATCH(ContinueStmt)
  EQUAL_DISPATCH(Function)
  EQUAL_DISPATCH(Program)

  throw pypto::TypeError("Unknown IR node type in StructuralEqualImpl::Equal: " + lhs->TypeName());
}

#undef EQUAL_DISPATCH
#undef EQUAL_DISPATCH_BASE

template <bool AssertMode>
bool StructuralEqualImpl<AssertMode>::EqualType(const TypePtr& lhs, const TypePtr& rhs) {
  if (lhs->TypeName() != rhs->TypeName()) {
    if constexpr (AssertMode) {
      std::ostringstream msg;
      msg << "Type name mismatch (" << lhs->TypeName() << " != " << rhs->TypeName() << ")";
      ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
    }
    return false;
  }

  if (auto lhs_scalar = As<ScalarType>(lhs)) {
    auto rhs_scalar = As<ScalarType>(rhs);
    if (!rhs_scalar) {
      if constexpr (AssertMode) {
        ThrowMismatch("Type cast failed for ScalarType", IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    if (lhs_scalar->dtype_ != rhs_scalar->dtype_) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "ScalarType dtype mismatch (" << lhs_scalar->dtype_.ToString()
            << " != " << rhs_scalar->dtype_.ToString() << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  } else if (auto lhs_tensor = As<TensorType>(lhs)) {
    auto rhs_tensor = As<TensorType>(rhs);
    if (!rhs_tensor) {
      if constexpr (AssertMode) {
        ThrowMismatch("Type cast failed for TensorType", IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    if (lhs_tensor->dtype_ != rhs_tensor->dtype_) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "TensorType dtype mismatch (" << lhs_tensor->dtype_.ToString()
            << " != " << rhs_tensor->dtype_.ToString() << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    if (lhs_tensor->shape_.size() != rhs_tensor->shape_.size()) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "TensorType shape rank mismatch (" << lhs_tensor->shape_.size()
            << " != " << rhs_tensor->shape_.size() << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    for (size_t i = 0; i < lhs_tensor->shape_.size(); ++i) {
      if (!Equal(lhs_tensor->shape_[i], rhs_tensor->shape_[i])) return false;
    }
    return true;
  } else if (auto lhs_tile = As<TileType>(lhs)) {
    auto rhs_tile = As<TileType>(rhs);
    if (!rhs_tile) {
      if constexpr (AssertMode) {
        ThrowMismatch("Type cast failed for TileType", IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    // Compare dtype
    if (lhs_tile->dtype_ != rhs_tile->dtype_) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "TileType dtype mismatch (" << lhs_tile->dtype_.ToString()
            << " != " << rhs_tile->dtype_.ToString() << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    // Compare shape size and dimensions
    if (lhs_tile->shape_.size() != rhs_tile->shape_.size()) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "TileType shape rank mismatch (" << lhs_tile->shape_.size()
            << " != " << rhs_tile->shape_.size() << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    for (size_t i = 0; i < lhs_tile->shape_.size(); ++i) {
      if (!Equal(lhs_tile->shape_[i], rhs_tile->shape_[i])) return false;
    }
    // Compare tile_view
    if (lhs_tile->tile_view_.has_value() != rhs_tile->tile_view_.has_value()) {
      if constexpr (AssertMode) {
        ThrowMismatch("TileType tile_view presence mismatch", IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    if (lhs_tile->tile_view_.has_value()) {
      const auto& lhs_tv = lhs_tile->tile_view_.value();
      const auto& rhs_tv = rhs_tile->tile_view_.value();
      // Compare valid_shape
      if (lhs_tv.valid_shape.size() != rhs_tv.valid_shape.size()) {
        if constexpr (AssertMode) {
          std::ostringstream msg;
          msg << "TileView valid_shape size mismatch (" << lhs_tv.valid_shape.size()
              << " != " << rhs_tv.valid_shape.size() << ")";
          ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
        }
        return false;
      }
      for (size_t i = 0; i < lhs_tv.valid_shape.size(); ++i) {
        if (!Equal(lhs_tv.valid_shape[i], rhs_tv.valid_shape[i])) return false;
      }
      // Compare stride
      if (lhs_tv.stride.size() != rhs_tv.stride.size()) {
        if constexpr (AssertMode) {
          std::ostringstream msg;
          msg << "TileView stride size mismatch (" << lhs_tv.stride.size() << " != " << rhs_tv.stride.size()
              << ")";
          ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
        }
        return false;
      }
      for (size_t i = 0; i < lhs_tv.stride.size(); ++i) {
        if (!Equal(lhs_tv.stride[i], rhs_tv.stride[i])) return false;
      }
      // Compare start_offset
      if (!Equal(lhs_tv.start_offset, rhs_tv.start_offset)) return false;
      // Compare blayout
      if (lhs_tv.blayout != rhs_tv.blayout) {
        if constexpr (AssertMode) {
          ThrowMismatch("TileView blayout mismatch", IRNodePtr(), IRNodePtr(), "", "");
        }
        return false;
      }
      // Compare slayout
      if (lhs_tv.slayout != rhs_tv.slayout) {
        if constexpr (AssertMode) {
          ThrowMismatch("TileView slayout mismatch", IRNodePtr(), IRNodePtr(), "", "");
        }
        return false;
      }
      // Compare fractal
      if (lhs_tv.fractal != rhs_tv.fractal) {
        if constexpr (AssertMode) {
          ThrowMismatch("TileView fractal mismatch", IRNodePtr(), IRNodePtr(), "", "");
        }
        return false;
      }
      // Compare pad
      if (lhs_tv.pad != rhs_tv.pad) {
        if constexpr (AssertMode) {
          ThrowMismatch("TileView pad mismatch", IRNodePtr(), IRNodePtr(), "", "");
        }
        return false;
      }
    }
    return true;
  } else if (auto lhs_tuple = As<TupleType>(lhs)) {
    auto rhs_tuple = As<TupleType>(rhs);
    if (!rhs_tuple) {
      if constexpr (AssertMode) {
        ThrowMismatch("Type cast failed for TupleType", IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    if (lhs_tuple->types_.size() != rhs_tuple->types_.size()) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "TupleType size mismatch (" << lhs_tuple->types_.size() << " != " << rhs_tuple->types_.size()
            << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    for (size_t i = 0; i < lhs_tuple->types_.size(); ++i) {
      if (!EqualType(lhs_tuple->types_[i], rhs_tuple->types_[i])) return false;
    }
    return true;
  } else if (IsA<MemRefType>(lhs) || IsA<UnknownType>(lhs)) {
    return true;  // Singleton type, both being MemRefType or UnknownType is sufficient
  }

  INTERNAL_UNREACHABLE << "EqualType encountered unhandled Type: " << lhs->TypeName();
  return false;
}

template <bool AssertMode>
bool StructuralEqualImpl<AssertMode>::EqualVar(const VarPtr& lhs, const VarPtr& rhs) {
  if (!enable_auto_mapping_) {
    auto lhs_it = lhs_to_rhs_var_map_.find(lhs);
    auto rhs_it = rhs_to_lhs_var_map_.find(rhs);
    // Case 1: already mapped to the same variable
    if (lhs_it != lhs_to_rhs_var_map_.end() && rhs_it != rhs_to_lhs_var_map_.end()) {
      if (lhs_it->second != rhs || rhs_it->second != lhs) {
        if constexpr (AssertMode) {
          ThrowMismatch("Variable mapping inconsistent (without auto-mapping)",
                        std::static_pointer_cast<const IRNode>(lhs),
                        std::static_pointer_cast<const IRNode>(rhs), "var " + lhs->name_,
                        "var " + rhs->name_);
        }
        return false;
      }
      return true;
    }
    // Case 2: different variables
    if (lhs.get() != rhs.get()) {
      if constexpr (AssertMode) {
        ThrowMismatch("Variable pointer mismatch (without auto-mapping)",
                      std::static_pointer_cast<const IRNode>(lhs),
                      std::static_pointer_cast<const IRNode>(rhs), "var " + lhs->name_, "var " + rhs->name_);
      }
      return false;
    }
    return true;
  }

  if (!EqualType(lhs->GetType(), rhs->GetType())) {
    if constexpr (AssertMode) {
      std::ostringstream msg;
      msg << "Variable type mismatch (" << lhs->GetType()->TypeName() << " != " << rhs->GetType()->TypeName()
          << ")";
      ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
    }
    return false;
  }

  auto it = lhs_to_rhs_var_map_.find(lhs);
  if (it != lhs_to_rhs_var_map_.end()) {
    if (it->second != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "Variable mapping inconsistent ('" << lhs->name_ << "' cannot map to both '"
            << it->second->name_ << "' and '" << rhs->name_ << "')";
        ThrowMismatch(msg.str(), std::static_pointer_cast<const IRNode>(lhs),
                      std::static_pointer_cast<const IRNode>(rhs));
      }
      return false;
    }
    return true;
  }

  auto rhs_it = rhs_to_lhs_var_map_.find(rhs);
  if (rhs_it != rhs_to_lhs_var_map_.end() && rhs_it->second != lhs) {
    if constexpr (AssertMode) {
      std::ostringstream msg;
      msg << "Variable mapping inconsistent ('" << rhs->name_ << "' is already mapped from '"
          << rhs_it->second->name_ << "', cannot map from '" << lhs->name_ << "')";
      ThrowMismatch(msg.str(), std::static_pointer_cast<const IRNode>(lhs),
                    std::static_pointer_cast<const IRNode>(rhs));
    }
    return false;
  }

  lhs_to_rhs_var_map_[lhs] = rhs;
  rhs_to_lhs_var_map_[rhs] = lhs;
  return true;
}

template <bool AssertMode>
bool StructuralEqualImpl<AssertMode>::EqualMemRef(const MemRefPtr& lhs, const MemRefPtr& rhs) {
  // 1. First, compare as Var (handles variable mapping and type comparison)
  if (!EqualVar(lhs, rhs)) {
    return false;
  }

  // 2. Then, compare MemRef-specific fields (except id_ which is a naming counter)
  if (lhs->memory_space_ != rhs->memory_space_) {
    if constexpr (AssertMode) {
      std::ostringstream msg;
      msg << "MemRef memory_space mismatch (" << MemorySpaceToString(lhs->memory_space_)
          << " != " << MemorySpaceToString(rhs->memory_space_) << ")";
      ThrowMismatch(msg.str(), std::static_pointer_cast<const IRNode>(lhs),
                    std::static_pointer_cast<const IRNode>(rhs));
    }
    return false;
  }

  if (!Equal(lhs->addr_, rhs->addr_)) {
    if constexpr (AssertMode) {
      ThrowMismatch("MemRef addr mismatch", std::static_pointer_cast<const IRNode>(lhs),
                    std::static_pointer_cast<const IRNode>(rhs));
    }
    return false;
  }

  if (lhs->size_ != rhs->size_) {
    if constexpr (AssertMode) {
      std::ostringstream msg;
      msg << "MemRef size mismatch (" << lhs->size_ << " != " << rhs->size_ << ")";
      ThrowMismatch(msg.str(), std::static_pointer_cast<const IRNode>(lhs),
                    std::static_pointer_cast<const IRNode>(rhs));
    }
    return false;
  }

  return true;
}

template <bool AssertMode>
bool StructuralEqualImpl<AssertMode>::EqualIterArg(const IterArgPtr& lhs, const IterArgPtr& rhs) {
  // 1. First, compare as Var (handles variable mapping)
  if (!EqualVar(lhs, rhs)) {
    return false;
  }

  // 2. Then, compare IterArg-specific field: initValue_
  if (!Equal(lhs->initValue_, rhs->initValue_)) {
    if constexpr (AssertMode) {
      ThrowMismatch("IterArg initValue mismatch", std::static_pointer_cast<const IRNode>(lhs),
                    std::static_pointer_cast<const IRNode>(rhs));
    }
    return false;
  }

  return true;
}

// Explicit template instantiations
template class StructuralEqualImpl<false>;  // For structural_equal
template class StructuralEqualImpl<true>;   // For assert_structural_equal

// Type aliases for cleaner code
using StructuralEqual = StructuralEqualImpl<false>;
using StructuralEqualAssert = StructuralEqualImpl<true>;

// Public API implementation
bool structural_equal(const IRNodePtr& lhs, const IRNodePtr& rhs, bool enable_auto_mapping) {
  StructuralEqual checker(enable_auto_mapping);
  return checker(lhs, rhs);
}

bool structural_equal(const TypePtr& lhs, const TypePtr& rhs, bool enable_auto_mapping) {
  StructuralEqual checker(enable_auto_mapping);
  return checker(lhs, rhs);
}

// Public assert API
void assert_structural_equal(const IRNodePtr& lhs, const IRNodePtr& rhs, bool enable_auto_mapping) {
  StructuralEqualAssert checker(enable_auto_mapping);
  checker(lhs, rhs);
}

void assert_structural_equal(const TypePtr& lhs, const TypePtr& rhs, bool enable_auto_mapping) {
  StructuralEqualAssert checker(enable_auto_mapping);
  checker(lhs, rhs);
}

}  // namespace ir
}  // namespace pypto
