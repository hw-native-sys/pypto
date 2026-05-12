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
#include "pypto/ir/memory_space.h"
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

namespace {

bool AreForSyntaxScalarDtypesEquivalent(const DataType& lhs, const DataType& rhs) {
  return (lhs == DataType::INT64 && rhs == DataType::INDEX) ||
         (lhs == DataType::INDEX && rhs == DataType::INT64);
}

}  // namespace

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
    INTERNAL_CHECK_SPAN(rhs, lhs->span_) << "structural_equal encountered null rhs IR node field";
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
      INTERNAL_CHECK_SPAN(rhs[i], lhs[i]->span_)
          << "structural_equal encountered null rhs IR node in vector at index " << i;

      if constexpr (AssertMode) {
        std::ostringstream index_str;
        index_str << "[" << i << "]";
        path_.emplace_back(index_str.str());
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
      INTERNAL_CHECK_SPAN(rhs_it->second, lhs_it->second->span_)
          << "structural_equal encountered null rhs value in map";

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
        path_.emplace_back(key_str.str());
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

  result_type VisitLeafField(const std::vector<int64_t>& lhs, const std::vector<int64_t>& rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "vector<int64_t> mismatch (size " << lhs.size() << " vs " << rhs.size() << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
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

  result_type VisitLeafField(const bool& lhs, const bool& rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "Bool value mismatch (" << (lhs ? "true" : "false") << " != " << (rhs ? "true" : "false")
            << ")";
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

  result_type VisitLeafField(const InlineLanguage& lhs, const InlineLanguage& rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "InlineLanguage mismatch (" << static_cast<unsigned int>(lhs)
            << " != " << static_cast<unsigned int>(rhs) << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  result_type VisitLeafField(const ChunkPolicy& lhs, const ChunkPolicy& rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "ChunkPolicy mismatch (" << ChunkPolicyToString(lhs) << " != " << ChunkPolicyToString(rhs)
            << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  result_type VisitLeafField(const std::optional<ChunkConfig>& lhs, const std::optional<ChunkConfig>& rhs) {
    if (lhs.has_value() != rhs.has_value()) {
      if constexpr (AssertMode) {
        ThrowMismatch("ChunkConfig presence mismatch", IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    if (!lhs.has_value()) return true;
    if (!VisitIRNodeField(lhs->size, rhs->size)) return false;
    return VisitLeafField(lhs->policy, rhs->policy);
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

  result_type VisitLeafField(const Level& lhs, const Level& rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "Level mismatch (" << LevelToString(lhs) << " != " << LevelToString(rhs) << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  result_type VisitLeafField(const Role& lhs, const Role& rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "Role mismatch (" << RoleToString(lhs) << " != " << RoleToString(rhs) << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  result_type VisitLeafField(const std::optional<Level>& lhs, const std::optional<Level>& rhs) {
    if (lhs.has_value() != rhs.has_value()) {
      if constexpr (AssertMode) {
        ThrowMismatch("Level optional presence mismatch", IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    if (lhs.has_value()) {
      return VisitLeafField(*lhs, *rhs);
    }
    return true;
  }

  result_type VisitLeafField(const std::optional<Role>& lhs, const std::optional<Role>& rhs) {
    if (lhs.has_value() != rhs.has_value()) {
      if constexpr (AssertMode) {
        ThrowMismatch("Role optional presence mismatch", IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    if (lhs.has_value()) {
      return VisitLeafField(*lhs, *rhs);
    }
    return true;
  }

  result_type VisitLeafField(const SplitMode& lhs, const SplitMode& rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        ThrowMismatch("SplitMode mismatch", IRNodePtr(), IRNodePtr(), SplitModeToString(lhs),
                      SplitModeToString(rhs));
      }
      return false;
    }
    return true;
  }

  result_type VisitLeafField(const std::optional<SplitMode>& lhs, const std::optional<SplitMode>& rhs) {
    if (lhs.has_value() != rhs.has_value()) {
      if constexpr (AssertMode) {
        ThrowMismatch("SplitMode optional presence mismatch", IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    if (lhs.has_value()) {
      return VisitLeafField(*lhs, *rhs);
    }
    return true;
  }

  result_type VisitLeafField(const std::optional<int>& lhs, const std::optional<int>& rhs) {
    if (lhs.has_value() != rhs.has_value()) {
      if constexpr (AssertMode) {
        ThrowMismatch("optional<int> presence mismatch", IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    if (lhs.has_value() && *lhs != *rhs) {
      if constexpr (AssertMode) {
        ThrowMismatch("optional<int> value mismatch", IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  result_type VisitLeafField(const std::optional<bool>& lhs, const std::optional<bool>& rhs) {
    if (lhs.has_value() != rhs.has_value()) {
      if constexpr (AssertMode) {
        ThrowMismatch("optional<bool> presence mismatch", IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    if (lhs.has_value() && *lhs != *rhs) {
      if constexpr (AssertMode) {
        ThrowMismatch("optional<bool> value mismatch", IRNodePtr(), IRNodePtr(), "", "");
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

  result_type VisitLeafField(const ArgDirection& lhs, const ArgDirection& rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "ArgDirection mismatch (" << ArgDirectionToString(lhs) << " != " << ArgDirectionToString(rhs)
            << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  result_type VisitLeafField(const std::vector<ArgDirection>& lhs, const std::vector<ArgDirection>& rhs) {
    if (lhs.size() != rhs.size()) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "ArgDirection vector size mismatch (" << lhs.size() << " != " << rhs.size() << ")";
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
      } else if (lhs_val.type() == typeid(MemorySpace)) {
        values_equal = (AnyCast<MemorySpace>(lhs_val, "comparing kwarg: " + lhs[i].first) ==
                        AnyCast<MemorySpace>(rhs_val, "comparing kwarg: " + lhs[i].first));
      } else if (lhs_val.type() == typeid(LoopOrigin)) {
        values_equal = (AnyCast<LoopOrigin>(lhs_val, "comparing kwarg: " + lhs[i].first) ==
                        AnyCast<LoopOrigin>(rhs_val, "comparing kwarg: " + lhs[i].first));
      } else if (lhs_val.type() == typeid(TensorLayout)) {
        values_equal = (AnyCast<TensorLayout>(lhs_val, "comparing kwarg: " + lhs[i].first) ==
                        AnyCast<TensorLayout>(rhs_val, "comparing kwarg: " + lhs[i].first));
      } else if (lhs_val.type() == typeid(TileLayout)) {
        values_equal = (AnyCast<TileLayout>(lhs_val, "comparing kwarg: " + lhs[i].first) ==
                        AnyCast<TileLayout>(rhs_val, "comparing kwarg: " + lhs[i].first));
      } else if (lhs_val.type() == typeid(PadValue)) {
        values_equal = (AnyCast<PadValue>(lhs_val, "comparing kwarg: " + lhs[i].first) ==
                        AnyCast<PadValue>(rhs_val, "comparing kwarg: " + lhs[i].first));
      } else if (lhs_val.type() == typeid(float)) {
        values_equal = (AnyCast<float>(lhs_val, "comparing kwarg: " + lhs[i].first) ==
                        AnyCast<float>(rhs_val, "comparing kwarg: " + lhs[i].first));
      } else if (lhs_val.type() == typeid(std::vector<ArgDirection>)) {
        const auto& lhs_dirs =
            AnyCast<std::vector<ArgDirection>>(lhs_val, "comparing kwarg: " + lhs[i].first);
        const auto& rhs_dirs =
            AnyCast<std::vector<ArgDirection>>(rhs_val, "comparing kwarg: " + lhs[i].first);
        values_equal = VisitLeafField(lhs_dirs, rhs_dirs);
      } else if (lhs_val.type() == typeid(ExprPtr)) {
        const auto& lhs_expr = AnyCast<ExprPtr>(lhs_val, "comparing kwarg: " + lhs[i].first);
        const auto& rhs_expr = AnyCast<ExprPtr>(rhs_val, "comparing kwarg: " + lhs[i].first);
        values_equal = Equal(lhs_expr, rhs_expr);
      } else {
        INTERNAL_CHECK(false) << "Unsupported kwargs value type for key '" << lhs[i].first
                              << "': " << DemangleTypeName(lhs_val.type().name());
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
    INTERNAL_UNREACHABLE_SPAN(lhs) << "structural_equal should not visit Span field";
    return true;  // Never reached
  }

  // Required by the template instantiation path for Stmt::leading_comments_
  // (IgnoreField). VisitIgnoreField discards the lambda at runtime, so this
  // overload should never be called — guard it like the Span overload.
  result_type VisitLeafField(const std::vector<std::string>& /*lhs*/,
                             const std::vector<std::string>& /*rhs*/) {
    INTERNAL_UNREACHABLE << "structural_equal should not visit leading_comments field";
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

  // Path tracking hooks called by FieldIterator::VisitFieldImpl for each field.
  // PushFieldName pushes ".name" only when not inside a transparent container.
  // Transparent containers (Program, SeqStmts) suppress their own field
  // names so that their vector/map element accessors ([i] / ['key']) attach directly
  // to the parent field name, producing paths like body[1] instead of body.stmts[1].
  void PushFieldName(const char* name) {
    if (transparent_depth_ == 0) {
      field_name_stack_.emplace_back(name);
    }
    if constexpr (AssertMode) {
      if (transparent_depth_ == 0) {
        path_.emplace_back(name);  // No dot prefix — ThrowMismatch adds '.' separators
      }
    }
  }

  void PopFieldName() {
    if (transparent_depth_ == 0) {
      field_name_stack_.pop_back();
    }
    if constexpr (AssertMode) {
      if (transparent_depth_ == 0) {
        path_.pop_back();
      }
    }
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
  bool IsLoopVarFieldContext() const {
    return !field_name_stack_.empty() && field_name_stack_.back() == "loop_var";
  }
  bool IsConstIntTypeContext() const {
    return !node_type_stack_.empty() && node_type_stack_.back() == "ConstInt" && !field_name_stack_.empty() &&
           field_name_stack_.back() == "type";
  }

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
  std::vector<std::string> field_name_stack_;
  std::vector<std::string> node_type_stack_;
  int transparent_depth_ = 0;  // Depth inside transparent containers (Program/SeqStmts)
};

// Type dispatch macro for generic field-based comparison.
// Saves and resets transparent_depth_ to 0 before entering EqualWithFields so that
// field names of this (non-transparent) node are always pushed into the path, even
// when Equal() is called recursively from within a transparent container's field visit.
#define EQUAL_DISPATCH(Type)                                             \
  if (auto lhs_##Type = As<Type>(lhs)) {                                 \
    auto rhs_##Type = As<Type>(rhs);                                     \
    node_type_stack_.emplace_back(#Type);                                \
    int saved_depth = transparent_depth_;                                \
    transparent_depth_ = 0;                                              \
    bool result = rhs_##Type && EqualWithFields(lhs_##Type, rhs_##Type); \
    transparent_depth_ = saved_depth;                                    \
    node_type_stack_.pop_back();                                         \
    return result;                                                       \
  }

// Dispatch macro for transparent container nodes (Program, SeqStmts).
// Increments transparent_depth_ so that their field names are suppressed in the path,
// allowing vector/map element accessors ([i] / ['key']) to attach directly to the
// parent field name: e.g., body[1] instead of body.stmts[1].
#define EQUAL_DISPATCH_TRANSPARENT(Type)                                 \
  if (auto lhs_##Type = As<Type>(lhs)) {                                 \
    transparent_depth_++;                                                \
    auto rhs_##Type = As<Type>(rhs);                                     \
    node_type_stack_.emplace_back(#Type);                                \
    bool result = rhs_##Type && EqualWithFields(lhs_##Type, rhs_##Type); \
    node_type_stack_.pop_back();                                         \
    transparent_depth_--;                                                \
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
    auto rhs_memref = std::static_pointer_cast<const MemRef>(rhs);
    bool result = rhs_memref && EqualMemRef(lhs_memref, rhs_memref);
    return result;
  }

  // Check IterArg before Var (IterArg inherits from Var)
  if (auto lhs_iter = As<IterArg>(lhs)) {
    bool result = EqualIterArg(lhs_iter, std::static_pointer_cast<const IterArg>(rhs));
    return result;
  }

  if (auto lhs_var = As<Var>(lhs)) {
    bool result = EqualVar(lhs_var, std::static_pointer_cast<const Var>(rhs));
    return result;
  }

  // All other types use generic field-based comparison
  EQUAL_DISPATCH(ConstInt)
  EQUAL_DISPATCH(ConstFloat)
  EQUAL_DISPATCH(ConstBool)
  EQUAL_DISPATCH(Call)
  EQUAL_DISPATCH(MakeTuple)
  EQUAL_DISPATCH(TupleGetItemExpr)

  // BinaryExpr and UnaryExpr are abstract base classes matching multiple kinds
  EQUAL_DISPATCH(BinaryExpr)
  EQUAL_DISPATCH(UnaryExpr)

  EQUAL_DISPATCH(AssignStmt)
  EQUAL_DISPATCH(IfStmt)
  EQUAL_DISPATCH(YieldStmt)
  EQUAL_DISPATCH(ReturnStmt)
  EQUAL_DISPATCH(ForStmt)
  EQUAL_DISPATCH(WhileStmt)
  EQUAL_DISPATCH(InCoreScopeStmt)
  EQUAL_DISPATCH(AutoInCoreScopeStmt)
  EQUAL_DISPATCH(ClusterScopeStmt)
  EQUAL_DISPATCH(HierarchyScopeStmt)
  EQUAL_DISPATCH(SpmdScopeStmt)
  EQUAL_DISPATCH(RuntimeScopeStmt)
  EQUAL_DISPATCH_TRANSPARENT(SeqStmts)
  EQUAL_DISPATCH(EvalStmt)
  EQUAL_DISPATCH(BreakStmt)
  EQUAL_DISPATCH(ContinueStmt)
  EQUAL_DISPATCH(InlineStmt)
  EQUAL_DISPATCH(Function)
  EQUAL_DISPATCH_TRANSPARENT(Program)
  EQUAL_DISPATCH(WindowBuffer)
  EQUAL_DISPATCH(CommGroup)

  throw pypto::TypeError("Unknown IR node type in StructuralEqualImpl::Equal: " + lhs->TypeName());
}

#undef EQUAL_DISPATCH
#undef EQUAL_DISPATCH_TRANSPARENT

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
    if ((IsLoopVarFieldContext() || IsConstIntTypeContext()) &&
        AreForSyntaxScalarDtypesEquivalent(lhs_scalar->dtype_, rhs_scalar->dtype_)) {
      return true;
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
  } else if (lhs->GetKind() == ObjectKind::TensorType ||
             lhs->GetKind() == ObjectKind::DistributedTensorType) {
    // DistributedTensorType has identical fields to TensorType — share the
    // comparison code via static_cast. The TypeName check above already
    // guarantees both sides have matching kinds.
    auto lhs_tensor = std::static_pointer_cast<const TensorType>(lhs);
    auto rhs_tensor = std::static_pointer_cast<const TensorType>(rhs);
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
    // Compare tensor_view
    if (lhs_tensor->tensor_view_.has_value() != rhs_tensor->tensor_view_.has_value()) {
      if constexpr (AssertMode) {
        ThrowMismatch("TensorType tensor_view presence mismatch", IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    if (lhs_tensor->tensor_view_.has_value()) {
      const auto& lhs_tv = lhs_tensor->tensor_view_.value();
      const auto& rhs_tv = rhs_tensor->tensor_view_.value();
      // Compare valid_shape
      if (lhs_tv.valid_shape.size() != rhs_tv.valid_shape.size()) {
        if constexpr (AssertMode) {
          std::ostringstream msg;
          msg << "TensorView valid_shape size mismatch (" << lhs_tv.valid_shape.size()
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
          msg << "TensorView stride size mismatch (" << lhs_tv.stride.size() << " != " << rhs_tv.stride.size()
              << ")";
          ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
        }
        return false;
      }
      for (size_t i = 0; i < lhs_tv.stride.size(); ++i) {
        if (!Equal(lhs_tv.stride[i], rhs_tv.stride[i])) return false;
      }
      // Compare layout
      if (lhs_tv.layout != rhs_tv.layout) {
        if constexpr (AssertMode) {
          ThrowMismatch("TensorView layout mismatch", IRNodePtr(), IRNodePtr(), "", "");
        }
        return false;
      }
      // Compare pad
      if (lhs_tv.pad != rhs_tv.pad) {
        if constexpr (AssertMode) {
          ThrowMismatch("TensorView pad mismatch", IRNodePtr(), IRNodePtr(), "", "");
        }
        return false;
      }
    }
    // DistributedTensorType-only field: window_buffer_ back-reference. Both
    // sides must agree on presence, and the underlying WindowBuffer Vars must
    // match through the regular Var-equality path (which falls into the Var
    // identity map / auto-mapping logic — same as MemRef inside ShapedType).
    if (lhs->GetKind() == ObjectKind::DistributedTensorType) {
      auto lhs_dt = std::static_pointer_cast<const DistributedTensorType>(lhs);
      auto rhs_dt = std::static_pointer_cast<const DistributedTensorType>(rhs);
      if (lhs_dt->window_buffer_.has_value() != rhs_dt->window_buffer_.has_value()) {
        if constexpr (AssertMode) {
          ThrowMismatch("DistributedTensorType window_buffer presence mismatch", IRNodePtr(), IRNodePtr(), "",
                        "");
        }
        return false;
      }
      if (lhs_dt->window_buffer_.has_value() && !EqualVar(*lhs_dt->window_buffer_, *rhs_dt->window_buffer_)) {
        return false;
      }
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
    // Field comparison is sufficient — TileType ctor canonicalizes implicit
    // views to nullopt, so equivalent semantic states have identical fields.
    const auto& lhs_tile_view = lhs_tile->tile_view_;
    const auto& rhs_tile_view = rhs_tile->tile_view_;
    if (lhs_tile_view.has_value() != rhs_tile_view.has_value()) {
      if constexpr (AssertMode) {
        ThrowMismatch("TileType tile_view presence mismatch", IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    if (lhs_tile_view.has_value()) {
      const auto& lhs_tv = lhs_tile_view.value();
      const auto& rhs_tv = rhs_tile_view.value();
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
    // Compare memory_space
    if (lhs_tile->memory_space_.has_value() != rhs_tile->memory_space_.has_value()) {
      if constexpr (AssertMode) {
        ThrowMismatch("TileType memory_space presence mismatch", IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    if (lhs_tile->memory_space_.has_value() &&
        lhs_tile->memory_space_.value() != rhs_tile->memory_space_.value()) {
      if constexpr (AssertMode) {
        ThrowMismatch("TileType memory_space mismatch", IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
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
  } else if (auto lhs_sa = As<ArrayType>(lhs)) {
    auto rhs_sa = As<ArrayType>(rhs);
    if (!rhs_sa) {
      if constexpr (AssertMode) {
        ThrowMismatch("Type cast failed for ArrayType", IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    if (lhs_sa->dtype_ != rhs_sa->dtype_) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "ArrayType dtype mismatch (" << lhs_sa->dtype_.ToString()
            << " != " << rhs_sa->dtype_.ToString() << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    if (!Equal(lhs_sa->extent(), rhs_sa->extent())) {
      if constexpr (AssertMode) {
        ThrowMismatch("ArrayType extent mismatch", IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  } else if (IsA<MemRefType>(lhs) || IsA<UnknownType>(lhs) || IsA<PtrType>(lhs) ||
             IsA<WindowBufferType>(lhs)) {
    return true;  // Singleton type, both being same type kind is sufficient
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
                        std::static_pointer_cast<const IRNode>(rhs), "var " + lhs->name_hint_,
                        "var " + rhs->name_hint_);
        }
        return false;
      }
      return true;
    }
    // Case 2: different variables
    if (lhs.get() != rhs.get()) {
      if constexpr (AssertMode) {
        ThrowMismatch(
            "Variable pointer mismatch (without auto-mapping)", std::static_pointer_cast<const IRNode>(lhs),
            std::static_pointer_cast<const IRNode>(rhs), "var " + lhs->name_hint_, "var " + rhs->name_hint_);
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
        msg << "Variable mapping inconsistent ('" << lhs->name_hint_ << "' cannot map to both '"
            << it->second->name_hint_ << "' and '" << rhs->name_hint_ << "')";
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
      msg << "Variable mapping inconsistent ('" << rhs->name_hint_ << "' is already mapped from '"
          << rhs_it->second->name_hint_ << "', cannot map from '" << lhs->name_hint_ << "')";
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

  // 2. Then, compare MemRef-specific fields: base_, byte_offset_, size_
  if (!EqualVar(lhs->base_, rhs->base_)) {
    if constexpr (AssertMode) {
      ThrowMismatch("MemRef base mismatch", std::static_pointer_cast<const IRNode>(lhs),
                    std::static_pointer_cast<const IRNode>(rhs));
    }
    return false;
  }

  if (!Equal(lhs->byte_offset_, rhs->byte_offset_)) {
    if constexpr (AssertMode) {
      ThrowMismatch("MemRef byte_offset mismatch", std::static_pointer_cast<const IRNode>(lhs),
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
