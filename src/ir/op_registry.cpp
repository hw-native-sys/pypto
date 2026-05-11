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

#include "pypto/ir/op_registry.h"

#include <algorithm>
#include <any>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/span.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

void ValidateKwargs(const std::vector<std::pair<std::string, std::any>>& kwargs,
                    const std::unordered_map<std::string, std::type_index>& allowed_kwargs,
                    const std::string& op_name) {
  for (const auto& [key, value] : kwargs) {
    auto it = allowed_kwargs.find(key);
    if (it == allowed_kwargs.end()) {
      throw ValueError("Unknown kwarg '" + key + "' for operator '" + op_name + "'");
    }

    // For DataType, accept both DataType and int (since Python may pass as int for backward compatibility)
    if (it->second == std::type_index(typeid(DataType))) {
      std::type_index value_type(value.type());
      if (value_type != std::type_index(typeid(DataType)) && value_type != std::type_index(typeid(int))) {
        throw TypeError("Kwarg '" + key + "' for operator '" + op_name +
                        "' expects DataType or int, but got incompatible type");
      }
    } else if (it->second == std::type_index(typeid(MemorySpace))) {
      if (std::type_index(value.type()) != std::type_index(typeid(MemorySpace))) {
        throw TypeError("Kwarg '" + key + "' for operator '" + op_name +
                        "' expects MemorySpace, but got incompatible type");
      }
    } else if (it->second == std::type_index(typeid(TileLayout))) {
      if (std::type_index(value.type()) != std::type_index(typeid(TileLayout))) {
        throw TypeError("Kwarg '" + key + "' for operator '" + op_name +
                        "' expects TileLayout, but got incompatible type");
      }
    } else if (std::type_index(value.type()) != it->second) {
      throw TypeError("Kwarg '" + key + "' for operator '" + op_name + "' has incompatible type");
    }
  }
}

OpRegistry& OpRegistry::GetInstance() {
  static OpRegistry instance;
  return instance;
}

OpRegistryEntry& OpRegistry::Register(const std::string& op_name) {
  // Check if operator is already registered
  CHECK(registry_.find(op_name) == registry_.end()) << "Operator '" + op_name + "' is already registered";

  // Create and insert the entry into the registry
  auto result = registry_.emplace(op_name, OpRegistryEntry());
  auto& entry = result.first->second;
  entry.set_name(op_name);

  // Create the operator instance with the operator name
  entry.op_ = std::make_shared<Op>(op_name);

  return entry;
}

// ============================================================================
// OpRegistry Implementation
// ============================================================================

CallPtr OpRegistry::Create(const std::string& op_name, const std::vector<ExprPtr>& args, Span span) const {
  // Call new version with empty kwargs for backward compatibility
  return Create(op_name, args, {}, std::move(span));
}

CallPtr OpRegistry::Create(const std::string& op_name, const std::vector<ExprPtr>& args,
                           const std::vector<std::pair<std::string, std::any>>& kwargs, Span span) const {
  // Look up operator in registry
  auto it = registry_.find(op_name);
  if (it == registry_.end()) {
    std::string msg = "Operator '" + op_name + "' not found in registry";
    if (op_name.find('.') == std::string::npos) {
      msg +=
          ". This looks like a function name (GlobalVar), not a registered operator. "
          "Callers should check for GlobalVar before using OpRegistry::Create.";
    }
    throw ValueError(msg);
  }

  const auto& entry = it->second;

  // Get operator instance (shared definition)
  OpPtr op = entry.GetOp();

  // Validate kwargs against allowed attributes (stored in Op)
  if (!kwargs.empty()) {
    const auto& allowed_kwargs = op->GetAttrs();
    if (!allowed_kwargs.empty()) {
      ValidateKwargs(kwargs, allowed_kwargs, op_name);
    }
  }

  const auto& deduce_type_fn = entry.GetDeduceType();

  // Deduce result type (pass args and kwargs separately)
  TypePtr result_type;
  try {
    result_type = deduce_type_fn(args, kwargs);
  } catch (const std::exception& e) {
    std::string location = span.is_valid() ? " at " + span.to_string() : "";
    throw ValueError(std::string(e.what()) + location);
  }
  INTERNAL_CHECK_SPAN(result_type, span) << "Type deduction failed for '" + op_name + "'";

  // Apply OpMemorySpaceSpec to TileType results that lack memory_space.
  // This ensures the deduced type carries memory_space even when individual
  // type deduction functions omit it (fixes issue #553).
  //
  // Single-output ops: patch the result TileType directly.
  // Tuple-output ops (e.g. tile.gather_compare): patch each TileType element
  // that lacks a memory_space. Heterogeneous-output ops should set
  // memory_space_ inside f_deduce_type rather than relying on this fallback.
  const auto& mem_spec = entry.GetMemorySpec();
  if (mem_spec.has_value() && mem_spec->deduce_output_memory) {
    auto resolve_memory_space = [&]() -> std::optional<MemorySpace> {
      auto resolved = mem_spec->deduce_output_memory(kwargs);
      if (resolved.has_value()) {
        return resolved;
      }
      // Inherit from first tile-typed input
      for (const auto& arg : args) {
        if (auto input_tile = As<TileType>(arg->GetType())) {
          if (input_tile->memory_space_.has_value()) {
            return input_tile->memory_space_;
          }
        }
      }
      return std::nullopt;
    };
    auto apply_memory_space = [](const TileTypePtr& tile_type, MemorySpace space) {
      return std::make_shared<TileType>(tile_type->shape_, tile_type->dtype_, tile_type->memref_,
                                        tile_type->tile_view_, space);
    };

    if (auto tile_type = As<TileType>(result_type)) {
      // Single-output case: result is a TileType.
      if (!tile_type->memory_space_.has_value()) {
        if (auto space = resolve_memory_space(); space.has_value()) {
          result_type = apply_memory_space(tile_type, *space);
        }
      }
    } else if (auto tuple_type = As<TupleType>(result_type)) {
      // Multi-output case: result is a TupleType. Patch every TileType
      // element that is missing a memory_space.
      bool any_missing = false;
      for (const auto& elem_ty : tuple_type->types_) {
        if (auto elem_tile = As<TileType>(elem_ty); elem_tile && !elem_tile->memory_space_.has_value()) {
          any_missing = true;
          break;
        }
      }
      if (any_missing) {
        if (auto space = resolve_memory_space(); space.has_value()) {
          std::vector<TypePtr> new_elems;
          new_elems.reserve(tuple_type->types_.size());
          for (const auto& elem_ty : tuple_type->types_) {
            if (auto elem_tile = As<TileType>(elem_ty); elem_tile && !elem_tile->memory_space_.has_value()) {
              new_elems.push_back(apply_memory_space(elem_tile, *space));
            } else {
              new_elems.push_back(elem_ty);
            }
          }
          result_type = std::make_shared<TupleType>(std::move(new_elems));
        }
      }
    }
  }

  // Create Call with deduced type
  return std::make_shared<Call>(op, args, kwargs, result_type, std::move(span));
}

const OpRegistryEntry& OpRegistry::GetEntry(const std::string& op_name) const {
  auto it = registry_.find(op_name);
  CHECK(it != registry_.end()) << "Operator '" + op_name + "' not found in registry";
  return it->second;
}

OpPtr OpRegistry::GetOp(const std::string& op_name) const {
  auto it = registry_.find(op_name);
  CHECK(it != registry_.end()) << "Operator '" + op_name + "' not found in registry";
  return it->second.GetOp();
}

void OpRegistry::ValidateTileOps() const {
  std::vector<std::string> missing;
  for (const auto& [name, entry] : registry_) {
    if (name.rfind("tile.", 0) != 0) continue;
    if (entry.GetMemorySpec().has_value()) continue;
    missing.push_back(name);
  }
  if (!missing.empty()) {
    std::sort(missing.begin(), missing.end());
    std::string msg =
        "The following tile ops are missing a memory spec "
        "(add set_output_memory/set_input_memory or no_memory_spec()):";
    for (const auto& name : missing) {
      msg += "\n  - " + name;
    }
    throw ValueError(msg);
  }
}

}  // namespace ir
}  // namespace pypto
