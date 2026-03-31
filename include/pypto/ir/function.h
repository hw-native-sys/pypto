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

#ifndef PYPTO_IR_FUNCTION_H_
#define PYPTO_IR_FUNCTION_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/reflection/field_traits.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

/**
 * @brief Function type classification
 *
 * Categorizes functions by their execution context and purpose:
 * - Opaque: Unspecified (default)
 * - Orchestration: Runs on host/AICPU for control flow and dependency analysis
 * - InCore: Sub-graph on specific AICore (unspecialized)
 * - AIC: Cube core kernel (specialized InCore)
 * - AIV: Vector core kernel (specialized InCore)
 * - Group: Co-scheduled group of AIC + AIV kernels
 */
enum class FunctionType : uint8_t {
  Opaque = 0,         ///< Default: unspecified function type
  Orchestration = 1,  ///< Host/AICPU control and coordination
  InCore = 2,         ///< AICore sub-graph execution (unspecialized)
  AIC = 3,            ///< Cube core kernel (specialized InCore)
  AIV = 4,            ///< Vector core kernel (specialized InCore)
  Group = 5           ///< Co-scheduled group of AIC + AIV kernels
};

/**
 * @brief Hierarchy level in the Linqu machine model
 *
 * Levels map bottom-up from individual cores (Level 0) to the global
 * coordinator (Level 9). Multiple enum values may share the same underlying
 * integer when they are readability aliases for the same concept.
 */
enum class Level : uint8_t {
  AIV = 0,         ///< Single AIV (Vector) core
  AIC = 1,         ///< Single AIC (Cube) core
  CORE_GROUP = 2,  ///< Core-group (e.g. 1 AIC + 2 AIV)
  CHIP_DIE = 3,    ///< Chip die (optional in single-die models)
  CHIP = 4,        ///< Chip (UMA)
  HOST = 5,        ///< Host (single OS instance)
  CLUSTER_0 = 6,   ///< Cluster-level-0 (pod)
  CLUSTER_1 = 7,   ///< Cluster-level-1 (supernode)
  CLUSTER_2 = 8,   ///< Cluster-level-2 (cross-rack)
  GLOBAL = 9,      ///< Global coordinator

  // Readability aliases
  L2CACHE = 3,    ///< Alias for CHIP_DIE
  PROCESSOR = 4,  ///< Alias for CHIP
  UMA = 4,        ///< Alias for CHIP
  NODE = 5,       ///< Alias for HOST
  POD = 6,        ///< Alias for CLUSTER_0
  CLOS1 = 7,      ///< Alias for CLUSTER_1
  CLOS2 = 8,      ///< Alias for CLUSTER_2
};

/**
 * @brief Function role at L3-L7 hierarchy levels
 *
 * Distinguishes orchestrators (which build task DAGs and submit work)
 * from workers (which execute concrete compute or data tasks).
 */
enum class Role : uint8_t {
  Orchestrator = 0,  ///< Builds DAG, submits tasks, never computes directly
  Worker = 1,        ///< Executes compute/data tasks, never submits further tasks
};

/**
 * @brief Convert Level to string (primary name)
 */
inline std::string LevelToString(Level level) {
  switch (level) {
    case Level::AIV:
      return "AIV";
    case Level::AIC:
      return "AIC";
    case Level::CORE_GROUP:
      return "CORE_GROUP";
    case Level::CHIP_DIE:
      return "CHIP_DIE";
    case Level::CHIP:
      return "CHIP";
    case Level::HOST:
      return "HOST";
    case Level::CLUSTER_0:
      return "CLUSTER_0";
    case Level::CLUSTER_1:
      return "CLUSTER_1";
    case Level::CLUSTER_2:
      return "CLUSTER_2";
    case Level::GLOBAL:
      return "GLOBAL";
  }
  throw pypto::TypeError("Unknown Level");
}

/**
 * @brief Convert string to Level
 */
inline Level StringToLevel(const std::string& str) {
  static const std::unordered_map<std::string, Level> kMap = {
      {"AIV", Level::AIV},
      {"AIC", Level::AIC},
      {"CORE_GROUP", Level::CORE_GROUP},
      {"CHIP_DIE", Level::CHIP_DIE},
      {"L2CACHE", Level::CHIP_DIE},
      {"CHIP", Level::CHIP},
      {"PROCESSOR", Level::CHIP},
      {"UMA", Level::CHIP},
      {"HOST", Level::HOST},
      {"NODE", Level::HOST},
      {"CLUSTER_0", Level::CLUSTER_0},
      {"POD", Level::CLUSTER_0},
      {"CLUSTER_1", Level::CLUSTER_1},
      {"CLOS1", Level::CLUSTER_1},
      {"CLUSTER_2", Level::CLUSTER_2},
      {"CLOS2", Level::CLUSTER_2},
      {"GLOBAL", Level::GLOBAL},
  };
  auto it = kMap.find(str);
  if (it != kMap.end()) return it->second;
  throw pypto::TypeError("Unknown Level: " + str);
}

/**
 * @brief Map Level enum value to Linqu hierarchy level number (0-7)
 *
 * Multiple Level values may map to the same Linqu level (e.g. AIV, AIC, CORE_GROUP → 0).
 */
inline int LevelToLinquLevel(Level level) {
  switch (level) {
    case Level::AIV:
    case Level::AIC:
    case Level::CORE_GROUP:
      return 0;
    case Level::CHIP_DIE:
      return 1;
    case Level::CHIP:
      return 2;
    case Level::HOST:
      return 3;
    case Level::CLUSTER_0:
      return 4;
    case Level::CLUSTER_1:
      return 5;
    case Level::CLUSTER_2:
      return 6;
    case Level::GLOBAL:
      return 7;
  }
  throw pypto::TypeError("Unknown Level");
}

/**
 * @brief Convert Role to string
 */
inline std::string RoleToString(Role role) {
  switch (role) {
    case Role::Orchestrator:
      return "Orchestrator";
    case Role::Worker:
      return "Worker";
  }
  throw pypto::TypeError("Unknown Role");
}

/**
 * @brief Convert string to Role
 */
inline Role StringToRole(const std::string& str) {
  static const std::unordered_map<std::string, Role> kMap = {
      {"Orchestrator", Role::Orchestrator},
      {"ORCHESTRATOR", Role::Orchestrator},
      {"Worker", Role::Worker},
      {"WORKER", Role::Worker},
  };
  auto it = kMap.find(str);
  if (it != kMap.end()) return it->second;
  throw pypto::TypeError("Unknown Role: " + str);
}

/**
 * @brief Parameter direction classification
 *
 * Models kernel-style parameter directions:
 * - In: Read-only input parameter (default)
 * - Out: Write-only output parameter
 * - InOut: Read-write parameter
 */
enum class ParamDirection : uint8_t {
  In = 0,     ///< Read-only input (default)
  Out = 1,    ///< Write-only output
  InOut = 2,  ///< Read-write input/output
};

/**
 * @brief Convert FunctionType to string
 * @param type The function type
 * @return String representation
 */
inline std::string FunctionTypeToString(FunctionType type) {
  switch (type) {
    case FunctionType::Opaque:
      return "Opaque";
    case FunctionType::Orchestration:
      return "Orchestration";
    case FunctionType::InCore:
      return "InCore";
    case FunctionType::AIC:
      return "AIC";
    case FunctionType::AIV:
      return "AIV";
    case FunctionType::Group:
      return "Group";
  }
  throw pypto::TypeError("Unknown FunctionType");
}

/**
 * @brief Check if a FunctionType is an InCore variant (InCore, AIC, or AIV)
 */
inline bool IsInCoreType(FunctionType type) {
  return type == FunctionType::InCore || type == FunctionType::AIC || type == FunctionType::AIV;
}

/**
 * @brief Convert string to FunctionType
 * @param str String representation
 * @return FunctionType enum value
 * @throws pypto::TypeError if string is not recognized
 */
inline FunctionType StringToFunctionType(const std::string& str) {
  if (str == "Opaque") {
    return FunctionType::Opaque;
  } else if (str == "Orchestration") {
    return FunctionType::Orchestration;
  } else if (str == "InCore") {
    return FunctionType::InCore;
  } else if (str == "AIC") {
    return FunctionType::AIC;
  } else if (str == "AIV") {
    return FunctionType::AIV;
  } else if (str == "Group") {
    return FunctionType::Group;
  } else {
    throw pypto::TypeError("Unknown FunctionType: " + str);
  }
}

/**
 * @brief Convert ParamDirection to string
 * @param dir The parameter direction
 * @return String representation ("In", "Out", or "InOut")
 */
inline std::string ParamDirectionToString(ParamDirection dir) {
  switch (dir) {
    case ParamDirection::In:
      return "In";
    case ParamDirection::Out:
      return "Out";
    case ParamDirection::InOut:
      return "InOut";
  }
  throw pypto::TypeError("Unknown ParamDirection");
}

/**
 * @brief Convert string to ParamDirection
 * @param str String representation
 * @return ParamDirection enum value
 * @throws pypto::TypeError if string is not recognized
 */
inline ParamDirection StringToParamDirection(const std::string& str) {
  if (str == "In") {
    return ParamDirection::In;
  } else if (str == "Out") {
    return ParamDirection::Out;
  } else if (str == "InOut") {
    return ParamDirection::InOut;
  } else {
    throw pypto::TypeError("Unknown ParamDirection: " + str);
  }
}

/**
 * @brief Function definition
 *
 * Represents a complete function definition with name, parameters, return types, and body.
 * Functions are immutable IR nodes.
 *
 * Optional level_ and role_ fields carry hierarchy metadata for distributed programs.
 * When unset (nullopt), the function uses legacy FunctionType-only semantics.
 */
class Function : public IRNode {
 public:
  /**
   * @brief Create a function definition
   *
   * @param name Function name
   * @param params Parameter variables with directions
   * @param return_types Return types
   * @param body Function body statement (use SeqStmts for multiple statements)
   * @param span Source location
   * @param type Function type (default: Opaque)
   * @param level Hierarchy level (default: nullopt — unspecified)
   * @param role Function role (default: nullopt)
   * @param split Split mode for cross-core transfer (default: nullopt)
   */
  Function(std::string name, std::vector<VarPtr> params, std::vector<ParamDirection> param_directions,
           std::vector<TypePtr> return_types, StmtPtr body, Span span,
           FunctionType type = FunctionType::Opaque, std::optional<Level> level = std::nullopt,
           std::optional<Role> role = std::nullopt, std::optional<SplitMode> split = std::nullopt)
      : IRNode(std::move(span)),
        name_(std::move(name)),
        params_(std::move(params)),
        param_directions_(std::move(param_directions)),
        return_types_(std::move(return_types)),
        body_(std::move(body)),
        func_type_(type),
        level_(level),
        role_(role),
        split_(split) {
    CHECK(params_.size() == param_directions_.size())
        << "params and param_directions must have same size, got " << params_.size() << " vs "
        << param_directions_.size();
  }

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::Function; }
  [[nodiscard]] std::string TypeName() const override { return "Function"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors (params as DEF field, func_type, level, role, split,
   *         return_types and body as USUAL fields, name as an IGNORE field)
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(
        IRNode::GetFieldDescriptors(),
        std::make_tuple(reflection::DefField(&Function::params_, "params"),
                        reflection::UsualField(&Function::param_directions_, "param_directions"),
                        reflection::UsualField(&Function::func_type_, "func_type"),
                        reflection::UsualField(&Function::level_, "level"),
                        reflection::UsualField(&Function::role_, "role"),
                        reflection::UsualField(&Function::split_, "split"),
                        reflection::UsualField(&Function::return_types_, "return_types"),
                        reflection::UsualField(&Function::body_, "body"),
                        reflection::IgnoreField(&Function::name_, "name")));
  }

 public:
  std::string name_;                // Function name
  FunctionType func_type_;          // Function type (Opaque, Orchestration, InCore, AIC, AIV, or Group)
  std::optional<Level> level_;      // Hierarchy level (nullopt = infer from func_type)
  std::optional<Role> role_;        // Function role (nullopt = default per level)
  std::optional<SplitMode> split_;  // Split mode for cross-core transfer (nullopt = no split)
  std::vector<VarPtr> params_;      // Parameter variables
  std::vector<ParamDirection> param_directions_;  // Parameter directions (same length as params_)
  std::vector<TypePtr> return_types_;             // Return types
  StmtPtr body_;                                  // Function body statement
};

using FunctionPtr = std::shared_ptr<const Function>;

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_FUNCTION_H_
