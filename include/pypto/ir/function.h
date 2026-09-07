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

#include <algorithm>
#include <any>
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
 * - Spmd: SPMD data-parallel dispatch wrapper
 * - Inline: Whole-body substitution at every call site by the InlineFunctions
 *   pass. Eliminated from the program before any other pass runs.
 * - Graph: A callable orchestration fragment. Its body is orchestration code,
 *   but each call site is a single task launch that the host_build_graph
 *   runtime records once and replays thereafter. The runtime identifies the
 *   recording by the address of the emitted C++ function, so one Graph
 *   function is one recorded topology with no key to keep in sync.
 */
enum class FunctionType : uint8_t {
  Opaque = 0,         ///< Default: unspecified function type
  Orchestration = 1,  ///< Host/AICPU control and coordination
  InCore = 2,         ///< AICore sub-graph execution (unspecialized)
  AIC = 3,            ///< Cube core kernel (specialized InCore)
  AIV = 4,            ///< Vector core kernel (specialized InCore)
  Group = 5,          ///< Co-scheduled group of AIC + AIV kernels
  Spmd = 6,           ///< SPMD data-parallel dispatch
  Inline = 7,         ///< Whole-body substitution at every call site
  Graph = 8           ///< Recordable/replayable orchestration fragment
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

  UNDEFINED = 255
};

/**
 * @brief Function role at L3-L7 hierarchy levels
 *
 * Distinguishes orchestrators (which build task DAGs and submit work) from
 * sub-workers (which execute concrete compute or data tasks dispatched by an
 * orchestrator at the same level).
 */
enum class Role : uint8_t {
  Orchestrator = 0,  ///< Builds DAG, submits tasks, never computes directly
  SubWorker = 1,     ///< Executes compute/data tasks dispatched by the orchestrator at the same level
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
    case Level::UNDEFINED:
      break;
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
    case Level::UNDEFINED:
      break;
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
    case Role::SubWorker:
      return "SubWorker";
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
      {"SubWorker", Role::SubWorker},
      {"SUBWORKER", Role::SubWorker},
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
    case FunctionType::Spmd:
      return "Spmd";
    case FunctionType::Inline:
      return "Inline";
    case FunctionType::Graph:
      return "Graph";
  }
  throw pypto::TypeError("Unknown FunctionType");
}

/**
 * @brief Convert FunctionType to level
 * @param type The function type
 * @return Level
 */
inline Level FunctionTypeToLevel(FunctionType type) {
  switch (type) {
    case FunctionType::Orchestration:
    case FunctionType::Graph:
      // A Graph body is chip-level orchestration; only its call site differs.
      return Level::CHIP;
    case FunctionType::InCore:
      return Level::CHIP_DIE;
    case FunctionType::AIC:
      return Level::AIC;
    case FunctionType::AIV:
      return Level::AIV;
    case FunctionType::Group:
      return Level::CORE_GROUP;
    default:
      return Level::UNDEFINED;
  }
}

/**
 * @brief Check if a FunctionType is an InCore variant (InCore, AIC, or AIV)
 */
inline bool IsInCoreType(FunctionType type) {
  return type == FunctionType::InCore || type == FunctionType::AIC || type == FunctionType::AIV;
}

/**
 * @brief Check if a FunctionType is a scope wrapper (Group or Spmd)
 *
 * Wrappers are synthesised by the scope outliners and forward their params 1:1
 * to an inner kernel call, so passes routinely treat the two kinds alike
 * (direction propagation, return lineage, return-order normalization).
 */
inline bool IsWrapperType(FunctionType type) {
  return type == FunctionType::Group || type == FunctionType::Spmd;
}

/**
 * @brief Check if a FunctionType has an orchestration body (Orchestration or Graph)
 *
 * Orchestration and Graph bodies are both host/AICPU task-orchestration code, so
 * every pass that processes a function *because it orchestrates tasks* must
 * accept both. Prefer this over `== FunctionType::Orchestration`, which silently
 * skips Graph bodies and produces a program missing whatever that pass
 * contributes.
 *
 * The exception is code that means "the single compilation entry point" rather
 * than "an orchestration body" — that must stay a strict comparison, since a
 * Graph is called by the entry, never the entry itself.
 */
inline bool IsOrchestrationLike(FunctionType type) {
  return type == FunctionType::Orchestration || type == FunctionType::Graph;
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
  } else if (str == "Spmd") {
    return FunctionType::Spmd;
  } else if (str == "Inline") {
    return FunctionType::Inline;
  } else if (str == "Graph") {
    return FunctionType::Graph;
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
 * @brief Reserved Function attr key marking a Group whose body was a
 * ``pl.cluster(): with pl.spmd(...)`` region.
 *
 * Value type: ``bool`` (emitted only when true). Stamped by
 * ``OutlineClusterScopes``' ``UnwrapNestedSpmd`` when it replaces the nested
 * Spmd scope with its body and moves the launch spec onto the Group's dispatch
 * (``kAttrCoreNum`` there — the spec cannot live on the Group, whose scope does
 * not bind the caller-local Vars it references).
 *
 * This marker is what remains function-scoped, and legitimately so: it states a
 * property of THIS function's body, references nothing outside it, and is a
 * plain bool that round-trips through the decorator. It tells a launch-site
 * consumer that dispatching this Group launches the kernels its body calls,
 * rather than the Group itself as a mixed kernel — the distinction the
 * occupancy verifier used to draw from the presence of a ``core_num`` attr.
 */
inline constexpr const char* kAttrSpmdUnwrapped = "spmd_unwrapped";

/**
 * @brief Reserved Function attr key marking an AIV kernel that runs on BOTH
 * vector sub-lanes of a mixed kernel.
 *
 * Value type: ``bool``. Written by ``LowerAutoVectorSplit`` (pass 23) and
 * ``SplitVectorKernel`` (pass 26) onto the AIV lane, and by
 * ``ExpandMixedKernel`` (pass 24) for the backend-inferred no-split case
 * (``BackendHandler::RequiresNoSplitDualAivDispatch``). Read by PTO codegen
 * (``PTOCodegen::IsDualAivDispatchFunction`` — subblock-aware emission),
 * orchestration codegen (both-lanes MixedKernel dispatch) and
 * ``VerifyHardSyncallOccupancy`` (a dual-dispatched AIV kernel is a mixed-kernel
 * lane, not a standalone AIV launch). Never stripped — it survives to codegen.
 *
 * Mode-agnostic on purpose: it states only "both AIV sub-lanes execute this
 * body". The per-op split geometry rides the ``split`` attr / per-op ints, so a
 * multi-mode ``pl.split_aiv`` function carries this marker with no
 * function-level ``split`` mode at all.
 */
inline constexpr const char* kAttrDualAivDispatch = "dual_aiv_dispatch";

/**
 * @brief Reserved Function attr key marking an InCore function outlined from a
 * scope that held ``pl.split_aiv`` region(s).
 *
 * Value type: ``bool``. Written by ``ScopeOutliner`` (pass 8) when it outlines a
 * CORE_GROUP scope containing ``SplitAivScopeStmt`` regions, and re-stamped by
 * ``LowerAutoVectorSplit`` (pass 23) on the functions it lowers. Read by
 * ``SplitVectorKernel`` (pass 26, to stamp ``dual_aiv_dispatch`` without
 * re-halving an already-lowered body), ``MemoryReuse`` (pass 36 — it gates the
 * Ascend910B ``tile.load`` + ``tpop_from_aic`` in-place hazard guard) and
 * ``VerifyAivSplit`` (provenance for the boundary-op checks). Never stripped.
 *
 * ``MemoryReuse`` keys on this marker rather than on ``Function::GetSplitMode``
 * precisely because a multi-mode region function has no single function-level
 * mode once pass 23 has lowered and erased the per-region ones — dropping the
 * marker there silently disables a hardware-correctness guard.
 */
inline constexpr const char* kAttrSplitAiv = "split_aiv";

/**
 * @brief Reserved Function attr key recording that ``pl.split_aiv`` regions were
 * already transpose-hazard-checked per region.
 *
 * Value type: ``bool``. Written by ``LowerAutoVectorSplit`` (pass 23), which
 * validates each region against its own unambiguous mode. Read by
 * ``ExpandMixedKernel`` (pass 24) to skip its single-function-mode transpose
 * check, which would otherwise mis-check a multi-mode function against whichever
 * mode happened to be stamped function-level. Never stripped.
 */
inline constexpr const char* kAttrSplitAivRegionValidated = "split_aiv_region_validated";

/**
 * @brief Reserved Function attr key naming a hand-written external C++ kernel
 * source.
 *
 * Value type: ``std::string`` (an absolute path). Written by the
 * ``@pl.function(external_source=...)`` decorator; the function's DSL body is
 * then an empty ``...``. Read by ``ExpandMixedKernel`` (external members cannot
 * be ABI-normalised or inferred into dual dispatch),
 * ``VerifyReturnParamsExplicit`` (a declaration-only body legitimately has no
 * ``ReturnStmt``), the Python printer, and the backend, which compiles the named
 * ``.cpp`` in place of PyPTO codegen. Never stripped.
 *
 * Decorator-only: the printer emits it as an ``@pl.function(external_source=...)``
 * keyword rather than a ``pl.func_attr({...})`` body prologue, because the parser
 * must read it before it walks the body. Kept in sync with the parser's
 * ``_DECORATOR_ONLY_FUNC_ATTRS``.
 */
inline constexpr const char* kAttrExternalSource = "external_source";

/**
 * @brief Reserved Function attr key opting a function out of automatic
 * ``RuntimeScopeStmt`` materialization.
 *
 * Value type: ``bool``; **absent means true**, so only the opt-out (``false``)
 * is ever stored. Written by the ``@pl.function(auto_scope=False)`` decorator and
 * by ``MaterializeRuntimeScopes`` (pass 48), which stamps ``false`` on the
 * functions it has already processed. Read by that same pass (idempotence),
 * ``AutoDeriveTaskDependencies`` (pass 41), ``VerifyRuntimeScopesMaterialized``
 * and the Python printer.
 *
 * Decorator-only, for the same reason as ``kAttrExternalSource`` — see there.
 */
inline constexpr const char* kAttrAutoScope = "auto_scope";

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
   * @param attrs Function-level attributes (default: empty)
   * @param requires_runtime_binding True for SubWorkers declared with an
   *        abstract (`...`) body — the implementation is supplied at runtime
   *        rather than captured at compile time (default: false)
   */
  Function(std::string name, std::vector<VarPtr> params, std::vector<ParamDirection> param_directions,
           std::vector<TypePtr> return_types, StmtPtr body, Span span,
           FunctionType type = FunctionType::Opaque, std::optional<Level> level = std::nullopt,
           std::optional<Role> role = std::nullopt, std::vector<std::pair<std::string, std::any>> attrs = {},
           bool requires_runtime_binding = false)
      : IRNode(std::move(span)),
        name_(std::move(name)),
        params_(std::move(params)),
        param_directions_(std::move(param_directions)),
        return_types_(std::move(return_types)),
        body_(std::move(body)),
        func_type_(type),
        level_(level),
        role_(role),
        attrs_(std::move(attrs)),
        requires_runtime_binding_(requires_runtime_binding) {
    CHECK(params_.size() == param_directions_.size())
        << "params and param_directions must have same size, got " << params_.size() << " vs "
        << param_directions_.size();
    if (IsInCoreType(func_type_) || func_type_ == FunctionType::Group || IsOrchestrationLike(func_type_)) {
      Level derived_level = FunctionTypeToLevel(func_type_);
      // A Graph body orchestrates tasks, so it is an Orchestrator like any other
      // orchestration body. Code that must single out the compilation *entry*
      // therefore cannot key on the role alone — see IsChipOrch in
      // materialize_comm_domain_scopes_pass.cpp, which excludes Graph explicitly.
      Role derived_role = IsOrchestrationLike(func_type_) ? Role::Orchestrator : Role::SubWorker;
      if (level_.has_value()) {
        CHECK(*level_ == derived_level)
            << "Function '" << name_ << "' has func_type=" << FunctionTypeToString(func_type_)
            << " which implies level=" << LevelToString(derived_level)
            << ", but explicit level=" << LevelToString(*level_) << " was provided";
      } else {
        level_ = derived_level;
      }
      if (role_.has_value()) {
        CHECK(*role_ == derived_role)
            << "Function '" << name_ << "' has func_type=" << FunctionTypeToString(func_type_)
            << " which implies role=" << RoleToString(derived_role)
            << ", but explicit role=" << RoleToString(*role_) << " was provided";
      } else {
        role_ = derived_role;
      }
    }
  }

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::Function; }
  [[nodiscard]] std::string TypeName() const override { return "Function"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors (params as DEF field, func_type, level, role, attrs,
   *         return_types and body as USUAL fields, name as an IGNORE field)
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(
        IRNode::GetFieldDescriptors(),
        std::make_tuple(
            reflection::DefField(&Function::params_, "params"),
            reflection::UsualField(&Function::param_directions_, "param_directions"),
            reflection::UsualField(&Function::func_type_, "func_type"),
            reflection::UsualField(&Function::level_, "level"),
            reflection::UsualField(&Function::role_, "role"),
            reflection::UsualField(&Function::attrs_, "attrs"),
            reflection::UsualField(&Function::requires_runtime_binding_, "requires_runtime_binding"),
            reflection::UsualField(&Function::return_types_, "return_types"),
            reflection::UsualField(&Function::body_, "body"),
            reflection::IgnoreField(&Function::name_, "name")));
  }

 public:
  std::string name_;                                     // Function name
  FunctionType func_type_;                               // Function type (see the FunctionType enum)
  std::optional<Level> level_;                           // Hierarchy level (nullopt = infer from func_type)
  std::optional<Role> role_;                             // Function role (nullopt = default per level)
  std::vector<std::pair<std::string, std::any>> attrs_;  // Function-level attributes (key-value metadata)
  bool requires_runtime_binding_ = false;  // SubWorker with abstract (`...`) body: impl bound at runtime
  std::vector<VarPtr> params_;             // Parameter variables
  std::vector<ParamDirection> param_directions_;  // Parameter directions (same length as params_)
  std::vector<TypePtr> return_types_;             // Return types
  StmtPtr body_;                                  // Function body statement

  /**
   * @brief Get a typed attribute value
   * @tparam T Expected type of the attribute value
   * @param key Attribute key
   * @param default_value Default value if key doesn't exist
   * @return The attribute value or default
   */
  template <typename T>
  [[nodiscard]] T GetAttr(const std::string& key, const T& default_value = T{}) const {
    for (const auto& [k, v] : attrs_) {
      if (k == key) return AnyCast<T>(v, "func attr key: " + key);
    }
    return default_value;
  }

  /**
   * @brief Check if an attribute exists
   * @param key Attribute key
   * @return true if the attribute exists
   */
  [[nodiscard]] bool HasAttr(const std::string& key) const {
    return std::any_of(attrs_.begin(), attrs_.end(), [&key](const auto& pair) { return pair.first == key; });
  }

  /**
   * @brief Get all attributes
   * @return Vector of key-value attribute pairs
   */
  [[nodiscard]] const std::vector<std::pair<std::string, std::any>>& GetAttrs() const { return attrs_; }

  /**
   * @brief Convenience: extract SplitMode from attrs
   * @return SplitMode if "split" attr is set and non-zero, nullopt otherwise
   */
  [[nodiscard]] std::optional<SplitMode> GetSplitMode() const {
    if (!HasAttr("split")) return std::nullopt;
    int val = GetAttr<int>("split", 0);
    if (val == 0) return std::nullopt;
    CHECK(val >= 0 && val <= static_cast<int>(SplitMode::LeftRight))
        << "Invalid split mode value in attrs: " << val;
    return static_cast<SplitMode>(val);
  }
};

using FunctionPtr = std::shared_ptr<const Function>;

/**
 * @brief Null-safe overload of IsOrchestrationLike for a function handle
 *
 * A null handle is not orchestration-like, so callers that already hold a
 * possibly-null FunctionPtr need no separate guard.
 */
inline bool IsOrchestrationLike(const FunctionPtr& func) {
  return func != nullptr && IsOrchestrationLike(func->func_type_);
}

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_FUNCTION_H_
