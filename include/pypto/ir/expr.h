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

#ifndef PYPTO_IR_EXPR_H_
#define PYPTO_IR_EXPR_H_

#include <any>
#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeindex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/ir/core.h"
#include "pypto/ir/reflection/field_traits.h"
#include "pypto/ir/span.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

/**
 * @brief Base class for all expressions in the IR
 *
 * This is the root base class for all expression types (scalar, tensor, etc).
 * Expressions represent computations that produce values.
 * All expressions are immutable.
 */
class Expr : public IRNode {
 protected:
  TypePtr type_;  // Type of the expression result

 public:
  /**
   * @brief Create an expression
   *
   * @param span Source location
   * @param type Type of the expression result (defaults to UnknownType)
   */
  explicit Expr(Span s, TypePtr type = GetUnknownType()) : IRNode(std::move(s)), type_(std::move(type)) {}
  ~Expr() override = default;

  /**
   * @brief Get the type name of this expression
   *
   * @return Human-readable type name (e.g., "ScalarExpr", "Var", "Call")
   */
  [[nodiscard]] std::string TypeName() const override { return "Expr"; }

  /**
   * @brief Get the type of this expression
   *
   * @return Type pointer of the expression result
   */
  [[nodiscard]] const TypePtr& GetType() const { return type_; }

  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(IRNode::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&Expr::type_, "type")));
  }
};

using ExprPtr = std::shared_ptr<const Expr>;

// Forward declarations for enum types defined elsewhere
enum class MemorySpace;
enum class PadValue;

/**
 * @brief Base class for operations/functions
 *
 * Represents callable operations in the IR.
 * Stores the schema of allowed kwargs (key -> expected type mapping).
 * Actual kwarg values are stored per-Call instance in Call::kwargs_.
 */
class Op {
 public:
  std::string name_;

  explicit Op(std::string name) : name_(std::move(name)) {}
  virtual ~Op() = default;

  /**
   * @brief Register an allowed kwarg with its expected type
   *
   * Defines that this operator accepts a kwarg with the given key and type.
   * This is used for validation when creating Call expressions.
   *
   * Only specific types are allowed: bool, int, std::string, double, DataType, MemorySpace,
   * TensorLayout, and PadValue
   * This is enforced at compile-time via static_assert.
   *
   * @tparam T Expected type of the kwarg value (must be one of the allowed types)
   * @param key Kwarg key (string identifier)
   */
  template <typename T>
  void SetAttrType(const std::string& key) const {
    // Compile-time check: only allow specific types
    static_assert(
        std::is_same_v<T, bool> || std::is_same_v<T, int> || std::is_same_v<T, std::string> ||
            std::is_same_v<T, double> || std::is_same_v<T, DataType> || std::is_same_v<T, MemorySpace> ||
            std::is_same_v<T, TensorLayout> || std::is_same_v<T, TileLayout> || std::is_same_v<T, PadValue>,
        "SetAttrType only accepts: bool, int, std::string, double, DataType, MemorySpace, TensorLayout, "
        "TileLayout, PadValue");

    attrs_.emplace(key, std::type_index(typeid(T)));
  }

  /**
   * @brief Get the expected type for a kwarg
   *
   * @param key Kwarg key
   * @return type_index of the expected type
   * @throws pypto::ValueError if kwarg is not registered
   */
  [[nodiscard]] std::type_index GetAttrType(const std::string& key) const {
    auto it = attrs_.find(key);
    if (it == attrs_.end()) {
      throw pypto::ValueError("Attribute '" + key + "' not found in operator '" + name_ + "'");
    }
    return it->second;
  }

  /**
   * @brief Check if a kwarg is registered
   *
   * @param key Kwarg key
   * @return true if the kwarg is registered
   */
  [[nodiscard]] bool HasAttr(const std::string& key) const { return attrs_.find(key) != attrs_.end(); }

  /**
   * @brief Get all registered kwarg keys
   *
   * @return Vector of all kwarg keys
   */
  [[nodiscard]] std::vector<std::string> GetAttrKeys() const {
    std::vector<std::string> keys;
    keys.reserve(attrs_.size());
    for (const auto& pair : attrs_) {
      keys.push_back(pair.first);
    }
    return keys;
  }

  /**
   * @brief Get all registered kwargs as a map
   *
   * @return Map of kwarg keys to expected types
   */
  [[nodiscard]] const std::unordered_map<std::string, std::type_index>& GetAttrs() const { return attrs_; }

  [[nodiscard]] virtual ObjectKind GetKind() const { return ObjectKind::Op; }
  [[nodiscard]] virtual std::string TypeName() const { return "Op"; }

 private:
  mutable std::unordered_map<std::string, std::type_index> attrs_;  ///< Kwarg schema (key -> type)
};

using OpPtr = std::shared_ptr<const Op>;

/**
 * @brief Global variable reference for functions in a program
 *
 * Represents a reference to a function in the program's global scope.
 * Can be used as an operation in Call expressions to call functions within the same program.
 * The name of the GlobalVar should match the name of the function it references.
 */
class GlobalVar : public Op {
 public:
  explicit GlobalVar(std::string name) : Op(std::move(name)) {}
  ~GlobalVar() override = default;
  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::GlobalVar; }
  [[nodiscard]] std::string TypeName() const override { return "GlobalVar"; }
};

using GlobalVarPtr = std::shared_ptr<const GlobalVar>;

/**
 * @brief Custom comparator for ordering GlobalVarPtr by name
 *
 * Used in std::map to maintain deterministic ordering of functions in a Program.
 * Ensures consistent structural equality and hashing.
 */
struct GlobalVarPtrLess {
  bool operator()(const GlobalVarPtr& lhs, const GlobalVarPtr& rhs) const { return lhs->name_ < rhs->name_; }
};

/**
 * @brief Variable reference expression
 *
 * Represents a reference to a named variable.
 * Can represent both scalar and tensor variables based on its type.
 */
class Var : public Expr {
 public:
  std::string name_hint_;

  /**
   * @brief Create a variable reference
   *
   * @param name_hint Variable name hint (cosmetic label, not an identifier)
   * @param type Type of the variable (ScalarType, TensorType, or TileType)
   *             Memory reference information is stored in ShapedType for Tensor/Tile types
   * @param span Source location
   * @return Shared pointer to const Var expression
   */
  Var(std::string name_hint, TypePtr type, Span span)
      : Expr(std::move(span), std::move(type)),
        name_hint_(std::move(name_hint)),
        unique_id_(next_unique_id_.fetch_add(1, std::memory_order_relaxed)) {}

  /**
   * @brief Get the unique identity of this variable
   *
   * Monotonically increasing ID assigned at construction, providing
   * deterministic identity that is stable for the lifetime of the process.
   *
   * @return Process-unique identifier for this variable instance
   */
  [[nodiscard]] uint64_t UniqueId() const { return unique_id_; }

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::Var; }
  [[nodiscard]] std::string TypeName() const override { return "Var"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors (name_hint_ as IGNORE field)
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Expr::GetFieldDescriptors(),
                          std::make_tuple(reflection::IgnoreField(&Var::name_hint_, "name_hint")));
  }

 private:
  static inline std::atomic<uint64_t> next_unique_id_{0};
  uint64_t unique_id_;
};

using VarPtr = std::shared_ptr<const Var>;

/**
 * @brief Iteration argument variable
 *
 * Represents an iteration argument (loop-carried value) in for loops.
 * IterArgs implement SSA-style loop-carried dependencies where values are
 * carried from one iteration to the next via yield statements.
 *
 * **Scoping Rules:**
 * - IterArg variables are scoped to the loop body only
 * - Cannot be directly accessed outside the loop
 * - Must use return_vars to expose final values after the loop
 *
 * **Usage Pattern:**
 * 1. Create IterArg with initial value
 * 2. Use in ForStmt's iter_args list
 * 3. Update via YieldStmt in loop body
 * 4. Capture final value in ForStmt's return_vars
 *
 * @example
 * // for i, (sum,) in pl.range(0, n, 1, init_values=(0,)):
 * //     sum = pl.yield_(sum + i)
 * // sum_final = sum
 * auto sum_iter = std::make_shared<IterArg>("sum", type, init_val, span);
 * auto sum_final = std::make_shared<Var>("sum_final", type, span);
 * auto for_stmt = std::make_shared<ForStmt>(
 *     i, start, stop, step,
 *     std::vector{sum_iter},  // iter_args (loop-scoped)
 *     body,
 *     std::vector{sum_final}, // return_vars (accessible after loop)
 *     span
 * );
 */
class IterArg : public Var {
 public:
  ExprPtr initValue_;  // Initial value expression for first iteration

  /**
   * @brief Create an iteration argument
   *
   * @param name_hint Variable name hint (scoped to loop body)
   * @param type Type of the variable (ScalarType, TensorType, or TileType)
   *             Memory reference information is stored in ShapedType for Tensor/Tile types
   * @param initValue Initial value expression for first iteration
   * @param span Source location
   */
  IterArg(std::string name_hint, TypePtr type, ExprPtr initValue, Span span)
      : Var(std::move(name_hint), std::move(type), std::move(span)), initValue_(std::move(initValue)) {}

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::IterArg; }
  [[nodiscard]] std::string TypeName() const override { return "IterArg"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors (initValue_ as USUAL field)
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Var::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&IterArg::initValue_, "initValue")));
  }
};

using IterArgPtr = std::shared_ptr<const IterArg>;

/**
 * @brief Call-site argument direction classification
 *
 * Models the runtime task-submission semantics for each positional argument of a Call.
 * Mirrors the runtime TensorArgType enum (INPUT/OUTPUT/INOUT/OUTPUT_EXISTING/NO_DEP)
 * one-to-one, plus a Scalar tag for non-tensor arguments.
 *
 * Distinct from `ParamDirection` (which lives on the callee Function and describes
 * the function-signature contract — "I read/write this parameter"). `ArgDirection`
 * describes the call-site task behavior — "this submission establishes these
 * dependencies and uses this memory ownership model".
 *
 * - Input:           Read-only input → runtime add_input, TensorArgType::INPUT
 * - Output:          Runtime allocates a new buffer → add_output(create_info),
 *                    TensorArgType::OUTPUT
 * - InOut:           Read-then-write → add_inout, TensorArgType::INOUT
 * - OutputExisting:  Write into an already-existing tensor (skips OverlapMap,
 *                    keeps creator dependency) → add_output(tensor),
 *                    TensorArgType::OUTPUT_EXISTING
 * - NoDep:           No-dependency existing tensor (skips OverlapMap lookup
 *                    *and* publish). Legal regardless of the callee's
 *                    ``ParamDirection`` (In / Out / InOut): the user opts the
 *                    slot out of auto-dep tracking for both reads and writes
 *                    by guaranteeing disjoint access out-of-band — e.g. paged
 *                    KV-cache writes whose offset is data-dependent. → add_no_dep,
 *                    TensorArgType::NO_DEP
 * - Scalar:          Non-tensor scalar argument → add_scalar (must follow all
 *                    tensors in the runtime arg list)
 */
enum class ArgDirection : uint8_t {
  Input = 0,           ///< Read-only input (default for tensors)
  Output = 1,          ///< Runtime-allocated output buffer
  InOut = 2,           ///< Read-then-write
  OutputExisting = 3,  ///< Write-only into an existing tensor
  NoDep = 4,           ///< No-dependency existing tensor
  Scalar = 5,          ///< Scalar (non-tensor) argument
};

/**
 * @brief Convert ArgDirection to string
 */
inline std::string ArgDirectionToString(ArgDirection dir) {
  switch (dir) {
    case ArgDirection::Input:
      return "Input";
    case ArgDirection::Output:
      return "Output";
    case ArgDirection::InOut:
      return "InOut";
    case ArgDirection::OutputExisting:
      return "OutputExisting";
    case ArgDirection::NoDep:
      return "NoDep";
    case ArgDirection::Scalar:
      return "Scalar";
  }
  throw pypto::TypeError("Unknown ArgDirection");
}

/**
 * @brief Convert string to ArgDirection
 */
inline ArgDirection StringToArgDirection(const std::string& str) {
  if (str == "Input") {
    return ArgDirection::Input;
  } else if (str == "Output") {
    return ArgDirection::Output;
  } else if (str == "InOut") {
    return ArgDirection::InOut;
  } else if (str == "OutputExisting") {
    return ArgDirection::OutputExisting;
  } else if (str == "NoDep") {
    return ArgDirection::NoDep;
  } else if (str == "Scalar") {
    return ArgDirection::Scalar;
  }
  throw pypto::TypeError("Unknown ArgDirection: " + str);
}

/**
 * @brief Reserved attrs key for call-site argument directions
 *
 * The value stored under this key is a `std::vector<ArgDirection>` whose length
 * matches `Call::args_`. See `Call::GetArgDirections` for details.
 */
inline constexpr const char* kAttrArgDirections = "arg_directions";

/**
 * @brief Function call expression
 *
 * Represents a function call with an operation and arguments.
 * Can accept any Expr as arguments, not just scalar expressions.
 * Supports keyword arguments (kwargs) for operator metadata, plus a generic
 * `attrs_` map for compiler-internal node metadata (e.g., call-site argument
 * directions). Both containers share the `vector<pair<string, any>>` shape so
 * they reuse the same serialization / structural-comparison machinery.
 *
 * Call-site argument directions are stored as an optional attr under the
 * reserved key `kAttrArgDirections` (value type: `std::vector<ArgDirection>`).
 * Absence means legacy / pre-DeriveCallDirections state; consumers that require
 * resolved call-site behavior (codegen, runtime task submission) must run
 * DeriveCallDirections before they observe the IR.
 */
class Call : public Expr {
 public:
  OpPtr op_;                                              // Operation/function
  std::vector<ExprPtr> args_;                             // Positional arguments
  std::vector<std::pair<std::string, std::any>> attrs_;   // Compiler-internal metadata (ordered)
  std::vector<std::pair<std::string, std::any>> kwargs_;  // Keyword arguments (metadata, ordered)

  /**
   * @brief Create a function call expression
   *
   * @param op Operation/function to call
   * @param args List of argument expressions
   * @param span Source location
   */
  Call(OpPtr op, std::vector<ExprPtr> args, Span span)
      : Expr(std::move(span)), op_(std::move(op)), args_(std::move(args)), attrs_(), kwargs_() {}

  /**
   * @brief Create a function call expression with explicit type
   *
   * @param op Operation/function to call
   * @param args List of argument expressions
   * @param type Result type of the call
   * @param span Source location
   */
  Call(OpPtr op, std::vector<ExprPtr> args, TypePtr type, Span span)
      : Expr(std::move(span), std::move(type)),
        op_(std::move(op)),
        args_(std::move(args)),
        attrs_(),
        kwargs_() {}

  /**
   * @brief Create a function call expression with kwargs
   *
   * @param op Operation/function to call
   * @param args List of argument expressions
   * @param kwargs Keyword arguments (metadata)
   * @param span Source location
   */
  Call(OpPtr op, std::vector<ExprPtr> args, std::vector<std::pair<std::string, std::any>> kwargs, Span span)
      : Expr(std::move(span)),
        op_(std::move(op)),
        args_(std::move(args)),
        attrs_(),
        kwargs_(std::move(kwargs)) {}

  /**
   * @brief Create a function call expression with kwargs and explicit type
   *
   * @param op Operation/function to call
   * @param args List of argument expressions
   * @param kwargs Keyword arguments (metadata)
   * @param type Result type of the call
   * @param span Source location
   */
  Call(OpPtr op, std::vector<ExprPtr> args, std::vector<std::pair<std::string, std::any>> kwargs,
       TypePtr type, Span span)
      : Expr(std::move(span), std::move(type)),
        op_(std::move(op)),
        args_(std::move(args)),
        attrs_(),
        kwargs_(std::move(kwargs)) {}

  /**
   * @brief Create a function call expression with attrs, kwargs, and explicit type
   *
   * @param op Operation/function to call
   * @param args List of argument expressions
   * @param kwargs Keyword arguments (metadata)
   * @param attrs Compiler-internal metadata (e.g., reserved key `arg_directions`)
   * @param type Result type of the call
   * @param span Source location
   *
   * Validates that, when present, `attrs[kAttrArgDirections]` is a
   * `std::vector<ArgDirection>` with the same length as `args`.
   */
  Call(OpPtr op, std::vector<ExprPtr> args, std::vector<std::pair<std::string, std::any>> kwargs,
       std::vector<std::pair<std::string, std::any>> attrs, TypePtr type, Span span)
      : Expr(std::move(span), std::move(type)),
        op_(std::move(op)),
        args_(std::move(args)),
        attrs_(std::move(attrs)),
        kwargs_(std::move(kwargs)) {
    ValidateArgDirectionsAttr();
  }

  /**
   * @brief Get a kwarg value with type checking
   *
   * @tparam T Type of the kwarg value
   * @param key Kwarg key
   * @param default_value Default value if key doesn't exist
   * @return The kwarg value or default
   */
  template <typename T>
  T GetKwarg(const std::string& key, const T& default_value = T{}) const {
    for (const auto& [k, v] : kwargs_) {
      if (k == key) {
        return AnyCast<T>(v, "kwarg key: " + key);
      }
    }
    return default_value;
  }

  /**
   * @brief Check if a kwarg exists
   *
   * @param key Kwarg key
   * @return true if the kwarg exists
   */
  [[nodiscard]] bool HasKwarg(const std::string& key) const {
    for (const auto& [k, v] : kwargs_) {
      if (k == key) {
        return true;
      }
    }
    return false;
  }

  /**
   * @brief Get an attr value with type checking
   *
   * @tparam T Type of the attr value
   * @param key Attr key
   * @param default_value Default value if key doesn't exist
   * @return The attr value or default
   */
  template <typename T>
  [[nodiscard]] T GetAttr(const std::string& key, const T& default_value = T{}) const {
    for (const auto& [k, v] : attrs_) {
      if (k == key) {
        return AnyCast<T>(v, "attr key: " + key);
      }
    }
    return default_value;
  }

  /**
   * @brief Check if an attr exists
   *
   * @param key Attr key
   * @return true if the attr exists
   */
  [[nodiscard]] bool HasAttr(const std::string& key) const {
    for (const auto& [k, v] : attrs_) {
      if (k == key) {
        return true;
      }
    }
    return false;
  }

  /**
   * @brief Check if call-site arg directions have been resolved
   *
   * @return true if attrs_ contains a non-empty arg_directions vector
   */
  [[nodiscard]] bool HasArgDirections() const { return HasAttr(kAttrArgDirections); }

  /**
   * @brief Get the resolved call-site arg directions
   *
   * @return Per-argument directions, or empty vector if not yet resolved.
   */
  [[nodiscard]] std::vector<ArgDirection> GetArgDirections() const {
    return GetAttr<std::vector<ArgDirection>>(kAttrArgDirections);
  }

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::Call; }
  [[nodiscard]] std::string TypeName() const override { return "Call"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors (op, args, attrs, and kwargs as USUAL fields)
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Expr::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&Call::op_, "op"),
                                          reflection::UsualField(&Call::args_, "args"),
                                          reflection::UsualField(&Call::attrs_, "attrs"),
                                          reflection::UsualField(&Call::kwargs_, "kwargs")));
  }

 private:
  void ValidateArgDirectionsAttr() const {
    for (const auto& [k, v] : attrs_) {
      if (k != kAttrArgDirections) {
        continue;
      }
      if (v.type() != typeid(std::vector<ArgDirection>)) {
        throw pypto::TypeError("Call attrs['" + std::string(kAttrArgDirections) +
                               "'] must be std::vector<ArgDirection>");
      }
      const auto& dirs = AnyCast<std::vector<ArgDirection>>(v, "attr key: arg_directions");
      if (!dirs.empty() && dirs.size() != args_.size()) {
        throw pypto::TypeError("Call attrs['arg_directions'] size (" + std::to_string(dirs.size()) +
                               ") must match args size (" + std::to_string(args_.size()) + ")");
      }
      return;
    }
  }
};

using CallPtr = std::shared_ptr<const Call>;

/**
 * @brief Build a copy of `attrs` with `kAttrArgDirections` set to `dirs`
 *
 * Replaces an existing entry if present; otherwise appends.
 */
inline std::vector<std::pair<std::string, std::any>> WithArgDirectionsAttr(
    std::vector<std::pair<std::string, std::any>> attrs, std::vector<ArgDirection> dirs) {
  for (auto& [k, v] : attrs) {
    if (k == kAttrArgDirections) {
      v = std::move(dirs);
      return attrs;
    }
  }
  attrs.emplace_back(kAttrArgDirections, std::move(dirs));
  return attrs;
}

/**
 * @brief Reserved attr key for per-arg ``ArgDirection::NoDep`` overrides
 *
 * The parser sets this when the user wraps a kernel-call argument in
 * ``pl.no_dep(...)`` to mark a single arg position as no-dep without
 * forcing the whole call to a manual dep mode. Value type is
 * ``std::vector<int32_t>``: the argument indices to be set to NoDep.
 *
 * ``DeriveCallDirections`` reads this attr AFTER computing the per-arg
 * directions and overwrites each indicated slot to ``ArgDirection::NoDep``,
 * leaving every other position as the auto-derived direction.
 */
inline constexpr const char* kAttrArgDirectionOverrides = "arg_direction_overrides";

/**
 * Build a copy of ``attrs`` with ``kAttrArgDirectionOverrides`` set to
 * ``no_dep_indices``. Replaces an existing entry if present; otherwise appends.
 */
inline std::vector<std::pair<std::string, std::any>> WithArgDirectionOverridesAttr(
    std::vector<std::pair<std::string, std::any>> attrs, std::vector<int32_t> no_dep_indices) {
  for (auto& [k, v] : attrs) {
    if (k == kAttrArgDirectionOverrides) {
      v = std::move(no_dep_indices);
      return attrs;
    }
  }
  attrs.emplace_back(kAttrArgDirectionOverrides, std::move(no_dep_indices));
  return attrs;
}

/**
 * @brief Reserved attr key for the set of dep edges that codegen packs into
 * a stack ``PTO2TaskId[]`` array and emits as a single
 * ``params.set_dependencies(arr, count)`` call before the kernel submit.
 *
 * Value type: ``std::vector<VarPtr>`` where every entry is a Var of
 * ``ScalarType(DataType::TASK_ID)`` or ``ArrayType(..., TASK_ID)``. Written
 * directly by the parser when the user passes ``deps=[tid1, tid2, ...]`` on a
 * ``pl.submit(...)`` call inside ``with pl.manual_scope():``. Users obtain
 * each entry as the producer TaskId of a prior ``pl.submit(...)``, from
 * ``pl.array.create(N, pl.TASK_ID)``, the ``None`` sentinel, or by threading
 * a TaskId iter_arg through a ``pl.range`` / ``pl.parallel`` loop.
 */
inline constexpr const char* kAttrManualDepEdges = "manual_dep_edges";

/**
 * @brief Reserved attr key marking an internal dependency-only TaskId call.
 *
 * Value type: ``bool``. Written by the manual phase-fence expansion pass on
 * ``system.task_dummy`` calls. Orchestration codegen lowers the marked call to
 * ``rt_submit_dummy_task(...)`` and uses ``manual_dep_edges`` on that same call
 * as the dummy task fanin.
 */
inline constexpr const char* kAttrDummyTask = "dummy_task";

/**
 * @brief Reserved attr key for the producer-TaskId Var captured by a
 * ``with pl.at(...) as tid:`` block.
 *
 * Set by the parser as a key on the enclosing ``ScopeStmt``'s ``attrs_``
 * (``InCoreScopeStmt`` / ``AutoInCoreScopeStmt`` / ``HierarchyScopeStmt``).
 * Value type: ``VarPtr`` — a fresh ``Scalar[TASK_ID]`` Var allocated in the
 * outer scope. ``OutlineIncoreScopes`` / ``OutlineHierarchyScopes`` augment
 * the synthesised ``Call``'s return type with ``Scalar[TASK_ID]`` and emit
 * an ``AssignStmt(tid_var, TupleGetItem(call_lhs, last_idx))`` so the outer
 * scope's references to ``tid`` resolve to the producer TaskId.
 */
inline constexpr const char* kAttrTaskIdVar = "task_id_var";

/**
 * @brief Reserved attr key set on a ``ScopeStmt`` by the DSL parser for
 * ``pl.at(no_dep_args=[t1, t2])``. Holds ``std::vector<VarPtr>`` referencing
 * outer-scope tensor Vars that should be marked ``ArgDirection::NoDep`` on
 * the synthesised kernel Call.
 *
 * The outliner translates this list into positional indices into the
 * synthesised Call's ``args_`` (using the captured-var order) and writes the
 * result back as ``kAttrArgDirectionOverrides`` on the Call.
 * ``DeriveCallDirections`` then consumes it exactly like the per-arg overrides
 * produced by ``pl.no_dep(t)`` wrappers at explicit kernel call sites.
 *
 * Never appears on a ``Call`` — it is exclusively a scope-level marker.
 */
inline constexpr const char* kAttrArgDirOverrideVars = "arg_direction_overrides_vars";

/**
 * Build a copy of ``attrs`` with ``kAttrArgDirOverrideVars`` set to ``vars``.
 * Replaces an existing entry if present; otherwise appends.
 */
inline std::vector<std::pair<std::string, std::any>> WithArgDirOverrideVarsAttr(
    std::vector<std::pair<std::string, std::any>> attrs, std::vector<VarPtr> vars) {
  for (auto& [k, v] : attrs) {
    if (k == kAttrArgDirOverrideVars) {
      v = std::move(vars);
      return attrs;
    }
  }
  attrs.emplace_back(kAttrArgDirOverrideVars, std::move(vars));
  return attrs;
}

/**
 * Build a copy of ``attrs`` with ``kAttrManualDepEdges`` set to ``vars``.
 * Replaces an existing entry if present; otherwise appends.
 */
inline std::vector<std::pair<std::string, std::any>> WithManualDepEdgesAttr(
    std::vector<std::pair<std::string, std::any>> attrs, std::vector<VarPtr> vars) {
  for (auto& [k, v] : attrs) {
    if (k == kAttrManualDepEdges) {
      v = std::move(vars);
      return attrs;
    }
  }
  attrs.emplace_back(kAttrManualDepEdges, std::move(vars));
  return attrs;
}

/**
 * @brief Reserved attr key for the physical device selector on a
 * host-orchestrator dispatch to a chip-level Orchestration function.
 *
 * Value type: ``ExprPtr`` — either a ``ConstInt`` (fixed device) or a ``Var``
 * referring to the induction variable of an enclosing ``pl.range`` loop.
 * Written by the parser when the user writes ``self.chip_orch(..., device=r)``;
 * consumed by ``CollectCommGroups`` to derive per-CommGroup device subsets and
 * by distributed codegen to emit ``submit_next_level(..., worker=<r>)``.
 *
 * SSA passes MUST substitute ``Var`` references inside this attr's value the
 * same way they do for Call args, otherwise the stored ``Var`` becomes a dead
 * reference once the loop's induction variable is versioned.
 */
inline constexpr const char* kAttrDevice = "device";

/**
 * @brief Task-launch expression — ``pl.submit(self.kernel, args, deps=[...])``
 *
 * Produced by ``pl.submit(...)`` inside a ``pl.manual_scope`` body.
 * Semantically distinct from ``Call``: a ``Submit`` launches an asynchronous
 * task and produces a TaskId in addition to the callee's logical return. The
 * result type is always ``Tuple[<callee return>..., Scalar[TASK_ID]]`` (or
 * just ``Scalar[TASK_ID]`` when the callee has no value return); callers
 * unpack as ``out, tid = pl.submit(...)``.
 *
 * ``deps_`` is a first-class field carrying the explicit cross-task
 * dependencies passed as ``deps=[tid1, tid2, ...]`` — each entry is an
 * ``ExprPtr`` referencing a ``Scalar[TASK_ID]`` Var or an
 * ``Array[N, TASK_ID]`` Var. Passes that walk variable uses MUST include
 * ``deps_`` (see ``.claude/rules/pass-submit-awareness.md``).
 *
 * ``attrs_`` / ``kwargs_`` mirror ``Call``'s layout so reserved keys such as
 * ``kAttrArgDirections`` and ``kAttrArgDirectionOverrides`` still apply to
 * ``args_``. The ``kAttrManualDepEdges`` key is intentionally NOT read from a
 * ``Submit``'s ``attrs_`` — use the typed ``deps_`` field instead.
 */
class Submit : public Expr {
 public:
  OpPtr op_;                   // Callee (typically a GlobalVar)
  std::vector<ExprPtr> args_;  // Positional arguments
  std::vector<ExprPtr> deps_;  // TaskId dependencies (Scalar[TASK_ID] / Array[N, TASK_ID])
  // SPMD launch spec — populated only by ``pl.spmd_submit(...)``.
  // ``core_num_`` is the block count (an INDEX/INT-typed Expr — typically a
  // ConstInt or a closure Var); ``std::nullopt`` marks a plain
  // ``pl.submit(...)`` (single-block launch, no launch spec). ``sync_start_``
  // requires all logical blocks to launch atomically and is only meaningful
  // when ``core_num_`` is present. These lower to ``Arg::launch_spec`` in
  // orchestration codegen via SubmitToCallView (attrs ``"core_num"`` /
  // ``"sync_start"``) → EmitLaunchSpec. ``core_num_`` is a first-class field
  // (not an attr) because it is an SSA value in the use-def chain — passes
  // that substitute / DCE / dominance-check Vars must walk it (see
  // .claude/rules/pass-submit-awareness.md, rule 2).
  std::optional<ExprPtr> core_num_;
  bool sync_start_ = false;
  std::vector<std::pair<std::string, std::any>> attrs_;   // Compiler-internal metadata (ordered)
  std::vector<std::pair<std::string, std::any>> kwargs_;  // Keyword arguments (metadata, ordered)

  /**
   * @brief Create a Submit with the minimum fields.
   */
  Submit(OpPtr op, std::vector<ExprPtr> args, std::vector<ExprPtr> deps, TypePtr type, Span span)
      : Expr(std::move(span), std::move(type)),
        op_(std::move(op)),
        args_(std::move(args)),
        deps_(std::move(deps)),
        attrs_(),
        kwargs_() {}

  /**
   * @brief Create a Submit with attrs and kwargs.
   *
   * The trailing ``core_num`` / ``sync_start`` carry the SPMD launch spec for
   * ``pl.spmd_submit(...)``; they default to "no launch spec" so every
   * existing 7-arg construction site keeps building a plain submit.
   *
   * Validates that, when present, ``attrs[kAttrArgDirections]`` is a
   * ``std::vector<ArgDirection>`` whose length matches ``args``, and that the
   * launch spec is well-formed (``sync_start`` implies ``core_num``).
   */
  Submit(OpPtr op, std::vector<ExprPtr> args, std::vector<ExprPtr> deps,
         std::vector<std::pair<std::string, std::any>> kwargs,
         std::vector<std::pair<std::string, std::any>> attrs, TypePtr type, Span span,
         std::optional<ExprPtr> core_num = std::nullopt, bool sync_start = false)
      : Expr(std::move(span), std::move(type)),
        op_(std::move(op)),
        args_(std::move(args)),
        deps_(std::move(deps)),
        core_num_(std::move(core_num)),
        sync_start_(sync_start),
        attrs_(std::move(attrs)),
        kwargs_(std::move(kwargs)) {
    ValidateArgDirectionsAttr();
    ValidateLaunchSpec();
  }

  template <typename T>
  T GetKwarg(const std::string& key, const T& default_value = T{}) const {
    for (const auto& [k, v] : kwargs_) {
      if (k == key) {
        return AnyCast<T>(v, "kwarg key: " + key);
      }
    }
    return default_value;
  }

  [[nodiscard]] bool HasKwarg(const std::string& key) const {
    for (const auto& [k, v] : kwargs_) {
      if (k == key) {
        return true;
      }
    }
    return false;
  }

  template <typename T>
  [[nodiscard]] T GetAttr(const std::string& key, const T& default_value = T{}) const {
    for (const auto& [k, v] : attrs_) {
      if (k == key) {
        return AnyCast<T>(v, "attr key: " + key);
      }
    }
    return default_value;
  }

  [[nodiscard]] bool HasAttr(const std::string& key) const {
    for (const auto& [k, v] : attrs_) {
      if (k == key) {
        return true;
      }
    }
    return false;
  }

  [[nodiscard]] bool HasArgDirections() const { return HasAttr(kAttrArgDirections); }

  [[nodiscard]] std::vector<ArgDirection> GetArgDirections() const {
    return GetAttr<std::vector<ArgDirection>>(kAttrArgDirections);
  }

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::Submit; }
  [[nodiscard]] std::string TypeName() const override { return "Submit"; }

  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Expr::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&Submit::op_, "op"),
                                          reflection::UsualField(&Submit::args_, "args"),
                                          reflection::UsualField(&Submit::deps_, "deps"),
                                          reflection::UsualField(&Submit::core_num_, "core_num"),
                                          reflection::UsualField(&Submit::sync_start_, "sync_start"),
                                          reflection::UsualField(&Submit::attrs_, "attrs"),
                                          reflection::UsualField(&Submit::kwargs_, "kwargs")));
  }

 private:
  void ValidateLaunchSpec() const {
    if (sync_start_ && !core_num_.has_value()) {
      throw pypto::ValueError(
          "Submit sync_start=true requires core_num (an SPMD launch). Plain "
          "pl.submit(...) has no launch spec — use pl.spmd_submit(...) for SPMD.");
    }
    if (core_num_.has_value() && !*core_num_) {
      throw pypto::TypeError("Submit core_num must be a non-null Expr when present");
    }
    // core_num is an SPMD block count — reject a non-integer launch dimension
    // at the public boundary (the DSL parser already enforces this, but direct
    // C++/Python construction would otherwise let a float/bool expr through).
    // Only reject a *typed* non-integer scalar; leave other expr shapes alone.
    if (core_num_.has_value()) {
      if (auto scalar = std::dynamic_pointer_cast<const ScalarType>((*core_num_)->GetType())) {
        if (!scalar->dtype_.IsInt() && !scalar->dtype_.IsIndexLike()) {
          throw pypto::TypeError("Submit core_num must be an integer/index expression (SPMD block count)");
        }
      }
    }
  }

  void ValidateArgDirectionsAttr() const {
    for (const auto& [k, v] : attrs_) {
      if (k != kAttrArgDirections) {
        continue;
      }
      if (v.type() != typeid(std::vector<ArgDirection>)) {
        throw pypto::TypeError("Submit attrs['" + std::string(kAttrArgDirections) +
                               "'] must be std::vector<ArgDirection>");
      }
      const auto& dirs = AnyCast<std::vector<ArgDirection>>(v, "attr key: arg_directions");
      if (!dirs.empty() && dirs.size() != args_.size()) {
        throw pypto::TypeError("Submit attrs['arg_directions'] size (" + std::to_string(dirs.size()) +
                               ") must match args size (" + std::to_string(args_.size()) + ")");
      }
      return;
    }
  }
};

using SubmitPtr = std::shared_ptr<const Submit>;

/**
 * @brief Adapter that returns the equivalent augmented-Call view of a Submit.
 *
 * Synthesises a fresh ``Call`` with ``Submit::deps_`` re-encoded as the
 * ``kAttrManualDepEdges`` attr. Used by codegen and other consumers that
 * predate the Submit IR kind and still operate on Call: dispatch on
 * ``As<Submit>(expr)`` first, then funnel through this view so the existing
 * Call codepath sees the dep info via the legacy attrs interface.
 *
 * The original Submit is untouched — the returned Call is a transient
 * adapter, not a transformation of the IR.
 */
inline CallPtr SubmitToCallView(const SubmitPtr& submit) {
  // Strip any pre-existing kAttrManualDepEdges from submit->attrs_ — the
  // canonical encoding for Submit deps is the typed deps_ field, so a
  // stray attr entry would either duplicate the deps (when deps_ is
  // populated and we re-add the attr below) or survive as a stale stand-in
  // (when deps_ is empty and the original attr would silently propagate
  // into the synthesised Call). Filtering here keeps deps_ as the single
  // source of truth at the view boundary.
  std::vector<std::pair<std::string, std::any>> attrs;
  attrs.reserve(submit->attrs_.size());
  for (const auto& [k, v] : submit->attrs_) {
    // ``core_num`` / ``sync_start`` are first-class Submit fields and are
    // re-emitted below from core_num_ / sync_start_. Drop any stray attr of
    // the same key so the field stays the single source of truth (Call::GetAttr
    // returns the first match, so a stale attr would otherwise shadow it).
    if (k != kAttrManualDepEdges && k != "core_num" && k != "sync_start") {
      attrs.emplace_back(k, v);
    }
  }
  if (!submit->deps_.empty()) {
    std::vector<VarPtr> dep_vars;
    dep_vars.reserve(submit->deps_.size());
    for (size_t i = 0; i < submit->deps_.size(); ++i) {
      const auto& d = submit->deps_[i];
      // Submit::deps_ entries are Var-like by construction (Scalar[TASK_ID]
      // / Array[N, TASK_ID]) — the parser rejects anything else. Enforce
      // the invariant here so a stray non-Var entry surfaces as a clear
      // failure at the view boundary, rather than silently dropping the
      // dep and losing the manual dependency edge downstream. We throw
      // pypto::TypeError directly (matching Submit::ValidateArgDirectionsAttr
      // above) because INTERNAL_CHECK_SPAN lives in logging.h, which expr.h
      // does not transitively include. The Var/IterArg check is inlined
      // because kind_traits.h's AsVarLike depends on this header.
      if (!d) {
        throw pypto::TypeError("Submit dep at index " + std::to_string(i) + " is null");
      }
      auto kind = d->GetKind();
      if (kind != ObjectKind::Var && kind != ObjectKind::IterArg) {
        throw pypto::TypeError("Submit dep at index " + std::to_string(i) +
                               " is not Var-like (kind: " + std::to_string(static_cast<int>(kind)) + ")");
      }
      dep_vars.push_back(std::static_pointer_cast<const Var>(d));
    }
    attrs = WithManualDepEdgesAttr(std::move(attrs), std::move(dep_vars));
  }
  // Carry the SPMD launch spec (pl.spmd_submit) through to the Call view as
  // attrs so orchestration codegen's EmitLaunchSpec reads core_num/sync_start
  // uniformly with the scope-based pl.spmd path (which stores them on the
  // Spmd-wrapper function's attrs). core_num_/sync_start_ are first-class
  // Submit fields and never appear in submit->attrs_, so no duplication.
  if (submit->core_num_.has_value()) {
    attrs.emplace_back("core_num", std::any(*submit->core_num_));
    attrs.emplace_back("sync_start", std::any(submit->sync_start_));
  }
  return std::make_shared<Call>(submit->op_, submit->args_, submit->kwargs_, std::move(attrs),
                                submit->GetType(), submit->span_);
}

/**
 * @brief Expression to create a tuple from multiple expressions
 *
 * Takes a list of expressions and creates a tuple value.
 * The result type is TupleType containing the types of all input expressions.
 */
class MakeTuple : public Expr {
 public:
  std::vector<ExprPtr> elements_;  // Elements of the tuple

  /**
   * @brief Create a tuple construction expression
   *
   * @param elements Expressions to be tuple elements
   * @param span Source location
   */
  MakeTuple(std::vector<ExprPtr> elements, Span span);

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::MakeTuple; }
  [[nodiscard]] std::string TypeName() const override { return "MakeTuple"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Expr::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&MakeTuple::elements_, "elements")));
  }
};

using MakeTuplePtr = std::shared_ptr<const MakeTuple>;

/**
 * @brief Tuple element access expression
 *
 * Represents accessing an element from a tuple by index.
 * The tuple must have TupleType and index must be a compile-time constant.
 */
class TupleGetItemExpr : public Expr {
 public:
  ExprPtr tuple_;  // Tuple expression (must have TupleType)
  int index_;      // Index of the element to access (0-based)

  /**
   * @brief Create a tuple element access expression
   *
   * @param tuple Tuple expression (must have TupleType)
   * @param index Index of the element (0-based, must be within bounds)
   * @param span Source location
   */
  TupleGetItemExpr(ExprPtr tuple, int index, Span span);

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::TupleGetItemExpr; }
  [[nodiscard]] std::string TypeName() const override { return "TupleGetItemExpr"; }

  /**
   * @brief Get field descriptors for reflection-based visitation
   *
   * @return Tuple of field descriptors
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(Expr::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&TupleGetItemExpr::tuple_, "tuple"),
                                          reflection::UsualField(&TupleGetItemExpr::index_, "index")));
  }
};

using TupleGetItemExprPtr = std::shared_ptr<const TupleGetItemExpr>;

/**
 * @brief Compare two ExprPtr values: ConstInt by value, otherwise by pointer identity
 */
bool AreExprsEqual(const ExprPtr& e1, const ExprPtr& e2);

/**
 * @brief Compare two ExprPtr vectors element-wise
 */
bool AreExprVectorsEqual(const std::vector<ExprPtr>& v1, const std::vector<ExprPtr>& v2);

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_EXPR_H_
