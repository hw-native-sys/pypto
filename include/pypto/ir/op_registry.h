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

/**
 * @file op_registry.h
 * @brief Operator registration system for PyPTO IR
 *
 * This file provides a modern C++ template-based operator registration system
 * that enables compile-time type checking and automatic type deduction for
 * tensor, tile, and scalar operations.
 */

#ifndef PYPTO_IR_OP_REGISTRY_H_
#define PYPTO_IR_OP_REGISTRY_H_

#include <algorithm>
#include <any>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/common.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/core_affinity_kind.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/span.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

// Forward declaration
class Call;
using CallPtr = std::shared_ptr<const Call>;

/// Full memory space specification for one operator.
struct OpMemorySpaceSpec {
  /// Required memory spaces per input arg index.
  /// Each element is a set of allowed memory spaces.
  /// Empty vector at position i = any memory space accepted for arg i.
  std::vector<std::vector<MemorySpace>> input_constraints;

  /// Resolves output memory space from the Call's kwargs.
  /// Returns nullopt when the space cannot be resolved from kwargs alone — either
  /// because the op inherits from its input (see `output_inherits_input`) or
  /// because a retargetable kwarg is absent and InferTileMemorySpace must decide.
  using OutputResolver =
      std::function<std::optional<MemorySpace>(const std::vector<std::pair<std::string, std::any>>& kwargs)>;
  OutputResolver deduce_output_memory;

  /// When set, the output reuses the MemRef of the input argument at this index.
  /// Used by accumulate ops (matmul_acc, gemv_acc) where the output IS the input buffer.
  std::optional<size_t> output_reuses_input_arg;

  /// True when the output memory space is defined to equal the first tile-typed
  /// input's memory space (set via `set_output_memory_inherit_input`).
  /// InferTileMemorySpace uses this for forward inheritance and backward-demand
  /// propagation through view-like ops; memory reuse uses it to skip retargeting.
  bool output_inherits_input = false;
};

/**
 * @brief Evidence available to analyses of an operation's physical accesses.
 *
 * Unknown is deliberately the default. Functional means every tile operand is
 * read and every tile-typed SSA result is written, with no hidden tile
 * workspace or mutation; range-sensitive analyses must still prove that an
 * access covers the whole allocation. NoAccess marks declarations and
 * zero-copy metadata operations that execute no memory access.
 */
enum class ExecutionMemoryAccessEvidence : uint8_t {
  Unknown,
  Functional,
  NoAccess,
};

/**
 * @brief What executing an operation does to the buffer one argument names.
 *
 * This is an *effect* declaration, not a type. It answers the single question
 * every direction and dependency analysis asks: does running this call read
 * from, or write to, the memory this argument names?
 *
 * `Read` is the default for an argument the operator does not name, because
 * the overwhelming majority of operators are functional — they consume their
 * operands and produce a fresh SSA result. An operator that instead updates an
 * operand in place must say so: a missing `Write` is not a conservative
 * approximation, it silently erases a real dependency edge (the writer looks
 * like a pure reader, so nothing is ordered against it).
 */
enum class ArgEffect : uint8_t {
  Read = 0,       ///< Read, never written.
  Write = 1,      ///< Overwritten without being read first (a destination operand).
  ReadWrite = 2,  ///< Read *and* written — accumulate, atomic update, partial in-place rewrite.
};

/// Merge two independent observations of one argument's effect. Each is a lower
/// bound on the accesses, so the merge is a union along Read < {Write} < ReadWrite.
[[nodiscard]] inline ArgEffect MergeArgEffect(ArgEffect lhs, ArgEffect rhs) {
  if (lhs == rhs) return lhs;
  return ArgEffect::ReadWrite;
}

[[nodiscard]] inline bool ArgEffectReads(ArgEffect effect) { return effect != ArgEffect::Write; }
[[nodiscard]] inline bool ArgEffectWrites(ArgEffect effect) { return effect != ArgEffect::Read; }

[[nodiscard]] inline std::string ArgEffectToString(ArgEffect effect) {
  switch (effect) {
    case ArgEffect::Read:
      return "Read";
    case ArgEffect::Write:
      return "Write";
    case ArgEffect::ReadWrite:
      return "ReadWrite";
  }
  return "Unknown";
}

/// Read an integer-valued kwarg out of a call's kwargs, or `fallback` when the
/// call does not carry it. Effect resolvers use this to branch on the enum-backed
/// int kwargs (`atomic`, `op`, ...) that decide whether a destination operand is
/// overwritten or accumulated into.
[[nodiscard]] inline int GetIntKwarg(const std::vector<std::pair<std::string, std::any>>& kwargs,
                                     const std::string& key, int fallback) {
  for (const auto& [k, v] : kwargs) {
    if (k == key) return AnyCast<int>(v, key);
  }
  return fallback;
}

/// MemorySpace-valued counterpart of GetIntKwarg. A memory-space kwarg is stored
/// as a `MemorySpace`, not as an int (see `ConvertKwargsDict`), so it needs its
/// own accessor — `tile.mgather` selects which operand is its GM scratch from
/// `target_memory`.
[[nodiscard]] inline MemorySpace GetMemorySpaceKwarg(
    const std::vector<std::pair<std::string, std::any>>& kwargs, const std::string& key,
    MemorySpace fallback) {
  for (const auto& [k, v] : kwargs) {
    if (k == key) return AnyCast<MemorySpace>(v, key);
  }
  return fallback;
}

/// String-valued counterpart of GetIntKwarg, for the kwargs that select an
/// operator mode by name (`system.syncall`'s hard/soft form).
[[nodiscard]] inline std::string GetStringKwarg(const std::vector<std::pair<std::string, std::any>>& kwargs,
                                                const std::string& key, const std::string& fallback) {
  for (const auto& [k, v] : kwargs) {
    if (k == key) return AnyCast<std::string>(v, key);
  }
  return fallback;
}

/**
 * @brief The hardware path an operation's writes travel.
 *
 * PyPTO cannot order an MTE3 (DMA) store against a scalar D-cache write to the
 * same GM tensor, so a function that mixes both on one buffer is rejected. The
 * channel is a property of the operator's lowering, declared once here rather
 * than re-derived by the diagnostic.
 */
enum class WriteChannel : uint8_t {
  Dma,     ///< MTE3 / DMA store path (tile.store, tensor.assemble, cross-rank put/get).
  Scalar,  ///< Scalar D-cache write path (tensor.write).
};

/**
 * @brief Per-argument execution effects declared by one operator.
 *
 * Absent (`std::nullopt` on the entry) means *nobody has classified this
 * operator yet* — which is deliberately distinct from "declared read-only", so
 * an analysis can refuse to guess instead of defaulting an unclassified writer
 * to read-only.
 */
struct OpArgEffectSpec {
  /// Effect resolved from the call's kwargs, for arguments whose effect is not
  /// fixed by the operator alone (an atomic store reads its destination; a
  /// `NotifyOp::kAtomicAdd` accumulates into the peer slot).
  using Resolver = std::function<ArgEffect(const std::vector<std::pair<std::string, std::any>>& kwargs)>;

  /// Effect per positional argument index. Indices past the end are `Read`.
  std::vector<ArgEffect> per_arg;

  /// Kwarg-dependent effects, keyed by argument index. Overrides `per_arg`.
  std::map<size_t, Resolver> kwarg_dependent;

  /// Which path this operator's writes take. Set only for operators that write.
  std::optional<WriteChannel> write_channel;

  /// Argument indices a registration named explicitly. `per_arg` cannot answer
  /// this on its own: it is resized to *cover* the highest declared index, so a
  /// slot nobody named is indistinguishable there from one declared `Read`.
  /// Validation needs the difference — an operator that classified the wrong
  /// argument has not classified the one it updates in place.
  std::set<size_t> declared_args;

  /// True only when a registration called `no_arg_writes()`, which is a verdict
  /// about every argument at once. The spec's mere existence cannot stand in for
  /// this: `set_write_channel()` creates it too, and an operator that declared
  /// only a channel has classified nothing.
  bool declared_no_writes = false;
};

/**
 * @brief Type-erased operator registration entry
 *
 * This class represents a registered operator in the registry system. It stores
 * metadata about the operator including its name, description, expected arguments,
 * validation logic, and type deduction function. The entry provides a fluent
 * interface for configuring operator properties during registration.
 *
 * Example usage:
 * @code
 * OpRegistryEntry entry;
 * entry.set_name("tensor.add")
 *      .set_description("Element-wise addition of two tensors")
 *      .add_argument("lhs", "Left-hand side tensor")
 *      .add_argument("rhs", "Right-hand side tensor")
 *      .f_deduce_type([](const std::vector<ExprPtr>& args) {
 *          return args[0]->GetType();
 *      });
 * @endcode
 */
class OpRegistryEntry {
 public:
  /**
   * @brief Get the operator instance
   *
   * Validates that the operator is properly configured with all required fields
   * before returning the operator instance. This ensures that operators cannot
   * be used until they are fully defined.
   *
   * Required fields:
   * - name: Set automatically during registration
   * - description: Must be set via set_description()
   * - op_category: Must be set via set_op_category()
   * - arguments: Must be set via add_argument() or no_argument()
   * - deduce_type: Must be set via f_deduce_type()
   *
   * @return Const reference to the operator pointer
   * @throws ValueError if any required field is not set
   */
  [[nodiscard]] inline const OpPtr& GetOp() const {
    // Check operator instance
    CHECK(op_) << "Operator '" + name_ + "' has no operator instance";

    // Check description is set
    CHECK(description_.has_value()) << "Operator '" + name_ +
                                           "' has no description. Use .set_description() to provide one.";

    // Check op_category is set
    CHECK(op_category_.has_value()) << "Operator '" + name_ +
                                           "' has no category. Use .set_op_category() to provide one.";

    // Check arguments are defined (either with arguments or marked as no_argument)
    CHECK(arguments_.has_value())
        << "Operator '" + name_ +
               "' has no argument definition. Use .add_argument() or .no_argument() to define arguments.";

    // Check deduce_type is set
    CHECK(deduce_type_.has_value())
        << "Operator '" + name_ + "' has no type deduction function. Use .f_deduce_type() to provide one.";

    return op_;
  }

  /**
   * @brief Get the operator name
   *
   * @return Const reference to the operator name
   */
  [[nodiscard]] inline const std::string& GetName() const { return name_; }

  /**
   * @brief Get the operator description
   *
   * @return Const reference to the operator description
   * @throws ValueError if description is not set
   */
  [[nodiscard]] inline const std::string& GetDescription() const {
    CHECK(description_.has_value()) << "Operator '" + name_ + "' has no description";
    return *description_;
  }

  /**
   * @brief Get the operator category
   *
   * @return Const reference to the operator category (e.g., "TensorOp", "TileOp", "ScalarOp")
   * @throws ValueError if category is not set
   */
  [[nodiscard]] inline const std::string& GetOpCategory() const {
    CHECK(op_category_.has_value()) << "Operator '" + name_ + "' has no category";
    return *op_category_;
  }

  /**
   * @brief Get the type deduction function
   *
   * Validates that the type deduction function is properly registered.
   *
   * @return Const reference to the type deduction function
   * @throws ValueError if the type deduction function is not set
   */
  [[nodiscard]] inline const std::function<TypePtr(const std::vector<ExprPtr>&,
                                                   const std::vector<std::pair<std::string, std::any>>&)>&
  GetDeduceType() const {
    CHECK(deduce_type_.has_value()) << "Operator '" + name_ + "' has no type deduction function";
    return *deduce_type_;
  }

  /**
   * @brief Set the operator description
   *
   * Provides human-readable documentation for the operator. Should describe
   * what the operator does, its semantics, and any important constraints.
   *
   * @param description Human-readable description of the operator
   * @return Reference to this entry for method chaining
   */
  inline OpRegistryEntry& set_description(std::string description) {
    CHECK(!description_.has_value()) << "Operator '" + name_ + "' description is already set";
    description_ = std::move(description);
    return *this;
  }

  /**
   * @brief Set the operator category
   *
   * Specifies the category of the operator (e.g., "TensorOp", "TileOp", "ScalarOp").
   * This is used for categorization and type checking without requiring specific type details.
   *
   * @param category Operator category (e.g., "TensorOp", "TileOp", "ScalarOp")
   * @return Reference to this entry for method chaining
   */
  inline OpRegistryEntry& set_op_category(std::string category) {
    CHECK(!op_category_.has_value()) << "Operator '" + name_ + "' category is already set";
    op_category_ = std::move(category);
    return *this;
  }

  /**
   * @brief Add an argument specification
   *
   * Documents an expected argument with its name, type, and description.
   * Arguments should be added in the order they appear in the operator's
   * argument list.
   *
   * @param name Argument name (for documentation)
   * @param type Expected type of the argument (nullptr for any type)
   * @param description Description of the argument's purpose
   * @return Reference to this entry for method chaining
   */
  inline OpRegistryEntry& add_argument(std::string name, std::string description) {
    // Initialize the vector if not already initialized
    if (!arguments_.has_value()) {
      arguments_ = std::vector<std::pair<std::string, std::string>>();
    }
    arguments_->emplace_back(std::move(name), std::move(description));
    return *this;
  }

  /**
   * @brief Mark the operator as having no arguments
   *
   * This method must be called explicitly for operators that take no arguments
   * to distinguish from operators where arguments were simply not defined.
   *
   * @return Reference to this entry for method chaining
   */
  inline OpRegistryEntry& no_argument() {
    CHECK(!arguments_.has_value()) << "Operator '" + name_ +
                                          "' already has arguments defined. Cannot call no_argument() after "
                                          "add_argument().";
    arguments_ = std::vector<std::pair<std::string, std::string>>();
    return *this;
  }

  /**
   * @brief Set the type deduction function
   *
   * Provides a function that computes the result type of the operator given
   * its arguments and keyword arguments. This is called during operator creation
   * to determine the type of the resulting Call expression.
   *
   * The function should:
   * - Validate that argument types are compatible
   * - Read metadata from kwargs as needed
   * - Compute and return the result type
   * - Throw std::invalid_argument if types are incompatible
   *
   * @param dt Function that takes arguments, kwargs and returns the deduced result type
   * @return Reference to this entry for method chaining
   */
  inline OpRegistryEntry& f_deduce_type(
      std::function<TypePtr(const std::vector<ExprPtr>&,
                            const std::vector<std::pair<std::string, std::any>>&)>
          dt) {
    CHECK(!deduce_type_.has_value()) << "Operator '" + name_ + "' type deduction function is already set";
    deduce_type_ = std::move(dt);
    return *this;
  }

  /**
   * @brief Register an allowed kwarg for the operator
   *
   * Defines that this operator accepts a kwarg with the given key and expected type.
   * The type information is stored in the Op instance and used for validation
   * when creating Call expressions.
   *
   * Note: This only defines the kwarg schema (what kwargs are allowed and their types).
   * Actual kwarg values are provided per-Call instance when calling OpRegistry::Create().
   *
   * Only specific types are allowed: bool, int, std::string, double, DataType, MemorySpace
   * This is enforced at compile-time via static_assert in Op::SetAttrType.
   *
   * Example usage:
   * @code
   * REGISTER_OP("tensor.matmul")
   *     .set_attr<DataType>("out_dtype")       // OK: DataType is allowed
   *     .set_attr<bool>("a_trans")             // OK: bool is allowed
   *     .set_attr<MemorySpace>("target_memory") // OK: MemorySpace is allowed
   *
   * // The following would cause a compile-time error:
   * // .set_attr<float>("bad_attr")       // ERROR: float is not allowed
   * // .set_attr<std::vector<int>>("bad") // ERROR: vector is not allowed
   * @endcode
   *
   * @tparam T Expected type of the kwarg value (must be one of: bool, int, std::string, double, DataType,
   * MemorySpace)
   * @param key Kwarg key (string identifier)
   * @return Reference to this entry for method chaining
   */
  template <typename T>
  inline OpRegistryEntry& set_attr(const std::string& key) {
    CHECK(op_) << "Operator '" + name_ + "' has no operator instance";
    op_->SetAttrType<T>(key);  // Delegate to Op::SetAttrType (compile-time check happens there)
    return *this;
  }

  /// Set fixed output memory space (e.g., matmul -> Acc)
  inline OpRegistryEntry& set_output_memory(MemorySpace space) {
    EnsureMemorySpec();
    auto& spec = *memory_spec_;  // NOLINT(bugprone-unchecked-optional-access)
    spec.deduce_output_memory = [space](const std::vector<std::pair<std::string, std::any>>&) {
      return std::optional<MemorySpace>(space);
    };
    return *this;
  }

  /// Set output memory from kwarg (e.g., tile.load reads target_memory).
  /// When the kwarg is absent, the resolver falls back to `default_space`. Pass
  /// `std::nullopt` (the default) to mark the op as retargetable: the resolver
  /// returns nullopt and InferTileMemorySpace decides the final memory space
  /// from producer/consumer context.
  inline OpRegistryEntry& set_output_memory_from_kwarg(
      const std::string& kwarg_key = "target_memory",
      std::optional<MemorySpace> default_space = std::nullopt) {
    EnsureMemorySpec();
    auto& spec = *memory_spec_;  // NOLINT(bugprone-unchecked-optional-access)
    spec.deduce_output_memory = [kwarg_key,
                                 default_space](const std::vector<std::pair<std::string, std::any>>& kwargs) {
      for (const auto& [k, v] : kwargs) {
        if (k == kwarg_key) {
          return std::optional<MemorySpace>(AnyCast<MemorySpace>(v, kwarg_key));
        }
      }
      return default_space;
    };
    return *this;
  }

  /// Set output memory inherited from first tile-typed input (view ops).
  /// The resolver returns nullopt; InferTileMemorySpace resolves by copying the input's
  /// (already-resolved) memory space onto the output.
  inline OpRegistryEntry& set_output_memory_inherit_input() {
    EnsureMemorySpec();
    auto& spec = *memory_spec_;  // NOLINT(bugprone-unchecked-optional-access)
    spec.output_inherits_input = true;
    spec.deduce_output_memory =
        [](const std::vector<std::pair<std::string, std::any>>&) -> std::optional<MemorySpace> {
      return std::nullopt;
    };
    return *this;
  }

  /// Set input memory constraint (single allowed space)
  inline OpRegistryEntry& set_input_memory(size_t arg_index, MemorySpace required) {
    return set_input_memory(arg_index, std::vector<MemorySpace>{required});
  }

  /// Set input memory constraint (multiple allowed spaces)
  inline OpRegistryEntry& set_input_memory(size_t arg_index, std::vector<MemorySpace> allowed) {
    EnsureMemorySpec();
    auto& spec = *memory_spec_;  // NOLINT(bugprone-unchecked-optional-access)
    if (spec.input_constraints.size() <= arg_index) {
      spec.input_constraints.resize(arg_index + 1);
    }
    spec.input_constraints[arg_index] = std::move(allowed);
    return *this;
  }

  /// Mark this op as not needing a memory spec (e.g., returns MemRefType, not TileType).
  /// Creates an empty spec so ValidateTileOps() treats it as intentionally opted out.
  inline OpRegistryEntry& no_memory_spec() {
    EnsureMemorySpec();
    return *this;
  }

  /// Get memory spec (nullopt if not annotated)
  [[nodiscard]] const std::optional<OpMemorySpaceSpec>& GetMemorySpec() const { return memory_spec_; }

  /// True when this op's output memory space equals its first tile-typed input's
  /// (registered via `set_output_memory_inherit_input`). The single source of truth
  /// for passes that need to propagate memory-space information through view-like ops
  /// (InferTileMemorySpace, memory reuse).
  /// An op may combine this with `set_output_reuses_input(idx)` (e.g. in-place
  /// variants like tile.fillpad_inplace that reuse the input's MemRef in place);
  /// the memory-space-inheritance relation still holds.
  [[nodiscard]] bool OutputMemoryInheritsInput() const {
    return memory_spec_.has_value() && memory_spec_->output_inherits_input;
  }

  /// True when this op's output memory space can be chosen by the compiler
  /// (e.g. `tile.load`, `tile.create`): the op carries a writable `target_memory`
  /// kwarg that InferTileMemorySpace can rewrite to match consumer demand.
  /// Inherit-input and fixed-output ops don't participate in retargeting.
  /// Distinguishes true deferral (resolver returns nullopt when the kwarg is
  /// absent) from ops that carry a `target_memory` kwarg but still produce a
  /// concrete default (e.g. `tile.move` → Vec) — those are not retargetable.
  [[nodiscard]] bool HasRetargetableMemoryKwarg() const {
    if (!memory_spec_.has_value() || !memory_spec_->deduce_output_memory) return false;
    if (memory_spec_->output_inherits_input) return false;
    if (!op_ || !op_->HasAttr("target_memory")) return false;
    return !memory_spec_->deduce_output_memory({}).has_value();
  }

  /// Declare that this op's output reuses the MemRef of the input at arg_index.
  /// Used for accumulate ops where the output writes into the input buffer.
  inline OpRegistryEntry& set_output_reuses_input(size_t arg_index) {
    EnsureMemorySpec();
    auto& spec = *memory_spec_;  // NOLINT(bugprone-unchecked-optional-access)
    spec.output_reuses_input_arg = arg_index;
    return *this;
  }

  /// Returns the input arg index whose MemRef the output should reuse, or nullopt.
  [[nodiscard]] std::optional<size_t> GetOutputReusesInputArg() const {
    if (!memory_spec_.has_value()) return std::nullopt;
    return memory_spec_->output_reuses_input_arg;
  }

  /// Mark this operation as NOT safe for in-place execution (src buffer == dst buffer).
  /// The shared allocation-constraint analysis prevents producer-consumer reuse
  /// for such operations in both MemoryReuse and DSA-RP.
  inline OpRegistryEntry& not_inplace_safe() {
    is_inplace_safe_ = false;
    return *this;
  }

  /// Returns true if this operation supports in-place execution (src == dst buffer).
  /// Defaults to true (backward compatible). Ops that do not support src == dst must
  /// explicitly call not_inplace_safe() during registration.
  [[nodiscard]] bool IsInplaceSafe() const { return is_inplace_safe_; }

  /// Mark an IR-only declaration or zero-copy view that emits no execution-time
  /// memory access. This is distinct from output-memory inheritance: mutating
  /// operations such as tile.assemble also inherit an input memory space.
  inline OpRegistryEntry& no_execution_memory_access() {
    execution_memory_access_evidence_ = ExecutionMemoryAccessEvidence::NoAccess;
    return *this;
  }

  /// Mark an operation whose complete tile access contract is functional:
  /// every tile operand is read and every tile-typed SSA result is written.
  /// This annotation does not by itself prove a whole-allocation access.
  inline OpRegistryEntry& functional_execution_memory_access() {
    execution_memory_access_evidence_ = ExecutionMemoryAccessEvidence::Functional;
    return *this;
  }

  /// Access evidence used by conservative physical-hazard analyses.
  [[nodiscard]] ExecutionMemoryAccessEvidence GetExecutionMemoryAccessEvidence() const {
    return execution_memory_access_evidence_;
  }

  /// Mark input argument `arg_index` as one whose buffer must NOT be reused as
  /// this op's output buffer. Unlike not_inplace_safe() (which forbids the
  /// output aliasing ANY still-live input), this targets a *specific* operand
  /// that the op reads while writing its output, so aliasing the output with it
  /// corrupts results even though the op is otherwise in-place-safe — e.g.
  /// tile.sel's mask (and tmp scratch), which the TSEL intrinsic reads while
  /// writing dst. MemoryReuse consults this to forbid the output from landing
  /// on such an operand's buffer.
  inline OpRegistryEntry& forbid_output_alias(size_t arg_index) {
    forbid_output_alias_args_.insert(arg_index);
    return *this;
  }

  /// Input argument indices whose buffer the output must not alias. Empty for
  /// most ops (see forbid_output_alias()).
  [[nodiscard]] const std::set<size_t>& ForbidOutputAliasArgs() const { return forbid_output_alias_args_; }

  /// Declare which core executes this op. When unset, ClassifyCallAffinity
  /// derives the affinity from the op's memory spec (output memory space, or
  /// first tile input memory space for view/store ops). Use this for ops
  /// whose execution side is not encoded in any memory space — cross-core
  /// transfer ops (tpush/tpop/tfree/initialize_pipe), SPMD shared ops
  /// (get_block_idx, get_block_num), and tile.create (shared-by-policy).
  inline OpRegistryEntry& set_core_affinity(core_affinity::CoreAffinity a) {
    CHECK(!core_affinity_.has_value()) << "Operator '" << name_ << "' core affinity is already set";
    core_affinity_ = a;
    return *this;
  }

  /// Returns the explicitly declared core affinity, or nullopt if the op
  /// should be classified from its memory spec.
  [[nodiscard]] std::optional<core_affinity::CoreAffinity> GetCoreAffinity() const { return core_affinity_; }

  /// Mark an operation that MUST NOT RUN ON A SECOND CORE: replicating the call
  /// onto another lane changes what the program means. The canonical case is
  /// `pld.system.notify`, which publishes a cross-rank signal — a copy on the
  /// cube lane can release the peer before the vector lane's TPUT has landed
  /// the data that signal covers, so the peer reads stale bytes. (The
  /// atomic-add form additionally double-counts, but non-idempotence is not
  /// what the flag encodes: a `NotifyOp::kSet` fires the same race.)
  ///
  /// This axis says nothing about WHICH core the op runs on — placement stays
  /// entirely with set_core_affinity(). The two are orthogonal: an op may be
  /// core-agnostic (no declared affinity, hence SHARED) and still be
  /// no-duplicate, which is exactly the combination that makes the flag
  /// necessary — ExpandMixedKernel replicates SHARED statements onto both the
  /// AIC and the AIV lane, and an affinity declaration cannot express "runs on
  /// either core, but only one of them" without making a false claim about the
  /// ISA. Ops that are pinned to one lane by set_core_affinity() need no flag:
  /// they are never duplicated in the first place.
  ///
  /// The consumer is LowerAutoVectorSplit's `pl.split_aiv` region placement
  /// stamp: it pins exactly the no-duplicate calls inside a region to the AIV
  /// lane, so they are not copied onto the cube lane by ExpandMixedKernel. No
  /// verifier rejects anything on this axis — see the "NOT CHECKED,
  /// DELIBERATELY" note in verify_aiv_split.cpp.
  ///
  /// Do NOT use it for an op whose presence on the cube lane is load-bearing:
  /// pinning `pld.system.wait` to AIV would let the matmul race past the peer
  /// data it blocks on.
  inline OpRegistryEntry& set_no_duplicate() {
    no_duplicate_ = true;
    return *this;
  }

  /// True when duplicating this op onto a second core would change program
  /// meaning (see set_no_duplicate()). False for the vast majority of ops.
  [[nodiscard]] bool IsNoDuplicate() const { return no_duplicate_; }

  /// Declare the cross-core role of this op. Used for registry-driven predicates
  /// (IsTPop, IsInitializePipe, ...) so passes do not have to string-compare
  /// on specific op names.
  inline OpRegistryEntry& set_cross_core_role(core_affinity::CrossCoreRole role) {
    CHECK(!cross_core_role_.has_value()) << "Operator '" << name_ << "' cross-core role is already set";
    cross_core_role_ = role;
    return *this;
  }

  [[nodiscard]] std::optional<core_affinity::CrossCoreRole> GetCrossCoreRole() const {
    return cross_core_role_;
  }

  /// Declare what executing this operator does to positional argument
  /// `arg_index`. Every argument the operator does not name is `Read`; calling
  /// this at all marks the operator *classified*, so a later analysis can tell
  /// "reads everything" apart from "nobody looked yet".
  ///
  /// Declare the argument that carries the destination, not the one that
  /// carries the data: `tile.store(tile, offsets, output_tensor)` writes
  /// argument 2.
  inline OpRegistryEntry& set_arg_effect(size_t arg_index, ArgEffect effect) {
    auto& effects = EnsureArgEffects();
    CHECK(!effects.declared_no_writes) << "Operator '" << name_ << "' names argument " << arg_index
                                       << " after declaring no_arg_writes(); the two are contradictory";
    if (effects.per_arg.size() <= arg_index) {
      effects.per_arg.resize(arg_index + 1, ArgEffect::Read);
    }
    effects.per_arg[arg_index] = effect;
    effects.declared_args.insert(arg_index);
    return *this;
  }

  /// Declare an argument whose effect the operator alone does not fix, because
  /// a kwarg decides it — an atomic `tile.store` reads the accumulator it adds
  /// into, while a plain one overwrites it. The resolver sees the call's kwargs
  /// and must return the effect for that call.
  inline OpRegistryEntry& set_arg_effect(size_t arg_index, OpArgEffectSpec::Resolver resolver) {
    CHECK(resolver) << "Operator '" << name_ << "' argument " << arg_index
                    << " was given a null effect resolver";
    auto& effects = EnsureArgEffects();
    CHECK(!effects.declared_no_writes) << "Operator '" << name_ << "' names argument " << arg_index
                                       << " after declaring no_arg_writes(); the two are contradictory";
    effects.kwarg_dependent[arg_index] = std::move(resolver);
    effects.declared_args.insert(arg_index);
    return *this;
  }

  /// Declare that this operator writes through none of its arguments. Use it to
  /// classify an operator whose name or side-effect-only signature would
  /// otherwise leave a reader wondering — `pld.system.wait` polls a signal it
  /// never writes.
  inline OpRegistryEntry& no_arg_writes() {
    auto& effects = EnsureArgEffects();
    CHECK(effects.declared_args.empty())
        << "Operator '" << name_
        << "' declares no_arg_writes() after naming an argument; the two are contradictory";
    effects.declared_no_writes = true;
    return *this;
  }

  /// Declare which hardware path this operator's writes take. Required for an
  /// operator that writes a GM tensor, so the mixed-store diagnostic can tell
  /// an MTE3 store from a scalar one without re-listing operators.
  inline OpRegistryEntry& set_write_channel(WriteChannel channel) {
    auto& effects = EnsureArgEffects();
    CHECK(!effects.write_channel.has_value()) << "Operator '" << name_ << "' write channel is already set";
    effects.write_channel = channel;
    return *this;
  }

  /// True when this operator declared its per-argument effects. False means the
  /// operator has never been classified — an analysis that needs the answer must
  /// say so loudly rather than assume read-only.
  [[nodiscard]] bool HasDeclaredArgEffects() const { return arg_effects_.has_value(); }

  /// True when the registration reached a verdict about argument `arg_index` in
  /// particular: it named that argument, or it declared with `no_arg_writes()`
  /// that the operator writes through none of them. An operator that named some
  /// *other* argument — or that only set a write channel, which creates the spec
  /// as a side effect — has not decided about this one, and reading the resulting
  /// `Read` as a decision is what this distinguishes.
  [[nodiscard]] bool HasDeclaredArgEffect(size_t arg_index) const {
    if (!arg_effects_.has_value()) return false;
    if (arg_effects_->declared_no_writes) return true;
    return arg_effects_->declared_args.count(arg_index) > 0;
  }

  /// The effect on positional argument `arg_index` for a call carrying `kwargs`.
  /// `Read` for any argument the operator did not name.
  [[nodiscard]] ArgEffect GetArgEffect(size_t arg_index,
                                       const std::vector<std::pair<std::string, std::any>>& kwargs) const {
    if (!arg_effects_.has_value()) return ArgEffect::Read;
    auto resolver = arg_effects_->kwarg_dependent.find(arg_index);
    if (resolver != arg_effects_->kwarg_dependent.end()) return resolver->second(kwargs);
    if (arg_index >= arg_effects_->per_arg.size()) return ArgEffect::Read;
    return arg_effects_->per_arg[arg_index];
  }

  /// True when this operator writes through at least one argument under some
  /// kwargs. Cheap pre-filter for analyses that only care about writers.
  [[nodiscard]] bool WritesAnyArg() const {
    if (!arg_effects_.has_value()) return false;
    if (!arg_effects_->kwarg_dependent.empty()) return true;
    return std::any_of(arg_effects_->per_arg.begin(), arg_effects_->per_arg.end(), ArgEffectWrites);
  }

  /// The hardware path this operator's writes take, or nullopt when it declared
  /// none (either it writes nothing, or its writes are not GM stores).
  [[nodiscard]] std::optional<WriteChannel> GetWriteChannel() const {
    return arg_effects_.has_value() ? arg_effects_->write_channel : std::nullopt;
  }

  inline OpRegistryEntry& set_internal_only(bool value = true) {
    internal_only_ = value;
    return *this;
  }

  [[nodiscard]] bool IsInternalOnly() const { return internal_only_; }

  inline OpRegistryEntry& set_template_dir(std::string template_dir) {
    CHECK(!template_dir_.has_value()) << "Operator '" << name_ << "' template_dir is already set";
    template_dir_ = std::move(template_dir);
    return *this;
  }

  [[nodiscard]] const std::optional<std::string>& GetTemplateDir() const { return template_dir_; }

 private:
  void EnsureMemorySpec() {
    if (!memory_spec_.has_value()) {
      memory_spec_ = OpMemorySpaceSpec{};
    }
  }

  /// The effect spec, creating it when this is the operator's first declaration.
  /// Returns a reference rather than leaving callers to dereference the optional:
  /// engagement is obvious here and provable to a reader (and to clang-tidy's
  /// unchecked-optional-access analysis) only at this one site.
  OpArgEffectSpec& EnsureArgEffects() {
    if (!arg_effects_.has_value()) return arg_effects_.emplace();
    return *arg_effects_;
  }

  /**
   * @brief Set the operator name
   *
   * The name is used as the unique identifier for the operator in the registry.
   * Convention: use dotted notation like "tensor.add" or "tile.matmul".
   *
   * @param name The operator name (e.g., "tensor.add", "tile.conv2d")
   * @return Reference to this entry for method chaining
   */
  inline OpRegistryEntry& set_name(std::string name) {
    name_ = std::move(name);
    return *this;
  }
  friend class OpRegistry;

  OpPtr op_;                                ///< Operator instance
  std::string name_;                        ///< Operator name (unique identifier)
  std::optional<std::string> description_;  ///< Human-readable description
  std::optional<std::string> op_category_;  ///< Operator category (e.g., "TensorOp", "TileOp", "ScalarOp")
  std::optional<std::vector<std::pair<std::string, std::string>>>
      arguments_;  ///< Argument specifications (name, description)
  std::optional<std::function<TypePtr(const std::vector<ExprPtr>&,
                                      const std::vector<std::pair<std::string, std::any>>&)>>
      deduce_type_;                               ///< Type deduction function
  std::optional<OpMemorySpaceSpec> memory_spec_;  ///< Memory space specification
  std::optional<OpArgEffectSpec> arg_effects_;    ///< Per-argument execution effects; nullopt = unclassified
  bool is_inplace_safe_{true};  ///< Whether the op supports in-place execution (src == dst buffer)
  ExecutionMemoryAccessEvidence execution_memory_access_evidence_{ExecutionMemoryAccessEvidence::Unknown};
  std::set<size_t> forbid_output_alias_args_;  ///< Input args whose buffer the output must not reuse
  std::optional<core_affinity::CoreAffinity> core_affinity_;     ///< Explicit core-affinity override
  std::optional<core_affinity::CrossCoreRole> cross_core_role_;  ///< Cross-core role (for predicates)
  bool no_duplicate_{false};   ///< True when the op must not run on a second core (set_no_duplicate)
  bool internal_only_{false};  ///< True for compiler-created ops only.
  std::optional<std::string> template_dir_;  ///< Package resource for builtin templates.
};

/**
 * @brief Global operator registry (singleton)
 *
 * Manages registration and creation of operators with automatic type deduction.
 * Uses template metaprogramming to provide compile-time type safety while
 * supporting runtime operator lookup by name.
 *
 * Thread-safety: The registry is not thread-safe during registration.
 * Register all operators during initialization before concurrent access.
 */
class OpRegistry {
 public:
  // Disable copy and move
  OpRegistry(const OpRegistry&) = delete;
  OpRegistry& operator=(const OpRegistry&) = delete;
  OpRegistry(OpRegistry&&) = delete;
  OpRegistry& operator=(OpRegistry&&) = delete;

  /**
   * @brief Get the singleton instance
   *
   * @return Reference to the global operator registry
   */
  static OpRegistry& GetInstance();

  /**
   * @brief Register an operator by name
   *
   * Creates a new operator registry entry that can be configured using
   * the fluent API (set_description, add_argument, f_deduce_type, etc.).
   *
   * @param op_name Name of the operator (e.g., "tensor.add", "tile.mul")
   * @throws ValueError if operator is already registered
   */
  OpRegistryEntry& Register(const std::string& op_name);

  /**
   * @brief Create a Call expression for a registered operator
   *
   * Looks up the operator by name, validates arguments, deduces the result type,
   * and creates a Call expression with proper typing.
   *
   * @param op_name Name of the operator to call
   * @param args Arguments to pass to the operator
   * @param span Source location information
   * @return Shared pointer to Call expression with deduced type
   * @throws pypto::ValueError if operator not found or argument count invalid
   */
  [[nodiscard]] CallPtr Create(const std::string& op_name, const std::vector<ExprPtr>& args, Span span) const;

  /**
   * @brief Create a Call expression with kwargs for a registered operator
   *
   * Looks up the operator by name, validates arguments, deduces the result type
   * using both args and kwargs, and creates a Call expression with proper typing.
   *
   * @param op_name Name of the operator to call
   * @param args Positional Expr arguments
   * @param kwargs Keyword arguments (metadata)
   * @param span Source location information
   * @return Shared pointer to Call expression with deduced type
   * @throws ValueError if operator not found or invalid arguments
   */
  [[nodiscard]] CallPtr Create(const std::string& op_name, const std::vector<ExprPtr>& args,
                               const std::vector<std::pair<std::string, std::any>>& kwargs, Span span) const;

  /**
   * @brief Create a Call expression from user-facing parser/binding paths.
   *
   * Unlike compiler-internal ``Create`` calls, this path rejects operators
   * marked ``internal_only`` so builtin implementation details cannot be
   * reached by spelling their registry name in user code.
   */
  [[nodiscard]] CallPtr CreateUserFacing(const std::string& op_name, const std::vector<ExprPtr>& args,
                                         Span span) const;

  /**
   * @brief Create a user-facing Call expression with kwargs.
   */
  [[nodiscard]] CallPtr CreateUserFacing(const std::string& op_name, const std::vector<ExprPtr>& args,
                                         const std::vector<std::pair<std::string, std::any>>& kwargs,
                                         Span span) const;

  /**
   * @brief Create a Call expression for a compiler-internal operator.
   *
   * This explicit spelling is intended for passes that synthesize operators
   * marked ``internal_only``. User-facing bindings and parser helpers must keep
   * using ``CreateUserFacing`` so internal builtin ops cannot be reached by
   * name.
   */
  [[nodiscard]] CallPtr CreateInternal(const std::string& op_name, const std::vector<ExprPtr>& args,
                                       Span span) const;

  /**
   * @brief Create a compiler-internal Call expression with kwargs.
   */
  [[nodiscard]] CallPtr CreateInternal(const std::string& op_name, const std::vector<ExprPtr>& args,
                                       const std::vector<std::pair<std::string, std::any>>& kwargs,
                                       Span span) const;

  /**
   * @brief Check if an operator is registered
   *
   * @param op_name Name of the operator
   * @return true if the operator is registered
   */
  [[nodiscard]] bool IsRegistered(const std::string& op_name) const {
    return registry_.find(op_name) != registry_.end();
  }

  /**
   * @brief Get the operator registry entry by name
   *
   * @param op_name Name of the operator
   * @return Const reference to the operator registry entry
   * @throws ValueError if operator not found
   */
  [[nodiscard]] const OpRegistryEntry& GetEntry(const std::string& op_name) const;

  /**
   * @brief Get the operator instance by name
   *
   * @param op_name Name of the operator
   * @return Shared pointer to the operator instance
   * @throws ValueError if operator not found
   */
  [[nodiscard]] OpPtr GetOp(const std::string& op_name) const;

  /**
   * @brief Validate that all tile.* ops have a memory spec
   *
   * Checks every registered operator whose name starts with "tile." has either
   * a memory spec (via set_output_memory/set_input_memory/etc.) or an explicit
   * opt-out (via no_memory_spec()). Call at module init to catch missing specs
   * at import time.
   *
   * @throws ValueError listing all tile ops missing a memory spec
   */
  void ValidateTileOps() const;

  /**
   * @brief Validate that every operator which updates an argument in place has
   *        declared its per-argument effects.
   *
   * An operator declaring `set_output_reuses_input(N)` writes through argument
   * N — that is what reusing the buffer means. Direction inference, dependency
   * analysis and the parameter-direction verifier all read those effects, and
   * an undeclared operator reads as a pure consumer: the write vanishes, no
   * dependency edge is emitted, and the failure surfaces on device as a race
   * or a deadlock rather than at compile time.
   *
   * Classification, not a particular answer, is what is required: an operator
   * whose in-place slot is metadata rather than data may declare it `Read` (via
   * `no_arg_writes()`), which records that a human decided.
   *
   * Call at module init to catch an unclassified operator at import time.
   *
   * @throws ValueError listing every in-place operator with undeclared effects
   */
  void ValidateArgEffects() const;

 private:
  OpRegistry() = default;
  ~OpRegistry() = default;

  [[nodiscard]] CallPtr CreateImpl(const std::string& op_name, const std::vector<ExprPtr>& args,
                                   const std::vector<std::pair<std::string, std::any>>& kwargs, Span span,
                                   bool allow_internal) const;

  std::unordered_map<std::string, OpRegistryEntry> registry_;
};

/**
 * @brief Validate kwargs against allowed attributes
 *
 * Checks that all provided kwargs match registered attributes and have compatible types.
 * For DataType kwargs, accepts both DataType and int for backward compatibility.
 * MemorySpace kwargs require the MemorySpace enum type.
 *
 * @param kwargs The kwargs to validate
 * @param allowed_kwargs Map of allowed kwarg keys to expected types
 * @param op_name Operator name for error messages
 * @throws ValueError if unknown kwarg
 * @throws TypeError if type mismatch
 */
void ValidateKwargs(const std::vector<std::pair<std::string, std::any>>& kwargs,
                    const std::unordered_map<std::string, std::type_index>& allowed_kwargs,
                    const std::string& op_name);

/**
 * @brief Read a required kwarg by key from a deducer kwargs list, throwing if absent.
 *
 * Unlike `Call::GetKwarg` (which returns a default when the key is missing and
 * operates on an already-constructed Call), this is for op type-deduction
 * sites (`f_deduce_type`) that receive the raw kwargs vector before any Call
 * exists and treat the kwarg as mandatory. Shared by the distributed op
 * deducers (`pld.tensor.put`, `pld.system.notify`, `pld.system.wait`) so the
 * lookup-or-throw logic is defined once.
 *
 * @tparam T Expected type of the kwarg value
 * @param kwargs Keyword arguments (metadata) passed to the deducer
 * @param key Kwarg key to read
 * @param op_name Operator name, used in the error message
 * @return The kwarg value cast to T
 * @throws ValueError if the key is absent
 */
template <typename T>
T GetRequiredKwarg(const std::vector<std::pair<std::string, std::any>>& kwargs, const std::string& key,
                   const std::string& op_name) {
  for (const auto& [k, v] : kwargs) {
    if (k == key) {
      return AnyCast<T>(v, "kwarg key: " + key);
    }
  }
  throw ValueError("Missing kwarg '" + key + "' on " + op_name);
}

/**
 * @brief Operator check — a typo-safe alternative to a raw name-string literal.
 *
 * Tests whether `op` is the operator named `op_name`, routing the literal through
 * `OpRegistry::GetOp`, which throws `ValueError` if `op_name` is not a registered
 * operator. A mistyped or renamed operator literal therefore fails loudly at the
 * comparison site, instead of silently evaluating to `false` the way a bare
 * `op_->name_ == "..."` comparison does.
 *
 * The match itself is by *canonical name*, not pointer identity. `Op` instances
 * are constructed in several places — registry singletons, the `.pto`
 * deserializer (`deserializer.cpp`), and the MemRef alloc builders
 * (`memref_utils.h`) each create their own `Op` objects — so two `Op`s sharing a
 * name are the same operator yet distinct pointers. Name identity is the
 * invariant the IR maintains; pointer identity is not.
 *
 * Prefer these over raw name comparisons:
 * @code
 *   if (IsOp(call, "tile.reshape")) ...   // was: call->op_->name_ == "tile.reshape"
 *   if (!IsOp(call, "tile.store")) ...    // was: call->op_->name_ != "tile.store"
 * @endcode
 *
 * @param op       Operator pointer to test (a Call/Submit `op_`); may be null.
 * @param op_name  Registered operator name to compare against.
 * @return true iff `op` is non-null and names the registered operator `op_name`.
 * @throws ValueError if `op_name` is not a registered operator.
 */
[[nodiscard]] inline bool IsOp(const OpPtr& op, const std::string& op_name) {
  // GetOp throws on an unregistered name (typo-safety); evaluated unconditionally
  // so the guard fires even when `op` is null. Compare by canonical name.
  const OpPtr& canonical = OpRegistry::GetInstance().GetOp(op_name);
  return op && op->name_ == canonical->name_;
}

/// @overload Test a Call's operator (false when `call` or its op is null).
[[nodiscard]] inline bool IsOp(const CallPtr& call, const std::string& op_name) {
  return call && IsOp(call->op_, op_name);
}

/// @overload Test a Submit's operator (false when `submit` or its op is null).
[[nodiscard]] inline bool IsOp(const SubmitPtr& submit, const std::string& op_name) {
  return submit && IsOp(submit->op_, op_name);
}

/**
 * @brief Read an optional kwarg by key, returning a default when absent.
 *
 * The optional counterpart of `GetRequiredKwarg`: same lookup-and-`AnyCast`
 * logic, but returns `default_value` instead of throwing when the key is
 * missing. A present-but-wrong-typed value still raises a contextual
 * `TypeError` (via `AnyCast`). Shared by deducers and op-conversion lowerings
 * so the "scan kwargs, value-or-default" loop lives in one place.
 *
 * @tparam T Expected type of the kwarg value
 * @param kwargs Keyword arguments (metadata) passed to the deducer / conversion
 * @param key Kwarg key to read
 * @param default_value Value returned when the key is absent
 * @return The kwarg value cast to T, or `default_value` if the key is absent
 * @throws TypeError if the key is present but holds a different type
 */
template <typename T>
T GetKwargOr(const std::vector<std::pair<std::string, std::any>>& kwargs, const std::string& key,
             const T& default_value) {
  for (const auto& [k, v] : kwargs) {
    if (k == key) {
      return AnyCast<T>(v, "kwarg key: " + key);
    }
  }
  return default_value;
}

/**
 * @brief Helper macro for operator registration
 *
 * Use this macro to register operators in initialization code:
 * @code
 * REGISTER_OP("TensorAdd");
 * REGISTER_OP("TensorAdd");
 * @endcode
 */
#define REGISTER_OP(OpName)                                                                           \
  static PYPTO_STR_CONCAT(PYPTO_UNUSED ::pypto::ir::OpRegistryEntry& OpRegistryEntry_, __COUNTER__) = \
      ::pypto::ir::OpRegistry::GetInstance().Register(OpName)

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_OP_REGISTRY_H_
