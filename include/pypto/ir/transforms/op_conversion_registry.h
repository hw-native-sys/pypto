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

#ifndef PYPTO_IR_TRANSFORMS_OP_CONVERSION_REGISTRY_H_
#define PYPTO_IR_TRANSFORMS_OP_CONVERSION_REGISTRY_H_

#include <any>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/common.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"

namespace pypto {
namespace ir {

/**
 * @brief Result of an op conversion rule
 *
 * A conversion may produce:
 * - Simple: Just one tile op call (empty prologue, result expr only)
 * - Complex: Multiple prologue statements + a final result expression
 */
struct ConversionResult {
  std::vector<StmtPtr> prologue;  ///< Statements to insert before the assignment
  ExprPtr result;                 ///< The result expression

  /// Convenience: construct from Expr only (simple case)
  explicit ConversionResult(ExprPtr expr) : prologue{}, result{std::move(expr)} {}

  /// Full constructor (complex case)
  ConversionResult(std::vector<StmtPtr> stmts, ExprPtr expr)
      : prologue{std::move(stmts)}, result{std::move(expr)} {}
};

/**
 * @brief Signature for custom conversion functions
 *
 * @param args Positional arguments (already substituted to tile types)
 * @param kwargs Keyword arguments from the original call
 * @param span Source location of the original call
 * @return ConversionResult with optional prologue and result expression
 */
using ConversionFunc = std::function<ConversionResult(
    const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs,
    const Span& span)>;

/**
 * @brief A memory space that was *derived* from an operator's own declaration.
 *
 * The whole point of this type is that a conversion cannot write one. It has no
 * public constructor and no conversion from `MemorySpace`, so the only way to
 * obtain one is `BridgeSpaceOf` / `OperandSpaceOf` below, which read the value
 * out of the consuming operator's `set_input_memory`. `OpRegistry` therefore
 * stays the single place an operand's memory space is written -- enforced by the
 * type, not by a convention a new registration can quietly ignore.
 */
class OperandSpace {
 public:
  /// The derived space. Call this at the point of use; do not cache it beside
  /// the requirement, or the derivation stops being the single source again.
  [[nodiscard]] MemorySpace Get() const { return space_; }

 private:
  explicit OperandSpace(MemorySpace space) : space_(space) {}
  MemorySpace space_;

  friend OperandSpace OperandSpaceOf(std::initializer_list<std::string_view> op_names, size_t arg_index);
  friend OperandSpace BridgeSpaceOf(std::initializer_list<std::string_view> op_names, size_t arg_index);
};

/**
 * @brief The space `arg_index` is constrained to, read from the operators that
 *        consume it, for an operand materialised **in place**.
 *
 * Such an operand has no staging space at all: today that is
 * `tile.matmul_acc`'s accumulator, which nothing but the MAD unit writes, so
 * the requirement exists to reach the allocation site rather than to place a
 * load.
 *
 * `op_names` lists **every** operator the converter can emit for this operand,
 * and is a braced list even when there is one -- `OperandSpaceOf({"tile.foo"},
 * 0)`. A converter that dispatches on rank reaches more than one:
 * `tensor.matmul` emits `tile.matmul` or `tile.batch_matmul`, and naming only
 * the 2-D one would silently derive against an operator the call may never
 * become. All listed operators must declare the same constraint, which is an
 * invariant rather than a coincidence: `FlattenTileNdTo2D` unrolls the batched
 * op into the 2-D one, so their operands land in the same buffer by
 * construction. A divergence is a bug and throws here.
 *
 * Throws (`InternalError`) while the conversion registry is built when a listed
 * operator is unregistered, declares no constraint for `arg_index`, or
 * disagrees with its siblings.
 */
[[nodiscard]] OperandSpace OperandSpaceOf(std::initializer_list<std::string_view> op_names, size_t arg_index);

/**
 * @brief The space a *bridged* `tile.load` must target for that same operand.
 *
 * The constraint and the load target are not the same space, which is why this
 * is not `OperandSpaceOf`: `tile.matmul` constrains its operands to `Left` /
 * `Right`, but MTE2 fills no L0 buffer, so the bridge loads into `Mat` and
 * `InferTileMemorySpace` Phase 2 adds the `Mat -> L0` move. The mapping is
 * `StagingSpaceForLoad`, shared with that pass, which is what keeps a bridged
 * load and an inferred load placing the same operand in the same buffer.
 *
 * `op_names` follows the same rule as `OperandSpaceOf`. Additionally throws
 * when the constraint is reachable by no load even indirectly -- such an
 * operand has to be created where it is needed and takes `OperandSpaceOf`.
 */
[[nodiscard]] OperandSpace BridgeSpaceOf(std::initializer_list<std::string_view> op_names, size_t arg_index);

/**
 * @brief Per-input bridging requirement for a converter.
 *
 * Declares that a specific input operand has to reach the converter as a tile.
 * The framework inserts the `tile.load` for a `TensorType` input, and this
 * struct says where it lands and how it is shaped.
 *
 * `demanded_space` is **derived, not authored** -- see `OperandSpace`. A
 * registration names the operand it belongs to in one of two spellings and the
 * value is read from that operator's declaration:
 *
 * - `BridgeSpaceOf(...)` for an operand the bridge **loads**, which stages
 *   through `StagingSpaceForLoad`;
 * - `OperandSpaceOf(...)` for an operand materialised **in place**, which has
 *   no staging space at all.
 *
 * Which kind an entry is is visible at the registration instead of being
 * implied by a bare `MemorySpace` literal -- there is no way to write one here.
 */
struct InputSpaceReq {
  OperandSpace demanded_space;             ///< Derived via `BridgeSpaceOf` / `OperandSpaceOf`; never authored
  std::optional<std::string> trans_kwarg;  ///< Read transpose flag from this kwarg (if any)
  /// Whether this operand carries the matmul's **M** axis, i.e. an axis the MAD
  /// reads out of a whole number of NZ fractal boxes.  The logical extent of a
  /// cube operand is essentially free (``pto.mad`` derives ``%m`` from the
  /// operand's valid extent), but its *physical* extent on that axis must be a
  /// multiple of the box, so the bridged tile allocates the boxed extent and
  /// carries the tensor's true extent in ``valid_shape``.
  ///
  /// Which axis M lands on depends on this operand's own ``trans_kwarg``: an
  /// untransposed operand has M on rows, while a transposed one is loaded
  /// naturally with K on rows, so its M is the *column* axis.  The reduction
  /// axis (K) and the output axis (N) are never boxed here — padding K would
  /// feed uninitialised L1 into the sum.
  bool cube_m_axis = false;
  /// Index of the operand whose layout fixes the M alignment for the *whole*
  /// call, when that is not this operand itself.  Every cube tile M runs
  /// through has to agree on one padded extent — ``tile.matmul_acc`` requires
  /// the accumulator and the product to have the same physical M — but the
  /// granularity differs per tile: an Acc box is 16 rows for every dtype, while
  /// a transposed left operand's column box is ``32 / sizeof(dtype)`` (32 for
  /// INT8).  Naming one decider makes them share a single alignment.
  std::optional<size_t> m_align_from_arg;
  /// Whether this operand carries the matmul's **N** axis. The output axis is
  /// boxed on exactly the same grounds as M: only the *physical* extent is
  /// constrained, and N's padded cells land outside the result's valid region,
  /// so nothing reads them. K remains unboxable -- padding it would feed
  /// uninitialised L1 into the sum -- which is why an operand declares at most
  /// one of ``cube_m_axis`` / ``cube_n_axis``: its remaining axis is K.
  ///
  /// Which axis N lands on mirrors M: an untransposed right operand has N on
  /// its columns, while a transposed one is loaded naturally with K on columns,
  /// so its N is the *row* axis.
  bool cube_n_axis = false;
};

/**
 * @brief Full conversion entry: converter function + per-input space requirements.
 */
struct ConversionEntry {
  ConversionFunc func;
  std::unordered_map<size_t, InputSpaceReq> input_reqs;  ///< Per-input space requirements (key = arg index)
};

/**
 * @brief Registry mapping tensor op names to tile op conversion rules
 *
 * Supports two registration styles:
 * - Simple name mapping: tensor.add -> tile.add (auto-creates conversion)
 * - Custom converter: full ConversionFunc for complex conversions
 *
 * Re-registering the same op name replaces the previous rule (override semantics).
 */
class OpConversionRegistry {
 public:
  OpConversionRegistry(const OpConversionRegistry&) = delete;
  OpConversionRegistry& operator=(const OpConversionRegistry&) = delete;

  /**
   * @brief Get the singleton instance
   */
  static OpConversionRegistry& GetInstance();

  /**
   * @brief Register a simple name mapping (tensor op -> tile op)
   *
   * Creates a ConversionFunc that calls OpRegistry::Create with the target name.
   * Re-registering the same from_op replaces the previous rule.
   *
   * @param from_op Source op name (e.g., "tensor.add")
   * @param to_op Target op name (e.g., "tile.add")
   * @param input_reqs Per-input memory space requirements (default: none)
   */
  void RegisterSimple(const std::string& from_op, const std::string& to_op,
                      std::unordered_map<size_t, InputSpaceReq> input_reqs = {});

  /**
   * @brief Register a custom conversion function
   *
   * Re-registering the same from_op replaces the previous rule.
   *
   * @param from_op Source op name (e.g., "tensor.matmul")
   * @param func Custom conversion function
   * @param input_reqs Per-input memory space requirements (default: none)
   */
  void RegisterCustom(const std::string& from_op, ConversionFunc func,
                      std::unordered_map<size_t, InputSpaceReq> input_reqs = {});

  /**
   * @brief Look up a conversion entry for an op
   *
   * @param op_name The operator name to look up
   * @return Pointer to the ConversionEntry, or nullptr if not registered
   */
  [[nodiscard]] const ConversionEntry* Lookup(const std::string& op_name) const;

  /**
   * @brief Check if a conversion rule exists for an op
   */
  [[nodiscard]] bool HasConversion(const std::string& op_name) const;

 private:
  OpConversionRegistry();

  void RegisterScalarAndUnaryOps();
  void RegisterBroadcastAndTransformOps();
  void RegisterElementwiseBinaryOps();
  void RegisterMemoryOps();
  void RegisterMatmulOps();
  void RegisterReductionOps();
  void RegisterSortOps();
  void RegisterGatherOps();
  void RegisterPagedGatherOps();
  void RegisterScatterOps();
  void RegisterCmpOps();
  void RegisterDistributedOps();
  void RegisterCrossCoreOps();

  std::unordered_map<std::string, ConversionEntry> conversions_;
};

/**
 * @brief Helper macro for simple op conversion registration
 *
 * Currently unused, and adding a use needs care: it registers from a static
 * initializer, which would build the `OpConversionRegistry` singleton while
 * `OpRegistry` may still be partly empty. `BridgeSpaceOf` / `OperandSpaceOf`
 * read `OpRegistry` where they are called, so a requirement built from a static
 * initializer can race operator registration. Both are callable from any
 * translation unit; call them from registration code that runs after static
 * init -- as all of it does today, where the singleton is first built from pass
 * code.
 */
#define REGISTER_OP_CONVERSION(FromOp, ToOp)                \
  static bool PYPTO_STR_CONCAT(op_conv_reg_, __COUNTER__) = \
      (::pypto::ir::OpConversionRegistry::GetInstance().RegisterSimple(FromOp, ToOp), true)

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_OP_CONVERSION_REGISTRY_H_
