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
#include <optional>
#include <string>
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
 * @brief Per-input memory space requirement for a converter.
 *
 * Declares that a specific input operand must reside in a particular memory space.
 * The framework uses this to automatically insert tile.load (for TensorType inputs)
 * before calling the converter.
 */
struct InputSpaceReq {
  MemorySpace space;                       ///< Required memory space
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
 */
#define REGISTER_OP_CONVERSION(FromOp, ToOp)                \
  static bool PYPTO_STR_CONCAT(op_conv_reg_, __COUNTER__) = \
      (::pypto::ir::OpConversionRegistry::GetInstance().RegisterSimple(FromOp, ToOp), true)

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_OP_CONVERSION_REGISTRY_H_
