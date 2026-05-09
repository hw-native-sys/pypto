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

#ifndef PYPTO_IR_PROGRAM_H_
#define PYPTO_IR_PROGRAM_H_

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/reflection/field_traits.h"
#include "pypto/ir/span.h"

namespace pypto {
namespace ir {

/**
 * @brief Per-rank allocation spec for one named CommGroup HCCL window buffer.
 *
 * Maps 1:1 to ``simpler.task_interface.ChipBufferSpec`` at submit-time. Pure
 * allocation metadata: does NOT describe how the buffer is used in code.
 * Code-level use is expressed in the function signature via
 * ``pld.DistributedTensor[[shape], dtype]``; the alloc op
 * (``pld.alloc_window_buffer``, added in N2) materialises one of these slots
 * into the program's :class:`CommGroup`.
 *
 * ``size_`` is the **element count** of one rank's slice (a single scalar; this
 * struct is allocation-only and intentionally does not carry a multi-dim
 * shape). ``size_`` may be a ``ConstInt`` (compile-time known) or a symbolic
 * expression referring to the world size.
 *
 * ``load_from_host_`` / ``store_to_host_`` are simple boolean flags marking
 * whether the slot participates in pre-fork H2D / post-task D2H staging. The
 * specific host tensor that supplies / receives the staged data is recorded
 * on the alloc op, not on this allocation spec.
 */
class WindowBuffer : public IRNode {
 public:
  std::string name_;             ///< Buffer name (parser-extracted from alloc-op LHS)
  ExprPtr size_;                 ///< Per-rank element count (ConstInt or symbolic Expr)
  DataType dtype_;               ///< Element data type
  bool load_from_host_ = false;  ///< Pre-fork H2D copy from a host staging tensor
  bool store_to_host_ = false;   ///< Post-task D2H copy back into a host staging tensor

  WindowBuffer(std::string name, ExprPtr size, DataType dtype, bool load_from_host = false,
               bool store_to_host = false, Span span = Span::unknown())
      : IRNode(std::move(span)),
        name_(std::move(name)),
        size_(std::move(size)),
        dtype_(dtype),
        load_from_host_(load_from_host),
        store_to_host_(store_to_host) {}

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::WindowBuffer; }
  [[nodiscard]] std::string TypeName() const override { return "WindowBuffer"; }

  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(
        IRNode::GetFieldDescriptors(),
        std::make_tuple(reflection::UsualField(&WindowBuffer::name_, "name"),
                        reflection::UsualField(&WindowBuffer::size_, "size"),
                        reflection::UsualField(&WindowBuffer::dtype_, "dtype"),
                        reflection::UsualField(&WindowBuffer::load_from_host_, "load_from_host"),
                        reflection::UsualField(&WindowBuffer::store_to_host_, "store_to_host")));
  }
};

using WindowBufferPtr = std::shared_ptr<const WindowBuffer>;

/**
 * @brief A communication group inferred for a ``@pl.program``.
 *
 * The ``CollectCommGroups`` pass (N4) builds these from
 * ``pld.alloc_window_buffer`` ops and their dispatch coverage. The runtime
 * (``distributed_runner``) uses this to compose ``ChipBootstrapConfig`` before
 * bringing the workers up.
 *
 * ``devices_`` is the ascending-sorted set of physical device ids covered by
 * the group. **An empty vector means "all devices"** (every entry of
 * ``DistributedConfig.device_ids``, resolved by the driver at submit-time).
 */
class CommGroup : public IRNode {
 public:
  std::vector<int64_t> devices_;        ///< Covered device ids (ascending); empty = all devices
  std::vector<WindowBufferPtr> slots_;  ///< Allocation slots in this group (alloc-order)

  CommGroup(std::vector<int64_t> devices, std::vector<WindowBufferPtr> slots, Span span = Span::unknown())
      : IRNode(std::move(span)), devices_(std::move(devices)), slots_(std::move(slots)) {}

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::CommGroup; }
  [[nodiscard]] std::string TypeName() const override { return "CommGroup"; }

  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(IRNode::GetFieldDescriptors(),
                          std::make_tuple(reflection::UsualField(&CommGroup::devices_, "devices"),
                                          reflection::UsualField(&CommGroup::slots_, "slots")));
  }
};

using CommGroupPtr = std::shared_ptr<const CommGroup>;

/**
 * @brief Program definition
 *
 * Represents a complete program with functions mapped by GlobalVar references.
 * Programs are immutable IR nodes.
 *
 * Functions are stored in a sorted map (by GlobalVar name) to ensure deterministic
 * ordering for structural equality and hashing.
 *
 * @note The GlobalVar name must match the function name and be unique within the program.
 *       Validation of this constraint may be added in future passes.
 */
class Program : public IRNode {
 public:
  /**
   * @brief Create a program from a map of GlobalVars to Functions
   *
   * @param functions Map of GlobalVar references to their corresponding functions
   * @param name Program name (optional)
   * @param span Source location
   */
  Program(std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> functions, std::string name, Span span)
      : IRNode(std::move(span)), functions_(std::move(functions)), name_(std::move(name)) {}

  /**
   * @brief Map-based ctor with CommGroup metadata (used by the deserializer).
   */
  Program(std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> functions,
          std::vector<CommGroupPtr> comm_groups, std::string name, Span span)
      : IRNode(std::move(span)),
        functions_(std::move(functions)),
        name_(std::move(name)),
        comm_groups_(std::move(comm_groups)) {}

  /**
   * @brief Create a program from a list of functions
   *
   * Convenience constructor that creates GlobalVar references for each function
   * using the function's name. Functions are automatically sorted by name in the map.
   *
   * @param functions List of functions
   * @param name Program name (optional)
   * @param span Source location
   */
  Program(const std::vector<FunctionPtr>& functions, std::string name, Span span);

  /**
   * @brief Create a program from a list of functions and CommGroup metadata.
   *
   * @param functions List of functions
   * @param comm_groups List of CommGroups declared on the program
   * @param name Program name (optional)
   * @param span Source location
   */
  Program(const std::vector<FunctionPtr>& functions, std::vector<CommGroupPtr> comm_groups, std::string name,
          Span span);

  [[nodiscard]] ObjectKind GetKind() const override { return ObjectKind::Program; }
  [[nodiscard]] std::string TypeName() const override { return "Program"; }

  /**
   * @brief Get a function by name
   *
   * @param name Function name to look up
   * @return Shared pointer to the function, or nullptr if not found
   */
  [[nodiscard]] FunctionPtr GetFunction(const std::string& name) const;

  /**
   * @brief Get a GlobalVar by name
   *
   * @param name GlobalVar name to look up
   * @return Shared pointer to the GlobalVar, or nullptr if not found
   */
  [[nodiscard]] GlobalVarPtr GetGlobalVar(const std::string& name) const;

  /**
   * @brief Get field descriptors for reflection-based visitation.
   *
   * ``comm_groups_`` participates in structural equality / hashing via
   * ``UsualField``: two programs declaring the same CommGroups (same names,
   * sizes, dtypes, host-staging flags) are structurally equivalent.
   */
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(IRNode::GetFieldDescriptors(),
                          std::make_tuple(reflection::IgnoreField(&Program::name_, "name"),
                                          reflection::UsualField(&Program::functions_, "functions"),
                                          reflection::UsualField(&Program::comm_groups_, "comm_groups")));
  }

 public:
  std::string name_;                                                 // Program name
  std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> functions_;  // Map of GlobalVars to Functions
  std::vector<CommGroupPtr> comm_groups_;                            // CommGroups (host-side metadata)
};

using ProgramPtr = std::shared_ptr<const Program>;

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_PROGRAM_H_
