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

#ifndef PYPTO_CODEGEN_DISTRIBUTED_DISTRIBUTED_OP_REGISTRY_H_
#define PYPTO_CODEGEN_DISTRIBUTED_DISTRIBUTED_OP_REGISTRY_H_

#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

#include "pypto/codegen/codegen_base.h"
#include "pypto/ir/expr.h"

namespace pypto {
namespace codegen {

/**
 * @brief Registry for distributed host-orch Python codegen functions.
 *
 * Mirrors :class:`OrchestrationOpRegistry` (host C++ codegen) and the
 * per-backend ``Backend::RegisterOp`` mechanism (PTO MLIR codegen). Each
 * registered function generates the **Python** form of an op for the
 * distributed host_orch module emitted by :class:`DistributedCodegen`.
 *
 * Registered functions either:
 *   * return a non-empty string — used by the caller as the RHS expression
 *     of a Python assignment (``var = <returned>``), or
 *   * return the empty string after calling ``codegen.Emit(...)`` to write
 *     one or more lines directly — used when the op needs a different LHS
 *     shape (e.g. ``tensors["x"] = ...``) or no LHS at all (markers).
 */
class DistributedOpRegistry {
 public:
  using CodegenFunc = std::function<std::string(const ir::CallPtr&, CodegenBase&)>;

  static DistributedOpRegistry& GetInstance();

  void Register(const std::string& op_name, CodegenFunc func);

  [[nodiscard]] std::optional<CodegenFunc> Get(const std::string& op_name) const;

 private:
  DistributedOpRegistry() = default;
  std::unordered_map<std::string, CodegenFunc> registry_;
};

/**
 * @brief Helper for static-initialisation-time op registration.
 *
 * Used by the :c:macro:`REGISTER_DISTRIBUTED_OP` macro.
 */
class DistributedOpRegistryEntry {
 public:
  explicit DistributedOpRegistryEntry(std::string op_name) : op_name_(std::move(op_name)) {}

  DistributedOpRegistryEntry& SetCodegen(DistributedOpRegistry::CodegenFunc func) {
    DistributedOpRegistry::GetInstance().Register(op_name_, std::move(func));
    return *this;
  }

 private:
  std::string op_name_;
};

}  // namespace codegen
}  // namespace pypto

/**
 * @brief Register a distributed host-orch Python codegen function.
 *
 * Usage:
 *   REGISTER_DISTRIBUTED_OP(tensor_slice, "tensor.slice") {
 *     // op:      const ir::CallPtr&
 *     // codegen: codegen::CodegenBase&  (dynamic_cast to DistributedCodegen if needed)
 *     codegen.Emit("tensors[\"x\"] = ...");
 *     return "";  // returning empty signals "already emitted, no RHS to splice"
 *   }
 */
#define REGISTER_DISTRIBUTED_OP(func_name, op_name_str)                                                     \
  static std::string DistributedCodegen_##func_name(const ::pypto::ir::CallPtr& op,                         \
                                                    ::pypto::codegen::CodegenBase& codegen);                \
  static ::pypto::codegen::DistributedOpRegistryEntry __dist_op_entry_##func_name =                         \
      ::pypto::codegen::DistributedOpRegistryEntry(op_name_str).SetCodegen(DistributedCodegen_##func_name); \
  static std::string DistributedCodegen_##func_name(const ::pypto::ir::CallPtr& op,                         \
                                                    ::pypto::codegen::CodegenBase& codegen)

#endif  // PYPTO_CODEGEN_DISTRIBUTED_DISTRIBUTED_OP_REGISTRY_H_
