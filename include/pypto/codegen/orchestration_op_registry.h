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

#ifndef PYPTO_CODEGEN_ORCHESTRATION_OP_REGISTRY_H_
#define PYPTO_CODEGEN_ORCHESTRATION_OP_REGISTRY_H_

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
 * @brief Registry for orchestration-level operation codegen functions
 *
 * Orchestration operations (tensor.create, tensor.read, etc.) are host-side operations
 * that don't depend on backend hardware. This registry provides a mechanism to register
 * codegen functions for these operations, similar to backend operation registration but
 * at the orchestration layer.
 */
class OrchestrationOpRegistry {
 public:
  /**
   * @brief Codegen function signature for orchestration operations
   *
   * Takes a Call expression and a CodegenBase reference, returns generated C++ code string.
   */
  using CodegenFunc = std::function<std::string(const ir::CallPtr&, CodegenBase&)>;

  /**
   * @brief Get the singleton instance
   */
  static OrchestrationOpRegistry& GetInstance();

  /**
   * @brief Register a codegen function for an operation
   *
   * @param op_name Operation name (e.g., "tensor.create")
   * @param func Codegen function
   */
  void Register(const std::string& op_name, CodegenFunc func);

  /**
   * @brief Get the codegen function for an operation
   *
   * @param op_name Operation name
   * @return Codegen function if registered, nullopt otherwise
   */
  [[nodiscard]] std::optional<CodegenFunc> Get(const std::string& op_name) const;

 private:
  OrchestrationOpRegistry() = default;
  std::unordered_map<std::string, CodegenFunc> registry_;
};

/**
 * @brief Helper class for registering orchestration operations
 *
 * Used by REGISTER_ORCHESTRATION_OP macro to enable registration at static initialization time.
 */
class OrchestrationOpRegistryEntry {
 public:
  explicit OrchestrationOpRegistryEntry(std::string op_name) : op_name_(std::move(op_name)) {}

  /**
   * @brief Set the codegen function
   */
  OrchestrationOpRegistryEntry& SetCodegen(OrchestrationOpRegistry::CodegenFunc func) {
    OrchestrationOpRegistry::GetInstance().Register(op_name_, std::move(func));
    return *this;
  }

 private:
  std::string op_name_;
};

}  // namespace codegen
}  // namespace pypto

/**
 * @brief Macro for registering orchestration operation codegen
 *
 * Usage:
 *   REGISTER_ORCHESTRATION_OP(tensor_create, "tensor.create") {
 *     // op: const ir::CallPtr&
 *     // codegen: CodegenBase&
 *     return "generated code";
 *   }
 */
#define REGISTER_ORCHESTRATION_OP(func_name, op_name_str)                                      \
  static std::string OrchestrationCodegen_##func_name(const ::pypto::ir::CallPtr& op,          \
                                                      ::pypto::codegen::CodegenBase& codegen); \
  static ::pypto::codegen::OrchestrationOpRegistryEntry __orch_op_entry_##func_name =          \
      ::pypto::codegen::OrchestrationOpRegistryEntry(op_name_str)                              \
          .SetCodegen(OrchestrationCodegen_##func_name);                                       \
  static std::string OrchestrationCodegen_##func_name(const ::pypto::ir::CallPtr& op,          \
                                                      ::pypto::codegen::CodegenBase& codegen)

#endif  // PYPTO_CODEGEN_ORCHESTRATION_OP_REGISTRY_H_
