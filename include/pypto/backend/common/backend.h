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

#ifndef PYPTO_BACKEND_COMMON_BACKEND_H_
#define PYPTO_BACKEND_COMMON_BACKEND_H_

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/backend/common/soc.h"
#include "pypto/core/common.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/pipe.h"

namespace pypto {

// Forward declarations
namespace codegen {
class CodegenBase;
}  // namespace codegen

namespace ir {
class Call;
using CallPtr = std::shared_ptr<const Call>;
}  // namespace ir

namespace backend {

// Forward declaration (required for GetBackendInstance return type)
class Backend;

/**
 * @brief Backend type identifier for selecting backend instance
 *
 * Used by InsertSyncPass, InferFunctionCoreType, GenerateOrchestration and compile
 * to obtain the corresponding backend instance via GetBackendInstance().
 */
enum class BackendType {
  CCE,  ///< 910B CCE backend (C++ codegen)
  PTO   ///< 910B PTO backend (PTO assembly codegen)
};

/**
 * @brief Get the singleton backend instance for the given type
 *
 * @param type Backend type (CCE or PTO)
 * @return Pointer to the backend instance (never null)
 */
const Backend* GetBackendInstance(BackendType type);

// Backend op code generation function type
using BackendCodegenFunc = std::function<std::string(const ir::CallPtr& op, codegen::CodegenBase& codegen)>;

/**
 * @brief Backend op registration entry for fluent interface
 *
 * Provides a fluent interface for registering backend-specific operator
 * information (pipe type and code generation function). The entry is
 * automatically finalized in the destructor.
 */
class BackendOpRegistryEntry {
 public:
  /**
   * @brief Construct registration entry
   *
   * @param backend Backend instance to register to
   * @param op_name Operator name
   */
  BackendOpRegistryEntry(Backend* backend, const std::string& op_name)
      : backend_(backend), op_name_(op_name) {}

  /**
   * @brief Set pipeline type
   *
   * @param pipe Pipeline type (e.g., M, V, MTE2)
   * @return Reference to this entry for method chaining
   */
  BackendOpRegistryEntry& set_pipe(ir::PipeType pipe);

  /**
   * @brief Set code generation function
   *
   * @param func Code generation function
   * @return Reference to this entry for method chaining
   */
  BackendOpRegistryEntry& f_codegen(BackendCodegenFunc func);

  /**
   * @brief Finalize registration in destructor
   *
   * Automatically registers the operator with the backend if both
   * pipe type and codegen function are set.
   */
  ~BackendOpRegistryEntry();

  // Disable copy and move to prevent duplicate registration
  BackendOpRegistryEntry(const BackendOpRegistryEntry&) = delete;
  BackendOpRegistryEntry& operator=(const BackendOpRegistryEntry&) = delete;
  BackendOpRegistryEntry(BackendOpRegistryEntry&&) = delete;
  BackendOpRegistryEntry& operator=(BackendOpRegistryEntry&&) = delete;

 private:
  Backend* backend_;
  std::string op_name_;
  std::optional<ir::PipeType> pipe_;
  std::optional<BackendCodegenFunc> codegen_func_;
};

// Macro for registering backend operators with fluent interface
#define REGISTER_BACKEND_OP(BackendClass, OpName)                                                 \
  static PYPTO_STR_CONCAT(PYPTO_UNUSED ::pypto::backend::BackendOpRegistryEntry& BackendOpEntry_, \
                          __COUNTER__) = BackendClass::Instance().RegisterOp(OpName)

/**
 * @brief Abstract backend base class
 *
 * Represents a hardware backend configuration with SoC structure.
 * Provides serialization/deserialization, operator registration,
 * and abstract methods for backend-specific operations.
 */
class Backend {
 public:
  /**
   * @brief Backend operator information
   *
   * Stores backend-specific operator metadata including pipeline type
   * and code generation function.
   */
  struct BackendOpInfo {
    ir::PipeType pipe;
    BackendCodegenFunc codegen_func;
  };

  virtual ~Backend() = default;

  // Disable copy and move to enforce unique ownership
  Backend(const Backend&) = delete;
  Backend& operator=(const Backend&) = delete;
  Backend(Backend&&) = delete;
  Backend& operator=(Backend&&) = delete;

  /**
   * @brief Register an operator with backend-specific information
   *
   * Returns a registration entry for fluent interface configuration.
   *
   * @param op_name Operator name
   * @return Registration entry for method chaining
   */
  BackendOpRegistryEntry RegisterOp(const std::string& op_name);

  /**
   * @brief Finalize operator registration
   *
   * Internal method called by BackendOpRegistryEntry destructor.
   *
   * @param op_name Operator name
   * @param pipe Pipeline type
   * @param func Code generation function
   */
  void FinalizeOpRegistration(const std::string& op_name, ir::PipeType pipe, BackendCodegenFunc func);

  /**
   * @brief Get backend-specific operator information
   *
   * @param op_name Operator name
   * @return Pointer to operator info, or nullptr if not registered
   */
  [[nodiscard]] const BackendOpInfo* GetOpInfo(const std::string& op_name) const;

  /**
   * @brief Export backend to msgpack file
   *
   * @param path File path to export to
   * @throws RuntimeError if file cannot be written
   */
  void ExportToFile(const std::string& path) const;

  /**
   * @brief Import backend from msgpack file
   *
   * @param path File path to import from
   * @return Unique pointer to backend instance
   * @throws RuntimeError if file cannot be read or parsed
   */
  static std::unique_ptr<Backend> ImportFromFile(const std::string& path);

  /**
   * @brief Find memory path from source to destination
   *
   * Uses BFS to find shortest path through memory hierarchy.
   *
   * @param from Source memory space
   * @param to Destination memory space
   * @return Vector of memory spaces in the path (including from and to)
   */
  [[nodiscard]] std::vector<ir::MemorySpace> FindMemPath(ir::MemorySpace from, ir::MemorySpace to) const;

  /**
   * @brief Get memory size for a specific memory type
   *
   * Returns the size of a single memory component of the given type.
   * If the type exists in multiple cores, returns the size from the first occurrence.
   *
   * @param mem_type Memory space type
   * @return Memory size in bytes, or 0 if not found
   */
  [[nodiscard]] uint64_t GetMemSize(ir::MemorySpace mem_type) const;

  /**
   * @brief Get backend type name for serialization
   *
   * @return Backend type name (e.g., "910B_CCE", "910B_PTO")
   */
  [[nodiscard]] virtual std::string GetTypeName() const = 0;

  /**
   * @brief Get the SoC structure
   *
   * @return Const reference to SoC
   */
  [[nodiscard]] const SoC& GetSoC() const { return *soc_; }

 protected:
  /**
   * @brief Construct backend with SoC
   *
   * Protected constructor - only derived classes can instantiate Backend.
   *
   * @param soc Immutable SoC structure (includes memory hierarchy)
   */
  explicit Backend(const SoC& soc) : soc_(&soc) {}

  Backend() = default;

  const SoC* soc_{nullptr};
  std::unordered_map<std::string, BackendOpInfo> backend_op_registry_{};
};

}  // namespace backend
}  // namespace pypto

#endif  // PYPTO_BACKEND_COMMON_BACKEND_H_
