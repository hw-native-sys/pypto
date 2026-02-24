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

#ifndef PYPTO_BACKEND_COMMON_BACKEND_REGISTRY_H_
#define PYPTO_BACKEND_COMMON_BACKEND_REGISTRY_H_

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/soc.h"

namespace pypto {
namespace backend {

/**
 * @brief Registry for backend types
 *
 * Singleton registry that maps backend type names to factory functions.
 * Used for deserialization to create the correct backend subclass.
 */
class BackendRegistry {
 public:
  using CreateFunc = std::function<std::unique_ptr<Backend>(std::shared_ptr<const SoC>)>;

  /**
   * @brief Get the singleton instance
   *
   * @return Reference to the global registry
   */
  static BackendRegistry& Instance();

  /**
   * @brief Register a backend type
   *
   * @param type_name Backend type name (e.g., "910B_CCE", "910B_PTO")
   * @param func Factory function that creates backend from SoC
   */
  void Register(const std::string& type_name, CreateFunc func);

  /**
   * @brief Create a backend instance (deprecated for singleton backends)
   *
   * Backends are singletons; this always throws. Use Backend910B_CCE::Instance()
   * or Backend910B_PTO::Instance() instead.
   *
   * @param type_name Backend type name
   * @param soc SoC structure (includes memory hierarchy)
   * @throws ValueError always (backends are singletons, not created via registry)
   */
  std::unique_ptr<Backend> Create(const std::string& type_name, const std::shared_ptr<const SoC>& soc);

  /**
   * @brief Check if a type is registered
   *
   * @param type_name Backend type name to check
   * @return true if registered, false otherwise
   */
  [[nodiscard]] bool IsRegistered(const std::string& type_name) const;

 private:
  BackendRegistry() = default;
  std::unordered_map<std::string, CreateFunc> registry_;
};

/**
 * @brief Create backend from registry (deprecated for singleton backends)
 *
 * Backends are singletons; this always throws. Use Backend910B_CCE::Instance()
 * or Backend910B_PTO::Instance() instead.
 *
 * @param type_name Backend type name
 * @param soc SoC structure (includes memory hierarchy)
 * @throws ValueError always (backends are singletons, not created via registry)
 */
std::unique_ptr<Backend> CreateBackendFromRegistry(const std::string& type_name,
                                                   const std::shared_ptr<const SoC>& soc);

}  // namespace backend
}  // namespace pypto

#endif  // PYPTO_BACKEND_COMMON_BACKEND_REGISTRY_H_
