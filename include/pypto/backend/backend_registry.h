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

#ifndef PYPTO_BACKEND_BACKEND_REGISTRY_H_
#define PYPTO_BACKEND_BACKEND_REGISTRY_H_

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "pypto/backend/backend.h"
#include "pypto/backend/soc.h"

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
  using CreateFunc = std::function<std::unique_ptr<Backend>(
      std::shared_ptr<const SoC>, std::map<ir::MemorySpace, std::vector<ir::MemorySpace>>)>;

  /**
   * @brief Get the singleton instance
   *
   * @return Reference to the global registry
   */
  static BackendRegistry& Instance();

  /**
   * @brief Register a backend type
   *
   * @param type_name Backend type name (e.g., "910B")
   * @param func Factory function that creates backend from SoC and memory graph
   */
  void Register(const std::string& type_name, CreateFunc func);

  /**
   * @brief Create a backend instance
   *
   * @param type_name Backend type name
   * @param soc SoC structure
   * @param mem_graph Memory hierarchy adjacency list
   * @return Unique pointer to created backend
   * @throws RuntimeError if type_name is not registered
   */
  std::unique_ptr<Backend> Create(const std::string& type_name, std::shared_ptr<const SoC> soc,
                                  const std::map<ir::MemorySpace, std::vector<ir::MemorySpace>>& mem_graph);

  /**
   * @brief Check if a type is registered
   *
   * @param type_name Backend type name to check
   * @return true if registered, false otherwise
   */
  bool IsRegistered(const std::string& type_name) const;

 private:
  BackendRegistry() = default;
  std::unordered_map<std::string, CreateFunc> registry_;
};

/**
 * @brief Helper function for backend deserialization
 *
 * Called by Backend::ImportFromFile to create backend from registry.
 *
 * @param type_name Backend type name
 * @param soc SoC structure
 * @param mem_graph Memory hierarchy adjacency list
 * @return Unique pointer to created backend
 */
std::unique_ptr<Backend> CreateBackendFromRegistry(
    const std::string& type_name, std::shared_ptr<const SoC> soc,
    const std::map<ir::MemorySpace, std::vector<ir::MemorySpace>>& mem_graph);

}  // namespace backend
}  // namespace pypto

#endif  // PYPTO_BACKEND_BACKEND_REGISTRY_H_
