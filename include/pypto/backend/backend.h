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

#ifndef PYPTO_BACKEND_BACKEND_H_
#define PYPTO_BACKEND_BACKEND_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "pypto/backend/soc.h"
#include "pypto/ir/memref.h"

namespace pypto {
namespace backend {

/**
 * @brief Abstract backend base class
 *
 * Represents a hardware backend configuration with SoC structure.
 * Provides serialization/deserialization and abstract methods for
 * backend-specific operations.
 */
class Backend {
 public:
  virtual ~Backend() = default;

  // Disable copy and move to enforce unique ownership
  Backend(const Backend&) = delete;
  Backend& operator=(const Backend&) = delete;
  Backend(Backend&&) = delete;
  Backend& operator=(Backend&&) = delete;

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
   * @return Backend type name (e.g., "910B")
   */
  [[nodiscard]] virtual std::string GetTypeName() const = 0;

  /**
   * @brief Get the SoC structure
   *
   * @return Shared pointer to const SoC
   */
  [[nodiscard]] SoCPtr GetSoC() const { return soc_; }

  /**
   * @brief Get the memory hierarchy graph
   *
   * @return Const reference to memory adjacency list
   */
  [[nodiscard]] const std::map<ir::MemorySpace, std::vector<ir::MemorySpace>>& GetMemoryGraph() const {
    return mem_graph_;
  }

 protected:
  /**
   * @brief Construct backend with SoC and memory hierarchy
   *
   * Protected constructor - only derived classes can instantiate Backend.
   *
   * @param soc Immutable SoC structure
   * @param mem_graph Memory hierarchy adjacency list
   */
  Backend() = default;

  SoCPtr soc_{nullptr};
  std::map<ir::MemorySpace, std::vector<ir::MemorySpace>> mem_graph_{};
};

}  // namespace backend
}  // namespace pypto

#endif  // PYPTO_BACKEND_BACKEND_H_
