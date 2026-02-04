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

#ifndef PYPTO_BACKEND_SOC_H_
#define PYPTO_BACKEND_SOC_H_

#include <cstdint>
#include <map>
#include <memory>
#include <vector>

#include "pypto/ir/memref.h"
#include "pypto/ir/pipe.h"

namespace pypto {
namespace backend {

// Forward declarations
class Mem;
class Core;
class Cluster;
class Die;
class SoC;

using MemPtr = std::shared_ptr<const Mem>;
using CorePtr = std::shared_ptr<const Core>;
using ClusterPtr = std::shared_ptr<const Cluster>;
using DiePtr = std::shared_ptr<const Die>;
using SoCPtr = std::shared_ptr<const SoC>;

/**
 * @brief Memory component
 *
 * Represents a memory unit with type, size, and alignment requirements.
 * Immutable once constructed.
 */
class Mem {
 public:
  /**
   * @brief Construct a memory component
   *
   * @param mem_type Memory space type
   * @param mem_size Memory size in bytes
   * @param alignment Memory alignment requirement in bytes
   */
  Mem(ir::MemorySpace mem_type, uint64_t mem_size, uint64_t alignment);

  // Allow copy and move for container storage
  Mem(const Mem&) = default;
  Mem& operator=(const Mem&) = default;
  Mem(Mem&&) = default;
  Mem& operator=(Mem&&) = default;

  [[nodiscard]] ir::MemorySpace GetMemType() const { return mem_type_; }
  [[nodiscard]] uint64_t GetMemSize() const { return mem_size_; }
  [[nodiscard]] uint64_t GetAlignment() const { return alignment_; }

  // Comparison operators for map key support
  bool operator<(const Mem& other) const;
  bool operator==(const Mem& other) const;

 private:
  ir::MemorySpace mem_type_;
  uint64_t mem_size_;
  uint64_t alignment_;
};

/**
 * @brief Processing core with associated memories
 *
 * Contains core type and list of memory components.
 * Immutable once constructed.
 */
class Core {
 public:
  /**
   * @brief Construct a processing core
   *
   * @param core_type Type of core (CUBE or VECTOR)
   * @param mems Vector of memory components
   */
  Core(ir::CoreType core_type, std::vector<Mem> mems);

  // Allow copy and move for container storage
  Core(const Core&) = default;
  Core& operator=(const Core&) = default;
  Core(Core&&) = default;
  Core& operator=(Core&&) = default;

  [[nodiscard]] ir::CoreType GetCoreType() const { return core_type_; }
  [[nodiscard]] const std::vector<Mem>& GetMems() const { return mems_; }

  // Comparison operators for map key support
  bool operator<(const Core& other) const;
  bool operator==(const Core& other) const;

 private:
  ir::CoreType core_type_;
  std::vector<Mem> mems_;
};

/**
 * @brief Cluster of processing cores
 *
 * Contains a map of core configurations to their counts.
 * map<Core, int> stores different core types and how many of each.
 * Immutable once constructed.
 */
class Cluster {
 public:
  /**
   * @brief Construct a cluster
   *
   * @param core_counts Map from core configuration to count
   */
  explicit Cluster(std::map<Core, int> core_counts);

  /**
   * @brief Convenience constructor for single core type
   *
   * @param core Single core configuration
   * @param count Number of cores with this configuration
   */
  Cluster(const Core& core, int count);

  // Allow copy and move for container storage
  Cluster(const Cluster&) = default;
  Cluster& operator=(const Cluster&) = default;
  Cluster(Cluster&&) = default;
  Cluster& operator=(Cluster&&) = default;

  [[nodiscard]] const std::map<Core, int>& GetCoreCounts() const { return core_counts_; }
  [[nodiscard]] int TotalCoreCount() const;

  // Comparison operators for map key support
  bool operator<(const Cluster& other) const;
  bool operator==(const Cluster& other) const;

 private:
  std::map<Core, int> core_counts_;
};

/**
 * @brief Die containing clusters
 *
 * Contains a map of cluster configurations to their counts.
 * map<Cluster, int> stores different cluster types and how many of each.
 * Immutable once constructed.
 */
class Die {
 public:
  /**
   * @brief Construct a die
   *
   * @param cluster_counts Map from cluster configuration to count
   */
  explicit Die(std::map<Cluster, int> cluster_counts);

  /**
   * @brief Convenience constructor for single cluster type
   *
   * @param cluster Single cluster configuration
   * @param count Number of clusters with this configuration
   */
  Die(const Cluster& cluster, int count);

  // Allow copy and move for container storage
  Die(const Die&) = default;
  Die& operator=(const Die&) = default;
  Die(Die&&) = default;
  Die& operator=(Die&&) = default;

  [[nodiscard]] const std::map<Cluster, int>& GetClusterCounts() const { return cluster_counts_; }
  [[nodiscard]] int TotalClusterCount() const;
  [[nodiscard]] int TotalCoreCount() const;

  // Comparison operators for map key support
  bool operator<(const Die& other) const;
  bool operator==(const Die& other) const;

 private:
  std::map<Cluster, int> cluster_counts_;
};

/**
 * @brief System on Chip (SoC)
 *
 * Top-level structure containing dies.
 * map<Die, int> stores different die types and how many of each.
 * Immutable once constructed.
 */
class SoC {
 public:
  /**
   * @brief Construct a SoC
   *
   * @param die_counts Map from die configuration to count
   */
  explicit SoC(std::map<Die, int> die_counts);

  /**
   * @brief Convenience constructor for single die type
   *
   * @param die Single die configuration
   * @param count Number of dies with this configuration
   */
  SoC(const Die& die, int count);

  // Disable copy and move to enforce immutability
  SoC(const SoC&) = delete;
  SoC& operator=(const SoC&) = delete;
  SoC(SoC&&) = delete;
  SoC& operator=(SoC&&) = delete;

  [[nodiscard]] const std::map<Die, int>& GetDieCounts() const { return die_counts_; }
  [[nodiscard]] int TotalDieCount() const;
  [[nodiscard]] int TotalClusterCount() const;
  [[nodiscard]] int TotalCoreCount() const;

 private:
  std::map<Die, int> die_counts_;
};

}  // namespace backend
}  // namespace pypto

#endif  // PYPTO_BACKEND_SOC_H_
