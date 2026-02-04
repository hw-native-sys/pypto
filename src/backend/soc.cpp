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

#include "pypto/backend/soc.h"

#include <map>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

namespace pypto {
namespace backend {

// ========== Mem Implementation ==========

Mem::Mem(ir::MemorySpace mem_type, uint64_t mem_size, uint64_t alignment)
    : mem_type_(mem_type), mem_size_(mem_size), alignment_(alignment) {}

bool Mem::operator<(const Mem& other) const {
  return std::tie(mem_type_, mem_size_, alignment_) <
         std::tie(other.mem_type_, other.mem_size_, other.alignment_);
}

bool Mem::operator==(const Mem& other) const {
  return mem_type_ == other.mem_type_ && mem_size_ == other.mem_size_ && alignment_ == other.alignment_;
}

// ========== Core Implementation ==========

Core::Core(ir::CoreType core_type, std::vector<Mem> mems) : core_type_(core_type), mems_(std::move(mems)) {}

bool Core::operator<(const Core& other) const {
  if (core_type_ != other.core_type_) {
    return core_type_ < other.core_type_;
  }
  return mems_ < other.mems_;
}

bool Core::operator==(const Core& other) const {
  return core_type_ == other.core_type_ && mems_ == other.mems_;
}

// ========== Cluster Implementation ==========

Cluster::Cluster(std::map<Core, int> core_counts) : core_counts_(std::move(core_counts)) {}

Cluster::Cluster(const Core& core, int count) : core_counts_({{core, count}}) {}

int Cluster::TotalCoreCount() const {
  return std::accumulate(core_counts_.begin(), core_counts_.end(), 0,
                         [](int sum, const auto& pair) { return sum + pair.second; });
}

bool Cluster::operator<(const Cluster& other) const { return core_counts_ < other.core_counts_; }

bool Cluster::operator==(const Cluster& other) const { return core_counts_ == other.core_counts_; }

// ========== Die Implementation ==========

Die::Die(std::map<Cluster, int> cluster_counts) : cluster_counts_(std::move(cluster_counts)) {}

Die::Die(const Cluster& cluster, int count) : cluster_counts_({{cluster, count}}) {}

int Die::TotalClusterCount() const {
  return std::accumulate(cluster_counts_.begin(), cluster_counts_.end(), 0,
                         [](int sum, const auto& pair) { return sum + pair.second; });
}

int Die::TotalCoreCount() const {
  return std::accumulate(cluster_counts_.begin(), cluster_counts_.end(), 0, [](int sum, const auto& pair) {
    return sum + pair.first.TotalCoreCount() * pair.second;
  });
}

bool Die::operator<(const Die& other) const { return cluster_counts_ < other.cluster_counts_; }

bool Die::operator==(const Die& other) const { return cluster_counts_ == other.cluster_counts_; }

// ========== SoC Implementation ==========

SoC::SoC(std::map<Die, int> die_counts) : die_counts_(std::move(die_counts)) {}

SoC::SoC(const Die& die, int count) : die_counts_({{die, count}}) {}

int SoC::TotalDieCount() const {
  return std::accumulate(die_counts_.begin(), die_counts_.end(), 0,
                         [](int sum, const auto& pair) { return sum + pair.second; });
}

int SoC::TotalClusterCount() const {
  return std::accumulate(die_counts_.begin(), die_counts_.end(), 0, [](int sum, const auto& pair) {
    return sum + pair.first.TotalClusterCount() * pair.second;
  });
}

int SoC::TotalCoreCount() const {
  return std::accumulate(die_counts_.begin(), die_counts_.end(), 0, [](int sum, const auto& pair) {
    return sum + pair.first.TotalCoreCount() * pair.second;
  });
}

}  // namespace backend
}  // namespace pypto
