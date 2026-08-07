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

#include "pypto/backend/common/tcvt_path.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pypto {
namespace backend {
namespace {

using AdjList = std::unordered_map<uint8_t, std::vector<DataType>>;

AdjList BuildAdjacency(const TcvtAdjacency& table) {
  AdjList adjacency;
  for (const auto& [from, to] : table.edges) {
    if (from != to) adjacency[from.Code()].push_back(to);
  }
  return adjacency;
}

std::optional<DataType> SameWidthFloat(DataType dtype) {
  if (dtype.IsFloat()) return std::nullopt;
  if (dtype.GetBit() == 32) return DataType::FP32;
  if (dtype.GetBit() == 16) return DataType::FP16;
  return std::nullopt;
}

std::optional<std::pair<int, int>> FloatFormat(DataType dtype) {
  if (dtype == DataType::FP32) return std::make_pair(24, 8);
  if (dtype == DataType::FP16) return std::make_pair(11, 5);
  if (dtype == DataType::BF16) return std::make_pair(8, 8);
  return std::nullopt;
}

size_t IntValueBits(DataType dtype) { return dtype.IsSignedInt() ? dtype.GetBit() - 1 : dtype.GetBit(); }

bool NarrowsRelativeTo(DataType intermediate, DataType destination) {
  if (intermediate == destination) return false;
  if (intermediate.IsInt() && destination.IsFloat()) return true;
  if (intermediate.IsFloat() && destination.IsFloat()) {
    const auto intermediate_format = FloatFormat(intermediate);
    const auto destination_format = FloatFormat(destination);
    if (!intermediate_format || !destination_format) return false;
    return intermediate_format->first < destination_format->first ||
           intermediate_format->second < destination_format->second;
  }
  if (intermediate.IsFloat() && destination.IsInt()) {
    const auto format = FloatFormat(intermediate);
    return format && static_cast<size_t>(format->first) < IntValueBits(destination);
  }
  if (intermediate.IsInt() && destination.IsInt()) {
    if (intermediate.IsUnsignedInt() && destination.IsSignedInt()) return true;
    return IntValueBits(intermediate) < IntValueBits(destination);
  }
  return false;
}

int EdgePreferenceCost(DataType from, DataType to) {
  if (!from.IsFloat() && to.IsFloat() && from.GetBit() == to.GetBit()) return 0;
  if (from.IsFloat() && to.IsFloat()) return 1;
  return 2;
}

}  // namespace

bool IsNativeTcvt(const TcvtAdjacency& table, DataType from, DataType to) {
  if (from == to) return false;
  for (const auto& [source, destination] : table.edges) {
    if (source == from && destination == to) return true;
  }
  return false;
}

std::vector<DataType> FindTcvtPath(const TcvtAdjacency& table, DataType from, DataType to) {
  if (from == to) return {};
  if (IsNativeTcvt(table, from, to)) return {to};

  const AdjList adjacency = BuildAdjacency(table);
  struct NodeInfo {
    uint8_t parent = 0;
    DataType dtype = DataType::BOOL;
    int distance = -1;
    int preference = 0;
  };
  std::array<NodeInfo, 256> info{};
  std::queue<uint8_t> queue;
  info[from.Code()] = NodeInfo{from.Code(), from, 0, 0};
  queue.push(from.Code());

  while (!queue.empty()) {
    const uint8_t current = queue.front();
    queue.pop();
    const NodeInfo& current_info = info[current];
    auto found = adjacency.find(current);
    if (found == adjacency.end()) continue;

    std::vector<DataType> neighbors = found->second;
    if (auto preferred = SameWidthFloat(current_info.dtype)) {
      auto preferred_it = std::find(neighbors.begin(), neighbors.end(), *preferred);
      if (preferred_it != neighbors.end()) std::iter_swap(neighbors.begin(), preferred_it);
    }
    for (const DataType& next : neighbors) {
      if (next != to && NarrowsRelativeTo(next, to)) continue;
      const int distance = current_info.distance + 1;
      const int preference = current_info.preference + EdgePreferenceCost(current_info.dtype, next);
      NodeInfo& next_info = info[next.Code()];
      if (next_info.distance < 0) {
        next_info = NodeInfo{current, next, distance, preference};
        queue.push(next.Code());
      } else if (next_info.distance == distance && preference < next_info.preference) {
        next_info.parent = current;
        next_info.dtype = next;
        next_info.preference = preference;
      }
    }
  }

  if (info[to.Code()].distance < 0) return {};
  std::vector<DataType> reversed;
  for (uint8_t current = to.Code(); current != from.Code(); current = info[current].parent) {
    reversed.push_back(info[current].dtype);
  }
  std::reverse(reversed.begin(), reversed.end());
  return reversed;
}

}  // namespace backend
}  // namespace pypto
