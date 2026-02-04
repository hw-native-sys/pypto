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

#include "pypto/backend/backend.h"

#include <algorithm>
#include <fstream>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// clang-format off
#include <msgpack.hpp>
// clang-format on

#include "pypto/core/logging.h"

namespace pypto {
namespace backend {

// Forward declaration of registry
class BackendRegistry;

// ========== Serialization Helpers ==========

namespace {

// Serialize Mem to msgpack object
msgpack::object SerializeMem(const Mem& mem, msgpack::zone& zone) {
  std::map<std::string, msgpack::object> mem_map;
  mem_map["mem_type"] = msgpack::object(static_cast<int>(mem.GetMemType()), zone);
  mem_map["mem_size"] = msgpack::object(mem.GetMemSize(), zone);
  mem_map["alignment"] = msgpack::object(mem.GetAlignment(), zone);
  return msgpack::object(mem_map, zone);
}

// Serialize Core to msgpack object
msgpack::object SerializeCore(const Core& core, msgpack::zone& zone) {
  std::map<std::string, msgpack::object> core_map;
  core_map["core_type"] = msgpack::object(static_cast<int>(core.GetCoreType()), zone);

  // Serialize mems vector
  std::vector<msgpack::object> mems_vec;
  for (const auto& mem : core.GetMems()) {
    mems_vec.emplace_back(SerializeMem(mem, zone));
  }
  core_map["mems"] = msgpack::object(mems_vec, zone);

  return msgpack::object(core_map, zone);
}

// Serialize Cluster to msgpack object
msgpack::object SerializeCluster(const Cluster& cluster, msgpack::zone& zone) {
  std::map<std::string, msgpack::object> cluster_map;

  // Serialize core_counts map as array of [core, count] pairs
  std::vector<msgpack::object> cores_vec;
  for (const auto& [core, count] : cluster.GetCoreCounts()) {
    std::map<std::string, msgpack::object> entry;
    entry["core"] = SerializeCore(core, zone);
    entry["count"] = msgpack::object(count, zone);
    cores_vec.emplace_back(msgpack::object(entry, zone));
  }
  cluster_map["cores"] = msgpack::object(cores_vec, zone);

  return msgpack::object(cluster_map, zone);
}

// Serialize Die to msgpack object
msgpack::object SerializeDie(const Die& die, msgpack::zone& zone) {
  std::map<std::string, msgpack::object> die_map;

  // Serialize cluster_counts map
  std::vector<msgpack::object> clusters_vec;
  for (const auto& [cluster, count] : die.GetClusterCounts()) {
    std::map<std::string, msgpack::object> entry;
    entry["cluster"] = SerializeCluster(cluster, zone);
    entry["count"] = msgpack::object(count, zone);
    clusters_vec.emplace_back(msgpack::object(entry, zone));
  }
  die_map["clusters"] = msgpack::object(clusters_vec, zone);

  return msgpack::object(die_map, zone);
}

// Serialize SoC to msgpack object
msgpack::object SerializeSoC(const SoC& soc, msgpack::zone& zone) {
  std::map<std::string, msgpack::object> soc_map;

  // Serialize die_counts map
  std::vector<msgpack::object> dies_vec;
  for (const auto& [die, count] : soc.GetDieCounts()) {
    std::map<std::string, msgpack::object> entry;
    entry["die"] = SerializeDie(die, zone);
    entry["count"] = msgpack::object(count, zone);
    dies_vec.emplace_back(msgpack::object(entry, zone));
  }
  soc_map["dies"] = msgpack::object(dies_vec, zone);

  return msgpack::object(soc_map, zone);
}

// Serialize Backend to bytes
std::vector<uint8_t> SerializeBackend(const Backend& backend) {
  msgpack::sbuffer buffer;
  msgpack::packer<msgpack::sbuffer> packer(buffer);

  msgpack::zone zone;

  // Create root object
  std::map<std::string, msgpack::object> root;
  root["type"] = msgpack::object(backend.GetTypeName(), zone);
  root["soc"] = SerializeSoC(*backend.GetSoC(), zone);

  // Serialize memory graph
  std::vector<msgpack::object> mem_graph_vec;
  for (const auto& [mem_space, neighbors] : backend.GetMemoryGraph()) {
    std::map<std::string, msgpack::object> edge_entry;
    edge_entry["from"] = msgpack::object(static_cast<int>(mem_space), zone);

    std::vector<msgpack::object> neighbors_vec;
    for (const auto& neighbor : neighbors) {
      neighbors_vec.emplace_back(msgpack::object(static_cast<int>(neighbor), zone));
    }
    edge_entry["to"] = msgpack::object(neighbors_vec, zone);

    mem_graph_vec.emplace_back(msgpack::object(edge_entry, zone));
  }
  root["mem_graph"] = msgpack::object(mem_graph_vec, zone);

  packer.pack(msgpack::object(root, zone));

  return std::vector<uint8_t>(buffer.data(), buffer.data() + buffer.size());
}

// ========== Deserialization Helpers ==========

// Deserialize Mem from msgpack object
Mem DeserializeMem(const msgpack::object& obj) {
  auto mem_map = obj.as<std::map<std::string, msgpack::object>>();

  auto mem_type = static_cast<ir::MemorySpace>(mem_map.at("mem_type").as<int>());
  auto mem_size = mem_map.at("mem_size").as<uint64_t>();
  auto alignment = mem_map.at("alignment").as<uint64_t>();

  return Mem(mem_type, mem_size, alignment);
}

// Deserialize Core from msgpack object
Core DeserializeCore(const msgpack::object& obj) {
  auto core_map = obj.as<std::map<std::string, msgpack::object>>();

  auto core_type = static_cast<ir::CoreType>(core_map.at("core_type").as<int>());

  std::vector<Mem> mems;
  auto mems_vec = core_map.at("mems").as<std::vector<msgpack::object>>();
  for (const auto& mem_obj : mems_vec) {
    mems.push_back(DeserializeMem(mem_obj));
  }

  return Core(core_type, std::move(mems));
}

// Deserialize Cluster from msgpack object
std::shared_ptr<const Cluster> DeserializeCluster(const msgpack::object& obj) {
  auto cluster_map = obj.as<std::map<std::string, msgpack::object>>();

  std::map<Core, int> core_counts;
  auto cores_vec = cluster_map.at("cores").as<std::vector<msgpack::object>>();
  for (const auto& entry_obj : cores_vec) {
    auto entry = entry_obj.as<std::map<std::string, msgpack::object>>();
    auto core = DeserializeCore(entry.at("core"));
    auto count = entry.at("count").as<int>();
    core_counts[core] = count;
  }

  return std::make_shared<Cluster>(std::move(core_counts));
}

// Deserialize Die from msgpack object
std::shared_ptr<const Die> DeserializeDie(const msgpack::object& obj) {
  auto die_map = obj.as<std::map<std::string, msgpack::object>>();

  std::map<Cluster, int> cluster_counts;
  auto clusters_vec = die_map.at("clusters").as<std::vector<msgpack::object>>();
  for (const auto& entry_obj : clusters_vec) {
    auto entry = entry_obj.as<std::map<std::string, msgpack::object>>();
    auto cluster = DeserializeCluster(entry.at("cluster"));
    auto count = entry.at("count").as<int>();
    cluster_counts[*cluster] = count;
  }

  return std::make_shared<Die>(std::move(cluster_counts));
}

// Deserialize SoC from msgpack object
std::shared_ptr<const SoC> DeserializeSoC(const msgpack::object& obj) {
  auto soc_map = obj.as<std::map<std::string, msgpack::object>>();

  std::map<Die, int> die_counts;
  auto dies_vec = soc_map.at("dies").as<std::vector<msgpack::object>>();
  for (const auto& entry_obj : dies_vec) {
    auto entry = entry_obj.as<std::map<std::string, msgpack::object>>();
    auto die = DeserializeDie(entry.at("die"));
    auto count = entry.at("count").as<int>();
    die_counts[*die] = count;
  }

  return std::make_shared<SoC>(std::move(die_counts));
}

}  // namespace

// ========== Backend Implementation ==========

void Backend::ExportToFile(const std::string& path) const {
  auto data = SerializeBackend(*this);
  std::ofstream file(path, std::ios::binary);
  CHECK(file) << "Failed to open file for writing: " + path;
  file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
  CHECK(file) << "Failed to write to file: " + path;
}

std::unique_ptr<Backend> Backend::ImportFromFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  CHECK(file.is_open()) << "Failed to open file for reading: " + path;

  std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  CHECK(!file.fail()) << "Failed to read from file: " + path;

  // Unpack msgpack data
  msgpack::object_handle oh = msgpack::unpack(reinterpret_cast<const char*>(data.data()), data.size());
  msgpack::object obj = oh.get();

  auto root = obj.as<std::map<std::string, msgpack::object>>();
  auto type_name = root.at("type").as<std::string>();
  auto soc = DeserializeSoC(root.at("soc"));

  // Deserialize memory graph
  std::map<ir::MemorySpace, std::vector<ir::MemorySpace>> mem_graph;
  auto mem_graph_vec = root.at("mem_graph").as<std::vector<msgpack::object>>();
  for (const auto& edge_obj : mem_graph_vec) {
    auto edge_entry = edge_obj.as<std::map<std::string, msgpack::object>>();
    auto from = static_cast<ir::MemorySpace>(edge_entry.at("from").as<int>());

    std::vector<ir::MemorySpace> neighbors;
    auto neighbors_vec = edge_entry.at("to").as<std::vector<msgpack::object>>();
    for (const auto& neighbor_obj : neighbors_vec) {
      neighbors.push_back(static_cast<ir::MemorySpace>(neighbor_obj.as<int>()));
    }

    mem_graph[from] = neighbors;
  }

  // Use registry to create appropriate backend type
  extern std::unique_ptr<Backend> CreateBackendFromRegistry(
      const std::string& type_name, std::shared_ptr<const SoC> soc,
      const std::map<ir::MemorySpace, std::vector<ir::MemorySpace>>& mem_graph);
  return CreateBackendFromRegistry(type_name, soc, mem_graph);
}

std::vector<ir::MemorySpace> Backend::FindMemPath(ir::MemorySpace from, ir::MemorySpace to) const {
  if (from == to) {
    return {from};
  }

  // BFS to find shortest path
  std::queue<ir::MemorySpace> queue;
  std::unordered_map<ir::MemorySpace, ir::MemorySpace> parent;
  std::set<ir::MemorySpace> visited;

  queue.push(from);
  visited.insert(from);
  parent[from] = from;  // Mark root

  bool found = false;
  while (!queue.empty() && !found) {
    auto current = queue.front();
    queue.pop();

    auto it = mem_graph_.find(current);
    if (it == mem_graph_.end()) {
      continue;
    }

    for (auto neighbor : it->second) {
      if (visited.find(neighbor) == visited.end()) {
        visited.insert(neighbor);
        parent[neighbor] = current;
        queue.push(neighbor);

        if (neighbor == to) {
          found = true;
          break;
        }
      }
    }
  }

  CHECK(found) << "No path found from " << static_cast<int>(from) << " to " << static_cast<int>(to);

  // Reconstruct path
  std::vector<ir::MemorySpace> path;
  ir::MemorySpace current = to;
  while (current != from) {
    path.push_back(current);
    current = parent[current];
  }
  path.push_back(from);
  std::reverse(path.begin(), path.end());

  return path;
}

uint64_t Backend::GetMemSize(ir::MemorySpace mem_type) const {
  // Find the first memory of the requested type and return its size
  for (const auto& [die, die_count] : soc_->GetDieCounts()) {
    for (const auto& [cluster, cluster_count] : die.GetClusterCounts()) {
      for (const auto& [core, core_count] : cluster.GetCoreCounts()) {
        for (const auto& mem : core.GetMems()) {
          if (mem.GetMemType() == mem_type) {
            return mem.GetMemSize();
          }
        }
      }
    }
  }

  // Memory type not found in SoC
  return 0;
}

}  // namespace backend
}  // namespace pypto
