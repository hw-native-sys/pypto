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

#include "pypto/hierarchy/level_handler.h"

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "pypto/ir/function.h"

namespace pypto {
namespace hierarchy {
namespace {

class HostLevelHandler final : public LevelHandler {
 public:
  std::string GetDomainAllocator() const override { return "(_domain_provider or orch.allocate_domain)"; }

  std::string FormatDomainMembership(const std::vector<int64_t>& children) const override {
    std::ostringstream workers;
    workers << "[";
    if (children.empty()) {
      workers << "*range(world_size)";
    } else {
      for (size_t i = 0; i < children.size(); ++i) {
        if (i > 0) workers << ", ";
        workers << children[i];
      }
    }
    workers << "]";
    return workers.str();
  }

  std::string FormatCommCtxValue(const std::string& table, const std::string& child) const override {
    return table + "[" + child + "].device_ctx";
  }

  std::string FormatHostTensorAppend(const HostArgSpec& spec) const override {
    return "add_chip_tensor_arg(" + spec.task_args + ", " + spec.worker_expr + ", " + spec.tensor_expr +
           ", " + spec.tag + ", callables, \"" + spec.callee_name + "\", " +
           std::to_string(spec.tensor_index) + ")";
  }

  bool MaterializesWindowTensor() const override { return true; }

  std::vector<std::string> GetEntryKeywordParams(bool accepts_domain_provider) const override {
    // Keep the runner ABI and its ordering, including the persistent-domain hook.
    std::vector<std::string> params = {"tensors", "callables", "sub_ids", "_keep", "world_size"};
    if (accepts_domain_provider) params.push_back("_domain_provider=None");
    return params;
  }

  std::vector<std::string> GetImports() const override {
    // DataType builds address-free window tensors; CommBufferSpec describes
    // domain allocations. Import both even for comm-less programs, as before.
    // _submit_chip also namespaces per-dispatch DFX output_prefix values.
    return {"import torch",
            "from simpler.task_interface import "
            "CallConfig, CommBufferSpec, DataType, TaskArgs, TensorArgType",
            "from pypto.runtime.tensor_arg import make_tensor_arg",
            "from pypto.runtime.distributed_runner import add_chip_tensor_arg, _submit_chip"};
  }
};

class ClusterLevelHandler final : public LevelHandler {
 public:
  std::string GetDomainAllocator() const override {
    return "(_domain_provider or orch.allocate_global_domain)";
  }

  std::string FormatDomainMembership(const std::vector<int64_t>& children) const override {
    // IR children are logical node indexes. Runtime global membership instead
    // names each actual L3 worker and its local L2 children, in topology order.
    std::ostringstream members;
    members << "[(nodes[n], d) for n in ";
    if (children.empty()) {
      members << "range(num_workers)";
    } else {
      members << "(";
      for (const auto child : children) members << child << ", ";
      members << ")";
    }
    members << " for d in range(node_sizes[n])]";
    return members.str();
  }

  std::string FormatCommCtxValue(const std::string& table, const std::string&) const override {
    // Only the domain identity crosses the node boundary. Device contexts are
    // resolved by the receiving HOST entry on its own local Worker.
    return table + ".domain_id";
  }

  std::string FormatHostTensorAppend(const HostArgSpec& spec) const override {
    return spec.task_args + ".add_tensor(_io_arg(node_bindings[" + spec.worker_expr + "], " +
           spec.tensor_expr + ", " + spec.tag + "), " + spec.tag + ")";
  }

  bool MaterializesWindowTensor() const override { return false; }

  std::vector<std::string> GetEntryKeywordParams(bool accepts_domain_provider) const override {
    std::vector<std::string> params = {"tensors",       "callables", "nodes",  "node_sizes",
                                       "node_bindings", "_keep",     "_retain"};
    if (accepts_domain_provider) params.push_back("_domain_provider=None");
    return params;
  }

  std::vector<std::string> GetImports() const override {
    // _io_arg is supplied by cluster runner integration. Do not import the
    // HOST CHIP helpers or require a runner module before that layer exists.
    return {"import torch",
            "from simpler.task_interface import "
            "CallConfig, CommBufferSpec, DataType, TaskArgs, TensorArgType"};
  }
};

}  // namespace

const LevelHandler* GetLevelHandler(ir::Level level) {
  if (level == ir::Level::HOST) {
    static const HostLevelHandler handler;
    return &handler;
  }
  if (level == ir::Level::CLUSTER_0) {
    static const ClusterLevelHandler handler;
    return &handler;
  }
  return nullptr;
}

}  // namespace hierarchy
}  // namespace pypto
