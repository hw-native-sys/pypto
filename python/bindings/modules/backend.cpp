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

#include "pypto/backend/common/backend.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "../module.h"
#include "pypto/backend/910B/backend_910b.h"
#include "pypto/backend/950/backend_950.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/backend/common/soc.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/pipe.h"

namespace nb = nanobind;

namespace pypto {
namespace python {

using pypto::backend::Backend;
using pypto::backend::Backend910B;
using pypto::backend::Backend950;
using pypto::backend::BackendHandler;
using pypto::backend::BackendType;
using pypto::backend::Cluster;
using pypto::backend::Core;
using pypto::backend::Die;
using pypto::backend::Mem;
using pypto::backend::SoC;
using pypto::ir::CoreType;
using pypto::ir::MemorySpace;

void BindBackend(nb::module_& m) {
  nb::module_ backend_mod = m.def_submodule("backend", "PyPTO Backend module");

  // ========== BackendType enum ==========
  nb::enum_<BackendType>(backend_mod, "BackendType",
                         "Backend type for passes and codegen (use Instance internally)")
      .value("Ascend910B", BackendType::Ascend910B, "910B backend (PTO assembly codegen)")
      .value("Ascend950", BackendType::Ascend950, "950 PTO backend");

  // ========== Mem class ==========
  nb::class_<Mem>(backend_mod, "Mem", "Memory component")
      .def(nb::init<MemorySpace, uint64_t, uint64_t>(), nb::arg("mem_type"), nb::arg("mem_size"),
           nb::arg("alignment"), "Create a memory component")
      .def_prop_ro("mem_type", &Mem::GetMemType, "Memory space type")
      .def_prop_ro("mem_size", &Mem::GetMemSize, "Memory size in bytes")
      .def_prop_ro("alignment", &Mem::GetAlignment, "Memory alignment in bytes")
      .def("__repr__", [](const Mem& mem) {
        return "Mem(type=" + ir::MemorySpaceToString(mem.GetMemType()) +
               ", size=" + std::to_string(mem.GetMemSize()) +
               ", alignment=" + std::to_string(mem.GetAlignment()) + ")";
      });

  // ========== Core class ==========
  nb::class_<Core>(backend_mod, "Core", "Processing core")
      .def(nb::init<CoreType, std::vector<Mem>>(), nb::arg("core_type"), nb::arg("mems"),
           "Create a processing core")
      .def_prop_ro("core_type", &Core::GetCoreType, "Core type (CUBE or VECTOR)")
      .def_prop_ro("mems", &Core::GetMems, "List of memory components")
      .def("__repr__", [](const Core& core) {
        return "Core(type=" + std::to_string(static_cast<int>(core.GetCoreType())) +
               ", mems=" + std::to_string(core.GetMems().size()) + ")";
      });

  // ========== Cluster class ==========
  nb::class_<Cluster>(backend_mod, "Cluster", "Cluster of processing cores")
      .def(nb::init<std::map<Core, int>>(), nb::arg("core_counts"), "Create cluster from core counts map")
      .def(nb::init<const Core&, int>(), nb::arg("core"), nb::arg("count"),
           "Create cluster with single core type")
      .def_prop_ro("core_counts", &Cluster::GetCoreCounts, "Map of core configurations to counts")
      .def("total_core_count", &Cluster::TotalCoreCount, "Get total number of cores in cluster")
      .def("__repr__", [](const Cluster& cluster) {
        return "Cluster(total_cores=" + std::to_string(cluster.TotalCoreCount()) + ")";
      });

  // ========== Die class ==========
  nb::class_<Die>(backend_mod, "Die", "Die containing clusters")
      .def(nb::init<std::map<Cluster, int>>(), nb::arg("cluster_counts"),
           "Create die from cluster counts map")
      .def(nb::init<const Cluster&, int>(), nb::arg("cluster"), nb::arg("count"),
           "Create die with single cluster type")
      .def_prop_ro("cluster_counts", &Die::GetClusterCounts, "Map of cluster configurations to counts")
      .def("total_cluster_count", &Die::TotalClusterCount, "Get total number of clusters in die")
      .def("total_core_count", &Die::TotalCoreCount, "Get total number of cores in die")
      .def("__repr__", [](const Die& die) {
        return "Die(clusters=" + std::to_string(die.TotalClusterCount()) +
               ", cores=" + std::to_string(die.TotalCoreCount()) + ")";
      });

  // ========== SoC class ==========
  nb::class_<SoC>(backend_mod, "SoC", "System on Chip")
      .def(nb::init<std::map<Die, int>>(), nb::arg("die_counts"), "Create SoC from die counts map")
      .def(nb::init<const Die&, int>(), nb::arg("die"), nb::arg("count"), "Create SoC with single die type")
      .def_prop_ro("die_counts", &SoC::GetDieCounts, "Map of die configurations to counts")
      .def("total_die_count", &SoC::TotalDieCount, "Get total number of dies in SoC")
      .def("total_cluster_count", &SoC::TotalClusterCount, "Get total number of clusters in SoC")
      .def("total_core_count", &SoC::TotalCoreCount, "Get total number of cores in SoC")
      .def("__repr__", [](const SoC& soc) {
        return "SoC(dies=" + std::to_string(soc.TotalDieCount()) +
               ", clusters=" + std::to_string(soc.TotalClusterCount()) +
               ", cores=" + std::to_string(soc.TotalCoreCount()) + ")";
      });

  // ========== BackendHandler ==========
  // Per-backend behaviour dispatch. Passes / codegen / Python all consume this
  // interface rather than branching on BackendType.
  nb::class_<BackendHandler>(backend_mod, "BackendHandler", "Per-backend behaviour dispatch interface")
      .def("get_pto_target_arch", &BackendHandler::GetPtoTargetArch,
           "PTO MLIR target arch attribute string (e.g. 'a2a3', 'a5')")
      .def("get_launch_spec_core_count_method", &BackendHandler::GetLaunchSpecCoreCountMethod,
           "Method name on launch_spec for setting core count "
           "('set_block_num' on Ascend910B, 'set_core_num' on Ascend950)")
      .def("get_default_sim_platform", &BackendHandler::GetDefaultSimPlatform,
           "Default simulator platform name (e.g. 'a2a3sim', 'a5sim')")
      .def("get_extra_ptoas_flags", &BackendHandler::GetExtraPtoasFlags,
           "Extra flags appended to ptoas invocation for this backend")
      .def("requires_gm_pipe_buffer", &BackendHandler::RequiresGMPipeBuffer,
           "Whether ExpandMixedKernel must inject the GM-backed pipe slot buffer")
      .def("requires_split_load_tpop_workaround", &BackendHandler::RequiresSplitLoadTpopWorkaround,
           "Whether LegalizePtoBufferReuse must apply the split-load tpop hazard workaround")
      .def("requires_vto_c_fractal_adapt", &BackendHandler::RequiresVtoCFractalAdapt,
           "Whether AIV-side V-to-C tpush must materialise a fractal-layout adapter move")
      .def("requires_runtime_subblock_bridge", &BackendHandler::RequiresRuntimeSubblockBridge,
           "Whether split AIV wrappers must source the subblock id from the runtime context")
      .def("requires_no_split_dual_aiv_dispatch", &BackendHandler::RequiresNoSplitDualAivDispatch,
           "Whether mixed kernels with no split mode must still be dispatched on both AIV lanes")
      .def("get_gm_access_granularity_bytes", &BackendHandler::GetGmAccessGranularityBytes,
           "GM access granularity in bytes (issue #1180 perf-hint threshold input)")
      .def("get_l2_cache_line_bytes", &BackendHandler::GetL2CacheLineBytes, "L2 cache line size in bytes")
      .def("get_recommended_innermost_dim_bytes", &BackendHandler::GetRecommendedInnermostDimBytes,
           "Recommended minimum innermost-dim size, in bytes, for tile ops touching GM")
      .def("get_l0a_capacity_bytes", &BackendHandler::GetL0aCapacityBytes,
           "L0a (Left) on-chip SRAM capacity in bytes")
      .def("get_l0b_capacity_bytes", &BackendHandler::GetL0bCapacityBytes,
           "L0b (Right) on-chip SRAM capacity in bytes")
      .def("get_l0c_capacity_bytes", &BackendHandler::GetL0cCapacityBytes,
           "L0c (Acc) on-chip SRAM capacity in bytes")
      .def("get_l0_fractal_alignment", &BackendHandler::GetL0FractalAlignment,
           "Cube fractal alignment in elements for L0 tile dimensions m, n, k")
      .def("get_min_l0_tile_dim", &BackendHandler::GetMinL0TileDim,
           "Minimum legal value for L0 tile dimensions m, n, k");

  // ========== Backend abstract base class ==========
  nb::class_<Backend>(backend_mod, "Backend", "Abstract backend base class")
      .def("get_type_name", &Backend::GetTypeName, "Get backend type name")
      .def("get_handler", &Backend::GetHandler, nb::rv_policy::reference,
           "Get the per-backend BackendHandler singleton")
      .def("export_to_file", &Backend::ExportToFile, nb::arg("path"), "Export backend to msgpack file")
      .def_static(
          "import_from_file",
          [](const std::string& path) -> Backend* {
            // Return raw pointer that nanobind will manage
            return Backend::ImportFromFile(path).release();
          },
          nb::arg("path"), nb::rv_policy::take_ownership, "Import backend from msgpack file")
      .def("find_mem_path", &Backend::FindMemPath, nb::arg("from_mem"), nb::arg("to_mem"),
           "Find memory path from source to destination")
      .def("get_mem_size", &Backend::GetMemSize, nb::arg("mem_type"),
           "Get total memory size for given memory type")
      .def_prop_ro(
          "soc", [](const Backend& backend) -> const SoC& { return backend.GetSoC(); }, "Get SoC object");

  // ========== Backend910B concrete implementation ==========
  nb::class_<Backend910B, Backend>(backend_mod, "Backend910B", "910B backend implementation")
      .def_static("instance", &Backend910B::Instance, nb::rv_policy::reference,
                  "Get singleton instance of 910B backend");

  // ========== Backend950 concrete implementation ==========
  nb::class_<Backend950, Backend>(backend_mod, "Backend950", "950 PTO backend implementation")
      .def_static("instance", &Backend950::Instance, nb::rv_policy::reference,
                  "Get singleton instance of 950 backend");

  // ========== Backend configuration functions ==========
  backend_mod.def("set_backend_type", &backend::BackendConfig::SetBackendType, nb::arg("backend_type"),
                  "Set the global backend type. Must be called before any backend operations. "
                  "Can be called multiple times with the same type (idempotent).");

  backend_mod.def("get_backend_type", &backend::BackendConfig::GetBackendType,
                  "Get the configured backend type. Throws error if not configured.");

  backend_mod.def(
      "get_backend_instance", [](BackendType type) { return backend::GetBackendInstance(type); },
      nb::arg("backend_type"), nb::rv_policy::reference,
      "Get the singleton Backend instance for a specific BackendType. "
      "Useful when the caller knows the desired backend independently of global config.");

  backend_mod.def(
      "get_handler", []() { return backend::BackendConfig::GetBackend()->GetHandler(); },
      nb::rv_policy::reference,
      "Get the BackendHandler for the currently configured backend. "
      "Throws if backend type has not been configured.");

  backend_mod.def("is_backend_configured", &backend::BackendConfig::IsConfigured,
                  "Check if backend type has been configured.");

  backend_mod.def("reset_for_testing", &backend::BackendConfig::ResetForTesting,
                  "Reset backend configuration (for testing only). "
                  "WARNING: Only use in tests to reset between test cases.");
}

}  // namespace python
}  // namespace pypto
