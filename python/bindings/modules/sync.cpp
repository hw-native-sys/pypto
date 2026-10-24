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

/**
 * @file sync.cpp
 * @brief Expose typed hardware resource domains and immutable built-in catalogs.
 *
 * Queries require neither an installed ISA nor global backend selection. Python
 * can construct lookup keys, but cannot register catalogs or modify target facts.
 * Pool identity and capacity do not certify an instruction's backend legality.
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <new>

#include "../module.h"
#include "pypto/ir/pipe.h"
#include "pypto/sync/resource_catalog.h"

namespace nb = nanobind;

namespace pypto::python {

/// Register hardware queries after IR pipe enums; built-in catalogs remain native-owned for process lifetime.
void BindSync(nb::module_& m) {
  namespace s = pypto::sync;
  auto mod = m.def_submodule("sync", "Hardware synchronization resource catalogs");

  // Separate resource families even when their numeric IDs or user-facing names coincide.
  nb::enum_<s::ResourceKind>(mod, "ResourceKind")
      .value("PipelineEvent", s::ResourceKind::PipelineEvent)
      .value("IntraBlockSignal", s::ResourceKind::IntraBlockSignal)
      .value("BufferCredit", s::ResourceKind::BufferCredit);
  nb::enum_<s::CoreScope>(mod, "CoreScope")
      .value("AIC", s::CoreScope::AIC)
      .value("AIV", s::CoreScope::AIV)
      .value("CoreGroup", s::CoreScope::CoreGroup);

  // Keys are Python-owned values. Native query validation decides whether their domain is modeled.
  nb::class_<s::PoolKey>(mod, "PoolKey")
      .def(
          "__init__",
          [](s::PoolKey* self, s::ResourceKind kind, s::CoreScope core, ir::PipeType src, ir::PipeType dst,
             std::uint32_t replica) { new (self) s::PoolKey{kind, core, src, dst, replica}; },
          nb::arg("kind") = s::ResourceKind::PipelineEvent, nb::arg("core") = s::CoreScope::AIV,
          nb::arg("src") = ir::PipeType::ALL, nb::arg("dst") = ir::PipeType::ALL, nb::arg("replica") = 0)
      .def_rw("kind", &s::PoolKey::kind)
      .def_rw("core", &s::PoolKey::core)
      .def_rw("src", &s::PoolKey::src)
      .def_rw("dst", &s::PoolKey::dst)
      .def_rw("replica", &s::PoolKey::replica)
      .def("__eq__", &s::PoolKey::operator==, nb::is_operator());

  // Return the writable key by value, preventing a nested setter from changing a read-only row.
  // STL ID lists are snapshots as well; mutating a returned list never changes catalog storage.
  nb::class_<s::PoolSpec>(mod, "PoolSpec")
      .def_prop_ro("key", [](const s::PoolSpec& spec) { return spec.key; })
      .def_ro("capacity", &s::PoolSpec::capacity)
      .def_ro("legal_ids", &s::PoolSpec::legal_ids)
      .def_ro("reserved_ids", &s::PoolSpec::reserved_ids)
      .def_ro("requires_contiguous_ids", &s::PoolSpec::requires_contiguous_ids);
  // No Python constructor or registration method can bypass native catalog validation.
  nb::class_<s::ResourceCatalog>(mod, "ResourceCatalog")
      .def_prop_ro("target", &s::ResourceCatalog::GetTarget)
      .def("find_pool", &s::ResourceCatalog::FindPool, nb::arg("key"),
           "Return an owned pool snapshot, or None for an invalid or unmodeled domain");

  // Backends own these immutable objects. Reference policy prevents Python from deleting them.
  mod.def("find_resource_catalog", &s::FindResourceCatalog, nb::arg("target"), nb::rv_policy::reference,
          "Borrow a backend's process-lifetime catalog, or return None for an unknown target");
}

}  // namespace pypto::python
