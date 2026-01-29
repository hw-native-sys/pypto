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

#include "pypto/ir/transforms/passes.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

namespace pypto {
namespace python {

using namespace pypto::ir;  // NOLINT(build/namespaces)

void BindPass(nb::module_& m) {
  // Create a new 'passes' submodule (using 'passes' instead of 'pass' to avoid Python keyword)
  nb::module_ passes = m.def_submodule("passes", "IR transformation passes");

  // Pass class - opaque to Python, only expose call operators
  nb::class_<Pass>(passes, "Pass", "Opaque pass object. Do not instantiate directly - use factory functions.")
      .def("__call__", &Pass::operator(), nb::arg("program"), "Execute pass on program");

  // Factory functions with snake_case names
  passes.def("identity", &pass::Identity,
             "Create an identity pass for testing\n\n"
             "Appends \"_identity\" to function names to verify pass execution.");

  passes.def("init_mem_ref", &pass::InitMemRef,
             "Create an init memref pass\n\n"
             "Initializes MemRef for all variables in functions.\n"
             "Sets memory space to UB by default, or DDR for block.load/block.store operands.");

  passes.def("basic_memory_reuse", &pass::BasicMemoryReuse,
             "Create a basic memory reuse pass\n\n"
             "Uses dependency analysis to identify memory reuse opportunities.\n"
             "Variables with non-overlapping lifetimes in the same memory space can share MemRef objects.");

  passes.def("insert_sync", &pass::InsertSync,
             "Create an insert sync pass\n\n"
             "Analyzes data dependencies and inserts synchronization operations\n"
             "(sync_src, sync_dst, bar_v, bar_m) for correct execution across hardware pipes.");

  passes.def("add_alloc", &pass::AddAlloc,
             "Create an add alloc pass\n\n"
             "This pass traverses all TileType variables in each Function and creates alloc operations\n"
             "for each unique MemRef. The alloc operations are added at the beginning of the function.\n\n"
             "The pass:\n"
             "1. Identifies all TileType variables in the function\n"
             "2. Collects all unique MemRef objects from these TileType variables\n"
             "3. Creates an alloc operation for each unique MemRef\n"
             "4. Prepends these alloc operations to the function body\n\n"
             "Each alloc operation has no input/output arguments but is bound to a MemRef pointer\n"
             "to track memory allocation for that specific buffer.");
}

}  // namespace python
}  // namespace pypto
