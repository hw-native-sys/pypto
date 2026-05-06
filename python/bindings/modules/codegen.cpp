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

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "pypto/backend/common/backend.h"
#include "pypto/codegen/distributed/distributed_codegen.h"
#include "pypto/codegen/orchestration/orchestration_codegen.h"
#include "pypto/codegen/pto/pto_codegen.h"

namespace nb = nanobind;

namespace pypto {
namespace python {

using namespace pypto::backend;  // NOLINT(build/namespaces)
using namespace pypto::codegen;  // NOLINT(build/namespaces)
using namespace pypto::ir;       // NOLINT(build/namespaces)

void BindCodegen(nb::module_& m) {
  // Create a new 'codegen' submodule
  nb::module_ codegen_module =
      m.def_submodule("codegen", "Code generation module for converting IR to pto-isa C++");

  // PTOCodegen - PTO assembly code generator
  nb::class_<PTOCodegen>(
      codegen_module, "PTOCodegen",
      "Code generator that transforms PyPTO IR to PTO assembly (.pto files). "
      "Generates PTO ISA instructions in SSA form with tile operations, control flow, and type "
      "annotations.")
      .def(nb::init<>(), "Create a PTO code generator (backend is always PTO)")
      .def("generate", &PTOCodegen::Generate, nb::arg("program"),
           "Generate PTO assembly from PyPTO IR Program. Returns PTO assembly code string (.pto format) with "
           "instructions like tmul, tadd, FOR/ENDFOR, etc.");

  // OrchestrationResult - result of orchestration code generation
  nb::class_<OrchestrationResult>(codegen_module, "OrchestrationResult",
                                  "Result of orchestration code generation")
      .def_ro("code", &OrchestrationResult::code, "Generated C++ orchestration code")
      .def_ro("func_name_to_id", &OrchestrationResult::func_name_to_id,
              "Kernel function name to func_id mapping")
      .def_ro("func_name_to_core_type", &OrchestrationResult::func_name_to_core_type,
              "Kernel function name to core type mapping");

  // Free functions for orchestration codegen (backend-agnostic)
  codegen_module.def("generate_orchestration", &GenerateOrchestration, nb::arg("program"), nb::arg("func"),
                     "Generate C++ orchestration code for a function.\n\n"
                     "Uses PTO2 runtime API (rt_submit_task, make_tensor_external, etc.).\n"
                     "This is backend-agnostic.\n\n"
                     "Args:\n"
                     "    program: The IR Program containing all functions\n"
                     "    func: The orchestration function to generate code for\n\n"
                     "Returns:\n"
                     "    OrchestrationResult with generated code and function metadata");

  // DistributedCodegen - Distributed C++ code generator for Linqu runtime
  nb::class_<DistributedCodegen>(codegen_module, "DistributedCodegen",
                                 "Distributed codegen for Linqu hierarchy runtime C++ code")
      .def(nb::init<>(), "Create a distributed code generator")
      .def("generate", &DistributedCodegen::Generate, nb::arg("program"),
           "Generate distributed C++ code from IR Program.\n\n"
           "Args:\n"
           "    program: The IR Program (after OutlineHierarchyScopes)\n\n"
           "Returns:\n"
           "    Complete C++ source code as a string");

  codegen_module.def("infer_function_core_type", &InferFunctionCoreType, nb::arg("func"),
                     "Infer the core type (CUBE or VECTOR) of a function from its operations.\n\n"
                     "Args:\n"
                     "    func: The function to infer core type for\n\n"
                     "Returns:\n"
                     "    CoreType.CUBE or CoreType.VECTOR");
}

}  // namespace python
}  // namespace pypto
