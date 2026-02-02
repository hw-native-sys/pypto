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

#include "pypto/codegen/cce/cce_codegen.h"
#include "pypto/codegen/cce/type_converter.h"
#include "pypto/codegen/pto/pto_codegen.h"

namespace nb = nanobind;

namespace pypto {
namespace python {

using namespace pypto::codegen;  // NOLINT(build/namespaces)
using namespace pypto::ir;       // NOLINT(build/namespaces)

void BindCodegen(nb::module_& m) {
  // Create a new 'codegen' submodule
  nb::module_ codegen_module =
      m.def_submodule("codegen", "Code generation module for converting IR to pto-isa C++");

  // TypeConverter class for type conversions
  nb::class_<TypeConverter>(codegen_module, "TypeConverter",
                            "Utility for converting IR types to pto-isa C++ types")
      .def(nb::init<>(), "Create a type converter")
      .def("ConvertDataType", &TypeConverter::ConvertDataType, nb::arg("dtype"),
           "Convert DataType to C++ type string\n\n"
           "Args:\n"
           "    dtype: PyPTO DataType\n\n"
           "Returns:\n"
           "    C++ type string (e.g., 'float', 'half', 'int32_t')")
      .def("ConvertPipeType", &TypeConverter::ConvertPipeType, nb::arg("pipe"),
           "Convert PipeType to pto-isa pipe type string\n\n"
           "Args:\n"
           "    pipe: Pipeline type\n\n"
           "Returns:\n"
           "    C++ pipe type string with 'PIPE_' prefix (e.g., 'PIPE_MTE1', 'PIPE_V')")
      .def("ConvertEventId", &TypeConverter::ConvertEventId, nb::arg("event_id"),
           "Convert event ID to pto-isa event ID string\n\n"
           "Args:\n"
           "    event_id: Event ID (must be in range [0, 7])\n\n"
           "Returns:\n"
           "    C++ event ID string with 'EVENT_ID' prefix (e.g., 'EVENT_ID0')")
      .def("GenerateShapeType", &TypeConverter::GenerateShapeType, nb::arg("dims"),
           "Generate Shape type instantiation\n\n"
           "Args:\n"
           "    dims: Shape dimensions\n\n"
           "Returns:\n"
           "    Shape type string with 5D padding (e.g., 'Shape<1, 1, 1, 128, 64>')")
      .def("GenerateStrideType", &TypeConverter::GenerateStrideType, nb::arg("shape"),
           "Generate Stride type instantiation for row-major layout\n\n"
           "Args:\n"
           "    shape: Shape dimensions\n\n"
           "Returns:\n"
           "    Stride type string with 5D padding");

  // PTOCodegen - PTO assembly code generator
  nb::class_<PTOCodegen>(
      codegen_module, "PTOCodegen",
      "Code generator that transforms PyPTO IR to PTO assembly (.pto files). "
      "Generates PTO ISA instructions in SSA form with tile operations, control flow, and type "
      "annotations.")
      .def(nb::init<>(), "Create a new PTO assembly code generator")
      .def("generate", &PTOCodegen::Generate, nb::arg("program"),
           "Generate PTO assembly from PyPTO IR Program. Returns PTO assembly code string (.pto format) with "
           "instructions like tmul, tadd, FOR/ENDFOR, etc.");

  // CCECodegen - CCE/pto-isa C++ code generator (unified in codegen module)
  nb::class_<CCECodegen>(codegen_module, "CCECodegen",
                         "CCE code generator for converting PyPTO IR to pto-isa C++ code")
      .def(nb::init<>(), "Create a code generator")
      .def(
          "generate",
          [](CCECodegen& self, const ProgramPtr& program) {
            auto files_map = self.Generate(program);
            nb::dict result;
            for (const auto& pair : files_map) {
              result[pair.first.c_str()] = pair.second;
            }
            return result;
          },
          nb::arg("program"),
          "Generate C++ code from a PyPTO IR Program. Returns a dict mapping file paths to "
          "content. Kernel functions -> kernels/<func_name>.cpp, orchestration -> "
          "orchestration/<func_name>.cpp.");
}

}  // namespace python
}  // namespace pypto
