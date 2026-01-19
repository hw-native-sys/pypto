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

#include "pypto/ir/transform/base/pass.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

#include "pypto/ir/transform/passes/identity_pass.h"

namespace nb = nanobind;

namespace pypto {
namespace python {

using namespace pypto::ir;  // NOLINT(build/namespaces)

void BindPass(nb::module_& m) {
  // Create a new 'passes' submodule (using 'passes' instead of 'pass' to avoid Python keyword)
  nb::module_ passes = m.def_submodule("passes", "IR transformation passes");

  // Pass base class for IR transformations
  nb::class_<Pass>(passes, "Pass", "Base class for IR transformation passes")
      .def("run", &Pass::Run, nb::arg("func"), "Execute the pass on a function");

  // IdentityPass - a pass that appends a suffix to function name
  nb::class_<IdentityPass, Pass>(passes, "IdentityPass",
                                 "A pass that appends '_identity' suffix to function name for testing")
      .def(nb::init<>(), "Create an identity pass");
}

}  // namespace python
}  // namespace pypto
