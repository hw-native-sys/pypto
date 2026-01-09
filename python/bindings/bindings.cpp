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
 * @file bindings.cpp
 * @brief Main Python module definition
 *
 * This file serves as the entry point for the PyPTO Python module.
 * It imports and registers all sub-module bindings (errors, tensors, ops, etc.)
 * to create the complete Python API.
 */

#include <nanobind/nanobind.h>

#include "./module.h"
#include "pypto/core/common.h"

namespace nb = nanobind;

NB_MODULE(pypto_core, m) {
  m.doc() = PYPTO_NANOBIND_MODULE_DOC;

  // Register error handling bindings
  pypto::python::BindErrors(m);

  // Register core types (DataType enum and utilities)
  pypto::python::BindCore(m);

  // Register testing utilities (exposed as pypto.testing)
  pypto::python::BindTesting(m);

  // Register IR (Intermediate Representation) bindings
  pypto::python::BindIR(m);

  // Register logging framework bindings
  pypto::python::BindLogging(m);

  // Future bindings can be added here:
  // pypto::python::BindTensors(m);
  // pypto::python::BindOps(m);
  // pypto::python::BindDevices(m);
}
