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
 * @file testing.cpp
 * @brief Implementation of Python bindings for testing utilities
 *
 * This module provides internal testing utilities that should not be used
 * in production code. It is exposed as pypto.testing in Python.
 */

#include <pybind11/pybind11.h>

#include <cassert>
#include <string>

#include "../bindings.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"

namespace py = pybind11;

namespace pypto {
namespace python {

// ============================================================================
// Helper functions to demonstrate error raising from C++
// ============================================================================

/**
 * @brief Raise a ValueError from C++ for testing purposes
 * @param message Error message to include in the exception
 */
[[noreturn]] void raise_value_error(const std::string& message) { throw pypto::ValueError(message); }

/**
 * @brief Raise a TypeError from C++ for testing purposes
 * @param message Error message to include in the exception
 */
[[noreturn]] void raise_type_error(const std::string& message) { throw pypto::TypeError(message); }

/**
 * @brief Raise a RuntimeError from C++ for testing purposes
 * @param message Error message to include in the exception
 */
[[noreturn]] void raise_runtime_error(const std::string& message) { throw pypto::RuntimeError(message); }

/**
 * @brief Raise a NotImplementedError from C++ for testing purposes
 * @param message Error message to include in the exception
 */
[[noreturn]] void raise_not_implemented_error(const std::string& message) {
  throw pypto::NotImplementedError(message);
}

/**
 * @brief Raise an IndexError from C++ for testing purposes
 * @param message Error message to include in the exception
 */
[[noreturn]] void raise_index_error(const std::string& message) { throw pypto::IndexError(message); }

/**
 * @brief Raise a generic Error from C++ for testing purposes
 * @param message Error message to include in the exception
 */
[[noreturn]] void raise_generic_error(const std::string& message) { throw pypto::Error(message); }

/**
 * @brief Raise an AssertionError from C++ for testing purposes
 * @param message Error message to include in the exception
 */
[[noreturn]] void raise_assertion_error(const std::string& message) { throw pypto::AssertionError(message); }

/**
 * @brief Raise an InternalError from C++ for testing purposes
 * @param message Error message to include in the exception
 */
[[noreturn]] void raise_internal_error(const std::string& message) { throw pypto::InternalError(message); }

// ============================================================================
// Module binding
// ============================================================================

void BindTesting(py::module_& m) {
  // Create a protected submodule for testing utilities
  // This will be accessible as pypto.testing in Python
  py::module_ testing = m.def_submodule("testing", "Internal testing utilities (do not use in production)");

  // Register error-raising helper functions
  testing.def("raise_value_error", &raise_value_error, py::arg("message"),
              "Raise a ValueError from C++ for testing error handling");

  testing.def("raise_type_error", &raise_type_error, py::arg("message"),
              "Raise a TypeError from C++ for testing error handling");

  testing.def("raise_runtime_error", &raise_runtime_error, py::arg("message"),
              "Raise a RuntimeError from C++ for testing error handling");

  testing.def("raise_not_implemented_error", &raise_not_implemented_error, py::arg("message"),
              "Raise a NotImplementedError from C++ for testing error handling");

  testing.def("raise_index_error", &raise_index_error, py::arg("message"),
              "Raise an IndexError from C++ for testing error handling");

  testing.def("raise_generic_error", &raise_generic_error, py::arg("message"),
              "Raise a generic Error from C++ for testing error handling");

  testing.def("raise_assertion_error", &raise_assertion_error, py::arg("message"),
              "Raise an AssertionError from C++ for testing error handling");

  testing.def("raise_internal_error", &raise_internal_error, py::arg("message"),
              "Raise an InternalError from C++ for testing error handling");
}

}  // namespace python
}  // namespace pypto
