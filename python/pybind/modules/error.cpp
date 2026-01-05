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
 * @file error.cpp
 * @brief Implementation of Python bindings for PyPTO error classes
 */

#include "pypto/core/error.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../bindings.h"

namespace py = pybind11;

namespace pypto {
namespace python {

void BindErrors(py::module_& m) {
  // Register custom exception types and map them to Python exceptions
  // These static objects ensure exceptions persist for the lifetime of the module
  static py::exception<pypto::Error> exc_error(m, "Error", PyExc_Exception);
  static py::exception<pypto::ValueError> exc_value_error(m, "ValueError", PyExc_ValueError);
  static py::exception<pypto::TypeError> exc_type_error(m, "TypeError", PyExc_TypeError);
  static py::exception<pypto::RuntimeError> exc_runtime_error(m, "RuntimeError", PyExc_RuntimeError);
  static py::exception<pypto::NotImplementedError> exc_not_implemented_error(m, "NotImplementedError",
                                                                             PyExc_NotImplementedError);
  static py::exception<pypto::IndexError> exc_index_error(m, "IndexError", PyExc_IndexError);
  static py::exception<pypto::AssertionError> exc_assertion_error(m, "AssertionError", PyExc_AssertionError);
  static py::exception<pypto::InternalError> exc_internal_error(m, "InternalError", PyExc_RuntimeError);

  // Set __module__ to "pypto" so the exception displays as "pypto.InternalError" instead of
  // "pypto.pypto_core.InternalError"
  PyObject* internal_error_type = exc_internal_error.ptr();
  PyObject_SetAttrString(internal_error_type, "__module__", PyUnicode_FromString("pypto"));

  // Register exception translator to convert C++ exceptions to Python exceptions
  // This translator includes the full stack trace in the Python exception message
  py::register_exception_translator([](std::exception_ptr p) {  // NOLINT(performance-unnecessary-value-param)
    try {
      if (p) std::rethrow_exception(p);
    } catch (const pypto::ValueError& e) {
      // Catch most specific exceptions first
      PyErr_SetString(PyExc_ValueError, e.GetFullMessage().c_str());
    } catch (const pypto::TypeError& e) {
      PyErr_SetString(PyExc_TypeError, e.GetFullMessage().c_str());
    } catch (const pypto::RuntimeError& e) {
      PyErr_SetString(PyExc_RuntimeError, e.GetFullMessage().c_str());
    } catch (const pypto::NotImplementedError& e) {
      PyErr_SetString(PyExc_NotImplementedError, e.GetFullMessage().c_str());
    } catch (const pypto::IndexError& e) {
      PyErr_SetString(PyExc_IndexError, e.GetFullMessage().c_str());
    } catch (const pypto::AssertionError& e) {
      PyErr_SetString(PyExc_AssertionError, e.GetFullMessage().c_str());
    } catch (const pypto::InternalError& e) {
      PyErr_SetString(exc_internal_error.ptr(), e.GetFullMessage().c_str());
    } catch (const pypto::Error& e) {
      // Catch base Error last as a fallback
      PyErr_SetString(PyExc_Exception, e.GetFullMessage().c_str());
    }
  });
}

}  // namespace python
}  // namespace pypto
