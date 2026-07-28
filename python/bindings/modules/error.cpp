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

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <cstdlib>
#include <string>
#include <string_view>

#include "../module.h"

namespace nb = nanobind;

namespace pypto {
namespace python {

namespace {

/// Whether the user asked for C++ tracebacks on every error via `PTO_BACKTRACE=1`.
///
/// Read on each throw rather than cached, so it can be toggled in-process (tests, REPL). The
/// lookup is negligible next to the stack capture the exception already paid for. Only the exact
/// value `1` enables traces; unlike the DSL parser, which rejects anything other than `0` / `1`,
/// an unrecognised value is ignored rather than reported — throwing from inside an exception
/// translator would replace the error the user is actually trying to read.
bool TracebackRequested() {
  const char* env = std::getenv("PTO_BACKTRACE");
  return env != nullptr && std::string_view(env) == "1";
}

/// Build the Python-visible message for a *user* error (the `CHECK` family).
///
/// The C++ frames name PyPTO internals that mean nothing to the caller, and they push the DSL
/// source snippet further down the message, so they are opt-in through `PTO_BACKTRACE=1` — the
/// same switch the DSL diagnostics already advertise. Bug-class exceptions (the `INTERNAL_CHECK`
/// family) bypass this helper and always report `GetFullMessage()`.
std::string UserErrorMessage(const Error& e) { return TracebackRequested() ? e.GetFullMessage() : e.what(); }

}  // namespace

void BindErrors(nb::module_& m) {
  // Register custom exception types and map them to Python exceptions
  // These static objects ensure exceptions persist for the lifetime of the module
  static nb::exception<pypto::Error> exc_error(m, "Error", PyExc_Exception);
  static nb::exception<pypto::ValueError> exc_value_error(m, "ValueError", PyExc_ValueError);
  static nb::exception<pypto::TypeError> exc_type_error(m, "TypeError", PyExc_TypeError);
  static nb::exception<pypto::RuntimeError> exc_runtime_error(m, "RuntimeError", PyExc_RuntimeError);
  static nb::exception<pypto::NotImplementedError> exc_not_implemented_error(m, "NotImplementedError",
                                                                             PyExc_NotImplementedError);
  static nb::exception<pypto::IndexError> exc_index_error(m, "IndexError", PyExc_IndexError);
  static nb::exception<pypto::AssertionError> exc_assertion_error(m, "AssertionError", PyExc_AssertionError);
  static nb::exception<pypto::InternalError> exc_internal_error(m, "InternalError", PyExc_RuntimeError);

  // Set __module__ to "pypto" so the exception displays as "pypto.InternalError" instead of
  // "pypto.pypto_core.InternalError"
  PyObject* internal_error_type = exc_internal_error.ptr();
  PyObject_SetAttrString(internal_error_type, "__module__", PyUnicode_FromString("pypto"));

  // Register exception translator to convert C++ exceptions to Python exceptions.
  // See UserErrorMessage() above for when the C++ stack trace is included in the message.
  nb::register_exception_translator([](const std::exception_ptr& p, void*) {
    try {
      if (p) std::rethrow_exception(p);
    } catch (const pypto::ValueError& e) {
      // Catch most specific exceptions first
      PyErr_SetString(PyExc_ValueError, UserErrorMessage(e).c_str());
    } catch (const pypto::TypeError& e) {
      PyErr_SetString(PyExc_TypeError, UserErrorMessage(e).c_str());
    } catch (const pypto::RuntimeError& e) {
      PyErr_SetString(PyExc_RuntimeError, UserErrorMessage(e).c_str());
    } catch (const pypto::NotImplementedError& e) {
      // User class, not bug class: an unlowered feature is a documented limitation surfaced to the
      // caller — the same category error-checking.md assigns to CHECK — not a failed invariant.
      PyErr_SetString(PyExc_NotImplementedError, UserErrorMessage(e).c_str());
    } catch (const pypto::IndexError& e) {
      PyErr_SetString(PyExc_IndexError, UserErrorMessage(e).c_str());
    } catch (const pypto::AssertionError& e) {
      // Bug class (the `INTERNAL_CHECK` family, here and in the next handler): a failed internal
      // invariant is a PyPTO bug, and the traceback is the primary artefact for diagnosing it, so
      // it is always reported.
      PyErr_SetString(PyExc_AssertionError, e.GetFullMessage().c_str());
    } catch (const pypto::InternalError& e) {
      PyErr_SetString(exc_internal_error.ptr(), e.GetFullMessage().c_str());
    } catch (const pypto::Error& e) {
      // Catch base Error last as a fallback. VerificationError lands here too: its diagnostic
      // report already pinpoints the offending IR, and the frames above the throw are always the
      // same verifier-registry entry, so it follows the user-facing default.
      PyErr_SetString(PyExc_Exception, UserErrorMessage(e).c_str());
    }
  });
}

}  // namespace python
}  // namespace pypto
