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

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>  // NOLINT(misc-include-cleaner) -- registers shared_ptr casters
#include <nanobind/stl/string.h>      // NOLINT(misc-include-cleaner) -- registers std::string casters

#include <cassert>
#include <string>
#include <utility>

#include "../module.h"
#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/function.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/span.h"
#include "pypto/ir/transforms/dsa/allocation_plan.h"
#include "pypto/ir/transforms/dsa/reuse_penalty_recognizer.h"

namespace nb = nanobind;

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

[[noreturn]] void raise_internal_error_with_span(const std::string& message, const std::string& filename,
                                                 int line, int col) {
  ir::Span span(filename, line, col);
  INTERNAL_CHECK_SPAN(false, span) << message;
}

/**
 * @brief Return DSA-RP recognizer output without running placement.
 *
 * This intentionally lives in the internal testing module: production callers
 * consume the same recognizer through MemRefDsaAdapter, while unit tests need
 * to distinguish edge construction from solver tie-breaking.
 */
nb::list RecognizeDsaReusePenaltiesForTesting(const ir::FunctionPtr& func) {
  const ir::dsa_adapter::AllocationPlan plan = ir::dsa_adapter::BuildDsaAllocationPlan(func);
  const auto penalties =
      ir::dsa_adapter::RecognizeReusePenalties(func, plan, *backend::BackendConfig::GetBackend());

  nb::list result;
  for (const auto& penalty : penalties) {
    INTERNAL_CHECK(penalty.first_interval < plan.intervals.size());
    INTERNAL_CHECK(penalty.second_interval < plan.intervals.size());
    nb::dict edge;
    edge["first_interval"] = penalty.first_interval;
    edge["second_interval"] = penalty.second_interval;
    edge["first_name"] = plan.intervals[penalty.first_interval].variable->name_hint_;
    edge["second_name"] = plan.intervals[penalty.second_interval].variable->name_hint_;
    edge["cost"] = penalty.cost;
    result.append(std::move(edge));
  }
  return result;
}

/**
 * @brief Return exact backend pipe inference for a Call, or None.
 */
nb::object TryInferPipeForTesting(const ir::CallPtr& call) {
  const auto pipe = backend::BackendConfig::GetBackend()->TryInferPipe(call);
  if (!pipe) return nb::none();
  return nb::int_(static_cast<int>(*pipe));
}

/**
 * @brief Return an operation's registered execution-memory-access evidence.
 */
std::string GetExecutionMemoryAccessEvidenceForTesting(const std::string& op_name) {
  const auto& registry = ir::OpRegistry::GetInstance();
  CHECK(registry.IsRegistered(op_name)) << "Unknown operation '" << op_name << "'";
  switch (registry.GetEntry(op_name).GetExecutionMemoryAccessEvidence()) {
    case ir::ExecutionMemoryAccessEvidence::Unknown:
      return "unknown";
    case ir::ExecutionMemoryAccessEvidence::Functional:
      return "functional";
    case ir::ExecutionMemoryAccessEvidence::NoAccess:
      return "no_access";
  }
  INTERNAL_UNREACHABLE << "Unknown execution-memory-access evidence";
}

// ============================================================================
// Module binding
// ============================================================================

void BindTesting(nb::module_& m) {
  // Create a protected submodule for testing utilities
  // This will be accessible as pypto.testing in Python
  nb::module_ testing = m.def_submodule("testing", "Internal testing utilities (do not use in production)");

  // Register error-raising helper functions
  testing.def("raise_value_error", &raise_value_error, nb::arg("message"),
              "Raise a ValueError from C++ for testing error handling");

  testing.def("raise_type_error", &raise_type_error, nb::arg("message"),
              "Raise a TypeError from C++ for testing error handling");

  testing.def("raise_runtime_error", &raise_runtime_error, nb::arg("message"),
              "Raise a RuntimeError from C++ for testing error handling");

  testing.def("raise_not_implemented_error", &raise_not_implemented_error, nb::arg("message"),
              "Raise a NotImplementedError from C++ for testing error handling");

  testing.def("raise_index_error", &raise_index_error, nb::arg("message"),
              "Raise an IndexError from C++ for testing error handling");

  testing.def("raise_generic_error", &raise_generic_error, nb::arg("message"),
              "Raise a generic Error from C++ for testing error handling");

  testing.def("raise_assertion_error", &raise_assertion_error, nb::arg("message"),
              "Raise an AssertionError from C++ for testing error handling");

  testing.def("raise_internal_error", &raise_internal_error, nb::arg("message"),
              "Raise an InternalError from C++ for testing error handling");

  testing.def("raise_internal_error_with_span", &raise_internal_error_with_span, nb::arg("message"),
              nb::arg("filename"), nb::arg("line"), nb::arg("col"),
              "Raise an InternalError with IR source span for testing");

  testing.def("recognize_dsa_reuse_penalties", &RecognizeDsaReusePenaltiesForTesting, nb::arg("function"),
              "Return recognized DSA-RP edges without running placement");

  testing.def("try_infer_pipe", &TryInferPipeForTesting, nb::arg("call"),
              "Return the exact backend pipe for a Call, or None");

  testing.def("get_execution_memory_access_evidence", &GetExecutionMemoryAccessEvidenceForTesting,
              nb::arg("op_name"), "Return an operation's execution-memory-access evidence");
}

}  // namespace python
}  // namespace pypto
