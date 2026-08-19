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
#include "pypto/ir/core_affinity_kind.h"
#include "pypto/ir/function.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/span.h"
#include "pypto/ir/transforms/dsa/allocation_plan.h"
#include "pypto/ir/transforms/dsa/reuse_penalty_recognizer.h"
#include "pypto/ir/transforms/utils/core_affinity.h"

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
 * @brief Throw the PyPTO exception named by `kind`
 *
 * Sole throw site for `rethrow_with_message` below, so tests can assert that the
 * reported stack trace still names this function after the rethrow.
 *
 * @param kind Exception class name, e.g. "InternalError"
 * @param message Error message to include in the exception
 */
[[noreturn]] void RaiseByKind(const std::string& kind, const std::string& message) {
  if (kind == "ValueError") throw pypto::ValueError(message);
  if (kind == "TypeError") throw pypto::TypeError(message);
  if (kind == "RuntimeError") throw pypto::RuntimeError(message);
  if (kind == "NotImplementedError") throw pypto::NotImplementedError(message);
  if (kind == "IndexError") throw pypto::IndexError(message);
  if (kind == "AssertionError") throw pypto::AssertionError(message);
  if (kind == "InternalError") throw pypto::InternalError(message);
  if (kind == "Error") throw pypto::Error(message);
  throw pypto::ValueError("Unknown exception kind for testing: " + kind);
}

/**
 * @brief Raise `kind`, then route it through Error::RethrowWithMessage
 *
 * Exercises the trace-preserving typed rethrow that OpRegistry::CreateImpl depends on:
 * the exception reaching Python must keep the concrete type and the frames captured at
 * the original throw inside RaiseByKind, while carrying the replacement message.
 *
 * @param kind Exception class name to raise
 * @param original Message given to the original exception
 * @param replacement Message the rethrown exception should carry instead
 */
[[noreturn]] void rethrow_with_message(const std::string& kind, const std::string& original,
                                       const std::string& replacement) {
  try {
    RaiseByKind(kind, original);
  } catch (const pypto::Error& e) {
    e.RethrowWithMessage(replacement);
  }
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

/**
 * @brief Spell a CoreAffinity as the lowercase string the Python tests assert on.
 */
const char* CoreAffinityToString(ir::core_affinity::CoreAffinity affinity) {
  switch (affinity) {
    case ir::core_affinity::CoreAffinity::CUBE:
      return "cube";
    case ir::core_affinity::CoreAffinity::VECTOR:
      return "vector";
    case ir::core_affinity::CoreAffinity::SHARED:
      return "shared";
    case ir::core_affinity::CoreAffinity::MIXED:
      return "mixed";
  }
  INTERNAL_UNREACHABLE << "Unknown core affinity";
}

/**
 * @brief Return an operation's explicitly declared core affinity, or None.
 *
 * None means the op declares no affinity and ClassifyCallAffinity derives it
 * from the call (memory spec, operand tiles, result tile).
 */
nb::object GetDeclaredCoreAffinityForTesting(const std::string& op_name) {
  const auto& registry = ir::OpRegistry::GetInstance();
  CHECK(registry.IsRegistered(op_name)) << "Unknown operation '" << op_name << "'";
  const auto affinity = registry.GetEntry(op_name).GetCoreAffinity();
  if (!affinity) return nb::none();
  return nb::str(CoreAffinityToString(*affinity));
}

/**
 * @brief Return the core affinity ClassifyCallAffinity derives for a Call.
 *
 * Unlike get_declared_core_affinity this is the *effective* placement: it runs
 * the full classification chain (declared affinity, then the dynamic special
 * cases, then output memory spec, first tile argument, and result tile memory),
 * so it depends on how far the call has been lowered.
 */
std::string ClassifyCallAffinityForTesting(const ir::CallPtr& call) {
  return CoreAffinityToString(ir::core_affinity::ClassifyCallAffinity(call));
}

/**
 * @brief Return whether an operation must not run on a second core.
 *
 * The set_no_duplicate() axis: replicating such an op onto the other lane of a
 * mixed kernel changes what the program means. Read by LowerAutoVectorSplit's
 * pl.split_aiv region placement stamp.
 */
bool IsNoDuplicateOpForTesting(const std::string& op_name) {
  const auto& registry = ir::OpRegistry::GetInstance();
  CHECK(registry.IsRegistered(op_name)) << "Unknown operation '" << op_name << "'";
  return registry.GetEntry(op_name).IsNoDuplicate();
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

  testing.def("rethrow_with_message", &rethrow_with_message, nb::arg("kind"), nb::arg("original"),
              nb::arg("replacement"),
              "Raise `kind` and rethrow it via Error::RethrowWithMessage for testing");

  testing.def("recognize_dsa_reuse_penalties", &RecognizeDsaReusePenaltiesForTesting, nb::arg("function"),
              "Return recognized DSA-RP edges without running placement");

  testing.def("try_infer_pipe", &TryInferPipeForTesting, nb::arg("call"),
              "Return the exact backend pipe for a Call, or None");

  testing.def("get_execution_memory_access_evidence", &GetExecutionMemoryAccessEvidenceForTesting,
              nb::arg("op_name"), "Return an operation's execution-memory-access evidence");

  testing.def("get_declared_core_affinity", &GetDeclaredCoreAffinityForTesting, nb::arg("op_name"),
              "Return an operation's explicitly declared core affinity, or None");

  testing.def("is_no_duplicate_op", &IsNoDuplicateOpForTesting, nb::arg("op_name"),
              "Return whether an operation must not run on a second core");

  testing.def("classify_call_affinity", &ClassifyCallAffinityForTesting, nb::arg("call"),
              "Return the core affinity ClassifyCallAffinity derives for a Call");
}

}  // namespace python
}  // namespace pypto
