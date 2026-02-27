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
 * @file module.h
 * @brief Common header for all Python bindings in PyPTO
 *
 * This header declares all the binding functions for different modules.
 * Each module (error, testing, tensor, ops, etc.) should implement its
 * Bind* function and include this header.
 */

#ifndef PYTHON_BINDINGS_MODULE_H_
#define PYTHON_BINDINGS_MODULE_H_

#include <nanobind/nanobind.h>

namespace pypto {
namespace python {

/**
 * @brief Register error exception types and exception translator
 *
 * This function registers all PyPTO error classes with Python and sets up
 * an exception translator that converts C++ exceptions to Python exceptions
 * with full stack trace information.
 *
 * @param m The nanobind module object
 */
void BindErrors(nanobind::module_& m);

/**
 * @brief Register testing utilities as a submodule
 *
 * Creates a protected testing submodule containing helper functions
 * for testing error handling and other internal functionality.
 *
 * @param m The parent nanobind module object
 */
void BindTesting(nanobind::module_& m);

/**
 * @brief Register core types and utilities
 *
 * Registers core PyPTO types including DataType enum and related utility functions.
 *
 * @param m The nanobind module object
 */
void BindCore(nanobind::module_& m);

/**
 * @brief Register IR (Intermediate Representation) classes
 *
 * Registers all IR node classes including Span, IRNodeNode, Expr, Var, Const,
 * BinaryExpr, and all binary operations (Add, Sub, Mul, Div, Mod).
 *
 * @param m The nanobind module object
 */
void BindIR(nanobind::module_& m);

/**
 * @brief Register IR Builder for incremental IR construction
 *
 * Registers the IRBuilder class and related context management classes
 * for building IR incrementally with context managers.
 *
 * @param m The nanobind module object
 */
void BindIRBuilder(nanobind::module_& m);

/**
 * @brief Register Pass classes for IR transformations
 *
 * Registers the Pass base class and concrete pass implementations
 * (e.g., InitMemRef, AllocateMemoryAddr) for IR transformations.
 *
 * @param m The nanobind module object
 */
void BindPass(nanobind::module_& m);

/**
 * @brief Register logging framework types and functions
 *
 * Registers the LogLevel enum and LoggerManager functions for controlling
 * the logging system from Python.
 *
 * @param m The nanobind module object
 */
void BindLogging(nanobind::module_& m);

/**
 * @brief Register code generation (codegen) classes
 *
 * Registers the CCECodegen class and related code generation functionality
 * for converting PyPTO IR to pto-isa C++ code.
 *
 * @param m The nanobind module object
 */
void BindCodegen(nanobind::module_& m);

/**
 * @brief Register backend classes
 *
 * Registers SoC hierarchy, builders, and backend implementations.
 *
 * @param m The nanobind module object
 */
void BindBackend(nanobind::module_& m);

// Future binding declarations can be added here:
// void BindTensors(nanobind::module_& m);
// void BindOps(nanobind::module_& m);
// void BindDevices(nanobind::module_& m);

}  // namespace python
}  // namespace pypto

#endif  // PYTHON_BINDINGS_MODULE_H_
