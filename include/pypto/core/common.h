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
 * @file common.h
 * @brief Common macros, constants, and utility definitions
 *
 * This header provides commonly used macros and constants that are shared
 * across the PyPTO codebase, including:
 * - Compiler hints and attributes
 * - Utility macros for code generation
 * - Build configuration constants
 * - nanobind module configuration
 */

#ifndef PYPTO_CORE_COMMON_H_
#define PYPTO_CORE_COMMON_H_

namespace pypto {

// ============================================================================
// Version Information
// ============================================================================

#define PYPTO_VERSION_MAJOR 0
#define PYPTO_VERSION_MINOR 1
#define PYPTO_VERSION_PATCH 0

// ============================================================================
// nanobind Module Configuration
// ============================================================================

// Default docstring for the nanobind module
#define PYPTO_NANOBIND_MODULE_DOC "PyPTO core library"

// ============================================================================
// Compiler Hints and Attributes
// ============================================================================

#define PYPTO_ALWAYS_INLINE __attribute__((always_inline))
}  // namespace pypto
#endif  // PYPTO_CORE_COMMON_H_
