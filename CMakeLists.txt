# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.15)

# Support both scikit-build and native CMake builds
if(DEFINED SKBUILD_PROJECT_NAME)
    # Building with scikit-build-core
    project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION})
    set(SKBUILD_MODE ON)
else()
    # Building with native CMake
    file(STRINGS "${CMAKE_SOURCE_DIR}/pyproject.toml" _version_line LIMIT_COUNT 1 REGEX "^version ?= ?\"")

    if(_version_line)
        string(REGEX REPLACE "^version ?= ?\"([^\"]*)\".*" "\\1" _pypto_version "${_version_line}")
        project(pypto VERSION ${_pypto_version})
    else()
        message(FATAL_ERROR "Failed to parse version from pyproject.toml")
    endif()

    set(SKBUILD_MODE OFF)
endif()

set(LIBRARY_NAME "${PROJECT_NAME}_core")

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable ccache if available
find_program(CCACHE_PROGRAM ccache)

if(CCACHE_PROGRAM)
    message(STATUS "Found ccache: ${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
else()
    message(STATUS "ccache not found")
endif()

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build type (Debug or Release)" FORCE)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -O2")

# Fix debug info paths when building in a temp directory (e.g., pip install)
# This remaps paths in DWARF debug info so backtraces show clean relative paths
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    # Remap source directory to "." so absolute source paths become relative
    add_compile_options(-fdebug-prefix-map=${CMAKE_SOURCE_DIR}=.)
endif()

# Find Python and nanobind
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(nanobind CONFIG REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# External dependencies
include(${CMAKE_SOURCE_DIR}/cmake/libbacktrace.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/msgpack.cmake)

# PyPTO core sources
set(PYPTO_SOURCES
    src/codegen/pto/pto_codegen.cpp
    src/core/backtrace.cpp
    src/core/error.cpp
    src/ir/core.cpp
    src/ir/expr.cpp
    src/ir/stmt.cpp
    src/ir/function.cpp
    src/ir/program.cpp
    src/ir/type.cpp
    src/ir/builder.cpp
    src/ir/memref.cpp
    src/ir/op_registry.cpp
    src/ir/op/type_inference.cpp
    src/ir/op/tensor_ops/elementwise.cpp
    src/ir/op/tensor_ops/matmul.cpp
    src/ir/op/tensor_ops/reduction.cpp
    src/ir/op/tensor_ops/unary.cpp
    src/ir/op/tensor_ops/memory.cpp
    src/ir/op/tensor_ops/transform.cpp
    src/ir/op/block_ops/elementwise.cpp
    src/ir/op/block_ops/reduction.cpp
    src/ir/op/block_ops/unary.cpp
    src/ir/op/block_ops/memory.cpp
    src/ir/op/block_ops/matmul.cpp
    src/ir/op/block_ops/batch_matmul.cpp
    src/ir/op/block_ops/broadcast.cpp
    src/ir/op/block_ops/transform.cpp
    src/ir/op/sync_ops/sync.cpp
    src/ir/op/testing.cpp
    src/ir/transforms/visitor.cpp
    src/ir/transforms/mutator.cpp
    src/ir/transforms/python_printer.cpp
    src/ir/transforms/structural_hash.cpp
    src/ir/transforms/structural_equal.cpp
    src/ir/transforms/dependency_analyzer.cpp
    src/ir/transforms/passes.cpp
    src/ir/transforms/init_memref.cpp
    src/ir/transforms/basic_memory_reuse_pass.cpp
    src/ir/transforms/insert_sync_pass.cpp
    src/ir/transforms/verifier.cpp
    src/codegen/cce/code_emitter.cpp
    src/codegen/cce/code_context.cpp
    src/codegen/cce/type_converter.cpp
    src/codegen/cce/cce_codegen.cpp
    src/codegen/codegen_base.cpp
    src/codegen/orchestration_op_registry.cpp
    src/codegen/tensor_op_codegen.cpp
    src/codegen/orchestration/orchestration_codegen.cpp
    src/ir/transforms/add_alloc_pass.cpp
    src/ir/transforms/verify_ssa_pass.cpp
    src/ir/transforms/type_check_pass.cpp
    src/ir/transforms/convert_to_ssa_pass.cpp
    src/ir/transforms/outline_incore_scopes_pass.cpp
    src/ir/transforms/flatten_call_expr_pass.cpp
    src/ir/transforms/verify_no_nested_call_pass.cpp
    src/ir/transforms/utils/parent_stmt_analysis.cpp
    src/ir/transforms/utils/normalize_stmt_structure.cpp
    src/ir/transforms/utils/flatten_single_stmt.cpp
    src/ir/serialization/serializer.cpp
    src/ir/serialization/deserializer.cpp
    src/ir/serialization/type_registry.cpp
    src/ir/serialization/type_deserializers.cpp
    src/backend/common/soc.cpp
    src/backend/common/backend.cpp
    src/backend/common/backend_config.cpp
    src/backend/common/backend_registry.cpp
    src/backend/910B_CCE/backend_910b_cce.cpp
    src/backend/910B_CCE/backend_910b_cce_ops.cpp
    src/backend/910B_PTO/backend_910b_pto.cpp
    src/backend/910B_PTO/backend_910b_pto_ops.cpp
)

# Include Python bindings
add_subdirectory(python/bindings)
