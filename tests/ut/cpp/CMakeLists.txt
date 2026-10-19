# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_executable(pypto_dsa_reuse_penalty_solver_test
    dsa_reuse_penalty_solver_test.cpp
    ${CMAKE_SOURCE_DIR}/src/ir/transforms/dsa/dsa_reuse_penalty_solver.cpp
)

target_include_directories(pypto_dsa_reuse_penalty_solver_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

add_test(
    NAME pypto_dsa_reuse_penalty_solver_test
    COMMAND pypto_dsa_reuse_penalty_solver_test
)

add_executable(pypto_structural_comparison_arena_test structural_comparison_arena_test.cpp)
target_include_directories(pypto_structural_comparison_arena_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
# Keep assert-based checks enabled in release builds as well.
target_compile_options(pypto_structural_comparison_arena_test PRIVATE -UNDEBUG)
add_test(NAME pypto_structural_comparison_arena_test COMMAND pypto_structural_comparison_arena_test)

add_executable(pypto_bounded_kernel_index_test bounded_kernel_index_test.cpp)
target_include_directories(pypto_bounded_kernel_index_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_options(pypto_bounded_kernel_index_test PRIVATE -UNDEBUG)
add_test(NAME pypto_bounded_kernel_index_test COMMAND pypto_bounded_kernel_index_test)
