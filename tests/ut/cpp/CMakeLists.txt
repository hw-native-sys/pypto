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

add_executable(pypto_tcvt_path_test
    tcvt_path_test.cpp
    ${CMAKE_SOURCE_DIR}/src/backend/common/tcvt_path.cpp
)

target_include_directories(pypto_tcvt_path_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

add_test(
    NAME pypto_tcvt_path_test
    COMMAND pypto_tcvt_path_test
)
