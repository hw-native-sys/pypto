# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# This optional module uses the exact pinned SDK C++ implementation. Simpler's
# Python DSO hides its C++ symbols, so link SDK sources rather than extracting
# private context addresses or calling Python from a taskQueue callback.
execute_process(COMMAND "${Python_EXECUTABLE}" -c
    "import pathlib,torch,importlib.util; assert torch._C._GLIBCXX_USE_CXX11_ABI, 'The pinned Simpler binding requires the C++11 libstdc++ ABI'; print(torch.utils.cmake_prefix_path); print(pathlib.Path(importlib.util.find_spec('torch_npu').origin).parent)"
    OUTPUT_VARIABLE _torch_paths OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _torch_result)
if(NOT _torch_result EQUAL 0)
    message(FATAL_ERROR "PYPTO_BUILD_TORCH_NPU requires matching torch and torch_npu")
endif()
string(REPLACE "\n" ";" _torch_paths "${_torch_paths}")
list(GET _torch_paths 0 _torch_cmake)
list(GET _torch_paths 1 _torch_npu)
list(APPEND CMAKE_PREFIX_PATH "${_torch_cmake}")
find_package(Torch REQUIRED)
find_path(PYPTO_ACL_INCLUDE acl/acl.h HINTS "$ENV{ASCEND_HOME_PATH}/include")
find_library(PYPTO_ASCENDCL ascendcl HINTS "$ENV{ASCEND_HOME_PATH}/lib64")
if(NOT PYPTO_ACL_INCLUDE OR NOT PYPTO_ASCENDCL)
    message(FATAL_ERROR "Source CANN set_env.sh before building the torch_npu adapter")
endif()
set(_sdk "${CMAKE_SOURCE_DIR}/runtime/src/common")
execute_process(COMMAND git rev-parse HEAD WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/runtime"
    OUTPUT_VARIABLE _sdk_revision OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _sdk_result)
if(NOT _sdk_result EQUAL 0)
    message(FATAL_ERROR "Cannot identify the pinned Simpler source revision")
endif()
nanobind_add_module(_torch_npu torch_npu_adapter.cpp
    "${_sdk}/worker/chip_worker.cpp" "${_sdk}/worker/chip_run_lane.cpp"
    "${_sdk}/log/host_log.cpp" "${_sdk}/log/unified_log_host.cpp"
    "${CMAKE_SOURCE_DIR}/src/core/backtrace.cpp" "${CMAKE_SOURCE_DIR}/src/core/error.cpp")
target_include_directories(_torch_npu PRIVATE
    "${LIBBACKTRACE_INSTALL_DIR}/include" "${_torch_npu}/include" "${PYPTO_ACL_INCLUDE}"
    "${_sdk}" "${_sdk}/worker" "${_sdk}/task_interface" "${_sdk}/hierarchical"
    "${_sdk}/platform/include" "${_sdk}/platform/include/common"
    "${_sdk}/platform/include/host" "${_sdk}/log/include")
separate_arguments(_torch_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(_torch_npu PRIVATE ${_torch_flags})
target_compile_definitions(_torch_npu PRIVATE PYPTO_SIMPLER_REVISION="${_sdk_revision}")
target_link_libraries(_torch_npu PRIVATE libbacktrace ${TORCH_LIBRARIES}
    "${TORCH_INSTALL_PREFIX}/lib/libtorch_python.so"
    "${_torch_npu}/lib/libtorch_npu.so" "${PYPTO_ASCENDCL}" ${CMAKE_DL_LIBS} pthread)
set_target_properties(_torch_npu PROPERTIES INSTALL_RPATH
    "$ORIGIN/../torch/lib;$ORIGIN/../torch_npu/lib")
if(SKBUILD_MODE)
    install(TARGETS _torch_npu DESTINATION .)
else()
    set_target_properties(_torch_npu PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/pypto")
endif()

option(PYPTO_BUILD_TORCH_NPU_TESTS "Build test-only queue gates and adapter counters" OFF)
if(PYPTO_BUILD_TORCH_NPU_TESTS)
    target_compile_definitions(_torch_npu PRIVATE PYPTO_KERNEL_TEST_COUNTERS=1)
    nanobind_add_module(_torch_npu_test "${CMAKE_SOURCE_DIR}/tests/st/runtime/kernel/queue_gate.cpp"
        "${CMAKE_SOURCE_DIR}/src/core/backtrace.cpp" "${CMAKE_SOURCE_DIR}/src/core/error.cpp")
    target_include_directories(_torch_npu_test PRIVATE
        "${LIBBACKTRACE_INSTALL_DIR}/include" "${_torch_npu}/include" "${PYPTO_ACL_INCLUDE}")
    target_compile_options(_torch_npu_test PRIVATE ${_torch_flags})
    target_link_libraries(_torch_npu_test PRIVATE libbacktrace ${TORCH_LIBRARIES}
        "${_torch_npu}/lib/libtorch_npu.so" "${PYPTO_ASCENDCL}")
    set_target_properties(_torch_npu_test PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build/torch_npu_tests")
endif()
