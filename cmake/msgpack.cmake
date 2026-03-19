# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# Auto-update git submodules if not initialized
find_package(Git QUIET)

if(GIT_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/3rdparty/msgpack-c/.git")
        message(STATUS "Initializing msgpack-c git submodule...")
        execute_process(
            COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            RESULT_VARIABLE GIT_SUBMOD_RESULT
        )

        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}")
        endif()
    endif()
endif()

# Set up msgpack-c (header-only library)
set(MSGPACK_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/msgpack-c")

if(NOT EXISTS "${MSGPACK_SOURCE_DIR}/include/msgpack.hpp")
    message(FATAL_ERROR "msgpack-c submodule not found. Please run: git submodule update --init --recursive")
endif()

# Create imported interface target for header-only library
add_library(msgpackc-cxx INTERFACE)
target_include_directories(msgpackc-cxx INTERFACE "${MSGPACK_SOURCE_DIR}/include")

# Set C++17 standard for msgpack
target_compile_features(msgpackc-cxx INTERFACE cxx_std_17)

# Disable Boost dependency in msgpack (use standalone mode)
target_compile_definitions(msgpackc-cxx INTERFACE MSGPACK_NO_BOOST)

message(STATUS "msgpack-c configured from: ${MSGPACK_SOURCE_DIR}")
