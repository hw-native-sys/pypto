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
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/3rdparty/libbacktrace/.git")
        message(STATUS "Initializing git submodules...")
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

# Set up libbacktrace as an external project
include(ExternalProject)

set(LIBBACKTRACE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/libbacktrace")
set(LIBBACKTRACE_INSTALL_DIR "${CMAKE_BINARY_DIR}/3rdparty/libbacktrace")
set(LIBBACKTRACE_BUILD_DIR "${CMAKE_BINARY_DIR}/3rdparty/libbacktrace/build")

# Create installation directories
file(MAKE_DIRECTORY ${LIBBACKTRACE_INSTALL_DIR}/include)
file(MAKE_DIRECTORY ${LIBBACKTRACE_INSTALL_DIR}/lib)

ExternalProject_Add(project_libbacktrace
    PREFIX ${LIBBACKTRACE_BUILD_DIR}
    SOURCE_DIR ${LIBBACKTRACE_SOURCE_DIR}
    BINARY_DIR ${LIBBACKTRACE_BUILD_DIR}
    CONFIGURE_COMMAND ${LIBBACKTRACE_SOURCE_DIR}/configure
    "--prefix=${LIBBACKTRACE_INSTALL_DIR}"
    --with-pic
    BUILD_COMMAND make -j${CMAKE_BUILD_PARALLEL_LEVEL}
    INSTALL_COMMAND make install
    BUILD_BYPRODUCTS "${LIBBACKTRACE_INSTALL_DIR}/lib/libbacktrace.a"
    "${LIBBACKTRACE_INSTALL_DIR}/include/backtrace.h"
)

# Create imported target
add_library(libbacktrace STATIC IMPORTED)
set_target_properties(libbacktrace PROPERTIES
    IMPORTED_LOCATION ${LIBBACKTRACE_INSTALL_DIR}/lib/libbacktrace.a
    INTERFACE_INCLUDE_DIRECTORIES ${LIBBACKTRACE_INSTALL_DIR}/include
)
add_dependencies(libbacktrace project_libbacktrace)

# Function to run dsymutil on macOS for libbacktrace debug symbols
function(pypto_add_apple_dsymutil target_name)
    if(APPLE)
        find_program(DSYMUTIL dsymutil)
        mark_as_advanced(DSYMUTIL)
        if(DSYMUTIL)
            add_custom_command(
                TARGET ${target_name}
                POST_BUILD
                COMMAND ${DSYMUTIL} ARGS $<TARGET_FILE:${target_name}>
                COMMENT "Generating dSYM for ${target_name}"
                VERBATIM
            )
        endif()
    endif()
endfunction()
