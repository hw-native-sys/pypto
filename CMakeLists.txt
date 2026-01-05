# Copyright (c) Qijizhifeng. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

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

# PyPTO core sources
set(PYPTO_SOURCES
    src/core/backtrace.cpp
    src/core/error.cpp
    src/ir/core.cpp
    src/ir/transform/visitor.cpp
    src/ir/transform/mutator.cpp
    src/ir/transform/printer.cpp
    src/ir/transform/structural_hash.cpp
    src/ir/transform/structural_equal.cpp
)

# Include Python bindings
add_subdirectory(python/pybind)
