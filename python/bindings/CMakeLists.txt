# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# nanobind Python bindings

# Collect all binding source files
set(PYPTO_BINDING_SOURCES
    bindings.cpp
    modules/error.cpp
    modules/core.cpp
    modules/logging.cpp
    modules/testing.cpp
    modules/ir.cpp
    modules/ir_builder.cpp
    modules/passes.cpp
    modules/codegen.cpp
    modules/backend.cpp
)

# Make paths absolute
list(TRANSFORM PYPTO_SOURCES PREPEND "${CMAKE_SOURCE_DIR}/")

# Make paths absolute
list(TRANSFORM PYPTO_BINDING_SOURCES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

# Export to parent scope
set(PYPTO_BINDING_SOURCES ${PYPTO_BINDING_SOURCES} PARENT_SCOPE)

# Set binding include directory
set(PYPTO_BINDING_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)

# Create the nanobind module
nanobind_add_module(${LIBRARY_NAME}
    ${PYPTO_BINDING_SOURCES}
    ${PYPTO_SOURCES}
)

# Set visibility to hidden by default (best practice for Python modules)
set_target_properties(${LIBRARY_NAME} PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# Link with libbacktrace and msgpack
target_include_directories(${LIBRARY_NAME} PRIVATE
    ${LIBBACKTRACE_INSTALL_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(${LIBRARY_NAME} PRIVATE
    libbacktrace
    msgpackc-cxx
)

# Handle installation based on build mode
if(SKBUILD_MODE)
    # scikit-build-core: install to the wheel
    install(TARGETS ${LIBRARY_NAME} DESTINATION .)
    # Install dSYM debug symbols on macOS
    if(APPLE)
        install(DIRECTORY $<TARGET_FILE:${LIBRARY_NAME}>.dSYM
            DESTINATION .
            OPTIONAL
        )
    endif()
else()
    # Native CMake: output to local python directory for development
    set_target_properties(${LIBRARY_NAME} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/pypto
    )
endif()

# Generate dSYM on macOS for libbacktrace debug symbols
pypto_add_apple_dsymutil(${LIBRARY_NAME})
