# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)

# The `ASC` language is what supplies the kernel-launch shim (`__cce_rtLaunch`)
# and the `<<<>>>` lowering. Linking the same objects with a plain C++ driver
# leaves those symbols undefined, which is why this project exists rather than a
# hand-written link line: the shim is a language driver, not a library.
find_package(ASC REQUIRED)

project(vec_add_run LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)

# dav-2201 is Atlas A2 (910B). dav-3510 is A5 (910C).
# The ASC driver's target feature. `setup.sh` has always taken an `arch`
# argument and NEVER used it: nothing passed it to cmake or to the driver, which
# went unnoticed because the hand-written AscendC artefacts compile under the
# driver's default. The pto-isa artefacts do not — every intrinsic in
# pto/npu/a2a3/*.hpp carries a target attribute, and without this the compile
# fails with "does not support the given target feature" on TLoad, TAdd,
# TStore and even set_vector_mask. The spelling is the one this bench's own
# receipts use for a direct driver invocation: `-x asc --npu-arch=dav-2201`.
set(ARCH "dav-2201" CACHE STRING "ASC target feature, e.g. dav-2201 (910B) or dav-c220 (910C)")
add_compile_options(--npu-arch=${ARCH})

add_executable(vec_add_run op_host/main.asc)

target_link_libraries(vec_add_run PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

target_include_directories(vec_add_run PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

# `KERNEL_DO` is the launcher symbol the kernel file defines; `KERNEL_N` is the
# element count it processes. Both come from the emitted file, not from here.
set(KERNEL_DO "vec_add_do" CACHE STRING "launcher symbol")
set(KERNEL_N "256" CACHE STRING "elements processed")
set(KERNEL_ARGS "3" CACHE STRING "GM arguments the launcher takes")
set(KERNEL_REPEAT "1" CACHE STRING "launches to time; 1 disables timing")
set(MV_N "64" CACHE STRING "matvec oracle: output rows")
set(MV_K "256" CACHE STRING "matvec oracle: contraction width")
# The THIRD extent. 0 means "not passed"; an oracle needing it REFUSES on 0 rather
# than assuming a square shape. Declared AND forwarded below -- setting the cache
# variable alone does nothing, which cost a run to find.
set(MV_P "0" CACHE STRING "gemm oracle: the N extent")
# 0 means DO NOT OVERRIDE: EMB_ROWS is `#ifndef`-guarded in main.asc, so defining it
# as 0 would satisfy the guard with a zero-row table. Only a nonzero value is passed.
set(EMB_ROWS_OVERRIDE "0" CACHE STRING "embedding oracle: table row count, 0 = default")
set(ORACLE "0" CACHE STRING "0 sum, 1 rmsnorm, 2 matvec")
set(RMS_EPS "1e-5f" CACHE STRING "epsilon the kernel bakes in")
set(TOLERANCE "1e-6" CACHE STRING "relative tolerance")
target_compile_definitions(vec_add_run PRIVATE
    KERNEL_DO=${KERNEL_DO}
    KERNEL_N=${KERNEL_N}
    KERNEL_ARGS=${KERNEL_ARGS}
    KERNEL_REPEAT=${KERNEL_REPEAT}
    MV_N=${MV_N}
    MV_K=${MV_K}
    MV_P=${MV_P}
    $<$<NOT:$<STREQUAL:${EMB_ROWS_OVERRIDE},0>>:EMB_ROWS=${EMB_ROWS_OVERRIDE}>
    ORACLE=${ORACLE}
    RMS_EPS=${RMS_EPS}
    TOLERANCE=${TOLERANCE}
)

target_compile_options(vec_add_run PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)
