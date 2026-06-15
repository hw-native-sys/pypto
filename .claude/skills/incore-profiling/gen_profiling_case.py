# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Generate a standalone camodel (msprof op-simulator) testcase for ONE PTOAS kernel.

This is the **profiling-focused** testcase generator for the incore-profiling skill.
It is a drop-in replacement for the legacy PTOAS validation-harness generator:
same CLI, same output files, but ~10x smaller because it does ONE job — produce a
buildable+runnable sim testcase so the camodel can time the kernel — and drops the
entire correctness-validation machinery (golden compare, ULP tolerances, scatter/
gather/mrgsort special-casing, runtime int-expression buffer-size inference).

Why it can be small: buffer sizes come straight from the sibling ``<kernel>.pto``'s
``make_tensor_view`` shape constants (static), instead of being inferred from the
compiled C++ kernel's runtime pointer arithmetic. The kernel ABI (name, arg types,
order) comes from the one ``__global__``/``_aic`` declaration line in the ``.cpp``.

It emits, at ``<output-root>/ptoas/<testcase>/``:
  - ``<testcase>_kernel.cpp`` : the input .cpp + a compat preamble (+ a merged
    ``__global__`` dispatcher for mixed cube+vector kernels).
  - ``launch.cpp``            : host->device launch shim (``<<<1, …>>>`` single core).
  - ``main.cpp``             : alloc -> read input bins -> Launch -> sync.
  - ``CMakeLists.txt``        : builds ``<testcase>_sim`` against the camodel.
  - ``golden.py``            : writes input ``vN.bin`` (zeros for ints, random for floats).

Profiling is data-independent for per-instruction cost, so input *values* are
irrelevant; only buffer sizes (no OOB) and loop trip-counts matter. Data-dependent
kernels (e.g. a flash-attention work table read from GM) still need real control
tensors wired in afterwards — see the skill's "Caveats".
"""

import argparse
import re
import sys
from pathlib import Path

# ── Type maps: .pto pointee / C++ type -> (host C type, numpy dtype) ──────────
# host type sizes the .bin and the host-side buffers; bf16/half are carried as
# raw bits on the host (uint16 / aclFloat16) since they aren't native host types.
_CPP_TO_HOST = {"bfloat16_t": "uint16_t", "__bf16": "uint16_t", "half": "aclFloat16"}
_CPP_TO_NP = {
    "int32_t": "np.int32",
    "float": "np.float32",
    "bfloat16_t": "np.uint16",
    "__bf16": "np.uint16",
    "half": "np.float16",
    "aclFloat16": "np.float16",
    "uint16_t": "np.uint16",
    "int16_t": "np.int16",
    "int8_t": "np.int8",
    "uint8_t": "np.uint8",
    "uint32_t": "np.uint32",
    "int64_t": "np.int64",
    "uint64_t": "np.uint64",
}
# Fallback sizes (elements) when a shape can't be read statically from the .pto.
_DEFAULT_DYNAMIC = 256  # for make_tensor_view shape = [%argN] (runtime-sized, e.g. seq_lens)
_DEFAULT_SCRATCH = 1 << 20  # GM pointers with no make_tensor_view (e.g. the cube<->vector pipe slot)


def host_type(cpp_type: str) -> str:
    return _CPP_TO_HOST.get(cpp_type, cpp_type)


def np_dtype(cpp_type: str) -> str:
    return _CPP_TO_NP.get(cpp_type, "np.float32")


def is_integer_np(dt: str) -> bool:
    return dt.startswith("np.int") or dt.startswith("np.uint")


# ── Parse the kernel C++ signature (the launch ABI) ──────────────────────────
class Param:
    """One kernel parameter: a GM pointer or a scalar tail arg."""

    def __init__(self, cpp_type: str, name: str, is_ptr: bool):
        self.cpp_type = cpp_type
        self.name = name
        self.is_ptr = is_ptr


def _split_params(blob: str) -> list[str]:
    return [p.strip() for p in blob.split(",") if p.strip()]


def _parse_param(text: str) -> Param:
    """Parse one decl, e.g. '__gm__ bfloat16_t* v7' or 'int32_t v11'."""
    is_ptr = "__gm__" in text and "*" in text
    if is_ptr:
        m = re.search(r"__gm__\s+([\w:]+)\s*\*\s*(\w+)", text)
        if not m:
            raise ValueError(f"cannot parse pointer param: {text!r}")
        return Param(m.group(1), m.group(2), True)
    toks = text.split()
    return Param(" ".join(toks[:-1]), toks[-1], False)


def parse_cpp(cpp_text: str) -> tuple[str, bool, list[Param]]:
    """Return (kernel_name, is_mixed, params) from the kernel .cpp.

    A *pure* kernel exposes one ``__global__ AICORE void <name>(...)``.
    A *mixed* (cube+vector) kernel exposes ``AICORE void <name>_aic(...)`` and
    ``<name>_aiv(...)`` with no merged ``__global__`` — we synthesize one.
    """
    m = re.search(r"__global__\s+AICORE\s+void\s+(\w+)\s*\(([^)]*)\)", cpp_text)
    if m:
        return m.group(1), False, [_parse_param(p) for p in _split_params(m.group(2))]
    m = re.search(r"\bAICORE\s+void\s+(\w+)_aic\s*\(([^)]*)\)", cpp_text)
    if m:
        return m.group(1), True, [_parse_param(p) for p in _split_params(m.group(2))]
    raise ValueError("no '__global__ AICORE void <name>' or '<name>_aic' decl found in kernel .cpp")


# ── Parse the sibling .pto for static buffer sizes ───────────────────────────
def _parse_dim_list(blob: str) -> list[int]:
    """Parse a .pto ``[%cN_index, %argM, ...]`` dim/stride list to ints.

    A constant ``%cN_index`` yields its value; a dynamic ``%argN`` (runtime) or
    any other non-constant token yields ``_DEFAULT_DYNAMIC``.
    """
    out: list[int] = []
    for tok in blob.split(","):
        cm = re.match(r"%c(\d+)_index", tok.strip())
        out.append(int(cm.group(1)) if cm else _DEFAULT_DYNAMIC)
    return out


def parse_pto_sizes(pto_text: str) -> dict[int, int]:
    """Map GM arg index -> element count, from ``make_tensor_view`` shape + strides.

    Allocates the true linear footprint ``1 + Σ_d (shape[d]-1)*stride[d]`` rather
    than ``prod(shape)``, so a padded/strided view (physical stride larger than
    the shape) is not under-allocated. Constant dims use their value; a dynamic
    dim (``%argN``) uses ``_DEFAULT_DYNAMIC``. Keeps the largest footprint across
    an arg's views (a safe upper bound on what the kernel touches).
    """
    sizes: dict[int, int] = {}
    pat = re.compile(
        r"make_tensor_view\s+%arg(\d+),\s*shape\s*=\s*\[([^\]]*)\]"
        r"(?:,\s*strides\s*=\s*\[([^\]]*)\])?"
    )
    for m in pat.finditer(pto_text):
        argn = int(m.group(1))
        shape = _parse_dim_list(m.group(2))
        strides = _parse_dim_list(m.group(3)) if m.group(3) else None
        if strides and len(strides) == len(shape):
            footprint = 1 + sum((s - 1) * st for s, st in zip(shape, strides))
        else:  # no/mismatched strides -> contiguous product
            footprint = 1
            for s in shape:
                footprint *= s
        sizes[argn] = max(sizes.get(argn, 0), footprint)
    return sizes


def elem_count_for(param_idx: int, pto_sizes: dict[int, int]) -> int:
    """Resolve a GM pointer's element count, with safe fallbacks."""
    if param_idx in pto_sizes:
        return pto_sizes[param_idx]
    return _DEFAULT_SCRATCH  # e.g. the cube<->vector pipe slot buffer (no make_tensor_view)


# ── Code emission ────────────────────────────────────────────────────────────
# The PTOAS compat preamble (FP8/FP4 + __VEC_SCOPE__ fallbacks) shared by the
# kernel.cpp and launch.cpp so they compile on dav-c220 / dav-c310.
_PREAMBLE = """\
// ---------------------------------------------------------------------------
// PTOAS compatibility layer: minimal FP8/FP4 + __VEC_SCOPE__ fallbacks so the
// pto-isa headers compile across AICore arch/toolchain combinations.
// ---------------------------------------------------------------------------
#ifndef __VEC_SCOPE__
#define __VEC_SCOPE__
#endif

#if defined(__CCE_AICORE__) && defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
typedef struct { unsigned char v; } hifloat8_t;
typedef struct { unsigned char v; } float8_e4m3_t;
typedef struct { unsigned char v; } float8_e5m2_t;
typedef struct { unsigned char v; } float8_e8m0_t;
typedef struct { unsigned char v; } float4_e1m2x2_t;
typedef struct { unsigned char v; } float4_e2m1x2_t;
#endif
#include <stdint.h>

#if defined(__CCE_AICORE__) && defined(PTOAS_ENABLE_CCE_PRINT)
#include <ccelib/print/print.h>
#endif
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

#if !defined(__CCE_AICORE__) && !defined(TMRGSORT_HPP)
namespace pto {
struct MrgSortExecutedNumList {
    uint16_t mrgSortList0;
    uint16_t mrgSortList1;
    uint16_t mrgSortList2;
    uint16_t mrgSortList3;
};
} // namespace pto
#endif
#ifndef __CPU_SIM
#include "acl/acl.h"
#endif
"""

_MAIN_TEMPLATE = """\
#include "test_common.h"
#include "acl/acl.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>

using namespace PtoTestCommon;

#define ACL_CHECK(expr)                                                                          \\
    do {                                                                                         \\
        const aclError _ret = (expr);                                                            \\
        if (_ret != ACL_SUCCESS) {                                                               \\
            std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\\n", #expr, (int)_ret, __FILE__, __LINE__); \\
            const char *_recent = aclGetRecentErrMsg();                                          \\
            if (_recent != nullptr && _recent[0] != '\\0') {                                      \\
                std::fprintf(stderr, "[ERROR] RecentErrMsg: %s\\n", _recent);                     \\
            }                                                                                    \\
            rc = 1;                                                                              \\
            goto cleanup;                                                                        \\
        }                                                                                        \\
    } while (0)

@LAUNCH_DECL@

int main() {
@PARAM_DECLS@

    int rc = 0;
    bool aclInited = false;
    bool deviceSet = false;
    int deviceId = 0;
    aclrtStream stream = nullptr;

    ACL_CHECK(aclInit(nullptr));
    aclInited = true;
    if (const char *envDevice = std::getenv("ACL_DEVICE_ID")) {
        deviceId = std::atoi(envDevice);
    }
    ACL_CHECK(aclrtSetDevice(deviceId));
    deviceSet = true;
    ACL_CHECK(aclrtCreateStream(&stream));

@ALLOC@
@READ_INPUTS@
@COPY_TO_DEVICE@
    @LAUNCH_CALL@

    ACL_CHECK(aclrtSynchronizeStream(stream));

cleanup:
@FREE@
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
        stream = nullptr;
    }
    if (deviceSet) {
        aclrtResetDevice(deviceId);
    }
    if (aclInited) {
        aclFinalize();
    }
    return rc;
}
"""

_CMAKE_TEMPLATE = """\
cmake_minimum_required(VERSION 3.16)
set(CMAKE_C_COMPILER bisheng)
set(CMAKE_CXX_COMPILER bisheng)
project(@TESTCASE@_incore_profiling)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT DEFINED SOC_VERSION)
    set(SOC_VERSION Ascend910)
endif()
option(ENABLE_SIM_GOLDEN "Build Ascend simulator (camodel) executable" ON)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH, please source the CANN set_env.sh.")
else()
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()

set(PTO_ISA_ROOT "" CACHE PATH "Path to pto-isa repo")
if(NOT PTO_ISA_ROOT)
    message(FATAL_ERROR "Cannot find PTO_ISA_ROOT, please pass -DPTO_ISA_ROOT=/path/to/pto-isa.")
endif()

set(ASCEND_DRIVER_PATH /usr/local/Ascend/driver)

add_compile_options(-D_FORTIFY_SOURCE=2 -O2 -std=c++17
    -Wno-macro-redefined -Wno-ignored-attributes -fstack-protector-strong -fPIC)
add_link_options(-s -Wl,-z,relro -Wl,-z,now)

set(CMAKE_CCE_COMPILE_OPTIONS
    -xcce -fenable-matrix --cce-aicore-enable-tl -fPIC
    -Xhost-start -Xhost-end
    "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-record-overflow=true"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false")
set(CMAKE_CPP_COMPILE_OPTIONS -xc++ "SHELL:-include stdint.h" "SHELL:-include stddef.h")

include_directories(${PTO_ISA_ROOT}/include ${PTO_ISA_ROOT}/tests/common
    ${ASCEND_HOME_PATH}/include ${ASCEND_DRIVER_PATH}/kernel/inc)

add_library(@TESTCASE@_kernel SHARED @TESTCASE@_kernel.cpp launch.cpp)
target_compile_options(@TESTCASE@_kernel PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS}
    --cce-aicore-arch=@AICORE_ARCH@ -DREGISTER_BASE -std=c++17)
target_include_directories(@TESTCASE@_kernel PRIVATE
    ${ASCEND_HOME_PATH}/pkg_inc/ ${ASCEND_HOME_PATH}/pkg_inc/profiling/
    ${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime)
target_link_options(@TESTCASE@_kernel PRIVATE --cce-fatobj-link)

if(ENABLE_SIM_GOLDEN)
    add_executable(@TESTCASE@_sim main.cpp)
    target_compile_options(@TESTCASE@_sim PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
    target_include_directories(@TESTCASE@_sim PRIVATE
        ${PTO_ISA_ROOT}/include ${PTO_ISA_ROOT}/tests/common)
    target_link_directories(@TESTCASE@_sim PUBLIC
        ${ASCEND_HOME_PATH}/lib64
        ${ASCEND_HOME_PATH}/aarch64-linux/simulator/${SOC_VERSION}/lib
        ${ASCEND_HOME_PATH}/simulator/${SOC_VERSION}/lib
        ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib)
    target_link_libraries(@TESTCASE@_sim PRIVATE
        @TESTCASE@_kernel runtime_camodel
        stdc++ ascendcl m tiling_api platform c_sec dl nnopbase)
endif()
"""

_GOLDEN_TEMPLATE = """\
#!/usr/bin/python3
import numpy as np

def main():
    np.random.seed(19)
@INPUT_GENERATE@


if __name__ == "__main__":
    main()
"""


def emit_kernel_cpp(cpp_text: str, name: str, is_mixed: bool, params: list[Param]) -> str:
    """Compat preamble + the original kernel + (mixed) a merged __global__ dispatcher.

    For a mixed kernel the standalone ``<name>_aic`` / ``<name>_aiv`` are
    self-contained (each builds its own GM pipe from the slot buffer), so the
    merged entry just dispatches by core type — no body inlining needed.
    """
    out = _PREAMBLE + "\n" + cpp_text
    if is_mixed:
        decl = ", ".join(
            (f"__gm__ {p.cpp_type}* {p.name}" if p.is_ptr else f"{p.cpp_type} {p.name}") for p in params
        )
        call = ", ".join(p.name for p in params)
        # extern "C" so the merged dispatcher's launch-ABI symbol matches the
        # non-mangled forward decl in launch.cpp (mirrors the ptoas pure-kernel
        # convention, which is always `extern "C" __global__`).
        out += (
            f'\n\nextern "C" __global__ AICORE void {name}({decl}) {{\n'
            f"#if defined(__DAV_CUBE__)\n  {name}_aic({call});\n#endif\n"
            f"#if defined(__DAV_VEC__)\n  {name}_aiv({call});\n#endif\n}}\n"
        )
    return out


def emit_launch_cpp(name: str, params: list[Param]) -> str:
    launch_name = "Launch" + name[:1].upper() + name[1:]
    dev_decl = ", ".join(
        (f"__gm__ {p.cpp_type}* {p.name}" if p.is_ptr else f"{p.cpp_type} {p.name}") for p in params
    )
    host_params = ", ".join(
        (f"{host_type(p.cpp_type)} *{p.name}" if p.is_ptr else f"{p.cpp_type} {p.name}") for p in params
    )
    casts = ", ".join((f"(__gm__ {p.cpp_type}*){p.name}" if p.is_ptr else p.name) for p in params)
    # extern "C" so this forward decl resolves to the kernel's unmangled symbol.
    # ptoas pure kernels are emitted as `extern "C" __global__ AICORE void <name>`
    # (and the synthesized mixed dispatcher matches via emit_kernel_cpp); without
    # extern "C" here the call mangles and fails to link (undefined reference).
    return (
        _PREAMBLE
        + f'\nextern "C" __global__ AICORE void {name}({dev_decl});\n\n'
        + f"void {launch_name}({host_params}, void *stream) {{\n"
        + f"    {name}<<<1, nullptr, stream>>>({casts});\n}}\n"
    )


def emit_main_cpp(name: str, params: list[Param], counts: dict[str, int]) -> str:
    launch_name = "Launch" + name[:1].upper() + name[1:]
    ptrs = [p for p in params if p.is_ptr]
    scalars = [p for p in params if not p.is_ptr]

    decls, alloc, reads, copy, free = [], [], [], [], []
    for p in ptrs:
        ht, n = host_type(p.cpp_type), p.name
        decls.append(f"    size_t elemCount_{n} = {counts[n]};")
        decls.append(f"    size_t fileSize_{n} = elemCount_{n} * sizeof({ht});")
        decls.append(f"    {ht} *{n}Host = nullptr;")
        decls.append(f"    {ht} *{n}Device = nullptr;")
        alloc.append(f"    ACL_CHECK(aclrtMallocHost((void **)(&{n}Host), fileSize_{n}));")
        alloc.append(
            f"    ACL_CHECK(aclrtMalloc((void **)&{n}Device, fileSize_{n}, ACL_MEM_MALLOC_HUGE_FIRST));"
        )
        reads.append(f'    ReadFile("./{n}.bin", fileSize_{n}, {n}Host, fileSize_{n});')
        copy.append(
            f"    ACL_CHECK(aclrtMemcpy({n}Device, fileSize_{n}, {n}Host, fileSize_{n}, "
            "ACL_MEMCPY_HOST_TO_DEVICE));"
        )
        free.append(f"    if ({n}Device) aclrtFree({n}Device);")
        free.append(f"    if ({n}Host) aclrtFreeHost({n}Host);")
    for p in scalars:
        decls.append(f"    {p.cpp_type} {p.name} = 1;")  # safe default (matches legacy)

    launch_args = ", ".join((f"{p.name}Device" if p.is_ptr else p.name) for p in params)
    launch_decl_params = ", ".join(
        (f"{host_type(p.cpp_type)} *{p.name}" if p.is_ptr else f"{p.cpp_type} {p.name}") for p in params
    )
    text = _MAIN_TEMPLATE
    text = text.replace("@LAUNCH_DECL@", f"void {launch_name}({launch_decl_params}, void *stream);")
    text = text.replace("@PARAM_DECLS@", "\n".join(decls))
    text = text.replace("@ALLOC@", "\n".join(alloc))
    text = text.replace("@READ_INPUTS@", "\n".join(reads))
    text = text.replace("@COPY_TO_DEVICE@", "\n".join(copy))
    text = text.replace("@LAUNCH_CALL@", f"{launch_name}({launch_args}, stream);")
    text = text.replace("@FREE@", "\n".join(free))
    return text


def emit_golden(params: list[Param], counts: dict[str, int]) -> str:
    lines = []
    for p in params:
        if not p.is_ptr:
            continue
        dt = np_dtype(p.cpp_type)
        n, size = p.name, counts[p.name]
        if is_integer_np(dt):
            lines.append(f"    {n} = np.zeros(({size},), dtype={dt})")
        else:
            lines.append(f"    {n} = np.random.random(size=({size},)).astype({dt})")
        lines.append(f'    {n}.tofile("{n}.bin")')
    return _GOLDEN_TEMPLATE.replace("@INPUT_GENERATE@", "\n".join(lines))


def generate(input_cpp: Path, testcase: str, output_root: Path, aicore_arch: str) -> Path:
    cpp_text = input_cpp.read_text(encoding="utf-8")
    pto_path = input_cpp.with_suffix(".pto")
    if not pto_path.is_file():
        raise FileNotFoundError(
            f"sibling .pto not found next to the kernel: {pto_path}. "
            "The .pto carries the static tensor shapes used for buffer sizing."
        )
    name, is_mixed, params = parse_cpp(cpp_text)
    pto_sizes = parse_pto_sizes(pto_path.read_text(encoding="utf-8"))

    counts: dict[str, int] = {}
    for i, p in enumerate(params):
        if p.is_ptr:
            counts[p.name] = elem_count_for(i, pto_sizes)

    out_dir = output_root / "ptoas" / testcase
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{testcase}_kernel.cpp").write_text(
        emit_kernel_cpp(cpp_text, name, is_mixed, params), encoding="utf-8"
    )
    (out_dir / "launch.cpp").write_text(emit_launch_cpp(name, params), encoding="utf-8")
    (out_dir / "main.cpp").write_text(emit_main_cpp(name, params, counts), encoding="utf-8")
    (out_dir / "CMakeLists.txt").write_text(
        _CMAKE_TEMPLATE.replace("@TESTCASE@", testcase).replace("@AICORE_ARCH@", aicore_arch),
        encoding="utf-8",
    )
    (out_dir / "golden.py").write_text(emit_golden(params, counts), encoding="utf-8")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate a camodel profiling testcase from a .cpp + .pto")
    ap.add_argument("--input", required=True, help="PTOAS kernel .cpp (.pto sibling read for buffer sizes)")
    ap.add_argument("--testcase", required=True, help="Testcase name, e.g. <func>_msprof")
    ap.add_argument("--output-root", required=True, help="Root dir; case -> <root>/ptoas/<testcase>/")
    ap.add_argument("--run-mode", default="sim", choices=["sim", "npu"], help="CLI compat (sim only)")
    ap.add_argument("--soc-version", default="Ascend910B1", help="CLI compat (cmake -DSOC_VERSION)")
    ap.add_argument("--aicore-arch", default="dav-c220", help="--cce-aicore-arch (a2a3 / a5)")
    args = ap.parse_args(argv)

    arch = args.aicore_arch or "dav-c220"
    out_dir = generate(Path(args.input), args.testcase, Path(args.output_root), arch)
    print(f"[gen_profiling_case] wrote testcase -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
