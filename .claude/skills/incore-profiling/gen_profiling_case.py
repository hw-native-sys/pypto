# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Generate a standalone simulator or real-NPU testcase for one PTOAS kernel.

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

Simulator mode sizes buffers from the sibling ``.pto`` and uses synthetic data.
NPU mode accepts deterministic synthetic ABI inputs, caller-supplied input
files, or one exact pure-kernel invocation reconstructed from PyPTO's level-2
argument dump. Synthetic inputs avoid full-model DFX capture for large kernels;
exact scalar ABI values remain mandatory.

It emits, at ``<output-root>/ptoas/<testcase>/``:
  - ``<testcase>_kernel.cpp`` : the input .cpp + a compat preamble (+ a merged
    ``__global__`` dispatcher for mixed cube+vector kernels).
  - ``launch.cpp``            : host launch shim (sim defaults to one core;
    NPU uses the captured block count).
  - ``main.cpp``             : simulator launch or ACL event-timed NPU loop.
  - ``CMakeLists.txt``        : builds ``<testcase>_sim`` and/or ``*_npu``.
  - ``golden.py``            : writes input ``vN.bin`` (zeros for ints, random for floats).

Profiling is data-independent for per-instruction cost, so input *values* are
irrelevant; only buffer sizes (no OOB) and loop trip-counts matter. Data-dependent
kernels (e.g. a flash-attention work table read from GM) still need real control
tensors wired in afterwards — see the skill's "Caveats".
"""

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np

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
_CPP_BYTE_SIZES = {
    "int32_t": 4,
    "float": 4,
    "bfloat16_t": 2,
    "__bf16": 2,
    "half": 2,
    "aclFloat16": 2,
    "uint16_t": 2,
    "int16_t": 2,
    "int8_t": 1,
    "uint8_t": 1,
    "uint32_t": 4,
    "int64_t": 8,
    "uint64_t": 8,
}
_SYNTHETIC_INTEGER_DTYPES = {
    "int8_t": np.int8,
    "uint8_t": np.uint8,
    "int16_t": np.int16,
    "uint16_t": np.uint16,
    "int32_t": np.int32,
    "uint32_t": np.uint32,
    "int64_t": np.int64,
    "uint64_t": np.uint64,
}
_SYNTHETIC_SEED = 19
_SYNTHETIC_CHUNK_ELEMENTS = 1 << 20


def host_type(cpp_type: str) -> str:
    return _CPP_TO_HOST.get(cpp_type, cpp_type)


def np_dtype(cpp_type: str) -> str:
    return _CPP_TO_NP.get(cpp_type, "np.float32")


def is_integer_np(dt: str) -> bool:
    return dt.startswith("np.int") or dt.startswith("np.uint")


def byte_size(cpp_type: str) -> int:
    """Return the host representation size for one kernel ABI element."""
    try:
        return _CPP_BYTE_SIZES[cpp_type]
    except KeyError as exc:
        raise ValueError(
            f"unsupported pointer element type {cpp_type!r}; add its byte size before importing real inputs"
        ) from exc


# ── Parse the kernel C++ signature (the launch ABI) ──────────────────────────
class Param:
    """One kernel parameter: a GM pointer or a scalar tail arg."""

    def __init__(self, cpp_type: str, name: str, is_ptr: bool):
        self.cpp_type = cpp_type
        self.name = name
        self.is_ptr = is_ptr


@dataclass(frozen=True)
class DumpSelection:
    """One pure-kernel dispatch to reconstruct from an argument dump."""

    manifest: Path
    func_id: int
    task_id: str | None = None
    task_occurrence: int | None = None


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

    A *pure* kernel exposes one ``__global__ AICORE void <name>(...)`` on older
    PTOAS, or a bare ``AICORE void <name>(...)`` body on newer PTOAS.
    A *mixed* (cube+vector) kernel exposes ``AICORE void <name>_aic(...)`` and
    ``<name>_aiv(...)`` with no merged ``__global__`` — we synthesize one.
    """
    m = re.search(r"__global__\s+AICORE\s+void\s+(\w+)\s*\(([^)]*)\)", cpp_text)
    if m:
        return m.group(1), False, [_parse_param(p) for p in _split_params(m.group(2))]
    m = re.search(r"\bAICORE\s+void\s+(\w+)_aic\s*\(([^)]*)\)", cpp_text)
    if m:
        return m.group(1), True, [_parse_param(p) for p in _split_params(m.group(2))]
    m = re.search(
        r'(?<!static\s)(?<!inline\s)(?:extern\s+"C"\s+)?AICORE\s+void\s+(\w+)\s*\(([^)]*)\)',
        cpp_text,
    )
    if m and not m.group(1).endswith(("_aic", "_aiv")):
        return m.group(1), False, [_parse_param(p) for p in _split_params(m.group(2))]
    raise ValueError(
        "no '__global__ AICORE void <name>', bare 'AICORE void <name>', "
        "or '<name>_aic' decl found in kernel .cpp"
    )


def _load_ptoas_generator(ptoas_root: Path) -> tuple[ModuleType, Path, str]:
    """Load PTOAS's canonical validation generator for mixed-kernel wrapping."""
    script = ptoas_root / "test" / "npu_validation" / "scripts" / "generate_testcase.py"
    if not script.is_file():
        raise FileNotFoundError(f"PTOAS mixed-kernel generator not found: {script}")
    spec = importlib.util.spec_from_file_location("_pypto_ptoas_validation_generator", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load PTOAS mixed-kernel generator: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for symbol in ("_describe_kernel_source", "_append_mixed_kernel_wrapper"):
        if not callable(getattr(module, symbol, None)):
            raise RuntimeError(f"PTOAS generator lacks required mixed-kernel helper {symbol}: {script}")
    digest = hashlib.sha256(script.read_bytes()).hexdigest()
    return module, script, digest


def _prepare_mixed_group(cpp_text: str, ptoas_root: Path) -> tuple[str, list[Param], str, dict[str, str]]:
    """Create the same mixed AIC/AIV group wrapper used by PTOAS validation."""
    module, script, digest = _load_ptoas_generator(ptoas_root)
    info = module._describe_kernel_source(cpp_text)
    if info.get("kind") != "mixed":
        raise ValueError(f"PTOAS did not classify the input as a mixed AIC/AIV kernel: {script}")
    raw_params = info.get("raw_params")
    if not isinstance(raw_params, list) or not raw_params:
        raise ValueError(f"PTOAS mixed-kernel description has no launch ABI: {script}")
    name = info["kernel_name"]
    wrapped = module._append_mixed_kernel_wrapper(
        cpp_text,
        name,
        raw_params,
        info["aic_text"],
        info["aiv_text"],
    )
    if not re.search(rf'extern\s+"C"\s+__global__\s+AICORE\s+void\s+{re.escape(name)}\s*\(', wrapped):
        raise RuntimeError("PTOAS mixed-kernel wrapper did not emit the expected global entry")
    provenance = {
        "kind": "ptoas_validation_group_wrapper",
        "generator": str(script.resolve()),
        "generator_sha256": digest,
    }
    return name, [_parse_param(param) for param in raw_params], wrapped, provenance


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

_NPU_BENCHMARK_MAIN_TEMPLATE = """\
#include "test_common.h"
#include "acl/acl.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

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

static int PositiveEnv(const char *name, int defaultValue) {
    const char *raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\\0') return defaultValue;
    const int value = std::atoi(raw);
    if (value <= 0) {
        std::fprintf(stderr, "[ERROR] %s must be positive, got %s\\n", name, raw);
        return -1;
    }
    return value;
}

static bool WriteBinary(const std::string &path, const void *data, size_t size) {
    std::ofstream output(path, std::ios::binary);
    if (!output) return false;
    output.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
    return output.good();
}

int main() {
@PARAM_DECLS@

    int rc = 0;
    bool aclInited = false;
    bool deviceSet = false;
    int deviceId = 0;
    aclrtStream stream = nullptr;
    const int warmup = PositiveEnv("PYPTO_BENCH_WARMUP", 10);
    const int rounds = PositiveEnv("PYPTO_BENCH_ROUNDS", 100);
    const char *timingPath = std::getenv("PYPTO_BENCH_OUTPUT");
    const char *dumpDir = std::getenv("PYPTO_BENCH_DUMP_DIR");
    std::vector<aclrtEvent> startEvents;
    std::vector<aclrtEvent> endEvents;
    std::vector<float> elapsedUs;

    if (warmup <= 0 || rounds <= 0) return 2;
    startEvents.assign(static_cast<size_t>(rounds), nullptr);
    endEvents.assign(static_cast<size_t>(rounds), nullptr);
    elapsedUs.assign(static_cast<size_t>(rounds), 0.0F);

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

    for (int i = 0; i < warmup; ++i) {
@COPY_TO_DEVICE@
        @LAUNCH_CALL@
        ACL_CHECK(aclrtSynchronizeStream(stream));
    }

    for (int i = 0; i < rounds; ++i) {
@COPY_TO_DEVICE@
        ACL_CHECK(aclrtCreateEvent(&startEvents[static_cast<size_t>(i)]));
        ACL_CHECK(aclrtCreateEvent(&endEvents[static_cast<size_t>(i)]));
        ACL_CHECK(aclrtRecordEvent(startEvents[static_cast<size_t>(i)], stream));
        @LAUNCH_CALL@
        ACL_CHECK(aclrtRecordEvent(endEvents[static_cast<size_t>(i)], stream));
        ACL_CHECK(aclrtSynchronizeEvent(endEvents[static_cast<size_t>(i)]));
        float elapsedMs = 0.0F;
        ACL_CHECK(aclrtEventElapsedTime(&elapsedMs, startEvents[static_cast<size_t>(i)],
                                       endEvents[static_cast<size_t>(i)]));
        elapsedUs[static_cast<size_t>(i)] = elapsedMs * 1000.0F;
    }

@COPY_FROM_DEVICE@

    {
        const std::string outputPath = timingPath != nullptr ? timingPath : "timings.tsv";
        std::ofstream timing(outputPath);
        if (!timing) {
            std::fprintf(stderr, "[ERROR] cannot open timing output %s\\n", outputPath.c_str());
            rc = 1;
            goto cleanup;
        }
        timing << "sample\\telapsed_us\\n";
        for (size_t i = 0; i < elapsedUs.size(); ++i) timing << i << '\\t' << elapsedUs[i] << '\\n';
    }

    if (dumpDir != nullptr && dumpDir[0] != '\\0') {
@WRITE_OUTPUTS@
    }

cleanup:
    for (aclrtEvent event : startEvents) {
        if (event != nullptr) aclrtDestroyEvent(event);
    }
    for (aclrtEvent event : endEvents) {
        if (event != nullptr) aclrtDestroyEvent(event);
    }
@FREE@
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
        stream = nullptr;
    }
    if (deviceSet) aclrtResetDevice(deviceId);
    if (aclInited) aclFinalize();
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
option(ENABLE_SIM_GOLDEN "Build Ascend simulator (camodel) executable" @SIM_DEFAULT@)
option(ENABLE_NPU_BENCHMARK "Build real-device standalone benchmark executable" @NPU_DEFAULT@)

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

if(ENABLE_NPU_BENCHMARK)
    add_executable(@TESTCASE@_npu main.cpp)
    target_compile_options(@TESTCASE@_npu PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
    target_include_directories(@TESTCASE@_npu PRIVATE
        ${PTO_ISA_ROOT}/include ${PTO_ISA_ROOT}/tests/common)
    target_link_directories(@TESTCASE@_npu PUBLIC ${ASCEND_HOME_PATH}/lib64)
    target_link_libraries(@TESTCASE@_npu PRIVATE
        @TESTCASE@_kernel stdc++ ascendcl m c_sec dl pthread)
    set_target_properties(@TESTCASE@_npu PROPERTIES BUILD_RPATH "$ORIGIN")
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


def emit_kernel_cpp(
    cpp_text: str,
    name: str,
    is_mixed: bool,
    params: list[Param],
    *,
    mixed_group_wrapped: bool = False,
) -> str:
    """Compat preamble + the original kernel + (mixed) a merged __global__ dispatcher.

    For a mixed kernel the standalone ``<name>_aic`` / ``<name>_aiv`` are
    self-contained (each builds its own GM pipe from the slot buffer), so the
    merged entry just dispatches by core type — no body inlining needed.
    """
    decl = ", ".join(
        (f"__gm__ {p.cpp_type}* {p.name}" if p.is_ptr else f"{p.cpp_type} {p.name}") for p in params
    )
    call = ", ".join(p.name for p in params)

    has_global_entry = re.search(rf"__global__\s+AICORE\s+void\s+{re.escape(name)}\s*\(", cpp_text)
    if not is_mixed and not has_global_entry:
        impl_name = f"{name}_impl"
        cpp_text = re.sub(
            rf'(?:extern\s+"C"\s+)?AICORE\s+void\s+{re.escape(name)}\s*\(',
            f"static AICORE void {impl_name}(",
            cpp_text,
        )

    out = _PREAMBLE + "\n" + cpp_text
    if not is_mixed and not has_global_entry:
        out += f'\n\nextern "C" __global__ AICORE void {name}({decl}) {{\n  {name}_impl({call});\n}}\n'
    if is_mixed and not mixed_group_wrapped:
        # The AIV side of a mixed kernel may take extra trailing scalar args
        # beyond the AIC launch ABI (e.g. block-partition offsets the runtime
        # derives per AIV subblock). The synthesized single-core dispatcher has
        # no such runtime, so we pass 0 for each extra arg — that profiles the
        # first partition, and per-instruction cost is partition-independent.
        aiv_call = call
        m_aiv = re.search(rf"\bAICORE\s+void\s+{re.escape(name)}_aiv\s*\(([^)]*)\)", cpp_text)
        if m_aiv:
            n_extra = len(_split_params(m_aiv.group(1))) - len(params)
            if n_extra > 0:
                extra_args = ", ".join(["0"] * n_extra)
                aiv_call = f"{call}, {extra_args}" if call else extra_args
        # extern "C" so the merged dispatcher's launch-ABI symbol matches the
        # non-mangled forward decl in launch.cpp (mirrors the ptoas pure-kernel
        # convention, which is always `extern "C" __global__`).
        out += (
            f'\n\nextern "C" __global__ AICORE void {name}({decl}) {{\n'
            f"#if defined(__DAV_CUBE__)\n  {name}_aic({call});\n#endif\n"
            f"#if defined(__DAV_VEC__)\n  {name}_aiv({aiv_call});\n#endif\n}}\n"
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
        + f"void {launch_name}({host_params}, void *stream, uint32_t blockDim) {{\n"
        + f"    {name}<<<blockDim, nullptr, stream>>>({casts});\n}}\n"
    )


def _scalar_literal(param: Param, scalar_values: dict[str, str]) -> str:
    value = scalar_values.get(param.name, "1")
    if not re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", value):
        raise ValueError(f"scalar value for {param.name!r} must be a numeric literal, got {value!r}")
    return f"static_cast<{param.cpp_type}>({value})"


def emit_main_cpp(
    name: str,
    params: list[Param],
    counts: dict[str, int],
    scalar_values: dict[str, str],
    block_dim: int,
) -> str:
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
        decls.append(f"    {p.cpp_type} {p.name} = {_scalar_literal(p, scalar_values)};")

    launch_args = ", ".join((f"{p.name}Device" if p.is_ptr else p.name) for p in params)
    launch_decl_params = ", ".join(
        (f"{host_type(p.cpp_type)} *{p.name}" if p.is_ptr else f"{p.cpp_type} {p.name}") for p in params
    )
    text = _MAIN_TEMPLATE
    text = text.replace(
        "@LAUNCH_DECL@", f"void {launch_name}({launch_decl_params}, void *stream, uint32_t blockDim);"
    )
    text = text.replace("@PARAM_DECLS@", "\n".join(decls))
    text = text.replace("@ALLOC@", "\n".join(alloc))
    text = text.replace("@READ_INPUTS@", "\n".join(reads))
    text = text.replace("@COPY_TO_DEVICE@", "\n".join(copy))
    text = text.replace("@LAUNCH_CALL@", f"{launch_name}({launch_args}, stream, {block_dim});")
    text = text.replace("@FREE@", "\n".join(free))
    return text


def emit_npu_benchmark_main_cpp(
    name: str,
    params: list[Param],
    counts: dict[str, int],
    scalar_values: dict[str, str],
    block_dim: int,
) -> str:
    """Emit a real-device runner that records one device-event duration per launch."""
    launch_name = "Launch" + name[:1].upper() + name[1:]
    ptrs = [p for p in params if p.is_ptr]
    scalars = [p for p in params if not p.is_ptr]

    decls: list[str] = []
    alloc: list[str] = []
    reads: list[str] = []
    copy_to_device: list[str] = []
    copy_from_device: list[str] = []
    writes: list[str] = []
    free: list[str] = []
    for p in ptrs:
        host = host_type(p.cpp_type)
        name_part = p.name
        decls.extend(
            [
                f"    size_t elemCount_{name_part} = {counts[name_part]};",
                f"    size_t fileSize_{name_part} = elemCount_{name_part} * sizeof({host});",
                f"    {host} *{name_part}Host = nullptr;",
                f"    {host} *{name_part}Device = nullptr;",
            ]
        )
        alloc.append(f"    ACL_CHECK(aclrtMallocHost((void **)(&{name_part}Host), fileSize_{name_part}));")
        alloc.append(
            f"    ACL_CHECK(aclrtMalloc((void **)&{name_part}Device, fileSize_{name_part}, "
            "ACL_MEM_MALLOC_HUGE_FIRST));"
        )
        reads.append(
            f'    ReadFile("./{name_part}.bin", fileSize_{name_part}, {name_part}Host, fileSize_{name_part});'
        )
        copy_to_device.append(
            f"        ACL_CHECK(aclrtMemcpy({name_part}Device, fileSize_{name_part}, {name_part}Host, "
            f"fileSize_{name_part}, ACL_MEMCPY_HOST_TO_DEVICE));"
        )
        copy_from_device.append(
            f"    ACL_CHECK(aclrtMemcpy({name_part}Host, fileSize_{name_part}, {name_part}Device, "
            f"fileSize_{name_part}, ACL_MEMCPY_DEVICE_TO_HOST));"
        )
        writes.append(
            f'        if (!WriteBinary(std::string(dumpDir) + "/{name_part}.bin", '
            f"{name_part}Host, fileSize_{name_part})) {{"
        )
        writes.append(f'            std::fprintf(stderr, "[ERROR] cannot write output {name_part}.bin\\n");')
        writes.extend(["            rc = 1;", "            goto cleanup;", "        }"])
        free.append(f"    if ({name_part}Device) aclrtFree({name_part}Device);")
        free.append(f"    if ({name_part}Host) aclrtFreeHost({name_part}Host);")
    for p in scalars:
        decls.append(f"    {p.cpp_type} {p.name} = {_scalar_literal(p, scalar_values)};")

    launch_args = ", ".join((f"{p.name}Device" if p.is_ptr else p.name) for p in params)
    launch_decl_params = ", ".join(
        (f"{host_type(p.cpp_type)} *{p.name}" if p.is_ptr else f"{p.cpp_type} {p.name}") for p in params
    )
    launch_call = f"{launch_name}({launch_args}, stream, {block_dim});"
    text = _NPU_BENCHMARK_MAIN_TEMPLATE
    text = text.replace(
        "@LAUNCH_DECL@", f"void {launch_name}({launch_decl_params}, void *stream, uint32_t blockDim);"
    )
    text = text.replace("@PARAM_DECLS@", "\n".join(decls))
    text = text.replace("@ALLOC@", "\n".join(alloc))
    text = text.replace("@READ_INPUTS@", "\n".join(reads))
    text = text.replace("@COPY_TO_DEVICE@", "\n".join(copy_to_device))
    text = text.replace("@LAUNCH_CALL@", launch_call)
    text = text.replace("@COPY_FROM_DEVICE@", "\n".join(copy_from_device))
    text = text.replace("@WRITE_OUTPUTS@", "\n".join(writes))
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


def _synthetic_chunk(cpp_type: str, count: int, rng: np.random.Generator) -> np.ndarray:
    """Create bounded deterministic data represented in the kernel ABI type."""
    if cpp_type in {"bfloat16_t", "__bf16"}:
        fp32 = rng.uniform(-0.125, 0.125, size=count).astype(np.float32)
        return (fp32.view(np.uint32) >> 16).astype(np.uint16)
    if cpp_type in {"half", "aclFloat16"}:
        return rng.uniform(-0.125, 0.125, size=count).astype(np.float16)
    if cpp_type == "float":
        return rng.uniform(-0.125, 0.125, size=count).astype(np.float32)
    dtype = _SYNTHETIC_INTEGER_DTYPES.get(cpp_type)
    if dtype is not None:
        # Zero is the safest generic value for index/control tensors: it avoids
        # manufacturing out-of-range dynamic addresses while still exercising
        # the exact kernel control flow selected by explicit scalar arguments.
        return np.zeros(count, dtype=dtype)
    raise ValueError(f"unsupported synthetic-input ABI type {cpp_type!r}")


def _write_synthetic_inputs(
    output_dir: Path,
    params: list[Param],
    counts: dict[str, int],
    *,
    seed: int,
) -> None:
    """Write deterministic finite ABI inputs without retaining large tensors in RAM."""
    rng = np.random.default_rng(seed)
    for param in params:
        if not param.is_ptr:
            continue
        remaining = counts[param.name]
        path = output_dir / f"{param.name}.bin"
        with path.open("wb") as stream:
            while remaining:
                count = min(remaining, _SYNTHETIC_CHUNK_ELEMENTS)
                stream.write(_synthetic_chunk(param.cpp_type, count, rng).tobytes())
                remaining -= count


def _copy_real_inputs(
    input_dir: Path,
    output_dir: Path,
    params: list[Param],
    counts: dict[str, int],
) -> None:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"real-input directory does not exist: {input_dir}")
    for param in params:
        if not param.is_ptr:
            continue
        source = input_dir / f"{param.name}.bin"
        if not source.is_file():
            raise FileNotFoundError(f"real-input directory is missing ABI buffer {source.name}: {input_dir}")
        item_size = byte_size(param.cpp_type)
        file_size = source.stat().st_size
        if file_size == 0 or file_size % item_size != 0:
            raise ValueError(
                f"input {source} has {file_size} bytes, which is not a positive multiple of "
                f"the {item_size}-byte ABI type {param.cpp_type}"
            )
        counts[param.name] = file_size // item_size
        shutil.copy2(source, output_dir / source.name)


def _parse_scalar_assignments(assignments: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for assignment in assignments:
        name, separator, value = assignment.partition("=")
        if not separator or not name or not value:
            raise ValueError(f"scalar assignment must have the form NAME=VALUE, got {assignment!r}")
        if not re.fullmatch(r"[A-Za-z_]\w*", name):
            raise ValueError(f"invalid scalar parameter name {name!r}")
        if name in values:
            raise ValueError(f"scalar parameter {name!r} was specified more than once")
        values[name] = value
    return values


def _validate_input_source(
    *,
    run_mode: str,
    input_dir: Path | None,
    dump_selection: DumpSelection | None,
    synthetic_seed: int | None,
) -> None:
    """Validate the mutually-exclusive standalone input modes."""
    synthetic_inputs = synthetic_seed is not None
    input_sources = sum((input_dir is not None, dump_selection is not None, synthetic_inputs))
    if input_sources > 1:
        raise ValueError("input_dir, args_dump, and synthetic_inputs are mutually exclusive")
    if synthetic_inputs and run_mode != "npu":
        raise ValueError("synthetic_inputs is only supported for real-NPU standalone cases")
    if synthetic_seed is not None and synthetic_seed < 0:
        raise ValueError(f"synthetic_seed must be nonnegative, got {synthetic_seed}")
    if run_mode == "npu" and input_sources == 0:
        raise ValueError(
            "real-NPU standalone cases require one input source: input_dir, args_dump, or synthetic_inputs"
        )


def _select_dump_task(
    entries: list[dict],
    func_id: int,
    *,
    task_id: str | None,
    task_occurrence: int | None,
) -> tuple[str, list[dict]]:
    """Select one pure-kernel dispatch from an args-dump manifest."""
    matching = [entry for entry in entries if entry.get("func_id") == [func_id]]
    task_ids = sorted({str(entry.get("task_id")) for entry in matching})
    if task_id is not None:
        if task_id not in task_ids:
            raise ValueError(f"args dump has no pure func_id={func_id} dispatch with task_id={task_id}")
        selected = task_id
    elif task_occurrence is not None:
        if task_occurrence < 0 or task_occurrence >= len(task_ids):
            raise ValueError(
                f"task occurrence {task_occurrence} is outside the {len(task_ids)} "
                f"pure func_id={func_id} dispatches"
            )
        selected = task_ids[task_occurrence]
    elif len(task_ids) == 1:
        selected = task_ids[0]
    else:
        raise ValueError(
            f"func_id={func_id} has {len(task_ids)} pure dispatches; select one with "
            "--task-id or --task-occurrence"
        )
    return selected, [entry for entry in matching if entry.get("task_id") == selected]


def _usable_dump_entry(entry: dict, *, context: str) -> None:
    if entry.get("truncated"):
        raise ValueError(f"{context} is truncated")
    if entry.get("overwritten"):
        raise ValueError(f"{context} was overwritten in the dump ring")
    if entry.get("arg_index_ambiguous"):
        raise ValueError(f"{context} has an ambiguous ABI argument index")
    if not entry.get("is_contiguous"):
        raise ValueError(f"{context} is non-contiguous; standalone reconstruction would change its layout")


def _dump_entry(entries: list[dict], arg_index: int, *, kind: str, stage: str) -> dict | None:
    matches = [
        entry
        for entry in entries
        if entry.get("arg_index") == arg_index and entry.get("kind") == kind and entry.get("stage") == stage
    ]
    if len(matches) > 1:
        raise ValueError(f"selected dispatch has duplicate {stage} {kind} records for ABI arg {arg_index}")
    return matches[0] if matches else None


def _payload_slice(payload: bytes, entry: dict, *, context: str, expected_size: int | None = None) -> bytes:
    _usable_dump_entry(entry, context=context)
    begin = int(entry.get("bin_offset", -1))
    size = int(entry.get("bin_size", -1))
    if begin < 0 or size <= 0 or begin + size > len(payload):
        raise ValueError(f"{context} references an invalid payload slice")
    if expected_size is not None and size != expected_size:
        raise ValueError(f"{context} has {size} bytes, expected {expected_size}")
    return payload[begin : begin + size]


def _extract_dump_tensor(
    task_entries: list[dict],
    payload: bytes,
    out_dir: Path,
    param: Param,
    arg_index: int,
    task_id: str,
) -> tuple[int, str]:
    before = _dump_entry(task_entries, arg_index, kind="tensor", stage="before_dispatch")
    after = _dump_entry(task_entries, arg_index, kind="tensor", stage="after_completion")
    record = before or after
    if record is None:
        raise ValueError(f"args dump is missing tensor ABI arg {arg_index} ({param.name})")
    context = f"task {task_id} tensor arg {arg_index} ({param.name})"
    _usable_dump_entry(record, context=context)
    role = str(record.get("role"))
    if role not in {"input", "output", "inout"}:
        raise ValueError(f"{context} has invalid role {role!r}")
    if role in {"input", "inout"} and before is None:
        raise ValueError(f"{context} has no before_dispatch payload")
    if before is None:
        raw = bytes(int(record.get("bin_size", 0)))
    else:
        raw = _payload_slice(payload, before, context=context)
    item_size = byte_size(param.cpp_type)
    if not raw or len(raw) % item_size:
        raise ValueError(
            f"{context} has {len(raw)} bytes, incompatible with {param.cpp_type} ({item_size} bytes)"
        )
    (out_dir / f"{param.name}.bin").write_bytes(raw)

    expected_dir = out_dir / "captured_expected"
    expected_dir.mkdir(exist_ok=True)
    if role == "input":
        # A read-only ABI input must remain unchanged after the standalone call.
        (expected_dir / f"{param.name}.bin").write_bytes(raw)
    else:
        if after is None:
            raise ValueError(f"{context} has no after_completion payload for correctness checking")
        expected = _payload_slice(payload, after, context=context, expected_size=len(raw))
        (expected_dir / f"{param.name}.bin").write_bytes(expected)
    return len(raw) // item_size, role


def _extract_dump_scalar(task_entries: list[dict], param: Param, arg_index: int) -> str:
    scalar = _dump_entry(task_entries, arg_index, kind="scalar", stage="before_dispatch")
    if scalar is None or "value" not in scalar:
        raise ValueError(f"args dump is missing scalar ABI arg {arg_index} ({param.name})")
    value = scalar["value"]
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return repr(value)
    raise ValueError(f"scalar ABI arg {arg_index} ({param.name}) has non-numeric value {value!r}")


def _extract_dump_invocation(
    selection: DumpSelection,
    out_dir: Path,
    params: list[Param],
    counts: dict[str, int],
) -> tuple[dict[str, str], dict]:
    """Materialize one exact pure-kernel invocation from a level-2 args dump."""
    manifest_path = selection.manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("args")
    bin_name = manifest.get("bin_file")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"args dump has no entries: {manifest_path}")
    if not isinstance(bin_name, str) or not bin_name:
        raise ValueError("args dump has no payload file; capture with enable_dump_args=2")
    bin_path = manifest_path.parent / bin_name
    if not bin_path.is_file():
        raise FileNotFoundError(f"args-dump payload does not exist: {bin_path}")

    selected_task, task_entries = _select_dump_task(
        entries,
        selection.func_id,
        task_id=selection.task_id,
        task_occurrence=selection.task_occurrence,
    )
    payload = bin_path.read_bytes()
    scalar_values: dict[str, str] = {}
    roles: dict[str, str] = {}

    for arg_index, param in enumerate(params):
        if param.is_ptr:
            counts[param.name], role = _extract_dump_tensor(
                task_entries,
                payload,
                out_dir,
                param,
                arg_index,
                selected_task,
            )
            roles[param.name] = role
        else:
            scalar_values[param.name] = _extract_dump_scalar(task_entries, param, arg_index)

    capture = {
        "task_id": selected_task,
        "func_id": selection.func_id,
        "roles": roles,
        "recommended_outputs": sorted(name for name, role in roles.items() if role in {"output", "inout"}),
    }
    return scalar_values, capture


def generate(  # noqa: PLR0913
    input_cpp: Path,
    testcase: str,
    output_root: Path,
    aicore_arch: str,
    *,
    run_mode: str = "sim",
    block_dim: int = 1,
    input_dir: Path | None = None,
    scalar_values: dict[str, str] | None = None,
    dump_selection: DumpSelection | None = None,
    synthetic_seed: int | None = None,
    ptoas_root: Path | None = None,
) -> Path:
    if run_mode not in {"sim", "npu"}:
        raise ValueError(f"run_mode must be 'sim' or 'npu', got {run_mode!r}")
    if block_dim <= 0:
        raise ValueError(f"block_dim must be positive, got {block_dim}")
    cpp_text = input_cpp.read_text(encoding="utf-8")
    pto_path = input_cpp.with_suffix(".pto")
    if not pto_path.is_file():
        raise FileNotFoundError(
            f"sibling .pto not found next to the kernel: {pto_path}. "
            "The .pto carries the static tensor shapes used for buffer sizing."
        )
    name, is_mixed, params = parse_cpp(cpp_text)
    mixed_wrapper: dict[str, str] | None = None
    mixed_group_wrapped = False
    if run_mode == "npu" and is_mixed:
        if ptoas_root is None:
            raise ValueError(
                "real-device mixed AIC/AIV timing requires --ptoas-root so the canonical PTOAS "
                "group wrapper is used"
            )
        name, params, cpp_text, mixed_wrapper = _prepare_mixed_group(cpp_text, ptoas_root)
        mixed_group_wrapped = True
    pto_sizes = parse_pto_sizes(pto_path.read_text(encoding="utf-8"))

    counts: dict[str, int] = {}
    for i, p in enumerate(params):
        if p.is_ptr:
            counts[p.name] = elem_count_for(i, pto_sizes)

    out_dir = output_root / "ptoas" / testcase
    out_dir.mkdir(parents=True, exist_ok=True)
    _validate_input_source(
        run_mode=run_mode,
        input_dir=input_dir,
        dump_selection=dump_selection,
        synthetic_seed=synthetic_seed,
    )
    capture: dict | None = None
    captured_scalars: dict[str, str] = {}
    if dump_selection is not None:
        captured_scalars, capture = _extract_dump_invocation(
            dump_selection,
            out_dir,
            params,
            counts,
        )
    if input_dir is not None:
        _copy_real_inputs(input_dir, out_dir, params, counts)
    scalar_values = scalar_values or {}
    conflicting_scalars = {
        name
        for name in set(scalar_values) & set(captured_scalars)
        if scalar_values[name] != captured_scalars[name]
    }
    if conflicting_scalars:
        raise ValueError(
            f"explicit scalar values disagree with the captured invocation: {sorted(conflicting_scalars)}"
        )
    scalar_values = {**captured_scalars, **scalar_values}
    scalar_names = {param.name for param in params if not param.is_ptr}
    unknown_scalars = sorted(set(scalar_values) - scalar_names)
    if unknown_scalars:
        raise ValueError(f"scalar values name parameters absent from the kernel ABI: {unknown_scalars}")
    missing_scalars = sorted(scalar_names - set(scalar_values))
    if run_mode == "npu" and missing_scalars:
        raise ValueError(
            "real-NPU standalone cases require every scalar ABI argument explicitly; "
            f"missing: {missing_scalars}"
        )
    if synthetic_seed is not None:
        _write_synthetic_inputs(out_dir, params, counts, seed=synthetic_seed)
    (out_dir / f"{testcase}_kernel.cpp").write_text(
        emit_kernel_cpp(
            cpp_text,
            name,
            is_mixed,
            params,
            mixed_group_wrapped=mixed_group_wrapped,
        ),
        encoding="utf-8",
    )
    (out_dir / "launch.cpp").write_text(emit_launch_cpp(name, params), encoding="utf-8")
    main_cpp = (
        emit_npu_benchmark_main_cpp(name, params, counts, scalar_values, block_dim)
        if run_mode == "npu"
        else emit_main_cpp(name, params, counts, scalar_values, block_dim)
    )
    (out_dir / "main.cpp").write_text(main_cpp, encoding="utf-8")
    sim_default = "ON" if run_mode == "sim" else "OFF"
    npu_default = "ON" if run_mode == "npu" else "OFF"
    (out_dir / "CMakeLists.txt").write_text(
        _CMAKE_TEMPLATE.replace("@TESTCASE@", testcase)
        .replace("@AICORE_ARCH@", aicore_arch)
        .replace("@SIM_DEFAULT@", sim_default)
        .replace("@NPU_DEFAULT@", npu_default),
        encoding="utf-8",
    )
    (out_dir / "golden.py").write_text(emit_golden(params, counts), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "testcase": testcase,
        "kernel": name,
        "run_mode": run_mode,
        "aicore_arch": aicore_arch,
        "block_dim": block_dim,
        "mixed": is_mixed,
        **({"mixed_runner": mixed_wrapper} if mixed_wrapper is not None else {}),
        **(
            {"input_source": {"kind": "synthetic", "seed": synthetic_seed}}
            if synthetic_seed is not None
            else {}
        ),
        **({"capture": capture} if capture is not None else {}),
        "parameters": [
            {
                "name": param.name,
                "cpp_type": param.cpp_type,
                "kind": "pointer" if param.is_ptr else "scalar",
                **(
                    {"elements": counts[param.name]}
                    if param.is_ptr
                    else {"value": scalar_values.get(param.name, "1")}
                ),
            }
            for param in params
        ],
    }
    (out_dir / "standalone_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate a camodel profiling testcase from a .cpp + .pto")
    ap.add_argument("--input", required=True, help="PTOAS kernel .cpp (.pto sibling read for buffer sizes)")
    ap.add_argument("--testcase", required=True, help="Testcase name, e.g. <func>_msprof")
    ap.add_argument("--output-root", required=True, help="Root dir; case -> <root>/ptoas/<testcase>/")
    ap.add_argument(
        "--run-mode",
        default="sim",
        choices=["sim", "npu"],
        help="emit an op-simulator case or a real-device event-timed benchmark",
    )
    ap.add_argument("--soc-version", default="Ascend910B1", help="CLI compat (cmake -DSOC_VERSION)")
    ap.add_argument("--aicore-arch", default="dav-c220", help="--cce-aicore-arch (a2a3 / a5)")
    ap.add_argument("--block-dim", type=int, default=1, help="exact launch block dimension")
    ap.add_argument(
        "--ptoas-root",
        type=Path,
        help="PTOAS checkout supplying its canonical mixed AIC/AIV group wrapper",
    )
    ap.add_argument(
        "--input-dir",
        type=Path,
        help="directory containing one real <ABI-name>.bin file per pointer argument",
    )
    ap.add_argument(
        "--synthetic-inputs",
        action="store_true",
        help="write deterministic bounded ABI inputs directly (NPU mode; no model args dump)",
    )
    ap.add_argument(
        "--synthetic-seed",
        type=int,
        default=_SYNTHETIC_SEED,
        help=f"seed used by --synthetic-inputs (default: {_SYNTHETIC_SEED})",
    )
    ap.add_argument(
        "--scalar",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="exact scalar-tail argument; repeat for multiple scalars",
    )
    ap.add_argument(
        "--args-dump",
        type=Path,
        help="args_dump.json from an enable_dump_args=2 run (small workloads only)",
    )
    ap.add_argument("--func-id", type=int, help="kernel func_id to extract from --args-dump")
    ap.add_argument("--task-id", help="exact task dispatch ID to extract from --args-dump")
    ap.add_argument(
        "--task-occurrence",
        type=int,
        help="zero-based task occurrence when the selected func_id was dispatched more than once",
    )
    args = ap.parse_args(argv)

    arch = args.aicore_arch or "dav-c220"
    scalar_values = _parse_scalar_assignments(args.scalar)
    dump_selection = None
    if args.args_dump is not None:
        if args.func_id is None:
            ap.error("--func-id is required with --args-dump")
        dump_selection = DumpSelection(
            args.args_dump,
            args.func_id,
            task_id=args.task_id,
            task_occurrence=args.task_occurrence,
        )
    out_dir = generate(
        Path(args.input),
        args.testcase,
        Path(args.output_root),
        arch,
        run_mode=args.run_mode,
        block_dim=args.block_dim,
        input_dir=args.input_dir,
        scalar_values=scalar_values,
        dump_selection=dump_selection,
        synthetic_seed=args.synthetic_seed if args.synthetic_inputs else None,
        ptoas_root=args.ptoas_root,
    )
    print(f"[gen_profiling_case] wrote testcase -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
