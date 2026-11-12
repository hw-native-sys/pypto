# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Execute generated bridge code against pinned TMR descriptors and argument slots."""

import hashlib
import shutil
import subprocess
import textwrap
from dataclasses import replace
from pathlib import Path

import pypto.language as pl
import pytest
from _orchestration_codegen_common import (
    _assert_orchestration_compiles,
    _generate_orch_code,
    _generate_orch_full_pipeline,
    _runtime_include_args,
)
from pypto import backend
from pypto.backend.external_bridge import RuntimeTarget
from pypto.backend.tmr_bridge import TMRBridge, materialize
from pypto.jit._specialize_types import DynDim, TensorMeta
from pypto.jit.external.compatibility import request_from_context
from pypto.jit.external.contracts import (
    ABIParameter,
    ABIValue,
    Access,
    AliasPermission,
    CompileTarget,
    Completion,
    Core,
    Coverage,
    ExecutionRequirements,
    KernelContract,
    KernelRequest,
    NativeABI,
    ParameterSpec,
    PreparationIdentity,
    PreparedKernel,
    ScalarSpec,
    SourceArtifact,
    SourceFile,
    TensorEffect,
    TensorSpec,
    Unit,
    WritebackBinding,
)
from pypto.jit.external.validation import ContractError
from pypto.jit.specializer import SpecializeContext, Specializer
from pypto.pypto_core import DataType


def _source(content: str) -> SourceArtifact:
    data = content.encode()
    return SourceArtifact(
        (SourceFile("entry.hpp", hashlib.sha256(data).hexdigest(), data),), "entry.hpp", "native"
    )


@pytest.fixture
def prepared():
    params = (
        ParameterSpec("x", TensorSpec(("d0",), DataType.FP32, alignment=4)),
        ParameterSpec("scale", ScalarSpec(DataType.FP32)),
        ParameterSpec("out", TensorSpec(("d0",), DataType.FP32, alignment=4)),
    )
    abi = [
        ABIParameter(1, ABIValue.SCALAR, 4, 4),
        ABIParameter(2, ABIValue.DATA_POINTER, 8, 8),
        ABIParameter(0, ABIValue.DATA_POINTER, 8, 8),
    ]
    for index, unit in ((0, Unit.ELEMENTS), (2, Unit.BYTES)):
        abi.extend(
            (
                ABIParameter(index, ABIValue.DIMENSION, 4, 4, axis=0, unit=unit),
                ABIParameter(index, ABIValue.STRIDE, 4, 4, axis=0, unit=unit),
                ABIParameter(index, ABIValue.OFFSET, 8, 8, unit=unit),
            )
        )
    return PreparedKernel(
        KernelRequest(params, CompileTarget("a5", "ccec", "test")),
        KernelContract(
            params,
            (TensorEffect(0, Access.READ), TensorEffect(2, Access.WRITE, Coverage.FULL)),
            writebacks=(WritebackBinding(2),),
        ),
        _source("""inline __aicore__ void native(float scale, __gm__ float* out, const __gm__ float* x,
                                     uint32_t n, uint32_t stride, uint64_t offset,
                                     uint32_t out_bytes, uint32_t out_stride, uint64_t out_offset) {
    always_assert(out_bytes == n * 4);
    for (uint32_t i = 0; i < n; ++i)
        out[(out_offset + i * out_stride) / 4] = x[offset + i * stride] * scale;
}"""),
        NativeABI("c_pointer", 1, tuple(abi)),
        ExecutionRequirements((Core.AIV,), 1, Completion.ENTRY_RETURN),
        PreparationIdentity("fixture", "1", "1", "b" * 64, True),
    )


def _target(prepared):
    return RuntimeTarget(
        "tmr",
        "1",
        prepared.request.target,
        frozenset({("c_pointer", 1)}),
        frozenset(
            {
                "core:aic",
                "core:aiv",
                "aiv_lanes:0",
                "aiv_lanes:1",
                "aiv_lanes:2",
                "completion:entry_return_visible",
                "parameter_returns",
                "tensor_writeback",
                "partial_write",
                "aliasing",
                "full_residency",
                "sync_start",
                "logical_launch_context",
                "invented",
            }
        ),
    )


def _compile_run(path: Path, source: str, *, includes: tuple[Path, ...] = ()) -> None:
    compiler = shutil.which("g++-15") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    path.write_text(source)
    args = _runtime_include_args(
        Path(__file__).resolve().parents[3], runtime_name="tensormap_and_ringbuffer", arch="a5"
    )
    binary = path.with_suffix("")
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-DNDEBUG",
            *(f"-I{p}" for p in includes),
            *args,
            str(path),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("permission", [False, True])
def test_generated_guard_checks_each_invocation(prepared, tmp_path, permission):
    if permission:
        prepared = replace(
            prepared, contract=replace(prepared.contract, alias_permissions=(AliasPermission(0, 2),))
        )
    task = TMRBridge().lower(prepared, _target(prepared))
    # Use the real TensorRef union, tags, scalar slots and strided descriptors.
    # Each mutation simulates a new invocation of the same compiled artifact.
    source = (
        """#include <cstring>
#include "types.h"
[[noreturn]] void assert_impl(const char* condition, const char*, int) { throw condition; }
"""
        + task.preflight_source
        + f"""
int main() {{
    float xbuf[64]{{}}, ybuf[64]{{}};
    simpler::tmr::Tensor x{{}}, y{{}};
    x.buffer = {{uint64_t(xbuf), sizeof(xbuf)}}; x.ndims = 1;
    x.dtype = DataType::FLOAT32; x.shapes[0] = 3; x.strides[0] = 2; x.start_offset = 1;
    y.buffer = {{uint64_t(ybuf), sizeof(ybuf)}}; y.ndims = 1;
    y.dtype = DataType::FLOAT32; y.shapes[0] = 3; y.strides[0] = 3; y.start_offset = 2;
    CoreTaskArgs args; args.add_input(x); args.add_output(y); args.add_scalar(2.0f);
    auto check = [&]() {{ {task.preflight_symbol}(args); }};
    auto rejects = [&](const char* message) {{
        try {{ check(); }} catch (const char* why) {{ return std::strstr(why, message) != nullptr; }}
        return false;
    }};
    check();
    y.shapes[0] = 2; if (!rejects("shared symbol")) return 1; y.shapes[0] = 3;
    y.strides[0] = 0; if (!rejects("positive stride")) return 2; y.strides[0] = 3;
    y.start_offset = 64; if (!rejects("storage bounds")) return 3; y.start_offset = 2;
    y.buffer.addr++; if (!rejects("alignment")) return 4; y.buffer.addr--;
    y.dtype = DataType::INT32; if (!rejects("dtype")) return 5; y.dtype = DataType::FLOAT32;
    y.ndims = 2; if (!rejects("rank")) return 6; y.ndims = 1;
    y.start_offset = UINT64_MAX; if (!rejects("overflow")) return 7; y.start_offset = 2;
    auto addr = y.buffer.addr; y.buffer.addr = x.buffer.addr;
    if (rejects("overlap") == {str(permission).lower()}) return 8;
    // Disjoint sibling views share storage without needing alias permission.
    y.start_offset = 32; check(); y.start_offset = 2; y.buffer.addr = addr;
    // Runtime-allocated OUTPUT is a different union member: reject before dereferencing.
    args.tag(1) = TensorArgType::OUTPUT;
    if (!rejects("runtime-allocated")) return 9;
    args.tag(1) = TensorArgType::OUTPUT_EXISTING;
    // Metadata can fit the descriptor while overflowing the selected native width.
    x.shapes[0] = y.shapes[0] = 1U << 30;
    x.strides[0] = y.strides[0] = 1; x.start_offset = y.start_offset = 0;
    x.buffer.size = y.buffer.size = 1ULL << 34;
    if (!rejects("native tensor_dimension width")) return 10;
    x.shapes[0] = y.shapes[0] = 0; check();
}}
"""
    )
    _compile_run(tmp_path / "guard.cpp", source)


@pytest.mark.parametrize(
    "address,offset,accepted",
    [(0x1004, 3, False), (0x1004, 0, False), (0x1000, 1, True), (0x1000, 4, True)],
)
def test_pointer_alignment_is_independent_of_view_offset(prepared, tmp_path, address, offset, accepted):
    parameter = replace(prepared.request.parameters[0], type=TensorSpec(("d0",), DataType.FP32, alignment=16))
    parameters = (parameter, *prepared.request.parameters[1:])
    prepared = replace(
        prepared,
        request=replace(prepared.request, parameters=parameters),
        contract=replace(prepared.contract, parameters=parameters),
    )
    task = TMRBridge().lower(prepared, _target(prepared))
    # The checker does not dereference storage. Synthetic addresses distinguish
    # allocation-base alignment from an offset that aligns/misaligns the view.
    source = (
        """#include <cstring>
#include "types.h"
[[noreturn]] void assert_impl(const char* condition, const char*, int) { throw condition; }
"""
        + task.preflight_source
        + f"""
int main() {{
    simpler::tmr::Tensor x{{}}, y{{}};
    x.buffer = {{{address}, 256}}; x.ndims = 1;
    x.dtype = DataType::FLOAT32; x.shapes[0] = 3; x.strides[0] = 2; x.start_offset = {offset};
    y = x; y.buffer.addr = 0x2000;
    CoreTaskArgs args; args.add_input(x); args.add_output(y); args.add_scalar(2.0f);
    try {{
        {task.preflight_symbol}(args);
        return {0 if accepted else 1};
    }} catch (const char* why) {{
        return {1 if accepted else 0} || std::strstr(why, "x: alignment") == nullptr;
    }}
}}
"""
    )
    _compile_run(tmp_path / "alignment.cpp", source)


@pytest.mark.parametrize("cube", [False, True])
@pytest.mark.parametrize("inherited_hf32", [False, True])
@pytest.mark.parametrize("atomic_type", range(7))
def test_native_order_bits_offsets_and_completion(prepared, tmp_path, cube, inherited_hf32, atomic_type):
    # Only hardware intrinsics are mocked. Descriptor layout and argument packing
    # come from the pinned runtime. The native body touches strided view elements.
    asc = tmp_path / "mock_device_intrinsics.h"
    restored_ctrl = 42 | ((1 << 46) if inherited_hf32 else 0)
    # CTRL[8:6] encodes disabled mode or one of six atomic data types. Saving
    # CTRL before clearing these bits used to undo the reset at entry and exit.
    initial_ctrl = restored_ctrl | (atomic_type << 6)
    expected_ctrl = 42 if cube else restored_ctrl
    native = (
        prepared.artifact.files[0]
        .content.decode()
        .replace(
            "always_assert(out_bytes == n * 4);",
            "always_assert(out_bytes == n * 4);\n"
            f"    always_assert(mock_ctrl == {expected_ctrl}ULL);\n"
            # Native code changes both atomic mode and an unrelated control bit.
            # Exit must disable atomics while restoring the unrelated state.
            "    mock_ctrl ^= (1ULL << 48) | 128;",
        )
    )
    prepared = replace(
        prepared,
        artifact=_source(native),
        execution=replace(
            prepared.execution, cores=(Core.AIC,) if cube else (Core.AIV,), aiv_lanes=0 if cube else 1
        ),
    )
    asc.write_text(
        ("#define __DAV_CUBE__\n" if cube else "")
        + f"static uint64_t mock_ctrl = {initial_ctrl}ULL;\n"
        + """#pragma once
#define __gm__
#define __aicore__
constexpr int PIPE_ALL = 0, DSB_DDR = 0;
static int events = 0;
inline uint64_t get_ctrl() { events = events * 10 + 1; return mock_ctrl; }
inline void set_atomic_none() { mock_ctrl &= ~(7ULL << 6); events = events * 10 + 2; }
inline void pipe_barrier(int) { events = events * 10 + 3; }
enum class cache_line_t { ENTIRE_DATA_CACHE };
enum class dcci_dst_t { CACHELINE_OUT };
inline void dcci(int64_t*, cache_line_t, dcci_dst_t) { events = events * 10 + 4; }
inline void dsb(int) { events = events * 10 + 5; }
inline void set_ctrl(uint64_t v) { mock_ctrl = v; events = events * 10 + 6; }
"""
    )
    task = TMRBridge().lower(prepared, _target(prepared))
    with materialize(task) as artifact:
        source = f'''#include <cstring>
#include "types.h"
[[noreturn]] void assert_impl(const char* c, const char*, int) {{ throw c; }}
#include "mock_device_intrinsics.h"
#include "{artifact.source}"
int main() {{
    float xbuf[16]{{}}, ybuf[16]{{}};
    xbuf[1] = 2; xbuf[3] = 3; xbuf[5] = 4;
    simpler::tmr::Tensor x{{}}, y{{}};
    x.buffer = {{uint64_t(xbuf), sizeof(xbuf)}}; x.ndims = 1;
    x.dtype = DataType::FLOAT32; x.shapes[0] = 3; x.strides[0] = 2; x.start_offset = 1;
    y.buffer = {{uint64_t(ybuf), sizeof(ybuf)}}; y.ndims = 1;
    y.dtype = DataType::FLOAT32; y.shapes[0] = 3; y.strides[0] = 3; y.start_offset = 2;
    int64_t args[] = {{int64_t(&x), int64_t(&y), 0x40000000}};
    kernel_entry(args);
    if (events != {21634526 if cube else 2134526} || mock_ctrl != {restored_ctrl}ULL) return 1;
    if (ybuf[2] != 4 || ybuf[5] != 6 || ybuf[8] != 8 || ybuf[0] != 0) return 2;
    events = 0;
    args[2] = 0x80000000; kernel_entry(args); // signed zero survives bit unpacking
    uint32_t bits; std::memcpy(&bits, &ybuf[2], 4); if (bits != 0x80000000) return 3;
    if (mock_ctrl != {restored_ctrl}ULL) return 4;
}}
'''
        _compile_run(tmp_path / "wrapper.cpp", source, includes=(tmp_path,))
    assert not Path(artifact.source).exists()


@pytest.mark.parametrize("requirement", ["sync_start", "logical_launch_context"])
def test_capability_claim_does_not_implement_launch_requirement(prepared, requirement):
    prepared = replace(prepared, execution=replace(prepared.execution, **{requirement: True}))
    with pytest.raises(ContractError, match=f"{requirement} is not implemented"):
        TMRBridge().lower(prepared, _target(prepared))


def test_reject_unknown_capability_and_runtime(prepared):
    prepared = replace(
        prepared, contract=replace(prepared.contract, required_capabilities=frozenset({"invented"}))
    )
    with pytest.raises(ContractError, match="unsupported.*capabilities"):
        TMRBridge().lower(prepared, _target(prepared))
    original = replace(prepared, contract=replace(prepared.contract, required_capabilities=frozenset()))
    with pytest.raises(ContractError, match="expected runtime"):
        TMRBridge().lower(original, replace(_target(original), runtime="hbg"))


def test_materialization_rejects_modified_payload(prepared):
    task = TMRBridge().lower(prepared, _target(prepared))
    with pytest.raises(ContractError, match="unmodified"):
        with materialize(replace(task, preflight_source="")):
            pytest.fail("modified checker accepted")


@pytest.mark.parametrize("core", ["AIC", "AIV", "mixed"])
@pytest.mark.parametrize("wrapper", [False, True])
@pytest.mark.parametrize("explicit", [False, True], ids=["call", "submit"])
@pytest.mark.parametrize("full_pipeline", [False, True])
def test_codegen_preflight_precedes_submit(prepared, tmp_path, core, wrapper, explicit, full_pipeline):
    backend.reset_for_testing()
    backend.set_backend_type(backend.BackendType.Ascend950)
    task = TMRBridge().lower(prepared, _target(prepared))
    attrs = {
        "external_preflight_source": task.preflight_source,
        "external_preflight_symbol": task.preflight_symbol,
    }
    with materialize(task) as artifact:
        members = ("AIC", "AIV") if core == "mixed" else (core,)
        declarations = ""
        for member in members:
            declarations += f"""
    @pl.function(type=pl.FunctionType.{member}, external_source={artifact.source!r})
    def native_{member}(self, x: pl.Tensor[[3], pl.FP32], scale: pl.Scalar[pl.FP32],
                out: pl.InOut[pl.Tensor[[3], pl.FP32]]):
        pl.func_attr({attrs!r})
        ...
"""
        callee = f"native_{members[0]}"
        if core == "mixed":
            declarations += """
    @pl.function(type=pl.FunctionType.Group)
    def grouped(self, x: pl.Tensor[[3], pl.FP32], scale: pl.Scalar[pl.FP32],
                out: pl.InOut[pl.Tensor[[3], pl.FP32]]):
        self.native_AIC(x, scale, out)
        self.native_AIV(x, scale, out)
"""
            callee = "grouped"
        if wrapper:
            declarations += f"""
    @pl.function(type=pl.FunctionType.Spmd)
    def wrapped(self, scale: pl.Scalar[pl.FP32], out: pl.InOut[pl.Tensor[[3], pl.FP32]],
                x: pl.Tensor[[3], pl.FP32]):
        pl.func_attr({{"core_num": 1}})
        self.{callee}(x, scale, out)
"""
            callee = "wrapped"
        args = "scale, out, x" if wrapper else "x, scale, out"
        call = f"self.{callee}({args})"
        if explicit:
            call = f"with pl.manual_scope():\n    (), task = pl.submit(self.{callee}, {args})"
        source = f"""import pypto.language as pl
@pl.program
class P:
{declarations}
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[3], pl.FP32], scale: pl.Scalar[pl.FP32],
                out: pl.InOut[pl.Tensor[[3], pl.FP32]]):
{textwrap.indent(call, "        ")}
"""
        program = pl.parse(source)
        output = _generate_orch_full_pipeline(program) if full_pipeline else _generate_orch_code(program)
        checker = f"{task.preflight_symbol}(params_t0);"
        assert output.count(checker) == 1
        assert output.index(".add_scalar(") < output.index(checker) < output.index("rt_submit_")
        path = tmp_path / "main.cpp"
        path.write_text(output)
        _assert_orchestration_compiles(path, runtime_name="tensormap_and_ringbuffer", arch="a5")


@pytest.mark.parametrize(
    "core,lanes", [((Core.AIC,), 0), ((Core.AIV,), 1), ((Core.AIC, Core.AIV), 1), ((Core.AIC, Core.AIV), 2)]
)
def test_context_handoff_uses_contract_topology(prepared, core, lanes):
    dim = DynDim("M", "M", 3)
    context = SpecializeContext(
        func_name="native",
        source="""def native(x: pl.Tensor, scale: pl.Scalar[pl.FP32],
           out: pl.InOut[pl.Tensor]):
    ...
""",
        func_type="extern",
        level=None,
        param_names=["x", "scale", "out"],
        tensor_meta={name: TensorMeta((dim,), DataType.FP32) for name in ("x", "out")},
        scalar_dtypes={"scale": DataType.FP32},
        py_globals={"pl": pl},
    )
    request = request_from_context(context, prepared.request.target)
    prepared = replace(
        prepared,
        request=request,
        contract=replace(prepared.contract, parameters=request.parameters),
        execution=replace(prepared.execution, cores=core, aiv_lanes=lanes),
    )
    with materialize(TMRBridge().lower(prepared, _target(prepared))) as artifact:
        artifact.configure_context(context)
        source = Specializer("Bridge", [context]).specialize()
        assert "external_preflight_source" in source
        assert ("FunctionType.Group" in source) == (len(core) == 2)
        assert ('"dual_aiv_dispatch": True' in source) == (lanes == 2)
        pl.parse(source)
        # Reject reuse across a changed specialization before mutating the context.
        context.scalar_dtypes["scale"] = DataType.INT32
        with pytest.raises(ContractError, match="prepare it again"):
            artifact.configure_context(context)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("strides", (0,), "positive strides"),
        ("shape", (1 << 32,), "exceeds uint32"),
        ("offset", 1 << 64, "offset exceeds"),
        ("alignment", 1 << 64, "alignment exceeds"),
    ],
)
def test_reject_unrepresentable_runtime_constraints(prepared, field, value, message):
    parameter = replace(
        prepared.request.parameters[0], type=replace(prepared.request.parameters[0].type, **{field: value})
    )
    parameters = (parameter, *prepared.request.parameters[1:])
    prepared = replace(
        prepared,
        request=replace(prepared.request, parameters=parameters),
        contract=replace(prepared.contract, parameters=parameters),
    )
    with pytest.raises(ContractError, match=message):
        TMRBridge().lower(prepared, _target(prepared))


@pytest.mark.parametrize("core", ["cube", "vec"])
def test_native_wrapper_compiles_with_ccec(prepared, tmp_path, core):
    compiler = shutil.which("ccec")
    if compiler is None:
        pytest.skip("CANN device compiler unavailable")
    repo = Path(__file__).resolve().parents[3]
    includes = [repo / "runtime/src/common", repo / "runtime/src/common/task_interface"]
    # A real device function, with no test-only assertion handler or mocked
    # completion operations. Both code variants must accept the generated ABI.
    prepared = replace(
        prepared,
        artifact=_source("""
inline __aicore__ void native(float scale, __gm__ float* out, const __gm__ float* x,
                            uint32_t n, uint32_t stride, uint64_t offset,
                            uint32_t out_bytes, uint32_t out_stride, uint64_t out_offset) {
    for (uint32_t i = 0; i < n; ++i)
        out[(out_offset + i * out_stride) / 4] = x[offset + i * stride] * scale;
}
"""),
    )
    with materialize(TMRBridge().lower(prepared, _target(prepared))) as artifact:
        result = subprocess.run(
            [
                compiler,
                "-c",
                "-O2",
                "-std=c++17",
                "-x",
                "cce",
                "--cce-aicore-only",
                f"--cce-aicore-arch=dav-c310-{core}",
                *(f"-I{p}" for p in includes),
                artifact.source,
                "-o",
                str(tmp_path / "kernel.o"),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("symbol", ["kernel_entry", "broken();", "kernel\nentry"])
def test_reject_invalid_native_symbol(prepared, symbol):
    prepared = replace(prepared, artifact=replace(prepared.artifact, symbol=symbol))
    with pytest.raises(ContractError, match="reserved|qualified C\\+\\+ identifier"):
        TMRBridge().lower(prepared, _target(prepared))


def test_reject_descriptor_abi(prepared):
    abi = replace(
        prepared.native_abi,
        parameters=(
            ABIParameter(1, ABIValue.SCALAR, 4, 4),
            ABIParameter(0, ABIValue.DESCRIPTOR, 32, 8),
            ABIParameter(2, ABIValue.DESCRIPTOR, 32, 8),
        ),
    )
    with pytest.raises(ContractError, match="does not define descriptor layout"):
        TMRBridge().lower(replace(prepared, native_abi=abi), _target(prepared))
