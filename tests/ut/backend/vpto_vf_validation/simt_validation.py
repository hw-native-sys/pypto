# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Native SIMT fixture and linked-image validation support."""

import os
import re
import runpy
from pathlib import Path

from pypto import DataType, backend, ir
from pypto.backend.pto_backend import _generate_kernel_wrapper
from pypto.ir import IRBuilder
from pypto.runtime._vpto import validate_device_image
from pypto.runtime.elf_parser import extract_text_section

SOURCE = Path(__file__).resolve().parent
SUPPORT = runpy.run_path(str(SOURCE / "test_validation.py"))


def make_wrapper() -> str:
    """Use production Tensor origin/scalar unpacking for the native export ABI."""
    builder = IRBuilder()
    with builder.function("run_simt_copy", type=ir.FunctionType.InCore) as function:
        function.param("src", ir.TensorType([1, 128], DataType.FP32))
        function.param("dst", ir.TensorType([1, 128], DataType.FP32))
        n = function.param("n", ir.ScalarType(DataType.INT32))
        function.return_type(ir.ScalarType(DataType.INT32))
        builder.return_stmt(n)
    backend.reset_for_testing()
    backend.set_backend_type(backend.BackendType.Ascend950)
    try:
        return _generate_kernel_wrapper(
            function.get_result(),
            'extern "C" __aicore__ void run_simt_copy(__gm__ float*, __gm__ float*, int32_t);\n',
        )
    finally:
        backend.reset_for_testing()


def check_native_image(evidence: Path) -> None:
    """Inspect actual compiler outputs, including the SIMT launch size operand.

    LLVM export bypasses PTOAS' object size patcher. This small, intrinsic-only
    fixture must have a correct size without binary patching; inline-assembly
    SIMT bodies are deliberately outside this acceptance probe.
    """
    llvm = (evidence / "vpto_0.ll").read_text()
    signature = next(line for line in llvm.splitlines() if line.startswith("define void @run_simt_copy("))
    assert signature.count("ptr addrspace(1)") == 2 and "i32" in signature
    assert "call simt_entry void @copy_body" in llvm
    assert '!"simt-max-threads", i32 128' in llvm
    assert '!"simt-max-registers", i32 128' in llvm
    assert "@llvm.hivm.ldg.cache.s32" in llvm and "@llvm.hivm.stg.cache.b32" in llvm
    assert "addrspace(6)" not in llvm, "Direct GM copy must not stage through UB"
    check_simt_launch_size(evidence, "copy_body")


def check_simt_launch_size(evidence: Path, helper_name: str) -> None:
    """Verify the linked instruction's size against its actual SIMT entry symbol."""
    evidence = evidence.resolve()
    image = (evidence / "kernel.elf").read_bytes()
    validate_device_image(image)
    assert extract_text_section(image), "The production loader must extract nonempty device code"
    cann = Path(os.environ["ASCEND_HOME_PATH"])
    run = SUPPORT["run"]
    run(evidence, "ptoas-version", [SUPPORT["require_ptoas"](), "--version"])
    run(evidence, "bisheng-version", [cann / "bin/bisheng", "--version"])
    disassembly = run(evidence, "disassembly", [cann / "bin/llvm-objdump", "-d", evidence / "kernel.elf"])
    symbols = run(evidence, "symbols", ["readelf", "-Ws", evidence / "kernel.elf"])
    symbol = re.search(
        rf"\s+(\d+)\s+FUNC\s+WEAK\s+DEFAULT\s+\d+\s+{re.escape(helper_name)}_simt_entry$",
        symbols,
        re.M,
    )
    assert symbol is not None, "Device linking dropped the SIMT entry"
    sizes = re.findall(r"\bVF_SIMT\s+[^\n#]*#(\d+)", disassembly)
    assert len(sizes) == 1 and int(sizes[0]) != 65535, "Missing or unpatched VF_SIMT size"
    assert int(sizes[0]) * 4 == int(symbol[1]), "VF_SIMT size disagrees with the linked callee"
