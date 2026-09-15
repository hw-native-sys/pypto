# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Build and restore real program artifacts without a device Worker."""

import pypto.language as pl
import pytest
from pypto import passes


@pl.jit
def add_scalar(x: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]):
    """Add a scalar through the DSL to exercise real program binary generation."""
    with pl.at(level=pl.Level.CORE_GROUP):
        pl.store(pl.add(pl.load(x, [0, 0], [16, 16]), 3.0), [0, 0], out)
    return out


@pytest.mark.parametrize(
    "runtime", [passes.RuntimeKind.HOST_BUILD_GRAPH, passes.RuntimeKind.TENSORMAP_AND_RINGBUFFER]
)
def test_program_binary_build_without_worker(test_config, monkeypatch, tmp_path, runtime):
    """Real DSL binaries build and restore without SDK builds or Worker initialization."""
    if test_config.codegen_only:
        pytest.skip("Binary builds require the target compiler and SDK")

    from pypto.jit._artifact_manifest import BuildKind  # noqa: PLC0415
    from pypto.runtime._prebuilt import load_prebuilt, prepare_prebuilt, read_prebuilt  # noqa: PLC0415
    from pypto.runtime.kernel_compiler import KernelCompiler  # noqa: PLC0415
    from simpler.worker import Worker  # noqa: PLC0415
    from simpler_setup import KernelCompiler as SDKCompiler  # noqa: PLC0415

    def forbidden(*args, **kwargs):
        """Fail if building or restoring crosses a forbidden SDK or Worker boundary."""
        pytest.fail("program binary build used an SDK build or initialized a Worker")

    monkeypatch.setattr(Worker, "__init__", forbidden)
    monkeypatch.setattr(SDKCompiler, "compile_incore", forbidden)
    monkeypatch.setattr(SDKCompiler, "compile_orchestration", forbidden)
    monkeypatch.setenv("PYPTO_PROG_BUILD_DIR", str(tmp_path / "generated"))
    with passes.PassContext([], runtime=runtime):
        compiled = add_scalar.compile(config=test_config)
    assert compiled._chip_callable is None
    prepare_prebuilt(compiled.output_dir, compiled.platform, BuildKind.SINGLE_CHIP)
    record = read_prebuilt(compiled.output_dir, compiled.platform, BuildKind.SINGLE_CHIP)["."]
    assert record["orchestration"]["binary"].startswith(b"\x7fELF")
    assert record["kernels"] and all(kernel["binary"] for kernel in record["kernels"])
    monkeypatch.setattr(KernelCompiler, "compile_incore", forbidden)
    monkeypatch.setattr(KernelCompiler, "compile_orchestration", forbidden)
    chips = load_prebuilt(compiled.output_dir, compiled.platform, BuildKind.SINGLE_CHIP)
    assert chips["."][0] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
