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


@pl.jit
def kernel_with_scalar(
    x: pl.Tensor[[16, 16], pl.FP32],
    scale: pl.Scalar[pl.FP32],
    out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
):
    """Exercise interleaved pools and a returned external output in real codegen."""
    with pl.at(level=pl.Level.CORE_GROUP):
        pl.store(pl.mul(pl.load(x, [0, 0], [16, 16]), scale), [0, 0], out)
    return out


@pytest.mark.parametrize(
    "runtime", [passes.RuntimeKind.HOST_BUILD_GRAPH, passes.RuntimeKind.TENSORMAP_AND_RINGBUFFER]
)
def test_kernel_binary_build_without_worker(test_config, monkeypatch, tmp_path, runtime):
    """Build actual kernel bytes, then restore them with every compiler/Worker forbidden."""
    from pypto import CacheConfig  # noqa: PLC0415
    from pypto.ir.compiled_program import CompiledProgram  # noqa: PLC0415
    from pypto.jit._artifact_manifest import BuildKind  # noqa: PLC0415
    from pypto.runtime._artifact_sources import package_generated_sources  # noqa: PLC0415
    from pypto.runtime._prebuilt import load_prebuilt, read_prebuilt  # noqa: PLC0415
    from pypto.runtime.kernel_compiler import KernelCompiler  # noqa: PLC0415
    from simpler.task_interface import ChipWorker  # noqa: PLC0415
    from simpler.worker import Worker  # noqa: PLC0415
    from simpler_setup import KernelCompiler as SDKCompiler  # noqa: PLC0415

    if test_config.codegen_only or test_config.platform.endswith("sim"):
        pytest.skip("The kernel descriptor currently targets a2a3/a5 device binaries")

    def forbidden(*args, **kwargs):
        pytest.fail("kernel build/restore initialized a Worker or invoked a forbidden compiler")

    monkeypatch.setattr(Worker, "__init__", forbidden)
    monkeypatch.setattr(ChipWorker, "__init__", forbidden)
    monkeypatch.setattr(SDKCompiler, "compile_incore", forbidden)
    monkeypatch.setattr(SDKCompiler, "compile_orchestration", forbidden)
    monkeypatch.setenv("PYPTO_PROG_BUILD_DIR", str(tmp_path / "generated"))
    test_config.cache_config = CacheConfig(enabled=False)
    with passes.PassContext([], runtime=runtime):
        artifact = kernel_with_scalar._resolve_kernel_artifact(
            (), {"config": test_config}, allow_signature_mode=True
        )
    assert artifact.kernel_abi.return_aliases == (2,)
    package_generated_sources(artifact.output_dir, BuildKind.SINGLE_CHIP)
    callable_ = artifact.load()
    assert callable_ is artifact.load()
    record = read_prebuilt(artifact.output_dir, artifact.platform, BuildKind.SINGLE_CHIP)["."]
    assert record["orchestration"]["binary"].startswith(b"\x7fELF")
    assert artifact.kernel_abi.binary_tag() in record["orchestration"]["binary"]
    assert record["kernels"] and all(k["binary"] for k in record["kernels"])
    with pytest.raises(ValueError, match="requires 'program'"):
        CompiledProgram.from_dir(artifact.output_dir)
    monkeypatch.setattr(KernelCompiler, "compile_incore", forbidden)
    monkeypatch.setattr(KernelCompiler, "compile_orchestration", forbidden)
    # No config execution, SDK discovery, compilation, or Worker is allowed on recovery.
    monkeypatch.setattr("pypto.runtime._artifact_sources.read_kernel_config", forbidden)
    monkeypatch.setattr(KernelCompiler, "__init__", forbidden)
    restored = load_prebuilt(
        artifact.output_dir, artifact.platform, BuildKind.SINGLE_CHIP, kernel_abi=artifact.kernel_abi
    )["."][0]
    assert restored is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
