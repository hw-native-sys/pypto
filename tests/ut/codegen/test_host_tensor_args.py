# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""HBG control tensor transfer selection and rejection of mixed storage roles."""

import sys
from types import ModuleType, SimpleNamespace

import pypto.language as pl
import pytest
from pypto.backend._host_tensor_args import host_tensor_indices
from pypto.backend.pto_backend import _generate_config_file
from pypto.pypto_core import ir, passes


@pl.program
class HostControl:
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        data: pl.Tensor[[8], pl.FP32],
        n: pl.Scalar[pl.INT32],
        control: pl.InOut[pl.Tensor[[2], pl.INT32]],
    ):
        value = pl.tensor.read(control, [0])
        pl.tensor.write(control, [1], value + n)


@pl.program
class MixedControl:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, data: pl.Tensor[[8], pl.INT32]):
        pass

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, control: pl.Tensor[[8], pl.INT32]):
        pl.tensor.read(control, [0])
        self.kernel(control)


@pl.program
class SlicedControl:
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, control: pl.Tensor[[8], pl.INT32]):
        view = pl.slice(control, [2], [2])
        pl.tensor.read(view, [0])


def _main(program: ir.Program) -> ir.Function:
    func = program.get_function("main")
    assert func is not None
    return func


def test_host_reads_and_writes_use_tensor_pool_indices():
    assert host_tensor_indices(_main(HostControl)) == [1]


def test_hbg_manifest_retains_host_control_transfers():
    config = _generate_config_file(
        "main",
        [],
        runtime=passes.RuntimeKind.HOST_BUILD_GRAPH,
        host_tensor_indices=host_tensor_indices(_main(HostControl)),
    )
    namespace: dict[str, object] = {"__file__": "kernel_config.py"}
    exec(config, namespace)
    runtime_config = namespace["RUNTIME_CONFIG"]
    assert isinstance(runtime_config, dict)
    assert runtime_config["host_tensor_indices"] == [1]


def test_mixed_host_and_device_control_is_rejected():
    with pytest.raises(ValueError, match="separate tensor arguments"):
        host_tensor_indices(_main(MixedControl))


def test_host_access_through_slice_selects_source_parameter():
    assert host_tensor_indices(_main(SlicedControl)) == [0]


def test_compile_emits_host_control_metadata(tmp_path, monkeypatch):
    from pypto.ir.compile import compile as ir_compile  # noqa: PLC0415
    from pypto.ir.compiled_program import CompiledProgram  # noqa: PLC0415
    from pypto.runtime._artifact_sources import read_kernel_config  # noqa: PLC0415

    # Configuration loading needs only signature enum names, not a native runtime.
    task_interface = ModuleType("simpler.task_interface")
    setattr(task_interface, "ArgDirection", SimpleNamespace(IN=0, OUT=1, INOUT=2, SCALAR=3))
    monkeypatch.setitem(sys.modules, "simpler.task_interface", task_interface)
    with passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH):
        compiled = ir_compile(HostControl, output_dir=str(tmp_path), platform="a2a3sim")
    config = read_kernel_config(tmp_path / "kernel_config.py")
    assert config.RUNTIME_CONFIG["host_tensor_indices"] == [1]
    assert isinstance(compiled, CompiledProgram)
    assert compiled._host_tensor_indices() == [1]


@pl.program
class PredicateControl:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, data: pl.Tensor[[8], pl.INT32]):
        pass

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, data: pl.Tensor[[8], pl.INT32], gate: pl.Tensor[[8], pl.INT32]):
        with pl.manual_scope():
            _ = pl.spmd_submit(self.kernel, data, core_num=1, predicate=(gate[0] > 0))


@pl.program
class MixedPredicateControl:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, data: pl.Tensor[[8], pl.INT32]):
        pass

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, data: pl.Tensor[[8], pl.INT32], gate: pl.Tensor[[8], pl.INT32]):
        pl.tensor.read(gate, [0])
        with pl.manual_scope():
            _ = pl.spmd_submit(self.kernel, data, core_num=1, predicate=(gate[0] > 0))


@pl.program
class InternalPredicateControl:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, data: pl.Tensor[[8], pl.INT32]):
        pass

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, data: pl.Tensor[[8], pl.INT32]):
        gate = pl.create_tensor([8], dtype=pl.INT32)
        with pl.manual_scope():
            _ = pl.spmd_submit(self.kernel, data, core_num=1, predicate=(gate[0] > 0))


@pytest.mark.parametrize("program", [PredicateControl, InternalPredicateControl])
def test_scheduler_predicate_is_not_a_host_read(program):
    assert host_tensor_indices(_main(program)) == []


def test_predicate_and_host_read_of_same_storage_is_rejected():
    with pytest.raises(ValueError, match="separate tensor arguments"):
        host_tensor_indices(_main(MixedPredicateControl))
