# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Provider preparation through ordinary JIT, including all native topologies."""

from pathlib import Path
from typing import Any, cast

import pypto.language as pl
import pytest
from pypto import backend, passes
from pypto.backend.cannbotdsl_jit import CANNBotDSLBinding
from pypto.jit.external.compatibility import request_from_context
from pypto.jit.external.contracts import (
    Access,
    CompileTarget,
    Coverage,
    KernelContract,
    TensorEffect,
    WritebackBinding,
)
from pypto.jit.external.providers.cannbotdsl import CANNBotDSLProvider
from pypto.jit.external.validation import ContractError
from pypto.jit.specializer import TensorMeta, build_specialize_context
from pypto.runtime import ExecutionMode, RunConfig


class _Binding:
    def __init__(self, core):
        self.core = core
        self.requests = []

    def prepare(self, context, platform, runtime, output_dir):
        ctx = context
        assert platform == "a5"
        assert runtime == passes.RuntimeKind.TENSORMAP_AND_RINGBUFFER
        request = request_from_context(ctx, CompileTarget("a5", "ccec", "test"))
        self.requests.append(request)
        path = Path(output_dir) / "callback.cpp"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"// export {len(self.requests)}\n")
        ctx.external_core_type = self.core
        ctx.external_aic_source = str(path) if self.core != "aiv" else None
        ctx.external_aiv_source = str(path) if self.core != "aic" else None
        ctx.external_dual_aiv_dispatch = self.core == "mixed"
        ctx.external_binary_policy = "a5-text-only"
        ctx.external_contract = KernelContract(
            request.parameters,
            (TensorEffect(0, Access.READ), TensorEffect(1, Access.WRITE, Coverage.FULL)),
            writebacks=(WritebackBinding(1),),
        )


def _entry(binding):
    @pl.jit.extern(binding=binding)
    def native(x: pl.Tensor, out: pl.Tensor): ...

    @pl.jit
    def entry(
        x: pl.Tensor[[32], pl.FP32], tmp: pl.Tensor[[32], pl.FP32], out: pl.Out[pl.Tensor[[32], pl.FP32]]
    ):
        native(x, tmp)
        native(tmp, out)
        return out

    return entry


@pytest.mark.parametrize("core", ["aic", "aiv", "mixed"])
def test_binding_prepares_owned_variants_without_jit_cache(monkeypatch, core):
    backend.reset_for_testing()
    monkeypatch.setattr("pypto.jit.decorator._ptoas_available", lambda: False)
    binding = _Binding(core)
    entry = _entry(binding)
    config = RunConfig(backend_type=backend.BackendType.Ascend950, execution_mode=ExecutionMode.ONBOARD)
    first = entry.compile(config=config)
    second = entry.compile(config=config)
    assert first is not second
    assert len(binding.requests) == 2
    assert not entry._cache
    with pytest.raises(RuntimeError, match="cannot request recompilation"):
        entry._resolve_compiled((), {"config": config}, allow_signature_mode=True, _lookup_only=True)
    # A second preparation cannot replace the first program's owned export.
    first_sources = [f for f in first.program.functions.values() if "external_source" in f.attrs]
    assert len(first_sources) == (2 if core == "mixed" else 1)
    for function in first_sources:
        assert function.attrs["external_binary_policy"] == "a5-text-only"
        assert Path(function.attrs["external_source"]).read_text() == "// export 1\n"
    output = Path(first.output_dir)
    config_source = (output / "kernel_config.py").read_text()
    assert config_source.count("a5-text-only") == len(first_sources)
    orchestration = "\n".join(path.read_text() for path in output.rglob("*.cpp"))
    assert orchestration.count(".set_allow_early_resolve(false)") >= 2
    if core == "mixed":
        assert "MixedKernels" in orchestration


def test_binding_rejects_source_options_and_targetless_specialization():
    with pytest.raises(ValueError, match="cannot be combined"):
        pl.jit.extern(binding=_Binding("aiv"), source="ignored.cpp")(lambda: None)
    with pytest.raises(ValueError, match="determines core_type"):
        pl.jit.extern(binding=_Binding("aiv"), core_type="aic")
    with pytest.raises(TypeError, match="implement prepare"):
        pl.jit.extern(binding=cast(Any, object()))
    with pytest.raises(ValueError, match="explicit target"):
        _entry(_Binding("aiv")).specialize()


def _context(shape):
    def native(x: pl.Tensor, out: pl.Tensor): ...

    return build_specialize_context(
        native,
        "native",
        "extern",
        None,
        {name: TensorMeta(shape, pl.FP32) for name in ("x", "out")},
        {},
        [],
    )


@pytest.mark.parametrize("shape", [(3,), (0,), ("n",)])
def test_cannbot_binding_rejects_unproven_cache_line_views(tmp_path, shape):
    binding = CANNBotDSLBinding(CANNBotDSLProvider(), object())
    with pytest.raises(ContractError, match="positive static|whole 64-byte"):
        binding.prepare(_context(shape), "a5", passes.RuntimeKind.TENSORMAP_AND_RINGBUFFER, str(tmp_path))


@pytest.mark.parametrize(
    "platform,runtime",
    [
        ("a5sim", passes.RuntimeKind.TENSORMAP_AND_RINGBUFFER),
        ("a2a3", passes.RuntimeKind.TENSORMAP_AND_RINGBUFFER),
        ("a5", passes.RuntimeKind.HOST_BUILD_GRAPH),
    ],
)
def test_cannbot_binding_rejects_unsupported_target(tmp_path, platform, runtime):
    binding = CANNBotDSLBinding(CANNBotDSLProvider(), object())
    with pytest.raises(ContractError, match="requires platform a5"):
        binding.prepare(_context((32,)), platform, runtime, str(tmp_path))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
