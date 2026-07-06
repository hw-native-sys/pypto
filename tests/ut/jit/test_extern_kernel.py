# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for @pl.jit.extern — integrating hand-written C++ kernels via JIT.

An external kernel is a signature-only ``@pl.jit.extern`` stub backed by a
hand-written ``.cpp``. The specializer renders it into a header-only
``@pl.function(external_source=...)`` declaration; a ``core_type="mixed"``
kernel expands to an AIC member + AIV member + Group wrapper so the entry's
call lowers to a single MixedKernels submit.
"""

from pathlib import Path

import pypto.language as pl
import pytest
import torch
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.jit.specializer import Specializer
from pypto.pypto_core import ir

_KERNEL_SRC = '#include <cstdint>\nextern "C" void kernel_entry(int64_t* args) { (void)args; }\n'


def _write_kernel(tmp_path: Path, name: str = "ext.cpp") -> Path:
    src = tmp_path / name
    src.write_text(_KERNEL_SRC)
    return src


def _specialize(entry, *args):
    """Run the JIT front-half (bind -> contexts -> specialize) and return source."""
    pn, _, tmeta, sv, sd, pfd = entry._bind_args(args, {})
    contexts = entry._build_contexts(tmeta, sv, sd, pfd)
    return Specializer(f"_jit_{entry.__name__}", contexts).specialize()


def test_mixed_extern_expands_to_group(tmp_path):
    cpp = _write_kernel(tmp_path)

    @pl.jit.extern(core_type="mixed", aic_source=cpp, aiv_source=cpp)
    def pa(
        a: pl.Tensor[[128, 128], pl.FP16],
        out: pl.Out[pl.Tensor[[128, 128], pl.FP16]],
    ) -> pl.Tensor[[128, 128], pl.FP16]: ...

    @pl.jit
    def entry(a: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
        out = pa(a, out)
        return out

    a = torch.zeros(128, 128, dtype=torch.float16)
    out = torch.zeros(128, 128, dtype=torch.float16)
    src = _specialize(entry, a, out)

    # AIC member + AIV member + Group wrapper, entry dispatches the group.
    assert "type=pl.FunctionType.AIC, external_source=" in src
    assert "type=pl.FunctionType.AIV, external_source=" in src
    assert "type=pl.FunctionType.Group" in src
    assert "def pa_aic(self," in src
    assert "def pa_aiv(self," in src
    assert "self.pa_aic(a, out)" in src
    assert "self.pa_aiv(a, out)" in src
    assert "self.pa(a, out)" in src

    # The generated program parses and survives the full pass pipeline.
    program = pl.parse(src)
    assert isinstance(program, ir.Program)
    after = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
    names = {f.name for f in after.functions.values()}
    assert {"entry", "pa", "pa_aic", "pa_aiv"} <= names


def test_single_core_extern(tmp_path):
    cpp = _write_kernel(tmp_path)

    @pl.jit.extern(core_type="aiv", source=cpp)
    def relu(
        a: pl.Tensor[[64, 64], pl.FP32],
        out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]: ...

    @pl.jit
    def entry(a: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
        out = relu(a, out)
        return out

    a = torch.zeros(64, 64, dtype=torch.float32)
    out = torch.zeros(64, 64, dtype=torch.float32)
    src = _specialize(entry, a, out)

    # One AIV declaration, no Group / member split.
    assert "type=pl.FunctionType.AIV, external_source=" in src
    assert "def relu(self," in src
    assert "FunctionType.Group" not in src
    assert "relu_aic" not in src
    assert "self.relu(a, out)" in src


def test_source_hash_tracks_cpp_edits(tmp_path):
    """Editing the .cpp changes the JIT cache key (the Python stub never does)."""
    cpp = _write_kernel(tmp_path)

    @pl.jit.extern(core_type="aiv", source=cpp)
    def k(
        a: pl.Tensor[[16, 16], pl.FP32],
        out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ) -> pl.Tensor[[16, 16], pl.FP32]: ...

    @pl.jit
    def entry(a: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
        out = k(a, out)
        return out

    h1 = entry._get_source_hash()
    cpp.write_text(_KERNEL_SRC + "\n// changed\n")
    entry._source_hash = None  # force recompute (normally a fresh interpreter)
    h2 = entry._get_source_hash()
    assert h1 != h2


def test_extern_bad_core_type(tmp_path):
    cpp = _write_kernel(tmp_path)
    with pytest.raises(ValueError, match="core_type must be"):

        @pl.jit.extern(core_type="cube", source=cpp)
        def k(a: pl.Tensor, out: pl.Out[pl.Tensor]): ...


def test_extern_mixed_requires_both_sources(tmp_path):
    cpp = _write_kernel(tmp_path)
    with pytest.raises(ValueError, match="requires both aic_source"):

        @pl.jit.extern(core_type="mixed", aic_source=cpp)
        def k(a: pl.Tensor, out: pl.Out[pl.Tensor]): ...


def test_extern_source_not_found(tmp_path):
    missing = tmp_path / "nope.cpp"
    with pytest.raises(ValueError, match="source file not found"):

        @pl.jit.extern(core_type="aiv", source=missing)
        def k(a: pl.Tensor, out: pl.Out[pl.Tensor]): ...


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
