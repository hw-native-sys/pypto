# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Device-free checks of Buffer mode in the real inline and pool compile paths."""

from concurrent.futures import ThreadPoolExecutor

import pypto.language as pl
import pytest
import torch
from pypto import ir, passes
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.runtime.runner import RunConfig

from harness import st
from harness.core import test_runner as tr


@pl.jit.incore
def _copy_kernel(a: pl.Tensor, out: pl.Out[pl.Tensor]):
    value = pl.load(a, [0, 0], [16, 32])
    return pl.store(value, [0, 0], out)


@pl.jit
def _copy_entry(a: pl.Tensor, out: pl.Out[pl.Tensor]):
    out = _copy_kernel(a, out)
    return out


def _case(planner=passes.MemoryPlanner.PYPTO, enabled=True):
    return st.case(
        _copy_entry,
        torch.arange(512, dtype=torch.float32).reshape(16, 32),
        torch.zeros(16, 32),
        name="representation_case",
        golden=lambda tensors: tensors["a"],
        platform="a2a3",
        memory_planner=planner,
        enable_buffer_ir=enabled,
    )


def test_buffer_mode_has_a_distinct_cache_key_without_changing_ordinary_cases():
    legacy = _case(enabled=False)
    explicit = _case(enabled=True)
    assert tr._cache_key(legacy) == "representation_case@a2a3@pypto"
    assert tr._cache_key(explicit) == "representation_case@a2a3@pypto@buffer_ir"
    explicit.platform = None
    bound = explicit.for_platform("a2a3")
    assert bound is not explicit and bound.get_enable_buffer_ir()


@pytest.mark.skipif(find_ptoas_binary() is None, reason="PTOAS is required for native source compilation")
@pytest.mark.parametrize("in_pool", [False, True], ids=["inline", "pool_thread"])
@pytest.mark.parametrize(
    "planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS]
)
def test_actual_compile_paths_preserve_case_mode_and_planner(tmp_path, planner, in_pool):
    case = _case(planner)
    tr._install_backend(BackendType.Ascend910B)
    if in_pool:
        work_dir = tmp_path / "pool"
        # This context lives on the parent thread and intentionally disagrees.
        # The worker must reconstruct settings from the case, not inherit it.
        with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO, enable_buffer_ir=False):
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(tr._compile_for_cache, case, work_dir, "a2a3", False, False, None).result()
    else:
        runner = tr.TestRunner(
            RunConfig(platform="a2a3", codegen_only=True, save_kernels=True, save_kernels_dir=str(tmp_path))
        )
        result = runner.run(case)
        assert result.passed, result.error
        work_dir = tmp_path / case.get_name()

    restored = ir.deserialize((work_dir / "buffer_ir.msgpack").read_bytes())
    assert isinstance(restored, ir.Program)
    devices = [f for f in restored.functions.values() if ir.is_incore_type(f.func_type)]
    assert devices and all(f.ir_stage == ir.FunctionIRStage.Buffer for f in devices)
    native = list((work_dir / "ptoas").glob("*.pto"))
    assert native
    text = "\n".join(path.read_text() for path in native)
    assert "pto.tload ins(" in text and "pto.tstore ins(" in text
    assert (" addr = " in text) == (planner != passes.MemoryPlanner.PTOAS)
    assert passes.PassContext.current() is None


def test_buffer_case_rejects_a_compiler_that_leaves_functional_device_ir(tmp_path, monkeypatch):
    case = _case()
    program = case.get_program()

    def leave_functional(program, **_kwargs):
        # Invoke a real pass/instrument callback, but omit final conversion.
        return passes.simplify()(program)

    monkeypatch.setattr(tr.ir, "compile", leave_functional)
    with pytest.raises(ValueError, match="did not convert every device function"):
        tr._compile_case_program(case, program, output_dir=str(tmp_path), memory_planner=None)
    assert not (tmp_path / "buffer_ir.msgpack").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
