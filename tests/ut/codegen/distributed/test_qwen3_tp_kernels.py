# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Lower every standalone Qwen3 TP2 entry without allocating model weights."""

import ast
import importlib.util
import sys
from pathlib import Path
from typing import cast

import pytest
import torch
from pypto.runtime import RunConfig

SOURCE = Path(__file__).resolve().parents[4] / "examples/models/qwen3_tp/kernels.py"
ENTRIES = {
    node.name: node
    for node in ast.parse(SOURCE.read_text()).body
    if isinstance(node, ast.FunctionDef) and node.name.endswith("_chip")
}


@pytest.fixture(scope="module")
def kernels():
    spec = importlib.util.spec_from_file_location("qwen3_tp_kernels", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


@pytest.mark.parametrize("name", list(ENTRIES))
def test_qwen3_chip_lowering(kernels, tmp_path, name):
    values = []
    dtypes = {"BF16": torch.bfloat16, "FP32": torch.float32, "INT32": torch.int32}
    for arg in ENTRIES[name].args.args:
        annotation = cast(ast.Subscript, arg.annotation)
        if cast(ast.Attribute, annotation.value).attr in ("InOut", "Out"):
            annotation = cast(ast.Subscript, annotation.slice)
        if cast(ast.Attribute, annotation.value).attr == "Scalar":
            values.append(1)
        else:
            shape_node, dtype_node = cast(ast.Tuple, annotation.slice).elts
            shape = eval(compile(ast.Expression(shape_node), str(SOURCE), "eval"), vars(kernels))
            values.append(
                torch.empty(shape, dtype=dtypes[cast(ast.Attribute, dtype_node).attr], device="meta")
            )
    compiled = getattr(kernels, name).compile(
        *values,
        config=RunConfig(platform="a2a3", save_kernels=True, save_kernels_dir=str(tmp_path)),
    )
    assert list(Path(compiled.output_dir).rglob("*.pto"))
    assert (Path(compiled.output_dir) / "kernel_config.py").is_file()
    if name == "tp_ffn_swiglu_down_step_chip":
        orchestration = "\n".join(
            path.read_text() for path in (Path(compiled.output_dir) / "orchestration").glob("*.cpp")
        )
        assert orchestration.count("rt_submit_aiv_task(") == 1
        assert orchestration.count("rt_submit_aic_task(") == 1
        assert "params_t0.add_inout(ext_act);" in orchestration
        assert "params_t1.add_input(ext_act);" in orchestration
        assert orchestration.index("rt_submit_aiv_task(") < orchestration.index("rt_submit_aic_task(")
    if name == "tp_ffn_step_chip":
        orchestration = "\n".join(
            path.read_text() for path in (Path(compiled.output_dir) / "orchestration").glob("*.cpp")
        )
        assert orchestration.count("rt_submit_aiv_task(") == 2
        assert orchestration.count("rt_submit_aic_task(") == 2
        for producer, consumer, tensor in ((0, 1, "xn"), (1, 2, "gate"), (1, 2, "up"), (2, 3, "act")):
            assert f"params_t{producer}.add_inout(ext_{tensor});" in orchestration
            assert f"params_t{consumer}.add_input(ext_{tensor});" in orchestration
    for path in Path(compiled.output_dir).rglob("*.pto"):
        for line in path.read_text().splitlines():
            if "pto.tcvt " in line and "dtype=bf16" in line.split("outs(")[-1]:
                assert "round_mode RINT" in line, line
    if name in (
        "rms_hidden_step_chip",
        "tp_hidden_rms_step_chip",
        "tp_qkv_post_step_chip",
        "tp_ffn_residual_rms_step_chip",
        "tp_ffn_step_chip",
    ):
        lines = [
            line
            for path in Path(compiled.output_dir).rglob("*.pto")
            for line in path.read_text().splitlines()
            if "pto.trsqrt " in line
        ]
        assert lines, f"Missing normalization lowering for {name}"
        assert all("rsqrt_precision high_precision" in line for line in lines), (
            "Normalization must not use the approximate default vrsqrt path",
            lines,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
