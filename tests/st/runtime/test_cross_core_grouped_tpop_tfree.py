# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime regression for consecutive cross-core ``tpop`` and ``tfree`` order.

The test rewrites the compiled V2CUD AIC block from:

1. ``tpop0, alloc0, use0, tfree0``
2. ``tpop1, alloc1, use1, tfree1``

into:

1. ``tpop0, tpop1``
2. ``alloc0, use0, alloc1, use1``
3. ``tfree0, tfree1``

so the reordered artifact visually exposes consecutive pops and consecutive
frees.
"""

import importlib.util
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import torch
from pypto.backend import BackendType
from pypto.backend.pto_backend import _preprocess_ptoas_output, _run_ptoas
from pypto.runtime import compile_program
from pypto.runtime.device_runner import (
    build_orch_args_from_inputs,
    compile_and_assemble,
    execute_on_device,
    validate_golden,
)

from tests.st.runtime.test_cross_core import V2CUDProgram

_PLATFORM_TO_BACKEND: dict[str, BackendType] = {
    "a2a3": BackendType.Ascend910B,
}
_DEFAULT_PLATFORM = "a2a3"
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_platform(config: pytest.Config) -> str:
    """Resolve the effective platform from the session-wide allowlist."""
    raw_platform = str(config.getoption("--platform") or "")
    tokens = [tok.strip() for tok in raw_platform.split(",") if tok.strip()]
    if tokens and _DEFAULT_PLATFORM not in tokens:
        raise pytest.UsageError(
            "tests/st/runtime/test_cross_core_grouped_tpop_tfree.py only supports --platform=a2a3"
        )
    return _DEFAULT_PLATFORM


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Provide the single effective platform to the standalone runtime regression."""
    if "platform" not in metafunc.fixturenames:
        return
    platform = _resolve_platform(metafunc.config)
    metafunc.parametrize("platform", [platform], ids=[platform])


def _load_kernel_config(work_dir: Path) -> Any:
    config_path = work_dir / "kernel_config.py"
    spec = importlib.util.spec_from_file_location("_cross_core_grouped_tpop_kernel_config", str(config_path))
    assert spec is not None and spec.loader is not None, f"Cannot load kernel_config.py from {config_path}"
    kernel_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel_config)
    return kernel_config


def _default_saved_work_dir(test_name: str) -> Path:
    """Return the default saved-artifact path under ``build_output/``."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _PROJECT_ROOT / "build_output" / f"{test_name}_{timestamp}"


def _require_ptoas() -> None:
    if os.environ.get("PTOAS_ROOT") or shutil.which("ptoas"):
        return
    pytest.skip("ptoas is unavailable for the consecutive-tpop/consecutive-tfree runtime regression")


def _resolve_backend_type(platform: str) -> BackendType:
    try:
        return _PLATFORM_TO_BACKEND[platform]
    except KeyError as exc:
        raise pytest.UsageError(
            "tests/st/runtime/test_cross_core_grouped_tpop_tfree.py only supports --platform=a2a3"
        ) from exc


def _rewrite_consecutive_tpop_tfree_pto(pto_path: Path) -> None:
    lines = pto_path.read_text(encoding="utf-8").splitlines()

    try:
        func_start = next(
            i for i, line in enumerate(lines) if line.startswith("  func.func @main_incore_0_aic(")
        )
    except StopIteration as exc:
        raise AssertionError(f"AIC function not found in {pto_path}") from exc

    try:
        func_end = next(i for i in range(func_start + 1, len(lines)) if lines[i] == "  }")
    except StopIteration as exc:
        raise AssertionError(f"AIC function end not found in {pto_path}") from exc

    tpop_indices = [i for i in range(func_start, func_end) if "pto.tpop_from_aiv" in lines[i]]
    tfree_indices = [
        i for i in range(func_start, func_end) if lines[i].strip() == "pto.tfree_from_aiv {split = 1}"
    ]
    assert len(tpop_indices) == 2, (
        f"Expected 2 tpop lines in reordered-artifact test, got {len(tpop_indices)}"
    )
    assert len(tfree_indices) == 2, (
        f"Expected 2 tfree lines in reordered-artifact test, got {len(tfree_indices)}"
    )

    original_block = lines[tpop_indices[0] : tfree_indices[1] + 1]
    assert len(original_block) == 8, (
        "Expected the V2CUD AIC cross-core block to contain exactly 8 lines "
        f"(tpop/alloc/tmov/tfree x2), got {len(original_block)}"
    )
    pop0, alloc0, use0, free0, pop1, alloc1, use1, free1 = original_block
    assert use0.strip().startswith("pto.tmov ins("), use0
    assert use1.strip().startswith("pto.tmov ins("), use1
    assert free0.strip() == "pto.tfree_from_aiv {split = 1}", free0
    assert free1.strip() == "pto.tfree_from_aiv {split = 1}", free1

    consecutive_pop_then_free_block = [
        pop0,
        pop1,
        alloc0,
        use0,
        alloc1,
        use1,
        free0,
        free1,
    ]
    assert "pto.tpop_from_aiv" in consecutive_pop_then_free_block[0], consecutive_pop_then_free_block[0]
    assert "pto.tpop_from_aiv" in consecutive_pop_then_free_block[1], consecutive_pop_then_free_block[1]
    assert consecutive_pop_then_free_block[-2].strip() == "pto.tfree_from_aiv {split = 1}"
    assert consecutive_pop_then_free_block[-1].strip() == "pto.tfree_from_aiv {split = 1}"

    rewritten = (
        "\n".join(lines[: tpop_indices[0]] + consecutive_pop_then_free_block + lines[tfree_indices[1] + 1 :])
        + "\n"
    )
    pto_path.write_text(rewritten, encoding="utf-8")


def _replace_ptoas_body(wrapper_text: str, new_body: str) -> str:
    start = wrapper_text.index("// --- ptoas-generated code ---")
    end = wrapper_text.index("// --- Kernel entry point ---")
    return f"{wrapper_text[:start]}// --- ptoas-generated code ---\n\n{new_body}\n{wrapper_text[end:]}"


def _rebuild_consecutive_tpop_tfree_artifact(work_dir: Path) -> None:
    pto_path = work_dir / "ptoas" / "main_incore_0.pto"
    cpp_path = work_dir / "ptoas" / "main_incore_0.cpp"

    _rewrite_consecutive_tpop_tfree_pto(pto_path)
    _run_ptoas(
        str(pto_path),
        str(cpp_path),
        ptoas_flags=["--enable-insert-sync", "--pto-level=level3", "--pto-arch", "a3"],
    )

    new_body = _preprocess_ptoas_output(cpp_path.read_text(encoding="utf-8"))
    kernel_config = _load_kernel_config(work_dir)
    for kernel in kernel_config.KERNELS:
        wrapper_path = Path(kernel["source"])
        wrapper_text = wrapper_path.read_text(encoding="utf-8")
        wrapper_path.write_text(_replace_ptoas_body(wrapper_text, new_body), encoding="utf-8")


class TestCrossCoreGroupedTpopTfree:
    """Consecutive-tpop/consecutive-tfree cross-core runtime regression."""

    @pytest.mark.platforms("a2a3")
    def test_v2c_updown_consecutive_tpop_then_tfree_reordered_artifact(self, platform, test_config):
        """Rewrite compiled V2C PTO into consecutive-pop and consecutive-free order."""
        from harness.core.test_runner import _last_device  # noqa: PLC0415

        _require_ptoas()
        backend_type = _resolve_backend_type(platform)
        device_id = int(os.environ.get("TASK_DEVICE", str(test_config.device_id)))

        test_name = "cross_core_v2c_updown_consecutive_tpop_then_tfree"
        if test_config.save_kernels:
            if test_config.save_kernels_dir:
                work_dir = Path(test_config.save_kernels_dir) / test_name
            else:
                work_dir = _default_saved_work_dir(test_name)
            work_dir.mkdir(parents=True, exist_ok=True)
            cleanup_dir = None
        else:
            work_dir = Path(tempfile.mkdtemp(prefix="pypto_cross_core_consecutive_pop_free_"))
            cleanup_dir = work_dir

        try:
            compile_program(
                V2CUDProgram,
                work_dir,
                strategy=test_config.strategy,
                backend_type=backend_type,
                dump_passes=test_config.dump_passes,
            )
            _rebuild_consecutive_tpop_tfree_artifact(work_dir)

            kernel_config = _load_kernel_config(work_dir)
            runtime_cfg = getattr(kernel_config, "RUNTIME_CONFIG", {})
            chip_callable, runtime_name = compile_and_assemble(
                work_dir, platform, pto_isa_commit=test_config.pto_isa_commit
            )

            for seed in (0, 1, 2):
                torch.manual_seed(seed)
                a = torch.randn(32, 32, dtype=torch.float32)
                b = torch.randn(32, 32, dtype=torch.float32)
                output = torch.zeros(32, 32, dtype=torch.float32)

                orch_args, _, _, outputs = build_orch_args_from_inputs(
                    [("a", a), ("b", b), ("output", output)],
                    {"output"},
                )
                if test_config.any_dfx_enabled():
                    dfx_dir = work_dir / "dfx_outputs"
                    dfx_dir.mkdir(parents=True, exist_ok=True)
                    output_prefix: str | None = str(dfx_dir)
                else:
                    output_prefix = None
                execute_on_device(
                    chip_callable,
                    orch_args,
                    platform,
                    runtime_name,
                    device_id,
                    block_dim=runtime_cfg.get("block_dim", 24),
                    aicpu_thread_num=runtime_cfg.get("aicpu_thread_num", 4),
                    output_prefix=output_prefix,
                    enable_l2_swimlane=test_config.enable_l2_swimlane,
                    enable_dump_tensor=test_config.enable_dump_tensor,
                    enable_pmu=test_config.enable_pmu,
                    enable_dep_gen=test_config.enable_dep_gen,
                )
                validate_golden(
                    outputs,
                    {"output": torch.matmul(a + b, a - b)},
                    rtol=test_config.rtol,
                    atol=test_config.atol,
                )
        finally:
            if cleanup_dir is not None:
                shutil.rmtree(cleanup_dir, ignore_errors=True)

        _last_device["value"] = device_id


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))
