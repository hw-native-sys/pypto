# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Test runner for executing PTO test cases.

Orchestrates the full test execution pipeline:
1. Get program from test case (@pl.program or IRBuilder)
2. Generate kernel and orchestration code via PyPTO ir.compile()
3. Generate golden.py
4. Execute via simpler's CodeRunner
5. Validate results
"""

import shutil
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path

import pytest
from pypto.backend import BackendType, set_backend_type
from pypto.runtime import compile_program
from pypto.runtime.golden_writer import _extract_compute_golden, generate_golden_source
from pypto.runtime.runner import RunConfig, RunResult, _execute_on_device
from pypto.runtime.tensor_spec import TensorSpec as RuntimeTensorSpec

from harness.core.harness import PTOTestCase

# tests/st/harness/core/test_runner.py -> tests/st/ -> project root
_ST_DIR = Path(__file__).parent.parent.parent
_PROJECT_ROOT = _ST_DIR.parent.parent

# Map BackendType to the architecture prefix used by the platform string.
# "a2a3" covers Ascend 910B (PTO and CCE backends); "a5" covers Ascend 950.
_BACKEND_TO_ARCH: dict[BackendType, str] = {
    BackendType.Ascend910B_PTO: "a2a3",
    BackendType.Ascend910B_CCE: "a2a3",
    BackendType.Ascend950: "a5",
}


def _resolve_platform(config_platform: str, backend_type: BackendType) -> str:
    """Return the platform string required to compile for *backend_type*.

    Preserves the sim/hardware distinction from *config_platform* (i.e. the
    ``sim`` suffix) while replacing the architecture prefix to match the
    backend.  For example, if the global config says ``"a2a3sim"`` but the
    test case requests ``Ascend950``, this returns ``"a5sim"``.
    """
    is_sim = config_platform.endswith("sim")
    arch = _BACKEND_TO_ARCH.get(backend_type, config_platform.rstrip("sim").rstrip("_"))
    return f"{arch}sim" if is_sim else arch


def _default_work_dir(test_name: str) -> Path:
    """Return the default output path for a saved test: build_output/{testName}_{timestamp}."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _PROJECT_ROOT / "build_output" / f"{test_name}_{timestamp}"


def _write_golden_for_test_case(test_case: PTOTestCase, output_path: Path) -> None:
    """Generate and write golden.py for *test_case*.

    Converts harness TensorSpec (DataType) to runtime TensorSpec (torch.dtype),
    extracts compute_golden from the compute_expected method, and writes golden.py.

    Args:
        test_case: The PTOTestCase to generate golden for.
        output_path: Destination path for the generated golden.py.
    """
    runtime_specs = [
        RuntimeTensorSpec(
            name=spec.name,
            shape=spec.shape,
            dtype=spec.dtype.torch_dtype,
            init_value=spec.init_value,
            is_output=spec.is_output,
        )
        for spec in test_case.tensor_specs
    ]

    try:
        compute_golden_src = _extract_compute_golden(test_case.compute_expected)
    except RuntimeError:
        output_specs = [s for s in test_case.tensor_specs if s.is_output]
        lines = [
            "def compute_golden(tensors, params):",
            '    """Compute expected outputs - PLACEHOLDER."""',
            "    # TODO: Could not extract compute_expected source.",
            "    # Please implement the expected computation here.",
        ]
        for spec in output_specs:
            lines.append(f'    # tensors["{spec.name}"][:] = ...')
        lines.append("")
        lines.append('    raise NotImplementedError("compute_expected source extraction failed")')
        compute_golden_src = "\n".join(lines)

    write_golden_src = generate_golden_source(
        runtime_specs,
        None,
        test_case.config.rtol,
        test_case.config.atol,
        compute_golden_src=compute_golden_src,
        scalar_specs=test_case.scalar_specs or None,
    )
    output_path.write_text(write_golden_src, encoding="utf-8")


class TestRunner:
    """Executes PTO test cases via simpler's CodeRunner.

    This runner integrates with simpler's CodeRunner to execute tests:
    1. Generate kernel and orchestration C++ from PyPTO program via ir.compile()
    2. Generate golden.py for reference computation
    3. Use CodeRunner to compile, execute, and validate

    Example:
        runner = TestRunner(RunConfig(platform="a2a3sim"))
        result = runner.run(my_test_case)
        assert result.passed
    """

    __test__ = False  # Not a pytest test class

    def __init__(self, config: RunConfig | None = None):
        """Initialize test runner.

        Args:
            config: Test configuration. If None, uses default config.
        """
        self.config = config or RunConfig()

    def run(self, test_case: PTOTestCase) -> RunResult:
        """Run a test case and return results.

        Args:
            test_case: The test case to run.

        Returns:
            RunResult with pass/fail status and details.
        """
        start_time = time.time()
        test_name = test_case.get_name()

        # Determine work directory based on save_kernels configuration
        if self.config.save_kernels:
            if self.config.save_kernels_dir:
                work_dir = Path(self.config.save_kernels_dir) / test_name
            else:
                work_dir = _default_work_dir(test_name)
            work_dir.mkdir(parents=True, exist_ok=True)
            use_temp = False
        else:
            work_dir = Path(tempfile.mkdtemp(prefix=f"pypto_test_{test_name}_"))
            use_temp = True

        try:
            # Set PyPTO backend type for code generation
            backend_type = test_case.get_backend_type()
            set_backend_type(backend_type)

            # 1. Get program
            program = test_case.get_program()
            if program is None:
                raise ValueError(
                    f"Test case {test_name} must implement get_program() "
                    "to return a @pl.program class or ir.Program"
                )

            # 2. Compile: generates kernels/, orchestration/ and patches headers
            strategy = test_case.get_strategy()
            compile_program(
                program,
                work_dir,
                strategy=strategy,
                backend_type=backend_type,
                dump_passes=self.config.dump_passes,
            )

            # 3. Validate that kernels and orchestration were generated
            if not list((work_dir / "kernels").rglob("*.cpp")):
                raise ValueError(f"No kernels generated for {test_name}")
            if not list((work_dir / "orchestration").glob("*.cpp")):
                raise ValueError(
                    f"No orchestration generated for {test_name}. "
                    "Ensure your @pl.program includes an orchestration function "
                    "(decorated with @pl.function(type=pl.FunctionType.Orchestration))."
                )

            # 4. Generate golden.py in work_dir
            golden_path = work_dir / "golden.py"
            _write_golden_for_test_case(test_case, golden_path)

            # 5. Execute via CodeRunner (skip if codegen_only)
            if self.config.codegen_only:
                return RunResult(
                    passed=True,
                    test_name=test_name,
                    execution_time=time.time() - start_time,
                )

            platform = _resolve_platform(self.config.platform, backend_type)
            _execute_on_device(work_dir, golden_path, platform, self.config.device_id)

            return RunResult(
                passed=True,
                test_name=test_name,
                execution_time=time.time() - start_time,
            )

        except Exception as e:
            return RunResult(
                passed=False,
                test_name=test_name,
                error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                execution_time=time.time() - start_time,
            )
        finally:
            if use_temp and work_dir.exists():
                shutil.rmtree(work_dir)


class TestSuite:
    """Collection of test cases that can be run together."""

    __test__ = False  # Not a pytest test class

    def __init__(self, name: str, config: RunConfig | None = None):
        """Initialize test suite.

        Args:
            name: Suite name.
            config: Configuration for all tests in suite.
        """
        self.name = name
        self.config = config or RunConfig()
        self._test_cases: list = []

    def add_test(self, test_case: PTOTestCase) -> "TestSuite":
        """Add a test case to the suite."""
        self._test_cases.append(test_case)
        return self

    def run_all(self, runner: TestRunner | None = None) -> dict[str, RunResult]:
        """Run all test cases in the suite."""
        if runner is None:
            runner = TestRunner(self.config)

        results = {}
        for test_case in self._test_cases:
            result = runner.run(test_case)
            results[test_case.get_name()] = result
            print(result)

        return results

    def summary(self, results: dict[str, RunResult]) -> str:
        """Generate summary of test results."""
        passed = sum(1 for r in results.values() if r.passed)
        total = len(results)
        failed = total - passed

        lines = [
            f"\n{'=' * 50}",
            f"Test Suite: {self.name}",
            f"{'=' * 50}",
            f"Passed: {passed}/{total}",
            f"Failed: {failed}/{total}",
        ]

        if failed > 0:
            lines.append("\nFailed tests:")
            for name, result in results.items():
                if not result.passed:
                    lines.append(f"  - {name}: {result.error}")

        return "\n".join(lines)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
