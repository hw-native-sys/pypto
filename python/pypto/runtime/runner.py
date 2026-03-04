# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Runner for executing PTO test cases.

Orchestrates the full execution pipeline:
1. Get program from test case (@pl.program or IRBuilder)
2. Generate kernel and orchestration code via PyPTO ir.compile()
3. Generate golden.py
4. Execute via simpler's CodeRunner
5. Validate results
"""

import time
import traceback
from datetime import datetime
from pathlib import Path

from pypto.backend import set_backend_type
from pypto.runtime.environment import setup_simpler_paths
from pypto.runtime.golden_generator import GoldenGenerator
from pypto.runtime.harness import PTOTestCase, RunConfig, RunResult
from pypto.runtime.program_generator import ProgramCodeGenerator


class Runner:
    """Executes PTO test cases via simpler's CodeRunner.

    This runner integrates with simpler's CodeRunner to execute tests:
    1. Generate kernel and orchestration C++ from PyPTO program via ir.compile()
    2. Generate golden.py for reference computation
    3. Use CodeRunner to compile, execute, and validate

    Example:
        runner = Runner(RunConfig(platform="a2a3sim"))
        result = runner.run(my_test_case)
        assert result.passed
    """

    __test__ = False  # Not a pytest test class

    def __init__(self, config: RunConfig | None = None):
        """Initialize runner.

        Args:
            config: Runtime configuration. If None, uses default config.
        """
        self.config = config or RunConfig()

    def run(self, test_case: PTOTestCase) -> RunResult:
        """Run a test case and return results.

        All build artifacts (kernels, orchestration, golden.py, pass dumps)
        are always persisted under the output directory.

        Args:
            test_case: The test case to run.

        Returns:
            RunResult with pass/fail status and details.
        """
        start_time = time.time()
        test_name = test_case.get_name()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Always use a persistent output directory: build_output/<test_name>_<timestamp>
        if self.config.output_dir:
            work_dir = Path(self.config.output_dir)
        else:
            work_dir = Path.cwd() / "build_output" / f"{test_name}_{timestamp}"
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Set PyPTO backend type for code generation
            backend_type = test_case.get_backend_type()
            set_backend_type(backend_type)

            # 1. Generate kernel C++ files
            program = test_case.get_program()
            if program is None:
                raise ValueError(
                    f"Test case {test_name} must implement get_program() "
                    "to return a @pl.program class or ir.Program"
                )

            strategy = test_case.get_strategy()
            codegen = ProgramCodeGenerator(strategy=strategy, backend_type=backend_type)
            codegen_result = codegen.generate(
                program,
                work_dir,
                dump_passes=self.config.dump_passes,
            )

            # Extract results
            kernel_configs = codegen_result["kernels"]
            orch_info = codegen_result.get("orchestration")

            if not kernel_configs:
                raise ValueError(f"No kernels generated for {test_name}")

            # 2. Verify orchestration was generated
            if orch_info is None:
                raise ValueError(
                    f"No orchestration generated for {test_name}. "
                    "Ensure your @pl.program includes an orchestration function "
                    "(decorated with @pl.function(type=pl.FunctionType.Orchestration))."
                )

            # 3. Generate golden.py in work_dir
            golden_path = work_dir / "golden.py"
            golden_gen = GoldenGenerator()
            golden_gen.write(test_case, golden_path)

            # 4. Execute via CodeRunner (skip if codegen_only)
            if self.config.codegen_only:
                return RunResult(
                    passed=True,
                    test_name=test_name,
                    execution_time=time.time() - start_time,
                )

            self._execute_with_code_runner(work_dir, golden_path, test_name)

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

    def _execute_with_code_runner(
        self,
        work_dir: Path,
        golden_path: Path,
        test_name: str,
    ) -> None:
        """Execute test using simpler's CodeRunner.

        Args:
            work_dir: Path to work directory with kernel_config.py and golden.py
            golden_path: Path to golden.py
            test_name: Name of the test (for logging)

        Raises:
            Exception: If test execution fails
        """
        from code_runner import CodeRunner  # noqa: PLC0415

        runner = CodeRunner(
            kernels_dir=str(work_dir),
            golden_path=str(golden_path),
            platform=self.config.platform,
            device_id=self.config.device_id,
        )

        runner.run()


# Backward-compatible alias
TestRunner = Runner


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

    def run_all(self, runner: Runner | None = None) -> dict[str, RunResult]:
        """Run all test cases in the suite."""
        if runner is None:
            runner = Runner(self.config)

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


def run(test_case: PTOTestCase, config: RunConfig | None = None) -> RunResult:
    """Convenience function to run a single PTO test case.

    Automatically sets up the Simpler environment (paths, sys.path) and
    executes the test case through the Runner pipeline.

    Args:
        test_case: The PTOTestCase to run.
        config: Runtime configuration. If None, uses the test_case's own config.

    Returns:
        RunResult with pass/fail status and details.

    Example:
        from pypto.runtime import run, RunConfig

        result = run(my_test_case, RunConfig(platform="a2a3"))
        assert result.passed, result.error
    """
    if config is None:
        config = test_case.config

    # Setup Simpler environment if not codegen-only
    if not config.codegen_only:
        setup_simpler_paths()

    runner = Runner(config)
    return runner.run(test_case)
