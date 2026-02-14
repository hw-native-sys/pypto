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

from pypto.backend import BackendType, set_backend_type

from harness.adapters.golden_generator import GoldenGenerator
from harness.adapters.program_generator import ProgramCodeGenerator
from harness.core.harness import PTOTestCase, TestConfig, TestResult

# tests/st/harness/core/test_runner.py -> tests/st/ -> project root
_ST_DIR = Path(__file__).parent.parent.parent
_PROJECT_ROOT = _ST_DIR.parent.parent

# Session-level output directory (shared across all tests in a pytest session)
_SESSION_OUTPUT_DIR = None

# Counter for unique test numbering within a session
_TEST_COUNTER = 0


def _get_next_test_number() -> int:
    """Get the next sequential test number for unique naming."""
    global _TEST_COUNTER  # noqa: PLW0603
    _TEST_COUNTER += 1
    return _TEST_COUNTER


def _get_session_output_dir() -> Path:
    """Get or create session-level output directory with timestamp.

    Returns:
        Path to the session output directory (build/tests/st/outputs/output_{timestamp}/).
    """
    global _SESSION_OUTPUT_DIR  # noqa: PLW0603
    if _SESSION_OUTPUT_DIR is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = _PROJECT_ROOT / "build" / "tests" / "st" / "outputs"
        _SESSION_OUTPUT_DIR = output_base / f"output_{timestamp}"
        _SESSION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return _SESSION_OUTPUT_DIR


class TestRunner:
    """Executes PTO test cases via simpler's CodeRunner.

    This runner integrates with simpler's CodeRunner to execute tests:
    1. Generate kernel and orchestration C++ from PyPTO program via ir.compile()
    2. Generate golden.py for reference computation
    3. Use CodeRunner to compile, execute, and validate

    Example:
        runner = TestRunner(TestConfig(platform="a2a3sim"))
        result = runner.run(my_test_case)
        assert result.passed
    """

    __test__ = False  # Not a pytest test class

    def __init__(self, config: TestConfig | None = None):
        """Initialize test runner.

        Args:
            config: Test configuration. If None, uses default config.
        """
        self.config = config or TestConfig()

    def run(self, test_case: PTOTestCase) -> TestResult:
        """Run a test case and return results.

        Args:
            test_case: The test case to run.

        Returns:
            TestResult with pass/fail status and details.
        """
        start_time = time.time()
        test_name = test_case.get_name()

        # Determine work directory based on save_kernels configuration
        if self.config.save_kernels:
            # Always save mode: use persistent directory directly
            # Add sequential number prefix to avoid name collisions
            test_num = _get_next_test_number()
            numbered_name = f"{test_num:03d}_{test_name}"
            if self.config.save_kernels_dir:
                work_dir = Path(self.config.save_kernels_dir) / numbered_name
            else:
                session_dir = _get_session_output_dir()
                work_dir = session_dir / numbered_name
            work_dir.mkdir(parents=True, exist_ok=True)
            use_temp = False
        else:
            # Temporary mode: use temp directory for execution
            work_dir = Path(tempfile.mkdtemp(prefix=f"pypto_test_{test_name}_"))
            use_temp = True

        try:
            # Set PyPTO backend type to CCE for code generation
            set_backend_type(BackendType.CCE)

            # 1. Generate kernel C++ files
            program = test_case.get_program()
            if program is None:
                raise ValueError(
                    f"Test case {test_name} must implement get_program() "
                    "to return a @pl.program class or ir.Program"
                )

            strategy = test_case.get_strategy()
            codegen = ProgramCodeGenerator(strategy=strategy)
            codegen_result = codegen.generate(
                program,
                work_dir,  # Pass work_dir instead of kernels_dir
                dump_passes=self.config.dump_passes,
            )

            # Extract results
            kernel_configs = codegen_result["kernels"]
            orch_info = codegen_result.get("orchestration")

            if not kernel_configs:
                raise ValueError(f"No kernels generated for {test_name}")

            # 2. Verify orchestration was generated
            # ir.compile() should generate orchestration/*.cpp and kernel_config.py
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
                # Codegen-only mode: skip runtime execution
                return TestResult(
                    passed=True,
                    test_name=test_name,
                    execution_time=time.time() - start_time,
                )

            self._execute_with_code_runner(work_dir, golden_path, test_name)

            return TestResult(
                passed=True,
                test_name=test_name,
                execution_time=time.time() - start_time,
            )

        except Exception as e:
            return TestResult(
                passed=False,
                test_name=test_name,
                error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                execution_time=time.time() - start_time,
            )
        finally:
            # Clean up temporary directory if used
            if use_temp and work_dir.exists():
                shutil.rmtree(work_dir)

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
        # code_runner is from simpler and may not be available at import time
        from code_runner import CodeRunner  # noqa: PLC0415

        runner = CodeRunner(
            kernels_dir=str(work_dir),
            golden_path=str(golden_path),
            platform=self.config.platform,
            device_id=self.config.device_id,
        )

        # Run the test
        runner.run()


class TestSuite:
    """Collection of test cases that can be run together."""

    __test__ = False  # Not a pytest test class

    def __init__(self, name: str, config: TestConfig | None = None):
        """Initialize test suite.

        Args:
            name: Suite name.
            config: Configuration for all tests in suite.
        """
        self.name = name
        self.config = config or TestConfig()
        self._test_cases: list = []

    def add_test(self, test_case: PTOTestCase) -> "TestSuite":
        """Add a test case to the suite."""
        self._test_cases.append(test_case)
        return self

    def run_all(self, runner: TestRunner | None = None) -> dict[str, TestResult]:
        """Run all test cases in the suite."""
        if runner is None:
            runner = TestRunner(self.config)

        results = {}
        for test_case in self._test_cases:
            result = runner.run(test_case)
            results[test_case.get_name()] = result
            print(result)

        return results

    def summary(self, results: dict[str, TestResult]) -> str:
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
