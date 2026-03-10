# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
PyPTO runtime runner.

Provides :func:`run`, the main entry point for compiling a ``@pl.program`` and
executing it on an Ascend NPU (or simulator), with correctness validation against
a user-supplied golden function.

Typical usage::

    import torch
    from pypto.runtime import run, RunConfig, TensorSpec

    def golden(tensors, params):
        tensors["out"][:] = tensors["a"] + tensors["b"]

    result = run(
        program=MyProgram,
        tensor_specs=[
            TensorSpec("a",   [128, 128], torch.float32, init_value=2.0),
            TensorSpec("b",   [128, 128], torch.float32, init_value=3.0),
            TensorSpec("out", [128, 128], torch.float32, is_output=True),
        ],
        golden=golden,
        config=RunConfig(platform="a2a3sim"),
    )
    print(result)  # PASS / FAIL: ...
"""

import shutil
import tempfile
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pypto import ir
from pypto.backend import BackendType, set_backend_type
from pypto.ir.pass_manager import OptimizationStrategy

from .golden_writer import write_golden
from .tensor_spec import TensorSpec


@dataclass
class RunConfig:
    """Configuration for a :func:`run` invocation.

    Attributes:
        platform: Target execution platform — ``"a2a3sim"`` (simulator) or
            ``"a2a3"`` (real Ascend hardware).
        device_id: Hardware device index (ignored for simulator).
        rtol: Relative tolerance for result comparison.
        atol: Absolute tolerance for result comparison.
        strategy: PyPTO optimisation strategy applied during compilation.
        backend_type: Code-generation backend (:attr:`BackendType.Ascend910B_CCE` by default).
        dump_passes: If ``True``, dump intermediate IR after each pass.
        work_dir: Directory for generated artefacts.  If ``None`` a temporary
            directory is created and removed after execution.  Set to a path to
            retain the generated C++ files for inspection.
    """

    platform: str = "a2a3sim"
    device_id: int = 0
    rtol: float = 1e-5
    atol: float = 1e-5
    strategy: OptimizationStrategy = field(default_factory=lambda: OptimizationStrategy.Default)
    backend_type: BackendType = field(default_factory=lambda: BackendType.Ascend910B_PTO)
    dump_passes: bool = False
    work_dir: str | None = None

    def __post_init__(self) -> None:
        if self.platform not in ("a2a3sim", "a2a3"):
            raise ValueError(f"Invalid platform {self.platform!r}. Expected 'a2a3sim' or 'a2a3'.")


@dataclass
class RunResult:
    """Result returned by :func:`run`.

    Attributes:
        passed: ``True`` if the program executed and results matched the golden
            reference within the configured tolerances.
        error: Human-readable error message when ``passed`` is ``False``.
        execution_time: Wall-clock time in seconds for the full run (compile +
            execute + validate).
    """

    passed: bool
    error: str | None = None
    execution_time: float | None = None

    def __str__(self) -> str:
        if self.passed:
            return f"PASS ({self.execution_time:.2f}s)" if self.execution_time else "PASS"
        msg = "FAIL"
        if self.error:
            msg += f": {self.error}"
        return msg


def run(
    program: Any,
    tensor_specs: list[TensorSpec],
    golden: Callable,
    config: RunConfig | None = None,
) -> RunResult:
    """Compile *program* and run it on device, validating against *golden*.

    The full pipeline executed by this function:

    1. Call :func:`ir.compile` to generate CCE C++ kernel and orchestration files.
    2. Patch the orchestration file with the required ``runtime.h`` header.
    3. Write a ``golden.py`` file from *tensor_specs* and *golden*.
    4. Invoke Simpler's ``CodeRunner`` to compile, load, execute, and validate.

    Args:
        program: A ``@pl.program`` decorated class or an ``ir.Program`` object.
        tensor_specs: Ordered list of tensor specifications.  The order must match
            the parameter order of the program's orchestration function.
        golden: A function with signature ``golden(tensors, params)`` that
            computes the expected outputs in-place (writes to
            ``tensors[output_name]``).  The function name does not matter.
        config: Run configuration.  Uses default :class:`RunConfig` if ``None``.

    Returns:
        :class:`RunResult` with ``passed=True`` on success, or ``passed=False``
        with an ``error`` message on failure.

    Example:
        >>> result = run(MyProgram, specs, my_golden, RunConfig(platform="a2a3sim"))
        >>> assert result.passed, str(result)
    """
    if config is None:
        config = RunConfig()

    start_time = time.time()
    if config.work_dir is None:
        use_temp = True
        work_dir = Path(tempfile.mkdtemp(prefix="pypto_run_"))
    else:
        use_temp = False
        work_dir = Path(config.work_dir).resolve()
    if not use_temp:
        work_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Set backend for code generation
        set_backend_type(config.backend_type)

        # 2. Compile: generates kernels/, orchestration/, kernel_config.py
        ir.compile(
            program,
            output_dir=str(work_dir),
            strategy=config.strategy,
            dump_passes=config.dump_passes,
            backend_type=config.backend_type,
        )

        # 3. Patch orchestration files with required headers
        _patch_orchestration_headers(work_dir)

        # 4. Write golden.py
        golden_path = work_dir / "golden.py"
        write_golden(tensor_specs, golden, golden_path, rtol=config.rtol, atol=config.atol)

        # 5. Execute via Simpler's CodeRunner (lazy import — optional dependency)
        # Automatically add Simpler paths if SIMPLER_ROOT is set (mirrors conftest.py behaviour)
        import os  # noqa: PLC0415
        import sys  # noqa: PLC0415

        simpler_root = os.environ.get("SIMPLER_ROOT")
        if simpler_root:
            for sub in ("examples/scripts", "python"):
                p = str(Path(simpler_root) / sub)
                if p not in sys.path:
                    sys.path.insert(0, p)

        from code_runner import CodeRunner  # type: ignore[import]  # noqa: PLC0415

        code_runner = CodeRunner(
            kernels_dir=str(work_dir),
            golden_path=str(golden_path),
            platform=config.platform,
            device_id=config.device_id,
        )
        code_runner.run()

        return RunResult(passed=True, execution_time=time.time() - start_time)

    except Exception as exc:
        return RunResult(
            passed=False,
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            execution_time=time.time() - start_time,
        )
    finally:
        if use_temp and work_dir.exists():
            shutil.rmtree(work_dir)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _patch_orchestration_headers(work_dir: Path) -> None:
    """Add ``runtime.h`` and ``<iostream>`` includes to orchestration C++ files.

    Simpler's CodeRunner requires these headers in the orchestration translation
    unit.  They are added here rather than in the code generator so that the
    compiler back-end remains unaware of runtime-specific requirements.

    Args:
        work_dir: Root output directory produced by :func:`ir.compile`.
    """
    orch_dir = work_dir / "orchestration"
    if not orch_dir.exists():
        return
    for cpp_file in orch_dir.glob("*.cpp"):
        _add_headers_to_file(cpp_file)


def _add_headers_to_file(cpp_file: Path) -> None:
    """Insert missing ``runtime.h`` / ``<iostream>`` headers into *cpp_file*.

    Args:
        cpp_file: Path to a C++ source file that may be missing the headers.
    """
    content = cpp_file.read_text(encoding="utf-8")

    has_runtime_h = '#include "runtime.h"' in content
    has_iostream = "#include <iostream>" in content

    if has_runtime_h and has_iostream:
        return  # Nothing to do

    headers: list[str] = []
    if not has_runtime_h:
        headers.append('#include "runtime.h"')
    if not has_iostream:
        headers.append("#include <iostream>")

    # Find the first non-comment, non-blank line as the insertion point.
    lines = content.splitlines(keepends=True)
    insert_pos = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith(("//", "/*", "*")):
            insert_pos = i
            break

    header_block = "\n".join(headers) + "\n"
    if insert_pos > 0:
        header_block += "\n"

    lines.insert(insert_pos, header_block)
    cpp_file.write_text("".join(lines), encoding="utf-8")
