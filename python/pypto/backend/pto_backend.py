# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PTO backend driver.

Orchestrates the full PTO backend output pipeline:

- **Kernel files**: InCore functions go through C++ PTOCodegen (IR → MLIR) → ptoas → kernel wrapper
- **Orchestration**: Shared C++ orchestration codegen (PTO2 runtime API)
- **Config**: Generates kernel_config.py with runtime/orchestration/kernel metadata

Entry point: ``generate(program, output_dir) -> dict[str, str]``
"""

import logging
import os
import re
import shutil
import subprocess
import textwrap
from collections import OrderedDict

from pypto.pypto_core import backend as _backend_core
from pypto.pypto_core import codegen as _codegen_core
from pypto.pypto_core import ir as _ir_core

logger = logging.getLogger(__name__)

_PTOAS_RELEASE_URL = "https://github.com/zhangstevenunity/PTOAS/releases"


class PartialCodegenError(RuntimeError):
    """Codegen failed after producing some output files."""

    def __init__(self, message: str, files: dict[str, str]) -> None:
        super().__init__(message)
        self.files = files


def _get_error_summary(exc: Exception, func_name: str) -> str:
    """Extract the first meaningful line from an exception, without the function name.

    Strips the C++ Traceback tail, takes only the first line, and removes
    occurrences of *func_name* so the summary column focuses on the error
    itself instead of repeating the function name.
    """
    msg = str(exc)
    traceback_marker = msg.find("\n\nC++ Traceback")
    if traceback_marker != -1:
        msg = msg[:traceback_marker]
    first_line = msg.split("\n", 1)[0].strip()
    if not first_line:
        return type(exc).__name__
    summary = first_line.replace(func_name, "").replace("  ", " ").strip()
    return summary or first_line


def _format_error_report(
    errors: list[tuple[str, Exception]],
    output_dir: str,
) -> str:
    """Build a concise error summary table and write full details to a log file.

    Each failed function is shown on its own row, with ``Function`` as the
    first column and ``Error`` as the second column. Returns the summary string
    for use in the ``RuntimeError`` message.
    """
    max_error_col = 60

    summaries = OrderedDict((name, _get_error_summary(exc, name)) for name, exc in errors)
    longest_error = max(len(summary) for summary in summaries.values())
    error_col_width = min(longest_error, max_error_col) + 2
    error_col_width = max(error_col_width, len("Error") + 2)
    func_col_width = max(len(n) for n, _ in errors) + 2
    func_col_width = max(func_col_width, len("Function") + 2)

    lines: list[str] = [f"{len(errors)} function(s) failed to compile:\n"]
    lines.append(f"  {'Function':<{func_col_width}}| {'Error'}")
    lines.append(f"  {'-' * func_col_width}+{'-' * error_col_width}")

    sep_line = f"  {'-' * func_col_width}+{'-' * error_col_width}"
    for func_name, summary in summaries.items():
        wrapped = textwrap.wrap(summary, width=max_error_col) or [summary]
        lines.append(f"  {func_name:<{func_col_width}}| {wrapped[0]}")
        for err_part in wrapped[1:]:
            lines.append(f"  {'':<{func_col_width}}| {err_part}")
        lines.append(sep_line)

    summary_text = "\n".join(lines)

    report_dir = os.path.join(output_dir, "report")
    detail_path = os.path.join(report_dir, "codegen_errors.txt")
    separator = "\n" + "=" * 72 + "\n"
    detail_parts = [f"  [{name}]\n{exc}" for name, exc in errors]
    detail_content = summary_text + "\n\n" + separator.join(detail_parts)
    try:
        os.makedirs(report_dir, exist_ok=True)
        with open(detail_path, "w") as f:
            f.write(detail_content)
        lines.append(f"\n  Full details: {detail_path}")
    except OSError:
        pass

    return "\n".join(lines)


def _run_ptoas(
    pto_path: str,
    output_path: str,
    ptoas_flags: list[str] | None = None,
) -> None:
    """Run the ptoas tool to compile a .pto file to C++.

    Locates ptoas via PTOAS_ROOT env var (``$PTOAS_ROOT/ptoas``) or PATH fallback.

    Args:
        pto_path: Path to the input .pto file
        output_path: Path for the output .cpp file
        ptoas_flags: Additional flags to pass to ptoas (optional)

    Raises:
        FileNotFoundError: If the ptoas binary cannot be found
        RuntimeError: If ptoas compilation fails
    """
    ptoas_root = os.environ.get("PTOAS_ROOT")
    if ptoas_root:
        ptoas_bin = os.path.join(ptoas_root, "ptoas")
        if not (os.path.isfile(ptoas_bin) and os.access(ptoas_bin, os.X_OK)):
            raise FileNotFoundError(
                f"PTOAS_ROOT is set to '{ptoas_root}' but '{ptoas_bin}' does not exist or is not executable. "
            )
    else:
        ptoas_bin = shutil.which("ptoas")
        if not ptoas_bin:
            raise FileNotFoundError(
                "ptoas binary not found. Set PTOAS_ROOT to the extracted release directory, "
                f"or add ptoas to your PATH.\nDownload from: {_PTOAS_RELEASE_URL}"
            )

    cmd = [ptoas_bin, pto_path, "-o", output_path]
    if ptoas_flags:
        cmd.extend(ptoas_flags)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ptoas compilation timed out after {exc.timeout}s") from exc
    if result.returncode != 0:
        raise RuntimeError(f"ptoas compilation failed: {result.stderr.strip()}")


_KERNEL_HEADER = """\
// Kernel Function: {func_name}
// Generated by PyPTO IR Compiler (PTO backend)

#include <cstdint>
#include <pto/pto-inst.hpp>
#include "tensor.h"

using namespace pto;

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

"""


def _preprocess_ptoas_output(content: str) -> str:
    """Strip includes/using and make functions static in ptoas output.

    Removes the header lines that the wrapper already provides, and replaces
    ``__global__ AICORE void`` with ``static __aicore__ void`` so the wrapper's
    ``kernel_entry`` is the actual entry point.
    """
    lines = content.splitlines(keepends=True)
    filtered: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#include") and (
            "pto-inst" in stripped or "cstdint" in stripped or "tensor.h" in stripped
        ):
            continue
        if stripped == "using namespace pto;":
            continue
        filtered.append(line)
    result = "".join(filtered)
    result = re.sub(r"__global__\s+AICORE\s+void", "static __aicore__ void", result)
    return result


def _generate_arg_unpacking(func: _ir_core.Function) -> tuple[str, list[str]]:
    """Generate C++ code to unpack ``int64_t* args`` into typed locals.

    Returns:
        A tuple of (C++ unpacking code, list of local variable names in order).
    """
    lines: list[str] = []
    var_names: list[str] = []

    for i, param in enumerate(func.params):
        param_name = param.name_hint
        param_type = param.type

        if isinstance(param_type, _ir_core.TensorType):
            c_type = param_type.dtype.to_c_type_string()
            lines.append(f"    // Unpack tensor: {param_name}")
            lines.append(
                f"    __gm__ TensorData* {param_name}_tensor = "
                f"reinterpret_cast<__gm__ TensorData*>(args[{i}]);"
            )
            lines.append(
                f"    __gm__ {c_type}* {param_name} = "
                f"reinterpret_cast<__gm__ {c_type}*>("
                f"{param_name}_tensor->buffer.addr) + {param_name}_tensor->start_offset;"
            )
            var_names.append(param_name)

        elif isinstance(param_type, _ir_core.ScalarType):
            c_type = param_type.dtype.to_c_type_string()
            lines.append(f"    // Unpack scalar: {param_name}")
            lines.append(f"    union {{ uint64_t u64; {c_type} val; }} {param_name}_conv;")
            lines.append(f"    {param_name}_conv.u64 = args[{i}];")
            lines.append(f"    {c_type} {param_name} = {param_name}_conv.val;")
            var_names.append(param_name)

        else:
            raise ValueError(
                f"Unsupported parameter type for wrapper generation: "
                f"{type(param_type).__name__} in function {func.name}"
            )

        lines.append("")  # blank line between params

    # Extract dynamic dimension values from tensor structs (shapes[] holds current view shape at runtime)
    seen_dyn_vars: set[str] = set()
    for param in func.params:
        if not isinstance(param.type, _ir_core.TensorType):
            continue
        for dim_idx, dim in enumerate(param.type.shape):
            if isinstance(dim, _ir_core.Var) and dim.name_hint not in seen_dyn_vars:
                var_name = dim.name_hint
                seen_dyn_vars.add(var_name)
                lines.append(f"    // Extract dynamic dim: {var_name}")
                lines.append(
                    f"    int64_t {var_name} = static_cast<int64_t>"
                    f"({param.name_hint}_tensor->shapes[{dim_idx}]);"
                )
                lines.append("")
                var_names.append(var_name)

    return "\n".join(lines), var_names


def _generate_kernel_wrapper(func: _ir_core.Function, ptoas_code: str) -> str:
    """Generate a complete CCE-compatible wrapper file for one InCore function.

    Combines:
    1. CCE kernel header (includes, macros)
    2. Preprocessed ptoas code (static, no duplicate includes)
    3. ``kernel_entry`` wrapper with arg unpacking and forward call
    """
    header = _KERNEL_HEADER.format(func_name=func.name)
    ptoas_body = _preprocess_ptoas_output(ptoas_code)
    unpacking_code, var_names = _generate_arg_unpacking(func)
    call_args = ", ".join(var_names)

    wrapper_func = (
        "// --- CCE-compatible entry point ---\n"
        'extern "C" __aicore__ __attribute__((always_inline)) '
        "void kernel_entry(__gm__ int64_t* args)\n"
        "{\n"
        f"{unpacking_code}\n"
        f"    // Forward to ptoas-generated function\n"
        f"    {func.name}({call_args});\n"
        "}\n"
    )

    return f"{header}\n// --- ptoas-generated code ---\n{ptoas_body}\n{wrapper_func}"


def _generate_config_file(
    orch_func_name: str,
    func_name_to_id: dict[str, int],
    func_name_to_core_type: dict[str, _ir_core.CoreType],
) -> str:
    """Generate kernel_config.py content matching the CCE format."""
    lines = [
        "# Kernel and Orchestration Configuration\n",
        "from pathlib import Path\n",
        "_ROOT_DIR = Path(__file__).parent\n",
        "# Runtime configuration for tensormap_and_ringbuffer",
        "# This runtime requires 4 AICPU threads (3 schedulers + 1 orchestrator on thread 3)",
        "RUNTIME_CONFIG = {",
        '\t"runtime": "tensormap_and_ringbuffer",',
        '\t"aicpu_thread_num": 4,',
        '\t"orch_thread_num": 1,',
        '\t"block_dim": 24,',
        "}\n",
        "ORCHESTRATION = {",
        f'\t"source": str(_ROOT_DIR / "orchestration" / "{orch_func_name}.cpp"),',
        '\t"function_name": "aicpu_orchestration_entry"',
        "}\n",
        "KERNELS = [",
    ]

    for name, func_id in sorted(func_name_to_id.items(), key=lambda x: x[1]):
        core_type = func_name_to_core_type[name]
        ct_str = "aiv" if core_type == _ir_core.CoreType.VECTOR else "aic"
        lines.append(
            f'\t{{"func_id": {func_id}, '
            f'"source": str(_ROOT_DIR / "kernels" / "{ct_str}" / "{name}.cpp"), '
            f'"core_type": "{ct_str}"}},'
        )

    lines.append("]")
    return "\n".join(lines) + "\n"


def _extract_group_member_names(
    group_func: _ir_core.Function,
) -> list[str]:
    """Extract function names called by a Group function from its body."""
    names: list[str] = []
    stmts = _ir_core.flatten_to_stmts(group_func.body)
    for stmt in stmts:
        call = None
        if isinstance(stmt, _ir_core.EvalStmt):
            call = stmt.expr
        elif isinstance(stmt, _ir_core.AssignStmt):
            call = stmt.value
        if isinstance(call, _ir_core.Call) and isinstance(call.op, _ir_core.GlobalVar):
            names.append(call.op.name)
    return names


def _build_group_mapping(
    program: _ir_core.Program,
) -> tuple[dict[str, list[_ir_core.Function]], list[_ir_core.Function]]:
    """Partition InCore functions into groups and ungrouped.

    Returns:
        (groups, ungrouped) where groups maps group_name to list of member
        InCore functions, and ungrouped is a list of InCore functions not
        belonging to any group.
    """
    func_by_name: dict[str, _ir_core.Function] = {f.name: f for f in program.functions.values()}
    grouped_names: set[str] = set()
    groups: dict[str, list[_ir_core.Function]] = {}

    for func in program.functions.values():
        if func.func_type != _ir_core.FunctionType.Group:
            continue
        member_names = _extract_group_member_names(func)
        members = [func_by_name[n] for n in member_names if n in func_by_name]
        if members:
            groups[func.name] = members
            grouped_names.update(n for n in member_names if n in func_by_name)

    ungrouped = [
        f
        for f in program.functions.values()
        if _ir_core.is_incore_type(f.func_type) and f.name not in grouped_names
    ]
    return groups, ungrouped


def generate(
    transformed_program: _ir_core.Program,
    output_dir: str,
    skip_ptoas: bool = False,
) -> dict[str, str]:
    """Generate all PTO backend output files (kernels + orchestration + config).

    Analogous to ``CCECodegen.generate()`` — returns a complete file map for the
    PTO backend. Kernel InCore functions go through the ptoas pipeline by default;
    when ``skip_ptoas=True``, the raw MLIR (.pto) content is returned directly
    without invoking ptoas.

    Args:
        transformed_program: Program after pass pipeline
        output_dir: Base output directory (used for ptoas intermediates when skip_ptoas=False)
        skip_ptoas: When True, skip the ptoas compilation step and return raw MLIR
            content in result_files with .pto extension instead of compiled .cpp wrappers.

    Returns:
        Dict mapping relative file paths to their content.
    """
    result_files: dict[str, str] = {}
    orch_func: _ir_core.Function | None = None
    errors: list[tuple[str, Exception]] = []

    for func in transformed_program.functions.values():
        if func.func_type == _ir_core.FunctionType.Orchestration:
            orch_func = func

    groups, ungrouped = _build_group_mapping(transformed_program)

    def _codegen_single_function(func: _ir_core.Function, pto_code: str) -> None:
        """Compile one InCore function's MLIR through ptoas and emit its wrapper."""
        core_type = _codegen_core.infer_function_core_type(func)
        ct_str = "aiv" if core_type == _ir_core.CoreType.VECTOR else "aic"

        if skip_ptoas:
            kernel_rel = os.path.join("kernels", ct_str, f"{func.name}.pto")
            result_files[kernel_rel] = pto_code
        else:
            ptoas_dir = os.path.join(output_dir, "ptoas")
            os.makedirs(ptoas_dir, exist_ok=True)

            pto_path = os.path.join(ptoas_dir, f"{func.name}.pto")
            with open(pto_path, "w") as f:
                f.write(pto_code)

            cpp_path = os.path.join(ptoas_dir, f"{func.name}.cpp")
            _run_ptoas(
                pto_path,
                cpp_path,
                ptoas_flags=[
                    "--enable-insert-sync",
                    "--pto-level=level3",
                ]
                + (
                    ["--pto-arch", "a5"]
                    if _backend_core.get_backend_type() == _backend_core.BackendType.Ascend950
                    else []
                ),
            )

            with open(cpp_path) as f:
                ptoas_cpp = f.read()

            wrapper_code = _generate_kernel_wrapper(func, ptoas_cpp)
            kernel_rel = os.path.join("kernels", ct_str, f"{func.name}.cpp")
            result_files[kernel_rel] = wrapper_code

    # Grouped functions: one MLIR module per group
    for group_name, members in groups.items():
        try:
            grouped_program = _ir_core.Program(members, group_name, transformed_program.span)
            codegen_instance = _codegen_core.PTOCodegen()
            pto_code = codegen_instance.generate(grouped_program)

            if skip_ptoas:
                # One .pto file per group, keyed by group name
                kernel_rel = os.path.join("kernels", f"{group_name}.pto")
                result_files[kernel_rel] = pto_code
            else:
                # ptoas: compile grouped .pto, then generate per-function wrappers
                ptoas_dir = os.path.join(output_dir, "ptoas")
                os.makedirs(ptoas_dir, exist_ok=True)

                pto_path = os.path.join(ptoas_dir, f"{group_name}.pto")
                with open(pto_path, "w") as f:
                    f.write(pto_code)

                cpp_path = os.path.join(ptoas_dir, f"{group_name}.cpp")
                _run_ptoas(
                    pto_path,
                    cpp_path,
                    ptoas_flags=[
                        "--enable-insert-sync",
                        "--pto-level=level3",
                    ]
                    + (
                        ["--pto-arch", "a5"]
                        if _backend_core.get_backend_type() == _backend_core.BackendType.Ascend950
                        else []
                    ),
                )

                with open(cpp_path) as f:
                    ptoas_cpp = f.read()

                for func in members:
                    wrapper_code = _generate_kernel_wrapper(func, ptoas_cpp)
                    core_type = _codegen_core.infer_function_core_type(func)
                    ct_str = "aiv" if core_type == _ir_core.CoreType.VECTOR else "aic"
                    kernel_rel = os.path.join("kernels", ct_str, f"{func.name}.cpp")
                    result_files[kernel_rel] = wrapper_code
        except Exception as e:
            func_names = ", ".join(m.name for m in members)
            logger.error("Failed to compile group '%s' [%s]: %s", group_name, func_names, e)
            errors.append((group_name, e))

    # Ungrouped functions: one MLIR module per function (existing behavior)
    for func in ungrouped:
        try:
            single_program = _ir_core.Program([func], func.name, transformed_program.span)
            codegen_instance = _codegen_core.PTOCodegen()
            pto_code = codegen_instance.generate(single_program)
            _codegen_single_function(func, pto_code)
        except Exception as e:
            logger.error("Failed to compile function '%s': %s", func.name, e)
            errors.append((func.name, e))

    # Orchestration + config (shared codegen with CCE)
    if orch_func is not None:
        try:
            orch_result = _codegen_core.generate_orchestration(transformed_program, orch_func)
            result_files[f"orchestration/{orch_func.name}.cpp"] = (
                f"// Orchestration Function: {orch_func.name}\n"
                f"// Generated by PyPTO IR Compiler\n\n"
                f"{orch_result.code}"
            )
            if not skip_ptoas:
                result_files["kernel_config.py"] = _generate_config_file(
                    orch_func.name,
                    orch_result.func_name_to_id,
                    orch_result.func_name_to_core_type,
                )
        except Exception as e:
            logger.error("Failed to generate orchestration '%s': %s", orch_func.name, e)
            errors.append((orch_func.name, e))

    if errors:
        report = _format_error_report(errors, output_dir)
        if result_files:
            raise PartialCodegenError(report, result_files)
        raise RuntimeError(report)

    return result_files
