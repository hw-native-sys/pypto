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
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any

from pypto.compile_profiling import CompileProfiler, StageRecord
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


_ERROR_LINE_RE = re.compile(r"(?:^Error:|\berror:)", re.IGNORECASE)


def _strip_function_name_prefix(summary: str, func_name: str) -> str:
    """Remove a leading function-name prefix without corrupting file paths."""
    prefixes = (
        f"Failed to compile function '{func_name}':",
        f'Failed to compile function "{func_name}":',
        f"{func_name}:",
        func_name,
    )
    for prefix in prefixes:
        if summary.startswith(prefix):
            stripped = summary[len(prefix) :].strip()
            if stripped:
                return stripped
    return summary


def _get_error_summary(exc: Exception, func_name: str) -> str:
    """Extract the most useful error line from an exception.

    Strips the C++ Traceback tail, prefers the first line that contains an
    actual error marker, and removes only a leading *func_name* prefix so file
    paths remain intact.
    """
    msg = str(exc)
    traceback_marker = msg.find("\n\nC++ Traceback")
    if traceback_marker != -1:
        msg = msg[:traceback_marker]
    lines = [line.strip() for line in msg.splitlines() if line.strip()]
    if not lines:
        return type(exc).__name__

    first_line = lines[0]
    ptoas_prefix = "ptoas compilation failed:"
    if first_line.startswith(ptoas_prefix):
        first_detail = first_line[len(ptoas_prefix) :].strip()
        detail_lines = []
        if first_detail:
            detail_lines.append(first_detail)
        detail_lines.extend(lines[1:])
        for line in detail_lines:
            if _ERROR_LINE_RE.search(line):
                return f"{ptoas_prefix} {line}"

    for line in lines:
        if _ERROR_LINE_RE.search(line):
            return _strip_function_name_prefix(line, func_name)

    return _strip_function_name_prefix(first_line, func_name)


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

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#if defined(__CPU_SIM)
#define __aicore__
#else
#define __aicore__ [aicore]
#endif
#endif

{subblock_override}#include <pto/pto-inst.hpp>
#include "tensor.h"
{spmd_override}

using namespace pto;

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
    # Non-mixed kernels: optional __global__ AICORE void -> static __aicore__ void.
    result = re.sub(r"(?:__global__\s+)?AICORE\s+void", "static __aicore__ void", result)
    # Mixed-kernel sub-functions and helpers: normalize remaining AICORE qualifiers.
    result = re.sub(r"\bAICORE\b", "__aicore__", result)
    return result


def _generate_arg_unpacking(func: _ir_core.Function, *, uses_spmd: bool = False) -> tuple[str, list[str]]:
    """Generate C++ code to unpack ``int64_t* args`` into typed locals.

    Args[] are dispatched in tensors-first order (all tensors, then scalars),
    matching the PTOParam dispatch convention and the MLIR func.func signature
    order emitted by PTOCodegen. The returned var_names list is also in
    tensors-first order, matching the compiled ptoas function parameter order.

    Returns:
        A tuple of (C++ unpacking code, list of local variable names in
        tensors-first order).
    """
    lines: list[str] = []
    var_names: list[str] = []

    # Separate params into tensors and scalars for tensors-first dispatch order
    tensor_params = [p for p in func.params if isinstance(p.type, _ir_core.TensorType)]
    scalar_params = [p for p in func.params if isinstance(p.type, _ir_core.ScalarType)]
    other_params = [
        p for p in func.params if not isinstance(p.type, (_ir_core.TensorType, _ir_core.ScalarType))
    ]
    if other_params:
        raise ValueError(
            f"Unsupported parameter type(s) for wrapper generation in function {func.name}: "
            + ", ".join(f"{p.name_hint}: {type(p.type).__name__}" for p in other_params)
        )

    scalar_start_idx = len(tensor_params)

    # Unpack tensors: args[0..N_tensors-1]
    for i, param in enumerate(tensor_params):
        param_name = param.name_hint
        assert isinstance(param.type, _ir_core.TensorType)
        c_type = param.type.dtype.to_c_type_string()
        lines.append(f"    // Unpack tensor: {param_name}")
        lines.append(f"    __gm__ Tensor* {param_name}_tensor = reinterpret_cast<__gm__ Tensor*>(args[{i}]);")
        if param_name == "__gm_pipe_buffer" and uses_spmd:
            lines.append("    // SPMD: shard GM pipe workspace by logical block_idx to avoid overlap.")
            lines.append("    int64_t __pypto_gm_block_num = static_cast<int64_t>(__pypto_spmd_block_num);")
            lines.append("    if (__pypto_gm_block_num <= 0) __pypto_gm_block_num = 1;")
            lines.append(
                f"    int64_t __pypto_gm_total_elems = static_cast<int64_t>({param_name}_tensor->shapes[0]);"
            )
            lines.append(
                "    int64_t __pypto_gm_elems_per_block = __pypto_gm_total_elems / __pypto_gm_block_num;"
            )
            lines.append(
                "    int64_t __pypto_gm_block_offset = static_cast<int64_t>(__pypto_spmd_block_idx) * "
                "__pypto_gm_elems_per_block;"
            )
            lines.append(
                f"    __gm__ {c_type}* {param_name} = "
                f"reinterpret_cast<__gm__ {c_type}*>({param_name}_tensor->buffer.addr) + "
                f"{param_name}_tensor->start_offset + __pypto_gm_block_offset;"
            )
        else:
            lines.append(
                f"    __gm__ {c_type}* {param_name} = "
                f"reinterpret_cast<__gm__ {c_type}*>("
                f"{param_name}_tensor->buffer.addr) + {param_name}_tensor->start_offset;"
            )
        lines.append("")
        var_names.append(param_name)

    # Unpack scalars: args[N_tensors..]
    for j, param in enumerate(scalar_params):
        param_name = param.name_hint
        assert isinstance(param.type, _ir_core.ScalarType)
        c_type = param.type.dtype.to_c_type_string()
        arg_idx = scalar_start_idx + j
        lines.append(f"    // Unpack scalar: {param_name}")
        lines.append(f"    union {{ uint64_t u64; {c_type} val; }} {param_name}_conv;")
        lines.append(f"    {param_name}_conv.u64 = args[{arg_idx}];")
        lines.append(f"    {c_type} {param_name} = {param_name}_conv.val;")
        lines.append("")
        var_names.append(param_name)

    # Extract dynamic dimension values from tensor structs (shapes[] holds current view shape at runtime).
    # Deduplicate by IR variable identity (same_as), not by name_hint, so that
    # distinct Var objects sharing a cosmetic name are never incorrectly merged.
    seen_dyn_vars: list[_ir_core.Var] = []
    used_c_names: set[str] = set(var_names)
    used_c_names.update(f"{p.name_hint}_tensor" for p in tensor_params)
    used_c_names.update(f"{p.name_hint}_conv" for p in scalar_params)
    for param in tensor_params:
        assert isinstance(param.type, _ir_core.TensorType)
        for dim_idx, dim in enumerate(param.type.shape):
            if isinstance(dim, _ir_core.Var) and not any(dim.same_as(v) for v in seen_dyn_vars):
                seen_dyn_vars.append(dim)
                var_name = dim.name_hint
                if var_name in used_c_names:
                    suffix = 1
                    while f"{var_name}_{suffix}" in used_c_names:
                        suffix += 1
                    var_name = f"{var_name}_{suffix}"
                used_c_names.add(var_name)
                lines.append(f"    // Extract dynamic dim: {var_name}")
                lines.append(
                    f"    int64_t {var_name} = static_cast<int64_t>"
                    f"({param.name_hint}_tensor->shapes[{dim_idx}]);"
                )
                lines.append("")
                var_names.append(var_name)

    return "\n".join(lines), var_names


def _get_fixed_subblock_id(func: _ir_core.Function) -> int | None:
    """Return the fixed lane id for legacy split-specialized AIV wrappers."""
    split_mode = getattr(func, "split", None)
    if split_mode is None or split_mode == _ir_core.SplitMode.NONE:
        return None
    if _codegen_core.infer_function_core_type(func) != _ir_core.CoreType.VECTOR:
        return None
    return 1 if func.name.endswith("__aiv1") else None


def _uses_dynamic_subblock_id(func: _ir_core.Function) -> bool:
    """Return whether the function reads subblock id from the runtime lane context."""
    stmts = _ir_core.flatten_to_stmts(func.body)
    for stmt in stmts:
        call = None
        if isinstance(stmt, _ir_core.EvalStmt):
            call = stmt.expr
        elif isinstance(stmt, _ir_core.AssignStmt):
            call = stmt.value
        if not isinstance(call, _ir_core.Call):
            continue
        op = getattr(call, "op", None)
        if isinstance(op, _ir_core.Op) and op.name == "tile.get_subblock_idx":
            return True
    return False


def _requires_dual_aiv_dispatch(func: _ir_core.Function) -> bool:
    """Return whether the function must be dispatched on both AIV lanes."""
    split_mode = getattr(func, "split", None)
    if split_mode is not None and split_mode != _ir_core.SplitMode.NONE:
        return True
    return bool(getattr(func, "attrs", {}).get("dual_aiv_dispatch", False))


_SPMD_BLOCK_OPS = frozenset({"tile.get_block_idx", "tile.get_block_num"})


def _uses_spmd_block_ops(func: _ir_core.Function) -> bool:
    """Return whether the function uses SPMD block identity ops (get_block_idx/num).

    These ops compile to ccec built-ins that return physical core indices.
    In the tensormap_and_ringbuffer runtime the logical block index must be
    read from the dispatch payload via ``get_block_idx(args)`` / ``get_block_num(args)``
    (defined in ``intrinsic.h``), so the wrapper needs a macro bridge.
    """

    class _SpmdOpFinder(_ir_core.IRVisitor):
        def __init__(self) -> None:
            super().__init__()
            self.found = False

        def visit_call(self, op: _ir_core.Call) -> None:
            if self.found:
                return
            ir_op = getattr(op, "op", None)
            if isinstance(ir_op, _ir_core.Op) and ir_op.name in _SPMD_BLOCK_OPS:
                self.found = True
                return
            super().visit_call(op)

    finder = _SpmdOpFinder()
    finder.visit_stmt(func.body)
    return finder.found


def _needs_runtime_subblock_bridge(func: _ir_core.Function) -> bool:
    """Return whether A2A3 split AIV wrappers must source subblock id from runtime context."""
    if not _requires_dual_aiv_dispatch(func):
        return False
    if _codegen_core.infer_function_core_type(func) != _ir_core.CoreType.VECTOR:
        return False
    if not _backend_core.get_handler().requires_runtime_subblock_bridge():
        return False
    return _uses_dynamic_subblock_id(func)


def _generate_kernel_header(func: _ir_core.Function, *, uses_spmd: bool | None = None) -> str:
    """Generate the wrapper header, including split lane overrides when needed."""
    fixed_subblock_id = _get_fixed_subblock_id(func)
    subblock_override = ""
    if fixed_subblock_id is not None:
        subblock_override = textwrap.dedent(
            f"""\
            // Keep split-specialized AIV wrappers aligned with PTO-ISA pipe slot offsets.
            #if !defined(__CPU_SIM)
            #define PYPTO_FIXED_SUBBLOCK_ID {fixed_subblock_id}
            #define get_subblockid() PYPTO_FIXED_SUBBLOCK_ID
            #endif

            """
        )
    elif _needs_runtime_subblock_bridge(func):
        subblock_override = textwrap.dedent(
            """\
            #if !defined(__CPU_SIM)
            #include "intrinsic.h"

            // A2A3 mixed tasks run the same AIV kernel on two vector cores.
            // Bridge the runtime-provided lane id into PTO-ISA get_subblockid().
            [[block_local]] static int32_t pypto_runtime_subblock_id;
            #define get_subblockid() pypto_runtime_subblock_id
            #endif

            """
        )

    # SPMD: include intrinsic.h so the wrapper can call get_block_idx(args) /
    # get_block_num(args). Block identity now flows into the kernel as the
    # first two wrapper-passed parameters, so there is no macro shadow, no
    # [[block_local]] static / thread_local storage, and no __CPU_SIM fork.
    if uses_spmd is None:
        uses_spmd = _uses_spmd_block_ops(func)
    spmd_override = '#include "intrinsic.h"\n' if uses_spmd else ""

    return _KERNEL_HEADER.format(
        func_name=func.name,
        subblock_override=subblock_override,
        spmd_override=spmd_override,
    )


def _generate_kernel_wrapper(
    func: _ir_core.Function,
    ptoas_code: str,
    *,
    group_uses_spmd: bool = False,
) -> str:
    """Generate a complete kernel wrapper file for one InCore function.

    Combines:
    1. Kernel header (includes, macros)
    2. Preprocessed ptoas code (static, no duplicate includes)
    3. ``kernel_entry`` wrapper with arg unpacking and forward call
    """
    func_uses_spmd = _uses_spmd_block_ops(func)
    uses_spmd = group_uses_spmd or func_uses_spmd
    header = _generate_kernel_header(func, uses_spmd=uses_spmd)
    ptoas_body = _preprocess_ptoas_output(ptoas_code)
    unpacking_code, var_names = _generate_arg_unpacking(func, uses_spmd=uses_spmd)
    runtime_subblock_setup = ""
    if _needs_runtime_subblock_bridge(func):
        runtime_subblock_setup = (
            "#if !defined(__CPU_SIM)\n"
            "    // Read A2A3 mixed-task subblock id from runtime dispatch context\n"
            "    pypto_runtime_subblock_id = get_sub_block_id(args);\n"
            "#endif\n\n"
        )

    # Resolve SPMD block identity once from intrinsic.h::get_block_idx(args) /
    # get_block_num(args). Locals are declared whenever any function in the
    # group needs them (e.g., __gm_pipe_buffer sharding in _generate_arg_unpacking
    # references them even for non-SPMD members of an SPMD group).
    spmd_args_setup = ""
    if uses_spmd:
        spmd_args_setup = (
            "    // Read logical SPMD block identity from runtime dispatch payload\n"
            "    int32_t __pypto_spmd_block_idx = get_block_idx(args);\n"
            "    int32_t __pypto_spmd_block_num = get_block_num(args);\n\n"
        )

    # PTOCodegen appends two synthetic i32 params (block_idx, block_num) at
    # the end of the func.func signature when func itself uses
    # tile.get_block_idx / tile.get_block_num. Mirror that order in the call.
    call_args_list = list(var_names)
    if func_uses_spmd:
        call_args_list = call_args_list + ["__pypto_spmd_block_idx", "__pypto_spmd_block_num"]
    call_args = ", ".join(call_args_list)

    wrapper_func = (
        "// --- Kernel entry point ---\n"
        'extern "C" __aicore__ __attribute__((always_inline)) '
        "void kernel_entry(__gm__ int64_t* args)\n"
        "{\n"
        f"{runtime_subblock_setup}"
        f"{spmd_args_setup}"
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
    """Generate kernel_config.py content."""
    lines = [
        "# Kernel and Orchestration Configuration\n",
        "from pathlib import Path\n",
        "_ROOT_DIR = Path(__file__).parent\n",
        "# Runtime configuration for tensormap_and_ringbuffer",
        "# This runtime requires 4 AICPU threads (3 schedulers + 1 orchestrator on thread 3)",
        "RUNTIME_CONFIG = {",
        '\t"runtime": "tensormap_and_ringbuffer",',
        '\t"aicpu_thread_num": 4,',
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
            f'"name": "{name}", '
            f'"source": str(_ROOT_DIR / "kernels" / "{ct_str}" / "{name}.cpp"), '
            f'"core_type": "{ct_str}"}},'
        )

    lines.append("]")
    return "\n".join(lines) + "\n"


class _CallCollector(_ir_core.IRVisitor):
    """Collect all GlobalVar callee names reachable from a function body."""

    def __init__(self) -> None:
        super().__init__()
        self.callee_names: list[str] = []

    def visit_call(self, op: _ir_core.Call) -> None:
        if isinstance(op.op, _ir_core.GlobalVar):
            self.callee_names.append(op.op.name)
        super().visit_call(op)


def _extract_group_member_names(
    group_func: _ir_core.Function,
) -> list[str]:
    """Extract function names called by a Group function from its body."""
    collector = _CallCollector()
    collector.visit_stmt(group_func.body)
    return collector.callee_names


def _extract_peer_function_names(
    func: _ir_core.Function,
) -> list[str]:
    """Extract peer function names referenced by import_peer_buffer ops."""
    peer_names: list[str] = []
    seen_names: set[str] = set()
    stmts = _ir_core.flatten_to_stmts(func.body)
    for stmt in stmts:
        call = None
        if isinstance(stmt, _ir_core.EvalStmt):
            call = stmt.expr
        elif isinstance(stmt, _ir_core.AssignStmt):
            call = stmt.value
        if not isinstance(call, _ir_core.Call):
            continue
        op = getattr(call, "op", None)
        if not isinstance(op, _ir_core.Op) or op.name != "system.import_peer_buffer":
            continue
        kwargs = getattr(call, "kwargs", {}) or {}
        peer_func = kwargs.get("peer_func", "")
        if peer_func and peer_func not in seen_names:
            peer_names.append(peer_func)
            seen_names.add(peer_func)
    return peer_names


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


def _get_ptoas_flags() -> list[str]:
    """Build the common ptoas flag list for kernel compilation."""
    flags = [
        "--enable-insert-sync",
        "--pto-level=level3",
    ]
    flags.extend(_backend_core.get_handler().get_extra_ptoas_flags())
    return flags


def _get_kernel_output_path(
    func: _ir_core.Function,
    suffix: str,
) -> str:
    """Return the per-core kernel output path for a function."""
    core_type = _codegen_core.infer_function_core_type(func)
    ct_str = "aiv" if core_type == _ir_core.CoreType.VECTOR else "aic"
    return os.path.join("kernels", ct_str, f"{func.name}.{suffix}")


def _compile_pto_module(
    pto_code: str,
    unit_name: str,
    output_dir: str,
) -> str:
    """Run ptoas for one MLIR module and return the generated C++."""
    ptoas_dir = os.path.join(output_dir, "ptoas")
    os.makedirs(ptoas_dir, exist_ok=True)

    pto_path = os.path.join(ptoas_dir, f"{unit_name}.pto")
    with open(pto_path, "w") as f:
        f.write(pto_code)

    cpp_path = os.path.join(ptoas_dir, f"{unit_name}.cpp")
    _run_ptoas(
        pto_path,
        cpp_path,
        ptoas_flags=_get_ptoas_flags(),
    )

    with open(cpp_path) as f:
        return f.read()


def _emit_single_function_output(
    result_files: dict[str, str],
    func: _ir_core.Function,
    pto_code: str,
    output_dir: str,
    skip_ptoas: bool,
) -> None:
    """Emit output files for one InCore function."""
    suffix = "pto" if skip_ptoas else "cpp"
    kernel_rel = _get_kernel_output_path(func, suffix)
    if skip_ptoas:
        result_files[kernel_rel] = pto_code
        return

    ptoas_cpp = _compile_pto_module(pto_code, func.name, output_dir)
    result_files[kernel_rel] = _generate_kernel_wrapper(func, ptoas_cpp)


def _emit_group_output(
    result_files: dict[str, str],
    group_name: str,
    members: list[_ir_core.Function],
    pto_code: str,
    output_dir: str,
    skip_ptoas: bool,
) -> None:
    """Emit output files for one grouped MLIR module."""
    if skip_ptoas:
        result_files[os.path.join("kernels", f"{group_name}.pto")] = pto_code
        return

    ptoas_cpp = _compile_pto_module(pto_code, group_name, output_dir)
    group_uses_spmd = any(_uses_spmd_block_ops(f) for f in members)
    for func in members:
        result_files[_get_kernel_output_path(func, "cpp")] = _generate_kernel_wrapper(
            func, ptoas_cpp, group_uses_spmd=group_uses_spmd
        )


def _profiling_stage(prof: CompileProfiler | None, name: str) -> AbstractContextManager[Any]:
    """Return a profiling stage context if a profiler is active, else a no-op context."""
    if prof is not None:
        return prof.stage(name)
    return nullcontext()


# ---------------------------------------------------------------------------
# Parallel ptoas helpers
# ---------------------------------------------------------------------------

_CODEGEN_MAX_WORKERS_ENV = "PYPTO_CODEGEN_MAX_WORKERS"


def _get_max_workers() -> int | None:
    """Determine max worker threads for parallel ptoas invocations.

    Returns:
        ``None`` to use the ``ThreadPoolExecutor`` default, or an explicit
        thread count.  ``1`` means sequential execution (no thread pool).
    """
    env_val = os.environ.get(_CODEGEN_MAX_WORKERS_ENV, "").strip()
    if not env_val:
        return None  # ThreadPoolExecutor default
    try:
        n = int(env_val)
    except ValueError:
        logger.warning("Invalid %s='%s', using default", _CODEGEN_MAX_WORKERS_ENV, env_val)
        return None
    return max(1, n)


@dataclass
class _CodegenUnit:
    """One kernel codegen unit prepared for ptoas emission."""

    name: str
    pto_code: str  # MLIR from PTOCodegen (Phase 1 output)
    funcs: list[_ir_core.Function]
    is_group: bool
    stage_record: StageRecord  # profiling record (started in Phase 1)


@dataclass
class _EmitResult:
    """Result of emitting one codegen unit (ptoas + wrapper generation)."""

    name: str
    files: dict[str, str]
    ptoas_record: StageRecord
    error: Exception | None = None


def _emit_unit(
    unit: _CodegenUnit,
    output_dir: str,
    skip_ptoas: bool,
) -> _EmitResult:
    """Run ptoas + wrapper generation for one codegen unit.

    This is the Phase 2 worker — called from a thread pool or sequentially.
    PTOCodegen has already run; ``unit.pto_code`` contains the MLIR.
    """
    local_files: dict[str, str] = {}
    ptoas_record = StageRecord(name="ptoas", start=time.perf_counter())
    try:
        if unit.is_group:
            _emit_group_output(local_files, unit.name, unit.funcs, unit.pto_code, output_dir, skip_ptoas)
        else:
            _emit_single_function_output(local_files, unit.funcs[0], unit.pto_code, output_dir, skip_ptoas)
        ptoas_record.end = time.perf_counter()
        return _EmitResult(name=unit.name, files=local_files, ptoas_record=ptoas_record)
    except Exception as e:
        ptoas_record.end = time.perf_counter()
        if unit.is_group:
            func_names = ", ".join(m.name for m in unit.funcs)
            logger.error("Failed to compile group '%s' [%s]: %s", unit.name, func_names, e)
        else:
            logger.error("Failed to compile function '%s': %s", unit.name, e)
        return _EmitResult(name=unit.name, files={}, ptoas_record=ptoas_record, error=e)


def _merge_stage_record(prof: CompileProfiler | None, record: StageRecord) -> None:
    """Append a completed stage record into the profiler's current nesting level."""
    if prof is not None:
        prof.add_stage_record(record)


def _collect_emit_result(
    result: _EmitResult,
    unit: _CodegenUnit,
    prof: CompileProfiler | None,
    result_files: dict[str, str],
    errors: list[tuple[str, Exception]],
) -> None:
    """Finalize one emit result: merge profiling, collect files and errors."""
    unit.stage_record.children.append(result.ptoas_record)
    unit.stage_record.end = result.ptoas_record.end
    _merge_stage_record(prof, unit.stage_record)
    result_files.update(result.files)
    if result.error is not None:
        errors.append((result.name, result.error))


def _run_ptoas_phase(
    units: list[_CodegenUnit],
    output_dir: str,
    skip_ptoas: bool,
    prof: CompileProfiler | None,
    result_files: dict[str, str],
    errors: list[tuple[str, Exception]],
) -> None:
    """Phase 2: run ptoas for all codegen units, sequentially or in parallel."""
    max_workers = _get_max_workers()

    if max_workers == 1 or len(units) <= 1:
        for unit in units:
            result = _emit_unit(unit, output_dir, skip_ptoas)
            _collect_emit_result(result, unit, prof, result_files, errors)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_emit_unit, unit, output_dir, skip_ptoas) for unit in units]
            for unit, future in zip(units, futures):
                result = future.result()  # exceptions caught inside _emit_unit
                _collect_emit_result(result, unit, prof, result_files, errors)


def generate(
    transformed_program: _ir_core.Program,
    output_dir: str,
    skip_ptoas: bool = False,
) -> dict[str, str]:
    """Generate all PTO backend output files (kernels + orchestration + config).

    Analogous to the previous codegen pipeline — returns a complete file map for the
    PTO backend. Kernel InCore functions go through the ptoas pipeline by default;
    when ``skip_ptoas=True``, the raw MLIR (.pto) content is returned directly
    without invoking ptoas.

    For programs containing L3+ distributed functions (level=HOST or above),
    the distributed codegen path generates Python orchestration code alongside
    the standard PTO kernel artifacts.

    Args:
        transformed_program: Program after pass pipeline
        output_dir: Base output directory (used for ptoas intermediates when skip_ptoas=False)
        skip_ptoas: When True, skip the ptoas compilation step and return raw MLIR
            content in result_files with .pto extension instead of compiled .cpp wrappers.

    Returns:
        Dict mapping relative file paths to their content.
    """
    # Check for distributed functions (level >= HOST = Linqu level 3)
    has_distributed = any(
        f.level is not None and _ir_core.level_to_linqu_level(f.level) >= 3
        for f in transformed_program.functions.values()
    )

    if has_distributed:
        return _generate_with_distributed(transformed_program, output_dir, skip_ptoas)

    return _generate_single_chip(transformed_program, output_dir, skip_ptoas)


def _generate_with_distributed(
    transformed_program: _ir_core.Program,
    output_dir: str,
    skip_ptoas: bool,
) -> dict[str, str]:
    """Generate artifacts for a distributed (L3+) program.

    Output directory layout mirrors the distributed execution hierarchy::

      orchestration/host_orch.py        — L3 HOST orchestrator (Python)
      next_levels/{chip_task_name}/     — each L2 chip task (complete sub-dir)
          orchestration/{name}.cpp
          kernels/{core_type}/{kernel}.pto
          kernel_config.py
      sub_workers/{name}.py             — each SubWorker callable
    """
    result_files: dict[str, str] = {}

    # 1. L3 HOST orchestrator → orchestration/host_orch.py
    cg = _codegen_core.DistributedCodegen()
    orch_code = cg.generate(transformed_program)
    result_files["orchestration/host_orch.py"] = orch_code

    # 2. Each chip-level Orchestration → next_levels/{name}/...
    for func in transformed_program.functions.values():
        if func.func_type == _ir_core.FunctionType.Orchestration:
            chip_funcs = _collect_chip_task_functions(func, transformed_program)
            chip_program = _ir_core.Program(chip_funcs, func.name, transformed_program.span)
            chip_subdir = os.path.join(output_dir, "next_levels", func.name)
            chip_files = _generate_single_chip(chip_program, chip_subdir, skip_ptoas)
            for path, content in chip_files.items():
                result_files[f"next_levels/{func.name}/{path}"] = content

    # 3. HOST SubWorker functions → sub_workers/{name}.py
    # Only HOST-level (Linqu level >= 3) SubWorkers carry an InlineStmt body
    # captured by the decorator. Lower-level role=SubWorker kernels (e.g. CHIP
    # InCore promoted by auto-derive) keep their DSL body and are emitted by
    # chip-level codegen above.
    for func in transformed_program.functions.values():
        if (
            func.role == _ir_core.Role.SubWorker
            and func.level is not None
            and _ir_core.level_to_linqu_level(func.level) >= 3
        ):
            result_files[f"sub_workers/{func.name}.py"] = _emit_sub_worker_module(func)

    return result_files


def _emit_sub_worker_module(func: _ir_core.Function) -> str:
    """Emit a self-contained SubWorker module callable as ``fn(args: TaskArgs)``.

    The user's function body lives on ``func.body`` as an :class:`InlineStmt`
    captured by the decorator. The emitted module wraps the body in
    ``def _user_{name}(<params>)`` plus a dispatcher ``{name}(args)`` that
    unpacks tensors from ``TaskArgs``.
    """
    import textwrap  # noqa: PLC0415

    body = func.body
    if not isinstance(body, _ir_core.InlineStmt):
        raise RuntimeError(f"SubWorker '{func.name}' must have an InlineStmt body, got {type(body).__name__}")

    param_names = [p.name_hint for p in func.params]
    params_str = ", ".join(param_names)
    indented_body = textwrap.indent(body.body, "    ") if body.body else "    pass"
    unpack_block = (
        "\n".join(
            f"    {name} = _tensor_from_continuous(args.tensor({i}))" for i, name in enumerate(param_names)
        )
        or "    pass"
    )

    return (
        f'"""SubWorker: {func.name} — auto-generated, callable as fn(args: TaskArgs)."""\n'
        f"\n"
        f"import torch\n"
        f"\n"
        f"from pypto.runtime.distributed_runner import _tensor_from_continuous\n"
        f"\n"
        f"\n"
        f"def _user_{func.name}({params_str}):\n"
        f"{indented_body}\n"
        f"\n"
        f"\n"
        f"def {func.name}(args):\n"
        f"{unpack_block}\n"
        f"    _user_{func.name}({params_str})\n"
    )


def _collect_chip_task_functions(
    orch_func: _ir_core.Function,
    program: _ir_core.Program,
) -> list[_ir_core.Function]:
    """Collect ``orch_func`` and all chip-level callees reachable from its body.

    Walks the call graph starting from ``orch_func`` and returns any
    InCore/AIC/AIV/Group/Spmd function transitively called. This filters out
    chip kernels belonging to *other* orchestrations in multi-orchestration
    programs (e.g., L3 programs with multiple ``with pl.at(level=CHIP)``
    scopes), preventing redundant compilation and cross-orchestration name
    collisions in ``next_levels/{orch}/`` artifacts.
    """
    chip_func_types = (
        _ir_core.FunctionType.InCore,
        _ir_core.FunctionType.AIC,
        _ir_core.FunctionType.AIV,
        _ir_core.FunctionType.Group,
        _ir_core.FunctionType.Spmd,
    )

    result: list[_ir_core.Function] = [orch_func]
    visited: set[str] = {orch_func.name}
    work: list[_ir_core.Function] = [orch_func]

    while work:
        func = work.pop()
        collector = _CallCollector()
        collector.visit_stmt(func.body)
        for callee_name in collector.callee_names:
            if callee_name in visited:
                continue
            callee = program.get_function(callee_name)
            if callee is None or callee.func_type not in chip_func_types:
                continue
            visited.add(callee_name)
            result.append(callee)
            work.append(callee)

    return result


def _generate_single_chip(
    transformed_program: _ir_core.Program,
    output_dir: str,
    skip_ptoas: bool = False,
) -> dict[str, str]:
    """Generate artifacts for a single-chip (L0-L2) program. Original generate() logic."""
    result_files: dict[str, str] = {}
    errors: list[tuple[str, Exception]] = []
    prof = CompileProfiler.current()

    orch_func = next(
        (
            f
            for f in transformed_program.functions.values()
            if f.func_type == _ir_core.FunctionType.Orchestration
        ),
        None,
    )

    groups, ungrouped = _build_group_mapping(transformed_program)

    # ── Phase 1: IR → MLIR (sequential, fast) ────────────────────────
    # PTOCodegen converts IR to MLIR strings. This is cheap (pure string
    # generation) and runs sequentially so that we don't contend on the GIL.
    units: list[_CodegenUnit] = []

    # Grouped functions: one MLIR module per group
    for group_name, members in groups.items():
        try:
            grouped_program = _ir_core.Program(members, group_name, transformed_program.span)
            stage = StageRecord(name=f"kernel_codegen:{group_name}", start=time.perf_counter())
            ir_record = StageRecord(name="ir_to_mlir", start=time.perf_counter())
            pto_code = _codegen_core.PTOCodegen().generate(grouped_program)
            ir_record.end = time.perf_counter()
            stage.children.append(ir_record)
            units.append(_CodegenUnit(group_name, pto_code, members, is_group=True, stage_record=stage))
        except Exception as e:
            func_names = ", ".join(m.name for m in members)
            logger.error("Failed to compile group '%s' [%s]: %s", group_name, func_names, e)
            errors.append((group_name, e))

    for func in ungrouped:
        try:
            peer_names = _extract_peer_function_names(func)
            peer_funcs: list[_ir_core.Function] = []
            for name in peer_names:
                peer_func = transformed_program.get_function(name)
                if peer_func is not None:
                    peer_funcs.append(peer_func)
            single_program = _ir_core.Program([*peer_funcs, func], func.name, transformed_program.span)
            stage = StageRecord(name=f"kernel_codegen:{func.name}", start=time.perf_counter())
            ir_record = StageRecord(name="ir_to_mlir", start=time.perf_counter())
            pto_code = _codegen_core.PTOCodegen().generate(single_program)
            ir_record.end = time.perf_counter()
            stage.children.append(ir_record)
            units.append(_CodegenUnit(func.name, pto_code, [func], is_group=False, stage_record=stage))
        except Exception as e:
            logger.error("Failed to compile function '%s': %s", func.name, e)
            errors.append((func.name, e))

    # ── Phase 2: ptoas (parallel, slow) ──────────────────────────────
    # Each _emit_unit call runs the ptoas subprocess and generates the
    # kernel wrapper.  These are data-independent and subprocess-heavy, so
    # a thread pool gives real parallelism (subprocess.run releases the GIL).
    _run_ptoas_phase(units, output_dir, skip_ptoas, prof, result_files, errors)

    # Orchestration + config
    if orch_func is not None:
        try:
            with _profiling_stage(prof, "orchestration_codegen"):
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
