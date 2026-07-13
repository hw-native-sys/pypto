# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Emit a self-contained debug re-runner at ``<work_dir>/debug/run.py``.

The emitted script provides a uniform debug entry point for every
``build_output/<jit_dir>/`` directory:

- When a sibling ``golden.py`` exists (non-JIT path), inputs are loaded via
  :func:`golden.generate_inputs` and the run is validated against
  :func:`compute_golden` using :func:`pypto.runtime.debug.replay`.
- Otherwise (JIT path), inputs are materialised from shape / dtype
  information embedded in the generated script. The user can edit them
  freely to experiment.

So the user only needs to remember one command::

    python build_output/<jit_dir>/debug/run.py
"""

from pathlib import Path

import torch

from pypto.ir.compiled_program import ParamInfo, _to_torch_dtype
from pypto.pypto_core.ir import ParamDirection

_TORCH_DTYPE_NAMES: dict[torch.dtype, str] = {
    torch.float16: "torch.float16",
    torch.float32: "torch.float32",
    torch.float64: "torch.float64",
    torch.bfloat16: "torch.bfloat16",
    torch.int8: "torch.int8",
    torch.int16: "torch.int16",
    torch.int32: "torch.int32",
    torch.int64: "torch.int64",
    torch.uint8: "torch.uint8",
    torch.bool: "torch.bool",
}
for _name in ("uint16", "uint32", "uint64"):
    _dt = getattr(torch, _name, None)
    if _dt is not None:
        _TORCH_DTYPE_NAMES[_dt] = f"torch.{_name}"
del _name, _dt


def write_run_script(
    work_dir: Path | str,
    param_infos: list[ParamInfo],
    *,
    platform: str | None = None,
) -> Path:
    """Emit ``<work_dir>/debug/run.py``.

    Args:
        work_dir: A ``build_output/<jit_dir>/`` produced by ``ir.compile()``.
        param_infos: Orchestration parameter metadata, in call order.
            Obtain via :func:`pypto.ir.compiled_program.extract_param_infos`.
        platform: Default platform baked into the generated CLI's
            ``--platform`` argument. ``None`` falls back to ``"a2a3sim"``.

    Returns:
        The path to the written ``debug/run.py``.
    """
    work_dir = Path(work_dir)
    debug_dir = work_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    out_path = debug_dir / "run.py"
    out_path.write_text(_render(param_infos, platform), encoding="utf-8")
    return out_path


def _render(param_infos: list[ParamInfo], platform: str | None) -> str:
    init_lines = [_init_expr_for(p) for p in param_infos]
    names = ", ".join(p.name for p in param_infos)
    default_platform = platform or "a2a3sim"
    indented_inits = "\n".join("    " + line for line in init_lines) or "    pass"

    return f'''"""Auto-generated debug runner — only ``_inline_inputs`` and ``_user_compare`` are editable.

Re-execute the kernel in the parent build_output directory:

    python {{this_file}}

When a sibling ``golden.py`` exists, inputs are loaded via
``golden.generate_inputs`` and the run is validated against
``compute_golden``. Otherwise (JIT path) the inputs declared inline in
``_inline_inputs`` are used, and ``_user_compare`` is invoked after
replay so you can write your own assertions.

------------------------------------------------------------------------
CLI flags (forwarded to ``pypto.runtime.debug.replay._main``)
------------------------------------------------------------------------
  --platform PLAT          target platform (default: {default_platform})
  --device-id N            hardware device index (default: 0)
  --pmu LEVEL              enable PMU profiling at LEVEL
  --swimlane               enable L2 swimlane capture
  --dump-args [LEVEL]      dump per-task arguments to disk (bare=1 partial, 2 full)
  --dep-gen                enable dep_gen profiling
  --no-recompile           reuse cached .so/.bin (ignores cpp edits)
  --no-rebuild-from-pto    skip ptoas/*.pto -> kernels/*.cpp rebuild
  --log-level LEVEL        runtime log level: debug/v0..v9/info/warn/error/null
  --log-sync-pypto         also push --log-level to PyPTO's C++ logger
  --validate               compare outputs vs. golden.py (auto-on if golden.py exists)
  --no-validate            skip golden validation; runs ``_user_compare`` instead

Examples:
  python {{this_file}} --pmu 2 --swimlane              # DFX-on run
  python {{this_file}} --log-level debug               # verbose runtime trace
  python {{this_file}} --no-validate                   # skip golden, run _user_compare
  python {{this_file}} --no-recompile --no-rebuild-from-pto   # fast re-run, no rebuild

Run ``python {{this_file}} --help`` for the authoritative list — flags are
defined in :func:`pypto.runtime.debug.replay._main`, not here, so the
``--help`` text is always in sync.

Generated by pypto.runtime.debug.run_script_writer.
"""

import sys
from pathlib import Path

import torch

from pypto.runtime.debug.replay import _main

_WORK_DIR = Path(__file__).resolve().parent.parent


def _inline_inputs():
    """Materialise inputs from shape / dtype info embedded at compile time."""
{indented_inits}
    return [{names}]


def _user_compare({names}):
    """Hand-written comparison for the JIT path (no ``golden.py``).

    Invoked from ``_main`` after ``replay`` finishes when golden validation
    did NOT run. Output tensors have been populated in-place by the kernel.
    Edit the body to add your own assertions, for example::

        expected = a + b
        assert torch.allclose(c, expected, rtol=1e-5, atol=1e-5), (
            f"max diff = {{(c - expected).abs().max().item()}}"
        )
    """
    return  # TODO: add your comparison here


if __name__ == "__main__":
    sys.exit(_main(
        [str(_WORK_DIR), *sys.argv[1:]],
        inline_inputs=_inline_inputs,
        user_compare=_user_compare,
        default_platform="{default_platform}",
    ))
'''


_INT_DTYPES: set[torch.dtype] = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}
for _uint_name in ("uint16", "uint32", "uint64"):
    _uint_dt = getattr(torch, _uint_name, None)
    if _uint_dt is not None:
        _INT_DTYPES.add(_uint_dt)
del _uint_name, _uint_dt


def _init_expr_for(p: ParamInfo) -> str:
    """Return a Python statement initialising parameter *p*.

    Scalars (no shape) are not auto-materialised in v1 — emit a placeholder
    line so the user can hand-edit. Dynamic dimensions (``-1``) are filled
    with ``1`` so the generated file is at least syntactically valid.

    ``torch.randn`` only supports floating-point dtypes, so integer / bool
    inputs use ``torch.randint`` (the only API that accepts those dtypes
    natively without a downstream cast).
    """
    if p.shape is None:
        return f"{p.name} = None  # TODO: scalar param — set a ctypes value"

    torch_dtype = _to_torch_dtype(p.dtype)
    dtype_str = _TORCH_DTYPE_NAMES.get(torch_dtype, "torch.float32") if torch_dtype else "torch.float32"
    concrete_shape = tuple(d if d >= 0 else 1 for d in p.shape)
    has_dynamic = any(d < 0 for d in p.shape)
    comment = "  # NOTE: dynamic dim(s) filled with 1 — edit as needed" if has_dynamic else ""

    if p.direction == ParamDirection.Out:
        return f"{p.name} = torch.zeros({concrete_shape!r}, dtype={dtype_str}){comment}"
    if torch_dtype is torch.bool:
        return f"{p.name} = torch.randint(0, 2, {concrete_shape!r}, dtype={dtype_str}){comment}"
    if torch_dtype in _INT_DTYPES:
        return f"{p.name} = torch.randint(0, 8, {concrete_shape!r}, dtype={dtype_str}){comment}"
    return f"{p.name} = torch.randn({concrete_shape!r}, dtype={dtype_str}){comment}"
