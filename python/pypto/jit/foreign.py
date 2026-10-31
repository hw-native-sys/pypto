# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Lazy external-kernel integration for Python DSLs.

A callable's ``__pypto_jit__()`` hook returns a stable ``JITFunction`` with an
``external_builder``. The builder receives a specialization context and returns
this descriptor. PyPTO owns argument binding, task dependencies and compilation;
the producer owns device source generation and the runtime ABI wrapper.
"""

from dataclasses import dataclass
from pathlib import Path

from pypto._external_source import external_source_digest

from .specializer import SpecializeContext


@dataclass(frozen=True)
class ExternalKernelSpec:
    """One specialized external task, including its tensor effects.

    ``parameter_access`` maps every tensor parameter to ``read``, ``write`` or
    ``read_write``. Written parameters become implicit SSA results at a bare
    Python call site. A mixed task compiles ``source`` for both core types.
    """

    source: str
    core_type: str
    parameter_access: dict[str, str]
    include_dirs: tuple[str, ...] = ()
    dual_aiv_dispatch: bool = False
    backend: str | None = None
    runtime: str | None = None


def apply_external_spec(ctx: SpecializeContext, spec: object) -> None:
    """Validate a producer result and attach it to this request's context."""
    if not isinstance(spec, ExternalKernelSpec):
        raise TypeError(f"External builder for {ctx.func_name!r} must return ExternalKernelSpec")
    if spec.core_type not in ("aic", "aiv", "mixed"):
        raise ValueError(f"External kernel {ctx.func_name!r}: invalid core type {spec.core_type!r}")
    if spec.dual_aiv_dispatch and spec.core_type != "mixed":
        raise ValueError("dual_aiv_dispatch requires a mixed external kernel")
    if set(spec.parameter_access) != set(ctx.tensor_meta) or any(
        access not in ("read", "write", "read_write") for access in spec.parameter_access.values()
    ):
        raise ValueError(f"External kernel {ctx.func_name!r}: provide access for every tensor parameter")
    source = Path(spec.source).resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"External kernel source is not a file: {source}")
    includes = tuple(str(Path(path).resolve(strict=True)) for path in spec.include_dirs)
    if any(not Path(path).is_dir() for path in includes):
        raise ValueError(f"External kernel {ctx.func_name!r}: include paths must be directories")
    ctx.external_core_type = spec.core_type
    ctx.external_aic_source = str(source) if spec.core_type in ("aic", "mixed") else None
    ctx.external_aiv_source = str(source) if spec.core_type in ("aiv", "mixed") else None
    ctx.external_dual_aiv_dispatch = spec.dual_aiv_dispatch
    ctx.external_include_dirs = includes
    ctx.external_parameter_access = dict(spec.parameter_access)
    ctx.external_backend = spec.backend
    ctx.external_runtime = spec.runtime


def external_context_digest(contexts: list[SpecializeContext], backend: str, runtime: str) -> str:
    """Identify generated sources before either compiled-program cache lookup."""
    records: list[str] = []
    for ctx in contexts:
        if ctx.external_parameter_access is None:
            continue
        if ctx.external_backend is not None and ctx.external_backend != backend:
            raise ValueError(
                f"External kernel {ctx.func_name!r} requires backend {ctx.external_backend}, got {backend}"
            )
        if ctx.external_runtime is not None and ctx.external_runtime != runtime:
            raise ValueError(
                f"External kernel {ctx.func_name!r} requires runtime {ctx.external_runtime}, got {runtime}"
            )
        paths = [path for path in (ctx.external_aic_source, ctx.external_aiv_source) if path is not None]
        records.append(
            external_source_digest(
                paths,
                include_dirs=ctx.external_include_dirs,
                metadata=(
                    ctx.func_name,
                    str(ctx.external_core_type),
                    str(ctx.external_dual_aiv_dispatch),
                    repr(sorted(ctx.external_parameter_access.items())),
                    *ctx.external_include_dirs,
                ),
            )
        )
    return "|".join(records)
