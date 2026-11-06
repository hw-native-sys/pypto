# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""为 Python DSL 提供延迟外部核函数集成。

可调用对象的 ``__pypto_jit__()`` 钩子返回带 ``external_builder`` 的稳定
``JITFunction``。构建器接收特化上下文并返回此描述符。
PyPTO 负责参数绑定、任务依赖和编译；提供方提供设备代码。
``jit.cannbotdsl`` 等提供方适配器也可由 PyPTO 维护，负责类型转换和运行时 ABI 包装。
"""

from dataclasses import dataclass
from pathlib import Path

from pypto._external_source import external_source_digest

from .specializer import SpecializeContext


@dataclass(frozen=True)
class ExternalKernelSpec:
    """描述一个特化后的外部任务及其张量读写行为。

    ``parameter_access`` 将每个张量参数映射为 ``read``、``write`` 或
    ``read_write``。在无显式结果的 Python 调用处，被写入的参数成为隐式 SSA 结果。
    混合任务为两种核类型分别编译 ``source``。
    """

    source: str
    core_type: str
    parameter_access: dict[str, str]
    include_dirs: tuple[str, ...] = ()
    dual_aiv_dispatch: bool = False
    backend: str | None = None
    runtime: str | None = None
    # 外部源码中的硬件全核屏障参与方；None 表示没有全核驻留要求。
    hard_syncall: str | None = None


def apply_external_spec(ctx: SpecializeContext, spec: object) -> None:
    """校验提供方的结果，并将其附加到本次请求的上下文。"""
    if not isinstance(spec, ExternalKernelSpec):
        raise TypeError(f"External builder for {ctx.func_name!r} must return ExternalKernelSpec")
    if spec.core_type not in ("aic", "aiv", "mixed"):
        raise ValueError(f"External kernel {ctx.func_name!r}: invalid core type {spec.core_type!r}")
    if spec.dual_aiv_dispatch and spec.core_type != "mixed":
        raise ValueError("dual_aiv_dispatch requires a mixed external kernel")
    expected_syncall = {"aic": "aic_only", "aiv": "aiv_only", "mixed": "mix"}[spec.core_type]
    if spec.hard_syncall is not None and spec.hard_syncall != expected_syncall:
        raise ValueError(f"External {spec.core_type} kernel requires hard_syncall={expected_syncall!r}")
    if spec.hard_syncall == "mix" and not spec.dual_aiv_dispatch:
        raise ValueError("Mixed hard_syncall requires dual_aiv_dispatch=True")
    unknown = [
        name for name in ctx.param_names if name not in ctx.tensor_meta and name not in ctx.scalar_dtypes
    ]
    if unknown:
        raise ValueError(
            f"External kernel {ctx.func_name!r}: cannot infer parameters {unknown}; "
            "pass tensors or explicitly typed runtime scalars"
        )
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
    ctx.external_hard_syncall = spec.hard_syncall


def external_context_digest(contexts: list[SpecializeContext], backend: str, runtime: str) -> str:
    """在查询任一已编译程序缓存前，识别生成的源码。"""
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
                    str(ctx.external_hard_syncall),
                    repr(sorted(ctx.external_parameter_access.items())),
                    *ctx.external_include_dirs,
                ),
            )
        )
    return "|".join(records)
