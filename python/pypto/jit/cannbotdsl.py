# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# Copyright (c) 2026 Huawei Technologies Co., Ltd.

"""由 PyPTO 维护的 CANNBotDSL A5 设备可调用核函数适配器。

CANNBotDSL 提供 TensorSpec、降级流程、SourceBundle 源码和设备 ABI。
本适配器负责核函数识别、元数据转换、运行时包装和外部任务设置，
无需在提供方包中增加钩子。CANNBotDSL 为可选依赖，仅在使用适配器时导入。
"""

import hashlib
import importlib
import inspect
import json
import os
import sys
import tempfile
import threading
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .decorator import JITFunction
from .foreign import ExternalKernelSpec
from .specializer import DynDim

if TYPE_CHECKING:
    from ._cannbotdsl_source import SourceBundle
    from .specializer import SpecializeContext


def write_wrapper(exported: "SourceBundle", directory: str | Path) -> Path:
    """将 CANNBotDSL 原生源码和编译器报告的 ABI 适配到 simpler。"""
    manifest = exported.manifest
    directory = exported.write(directory)
    symbol = manifest["symbol"]
    lines = [
        '#include "c_api/asc_simd.h"',
        '#include "tensormap_and_ringbuffer/tensor.h"',
    ]
    # 将提供方启动入口嵌入为内联设备函数。
    # 计算与同步逻辑保留在提供方生成的源码中。
    macros = {"__global__": "inline __attribute__((always_inline))", "__cube__": "", "__vector__": ""}
    for name, value in macros.items():
        lines += [f'#pragma push_macro("{name}")', f"#undef {name}", f"#define {name} {value}"]
    lines += [
        '#pragma push_macro("KERNEL_TASK_TYPE_DEFAULT")',
        "#undef KERNEL_TASK_TYPE_DEFAULT",
        "#define KERNEL_TASK_TYPE_DEFAULT(...) ((void)0)",
        '#include "device_source.hpp"',
        '#pragma pop_macro("KERNEL_TASK_TYPE_DEFAULT")',
    ]
    for name in reversed(macros):
        lines.append(f'#pragma pop_macro("{name}")')
    lines += [
        "template<unsigned I, typename T, typename... Rest> struct PyPTOArgAt : PyPTOArgAt<I-1, Rest...> {};",
        "template<typename T, typename... Rest> struct PyPTOArgAt<0, T, Rest...> { using type = T; };",
        "template<unsigned I, typename Signature> struct PyPTODeviceArg;",
        "template<unsigned I, typename R, typename... Args>",
        "struct PyPTODeviceArg<I, R(Args...)> : PyPTOArgAt<I, Args...> {};",
        'extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {',
    ]
    parameters = {p["logical_index"]: p for p in manifest["parameters"]}
    arguments: list[str] = []
    for position, physical in enumerate(manifest["device_abi"]["parameters"]):
        index = physical["logical_index"]
        parameter = parameters[index]
        local = f"arg{index}"
        cpp_type = f"Arg{index}"
        lines += [
            f"    using {cpp_type} = typename PyPTODeviceArg<{position}, decltype({symbol})>::type;",
            f'    static_assert(sizeof({cpp_type}) == {physical["size"]}, "device argument size mismatch");',
        ]
        if parameter["kind"] == "tensor":
            lines += [
                f"    auto* t{index} = reinterpret_cast<__gm__ simpler::tmr::Tensor*>(args[{index}]);",
                f"    {cpp_type} {local}{{}};",
                f"    {local}.ptr = reinterpret_cast<decltype({local}.ptr)>(t{index}->buffer.addr)"
                f" + t{index}->start_offset;",
            ]
            for field in parameter["dynamic_fields"]:
                member = field["field"]
                lines.append(
                    f"    {local}.{member}[{field['slot']}] = t{index}->{member}s[{field['dimension']}];"
                )
        else:
            lines.append(f"    {cpp_type} {local} = *reinterpret_cast<__gm__ {cpp_type}*>(&args[{index}]);")
        arguments.append(local)
    lines += [
        "    uint64_t saved_ctrl = asc_get_ctrl();",
        "    set_atomic_none();",
        f"    {symbol}({', '.join(arguments)});",
        "    pipe_barrier(PIPE_ALL);",
        "    asc_dcci_entire_out();",
        "    dsb(DSB_DDR);",
        "    set_atomic_none();",
        "    asc_set_ctrl(saved_ctrl);",
        "}",
        "",
    ]
    path = Path(directory) / "entry.cpp"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def include_directories() -> tuple[str, ...]:
    cb = importlib.import_module("cannbotdsl")
    package_file = cb.__file__
    if package_file is None:
        raise RuntimeError("PyPTO DSL integration requires an installed CANNBotDSL package with headers")

    root = os.environ.get("ASCEND_HOME_PATH")
    if not root:
        raise RuntimeError("PyPTO DSL integration requires the CANN environment (ASCEND_HOME_PATH)")
    asc = Path(root) / "aarch64-linux" / "asc"
    return (
        str(asc),
        str(asc / "include"),
        str(Path(root) / "include"),
        str(Path(package_file).parent / "include"),
    )


def _materialize(exported: "SourceBundle") -> Path:
    # 发布完整且不可变的源码；其他编译器可能保留此路径。
    digest = hashlib.sha256(
        (
            exported.source + json.dumps(exported.manifest, sort_keys=True) + Path(__file__).read_text()
        ).encode()
    ).hexdigest()
    root = Path(os.environ.get("CANNBOTDSL_PYPTO_CACHE_DIR", Path.home() / ".cache/cannbotdsl/pypto"))
    root.mkdir(parents=True, exist_ok=True)
    target = root / digest
    if not target.is_dir():
        with tempfile.TemporaryDirectory(dir=root) as staging:
            staged = Path(staging) / "kernel"
            write_wrapper(exported, staged)
            try:
                staged.rename(target)
            except OSError:
                if not target.is_dir():
                    raise
    return target / "entry.cpp"


def _dtype(dtype: Any) -> Any:
    cb = importlib.import_module("cannbotdsl")

    names = {
        "FP16": "float16",
        "BF16": "bfloat16",
        "BFLOAT16": "bfloat16",
        "FP32": "float32",
        "FP64": "float64",
        "FP8E4M3FN": "float8_e4m3fn",
        "FP8_E4M3FN": "float8_e4m3fn",
        "FP8E8M0": "float8_e8m0",
        "FP8_E8M0": "float8_e8m0",
        "INT8": "int8",
        "INT16": "int16",
        "INT32": "int32",
        "INT64": "int64",
        "INDEX": "int64",
        "UINT8": "uint8",
        "UINT16": "uint16",
        "UINT32": "uint32",
        "UINT64": "uint64",
        "BOOL": "bool",
    }
    name = names.get(str(dtype).upper())
    if name is None or not hasattr(cb.dtypes, name):
        raise TypeError(f"PyPTO DSL integration does not support dtype {dtype}")
    return getattr(cb.dtypes, name)


def _build(kernel: Any, ctx: "SpecializeContext") -> "ExternalKernelSpec":
    cb = importlib.import_module("cannbotdsl")

    if ctx.constexpr_values:
        raise TypeError("PyPTO DSL integration currently accepts tensor and runtime scalar parameters only")
    dims: dict[str, Any] = {}
    arguments: list[Any] = []
    for name in ctx.param_names:
        if name in ctx.tensor_meta:
            meta = ctx.tensor_meta[name]
            if meta.layout is not None and meta.layout.name != "ND":
                raise ValueError(f"PyPTO DSL parameter {name!r} requires ND layout, got {meta.layout}")
            shape: list[Any] = []
            for dim in meta.shape:
                if isinstance(dim, DynDim):
                    shape.append(dims.setdefault(dim.name, cb.Dim(dim.name)))
                else:
                    shape.append(dim)
            # ND 视图保留父张量的前导步长，只有最内层维度连续；
            # 若完全按形状推导步长，会使切片后的 PyPTO 张量寻址错误。
            stride = tuple(cb.Dim(f"__pypto_{name}_stride_{i}") for i in range(len(shape) - 1)) + (1,)
            arguments.append(cb.TensorSpec(tuple(shape), _dtype(meta.dtype), stride=stride))
        elif name in ctx.scalar_dtypes:
            arguments.append(_dtype(ctx.scalar_dtypes[name]))
        else:
            raise TypeError(
                f"Cannot infer PyPTO DSL parameter {name!r}; pass a tensor or typed runtime scalar"
            )
    exporter = importlib.import_module("._cannbotdsl_source", __package__)
    exported = exporter.export_source_bundle(kernel, arguments)
    manifest = exported.manifest
    return ExternalKernelSpec(
        source=str(_materialize(exported)),
        core_type=manifest["core_type"],
        parameter_access={
            p["name"]: "read" if p["access"] == "none" else p["access"]
            for p in manifest["parameters"]
            if p["kind"] == "tensor"
        },
        include_dirs=include_directories(),
        dual_aiv_dispatch=manifest["core_type"] == "mixed",
        backend="Ascend950",
        runtime="tensormap_and_ringbuffer",
    )


_KERNELS: weakref.WeakKeyDictionary[Any, JITFunction] = weakref.WeakKeyDictionary()
_KERNELS_LOCK = threading.RLock()


def try_bind_kernel(kernel: Any) -> JITFunction | None:
    """识别可选提供方，避免普通 PyPTO 调用导入该依赖。"""
    module = sys.modules.get("cannbotdsl.lang.kernel")
    if module is None:
        return None
    if isinstance(kernel, module._BoundKernelLauncher):
        raise TypeError("PyPTO CANNBotDSL calls require an unbound @cb.kernel function")
    if isinstance(kernel, module.KernelLauncher):
        return bind_kernel(kernel)
    return None


def bind_kernel(kernel: Any) -> JITFunction:
    """为未配置的 DSL 核函数返回由 PyPTO 维护的稳定依赖。"""
    if kernel._block_dim is not None or kernel._thread_dim is not None or kernel._dyn_ub_buf:
        raise ValueError("PyPTO calls require an unconfigured @cb.kernel without host launch dimensions")
    if kernel.profile is not None:
        raise ValueError("PyPTO CANNBotDSL calls do not support provider-managed profiling")
    for param in inspect.signature(kernel.fn).parameters.values():
        if (
            param.kind not in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
            or param.default is not param.empty
        ):
            raise TypeError("PyPTO DSL kernels require explicit positional parameters without defaults")
    with _KERNELS_LOCK:
        if kernel not in _KERNELS:
            reference = weakref.ref(kernel)

            def build(context: "SpecializeContext") -> ExternalKernelSpec:
                target = reference()
                if target is None:
                    raise ReferenceError("CANNBotDSL kernel was released before PyPTO specialization")
                return _build(target, context)

            _KERNELS[kernel] = JITFunction(kernel.fn, func_type="extern", external_builder=build)
        return _KERNELS[kernel]
