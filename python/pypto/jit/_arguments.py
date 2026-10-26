# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tensor arguments and annotations used to specialize a JIT call.

Torch stays optional and is loaded through one module-owned cache. These helpers
read captured namespaces through the existing specializer services."""

import inspect
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir

from .specializer import (
    DynDim,
    ShapeDim,
    TensorMeta,
    func_name_lookup,
)

# ---------------------------------------------------------------------------
# torch-optional dtype conversion
# ---------------------------------------------------------------------------

# Sentinel list: empty means not yet loaded; [None] means torch unavailable;
# [torch_module] means torch is loaded.
_TORCH_CACHE: list[Any] = []
_TORCH_DTYPE_MAP: dict[Any, DataType] = {}


def _get_torch() -> Any:
    """Return the torch module, or None if not installed. Result is cached."""
    if not _TORCH_CACHE:
        try:
            import torch  # noqa: PLC0415

            _TORCH_CACHE.append(torch)
            _TORCH_DTYPE_MAP.update(
                {
                    torch.float16: DataType.FP16,
                    torch.float32: DataType.FP32,
                    torch.bfloat16: DataType.BF16,
                    torch.int8: DataType.INT8,
                    torch.int16: DataType.INT16,
                    torch.int32: DataType.INT32,
                    torch.int64: DataType.INT64,
                    torch.uint8: DataType.UINT8,
                    torch.bool: DataType.BOOL,
                }
            )
            # Optional low-precision MX dtypes (PyTorch 2.1+/2.3+/2.7+); required
            # for MX DSL ST. float4_e2m1fn_x2 is the packed MXFP4 weight dtype.
            for _torch_name, _pto_dt in (
                ("float8_e4m3fn", DataType.FP8E4M3FN),
                ("float8_e5m2", DataType.FP8E5M2),
                ("float8_e8m0fnu", DataType.FP8E8M0),
                ("float4_e2m1fn_x2", DataType.FP4E2M1X2),
            ):
                _td = getattr(torch, _torch_name, None)
                if _td is not None:
                    _TORCH_DTYPE_MAP[_td] = _pto_dt

        except ImportError:
            _TORCH_CACHE.append(None)
    return _TORCH_CACHE[0]


def torch_dtype_to_pypto(torch_dtype: Any) -> DataType:
    _get_torch()
    if torch_dtype not in _TORCH_DTYPE_MAP:
        raise TypeError(
            f"Unsupported torch dtype {torch_dtype}. "
            "Supported: float16, float32, bfloat16, int8/16/32/64, uint8, bool, "
            "float8_e4m3fn/e5m2/e8m0fnu, float4_e2m1fn_x2 (where torch supports them)."
        )
    return _TORCH_DTYPE_MAP[torch_dtype]


def is_tensor(obj: Any) -> bool:
    """Return True if obj is a torch.Tensor (without hard-importing torch)."""
    torch = _get_torch()
    if torch is None:
        return False
    return isinstance(obj, torch.Tensor)


def _build_tensor_meta(
    extents: Sequence[int],
    dtype: DataType,
    dyn_dims: dict[int, DynDim] | None = None,
    layout: _ir.TensorLayout | None = None,
) -> TensorMeta:
    """Build a ``TensorMeta`` from per-dim extents and a resolved dtype.

    ``dyn_dims`` maps ``dim_idx → DynDim`` for dims declared dynamic at this
    parameter (via ``bind_dynamic`` or an annotation-embedded ``pl.dynamic()``).
    The DynDim's ``static_bound`` is filled from the corresponding extent.
    Shared by the torch-tensor path (``extract_tensor_meta``, extent = the
    real tensor dim) and the signature path (``JITFunction._bind_args_from_signature``,
    extent = the static annotation dim or a placeholder for dynamic dims).

    ``layout`` is the annotation's third slot. It never comes from a runtime
    tensor — a torch tensor carries no PyPTO layout — so both paths read it
    from the same place, the parameter's annotation.
    """
    dyn = dyn_dims or {}
    shape: list[ShapeDim] = []
    for i, d in enumerate(extents):
        extent = int(d)
        bound = dyn.get(i)
        if bound is None:
            shape.append(extent)
        else:
            shape.append(DynDim(name=bound.name, literal=bound.literal, static_bound=extent))
    return TensorMeta(shape=tuple(shape), dtype=dtype, layout=layout)


def extract_tensor_meta(
    tensor: Any,
    dyn_dims: dict[int, DynDim] | None = None,
    layout: _ir.TensorLayout | None = None,
    expected_dtype: DataType | None = None,
) -> TensorMeta:
    """Extract TensorMeta from a torch.Tensor (shape/dtype only — no data read).

    ``layout`` and ``expected_dtype`` come from the parameter's annotation.
    Torch's ``float4_e2m1fn_x2`` maps directly to packed ``pl.FP4E2M1X2``.
    Explicit packed annotations also accept raw UINT8 carriers; E1M2 uses this
    ABI because Torch has no E1M2 dtype. Shapes always count carrier elements.
    Other dtype annotations are ignored here.
    """
    torch_dtype = torch_dtype_to_pypto(tensor.dtype)
    dtype = torch_dtype
    extents = list(tensor.shape)
    if torch_dtype == DataType.FP4E2M1X2 or (
        expected_dtype is not None and expected_dtype in (DataType.FP4E2M1X2, DataType.FP4E1M2X2)
    ):
        if not extents:
            raise TypeError("Packed FP4 tensors must have rank >= 1")
        if extents[-1] <= 0:
            raise TypeError(
                "Packed FP4 tensors require a positive runtime x2 carrier last "
                f"dimension; got shape {tuple(extents)}"
            )

    # Only packed annotations give raw carriers their numerical format. Other
    # annotations such as pl.INDEX are semantic IR types with an INT64 ABI and
    # must not introduce a general annotation-to-Torch strict dtype check here.
    if expected_dtype is not None and expected_dtype in (DataType.FP4E2M1X2, DataType.FP4E1M2X2):
        if torch_dtype not in (DataType.UINT8, expected_dtype):
            raise TypeError(
                f"Parameter annotated packed {expected_dtype} but got torch dtype mapped to {torch_dtype}; "
                "pass torch.uint8 carriers (or torch.float4_e2m1fn_x2 for E2M1)"
            )
        dtype = expected_dtype

    return _build_tensor_meta(extents, dtype, dyn_dims, layout)


def resolve_annotation(annotation: Any, ann_ns: Mapping[str, Any] | None) -> Any:
    """Resolve one parameter annotation, evaluating the string form if needed.

    ``from __future__ import annotations`` in the *user's* module leaves every
    annotation as a string; ``ann_ns`` (from ``func_name_lookup``) is the
    namespace to evaluate it in.

    Args:
        annotation: Raw ``inspect.Parameter.annotation``
        ann_ns: Namespace for string annotations, or None to leave them as-is

    Returns:
        The resolved annotation object, or the original string when it cannot
        be evaluated
    """
    if not isinstance(annotation, str) or ann_ns is None:
        return annotation
    try:
        # Trusted input: the kernel's own annotation source, evaluated in its
        # own globals+closure namespace (same as Python would).
        return eval(annotation, dict(ann_ns))  # noqa: S307
    except Exception:  # noqa: BLE001 - leave as string; callers treat it as "cannot infer"
        return annotation


def _annotation_namespace(func: Any, sig: inspect.Signature) -> Mapping[str, Any] | None:
    """Namespace for resolving ``func``'s string annotations, or None if unneeded."""
    if any(isinstance(p.annotation, str) for n, p in sig.parameters.items() if n != "self"):
        return func_name_lookup(func)
    return None


def _annotation_layout(annotation: Any, param_name: str, func_name: str) -> _ir.TensorLayout | None:
    """Read the layout slot off an already-resolved tensor annotation.

    ``pl.Tensor[[...], dtype, pl.NZ]`` evaluates to a ``Tensor`` instance whose
    ``layout`` holds the third slot; the two-slot form leaves it None.

    Args:
        annotation: Resolved parameter annotation (any object — non-tensor
            annotations simply carry no layout)
        param_name: Parameter the annotation belongs to, for diagnostics
        func_name: Enclosing kernel name, for diagnostics

    Returns:
        The annotated layout, or None when the annotation declares none

    Raises:
        TypeError: If the slot holds a ``pl.TensorView`` — specialization has
            nowhere to carry it, and silently dropping it would mis-declare the
            parameter
    """
    layout = getattr(annotation, "layout", None)
    if layout is None or isinstance(layout, _ir.TensorLayout):
        return layout
    # ``Tensor.__getitem__`` routes any non-MemRef third element into ``layout``,
    # so a pl.TensorView(...) lands here. TensorMeta has no field for it, and a
    # dropped stride is silently wrong code — refuse instead. This is reachable
    # from the DN rejection's own migration hint, so the message has to be plain.
    raise TypeError(
        f"@pl.jit function {func_name!r}: parameter {param_name!r} annotates a "
        f"{type(layout).__name__} in its layout slot, which @pl.jit does not yet "
        f"support — it would be dropped and the parameter compiled as ND. Use a "
        f"plain layout (e.g. pl.MX_A_ZZ), or declare the kernel with @pl.function, "
        f"which resolves the annotation directly."
    )


def param_layouts(func: Any, func_name: str) -> dict[str, _ir.TensorLayout]:
    """Map parameter name → annotated layout, for params that declare one.

    The torch-argument path derives shape and dtype from the passed tensors, so
    it never looks at annotations — but a layout has no runtime counterpart to
    read, making the annotation its only source. This recovers it. Also used to
    recover a dep function's own declarations, which no caller argument carries.

    Args:
        func: The Python function whose annotations to read
        func_name: Name to use in diagnostics

    Returns:
        Layout per parameter name; parameters without one are absent
    """
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return {}
    ann_ns = _annotation_namespace(func, sig)

    layouts: dict[str, _ir.TensorLayout] = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        layout = _annotation_layout(resolve_annotation(param.annotation, ann_ns), name, func_name)
        if layout is not None:
            layouts[name] = layout
    return layouts


def param_dtypes(func: Any) -> dict[str, DataType]:
    """Map shaped parameter names to their annotated packed FP4 formats.

    Explicit annotations give raw UINT8 carriers their numerical format at
    runtime and preserve that format across dependency boundaries. Other
    annotations (e.g. ``pl.INDEX``) must not drive a general dtype check here.
    """
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return {}
    ann_ns = _annotation_namespace(func, sig)

    dtypes: dict[str, DataType] = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        annotation = resolve_annotation(param.annotation, ann_ns)
        dtype = getattr(annotation, "dtype", None)
        if isinstance(dtype, DataType) and dtype in (DataType.FP4E2M1X2, DataType.FP4E1M2X2):
            dtypes[name] = dtype
    return dtypes


def constexpr_params(func: Any) -> list[str]:
    """Parameter names annotated ``pl.constexpr``, in declaration order.

    Constexpr-ness lives only in the annotation: the call site passes an
    ordinary Python value, and nothing about that value says whether the kernel
    wants it folded or dispatched. This recovers the declaration, exactly as
    :func:`param_layouts` recovers a layout the passed tensor cannot carry.

    Order is preserved so diagnostics list parameters the way the signature
    does.

    Args:
        func: The Python function whose annotations to read

    Returns:
        Constexpr parameter names; every other parameter is absent
    """
    from pypto.language.typing.constexpr import ConstexprMarker  # noqa: PLC0415

    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return []
    ann_ns = _annotation_namespace(func, sig)

    return [
        name
        for name, param in sig.parameters.items()
        if name != "self" and isinstance(resolve_annotation(param.annotation, ann_ns), ConstexprMarker)
    ]


def resolve_constexpr_value(func_name: str, name: str, value: Any) -> str:
    """Render one constexpr argument into the source text it folds to.

    The rendered text *is* the value's identity: it is what reaches the
    generated program and what :meth:`JITFunction._get_source_hash` keys on, so
    hashing it keeps the compilation key and the emitted code from drifting
    apart. This is the same rule a module-level constant already follows.

    Args:
        func_name: Kernel name, for diagnostics
        name: Parameter name
        value: The value bound to it at this call site

    Returns:
        The generated-source text this value folds to

    Raises:
        TypeError: if the specializer cannot render @p value into source.
    """
    from .specializer import constant_source  # noqa: PLC0415

    rendered = constant_source(value)
    if rendered is None:
        raise TypeError(
            f"@pl.jit function '{func_name}': constexpr parameter '{name}' got "
            f"{value!r}, which has no source form the specializer can fold. A "
            f"'pl.constexpr' value must be an int, float, bool, str, a pl dtype, an "
            f"enum member such as pl.Mem.Vec, or a list of those. Annotate '{name}' "
            f"as 'pl.Scalar[dtype]' if it should instead be a runtime value."
        )
    return rendered


def signature_tensor_meta(
    annotation: Any,
    dtype: DataType,
    dyn_for_param: dict[int, DynDim],
    dynvar_cls: type,
    param_name: str = "",
    func_name: str = "",
) -> TensorMeta:
    """Build TensorMeta from a shaped ``pl.Tensor[[...], dtype]`` annotation.

    Static dims use the annotation integer; dynamic dims (``pl.dynamic`` /
    ``bind_dynamic``) get a placeholder extent because the specialized program
    remains extent-independent. ``dynvar_cls`` is the lazily-imported ``DynVar``
    type. ``param_name`` / ``func_name`` only feed diagnostics.
    """
    shape = annotation.shape
    extents = [
        1 if (i in dyn_for_param or isinstance(dim, dynvar_cls)) else int(dim) for i, dim in enumerate(shape)
    ]
    # Record annotation-only DynVars not already bound via bind_dynamic.
    dyn_dims = dict(dyn_for_param)
    for i, dim in enumerate(shape):
        if isinstance(dim, dynvar_cls) and i not in dyn_dims:
            dyn_dims[i] = DynDim(name=dim.name, literal=dim.name, static_bound=0)
    layout = _annotation_layout(annotation, param_name, func_name)
    return _build_tensor_meta(extents, dtype, dyn_dims, layout)


def signature_constexpr_value(
    func_name: str,
    name: str,
    param: inspect.Parameter,
    kwargs: dict[str, Any],
) -> Any:
    """Resolve a constexpr parameter's value in annotation-driven signature mode.

    Unlike a scalar, a constexpr parameter has no runtime slot to fall back on:
    the value *is* what gets compiled, so one must be supplied here. It comes
    from ``kwargs`` or the signature default.

    Args:
        func_name: Kernel name, for diagnostics
        name: Parameter name
        param: The ``inspect.Parameter``, consulted for its default
        kwargs: Keyword arguments passed to ``compile()`` / ``lower()``

    Returns:
        The value to fold into the artifact

    Raises:
        TypeError: if the parameter has neither a keyword value nor a default.
    """
    if name in kwargs:
        return kwargs[name]
    if param.default is not inspect.Parameter.empty:
        return param.default
    raise TypeError(
        f"@pl.jit function '{func_name}': constexpr parameter '{name}' has no value. Its "
        f"value is resolved at compile time, so it cannot be left to dispatch — pass it as "
        f"a keyword, e.g. {func_name}.compile({name}=...), or give it a signature default. "
        f"Annotate '{name}' as 'pl.Scalar[dtype]' instead to make it a runtime value."
    )


def check_signature_scalar_value(
    func_name: str,
    name: str,
    param: inspect.Parameter,
    kwargs: dict[str, Any],
) -> None:
    """Validate the value a scalar parameter was given in signature mode.

    A scalar parameter is a runtime value, so signature mode needs no value for
    it and any value supplied is inert. One is still accepted rather than
    rejected: a signature default (``value: pl.Scalar[pl.FP32] = 3.0``) is
    legitimate, and refusing a literal here would make signature mode disagree
    with a direct call, which takes the same literal and dispatches it.

    An explicit keyword is the one form that used to mean "specialize this
    value", so it warns rather than changing meaning silently.

    Args:
        func_name: Kernel name, for diagnostics
        name: Parameter name
        param: The ``inspect.Parameter``, consulted for its default
        kwargs: Keyword arguments passed to ``compile()`` / ``lower()``

    Raises:
        TypeError: if the supplied value is neither a number nor ``pl.RUNTIME``.
    """
    from pypto.language.typing.scalar import RUNTIME  # noqa: PLC0415

    if name in kwargs:
        value = kwargs[name]
        explicit_keyword = True
    elif param.default is not inspect.Parameter.empty:
        value = param.default
        explicit_keyword = False
    else:
        return

    if value is RUNTIME:
        return
    if not isinstance(value, (int, float, bool)):
        raise TypeError(
            f"@pl.jit function '{func_name}': scalar parameter '{name}' must be an int/float/bool "
            f"or pl.RUNTIME, got {type(value).__name__}."
        )
    if explicit_keyword:
        # The message, not the reported line, carries the identification: it
        # names the kernel, the parameter and the value, so Python's
        # (text, category, lineno) dedup still emits one warning per affected
        # call form. Walking out to the user's frame is not available here —
        # ``compile()`` reaches this through a ``contextlib`` decorator, so the
        # first non-PyPTO frame is stdlib rather than user code.
        warnings.warn(
            f"@pl.jit function '{func_name}': '{name}' is a scalar parameter, so "
            f"{func_name}.compile({name}={value!r}) no longer folds that value into the "
            f"artifact — the parameter stays a runtime value whose value arrives at dispatch. "
            f"Drop the keyword; to compile against a fixed constant, read it from a "
            f"module-level or closure name in the body instead of taking it as a parameter.",
            DeprecationWarning,
            stacklevel=2,
        )
