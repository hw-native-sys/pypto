# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Legacy hooks and builder callbacks at the edge of the provider boundary."""

import ast
from collections.abc import Callable, Mapping, Sequence
from typing import TypeVar, cast

from pypto.jit._specialize_types import DynDim
from pypto.jit.specializer import SpecializeContext

from .contracts import (
    CompileTarget,
    Constant,
    KernelRequest,
    ParameterSpec,
    PreparedKernel,
    ScalarSpec,
    TensorSpec,
)
from .registry import KernelRef
from .validation import ContractError, validate_request

T = TypeVar("T")


def make_request(
    parameters: Sequence[ParameterSpec],
    target: CompileTarget,
    *,
    constants: Mapping[str, Constant] | None = None,
    options: Mapping[str, Constant] | None = None,
) -> KernelRequest:
    """Own the input containers and reject any mutable nested signature data."""
    # tuple(set(...)) would freeze an arbitrary physical parameter order.
    if not issubclass(type(parameters), Sequence):
        raise ContractError("request: expected an ordered parameter sequence")
    request = KernelRequest(
        tuple(parameters), target, tuple((constants or {}).items()), tuple((options or {}).items())
    )
    validate_request(request)
    return request


def request_from_context(
    context: SpecializeContext,
    target: CompileTarget,
    *,
    options: Mapping[str, Constant] | None = None,
) -> KernelRequest:
    """Copy only normalized metadata; never expose the context or its namespace.

    Dynamic dimensions are renamed in first-use order while retaining equality
    relationships. Inexact numeric bounds become independent dynamic symbols,
    rather than falsely promising a fixed extent. Missing layout, strides and
    offsets are rejected or left unconstrained; they are not inferred from shape.
    """
    parameters: list[ParameterSpec] = []
    constants: dict[str, Constant] = {}
    symbols: dict[str, str] = {}
    declared = set(context.param_names)
    categorized = [set(context.tensor_meta), set(context.scalar_dtypes), set(context.constexpr_values)]
    if set[str]().union(*categorized) != declared or sum(map(len, categorized)) != len(declared):
        raise ContractError(
            "request: each parameter must have exactly one tensor, scalar or constant binding"
        )
    if len(declared) != len(context.param_names):
        raise ContractError("request: duplicate parameter names in specialization context")
    for name in context.param_names:
        if name in context.tensor_meta:
            meta = context.tensor_meta[name]
            if meta.layout is None:
                raise ContractError(f"request: tensor {name!r} needs a resolved layout")
            shape: list[int | str] = []
            for axis, dimension in enumerate(meta.shape):
                if isinstance(dimension, DynDim):
                    shape.append(symbols.setdefault(dimension.name, f"d{len(symbols)}"))
                elif axis in meta.inexact_axes:
                    # Prefix differs from shared symbols, so independent runtime
                    # extents cannot accidentally acquire equality constraints.
                    shape.append(f"extent_{len(parameters)}_{axis}")
                else:
                    shape.append(dimension)
            parameters.append(ParameterSpec(name, TensorSpec(tuple(shape), meta.dtype, layout=meta.layout)))
        elif name in context.scalar_dtypes:
            parameters.append(ParameterSpec(name, ScalarSpec(context.scalar_dtypes[name])))
        else:
            # The legacy context stores source spellings, not constant values.
            # literal_eval accepts literals only; it never executes globals/AST.
            try:
                value = ast.literal_eval(context.constexpr_values[name])
            except (ValueError, SyntaxError) as error:
                raise ContractError(f"request: constant {name!r} must be a literal") from error
            constants[name] = cast(Constant, value)
    return make_request(parameters, target, constants=constants, options=options)


def legacy_builder(
    reference: KernelRef,
    target: CompileTarget,
    lower: Callable[[PreparedKernel], T],
    *,
    options: Mapping[str, Constant] | None = None,
) -> Callable[[SpecializeContext], T]:
    """Adapt a context-based builder to capture/prepare and an explicit lowering.

    The caller supplies runtime-specific conversion of PreparedKernel to its
    existing builder result. The provider sees only owned immutable inputs.
    Each invocation captures again; incomplete dependency identity is never
    silently promoted to a reusable preparation-cache entry.
    """
    frozen_options = dict(options or {})

    def build(context: SpecializeContext) -> T:
        request = request_from_context(context, target, options=frozen_options)
        return lower(reference.capture(request).prepare())

    return build
