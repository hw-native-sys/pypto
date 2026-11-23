# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Request-owned native JIT adapters, separate from immutable capture providers."""

import inspect
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Protocol

from pypto.jit._source import cache_in_snapshot

from .protocols import ExternalJITBinding
from .validation import ContractError

if TYPE_CHECKING:
    from pypto.jit.decorator import JITFunction
    from pypto.jit.specializer import SpecializeContext
    from pypto.pypto_core.passes import RuntimeKind


@dataclass(frozen=True)
class JITDeclaration:
    """Provider-selected signature and preparation, without a native body in JIT."""

    signature: inspect.Signature
    binding: ExternalJITBinding


class JITProvider(Protocol):
    def discover(self, kernel: object) -> JITDeclaration | None:
        """Inspect an already-loaded DSL object without tracing or compilation."""
        ...


@dataclass(frozen=True, eq=False)
class _Identity:
    value: object

    def __hash__(self) -> int:
        return id(self.value)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Identity) and self.value is other.value


@dataclass(frozen=True)
class _OwnedBinding:
    binding: ExternalJITBinding
    source: TemporaryDirectory[str]

    def prepare(
        self, context: "SpecializeContext", platform: str, runtime: "RuntimeKind", output_dir: str
    ) -> None:
        self.binding.prepare(context, platform, runtime, output_dir)
        # Metadata is now normalized and validated. Classify inferred tensors
        # for the ordinary extern emitter; scalar dtypes already live in context.
        parameters = [
            f"{name}: pl.Tensor" if name in context.tensor_meta else name for name in context.param_names
        ]
        context.source = f"def native_kernel({', '.join(parameters)}):\n    ...\n"


def _providers() -> tuple[JITProvider, ...]:
    # The adapter imports no optional DSL. Its public inspection API is used
    # only if that DSL is already loaded by the caller.
    from .providers.cannbotdsl import CANNBotDSLProvider  # noqa: PLC0415

    return (CANNBotDSLProvider(),)


@cache_in_snapshot
def _discover(identity: _Identity) -> "JITFunction | None":
    from pypto.jit.decorator import JITFunction  # noqa: PLC0415

    declarations = [
        declaration
        for provider in _providers()
        if (declaration := provider.discover(identity.value)) is not None
    ]
    if not declarations:
        return None
    if len(declarations) != 1:
        raise ContractError("JIT discovery: ambiguous providers; use an explicit extern binding")
    declaration = declarations[0]
    parameters = tuple(declaration.signature.parameters.values())
    if any(
        p.kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        or p.default is not inspect.Parameter.empty
        or p.name == "self"
        for p in parameters
    ):
        raise ContractError("JIT discovery requires explicit positional parameters without defaults or self")
    # Only names cross this boundary. Tensor/scalar types come from the call
    # site; provider annotations remain owned and checked by the native exporter.
    # An owned on-disk declaration lets existing source/AST analysis operate
    # normally, without inspecting or traversing the optional DSL's body.
    source = f"def native_kernel({', '.join(p.name for p in parameters)}):\n    ...\n"
    directory = TemporaryDirectory(prefix="pypto-native-jit-")
    path = Path(directory.name) / "declaration.py"
    path.write_text(source, encoding="utf-8")
    namespace: dict[str, Any] = {"__name__": __name__}
    exec(compile(source, str(path), "exec"), namespace)  # noqa: S102
    return JITFunction(
        namespace["native_kernel"],
        func_type="extern",
        external_core_type="mixed",
        external_binding=_OwnedBinding(declaration.binding, directory),
    )


def discover_jit(kernel: object) -> "JITFunction | None":
    """Deduplicate aliases within one request, without a global kernel cache.

    Identity keys support unhashable kernels and never invoke user equality.
    The resulting graph owns each kernel and its generated declaration. No DSL
    object is placed in the declaration's globals or the source-analysis caches.
    """
    return _discover(_Identity(kernel))
