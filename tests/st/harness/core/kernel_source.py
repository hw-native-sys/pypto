# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Where a test case's IR comes from.

The compile pipeline needs exactly one thing from a test case: an
``ir.Program`` that **no pass has run on yet**, because ``ir.compile()`` runs
the pass pipeline *and* code generation itself.  Everything else about how the
kernel was authored is irrelevant to it.

``KernelSource`` is that one thing, so the three authoring surfaces stop being
three execution paths:

======================  =========================================
Authoring surface       KernelSource
======================  =========================================
``@pl.jit`` entry       :class:`JitKernel`
``@pl.program`` class   :class:`ProgramKernel`
an ``ir.Program``       :class:`IRKernel`
======================  =========================================

A source may also derive its own :class:`~harness.core.harness.TensorSpec`
list.  ``JitKernel`` can — parameter directions come from the kernel's
``pl.Out`` / ``pl.InOut`` annotations and shapes come from the sample tensors —
so a JIT-authored case never writes its shapes a second time.  The other two
return ``None`` and let the case declare them.
"""

import threading
from typing import Any, Protocol, runtime_checkable

import torch
from pypto.jit import JITFunction

from harness.core.harness import DataType, TensorSpec

# Serialises every program build across the compile pool.
#
# Neither authoring surface is thread-safe: the ``@pl.program`` decorator
# mutates module-level builder state, and ``JITFunction``'s specializer walks
# and memoises a shared dep graph.  The compile pool runs ``build_program()``
# under this lock and everything after it — ``ir.compile()``, golden
# materialisation, the .so build — concurrently, which is where the parallelism
# actually is.
#
# One lock for all sources, on purpose: a per-source lock would let a
# ``@pl.program`` build race a JIT specialization, and the JIT specializer
# emits ``@pl.program`` source.
#
# Reentrant, because the acquisition nests: ``_compile_for_cache`` takes the
# lock around ``get_program()``, and for a ``Case`` that call lands right back
# here in ``build_program()``. Re-entering from the owning thread is safe by
# construction — the thread already has exclusive access and the nested build
# finishes before it releases — whereas a plain Lock self-deadlocks.
program_build_lock = threading.RLock()


@runtime_checkable
class KernelSource(Protocol):
    """Supplies the pre-pass IR for one test case."""

    def build_program(self) -> Any:
        """Return the ``ir.Program`` to compile, before any pass has run.

        Called on a compile-pool worker. Implementations MUST hold
        :data:`program_build_lock` for the duration of the build.
        """
        ...

    def tensor_specs(self) -> list[TensorSpec] | None:
        """Return the case's tensors, or ``None`` to let the case declare them."""
        ...

    def cache_id(self) -> str:
        """Stable identity for the compiled artifact of this kernel."""
        ...


# ---------------------------------------------------------------------------
# torch dtype -> harness DataType
# ---------------------------------------------------------------------------

# ``DataType.torch_dtype`` is many-to-one (UINT32 and INT32 both map to
# torch.int32, UINT16 and INT16 both to torch.int16), so the reverse map is
# built explicitly and resolves each collision to the signed member. A test
# that needs the unsigned reading declares that TensorSpec itself.
_TORCH_TO_DATATYPE: "dict[torch.dtype, DataType]" = {}
for _member in DataType:
    try:
        _torch_dtype = _member.torch_dtype
    except (ValueError, KeyError):
        continue  # optional MX dtype this torch build does not provide
    _TORCH_TO_DATATYPE.setdefault(_torch_dtype, _member)
del _member


def datatype_from_torch(dtype: torch.dtype) -> DataType:
    """Return the harness :class:`DataType` matching a torch dtype.

    Raises:
        ValueError: The dtype has no harness equivalent, naming what was
            received and what is available.
    """
    try:
        return _TORCH_TO_DATATYPE[dtype]
    except KeyError:
        known = ", ".join(sorted(str(d) for d in _TORCH_TO_DATATYPE))
        raise ValueError(f"No harness DataType for torch dtype {dtype}. Known dtypes: {known}") from None


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


class JitKernel:
    """A ``@pl.jit`` entry plus the sample arguments that specialize it.

    The sample arguments carry the shape/dtype contract, exactly as they do
    when a test calls the kernel directly.  They are the case's real inputs, so
    :meth:`tensor_specs` derives the whole tensor list from them and the test
    never restates a shape.

    Sample arguments may be omitted only when every tensor parameter carries a
    fully-shaped annotation (``pl.Tensor[[128, 128], pl.FP32]``); a bare
    ``pl.Tensor`` has no shape to read and ``specialize()`` says so.
    """

    def __init__(self, entry: Any, *args: Any, **kwargs: Any):
        # ``entry`` is typed Any because the guard below is what test authors
        # actually hit: passing a @pl.program class here is the natural mistake,
        # and it must fail with a message that names the right source.
        if not isinstance(entry, JITFunction):
            raise TypeError(
                f"JitKernel expects a @pl.jit function, got {type(entry).__name__}. "
                "Use ProgramKernel for a @pl.program class or IRKernel for an ir.Program."
            )
        if "config" in kwargs:
            raise ValueError(
                f"JitKernel({entry.__name__}) does not take config=: compile knobs "
                "(strategy, memory_planner, platform) belong on the Case, which applies "
                "them to ir.compile() for every source alike."
            )
        self.entry: JITFunction = entry
        self.args = args
        self.kwargs = kwargs
        self._classified: tuple[set[str], set[str]] | None = None

    def build_program(self) -> Any:
        with program_build_lock:
            return self.entry.specialize(*self.args, **self.kwargs)

    def tensor_specs(self) -> list[TensorSpec] | None:
        """Derive the tensor list from the kernel signature and sample args.

        A parameter is an output when the kernel annotates it ``pl.Out[...]``
        or ``pl.InOut[...]``.  ``pl.InOut`` additionally keeps its incoming
        data, so it is materialised from the sample tensor rather than zeroed.

        Returns ``None`` when the sample arguments do not cover every tensor
        parameter (annotation-only specialization) — the case declares its
        tensors in that case.
        """
        bound = self._bound_tensors()
        if bound is None:
            return None
        outputs = set(self.entry.output_param_names)
        inout_names = self._classify()[0]
        specs: list[TensorSpec] = []
        for name, value in bound.items():
            is_output = name in outputs
            # An Out param's incoming buffer is scratch, so it is left to be
            # zero-initialised; an InOut param's is live input and is seeded.
            keep_input = not is_output or name in inout_names
            specs.append(
                TensorSpec(
                    name=name,
                    shape=list(value.shape),
                    dtype=datatype_from_torch(value.dtype),
                    init_value=value if keep_input else None,
                    is_output=is_output,
                )
            )
        return specs

    def cache_id(self) -> str:
        parts = [self.entry.__name__]
        bound = self._bound_tensors() or {}
        for name in self.entry.param_names:
            value = bound.get(name)
            if value is not None:
                shape = "x".join(str(d) for d in value.shape)
                parts.append(f"{name}_{shape}_{datatype_from_torch(value.dtype).value}")
        for name, value in sorted(self.kwargs.items()):
            if not isinstance(value, torch.Tensor):
                parts.append(f"{name}_{value}")
        return "__".join(parts)

    def _classify(self) -> "tuple[set[str], set[str]]":
        """Return ``(inout_param_names, tensor_param_names)``, parsed once.

        ``output_param_names`` already covers Out and InOut together; what it
        cannot say is which of those *also* read their incoming data, and which
        params own a tensor slot at all. Both come from the same annotation
        walk, so it runs once per kernel and is cached.
        """
        if self._classified is None:
            from pypto.jit.decorator import _get_func_def  # noqa: PLC0415
            from pypto.jit.specializer import _classify_params  # noqa: PLC0415

            classified = _classify_params(_get_func_def(self.entry._func))
            self._classified = (set(classified[1]), set(classified[2]))
        return self._classified

    def _bound_tensors(self) -> dict[str, torch.Tensor] | None:
        """Sample tensors keyed by parameter name, or ``None`` if incomplete."""
        names = self.entry.param_names
        bound: dict[str, torch.Tensor] = {}
        for name, value in zip(names, self.args):
            if isinstance(value, torch.Tensor):
                bound[name] = value
        for name, value in self.kwargs.items():
            if isinstance(value, torch.Tensor):
                bound[name] = value
        # Every *tensor* parameter must be covered. A scalar parameter is
        # specialized as a literal and owns no tensor slot, so it is not
        # missing when absent from ``bound``.
        tensor_params = self._classify()[1]
        missing = [n for n in names if n not in bound and n in tensor_params]
        return None if missing else {n: bound[n] for n in names if n in bound}

    def __repr__(self) -> str:
        return f"JitKernel({self.entry.__name__})"


class ProgramKernel:
    """A ``@pl.program`` class (or a zero-argument factory returning one)."""

    def __init__(self, program: Any, *, name: str | None = None):
        if program is None:
            raise ValueError("ProgramKernel requires a @pl.program class or a factory returning one")
        self.program = program
        self._name = name or getattr(program, "__name__", type(program).__name__)

    def build_program(self) -> Any:
        # A @pl.program class is itself the program, so only a *plain* callable
        # (``PTOTestCase.get_program``, a factory) is invoked here.
        is_factory = callable(self.program) and not isinstance(self.program, type)
        with program_build_lock:
            built = self.program() if is_factory else self.program
            if built is None:
                raise ValueError(f"ProgramKernel({self._name}) factory returned None")
            return built

    def tensor_specs(self) -> list[TensorSpec] | None:
        # A @pl.program's parameter shapes live in its annotations, but the
        # initial values do not; the case declares the tensors.
        return None

    def cache_id(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"ProgramKernel({self._name})"


class IRKernel:
    """An already-built ``ir.Program`` — parser round-trips, hand-built IR."""

    def __init__(self, program: Any, *, name: str):
        if program is None:
            raise ValueError("IRKernel requires an ir.Program")
        self.program = program
        self._name = name

    def build_program(self) -> Any:
        return self.program

    def tensor_specs(self) -> list[TensorSpec] | None:
        return None

    def cache_id(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"IRKernel({self._name})"


__all__ = [
    "IRKernel",
    "JitKernel",
    "KernelSource",
    "ProgramKernel",
    "datatype_from_torch",
    "program_build_lock",
]
