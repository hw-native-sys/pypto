# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A test case as a value.

``PTOTestCase`` welds three unrelated questions into one abstract base class —
where the IR comes from, what the data and golden are, and how it executes.
A ``Case`` separates them: it *composes* a
:class:`~harness.core.kernel_source.KernelSource` with a tensor list and a
golden callable, and says nothing at all about execution.  The harness decides
that.

The golden is an ordinary Python callable — a lambda or a closure is fine.
That works because the golden runs in the **parent** process during
pre-compilation and its result is persisted to ``data/out/*.pt``; the batched
device run in the task-submit child only compares against those files (see
``_write_golden_for_test_case`` and ``pypto.runtime.runner``). Nothing about
the golden has to survive a process boundary.

A ``Case`` implements the same duck-typed surface the compile pipeline already
calls on a ``PTOTestCase``, so it flows through the existing pipeline
unchanged. That is deliberate: the pipeline was never the thing that required
subclassing.
"""

import copy
from dataclasses import dataclass, field
from typing import Any

import torch
from pypto.ir.pass_manager import OptimizationStrategy
from pypto.pypto_core.passes import MemoryPlanner
from pypto.runtime.runner import RunConfig
from pypto.runtime.tensor_spec import ScalarSpec

from harness.core.harness import ALL_PLATFORM_IDS, PTOTestCase, TensorSpec
from harness.core.kernel_source import IRKernel, JitKernel, KernelSource, ProgramKernel


@dataclass(eq=False)
class Case:
    """One executable system-test case.

    Attributes:
        kernel: Where the IR comes from — a :class:`JitKernel`,
            :class:`ProgramKernel`, or :class:`IRKernel`.
        name: Identity of the case. Used for the artifact cache key and the
            pytest parametrize id, so it must be unique within a run.
        tensors: The case's tensors. ``None`` asks the kernel source to derive
            them, which :class:`JitKernel` can do from its sample arguments.
        golden: Called once in the parent process with a dict of all tensors by
            name. It returns the single output tensor, a dict of output name
            to tensor, or ``None`` after mutating *tensors* in place.
            ``None`` means the case has no golden (a compile-only case).
        scalars: Scalar TaskArg slots, appended after all tensor slots.
        platform: Pin the case to one platform, or ``None`` to let the item's
            parametrize variant (or the session ``--platform``) decide.
        strategy / memory_planner / enable_pypto_l0c_double_buffer: Compile
            knobs, applied identically whichever kernel source is used.
        rtol / atol: Comparison tolerance for the default elementwise check,
            applied in the parent after the device run persists its actual
            outputs. Ignored when *compare* is set.
        compare: Replace the elementwise check entirely. Called in the parent
            with ``(actual, expected)`` — both dicts of output name to tensor,
            read back from ``data/actual`` and ``data/out`` — and raises
            ``AssertionError`` to fail the case, exactly as the test's own
            ``assert`` did before it became a case. Use it when the assertion is
            not per-element: a Frobenius relative error over the whole tensor,
            a rank or sparsity property, a tolerance that varies by region.
    """

    kernel: KernelSource
    name: str
    tensors: list[TensorSpec] | None = None
    golden: Any | None = None
    scalars: "list[ScalarSpec]" = field(default_factory=list)
    platform: str | None = None
    strategy: OptimizationStrategy = OptimizationStrategy.Default
    memory_planner: MemoryPlanner | None = None
    enable_pypto_l0c_double_buffer: bool | None = None
    rtol: float = 1e-5
    atol: float = 1e-5
    compare: Any | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kernel, (JitKernel, ProgramKernel, IRKernel)):
            raise TypeError(
                f"Case '{self.name}': kernel must be a JitKernel / ProgramKernel / IRKernel, "
                f"got {type(self.kernel).__name__}"
            )
        if self.platform is not None and self.platform not in ALL_PLATFORM_IDS:
            raise ValueError(
                f"Case '{self.name}': unknown platform '{self.platform}'. Expected one of {ALL_PLATFORM_IDS}."
            )
        if self.tensors is None:
            self.tensors = self.kernel.tensor_specs()
        if self.tensors is None:
            raise ValueError(
                f"Case '{self.name}': no tensors. {self.kernel!r} cannot derive them "
                "(a @pl.program source never can, and a JitKernel needs a sample tensor for "
                "every tensor parameter), so pass tensors=[TensorSpec(...), ...]."
            )
        outputs = [t for t in self.tensors if t.is_output]
        if not outputs:
            raise ValueError(
                f"Case '{self.name}': no output tensor. Mark the result with is_output=True, "
                "or annotate the kernel parameter pl.Out[...] / pl.InOut[...] so it is derived."
            )
        if self.compare is not None and not callable(self.compare):
            raise TypeError(
                f"Case '{self.name}': compare must be callable as compare(actual, expected), "
                f"got {type(self.compare).__name__}"
            )
        self.config = RunConfig(rtol=self.rtol, atol=self.atol)

    # -- KernelSource / data ------------------------------------------------

    @property
    def tensor_specs(self) -> "list[TensorSpec]":
        """The case's tensors (pipeline surface; mirrors ``PTOTestCase``)."""
        assert self.tensors is not None  # established in __post_init__
        return self.tensors

    @property
    def scalar_specs(self) -> "list[ScalarSpec]":
        return self.scalars

    def get_name(self) -> str:
        return self.name

    def get_program(self) -> Any:
        return self.kernel.build_program()

    def get_strategy(self) -> OptimizationStrategy:
        return self.strategy

    def get_memory_planner(self) -> MemoryPlanner | None:
        return self.memory_planner

    def get_enable_pypto_l0c_double_buffer(self) -> bool | None:
        return self.enable_pypto_l0c_double_buffer

    def get_platform(self) -> str | None:
        return self.platform

    def bind_platform(self, platform: str) -> None:
        """Bind *platform* unless the case pinned one itself."""
        if platform not in ALL_PLATFORM_IDS:
            raise ValueError(f"Unknown platform '{platform}'. Expected one of {ALL_PLATFORM_IDS}.")
        if self.platform is None:
            self.platform = platform

    def for_platform(self, platform: str) -> "Case":
        """Return this case bound to *platform*, without mutating the original.

        A case is declared once, at collection time, and pytest hands that same
        object to every platform variant of the item. Binding onto it in place
        would pin the *first* variant's platform for all of them — and a case's
        own platform outranks the item's in ``_resolve_platform``, so every
        variant would then key, compile and run the first platform's artifact
        instead of its own. Each variant therefore gets its own copy, exactly
        as the legacy ``PTOTestCase`` path gets a freshly constructed instance
        per item.

        A case that pinned a platform itself is returned unchanged: the pin is
        the author's statement that this case runs nowhere else, and it must
        keep outranking the item.

        Args:
            platform: The platform resolved for the item being bound.

        Returns:
            ``self`` when the case pinned a platform, otherwise a copy bound to
            *platform*. The copy differs in that one field only; the kernel
            source, tensors and golden are deliberately shared, since they are
            read-only for the duration of a run.
        """
        if self.platform is not None:
            return self
        bound = copy.copy(self)
        bound.bind_platform(platform)
        return bound

    # -- golden -------------------------------------------------------------

    def compute_expected(
        self, tensors: dict[str, torch.Tensor], params: dict[str, Any] | None = None
    ) -> None:
        """Write the expected outputs into *tensors*, in place.

        Adapts the case's ``golden`` callable to the in-place contract the
        pipeline uses. Runs in the parent process, so the callable may close
        over anything.
        """
        del params  # the golden closes over what it needs; no params channel
        if self.golden is None:
            raise ValueError(
                f"Case '{self.name}' has no golden, so its outputs cannot be validated. "
                "Pass golden=... , or route the case through a compile-only check."
            )
        produced = self.golden(tensors)
        if produced is None:
            return  # the callable mutated *tensors* itself
        output_names = [t.name for t in self.tensor_specs if t.is_output]
        if isinstance(produced, torch.Tensor):
            if len(output_names) != 1:
                raise ValueError(
                    f"Case '{self.name}': golden returned one tensor but the kernel has "
                    f"{len(output_names)} outputs {output_names}. Return a dict keyed by output name."
                )
            tensors[output_names[0]][:] = produced.to(tensors[output_names[0]].dtype)
            return
        if isinstance(produced, dict):
            unknown = sorted(set(produced) - set(output_names))
            if unknown:
                raise ValueError(
                    f"Case '{self.name}': golden returned unknown output(s) {unknown}. "
                    f"Kernel outputs are {output_names}."
                )
            missing = sorted(set(output_names) - set(produced))
            if missing:
                raise ValueError(
                    f"Case '{self.name}': golden did not produce output(s) {missing}. "
                    f"Kernel outputs are {output_names}."
                )
            for out_name, value in produced.items():
                tensors[out_name][:] = value.to(tensors[out_name].dtype)
            return
        raise TypeError(
            f"Case '{self.name}': golden returned {type(produced).__name__}; expected a torch.Tensor, "
            "a dict of output name -> tensor, or None after mutating the tensors in place."
        )

    def __repr__(self) -> str:
        return f"Case({self.name!r}, {self.kernel!r})"


def _legacy_memory_planner(test_case: PTOTestCase) -> MemoryPlanner | None:
    """Return the planner a legacy case selects, by the harness's own precedence.

    ``_resolve_case_memory_planner`` reads two channels: ``get_memory_planner()``
    first, then a planner carried on the case's own ``RunConfig``. A ``Case``
    rebuilds that ``RunConfig`` from ``rtol`` / ``atol`` alone, so the second
    channel has to be folded into the first here or the wrapped case would
    silently fall through to the session planner.
    """
    planner = test_case.get_memory_planner()
    if planner is not None:
        return planner
    return getattr(getattr(test_case, "config", None), "memory_planner", None)


def from_legacy(test_case: PTOTestCase) -> Case:
    """Wrap an existing ``PTOTestCase`` instance as a :class:`Case`.

    Lets the 88 existing ``PTOTestCase`` files move to the collection-time
    declaration (``st.cases(...)``) without rewriting their bodies: the case
    keeps its own ``get_program`` / ``define_tensors`` / ``compute_expected``,
    and only how the harness *finds* it changes.
    """
    return Case(
        kernel=ProgramKernel(test_case.get_program, name=test_case.get_name()),
        name=test_case.get_name(),
        tensors=list(test_case.tensor_specs),
        golden=lambda tensors: test_case.compute_expected(tensors),
        scalars=list(test_case.scalar_specs),
        platform=test_case.get_platform(),
        strategy=test_case.get_strategy(),
        memory_planner=_legacy_memory_planner(test_case),
        enable_pypto_l0c_double_buffer=test_case.get_enable_pypto_l0c_double_buffer(),
        rtol=test_case.config.rtol,
        atol=test_case.config.atol,
    )


__all__ = ["Case", "from_legacy"]
