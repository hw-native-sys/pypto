# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""How a system test declares its cases.

A case must be a **collection-time value**, not a statement inside the test
body.  The pre-compile pool has to know every case before any test body runs,
and the only way to hand pytest a value that early is to parametrize with it::

    from harness import st
    from examples.beginner.abs import tile_abs   # @pl.jit.incore

    @st.cases(
        st.case(tile_abs, a, out, golden=lambda t: torch.abs(t["a"])),
    )
    def test_tile_abs(case_run):
        case_run.assert_passed()

``st.cases`` is a thin wrapper over ``pytest.mark.parametrize``, so the case
object lands in ``item.callspec.params["_st_case"]`` and the harness reads it
directly.  It replaces the previous discovery route, which parsed the test
function's source and re-evaluated its constructor call with a miniature AST
interpreter — a route that could not see a ``@pl.jit`` test at all, and that
fell back silently to per-case inline compilation whenever an argument would
not resolve.

Usage note: declare the case at module level or inside the decorator call.
Building it inside the test body puts it back out of reach of collection.
"""

from pathlib import Path
from typing import Any

import pytest
from pypto.ir.pass_manager import OptimizationStrategy
from pypto.jit import JITFunction
from pypto.pypto_core import ir as _ir
from pypto.pypto_core.passes import MemoryPlanner
from pypto.runtime.runner import RunResult
from pypto.runtime.tensor_spec import ScalarSpec

from harness.core.case import Case, from_legacy
from harness.core.harness import TensorSpec
from harness.core.kernel_source import IRKernel, JitKernel, ProgramKernel


def case(  # noqa: PLR0913 — every knob mirrors one Case field; grouping them would only hide them
    kernel: Any,
    *sample_args: Any,
    name: str | None = None,
    inputs: dict[str, Any] | None = None,
    golden: Any | None = None,
    tensors: list[TensorSpec] | None = None,
    scalars: list[ScalarSpec] | None = None,
    platform: str | None = None,
    strategy: OptimizationStrategy = OptimizationStrategy.Default,
    memory_planner: MemoryPlanner | None = None,
    enable_pypto_l0c_double_buffer: bool | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    compare: Any | None = None,
) -> Case:
    """Build one :class:`~harness.core.case.Case`.

    The kernel source is chosen from what *kernel* is, so the test does not
    name it: a ``@pl.jit`` function becomes a ``JitKernel``, an ``ir.Program``
    an ``IRKernel``, anything else (a ``@pl.program`` class, or a factory
    returning one) a ``ProgramKernel``.

    Args:
        kernel: The kernel under test.
        *sample_args: Sample tensors, positionally, for a ``@pl.jit`` kernel.
            They carry the shape/dtype contract and are the case's real inputs,
            so the tensor list is derived from them.
        name: Case identity — the artifact cache key and the parametrize id.
            Defaults to a name derived from the kernel and its specialization.
        inputs: Sample tensors by parameter name, for a ``@pl.jit`` kernel that
            reads better bound by keyword. Combines with *sample_args*.
        golden: Called once in the parent with every tensor by name; returns
            the output tensor, a dict of output name to tensor, or ``None``
            after mutating in place.
        tensors: Declare the tensor list explicitly. Required for a
            ``@pl.program`` / ``ir.Program`` kernel, which cannot derive it.
        scalars: Scalar TaskArg slots, after all tensor slots.
        platform: Pin one platform; ``None`` follows the session matrix.
        strategy: Pass-pipeline optimization strategy.
        memory_planner: On-chip memory planner.
        enable_pypto_l0c_double_buffer: Opt in to dbC=2 under the PyPTO planner.
        rtol: Relative tolerance for the default elementwise comparison.
        atol: Absolute tolerance for the default elementwise comparison.
        compare: Replace the elementwise comparison. Called in the parent as
            ``compare(actual, expected)`` with dicts of output name to tensor,
            and raises ``AssertionError`` to fail — the same ``assert`` the
            test wrote before it became a case. For an assertion that is not
            per-element, such as a Frobenius relative error::

                def rel_err_under(limit):
                    def compare(actual, expected):
                        for name, got in actual.items():
                            ref = expected[name].float()
                            rel = (got.float() - ref).norm() / ref.norm()
                            assert rel < limit, f"{name}: rel_err {rel:.3e} > {limit}"
                    return compare

            ``rtol`` / ``atol`` are ignored when this is set.

    Returns:
        The case, ready to hand to :func:`cases`.
    """
    if isinstance(kernel, JITFunction):
        source = JitKernel(kernel, *sample_args, **(inputs or {}))
    elif isinstance(kernel, _ir.Program):
        if name is None:
            raise ValueError("st.case(ir.Program) needs an explicit name= — a Program carries none")
        source = IRKernel(kernel, name=name)
    else:
        if sample_args or inputs:
            raise ValueError(
                f"st.case({kernel!r}): sample arguments only apply to a @pl.jit kernel. "
                "A @pl.program declares its shapes in its annotations; pass tensors=[...] "
                "to say how they are filled."
            )
        source = ProgramKernel(kernel, name=name)

    return Case(
        kernel=source,
        name=name or source.cache_id(),
        tensors=tensors,
        golden=golden,
        scalars=scalars or [],
        platform=platform,
        strategy=strategy,
        memory_planner=memory_planner,
        enable_pypto_l0c_double_buffer=enable_pypto_l0c_double_buffer,
        rtol=rtol,
        atol=atol,
        compare=compare,
    )


def _unwrap_entry(entry: Any) -> "tuple[Case, str]":
    """Return ``(case, id)`` for one :func:`cases` entry.

    An entry is either a bare :class:`Case` or a ``pytest.param(case, marks=...)``
    wrapping one. ``pytest.param`` returns a ``ParameterSet`` namedtuple carrying
    ``values`` / ``marks`` / ``id``; it is matched structurally rather than by
    importing pytest's private class.
    """
    if isinstance(entry, Case):
        return entry, entry.name
    values = getattr(entry, "values", None)
    if values is not None and hasattr(entry, "marks"):
        if len(values) != 1 or not isinstance(values[0], Case):
            raise ValueError(
                "st.cases(): pytest.param(...) must wrap exactly one Case, got "
                f"{[type(v).__name__ for v in values]}"
            )
        return values[0], entry.id or values[0].name
    raise TypeError(
        f"st.cases(): expected a Case or pytest.param(Case, marks=...), got {type(entry).__name__}"
    )


def cases(*entries: Any) -> Any:
    """Parametrize a test over *entries*, one item per case.

    Per-case markers go through pytest's own mechanism::

        @st.cases(
            st.case(kernel_a, x, out, golden=...),
            pytest.param(
                st.case(kernel_b, x, out, golden=...),
                marks=pytest.mark.skip(reason="codegen bug: ..."),
            ),
        )
        def test_kernels(case_run):
            case_run.assert_passed()

    A skipped case is also left out of pre-compilation, so it costs nothing.
    Markers that apply to every case (``@pytest.mark.platforms``) stay on the
    test function or its class as usual.

    Args:
        *entries: Each a :class:`Case` from :func:`case`, or a
            ``pytest.param(case, marks=...)`` wrapping one.

    Returns:
        A ``pytest.mark.parametrize`` decorator binding the harness-owned
        ``_st_case`` argument; tests reach the case through ``case_run``.

    Raises:
        ValueError: No entries were given, an entry wraps something other than a
            single case, or two cases share a name — duplicate names would
            collide in the artifact cache and produce indistinguishable ids.
        TypeError: An entry is neither a Case nor a ``pytest.param``.
    """
    if not entries:
        raise ValueError("st.cases() needs at least one case")
    unwrapped = [_unwrap_entry(e) for e in entries]
    names = [c.name for c, _ in unwrapped]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(
            f"st.cases(): duplicate case name(s) {duplicates}; names must be unique within a run"
        )
    # ``_st_case``, not ``case``: several suites already parametrize their own
    # ``case`` argument (test_expand_ops, the all_to_all_v skew tests), and a
    # harness-owned param must not collide with a test's own. Mirrors the
    # existing ``_st_platform``. The pytest id still comes from ids, so the
    # underscore never shows up in a node id.
    return pytest.mark.parametrize("_st_case", list(entries), ids=[i for _, i in unwrapped])


def rel_err_under(limit: float) -> Any:
    """A :func:`case` ``compare=`` that bounds the Frobenius relative error.

    ``‖actual - expected‖_F / ‖expected‖_F < limit``, per output. This is the
    assertion a matmul-shaped test wants: accumulation order across cores and
    tiles moves individual elements far more than it moves the tensor as a
    whole, so an elementwise tolerance loose enough to pass would stop catching
    anything.

    Args:
        limit: The exclusive upper bound on the relative error.

    Returns:
        A comparator for ``st.case(..., compare=...)``.
    """

    def compare(actual: "dict[str, Any]", expected: "dict[str, Any]") -> None:
        for name, got in actual.items():
            reference = expected[name].float()
            rel_err = ((got.float() - reference).norm() / reference.norm()).item()
            assert rel_err < limit, f"{name}: Frobenius rel_err {rel_err:.3e} exceeds {limit}"

    return compare


class CaseRun:
    """What a test asserts on after its case has run.

    Attributes:
        case: The case that produced this run.
        result: The raw ``RunResult`` from the harness.
    """

    __test__ = False  # not a pytest test class

    def __init__(self, case_obj: Case, result: RunResult):
        self.case = case_obj
        self.result = result

    @property
    def passed(self) -> bool:
        return self.result.passed

    @property
    def error(self) -> str | None:
        return self.result.error

    @property
    def work_dir(self) -> Path | None:
        """The compiled artifact directory, or ``None`` if it was not cached.

        Where the generated kernels, ``golden.py`` and any ``dfx_outputs/``
        live — for a test that asserts on artifacts rather than only on the
        numeric result.
        """
        from harness.core.test_runner import artifact_work_dir  # noqa: PLC0415

        return artifact_work_dir(self.case)

    def assert_passed(self) -> None:
        """Fail the test with the harness's own error when the case did not pass."""
        assert self.passed, f"{self.case.name} failed: {self.error}"

    def __repr__(self) -> str:
        return f"CaseRun({self.case.name!r}, passed={self.passed})"


@pytest.fixture
def case_run(_st_case: Case, test_runner: Any) -> CaseRun:
    """Run the item's case and return its result.

    Requesting ``test_runner`` is what routes the case through the pre-compile
    and batched-execute pipeline: ``pytest_itemcollected`` marks any item whose
    fixture closure reaches ``test_runner`` as ``device_batch``.
    """
    return CaseRun(_st_case, test_runner.run(_st_case))


__all__ = ["Case", "CaseRun", "case", "case_run", "cases", "from_legacy", "rel_err_under"]
