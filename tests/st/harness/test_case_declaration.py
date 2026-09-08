# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Device-free tests for the ``st.case`` declaration surface.

Covers the three things a case has to get right before any card is involved:

1. a ``@pl.jit`` kernel and a ``@pl.program`` class produce the *same* compiled
   artifact — the claim that lets one execution path serve both;
2. the tensor list a JIT case derives from its sample arguments matches what a
   hand-written ``define_tensors()`` would have declared;
3. the golden callable is adapted to the pipeline's in-place contract, in every
   return shape it accepts.

Nothing here touches a device, and the compile check is skipped when ``ptoas``
is unavailable.
"""

import ast
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pypto.language as pl
import pytest
import torch
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.jit.decorator import _ptoas_available, jit
from pypto.pypto_core import ir as _ir
from pypto.pypto_core.passes import MemoryPlanner
from pypto.runtime.runner import RunConfig, validate_persisted_outputs

from harness import st
from harness.core.case import Case, from_legacy
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from harness.core.kernel_source import JitKernel, ProgramKernel, datatype_from_torch
from harness.core.test_runner import (
    _case_comparator,
    _compare_persisted_outputs,
    _compile_for_cache,
)

M = 16
N = 16


# ---------------------------------------------------------------------------
# The same kernel, authored both ways. Function names match on purpose: they
# are part of the IR, so a naming difference would mask a structural one.
# ---------------------------------------------------------------------------


@jit.incore
def kernel(a: pl.Tensor, out: pl.Out[pl.Tensor]):
    tile_a = pl.load(a, [0, 0], [M, N])
    return pl.store(pl.tile.abs(tile_a), [0, 0], out)


@jit
def orchestrator(a: pl.Tensor, out: pl.Out[pl.Tensor]):
    out = kernel(a, out)
    return out


@pl.program
class AbsProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[M, N], pl.FP32],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        tile_a = pl.load(a, [0, 0], [M, N])
        out = pl.store(pl.tile.abs(tile_a), [0, 0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[M, N], pl.FP32],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        out = self.kernel(a, out)
        return out


class AbsLegacyCase(PTOTestCase):
    """The pre-``Case`` way of declaring the very same test."""

    __test__ = False

    def get_name(self) -> str:
        return "abs_legacy"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [M, N], DataType.FP32, init_value=torch.randn(M, N)),
            TensorSpec("out", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return AbsProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = torch.abs(tensors["a"])


@jit.incore
def scaled_kernel(a: pl.Tensor, out: pl.Out[pl.Tensor], scale: float):
    """Abs then scale — gives the declaration surface a scalar to specialize on."""
    return pl.store(pl.tile.abs(pl.load(a, [0, 0], [M, N])) * scale, [0, 0], out)


@jit
def scaled_entry(a: pl.Tensor, out: pl.Out[pl.Tensor], scale: float):
    """Entry over :func:`scaled_kernel`."""
    out = scaled_kernel(a, out, scale)
    return out


def _jit_case(**kwargs: Any) -> Case:
    """A JIT-authored case over the shared ``a`` / ``out`` signature."""
    kwargs.setdefault("name", "abs_jit")
    kwargs.setdefault("golden", lambda t: torch.abs(t["a"]))
    return st.case(orchestrator, torch.randn(M, N), torch.zeros(M, N), **kwargs)


# ---------------------------------------------------------------------------


class TestKernelSourceEquivalence:
    """A JIT kernel and a @pl.program class compile to the same thing."""

    def test_same_ir_after_passes(self):
        """The two sources' programs are structurally equal once lowered."""
        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        jit_program = JitKernel(orchestrator, torch.randn(M, N), torch.zeros(M, N)).build_program()
        _ir.assert_structural_equal(pm.run_passes(jit_program), pm.run_passes(AbsProgram))

    @pytest.mark.skipif(not _ptoas_available(), reason="ptoas is required to generate kernel sources")
    def test_same_generated_artifacts(self):
        """Both sources satisfy ``_compile_for_cache`` and emit the same files.

        This is the integration claim: the existing compile task, unchanged,
        accepts a ``Case`` whichever surface authored its kernel.
        """
        produced: dict[str, tuple[list[str], list[str], bool]] = {}
        for label, case_obj in (
            ("jit", _jit_case(platform="a2a3")),
            (
                "program",
                st.case(
                    AbsProgram,
                    name="abs_program",
                    platform="a2a3",
                    tensors=AbsLegacyCase().define_tensors(),
                    golden=lambda t: torch.abs(t["a"]),
                ),
            ),
        ):
            work_dir = Path(tempfile.mkdtemp(prefix=f"case_decl_{label}_"))
            try:
                _compile_for_cache(case_obj, work_dir, "a2a3", False, False, None)
                produced[label] = (
                    sorted(p.name for p in (work_dir / "kernels").rglob("*.cpp")),
                    sorted(p.name for p in (work_dir / "orchestration").glob("*.cpp")),
                    (work_dir / "golden.py").exists(),
                )
                # The golden ran in this process and was persisted for the child.
                assert (work_dir / "data" / "out" / "out.pt").exists(), (
                    f"{label}: golden outputs were not persisted to data/out/"
                )
            finally:
                shutil.rmtree(work_dir, ignore_errors=True)

        assert produced["jit"][0] == produced["program"][0], "kernel sources differ"
        assert produced["jit"][1] == produced["program"][1], "orchestration sources differ"
        assert produced["jit"][2] and produced["program"][2], "golden.py was not written"

    def test_program_kernel_accepts_a_factory(self):
        """``ProgramKernel`` invokes a plain callable, and passes a class through."""
        assert ProgramKernel(AbsProgram).build_program() is AbsProgram
        assert ProgramKernel(lambda: AbsProgram, name="factory").build_program() is AbsProgram

    def test_jit_kernel_rejects_a_program_class(self):
        """The natural mistake fails with a message naming the right source."""
        with pytest.raises(TypeError, match="Use ProgramKernel"):
            JitKernel(AbsProgram)


class TestDerivedTensorSpecs:
    """A JIT case derives the tensor list its author would have written."""

    def test_matches_the_hand_written_declaration(self):
        derived = _jit_case().tensor_specs
        expected = AbsLegacyCase().define_tensors()
        assert [s.name for s in derived] == [s.name for s in expected]
        assert [s.shape for s in derived] == [s.shape for s in expected]
        assert [s.dtype for s in derived] == [s.dtype for s in expected]
        assert [s.is_output for s in derived] == [s.is_output for s in expected]

    def test_every_tensor_is_seeded_from_its_sample_argument(self):
        """Outputs included — the sample argument is the buffer the test prepared.

        A kernel that accumulates onto its destination (atomic-add, and any
        ``pl.InOut``) reads what it is handed, so the baseline a test fills in
        must reach the device. Deciding from the annotation alone that a
        ``pl.Out`` buffer is scratch would silently zero that baseline and
        compare against a golden that assumed it.
        """
        baseline = torch.full((M, N), 7.0)
        case_obj = st.case(
            orchestrator,
            torch.randn(M, N),
            baseline,
            name="abs_seeded_out",
            golden=lambda t: torch.abs(t["a"]),
        )
        specs = {s.name: s for s in case_obj.tensor_specs}
        assert specs["out"].is_output
        assert specs["out"].init_value is baseline, "an Out buffer's contents are the test's statement"
        assert specs["a"].init_value is not None

    def test_inout_param_is_an_output_and_keeps_its_input(self):
        """``pl.InOut`` is reported as an output and seeded."""

        @jit.incore
        def accumulate_k(x: pl.Tensor, acc: pl.InOut[pl.Tensor]):
            total = pl.tile.add(pl.load(x, [0, 0], [M, N]), pl.load(acc, [0, 0], [M, N]))
            return pl.store(total, [0, 0], acc)

        @jit
        def accumulate(x: pl.Tensor, acc: pl.InOut[pl.Tensor]):
            acc = accumulate_k(x, acc)
            return acc

        specs = {
            s.name: s
            for s in st.case(
                accumulate,
                torch.randn(M, N),
                torch.randn(M, N),
                name="acc_inout",
                golden=lambda t: t["x"] + t["acc"],
            ).tensor_specs
        }
        assert specs["acc"].is_output and specs["acc"].init_value is not None, "InOut keeps its input"
        assert specs["x"].init_value is not None

    def test_dtype_round_trips(self):
        """Every harness dtype this torch build supports maps back to itself."""
        for member in DataType:
            try:
                torch_dtype = member.torch_dtype
            except ValueError:
                continue  # optional MX dtype this build lacks
            assert datatype_from_torch(torch_dtype).torch_dtype is torch_dtype

    def test_unknown_dtype_names_what_is_available(self):
        with pytest.raises(ValueError, match="No harness DataType for torch dtype"):
            datatype_from_torch(torch.complex64)

    def test_program_case_without_tensors_is_rejected(self):
        """A @pl.program cannot derive its tensors, and the error says so."""
        with pytest.raises(ValueError, match="pass tensors="):
            st.case(AbsProgram, name="no_tensors", golden=lambda t: t["a"])


class TestGoldenAdaptation:
    """The golden callable reaches the pipeline's in-place contract."""

    def _tensors(self) -> dict[str, torch.Tensor]:
        return {"a": torch.randn(M, N), "out": torch.zeros(M, N)}

    def test_returned_tensor_is_written_to_the_single_output(self):
        case_obj = _jit_case()
        tensors = self._tensors()
        case_obj.compute_expected(tensors)
        assert torch.equal(tensors["out"], torch.abs(tensors["a"]))

    def test_closure_over_test_locals_is_allowed(self):
        """The golden runs in this process, so it may capture anything."""
        scale = 3.0
        case_obj = _jit_case(name="abs_closure", golden=lambda t: torch.abs(t["a"]) * scale)
        tensors = self._tensors()
        case_obj.compute_expected(tensors)
        assert torch.equal(tensors["out"], torch.abs(tensors["a"]) * scale)

    def test_in_place_golden_is_left_alone(self):
        def golden(t):
            t["out"][:] = torch.abs(t["a"])

        tensors = self._tensors()
        _jit_case(name="abs_inplace", golden=golden).compute_expected(tensors)
        assert torch.equal(tensors["out"], torch.abs(tensors["a"]))

    def test_dict_return_must_name_the_outputs(self):
        tensors = self._tensors()
        _jit_case(name="abs_dict", golden=lambda t: {"out": torch.abs(t["a"])}).compute_expected(tensors)
        assert torch.equal(tensors["out"], torch.abs(tensors["a"]))

        with pytest.raises(ValueError, match="unknown output"):
            bad = _jit_case(name="abs_bad_key", golden=lambda t: {"wrong": t["a"]})
            bad.compute_expected(self._tensors())

    def test_wrong_return_type_is_reported(self):
        with pytest.raises(TypeError, match="expected a torch.Tensor"):
            _jit_case(name="abs_bad_type", golden=lambda t: "nope").compute_expected(self._tensors())

    def test_missing_golden_is_reported(self):
        with pytest.raises(ValueError, match="has no golden"):
            st.case(
                orchestrator, torch.randn(M, N), torch.zeros(M, N), name="abs_no_golden"
            ).compute_expected(self._tensors())


class TestDeclaration:
    """``st.cases`` and the legacy adapter."""

    def test_duplicate_names_are_rejected(self):
        with pytest.raises(ValueError, match="duplicate case name"):
            st.cases(_jit_case(name="dup"), _jit_case(name="dup"))

    def test_empty_declaration_is_rejected(self):
        with pytest.raises(ValueError, match="at least one case"):
            st.cases()

    def test_from_legacy_preserves_the_case(self):
        """A wrapped ``PTOTestCase`` keeps its program, tensors and golden."""
        legacy = AbsLegacyCase()
        wrapped = from_legacy(legacy)
        assert wrapped.get_name() == legacy.get_name()
        assert wrapped.get_program() is AbsProgram
        assert [s.name for s in wrapped.tensor_specs] == [s.name for s in legacy.tensor_specs]

        tensors = {"a": torch.randn(M, N), "out": torch.zeros(M, N)}
        wrapped.compute_expected(tensors)
        assert torch.equal(tensors["out"], torch.abs(tensors["a"]))

    def test_from_legacy_preserves_a_run_config_memory_planner(self):
        """A planner carried only by the legacy ``RunConfig`` survives the wrap.

        ``_resolve_case_memory_planner`` reads ``get_memory_planner()`` first
        and the case's own ``RunConfig`` second. A ``Case`` rebuilds that
        ``RunConfig`` from the tolerances alone, so without folding the second
        channel into the first the wrapped case would silently fall through to
        the session planner.
        """
        legacy = AbsLegacyCase(RunConfig(memory_planner=MemoryPlanner.DSA_RP))
        assert legacy.get_memory_planner() is None, "the planner rides on the config only"
        assert from_legacy(legacy).get_memory_planner() == MemoryPlanner.DSA_RP

    def test_from_legacy_prefers_the_explicit_planner(self):
        """``get_memory_planner()`` still outranks the config's planner."""
        legacy = AbsLegacyCase(
            RunConfig(memory_planner=MemoryPlanner.DSA_RP), memory_planner=MemoryPlanner.PYPTO
        )
        assert from_legacy(legacy).get_memory_planner() == MemoryPlanner.PYPTO

    def test_platform_binding_respects_a_pin(self):
        pinned = _jit_case(name="abs_pinned", platform="a5")
        pinned.bind_platform("a2a3")
        assert pinned.get_platform() == "a5", "an explicit pin wins over the item's platform"

        floating = _jit_case(name="abs_floating")
        floating.bind_platform("a2a3")
        assert floating.get_platform() == "a2a3"

    def test_unknown_platform_is_rejected(self):
        with pytest.raises(ValueError, match="unknown platform"):
            _jit_case(name="abs_bad_platform", platform="a9")

    def test_for_platform_binds_each_variant_independently(self):
        """One declared case serves N platform variants without cross-talk.

        ``st.cases`` hands the *same* object to every variant of a
        multi-platform item, so binding in place would pin the first variant's
        platform for all of them — and a case's own platform outranks the
        item's, so every variant would go on to compile and run that first
        platform's artifact.
        """
        declared = _jit_case(name="abs_matrix")
        variants = {p: declared.for_platform(p) for p in ("a2a3", "a2a3sim", "a5", "a5sim")}

        assert declared.get_platform() is None, "the declaration itself stays unbound"
        for platform, bound in variants.items():
            assert bound is not declared
            assert bound.get_platform() == platform
        assert len({id(v) for v in variants.values()}) == 4, "one case object per variant"

    def test_for_platform_keeps_a_pin_and_its_identity(self):
        """A pinned case is returned as-is: the pin must keep outranking the item."""
        pinned = _jit_case(name="abs_pinned_matrix", platform="a5")
        assert pinned.for_platform("a2a3") is pinned
        assert pinned.get_platform() == "a5"

    def test_cache_id_separates_positional_scalars(self):
        """A scalar passed positionally must change the identity.

        It reaches the specializer exactly as a keyword scalar does, and the
        id is the default case name -- which is the artifact cache key, so a
        collision makes the second declaration reuse the first one's artifact.
        """
        a, out = torch.randn(M, N), torch.zeros(M, N)
        one = JitKernel(scaled_entry, a, out, 1.0)
        two = JitKernel(scaled_entry, a, out, 2.0)
        assert one.cache_id() != two.cache_id(), f"positional scalar dropped from the id: {one.cache_id()}"

    def test_cache_id_separates_scalar_types(self):
        """``1`` and ``1.0`` specialize differently and must not collide."""
        a, out = torch.randn(M, N), torch.zeros(M, N)
        assert JitKernel(scaled_entry, a, out, 1).cache_id() != (
            JitKernel(scaled_entry, a, out, 1.0).cache_id()
        )

    def test_cache_id_agrees_across_binding_styles(self):
        """The same scalar gives the same id positionally or by keyword."""
        a, out = torch.randn(M, N), torch.zeros(M, N)
        assert (
            JitKernel(scaled_entry, a, out, 2.0).cache_id()
            == JitKernel(scaled_entry, a, out, scale=2.0).cache_id()
        )

    def test_for_platform_carries_the_declaration_over(self):
        """The copy differs in platform alone — every other field is the case's."""
        declared = _jit_case(name="abs_carry", rtol=1e-3, atol=1e-2)
        bound = declared.for_platform("a5")
        assert bound.name == declared.name
        assert bound.kernel is declared.kernel
        assert bound.golden is declared.golden
        assert bound.tensor_specs == declared.tensor_specs
        assert (bound.config.rtol, bound.config.atol) == (1e-3, 1e-2)


class TestPlatformMatrixCollection:
    """A declared case against the multi-platform CLI, through the real hooks.

    Both halves are driven here because they fail independently: the collection
    hook decides what gets compiled and under which key, and the deselect hook
    decides which items exist at all. A pin that is honoured by one and ignored
    by the other still reports coverage it does not have.
    """

    PLATFORMS = ("a2a3", "a2a3sim", "a5", "a5sim")

    @staticmethod
    def _conftest() -> Any:
        """Load ``tests/st/conftest.py`` as a module; it is not importable by name."""
        import importlib.util  # noqa: PLC0415

        path = Path(__file__).resolve().parents[1] / "conftest.py"
        spec = importlib.util.spec_from_file_location("_st_conftest_under_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @classmethod
    def _items(cls, case_obj: Case) -> "list[Any]":
        """One stub item per matrix variant, shaped like the pytest items."""

        class _CallSpec:
            def __init__(self, params: dict) -> None:
                self.params = params

        class _Item:
            module = None

            def __init__(self, params: dict, name: str) -> None:
                self.callspec = _CallSpec(params)
                self.name = name

            def iter_markers(self, name: str | None = None) -> Any:
                return iter(())

        return [_Item({"_st_case": case_obj, "_st_platform": p}, f"test_it[{p}]") for p in cls.PLATFORMS]

    @staticmethod
    def _config(platform_opt: str) -> Any:
        class _Hook:
            def __init__(self) -> None:
                self.deselected: list[Any] = []

            def pytest_deselected(self, items: "list[Any]") -> None:
                self.deselected.extend(items)

        class _Config:
            def __init__(self) -> None:
                self.hook = _Hook()

            def getoption(self, name: str) -> Any:
                assert name == "--platform", name
                return platform_opt

        return _Config()

    def test_a_pinned_case_keeps_only_its_own_matrix_variant(self):
        """The other three variants would run A5 while claiming to be themselves."""
        conf = self._conftest()
        items = self._items(_jit_case(name="abs_pin_deselect", platform="a5"))
        config = self._config(",".join(self.PLATFORMS))

        kept = list(items)
        conf.pytest_collection_modifyitems(config, kept)

        assert [i.name for i in kept] == ["test_it[a5]"]
        assert [i.name for i in config.hook.deselected] == [
            "test_it[a2a3]",
            "test_it[a2a3sim]",
            "test_it[a5sim]",
        ]

    @classmethod
    def _unexpanded_item(cls, case_obj: Case) -> "list[Any]":
        """The single-platform shape: no matrix, so no platform param at all."""
        items = cls._items(case_obj)[:1]
        del items[0].callspec.params["_st_platform"]
        items[0].name = "test_it"
        return items

    def test_a_pinned_case_is_deselected_when_the_only_platform_is_not_its_own(self):
        """The matrix expands only for a multi-platform CLI.

        A plain ``--platform=a2a3`` run -- the shape CI uses -- grows no
        variants, so the pin had no platform param to disagree with and every
        pinned case survived. An A5-pinned case would then have been handed an
        A2A3 card.
        """
        conf = self._conftest()
        items = self._unexpanded_item(_jit_case(name="abs_pin_single", platform="a5"))
        config = self._config("a2a3")

        kept = list(items)
        conf.pytest_collection_modifyitems(config, kept)

        assert kept == []
        assert [i.name for i in config.hook.deselected] == ["test_it"]

    def test_a_pinned_case_survives_a_single_platform_run_naming_its_pin(self):
        conf = self._conftest()
        items = self._unexpanded_item(_jit_case(name="abs_pin_single_kept", platform="a5"))
        config = self._config("a5")

        kept = list(items)
        conf.pytest_collection_modifyitems(config, kept)

        assert [i.name for i in kept] == ["test_it"]
        assert config.hook.deselected == []

    def test_a_pinned_case_is_collected_once_under_its_pin(self):
        """Keyed by the pin, not by the item — even with no deselect in front.

        Keying by the item's platform filed one object under a key per variant.
        The pipeline resolves each of them back to the pin, so it compiled the
        same artifact directory once per variant, concurrently.
        """
        conf = self._conftest()
        pinned = _jit_case(name="abs_pin_key", platform="a5")

        seen: dict[str, Any] = {}
        for item in self._items(pinned):
            conf._collect_test_case_from_item(item, seen, None, "a2a3")

        assert list(seen) == ["abs_pin_key@a5@default"]
        assert all(c is pinned for c in seen.values()), "a pinned case is never copied"

    def test_an_unpinned_case_still_spans_the_matrix(self):
        """The pin path must not narrow a case that never asked for one."""
        conf = self._conftest()
        declared = _jit_case(name="abs_free_key")

        seen: dict[str, Any] = {}
        items = self._items(declared)
        for item in items:
            conf._collect_test_case_from_item(item, seen, None, "a2a3")

        assert sorted(seen) == [f"abs_free_key@{p}@default" for p in sorted(self.PLATFORMS)]
        assert len({id(c) for c in seen.values()}) == 4
        assert declared.get_platform() is None, "the declaration itself stays unbound"

        kept = list(items)
        conf.pytest_collection_modifyitems(self._config(",".join(self.PLATFORMS)), kept)
        assert len(kept) == 4, "an unpinned case keeps every variant"


class TestCustomCompare:
    """``compare=`` replaces the elementwise check the harness performs."""

    def test_non_callable_is_rejected(self):
        with pytest.raises(TypeError, match="compare must be callable"):
            _jit_case(name="abs_bad_compare", compare="nope")

    def test_only_a_case_carries_a_comparator(self):
        """Matched by type, so nothing else can look like it carries one.

        ``PTOTestCase`` has no such attribute, and a ``Mock`` answers truthily
        to *any* attribute — an attribute check would send it down the
        persist-then-compare path with no artifacts to read, which is how this
        was found.
        """
        assert _case_comparator(AbsLegacyCase()) is None
        assert _case_comparator(Mock()) is None, "a Mock must not look like it has a comparator"
        assert _case_comparator(_jit_case(name="abs_no_compare")) is None
        assert _case_comparator(_jit_case(name="abs_has_compare", compare=lambda a, e: None)) is not None

    @pytest.mark.skipif(not _ptoas_available(), reason="ptoas is required to generate kernel sources")
    def test_comparator_sees_the_persisted_outputs(self):
        """It receives ``data/actual`` and ``data/out``, keyed by output name.

        Compiled for real so the golden and ``golden.py``'s ``__outputs__`` are
        the genuine artefacts; only the device run is stood in for, by writing
        the actuals the run would have persisted.
        """
        case_obj = _jit_case(platform="a2a3")
        work_dir = Path(tempfile.mkdtemp(prefix="case_compare_"))
        try:
            _compile_for_cache(case_obj, work_dir, "a2a3", False, False, None)
            expected = torch.load(work_dir / "data" / "out" / "out.pt")

            # Stand in for the device run: persist an output that is close but
            # not elementwise-equal, so a comparator that looks at the whole
            # tensor can accept what the default check would reject.
            actual = expected + 1e-2
            (work_dir / "data" / "actual").mkdir(parents=True, exist_ok=True)
            torch.save(actual, work_dir / "data" / "actual" / "out.pt")

            seen: dict[str, Any] = {}

            def record(actual_map, expected_map):
                seen["names"] = sorted(actual_map)
                seen["match"] = torch.equal(actual_map["out"], actual) and torch.equal(
                    expected_map["out"], expected
                )

            _compare_persisted_outputs(work_dir, record)
            assert seen["names"] == ["out"], "keyed by the output names golden.py declares"
            assert seen["match"], "the comparator receives the persisted tensors verbatim"

            # The default check rejects this pair; a whole-tensor comparator accepts it.
            with pytest.raises(AssertionError):
                validate_persisted_outputs(work_dir, rtol=1e-5, atol=1e-5)

            def rel_err_under(limit):
                def compare(actual_map, expected_map):
                    for name, got in actual_map.items():
                        ref = expected_map[name].float()
                        rel = (got.float() - ref).norm() / ref.norm()
                        assert rel < limit, f"{name}: rel_err {rel:.3e} exceeds {limit}"

                return compare

            _compare_persisted_outputs(work_dir, rel_err_under(2e-2))
            with pytest.raises(AssertionError, match="rel_err"):
                _compare_persisted_outputs(work_dir, rel_err_under(1e-9))
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def test_missing_persisted_outputs_are_reported(self):
        empty = Path(tempfile.mkdtemp(prefix="case_compare_empty_"))
        try:
            with pytest.raises((AssertionError, FileNotFoundError)):
                _compare_persisted_outputs(empty, lambda a, e: None)
        finally:
            shutil.rmtree(empty, ignore_errors=True)


class _TwoNameCase(AbsLegacyCase):
    """A legacy case whose name is an argument, so one body can build two."""

    __test__ = False

    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name

    def get_name(self) -> str:
        return self._name


def _body_that_runs_two_distinct_cases(test_runner):
    """Stand-in for a test body that runs more than one case."""
    test_runner.run(_TwoNameCase("first_of_two"))
    test_runner.run(_TwoNameCase("second_of_two"))


def _body_that_skips_on_a_missing_optional_dep():
    """Stand-in for a unit-test body guarding an optional dependency.

    ``pytest.importorskip`` raises ``Skipped``, which derives from
    ``BaseException`` — the exact shape that used to escape discovery.
    """
    msgpack = pytest.importorskip("_pypto_no_such_optional_module")
    return msgpack


class TestCollectionIsNeverAbortedByADiscoveredCall:
    """Discovery runs code out of test bodies; it must never take the session down.

    ``_eval_arg_node`` resolves a constructor argument by *invoking* the callee,
    so a body containing ``pytest.importorskip("...")`` for an absent module
    raised ``Skipped`` inside ``pytest_collection_finish``. ``Skipped`` is a
    ``BaseException``, so both call sites' ``except Exception`` guards missed it,
    and a skip escaping a collection hook is not a skip — under xdist it is a
    session-wide INTERNALERROR that reports zero tests.
    """

    @staticmethod
    def _conftest() -> Any:
        return TestPlatformMatrixCollection._conftest()

    @staticmethod
    def _item(func: Any) -> Any:
        """A stub item shaped like the attributes discovery actually reads."""

        class _Item:
            module = sys.modules[__name__]
            callspec = None

            def __init__(self) -> None:
                self.function = func
                self.path = Path(__file__)

            def iter_markers(self, name: str | None = None) -> Any:
                return iter(())

        return _Item()

    def test_a_call_that_raises_skipped_is_merely_unresolvable(self):
        """The BaseException is converted at the one place discovery invokes code."""
        conf = self._conftest()
        node = ast.parse('pytest.importorskip("_pypto_no_such_optional_module")').body[0].value

        with pytest.raises(conf._Unresolvable):
            conf._eval_arg_node(node, {}, {}, {"pytest": pytest})

    def test_such_a_body_leaves_collection_intact(self):
        """End to end: the body is walked, nothing is discovered, nothing raises."""
        conf = self._conftest()
        seen: dict[str, Any] = {}

        item = self._item(_body_that_skips_on_a_missing_optional_dep)
        conf._collect_test_case_from_item(item, seen, None, "a2a3")

        assert seen == {}, "no PTOTestCase in that body — and no crash reaching that conclusion"

    def test_every_constructor_in_a_body_is_filed(self):
        """Not just the first. A body running two cases needs a future for both.

        Returning after the first left the second to the serial inline path
        *and* reported ``True``, so the undiscovered inventory never named it --
        the miss was invisible to the very guard meant to catch it.
        """
        conf = self._conftest()
        seen: dict[str, Any] = {}

        found = conf._collect_test_case_from_item(
            self._item(_body_that_runs_two_distinct_cases), seen, None, "a2a3"
        )

        assert found is True
        assert sorted(c.get_name() for c in seen.values()) == ["first_of_two", "second_of_two"]

    def test_only_items_under_tests_st_are_walked(self):
        """A session hook fires for every item; only ST bodies are ours to parse."""
        conf = self._conftest()
        tests_dir = Path(__file__).resolve().parents[2]

        class _PathItem:
            def __init__(self, path: Path) -> None:
                self.path = path

        ut_case = tests_dir / "ut" / "ir" / "expressions" / "test_call_arg_directions.py"
        assert conf._is_st_item(_PathItem(Path(__file__)))
        assert not conf._is_st_item(_PathItem(ut_case))


class TestInlineCaseGuard:
    """Collection fails when a test reaches the runner with no declared case.

    tests/st is at zero inline-compiled cases. Keeping it there needs a hard
    failure rather than a summary line: the advisory-only version of this report
    sat unread while the count reached 76.
    """

    @staticmethod
    def _conftest() -> Any:
        return TestPlatformMatrixCollection._conftest()

    @staticmethod
    def _item(node_id: str, markers: "list[Any]", swimlane_level: int = 0) -> Any:
        opts = {
            "--chip-swimlane-level": swimlane_level,
            "--enable-chip-swimlane": swimlane_level,
            "--codegen-only": False,
            "--device": "0",
        }

        class _Item:
            nodeid = node_id
            name = node_id.rsplit("::", 1)[-1]
            module = None
            fixturenames = ("test_runner",)
            config = SimpleNamespace(getoption=lambda n, default=None: opts.get(n, default))

            def iter_markers(self, name: str | None = None) -> Any:
                return iter([m for m in markers if name is None or m.name == name])

            def get_closest_marker(self, name: str) -> Any:
                return next((m for m in markers if m.name == name), None)

        return _Item()

    @staticmethod
    def _marker(name: str, *args: Any, **kwargs: Any) -> Any:
        class _Marker:
            pass

        m = _Marker()
        m.name, m.args, m.kwargs = name, args, kwargs
        return m

    def test_a_fixture_shaped_skip_is_stated_as_a_marker(self):
        """`without_swimlane` is the inverse of `swimlane`, and readable at collection.

        As an autouse fixture this condition was invisible to `_will_not_run`,
        so the class's cases were registered -- and the device-pool submitter
        puts every registered case on a card before the item loop starts. The
        case was compiled and run for a test that then skipped.
        """
        conf = self._conftest()
        marker = self._marker("without_swimlane", reason="runs without the record")

        on = self._item("t.py::test_x", [marker], swimlane_level=4)
        off = self._item("t.py::test_x", [marker], swimlane_level=0)

        assert conf.marker_skip_reason(on) == "runs without the record"
        assert conf.marker_skip_reason(off) is None

    def test_a_reasonless_exclusion_is_refused(self):
        conf = self._conftest()
        item = self._item("t.py::test_x", [self._marker("without_swimlane")], swimlane_level=4)
        with pytest.raises(pytest.UsageError, match="needs a reason"):
            conf.marker_skip_reason(item)

    def test_skipif_counts_as_will_not_run(self):
        """`iter_markers` names it `skipif`; matching only `skip` missed it.

        Registering a case behind either marker now costs a device run, not just
        a compile, because the submitter runs what the pool holds.
        """
        conf = self._conftest()
        for name in ("skip", "skipif"):
            item = self._item(f"t.py::test_{name}", [self._marker(name)], swimlane_level=0)
            assert conf._will_not_run(item), name
        plain = self._item("t.py::test_plain", [], swimlane_level=0)
        assert not conf._will_not_run(plain)

    def test_an_unmarked_reason_is_refused(self):
        """A bare `inline_case` reads exactly like a test nobody declared."""
        conf = self._conftest()
        item = self._item("t.py::test_bare", [self._marker("inline_case")])
        with pytest.raises(pytest.UsageError, match="needs a reason"):
            conf._inline_case_reason(item)

    def test_a_reasoned_marker_exempts_the_test(self):
        conf = self._conftest()
        item = self._item("t.py::test_ok", [self._marker("inline_case", reason="probes the card first")])
        assert conf._inline_case_reason(item) == "probes the card first"

    def test_an_unmarked_test_is_not_exempt(self):
        conf = self._conftest()
        assert conf._inline_case_reason(self._item("t.py::test_plain", [])) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
