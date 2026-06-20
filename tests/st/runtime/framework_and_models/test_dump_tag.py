# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""System tests for selective tensor dump end-to-end (simpler#844).

Selective dump has three front-ends, all backed by the same per-call
``dump_vars`` attr → ``Arg::dump(...)`` runtime path (the runtime latches the
dump level host-side; no orch-body toggle since simpler#953). This file
exercises two of them on-board:

1. ``pl.dump_tag`` via a ``@pl.jit`` entry — ``TestDumpTag*`` below.
   A tiny ``(a + 1) * 2`` kernel built as a ``@pl.jit`` entry composed of two
   ``@pl.jit.inline`` helpers. ``pl.dump_tag`` markers live in both scopes:

     - Inline-scope: ``pl.dump_tag(x)`` inside ``add_inline``. It desugars to
       ``dump_vars`` on the inline body's call; after ``InlineFunctions``
       splices the body in, the inline param ``x`` is substituted with the
       entry's ``a``, so the dump rides through to the inlined call.
     - Entry-scope: ``pl.dump_tag(intermediate)`` on the body-local
       ``pl.create_tensor`` result.

   The entry output ``c`` is intentionally never tagged.

2. ``pl.submit(..., dumps=[...])`` via the ``PTOTestCase`` harness —
   ``TestSubmitDumps*`` below. ``dumps=`` is the submit-side selective-dump
   surface (symmetric with ``deps=``); ``pl.dump_tag`` / ``pl.dump`` cannot be
   used here because ``@pl.jit`` has no ``pl.submit`` / ``pl.manual_scope``.
   A 2-stage submit pipeline tags only stage1, so stage2 must be filtered out.

In both scenarios, with ``--dump-tensor`` the runtime's selective-dump filter
retains only the tagged bindings in the manifest, so each test asserts both the
positive (tagged present) and negative (untagged filtered) paths in one pass.

Correctness always runs. Manifest validation (parsing the JSON, and for the
dump_tag case decoding sample bytes via ``simpler_setup.tools.dump_viewer``) is
gated behind ``--dump-tensor`` and ``not codegen_only``.
"""

import dataclasses
import json
import shutil
from pathlib import Path
from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import PLATFORMS, DataType, PTOTestCase, TensorSpec
from pypto.ir.pass_manager import OptimizationStrategy

_BUILD_OUTPUT_DIR = Path(__file__).resolve().parents[4] / "build_output"
_DUMP_TAG_WORK_DIR = _BUILD_OUTPUT_DIR / "dump_tag_test"

_REQUIRED_FIELDS: dict[str, type | tuple[type, ...]] = {
    "task_id": str,
    "role": str,
    "stage": str,
    "arg_index": int,
    "dtype": str,
    "shape": list,
    "strides": list,
    "start_offset": int,
    "bin_offset": int,
    "bin_size": int,
    "is_contiguous": bool,
}


@pl.jit.inline
def add_inline(a: pl.Tensor, c: pl.Tensor):
    """c = a + 1.0. Inline-scope dump_tag — desugars to ``dump_vars`` on the
    inline body's kernel call; after inlining, the mutator substitutes the
    caller's arg for the inline param ``a``, so the dump rides through to the
    inlined call site (tracked by Var identity).
    """
    pl.dump_tag(a)
    with pl.at(level=pl.Level.CORE_GROUP):
        tile_a = pl.load(a, [0, 0], [128, 128])
        tile_c = pl.add(tile_a, 1.0)
        pl.store(tile_c, [0, 0], c)
    return c


@pl.jit.inline
def mul_inline(a: pl.Tensor, c: pl.Tensor):
    """c = a * 2.0. No dump_tag here — its bindings should be filtered out."""
    with pl.at(level=pl.Level.CORE_GROUP):
        tile_a = pl.load(a, [0, 0], [128, 128])
        tile_c = pl.mul(tile_a, 2.0)
        pl.store(tile_c, [0, 0], c)
    return c


@pl.jit
def add_mul_with_dump_tags(a: pl.Tensor, c: pl.Out[pl.Tensor]):
    """Entry: c = (a + 1) * 2 with mixed-scope dump_tag markers."""
    intermediate = pl.create_tensor([128, 128], dtype=pl.FP32)
    pl.dump_tag(intermediate)
    intermediate = add_inline(a, intermediate)
    c = mul_inline(intermediate, c)
    return c


def _make_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    a = torch.randn(128, 128, dtype=torch.float32)
    c = torch.zeros(128, 128, dtype=torch.float32)
    expected = (a + 1.0) * 2.0
    return a, c, expected


@pytest.fixture(scope="session")
def dump_tag_run(test_config):
    """Run the dump_tag kernel once and return ``(work_dir, c, expected)``.

    ``work_dir`` is pinned to ``build_output/dump_tag_test/`` so the
    generated artefacts (compiled kernels, ``dfx_outputs/tensor_dump/``)
    survive across pytest sessions and can be inspected directly with
    ``python -m simpler_setup.tools.dump_viewer
    build_output/dump_tag_test/dfx_outputs/tensor_dump``.

    The directory is wiped at the start of each session so stale entries
    from a previous run can't be confused with the current one. Forced
    via ``dataclasses.replace`` on ``test_config.save_kernels_dir`` so
    manifest validation does not have to glob the timestamped default.

    Session-scoped so the (expensive) compile happens once regardless of
    how many manifest assertions follow.
    """
    add_mul_with_dump_tags._cache.clear()

    shutil.rmtree(_DUMP_TAG_WORK_DIR, ignore_errors=True)
    _DUMP_TAG_WORK_DIR.mkdir(parents=True, exist_ok=True)
    config = dataclasses.replace(test_config, save_kernels_dir=str(_DUMP_TAG_WORK_DIR))

    a, c, expected = _make_inputs()
    add_mul_with_dump_tags(a, c, config=config)
    return _DUMP_TAG_WORK_DIR, c, expected


@pytest.fixture(scope="session")
def dump_manifest(dump_tag_run, test_config) -> tuple[list[dict], Path, Path]:
    """Load ``tensor_dump.json`` and resolve the companion bin path.

    Returns ``(entries, manifest_path, bin_path)``. Mirrors how
    ``simpler_setup.tools.dump_viewer`` parses the manifest:
    top-level is a dict with ``tensors`` (entry list) and ``bin_file``
    (bin filename relative to the manifest directory).

    Skips when ``--dump-tensor`` is not set or when ``--codegen-only`` is
    set (no device execution means no manifest is written).
    """
    if not test_config.enable_dump_tensor:
        pytest.skip("pass --dump-tensor to exercise the dump pipeline")
    if test_config.codegen_only:
        pytest.skip("--codegen-only skips device execution; no manifest is written")

    work_dir, _, _ = dump_tag_run
    manifest_path = work_dir / "dfx_outputs" / "tensor_dump" / "tensor_dump.json"
    assert manifest_path.exists(), f"tensor_dump.json not found at {manifest_path}"

    manifest = json.loads(manifest_path.read_text())
    assert isinstance(manifest, dict), f"tensor_dump.json should hold a dict, got {type(manifest).__name__}"
    entries = manifest.get("tensors")
    assert isinstance(entries, list), (
        f"tensor_dump.json['tensors'] should be a list, got {type(entries).__name__}"
    )
    assert entries, "tensor_dump.json['tensors'] is empty — dump pipeline produced no entries"

    bin_name = manifest.get("bin_file")
    assert isinstance(bin_name, str) and bin_name, (
        f"tensor_dump.json missing 'bin_file' key (or empty): manifest keys = {sorted(manifest)}"
    )
    bin_path = manifest_path.parent / bin_name
    return entries, manifest_path, bin_path


class TestDumpTagCorrectness:
    """Correctness check — always runs regardless of ``--dump-tensor``."""

    def test_add_mul_matches_torch_reference(self, dump_tag_run):
        _, c, expected = dump_tag_run
        assert torch.allclose(c, expected, rtol=1e-5, atol=1e-5), (
            f"add_mul_with_dump_tags numerical mismatch: max diff = {(c - expected).abs().max().item()}"
        )


class TestDumpTagManifest:
    """Manifest validation — only runs when ``--dump-tensor`` is enabled."""

    def test_entries_have_required_fields(self, dump_manifest):
        entries, manifest_path, _ = dump_manifest
        for i, entry in enumerate(entries):
            for field, expected_type in _REQUIRED_FIELDS.items():
                assert field in entry, f"{manifest_path}: entry[{i}] missing required field {field!r}"
                assert isinstance(entry[field], expected_type), (
                    f"{manifest_path}: entry[{i}].{field} has type "
                    f"{type(entry[field]).__name__}, expected {expected_type}"
                )
            assert entry["role"] in {"input", "output", "inout"}, (
                f"unexpected role {entry['role']!r} in entry[{i}]"
            )
            assert entry["stage"] in {"before_dispatch", "after_completion"}, (
                f"unexpected stage {entry['stage']!r} in entry[{i}]"
            )

    def test_bin_offsets_fit_within_bin_file(self, dump_manifest):
        entries, manifest_path, bin_path = dump_manifest
        assert bin_path.exists(), f"bin file {bin_path} not found alongside {manifest_path}"
        bin_size = bin_path.stat().st_size
        for i, entry in enumerate(entries):
            if entry.get("overwritten") or entry.get("truncated"):
                continue
            end = entry["bin_offset"] + entry["bin_size"]
            assert end <= bin_size, (
                f"entry[{i}] (task_id={entry['task_id']}, role={entry['role']}, "
                f"stage={entry['stage']}, arg={entry['arg_index']}) references "
                f"bytes [{entry['bin_offset']}, {end}) but {bin_path.name} is only {bin_size} bytes"
            )

    def test_simpler_dump_viewer_can_decode_a_sample(self, dump_manifest):
        """The simpler-provided ``dump_viewer`` parses the manifest + binary."""
        from simpler_setup.tools.dump_viewer import decode_elements, read_tensor_data  # noqa: PLC0415

        entries, _, bin_path = dump_manifest

        sample = next(
            (e for e in entries if e["bin_size"] > 0 and not e.get("overwritten") and not e.get("truncated")),
            None,
        )
        assert sample is not None, "no decodable entry in tensor_dump.json (all overwritten/truncated/empty)"

        data = read_tensor_data(bin_path, sample["bin_offset"], sample["bin_size"])
        assert len(data) == sample["bin_size"], (
            f"read_tensor_data returned {len(data)} bytes, expected {sample['bin_size']}"
        )

        numel = 1
        for d in sample["shape"]:
            numel *= d
        # Decode at most 16 elements — enough to prove parseability without
        # quadratic cost on large tensors.
        elements = decode_elements(data, sample["dtype"], min(numel, 16))
        assert len(elements) == min(numel, 16), (
            f"decode_elements returned {len(elements)} elements, expected {min(numel, 16)}"
        )

    def test_only_tagged_kernel_dumps(self, dump_manifest):
        """Selective dump must drop kernel2 entirely.

        The tagged values (``a`` and the ``intermediate`` produced by kernel1)
        ride on each consuming call's ``dump_vars`` by Var identity, so kernel1
        (``add_inline``) dumps both. Kernel2 (``mul_inline``) consumes the
        value rebound after kernel1 (a distinct Var from the tagged
        ``intermediate``) and writes ``c`` — neither is tagged, so codegen
        emits no ``params_t1.dump`` call. The manifest must therefore contain
        entries from a single task (one ``task_id``) only.

        The runtime manifest identifies each kernel dispatch by ``task_id``
        (``(ring_id << 32) | local_id``); ``func_id`` / ``subtask_id`` are
        runtime-internal and not surfaced per entry. One ``task_id`` per
        dispatch makes it the right discriminator for "single kernel".
        """
        entries, manifest_path, _ = dump_manifest
        task_ids = {e["task_id"] for e in entries}
        assert len(task_ids) == 1, (
            f"{manifest_path}: selective dump should retain entries from a single kernel, "
            f"found {len(task_ids)} task_ids={sorted(task_ids)}"
        )
        roles = {e["role"] for e in entries}
        assert "input" in roles, f"{manifest_path}: missing role=input entries; have {sorted(roles)}"
        assert "inout" in roles, f"{manifest_path}: missing role=inout entries; have {sorted(roles)}"


# ===========================================================================
# pl.submit(..., dumps=[...]) — submit-side selective dump (PTOTestCase harness)
# ===========================================================================
#
# ``@pl.jit`` (above) cannot express ``pl.submit`` / ``pl.manual_scope``, so the
# submit ``dumps=`` surface is exercised through the PTOTestCase / test_runner
# harness instead. Same runtime dump pipeline, different DSL front-end.

_SUBMIT_DUMPS_ROWS = 128
_SUBMIT_DUMPS_COLS = 128


def _build_submit_dumps_program():
    """Build a 2-stage submit pipeline with ``dumps=`` on stage1 only."""
    ROWS, COLS = _SUBMIT_DUMPS_ROWS, _SUBMIT_DUMPS_COLS

    @pl.program
    class SubmitDumpsProgram:
        """``out = (x + 1) * 2`` via two submitted stages; stage1 dumps x + scratch."""

        @pl.function(type=pl.FunctionType.InCore)
        def stage1(
            self,
            x: pl.Tensor[[ROWS, COLS], pl.FP32],
            scratch: pl.InOut[pl.Tensor[[ROWS, COLS], pl.FP32]],
        ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
            t: pl.Tile[[ROWS, COLS], pl.FP32] = pl.load(x, [0, 0], [ROWS, COLS])
            r: pl.Tile[[ROWS, COLS], pl.FP32] = pl.add(t, 1.0)  # x + 1
            # Read scratch (init 0.0) so it is a genuine InOut slot: adding the
            # zero-initialised buffer leaves the result == x + 1, and the read
            # makes codegen register scratch via add_inout (dump role "inout").
            s: pl.Tile[[ROWS, COLS], pl.FP32] = pl.load(scratch, [0, 0], [ROWS, COLS])
            acc: pl.Tile[[ROWS, COLS], pl.FP32] = pl.add(r, s)  # (x + 1) + 0
            ret: pl.Tensor[[ROWS, COLS], pl.FP32] = pl.store(acc, [0, 0], scratch)
            return ret

        @pl.function(type=pl.FunctionType.InCore)
        def stage2(
            self,
            scratch: pl.Tensor[[ROWS, COLS], pl.FP32],
            out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
        ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
            t: pl.Tile[[ROWS, COLS], pl.FP32] = pl.load(scratch, [0, 0], [ROWS, COLS])
            r: pl.Tile[[ROWS, COLS], pl.FP32] = pl.mul(t, 2.0)  # scratch * 2
            ret: pl.Tensor[[ROWS, COLS], pl.FP32] = pl.store(r, [0, 0], out)
            return ret

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            x: pl.Tensor[[ROWS, COLS], pl.FP32],
            scratch: pl.InOut[pl.Tensor[[ROWS, COLS], pl.FP32]],
            out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
        ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
            with pl.manual_scope():
                # stage1 dumps its input x and inout scratch; stage2 dumps
                # nothing, so selective dump must filter stage2 out entirely.
                scratch, stage1_tid = pl.submit(self.stage1, x, scratch, dumps=[x, scratch])
                out, _ = pl.submit(self.stage2, scratch, out, deps=[stage1_tid])
            return out

    return SubmitDumpsProgram


class _SubmitDumpsPipelinePTO(PTOTestCase):
    """``out = (x + 1) * 2`` via a 2-stage submit pipeline with stage1 dumps=."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return f"submit_dumps_pipeline_{_SUBMIT_DUMPS_ROWS}x{_SUBMIT_DUMPS_COLS}"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [_SUBMIT_DUMPS_ROWS, _SUBMIT_DUMPS_COLS], DataType.FP32, init_value=torch.randn),
            TensorSpec("scratch", [_SUBMIT_DUMPS_ROWS, _SUBMIT_DUMPS_COLS], DataType.FP32, init_value=0.0),
            TensorSpec(
                "out", [_SUBMIT_DUMPS_ROWS, _SUBMIT_DUMPS_COLS], DataType.FP32, init_value=0.0, is_output=True
            ),
        ]

    def get_program(self) -> Any:
        return _build_submit_dumps_program()

    def compute_expected(self, tensors, params=None):
        # out = (x + 1) * 2 element-wise.
        tensors["out"][:] = (tensors["x"] + 1.0) * 2.0


class TestSubmitDumpsCorrectness:
    """Numerical correctness — runs on every supported platform.

    Guards that ``dumps=`` is inert to the computed result: marking args for
    selective dump must not perturb the kernel output.
    """

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_pipeline_correctness(self, test_runner, platform):
        result = test_runner.run(_SubmitDumpsPipelinePTO(platform=platform))
        assert result.passed, f"submit dumps= pipeline execution failed: {result.error}"


@pytest.fixture(scope="module")
def submit_dumps_manifest_file(test_runner) -> Path:
    """Run the submit pipeline once with --dump-tensor and return the manifest path."""
    if not test_runner.config.enable_dump_tensor:
        pytest.skip("pass --dump-tensor to validate the submit dumps= manifest")
    if test_runner.config.codegen_only:
        pytest.skip("--codegen-only skips device execution; no manifest is written")

    pattern = "*/dfx_outputs/tensor_dump/tensor_dump.json"
    before: set[Path] = set(_BUILD_OUTPUT_DIR.glob(pattern))
    result = test_runner.run(_SubmitDumpsPipelinePTO())
    assert result.passed, f"submit dumps= pipeline failed: {result.error}"

    after: set[Path] = set(_BUILD_OUTPUT_DIR.glob(pattern))
    new_files = after - before
    assert new_files, "No tensor_dump.json was generated for the submit dumps= run"
    return max(new_files, key=lambda p: p.stat().st_mtime)


@pytest.fixture(scope="module")
def submit_dumps_manifest(submit_dumps_manifest_file: Path) -> list[dict]:
    """Parse ``tensor_dump.json`` and return the entry list (the ``tensors`` key)."""
    manifest = json.loads(submit_dumps_manifest_file.read_text())
    assert isinstance(manifest, dict), (
        f"{submit_dumps_manifest_file}: expected a dict, got {type(manifest).__name__}"
    )
    entries = manifest.get("tensors")
    assert isinstance(entries, list) and entries, (
        f"{submit_dumps_manifest_file}: 'tensors' missing or empty — dump pipeline produced no entries"
    )
    return entries


class TestSubmitDumpsManifest:
    """Manifest validation for ``dumps=`` — only runs when ``--dump-tensor`` is enabled."""

    def test_only_dumped_submit_appears(self, submit_dumps_manifest):
        """Selective dump must drop stage2 entirely.

        Only stage1 carries ``dumps=[x, scratch]``; stage2 has no ``dumps=``,
        so codegen emits no ``.dump(...)`` for it. The manifest must therefore
        contain entries from a single task (one ``task_id``, stage1) only — the
        submit analogue of ``test_only_tagged_kernel_dumps`` above. The runtime
        identifies each dispatch by ``task_id``; ``func_id`` / ``subtask_id``
        are runtime-internal and not surfaced per manifest entry.
        """
        entries = submit_dumps_manifest
        task_ids = {e["task_id"] for e in entries}
        assert len(task_ids) == 1, (
            f"selective dump should retain entries from a single submitted kernel, "
            f"found {len(task_ids)} task_ids={sorted(task_ids)}"
        )

    def test_dumped_roles_cover_input_and_inout(self, submit_dumps_manifest):
        """``dumps=[x, scratch]`` dumps one input (x) and one inout (scratch) slot."""
        roles = {e["role"] for e in submit_dumps_manifest}
        assert "input" in roles, f"missing role=input entries; have {sorted(roles)}"
        assert "inout" in roles, f"missing role=inout entries; have {sorted(roles)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
