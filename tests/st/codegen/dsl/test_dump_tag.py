# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Lightweight system test for ``pl.dump_tag`` end-to-end.

A tiny ``(a + 1) * 2`` kernel built as a ``@pl.jit`` entry composed of two
``@pl.jit.inline`` helpers. ``pl.dump_tag`` markers live in both scopes:

  - Inline-scope: ``pl.dump_tag(x)`` inside ``add_inline``. The
    ``InlineFunctions`` pass migrates this marker onto the caller orch
    after splicing, where the inline param ``x`` resolves to the entry's
    ``a``.
  - Entry-scope: ``pl.dump_tag(intermediate)`` on the body-local
    ``pl.create_tensor`` result.

The entry output ``c`` is intentionally never tagged. With
``--dump-tensor`` the runtime's selective-dump filter should retain only
the tagged bindings in the manifest, so the test can assert both the
positive (``a`` / ``intermediate`` present) and negative (``c`` filtered)
paths in one pass.

Correctness always runs. Manifest validation (parsing the JSON via
``simpler_setup.tools.dump_viewer`` and decoding sample bytes from the
companion bin file referenced by ``manifest["bin_file"]``) is gated
behind ``--dump-tensor`` and ``not codegen_only``.
"""

import dataclasses
import json
import shutil
from pathlib import Path

import pypto.language as pl
import pytest
import torch

_DUMP_TAG_WORK_DIR = Path(__file__).resolve().parents[4] / "build_output" / "dump_tag_test"

_REQUIRED_FIELDS: dict[str, type | tuple[type, ...]] = {
    "task_id": str,
    "subtask_id": int,
    "role": str,
    "stage": str,
    "arg_index": int,
    "func_id": int,
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
    """c = a + 1.0. Inline-scope dump_tag — ``MergeDumpTaggedNames`` migrates
    the literal name ``a`` to the caller's ``dump_tagged_names``; codegen
    matches it against the call-site arg's IR Var name, which is the entry's
    ``a`` after inline param-substitution.
    """
    pl.dump_tag(a)
    with pl.incore():
        tile_a = pl.load(a, [0, 0], [128, 128])
        tile_c = pl.add(tile_a, 1.0)
        pl.store(tile_c, [0, 0], c)
    return c


@pl.jit.inline
def mul_inline(a: pl.Tensor, c: pl.Tensor):
    """c = a * 2.0. No dump_tag here — its bindings should be filtered out."""
    with pl.incore():
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

        After ``InlineFunctions`` migration the orch carries
        ``dump_tagged_names = {"a", "intermediate"}``. Kernel1 (``add_inline``)
        consumes ``a`` and writes ``intermediate`` — both tagged — so codegen
        emits ``params_t0.dump(ext_a, intermediate)``.

        Kernel2 (``mul_inline``) consumes the SSA-rebound ``intermediate_v1``
        and writes ``c``. Neither name matches the tag set (``intermediate_v1``
        is a distinct IR binding from ``intermediate`` — dump_tag matching is
        documented as IR-name-based), so codegen emits no ``params_t1.dump``
        call. The manifest must therefore contain entries from a single
        ``func_id`` only.
        """
        entries, manifest_path, _ = dump_manifest
        func_ids = {e["func_id"] for e in entries}
        assert len(func_ids) == 1, (
            f"{manifest_path}: selective dump should retain entries from a single kernel, "
            f"found {len(func_ids)} func_ids={sorted(func_ids)}"
        )
        roles = {e["role"] for e in entries}
        assert "input" in roles, f"{manifest_path}: missing role=input entries; have {sorted(roles)}"
        assert "inout" in roles, f"{manifest_path}: missing role=inout entries; have {sorted(roles)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
