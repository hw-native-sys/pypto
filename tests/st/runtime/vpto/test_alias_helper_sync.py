# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Distinct captured Tile views through production outlining and persistent A5.

No generated-source rewriting: the helper must recover a common UB base from
the planner while the adapter retains the separate Tile effects.
"""

import json
import os
import re
import runpy
import sys
from pathlib import Path

import pypto.language as pl
import pytest
import torch
from pypto import CacheConfig
from pypto.backend import BackendType
from pypto.runtime import ChipWorker, ExecutionMode, RunConfig
from pypto.runtime.runtime_pin import check_runtime_pin
from test_tile_helper_sync import _environment, _record_compilation
from test_vpto_vf_onboard import FIXTURE, ROOT

pytestmark = pytest.mark.platforms("a5", reason="VPTO helpers require the A5 AIV backend")


def _audit_machine_code(evidence: Path, kind: str, offset: int) -> dict[str, list[str]]:
    """Check the helper object and linked image, allowing observed dead-code elimination."""
    support = runpy.run_path(str(FIXTURE / "test_validation.py"))
    objdump = Path(os.environ["ASCEND_HOME_PATH"]) / "bin/llvm-objdump"
    expected_kind = {"wr": "VST_VLD", "rw": "VLD_VST", "ww": "VST_VST"}[kind]
    # For exact-address WR, Bisheng forwards the stored register to the result.
    # For exact-address WW, the overwritten first store is eliminated.
    eliminated = offset == 0 and kind in ("wr", "ww")
    expected = [expected_kind] if offset < 64 and not eliminated else []
    results = {}
    for filename in ("vpto_0.o", "kernel.elf"):
        disassembly = support["run"](
            evidence, f"{filename}-disassembly", [objdump, "-d", "-r", evidence / filename]
        )
        barriers = re.findall(r"\bSMEM_BAR\.([A-Z_]+)", disassembly)
        assert barriers == expected, (kind, offset, filename, barriers)
        results[filename] = barriers
    return results


def _prepare(directory: Path, kind: str, offset: int):
    if kind == "wr":

        @pl.jit.incore
        def body(
            out: pl.InOut[pl.Tensor[[1, 320], pl.FP32]],
            first: pl.Scalar[pl.FP32],
            second: pl.Scalar[pl.FP32],
        ) -> pl.Tensor[[1, 320], pl.FP32]:
            a = pl.load(out, [0, 0], [1, 320], target_memory=pl.MemorySpace.Vec)
            left = pl.reshape(pl.slice(a, [1, 64], [0, 0]), [64])
            right = pl.reshape(pl.slice(a, [1, 64], [0, offset]), [64])
            saved = pl.reshape(pl.slice(a, [1, 64], [0, 256]), [64])
            with pl.reg_scope():
                x = pl.reg.broadcast(first, lanes=64, dtype=pl.FP32)
                left = pl.reg.store(left, offset=0, value=x)
                y = pl.reg.load(right, offset=0, lanes=64)
                saved = pl.reg.store(saved, offset=0, value=y)
            pl.store(a, [0, 0], out)
            return out

    elif kind == "rw":

        @pl.jit.incore
        def body(
            out: pl.InOut[pl.Tensor[[1, 320], pl.FP32]],
            first: pl.Scalar[pl.FP32],
            second: pl.Scalar[pl.FP32],
        ) -> pl.Tensor[[1, 320], pl.FP32]:
            a = pl.load(out, [0, 0], [1, 320], target_memory=pl.MemorySpace.Vec)
            left = pl.reshape(pl.slice(a, [1, 64], [0, 0]), [64])
            right = pl.reshape(pl.slice(a, [1, 64], [0, offset]), [64])
            saved = pl.reshape(pl.slice(a, [1, 64], [0, 256]), [64])
            with pl.reg_scope():
                y = pl.reg.load(left, offset=0, lanes=64)
                x = pl.reg.broadcast(first, lanes=64, dtype=pl.FP32)
                right = pl.reg.store(right, offset=0, value=x)
                saved = pl.reg.store(saved, offset=0, value=y)
            pl.store(a, [0, 0], out)
            return out

    elif kind == "ww":

        @pl.jit.incore
        def body(
            out: pl.InOut[pl.Tensor[[1, 320], pl.FP32]],
            first: pl.Scalar[pl.FP32],
            second: pl.Scalar[pl.FP32],
        ) -> pl.Tensor[[1, 320], pl.FP32]:
            a = pl.load(out, [0, 0], [1, 320], target_memory=pl.MemorySpace.Vec)
            left = pl.reshape(pl.slice(a, [1, 64], [0, 0]), [64])
            right = pl.reshape(pl.slice(a, [1, 64], [0, offset]), [64])
            with pl.reg_scope():
                x = pl.reg.broadcast(first, lanes=64, dtype=pl.FP32)
                y = pl.reg.broadcast(second, lanes=64, dtype=pl.FP32)
                left = pl.reg.store(left, offset=0, value=x)
                right = pl.reg.store(right, offset=0, value=y)
            pl.store(a, [0, 0], out)
            return out

    else:
        raise ValueError(f"Unsupported hazard kind: {kind}")

    @pl.jit
    def dispatch(
        out: pl.InOut[pl.Tensor[[1, 384], pl.FP32]],
        first: pl.Scalar[pl.FP32],
        second: pl.Scalar[pl.FP32],
    ) -> pl.Tensor[[1, 384], pl.FP32]:
        view = pl.slice(out, [1, 320], [0, 32])
        result = body(view, first, second)
        out = pl.assemble(out, result, [0, 32])
        return out

    return dispatch.compile(
        config=RunConfig(
            backend_type=BackendType.Ascend950,
            execution_mode=ExecutionMode.ONBOARD,
            cache_config=CacheConfig(enabled=False),
            save_kernels_dir=str(directory),
        )
    )


def _run(output: Path, device: int) -> None:
    pin = check_runtime_pin(checkout_root=ROOT)
    assert pin.matches and pin.pinned == pin.installed, pin
    (output / "executed-test-source.py").write_bytes(Path(__file__).read_bytes())
    records = []
    programs = []
    for kind in ("wr", "rw", "ww"):
        for offset in (0, 32, 64, 128):
            case = output / f"{kind}-{offset}"
            compiled = _prepare(case / "artifact", kind, offset)
            with _record_compilation(case / "objects") as calls:
                compiled.load()
                assert len(calls) == 1
            evidence = case / "objects/compilation-1"
            llvm = (evidence / "vpto_0.ll").read_text()
            barriers = [line.strip() for line in llvm.splitlines() if "call " in line and "mem.bar" in line]
            assert len(barriers) == (1 if offset < 64 else 0), barriers
            machine_barriers = _audit_machine_code(evidence, kind, offset)
            manifest = json.loads((case / "artifact/vpto/body/manifest.json").read_text())
            helper = manifest["helpers"][0]
            assert manifest["schema_version"] == 3
            assert helper["parameter_types"].count("!pto.ptr<f32, ub>") == 1
            expected_offsets = [0, offset * 4] + ([1024] if kind != "ww" else [])
            assert helper["capture_byte_offsets"][: len(expected_offsets)] == expected_offsets
            assert len([e for e in helper["parameter_effects"] if e != "none"]) == len(expected_offsets)
            (case / "artifact/alias-check.json").write_text(
                json.dumps({"kind": kind, "offset": offset, "barriers": barriers}, indent=2) + "\n"
            )
            programs.append((compiled, kind, offset))
            records.append(
                {"kind": kind, "offset": offset, "barriers": barriers, "machine_barriers": machine_barriers}
            )

    checks = 0
    config = RunConfig(
        backend_type=BackendType.Ascend950, execution_mode=ExecutionMode.ONBOARD, device_id=device
    )
    with ChipWorker(config=config, runtime=programs[0][0].runtime_name) as worker:
        for compiled, kind, offset in programs:
            handle = worker.register(compiled)
            for repeat in range(10):
                out = torch.arange(384, dtype=torch.float32).reshape(1, 384) + repeat * 100
                out[:, :32] = -777
                out[:, 352:] = -888
                expected = out.clone()
                view = expected[:, 32:352]
                first, second = float(1000 + repeat), float(-2000 - repeat)
                if kind == "wr":
                    view[:, :64] = first
                    view[:, 256:320] = view[:, offset : offset + 64].clone()
                elif kind == "rw":
                    old = view[:, :64].clone()
                    view[:, offset : offset + 64] = first
                    view[:, 256:320] = old
                else:
                    view[:, :64] = first
                    view[:, offset : offset + 64] = second
                handle(out, first, second)
                torch.testing.assert_close(out, expected, rtol=0, atol=0)
                checks += 1
                print(f"PASS {kind} offset={offset} repeat={repeat} guards=32", flush=True)
    assert checks == 120
    (output / "result.json").write_text(
        json.dumps(
            {
                "device_id": device,
                "runtime_pin": pin.installed,
                "runtime_pin_matches": pin.matches,
                "hardware_executed": True,
                "numerical_checks": checks,
                "cases": records,
                "generated_source_rewritten": False,
                "automatic_view_merging": True,
            },
            indent=2,
        )
        + "\n"
    )


def test_static_alias_persistent_onboard():
    device, timeout, output = _environment()
    runpy.run_path(str(FIXTURE / "test_validation.py"))["run"](
        output,
        "alias-worker",
        [sys.executable, Path(__file__).resolve(), "--worker", output, device],
        timeout=timeout,
    )
    assert json.loads((output / "result.json").read_text())["numerical_checks"] == 120


MASK_LENGTHS = (-1, 0, 1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 127, 128, 1023, 1024)
COMPLEX_CASES = (
    [(kind, offset, False, False) for kind in ("wr", "rw", "ww") for offset in (0, 8, 32, 56, 64, 96)]
    + [(kind, 32, True, False) for kind in ("wr", "rw", "ww")]
    + [("chain", offset, False, False) for offset in (8, 32, 64)]
    + [("ww", offset, False, True) for offset in (0, 32, 64)]
)


EXTENDED_CASES = [("ww", 32, True, True), ("triangle", 32, False, False), ("triangle", 64, False, False)]


def _triangle_body(left_offset: int, right_offset: int):
    @pl.jit.incore
    def body(
        out: pl.InOut[pl.Tensor[[1, 640], pl.FP32]],
        first: pl.Scalar[pl.FP32],
        second: pl.Scalar[pl.FP32],
        n: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[1, 640], pl.FP32]:
        a = pl.load(out, [0, 0], [1, 640], target_memory=pl.MemorySpace.Vec)
        outer = pl.slice(a, [1, 576], [0, 64])
        left = pl.reshape(pl.slice(outer, [1, 64], [0, left_offset]), [64])
        right = pl.reshape(pl.slice(outer, [1, 64], [0, right_offset]), [64])
        third = pl.reshape(pl.slice(outer, [1, 64], [0, 16]), [64])
        saved = pl.reshape(pl.slice(outer, [1, 64], [0, 256]), [64])
        saved2 = pl.reshape(pl.slice(outer, [1, 64], [0, 384]), [64])
        saved3 = pl.reshape(pl.slice(outer, [1, 64], [0, 512]), [64])
        with pl.reg_scope():
            index = pl.reg.index_gen(0, lanes=64, dtype=pl.INT32)
            limit = pl.reg.broadcast(n, lanes=64, dtype=pl.INT32)
            seed = pl.reg.full_mask(lanes=64)
            prefix = pl.reg.compare(index, limit, seed, predicate="lt")
            x = pl.reg.broadcast(first, lanes=64, dtype=pl.FP32)
            suffix = pl.reg.compare(index, limit, seed, predicate="ge")
            y = pl.reg.broadcast(second, lanes=64, dtype=pl.FP32)
            old = pl.reg.load(right, offset=0, lanes=64)
            left = pl.reg.store(left, offset=0, value=x, mask=prefix)
            middle = pl.reg.load(right, offset=0, lanes=64)
            right = pl.reg.store(right, offset=0, value=y, mask=prefix)
            after = pl.reg.load(third, offset=0, lanes=64)
            third = pl.reg.store(third, offset=0, value=x, mask=suffix)
            saved = pl.reg.store(saved, offset=0, value=old)
            saved2 = pl.reg.store(saved2, offset=0, value=middle)
            saved3 = pl.reg.store(saved3, offset=0, value=after)
        pl.store(a, [0, 0], out)
        return out

    return body


def _prepare_complex(directory: Path, kind: str, offset: int, reverse: bool, split: bool):
    left_offset, right_offset = (offset, 0) if reverse else (0, offset)
    second_predicate = "ge" if split else "lt"
    if kind == "wr":

        @pl.jit.incore
        def body(
            out: pl.InOut[pl.Tensor[[1, 640], pl.FP32]],
            first: pl.Scalar[pl.FP32],
            second: pl.Scalar[pl.FP32],
            n: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, 640], pl.FP32]:
            a = pl.load(out, [0, 0], [1, 640], target_memory=pl.MemorySpace.Vec)
            outer = pl.slice(a, [1, 576], [0, 64])
            left = pl.reshape(pl.slice(outer, [1, 64], [0, left_offset]), [64])
            right = pl.reshape(pl.slice(outer, [1, 64], [0, right_offset]), [64])
            saved = pl.reshape(pl.slice(outer, [1, 64], [0, 256]), [64])
            with pl.reg_scope():
                index = pl.reg.index_gen(0, lanes=64, dtype=pl.INT32)
                limit = pl.reg.broadcast(n, lanes=64, dtype=pl.INT32)
                seed = pl.reg.full_mask(lanes=64)
                prefix = pl.reg.compare(index, limit, seed, predicate="lt")
                x = pl.reg.broadcast(first, lanes=64, dtype=pl.FP32)
                left = pl.reg.store(left, offset=0, value=x, mask=prefix)
                y = pl.reg.load(right, offset=0, lanes=64)
                saved = pl.reg.store(saved, offset=0, value=y)
            pl.store(a, [0, 0], out)
            return out

    elif kind == "rw":

        @pl.jit.incore
        def body(
            out: pl.InOut[pl.Tensor[[1, 640], pl.FP32]],
            first: pl.Scalar[pl.FP32],
            second: pl.Scalar[pl.FP32],
            n: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, 640], pl.FP32]:
            a = pl.load(out, [0, 0], [1, 640], target_memory=pl.MemorySpace.Vec)
            outer = pl.slice(a, [1, 576], [0, 64])
            left = pl.reshape(pl.slice(outer, [1, 64], [0, left_offset]), [64])
            right = pl.reshape(pl.slice(outer, [1, 64], [0, right_offset]), [64])
            saved = pl.reshape(pl.slice(outer, [1, 64], [0, 256]), [64])
            with pl.reg_scope():
                index = pl.reg.index_gen(0, lanes=64, dtype=pl.INT32)
                limit = pl.reg.broadcast(n, lanes=64, dtype=pl.INT32)
                seed = pl.reg.full_mask(lanes=64)
                prefix = pl.reg.compare(index, limit, seed, predicate="lt")
                x = pl.reg.broadcast(first, lanes=64, dtype=pl.FP32)
                y = pl.reg.load(left, offset=0, lanes=64)
                right = pl.reg.store(right, offset=0, value=x, mask=prefix)
                saved = pl.reg.store(saved, offset=0, value=y)
            pl.store(a, [0, 0], out)
            return out

    elif kind == "ww":

        @pl.jit.incore
        def body(
            out: pl.InOut[pl.Tensor[[1, 640], pl.FP32]],
            first: pl.Scalar[pl.FP32],
            second: pl.Scalar[pl.FP32],
            n: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, 640], pl.FP32]:
            a = pl.load(out, [0, 0], [1, 640], target_memory=pl.MemorySpace.Vec)
            outer = pl.slice(a, [1, 576], [0, 64])
            left = pl.reshape(pl.slice(outer, [1, 64], [0, left_offset]), [64])
            right = pl.reshape(pl.slice(outer, [1, 64], [0, right_offset]), [64])
            with pl.reg_scope():
                index = pl.reg.index_gen(0, lanes=64, dtype=pl.INT32)
                limit = pl.reg.broadcast(n, lanes=64, dtype=pl.INT32)
                seed = pl.reg.full_mask(lanes=64)
                prefix = pl.reg.compare(index, limit, seed, predicate="lt")
                x = pl.reg.broadcast(first, lanes=64, dtype=pl.FP32)
                other_mask = pl.reg.compare(index, limit, seed, predicate=second_predicate)
                y = pl.reg.broadcast(second, lanes=64, dtype=pl.FP32)
                left = pl.reg.store(left, offset=0, value=x, mask=prefix)
                right = pl.reg.store(right, offset=0, value=y, mask=other_mask)
            pl.store(a, [0, 0], out)
            return out

    elif kind == "chain":

        @pl.jit.incore
        def body(
            out: pl.InOut[pl.Tensor[[1, 640], pl.FP32]],
            first: pl.Scalar[pl.FP32],
            second: pl.Scalar[pl.FP32],
            n: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, 640], pl.FP32]:
            a = pl.load(out, [0, 0], [1, 640], target_memory=pl.MemorySpace.Vec)
            outer = pl.slice(a, [1, 576], [0, 64])
            left = pl.reshape(pl.slice(outer, [1, 64], [0, left_offset]), [64])
            right = pl.reshape(pl.slice(outer, [1, 64], [0, right_offset]), [64])
            saved = pl.reshape(pl.slice(outer, [1, 64], [0, 256]), [64])
            saved2 = pl.reshape(pl.slice(outer, [1, 64], [0, 384]), [64])
            saved3 = pl.reshape(pl.slice(outer, [1, 64], [0, 512]), [64])
            with pl.reg_scope():
                index = pl.reg.index_gen(0, lanes=64, dtype=pl.INT32)
                limit = pl.reg.broadcast(n, lanes=64, dtype=pl.INT32)
                seed = pl.reg.full_mask(lanes=64)
                prefix = pl.reg.compare(index, limit, seed, predicate="lt")
                x = pl.reg.broadcast(first, lanes=64, dtype=pl.FP32)
                suffix = pl.reg.compare(index, limit, seed, predicate="ge")
                y = pl.reg.broadcast(second, lanes=64, dtype=pl.FP32)
                old = pl.reg.load(right, offset=0, lanes=64)
                left = pl.reg.store(left, offset=0, value=x, mask=prefix)
                middle = pl.reg.load(right, offset=0, lanes=64)
                right = pl.reg.store(right, offset=0, value=y, mask=prefix)
                after = pl.reg.load(left, offset=0, lanes=64)
                left = pl.reg.store(left, offset=0, value=y, mask=suffix)
                saved = pl.reg.store(saved, offset=0, value=old)
                saved2 = pl.reg.store(saved2, offset=0, value=middle)
                saved3 = pl.reg.store(saved3, offset=0, value=after)
            pl.store(a, [0, 0], out)
            return out

    elif kind == "triangle":
        body = _triangle_body(left_offset, right_offset)

    else:
        raise ValueError(kind)

    @pl.jit
    def dispatch(
        out: pl.InOut[pl.Tensor[[1, 704], pl.FP32]],
        first: pl.Scalar[pl.FP32],
        second: pl.Scalar[pl.FP32],
        n: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[1, 704], pl.FP32]:
        view = pl.slice(out, [1, 640], [0, 32])
        updated = body(view, first, second, n)
        out = pl.assemble(out, updated, [0, 32])
        return out

    return dispatch.compile(
        config=RunConfig(
            backend_type=BackendType.Ascend950,
            execution_mode=ExecutionMode.ONBOARD,
            cache_config=CacheConfig(enabled=False),
            save_kernels_dir=str(directory),
            dump_passes=True,
        )
    )


def _complex_expected(out, kind, offset, reverse, split, first, second, n):
    expected = out.clone()
    view = expected[:, 32:672]
    left_offset, right_offset = (offset, 0) if reverse else (0, offset)
    left = view[:, 64 + left_offset : 128 + left_offset]
    right = view[:, 64 + right_offset : 128 + right_offset]
    count = max(0, min(n, 64))
    if kind == "wr":
        left[:, :count] = first
        view[:, 320:384] = right.clone()
    elif kind == "rw":
        old = left.clone()
        right[:, :count] = first
        view[:, 320:384] = old
    elif kind == "ww":
        left[:, :count] = first
        if split:
            right[:, count:] = second
        else:
            right[:, :count] = second
    else:
        old = right.clone()
        left[:, :count] = first
        middle = right.clone()
        right[:, :count] = second
        if kind == "triangle":
            third = view[:, 80:144]
            after = third.clone()
            third[:, count:] = first
        else:
            after = left.clone()
            left[:, count:] = second
        view[:, 320:384] = old
        view[:, 448:512] = middle
        view[:, 576:640] = after
    return expected


def _run_complex(output: Path, device: int, *, extended: bool = False) -> None:
    selected_cases = EXTENDED_CASES if extended else COMPLEX_CASES
    pin = check_runtime_pin(checkout_root=ROOT)
    assert pin.matches and pin.pinned == pin.installed, pin
    (output / "executed-complex-test-source.py").write_bytes(Path(__file__).read_bytes())
    records, programs = [], []
    support = runpy.run_path(str(FIXTURE / "test_validation.py"))
    for kind, offset, reverse, split in selected_cases:
        name = f"{kind}-{offset}-reverse{int(reverse)}-split{int(split)}"
        case = output / name
        compiled = _prepare_complex(case / "artifact", kind, offset, reverse, split)
        with _record_compilation(case / "objects") as calls:
            compiled.load()
            assert len(calls) == 1
        evidence = case / "objects/compilation-1"
        llvm = (evidence / "vpto_0.ll").read_text()
        barriers = re.findall(r"call void @llvm.hivm.mem.bar.([a-z.]+)\(\)", llvm)
        if kind not in ("chain", "triangle"):
            assert len(barriers) == (1 if offset < 64 else 0), (name, barriers)
        else:
            expected = ["vld.vst", "vst.vld", "vld.vst"]
            if offset < 64:
                expected = ["vld.vst", "vst.vld"] + expected
            assert barriers == expected, (name, barriers)
        machine = {}
        for filename in ("vpto_0.o", "kernel.elf"):
            disassembly = support["run"](
                evidence,
                f"{filename}-disassembly",
                [Path(os.environ["ASCEND_HOME_PATH"]) / "bin/llvm-objdump", "-d", "-r", evidence / filename],
            )
            observed = re.findall(r"\bSMEM_BAR.([A-Z_]+)", disassembly)
            if kind not in ("chain", "triangle") and offset >= 64:
                assert not observed, (name, observed)
            assert observed == [bar.upper().replace(".", "_") for bar in barriers], (name, observed, barriers)
            machine[filename] = observed
        helper = json.loads((case / "artifact/vpto/body/manifest.json").read_text())["helpers"][0]
        memory_offsets = [
            b
            for b, effect in zip(helper["capture_byte_offsets"], helper["parameter_effects"], strict=True)
            if effect != "none"
        ]
        assert helper["parameter_types"].count("!pto.ptr<f32, ub>") == 1
        expected_offsets = {0, offset * 4}
        if kind != "ww":
            expected_offsets.add(256 * 4)
        if kind in ("chain", "triangle"):
            expected_offsets.update((384 * 4, 512 * 4))
        if kind == "triangle":
            expected_offsets.add(16 * 4)
        assert set(memory_offsets) == expected_offsets, (name, memory_offsets)
        records.append(
            {
                "name": name,
                "barriers": barriers,
                "machine_barriers": machine,
                "capture_byte_offsets": memory_offsets,
                "parameter_indices": helper["parameter_indices"],
            }
        )
        programs.append((compiled, kind, offset, reverse, split))

    checks = 0
    config = RunConfig(
        backend_type=BackendType.Ascend950, execution_mode=ExecutionMode.ONBOARD, device_id=device
    )
    with ChipWorker(config=config, runtime=programs[0][0].runtime_name) as worker:
        for compiled, kind, offset, reverse, split in programs:
            handle = worker.register(compiled)
            for repeat in range(3):
                for n in MASK_LENGTHS:
                    out = torch.arange(704, dtype=torch.float32).reshape(1, 704) + repeat * 100
                    out[:, :32] = -777
                    out[:, 672:] = -888
                    first, second = float(1000 + repeat), float(-2000 - repeat)
                    expected = _complex_expected(out, kind, offset, reverse, split, first, second, n)
                    handle(out, first, second, n)
                    torch.testing.assert_close(out, expected, rtol=0, atol=0)
                    checks += 1
                print(
                    f"PASS {kind} offset={offset} reverse={reverse} split={split} repeat={repeat}", flush=True
                )
    assert checks == len(selected_cases) * len(MASK_LENGTHS) * 3
    (output / "complex-result.json").write_text(
        json.dumps(
            {
                "device_id": device,
                "runtime_pin": pin.installed,
                "runtime_pin_matches": pin.matches,
                "hardware_executed": True,
                "numerical_checks": checks,
                "cases": records,
                "mask_lengths": MASK_LENGTHS,
                "generated_source_rewritten": False,
                "nested_tile_view_offset": 64,
                "tensor_offset": 32,
            },
            indent=2,
        )
        + "\n"
    )


def test_complex_masked_alias_persistent_onboard():
    device, timeout, output = _environment()
    runpy.run_path(str(FIXTURE / "test_validation.py"))["run"](
        output,
        "complex-alias-worker",
        [sys.executable, Path(__file__).resolve(), "--complex-worker", output, device],
        timeout=timeout,
    )
    assert (
        json.loads((output / "complex-result.json").read_text())["numerical_checks"]
        == len(COMPLEX_CASES) * len(MASK_LENGTHS) * 3
    )


@pytest.mark.skipif(
    os.environ.get("PYPTO_RUN_VPTO_ALIAS_VALIDATION") != "1", reason="Opt in to A5 compiler validation"
)
@pytest.mark.parametrize(
    "case",
    [
        ("wr", 8, False, False),
        ("rw", 32, True, False),
        ("ww", 0, False, True),
        ("chain", 32, False, False),
        ("triangle", 32, False, False),
        ("triangle", 64, False, False),
    ],
)
def test_complex_masked_alias_compilation(case):
    output = Path(os.environ["PYPTO_VPTO_ALIAS_OUTPUT"]) / ("preflight-" + "-".join(map(str, case)))
    compiled = _prepare_complex(output / "artifact", *case)
    with _record_compilation(output / "objects") as calls:
        compiled.load()
        assert len(calls) == 1
    assert (output / "objects/compilation-1/kernel.elf").stat().st_size > 0


def test_extended_masked_alias_persistent_onboard():
    device, timeout, output = _environment()
    output = output / "extended"
    output.mkdir(parents=True, exist_ok=True)
    runpy.run_path(str(FIXTURE / "test_validation.py"))["run"](
        output,
        "extended-alias-worker",
        [sys.executable, Path(__file__).resolve(), "--extended-worker", output, device],
        timeout=timeout,
    )
    assert (
        json.loads((output / "complex-result.json").read_text())["numerical_checks"]
        == len(EXTENDED_CASES) * len(MASK_LENGTHS) * 3
    )


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        _run(Path(sys.argv[2]), int(sys.argv[3]))
    elif len(sys.argv) == 4 and sys.argv[1] == "--complex-worker":
        _run_complex(Path(sys.argv[2]), int(sys.argv[3]))
    elif len(sys.argv) == 4 and sys.argv[1] == "--extended-worker":
        _run_complex(Path(sys.argv[2]), int(sys.argv[3]), extended=True)
    else:
        raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))
