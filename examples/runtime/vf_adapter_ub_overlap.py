# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Complete A5 Tile-adapter and overlapping-UB example, exercising both sync layers.

Run from a matching, locally built worktree with an allocated device:
    python examples/runtime/vf_adapter_ub_overlap.py --device 0 --output build/vf-example --timeout 600

The output directory must be new. All compiler and hardware subprocesses are
bounded by the parent timeout. The compilation recorder observes real results;
only the explicit changed stage edits helper source, from the first broadcast scalar to the second scalar.
"""

import argparse
import hashlib
import importlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import pypto.language as pl
import torch
from pypto import CacheConfig
from pypto.backend import BackendType
from pypto.ir.compiled_program import CompiledProgram
from pypto.jit._artifact_manifest import BuildKind
from pypto.runtime import ChipWorker, ExecutionMode, RunConfig
from pypto.runtime._artifact_sources import read_kernel_config, validate_generated_sources
from pypto.runtime.elf_parser import extract_text_section
from pypto.runtime.runtime_pin import check_runtime_pin

if TYPE_CHECKING:
    from pypto.runtime.kernel_compiler import KernelCompiler

LENGTHS = (-1, 0, 1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 127, 128, 1023, 1024)


class StageResult(TypedDict):
    stage: str
    checks: int


def _prepare(directory: Path) -> CompiledProgram:
    @pl.jit.incore
    def body(
        out: pl.InOut[pl.Tensor[[1, 640], pl.FP32]],
        first: pl.Scalar[pl.FP32],
        second: pl.Scalar[pl.FP32],
        n: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[1, 640], pl.FP32]:
        a = pl.load(out, [0, 0], [1, 640], target_memory=pl.MemorySpace.Vec)
        outer = pl.slice(a, [1, 576], [0, 64])
        left = pl.reshape(pl.slice(outer, [1, 64], [0, 0]), [64])
        right = pl.reshape(pl.slice(outer, [1, 64], [0, 32]), [64])
        saved = pl.reshape(pl.slice(outer, [1, 64], [0, 256]), [64])
        saved2 = pl.reshape(pl.slice(outer, [1, 64], [0, 384]), [64])
        saved3 = pl.reshape(pl.slice(outer, [1, 64], [0, 512]), [64])
        with pl.reg_scope():
            index = pl.reg.index_gen(0, lanes=64, dtype=pl.INT32)
            limit = pl.reg.broadcast(n, lanes=64, dtype=pl.INT32)
            seed = pl.reg.full_mask(lanes=64)
            prefix = pl.reg.compare(index, limit, seed, predicate="lt")
            suffix = pl.reg.compare(index, limit, seed, predicate="ge")
            x = pl.reg.broadcast(first, lanes=64, dtype=pl.FP32)
            y = pl.reg.broadcast(second, lanes=64, dtype=pl.FP32)
            old = pl.reg.load(right, offset=0, lanes=64)
            left = pl.reg.store(left, offset=0, value=x, mask=prefix)
            middle = pl.reg.load(right, offset=0, lanes=64)
            right = pl.reg.store(right, offset=0, value=y, mask=prefix)
            left = pl.reg.store(left, offset=0, value=x, mask=suffix)
            after = pl.reg.load(left, offset=0, lanes=64)
            saved = pl.reg.store(saved, offset=0, value=old)
            saved2 = pl.reg.store(saved2, offset=0, value=middle)
            saved3 = pl.reg.store(saved3, offset=0, value=after)
        pl.store(a, [0, 0], out)
        return out

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

    compiled = dispatch.compile(
        config=RunConfig(
            backend_type=BackendType.Ascend950,
            execution_mode=ExecutionMode.ONBOARD,
            cache_config=CacheConfig(enabled=False),
            save_kernels_dir=str(directory),
            dump_passes=True,
        ),
    )
    kernel = read_kernel_config(directory / "kernel_config.py").KERNELS[0]
    assert kernel["core_type"] == "aiv"
    assert kernel["vpto_sources"], "Normal JIT lost the structured helper bundle"
    assert all("pto.vmi." in Path(p).read_text() for p in kernel["vpto_sources"])
    wrapper = Path(kernel["source"]).read_text()
    assert "__pypto_vf_" in wrapper
    assert "args[3]" in wrapper
    _check_tile_sync(directory, wrapper)
    return compiled


def _check_tile_sync(directory: Path, wrapper: str) -> None:
    """Check the production caller/adapter contract without rewriting sources."""
    manifest = json.loads((directory / "vpto/body/manifest.json").read_text())
    assert manifest["schema_version"] == 3
    assert manifest["synchronization"] == "ptoas-insert-sync"
    assert manifest["effect_scope"] == "whole-tile"
    helper = manifest["helpers"][0]
    assert helper["parameter_effects"] == ["readwrite"] * 5 + ["none"] * 3
    assert helper["parameter_types"] == ["!pto.ptr<f32, ub>", "i32", "f32", "f32"]
    assert all(t.startswith("!pto.tile_buf<") for t in helper["caller_parameter_types"][:5])
    pto = (directory / "ptoas/body.pto").read_text()
    main, adapter = pto.split(f"func.func private @{helper['adapter_symbol']}(")
    assert f"func.call @{helper['adapter_symbol']}" in main
    assert "pto.tile_buf_addr" not in main
    assert "pto.barrier" not in main
    assert adapter.count("pto.tile_buf_addr") == 1
    assert 'pto.ptodsl.subkernel_helper = "simd"' in adapter
    assert "static static" not in wrapper
    call = re.search(rf"\n\s+{helper['adapter_symbol']}\(v", wrapper)
    assert call is not None
    prefix, suffix = wrapper[: call.start()], wrapper[call.end() :]
    assert "wait_flag(PIPE_MTE2, PIPE_V," in prefix
    assert "set_flag(PIPE_V, PIPE_MTE3," in suffix
    assert "wait_flag(PIPE_V, PIPE_MTE3," in suffix
    assert "pipe_barrier(PIPE_ALL)" not in prefix[prefix.index("  TLOAD(") :]
    assert "pipe_barrier(PIPE_ALL)" not in suffix[: suffix.index("  TSTORE(")]
    (directory.parent / "synchronization.json").write_text(
        json.dumps(
            {
                "production_codegen": True,
                "generated_source_rewritten": False,
                "ptoas_source_modified": False,
                "helper": helper,
                "automatic_mte2_v_mte3_events": True,
                "fixed_callsite_pipe_all": False,
                "kernel_tail_sync_retained": True,
            },
            indent=2,
        )
        + "\n"
    )


@contextmanager
def _record_compilation(root: Path) -> Iterator[list[str]]:
    """Retain real compiler inputs/outputs before temporary build cleanup."""
    compiler_class = importlib.import_module("pypto.runtime.kernel_compiler").KernelCompiler

    original_compile = compiler_class.compile_incore
    original_run = compiler_class._run
    calls: list[str] = []

    def counted_compile(self: "KernelCompiler", *args: Any, **kwargs: Any) -> bytes:
        assert kwargs.get("vpto_sources"), "JIT did not attach a VPTO helper"
        evidence = root / f"compilation-{len(calls) + 1}"
        evidence.mkdir(parents=True, exist_ok=True)
        inputs = {"caller.cpp": Path(args[0])}
        inputs.update({f"helper-{i}.pto": Path(p) for i, p in enumerate(kwargs["vpto_sources"])})
        hashes = {}
        for name, source in inputs.items():
            data = source.read_bytes()
            (evidence / name).write_bytes(data)
            hashes[name] = hashlib.sha256(data).hexdigest()
        (evidence / "inputs.json").write_text(json.dumps(hashes, indent=2) + "\n")
        binary = original_compile(self, *args, **kwargs)
        calls.append(str(args[0]))
        return binary

    def recorded_run(self: "KernelCompiler", cmd: list[str], output: Path, label: str) -> bytes:
        evidence = root / f"compilation-{len(calls) + 1}"
        evidence.mkdir(parents=True, exist_ok=True)
        (evidence / f"{output.name}.command.txt").write_text(
            f"cd {shlex.quote(str(self.project_root))} && {shlex.join(cmd)}\n"
        )
        for arg in cmd:
            if arg.startswith("--symbol-ordering-file="):
                order = Path(arg.split("=", 1)[1])
                (evidence / order.name).write_bytes(order.read_bytes())
        binary = original_run(self, cmd, output, label)
        (evidence / output.name).write_bytes(binary)
        return binary

    # These hooks record the original compiler's bytes. They do not inject code
    # or replace the runtime; this single-program example restores both hooks.
    compiler_class.compile_incore = counted_compile
    compiler_class._run = recorded_run
    try:
        yield calls
    finally:
        compiler_class.compile_incore = original_compile
        compiler_class._run = original_run


def _check_values(worker: ChipWorker, compiled: CompiledProgram, stage: str) -> int:
    handle = worker.register(compiled)
    count = 0
    for repeat in range(3):
        for n in LENGTHS:
            output = torch.arange(704, dtype=torch.float32).reshape(1, 704) + repeat * 100
            output[:, :32] = -777
            output[:, 672:] = -888
            first, second = float(1000 + repeat), float(-2000 - repeat)
            x = second if stage in ("changed", "relocated") else first
            expected = output.clone()
            left, right = expected[:, 96:160], expected[:, 128:192]
            active = max(0, min(n, 64))
            old = right.clone()
            left[:, :active] = x
            middle = right.clone()
            right[:, :active] = second
            left[:, active:] = x
            after = left.clone()
            expected[:, 352:416] = old
            expected[:, 480:544] = middle
            expected[:, 608:672] = after
            handle(output, first, second, n)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            count += 1
            print(f"PASS stage={stage} repeat={repeat} n={n} offset=32 guards=32", flush=True)
    return count


def _check_internal_sync(output: Path) -> list[str]:
    directory = output / "compilation-1"
    llvm = (directory / "vpto_0.ll").read_text()
    expected = ["vld.vst", "vst.vld", "vld.vst", "vst.vst", "vst.vld"]
    barriers = re.findall(r"call void @llvm.hivm.mem.bar.([a-z.]+)\(\)", llvm)
    assert barriers == expected, barriers
    objdump = Path(os.environ["ASCEND_HOME_PATH"]) / "bin/llvm-objdump"
    observed = {}
    for name in ("vpto_0.o", "kernel.elf"):
        command = [str(objdump), "-d", "-r", str(directory / name)]
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
        (directory / f"{name}-disassembly.log").write_text(result.stdout)
        (directory / f"{name}-disassembly.command.txt").write_text(shlex.join(command) + "\n")
        machine = re.findall(r"\bSMEM_BAR.([A-Z_]+)", result.stdout)
        assert machine == [item.upper().replace(".", "_") for item in expected], machine
        observed[name] = machine
    (output / "internal-sync.json").write_text(
        json.dumps({"llvm": barriers, "machine": observed, "ptoas_inserted": True}, indent=2) + "\n"
    )
    return barriers


def _run(output: Path, device: int) -> None:
    pin = check_runtime_pin(checkout_root=Path(__file__).resolve().parents[2])
    assert pin.matches and pin.pinned == pin.installed, pin
    directory = output / "original"
    compiled = _prepare(directory)
    config = RunConfig(
        backend_type=BackendType.Ascend950, execution_mode=ExecutionMode.ONBOARD, device_id=device
    )
    results: list[StageResult] = []
    with _record_compilation(output) as calls:
        # Build before acquiring the persistent runtime's device resources.
        compiled.load()
        assert len(calls) == 1
        internal_barriers = _check_internal_sync(output)
        binaries = list((directory / "cache").glob("incore_*.bin"))
        assert len(binaries) == 1
        cold_text = binaries[0].read_bytes()
        linked_text = extract_text_section((output / "compilation-1/kernel.elf").read_bytes())
        assert cold_text == linked_text and not cold_text.startswith(b"\x7fELF")
        (output / "cold.text.bin").write_bytes(cold_text)
        with ChipWorker(config=config, runtime=compiled.runtime_name) as worker:
            results.append({"stage": "cold", "checks": _check_values(worker, compiled, "cold")})
            warm = CompiledProgram.from_dir(directory)
            warm.load()
            assert len(calls) == 1, "Warm artifact missed the binary cache"
            results.append({"stage": "warm", "checks": _check_values(worker, warm, "warm")})

            # Deliberate invalidation experiment, after normal DSL execution.
            # The JIT generated the initial helper and vpto_sources itself.
            helper = Path(read_kernel_config(directory / "kernel_config.py").KERNELS[0]["vpto_sources"][0])
            source = helper.read_text()
            first_match = re.search(r"(%x[^ ]* = pto.vmi.vbrc )(%arg[0-9]+)", source)
            second_match = re.search(r"%y[^ ]* = pto.vmi.vbrc (%arg[0-9]+)", source)
            assert first_match is not None and second_match is not None
            source = source[: first_match.start(2)] + second_match[1] + source[first_match.end(2) :]
            helper.write_text(source)
            changed = CompiledProgram.from_dir(directory)
            changed.load()
            assert len(calls) == 2, "Helper mutation did not invalidate the cache"
            changed_text = extract_text_section((output / "compilation-2/kernel.elf").read_bytes())
            assert changed_text != cold_text
            assert any(p.read_bytes() == changed_text for p in (directory / "cache").glob("incore_*.bin"))
            results.append({"stage": "changed", "checks": _check_values(worker, changed, "changed")})

            relocated = output / "relocated"
            directory.rename(relocated)
            assert not directory.exists()
            validate_generated_sources(relocated, BuildKind.SINGLE_CHIP)
            moved = CompiledProgram.from_dir(relocated)
            moved.load()
            assert len(calls) == 2, "Relocation missed the content-addressed cache"
            results.append({"stage": "relocated", "checks": _check_values(worker, moved, "relocated")})

    report = {
        "device_id": device,
        "hardware_executed": True,
        "production_tile_adapter": True,
        "synchronization": "ptoas-insert-sync",
        "runtime": pin.installed,
        "runtime_pin_matches": pin.matches,
        "runtime_off_pin": pin.off_pin,
        "stages": results,
        "internal_barriers": internal_barriers,
        "lengths": LENGTHS,
        "numerical_checks": sum(item["checks"] for item in results),
        "compile_calls": len(calls),
        "text_bytes": len(cold_text),
        "cold_text_sha256": hashlib.sha256(cold_text).hexdigest(),
        "changed_text_sha256": hashlib.sha256(changed_text).hexdigest(),
        "elf_text_matches_worker_input": True,
        "original_path_removed": not directory.exists(),
    }
    assert report["numerical_checks"] == 192
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True, help="Explicit allocated A5 device ID")
    parser.add_argument("--output", type=Path, required=True, help="New evidence directory")
    parser.add_argument("--timeout", type=int, default=600, help="Whole child process timeout in seconds")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.device < 0 or args.timeout <= 0:
        parser.error("--device must be nonnegative and --timeout must be positive")
    if os.environ.get("PYPTO_SKIP_RUNTIME_PIN_CHECK", "0") != "0":
        parser.error("This example requires runtime pin checking")
    for key in ("LD_LIBRARY_PATH", "LD_PRELOAD"):
        if any(token in os.environ.get(key, "").lower() for token in ("simulator", "camodel", "msopprof")):
            parser.error(f"Remove simulator injection from {key} before ONBOARD execution")
    output = args.output.resolve()
    if args.child:
        _run(output, args.device)
        return 0
    if output.exists():
        parser.error(f"Output directory already exists; choose a new path: {output}")
    output.mkdir(parents=True)
    (output / "example.py").write_bytes(Path(__file__).read_bytes())
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--device",
        str(args.device),
        "--output",
        str(output),
        "--timeout",
        str(args.timeout),
    ]
    (output / "execution.command.txt").write_text(shlex.join(command) + "\n")
    with (output / "execution.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            # Kill the process group, including compiler/runtime descendants.
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            log.write(f"\nTIMEOUT after {args.timeout} seconds\n")
            code = 124
    print((output / "execution.log").read_text(), end="")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
