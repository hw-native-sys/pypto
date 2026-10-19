# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Audit EmitC/VPTO compilation, linked device ELF sections and optional execution.

This experiment uses identical PTO and a shared source path for the SDK wrapper.
VPTO requires a native ABI adapter; it does not fit PyPTO's current C++ interface.
All differences and failures remain in the output directory for CI inspection.
"""

import argparse
import difflib
import hashlib
import json
import re
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch


def elf_sections(binary: bytes, mask: int) -> dict[str, tuple[int, int, bytes]]:
    """Read allocated or executable sections of the ELF64 little-endian device ABI.

    Keep virtual addresses and flags: equal instruction bytes at different
    addresses do not establish an identical executable image. NOBITS sections
    retain their size without pretending they occupy bytes in the file.
    """
    if len(binary) < 64 or binary[:6] != b"\x7fELF\x02\x01":
        raise ValueError("Expected an ELF64 little-endian device binary")
    offset = struct.unpack_from("<Q", binary, 40)[0]
    stride, count, names_index = struct.unpack_from("<HHH", binary, 58)
    if stride != 64 or not count or names_index >= count or offset + count * stride > len(binary):
        raise ValueError("Invalid or unsupported ELF section table")
    headers = [struct.unpack_from("<IIQQQQIIQQ", binary, offset + i * stride) for i in range(count)]
    names_header = headers[names_index]
    names = binary[names_header[4] : names_header[4] + names_header[5]]
    sections = {}
    for name_offset, kind, flags, address, start, size, *_ in headers:
        if not flags & mask:
            continue
        end = names.find(b"\0", name_offset)
        if end < 0 or (kind != 8 and start + size > len(binary)):
            raise ValueError("Invalid ELF section name or payload")
        name = names[name_offset:end].decode("utf-8")
        if name in sections:
            raise ValueError(f"Duplicate ELF section: {name}")
        payload = struct.pack("<Q", size) if kind == 8 else binary[start : start + size]
        sections[name] = (address, flags, payload)
    if not sections:
        raise ValueError(f"ELF has no sections matching flags {mask:#x}")
    return sections


def binary_comparison(left: bytes, right: bytes) -> dict[str, Any]:
    """Distinguish whole-file identity from executable and allocated image identity."""
    executable = [elf_sections(data, 4) for data in (left, right)]
    return {
        "binary_equal": left == right,
        "executable_equal": executable[0] == executable[1],
        "executable_sizes": [sum(len(section[2]) for section in image.values()) for image in executable],
        # The current runtime uploads .text, not the entire ELF. Compare those
        # exact payload bytes independently of DWARF and ELF section placement.
        "runtime_text_equal": executable[0][".text"][2] == executable[1][".text"][2],
        "allocated_equal": elf_sections(left, 2) == elf_sections(right, 2),
        "sizes": [len(left), len(right)],
        "sha256": [hashlib.sha256(data).hexdigest() for data in (left, right)],
        "different_bytes": sum(a != b for a, b in zip(left, right)) + abs(len(left) - len(right)),
    }


def write_diff(left: str, right: str, target: Path) -> None:
    """Preserve readable C++ and disassembly differences alongside the hashes."""
    target.write_text(
        "".join(
            difflib.unified_diff(
                left.splitlines(keepends=True),
                right.splitlines(keepends=True),
                fromfile="emitc",
                tofile="vpto",
            )
        )
    )


def adapter_declarations(source: str, symbols: str) -> dict[str, str]:
    """Match C++ signatures to exactly one native VPTO export per function."""
    declarations = {}
    for name, arguments in re.findall(r"(?m)^AICORE void (\w+)\(([^)]*)\)\s*\{", source):
        exported = [
            f"{name}.{kind}"
            for kind in ("vector", "cube")
            if re.search(rf"\b{re.escape(name)}\.{kind}$", symbols, re.MULTILINE)
        ]
        if len(exported) != 1:
            raise RuntimeError(f"Cannot match VPTO entry for {name}: {exported}")
        declarations[name] = f'extern "C" AICORE void {name}({arguments}) __asm__("{exported[0]}");'
    if not declarations:
        raise RuntimeError("No VPTO functions matched the reference signatures")
    return declarations


def compile_backend(
    entry: Any,
    backend: str,
    config: Any,
    work: Path,
    destination: Path,
    inputs: list[Any],
    golden: Any,
    execute: bool,
) -> dict[str, Any]:
    """Compile from scratch and save the real SDK objects before temporary cleanup."""
    import torch  # noqa: PLC0415
    from pypto.backend import pto_backend  # noqa: PLC0415
    from pypto.runtime._artifact_sources import read_kernel_config  # noqa: PLC0415
    from pypto.runtime.debug.pto_rebuild import _splice_body  # noqa: PLC0415
    from pypto.runtime.kernel_compiler import KernelCompiler  # noqa: PLC0415
    from pypto.runtime.pto_isa import ensure_pto_isa_root  # noqa: PLC0415

    destination.mkdir(parents=True)
    pto_commands = []
    compiler_commands = []
    native_objects: dict[str, Path] = {}
    native_declarations: dict[str, str] = {}
    run_ptoas = pto_backend._run_ptoas
    sdk_compiler = KernelCompiler(config.platform)
    assert sdk_compiler.sdk.ccec is not None
    compiler_bin = Path(sdk_compiler.sdk.ccec.cxx_path).parent

    def selected_ptoas(pto_path, output_path, ptoas_flags=None):
        flags = [*(ptoas_flags or []), f"--pto-backend={backend}"]
        pto_commands.append(flags)
        run_ptoas(pto_path, output_path, flags)
        if backend == "emitc":
            return
        # VPTO emits a host fat object, not the C++ PyPTO expects here. Extract
        # the native payload and borrow only EmitC's signature to build an ABI
        # adapter. The EmitC function bodies are discarded before compilation.
        fat_object = destination / f"{Path(output_path).stem}.fat.o"
        shutil.copyfile(output_path, fat_object)
        native = fat_object.with_suffix(".device.o")
        subprocess.run(
            [
                str(compiler_bin / "llvm-objcopy"),
                f"--dump-section=__aicore_rel_binary={native}",
                str(fat_object),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        symbols = subprocess.run(
            [str(compiler_bin / "llvm-objdump"), "-t", str(native)],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout
        emitc_flags = [*(ptoas_flags or []), "--pto-backend=emitc"]
        run_ptoas(pto_path, output_path, emitc_flags)
        declarations = adapter_declarations(Path(output_path).read_text(), symbols)
        for name, declaration in declarations.items():
            native_objects[name] = native
            native_declarations[name] = declaration

    class RecordingCompiler(KernelCompiler):
        kernel_name = ""
        function_name = ""

        def _run(self, cmd, output, label):
            if label == "Incore-link" and backend == "vpto":
                cmd = [*cmd, str(native_objects[self.function_name])]
            compiler_commands.append(cmd)
            result = super()._run(cmd, output, label)
            # compile_incore deletes its private object directory after linking.
            (destination / f"{self.kernel_name}{output.suffix}").write_bytes(result)
            if output.suffix == ".elf":
                (destination / f"{self.kernel_name}.bin").write_bytes(elf_sections(result, 4)[".text"][2])
                assert self.sdk.ccec is not None
                objdump = Path(self.sdk.ccec.cxx_path).with_name("llvm-objdump")
                dump = subprocess.run(
                    [str(objdump), "-d", "-s", str(output)],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=60,
                )
                # Strip only the input filename banner. Instruction addresses,
                # symbols and encodings remain available in the actual diff.
                disassembly = dump.stdout.replace(str(output), "kernel.elf")
                (destination / f"{self.kernel_name}.asm").write_text(disassembly)
            return result

    result: dict[str, Any] = {"passed": False, "executed": False}
    try:
        if work.exists():
            shutil.rmtree(work)
        # Explicit output bypasses JIT caches; changing only a patched PTOAS
        # function would otherwise allow VPTO to reuse an EmitC specialization.
        with patch.object(pto_backend, "_run_ptoas", selected_ptoas):
            compiled = entry.compile(config=config)
        if not pto_commands:
            raise RuntimeError("Audit did not invoke PTOAS; a cache or compile-only fallback was used")
        compiler = RecordingCompiler(config.platform)
        assert compiler.sdk.ccec is not None
        result["compiler"] = compiler.sdk.ccec.cxx_path
        result["compiler_flags"] = {
            core: compiler.sdk.ccec.get_compile_flags(core_type=core) for core in ("aic", "aiv")
        }
        isa = ensure_pto_isa_root()
        # CompiledProgram.runtime_name lazily builds binaries; read metadata
        # directly so it cannot build the adapter without its native object.
        runtime_name = read_kernel_config(work / "kernel_config.py").RUNTIME_CONFIG["runtime"]
        kernels = sorted(work.glob("kernels/*/*.cpp"))
        if not kernels:
            raise RuntimeError("Audit found no generated kernel C++")
        for source in kernels:
            if backend == "vpto":
                _splice_body(source, "\n".join(native_declarations.values()))
            compiler.function_name = source.stem
            compiler.kernel_name = f"{source.parent.name}-{source.stem}"
            compiler.compile_incore(
                str(source),
                core_type=source.parent.name,
                pto_isa_root=isa,
                runtime_name=runtime_name,
            )
        if execute:
            out = torch.full_like(golden, float("nan"))
            loaded_kernels = []

            def audited_binary(_self, source_path, core_type="aiv", **_kwargs):
                # Execute exactly the ELF measured above, including the VPTO
                # adapter, rather than rebuilding through the production path.
                name = f"{core_type}-{Path(source_path).stem}.elf"
                loaded_kernels.append(name)
                return (destination / name).read_bytes()

            with patch.object(KernelCompiler, "compile_incore", audited_binary):
                compiled(*inputs, out, config=config)
            if len(loaded_kernels) != len(kernels):
                raise RuntimeError(
                    f"Execution loaded {loaded_kernels}, expected {len(kernels)} measured ELFs"
                )
            result["executed_kernels"] = loaded_kernels
            torch.testing.assert_close(out, golden, rtol=1e-4, atol=1e-4)
            torch.save(out, destination / "output.pt")
            result["output_sha256"] = hashlib.sha256(out.numpy().tobytes()).hexdigest()
            result["max_abs_error"] = float((out - golden).abs().max())
            result["executed"] = True
        result["passed"] = True
    except Exception as exc:
        # Keep the other backend running so one rejection does not erase the
        # baseline or prevent the CI artifact from containing both diagnostics.
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if work.exists():
            shutil.copytree(work, destination / "generated", dirs_exist_ok=True)
        result["ptoas_flags"] = pto_commands
        result["compiler_commands"] = compiler_commands
        result["requires_native_adapter"] = backend == "vpto"
        result["drop_in_compatible"] = backend == "emitc"
        (destination / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def compare_artifacts(root: Path) -> dict[str, Any]:
    """Require matching nonempty inventories before claiming backend parity."""
    emitc, vpto = root / "emitc", root / "vpto"
    binaries = [{p.name: p for p in directory.glob("*.elf")} for directory in (emitc, vpto)]
    pto = [
        {
            str(p.relative_to(directory / "generated")): p.read_bytes()
            for p in (directory / "generated").rglob("*.pto")
        }
        for directory in (emitc, vpto)
    ]
    result: dict[str, Any] = {
        "same_nonempty_pto": bool(pto[0]) and pto[0] == pto[1],
        "same_nonempty_kernel_inventory": bool(binaries[0]) and binaries[0].keys() == binaries[1].keys(),
        "kernels": {},
    }
    for name in sorted(binaries[0].keys() & binaries[1].keys()):
        result["kernels"][name] = binary_comparison(
            binaries[0][name].read_bytes(), binaries[1][name].read_bytes()
        )
        asm = Path(name).with_suffix(".asm")
        if (emitc / asm).exists() and (vpto / asm).exists():
            write_diff((emitc / asm).read_text(), (vpto / asm).read_text(), root / f"{asm}.diff")
    sources = [
        {
            str(p.relative_to(directory / "generated")): p
            for p in (directory / "generated/kernels").rglob("*.cpp")
        }
        for directory in (emitc, vpto)
    ]
    for name in sorted(sources[0].keys() & sources[1].keys()):
        write_diff(
            sources[0][name].read_text(), sources[1][name].read_text(), root / f"{Path(name).stem}.cpp.diff"
        )
    return result


def main() -> int:
    """Run the same corpus under both backends without acquiring a device by default."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("a2a3", "a5"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", help="Select named cases; default: entire audit corpus")
    parser.add_argument("--execute", action="store_true", help="Also run both backends against CPU goldens")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    from ptoas_backend_cases import cases  # noqa: PLC0415
    from pypto._target import ExecutionMode  # noqa: PLC0415
    from pypto.backend import BackendType  # noqa: PLC0415
    from pypto.backend._ptoas_locate import check_ptoas_version, find_ptoas_binary  # noqa: PLC0415
    from pypto.pypto_core.passes import MemoryPlanner  # noqa: PLC0415
    from pypto.runtime.runner import RunConfig  # noqa: PLC0415

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    ptoas = find_ptoas_binary()
    if ptoas is None:
        raise FileNotFoundError("The backend audit requires the pinned PTOAS executable")
    summary: dict[str, Any] = {"platform": args.platform, "ptoas": check_ptoas_version(ptoas), "cases": []}
    selected = [
        (entry, inputs, golden)
        for entry, inputs, golden in cases()
        if not args.case or entry.__name__ in args.case
    ]
    if not selected:
        raise ValueError(f"No audit cases match {args.case}")
    passed = True
    for entry, inputs, golden in selected:
        for planner in (MemoryPlanner.PYPTO, MemoryPlanner.PTOAS):
            name = f"{entry.__name__}-{planner.name.lower()}"
            root = output / name
            work = root / "work"
            config = RunConfig(
                backend_type=BackendType.Ascend910B if args.platform == "a2a3" else BackendType.Ascend950,
                execution_mode=ExecutionMode.ONBOARD,
                device_id=args.device,
                memory_planner=planner,
                save_kernels=True,
                save_kernels_dir=str(work),
            )
            backends = {}
            for backend in ("emitc", "vpto"):
                backends[backend] = compile_backend(
                    entry, backend, config, work, root / backend, inputs, golden, args.execute
                )
            comparison = compare_artifacts(root)
            ok = all(record["passed"] for record in backends.values())
            ok = ok and comparison["same_nonempty_pto"]
            ok = ok and comparison["same_nonempty_kernel_inventory"]
            record = {"name": name, "passed": ok, "backends": backends, **comparison}
            if all(item["executed"] for item in backends.values()):
                record["numerically_exact_equal"] = (
                    backends["emitc"]["output_sha256"] == backends["vpto"]["output_sha256"]
                )
            summary["cases"].append(record)
            passed = passed and ok
            (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(f"{name}: {'PASS' if ok else 'FAIL'} {comparison['kernels']}", flush=True)
    summary["passed"] = passed
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
