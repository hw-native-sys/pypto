# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Probe per-thread packed values with PTOAS and CANN, without reserving a device.

Opt in with PYPTO_RUN_VPTO_VF_VALIDATION=1. The standalone --probe command
retains all sources, LLVM, diagnostics and device objects for capability review.
This gate establishes compiler support, not a public PyPTO API or device numerics.
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import pytest
from pypto.backend._ptoas_locate import check_ptoas_version, find_ptoas_binary
from pypto.runtime._vpto import DEVICE_FLAGS, prepare_llvm_for_bisheng
from pypto.runtime.elf_parser import extract_text_section


@dataclass(frozen=True)
class Case:
    operation: str
    dtype: str
    space: str = "gm"
    target: str = ""
    rounding: str = "r"
    consume: bool = False
    width: int = 2

    @property
    def name(self) -> str:
        fields = [self.operation, self.dtype, self.space]
        if self.target:
            fields.extend([self.target, self.rounding])
        if self.operation.startswith("atomic_"):
            fields.append("old" if self.consume else "discard")
        if self.width != 2:
            fields.append(str(self.width))
        return "-".join(fields)


def cases() -> list[Case]:
    result = [
        Case(op, dtype, space)
        for op in ("copy", "construct", "add", "mul")
        for dtype in ("f16", "bf16", "f32")
        for space in ("gm", "ub")
    ]
    result += [Case(op, dtype) for op in ("absf", "sqrt") for dtype in ("f16", "bf16", "f32")]
    result += [
        Case("ftof", source, target=target, rounding=rounding)
        for source, target in (("f16", "f32"), ("bf16", "f32"), ("f32", "f16"), ("f32", "bf16"))
        for rounding in ("r", "z", "a", "f")
    ]
    result += [
        Case(op, dtype)
        for op in ("shuffle_idx", "shuffle_up", "shuffle_down", "shuffle_bfly")
        for dtype in ("f16", "bf16", "f32")
    ]
    result += [
        Case(op, dtype, space, consume=consume)
        for op in ("atomic_add", "atomic_min", "atomic_max", "atomic_exch", "atomic_cas")
        for dtype in ("f16", "bf16")
        for space in ("gm", "ub")
        for consume in (False, True)
    ]
    result += [
        Case("copy", dtype, space, width=4) for dtype in ("f16", "bf16", "f32") for space in ("gm", "ub")
    ]
    return result


def rejection(case: Case) -> str | None:
    """Expected PTOAS v0.74 diagnostic; every other failure is a regression."""
    if case.operation == "construct":
        return "Dialect `vector' not found for custom op 'vector.extract'"
    if case.operation == "sqrt" and case.dtype != "f16":
        return "'pto.sqrt' op operand #0 must be f16, f32 or vector<2xf16>"
    if case.operation.startswith("shuffle_") and case.dtype != "f16":
        return f"'pto.{case.operation}' op operand #0 must be i32, i64, f16, f32 or vector<2xf16>"
    if case.operation.startswith("atomic_") and case.consume:
        return "does not support using the old value result for packed atomics on beta.1"
    if case.width == 4 and case.space == "gm":
        return "'pto.ldg' op currently supports 8/16/32/64-bit integer"
    return None


def source(case: Case) -> str:
    value_type = f"vector<{case.width}x{case.dtype}>"
    output_type = f"vector<{case.width}x{case.target or case.dtype}>"
    pointer = f"!pto.ptr<{value_type}, {case.space}>"
    output_pointer = f"!pto.ptr<{output_type}, {case.space}>"
    load, store = ("ldg", "stg") if case.space == "gm" else ("load", "store")
    # Offset 1 is one packed element, not one scalar element. It preserves
    # natural alignment and makes the address-unit contract visible in LLVM.
    lines = [
        "%offset = arith.constant 1 : index",
        f"%value = pto.{load} %src[%offset] : {pointer} -> {value_type}",
    ]
    result = "%result"
    if case.operation == "copy":
        result = "%value"
    elif case.operation == "construct":
        lines += [
            f"%first = vector.extract %value[0] : {case.dtype} from {value_type}",
            f"%second = vector.extract %value[1] : {case.dtype} from {value_type}",
            f"%result = vector.from_elements %second, %first : {value_type}",
        ]
    elif case.operation in ("add", "mul"):
        lines += [f"%result = arith.{case.operation}f %value, %value : {value_type}"]
    elif case.operation == "ftof":
        lines += [f"%result = pto.ftof %value round({case.rounding}) : {value_type} -> {output_type}"]
    elif case.operation.startswith("shuffle_"):
        lines += [
            "%lane = arith.constant 1 : i32",
            f"%result = pto.{case.operation} %value, %lane : {value_type}, i32 -> {value_type}",
        ]
    elif case.operation.startswith("atomic_"):
        values = "%value, %value" if case.operation == "atomic_cas" else "%value"
        lines += [f"%result = pto.{case.operation} %dst, {values} : {pointer}, {value_type} -> {value_type}"]
        # Keep the update observable in both cases; only the old-value use
        # differs, exposing the packed-atomic return limitation independently.
        if not case.consume:
            result = ""
    elif case.operation == "absf":
        lines += [f"%result = pto.absf %value : {value_type}"]
    else:
        lines += [f"%result = pto.{case.operation} %value : {value_type} -> {value_type}"]
    if result:
        lines += [f"pto.{store} {result}, %dst[%offset] : {output_pointer}, {output_type}"]
    body = "\n    ".join([*lines, "return"])
    return f"""module attributes {{pto.target_arch = "a5", pto.backend = "vpto",
    pto.kernel_kind = #pto.kernel_kind<vector>}} {{
  func.func @run_packed_probe(%src: {pointer}, %dst: {output_pointer}) attributes {{pto.aiv}} {{
    %width = arith.constant 32 : i32
    %one = arith.constant 1 : i32
    pto.simt_launch @packed_probe<<<%width, %one, %one>>>(%src, %dst) : ({pointer}, {output_pointer}) -> ()
    pto.barrier #pto.pipe<PIPE_ALL>
    return
  }}
  func.func @packed_probe(%src: {pointer}, %dst: {output_pointer})
      attributes {{pto.simt_entry, pto.simt_max_threads = 32 : i32}} {{
    {body}
  }}
}}
"""


def toolchain() -> tuple[str, Path]:
    ptoas = find_ptoas_binary()
    assert ptoas, "PTOAS must be installed under PTOAS_ROOT or on PATH"
    check_ptoas_version(ptoas)
    compiler = Path(os.environ["ASCEND_HOME_PATH"]) / "bin" / "bisheng"
    assert compiler.is_file(), f"CANN compiler is missing: {compiler}"
    return ptoas, compiler


def probe(case: Case, output: Path, ptoas: str, compiler: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    pto_file, llvm, obj = output / "input.pto", output / "output.ll", output / "device.o"
    pto_file.write_text(source(case))
    commands = {
        "llvm": [
            ptoas,
            str(pto_file),
            "--pto-arch=a5",
            "--pto-backend=vpto",
            "--emit-vpto-llvm-ir",
            "-o",
            str(llvm),
        ],
        "device": [
            str(compiler),
            *DEVICE_FLAGS,
            "-c",
            "-x",
            "ir",
            str(output / "bisheng.ll"),
            "-o",
            str(obj),
        ],
    }
    stages: dict[str, int] = {}
    report = {"case": asdict(case), "name": case.name, "stages": stages}
    for stage, command in commands.items():
        (output / f"{stage}.command.json").write_text(json.dumps(command))
        if stage == "device":
            (output / "bisheng.ll").write_text(prepare_llvm_for_bisheng(llvm.read_text()))
        with subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True
        ) as process:
            try:
                log, _ = process.communicate(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                log, _ = process.communicate()
                (output / f"{stage}.log").write_text(log)
                raise
            returncode = process.returncode
        (output / f"{stage}.log").write_text(log)
        stages[stage] = returncode
        if returncode:
            break
    (output / "result.json").write_text(json.dumps(report, indent=2))
    return report


def check_result(case: Case, output: Path, report: dict) -> None:
    expected = rejection(case)
    if expected:
        assert report["stages"] == {"llvm": 1}, report
        assert expected in (output / "llvm.log").read_text(), report
        return
    assert report["stages"] == {"llvm": 0, "device": 0}, report
    image = (output / "device.o").read_bytes()
    assert image.startswith(b"\x7fELF") and extract_text_section(image), "Missing device code"
    assert b"packed_probe_simt_entry" in image, "Missing SIMT entry symbol"
    llvm = (output / "output.ll").read_text()
    element = {"f16": "half", "bf16": "bfloat", "f32": "float"}[case.dtype]
    address_space = 1 if case.space == "gm" else 6
    # Preserve packed-element addressing; a scalar-sized step would silently
    # overlap neighboring threads' values once the frontend uses this contract.
    assert re.search(
        rf"getelementptr <{case.width} x {element}>, ptr addrspace\({address_space}\) %\w+, i64 1", llvm
    )
    if case.space == "ub":
        alignment = case.width * (4 if case.dtype == "f32" else 2)
        assert re.search(
            rf"load <{case.width} x {element}>, ptr addrspace\(6\) %\w+, align {alignment}\b", llvm
        )


@pytest.mark.parametrize("case", cases(), ids=lambda case: case.name)
def test_packed_native_capability(case, tmp_path):
    if os.environ.get("PYPTO_RUN_VPTO_VF_VALIDATION") != "1":
        pytest.skip("Opt in with PYPTO_RUN_VPTO_VF_VALIDATION=1 and the pinned PTOAS/CANN toolchain")
    ptoas, compiler = toolchain()
    report = probe(case, tmp_path, ptoas, compiler)
    check_result(case, tmp_path, report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    args.probe.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="packed-", dir=args.probe.resolve()))
    ptoas, compiler = toolchain()
    versions = {}
    for name, command in (("ptoas", [ptoas, "--version"]), ("cann", [str(compiler), "--version"])):
        versions[name] = subprocess.run(
            command, capture_output=True, text=True, timeout=30, check=True
        ).stdout
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(lambda case: probe(case, output / case.name, ptoas, compiler), cases()))
    (output / "matrix.json").write_text(json.dumps({"versions": versions, "results": results}, indent=2))
    for case, result in zip(cases(), results, strict=True):
        check_result(case, output / case.name, result)
    accepted = sum(rejection(case) is None for case in cases())
    print(f"{accepted} device objects, {len(results) - accepted} expected rejections; evidence: {output}")


if __name__ == "__main__":
    if "--probe" in sys.argv:
        main()
    else:
        pytest.main([__file__, "-v", *sys.argv[1:]])
