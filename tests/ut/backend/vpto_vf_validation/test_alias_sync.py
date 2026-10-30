# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Opt-in separate-helper alias/synchronization acceptance against installed PTOAS.

These compiler probes deliberately expose the static address relation in VMI.
They do not claim that production PyPTO already specializes captured Tile views.
"""

import importlib
import json
import os
import shutil
from pathlib import Path

import pytest
from pypto.runtime import _vpto, _vpto_sync
from test_validation import ROOT, run

pytestmark = pytest.mark.skipif(
    os.environ.get("PYPTO_RUN_VPTO_ALIAS_VALIDATION") != "1",
    reason="Explicitly opt in to the installed A5 compiler toolchain",
)


def helper_source(kind: str, offset: int, relation: str = "static") -> str:
    """Build observable WR/RW/WW accesses, with a disjoint result at element 256."""
    ptr = "!pto.ptr<f32, ub>"
    reg = "!pto.vmi.vreg<64xf32>"
    mask = "!pto.vmi.mask<64xpred>"
    args = f"%base: {ptr}"
    second_base, second_offset = "%base", "%offset"
    setup = f"%offset = arith.constant {offset} : index"
    if relation == "unknown":
        args += f", %other: {ptr}"
        second_base, second_offset = "%other", "%zero"
    elif relation == "dynamic":
        args += ", %offset: index"
        setup = ""
    elif relation == "derived":
        setup += "\n    %other = pto.addptr %base, %offset : <f32, ub> -> <f32, ub>"
        second_base, second_offset = "%other", "%zero"

    def store(value: str, base: str, index: str) -> str:
        return f"pto.vmi.vstore {value}, {base}[{index}], %mask : {reg}, {ptr}, {mask}"

    def load(base: str, index: str) -> str:
        return f"%loaded = pto.vmi.vload {base}[{index}] : {ptr} -> {reg}"

    operations = {
        "wr": [
            store("%five", "%base", "%zero"),
            load(second_base, second_offset),
            store("%loaded", "%base", "%result"),
        ],
        "rw": [
            load("%base", "%zero"),
            store("%five", second_base, second_offset),
            store("%loaded", "%base", "%result"),
        ],
        "ww": [store("%five", "%base", "%zero"), store("%seven", second_base, second_offset)],
    }[kind]
    body = "\n      ".join(operations)
    return f"""module attributes {{pto.target_arch = "a5", pto.backend = "vpto"}} {{
  func.func @alias_helper({args}) attributes {{pto.aiv}} {{
    %zero = arith.constant 0 : index
    %result = arith.constant 256 : index
    {setup}
    %f5 = arith.constant 5.0 : f32
    %f7 = arith.constant 7.0 : f32
    pto.vecscope {{
      %mask = pto.vmi.pset "PAT_ALL" : {mask}
      %five = pto.vmi.vbrc %f5 : f32 -> {reg}
      %seven = pto.vmi.vbrc %f7 : f32 -> {reg}
      {body}
    }}
    return
  }}
}}
"""


def compile_helper(directory: Path, source: str, *, enabled: bool = True) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    ptoas = shutil.which("ptoas")
    assert ptoas, "PTOAS must be on PATH"
    (directory / "helper.pto").write_text(source)
    run(directory, "version", [ptoas, "--version"])
    flags = [*_vpto.PTOAS_FLAGS, f"--enable-vecscope-mem-bar={'true' if enabled else 'false'}"]
    run(
        directory,
        "lower",
        [ptoas, directory / "helper.pto", *flags, "--mlir-print-ir-after-all", "-o", directory / "helper.ll"],
    )
    llvm = (directory / "helper.ll").read_text()
    calls = [line.strip() for line in llvm.splitlines() if "call " in line and "bar" in line.lower()]
    report = {"barrier_calls": calls, "enabled": enabled}
    (directory / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


@pytest.mark.parametrize("relation", ["static", "derived"])
@pytest.mark.parametrize("kind", ["wr", "rw", "ww"])
@pytest.mark.parametrize("offset", [0, 32, 64, 128])
def test_static_alias_sync(kind: str, offset: int, relation: str):
    output = Path(os.environ.get("PYPTO_VPTO_ALIAS_OUTPUT", ROOT / "build/vf-alias-validation"))
    directory = (output / f"{kind}-{relation}-{offset}").resolve()
    report = compile_helper(directory, helper_source(kind, offset, relation))
    expected = 1 if offset < 64 else 0
    assert len(report["barrier_calls"]) == expected, report
    log = (directory / "lower.log").read_text()
    assert "PTOInsertVecScopeMemBar" in log
    _vpto_sync.validate_sync_contract((directory / "helper.ll").read_text(), str(directory / "helper.pto"))


@pytest.mark.parametrize("kind", ["wr", "rw", "ww"])
def test_disabled_auto_sync_control(kind: str):
    output = Path(os.environ.get("PYPTO_VPTO_ALIAS_OUTPUT", ROOT / "build/vf-alias-validation"))
    directory = (output / f"{kind}-disabled-32").resolve()
    report = compile_helper(directory, helper_source(kind, 32), enabled=False)
    assert not report["barrier_calls"]
    with pytest.raises(ValueError, match="missing .* barrier"):
        _vpto_sync.validate_sync_contract(
            (directory / "helper.ll").read_text(), str(directory / "helper.pto")
        )


@pytest.mark.parametrize("relation", ["unknown", "dynamic"])
@pytest.mark.parametrize("kind", ["wr", "rw", "ww"])
def test_unresolved_relation_is_not_rejected_by_ptoas(kind: str, relation: str):
    """Characterize the gap: successful compilation is NOT safe acceptance."""
    output = Path(os.environ.get("PYPTO_VPTO_ALIAS_OUTPUT", ROOT / "build/vf-alias-validation"))
    directory = (output / f"{kind}-{relation}").resolve()
    if relation == "dynamic" and kind in ("rw", "ww"):
        with pytest.raises(AssertionError, match="proven store alignment"):
            compile_helper(directory, helper_source(kind, 32, relation))
        return
    report = compile_helper(directory, helper_source(kind, 32, relation))
    assert not report["barrier_calls"], report
    (directory / "acceptance-gap.json").write_text(
        json.dumps(
            {
                "compiler_accepted": True,
                "required_contract": "reject unresolved relation",
                "contract_satisfied": False,
                "hardware_executed": False,
            },
            indent=2,
        )
        + "\n"
    )


@pytest.mark.parametrize("relation", ["unknown", "dynamic"])
@pytest.mark.parametrize("kind", ["wr", "rw", "ww"])
def test_production_compiler_rejects_unresolved_relationship(monkeypatch, kind, relation):
    compiler_class = importlib.import_module("pypto.runtime.kernel_compiler").KernelCompiler

    output = Path(os.environ.get("PYPTO_VPTO_ALIAS_OUTPUT", ROOT / "build/vf-alias-validation"))
    directory = (output / f"production-reject-{kind}-{relation}").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "helper.pto"
    source.write_text(helper_source(kind, 32, relation))
    compiler = compiler_class("a5")
    original = compiler._run
    commands = []

    def record(command, output, phase):
        commands.append(command)
        return original(command, output, phase)

    monkeypatch.setattr(compiler, "_run", record)
    with pytest.raises((ValueError, RuntimeError)) as error:
        compiler._compile_vpto_helpers((str(source),), directory)
    message = str(error.value)
    if relation == "dynamic" and kind in ("rw", "ww"):
        assert "proven store alignment" in message
    else:
        assert "unsupported VF synchronization contract" in message
    assert len(commands) == 1 and Path(commands[0][0]).name == "ptoas"
    assert not list(directory.glob("*.o"))
    (directory / "rejection.json").write_text(
        json.dumps({"error": message, "commands": commands, "device_compilation_started": False}, indent=2)
        + "\n"
    )


def masked_helper_source(kind: str, offset: int, mode: str) -> str:
    """Keep saved loads unmasked so every observed lane reaches the host oracle."""
    source = helper_source(kind, offset, "derived")
    mtype = "!pto.vmi.mask<64xpred>"
    itype = "!pto.vmi.vreg<64xsi32>"
    setup = ""
    if mode == "dynamic":
        source = source.replace("%base: !pto.ptr<f32, ub>)", "%base: !pto.ptr<f32, ub>, %n: i32)")
        setup = "%limit = builtin.unrealized_conversion_cast %n : i32 to si32"
    else:
        count = int(mode.removeprefix("prefix")) if mode.startswith("prefix") else 32
        setup = (
            f"%count = arith.constant {count} : i32\n"
            "    %limit = builtin.unrealized_conversion_cast %count : i32 to si32"
        )
    setup += "\n    %lane_zero = arith.constant 0 : i32"
    source = source.replace("    pto.vecscope {", setup + "\n    pto.vecscope {")
    masks = f"""%lanes = pto.vmi.vci %lane_zero {{order = "ASC"}} : i32 -> {itype}
      %limits = pto.vmi.vbrc %limit : si32 -> {itype}
      %prefix = pto.vmi.vcmp %lanes, %limits, %mask {{cmp = "lt"}} : {itype}, {itype}, {mtype} -> {mtype}
      %suffix = pto.vmi.vcmp %lanes, %limits, %mask {{cmp = "ge"}} : {itype}, {itype}, {mtype} -> {mtype}"""
    source = source.replace(
        f'%mask = pto.vmi.pset "PAT_ALL" : {mtype}',
        f'%mask = pto.vmi.pset "PAT_ALL" : {mtype}\n      {masks}',
    )
    lines = []
    for line in source.splitlines():
        if "pto.vmi.vstore %five" in line:
            line = line.replace(", %mask :", ", %prefix :")
        elif "pto.vmi.vstore %seven" in line:
            line = line.replace(", %mask :", ", %suffix :" if mode == "split" else ", %prefix :")
        lines.append(line)
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("mode", ["prefix0", "prefix32", "dynamic", "split"])
@pytest.mark.parametrize("offset", [0, 32, 64])
@pytest.mark.parametrize("kind", ["wr", "rw", "ww"])
def test_masked_alias_compiler_contract(kind, offset, mode):
    output = Path(os.environ.get("PYPTO_VPTO_ALIAS_OUTPUT", ROOT / "build/vf-masked-alias/compiler"))
    directory = (output / f"masked-{kind}-{offset}-{mode}").resolve()
    report = compile_helper(directory, masked_helper_source(kind, offset, mode))
    assert len(report["barrier_calls"]) == (1 if offset < 64 else 0), report
    _vpto_sync.validate_sync_contract((directory / "helper.ll").read_text(), str(directory / "helper.pto"))


def interleaved_helper_source(offset: int) -> str:
    source = masked_helper_source("wr", offset, "dynamic")
    ptr, reg, mask = "!pto.ptr<f32, ub>", "!pto.vmi.vreg<64xf32>", "!pto.vmi.mask<64xpred>"

    def load(name, base):
        return f"%{name} = pto.vmi.vload %{base}[%zero] : {ptr} -> {reg}"

    def store(value, base, index, predicate):
        return f"pto.vmi.vstore %{value}, %{base}[%{index}], %{predicate} : {reg}, {ptr}, {mask}"

    original = "\n      ".join(
        (
            store("five", "base", "zero", "prefix"),
            load("loaded", "other"),
            store("loaded", "base", "result", "mask"),
        )
    )
    replacement = "\n      ".join(
        (
            load("old", "other"),
            store("five", "base", "zero", "prefix"),
            load("middle", "other"),
            store("seven", "other", "zero", "prefix"),
            load("after", "base"),
            store("seven", "base", "zero", "suffix"),
            store("old", "base", "result", "mask"),
            store("middle", "base", "result2", "mask"),
            store("after", "base", "result3", "mask"),
        )
    )
    assert original in source
    return source.replace(original, replacement).replace(
        "    %result =",
        "    %result2 = arith.constant 384 : index\n    %result3 = arith.constant 512 : index\n    %result =",
    )


@pytest.mark.parametrize("offset", [8, 32, 64])
def test_interleaved_masked_barriers_and_missing_edge_rejection(offset):
    output = Path(os.environ.get("PYPTO_VPTO_ALIAS_OUTPUT", ROOT / "build/vf-masked-alias/compiler"))
    directory = (output / f"interleaved-{offset}").resolve()
    report = compile_helper(directory, interleaved_helper_source(offset))
    assert len(report["barrier_calls"]) == (5 if offset < 64 else 3)
    llvm = (directory / "helper.ll").read_text()
    _vpto_sync.validate_sync_contract(llvm, "interleaved.pto")
    lines = llvm.splitlines()
    rejected = []
    for i, line in enumerate(lines):
        if "call void @llvm.hivm.mem.bar" not in line:
            continue
        damaged = "\n".join(lines[:i] + lines[i + 1 :])
        with pytest.raises(ValueError, match="missing .* barrier") as error:
            _vpto_sync.validate_sync_contract(damaged, "removed-barrier.pto")
        rejected.append({"removed": line.strip(), "error": str(error.value)})
    (directory / "removed-barriers.json").write_text(json.dumps(rejected, indent=2) + "\n")


@pytest.mark.parametrize("mode", ["prefix32", "dynamic", "split"])
@pytest.mark.parametrize("kind", ["wr", "rw", "ww"])
def test_masked_full_footprint_policy_rejects_disabled_sync(kind, mode):
    output = Path(os.environ.get("PYPTO_VPTO_ALIAS_OUTPUT", ROOT / "build/vf-masked-alias/compiler"))
    directory = (output / f"disabled-masked-{kind}-{mode}").resolve()
    report = compile_helper(directory, masked_helper_source(kind, 0, mode), enabled=False)
    assert not report["barrier_calls"]
    # Complementary-mask cases document the current full-footprint policy,
    # not a claim that their active lanes really overlap.
    with pytest.raises(ValueError, match="missing .* barrier"):
        _vpto_sync.validate_sync_contract((directory / "helper.ll").read_text(), "disabled.pto")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
