# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Hardware evidence must name the right ELF region and match actual bytes."""

import hashlib
import struct
from pathlib import Path

import pytest
from pypto.runtime._hardware_loop_artifact import (
    lower_counters,
    requirements,
    validate_artifact,
)
from pypto.runtime._vpto_sync import validate_ordered_hardware_loops

LOOPS = [{"loop_id": "helper/loop0", "kernel": "helper", "trip_count": 16}]
ASSEMBLY = "0000000000001000 <helper.vector.thread>:\n  1000: c100171e VLOOPV2_V310 S2, #5, #1, #1\n"


def image(count=1):
    data = bytearray(832)
    data[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<HH", data, 16, 2, 0x1029)
    struct.pack_into("<Q", data, 24, 0x1000)
    struct.pack_into("<Q", data, 40, 512)
    struct.pack_into("<HHH", data, 58, 64, 5, 1)
    names = b"\0.shstrtab\0.text\0.symtab\0.strtab\0"
    data[64 : 64 + len(names)] = names
    code = struct.pack("<I", 0xC100171E) * count + struct.pack("<I", 0xC0000017)
    data[128 : 128 + len(code)] = code
    symbols = b"\0helper.vector.thread\0"
    data[320 : 320 + len(symbols)] = symbols
    struct.pack_into("<IBBHQQ", data, 280, 1, 2, 0, 2, 0x1000, len(code))
    for index, name, kind, flags, address, offset, size, link, stride in (
        (1, b".shstrtab", 3, 0, 0, 64, len(names), 0, 0),
        (2, b".text", 1, 6, 0x1000, 128, len(code), 0, 0),
        (3, b".symtab", 2, 0, 0, 256, 48, 4, 24),
        (4, b".strtab", 3, 0, 0, 320, len(symbols), 0, 0),
    ):
        struct.pack_into(
            "<IIQQQQIIQQ",
            data,
            512 + index * 64,
            names.index(name),
            kind,
            flags,
            address,
            offset,
            size,
            link,
            0,
            1,
            stride,
        )
    return bytes(data)


def test_verified_mapping_binds_final_image_digest_and_instruction():
    binary = image()
    evidence = validate_artifact(binary, ASSEMBLY, LOOPS, "ptoas 0.70")
    assert evidence["artifact_digest"] == hashlib.sha256(binary).hexdigest()
    assert evidence["loops"][0]["status"] == "verified"
    assert evidence["loops"][0]["region_id"] == "helper.vector.thread"


@pytest.mark.parametrize(
    "assembly",
    [
        "",
        ASSEMBLY.replace("VLOOPV2_V310", "NOP"),
        ASSEMBLY.replace("helper.vector.thread", "other.vector.thread"),
        ASSEMBLY + ASSEMBLY,
    ],
)
def test_missing_unrolled_wrong_kernel_and_ambiguous_evidence_fail(assembly):
    with pytest.raises(ValueError, match="evidence missing or ambiguous"):
        validate_artifact(image(), assembly, LOOPS, "ptoas 0.70")


def test_stale_disassembly_and_duplicate_mapping_fail():
    with pytest.raises(ValueError, match="artifact mismatch"):
        validate_artifact(image(), ASSEMBLY.replace("c100171e", "c100171f"), LOOPS, "ptoas 0.70")
    with pytest.raises(ValueError, match="ID mismatch or duplicate"):
        validate_artifact(image(), ASSEMBLY, LOOPS * 2, "ptoas 0.70")


def test_mapping_schema_and_source_obligation():
    source = "// pypto.hardware_loop.v2 helper/loop0 16\nscf.for"
    assert requirements(source) == LOOPS
    for invalid in (source + "\n" + source, source.replace("v2", "v3"), source.replace("scf.for", "")):
        with pytest.raises(ValueError):
            requirements(invalid)


@pytest.mark.parametrize("count", [1, 4])
def test_dynamic_nested_counter_lowering(count):
    fixture_name = "hardware_loop_dynamic.ll" if count == 1 else "hardware_loop_nested4.ll"
    fixture = Path(__file__).parents[1] / "backend/vpto_vf_validation" / fixture_name
    source = "\n".join(f"// pypto.hardware_loop.v2 helper/loop{i} dynamic" for i in range(count))
    source += "\n" + "scf.for\n" * count
    loops = requirements(source)
    assert len(loops) == count
    lowered = lower_counters(fixture.read_text(), loops)
    assert lowered.count("icmp ult i16") == count
    assert lowered.count("phi i16") == count
    assert "llvm.hivm.mem.bar.vv.all()" in lowered
    validate_ordered_hardware_loops(lowered, count)
    with pytest.raises(ValueError, match="ordering"):
        validate_ordered_hardware_loops(
            lowered.replace("  call void @llvm.hivm.mem.bar.vv.all()\n", "", 1), count
        )


@pytest.mark.parametrize("bound", [16, 32768, 65535])
def test_ordered_counter_lowering_preserves_unsigned_bounds(bound):
    fixture = Path(__file__).parents[1] / "backend/vpto_vf_validation/hardware_loop.ll"
    llvm = fixture.read_text().replace("i16", "i64").replace("%13, 16", f"%13, {bound}")
    lowered = lower_counters(llvm, [{**LOOPS[0], "trip_count": bound}])
    assert f"icmp ult i16 %13, {bound}" in lowered
    assert lowered.index("source_filename") < lowered.index("declare void @llvm.hivm.mem.bar.vv.all()")
    validate_ordered_hardware_loops(lowered, 1)


def test_four_loop_evidence_cannot_borrow_a_missing_instruction():
    loops = [{**LOOPS[0], "loop_id": f"helper/loop{i}"} for i in range(4)]
    assembly = "0000000000001000 <helper.vector.thread>:\n" + "".join(
        f"  {0x1000 + i * 4:x}: c100171e VLOOPV2_V310 S2, #5, #1, #1\n" for i in range(4)
    )
    with pytest.raises(ValueError, match="evidence missing or ambiguous"):
        validate_artifact(image(), ASSEMBLY + (ASSEMBLY.split("\n")[1] + "\n") * 3, loops, "ptoas 0.70")
    evidence = validate_artifact(image(4), assembly, loops, "ptoas 0.70")
    assert len(evidence["loops"]) == 4
    assert len({loop["instruction_id"] for loop in evidence["loops"]}) == 4
    with pytest.raises(ValueError, match="evidence missing"):
        validate_artifact(image(4), assembly.rsplit("\n", 2)[0] + "\n", loops, "ptoas 0.70")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
