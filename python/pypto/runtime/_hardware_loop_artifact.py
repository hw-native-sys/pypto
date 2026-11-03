# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Map VF hardware loops to real instructions in the linked ELF."""

import hashlib
import re
import struct
from typing import TypedDict

from ._vpto import validate_device_image


class ArtifactEvidence(TypedDict):
    schema_version: int
    artifact_digest: str
    toolchain_version: str
    target: str
    loops: list[dict[str, str | int]]


def requirements(source: str) -> list[dict[str, str | int]]:
    """Read versioned lowering obligations; malformed/duplicate markers fail closed."""
    markers = [line for line in source.splitlines() if "pypto.hardware_loop" in line]
    result: list[dict[str, str | int]] = []
    for line in markers:
        match = re.fullmatch(r"// pypto\.hardware_loop\.v2 ([-a-zA-Z$._0-9]+)/loop(\d+) (\d+|dynamic)", line)
        if match is None:
            raise ValueError("hardware loop mapping has an unsupported schema or bound")
        bound = int(match[3]) if match[3] != "dynamic" else "dynamic"
        if isinstance(bound, int) and not 0 <= bound <= 65535:
            raise ValueError("hardware loop mapping has an unsupported bound")
        result.append({"loop_id": f"{match[1]}/loop{match[2]}", "kernel": match[1], "trip_count": bound})
    ids = [str(loop["loop_id"]) for loop in result]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate hardware loop mapping")
    if result and len(re.findall(r"\bscf\.for\b", source)) != len(result):
        raise ValueError("hardware loop mapping must describe exactly every source loop")
    return result


def lower_counters(llvm: str, loops: list[dict[str, str | int]]) -> str:
    """Lower canonical counters, preserving IV uses with unsigned extension.

    Memory accesses receive full VF ordering in this general path. This is
    conservative, independent of dynamic bounds and alias relationships.
    """
    ssa = r"%[-a-zA-Z$._0-9]+"
    compares = list(
        re.finditer(rf"(?m)^([ \t]*({ssa}) = )icmp (slt|ult) i(16|32|64) ({ssa}), ({ssa}|-?\d+)$", llvm)
    )
    candidates = [match for match in compares if match[6] != "1"]
    if len(candidates) != len(loops):
        raise ValueError("hardware loop mapping/LLVM counter count mismatch")
    replacements: dict[str, str] = {}
    for index, (match, loop) in enumerate(zip(candidates, loops)):
        prefix, width, counter, bound = match[1], match[4], match[5], match[6]
        expected = loop["trip_count"]
        if expected != "dynamic" and bound != str(expected):
            raise ValueError("hardware loop bound/mapping mismatch")
        phi = re.search(rf"(?m)^[ \t]*{re.escape(counter)} = phi i{width} (.*)$", llvm)
        increment = re.search(rf"(?m)^([ \t]*({ssa}) = add )i{width} {re.escape(counter)}, 1$", llvm)
        if (
            phi is None
            or increment is None
            or not re.fullmatch(rf"\[ {ssa}, {ssa} \], \[ 0, {ssa} \]", phi[1])
        ):
            raise ValueError("hardware loop requires canonical zero-start unit-step recurrence")
        stem = f"%pypto_hwl{index}"
        bound_setup = ""
        if bound.startswith("%"):
            if match[3] == "slt":
                bound_setup = f"  {stem}_positive = icmp sgt i{width} {bound}, 0\n"
                bound_setup += (
                    f"  {stem}_nonnegative = select i1 {stem}_positive, i{width} {bound}, i{width} 0\n"
                )
                bound = stem + "_nonnegative"
            if width != "16":
                bound_setup += f"  {stem}_bound = trunc i{width} {bound} to i16\n"
                bound = stem + "_bound"
        extended = stem + "_iv"
        replacements[phi[0]] = phi[0].replace(f"phi i{width}", "phi i16")
        replacements[increment[0]] = increment[0].replace(f"add i{width}", "add i16")
        replacements[match[0]] = (
            (f"  {extended} = zext i16 {counter} to i{width}\n" if width != "16" else "")
            + bound_setup
            + f"{prefix}icmp ult i16 {counter}, {bound}"
        )
        # Other uses are ordinary address/value computations; never truncate them.
        if width != "16":
            pattern = rf"{re.escape(counter)}(?![-a-zA-Z$._0-9])"
            excluded = {phi[0], increment[0], match[0]}
            llvm = "\n".join(
                line if line in excluded else re.sub(pattern, extended, line) for line in llvm.split("\n")
            )
    for old, new in replacements.items():
        if old not in llvm:
            raise ValueError("overlapping hardware counter rewrite")
        llvm = llvm.replace(old, new, 1)
    barrier = "  call void @llvm.hivm.mem.bar.vv.all()"
    ordered: list[str] = []
    for line in llvm.splitlines():
        memory = "call " in line and re.search(r"@llvm\.hivm\.v(?:ld|st)", line)
        if memory:
            ordered.append(barrier)
        ordered.append(line)
        if memory:
            ordered.append(barrier)
    llvm = "\n".join(ordered) + "\n"
    if "declare void @llvm.hivm.mem.bar.vv.all()" not in llvm:
        # LLVM's source_filename/target directives must precede declarations.
        llvm += "\ndeclare void @llvm.hivm.mem.bar.vv.all()\n"
    return llvm


def _vector_symbols(binary: bytes) -> dict[str, tuple[int, bytes]]:
    table = struct.unpack_from("<Q", binary, 40)[0]
    stride, count = struct.unpack_from("<HH", binary, 58)
    sections: list[tuple[int, ...]] = [
        struct.unpack_from("<IIQQQQIIQQ", binary, table + i * stride) for i in range(count)
    ]
    result: dict[str, tuple[int, bytes]] = {}
    for section in sections:
        if section[1] != 2:
            continue
        strings = sections[section[6]]
        names = binary[strings[4] : strings[4] + strings[5]]
        for offset in range(section[4], section[4] + section[5], section[9]):
            name, _, _, raw_index, raw_address, raw_size = struct.unpack_from("<IBBHQQ", binary, offset)
            index, address, size = int(raw_index), int(raw_address), int(raw_size)
            symbol = names[name:].split(b"\0", 1)[0].decode("utf-8")
            if not symbol.endswith(".vector.thread"):
                continue
            if not index or index >= len(sections) or not size or symbol in result:
                raise ValueError("hardware loop artifact has an invalid/duplicate vector symbol")
            code = sections[index]
            start = code[4] + address - code[3]
            if address < code[3] or start + size > code[4] + code[5]:
                raise ValueError("hardware loop vector symbol exceeds its ELF section")
            result[symbol] = address, binary[start : start + size]
    return result


def validate_artifact(
    binary: bytes, disassembly: str, loops: list[dict[str, str | int]], toolchain_version: str
) -> ArtifactEvidence:
    """Require a unique mapped hardware instruction and check its actual ELF bytes.

    The compiler has already admitted canonical source/LLVM loops for each
    helper. A symbol-specific check therefore cannot borrow another helper's
    loop, the outer AIV marker, scalar branches, or an unrolled body as evidence.
    """
    validate_device_image(binary)
    symbols = _vector_symbols(binary)
    regions: dict[str, list[tuple[int, str]]] = {}
    current = ""
    for line in disassembly.splitlines():
        if match := re.fullmatch(r"[0-9a-fA-F]+ <([^>]+)>:", line.strip()):
            current = match[1]
            if current in regions:
                raise ValueError("hardware loop evidence missing or ambiguous: duplicate symbol")
            regions[current] = []
        elif match := re.fullmatch(r"\s*([0-9a-fA-F]+):\s+([0-9a-fA-F]{8})\s+VLOOPV2_V310\s+.*", line):
            regions.setdefault(current, []).append((int(match[1], 16), match[2]))
    ids = [str(loop["loop_id"]) for loop in loops]
    if len(set(ids)) != len(ids):
        raise ValueError("hardware loop mapping ID mismatch or duplicate")
    seen: set[str] = set()
    verified: list[dict[str, str | int]] = []
    positions: dict[str, int] = {}
    counts: dict[str, int] = {}
    for loop in loops:
        key = str(loop["kernel"])
        counts[key] = counts.get(key, 0) + 1
    for loop in loops:
        loop_id, kernel = str(loop["loop_id"]), str(loop["kernel"])
        index = positions.get(kernel, 0)
        positions[kernel] = index + 1
        if loop_id in seen or loop_id != f"{kernel}/loop{index}":
            raise ValueError("hardware loop mapping ID mismatch or duplicate")
        seen.add(loop_id)
        symbol = kernel + ".vector.thread"
        instructions = regions.get(symbol, [])
        if (
            symbol not in symbols
            or len(instructions) != counts[kernel]
            or len({address for address, _ in instructions}) != len(instructions)
        ):
            raise ValueError(f"hardware loop evidence missing or ambiguous for {loop_id}")
        start, code = symbols[symbol]
        address, word = instructions[index]
        offset = address - start
        if (
            offset < 0
            or offset + 4 > len(code)
            or code[offset : offset + 4] != int(word, 16).to_bytes(4, "little")
        ):
            raise ValueError(f"hardware loop disassembly/artifact mismatch for {loop_id}")
        verified.append(
            {
                **loop,
                "region_id": symbol + (f"/loop{index}" if counts[kernel] > 1 else ""),
                "instruction_id": hex(address),
                "instruction": "VLOOPV2_V310",
                "status": "verified",
            }
        )
    return {
        "schema_version": 1,
        "artifact_digest": hashlib.sha256(binary).hexdigest(),
        "toolchain_version": toolchain_version,
        "target": "a5/dav-c310-vec",
        "loops": verified,
    }
