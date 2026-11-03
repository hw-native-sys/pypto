# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Build contract for separately compiled, self-contained A5 VPTO helpers."""

import re
import struct
from collections.abc import Sequence
from pathlib import Path
from typing import cast

# PTOAS owns the helper's Bisheng profile, address spaces and scheduling.
# Export inspection LLVM separately to retain the existing synchronization gate.
PTOAS_FLAGS = (
    "--pto-arch=a5",
    "--pto-backend=vpto",
    "--vmi-fastmath=false",
    "--enable-vecscope-mem-bar=true",
)
LLVM_FLAGS = (*PTOAS_FLAGS, "--emit-vpto-llvm-ir")
OBJECT_ABI_VERSION = "ptoas-native-aiv-v1"
DEVICE_SECTION = "__aicore_rel_binary"
# Tile operations can leave out-of-line functions before kernel_entry in the
# caller object. Separate their sections so the linker can put the entry first;
# the persistent loader starts at byte zero of the final .text payload.
CALLER_FLAGS = ("-ffunction-sections",)
LINK_SYMBOL_ORDER = ("kernel_entry",)


def _device_object_sections(data: bytes) -> list[tuple[int, ...]]:
    """Reject incompatible or concatenated ELF payloads before reading exports."""
    if len(data) < 64 or data[:6] != b"\x7fELF\x02\x01":
        raise ValueError("VPTO helper requires a little-endian ELF64 device object")
    if struct.unpack_from("<HH", data, 16) != (1, 0x1029):
        raise ValueError("VPTO helper requires a relocatable AI Core ELF, not a host or launch image")
    (offset,) = struct.unpack_from("<Q", data, 40)
    size, count = struct.unpack_from("<HH", data, 58)
    end = offset + count * size
    if size != 64 or not count or offset < 64 or end > len(data):
        raise ValueError("VPTO helper has an invalid section table")
    sections = [struct.unpack_from("<IIQQQQIIQQ", data, offset + i * size) for i in range(count)]
    for section in sections:
        if section[1] != 8:  # SHT_NOBITS has no file payload.
            end = max(end, section[4] + section[5])
    if end > len(data):
        raise ValueError("VPTO helper has a truncated section")
    if any(data[end:]):
        raise ValueError("VPTO helper contains multiple device objects or unsupported trailing data")
    return sections


def device_object_aliases(data: bytes) -> dict[str, str]:
    """Validate one native AIV object and map its logical exports to device names.

    CANN fatobjs may contain multiple concatenated ELFs. This helper-only path
    requires exactly one relocatable device object; consuming only its first
    ELF would silently drop code. Native exports carry a CANN-dependent suffix.
    Read the actual symbol table instead of inferring a suffix from the version.
    Linker aliases preserve those exports and add no executable trampoline.
    """
    sections = _device_object_sections(data)

    aliases: dict[str, str] = {}
    for _, kind, _, _, start, length, link, _, _, stride in sections:
        if kind != 2:  # SHT_SYMTAB
            continue
        if stride != 24 or length % stride or link >= len(sections) or sections[link][1] != 3:
            raise ValueError("VPTO helper has an invalid symbol table")
        strings = sections[link]
        names = data[strings[4] : strings[4] + strings[5]]
        for position in range(start, start + length, stride):
            name, info, _, section = struct.unpack_from("<IBBH", data, position)
            if info != 0x12 or not section:  # Defined STB_GLOBAL / STT_FUNC only.
                continue
            terminator = names.find(b"\0", name)
            if name >= len(names) or terminator < 0:
                raise ValueError("VPTO helper has an invalid symbol name")
            native = names[name:terminator].decode("ascii")
            suffix = next((s for s in (".vector", "_mix_aiv") if native.endswith(s)), None)
            if suffix is None:
                continue
            logical = native[: -len(suffix)]
            if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", logical) or logical in aliases:
                raise ValueError(f"VPTO helper has an unsupported or ambiguous export: {native!r}")
            aliases[logical] = native
    if not aliases:
        raise ValueError("VPTO helper has no supported native AIV function exports")
    return aliases


def source_paths(value: object) -> tuple[str, ...]:
    """Validate config metadata without requiring the compiler or native SDK."""
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("vpto_sources must be a list of .pto source paths")
    paths: list[str] = []
    for path in cast(Sequence[object], value):
        if not isinstance(path, (str, Path)) or Path(path).suffix != ".pto":
            raise ValueError(f"vpto_sources requires .pto files, got {path!r}")
        paths.append(str(path))
    if len(paths) != len(set(paths)):
        raise ValueError("vpto_sources contains duplicate paths")
    return tuple(paths)


def validate_target(platform: str, core_type: str) -> None:
    """Reject CPU simulation and other device targets before cache lookup."""
    if platform != "a5" or core_type != "aiv":
        raise ValueError(
            f"VPTO helpers require platform='a5' and core_type='aiv', got {platform!r}/{core_type!r}. "
            "The a5sim platform is CPU simulation and cannot execute VPTO device objects."
        )


def validate_device_image(data: bytes) -> None:
    """Reject linked images whose dependencies would be lost by .text extraction.

    The persistent loader starts at byte zero and retains no ELF sections or
    relocation metadata. This structural check does not prove arbitrary code
    position-independent; helper authors must still obey the device-call ABI.
    """
    if len(data) < 64 or data[:6] != b"\x7fELF\x02\x01":
        raise ValueError("VPTO linking requires a little-endian ELF64 device image")
    elf_type, machine = struct.unpack_from("<HH", data, 16)
    if elf_type != 2 or machine != 0x1029:
        raise ValueError("VPTO linking requires an executable AI Core ELF, not a host fatobj")
    (entry,) = struct.unpack_from("<Q", data, 24)
    (offset,) = struct.unpack_from("<Q", data, 40)
    size, count, names_index = cast(tuple[int, int, int], struct.unpack_from("<HHH", data, 58))
    if size != 64 or not count or names_index >= count or offset + count * size > len(data):
        raise ValueError("VPTO linked image has an invalid section table")
    sections: list[tuple[int, ...]] = [
        struct.unpack_from("<IIQQQQIIQQ", data, offset + i * size) for i in range(count)
    ]
    names = sections[names_index]
    strings = data[names[4] : names[4] + names[5]]
    found_text = False
    for name_offset, kind, flags, address, start, length, _, _, _, stride in sections:
        name = strings[name_offset:].split(b"\x00", 1)[0]
        if kind != 8 and start + length > len(data):  # SHT_NOBITS has no file payload.
            raise ValueError("VPTO linked image has a truncated section")
        if name == b".text":
            found_text = bool(length)
            if entry != address:
                raise ValueError("VPTO kernel_entry must be at the start of .text")
        if flags & 2 and length and name != b".text":  # SHF_ALLOC
            raise ValueError(f"VPTO linked image needs unsupported allocated section {name!r}")
        if kind in (4, 9) and length:  # SHT_RELA / SHT_REL
            raise ValueError("VPTO linked image must have no remaining relocations")
        if kind in (2, 11):  # SHT_SYMTAB / SHT_DYNSYM
            if stride != 24 or length % stride:
                raise ValueError("VPTO linked image has an invalid symbol table")
            for position in range(start, start + length, stride):
                symbol, _, _, section = struct.unpack_from("<IBBH", data, position)
                if symbol and section == 0:
                    raise ValueError("VPTO linked image has an unresolved symbol")
    if not found_text:
        raise ValueError("VPTO linked image requires a nonempty .text section")
