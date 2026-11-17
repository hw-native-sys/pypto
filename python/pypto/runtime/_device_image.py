# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Structural admission checks for the persistent runtime's text-only loader."""

import struct
from dataclasses import dataclass
from typing import cast


@dataclass(frozen=True)
class _Section:
    name: int
    kind: int
    flags: int
    address: int
    offset: int
    size: int
    link: int
    info: int
    alignment: int
    stride: int

    def payload(self, data: bytes) -> bytes:
        return data[self.offset : self.offset + self.size]


def _string(table: bytes, offset: int) -> bytes:
    end = table.find(b"\0", offset)
    if offset >= len(table) or end < 0:
        raise ValueError("Device image has an invalid string-table offset or unterminated name")
    return table[offset:end]


def _sections(data: bytes) -> tuple[int, list[_Section], bytes]:
    if len(data) < 64 or data[:6] != b"\x7fELF\x02\x01":
        raise ValueError("Device image requires a little-endian ELF64 device image")
    elf_type, machine = struct.unpack_from("<HH", data, 16)
    if elf_type != 2 or machine != 0x1029:
        raise ValueError("Device image requires an executable AI Core ELF, not a host fatobj")
    (entry,) = struct.unpack_from("<Q", data, 24)
    (offset,) = struct.unpack_from("<Q", data, 40)
    size, count, names_index = cast(tuple[int, int, int], struct.unpack_from("<HHH", data, 58))
    if size != 64 or not count or not 0 < names_index < count or offset + count * size > len(data):
        raise ValueError("Device image has an invalid section table")
    sections = [_Section(*struct.unpack_from("<IIQQQQIIQQ", data, offset + i * size)) for i in range(count)]
    if sections[0] != _Section(0, 0, 0, 0, 0, 0, 0, 0, 0, 0):
        raise ValueError("Device image requires a null section at index 0")
    for section in sections:
        # NOBITS has an address extent but deliberately has no file payload.
        if section.kind != 8 and section.offset + section.size > len(data):
            raise ValueError("Device image has a truncated section")
    names = sections[names_index]
    if names.kind != 3 or not names.size:
        raise ValueError("Device image has an invalid section-name string table")
    return entry, sections, names.payload(data)


def _text_section(entry: int, sections: list[_Section], strings: bytes, *, plain_callee: bool) -> int:
    text_index = 0
    for index, section in enumerate(sections):
        name = _string(strings, section.name)
        # The loader ignores non-ASCII bytes when decoding names. Reject them
        # so it cannot select a different section after dropping those bytes.
        if not name.isascii():
            raise ValueError(f"Device image has a non-ASCII section name {name!r}")
        # Host-launched kernels can carry SU/SIMT stack and UB provisioning
        # metadata. A .su report does not replace that launch contract, and the
        # persistent text loader cannot consume it. Accept only plain callees.
        if plain_callee and name.startswith(b".ascend.meta") and section.size:
            raise ValueError("Device image requires unsupported kernel runtime metadata")
        if name == b".text":
            if text_index:
                raise ValueError("Device image has duplicate .text sections")
            if section.kind != 1 or section.flags != 6 or not section.size:
                raise ValueError("Device image requires a nonempty executable .text section")
            if entry != section.address:
                raise ValueError("Device image kernel_entry must be at the start of .text")
            text_index = index
        elif section.flags & 2 and section.size:
            raise ValueError(f"Device image needs unsupported allocated section {name!r}")
        if section.kind in (4, 9) and section.size:
            raise ValueError("Device image must have no remaining relocations")
    if not text_index:
        raise ValueError("Device image requires a nonempty .text section")
    return text_index


def _function_name(symbol: bytes) -> str:
    try:
        return symbol.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError("Device image has a non-ASCII function symbol") from error


def _functions(
    data: bytes, sections: list[_Section], text_index: int, entry_symbol: str | None
) -> frozenset[frozenset[str]]:
    text = sections[text_index]
    functions: dict[str, tuple[int, int]] = {}
    found_entry = False
    for section in sections:
        if section.kind not in (2, 11):
            continue
        if section.stride != 24 or section.size % 24:
            raise ValueError("Device image has an invalid symbol table")
        symbol_strings = b""
        if entry_symbol is not None:
            if not 0 < section.link < len(sections) or sections[section.link].kind != 3:
                raise ValueError("Device image has an invalid symbol string table")
            symbol_strings = sections[section.link].payload(data)
        for position in range(section.offset, section.offset + section.size, 24):
            name, info, _, defined_in, address, length = struct.unpack_from("<IBBHQQ", data, position)
            if name and defined_in == 0:
                raise ValueError("Device image has an unresolved symbol")
            if entry_symbol is None:
                continue
            symbol = _string(symbol_strings, name)
            # STT_FUNC gives the compiler's mangled name, also used by .su.
            # Do not treat data labels, debug symbols or $local aliases as
            # independent frames. Every actual function must be in .text.
            if info & 15 != 2:
                if symbol == entry_symbol.encode():
                    raise ValueError("Device image entry symbol must be a function")
                continue
            if (
                not symbol
                or defined_in != text_index
                or not text.address <= address < text.address + text.size
                or length > text.address + text.size - address
            ):
                raise ValueError("Device image has a function outside .text")
            function = _function_name(symbol)
            if function in functions:
                raise ValueError(f"Device image has duplicate function symbol {function!r}")
            functions[function] = (address, length)
            if function == entry_symbol:
                if found_entry or address != text.address or not length or info >> 4 != 1:
                    raise ValueError("Device image needs one global entry function at the start of .text")
                found_entry = True
    if entry_symbol is not None and not found_entry:
        raise ValueError(f"Device image is missing entry function {entry_symbol!r}")
    return _function_groups(functions)


def _function_groups(functions: dict[str, tuple[int, int]]) -> frozenset[frozenset[str]]:
    groups: dict[tuple[int, int, str], set[str]] = {}
    for name, (address, length) in functions.items():
        # Aliases describe one frame only when their complete, nonempty ranges
        # match. A zero-sized symbol does not establish a shared code extent.
        key = (address, length, name if length == 0 else "")
        groups.setdefault(key, set()).add(name)
    return frozenset(frozenset(names) for names in groups.values())


def validate_device_image(data: bytes, *, entry_symbol: str | None = None) -> frozenset[frozenset[str]]:
    """Validate a linked AI Core ELF and group retained function aliases by code range.

    The loader keeps only .text and calls its first byte. No allocated data,
    unresolved symbols or relocation records can survive. Requiring a named
    entry additionally retains symbol evidence for compiler stack validation.
    This is structural validation of compiler-produced code, not a proof that
    arbitrary machine code is position-independent or obeys the calling ABI.
    """
    entry, sections, strings = _sections(data)
    if entry_symbol is not None and (data[6] != 1 or struct.unpack_from("<I", data, 20)[0] != 1):
        raise ValueError("Device image has an unsupported ELF version")
    text_index = _text_section(entry, sections, strings, plain_callee=entry_symbol is not None)
    return _functions(data, sections, text_index, entry_symbol)
