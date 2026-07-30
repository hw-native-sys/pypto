# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""ELF / Mach-O object file parser for extracting .text sections and Build-IDs.

Pure Python implementation — no external dependencies.
"""

import logging
import struct
from pathlib import Path

logger = logging.getLogger(__name__)

# ELF Magic Numbers
ELFMAG0 = 0x7F
ELFMAG1 = ord("E")
ELFMAG2 = ord("L")
ELFMAG3 = ord("F")

# ELF class byte (e_ident[EI_CLASS]) for 64-bit objects.
ELFCLASS64 = 2

# Program-header type / note type for the GNU Build-ID note.
PT_NOTE = 4
NT_GNU_BUILD_ID = 3

# Mach-O Magic Numbers
MH_MAGIC_64 = 0xFEEDFACF

# Mach-O Load Command types
LC_SEGMENT_64 = 0x19

# ELF64 header / program-header field offsets and sizes.
_EHDR_SIZE = 64
_EHDR_PHOFF = 0x20
_EHDR_PHENTSIZE = 0x36
_PHDR_SIZE = 56
_PHDR_OFFSET = 8
_PHDR_FILESZ = 32
_NHDR_SIZE = 12

# FNV-1a 64-bit parameters (mirrors runtime ``src/common/utils/fnv1a_64.h``).
_FNV1A_64_OFFSET = 0xCBF29CE484222325
_FNV1A_64_PRIME = 0x100000001B3
_U64_MASK = 0xFFFFFFFFFFFFFFFF


def fnv1a_64(data: bytes) -> int:
    """FNV-1a 64-bit content hash of *data*.

    Mirrors the runtime's ``simpler::common::utils::fnv1a_64``
    (``src/common/utils/fnv1a_64.h``) so both sides agree bit-for-bit.
    """
    h = _FNV1A_64_OFFSET
    for byte in data:
        h = ((h ^ byte) * _FNV1A_64_PRIME) & _U64_MASK
    return h


def elf_build_id_64(data: bytes) -> int:
    """The 64-bit identifier the runtime derives from an ELF64's GNU Build-ID.

    The first 8 bytes of the ``NT_GNU_BUILD_ID`` note descriptor, read as a
    little-endian ``uint64``; falls back to :func:`fnv1a_64` over the whole
    buffer when *data* is not a well-formed ELF64 or carries no Build-ID (i.e.
    was linked without ``--build-id``).

    This mirrors the runtime's ``simpler::common::utils::elf_build_id_64``
    (``src/common/utils/elf_build_id.h``) exactly, because the value it returns
    for an orchestration ``.so`` is the ``hid=`` field of that callable's
    ``[STRACE]`` timing markers. Computing it host-side is what lets pypto map a
    marker back to the orchestration function name (see
    ``pypto.runtime.device_runner.register_callable_identity``).

    Args:
        data: The complete object-file bytes (the same buffer the runtime hashes).

    Returns:
        The 64-bit callable identity. Never raises — malformed input degrades to
        the FNV-1a fallback, matching the runtime.
    """
    if len(data) < _EHDR_SIZE:
        return fnv1a_64(data)
    if data[:4] != bytes((ELFMAG0, ELFMAG1, ELFMAG2, ELFMAG3)) or data[4] != ELFCLASS64:
        return fnv1a_64(data)

    (e_phoff,) = struct.unpack_from("<Q", data, _EHDR_PHOFF)
    e_phentsize, e_phnum = struct.unpack_from("<HH", data, _EHDR_PHENTSIZE)
    if e_phoff == 0 or e_phentsize < _PHDR_SIZE:
        return fnv1a_64(data)
    if e_phoff + e_phnum * e_phentsize > len(data):
        return fnv1a_64(data)

    for i in range(e_phnum):
        phdr = e_phoff + i * e_phentsize
        (p_type,) = struct.unpack_from("<I", data, phdr)
        if p_type != PT_NOTE:
            continue
        (p_offset,) = struct.unpack_from("<Q", data, phdr + _PHDR_OFFSET)
        (p_filesz,) = struct.unpack_from("<Q", data, phdr + _PHDR_FILESZ)
        if p_offset + p_filesz > len(data):
            continue  # Notes lie beyond the buffer we were given.
        build_id = _find_build_id_note(data, p_offset, p_offset + p_filesz)
        if build_id is not None:
            return build_id

    # No Build-ID found; the SO was likely linked without --build-id.
    return fnv1a_64(data)


def _find_build_id_note(data: bytes, start: int, end: int) -> int | None:
    """Scan the ELF note entries in ``data[start:end]`` for a GNU Build-ID."""
    note = start
    while note + _NHDR_SIZE <= end:
        namesz, descsz, ntype = struct.unpack_from("<III", data, note)
        name = note + _NHDR_SIZE
        desc = name + ((namesz + 3) & ~3)
        nxt = desc + ((descsz + 3) & ~3)
        if nxt > end:
            return None  # Malformed note entry.
        is_build_id = (
            ntype == NT_GNU_BUILD_ID and namesz == 4 and data[name : name + 4] == b"GNU\x00" and descsz >= 8
        )
        if is_build_id:
            return int(struct.unpack_from("<Q", data, desc)[0])
        note = nxt
    return None


def extract_text_section(obj_input: str | Path | bytes) -> bytes:
    """Extract .text section from an ELF64 or Mach-O .o file.

    Args:
        obj_input: Either a path to the .o file (str/Path) or the binary data (bytes).

    Returns:
        Binary data of the .text section.

    Raises:
        FileNotFoundError: If file path is provided and does not exist.
        ValueError: If data is not a valid object file or .text section not found.
    """
    if isinstance(obj_input, bytes):
        obj_data = obj_input
        source_name = "<bytes>"
    else:
        path = Path(obj_input)
        if not path.exists():
            raise FileNotFoundError(f"Object file not found: {obj_input}")
        with open(obj_input, "rb") as f:
            obj_data = f.read()
        source_name = str(obj_input)

    if len(obj_data) < 4:
        raise ValueError(f"Data too small to be a valid object file: {source_name}")

    # Detect format by magic number
    magic32 = struct.unpack("<I", obj_data[:4])[0]
    if magic32 == MH_MAGIC_64:
        return _extract_text_macho64(obj_data, source_name)

    if (
        obj_data[0] == ELFMAG0
        and obj_data[1] == ELFMAG1
        and obj_data[2] == ELFMAG2
        and obj_data[3] == ELFMAG3
    ):
        return _extract_text_elf64(obj_data, source_name)

    raise ValueError(f"Not a valid ELF or Mach-O file: {source_name}")


def _extract_text_elf64(elf_data: bytes, source_name: str) -> bytes:
    """Extract .text section from ELF64 data."""
    if len(elf_data) < 64:
        raise ValueError(f"Data too small to be a valid ELF: {source_name}")

    # Extract section header table info from ELF header
    e_shoff = struct.unpack("<Q", elf_data[40:48])[0]
    e_shnum = struct.unpack("<H", elf_data[60:62])[0]
    e_shstrndx = struct.unpack("<H", elf_data[62:64])[0]

    # Get string table section header
    shstr_offset = e_shoff + e_shstrndx * 64
    shstr_sh_offset = struct.unpack("<Q", elf_data[shstr_offset + 24 : shstr_offset + 32])[0]
    shstr_sh_size = struct.unpack("<Q", elf_data[shstr_offset + 32 : shstr_offset + 40])[0]

    # Extract string table
    strtab = elf_data[shstr_sh_offset : shstr_sh_offset + shstr_sh_size]

    # Find .text section
    for i in range(e_shnum):
        section_offset = e_shoff + i * 64
        sh_name = struct.unpack("<I", elf_data[section_offset : section_offset + 4])[0]
        sh_offset = struct.unpack("<Q", elf_data[section_offset + 24 : section_offset + 32])[0]
        sh_size = struct.unpack("<Q", elf_data[section_offset + 32 : section_offset + 40])[0]

        section_name = _extract_cstring(strtab, sh_name)
        if section_name == ".text":
            text_data = elf_data[sh_offset : sh_offset + sh_size]
            logger.debug(f"Loaded .text section from {source_name} (size: {sh_size} bytes)")
            return text_data

    raise ValueError(f".text section not found in: {source_name}")


def _extract_text_macho64(data: bytes, source_name: str) -> bytes:
    """Extract __text section from Mach-O 64-bit data."""
    if len(data) < 32:
        raise ValueError(f"Data too small to be a valid Mach-O: {source_name}")

    ncmds = struct.unpack("<I", data[16:20])[0]

    # Walk load commands starting at offset 32
    offset = 32
    for _ in range(ncmds):
        if offset + 8 > len(data):
            break
        cmd = struct.unpack("<I", data[offset : offset + 4])[0]
        cmdsize = struct.unpack("<I", data[offset + 4 : offset + 8])[0]

        if cmd == LC_SEGMENT_64:
            nsects = struct.unpack("<I", data[offset + 64 : offset + 68])[0]
            sect_base = offset + 72
            for s in range(nsects):
                sect_off = sect_base + s * 80
                sectname = data[sect_off : sect_off + 16].split(b"\x00")[0].decode("ascii")
                if sectname == "__text":
                    s_size = struct.unpack("<Q", data[sect_off + 40 : sect_off + 48])[0]
                    s_offset = struct.unpack("<I", data[sect_off + 48 : sect_off + 52])[0]
                    text_data = data[s_offset : s_offset + s_size]
                    logger.debug(f"Loaded __text section from {source_name} (size: {s_size} bytes)")
                    return text_data

        offset += cmdsize

    raise ValueError(f"__text section not found in: {source_name}")


def _extract_cstring(data: bytes, offset: int) -> str:
    """Extract a null-terminated C string from bytes."""
    end = data.find(b"\x00", offset)
    if end == -1:
        end = len(data)
    return data[offset:end].decode("ascii", errors="ignore")
