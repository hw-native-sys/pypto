# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared device-image fixtures independent of the optional toolchain."""

import struct

import pytest


def _device_elf(*, entry=0x1000, extra_flags=0, extra_kind=1, extra_size=0, text=b"\x10" * 24):
    """Construct a minimal linked image to exercise loader constraints without CANN."""
    section_offset = (128 + len(text) + extra_size + 7) & ~7
    data = bytearray(section_offset + 256)
    data[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<HH", data, 16, 2, 0x1029)
    struct.pack_into("<Q", data, 24, entry)
    struct.pack_into("<Q", data, 40, section_offset)
    struct.pack_into("<HHH", data, 58, 64, 4, 1)
    names = b"\0.shstrtab\0.text\0.extra\0"
    data[64 : 64 + len(names)] = names
    struct.pack_into("<IIQQQQIIQQ", data, section_offset + 64, 1, 3, 0, 0, 64, len(names), 0, 0, 1, 0)
    struct.pack_into("<IIQQQQIIQQ", data, section_offset + 128, 11, 1, 6, 0x1000, 96, len(text), 0, 0, 4, 0)
    struct.pack_into(
        "<IIQQQQIIQQ",
        data,
        section_offset + 192,
        17,
        extra_kind,
        extra_flags,
        0x2000,
        96 + len(text),
        extra_size,
        0,
        0,
        4,
        24 if extra_kind == 2 else 0,
    )
    if extra_kind == 2:
        struct.pack_into("<IBBH", data, 96 + len(text), 1, 0, 0, 0)
    data[96 : 96 + len(text)] = text
    return bytes(data)


@pytest.fixture
def device_elf():
    """Build executable AI Core ELF images with controlled entry and sections."""
    return _device_elf
