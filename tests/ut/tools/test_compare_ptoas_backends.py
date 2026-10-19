# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression coverage for backend binary comparison and missing evidence."""

import importlib.util
import struct
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "compare_ptoas_backends", Path(__file__).resolve().parents[2] / "ci/compare_ptoas_backends.py"
)
assert _spec is not None and _spec.loader is not None
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


def _elf(text=b"code", debug=b"debug", data=b"data", address=4096, bss_size=16):
    # A real ELF64 section layout, with debug bytes outside SHF_ALLOC. Changing
    # debug must affect whole-file identity without changing executable identity.
    names = b"\0.text\0.debug_info\0.data\0.bss\0.shstrtab\0"
    payloads = [
        (".text", 1, 6, text, address),
        (".debug_info", 1, 0, debug, 0),
        (".data", 1, 3, data, 8192),
        (".bss", 8, 3, b"", 12288),
        (".shstrtab", 3, 0, names, 0),
    ]
    binary = bytearray(64)
    binary[:6] = b"\x7fELF\x02\x01"
    headers = [bytes(64)]
    for name, kind, flags, payload, addr in payloads:
        offset = len(binary)
        binary.extend(payload)
        size = bss_size if kind == 8 else len(payload)
        headers.append(
            struct.pack(
                "<IIQQQQIIQQ", names.index(name.encode()), kind, flags, addr, offset, size, 0, 0, 1, 0
            )
        )
    struct.pack_into("<Q", binary, 40, len(binary))
    struct.pack_into("<HHH", binary, 58, 64, len(headers), len(headers) - 1)
    return bytes(binary) + b"".join(headers)


def test_equal_binary():
    result = audit.binary_comparison(_elf(), _elf())
    assert result["binary_equal"] and result["executable_equal"] and result["allocated_equal"]
    assert result["different_bytes"] == 0


def test_debug_only_difference():
    result = audit.binary_comparison(_elf(), _elf(debug=b"other debug path"))
    assert not result["binary_equal"]
    assert result["executable_equal"] and result["allocated_equal"]
    assert result["runtime_text_equal"]


@pytest.mark.parametrize("change", [{"text": b"other code"}, {"address": 4100}])
def test_executable_change_is_not_hidden(change):
    result = audit.binary_comparison(_elf(), _elf(**change))
    assert not result["executable_equal"] and not result["allocated_equal"]


@pytest.mark.parametrize("change", [{"data": b"changed constants"}, {"bss_size": 32}])
def test_data_and_zero_initialized_storage_matter(change):
    result = audit.binary_comparison(_elf(), _elf(**change))
    assert result["executable_equal"] and not result["allocated_equal"]


@pytest.mark.parametrize("binary", [b"", b"not an ELF", _elf()[:70]])
def test_invalid_or_truncated_elf_fails(binary):
    with pytest.raises(ValueError, match="ELF"):
        audit.elf_sections(binary, 4)


def test_empty_or_missing_backend_does_not_claim_parity(tmp_path):
    result = audit.compare_artifacts(tmp_path)
    assert not result["same_nonempty_pto"]
    assert not result["same_nonempty_kernel_inventory"]
    assert result["kernels"] == {}


@pytest.mark.parametrize("kind", ["vector", "cube"])
def test_adapter_uses_native_export_and_discards_reference_body(kind):
    source = "AICORE void kernel(__gm__ float* value) { reference_implementation(); }"
    symbols = f"00000000 g F .text 00000004 kernel.{kind}\n"
    declarations = audit.adapter_declarations(source, symbols)
    assert declarations == {
        "kernel": f'extern "C" AICORE void kernel(__gm__ float* value) __asm__("kernel.{kind}");'
    }
    assert "reference_implementation" not in declarations["kernel"]


@pytest.mark.parametrize("symbols", ["", "kernel.vector\nkernel.cube\n", "other_kernel.vector\n"])
def test_adapter_rejects_missing_or_ambiguous_native_entry(symbols):
    with pytest.raises(RuntimeError, match="Cannot match VPTO entry"):
        audit.adapter_declarations("AICORE void kernel() {}", symbols)


def test_adapter_rejects_no_reference_signature():
    with pytest.raises(RuntimeError, match="No VPTO functions"):
        audit.adapter_declarations("", "kernel.vector")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
