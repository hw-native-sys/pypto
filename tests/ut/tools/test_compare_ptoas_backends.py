# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression coverage for backend binary comparison and missing evidence."""

import importlib
import importlib.util
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

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


@pytest.fixture
def suite_backend(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "ci"))
    suite = importlib.import_module("ptoas_test_backend")
    pto_backend = importlib.import_module("pypto.backend.pto_backend")

    monkeypatch.setattr(suite, "_state", {"installed": False, "record_tests": True})
    monkeypatch.setenv("PYPTO_PTOAS_AUDIT_DIR", str(tmp_path / "audit"))
    monkeypatch.setenv("PYPTO_PTOAS_TEST_BACKEND", "vpto")
    calls = SimpleNamespace(ptoas=[], links=[], sources=[], fail_vpto=False)

    def run_ptoas(source, output, flags):
        selected = next(f for f in flags if f.startswith("--pto-backend="))
        calls.ptoas.append(selected)
        if calls.fail_vpto and selected == "--pto-backend=vpto":
            raise RuntimeError("unsupported VPTO operation")
        if selected == "--pto-backend=vpto":
            Path(output).write_bytes(b"native fat object")
        else:
            Path(output).write_text("AICORE void kernel(__gm__ float* value) { reference_implementation(); }")

    class Compiler:
        def __init__(self, platform="a2a3"):
            self.platform = platform
            self.sdk = SimpleNamespace(ccec=SimpleNamespace(cxx_path="compiler/bin/bisheng"))

        def _run(self, command, output, label):
            calls.links.append(command)
            native = any(str(arg).endswith("vpto.device.o") for arg in command)
            return _elf(text=b"vpto code" if native else b"emitc code")

        def compile_incore(self, source_path, *args, **kwargs):
            calls.sources.append(Path(source_path).read_text())
            return self._run(["link", "wrapper.o"], Path("unused.elf"), "Incore-link")

    def tool(command):
        if "llvm-objcopy" in command[0]:
            Path(command[1].split("=", 2)[2]).write_bytes(b"native object")
        return "kernel.vector\n"

    monkeypatch.setattr(pto_backend, "_run_ptoas", run_ptoas)
    monkeypatch.setitem(
        sys.modules, "pypto.runtime.kernel_compiler", SimpleNamespace(KernelCompiler=Compiler)
    )
    monkeypatch.setitem(
        sys.modules, "pypto.runtime.pto_isa", SimpleNamespace(ensure_pto_isa_root=lambda: "test-isa")
    )
    monkeypatch.setattr(suite, "_run_tool", tool)
    return SimpleNamespace(module=suite, compiler=Compiler, backend=pto_backend, calls=calls)


def _suite_wrapper(tmp_path, suite):
    source, output = tmp_path / "kernel.pto", tmp_path / "kernel.cpp"
    source.write_text("module {}")
    suite.backend._run_ptoas(str(source), str(output), ["--pto-arch=a3", "--pto-level=level2"])
    wrapper = tmp_path / "wrapper.cpp"
    wrapper.write_text(
        "// preserved header\n// --- ptoas-generated code ---\n"
        + output.read_text()
        + "\n// --- Kernel entry point ---\nvoid kernel_entry() {}\n"
    )
    return wrapper


@pytest.mark.parametrize("backend", ["emitc", "vpto"])
def test_existing_suite_executes_selected_backend_and_preserves_binary_evidence(
    suite_backend, monkeypatch, tmp_path, backend
):
    monkeypatch.setenv("PYPTO_PTOAS_TEST_BACKEND", backend)
    suite_backend.module.install()
    wrapper = _suite_wrapper(tmp_path, suite_backend)
    original = wrapper.read_bytes()
    binary = suite_backend.compiler().compile_incore(str(wrapper))
    assert binary == _elf(text=b"vpto code" if backend == "vpto" else b"emitc code")
    assert wrapper.read_bytes() == original
    assert f"--pto-backend={backend}" in suite_backend.calls.ptoas
    records = list((tmp_path / "audit").glob("module-*/kernel-*/result.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record["passed"]
    if backend == "vpto":
        assert not record["comparison"]["runtime_text_equal"]
        assert "reference_implementation" not in suite_backend.calls.sources[-1]
        assert 'extern "C" AICORE void kernel' in suite_backend.calls.sources[-1]
        assert suite_backend.calls.links[-1][-1].endswith("vpto.device.o")


def test_existing_suite_does_not_fallback_when_vpto_fails(suite_backend, tmp_path):
    suite_backend.calls.fail_vpto = True
    suite_backend.module.install()
    with pytest.raises(RuntimeError, match="unsupported VPTO"):
        _suite_wrapper(tmp_path, suite_backend)
    assert suite_backend.calls.ptoas == ["--pto-backend=vpto"]
    record = json.loads(next((tmp_path / "audit").glob("module-*/ptoas.json")).read_text())
    assert not record["passed"] and record["stage"] == "vpto"
    assert suite_backend.calls.links == []


def test_native_vpto_is_not_misreported_as_simulator_execution(suite_backend, tmp_path):
    suite_backend.module.install()
    wrapper = _suite_wrapper(tmp_path, suite_backend)
    with pytest.raises(RuntimeError, match="CPU simulator cannot execute"):
        suite_backend.compiler("a5sim").compile_incore(str(wrapper))
    assert suite_backend.calls.links == []


def test_bootstrap_stays_inactive_while_ci_dependencies_are_built(suite_backend, monkeypatch):
    monkeypatch.delenv("PYPTO_PTOAS_AUDIT_READY", raising=False)
    suite_backend.module.bootstrap()
    assert not suite_backend.module._state["installed"]
    assert suite_backend.calls.ptoas == []


def test_suite_backend_keeps_planner_and_target_flags(suite_backend):
    flags = ["--pto-level=level2", "--pto-arch", "a3", "--pto-backend=emitc"]
    selected = suite_backend.module.backend_flags(flags, "vpto")
    assert selected == ["--pto-level=level2", "--pto-arch", "a3", "--pto-backend=vpto"]
    assert flags[-1] == "--pto-backend=emitc"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
