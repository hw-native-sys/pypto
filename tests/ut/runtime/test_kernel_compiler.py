# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Build ownership, command compatibility and failure cleanup regressions."""

import importlib
import os
import shutil
import struct
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pypto.backend._ptoas_locate import PTOAS_MIN_VERSION
from pypto.runtime._device_image import validate_device_image
from pypto.runtime._stack_usage import validate_stack_usage
from pypto.runtime.elf_parser import extract_text_section


@pytest.fixture
def compiler_module(monkeypatch, tmp_path):
    """Exercise PyPTO commands without requiring an installed device SDK."""
    monkeypatch.delenv("PYPTO_COMPILER_TIMEOUT", raising=False)
    tool = SimpleNamespace(
        cxx_path="compiler",
        linker_path="linker",
        is_host=False,
        get_compile_flags=lambda **kwargs: ["-shared", "-fPIC"],
    )
    host = SimpleNamespace(
        cxx_path="host-compiler", is_host=True, get_compile_flags=lambda **kwargs: ["-shared", "-fPIC"]
    )
    sdk = SimpleNamespace(
        project_root=tmp_path,
        ccec=tool,
        gxx15=host,
        host_gxx=host,
        ccu=SimpleNamespace(host_toolchain=host),
        get_incore_include_dirs=lambda: [str(tmp_path / "sdk include")],
        get_orchestration_cache_inputs=lambda runtime: ([], []),
        _orchestration_toolchain=lambda runtime: host if runtime == "host_build_graph" else tool,
        incore_compile_cache_token=lambda core_type: {
            "flags": ["-shared", "-fPIC", "-DSDK_SIM_TARGET", "-include", "sdk_target.h"]
        },
    )
    monkeypatch.setitem(sys.modules, "simpler_setup", SimpleNamespace(KernelCompiler=lambda platform: sdk))
    monkeypatch.setitem(
        sys.modules,
        "simpler_setup.compile_paths",
        SimpleNamespace(compiler_visible_path=lambda path: Path(path)),
    )
    monkeypatch.setitem(
        sys.modules, "simpler_setup.toolchain", SimpleNamespace(GxxToolchain=lambda **kwargs: host)
    )
    name = "pypto.runtime.kernel_compiler"
    monkeypatch.delitem(sys.modules, name, raising=False)
    import pypto.runtime as runtime_package  # noqa: PLC0415

    monkeypatch.delattr(runtime_package, "kernel_compiler", raising=False)
    module = importlib.import_module(name)
    yield module
    # Import machinery writes these outside monkeypatch; restore the original
    # module/attribute (when present) through monkeypatch's saved deletion.
    sys.modules.pop(name, None)
    if hasattr(runtime_package, "kernel_compiler"):
        delattr(runtime_package, "kernel_compiler")


def _source(tmp_path):
    """Create a minimal kernel source at a path containing a space."""
    source = tmp_path / "kernel source.cpp"
    source.write_text('extern "C" void kernel_entry() {}')
    return str(source)


@pytest.mark.parametrize("platform", ["a2a3", "a5", "a2a3sim", "a5sim"])
def test_incore_build_and_link_are_owned_by_pypto(
    compiler_module, monkeypatch, tmp_path, device_elf, platform
):
    """Verify target-specific compiler commands, device linking and output cleanup."""
    commands = []

    def run(command, **kwargs):
        """Record invocations and emit distinct compiler and linker outputs."""
        commands.append(command)
        assert kwargs["cwd"] == tmp_path
        assert kwargs["timeout"] == 900
        Path(command[command.index("-o") + 1]).write_bytes(
            device_elf() if command[0] == "linker" else b"object"
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(compiler_module.subprocess, "run", run)
    compiler = compiler_module.KernelCompiler(platform)
    result = compiler.compile_incore(_source(tmp_path), pto_isa_root=str(tmp_path), build_dir=str(tmp_path))
    assert result == (b"object" if platform.endswith("sim") else device_elf())
    assert len(commands) == (1 if platform.endswith("sim") else 2)
    assert f"-I{tmp_path / 'sdk include'}" in commands[0]
    assert ("-DSDK_SIM_TARGET" in commands[0]) == platform.endswith("sim")
    if platform.endswith("sim"):
        assert commands[0][commands[0].index("-include") + 1] == "sdk_target.h"
    if not platform.endswith("sim"):
        assert commands[1][1:3] == ["-e", "kernel_entry"]
        assert commands[1][-1] == commands[0][commands[0].index("-o") + 1]
    assert not list(tmp_path.glob("pypto-incore-*"))


@pytest.mark.parametrize("failure", ["compile", "link", "empty", "missing", "unavailable"])
def test_failed_build_cleans_all_temporary_outputs(compiler_module, monkeypatch, tmp_path, failure):
    """Discard temporary binaries after compiler, linker or output-validation failures."""

    def run(command, **kwargs):
        """Inject the selected compiler failure or malformed output."""
        if failure == "unavailable":
            raise FileNotFoundError("compiler missing")
        output = Path(command[command.index("-o") + 1])
        if failure != "missing":
            output.write_bytes(b"" if failure == "empty" else b"partial")
        failed = (failure == "compile" and command[0] == "compiler") or (
            failure == "link" and command[0] == "linker"
        )
        return SimpleNamespace(returncode=1 if failed else 0, stderr="injected compiler error")

    monkeypatch.setattr(compiler_module.subprocess, "run", run)
    compiler = compiler_module.KernelCompiler("a2a3")
    with pytest.raises(RuntimeError, match="compilation failed|produced no binary|cannot run compiler"):
        compiler.compile_incore(_source(tmp_path), pto_isa_root=str(tmp_path), build_dir=str(tmp_path))
    assert not list(tmp_path.glob("pypto-incore-*"))


@pytest.mark.parametrize("runtime", ["host_build_graph", "tensormap_and_ringbuffer"])
def test_orchestration_uses_declared_sdk_sources(compiler_module, monkeypatch, tmp_path, runtime):
    """Compile all required helper sources with target-appropriate linking flags."""
    helper = tmp_path / "sdk helper.cpp"
    helper.write_text("// helper")
    compiler = compiler_module.KernelCompiler("a2a3")
    compiler.sdk.get_orchestration_cache_inputs = lambda runtime: ([str(tmp_path)], [str(helper)])
    commands = []

    def run(command, **kwargs):
        """Record orchestration commands and emit a stand-in shared library."""
        commands.append(command)
        Path(command[command.index("-o") + 1]).write_bytes(b"orchestration")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(compiler_module.subprocess, "run", run)
    assert (
        compiler.compile_orchestration(runtime, _source(tmp_path), build_dir=str(tmp_path))
        == b"orchestration"
    )
    assert str(helper) in commands[0]
    expected_link_flag = "-undefined" if sys.platform == "darwin" else "-Wl,--build-id=sha1"
    assert expected_link_flag in commands[0]
    assert ("-pthread" in commands[0]) == (runtime == "host_build_graph")
    assert not list(tmp_path.glob("pypto-orchestration-*"))
    helper.unlink()
    with pytest.raises(FileNotFoundError, match="sdk helper.cpp"):
        compiler.compile_orchestration(runtime, _source(tmp_path))
    assert len(commands) == 1


def test_invalid_inputs_fail_before_compiler(compiler_module, monkeypatch, tmp_path):
    """Reject unsupported core types and missing ISA roots without invoking a compiler."""
    run = Mock(side_effect=AssertionError("compiler must not run"))
    monkeypatch.setattr(compiler_module.subprocess, "run", run)
    compiler = compiler_module.KernelCompiler("a2a3")
    source = _source(tmp_path)
    with pytest.raises(ValueError, match="core_type"):
        compiler.compile_incore(source, core_type="invalid")
    with pytest.raises(ValueError, match="pto_isa_root"):
        compiler.compile_incore(source)
    run.assert_not_called()


@pytest.mark.parametrize("platform", ["a2a3sim", "a2a3"])
def test_sanitizer_flags_apply_only_to_host_builds(
    compiler_module, monkeypatch, tmp_path, device_elf, platform
):
    """Keep sanitizer instrumentation on host builds and out of device compiler commands."""
    monkeypatch.setattr(compiler_module.KernelCompiler, "_sanitizers", "address")
    compiler = compiler_module.KernelCompiler(platform)
    commands = []

    def run(command, **kwargs):
        """Record target flags while supplying a successful build output."""
        commands.append(command)
        Path(command[command.index("-o") + 1]).write_bytes(
            device_elf() if command[0] == "linker" else b"binary"
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(compiler_module.subprocess, "run", run)
    compiler.compile_incore(_source(tmp_path), pto_isa_root=str(tmp_path))
    assert ("-fsanitize=address" in commands[0]) == platform.endswith("sim")
    compiler.compile_orchestration("host_build_graph", _source(tmp_path))
    assert "-fsanitize=address" in commands[-1]


@pytest.mark.parametrize(
    ("stage", "label"),
    [("compiler", "Incore"), ("linker", "Incore-link"), ("host-compiler", "Orchestration")],
)
def test_compiler_timeout_reports_stage_and_cleans_outputs(
    compiler_module, monkeypatch, tmp_path, stage, label
):
    """All build stages honor the configured timeout and discard partial outputs."""
    monkeypatch.setenv("PYPTO_COMPILER_TIMEOUT", "0.25")
    commands = []

    def run(command, **kwargs):
        """Emit partial output and inject a timeout at the selected build stage."""
        commands.append(command[0])
        assert kwargs["timeout"] == 0.25
        Path(command[command.index("-o") + 1]).write_bytes(b"partial")
        if command[0] == stage:
            raise compiler_module.subprocess.TimeoutExpired(command, kwargs["timeout"])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(compiler_module.subprocess, "run", run)
    compiler = compiler_module.KernelCompiler("a2a3")
    with pytest.raises(RuntimeError, match=f"{label}: compiler timed out after 0.25 seconds") as error:
        if stage == "host-compiler":
            compiler.compile_orchestration("host_build_graph", _source(tmp_path), build_dir=str(tmp_path))
        else:
            compiler.compile_incore(_source(tmp_path), pto_isa_root=str(tmp_path), build_dir=str(tmp_path))
    assert isinstance(error.value.__cause__, compiler_module.subprocess.TimeoutExpired)
    assert commands[-1] == stage
    assert not list(tmp_path.glob("pypto-incore-*"))
    assert not list(tmp_path.glob("pypto-orchestration-*"))


def test_stalled_compiler_process_is_terminated(compiler_module, monkeypatch, tmp_path):
    """A real stalled subprocess exits through the bounded compiler error path."""
    monkeypatch.setenv("PYPTO_COMPILER_TIMEOUT", "0.1")
    compiler = compiler_module.KernelCompiler()
    with pytest.raises(RuntimeError, match="Incore: compiler timed out after 0.1 seconds"):
        compiler._run([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path / "kernel.o", "Incore")


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf", "-inf", "1e999", "invalid", ""])
def test_invalid_compiler_timeout_fails_before_sdk_setup(compiler_module, monkeypatch, timeout):
    """Reject values that would disable the timeout before discovering toolchains."""
    sdk = Mock(side_effect=AssertionError("SDK setup must not run"))
    monkeypatch.setattr(compiler_module, "_SimplerCompilerSDK", sdk)
    monkeypatch.setenv("PYPTO_COMPILER_TIMEOUT", timeout)
    with pytest.raises(ValueError, match="PYPTO_COMPILER_TIMEOUT must be positive finite seconds"):
        compiler_module.KernelCompiler()
    sdk.assert_not_called()


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"entry": 0x1010}, "start of .text"),
        ({"extra_flags": 2, "extra_size": 4}, "allocated section"),
        ({"extra_kind": 4, "extra_size": 4}, "relocations"),
        ({"extra_kind": 2, "extra_size": 24}, "unresolved symbol"),
    ],
)
def test_vpto_rejects_images_that_text_loader_cannot_load(device_elf, kwargs, error):
    from pypto.runtime._vpto import validate_device_image  # noqa: PLC0415

    validate_device_image(device_elf())
    with pytest.raises(ValueError, match=error):
        validate_device_image(device_elf(**kwargs))


def _text_only_elf():
    """An entry and a retained helper, each with independently required stack evidence."""
    data = bytearray(832)
    data[:7] = b"\x7fELF\x02\x01\x01"
    struct.pack_into("<HHI", data, 16, 2, 0x1029, 1)
    struct.pack_into("<Q", data, 24, 0x1000)
    struct.pack_into("<Q", data, 40, 512)
    struct.pack_into("<HHH", data, 58, 64, 5, 1)
    names = b"\0.shstrtab\0.text\0.strtab\0.symtab\0"
    symbols = b"\0kernel_entry\0helper\0"
    data[64 : 64 + len(names)] = names
    data[160 : 160 + len(symbols)] = symbols
    headers = [
        (1, 3, 0, 0, 64, len(names), 0, 0, 1, 0),
        (11, 1, 6, 0x1000, 128, 16, 0, 0, 4, 0),
        (17, 3, 0, 0, 160, len(symbols), 0, 0, 1, 0),
        (25, 2, 0, 0, 224, 72, 3, 1, 8, 24),
    ]
    for index, header in enumerate(headers, 1):
        struct.pack_into("<IIQQQQIIQQ", data, 512 + 64 * index, *header)
    struct.pack_into("<IBBHQQ", data, 248, 1, 0x12, 0, 2, 0x1000, 4)
    struct.pack_into("<IBBHQQ", data, 272, 14, 0x02, 0, 2, 0x1008, 4)
    return data


@pytest.mark.parametrize(
    "offset,fmt,value,error",
    [
        (6, "B", 0, "ELF version"),
        (20, "I", 0, "ELF version"),
        (24, "Q", 0x1004, "start of .text"),
        (40, "Q", 830, "section table"),
        (576 + 24, "Q", 830, "truncated section"),
        (576 + 4, "I", 1, "section-name string table"),
        (640, "I", 500, "string-table offset"),
        (640 + 4, "I", 8, "executable .text"),
        (640 + 8, "Q", 3, "executable .text"),
        (704, "I", 11, "duplicate .text"),
        (768 + 40, "I", 5, "symbol string table"),
        (768 + 56, "Q", 0, "symbol table"),
        (248, "I", 100, "string-table offset"),
        (248 + 4, "B", 0x11, "must be a function"),
        (248 + 6, "H", 0, "unresolved symbol"),
        (248 + 8, "Q", 0x1004, "global entry function"),
        (248 + 16, "Q", 0, "global entry function"),
        (272 + 8, "Q", 0x1010, "function outside .text"),
        (272 + 16, "Q", 100, "function outside .text"),
    ],
)
def test_text_only_rejects_malformed_images(offset, fmt, value, error):
    image = _text_only_elf()
    assert validate_device_image(bytes(image), entry_symbol="kernel_entry") == {
        frozenset({"kernel_entry"}),
        frozenset({"helper"}),
    }
    struct.pack_into(f"<{fmt}", image, offset, value)
    with pytest.raises(ValueError, match=error):
        validate_device_image(bytes(image), entry_symbol="kernel_entry")


@pytest.mark.parametrize("entry_symbol", [None, "kernel_entry"])
def test_device_image_rejects_text_in_section_zero(entry_symbol):
    image = _text_only_elf()
    # The loader selects the first .text, while the entry symbols refer to the
    # second one. Admission must reject this image before trusting its symbols.
    image[128:144] = b"validated bytes!"
    image[144:160] = b"different bytes!"
    image[512:576] = image[640:704]
    struct.pack_into("<Q", image, 512 + 24, 144)
    assert extract_text_section(bytes(image)) == b"different bytes!"
    with pytest.raises(ValueError, match="null section at index 0"):
        validate_device_image(bytes(image), entry_symbol=entry_symbol)


@pytest.mark.parametrize("offset,fmt", [(0, "I"), (4, "I"), (8, "Q"), (32, "Q")])
def test_device_image_requires_null_section_zero(offset, fmt):
    image = _text_only_elf()
    struct.pack_into(f"<{fmt}", image, 512 + offset, 1)
    with pytest.raises(ValueError, match="null section at index 0"):
        validate_device_image(bytes(image), entry_symbol="kernel_entry")


@pytest.mark.parametrize("entry_symbol", [None, "kernel_entry"])
@pytest.mark.parametrize("name", [b".te\xffxt", b"\x80.text", b".text\xfe"])
def test_device_image_rejects_section_names_decoded_as_text(entry_symbol, name):
    image = _text_only_elf()
    names_size = struct.unpack_from("<Q", image, 576 + 32)[0]
    image[64 + names_size : 64 + names_size + len(name) + 1] = name + b"\0"
    # Rename the earlier, nonallocated string table. The loader drops the
    # non-ASCII byte and selects its payload instead of the validated .text.
    struct.pack_into("<I", image, 576, names_size)
    struct.pack_into("<Q", image, 576 + 32, names_size + len(name) + 1)
    loaded = extract_text_section(bytes(image))
    assert loaded == bytes(image[64 : 64 + names_size + len(name) + 1])
    assert loaded != bytes(image[128:144])
    with pytest.raises(ValueError, match="non-ASCII section name"):
        validate_device_image(bytes(image), entry_symbol=entry_symbol)


@pytest.mark.parametrize("entry_symbol", [None, "kernel_entry"])
def test_device_image_rejects_non_ascii_section_names_outside_text(entry_symbol):
    image = _text_only_elf()
    # The .strtab name is not selected by the loader, but admission still
    # requires every section name to have an unambiguous ASCII spelling.
    image[64 + 18] = 0xFF
    assert extract_text_section(bytes(image)) == bytes(image[128:144])
    with pytest.raises(ValueError, match="non-ASCII section name"):
        validate_device_image(bytes(image), entry_symbol=entry_symbol)


@pytest.mark.parametrize("definition", ["kernel_entry", "helper"])
def test_text_only_accepts_stack_report_for_either_function_alias(tmp_path, definition):
    image = _text_only_elf()
    struct.pack_into("<Q", image, 272 + 8, 0x1000)
    functions = validate_device_image(bytes(image), entry_symbol="kernel_entry")
    assert functions == {frozenset({"kernel_entry", "helper"})}
    path = tmp_path / "kernel.su"
    path.write_text(f"source:{definition}\t0\tstatic\n")
    validate_stack_usage(path, functions)


@pytest.mark.parametrize(
    "report,error",
    [
        ("source:dead\t0\tstatic\n", "missing stack-usage evidence"),
        ("source:kernel_entry\t0\tstatic\nsource:helper\t16\tstatic\n", "zero-stack"),
        ("source:kernel_entry\t0\tstatic\nsource:helper\t0\tdynamic\n", "zero-stack"),
        ("source:kernel_entry\t16\tstatic\nsource:helper\t0\tstatic\n", "zero-stack"),
    ],
)
def test_text_only_rejects_missing_or_unsafe_alias_evidence(tmp_path, report, error):
    image = _text_only_elf()
    struct.pack_into("<Q", image, 272 + 8, 0x1000)
    path = tmp_path / "kernel.su"
    path.write_text(report)
    with pytest.raises(ValueError, match=error):
        validate_stack_usage(path, validate_device_image(bytes(image), entry_symbol="kernel_entry"))


@pytest.mark.parametrize("length", [0, 8])
def test_text_only_does_not_merge_different_function_extents(tmp_path, length):
    image = _text_only_elf()
    struct.pack_into("<QQ", image, 272 + 8, 0x1000, length)
    path = tmp_path / "kernel.su"
    path.write_text("source:kernel_entry\t0\tstatic\n")
    with pytest.raises(ValueError, match="missing stack-usage evidence.*helper"):
        validate_stack_usage(path, validate_device_image(bytes(image), entry_symbol="kernel_entry"))


def test_text_only_does_not_merge_zero_sized_symbols(tmp_path):
    image = _text_only_elf()
    struct.pack_into("<Q", image, 272 + 16, 0)
    image[181:187] = b"alias\0"
    struct.pack_into("<Q", image, 704 + 32, 27)
    struct.pack_into("<Q", image, 768 + 32, 96)
    struct.pack_into("<IBBHQQ", image, 296, 21, 0x12, 0, 2, 0x1008, 0)
    path = tmp_path / "kernel.su"
    path.write_text("source:kernel_entry\t0\tstatic\nsource:helper\t0\tstatic\n")
    with pytest.raises(ValueError, match="missing stack-usage evidence.*alias"):
        validate_stack_usage(path, validate_device_image(bytes(image), entry_symbol="kernel_entry"))


@pytest.mark.parametrize(
    "report,error",
    [
        (None, "no stack-usage report"),
        ("", "missing stack-usage evidence"),
        ("source:kernel_entry\t0\tstatic\n", "helper"),
        ("source:kernel_entry\t0\tstatic\nsource:helper\t16\tstatic\n", "zero-stack"),
        ("source:kernel_entry\t0\tstatic\nsource:helper\t0\tdynamic\n", "zero-stack"),
        ("bad record", "malformed"),
        ("source:helper\t-1\tstatic", "invalid or duplicate"),
        ("source:helper\t0\tstatic\nsource:helper\t0\tstatic", "invalid or duplicate"),
    ],
)
def test_text_only_requires_compiler_stack_evidence(tmp_path, report, error):
    path = tmp_path / "kernel.su"
    if report is not None:
        path.write_text(report)
    with pytest.raises(ValueError, match=error):
        validate_stack_usage(path, frozenset({frozenset({"kernel_entry"}), frozenset({"helper"})}))


def test_text_only_ignores_stack_frames_removed_by_linker(tmp_path):
    path = tmp_path / "kernel.su"
    path.write_text("path:kernel_entry\t0\tstatic\npath:helper\t0\tstatic\npath:dead\t64\tstatic\n")
    validate_stack_usage(path, validate_device_image(bytes(_text_only_elf()), entry_symbol="kernel_entry"))


def test_text_only_requires_unambiguous_retained_symbols():
    image = _text_only_elf()
    # A stripped binary cannot establish that kernel_entry is the first byte,
    # nor enumerate every function whose compiler stack evidence is required.
    struct.pack_into("<I", image, 768 + 4, 1)
    with pytest.raises(ValueError, match="missing entry function"):
        validate_device_image(bytes(image), entry_symbol="kernel_entry")
    image = _text_only_elf()
    struct.pack_into("<I", image, 272, 1)
    with pytest.raises(ValueError, match="duplicate function symbol"):
        validate_device_image(bytes(image), entry_symbol="kernel_entry")


def test_text_only_rejects_runtime_provisioning_metadata():
    image = _text_only_elf()
    names_size = struct.unpack_from("<Q", image, 576 + 32)[0]
    name = b".ascend.meta.kernel_entry\0"
    image[64 + names_size : 64 + names_size + len(name)] = name
    struct.pack_into("<Q", image, 576 + 32, names_size + len(name))
    struct.pack_into("<I", image, 704, names_size)
    with pytest.raises(ValueError, match="kernel runtime metadata"):
        validate_device_image(bytes(image), entry_symbol="kernel_entry")


@pytest.mark.parametrize("failure", [None, "stack", "missing_report", "image"])
def test_text_only_compile_validates_before_return_and_cleans_outputs(
    compiler_module, monkeypatch, tmp_path, failure
):
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        output = Path(command[command.index("-o") + 1])
        if command[0] == "compiler":
            output.write_bytes(b"object")
            if failure != "missing_report":
                size = 16 if failure == "stack" else 0
                output.with_suffix(".su").write_text(
                    f"source:kernel_entry\t0\tstatic\nsource:helper\t{size}\tstatic\n"
                )
        else:
            output.write_bytes(b"bad image" if failure == "image" else _text_only_elf())
            order = next(arg.split("=", 1)[1] for arg in command if arg.startswith("--symbol-ordering"))
            assert Path(order).read_text() == "kernel_entry\n"
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(compiler_module.subprocess, "run", run)
    compiler = compiler_module.KernelCompiler("a5")
    kwargs = dict(pto_isa_root=str(tmp_path), build_dir=str(tmp_path), text_only=True)
    if failure:
        with pytest.raises(ValueError, match="Device image"):
            compiler.compile_incore(_source(tmp_path), **kwargs)
    else:
        assert compiler.compile_incore(_source(tmp_path), **kwargs) == _text_only_elf()
    assert {"-fstack-usage", "-ffunction-sections", "-fdata-sections", "-fomit-frame-pointer"} <= set(
        commands[0]
    )
    assert "--gc-sections" in commands[1]
    assert not list(tmp_path.glob("pypto-incore-*"))


@pytest.mark.parametrize("platform,helpers", [("a5sim", None), ("a2a3", None), ("a5", ["helper.pto"])])
def test_text_only_rejects_unsupported_build_modes(compiler_module, monkeypatch, tmp_path, platform, helpers):
    run = Mock(side_effect=AssertionError("compiler must not run"))
    monkeypatch.setattr(compiler_module.subprocess, "run", run)
    compiler = compiler_module.KernelCompiler(platform)
    with pytest.raises(ValueError, match="text_only requires"):
        compiler.compile_incore(_source(tmp_path), text_only=True, vpto_sources=helpers)
    run.assert_not_called()


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("live_data", [False, True])
def test_real_cannbotdsl_text_only_compile(compiler_module, tmp_path, dynamic, live_data):
    """The real exporter includes SDK data; the strict link must remove only dead data."""
    cb = pytest.importorskip("cannbotdsl")
    if not hasattr(cb, "export_device"):
        pytest.skip("installed CANNBotDSL lacks public device export")
    ccec = shutil.which("ccec")
    if ccec is None:
        pytest.skip("CANN device compiler unavailable")
    sdk = Path(ccec).resolve().parents[3]
    headers = list(sdk.glob("*/asc/include/c_api/asc_simd.h"))
    if not headers:
        pytest.skip("CANN C API headers unavailable")

    from pypto import DataType  # noqa: PLC0415
    from pypto.backend.cannbotdsl_abi import build_call_adapter  # noqa: PLC0415
    from pypto.jit.external.contracts import (  # noqa: PLC0415
        CompileTarget,
        KernelRequest,
        ParameterSpec,
        TensorSpec,
    )
    from pypto.jit.external.providers.cannbotdsl import CANNBotDSLProvider  # noqa: PLC0415

    @cb.kernel
    def copy_elements(x, out):
        for i in cb.range(x.shape[0]):
            out[i] = x[i] + 1.0

    tensor = TensorSpec(("n" if dynamic else 32,), DataType.FP32, strides=(1,))
    request = KernelRequest(
        (ParameterSpec("x", tensor), ParameterSpec("out", tensor)), CompileTarget("a5", "ccec", "test")
    )
    artifact = build_call_adapter(CANNBotDSLProvider().prepare(copy_elements, request))
    for file in artifact.files:
        path = tmp_path / file.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(file.content)
    source = tmp_path / "entry.cpp"
    # A live reference must survive GC and fail admission. Dropping every data
    # section unconditionally would turn this into silently miscompiled code.
    read_data = "args[0] = g_tilingKey;" if live_data else ""
    source.write_text(
        f'#include "{artifact.entry}"\n'
        f'extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {{\n'
        f"  {read_data}\n  {artifact.symbol}(args);\n}}\n"
    )
    compiler = compiler_module.KernelCompiler("a5")
    compiler.sdk.ccec.cxx_path = ccec
    compiler.sdk.ccec.linker_path = str(Path(ccec).resolve().with_name("ld.lld"))
    compiler.sdk.ccec.get_compile_flags = lambda **kwargs: [
        "-std=c++17",
        "-O2",
        "-DNDEBUG",
        "-c",
        "-x",
        "cce",
        "--cce-aicore-only",
        "--cce-aicore-arch=dav-c310-vec",
    ]
    root = Path(__file__).resolve().parents[3]
    includes = [root / "runtime/src/common", root / "runtime/src/common/task_interface"]
    includes += [headers[0].parents[1], headers[0].parents[2]]
    includes.extend(tmp_path / path for path in artifact.include_dirs)
    kwargs = dict(pto_isa_root=str(tmp_path), extra_include_dirs=[str(p) for p in includes], text_only=True)
    if live_data:
        with pytest.raises(ValueError, match="allocated section"):
            compiler.compile_incore(str(source), **kwargs)
    else:
        binary = compiler.compile_incore(str(source), **kwargs)
        functions = validate_device_image(binary, entry_symbol="kernel_entry")
        assert any("kernel_entry" in names for names in functions)


@pytest.mark.parametrize("core_type", ["aic", "aiv", "mixed"])
@pytest.mark.parametrize("dynamic", [False, True])
def test_real_cannbotdsl_runtime_bridge_compile(compiler_module, tmp_path, dynamic, core_type):
    """Compile the actual resource/completion wrapper through strict binary admission."""
    cb = pytest.importorskip("cannbotdsl")
    if not hasattr(cb, "export_device"):
        pytest.skip("installed CANNBotDSL lacks public device export")
    ccec = shutil.which("ccec")
    if ccec is None:
        pytest.skip("CANN device compiler unavailable")
    sdk = Path(ccec).resolve().parents[3]
    if not (sdk / "asc/include/c_api/asc_simd.h").is_file():
        pytest.skip("CANN C API headers unavailable")

    from pypto import DataType  # noqa: PLC0415
    from pypto.backend.cannbotdsl_runtime import compile_task  # noqa: PLC0415
    from pypto.backend.external_bridge import RuntimeTarget  # noqa: PLC0415
    from pypto.backend.tmr_bridge import TMRBridge  # noqa: PLC0415
    from pypto.jit.external.contracts import (  # noqa: PLC0415
        CompileTarget,
        KernelRequest,
        ParameterSpec,
        TensorSpec,
    )
    from pypto.jit.external.providers.cannbotdsl import CANNBotDSLProvider  # noqa: PLC0415

    @cb.kernel
    def copy_elements(x, out):
        for i in cb.range(x.shape[0]):
            out[i] = x[i] + 1.0

    @cb.kernel
    def cube_copy(x, out):
        for i in cb.range(x.shape[0]):
            cb.scalar.cube_store_bypass(out.ptr(i), cb.scalar.load_bypass(x.ptr(i)) + 1)

    @cb.kernel
    def mixed_copy(x, out):
        for i in cb.range(x.shape[0]):
            if (i // 16) % 3 == 0:
                cb.scalar.cube_store_bypass(out.ptr(i), cb.scalar.load_bypass(x.ptr(i)) + 1)
            if (i // 16) % 3 == cb.get_subblock_id() + 1:
                cb.scalar.vec_store_bypass(out.ptr(i), cb.scalar.load_bypass(x.ptr(i)) + 1)

    tensor = TensorSpec(
        ("n" if dynamic else 96,), DataType.FP32 if core_type == "aiv" else DataType.INT32, strides=(1,)
    )
    request = KernelRequest(
        (ParameterSpec("x", tensor), ParameterSpec("out", tensor)), CompileTarget("a5", "ccec", "test")
    )
    prepared = CANNBotDSLProvider().prepare(
        {"aiv": copy_elements, "aic": cube_copy, "mixed": mixed_copy}[core_type], request
    )
    target = RuntimeTarget(
        "tmr",
        "1",
        request.target,
        frozenset({("cannbotdsl.device_callable", 1)}),
        frozenset(
            {
                *(f"core:{core.value}" for core in prepared.execution.cores),
                f"aiv_lanes:{prepared.execution.aiv_lanes}",
                "completion:entry_return_visible",
                "tensor_writeback",
                "partial_write",
            }
        ),
    )
    task = TMRBridge().lower(prepared, target)
    compiler = compiler_module.KernelCompiler("a5")
    compiler.project_root = Path(__file__).resolve().parents[3] / "runtime"
    compiler.sdk.ccec.get_compile_flags = lambda **kwargs: [
        "-std=c++17",
        "-O2",
        "-DNDEBUG",
        "-c",
        "-x",
        "cce",
        "--cce-aicore-only",
        f"--cce-aicore-arch=dav-c310-{'cube' if kwargs['core_type'] == 'aic' else 'vec'}",
    ]
    compiler.sdk.ccec.ascend_home_path = str(sdk)
    compiler.sdk.ccec.cxx_path = ccec
    compiler.sdk.ccec.linker_path = str(Path(ccec).resolve().with_name("ld.lld"))
    binary = compile_task(task, compiler, pto_isa_root=str(tmp_path))
    assert len(binary) == (2 if core_type == "mixed" else 1)
    assert all(binary)


@pytest.mark.parametrize("platform,core", [("a5sim", "aiv"), ("a2a3", "aiv"), ("a5", "aic")])
def test_vpto_rejects_unsupported_targets(compiler_module, tmp_path, platform, core):
    compiler = compiler_module.KernelCompiler(platform)
    with pytest.raises(ValueError, match="VPTO helpers require"):
        compiler.compile_incore(_source(tmp_path), core_type=core, vpto_sources=["helper.pto"])


@pytest.mark.parametrize(
    "helper_kind", ["empty", "ssa_writeback", "memory_attrs", "bridge", "simt_asm", "bridge_simt_asm"]
)
def test_vpto_compilation_links_helpers_and_rejects_bridge(
    compiler_module, monkeypatch, tmp_path, device_elf, helper_kind
):
    monkeypatch.setenv("LD_PRELOAD", "liballocator.so:")
    monkeypatch.setenv("LD_DEBUG", "libs")
    compiler = compiler_module.KernelCompiler("a5")
    monkeypatch.setattr(compiler, "_vpto_toolchain", lambda: ("ptoas", "bisheng"))
    # A load-dependent writeback needs no additional frontend synchronization
    # proof. Forward the LLVM produced by PTOAS unchanged to Bisheng.
    writeback = """declare <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6), i32, i32, i32)
declare void @llvm.hivm.vstsx1.v64f32(<64 x float>, ptr addrspace(6), i32, i32, i32, <256 x i1>)
define void @helper(ptr addrspace(6) %base, <256 x i1> %mask) {
  %acc = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %base, i32 0, i32 0, i32 0)
  %result = fadd <64 x float> %acc, %acc
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %result, ptr addrspace(6) %base,
    i32 0, i32 2, i32 0, <256 x i1> %mask)
  ret void
}
"""
    llvm = {
        "empty": "define void @helper() {\n  ret void\n}\n",
        "ssa_writeback": writeback,
        "memory_attrs": (
            "declare float @llvm.fabs.f32(float) #0\n"
            'attributes #0 = { nounwind memory(none) "comment"="memory(none)" }\n'
        ),
        "bridge": "declare void @bridge()",
        "simt_asm": (
            "define void @entry() { ret void }\n"
            'define simt_entry void @helper() {\n call void asm sideeffect "nop", ""()\n ret void\n}'
        ),
        "bridge_simt_asm": (
            'declare void @bridge()\ndefine simt_entry void @helper() { asm sideeffect "nop", ""() }'
        ),
    }[helper_kind]
    helpers = [tmp_path / "first.pto", tmp_path / "second.pto"]
    for helper in helpers:
        helper.write_text("module {}")
    commands = []

    def run(command, **kwargs):
        environment = kwargs.get("env")
        if environment is None:
            environment = os.environ
        assert environment["LD_PRELOAD"] == "liballocator.so:"
        assert environment["LD_DEBUG"] == "libs"
        commands.append(command)
        output = Path(command[command.index("-o") + 1])
        if command[0] == "ptoas":
            if "--emit-vpto-device-object" in command:
                assert "--emit-vpto-llvm-ir" not in command
                assert "--vpto-fix-vfsimt-size=auto" in command
                output.write_bytes(_native_simt_object(("entry.vector",)))
            else:
                output.write_text(llvm)
        else:
            if command[0] == "bisheng":
                expected_llvm = llvm.replace("nounwind memory(none) ", "nounwind ")
                assert Path(command[command.index("ir") + 1]).read_text() == expected_llvm
            if command[0] == "linker" and "-r" in command:
                assert "--defsym=entry=entry.vector" in command
            elif command[0] == "linker":
                order = next(
                    arg.split("=", 1)[1] for arg in command if arg.startswith("--symbol-ordering-file=")
                )
                assert Path(order).read_text() == "kernel_entry\n"
            output.write_bytes(device_elf() if command[0] == "linker" else b"object")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(compiler_module.subprocess, "run", run)
    kwargs = dict(pto_isa_root=str(tmp_path), build_dir=str(tmp_path), vpto_sources=list(map(str, helpers)))
    if helper_kind.startswith("bridge"):
        with pytest.raises(ValueError, match="not self-contained: bridge"):
            compiler.compile_incore(_source(tmp_path), **kwargs)
        assert not any(command[0] == "linker" for command in commands)
        assert not any("--emit-vpto-device-object" in command for command in commands)
    else:
        assert compiler.compile_incore(_source(tmp_path), **kwargs) == device_elf()
        helper_commands = ["ptoas", "ptoas", "linker"] if helper_kind == "simt_asm" else ["ptoas", "bisheng"]
        assert [command[0] for command in commands] == [
            "compiler",
            *helper_commands,
            *helper_commands,
            "linker",
        ]
        assert [Path(p).name for p in commands[-1][-3:]] == ["kernel.o", "vpto_0.o", "vpto_1.o"]
        if helper_kind != "simt_asm":
            assert "--cce-generic-addrspace=off" in commands[2]
        assert "-ffunction-sections" in commands[0]
    assert not list(tmp_path.glob("pypto-incore-*"))


def _native_simt_object(exports):
    data = _text_only_elf()
    struct.pack_into("<H", data, 16, 1)
    strings = b"\0" + b"".join(name.encode() + b"\0" for name in exports)
    data[160 : 160 + len(strings)] = strings
    struct.pack_into("<Q", data, 512 + 3 * 64 + 32, len(strings))
    data[248:296] = bytes(48)
    offset = 1
    for index, name in enumerate(exports):
        struct.pack_into("<IBBHQQ", data, 248 + index * 24, offset, 0x12, 0, 2, index * 8, 4)
        offset += len(name) + 1
    return bytes(data)


@pytest.mark.parametrize("native", ["entry.vector", "entry_mix_aiv", "entry"])
def test_simt_object_aliases_follow_emitted_abi(native):
    from pypto.runtime._vpto import simt_object_aliases  # noqa: PLC0415

    llvm = "define void @entry() { ret void }\ndefine simt_entry void @body() { ret void }"
    aliases = simt_object_aliases(llvm, _native_simt_object((native,)))
    assert aliases == (() if native == "entry" else (f"--defsym=entry={native}",))


@pytest.mark.parametrize("exports", [("unrelated.vector",), ("entry.vector", "entry_mix_aiv")])
def test_simt_object_aliases_reject_missing_or_ambiguous_definitions(exports):
    from pypto.runtime._vpto import simt_object_aliases  # noqa: PLC0415

    with pytest.raises(ValueError, match="needs one native definition"):
        simt_object_aliases("define void @entry() { ret void }", _native_simt_object(exports))


@pytest.mark.parametrize("info,visibility,section", [(0x02, 0, 2), (0x12, 2, 2), (0x12, 0, 0), (0x11, 0, 2)])
def test_simt_object_aliases_require_visible_defined_functions(info, visibility, section):
    from pypto.runtime._vpto import simt_object_aliases  # noqa: PLC0415

    data = bytearray(_native_simt_object(("entry.vector",)))
    struct.pack_into("<BBH", data, 252, info, visibility, section)
    with pytest.raises(ValueError, match="needs one native definition"):
        simt_object_aliases("define void @entry() { ret void }", bytes(data))


@pytest.mark.parametrize(
    "offset,fmt,value,error",
    [
        (16, "H", 2, "relocatable AI Core ELF"),
        (18, "H", 62, "relocatable AI Core ELF"),
        (824, "Q", 8, "invalid symbol table"),
        (808, "I", 99, "invalid symbol string table"),
        (248, "I", 99, "invalid string-table offset"),
        (254, "H", 99, "function outside its sections"),
    ],
)
def test_simt_object_rejects_invalid_exports(offset, fmt, value, error):
    from pypto.runtime._device_image import device_object_exports  # noqa: PLC0415

    data = bytearray(_native_simt_object(("entry.vector",)))
    struct.pack_into(f"<{fmt}", data, offset, value)
    with pytest.raises(ValueError, match=error):
        device_object_exports(bytes(data))


@pytest.mark.parametrize("stage", ["SIMT native object compilation", "SIMT logical export linking"])
def test_simt_object_failure_does_not_fall_back(compiler_module, monkeypatch, tmp_path, stage):
    compiler = compiler_module.KernelCompiler("a5")
    monkeypatch.setattr(compiler, "_vpto_toolchain", lambda: ("ptoas", "/selected/cann/bin/bisheng"))
    source = tmp_path / "helper.pto"
    source.write_text("module {}")
    labels = []

    def run(command, output, label, **kwargs):
        labels.append(label)
        if label == stage:
            raise RuntimeError("native compilation failed")
        if label == "VPTO lowering":
            output.write_text('define void @entry() { ret void }\ndefine simt_entry void @body() { asm "" }')
            return output.read_bytes()
        assert label == "SIMT native object compilation"
        assert kwargs["env"]["ASCEND_HOME_PATH"] == "/selected/cann"
        return _native_simt_object(("entry.vector",))

    monkeypatch.setattr(compiler, "_run", run)
    with pytest.raises(RuntimeError, match="native compilation failed"):
        compiler._compile_vpto_helpers((str(source),), tmp_path)
    assert labels[-1] == stage
    assert "VPTO device compilation" not in labels


@pytest.mark.parametrize("platform", ["a2a3", "a5"])
@pytest.mark.parametrize("core", ["aic", "aiv"])
@pytest.mark.parametrize("entry", [0x1000, 0x1008])
def test_native_entry_layout_without_vpto_helpers(
    compiler_module, monkeypatch, tmp_path, device_elf, platform, core, entry
):
    """An ordinary extern must never return an image that starts at a decoy."""
    image = device_elf(entry=entry)
    compiler = compiler_module.KernelCompiler(platform)
    monkeypatch.setattr(compiler, "_run", Mock(return_value=image))
    kwargs = dict(core_type=core, pto_isa_root=str(tmp_path), build_dir=str(tmp_path))
    if entry == 0x1000:
        assert compiler.compile_incore(_source(tmp_path), **kwargs) == image
    else:
        with pytest.raises(ValueError, match="kernel_entry.*start of .text") as error:
            compiler.compile_incore(_source(tmp_path), **kwargs)
        assert "0x1008" in str(error.value)
        assert "0x1000" in str(error.value)
        assert "offset=0x8" in str(error.value)
    assert not list(tmp_path.glob("pypto-incore-*"))


@pytest.mark.parametrize("kind", [1, 8])
def test_native_entry_check_preserves_block_local_sections(device_elf, kind):
    """Ordinary images need not satisfy VPTO's separate-data prohibition."""
    from pypto.runtime._device_image import validate_device_entry  # noqa: PLC0415

    image = device_elf(extra_flags=3, extra_kind=kind, extra_size=16)
    validate_device_entry(image)
    with pytest.raises(ValueError, match="allocated section"):
        validate_device_image(image)


@pytest.mark.parametrize(
    "damage",
    ["short", "host", "relocatable", "table", "strings", "name", "empty", "nobits", "nonexec", "duplicate"],
)
def test_device_image_rejects_invalid_entry_container(device_elf, damage):
    from pypto.runtime._device_image import validate_device_entry  # noqa: PLC0415

    data = bytearray(device_elf())
    (table,) = struct.unpack_from("<Q", data, 40)
    text = table + 128
    if damage == "short":
        data = data[:32]
    elif damage == "host":
        struct.pack_into("<H", data, 18, 183)  # EM_AARCH64, not AI Core.
    elif damage == "relocatable":
        struct.pack_into("<H", data, 16, 1)  # ET_REL has no executable entry.
    elif damage == "table":
        struct.pack_into("<Q", data, 40, len(data))
    elif damage == "strings":
        struct.pack_into("<Q", data, table + 64 + 24, len(data))
    elif damage == "name":
        struct.pack_into("<I", data, text, len(data))
    elif damage == "empty":
        struct.pack_into("<Q", data, text + 32, 0)
    elif damage == "nobits":
        struct.pack_into("<I", data, text + 4, 8)
    elif damage == "nonexec":
        struct.pack_into("<Q", data, text + 8, 2)
    else:
        data[table + 192 : table + 256] = data[text : text + 64]
    with pytest.raises(ValueError):
        validate_device_entry(bytes(data))


@pytest.mark.parametrize("with_decoy", [False, True])
def test_real_native_entry_layout(tmp_path, with_decoy):
    """Compile the issue's A5/AIV reduction without allocating or executing on an NPU."""
    pytest.importorskip("simpler_setup")
    if shutil.which("ccec") is None:
        pytest.skip("CCEC is required for the real device compiler")
    from pypto.runtime import pto_isa_include_dir  # noqa: PLC0415
    from pypto.runtime._device_image import validate_device_entry  # noqa: PLC0415
    from pypto.runtime.kernel_compiler import KernelCompiler  # noqa: PLC0415

    source = tmp_path / "entry_offset.cpp"
    decoy = 'extern "C" __aicore__ void decoy(__gm__ int64_t* args) { pipe_barrier(PIPE_ALL); }\n'
    source.write_text(
        "#include <pto/pto-inst.hpp>\n"
        + (decoy if with_decoy else "")
        + 'extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {\n'
        "  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);\n"
        "  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);\n"
        "  pipe_barrier(PIPE_ALL);\n}\n"
    )
    compiler = KernelCompiler("a5")
    pto_isa_root = str(Path(pto_isa_include_dir()).parent)
    if with_decoy:
        with pytest.raises(ValueError, match="kernel_entry.*offset=0x8"):
            compiler.compile_incore(str(source), core_type="aiv", pto_isa_root=pto_isa_root)
    else:
        validate_device_entry(
            compiler.compile_incore(str(source), core_type="aiv", pto_isa_root=pto_isa_root)
        )


def test_vpto_cache_tracks_contents_options_and_versions(compiler_module, monkeypatch, tmp_path):
    compiler = compiler_module.KernelCompiler("a5")
    monkeypatch.setattr(compiler, "_vpto_toolchain", lambda: ("ptoas", "bisheng"))
    monkeypatch.setattr(compiler_module, "check_ptoas_version", lambda _: "ptoas 0.70")
    version = SimpleNamespace(stdout="bisheng 1", stderr="")
    monkeypatch.setattr(compiler_module.subprocess, "run", lambda *a, **kw: version)
    helper = tmp_path / "helper.pto"
    helper.write_text("module {}")

    def key():
        return compiler.vpto_cache_identity((str(helper),), "aiv")

    identities = [key()]
    helper.write_text("module { /* changed */ }")
    identities.append(key())
    version.stdout = "bisheng 2"
    identities.append(key())
    monkeypatch.setattr(compiler_module._vpto, "DEVICE_FLAGS", ("-O3",))
    identities.append(key())
    monkeypatch.setattr(compiler_module._vpto, "CALLER_FLAGS", ())
    identities.append(key())
    monkeypatch.setattr(compiler_module._vpto, "LINK_SYMBOL_ORDER", ("another_entry",))
    identities.append(key())
    monkeypatch.setattr(compiler_module._vpto, "PTOAS_FLAGS", ("--enable-vecscope-mem-bar=true",))
    identities.append(key())
    monkeypatch.setattr(compiler_module._vpto, "SIMT_OBJECT_FLAGS", ("--emit-vpto-device-object",))
    identities.append(key())
    assert len(set(identities)) == 8


def test_vpto_identity_excludes_loader_diagnostics(compiler_module, monkeypatch, tmp_path):
    ignored = ("LD_PRELOAD", "LD_AUDIT", "LD_DEBUG", "LD_DEBUG_OUTPUT", "LD_PROFILE", "LD_PROFILE_OUTPUT")
    for name in ignored:
        monkeypatch.setenv(name, "runtime-only")
    monkeypatch.setenv("LD_LIBRARY_PATH", str(tmp_path / "libraries"))
    compiler = compiler_module.KernelCompiler("a5")
    monkeypatch.setattr(compiler, "_vpto_toolchain", lambda: ("ptoas", "bisheng"))
    monkeypatch.setattr(compiler_module, "check_ptoas_version", lambda _: "ptoas 0.72")
    calls = []

    def run(command, **kwargs):
        environment = kwargs.get("env")
        if environment is None:
            environment = os.environ
        calls.append(dict(environment))
        # Loader tracing includes a child PID, so inherited diagnostics would
        # give the cache lookup and publication different identities.
        diagnostics = f"{len(calls)}: loader trace" if environment.get("LD_DEBUG") else ""
        return SimpleNamespace(stdout="bisheng 1", stderr=diagnostics)

    monkeypatch.setattr(compiler_module.subprocess, "run", run)
    helper = tmp_path / "helper.pto"
    helper.write_text("module {}")
    first = compiler.vpto_cache_identity((str(helper),), "aiv")
    assert compiler.vpto_cache_identity((str(helper),), "aiv") == first
    assert len(calls) == 2
    for environment in calls:
        assert all(name not in environment for name in ignored)
        assert environment["LD_LIBRARY_PATH"] == str(tmp_path / "libraries")
    assert all(os.environ[name] == "runtime-only" for name in ignored)


def test_vpto_toolchain_preserves_relative_launcher_across_cwd(compiler_module, monkeypatch, tmp_path):
    caller = tmp_path / "caller"
    launcher = caller / "tools/ptoas"
    launcher.parent.mkdir(parents=True)
    # Keep this path-resolution test valid when the supported compiler advances.
    version_output = f"ptoas {PTOAS_MIN_VERSION.removeprefix('v')}"
    launcher.write_text(f"#!/bin/sh\nprintf '{version_output}\\n'\n")
    launcher.chmod(0o755)
    monkeypatch.chdir(caller)
    monkeypatch.setenv("PTOAS_ROOT", "tools")
    compiler = compiler_module.KernelCompiler("a5")
    ptoas, _ = compiler._vpto_toolchain()
    assert Path(ptoas).is_absolute()
    # The SDK cwd differs from discovery; execute the actual selected launcher.
    result = subprocess.run(
        [ptoas, "--version"], cwd=compiler.project_root, capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == version_output


def test_actual_compilation_inherits_runtime_injection(compiler_module, monkeypatch, tmp_path, device_elf):
    monkeypatch.setenv("LD_PRELOAD", "liballocator.so:")
    monkeypatch.setenv("LD_AUDIT", "libobserver.so")

    def run(command, **kwargs):
        # Metadata probes may normalize their environment; compilation may not.
        environment = kwargs.get("env")
        if environment is None:
            environment = os.environ
        assert environment["LD_PRELOAD"] == "liballocator.so:"
        assert environment["LD_AUDIT"] == "libobserver.so"
        Path(command[command.index("-o") + 1]).write_bytes(
            device_elf() if command[0] == "linker" else b"binary"
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(compiler_module.subprocess, "run", run)
    compiler = compiler_module.KernelCompiler("a2a3")
    assert compiler.compile_incore(_source(tmp_path), pto_isa_root=str(tmp_path)) == device_elf()


@pytest.mark.parametrize("effect", ["none", "read", "write", "argmem: read, inaccessiblemem: write"])
def test_bisheng_memory_annotations_preserve_other_llvm_text(effect):
    from pypto.runtime._vpto import prepare_llvm_for_bisheng  # noqa: PLC0415

    group = f'attributes #3 = {{ nounwind memory({effect}) "note"="memory(none)" }}'
    prefix = '; memory(none) in a comment\n@name = private constant [12 x i8] c"memory(none)"\n'
    text = prefix + group + "\n"
    expected = prefix + 'attributes #3 = { nounwind "note"="memory(none)" }\n'
    assert prepare_llvm_for_bisheng(text) == expected


def test_bisheng_removes_empty_attribute_groups_and_references():
    from pypto.runtime._vpto import prepare_llvm_for_bisheng  # noqa: PLC0415

    # Declarations, definitions and call sites may all reference the same group.
    # Group #10, quoted #1 text, and comments must survive removal of group #1.
    text = (
        "; keep #1 and #2 in comments\n"
        "declare void @callee() #1 #10\n"
        "define void @caller() #2 {\n"
        "  call void @callee() #1 ; keep #1 here too\n"
        "  ret void\n"
        "}\n"
        "attributes #1 = { memory(none) }\n"
        "attributes #2 = { memory(read) }\n"
        'attributes #10 = { nounwind memory(write) "note"="#1" }\n'
    )
    expected = (
        "; keep #1 and #2 in comments\n"
        "declare void @callee()  #10\n"
        "define void @caller()  {\n"
        "  call void @callee()  ; keep #1 here too\n"
        "  ret void\n"
        "}\n"
        "\n"
        "\n"
        'attributes #10 = { nounwind "note"="#1" }\n'
    )
    assert prepare_llvm_for_bisheng(text) == expected
    assert prepare_llvm_for_bisheng(expected) == expected


@pytest.mark.parametrize("policy", [None, "true", "false"])
def test_reg_auto_sync_flags(compiler_module, monkeypatch, tmp_path, policy):
    compiler = compiler_module.KernelCompiler("a5")
    monkeypatch.setattr(compiler, "_vpto_toolchain", lambda: ("ptoas", "bisheng"))
    helper = tmp_path / "helper.pto"
    helper.write_text((f"// pypto.reg_auto_sync: {policy}\n" if policy else "") + "module {}")
    commands = []

    def run(command, output, label):
        commands.append(command)
        output.write_text("define void @helper() { ret void }")
        return b"object"

    monkeypatch.setattr(compiler, "_run", run)
    compiler._compile_vpto_helpers((str(helper),), tmp_path)
    flags = [flag for flag in commands[0] if flag.startswith("--enable-vecscope-mem-bar=")]
    assert flags == [f"--enable-vecscope-mem-bar={'false' if policy == 'false' else 'true'}"]
    assert not any("mem-bar-all" in flag for flag in commands[0])


@pytest.mark.parametrize(
    "header",
    [
        "// pypto.reg_auto_sync: yes\n",
        "// pypto.reg_auto_sync:false\n",
        "// pypto.reg_auto_sync: false\n// pypto.reg_auto_sync: true\n",
        "module {}\n// pypto.reg_auto_sync: false\n",
    ],
)
def test_reg_auto_sync_rejects_invalid_markers(compiler_module, header):
    with pytest.raises(ValueError, match="reg_auto_sync"):
        compiler_module._vpto.ptoas_flags_for_source(header)


def test_reg_auto_sync_identity_and_compilation_share_snapshot(compiler_module, monkeypatch, tmp_path):
    compiler = compiler_module.KernelCompiler("a5")
    monkeypatch.setattr(compiler, "_vpto_toolchain", lambda: ("ptoas", "bisheng"))
    monkeypatch.setattr(compiler_module, "check_ptoas_version", lambda _: "ptoas 0.74")
    monkeypatch.setattr(
        compiler_module.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="bisheng", stderr="")
    )
    helper = tmp_path / "helper.pto"
    manual = "// pypto.reg_auto_sync: false\nmodule {}"
    helper.write_text(manual)
    snapshot = compiler_module._vpto.snapshot_sources((str(helper),))
    key = compiler.vpto_cache_identity((str(helper),), "aiv", snapshot=snapshot)
    helper.unlink()
    replacement = tmp_path / "replacement.pto"
    replacement.write_text("module {}")
    helper.symlink_to(replacement)
    commands = []
    compiled_sources = []

    def run(command, output, label):
        commands.append(command)
        if command[0] == "ptoas":
            compiled_sources.append(Path(command[1]).read_text())
        output.write_text("define void @helper() { ret void }")
        return b"object"

    monkeypatch.setattr(compiler, "_run", run)
    compiler._compile_vpto_helpers((str(helper),), tmp_path, snapshot)
    assert compiled_sources == [manual]
    assert "--enable-vecscope-mem-bar=false" in commands[0]
    # Identity snapshots are consumed once: a direct second compile sees edits.
    compiler._compile_vpto_helpers((str(helper),), tmp_path)
    assert compiled_sources[-1] == "module {}"
    assert "--enable-vecscope-mem-bar=true" in commands[2]
    assert compiler.vpto_cache_identity((str(helper),), "aiv") != key
    helper.write_text("// pypto.reg_auto_sync: invalid\nmodule {}")
    with pytest.raises(ValueError, match="reg_auto_sync"):
        compiler.vpto_cache_identity((str(helper),), "aiv")


def test_reg_auto_sync_snapshots_are_operation_local(compiler_module, monkeypatch, tmp_path):
    compiler = compiler_module.KernelCompiler("a5")
    monkeypatch.setattr(compiler, "_vpto_toolchain", lambda: ("ptoas", "bisheng"))
    monkeypatch.setattr(compiler_module, "check_ptoas_version", lambda _: "ptoas 0.74")
    monkeypatch.setattr(
        compiler_module.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="bisheng", stderr="")
    )
    helper = tmp_path / "shared.pto"
    first = "// pypto.reg_auto_sync: false\nmodule {}"
    second = "// pypto.reg_auto_sync: true\nmodule {}"
    first_identity_done = threading.Event()
    second_identity_done = threading.Event()
    compiled = {}

    def run(command, output, label):
        if command[0] == "ptoas":
            compiled[output.parent.name] = (Path(command[1]).read_text(), command)
        output.write_text("define void @helper() { ret void }")
        return b"object"

    monkeypatch.setattr(compiler, "_run", run)

    def compile_first():
        helper.write_text(first)
        snapshot = compiler_module._vpto.snapshot_sources((str(helper),))
        compiler.vpto_cache_identity((str(helper),), "aiv", snapshot=snapshot)
        first_identity_done.set()
        assert second_identity_done.wait(10)
        directory = tmp_path / "first"
        directory.mkdir()
        compiler._compile_vpto_helpers((str(helper),), directory, snapshot)

    def compile_second():
        assert first_identity_done.wait(10)
        helper.write_text(second)
        snapshot = compiler_module._vpto.snapshot_sources((str(helper),))
        compiler.vpto_cache_identity((str(helper),), "aiv", snapshot=snapshot)
        second_identity_done.set()
        directory = tmp_path / "second"
        directory.mkdir()
        compiler._compile_vpto_helpers((str(helper),), directory, snapshot)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(compile_first), executor.submit(compile_second)]
        for future in futures:
            future.result(timeout=20)
    # Both requests use the same path and compiler, but retain their own identity snapshot.
    assert compiled["first"][0] == first
    assert "--enable-vecscope-mem-bar=false" in compiled["first"][1]
    assert compiled["second"][0] == second
    assert "--enable-vecscope-mem-bar=true" in compiled["second"][1]


def test_reg_auto_sync_rejects_actual_simt_entry(compiler_module):
    source = "// pypto.reg_auto_sync: false\nmodule { func.func @helper() attributes {pto.simt_entry} {} }"
    with pytest.raises(ValueError, match="SIMT helpers"):
        compiler_module._vpto.ptoas_flags_for_source(source)


def test_reg_auto_sync_accepts_vf_name_containing_simt_entry(compiler_module):
    source = (
        "// pypto.reg_auto_sync: false\n"
        "// A comment mentioning pto.simt_entry is not an entry attribute.\n"
        'module { func.func @__pypto_vf_simt_entry_0() attributes {note = "pto.simt_entry"} {} }'
    )
    assert "--enable-vecscope-mem-bar=false" in compiler_module._vpto.ptoas_flags_for_source(source)


def test_ccu_host_module_uses_sdk_identity_and_preserves_elf(compiler_module, monkeypatch, tmp_path):
    """The SDK owns discovery; PyPTO invokes its command and retains Host ELF bytes."""
    compiler = compiler_module.KernelCompiler("a5")
    header = "// manifest identities supplied by Simpler\n"
    inputs = {
        "cxx_path": "sdk-ccu-compiler",
        "compile_flags": ["-fPIC", "-DSDK_CCU_FLAG"],
        "include_dirs": [str(tmp_path / "sdk include")],
        "link_flags": ["-shared", "-Wl,--build-id=sha1"],
        "library_paths": [str(tmp_path / "sdk lib/libccu.so"), str(tmp_path / "sdk lib/libhcomm.so")],
        "manifest_identity_header": header,
    }
    compiler.sdk.ccu_compile_cache_token = Mock(return_value=inputs)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        output = Path(command[command.index("-o") + 1])
        assert (output.parent / "ccu_manifest_identity.h").read_text() == header
        assert kwargs["timeout"] == 900
        output.write_bytes(b"complete host ELF")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(compiler_module.subprocess, "run", run)
    source = _source(tmp_path)
    assert compiler.compile_ccu(source) == b"complete host ELF"
    compiler.sdk.ccu_compile_cache_token.assert_called_once_with()
    assert len(commands) == 1
    command = commands[0]
    assert command[0:3] == ["sdk-ccu-compiler", "-fPIC", "-DSDK_CCU_FLAG"]
    assert f"-I{tmp_path / 'sdk include'}" in command
    assert command[command.index(source) + 1 : -2] == [*inputs["link_flags"], *inputs["library_paths"]]
    assert not Path(command[-1]).parent.exists()


@pytest.mark.parametrize("platform", ["a2a3", "a2a3sim", "a5sim"])
def test_ccu_rejects_unsupported_platform_before_discovery(compiler_module, tmp_path, platform):
    compiler = compiler_module.KernelCompiler(platform)
    compiler.sdk.ccu_compile_cache_token = Mock()
    with pytest.raises(ValueError, match="require a5"):
        compiler.compile_ccu(_source(tmp_path))
    compiler.sdk.ccu_compile_cache_token.assert_not_called()


@pytest.mark.parametrize("failure", ["compiler", "timeout", "empty"])
def test_ccu_failed_build_cleans_private_directory(compiler_module, monkeypatch, tmp_path, failure):
    compiler = compiler_module.KernelCompiler("a5")
    compiler.sdk.ccu_compile_cache_token = Mock(
        return_value={
            "cxx_path": "ccu-compiler",
            "compile_flags": [],
            "include_dirs": [],
            "link_flags": [],
            "library_paths": [],
            "manifest_identity_header": "// identities",
        }
    )
    outputs = []

    def run(command, **kwargs):
        outputs.append(Path(command[-1]))
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return SimpleNamespace(returncode=1 if failure == "compiler" else 0, stderr="CCU compile error")

    monkeypatch.setattr(compiler_module.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="CCU"):
        compiler.compile_ccu(_source(tmp_path))
    assert len(outputs) == 1
    assert not outputs[0].parent.exists()


def test_ccu_sanitizer_selection_is_shared_with_sdk(compiler_module, monkeypatch):
    monkeypatch.setattr(compiler_module.KernelCompiler, "_sanitizers", "address")
    compiler = compiler_module.KernelCompiler("a5")
    assert compiler.sdk._sanitizers == "address"
    assert compiler.sdk.ccu.host_toolchain is compiler.sdk.host_gxx


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
