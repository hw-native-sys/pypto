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
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


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
def test_incore_build_and_link_are_owned_by_pypto(compiler_module, monkeypatch, tmp_path, platform):
    """Verify target-specific compiler commands, device linking and output cleanup."""
    commands = []

    def run(command, **kwargs):
        """Record invocations and emit distinct compiler and linker outputs."""
        commands.append(command)
        assert kwargs["cwd"] == tmp_path
        assert kwargs["timeout"] == 900
        Path(command[command.index("-o") + 1]).write_bytes(b"linked" if command[0] == "linker" else b"object")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(compiler_module.subprocess, "run", run)
    compiler = compiler_module.KernelCompiler(platform)
    result = compiler.compile_incore(_source(tmp_path), pto_isa_root=str(tmp_path), build_dir=str(tmp_path))
    assert result == (b"object" if platform.endswith("sim") else b"linked")
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
def test_sanitizer_flags_apply_only_to_host_builds(compiler_module, monkeypatch, tmp_path, platform):
    """Keep sanitizer instrumentation on host builds and out of device compiler commands."""
    monkeypatch.setattr(compiler_module.KernelCompiler, "_sanitizers", "address")
    compiler = compiler_module.KernelCompiler(platform)
    commands = []

    def run(command, **kwargs):
        """Record target flags while supplying a successful build output."""
        commands.append(command)
        Path(command[command.index("-o") + 1]).write_bytes(b"binary")
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


def _vpto_elf(*, entry=0x1000, extra_flags=0, extra_kind=1, extra_size=0):
    """Construct a minimal linked image to exercise loader constraints without CANN."""
    data = bytearray(384)
    data[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<HH", data, 16, 2, 0x1029)
    struct.pack_into("<Q", data, 24, entry)
    struct.pack_into("<Q", data, 40, 128)
    struct.pack_into("<HHH", data, 58, 64, 4, 1)
    names = b"\0.shstrtab\0.text\0.extra\0"
    data[64 : 64 + len(names)] = names
    struct.pack_into("<IIQQQQIIQQ", data, 192, 1, 3, 0, 0, 64, len(names), 0, 0, 1, 0)
    struct.pack_into("<IIQQQQIIQQ", data, 256, 11, 1, 6, 0x1000, 96, 4, 0, 0, 4, 0)
    struct.pack_into(
        "<IIQQQQIIQQ",
        data,
        320,
        17,
        extra_kind,
        extra_flags,
        0x2000,
        100,
        extra_size,
        0,
        0,
        4,
        24 if extra_kind == 2 else 0,
    )
    if extra_kind == 2:
        struct.pack_into("<IBBH", data, 100, 1, 0, 0, 0)
    return bytes(data)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"entry": 0x1010}, "start of .text"),
        ({"extra_flags": 2, "extra_size": 4}, "allocated section"),
        ({"extra_kind": 4, "extra_size": 4}, "relocations"),
        ({"extra_kind": 2, "extra_size": 24}, "unresolved symbol"),
    ],
)
def test_vpto_rejects_images_that_text_loader_cannot_load(kwargs, error):
    from pypto.runtime._vpto import validate_device_image  # noqa: PLC0415

    validate_device_image(_vpto_elf())
    with pytest.raises(ValueError, match=error):
        validate_device_image(_vpto_elf(**kwargs))


@pytest.mark.parametrize("platform,core", [("a5sim", "aiv"), ("a2a3", "aiv"), ("a5", "aic")])
def test_vpto_rejects_unsupported_targets(compiler_module, tmp_path, platform, core):
    compiler = compiler_module.KernelCompiler(platform)
    with pytest.raises(ValueError, match="VPTO helpers require"):
        compiler.compile_incore(_source(tmp_path), core_type=core, vpto_sources=["helper.pto"])


@pytest.mark.parametrize("bridge", [False, True])
def test_vpto_compilation_links_helpers_and_rejects_bridge(compiler_module, monkeypatch, tmp_path, bridge):
    compiler = compiler_module.KernelCompiler("a5")
    monkeypatch.setattr(compiler, "_vpto_toolchain", lambda: ("ptoas", "bisheng"))
    helpers = [tmp_path / "first.pto", tmp_path / "second.pto"]
    for helper in helpers:
        helper.write_text("module {}")
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        output = Path(command[command.index("-o") + 1])
        if command[0] == "ptoas":
            output.write_text(
                "declare void @bridge()" if bridge else "define void @helper() {\n  ret void\n}\n"
            )
        else:
            if command[0] == "linker":
                order = next(
                    arg.split("=", 1)[1] for arg in command if arg.startswith("--symbol-ordering-file=")
                )
                assert Path(order).read_text() == "kernel_entry\n"
            output.write_bytes(_vpto_elf() if command[0] == "linker" else b"object")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(compiler_module.subprocess, "run", run)
    kwargs = dict(pto_isa_root=str(tmp_path), build_dir=str(tmp_path), vpto_sources=list(map(str, helpers)))
    if bridge:
        with pytest.raises(ValueError, match="not self-contained: bridge"):
            compiler.compile_incore(_source(tmp_path), **kwargs)
        assert not any(command[0] == "linker" for command in commands)
    else:
        assert compiler.compile_incore(_source(tmp_path), **kwargs) == _vpto_elf()
        assert [command[0] for command in commands] == [
            "compiler",
            "ptoas",
            "bisheng",
            "ptoas",
            "bisheng",
            "linker",
        ]
        assert [Path(p).name for p in commands[-1][-3:]] == ["kernel.o", "vpto_0.o", "vpto_1.o"]
        assert "--cce-generic-addrspace=off" in commands[2]
        assert "-ffunction-sections" in commands[0]
    assert not list(tmp_path.glob("pypto-incore-*"))


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
    monkeypatch.setattr(compiler_module._vpto_sync, "VERSION", "next-contract")
    identities.append(key())
    assert len(set(identities)) == 7


def test_vpto_toolchain_preserves_relative_launcher_across_cwd(compiler_module, monkeypatch, tmp_path):
    caller = tmp_path / "caller"
    launcher = caller / "tools/ptoas"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("#!/bin/sh\nprintf 'ptoas 0.70\\n'\n")
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
    assert result.stdout.strip() == "ptoas 0.70"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
