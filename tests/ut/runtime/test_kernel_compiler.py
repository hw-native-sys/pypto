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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
