# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Build identities follow effective imports and selected dependency versions."""

import ast
import hashlib
import json
import os
import runpy
import struct
import subprocess
import sys
import venv
from pathlib import Path
from types import SimpleNamespace

import pytest
from pypto._identity import InstallationIdentityCache
from pypto.jit import _build_identity as identity
from pypto.jit import _toolchain


def _elf(path, build_id):
    header = bytearray(64)
    header[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<Q", header, 32, 64)
    struct.pack_into("<HH", header, 54, 56, 1)
    note = struct.pack("<III", 4, len(build_id), 3) + b"GNU\0" + build_id
    program = bytearray(56)
    struct.pack_into("<I", program, 0, 4)
    struct.pack_into("<Q", program, 8, 120)
    struct.pack_into("<Q", program, 32, len(note))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + program + note)
    path.chmod(0o755)
    return path


def test_full_native_build_id_and_missing_id_fallback(tmp_path):
    path = _elf(tmp_path / "core.so", bytes(range(20)))
    assert identity.native_build_id(path) == ("gnu-build-id", bytes(range(20)).hex())
    # Changing bytes beyond the truncated trace ID still invalidates a build.
    _elf(path, bytes(range(19)) + b"x")
    assert identity.native_build_id(path)[1] != bytes(range(20)).hex()
    path.write_bytes(b"no build id")
    assert identity.native_build_id(path) == ("sha256", hashlib.sha256(path.read_bytes()).hexdigest())


def test_python_edit_with_preserved_size_and_mtime_changes_identity(tmp_path):
    path = tmp_path / "module.py"
    path.write_text("value = 1\n")
    first = identity.python_sources(tmp_path)
    stamp = path.stat()
    path.write_text("value = 2\n")
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    identity.python_sources.cache_clear()  # A new process starts with an empty memo.
    assert identity.python_sources(tmp_path) != first


@pytest.mark.parametrize("name", ["kernel.cpp.in", "kernel_config.py.in"])
def test_bundled_template_edit_changes_identity(tmp_path, name):
    (tmp_path / "__init__.py").write_text("pass\n")
    template = tmp_path / "templates" / name
    template.parent.mkdir()
    template.write_text("template = 1\n")
    first = identity.python_sources(tmp_path)
    stamp = template.stat()
    template.write_text("template = 2\n")
    os.utime(template, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    identity.python_sources.cache_clear()
    assert identity.python_sources(tmp_path) != first


def test_python_directory_cycle_is_unavailable(tmp_path):
    (tmp_path / "module.py").write_text("pass")
    (tmp_path / "cycle").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        identity.python_sources(tmp_path)


@pytest.fixture
def selected_build(tmp_path, monkeypatch):
    pypto = tmp_path / "checkout/pypto"
    pypto.mkdir(parents=True)
    (pypto / "__init__.py").write_text("pass")
    native = _elf(tmp_path / "wheel/pypto/core.so", b"p" * 20)
    runtime = _elf(tmp_path / "wheel/runtime.so", b"r" * 20)
    root = tmp_path / "runtime"
    root.mkdir()
    (root / "pto_isa.pin").write_text("a" * 40)
    cxx = _elf(tmp_path / "cxx", b"c" * 20)
    tools = {
        name: _elf(tmp_path / name, name.encode().ljust(20, b"x"))
        for name in ("cc1plus", "collect2", "as", "ld")
    }
    monkeypatch.setattr(
        identity,
        "_module_path",
        lambda name: {
            "pypto.pypto_core": native,
            "_task_interface": runtime,
        }[name],
    )
    monkeypatch.setattr(identity, "_python_package", lambda name: identity.python_sources(pypto))
    monkeypatch.setitem(sys.modules, "_task_interface", SimpleNamespace(__build_commit__="r1"))
    monkeypatch.setitem(
        sys.modules,
        "simpler_setup.pto_isa",
        SimpleNamespace(
            read_pto_isa_pin=lambda path: path.read_text().strip(),
        ),
    )
    monkeypatch.setattr(_toolchain, "_invocable", lambda path, **kwargs: Path(path))

    def run(args, **kwargs):
        if args[1] == "-E":
            return f"COLLECT_GCC={cxx}\n"
        if args[1].startswith("-print-prog-name="):
            return str(tools[args[1].partition("=")[2]])
        return "compiler version 1"

    monkeypatch.setattr(_toolchain, "_run", run)
    monkeypatch.setattr(identity, "_ptoas_identity", lambda selected: selected)
    compiler = SimpleNamespace(
        project_root=root,
        platform="a2a3sim",
        _sanitizers="",
        _orchestration_toolchain=lambda name: SimpleNamespace(cxx_path=str(cxx)),
        sdk=SimpleNamespace(gxx15=SimpleNamespace(cxx_path=str(cxx))),
    )
    return SimpleNamespace(compiler=compiler, root=root, native=native, runtime=runtime, tools=tools)


def _capture(build, assembler="ptoas-1"):
    return InstallationIdentityCache().capture(
        identity.discover_builds(lambda: build.compiler, assembler, "runtime")
    )


def test_editable_native_build_and_effective_dependencies_invalidate(selected_build, monkeypatch):
    build = selected_build
    first = _capture(build)
    assert first.usable
    # Native extension is outside the source tree, as with scikit-build editable installs.
    _elf(build.native, b"q" * 20)
    assert _capture(build).pypto != first.pypto
    _elf(build.runtime, b"s" * 20)
    assert _capture(build).runtime != first.runtime
    (build.root / "pto_isa.pin").write_text("b" * 40)
    assert _capture(build).pto_isa != first.pto_isa
    assert _capture(build, "ptoas-2").ptoas != first.ptoas
    # No ISA checkout exists: a READY hit only needs the resolver's selected version.
    assert not (build.root / "build/pto-isa").exists()
    monkeypatch.setitem(sys.modules, "_task_interface", SimpleNamespace(__build_commit__="r2"))
    assert _capture(build).runtime != first.runtime


@pytest.mark.parametrize("change", ["build_id", "content_fallback", "resolved_path"])
def test_running_interpreter_replacement_changes_identity(selected_build, tmp_path, monkeypatch, change):
    interpreter = _elf(tmp_path / "python", b"i" * 20)
    if change == "content_fallback":
        interpreter.write_bytes(b"interpreter build one")
    monkeypatch.setattr(sys, "executable", str(interpreter))
    first = _capture(selected_build)
    assert first.usable
    if change == "resolved_path":
        replacement = _elf(tmp_path / "other/python", b"i" * 20)
        monkeypatch.setattr(sys, "executable", str(replacement))
    elif change == "build_id":
        _elf(interpreter, b"j" * 20)
    else:
        interpreter.write_bytes(b"interpreter build two")
    # sys.version, cache_tag and all PyPTO files remain unchanged.
    second = _capture(selected_build)
    assert second.usable
    assert second.pypto != first.pypto
    assert second.runtime == first.runtime


def test_running_interpreter_symlink_alias_preserves_identity(selected_build, tmp_path, monkeypatch):
    interpreter = _elf(tmp_path / "python", b"i" * 20)
    monkeypatch.setattr(sys, "executable", str(interpreter))
    first = _capture(selected_build)
    alias = tmp_path / "python-alias"
    alias.symlink_to(interpreter)
    monkeypatch.setattr(sys, "executable", str(alias))
    assert _capture(selected_build).pypto == first.pypto


@pytest.mark.parametrize("program", ["as", "ld"])
def test_selected_gcc_helper_rebuild_changes_identity(selected_build, program):
    first = _capture(selected_build)
    _elf(selected_build.tools[program], b"z" * 20)
    assert _capture(selected_build).device_toolchain != first.device_toolchain


def test_unknown_elf_wrapper_does_not_get_a_build_identity(selected_build, monkeypatch, tmp_path):
    wrapper = _elf(tmp_path / "untracked-wrapper", b"w" * 20)
    real_driver = _elf(tmp_path / "real-gcc", b"g" * 20)

    def run(args, **kwargs):
        if args[1] == "-E":
            return f"COLLECT_GCC={real_driver}\n"
        if args[1].startswith("-print-prog-name="):
            return str(selected_build.tools[args[1].partition("=")[2]])
        return "compiler version 1"

    monkeypatch.setattr(_toolchain, "_run", run)
    with pytest.raises(ValueError, match="untracked selection inputs"):
        identity._compiler_identity(str(wrapper))


@pytest.mark.parametrize("kind", ["script", "script_symlink", "non_executable_elf"])
def test_gcc_wrapper_rejects_unsupported_executed_driver(selected_build, monkeypatch, tmp_path, kind):
    wrapper = _elf(tmp_path / "ccache", b"w" * 20)
    driver = _elf(tmp_path / "driver", b"d" * 20)
    if kind == "non_executable_elf":
        driver.chmod(0o644)
        reason = "Compiler executable is unavailable"
    else:
        driver.write_text('#!/bin/sh\nexec real-gcc "$@"\n')
        reason = "Unsupported compiler launcher"
        if kind == "script_symlink":
            alias = tmp_path / "driver-alias"
            alias.symlink_to(driver)
            driver = alias
    original_run = _toolchain._run

    def run(args, **kwargs):
        if args[1] == "-E":
            return f"COLLECT_GCC={driver}\n"
        return original_run(args, **kwargs)

    monkeypatch.setattr(_toolchain, "_run", run)
    with pytest.raises(ValueError, match=reason):
        identity._compiler_identity(str(wrapper))


def test_accepted_wrapper_rebuild_changes_compiler_identity(selected_build, tmp_path):
    wrapper = _elf(tmp_path / "ccache", b"w" * 20)
    first = identity._compiler_identity(str(wrapper))
    _elf(wrapper, b"v" * 20)
    assert identity._compiler_identity(str(wrapper)) != first


def test_dirty_runtime_checkout_has_no_build_identity(selected_build, monkeypatch):
    root = selected_build.root
    header = root / "runtime.h"
    header.write_text("original\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "pto_isa.pin", "runtime.h"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=PyPTO",
            "-c",
            "user.email=pypto@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    compiler_run = _toolchain._run

    def run(args, **kwargs):
        if args[0] == "git":
            result = subprocess.run(args, check=True, capture_output=True, text=True)
            return result.stdout + result.stderr
        return compiler_run(args, **kwargs)

    monkeypatch.setattr(_toolchain, "_run", run)
    assert _capture(selected_build).usable
    header.write_text("edited\n")
    with pytest.raises(ValueError, match="uncommitted changes"):
        _capture(selected_build)
    header.write_text("original\n")
    (root / "new_helper.py").write_text("pass\n")
    with pytest.raises(ValueError, match="uncommitted changes"):
        _capture(selected_build)


def test_benchmark_creates_report_directory(tmp_path, monkeypatch):
    benchmark = runpy.run_path(str(Path(__file__).resolve().parents[2] / "benchmarks/jit_cache_latency.py"))
    output = tmp_path / "nested/report.json"
    sample = {
        "identity_ms": 1,
        "warmup_ms": 2,
        "total_ms": 3,
        "stats": {
            "ready_hits": 1,
            "misses": 0,
            "bypasses": 0,
            "generation_builds": 0,
            "binary_builds": 0,
            "invalid_entries": 0,
            "storage_errors": 0,
        },
    }
    monkeypatch.setattr(
        sys,
        "argv",
        ["jit_cache_latency.py", "--cache-root", str(tmp_path), "--runs", "1", "--output", str(output)],
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=json.dumps(sample), stderr=""),
    )
    benchmark["main"]()
    assert json.loads(output.read_text())["summary_ms"]["total_ms"]["median"] == 3


def test_policy_and_epoch_changes_do_not_reuse_process_memo(selected_build, monkeypatch):
    monkeypatch.setattr(_toolchain, "_compiler", lambda *args: selected_build.compiler)
    monkeypatch.setattr(_toolchain, "_identities", {})
    monkeypatch.setattr(_toolchain, "find_ptoas_binary", lambda: "ptoas-1")
    monkeypatch.setitem(
        sys.modules,
        "pypto.runtime.kernel_compiler",
        SimpleNamespace(
            KernelCompiler=SimpleNamespace(_sanitizers=""),
        ),
    )
    monkeypatch.setenv("PYPTO_CACHE_IDENTITY", "build")
    first = _toolchain.capture_toolchain("a2a3sim", "runtime")
    assert first.usable
    monkeypatch.setenv("PYPTO_CACHE_EPOCH", "patched-sdk")
    assert _toolchain.capture_toolchain("a2a3sim", "runtime").digest != first.digest
    monkeypatch.setenv("PYPTO_CACHE_IDENTITY", "invalid")
    assert not _toolchain.capture_toolchain("a2a3sim", "runtime").usable


@pytest.mark.parametrize("selection", ["cwd", "PATH"])
def test_search_change_that_selects_another_tool_changes_identity(
    selected_build, monkeypatch, tmp_path, selection
):
    monkeypatch.setenv("PYPTO_CACHE_IDENTITY", "build")
    monkeypatch.setattr(_toolchain, "_compiler", lambda *args: selected_build.compiler)
    monkeypatch.setattr(_toolchain, "_identities", {})
    monkeypatch.setitem(
        sys.modules,
        "pypto.runtime.kernel_compiler",
        SimpleNamespace(KernelCompiler=SimpleNamespace(_sanitizers="")),
    )
    first_directory = tmp_path / "first"
    second_directory = tmp_path / "second"
    first_directory.mkdir()
    second_directory.mkdir()
    monkeypatch.chdir(first_directory)
    monkeypatch.setenv("PATH", str(first_directory))
    selections = []

    def find():
        selected = str(Path.cwd()) if selection == "cwd" else os.environ["PATH"]
        selections.append(selected)
        return selected

    monkeypatch.setattr(_toolchain, "find_ptoas_binary", find)
    first = _toolchain.capture_toolchain("a2a3sim", "runtime")
    if selection == "cwd":
        monkeypatch.chdir(second_directory)
    else:
        monkeypatch.setenv("PATH", str(second_directory))
    second = _toolchain.capture_toolchain("a2a3sim", "runtime")
    assert first.usable and second.usable
    assert first.pypto == second.pypto
    assert first.ptoas != second.ptoas and first.digest != second.digest
    assert selections == [str(first_directory), str(second_directory)]


@pytest.mark.parametrize(
    "setting", ["PYPTO_CACHE_IDENTITY", "PYPTO_CACHE_EPOCH", "PYTHONOPTIMIZE", "LD_LIBRARY_PATH"]
)
def test_policy_epoch_and_execution_overrides_remain_in_identity(monkeypatch, setting):
    component = _toolchain.ComponentInputs(unavailable_reason=None, reported_version="same-build")
    inputs = _toolchain.ToolchainInputs(component, component, component, component, component)
    monkeypatch.setenv("PYPTO_CACHE_IDENTITY", "build")
    monkeypatch.delenv("PYPTO_CACHE_EPOCH", raising=False)
    monkeypatch.delenv("PYTHONOPTIMIZE", raising=False)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.setattr(_toolchain, "_identities", {})
    monkeypatch.setattr(_toolchain, "find_ptoas_binary", lambda: "ptoas")
    monkeypatch.setattr(_toolchain, "_selected_compiler", lambda platform: None)
    monkeypatch.setattr(identity, "discover_builds", lambda *args: inputs)
    monkeypatch.setattr(_toolchain, "_discover", lambda *args: inputs)
    first = _toolchain.capture_toolchain("a2a3sim", "runtime")
    monkeypatch.setenv(setting, "content" if setting == "PYPTO_CACHE_IDENTITY" else "1")
    second = _toolchain.capture_toolchain("a2a3sim", "runtime")
    assert first.usable and second.usable and first.digest != second.digest


def test_relative_compiler_search_is_resolved_again_after_chdir(monkeypatch, tmp_path):
    class Compiler:
        _sanitizers = ""

        def __init__(self, platform):
            self.selected_path = Path("bin/c++").absolute()

    monkeypatch.setitem(
        sys.modules, "pypto.runtime.kernel_compiler", SimpleNamespace(KernelCompiler=Compiler)
    )
    monkeypatch.setenv("PATH", "bin")
    monkeypatch.chdir(tmp_path)
    _toolchain._compiler.cache_clear()
    try:
        first = _toolchain._selected_compiler("a2a3sim")
        assert _toolchain._selected_compiler("a2a3sim") is first
        other = tmp_path / "other"
        other.mkdir()
        monkeypatch.chdir(other)
        second = _toolchain._selected_compiler("a2a3sim")
        assert first.selected_path == tmp_path / "bin/c++"
        assert second.selected_path == other / "bin/c++"
    finally:
        _toolchain._compiler.cache_clear()


def test_ptoas_probe_uses_script_directory_before_importing_helpers(tmp_path):
    launcher_dir = tmp_path / "bin"
    package = launcher_dir / "ptoas"
    package.mkdir(parents=True)
    origin = package / "__init__.py"
    origin.write_text("raise AssertionError('must not import compiler')")
    # A -c probe starts with cwd on sys.path; the actual console script does not.
    (tmp_path / "json.py").write_text("raise AssertionError('must not import cwd helpers')")
    result = subprocess.run(
        [sys.executable, "-S", "-c", identity._PTOAS_PROBE, str(launcher_dir)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": ""},
        check=True,
        capture_output=True,
        text=True,
    )
    record = ast.literal_eval(result.stdout)
    assert record["ptoas"] == str(origin)
    assert record["interpreter_version"] == sys.version
    assert record["interpreter_cache_tag"] == sys.implementation.cache_tag
    assert isinstance(record["startup_hooks"], list)
    assert isinstance(record["startup_modules"], list)
    assert record["startup_unresolved"] == []


def test_ptoas_probe_records_selected_sitecustomize(tmp_path):
    hook = tmp_path / "sitecustomize.py"
    hook.write_text("sentinel = 1\n")
    result = subprocess.run(
        [sys.executable, "-c", identity._PTOAS_PROBE, str(tmp_path)],
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        check=True,
        capture_output=True,
        text=True,
    )
    record = ast.literal_eval(result.stdout)
    assert ("sitecustomize", str(hook)) in record["startup_modules"]


def test_unidentifiable_ptoas_startup_module_has_no_identity():
    with pytest.raises(ValueError, match="startup evidence is unavailable"):
        identity._startup_identity(
            {
                "startup_hooks": [],
                "startup_modules": [],
                "startup_path": [],
                "startup_unresolved": ["dynamic_hook"],
            }
        )


@pytest.fixture
def selected_ptoas_wheel(tmp_path, monkeypatch):
    launcher = tmp_path / "bin/ptoas"
    launcher.parent.mkdir()
    interpreter = _elf(tmp_path / "selected/python", b"i" * 20)
    launcher.write_text(f"#!{interpreter}\n")
    package = tmp_path / "selected/site-packages/ptoas"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("raise AssertionError('must not import')")
    core = _elf(package / "_core.so", b"a" * 20)
    metadata = package.parent / "ptoas-0.65.dev1.dist-info/METADATA"
    metadata.parent.mkdir()
    metadata.write_text("Name: ptoas\nVersion: 0.65.dev1\nRequires-Dist: numpy\n")
    for name in ("WHEEL", "RECORD"):
        (metadata.parent / name).write_text(name)
    package_origins = {}
    for name in ("ptoas", "ptodsl", "TileOps", "SoftOps"):
        root = package.parent / name
        root.mkdir(exist_ok=True)
        (root / "__init__.py").write_text("raise AssertionError('must not import compiler packages')")
        package_origins[name] = (str(root / "__init__.py"), [str(root)])
    package_evidence = {"ptoas_packages": package_origins, "loaded_ptoas_packages": []}
    numpy = package.parent / "numpy/__init__.py"
    numpy.parent.mkdir()
    numpy.write_text("raise AssertionError('must not import NumPy')")
    numpy_metadata = package.parent / "numpy-2.2.6.dist-info"
    numpy_metadata.mkdir()
    for name in ("METADATA", "WHEEL", "RECORD"):
        (numpy_metadata / name).write_text("Name: numpy\nVersion: 2.2.6\n" if name == "METADATA" else name)
    startup_hook = tmp_path / "selected/site-packages/startup.pth"
    startup_hook.write_text("/original/package/path\n")
    startup_module = tmp_path / "selected/site-packages/sitecustomize.py"
    startup_module.write_text("value = 1\n")
    calls = []
    monkeypatch.setattr(_toolchain, "_console_interpreter", lambda path: interpreter)
    interpreter_evidence = {
        "interpreter_version": "3.10.19 (selected build)",
        "interpreter_cache_tag": "cpython-310",
    }

    def probe(command):
        calls.append(command)
        return repr(
            {
                **interpreter_evidence,
                **package_evidence,
                "ptoas": str(package / "__init__.py"),
                "numpy": str(numpy),
                "startup_hooks": [str(startup_hook)],
                "startup_modules": [["startup_module", str(startup_module)]],
                "startup_path": [str(launcher.parent), str(package.parent)],
                "startup_unresolved": [],
            }
        )

    monkeypatch.setattr(_toolchain, "_run", probe)
    return SimpleNamespace(
        launcher=launcher,
        interpreter=interpreter,
        interpreter_evidence=interpreter_evidence,
        package_evidence=package_evidence,
        calls=calls,
        package=package,
        core=core,
        metadata=metadata,
        numpy_metadata=numpy_metadata,
        startup_hook=startup_hook,
        startup_module=startup_module,
    )


def test_ptoas_resolves_selected_interpreter_without_importing_compiler(selected_ptoas_wheel):
    wheel = selected_ptoas_wheel
    launcher, core, metadata = wheel.launcher, wheel.core, wheel.metadata
    package, numpy_metadata = wheel.package, wheel.numpy_metadata
    startup_hook, startup_module = wheel.startup_hook, wheel.startup_module
    first = identity._ptoas_identity(str(launcher))
    assert wheel.calls[0][0] == str(wheel.interpreter)
    _elf(core, b"b" * 20)
    assert identity._ptoas_identity(str(launcher)) != first
    _elf(core, b"a" * 20)
    metadata.write_text("Name: ptoas\nVersion: 0.65.dev2\nRequires-Dist: numpy\n")
    assert identity._ptoas_identity(str(launcher)) != first
    metadata.write_text("Name: ptoas\nVersion: 0.65.dev1\nRequires-Dist: numpy\n")
    assert identity._ptoas_identity(str(launcher)) == first
    startup_hook.write_text("/changed/package/path\n")
    assert identity._ptoas_identity(str(launcher)) != first
    startup_hook.write_text("/original/package/path\n")
    startup_module.write_text("value = 2\n")
    assert identity._ptoas_identity(str(launcher)) != first
    startup_module.write_text("value = 1\n")
    assert identity._ptoas_identity(str(launcher)) == first
    core.unlink()
    with pytest.raises(ValueError, match="no native compiler module"):
        identity._ptoas_identity(str(launcher))
    _elf(core, b"a" * 20)
    metadata.unlink()
    with pytest.raises(ValueError, match="metadata is unavailable or ambiguous"):
        identity._ptoas_identity(str(launcher))
    metadata.write_text("Name: ptoas\nVersion: 0.65.dev1\nRequires-Dist: numpy\n")
    second_metadata = package.parent / "ptoas-0.66.dist-info/METADATA"
    second_metadata.parent.mkdir()
    second_metadata.write_text("Name: ptoas\nVersion: 0.66\n")
    with pytest.raises(ValueError, match="metadata is unavailable or ambiguous"):
        identity._ptoas_identity(str(launcher))
    second_metadata.unlink()
    second_metadata.parent.rmdir()
    (numpy_metadata / "RECORD").write_text("new NumPy wheel build")
    assert identity._ptoas_identity(str(launcher)) != first
    (numpy_metadata / "RECORD").unlink()
    with pytest.raises(ValueError, match="NumPy wheel identity"):
        identity._ptoas_identity(str(launcher))
    (package / "_online").mkdir()
    with pytest.raises(ValueError, match="online build has no stable published identity"):
        identity._ptoas_identity(str(launcher))


@pytest.mark.parametrize("distribution", ["ptoas", "numpy"])
@pytest.mark.parametrize(
    "requirement", ["extra-compiler-package", "extra-compiler-package; python_version >= '3.10'"]
)
def test_ptoas_rejects_untracked_wheel_dependencies(selected_ptoas_wheel, distribution, requirement):
    wheel = selected_ptoas_wheel
    metadata = wheel.metadata if distribution == "ptoas" else wheel.numpy_metadata / "METADATA"
    assert identity._ptoas_identity(str(wheel.launcher))
    with metadata.open("a") as stream:
        stream.write(f"Requires-Dist: {requirement}\n")
    with pytest.raises(ValueError, match="Unsupported wheel dependency declarations"):
        identity._ptoas_identity(str(wheel.launcher))


def test_ptoas_requires_expected_numpy_declaration(selected_ptoas_wheel):
    wheel = selected_ptoas_wheel
    wheel.metadata.write_text("Name: ptoas\nVersion: 0.65.dev1\n")
    with pytest.raises(ValueError, match="Unsupported wheel dependency declarations"):
        identity._ptoas_identity(str(wheel.launcher))


@pytest.mark.parametrize("name", ["WHEEL", "RECORD"])
def test_ptoas_wheel_build_record_changes_identity(selected_ptoas_wheel, name):
    wheel = selected_ptoas_wheel
    first = identity._ptoas_identity(str(wheel.launcher))
    (wheel.metadata.parent / name).write_text("same version, different wheel build")
    assert identity._ptoas_identity(str(wheel.launcher)) != first


@pytest.mark.parametrize("name", ["WHEEL", "RECORD"])
def test_ptoas_requires_complete_wheel_metadata(selected_ptoas_wheel, name):
    wheel = selected_ptoas_wheel
    (wheel.metadata.parent / name).unlink()
    with pytest.raises(ValueError, match="metadata is unavailable or ambiguous"):
        identity._ptoas_identity(str(wheel.launcher))


@pytest.mark.parametrize("name", ["ptodsl", "TileOps", "SoftOps"])
@pytest.mark.parametrize(
    "redirect", ["origin", "namespace", "search_path", "symlink", "loaded_origin", "loaded_path"]
)
def test_ptoas_rejects_auxiliary_package_redirect(selected_ptoas_wheel, tmp_path, name, redirect):
    wheel = selected_ptoas_wheel
    outside = tmp_path / "editable" / name
    outside.mkdir(parents=True)
    origin = outside / "__init__.py"
    origin.write_text("value = 1")
    packages = wheel.package_evidence["ptoas_packages"]
    installed = wheel.package.parent / name
    if redirect == "origin":
        packages[name] = (str(origin), [str(outside)])
    elif redirect == "namespace":
        packages[name] = (None, [str(outside)])
    elif redirect == "search_path":
        packages[name] = (str(installed / "__init__.py"), [str(installed), str(outside)])
    elif redirect == "symlink":
        (installed / "__init__.py").unlink()
        installed.rmdir()
        installed.symlink_to(outside, target_is_directory=True)
    elif redirect == "loaded_origin":
        wheel.package_evidence["loaded_ptoas_packages"] = [(f"{name}.helper", str(origin), [])]
    else:
        wheel.package_evidence["loaded_ptoas_packages"] = [
            (name, str(installed / "__init__.py"), [str(outside)])
        ]
    with pytest.raises(ValueError, match="PTOAS.*(origin|redirect)"):
        identity._ptoas_identity(str(wheel.launcher))


@pytest.mark.parametrize("name", ["ptodsl", "TileOps", "SoftOps"])
@pytest.mark.parametrize("redirect", ["PYTHONPATH", "pth"])
def test_ptoas_probe_detects_auxiliary_redirect_without_importing(
    selected_ptoas_wheel, tmp_path, name, redirect
):
    wheel = selected_ptoas_wheel
    outside = tmp_path / "external"
    root = outside / name
    root.mkdir(parents=True)
    origin = root / "__init__.py"
    origin.write_text("raise AssertionError('must not import auxiliary compiler package')")
    prefix = ""
    search = os.pathsep.join((str(outside), str(wheel.package.parent)))
    if redirect == "pth":
        hooks = tmp_path / "hooks"
        hooks.mkdir()
        (hooks / "redirect.pth").write_text(f"import sys; sys.path.insert(1, {str(outside)!r})\n")
        prefix = f"import site; site.addsitedir({str(hooks)!r})\n"
        search = str(wheel.package.parent)
    result = subprocess.run(
        [sys.executable, "-S", "-c", prefix + identity._PTOAS_PROBE, str(wheel.launcher.parent)],
        env={**os.environ, "PYTHONPATH": search},
        check=True,
        capture_output=True,
        text=True,
    )
    record = ast.literal_eval(result.stdout)
    assert record["ptoas_packages"][name][0] == str(origin)
    with pytest.raises(ValueError, match="external redirect"):
        identity._ptoas_package_identity(record, wheel.package)


@pytest.mark.parametrize("change", ["version", "cache_tag", "build_id", "contents"])
def test_ptoas_interpreter_replacement_changes_identity(selected_ptoas_wheel, change):
    wheel = selected_ptoas_wheel
    if change == "contents":
        wheel.interpreter.write_bytes(b"\x7fELF-interpreter-a")
    first = identity._ptoas_identity(str(wheel.launcher))
    assert identity._ptoas_identity(str(wheel.launcher)) == first
    if change == "version":
        wheel.interpreter_evidence["interpreter_version"] = "3.10.20 (selected build)"
    elif change == "cache_tag":
        wheel.interpreter_evidence["interpreter_cache_tag"] = "cpython-311"
    elif change == "build_id":
        _elf(wheel.interpreter, b"j" * 20)
    else:
        wheel.interpreter.write_bytes(b"\x7fELF-interpreter-b")
    assert identity._ptoas_identity(str(wheel.launcher)) != first


@pytest.mark.parametrize("field", ["interpreter_version", "interpreter_cache_tag"])
@pytest.mark.parametrize("value", [None, "", 310])
def test_ptoas_requires_interpreter_evidence(selected_ptoas_wheel, field, value):
    wheel = selected_ptoas_wheel
    wheel.interpreter_evidence[field] = value
    with pytest.raises(ValueError, match="interpreter version/ABI evidence is unavailable"):
        identity._ptoas_identity(str(wheel.launcher))


def test_unknown_ptoas_launcher_has_no_build_identity(tmp_path, monkeypatch):
    launcher = tmp_path / "ptoas-wrapper"
    launcher.write_text('#!/bin/sh\nexec ptoas "$@"\n')

    def reject(path):
        raise ValueError(f"Unknown launcher: {path}")

    monkeypatch.setattr(_toolchain, "_console_interpreter", reject)
    with pytest.raises(ValueError, match="Unsupported PTOAS launcher for build identity"):
        identity._ptoas_identity(str(launcher))


@pytest.mark.parametrize("recognized_layout", [True, False])
def test_cann_without_install_version_has_no_build_identity(
    selected_build, monkeypatch, tmp_path, recognized_layout
):
    ccec_root = tmp_path / "tools/bisheng_compiler/bin" if recognized_layout else tmp_path / "unknown/bin"
    ccec = _elf(ccec_root / "ccec", b"e" * 20)
    linker = _elf(tmp_path / "ld.lld", b"l" * 20)
    selected_build.compiler.platform = "a2a3"
    selected_build.compiler.sdk.ccec = SimpleNamespace(cxx_path=str(ccec), linker_path=str(linker))
    monkeypatch.setattr(_toolchain, "_cann_install_version", lambda root: "" if recognized_layout else "v1")
    with pytest.raises(ValueError, match="CANN installation build version is unavailable"):
        identity.discover_builds(lambda: selected_build.compiler, "ptoas-1", "runtime")


@pytest.fixture
def selected_device_build(selected_build, monkeypatch, tmp_path):
    ccec = _elf(tmp_path / "tools/bisheng_compiler/bin/ccec", b"e" * 20)
    linker = _elf(tmp_path / "ld.lld", b"l" * 20)
    selected_build.compiler.platform = "a2a3"
    selected_build.compiler.sdk.ccec = SimpleNamespace(cxx_path=str(ccec), linker_path=str(linker))
    monkeypatch.setattr(_toolchain, "_cann_install_version", lambda root: "v1")
    return selected_build


@pytest.mark.parametrize("kind", ["script", "script_symlink", "non_executable_elf"])
@pytest.mark.parametrize("tool", ["cxx_path", "linker_path"])
def test_unsupported_device_tool_bypasses_persistence(
    selected_device_build, monkeypatch, tmp_path, kind, tool
):
    executable = Path(getattr(selected_device_build.compiler.sdk.ccec, tool))
    if kind == "non_executable_elf":
        executable.chmod(0o644)
        reason = "Compiler executable is unavailable"
    else:
        executable.write_text('#!/bin/sh\nexec device-tool "$@"\n')
        reason = "Unsupported compiler launcher"
        if kind == "script_symlink":
            alias = tmp_path / "executable-alias"
            alias.symlink_to(executable)
            setattr(selected_device_build.compiler.sdk.ccec, tool, str(alias))

    monkeypatch.setenv("PYPTO_CACHE_IDENTITY", "build")
    monkeypatch.setattr(_toolchain, "_identities", {})
    monkeypatch.setattr(_toolchain, "_selected_compiler", lambda platform: selected_device_build.compiler)
    monkeypatch.setattr(_toolchain, "find_ptoas_binary", lambda: "ptoas-1")
    # The same launcher can select different binaries without changing its bytes.
    for name, build_id in (("first", b"a" * 20), ("second", b"b" * 20)):
        target = _elf(tmp_path / name / "device-tool", build_id)
        monkeypatch.setenv("PATH", str(target.parent))
        captured = _toolchain.capture_toolchain("a2a3", "runtime")
        assert not captured.usable
        assert captured.digest is None
        assert any(reason in failure.reason for failure in captured.failures)


@pytest.mark.parametrize("kind", ["elf", "elf_symlink", "elf_without_build_id"])
def test_native_device_linker_changes_invalidate_identity(selected_device_build, tmp_path, kind):
    linker = Path(selected_device_build.compiler.sdk.ccec.linker_path)
    if kind == "elf_symlink":
        alias = tmp_path / "linker-alias"
        alias.symlink_to(linker)
        selected_device_build.compiler.sdk.ccec.linker_path = str(alias)
    elif kind == "elf_without_build_id":
        linker.write_bytes(b"\x7fELF-no-build-id-a")
    first = _capture(selected_device_build)
    assert first.usable
    if kind == "elf_without_build_id":
        linker.write_bytes(b"\x7fELF-no-build-id-b")
    else:
        _elf(linker, b"m" * 20)
    second = _capture(selected_device_build)
    assert second.usable
    assert first.device_toolchain != second.device_toolchain
    assert first.digest != second.digest


@pytest.mark.parametrize("hook", ["sitecustomize", "executable_pth"])
def test_script_only_startup_dependency_requires_manual_invalidation(tmp_path, hook):
    environment = tmp_path / "environment"
    venv.EnvBuilder(with_pip=False).create(environment)
    interpreter = environment / "bin/python"
    site = next((environment / "lib").glob("python*/site-packages"))
    helper = site / "actual_only.py"
    helper.write_text("value = 1\n")
    conditional = "import sys; sys.argv[0] != '-c' and __import__('actual_only')\n"
    (site / ("sitecustomize.py" if hook == "sitecustomize" else "startup.pth")).write_text(conditional)
    launcher = environment / "bin/ptoas"
    launcher.write_text("import sys; print('actual_only' in sys.modules)\n")
    env = {**os.environ, "PYTHONPATH": "", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    actual = subprocess.run(
        [str(interpreter), str(launcher)], env=env, check=True, capture_output=True, text=True
    )
    assert actual.stdout.strip() == "True"
    identities = []
    for value in (1, 2):
        helper.write_text(f"value = {value}\n")
        probe = subprocess.run(
            [str(interpreter), "-c", identity._PTOAS_PROBE, str(launcher.parent)],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        record = ast.literal_eval(probe.stdout)
        assert "actual_only" not in dict(record["startup_modules"])
        identities.append(identity._startup_identity(record))
    # This hidden dependency is outside the probe contract: callers must bump
    # PYPTO_CACHE_EPOCH (or disable persistence) when changing such inputs.
    assert identities[0] == identities[1]


def test_path_only_startup_does_not_execute_dependency_pth(tmp_path):
    environment = tmp_path / "environment"
    venv.EnvBuilder(with_pip=False).create(environment)
    site = next((environment / "lib").glob("python*/site-packages"))
    dependencies = tmp_path / "dependencies"
    dependencies.mkdir()
    (dependencies / "not_processed.pth").write_text("import nonexistent_startup_module\n")
    path_file = site / "dependencies.pth"
    path_file.write_text(f"{dependencies}\n")
    result = subprocess.run(
        [str(environment / "bin/python"), "-c", identity._PTOAS_PROBE, str(environment / "bin")],
        env={**os.environ, "PYTHONPATH": "", "PYTHONNOUSERSITE": "1"},
        check=True,
        capture_output=True,
        text=True,
    )
    record = ast.literal_eval(result.stdout)
    assert record["startup_hooks"] == [str(path_file)]
    assert str(dependencies) in record["startup_path"]
    assert identity._startup_identity(record)


@pytest.mark.parametrize("hook", ["sitecustomize", "usercustomize", "space", "tab", "bom"])
def test_executable_startup_files_participate_in_build_identity(selected_ptoas_wheel, monkeypatch, hook):
    wheel = selected_ptoas_wheel
    record = ast.literal_eval(_toolchain._run([]))
    if hook in ("sitecustomize", "usercustomize"):
        record["startup_modules"] = [(hook, str(wheel.startup_module))]
        changed_file = wheel.startup_module
    else:
        prefix = "\ufeffimport " if hook == "bom" else "import" + (" " if hook == "space" else "\t")
        wheel.startup_hook.write_text(prefix + "startup_hook\n")
        changed_file = wheel.startup_hook
    monkeypatch.setattr(_toolchain, "_run", lambda command: repr(record))
    first = identity._ptoas_identity(str(wheel.launcher))
    assert first == identity._ptoas_identity(str(wheel.launcher))
    changed_file.write_text(changed_file.read_text() + "# changed startup input\n")
    assert first != identity._ptoas_identity(str(wheel.launcher))


def test_missing_startup_evidence_has_no_build_identity(selected_ptoas_wheel, monkeypatch):
    record = ast.literal_eval(_toolchain._run([]))
    del record["startup_hooks"]
    monkeypatch.setattr(_toolchain, "_run", lambda command: repr(record))
    with pytest.raises(ValueError, match="startup"):
        identity._ptoas_identity(str(selected_ptoas_wheel.launcher))


@pytest.mark.parametrize(
    "loader", ["lib", "", ":/absolute", "/absolute:", "/absolute;lib", None, "/absolute"]
)
def test_loader_relative_entries_preserve_execution_cwd(tmp_path, monkeypatch, loader):
    component = _toolchain.ComponentInputs(unavailable_reason=None, reported_version="same-build")
    inputs = _toolchain.ToolchainInputs(component, component, component, component, component)
    monkeypatch.setattr(_toolchain, "_identities", {})
    monkeypatch.setattr(_toolchain, "find_ptoas_binary", lambda: "ptoas")
    monkeypatch.setattr(_toolchain, "_selected_compiler", lambda platform: None)
    monkeypatch.setattr(identity, "discover_builds", lambda *args: inputs)
    monkeypatch.setattr(_toolchain, "_discover", lambda *args: inputs)
    monkeypatch.setenv("PYPTO_CACHE_IDENTITY", "build")
    if loader is None:
        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    else:
        monkeypatch.setenv("LD_LIBRARY_PATH", loader)
    monkeypatch.chdir(tmp_path)
    first = _toolchain.capture_toolchain("a2a3", "runtime")
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    second = _toolchain.capture_toolchain("a2a3", "runtime")
    assert first.usable and second.usable
    if loader in (None, "/absolute"):
        assert first.digest == second.digest
    else:
        assert first.digest != second.digest


@pytest.mark.parametrize("selection", ["relative_compiler", "relative_path", "empty_path"])
def test_compiler_probe_uses_build_directory(tmp_path, monkeypatch, selection):
    caller, project = tmp_path / "caller", tmp_path / "project"
    for root in (caller, project):
        (root / "bin").mkdir(parents=True)
        _elf(root / "bin/g++", b"g" * 20)
        _elf(root / "g++", b"g" * 20)
        for name in ("cc1plus", "collect2", "as", "ld"):
            _elf(root / "bin" / name, root.name.encode().ljust(20, b"x"))
            _elf(root / name, root.name.encode().ljust(20, b"x"))
    monkeypatch.chdir(caller)
    relative = "" if selection == "empty_path" else "bin"
    monkeypatch.setenv("PATH", relative)
    assert _toolchain._run(
        [sys.executable, "-S", "-c", "import os; print(os.getcwd())"], cwd=project
    ).strip() == str(project)

    def probe(args, *, cwd=None):
        assert cwd == project
        if args[1] == "-E":
            return f"COLLECT_GCC={args[0]}\n"
        if args[1] == "--version":
            return "gcc test build"
        return args[1].partition("=")[2]

    monkeypatch.setattr(_toolchain, "_run", probe)
    selected = "bin/g++" if selection == "relative_compiler" else "g++"
    first = identity._compiler_identity(selected, project)
    helpers = {name: path for name, path, _ in first[-1]}
    assert helpers["as"] == str(project / relative / "as")
    assert helpers["ld"] == str(project / relative / "ld")
    _elf(caller / relative / "as", b"c" * 20)
    assert identity._compiler_identity(selected, project) == first
    _elf(project / relative / "as", b"p" * 20)
    assert identity._compiler_identity(selected, project) != first
    assert Path.cwd() == caller


def test_build_probes_and_wrapper_configuration_use_project_cwd(selected_build, monkeypatch, tmp_path):
    root = selected_build.root
    wrapper = _elf(tmp_path / "ccache", b"w" * 20)
    selected_build.compiler._orchestration_toolchain = lambda name: SimpleNamespace(cxx_path=str(wrapper))
    original_run = _toolchain._run
    calls = []

    def run(args, *, cwd=None):
        calls.append((args, cwd))
        return original_run(args, cwd=cwd)

    monkeypatch.setattr(_toolchain, "_run", run)
    assert _capture(selected_build).usable
    assert calls and all(cwd == root for _, cwd in calls)
    assert any(args[1] == "--show-config" for args, _ in calls)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
