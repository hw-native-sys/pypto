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
    monkeypatch.setattr(_toolchain, "_invocable", lambda path: Path(path))
    monkeypatch.setattr(_toolchain, "_executable", lambda path: Path(path))

    def run(args):
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


@pytest.mark.parametrize("program", ["as", "ld"])
def test_selected_gcc_helper_rebuild_changes_identity(selected_build, program):
    first = _capture(selected_build)
    _elf(selected_build.tools[program], b"z" * 20)
    assert _capture(selected_build).device_toolchain != first.device_toolchain


def test_unknown_elf_wrapper_does_not_get_a_build_identity(selected_build, monkeypatch, tmp_path):
    wrapper = _elf(tmp_path / "untracked-wrapper", b"w" * 20)
    real_driver = _elf(tmp_path / "real-gcc", b"g" * 20)

    def run(args):
        if args[1] == "-E":
            return f"COLLECT_GCC={real_driver}\n"
        if args[1].startswith("-print-prog-name="):
            return str(selected_build.tools[args[1].partition("=")[2]])
        return "compiler version 1"

    monkeypatch.setattr(_toolchain, "_run", run)
    with pytest.raises(ValueError, match="untracked selection inputs"):
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

    def run(args):
        if args[0] == "git":
            result = subprocess.run(args, check=True, capture_output=True, text=True)
            return result.stdout + result.stderr
        return compiler_run(args)

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


def test_ptoas_resolves_selected_interpreter_without_importing_compiler(tmp_path, monkeypatch):
    launcher = tmp_path / "bin/ptoas"
    launcher.parent.mkdir()
    launcher.write_text("#!/selected/python\n")
    package = tmp_path / "selected/site-packages/ptoas"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("raise AssertionError('must not import')")
    core = _elf(package / "_core.so", b"a" * 20)
    metadata = package.parent / "ptoas-0.65.dev1.dist-info/METADATA"
    metadata.parent.mkdir()
    metadata.write_text("Name: ptoas\nVersion: 0.65.dev1\n")
    numpy = package.parent / "numpy/__init__.py"
    numpy.parent.mkdir()
    numpy.write_text("raise AssertionError('must not import NumPy')")
    numpy_metadata = package.parent / "numpy-2.2.6.dist-info"
    numpy_metadata.mkdir()
    for name in ("METADATA", "WHEEL", "RECORD"):
        (numpy_metadata / name).write_text(name)
    startup_hook = tmp_path / "selected/site-packages/startup.pth"
    startup_hook.write_text("import sitecustomize\n")
    startup_module = tmp_path / "selected/site-packages/sitecustomize.py"
    startup_module.write_text("value = 1\n")
    calls = []
    monkeypatch.setattr(_toolchain, "_console_interpreter", lambda path: Path("/selected/python"))

    def probe(command):
        calls.append(command)
        return repr(
            {
                "ptoas": str(package / "__init__.py"),
                "numpy": str(numpy),
                "startup_hooks": [str(startup_hook)],
                "startup_modules": [["sitecustomize", str(startup_module)]],
                "startup_path": [str(launcher.parent), str(package.parent)],
                "startup_unresolved": [],
            }
        )

    monkeypatch.setattr(_toolchain, "_run", probe)
    first = identity._ptoas_identity(str(launcher))
    assert calls[0][0] == "/selected/python"
    _elf(core, b"b" * 20)
    assert identity._ptoas_identity(str(launcher)) != first
    _elf(core, b"a" * 20)
    metadata.write_text("Name: ptoas\nVersion: 0.65.dev2\n")
    assert identity._ptoas_identity(str(launcher)) != first
    metadata.write_text("Name: ptoas\nVersion: 0.65.dev1\n")
    assert identity._ptoas_identity(str(launcher)) == first
    startup_hook.write_text("import sitecustomize # changed\n")
    assert identity._ptoas_identity(str(launcher)) != first
    startup_hook.write_text("import sitecustomize\n")
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
    metadata.write_text("Name: ptoas\nVersion: 0.65.dev1\n")
    second_metadata = package.parent / "ptoas-0.66.dist-info/METADATA"
    second_metadata.parent.mkdir()
    second_metadata.write_text("Name: ptoas\nVersion: 0.66\n")
    with pytest.raises(ValueError, match="metadata is unavailable or ambiguous"):
        identity._ptoas_identity(str(launcher))
    second_metadata.unlink()
    (numpy_metadata / "RECORD").write_text("new NumPy wheel build")
    assert identity._ptoas_identity(str(launcher)) != first
    (numpy_metadata / "RECORD").unlink()
    with pytest.raises(ValueError, match="NumPy wheel identity"):
        identity._ptoas_identity(str(launcher))
    (package / "_online").mkdir()
    with pytest.raises(ValueError, match="online build has no stable published identity"):
        identity._ptoas_identity(str(launcher))


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
