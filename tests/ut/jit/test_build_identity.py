# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Build identities follow effective imports and selected dependency versions."""

import hashlib
import json
import os
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
    monkeypatch.setattr(_toolchain, "_run", lambda args: "compiler version 1")
    monkeypatch.setattr(identity, "_ptoas_identity", lambda selected: selected)
    compiler = SimpleNamespace(
        project_root=root,
        platform="a2a3sim",
        _sanitizers="",
        _orchestration_toolchain=lambda name: SimpleNamespace(cxx_path=str(cxx)),
        sdk=SimpleNamespace(gxx15=SimpleNamespace(cxx_path=str(cxx))),
    )
    return SimpleNamespace(compiler=compiler, root=root, native=native, runtime=runtime)


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
    assert json.loads(result.stdout) == str(origin)


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
    calls = []
    monkeypatch.setattr(_toolchain, "_console_interpreter", lambda path: Path("/selected/python"))

    def probe(command):
        calls.append(command)
        return json.dumps(str(package / "__init__.py"))

    monkeypatch.setattr(_toolchain, "_run", probe)
    first = identity._ptoas_identity(str(launcher))
    assert calls[0][0] == "/selected/python"
    _elf(core, b"b" * 20)
    assert identity._ptoas_identity(str(launcher)) != first
    metadata.write_text("Name: ptoas\nVersion: 0.65.dev2\n")
    assert identity._ptoas_identity(str(launcher)) != first


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
