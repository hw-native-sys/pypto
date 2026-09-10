# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Dependency inventories fail closed and hash compiler resource contents."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pypto._identity import fingerprint_content
from pypto.jit import _toolchain


def test_component_preserves_content_changes_without_metadata_change(tmp_path):
    header = tmp_path / "include/header.h"
    header.parent.mkdir()
    header.write_text("aaa")
    component = _toolchain._component({header.parent, header})
    before = fingerprint_content(component.roots)
    header.write_text("bbb")
    assert fingerprint_content(component.roots).digest != before.digest
    assert len(component.roots) == 1


def test_unknown_shell_launcher_is_not_an_executable_identity(tmp_path):
    script = tmp_path / "ptoas"
    script.write_text("#!/bin/sh\neval some_dynamic_command\n")
    script.chmod(0o755)
    with pytest.raises(ValueError, match="Unsupported"):
        _toolchain._ptoas_inputs(script)


def test_forwarding_launcher_cycles_are_unavailable(tmp_path, monkeypatch):
    first, second = tmp_path / "first", tmp_path / "second"
    first.write_text(f'#!/bin/bash\nexec "{second}" "$@"\n')
    second.write_text(f'#!/bin/bash\nexec "{first}" "$@"\n')
    monkeypatch.setattr(_toolchain, "_executable", lambda _: tmp_path / "bash")
    monkeypatch.setattr(_toolchain, "_elf_inputs", lambda _: set())
    with pytest.raises(ValueError, match="cycle"):
        _toolchain._ptoas_inputs(first)


def test_loader_dependencies_include_transitive_libraries(tmp_path, monkeypatch):
    binary = tmp_path / "compiler"
    binary.write_bytes(b"\x7fELF")
    library = tmp_path / "libcompiler.so"
    library.write_bytes(b"library")
    loader = tmp_path / "ld.so"
    loader.write_bytes(b"loader")
    output = f"linux-vdso.so.1 (0xabc)\nlibcompiler.so => {library} (0xabc)\n{loader} (0xabc)\n"
    monkeypatch.setattr(
        _toolchain.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=output, stderr=""),
    )
    assert _toolchain._elf_inputs(binary) == {binary, library, loader}
    output = "libcompiler.so => not found"
    with pytest.raises(ValueError, match="Unresolved"):
        _toolchain._elf_inputs(binary)


@pytest.mark.parametrize("name", ["CPATH", "LD_PRELOAD", "GCC_EXEC_PREFIX", "LIBRARY_PATH"])
def test_implicit_dependency_override_bypasses(monkeypatch, name):
    for variable in ("CPATH", "LD_PRELOAD", "GCC_EXEC_PREFIX", "LIBRARY_PATH"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv(name, "untracked")
    identity = _toolchain.capture_toolchain("a2a3", "tensormap_and_ringbuffer")
    assert not identity.usable and identity.digest is None
    assert name in identity.failures[0].reason


def test_implicit_include_roots_are_resolved_and_required(tmp_path):
    root = tmp_path / "headers"
    root.mkdir()
    output = f"#include <...> search starts here:\n {root}\nEnd of search list.\n"
    assert _toolchain._include_roots(output, Path("compiler")) == {root}
    with pytest.raises(ValueError, match="implicit"):
        _toolchain._include_roots("unrecognized output", Path("compiler"))


@pytest.mark.parametrize("left,right", [(True, 1), (1, 1.0), (0.0, -0.0)])
def test_persistent_specialization_preserves_scalar_types(left, right):
    from pypto._identity import digest_record  # noqa: PLC0415
    from pypto.jit._persistent import _record, _typed_specialization  # noqa: PLC0415
    from pypto.jit.cache import ScalarCacheInfo  # noqa: PLC0415

    assert digest_record(_record(ScalarCacheInfo("value", left))) != digest_record(
        _record(ScalarCacheInfo("value", right))
    )
    left_key = (None, None, None, None, (ScalarCacheInfo("value", left),))
    right_key = (None, None, None, None, (ScalarCacheInfo("value", right),))
    assert _typed_specialization(left_key) != _typed_specialization(right_key)


def test_persistent_specialization_preserves_tensor_dtype():
    from pypto import DataType  # noqa: PLC0415
    from pypto._identity import digest_record  # noqa: PLC0415
    from pypto.jit._persistent import _record  # noqa: PLC0415
    from pypto.jit.cache import TensorCacheInfo  # noqa: PLC0415

    left = TensorCacheInfo("x", (None, 128), DataType.FP32)
    right = TensorCacheInfo("x", (None, 128), DataType.INT32)
    assert digest_record(_record(left)) != digest_record(_record(right))


def test_linker_scripts_follow_sysroot_and_ignore_comment_paths(tmp_path):
    sysroot = tmp_path / "sysroot"
    library = sysroot / "lib/libc.so.6"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"\x7fELF")
    script = sysroot / "lib/libc.so"
    script.write_text(
        "/* documentation: https://www.gnu.org; build: /missing */\n"
        "# /also-missing\nGROUP ( /lib/libc.so.6 )\n"
    )
    assert _toolchain._linker_script_inputs({script}, sysroot) == {script, library}
    library.unlink()
    with pytest.raises(FileNotFoundError):
        _toolchain._linker_script_inputs({script}, sysroot)


def test_gcc_link_plan_selects_actual_inputs_only(tmp_path, monkeypatch):
    compiler = tmp_path / "g++"
    startup = tmp_path / "crt.o"
    library = tmp_path / "libstdc++.so"
    plugin = tmp_path / "lto-wrapper"
    unrelated = tmp_path / "unused.so"
    for path in (startup, library, plugin, unrelated):
        path.write_bytes(b"\x7fELF")

    def run(command):
        if "-###" in command:
            return f"/tool/collect2 {startup} -lstdc++ -plugin-opt={plugin}\n"
        if "-print-file-name=libstdc++.so" in command:
            return str(library)
        assert "-print-sysroot" in command
        return ""

    monkeypatch.setattr(_toolchain, "_run", run)
    assert _toolchain._gcc_link_inputs(compiler) == {startup, library, plugin}


@pytest.mark.parametrize(
    "body",
    [
        "import re\nimport sys\nfrom ptoas._cli import main\n"
        "if __name__ == '__main__':\n"
        "    sys.argv[0] = re.sub(r'(-script\\.pyw|\\.exe)?$', '', sys.argv[0])\n"
        "    sys.exit(main())\n",
        "import sys\nfrom ptoas._cli import main\n"
        "if __name__ == '__main__':\n"
        "    if sys.argv[0].endswith('-script.pyw'):\n"
        "        sys.argv[0] = sys.argv[0][:-11]\n"
        "    elif sys.argv[0].endswith('.exe'):\n"
        "        sys.argv[0] = sys.argv[0][:-4]\n"
        "    sys.exit(main())\n",
    ],
    ids=["pip", "uv"],
)
def test_wheel_console_script_preserves_virtualenv_interpreter(tmp_path, body):
    interpreter = tmp_path / "bin/python"
    interpreter.parent.mkdir()
    interpreter.symlink_to(sys.executable)
    launcher = interpreter.with_name("ptoas")
    launcher.write_text(f"#!{interpreter}\n# -*- coding: utf-8 -*-\n{body}")
    assert _toolchain._console_interpreter(launcher) == interpreter
    launcher.write_text(launcher.read_text() + "print('untracked launcher behavior')\n")
    with pytest.raises(ValueError, match="console-script grammar"):
        _toolchain._console_interpreter(launcher)


def test_wheel_inventory_covers_resources_numpy_and_native_libraries(tmp_path, monkeypatch):
    interpreter = Path(sys.executable).resolve()
    launcher = tmp_path / "ptoas"
    launcher.write_text("console entry point")
    stdlib = tmp_path / "stdlib"
    stdlib.mkdir()
    wheel = tmp_path / "site-packages"
    resource = wheel / "ptoas/_runtime/share/ptoas/TileOps/op.py"
    numpy_source = wheel / "numpy/__init__.py"
    native = wheel / "numpy.libs/libblas.so.1"
    dependency = tmp_path / "libdependency.so"
    for path in (resource, numpy_source, native, dependency):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x7fELF" if ".so" in path.name else b"original")
    roots = [wheel / "ptoas", wheel / "numpy", native.parent]
    monkeypatch.setattr(_toolchain, "_console_interpreter", lambda _: interpreter)
    monkeypatch.setattr(
        _toolchain, "_run", lambda _: json.dumps({"roots": [str(p) for p in roots], "stdlib": str(stdlib)})
    )
    monkeypatch.setattr(_toolchain, "_elf_inputs", lambda p: {p, dependency} if p == native else {p})
    paths = _toolchain._wheel_inputs(launcher)
    assert dependency in paths and set(roots) <= paths
    before = fingerprint_content(_toolchain._component(paths).roots)
    assert before.digest is not None
    resource.write_bytes(b"modified")
    after_resource = fingerprint_content(_toolchain._component(paths).roots)
    assert after_resource.digest != before.digest
    numpy_source.write_bytes(b"modified")
    assert fingerprint_content(_toolchain._component(paths).roots).digest != after_resource.digest


def test_python_optimization_splits_persistent_identity(monkeypatch):
    from pypto.jit._persistent import _semantic_environment  # noqa: PLC0415

    monkeypatch.delenv("PYTHONOPTIMIZE", raising=False)
    ordinary = _semantic_environment()
    monkeypatch.setenv("PYTHONOPTIMIZE", "1")
    assert _semantic_environment() != ordinary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
