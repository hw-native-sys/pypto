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
import os
import subprocess
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


def test_component_drops_every_enclosed_path_regardless_of_order(tmp_path):
    # Roots are accepted by walking a candidate's own parents rather than
    # rescanning the accepted roots, so an ancestor several levels up must still
    # absorb its descendants, and unrelated siblings must survive.
    deep = tmp_path / "sdk/lib/backend/plugin.so"
    deep.parent.mkdir(parents=True)
    deep.write_bytes(b"\x7fELF")
    sibling = tmp_path / "other/header.h"
    sibling.parent.mkdir()
    sibling.write_text("aaa")
    supplied = {tmp_path / "sdk", deep, deep.parent, deep.parent.parent, sibling, sibling.parent}

    component = _toolchain._component(supplied)

    assert [root.path for root in component.roots] == [sibling.parent, tmp_path / "sdk"]


def test_component_rejects_an_input_that_does_not_exist(tmp_path):
    present = tmp_path / "present.h"
    present.write_text("aaa")
    with pytest.raises(OSError):
        _toolchain._component({present, tmp_path / "missing.h"})


def test_component_rejects_a_dangling_symlink(tmp_path):
    link = tmp_path / "libmissing.so"
    link.symlink_to(tmp_path / "absent.so")
    with pytest.raises(OSError):
        _toolchain._component({link})


def test_elf_inputs_many_merges_every_closure(monkeypatch, tmp_path):
    first, second = tmp_path / "a.so", tmp_path / "b.so"
    shared = tmp_path / "libc.so"
    monkeypatch.setattr(_toolchain, "_elf_inputs", lambda p, *rest: {p, shared})
    assert _toolchain._elf_inputs_many([first, second]) == {first, second, shared}
    assert _toolchain._elf_inputs_many([]) == set()


def test_elf_inputs_many_propagates_a_worker_failure(monkeypatch, tmp_path):
    # A partial inventory is never returned: one unresolvable library fails the
    # whole component exactly as the serial loop did.
    natives = [tmp_path / f"lib{index}.so" for index in range(8)]

    def explode(path, *rest):
        if path == natives[5]:
            raise ValueError(f"Unresolved native dependencies for {path}")
        return {path}

    monkeypatch.setattr(_toolchain, "_elf_inputs", explode)
    with pytest.raises(ValueError, match="Unresolved native dependencies"):
        _toolchain._elf_inputs_many(natives)


def test_pto_isa_uses_the_revision_its_resolution_verified(tmp_path, monkeypatch):
    checkout = _git_checkout(tmp_path / "pto-isa")
    monkeypatch.setitem(
        sys.modules, "simpler_setup.pto_isa", SimpleNamespace(get_pto_isa_head=lambda root: "f" * 40)
    )
    component = _toolchain._pto_isa_component(checkout)
    assert component.verified_revision == "f" * 40
    assert component.roots == ()
    assert component.unavailable_reason is None


def _git_checkout(root: Path) -> Path:
    """A real committed checkout, so the git invocation itself is under test."""
    root.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    (root / ".gitignore").write_text("build/\n")
    (root / "isa.h").write_text("int isa;\n")
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "isa"]):
        subprocess.run(["git", *args], cwd=root, env=env, check=True, capture_output=True)
    return root


def test_ignored_files_are_not_invisible_to_the_checkout_check(tmp_path):
    # The resolver decides cleanliness with `git status --porcelain`, which
    # omits ignored paths; a generated file left in the tree would otherwise
    # leave both the revision and that check unchanged.
    checkout = _git_checkout(tmp_path / "pto-isa")
    assert _toolchain._unaccounted_checkout_state(checkout) == ""

    (checkout / "build").mkdir()
    (checkout / "build/generated.h").write_text("int generated;\n")

    assert "build/" in _toolchain._unaccounted_checkout_state(checkout)


def test_an_unusable_git_answer_is_not_treated_as_clean(tmp_path):
    not_a_checkout = tmp_path / "loose"
    not_a_checkout.mkdir()
    assert _toolchain._unaccounted_checkout_state(not_a_checkout) != ""


def test_pto_isa_falls_back_to_contents_for_an_unaccounted_tree(tmp_path, monkeypatch):
    checkout = tmp_path / "pto-isa"
    checkout.mkdir()
    (checkout / "isa.h").write_text("aaa")
    monkeypatch.setitem(
        sys.modules, "simpler_setup.pto_isa", SimpleNamespace(get_pto_isa_head=lambda root: "f" * 40)
    )
    monkeypatch.setattr(_toolchain, "_unaccounted_checkout_state", lambda root: "!! build/x.h")

    component = _toolchain._pto_isa_component(checkout)

    assert component.verified_revision is None
    assert [root.path for root in component.roots] == [checkout]


def test_pto_isa_falls_back_to_contents_when_the_revision_is_unknown(tmp_path, monkeypatch):
    # get_pto_isa_head reports failure as an empty string, which must never be
    # accepted as an identity.
    checkout = tmp_path / "pto-isa"
    checkout.mkdir()
    (checkout / "isa.h").write_text("aaa")
    monkeypatch.setitem(
        sys.modules, "simpler_setup.pto_isa", SimpleNamespace(get_pto_isa_head=lambda root: "")
    )
    component = _toolchain._pto_isa_component(checkout)
    assert component.verified_revision is None
    assert [root.path for root in component.roots] == [checkout]
    before = fingerprint_content(component.roots)
    (checkout / "isa.h").write_text("bbb")
    assert fingerprint_content(component.roots).digest != before.digest


def test_ptoas_is_identified_by_the_version_it_reports(tmp_path, monkeypatch):
    monkeypatch.setattr("pypto.backend._ptoas_locate.check_ptoas_version", lambda binary: "ptoas 0.61.dev3\n")
    component = _toolchain._ptoas_component(str(tmp_path / "ptoas"))
    assert component.reported_version == "ptoas 0.61.dev3"
    assert component.verified_revision is None
    assert component.roots == ()
    assert component.unavailable_reason is None


def test_ptoas_falls_back_to_contents_when_the_probe_fails(tmp_path, monkeypatch):
    def refuse(binary):
        raise RuntimeError("ptoas is version 0.55, but PyPTO requires PTOAS >= v0.61")

    monkeypatch.setattr("pypto.backend._ptoas_locate.check_ptoas_version", refuse)
    captured = {}

    def fake_inputs(launcher):
        captured["launcher"] = launcher
        return {tmp_path / "tree"}

    (tmp_path / "tree").mkdir()
    (tmp_path / "tree/ptoas.so").write_bytes(b"\x7fELF")
    monkeypatch.setattr(_toolchain, "_ptoas_inputs", fake_inputs)

    component = _toolchain._ptoas_component(str(tmp_path / "ptoas"))

    assert component.reported_version is None
    assert [root.path for root in component.roots] == [tmp_path / "tree"]
    assert captured["launcher"] == tmp_path / "ptoas"


def _cann_install(root: Path, arch: str = "aarch64-linux", **fields: str) -> Path:
    stated = {"package_name": "Ascend-cann-toolkit", "version": "9.0.0", **fields}
    (root / arch).mkdir(parents=True, exist_ok=True)
    (root / arch / "ascend_toolkit_install.info").write_text("".join(f"{k}={v}\n" for k, v in stated.items()))
    return root


def test_cann_version_prefers_the_build_over_the_release(tmp_path):
    # Two builds of one release share `version`; only `innerversion` separates
    # them, which is the whole point of covering the tree by what it states.
    root = _cann_install(tmp_path / "cann", innerversion="V100R001C10SPC001B250")
    assert _toolchain._cann_install_version(root) == "V100R001C10SPC001B250"


def test_cann_version_falls_back_to_the_release(tmp_path):
    root = _cann_install(tmp_path / "cann")
    assert _toolchain._cann_install_version(root) == "9.0.0"


def test_cann_version_counts_one_installation_reached_by_two_paths(tmp_path):
    # CANN ships `arm64-linux` as a symlink to `aarch64-linux`, so the glob
    # returns two paths naming one file. Counting paths would reject a normal
    # installation and silently fall back to reading every byte.
    root = _cann_install(tmp_path / "cann", innerversion="V100R001C10SPC001B250")
    (root / "arm64-linux").symlink_to(root / "aarch64-linux", target_is_directory=True)

    assert _toolchain._cann_install_version(root) == "V100R001C10SPC001B250"


def test_cann_version_rejects_two_installations(tmp_path):
    root = _cann_install(tmp_path / "cann", innerversion="B1")
    _cann_install(root, arch="x86_64-linux", innerversion="B2")
    assert _toolchain._cann_install_version(root) == ""


def test_cann_version_rejects_an_empty_statement(tmp_path):
    root = _cann_install(tmp_path / "cann", version="", innerversion="")
    assert _toolchain._cann_install_version(root) == ""


def test_cann_version_skips_an_empty_build_for_the_release(tmp_path):
    root = _cann_install(tmp_path / "cann", innerversion="")
    assert _toolchain._cann_install_version(root) == "9.0.0"


def test_cann_version_is_empty_without_an_installation(tmp_path):
    (tmp_path / "cann").mkdir()
    assert _toolchain._cann_install_version(tmp_path / "cann") == ""


def test_outside_keeps_only_what_the_installation_does_not_own(tmp_path):
    install = tmp_path / "cann"
    owned_root = install / "tools/bisheng_compiler"
    outside = tmp_path / "usr/include"
    paths = {install, owned_root, outside, tmp_path / "usr/lib64/libc.so.6"}

    assert _toolchain._outside(paths, install) == {outside, tmp_path / "usr/lib64/libc.so.6"}


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


def test_persistent_specialization_separates_scalar_semantics():
    """The scalar contract version reaches the persisted specialization digest.

    Scalar *values* no longer appear in the key at all (issue #2751), so what
    has to stay distinguishable is the contract that built the artifact, not
    the individual values.
    """
    from pypto.jit._persistent import _specialization_digest  # noqa: PLC0415
    from pypto.jit.cache import CacheKey  # noqa: PLC0415

    v1 = CacheKey("source", None, None, (), None, (("scalar_semantics", 1),))
    v2 = CacheKey("source", None, None, (), None, (("scalar_semantics", 2),))
    assert _specialization_digest(v1) != _specialization_digest(v2)


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


@pytest.mark.parametrize(
    "command",
    [
        "GROUP ( libdependency.a )",
        "INPUT ( -ldependency )",
        "GROUP ( AS_NEEDED ( -l:libdependency.a ) )",
        'INPUT ( "relative path/libdependency.a" )',
        "GROUP ( ../elsewhere/libdependency.a )",
        "INPUT ( =/lib/libdependency.a )",
        "INPUT ( $SYSROOT/lib/libdependency.a )",
    ],
)
def test_linker_search_dependent_inputs_disable_identity(tmp_path, monkeypatch, command):
    script = tmp_path / "libwrapper.so"
    script.write_text(command)
    # Even an existing local candidate is not proof of linker search resolution.
    dependency = tmp_path / "libdependency.a"
    dependency.write_bytes(b"!<arch>\nold")
    monkeypatch.chdir(tmp_path)
    for contents in (b"!<arch>\nold", b"!<arch>\nnew"):
        dependency.write_bytes(contents)
        with pytest.raises(ValueError, match="requires search-path resolution"):
            _toolchain._linker_script_inputs({script}, None)


def test_nested_absolute_linker_dependency_changes_fresh_identity(tmp_path):
    script = tmp_path / "installation/libwrapper.so"
    script.parent.mkdir()
    nested = tmp_path / "outside/nested script.ld"
    nested.parent.mkdir()
    dependency = nested.parent / "libdependency.a"
    dependency.write_bytes(b"!<arch>\nold")
    nested.write_text(f'INPUT ( "{dependency}" )')
    script.write_text(
        '/* GROUP ( -lignored ) */\nOUTPUT_FORMAT("elf64-littleaarch64")\n'
        f'OUTPUT_ARCH(aarch64) GROUP ( AS_NEEDED ( "{nested}" ) );'
    )

    def capture():
        paths = _toolchain._linker_script_inputs({script}, None)
        assert paths == {script, nested, dependency}
        component = _toolchain._component(paths)
        inputs = _toolchain.ToolchainInputs(component, component, component, component, component)
        return _toolchain.InstallationIdentityCache().capture(inputs)

    before = capture()
    dependency.write_bytes(b"!<arch>\nnew")
    after = capture()
    assert before.usable and after.usable and before.digest != after.digest
    nested.write_text("INPUT ( -ldependency )")
    with pytest.raises(ValueError, match="requires search-path resolution"):
        capture()


@pytest.mark.parametrize(
    "command",
    [
        'SEARCH_DIR("/untracked") GROUP ( -ldependency )',
        'INCLUDE "another.ld"',
        "STARTUP ( /untracked.o )",
        'INPUT ( "/unterminated )',
        "GROUP ( /* unterminated )",
        "GROUP ( AS_NEEDED ( /missing )",
    ],
)
def test_unknown_or_malformed_linker_scripts_are_unavailable(tmp_path, command):
    script = tmp_path / "script.ld"
    script.write_text(command)
    with pytest.raises(ValueError, match="linker script"):
        _toolchain._linker_script_inputs({script}, None)


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
        "import sys\nfrom ptoas._cli import main\n"
        "if __name__ == '__main__':\n"
        "    sys.argv[0] = sys.argv[0].removesuffix('.exe')\n"
        "    sys.exit(main())\n",
    ],
    ids=["pip", "uv", "pip26"],
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


@pytest.fixture
def compiler_metadata(monkeypatch):
    """Keep discovery tests independent of optional runtime installations."""
    monkeypatch.setitem(sys.modules, "simpler_setup", None)
    monkeypatch.setitem(sys.modules, "simpler", None)
    monkeypatch.setitem(
        sys.modules,
        "pypto.runtime.kernel_compiler",
        SimpleNamespace(KernelCompiler=SimpleNamespace(_sanitizers=None)),
    )


@pytest.mark.usefixtures("compiler_metadata")
@pytest.mark.parametrize("error_type", [AttributeError, KeyError])
def test_adapter_drift_returns_unavailable_evidence(monkeypatch, error_type):
    def fail(*args):
        raise error_type("changed compiler inventory")

    monkeypatch.setattr(_toolchain, "_compiler", fail)
    result = _toolchain.capture_toolchain("a2a3", "tensormap_and_ringbuffer")
    assert not result.usable and result.digest is None
    assert "changed compiler inventory" in result.failures[0].reason


@pytest.mark.usefixtures("compiler_metadata")
def test_chdir_rediscovers_but_reuses_identical_component_digests(tmp_path, monkeypatch):
    import pypto._identity as identity_module  # noqa: PLC0415

    payload = tmp_path / "toolchain"
    payload.write_bytes(b"compiler resources")
    component = _toolchain._component({payload})
    inputs = _toolchain.ToolchainInputs(component, component, component, component, component)
    compiler = SimpleNamespace(project_root=tmp_path, _sanitizers=None)
    monkeypatch.setattr(_toolchain, "_compiler", lambda *args: compiler)
    monkeypatch.setattr(_toolchain, "find_ptoas_binary", lambda: payload)
    monkeypatch.setattr(_toolchain, "_identities", {})
    monkeypatch.setattr(_toolchain, "_identity_cache", _toolchain.InstallationIdentityCache())
    discoveries, reads = [], []
    fingerprint = identity_module.fingerprint_content

    def discover(*args):
        discoveries.append(Path.cwd())
        return inputs

    def read(roots):
        reads.append(roots)
        return fingerprint(roots)

    monkeypatch.setattr(_toolchain, "_discover", discover)
    monkeypatch.setattr(identity_module, "fingerprint_content", read)
    monkeypatch.chdir(tmp_path)
    first = _toolchain.capture_toolchain("a2a3", "tensormap_and_ringbuffer")
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    second = _toolchain.capture_toolchain("a2a3", "tensormap_and_ringbuffer")
    assert first.usable and second == first
    assert discoveries == [tmp_path, other]
    assert len(reads) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
