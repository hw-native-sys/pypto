# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A compiled code unit can serve several runtime tensor signatures."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from pypto.runtime._binary_cache import BinaryCacheContext


def setup_assembly(device_runner, monkeypatch, tmp_path, kernels, *, fail=False):
    for kernel in kernels:
        source = tmp_path / Path(kernel["source"]).name
        source.write_text("// kernel source")
        kernel["source"] = str(source)
    (tmp_path / "kernel_config.py").write_text(
        f"KERNELS = {kernels!r}\nORCHESTRATION = {{'source': 'main.cpp', 'function_name': 'entry'}}\n"
    )
    compiled, core_calls, chip_calls = [], [], []
    monkeypatch.setattr(device_runner, "ensure_pto_isa_root", lambda: "/pto-isa")
    monkeypatch.setattr(
        device_runner,
        "KernelCompiler",
        lambda **kw: SimpleNamespace(
            get_incore_include_dirs=lambda: [], get_kernel_include_dirs=lambda _: []
        ),
    )
    monkeypatch.setattr(
        device_runner,
        "_current_binary_context",
        lambda *a, **kw: BinaryCacheContext(
            platform="a2a3sim",
            runtime_name="tensormap_and_ringbuffer",
            runtime_revision="a" * 40,
            pto_isa_revision="b" * 40,
        ),
    )
    monkeypatch.setattr(device_runner, "_compile_single_orchestration", lambda *a, **kw: b"orch")
    monkeypatch.setattr(device_runner, "register_callable_identity", lambda *a: None)

    def compile_kernel(kernel, *a, **kw):
        compiled.append(kernel)
        if fail:
            raise RuntimeError("compile failed")
        return b"raw", b"kernel"

    def core_build(**kw):
        core_calls.append(kw)
        return kw

    def chip_build(**kw):
        chip_calls.append(kw)
        return kw

    monkeypatch.setattr(device_runner, "_compile_single_kernel", compile_kernel)
    monkeypatch.setattr(device_runner, "CoreCallable", SimpleNamespace(build=core_build))
    monkeypatch.setattr(device_runner, "ChipCallable", SimpleNamespace(build=chip_build))
    return compiled, core_calls, chip_calls


def registrations(**changes):
    code = {"code_name": "step", "source": "step.cpp", "core_type": "aiv", **changes}
    return [{**code, "func_id": 0, "signature": ["IN"]}, {**code, "func_id": 1, "signature": ["INOUT"]}]


def test_two_signatures_compile_once(device_runner, monkeypatch, tmp_path):
    compiled, cores, chips = setup_assembly(device_runner, monkeypatch, tmp_path, registrations())
    result, _, _ = device_runner._compile_and_assemble_locked(tmp_path, "a2a3sim")
    assert len(compiled) == 1
    assert [c["signature"] for c in cores] == [["IN"], ["INOUT"]]
    assert cores[0]["binary"] is cores[1]["binary"]
    assert [fid for fid, _ in result["children"]] == [0, 1]
    assert len(chips) == 1


@pytest.mark.parametrize("change", [{"external": True}, {"builtin": True}, {"code_name": None}])
def test_external_and_legacy_configs_keep_separate_compiles(device_runner, monkeypatch, tmp_path, change):
    compiled, _, _ = setup_assembly(device_runner, monkeypatch, tmp_path, registrations(**change))
    device_runner._compile_and_assemble_locked(tmp_path, "a2a3sim")
    assert len(compiled) == 2


@pytest.mark.parametrize(
    "change", [{"core_type": "aic"}, {"extra_include_dirs": ["other"]}, {"source": "other.cpp"}]
)
def test_different_build_contracts_do_not_share(device_runner, monkeypatch, tmp_path, change):
    kernels = registrations()
    kernels[1].update(change)
    compiled, _, _ = setup_assembly(device_runner, monkeypatch, tmp_path, kernels)
    device_runner._compile_and_assemble_locked(tmp_path, "a2a3sim")
    assert len(compiled) == 2


def test_compile_failure_does_not_create_partial_callables(device_runner, monkeypatch, tmp_path):
    _, cores, chips = setup_assembly(device_runner, monkeypatch, tmp_path, registrations(), fail=True)
    with pytest.raises(RuntimeError, match="compile failed"):
        device_runner._compile_and_assemble_locked(tmp_path, "a2a3sim")
    assert not cores and not chips


def test_generated_cache_uses_code_identity_instead_of_registration_id(device_runner, tmp_path):
    a, b = registrations()
    source = tmp_path / "step.cpp"
    source.write_text("// generated source")
    a["source"] = b["source"] = str(source)
    args = ("a2a3sim", "/pto-isa", "runtime", object())
    assert device_runner._kernel_cache_file(tmp_path, a, *args) == device_runner._kernel_cache_file(
        tmp_path, b, *args
    )
    b["extra_include_dirs"] = ["other"]
    assert device_runner._kernel_cache_file(tmp_path, a, *args) != device_runner._kernel_cache_file(
        tmp_path, b, *args
    )


def test_generated_cache_tracks_source_and_local_includes(device_runner, tmp_path):
    source = tmp_path / "step.cpp"
    header = tmp_path / "step.h"
    source.write_text('#include "step.h"\n')
    header.write_text("constexpr int value = 1;")
    kernel = registrations()[0]
    kernel["source"] = str(source)
    args = ("a2a3sim", "/pto-isa", "runtime", object())
    original = device_runner._kernel_cache_file(tmp_path, kernel, *args)
    header.write_text("constexpr int value = 2;")
    changed_header = device_runner._kernel_cache_file(tmp_path, kernel, *args)
    source.write_text('#include "step.h"\n// changed source\n')
    assert original != changed_header
    assert changed_header != device_runner._kernel_cache_file(tmp_path, kernel, *args)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
