# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression tests for compiled kernel binary cache identity."""

import importlib

import pytest
from pypto._external_source import kernel_binary_cache_path


class _Compiler:
    def __init__(self, *, runtime_include_dirs=()):
        self.calls = []
        self.runtime_include_dirs = list(runtime_include_dirs)

    def get_incore_include_dirs(self):
        return []

    def get_kernel_include_dirs(self, runtime_name):
        del runtime_name
        return self.runtime_include_dirs

    def compile_incore(self, source, **kwargs):
        self.calls.append((source, kwargs))
        return f"{kwargs['core_type']} binary".encode()


def _kernel(source, core_type, func_id):
    return {
        "source": str(source),
        "core_type": core_type,
        "func_id": func_id,
        "external": True,
    }


def _cache_path(
    cache_dir,
    kernel,
    platform,
    *,
    pto_isa_root="/pto-isa",
    runtime_name="runtime",
    include_dirs=(),
):
    return kernel_binary_cache_path(
        cache_dir,
        source=kernel["source"],
        core_type=kernel["core_type"],
        func_id=kernel["func_id"],
        platform=platform,
        external=kernel["external"],
        pto_isa_root=pto_isa_root,
        runtime_name=runtime_name,
        include_dirs=include_dirs,
    )


def test_external_cache_separates_core_types_and_tracks_includes(tmp_path):
    include = tmp_path / "impl.cce"
    include.write_text("constexpr int VALUE = 1;\n")
    source = tmp_path / "entry.cpp"
    source.write_text('#include "impl.cce"\n')
    cache = tmp_path / "cache"
    aic = _kernel(source, "aic", 0)
    aiv = _kernel(source, "aiv", 1)

    aic_path = _cache_path(cache, aic, "a2a3")
    aiv_path = _cache_path(cache, aiv, "a2a3")
    assert aic_path != aiv_path

    include.write_text("constexpr int VALUE = 2;\n")
    changed_path = _cache_path(cache, aic, "a2a3")
    assert changed_path != aic_path


def test_external_cache_tracks_compiler_include_headers_at_same_root(tmp_path):
    pto_isa_root = tmp_path / "pto-isa"
    pto_include = pto_isa_root / "include"
    runtime_include = tmp_path / "runtime"
    pto_include.mkdir(parents=True)
    runtime_include.mkdir()
    intrinsic = pto_include / "intrinsic.h"
    runtime_header = runtime_include / "runtime_header.h"
    intrinsic.write_text("constexpr int PTO_VALUE = 1;\n")
    runtime_header.write_text("constexpr int RUNTIME_VALUE = 1;\n")
    source = tmp_path / "entry.cpp"
    source.write_text("#include <intrinsic.h>\n#include <runtime_header.h>\n")
    kernel = _kernel(source, "aiv", 0)
    include_dirs = [runtime_include]

    initial_path = _cache_path(
        tmp_path / "cache",
        kernel,
        "a2a3",
        pto_isa_root=str(pto_isa_root),
        include_dirs=include_dirs,
    )
    intrinsic.write_text("constexpr int PTO_VALUE = 2;\n")
    pto_changed_path = _cache_path(
        tmp_path / "cache",
        kernel,
        "a2a3",
        pto_isa_root=str(pto_isa_root),
        include_dirs=include_dirs,
    )
    runtime_header.write_text("constexpr int RUNTIME_VALUE = 2;\n")
    runtime_changed_path = _cache_path(
        tmp_path / "cache",
        kernel,
        "a2a3",
        pto_isa_root=str(pto_isa_root),
        include_dirs=include_dirs,
    )

    assert pto_changed_path != initial_path
    assert runtime_changed_path != pto_changed_path


def test_external_compile_ignores_source_sidecar(tmp_path):
    pytest.importorskip("simpler_setup", reason="requires the optional device compiler package")
    device_runner = importlib.import_module("pypto.runtime.device_runner")
    source = tmp_path / "entry.cpp"
    source.write_text("// external\n")
    sidecar = source.with_suffix(".so")
    sidecar.write_bytes(b"stale wrong-core binary")
    compiler_stub = _Compiler()

    raw, kernel_binary = device_runner.compile_single_kernel(
        _kernel(source, "aiv", 7),
        compiler_stub,
        "a2a3sim",
        "/pto-isa",
        "runtime",
        cache_dir=tmp_path / "cache",
    )

    assert raw == b"aiv binary"
    assert kernel_binary == raw
    assert [call[1]["core_type"] for call in compiler_stub.calls] == ["aiv"]
    assert sidecar.read_bytes() == b"stale wrong-core binary"


def test_external_compile_forwards_descriptor_include_dirs(tmp_path):
    pytest.importorskip("simpler_setup", reason="requires the optional device compiler package")
    device_runner = importlib.import_module("pypto.runtime.device_runner")
    source = tmp_path / "entry.cpp"
    source.write_text("// external\n")
    include_dir = tmp_path / "include"
    include_dir.mkdir()
    kernel = _kernel(source, "aiv", 8)
    kernel["extra_include_dirs"] = [str(include_dir)]
    compiler_stub = _Compiler()

    device_runner.compile_single_kernel(
        kernel,
        compiler_stub,
        "a2a3sim",
        "/pto-isa",
        "runtime",
    )

    assert compiler_stub.calls[0][1]["extra_include_dirs"] == [str(include_dir)]


def test_generated_cache_separates_simulator_and_device_platforms(tmp_path):
    source = tmp_path / "entry.cpp"
    source.write_text("// generated\n")
    kernel = _kernel(source, "aiv", 0)
    kernel["external"] = False

    sim_path = _cache_path(tmp_path, kernel, "a2a3sim")
    device_path = _cache_path(tmp_path, kernel, "a2a3")

    assert sim_path != device_path


def test_external_cache_is_relocatable(tmp_path):
    cache = tmp_path / "cache"

    def build_tree(root):
        pto_include = root / "pto-isa" / "include"
        runtime_include = root / "runtime"
        source_dir = root / "kernel"
        pto_include.mkdir(parents=True)
        runtime_include.mkdir()
        source_dir.mkdir()
        (pto_include / "intrinsic.h").write_text("constexpr int PTO_VALUE = 1;\n")
        (runtime_include / "runtime_header.h").write_text("constexpr int RUNTIME_VALUE = 1;\n")
        source = source_dir / "entry.cpp"
        source.write_text("#include <intrinsic.h>\n#include <runtime_header.h>\n")
        return source, root / "pto-isa", runtime_include

    source_a, pto_root_a, runtime_a = build_tree(tmp_path / "host-a")
    source_b, pto_root_b, runtime_b = build_tree(tmp_path / "host-b")

    path_a = _cache_path(
        cache,
        _kernel(source_a, "aiv", 0),
        "a2a3",
        pto_isa_root=pto_root_a,
        include_dirs=[runtime_a],
    )
    path_b = _cache_path(
        cache,
        _kernel(source_b, "aiv", 0),
        "a2a3",
        pto_isa_root=pto_root_b,
        include_dirs=[runtime_b],
    )

    assert path_a == path_b


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
