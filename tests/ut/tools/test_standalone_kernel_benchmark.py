# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the standalone InCore NPU benchmark infrastructure."""

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

_SKILL_DIR = Path(__file__).parents[3] / ".claude" / "skills" / "incore-profiling"


def _load_script(name: str) -> ModuleType:
    path = _SKILL_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(_SKILL_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(_SKILL_DIR))
    return module


@pytest.fixture(scope="module")
def generator() -> ModuleType:
    return _load_script("gen_profiling_case")


@pytest.fixture(scope="module")
def comparison() -> ModuleType:
    return _load_script("standalone_compare")


@pytest.fixture(scope="module")
def multi_comparison() -> ModuleType:
    return _load_script("standalone_multi_compare")


@pytest.fixture(scope="module")
def panel_summary() -> ModuleType:
    return _load_script("summarize_dsa_device_panel")


@pytest.fixture(scope="module")
def preflight() -> ModuleType:
    return _load_script("preflight_standalone_comparison")


def _write_kernel(root: Path, *, mixed: bool = False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    cpp = root / "kernel.cpp"
    if mixed:
        signature = "\n".join(
            (
                "AICORE void sample_aic(__gm__ float* v0, int32_t block_idx, int32_t block_num) {}",
                "AICORE void sample_aiv(__gm__ float* v0, int32_t block_idx, "
                "int32_t block_num, int32_t subblock_idx) {}",
            )
        )
    else:
        signature = 'extern "C" __global__ AICORE void sample(__gm__ float* v0, int32_t n) {}'
    cpp.write_text(signature + "\n", encoding="utf-8")
    pto = """\
func.func @sample(%arg0: !pto.ptr<f32>, %arg1: i32) {
  %view = pto.make_tensor_view %arg0, shape = [%c8_index], strides = [%c1_index]
}
"""
    if mixed:
        pto = """\
func.func @sample_aic(%arg0: !pto.ptr<f32>, %__pypto_spmd_block_idx: i32,
    %__pypto_spmd_block_num: i32) attributes {pto.kernel_kind = #pto.kernel_kind<cube>} {
}
func.func @sample_aiv(%arg0: !pto.ptr<f32>, %__pypto_spmd_block_idx: i32,
    %__pypto_spmd_block_num: i32, %__pypto_spmd_subblock_idx: i32)
    attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
  %view = pto.make_tensor_view %arg0, shape = [%c8_index], strides = [%c1_index]
}
"""
    cpp.with_suffix(".pto").write_text(pto, encoding="utf-8")
    return cpp


def test_generate_npu_case_with_real_inputs(generator: ModuleType, tmp_path: Path):
    kernel = _write_kernel(tmp_path)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "v0.bin").write_bytes(bytes(range(64)))

    case = generator.generate(
        kernel,
        "compact_sample",
        tmp_path / "output",
        "dav-c220",
        run_mode="npu",
        block_dim=8,
        input_dir=inputs,
        scalar_values={"n": "16"},
    )

    manifest = json.loads((case / "standalone_manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_mode"] == "npu"
    assert manifest["block_dim"] == 8
    assert manifest["parameters"] == [
        {"cpp_type": "float", "elements": 16, "kind": "pointer", "name": "v0"},
        {"cpp_type": "int32_t", "kind": "scalar", "name": "n", "value": "16"},
    ]
    assert (case / "v0.bin").read_bytes() == bytes(range(64))

    cmake = (case / "CMakeLists.txt").read_text(encoding="utf-8")
    assert 'option(ENABLE_SIM_GOLDEN "Build Ascend simulator (camodel) executable" OFF)' in cmake
    assert 'option(ENABLE_NPU_BENCHMARK "Build real-device standalone benchmark executable" ON)' in cmake
    assert "compact_sample_kernel runtime" in cmake
    main = (case / "main.cpp").read_text(encoding="utf-8")
    assert "aclrtEventElapsedTime" in main
    assert "PYPTO_BENCH_ROUNDS" in main
    assert "stream, 8);" in main
    launch = (case / "launch.cpp").read_text(encoding="utf-8")
    assert "sample<<<blockDim, nullptr, stream>>>" in launch


def test_physical_extent_analysis_unions_spmd_partition_offsets(generator: ModuleType):
    cpp = """\
AICORE void sample(
    __gm__ bfloat16_t* v4, int64_t row, int32_t block_idx, int32_t block_num) {
  const int64_t stop = 16;
  const int64_t step = 8;
  const int64_t tile_rows = 64;
  const int64_t row_stride = 1024;
  for (int64_t outer = (int64_t) block_idx; outer < stop; outer += step) {
    int64_t tile_offset = (int64_t) ((uint64_t) outer * (uint64_t) tile_rows);
    GlobalTensor<
        bfloat16_t,
        pto::Shape<1, 1, 1, 128, 64>,
        pto::Stride<131072, 131072, 131072, 1024, 1>,
        pto::Layout::ND> view = GlobalTensor<
            bfloat16_t,
            pto::Shape<1, 1, 1, 128, 64>,
            pto::Stride<131072, 131072, 131072, 1024, 1>,
            pto::Layout::ND>(v4 + row * row_stride + tile_offset, shape, stride);
  }
}
"""

    result = generator.analyze_pointer_extents(
        cpp,
        {"v4"},
        {"row": 4992, "block_idx": 0, "block_num": 8},
        spmd_block_index="block_idx",
    )

    assert result.unresolved == ()
    assert result.pointers["v4"].max_base_offset == 5_112_768
    assert result.pointers["v4"].required_elements == 5_242_880


def test_physical_extent_analysis_accepts_singleton_sentinel_strides(generator: ModuleType):
    cpp = """\
GlobalTensor<
    float,
    pto::Shape<1, 1, 1, 1, 8>,
    pto::Stride<-1, -1, -1, -1, 1>,
    pto::Layout::ND> view = GlobalTensor<
        float,
        pto::Shape<1, 1, 1, 1, 8>,
        pto::Stride<-1, -1, -1, -1, 1>,
        pto::Layout::ND>(v0, shape, stride);
"""

    result = generator.analyze_pointer_extents(cpp, {"v0"}, {})

    assert result.unresolved == ()
    assert result.pointers["v0"].required_elements == 8


def test_physical_extent_analysis_bounds_nonmonotone_offsets(generator: ModuleType):
    cpp = """\
for (int64_t i = 0; i < 10; i += 1) {
  GlobalTensor<
      float,
      pto::Shape<1>,
      pto::Stride<1>,
      pto::Layout::ND> view = GlobalTensor<
          float,
          pto::Shape<1>,
          pto::Stride<1>,
          pto::Layout::ND>(v0 + 100 - i, shape, stride);
}
"""

    result = generator.analyze_pointer_extents(cpp, {"v0"}, {})

    assert result.pointers["v0"].max_base_offset == 100
    assert result.pointers["v0"].required_elements == 101


@pytest.mark.parametrize(
    ("offset", "required_elements"),
    [
        (-5, 8),
        (12, 20),
    ],
)
@pytest.mark.parametrize(
    "clamp",
    [
        "offset < zero ? zero : offset",
        "offset > zero ? offset : zero",
    ],
)
def test_physical_extent_analysis_bounds_codegen_max_clamp(
    generator: ModuleType,
    offset: int,
    required_elements: int,
    clamp: str,
):
    cpp = f"""\
const int64_t zero = 0;
int64_t clamped = {clamp};
GlobalTensor<
    float,
    pto::Shape<8>,
    pto::Stride<1>,
    pto::Layout::ND> view = GlobalTensor<
        float,
        pto::Shape<8>,
        pto::Stride<1>,
        pto::Layout::ND>(v0 + clamped, shape, stride);
"""

    result = generator.analyze_pointer_extents(cpp, {"v0"}, {"offset": offset})

    assert result.unresolved == ()
    assert result.pointers["v0"].required_elements == required_elements


def test_physical_extent_analysis_bounds_inline_max_clamp_over_loop(generator: ModuleType):
    cpp = """\
const int64_t zero = 0;
const int64_t stride = 16;
for (int64_t i = -2; i < 4; i += 1) {
  GlobalTensor<
      float,
      pto::Shape<8>,
      pto::Stride<1>,
      pto::Layout::ND> view = GlobalTensor<
          float,
          pto::Shape<8>,
          pto::Stride<1>,
          pto::Layout::ND>(v0 + (i < zero ? zero : i) * stride, shape, stride);
}
"""

    result = generator.analyze_pointer_extents(cpp, {"v0"}, {})

    assert result.unresolved == ()
    assert result.pointers["v0"].max_base_offset == 48
    assert result.pointers["v0"].required_elements == 56


def test_physical_extent_analysis_rejects_non_max_ternary(generator: ModuleType):
    cpp = """\
const int64_t zero = 0;
int64_t selected = offset < zero ? offset : zero;
GlobalTensor<
    float,
    pto::Shape<8>,
    pto::Stride<1>,
    pto::Layout::ND> view = GlobalTensor<
        float,
        pto::Shape<8>,
        pto::Stride<1>,
        pto::Layout::ND>(v0 + selected, shape, stride);
"""

    result = generator.analyze_pointer_extents(cpp, {"v0"}, {"offset": 12})

    assert result.unresolved == ("v0: selected ('selected')",)


def test_physical_extent_analysis_rejects_composite_max_lookalike(generator: ModuleType):
    cpp = """\
const int64_t zero = 0;
const int64_t one = 1;
int64_t selected = offset + one < zero ? zero : one;
GlobalTensor<
    float,
    pto::Shape<8>,
    pto::Stride<1>,
    pto::Layout::ND> view = GlobalTensor<
        float,
        pto::Shape<8>,
        pto::Stride<1>,
        pto::Layout::ND>(v0 + selected, shape, stride);
"""

    result = generator.analyze_pointer_extents(cpp, {"v0"}, {"offset": 12})

    assert result.unresolved == ("v0: selected ('selected')",)


def test_physical_extent_analysis_reports_unsupported_pointer_bases(generator: ModuleType):
    cpp = """\
GlobalTensor<
    float,
    pto::Shape<8>,
    pto::Stride<1>,
    pto::Layout::ND> view = GlobalTensor<
        float,
        pto::Shape<8>,
        pto::Stride<1>,
        pto::Layout::ND>(select_pointer(v0), shape, stride);
"""

    result = generator.analyze_pointer_extents(cpp, {"v0"}, {})

    assert result.unresolved == ("v0: unsupported base expression 'select_pointer(v0)'",)


def _write_strided_partition_kernel(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    cpp = root / "partition.cpp"
    cpp.write_text(
        """\
extern "C" __global__ AICORE void sample(__gm__ float* v0, int64_t row) {
  const int64_t stride = 4;
  GlobalTensor<
      float,
      pto::Shape<1, 1, 1, 2>,
      pto::Stride<8, 8, 8, 4>,
      pto::Layout::ND> view = GlobalTensor<
          float,
          pto::Shape<1, 1, 1, 2>,
          pto::Stride<8, 8, 8, 4>,
          pto::Layout::ND>(v0 + row * stride, shape, layout);
}
""",
        encoding="utf-8",
    )
    cpp.with_suffix(".pto").write_text(
        """\
func.func @sample(%arg0: !pto.ptr<f32>, %arg1: index) {
  %view = pto.make_tensor_view %arg0, shape = [%c8_index], strides = [%c1_index]
}
""",
        encoding="utf-8",
    )
    return cpp


def test_generate_npu_case_sizes_full_partition_span(generator: ModuleType, tmp_path: Path):
    kernel = _write_strided_partition_kernel(tmp_path / "kernel")

    case = generator.generate(
        kernel,
        "partition",
        tmp_path / "output",
        "dav-c220",
        run_mode="npu",
        scalar_values={"row": "4"},
        synthetic_seed=19,
    )

    manifest = json.loads((case / "standalone_manifest.json").read_text(encoding="utf-8"))
    assert manifest["parameters"][0]["elements"] == 21
    assert manifest["physical_extent_analysis"]["pointers"]["v0"] == {
        "max_base_offset": 16,
        "required_elements": 21,
        "shape": [1, 1, 1, 2],
        "stride": [8, 8, 8, 4],
        "view_count": 1,
    }
    assert (case / "v0.bin").stat().st_size == 21 * 4


def test_generate_npu_case_rejects_truncated_partition_input(generator: ModuleType, tmp_path: Path):
    kernel = _write_strided_partition_kernel(tmp_path / "kernel")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    np.zeros(8, dtype=np.float32).tofile(inputs / "v0.bin")

    with pytest.raises(ValueError, match="full physical buffer"):
        generator.generate(
            kernel,
            "partition",
            tmp_path / "output",
            "dav-c220",
            run_mode="npu",
            input_dir=inputs,
            scalar_values={"row": "4"},
        )


def _write_fake_ptoas_generator(root: Path) -> Path:
    script = root / "test" / "npu_validation" / "scripts" / "generate_testcase.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        """
def _describe_kernel_source(text):
    return {
        "kind": "mixed",
        "kernel_name": "sample",
        "raw_params": ["__gm__ float* v0", "int32_t block_idx", "int32_t block_num", "int32_t subblock_idx"],
        "aic_text": "AICORE void sample_aic(__gm__ float* v0, int32_t block_idx, int32_t block_num) {}",
        "aiv_text": (
            "AICORE void sample_aiv(__gm__ float* v0, int32_t block_idx, "
            "int32_t block_num, int32_t subblock_idx) {}"
        ),
    }

def _append_mixed_kernel_wrapper(text, name, raw_params, aic_text, aiv_text):
    del aic_text, aiv_text
    return text + '\\n// PTOAS_CANONICAL_MIXED_WRAPPER\\n' + (
        'extern "C" __global__ AICORE void sample(__gm__ float* v0, int32_t block_idx, '
        'int32_t block_num, int32_t subblock_idx) {}\\n'
    )
""".lstrip(),
        encoding="utf-8",
    )
    return root


def test_generate_npu_mixed_requires_ptoas_group_wrapper(generator: ModuleType, tmp_path: Path):
    kernel = _write_kernel(tmp_path, mixed=True)
    with pytest.raises(ValueError, match="requires --ptoas-root"):
        generator.generate(
            kernel,
            "mixed",
            tmp_path / "output",
            "dav-c220",
            run_mode="npu",
        )


def test_generate_npu_mixed_uses_ptoas_group_wrapper(
    generator: ModuleType, comparison: ModuleType, tmp_path: Path
):
    kernel = _write_kernel(tmp_path / "kernel", mixed=True)
    ptoas_root = _write_fake_ptoas_generator(tmp_path / "PTOAS")
    compact = generator.generate(
        kernel,
        "mixed",
        tmp_path / "compact",
        "dav-c220",
        run_mode="npu",
        block_dim=8,
        synthetic_seed=19,
        ptoas_root=ptoas_root,
    )
    loose = generator.generate(
        kernel,
        "mixed",
        tmp_path / "loose",
        "dav-c220",
        run_mode="npu",
        block_dim=8,
        synthetic_seed=19,
        ptoas_root=ptoas_root,
    )

    generated = (compact / "mixed_kernel.cpp").read_text(encoding="utf-8")
    assert "PTOAS_CANONICAL_MIXED_WRAPPER" in generated
    assert "#if defined(__DAV_CUBE__)" not in generated
    assert "int32_t block_idx = static_cast<int32_t>(get_block_idx());" in generated
    assert "int32_t block_num = static_cast<int32_t>(get_block_num());" in generated
    assert "int32_t subblock_idx = static_cast<int32_t>(get_subblockid());" in generated
    assert "void sample(__gm__ float* v0)" in generated
    manifest = json.loads((compact / "standalone_manifest.json").read_text(encoding="utf-8"))
    assert manifest["mixed"] is True
    assert manifest["mixed_runner"]["kind"] == "ptoas_validation_group_wrapper"
    assert manifest["mixed_runner"]["identity_source"] == "direct_launch_builtins"
    assert manifest["parameters"] == [{"cpp_type": "float", "elements": 8, "kind": "pointer", "name": "v0"}]
    validated, pointers = comparison.validate_cases(compact, loose)
    assert validated["kernel"] == "sample"
    assert pointers == ["v0"]


def test_generate_npu_case_with_synthetic_inputs(generator: ModuleType, tmp_path: Path):
    kernel = tmp_path / "kernel.cpp"
    kernel.write_text(
        'extern "C" __global__ AICORE void sample(__gm__ bfloat16_t* v0, int32_t n) {}\n',
        encoding="utf-8",
    )
    kernel.with_suffix(".pto").write_text(
        """\
func.func @sample(%arg0: !pto.ptr<bf16>, %arg1: i32) {
  %view = pto.make_tensor_view %arg0, shape = [%c8_index], strides = [%c1_index]
}
""",
        encoding="utf-8",
    )

    case = generator.generate(
        kernel,
        "synthetic_sample",
        tmp_path / "output",
        "dav-c220",
        run_mode="npu",
        block_dim=4,
        scalar_values={"n": "8"},
        synthetic_seed=19,
    )
    repeated = generator.generate(
        kernel,
        "synthetic_sample_repeated",
        tmp_path / "repeated",
        "dav-c220",
        run_mode="npu",
        block_dim=4,
        scalar_values={"n": "8"},
        synthetic_seed=19,
    )

    raw = np.fromfile(case / "v0.bin", dtype=np.uint16)
    fp32 = (raw.astype(np.uint32) << 16).view(np.float32)
    assert len(raw) == 8
    assert np.isfinite(fp32).all()
    assert np.any(fp32 != 0.0)
    assert (case / "v0.bin").read_bytes() == (repeated / "v0.bin").read_bytes()
    manifest = json.loads((case / "standalone_manifest.json").read_text(encoding="utf-8"))
    assert manifest["input_source"] == {"kind": "synthetic", "seed": 19}


def test_generate_npu_case_from_invocation_profile(generator: ModuleType, tmp_path: Path):
    kernel = tmp_path / "integer_kernel.cpp"
    kernel.write_text(
        'extern "C" __global__ AICORE void sample(__gm__ int32_t* v0, int32_t n) {}\n',
        encoding="utf-8",
    )
    kernel.with_suffix(".pto").write_text(
        """\
func.func @sample(%arg0: !pto.ptr<i32>, %arg1: i32) {
  %view = pto.make_tensor_view %arg0, shape = [%c8_index], strides = [%c1_index]
}
""",
        encoding="utf-8",
    )
    profile = tmp_path / "invocation.json"
    profile.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "block_dim": 4,
                "input": {"kind": "synthetic", "seed": 23},
                "scalars": {"n": 8},
                "pointer_fills": {"v0": 7},
                "outputs": ["v0"],
            }
        ),
        encoding="utf-8",
    )

    assert (
        generator.main(
            [
                "--input",
                str(kernel),
                "--testcase",
                "profiled",
                "--output-root",
                str(tmp_path / "output"),
                "--run-mode",
                "npu",
                "--invocation-profile",
                str(profile),
            ]
        )
        == 0
    )
    case = tmp_path / "output" / "ptoas" / "profiled"
    assert np.array_equal(np.fromfile(case / "v0.bin", dtype=np.int32), np.full(8, 7, dtype=np.int32))
    manifest = json.loads((case / "standalone_manifest.json").read_text(encoding="utf-8"))
    assert manifest["block_dim"] == 4
    assert manifest["recommended_outputs"] == ["v0"]
    assert manifest["input_source"] == {
        "kind": "synthetic",
        "pointer_fills": {"v0": 7},
        "seed": 23,
    }
    assert manifest["invocation_profile"]["path"] == "invocation.json"
    assert len(manifest["invocation_profile"]["sha256"]) == 64


def test_invocation_profile_rejects_invalid_controls(generator: ModuleType, tmp_path: Path):
    profile = tmp_path / "invalid.json"
    profile.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "input": {"kind": "files"},
                "pointer_fills": {"indices": 1},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="pointer_fills require synthetic input"):
        generator.load_invocation_profile(profile)


def test_preflight_sync_summary_requires_one_target(preflight: ModuleType, tmp_path: Path):
    summary = tmp_path / "summary.jsonl"
    summary.write_text(
        "\n".join(
            [
                '{"function":"sample","active_sync_groups":2}',
                '{"function":"sibling","active_sync_groups":3}',
            ]
        ),
        encoding="utf-8",
    )
    assert preflight._load_target_sync_summary(summary, "sample")["active_sync_groups"] == 2
    with pytest.raises(ValueError, match="exactly one synchronization summary"):
        preflight._load_target_sync_summary(summary, "missing")


def test_generate_npu_requires_explicit_scalars(generator: ModuleType, tmp_path: Path):
    kernel = _write_kernel(tmp_path)
    with pytest.raises(ValueError, match="require every scalar ABI argument explicitly"):
        generator.generate(
            kernel,
            "missing_scalar",
            tmp_path / "output",
            "dav-c220",
            run_mode="npu",
            synthetic_seed=19,
        )


def test_generate_npu_synthetic_integer_input_is_safe(generator: ModuleType, tmp_path: Path):
    kernel = tmp_path / "integer_kernel.cpp"
    kernel.write_text(
        'extern "C" __global__ AICORE void sample(__gm__ int32_t* v0) {}\n',
        encoding="utf-8",
    )
    kernel.with_suffix(".pto").write_text(
        """\
func.func @sample(%arg0: !pto.ptr<i32>) {
  %view = pto.make_tensor_view %arg0, shape = [%c8_index], strides = [%c1_index]
}
""",
        encoding="utf-8",
    )

    case = generator.generate(
        kernel,
        "integer_sample",
        tmp_path / "output",
        "dav-c220",
        run_mode="npu",
        synthetic_seed=19,
    )

    assert np.array_equal(np.fromfile(case / "v0.bin", dtype=np.int32), np.zeros(8, dtype=np.int32))


def test_generate_npu_requires_input_source(generator: ModuleType, tmp_path: Path):
    kernel = _write_kernel(tmp_path)
    with pytest.raises(ValueError, match="require one input source"):
        generator.generate(
            kernel,
            "no_inputs",
            tmp_path / "output",
            "dav-c220",
            run_mode="npu",
            scalar_values={"n": "8"},
        )


def test_generate_npu_case_from_exact_args_dump(
    generator: ModuleType, comparison: ModuleType, tmp_path: Path
):
    kernel = _write_kernel(tmp_path)
    dump_dir = tmp_path / "args_dump"
    dump_dir.mkdir()
    before = bytes(range(32))
    after = bytes(reversed(range(32)))
    (dump_dir / "args.bin").write_bytes(before + after)

    def tensor(stage: str, offset: int) -> dict:
        return {
            "task_id": "0x0000000100000007",
            "func_id": [4],
            "arg_index": 0,
            "role": "inout",
            "stage": stage,
            "kind": "tensor",
            "dtype": "float32",
            "is_contiguous": True,
            "shape": [8],
            "strides": [1],
            "start_offset": 0,
            "bin_offset": offset,
            "bin_size": 32,
            "truncated": False,
            "overwritten": False,
        }

    dump = {
        "bin_file": "args.bin",
        "args": [
            tensor("before_dispatch", 0),
            tensor("after_completion", 32),
            {
                "task_id": "0x0000000100000007",
                "func_id": [4],
                "arg_index": 1,
                "role": "input",
                "stage": "before_dispatch",
                "kind": "scalar",
                "value": 8,
            },
        ],
    }
    manifest_path = dump_dir / "args_dump.json"
    manifest_path.write_text(json.dumps(dump), encoding="utf-8")

    case = generator.generate(
        kernel,
        "captured_sample",
        tmp_path / "output",
        "dav-c220",
        run_mode="npu",
        block_dim=8,
        dump_selection=generator.DumpSelection(manifest_path, 4),
    )

    manifest = json.loads((case / "standalone_manifest.json").read_text(encoding="utf-8"))
    assert manifest["capture"] == {
        "func_id": 4,
        "recommended_outputs": ["v0"],
        "roles": {"v0": "inout"},
        "task_id": "0x0000000100000007",
    }
    assert manifest["parameters"][0]["elements"] == 8
    assert manifest["parameters"][1]["value"] == "8"
    assert (case / "v0.bin").read_bytes() == before
    assert (case / "captured_expected" / "v0.bin").read_bytes() == after

    compact_dump = tmp_path / "compact_dump"
    loose_dump = tmp_path / "loose_dump"
    compact_dump.mkdir()
    loose_dump.mkdir()
    (compact_dump / "v0.bin").write_bytes(after)
    (loose_dump / "v0.bin").write_bytes(after)
    hashes = comparison._compare_outputs(
        compact_dump,
        loose_dump,
        ["v0"],
        expected_dir=case / "captured_expected",
    )
    assert len(set(hashes["v0"].values())) == 1


def test_args_dump_requires_unambiguous_dispatch(generator: ModuleType, tmp_path: Path):
    kernel = _write_kernel(tmp_path)
    dump_dir = tmp_path / "args_dump"
    dump_dir.mkdir()
    (dump_dir / "args.bin").write_bytes(bytes(64))
    entries = []
    for task_id, offset in (("0x1", 0), ("0x2", 32)):
        entries.extend(
            [
                {
                    "task_id": task_id,
                    "func_id": [4],
                    "arg_index": 0,
                    "role": "input",
                    "stage": "before_dispatch",
                    "kind": "tensor",
                    "is_contiguous": True,
                    "bin_offset": offset,
                    "bin_size": 32,
                    "truncated": False,
                    "overwritten": False,
                },
                {
                    "task_id": task_id,
                    "func_id": [4],
                    "arg_index": 1,
                    "role": "input",
                    "stage": "before_dispatch",
                    "kind": "scalar",
                    "value": 8,
                },
            ]
        )
    manifest_path = dump_dir / "args_dump.json"
    manifest_path.write_text(json.dumps({"bin_file": "args.bin", "args": entries}), encoding="utf-8")

    with pytest.raises(ValueError, match="select one"):
        generator.generate(
            kernel,
            "ambiguous",
            tmp_path / "output",
            "dav-c220",
            run_mode="npu",
            dump_selection=generator.DumpSelection(manifest_path, 4),
        )


def test_generate_sim_case_remains_single_core(generator: ModuleType, tmp_path: Path):
    kernel = _write_kernel(tmp_path)
    case = generator.generate(kernel, "sim_sample", tmp_path / "output", "dav-c220")
    cmake = (case / "CMakeLists.txt").read_text(encoding="utf-8")
    assert 'option(ENABLE_SIM_GOLDEN "Build Ascend simulator (camodel) executable" ON)' in cmake
    assert 'option(ENABLE_NPU_BENCHMARK "Build real-device standalone benchmark executable" OFF)' in cmake
    assert "stream, 1);" in (case / "main.cpp").read_text(encoding="utf-8")


def test_preflight_resolves_compiles_and_validates_endpoints(
    generator: ModuleType,
    preflight: ModuleType,
    tmp_path: Path,
    monkeypatch,
):
    del generator
    builds = {}
    for label in ("baseline", "candidate"):
        build = tmp_path / label
        ptoas = build / "ptoas"
        ptoas.mkdir(parents=True)
        (ptoas / "group.pto").write_text(
            """\
func.func @sample(%arg0: !pto.ptr<i32>, %arg1: i32) {
  %tile = pto.alloc_tile : !pto.tile<8xi32>, addr = 0
  %view = pto.make_tensor_view %arg0, shape = [%c8_index], strides = [%c1_index]
}
""",
            encoding="utf-8",
        )
        builds[label] = build

    profile = tmp_path / "invocation.json"
    profile.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "block_dim": 2,
                "input": {"kind": "synthetic", "seed": 19},
                "scalars": {"v1": 8},
                "pointer_fills": {"v0": 3},
                "outputs": ["v0"],
            }
        ),
        encoding="utf-8",
    )

    def fake_compile(unit, output_dir, ptoas_bin, *, timeout):
        del ptoas_bin, timeout
        output_dir.mkdir(parents=True, exist_ok=True)
        pto = output_dir / unit.path.name
        shutil.copy2(unit.path, pto)
        cpp = output_dir / "group.cpp"
        cpp.write_text(
            'extern "C" __global__ AICORE void sample(__gm__ int32_t* v0, int32_t v1) {}\n',
            encoding="utf-8",
        )
        summary = output_dir / "group.sync.jsonl"
        summary.write_text('{"function":"sample","active_sync_groups":0}\n', encoding="utf-8")
        return pto, cpp, summary

    monkeypatch.setattr(preflight, "compile_pto_unit", fake_compile)
    (tmp_path / "fake-ptoas").write_bytes(b"fake")
    output = preflight.prepare(
        builds["baseline"],
        builds["candidate"],
        "sample",
        profile,
        tmp_path / "prepared",
        tmp_path / "fake-ptoas",
    )

    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["post_insert_sync_compiled"] is True
    assert document["npu_cases_built"] is False
    assert document["ptoas"]["path"].endswith("fake-ptoas")
    assert document["recommended_outputs"] == ["v0"]
    assert document["endpoints"]["baseline"]["target_sync_summary"]["active_sync_groups"] == 0
    assert document["endpoints"]["baseline"]["functions"] == ["sample"]
    assert document["endpoints"]["candidate"]["functions"] == ["sample"]
    baseline_case = Path(document["endpoints"]["baseline"]["case_dir"])
    candidate_case = Path(document["endpoints"]["candidate"]["case_dir"])
    assert (baseline_case / "v0.bin").read_bytes() == (candidate_case / "v0.bin").read_bytes()


def test_validate_cases_and_summarize(generator: ModuleType, comparison: ModuleType, tmp_path: Path):
    kernel = _write_kernel(tmp_path)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "v0.bin").write_bytes(bytes(range(32)))
    compact = generator.generate(
        kernel,
        "sample",
        tmp_path / "compact",
        "dav-c220",
        run_mode="npu",
        block_dim=4,
        input_dir=inputs,
        scalar_values={"n": "8"},
    )
    loose = generator.generate(
        kernel,
        "sample",
        tmp_path / "loose",
        "dav-c220",
        run_mode="npu",
        block_dim=4,
        input_dir=inputs,
        scalar_values={"n": "8"},
    )
    manifest, pointers = comparison.validate_cases(compact, loose)
    assert manifest["kernel"] == "sample"
    assert pointers == ["v0"]

    summary = comparison.summarize(
        [10.0, 11.0, 10.5],
        [9.0, 9.5, 10.0],
        [-1.0, -0.5, -0.75, -0.25],
        bootstrap_samples=100,
    )
    assert summary["loose_minus_compact_us"] == pytest.approx(-1.0)
    assert summary["loose_minus_compact_percent"] < 0
    assert summary["paired_bootstrap_95_ci_us"][1] < 0


def test_multi_comparison_balances_orders_and_summarizes(multi_comparison: ModuleType):
    names = ["geometry_ff", "cypress", "dsa_rp_cg"]
    orders = multi_comparison.balanced_orders(names)
    assert len(orders) == 6
    for position in range(3):
        assert sorted(order[position] for order in orders) == sorted(names * 2)

    summary = multi_comparison.summarize_variants(
        {
            "first_fit": [12.0, 12.2, 11.8],
            "cypress": [11.0, 11.2, 10.8],
            "dsa_rp": [9.0, 9.2, 8.8],
        },
        [
            {"first_fit": 12.0, "cypress": 11.0, "dsa_rp": 9.0},
            {"first_fit": 12.2, "cypress": 11.1, "dsa_rp": 9.1},
            {"first_fit": 11.9, "cypress": 10.9, "dsa_rp": 8.9},
        ],
        bootstrap_samples=100,
    )
    assert summary["comparisons"]["cypress_minus_first_fit"]["median_delta_us"] == pytest.approx(-1.0)
    assert summary["comparisons"]["dsa_rp_minus_cypress"]["paired_bootstrap_95_ci_us"][1] < 0


def test_multi_comparison_requires_repeated_deterministic_outputs(
    multi_comparison: ModuleType, tmp_path: Path
):
    dumps: dict[str, list[Path]] = {}
    for arm in ("geometry_ff", "cypress", "dsa_rp_cg"):
        repetitions: list[Path] = []
        for repetition in range(3):
            dump = tmp_path / arm / f"run{repetition}"
            dump.mkdir(parents=True)
            (dump / "v0.bin").write_bytes(b"stable-output")
            repetitions.append(dump)
        dumps[arm] = repetitions

    stable, runs = multi_comparison._compare_all_outputs(dumps, ["v0"], None)
    assert len(set(stable["v0"].values())) == 1
    assert all(len(hashes) == 3 for hashes in runs["v0"].values())

    (dumps["cypress"][2] / "v0.bin").write_bytes(b"changed-output")
    with pytest.raises(ValueError, match="nondeterministic for cypress/v0.bin"):
        multi_comparison._compare_all_outputs(dumps, ["v0"], None)


def test_panel_summary_keeps_devices_separate(panel_summary: ModuleType):
    def report(scale: float) -> dict:
        medians = {
            "geometry_ff": 10.0,
            "cypress": 9.0,
            "dsa_rp_cg": 8.0,
        }
        return {
            "summary": {"variants": {arm: {"median_us": value * scale} for arm, value in medians.items()}}
        }

    reports = {
        ("kernel_a", "device4"): report(1.0),
        ("kernel_b", "device4"): report(2.0),
        ("kernel_a", "device5"): report(3.0),
        ("kernel_b", "device5"): report(4.0),
    }
    kernel_rows, panel_rows = panel_summary.summarize_panel(
        reports,
        bootstrap_samples=100,
        expected_tags={"kernel_a", "kernel_b"},
    )
    assert len(kernel_rows) == 12
    devices = {row["device"] for row in panel_rows}
    assert devices == {"device4", "device5"}
    candidate = next(
        row
        for row in panel_rows
        if row["device"] == "device4"
        and row["reference"] == "geometry_ff"
        and row["candidate"] == "dsa_rp_cg"
    )
    assert candidate["kernels"] == 2
    assert candidate["geomean_ratio"] == pytest.approx(0.8)


def test_panel_summary_rejects_different_kernel_sets(panel_summary: ModuleType):
    def report() -> dict:
        return {
            "summary": {
                "variants": {
                    arm: {"median_us": value}
                    for arm, value in {"geometry_ff": 10.0, "cypress": 9.0, "dsa_rp_cg": 8.0}.items()
                }
            }
        }

    reports = {
        ("kernel_a", "device4"): report(),
        ("kernel_b", "device4"): report(),
        ("kernel_a", "device5"): report(),
    }
    with pytest.raises(ValueError, match="device kernel sets differ"):
        panel_summary.summarize_panel(reports, bootstrap_samples=100)


def test_panel_summary_requires_frozen_panel_tags(panel_summary: ModuleType):
    report = {
        "summary": {
            "variants": {
                arm: {"median_us": value}
                for arm, value in {"geometry_ff": 10.0, "cypress": 9.0, "dsa_rp_cg": 8.0}.items()
            }
        }
    }
    with pytest.raises(ValueError, match="do not match frozen panel"):
        panel_summary.summarize_panel(
            {("kernel_a", "device4"): report},
            bootstrap_samples=100,
            expected_tags={"kernel_a", "kernel_b"},
        )


def test_panel_summary_loads_json_and_tsv_frozen_panels(panel_summary: ModuleType, tmp_path: Path):
    json_panel = tmp_path / "panel.json"
    json_panel.write_text(
        json.dumps({"kernels": [{"tag": "kernel_a"}, {"tag": "kernel_b"}]}),
        encoding="utf-8",
    )
    tsv_panel = tmp_path / "panel.tsv"
    tsv_panel.write_text("tag\tstatus\nkernel_a\tconfirm\nkernel_b\tconfirm\n", encoding="utf-8")

    assert panel_summary.load_expected_tags(json_panel) == {"kernel_a", "kernel_b"}
    assert panel_summary.load_expected_tags(tsv_panel) == {"kernel_a", "kernel_b"}


def test_panel_summary_rejects_mislabeled_report_device(panel_summary: ModuleType, tmp_path: Path):
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "device_id": 5,
                "summary": {
                    "variants": {
                        arm: {"median_us": value}
                        for arm, value in {
                            "geometry_ff": 10.0,
                            "cypress": 9.0,
                            "dsa_rp_cg": 8.0,
                        }.items()
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="contains a report for device 5"):
        panel_summary.load_reports([("kernel_a", "device4", report)])


def test_panel_summary_accepts_matching_device_alias(panel_summary: ModuleType, tmp_path: Path):
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "device_id": 4,
                "summary": {
                    "variants": {
                        arm: {"median_us": value}
                        for arm, value in {
                            "geometry_ff": 10.0,
                            "cypress": 9.0,
                            "dsa_rp_cg": 8.0,
                        }.items()
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    reports = panel_summary.load_reports([("kernel_a", "device4", report)])
    assert reports[("kernel_a", "device4")]["device_id"] == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
