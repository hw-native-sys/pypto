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
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator() -> ModuleType:
    return _load_script("gen_profiling_case")


@pytest.fixture(scope="module")
def comparison() -> ModuleType:
    return _load_script("standalone_compare")


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
    pto = "%view = pto.make_tensor_view %arg0, shape = [%c8_index], strides = [%c1_index]\n"
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
        "%view = pto.make_tensor_view %arg0, shape = [%c8_index], strides = [%c1_index]\n",
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
        "%view = pto.make_tensor_view %arg0, shape = [%c8_index], strides = [%c1_index]\n",
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
