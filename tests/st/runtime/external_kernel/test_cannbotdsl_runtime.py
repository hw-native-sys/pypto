# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Direct A5 AIC/AIV/MIX chains: strict compilation, ordered submission and repeated visibility.

Run with --platform=a5; --codegen-only builds both device and orchestration
binaries and constructs callables without starting a Worker. Both modes require
the pinned Simpler package and CANN toolchain: KernelCompiler uses simpler_setup.
Both modes also require CANNBotDSL with public device export; the tests skip otherwise.
This tests the direct bridge under the external-kernel test policy exception,
independently of automatic PyPTO JIT provider discovery.
"""

import pypto.language as pl
import pytest
from pypto import DataType, passes
from pypto.backend.cannbotdsl_runtime import build_submitter, compile_task
from pypto.backend.external_bridge import RuntimeTarget
from pypto.backend.tmr_bridge import TMRBridge
from pypto.jit.external.contracts import CompileTarget, KernelRequest, ParameterSpec, TensorSpec
from pypto.jit.external.providers.cannbotdsl import CANNBotDSLProvider


def _runtime_target(prepared):
    return RuntimeTarget(
        "tmr",
        "1",
        prepared.request.target,
        frozenset({("cannbotdsl.device_callable", 1)}),
        frozenset(
            {
                *(f"core:{core.value}" for core in prepared.execution.cores),
                f"aiv_lanes:{prepared.execution.aiv_lanes}",
                "completion:entry_return_visible",
                "tensor_writeback",
                "partial_write",
            }
        ),
    )


@pytest.mark.platforms("a5", reason="CANNBotDSL direct bridge requires real A5 text-only loading")
@pytest.mark.parametrize("core_type", ["aic", "aiv", "mixed", "chain"])
@pytest.mark.parametrize("dynamic", [False, True])
def test_cannbotdsl_runtime_chain(tmp_path, test_config, dynamic, core_type):
    cb = pytest.importorskip("cannbotdsl")
    if not hasattr(cb, "export_device"):
        pytest.skip("installed CANNBotDSL lacks public device export")

    from pypto.runtime.kernel_compiler import KernelCompiler  # noqa: PLC0415

    @cb.kernel
    def add_one(x, out):
        for i in cb.range(x.shape[0]):
            out[i] = x[i] + 1

    @cb.kernel
    def cube_add_one(x, out):
        for i in cb.range(x.shape[0]):
            cb.scalar.cube_store_bypass(out.ptr(i), cb.scalar.load_bypass(x.ptr(i)) + 1)

    @cb.kernel
    def mixed_add_one(x, out):
        # Each participant owns whole, disjoint 64-byte cache lines.
        for i in cb.range(x.shape[0]):
            if (i // 16) % 3 == 0:
                cb.scalar.cube_store_bypass(out.ptr(i), cb.scalar.load_bypass(x.ptr(i)) + 1)
            if (i // 16) % 3 == cb.get_subblock_id() + 1:
                cb.scalar.vec_store_bypass(out.ptr(i), cb.scalar.load_bypass(x.ptr(i)) + 1)

    tensor = TensorSpec(("n" if dynamic else 96,), DataType.INT32, strides=(1,), alignment=64)
    request = KernelRequest(
        (ParameterSpec("x", tensor), ParameterSpec("out", tensor)),
        CompileTarget("a5", "ccec", "runtime-test"),
    )
    compiler = KernelCompiler("a5")
    tasks, ids, images = [], [], []
    kinds = ("aic", "mixed", "aiv") if core_type == "chain" else (core_type,) * 3
    for kind in kinds:
        prepared = CANNBotDSLProvider().prepare(
            {"aic": cube_add_one, "aiv": add_one, "mixed": mixed_add_one}[kind], request
        )
        assert prepared.execution.aiv_lanes == {"aic": 0, "aiv": 1, "mixed": 2}[kind]
        target = _runtime_target(prepared)
        task = TMRBridge().lower(prepared, target)
        binaries = compile_task(task, compiler, pto_isa_root=str(tmp_path))
        tasks.append(task)
        ids.append(tuple(range(len(images), len(images) + len(binaries))))
        images.extend(binaries)
    submitter = build_submitter(tuple(tasks), tuple(ids))
    for file in submitter.files:
        path = tmp_path / file.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(file.content)
    source = tmp_path / "orchestration.cpp"
    source.write_text(
        f'''#include "{submitter.entry}"
extern "C" __attribute__((visibility("default")))
OrchestrationConfig cann_chain_config(const ChipTaskArgs&) {{
    return OrchestrationConfig{{.expected_arg_count = 4}};
}}
extern "C" __attribute__((visibility("default")))
void cann_chain(const ChipTaskArgs& args) {{
    CoreTaskArgs calls[3];
    for (int i = 0; i < 3; ++i) {{
        calls[i].add_input(args.tensor(i).ref());
        calls[i].add_output(args.tensor(i + 1).ref());
    }}
    {submitter.symbol}({{&calls[0], &calls[1], &calls[2]}});
}}
'''
    )
    orchestration = compiler.compile_orchestration(
        runtime_name="tensormap_and_ringbuffer", source_path=str(source)
    )
    assert all(images) and orchestration
    import torch  # noqa: PLC0415
    from simpler.task_interface import (  # noqa: PLC0415
        ArgDirection,
        CallConfig,
        ChipCallable,
        CoreCallable,
        TaskArgs,
        Tensor,
        TensorArgType,
    )
    from simpler.task_interface import (  # noqa: PLC0415
        DataType as RuntimeDType,
    )
    from simpler.worker import Worker  # noqa: PLC0415

    cores = [
        CoreCallable.build(signature=[ArgDirection.IN, ArgDirection.OUT], binary=image) for image in images
    ]
    chip = ChipCallable.build(
        signature=[ArgDirection.IN, ArgDirection.INOUT, ArgDirection.INOUT, ArgDirection.OUT],
        func_name="cann_chain",
        binary=orchestration,
        children=list(enumerate(cores)),
    )
    if test_config.codegen_only:
        return

    worker = Worker(
        level=2, platform="a5", runtime="tensormap_and_ringbuffer", device_id=test_config.device_id
    )
    buffers = []
    try:
        handle = worker.register(chip)
        worker.init()
        # Different shapes on successive dynamic runs also check fresh bindings.
        for count in (96, 17, 192) if dynamic else (96, 96, 96):
            buffers = [worker.malloc(count * 4) for _ in range(4)]
            args = TaskArgs()
            for i, buffer in enumerate(buffers):
                args.add_tensor(
                    Tensor(buffer, shapes=(count,), dtype=RuntimeDType.INT32),
                    (
                        TensorArgType.INPUT,
                        TensorArgType.INOUT,
                        TensorArgType.INOUT,
                        TensorArgType.OUTPUT_EXISTING,
                    )[i],
                )
            for repeat in range(3):
                x = torch.arange(count, dtype=torch.int32) + repeat
                worker.copy_to(buffers[0], x)
                for buffer in buffers[1:]:
                    worker.copy_to(buffer, torch.full_like(x, -999.0))
                worker.run(handle, args, CallConfig())
                for i, buffer in enumerate(buffers[1:], start=1):
                    actual = torch.empty_like(x)
                    worker.copy_from(actual, buffer)
                    torch.testing.assert_close(actual, x + i, rtol=0, atol=0)
            for buffer in buffers:
                worker.free(buffer)
            buffers = []
    finally:
        for buffer in buffers:
            worker.free(buffer)
        worker.close()


def _compute_kernels(cb):
    """Use actual Cube and SIMD instructions with callback-owned on-chip buffers."""

    @cb.jit
    def matmul_body(a, b, out):
        a_l1 = cb.Buffer(cb.MemLoc.L1, cb.dtypes.float16, cb.make_layout((16, 32)))
        b_l1 = cb.Buffer(cb.MemLoc.L1, cb.dtypes.float16, cb.make_layout((32, 16)))
        a_l0 = cb.Buffer(cb.MemLoc.L0A, cb.dtypes.float16, cb.make_layout((16, 32)))
        b_l0 = cb.Buffer(cb.MemLoc.L0B, cb.dtypes.float16, cb.make_layout((16, 32))).permute(1, 0)
        acc = cb.Buffer(cb.MemLoc.L0C, cb.dtypes.float32, cb.make_layout((16, 16)))
        cb.mem_copy(a_l1, a)
        cb.mem_copy(b_l1, b)
        cb.cube_sync_all()
        cb.mem_copy(a_l0, a_l1)
        cb.mem_copy(b_l0, b_l1)
        cb.cube_sync_all()
        result = cb.matmul(acc, a_l0, b_l0)
        cb.cube_sync_all()
        cb.mem_copy(out, result)

    @cb.jit
    def vector_body(x, y, out, tile):
        x_ub = cb.Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((128,)))
        y_ub = cb.Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((128,)))
        cb.mem_copy(x_ub, cb.tile_slice(x, (128,), (tile,)))
        cb.mem_copy(y_ub, cb.tile_slice(y, (128,), (tile,)))
        cb.vec_sync_all()
        with cb.vf():
            mask = cb.reg.full_mask()
            # A5 registers hold 256 bytes: process both 64-FP32 halves of the tile.
            # Keep buffer bases as VF arguments: unrolling adds per-half pointers
            # and spills a callee-saved scalar register in the MIX wrapper.
            for half in cb.range(2, unroll=1):
                offset = half * 64
                vx = cb.reg.vload(x_ub, offset)
                vy = cb.reg.vload(y_ub, offset)
                summed = cb.reg.vadd(vx, vy, mask=mask)
                result = cb.reg.vmul(summed, vx, mask=mask)
                cb.reg.vstore(x_ub, offset, result, mask)
        cb.vec_sync_all()
        cb.mem_copy(cb.tile_slice(out, (128,), (tile,)), x_ub)
        cb.vec_sync_all()

    @cb.kernel
    def cube_matmul(a, b, out):
        matmul_body(a, b, out)

    @cb.kernel
    def vector_compute(x, y, out):
        # Keep one tile live to fit the callback's zero-stack register budget.
        for tile in cb.range(2, unroll=1):
            vector_body(x, y, out, tile)

    @cb.kernel
    def mixed_compute(a, b, x, y, product, vector):
        matmul_body(a, b, product)
        # Each AIV writes 128 FP32 elements (eight whole 64-byte lines).
        vector_body(x, y, vector, cb.get_subblock_id())

    return cube_matmul, vector_compute, mixed_compute


def _compile_compute(tmp_path, entries, specs, flat, kind):
    from pypto.runtime.kernel_compiler import KernelCompiler  # noqa: PLC0415

    compiler = KernelCompiler("a5")
    tasks, ids, images, bindings, signatures = [], [], [], [], []
    for index, (kernel, names, slots, outputs) in enumerate(entries):
        parameters = tuple(
            ParameterSpec(name, flat if index else specs[slot]) for name, slot in zip(names, slots)
        )
        request = KernelRequest(parameters, CompileTarget("a5", "ccec", "runtime-test"))
        prepared = CANNBotDSLProvider().prepare(kernel, request)
        target = _runtime_target(prepared)
        expected_kind = "aiv" if index else kind
        assert prepared.execution.aiv_lanes == {"aic": 0, "aiv": 1, "mixed": 2}[expected_kind]
        exported_source = next(
            file.content.decode() for file in prepared.artifact.files if file.path == "device_callable.hpp"
        )
        if expected_kind != "aiv":
            assert "asc_mmad(" in exported_source
        if expected_kind != "aic":
            assert "asc_add(" in exported_source and "asc_mul(" in exported_source
        # Retain the real export for ISA/resource inspection in compile-only CI.
        for artifact in prepared.artifact.files:
            path = tmp_path / f"export_{index}" / artifact.path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(artifact.content)
        task = TMRBridge().lower(prepared, target)
        binaries = compile_task(task, compiler, pto_isa_root=str(tmp_path))
        tasks.append(task)
        ids.append(tuple(range(len(images), len(images) + len(binaries))))
        images.extend(binaries)
        signatures.extend([(slots, outputs)] * len(binaries))
        bindings.append((slots, outputs))
    submitter = build_submitter(tuple(tasks), tuple(ids))
    for file in submitter.files:
        path = tmp_path / file.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(file.content)
    calls = []
    for index, (slots, outputs) in enumerate(bindings):
        for slot in slots:
            direction = "output" if slot in outputs else "input"
            tensor = f"args.tensor({slot}).ref()"
            if index and slot == 4:
                tensor = "flat_product"
            calls.append(f"    calls[{index}].add_{direction}({tensor});")
    call_args = ", ".join(f"&calls[{i}]" for i in range(len(tasks)))
    source = tmp_path / "compute.cpp"
    source.write_text(
        f'''#include "{submitter.entry}"
extern "C" __attribute__((visibility("default")))
OrchestrationConfig cann_compute_config(const ChipTaskArgs&) {{
    return OrchestrationConfig{{.expected_arg_count = 7}};
}}
extern "C" __attribute__((visibility("default")))
void cann_compute(const ChipTaskArgs& args) {{
    uint32_t flat_shape[] = {{256}};
    auto flat_product = args.tensor(4).ref().reshape(flat_shape, 1);
    CoreTaskArgs calls[{len(tasks)}];
{chr(10).join(calls)}
    {submitter.symbol}({{{call_args}}});
}}
'''
    )
    orchestration = compiler.compile_orchestration(
        runtime_name="tensormap_and_ringbuffer", source_path=str(source)
    )
    return images, signatures, orchestration


@pytest.mark.platforms("a5", reason="CANNBotDSL direct bridge requires A5 text-only loading")
@pytest.mark.parametrize("kind", ["aic", "aiv", "mixed"])
def test_cannbotdsl_compute(tmp_path, test_config, kind):
    """Check Cube matmul, SIMD arithmetic, and a consumer of both MIX outputs."""
    cb = pytest.importorskip("cannbotdsl")
    if not hasattr(cb, "export_device"):
        pytest.skip("installed CANNBotDSL lacks public device export")

    cube, vector, mixed = _compute_kernels(cb)
    matrix_a = TensorSpec((16, 32), DataType.FP16, strides=(32, 1), alignment=64)
    matrix_b = TensorSpec((32, 16), DataType.FP16, strides=(16, 1), alignment=64)
    matrix_out = TensorSpec((16, 16), DataType.FP32, strides=(16, 1), alignment=64)
    flat = TensorSpec((256,), DataType.FP32, strides=(1,), alignment=64)
    specs = (matrix_a, matrix_b, flat, flat, matrix_out, flat, flat)
    # The consumer reads both MIX outputs, so an early group completion is visible.
    entries = {
        "aic": [(cube, ("a", "b", "out"), (0, 1, 4), {4})],
        "aiv": [(vector, ("x", "y", "out"), (2, 3, 5), {5})],
        "mixed": [
            (mixed, ("a", "b", "x", "y", "product", "vector"), (0, 1, 2, 3, 4, 5), {4, 5}),
            (vector, ("x", "y", "out"), (4, 5, 6), {6}),
        ],
    }[kind]
    images, signatures, orchestration = _compile_compute(tmp_path, entries, specs, flat, kind)
    import torch  # noqa: PLC0415
    from simpler.task_interface import (  # noqa: PLC0415
        ArgDirection,
        CallConfig,
        ChipCallable,
        CoreCallable,
        TaskArgs,
        Tensor,
        TensorArgType,
    )
    from simpler.task_interface import DataType as RuntimeDType  # noqa: PLC0415
    from simpler.worker import Worker  # noqa: PLC0415

    cores = []
    for image, (slots, outputs) in zip(images, signatures):
        cores.append(
            CoreCallable.build(
                signature=[ArgDirection.OUT if slot in outputs else ArgDirection.IN for slot in slots],
                binary=image,
            )
        )
    chip = ChipCallable.build(
        signature=[ArgDirection.IN] * 4 + [ArgDirection.INOUT] * 2 + [ArgDirection.OUT],
        func_name="cann_compute",
        binary=orchestration,
        children=list(enumerate(cores)),
    )
    if test_config.codegen_only:
        return
    worker = Worker(
        level=2, platform="a5", runtime="tensormap_and_ringbuffer", device_id=test_config.device_id
    )
    buffers = []
    try:
        handle = worker.register(chip)
        worker.init()
        for spec in specs:
            count = 1
            for extent in spec.shape:
                count *= extent
            buffers.append(worker.malloc(count * (2 if spec.dtype == DataType.FP16 else 4)))
        args = TaskArgs()
        for slot, (buffer, spec) in enumerate(zip(buffers, specs)):
            args.add_tensor(
                Tensor(
                    buffer,
                    shapes=spec.shape,
                    dtype=RuntimeDType.FLOAT16 if spec.dtype == DataType.FP16 else RuntimeDType.FLOAT32,
                ),
                TensorArgType.INPUT
                if slot < 4
                else TensorArgType.OUTPUT_EXISTING
                if slot == 6
                else TensorArgType.INOUT,
            )
        for repeat in range(3):
            a = ((torch.arange(512).reshape(16, 32) % 7 - 3 + repeat) / 4).half()
            b = ((torch.arange(512).reshape(32, 16) % 5 - 2 - repeat) / 4).half()
            x = (torch.arange(256) % 11 - 5 + repeat).float()
            y = (torch.arange(256) % 13 - 6 - repeat).float()
            for buffer, value in zip(buffers[:4], (a, b, x, y)):
                worker.copy_to(buffer, value)
            for buffer in buffers[4:]:
                worker.copy_to(buffer, torch.full((256,), float("nan")))
            worker.run(handle, args, CallConfig())
            product = (a.float() @ b.float()).flatten()
            vector_result = (x + y) * x
            expected = {
                "aic": {4: product},
                "aiv": {5: vector_result},
                "mixed": {4: product, 5: vector_result, 6: (product + vector_result) * product},
            }[kind]
            for slot, golden in expected.items():
                actual = torch.empty_like(golden)
                worker.copy_from(actual, buffers[slot])
                torch.testing.assert_close(actual, golden, rtol=0, atol=0)
    finally:
        for buffer in buffers:
            worker.free(buffer)
        worker.close()


@pl.jit.incore
def _jit_prepare(x: pl.Tensor, out: pl.Out[pl.Tensor]):
    tile = pl.load(x, [0, 0], [x.shape[0], x.shape[1]])
    value = pl.tile.adds(tile, 1.0)
    return pl.store(value, [0, 0], out)


@pl.jit.incore
def _jit_consume(x: pl.Tensor, y: pl.Tensor, out: pl.Out[pl.Tensor]):
    lhs = pl.load(x, [0, 0], [16, 16])
    rhs = pl.load(y, [0, 0], [16, 16])
    value = pl.tile.mul(pl.tile.add(lhs, rhs), lhs)
    return pl.store(value, [0, 0], out)


def _jit_compute_chain(cb, kind):
    native_cube, native_vector, native_mixed = _compute_kernels(cb)

    @pl.jit
    def cube_chain(a: pl.Tensor, b: pl.Tensor, x: pl.Tensor, y: pl.Tensor, out: pl.InOut[pl.Tensor]):
        a1 = pl.create_tensor([16, 32], dtype=pl.FP16)
        b1 = pl.create_tensor([32, 16], dtype=pl.FP16)
        a1 = _jit_prepare(a, a1)
        b1 = _jit_prepare(b, b1)
        product = pl.create_tensor([16, 16], dtype=pl.FP32)
        native_cube(a1, b1, product)
        out = _jit_consume(product, product, out)
        return out

    @pl.jit
    def vector_chain(a: pl.Tensor, b: pl.Tensor, x: pl.Tensor, y: pl.Tensor, out: pl.InOut[pl.Tensor]):
        x1 = pl.create_tensor([16, 16], dtype=pl.FP32)
        y1 = pl.create_tensor([16, 16], dtype=pl.FP32)
        x1 = _jit_prepare(x, x1)
        y1 = _jit_prepare(y, y1)
        vector = pl.create_tensor([256], dtype=pl.FP32)
        native_vector(pl.reshape(x1, [256]), pl.reshape(y1, [256]), vector)
        value = pl.reshape(vector, [16, 16])
        out = _jit_consume(value, value, out)
        return out

    @pl.jit
    def mixed_chain(a: pl.Tensor, b: pl.Tensor, x: pl.Tensor, y: pl.Tensor, out: pl.InOut[pl.Tensor]):
        a1 = pl.create_tensor([16, 32], dtype=pl.FP16)
        b1 = pl.create_tensor([32, 16], dtype=pl.FP16)
        x1 = pl.create_tensor([16, 16], dtype=pl.FP32)
        y1 = pl.create_tensor([16, 16], dtype=pl.FP32)
        a1 = _jit_prepare(a, a1)
        b1 = _jit_prepare(b, b1)
        x1 = _jit_prepare(x, x1)
        y1 = _jit_prepare(y, y1)
        product = pl.create_tensor([16, 16], dtype=pl.FP32)
        vector = pl.create_tensor([256], dtype=pl.FP32)
        native_mixed(a1, b1, pl.reshape(x1, [256]), pl.reshape(y1, [256]), product, vector)
        out = _jit_consume(product, pl.reshape(vector, [16, 16]), out)
        return out

    return {"aic": cube_chain, "aiv": vector_chain, "mixed": mixed_chain}[kind]


@pytest.mark.platforms("a5", reason="CANNBotDSL JIT requires A5 text-only callbacks")
@pytest.mark.parametrize("kind", ["aic", "aiv", "mixed"])
@pytest.mark.parametrize(
    "runtime", [passes.RuntimeKind.TENSORMAP_AND_RINGBUFFER, passes.RuntimeKind.HOST_BUILD_GRAPH]
)
def test_cannbotdsl_jit_compute_chain(tmp_path, test_config, kind, runtime):
    """Ordinary JIT producers and consumers surround real native Cube/SIMD work."""
    from dataclasses import replace  # noqa: PLC0415

    import torch  # noqa: PLC0415

    cb = pytest.importorskip("cannbotdsl")
    if not hasattr(cb, "export_device"):
        pytest.skip("installed CANNBotDSL lacks public device export")
    chain = _jit_compute_chain(cb, kind)
    config = replace(test_config, save_kernels=True, save_kernels_dir=str(tmp_path))
    a = torch.zeros((16, 32), dtype=torch.float16)
    b = torch.zeros((32, 16), dtype=torch.float16)
    x = torch.zeros((16, 16), dtype=torch.float32)
    y = torch.zeros_like(x)
    out = torch.empty_like(x)
    with passes.PassContext([], runtime=runtime):
        compiled = chain.compile(a, b, x, y, out, config=config)
    del chain
    compiled.load()
    # Loading after compile() returns also verifies exported source ownership.
    if config.codegen_only:
        return
    for repeat in range(4):
        a.copy_(((torch.arange(512).reshape(16, 32) % 7 - 3 + repeat) / 4).half())
        b.copy_(((torch.arange(512).reshape(32, 16) % 5 - 2 - repeat) / 4).half())
        x.copy_((torch.arange(256).reshape(16, 16) % 11 - 5 + repeat).float())
        y.copy_((torch.arange(256).reshape(16, 16) % 13 - 6 - repeat).float())
        out.fill_(float("nan"))
        compiled(a, b, x, y, out, config=config)
        product = (a.float() + 1) @ (b.float() + 1)
        vector = (x + y + 2) * (x + 1)
        lhs, rhs = {"aic": (product, product), "aiv": (vector, vector), "mixed": (product, vector)}[kind]
        torch.testing.assert_close(out, (lhs + rhs) * lhs, rtol=0, atol=0)


@pytest.mark.platforms("a5", reason="CANNBotDSL Graph replay requires the A5 percounter runtime")
def test_cannbotdsl_graph_rebinds_native_operands(tmp_path, test_config):
    from dataclasses import replace  # noqa: PLC0415

    import torch  # noqa: PLC0415

    cb = pytest.importorskip("cannbotdsl")
    if not hasattr(cb, "export_device"):
        pytest.skip("installed CANNBotDSL lacks public device export")
    _, native, _ = _compute_kernels(cb)

    @pl.jit.graph
    def stage(x: pl.Tensor, y: pl.Tensor, out: pl.InOut[pl.Tensor]):
        native(x, y, out)
        return out

    @pl.jit
    def entry(x: pl.Tensor, y: pl.Tensor, first: pl.InOut[pl.Tensor], second: pl.InOut[pl.Tensor]):
        first = stage(x, y, first)
        second = stage(y, x, second)
        return first, second

    x = torch.arange(256, dtype=torch.float32)
    y = x + 1
    first = torch.full_like(x, float("nan"))
    second = torch.full_like(y, float("nan"))
    config = replace(test_config, save_kernels=True, save_kernels_dir=str(tmp_path))
    with passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH):
        compiled = entry.compile(x, y, first, second, config=config)
    compiled.load()
    if config.codegen_only:
        return
    compiled(x, y, first, second, config=config)
    torch.testing.assert_close(first, (x + y) * x, rtol=0, atol=0)
    torch.testing.assert_close(second, (x + y) * y, rtol=0, atol=0)


@pytest.mark.platforms("a5", reason="CANNBotDSL BF16 view replay requires the A5 percounter runtime")
def test_cannbotdsl_graph_checks_derived_bf16_view(tmp_path, test_config):
    from dataclasses import replace  # noqa: PLC0415

    import torch  # noqa: PLC0415

    cb = pytest.importorskip("cannbotdsl")
    if not hasattr(cb, "export_device"):
        pytest.skip("installed CANNBotDSL lacks public device export")

    @cb.kernel
    def copy_tile(x, out):
        tile = cb.Buffer(cb.MemLoc.UB, cb.dtypes.bfloat16, cb.make_layout((64,)))
        cb.mem_copy(tile, x)
        cb.vec_sync_all()
        cb.mem_copy(out, tile)
        cb.vec_sync_all()

    @pl.jit.graph
    def stage(x: pl.Tensor, out: pl.InOut[pl.Tensor], offset: pl.Scalar[pl.INDEX]):
        view = pl.tensor.slice(x, [64], [offset])
        copy_tile(view, out)
        return out

    @pl.jit
    def entry(x: pl.Tensor, out: pl.InOut[pl.Tensor], step: pl.Scalar[pl.INDEX]):
        for i in pl.range(2):
            offset = i * step
            out = stage(x, out, offset)
        return out

    x = torch.arange(128, dtype=torch.float32).bfloat16()
    out = torch.empty(64, dtype=torch.bfloat16)
    config = replace(test_config, save_kernels=True, save_kernels_dir=str(tmp_path))
    with passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH):
        compiled = entry.compile(x, out, 32, config=config)
    compiled.load()
    if config.codegen_only:
        return
    compiled(x, out, 32, config=config)
    torch.testing.assert_close(out, x[32:96], rtol=0, atol=0)
    # First invocation is aligned. The same graph's next view starts two bytes later.
    with pytest.raises(RuntimeError, match="alignment"):
        compiled(x, out, 1, config=config)


@pytest.mark.platforms("a5", reason="CANNBotDSL JIT requires A5 text-only callbacks")
@pytest.mark.parametrize("postponed", [False, True])
def test_cannbotdsl_jit_constexpr_layout(tmp_path, test_config, postponed, monkeypatch):
    from dataclasses import replace  # noqa: PLC0415

    cb = pytest.importorskip("cannbotdsl")
    if not hasattr(cb, "export_device"):
        pytest.skip("installed CANNBotDSL lacks public device export")

    def copy_tile(x, out, width):
        tile = cb.Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((width,)))
        cb.mem_copy(tile, x)
        cb.vec_sync_all()
        cb.mem_copy(out, tile)

    # CANNBotDSL resolves postponed annotations in the kernel module namespace.
    monkeypatch.setitem(globals(), "cb", cb)
    copy_tile.__annotations__["width"] = "cb.Constexpr[int]" if postponed else cb.Constexpr[int]
    native = cb.kernel(copy_tile)

    @pl.jit
    def entry(x: pl.Tensor[[64], pl.FP32], out: pl.Out[pl.Tensor[[64], pl.FP32]]):
        native(x, out, 64)
        return out

    compiled = entry.compile(config=replace(test_config, save_kernels=True, save_kernels_dir=str(tmp_path)))
    compiled.load()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
