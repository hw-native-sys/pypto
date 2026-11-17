# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Direct A5 AIV chain: strict compilation, ordered submission and repeated visibility.

Run with --platform=a5; --codegen-only builds both device and orchestration
binaries without starting a Worker. This exercises the direct bridge, which is
independent of automatic PyPTO JIT provider discovery.
"""

import pytest
from pypto import DataType
from pypto.backend.cannbotdsl_aiv import build_aiv_submitter, compile_aiv_task
from pypto.backend.external_bridge import RuntimeTarget
from pypto.backend.tmr_bridge import TMRBridge
from pypto.jit.external.contracts import CompileTarget, KernelRequest, ParameterSpec, TensorSpec
from pypto.jit.external.providers.cannbotdsl import CANNBotDSLProvider


@pytest.mark.platforms("a5", reason="CANNBotDSL direct bridge requires real A5 text-only loading")
@pytest.mark.parametrize("dynamic", [False, True])
def test_cannbotdsl_aiv_chain(tmp_path, test_config, dynamic):
    cb = pytest.importorskip("cannbotdsl")
    if not hasattr(cb, "export_device"):
        pytest.skip("installed CANNBotDSL lacks public device export")

    from pypto.runtime.kernel_compiler import KernelCompiler  # noqa: PLC0415

    @cb.kernel
    def add_one(x, out):
        for i in cb.range(x.shape[0]):
            out[i] = x[i] + 1.0

    tensor = TensorSpec(("n" if dynamic else 32,), DataType.FP32, strides=(1,))
    request = KernelRequest(
        (ParameterSpec("x", tensor), ParameterSpec("out", tensor)),
        CompileTarget("a5", "ccec", "runtime-test"),
    )
    prepared = CANNBotDSLProvider().prepare(add_one, request)
    target = RuntimeTarget(
        "tmr",
        "1",
        request.target,
        frozenset({("cannbotdsl.device_callable", 1)}),
        frozenset(
            {
                "core:aiv",
                "aiv_lanes:1",
                "completion:entry_return_visible",
                "tensor_writeback",
                "partial_write",
            }
        ),
    )
    task = TMRBridge().lower(prepared, target)
    compiler = KernelCompiler("a5")
    kernel = compile_aiv_task(task, compiler, pto_isa_root=str(tmp_path))
    submitter = build_aiv_submitter((task,) * 3, (0, 1, 2))
    for file in submitter.files:
        path = tmp_path / file.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(file.content)
    source = tmp_path / "orchestration.cpp"
    source.write_text(
        f'''#include "{submitter.entry}"
extern "C" __attribute__((visibility("default")))
OrchestrationConfig cann_aiv_chain_config(const ChipTaskArgs&) {{
    return OrchestrationConfig{{.expected_arg_count = 4}};
}}
extern "C" __attribute__((visibility("default")))
void cann_aiv_chain(const ChipTaskArgs& args) {{
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
    assert kernel and orchestration
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

    core = CoreCallable.build(signature=[ArgDirection.IN, ArgDirection.OUT], binary=kernel)
    chip = ChipCallable.build(
        signature=[ArgDirection.IN, ArgDirection.INOUT, ArgDirection.INOUT, ArgDirection.OUT],
        func_name="cann_aiv_chain",
        binary=orchestration,
        children=[(i, core) for i in range(3)],
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
        for count in (32, 17, 64) if dynamic else (32, 32, 32):
            buffers = [worker.malloc(count * 4) for _ in range(4)]
            args = TaskArgs()
            for i, buffer in enumerate(buffers):
                args.add_tensor(
                    Tensor(buffer, shapes=(count,), dtype=RuntimeDType.FLOAT32),
                    (
                        TensorArgType.INPUT,
                        TensorArgType.INOUT,
                        TensorArgType.INOUT,
                        TensorArgType.OUTPUT_EXISTING,
                    )[i],
                )
            for repeat in range(3):
                x = torch.arange(count, dtype=torch.float32) + repeat
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
