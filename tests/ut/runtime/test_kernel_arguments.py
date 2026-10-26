# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Kernel argument packing stays independent of the program tensor adapter."""

import ctypes

import pytest
from pypto.ir.param_info import ParamInfo
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import ParamDirection
from pypto.runtime.kernel.arguments import BorrowedArguments, SourcePool
from pypto.torch.interop import CallSignature


def _frame(tensors, scalars=()):
    params = [
        ParamInfo(f"x{i}", ParamDirection.In, tuple(tensor.shape), DataType.FP32)
        for i, tensor in enumerate(tensors)
    ] + [ParamInfo(f"s{i}", ParamDirection.In, None, dtype) for i, (dtype, _) in enumerate(scalars)]
    return CallSignature(params, platform="a2a3").describe_call([*tensors, *(value for _, value in scalars)])


def test_kernel_packer_never_enters_program_helpers(eager_env, monkeypatch):
    from pypto.runtime import runner, tensor_arg  # noqa: PLC0415

    env = eager_env
    env.init()
    session = env.state.bound_worker(env.state.require_config())
    first, second = env.tensor(), env.tensor()
    frame = _frame(
        [first, second],
        [(DataType.INT32, -7), (DataType.UINT64, 2**63 + 1), (DataType.FP32, 1.25), (DataType.BOOL, True)],
    )
    monkeypatch.setattr(tensor_arg, "make_tensor_arg", lambda *a, **kw: pytest.fail("program tensor helper"))
    monkeypatch.setattr(runner, "_coerced_to_orch_args", lambda *a, **kw: pytest.fail("program packer"))
    arguments = BorrowedArguments(frame, session.sources, session.device_location)
    args = arguments.task_args
    assert [type(v) for v in args.scalars] == [ctypes.c_int32, ctypes.c_uint64, ctypes.c_float, ctypes.c_bool]
    assert [v.value for v in args.scalars] == [-7, 2**63 + 1, 1.25, True]
    assert [view.buffer.address for view, _ in args.tensors] == [
        first.untyped_storage().data_ptr(),
        second.untyped_storage().data_ptr(),
    ]
    assert args.transfers == ["NONE", "NONE"]
    arguments.release()
    env.state.close()


def test_external_storage_leases_span_multiple_calls(eager_env):
    env = eager_env
    env.init()
    pool, location = SourcePool(), env.workers[0].device_location
    base = env.tensor((32, 16))
    view = base[16:]
    first = BorrowedArguments(_frame([base, view]), pool, location)
    second = BorrowedArguments(_frame([view]), pool, location)
    args = first.task_args
    assert len(env.buffers) == 1
    assert args.tensors[0][0].buffer is args.tensors[1][0].buffer
    assert args.tensors[1][0].byte_offset == 16 * 16 * 4
    first.release()
    assert not env.buffers[0].closed
    second.release()
    assert env.buffers[0].closed
    env.state.close()


@pytest.mark.parametrize("close_fails", [False, True])
def test_partial_argument_construction_rolls_back_or_retains_storage(eager_env, monkeypatch, close_fails):
    env = eager_env
    env.init()
    pool = SourcePool()
    tensor = env.tensor()

    def fail_scalar(self, value):
        if close_fails:
            env.buffers[0].close_error = RuntimeError("source close failed")
        raise ValueError("scalar packing failed")

    monkeypatch.setattr(env.api.TaskArgs, "add_scalar", fail_scalar)
    with pytest.raises((ValueError, RuntimeError), match="scalar packing failed|source close failed"):
        BorrowedArguments(_frame([tensor], [(DataType.INT32, 1)]), pool, env.workers[0].device_location)
    assert not env.runs and not env.queue
    assert env.buffers[0].closed is not close_fails
    if close_fails:
        storage = tensor.untyped_storage()
        assert pool._sources[(storage.data_ptr(), storage.nbytes())].storage is storage
        env.buffers[0].close_error = None
        pool.close()
    assert not pool._sources
    env.state.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
