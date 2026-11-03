# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compute out = (a + b) * b with FP32 registers on an A5 device.

Each tile contains 1024 elements. Inside reg_scope, a hardware loop of 16 chunks loads
64 lanes, add and multiply in registers, and store back to vector memory.
The host checks three input patterns against PyTorch using one persistent worker.

Run in an activated PyPTO/CANN environment through the local device queue:
    task-submit --device auto 'python examples/advanced/05_register_add_mul.py --device-id "$TASK_DEVICE"'

Use --save-kernels-dir to retain generated sources and pass dumps for inspection.
Requires a backend that can prove hardware-loop emission; unsupported backends
reject compilation instead of silently using a software loop.
"""

import argparse

import pypto.language as pl
import torch
from pypto import CacheConfig
from pypto.backend import BackendType
from pypto.runtime import ChipWorker, ExecutionMode, RunConfig


@pl.jit.incore
def register_add_mul(
    lhs: pl.Tensor[[1, 1024], pl.FP32],
    rhs: pl.Tensor[[1, 1024], pl.FP32],
    out: pl.InOut[pl.Tensor[[1, 1024], pl.FP32]],
) -> pl.Tensor[[1, 1024], pl.FP32]:
    a = pl.load(lhs, [0, 0], [1, 1024], target_memory=pl.MemorySpace.Vec)
    b = pl.load(rhs, [0, 0], [1, 1024], target_memory=pl.MemorySpace.Vec)
    c = pl.load(out, [0, 0], [1, 1024], target_memory=pl.MemorySpace.Vec)
    a = pl.reshape(a, [1024])
    b = pl.reshape(b, [1024])
    c = pl.reshape(c, [1024])
    with pl.reg_scope():
        for chunk in pl.hardware_loop(16):
            x = pl.reg.load(a, offset=chunk * 64, lanes=64)
            y = pl.reg.load(b, offset=chunk * 64, lanes=64)
            summed = pl.reg.add(x, y)
            product = pl.reg.mul(summed, y)
            mask = pl.reg.full_mask(lanes=64)
            _stored = pl.reg.store(c, offset=chunk * 64, value=product, mask=mask)
    c = pl.reshape(c, [1, 1024])
    pl.store(c, [0, 0], out)
    return out


@pl.jit
def add_mul(
    lhs: pl.Tensor[[1, 1024], pl.FP32],
    rhs: pl.Tensor[[1, 1024], pl.FP32],
    out: pl.InOut[pl.Tensor[[1, 1024], pl.FP32]],
) -> pl.Tensor[[1, 1024], pl.FP32]:
    out = register_add_mul(lhs, rhs, out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--save-kernels-dir", help="Directory for generated kernel sources")
    args = parser.parse_args()
    config = RunConfig(
        arch=BackendType.Ascend950,
        execution_mode=ExecutionMode.ONBOARD,
        device_id=args.device_id,
        cache_config=CacheConfig(enabled=False),
        save_kernels_dir=args.save_kernels_dir,
        dump_passes=bool(args.save_kernels_dir),
    )
    compiled = add_mul.compile(config=config)
    # Build the real device binary before opening the persistent worker.
    compiled.load()
    generator = torch.Generator().manual_seed(0)
    with ChipWorker(config=config, runtime=compiled.runtime_name) as worker:
        run = worker.register(compiled)
        for case in range(3):
            a = torch.randn((1, 1024), generator=generator)
            b = torch.randn((1, 1024), generator=generator)
            if case == 1:
                b.zero_()
            elif case == 2:
                a = -b
            out = torch.full_like(a, -777.0)
            run(a, b, out)
            torch.testing.assert_close(out, (a + b) * b, rtol=1e-5, atol=1e-5)
            print(f"PASS case={case}: (a + b) * b, 1024 FP32 elements", flush=True)
    print(f"OK: reg_scope add + mul on A5 device {args.device_id}")


if __name__ == "__main__":
    main()
