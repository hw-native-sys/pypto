# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Generate A5 PTO for out = (a + b) * b with FP32 registers.

Each tile contains 1024 elements. Inside reg_scope, a hardware loop of 16 chunks loads
64 lanes, add and multiply in registers, and store back to vector memory.

Run in an activated PyPTO/PTOAS environment:
    python examples/advanced/05_register_add_mul.py --save-kernels-dir build/reg-add-mul

Use --save-kernels-dir to retain generated sources and pass dumps for inspection.
This example demonstrates hardware-loop PTO codegen. The unchanged runtime
synchronization admission rejects loop CFGs; this example only generates PTO.
"""

import argparse

import pypto.language as pl
from pypto import CacheConfig
from pypto.backend import BackendType
from pypto.runtime import ExecutionMode, RunConfig


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
    # Generate caller/helper PTO; runtime admission does not support loop CFGs.
    add_mul.compile(config=config)
    print("OK: generated reg_scope hardware-loop add + mul PTO (codegen only)")


if __name__ == "__main__":
    main()
