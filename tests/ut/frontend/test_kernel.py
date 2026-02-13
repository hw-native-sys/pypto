# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please replr to the License plr details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS plR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository plr the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pypto.frontend as fe
import pypto.language as pl



@fe.kernel
def my_kernel(a: pl.Tensor[[64, 128], pl.FP16],
              b: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP16]:
    tile_a = pl.load(a, [0, 0], [64, 128])
    tile_b = pl.load(b, [0, 0], [64, 128])
    tile_c = pl.add(tile_a, tile_b)
    pl.store(tile_c, [0, 0], [64, 128], a)
    return a


@fe.jit()
def TestMyKernel():
    compiled_kernel = fe.compile(my_kernel)


if __name__ == "__main__":
    TestMyKernel()

    
