# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""ST test for ``pl.at(name_hint=...)`` on a CORE_GROUP scope.

Verifies that a ``pl.at`` region annotated with
``level=pl.Level.CORE_GROUP, name_hint="GetKVCache"`` compiles and that the
supplied ``name_hint`` propagates into the generated ``kernel_config.py``
artifact (see issue #1113).
"""

import os
import sys
import tempfile

import pypto.language as pl
import pytest
from pypto import ir
from pypto.backend import BackendType

M = 32
K = 64
N = 32

_NAME_HINT = "GetKVCache"


@pl.program
class AtNameHintProgram:
    """CORE_GROUP scope annotated with ``name_hint="GetKVCache"``."""

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        a: pl.Tensor[[M, K], pl.FP32],
        b: pl.Tensor[[K, N], pl.FP32],
        output: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        # NOTE: pl.at name_hint must be a string literal (parser requirement),
        # so it cannot reference the _NAME_HINT module constant.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="GetKVCache"):
            out = pl.matmul(a, b)
            output = pl.assemble(output, out, [0, 0])
        return output


class TestAtNameHint:
    """Regression test for issue #1113 — ``name_hint`` on CORE_GROUP scopes."""

    def test_name_hint_propagates_to_kernel_config(self):
        """``name_hint`` appears in the generated ``kernel_config.py``."""
        with tempfile.TemporaryDirectory(prefix="pypto_at_name_hint_") as work_dir:
            ir.compile(
                AtNameHintProgram,
                output_dir=work_dir,
                backend_type=BackendType.Ascend910B,
                dump_passes=False,
            )

            config_path = os.path.join(work_dir, "kernel_config.py")
            assert os.path.exists(config_path), f"kernel_config.py not generated at {config_path}"

            content = open(config_path).read()
            # Guard against substring overlap with other names — anchor on the
            # full `"name": "GetKVCache"` key/value or `GetKVCache.cpp` filename.
            assert f'"name": "{_NAME_HINT}"' in content or f"{_NAME_HINT}.cpp" in content, (
                f"name_hint {_NAME_HINT!r} not found in kernel_config.py:\n{content}"
            )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))
