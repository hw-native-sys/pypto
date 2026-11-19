# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Export a model's actual input to the native AutoLayout pass for solver benchmarking."""

import argparse
import importlib
import sys
from pathlib import Path

from pypto import backend, ir, passes
from pypto.pypto_core import ir as core_ir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", type=Path)
    parser.add_argument("entry", help="JIT entry with a no-argument specialization")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.module.resolve().parent))
    backend.set_backend_type(backend.BackendType.Ascend910)
    program = getattr(importlib.import_module(args.module.stem), args.entry).specialize()
    prefix = passes.PassPipeline()
    for step in ir.PassManager.get_strategy(ir.OptimizationStrategy.AutoLayout).passes:
        if step.get_name() == "AutoLayout":
            args.output.write_bytes(core_ir.serialize(prefix.run(program)))
            return
        prefix.add_pass(step)
    raise RuntimeError("AutoLayout missing from the pipeline")


if __name__ == "__main__":
    main()
