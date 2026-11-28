# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

"""Prepare KERNEL artifacts and rank-local Qwen3 weights before allocating NPUs."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import cast

import torch
from pypto import passes

from examples.distributed.gloo_shmem_utils import compile_eager_kernel
from examples.models.qwen3_tp import kernels as k
from examples.models.qwen3_tp.weights import WeightLoader


def prepare_artifacts(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    dtypes = {"BF16": torch.bfloat16, "FP32": torch.float32, "INT32": torch.int32}
    context = passes.PassContext.current()
    group_eager_layers = (
        k.TP_WORLD == 2
        and context is not None
        and context.get_runtime() == passes.RuntimeKind.HOST_BUILD_GRAPH
    )
    for node in ast.parse(Path(k.__file__).read_text()).body:
        if not isinstance(node, ast.FunctionDef) or not node.name.endswith("_chip"):
            continue
        if node.name in k.EAGER_LAYER_ENTRIES and not group_eager_layers:
            continue
        values = []
        for arg in node.args.args:
            annotation = cast(ast.Subscript, arg.annotation)
            if cast(ast.Attribute, annotation.value).attr in ("InOut", "Out"):
                annotation = cast(ast.Subscript, annotation.slice)
            if cast(ast.Attribute, annotation.value).attr == "Scalar":
                values.append(1)
            else:
                shape_node, dtype_node = cast(ast.Tuple, annotation.slice).elts
                shape = eval(compile(ast.Expression(shape_node), k.__file__, "eval"), dict(vars(k)))
                values.append(
                    torch.empty(shape, dtype=dtypes[cast(ast.Attribute, dtype_node).attr], device="meta")
                )
        compile_eager_kernel(getattr(k, node.name), *values, output_dir=str(directory / node.name))
        entries.append(node.name)
        print(f"Prepared {node.name}", flush=True)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "world_size": k.TP_WORLD,
                "source_sha256": hashlib.sha256(Path(k.__file__).read_bytes()).hexdigest(),
                "entries": entries,
            },
            indent=2,
        )
        + "\n"
    )


def prepare_weights(model: str, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    inv = 1.0 / (1e6 ** (torch.arange(0, k.HEAD_DIM, 2).float() / k.HEAD_DIM))
    angles = torch.arange(k.MAX_SEQ).float()[:, None] * inv[None, :]
    angles = torch.cat((angles, angles), dim=-1)
    for rank in range(k.TP_WORLD):
        loader = WeightLoader(model, rank, "cpu", k.TP_WORLD)
        weights = {
            "world_size": k.TP_WORLD,
            "embed": loader.vocabulary("model.embed_tokens.weight"),
            "head": loader.vocabulary("lm_head.weight"),
            "final_norm": loader.norm("model.norm.weight"),
            "layers": [loader.layer(i) for i in range(k.NUM_LAYERS)],
            "cos": angles.cos(),
            "sin": angles.sin(),
        }
        torch.save(weights, directory / f"rank{rank}.pt")
        del weights
        print(f"Prepared weights for rank {rank}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--model")
    parser.add_argument(
        "--runtime",
        choices=("tensormap_and_ringbuffer", "host_build_graph"),
        default="host_build_graph",
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.artifacts:
        runtime = {
            "tensormap_and_ringbuffer": passes.RuntimeKind.TENSORMAP_AND_RINGBUFFER,
            "host_build_graph": passes.RuntimeKind.HOST_BUILD_GRAPH,
        }[args.runtime]
        with passes.PassContext([], runtime=runtime):
            prepare_artifacts(args.artifacts)
    if args.weights:
        if not args.model:
            parser.error("--weights requires --model")
        prepare_weights(args.model, args.weights)
