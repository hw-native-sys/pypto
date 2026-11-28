# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Measure warm native vLLM requests with pre-tokenized, fixed-length inputs.

Run only inside a device allocation. Engine timestamps include scheduling and
sampling, so these measurements must not be labelled device-kernel latency.
"""

import argparse
import dataclasses
import importlib
import json
import os
from pathlib import Path
from time import perf_counter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()
    devices = os.environ["TASK_DEVICE"].split(",")
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(devices)
    vllm = importlib.import_module("vllm")

    tokens = json.loads(args.prompt_ids.read_text())
    if not tokens or args.rounds < 2 or args.steps < 2:
        parser.error("Require nonempty input, >=2 rounds and >=2 output tokens")
    llm = vllm.LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=len(devices),
        enforce_eager=args.enforce_eager,
        max_model_len=128,
        max_num_seqs=1,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        async_scheduling=False,
        block_size=128,
        disable_log_stats=False,
    )
    params = vllm.SamplingParams(temperature=0, max_tokens=args.steps, ignore_eos=True)
    rows = []
    expected = None
    try:
        for trial in range(args.rounds + 1):
            start = perf_counter()
            outputs = llm.generate([{"prompt_token_ids": tokens}], params, use_tqdm=False)
            elapsed = (perf_counter() - start) * 1000
            assert len(outputs) == 1
            result = outputs[0]
            generated = list(result.outputs[0].token_ids)
            assert len(generated) == args.steps
            if expected is None:
                expected = generated
            assert generated == expected, "Greedy tokens changed between repetitions"
            stats = result.metrics
            assert stats is not None and not stats.is_corrupted
            row = {
                "trial": trial,
                "warmup": trial == 0,
                "tokens": generated,
                "request_wall_ms": elapsed,
                "metrics": dataclasses.asdict(stats),
                "prefill_engine_ms": (stats.first_token_ts - stats.scheduled_ts) * 1000,
                "decode_engine_mean_ms": (
                    (stats.last_token_ts - stats.first_token_ts) * 1000 / (args.steps - 1)
                ),
            }
            rows.append(row)
            # Retain completed observations even if a later round fails.
            args.output.write_text(
                json.dumps(
                    {
                        "backend": "native_vllm",
                        "enforce_eager": args.enforce_eager,
                        "batch_invariant": os.environ.get("VLLM_BATCH_INVARIANT", "0"),
                        "tp": len(devices),
                        "prompt_ids": tokens,
                        "rows": rows,
                    },
                    indent=2,
                )
                + "\n"
            )
            print(json.dumps(row), flush=True)
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
