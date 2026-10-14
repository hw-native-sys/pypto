# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compare software and hardware FP4 decoding, without attention or model dependencies."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np
import pypto.language as pl
import torch
from pypto.ir import CompiledProgram
from pypto.pypto_core.passes import MemoryPlanner
from pypto.runtime import RunConfig, benchmark, pto_isa_include_dir
from pypto.runtime.device_runner import _compile_and_assemble

N = pl.dynamic("ROWS")
K = 32
D = 512
WORKERS = 32
ROOT = Path(__file__).resolve().parent


@pl.jit(auto_scope=False)
def dsl_manual(packed: pl.Tensor[[N, D // 2], pl.UINT8], output: pl.Out[pl.Tensor[[N, D], pl.BF16]]):
    rows = pl.tensor.dim(packed, 0)
    with pl.spmd(WORKERS, sync_start=False):
        worker = pl.tile.get_block_idx()
        for row in pl.range(worker * K, rows, WORKERS * K):
            payload = pl.tile.load(packed, [row, 0], [K, D // 2])
            signed = pl.reinterpret_view(payload, pl.INT8)
            wide = pl.cast(signed, pl.INT16)
            byte_values = pl.ands(wide, 255)
            low = pl.ands(byte_values, 15)
            high = pl.shrs(byte_values, 4)
            low_code = pl.ands(low, 7)
            low_exponent = pl.shls(low_code, 9)
            low_base = pl.add(low_exponent, 0x3800)
            low_subnormal = pl.minimum(pl.sub(low_code, 2), 0)
            low_adjust = pl.shls(low_subnormal, 9)
            low_magnitude = pl.add(low_base, low_adjust)
            low_nonzero = pl.minimum(low_code, 1)
            low_bits = pl.mul(low_magnitude, low_nonzero)
            low_sign_code = pl.ands(low, 8)
            low_sign = pl.shls(low_sign_code, 12)
            low_signed = pl.or_(low_bits, low_sign)
            low_values = pl.reinterpret_view(low_signed, pl.FP16)
            high_code = pl.ands(high, 7)
            high_exponent = pl.shls(high_code, 9)
            high_base = pl.add(high_exponent, 0x3800)
            high_subnormal = pl.minimum(pl.sub(high_code, 2), 0)
            high_adjust = pl.shls(high_subnormal, 9)
            high_magnitude = pl.add(high_base, high_adjust)
            high_nonzero = pl.minimum(high_code, 1)
            high_bits = pl.mul(high_magnitude, high_nonzero)
            high_sign_code = pl.ands(high, 8)
            high_sign = pl.shls(high_sign_code, 12)
            high_signed = pl.or_(high_bits, high_sign)
            high_values = pl.reinterpret_view(high_signed, pl.FP16)
            low_lanes = pl.tile.full([K, D], dtype=pl.FP16, value=0.0)
            high_lanes = pl.tile.full([K, D], dtype=pl.FP16, value=0.0)
            low_lanes = pl.tile.scatter_mask(low_lanes, low_values, pl.tile.MaskPattern.P0101)
            high_lanes = pl.tile.scatter_mask(high_lanes, high_values, pl.tile.MaskPattern.P1010)
            low_bits = pl.reinterpret_view(low_lanes, pl.INT16)
            high_bits = pl.reinterpret_view(high_lanes, pl.INT16)
            bits = pl.or_(low_bits, high_bits)
            values = pl.reinterpret_view(bits, pl.FP16)
            decoded = pl.cast(values, pl.BF16, mode="rint")
            pl.tile.store(decoded, [row, 0], output)
    return output


@pl.jit(auto_scope=False)
def dsl_native(packed: pl.Tensor[[N, D // 2], pl.UINT8], output: pl.Out[pl.Tensor[[N, D], pl.BF16]]):
    rows = pl.tensor.dim(packed, 0)
    with pl.spmd(WORKERS, sync_start=False):
        worker = pl.tile.get_block_idx()
        for row in pl.range(worker * K, rows, WORKERS * K):
            payload = pl.tile.load(packed, [row, 0], [K, D // 2])
            fp4 = pl.reinterpret_view(payload, pl.FP4E2M1X2)
            decoded = pl.cast(fp4, pl.BF16)
            pl.tile.store(decoded, [row, 0], output)
    return output


@pl.jit.extern(core_type="aiv", source=ROOT / "native_decode.cpp", include_dirs=(pto_isa_include_dir(),))
def native_decode(
    output: pl.Out[pl.Tensor[[N, D], pl.BF16]],
    packed: pl.Tensor[[N, D // 2], pl.UINT8],
    rows: pl.Scalar[pl.INT32],
):
    return output


@pl.jit(auto_scope=False)
def cce_native(packed: pl.Tensor[[N, D // 2], pl.UINT8], output: pl.Out[pl.Tensor[[N, D], pl.BF16]]):
    rows = pl.tensor.dim(packed, 0)
    with pl.spmd(WORKERS, sync_start=False):
        output = native_decode(output, packed, rows)
    return output


def digest():
    return {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ["repro.py", "native_decode.cpp", "baseline/sources.json"]
    }


def inputs(rows):
    generator = torch.Generator().manual_seed(4102026)
    packed = torch.randint(0, 256, (rows, D // 2), dtype=torch.uint8, generator=generator)
    packed.flatten()[:256] = torch.arange(256, dtype=torch.uint8)
    codes = torch.stack([packed & 15, packed >> 4], dim=-1).long()
    lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
    )
    expected = lut[codes].reshape(rows, D).bfloat16()
    return packed, expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["build", "run"], required=True)
    parser.add_argument("--rows", nargs="+", type=int, default=[1024, 8192, 32768])
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["cce_native", "dsl_manual", "dsl_native"],
        default=["cce_native", "dsl_manual", "dsl_native"],
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--memory-planner", choices=["pypto", "dsa_rp"], default="pypto")
    args = parser.parse_args()
    for rows in args.rows:
        if rows < K or rows % K:
            raise ValueError("rows must be a positive multiple of 32")
    planner = {"pypto": MemoryPlanner.PYPTO, "dsa_rp": MemoryPlanner.DSA_RP}[args.memory_planner]
    cfg = RunConfig(platform="a5", device_id=args.device, memory_planner=planner)
    manifest_file = ROOT / f"builds-{args.memory_planner}.json"
    if args.phase == "build":
        builds = {}
        for rows in args.rows:
            packed, expected = inputs(rows)
            for variant in args.variants:
                x = packed
                compiled = globals()[variant].compile(x, torch.empty_like(expected), config=cfg)
                _compile_and_assemble(Path(compiled.output_dir), "a5")
                builds[f"{rows}/{variant}"] = str(compiled.output_dir)
                print("CPU_BUILD_PASS", rows, variant, flush=True)
        manifest_file.write_text(json.dumps(dict(sources=digest(), builds=builds), indent=2))
        return
    manifest = json.loads(manifest_file.read_text())
    if manifest["sources"] != digest():
        raise RuntimeError("Rebuild: source differs from the CPU-prebuilt artifact")
    results = []
    for rows in args.rows:
        packed, expected = inputs(rows)
        input_sha = hashlib.sha256(packed.numpy().tobytes()).hexdigest()
        for variant in args.variants:
            x = packed
            output = torch.full_like(expected, float("nan"))
            compiled = CompiledProgram.from_dir(manifest["builds"][f"{rows}/{variant}"], platform="a5")
            compiled(x, output, config=cfg)
            different_bits = int((output.view(torch.int16) != expected.view(torch.int16)).sum())
            if different_bits:
                raise RuntimeError(f"{variant}: {different_bits} mismatched BF16 bit patterns")
            stats = benchmark(compiled, [x, output], rounds=args.rounds, warmup=args.warmup, config=cfg)
            samples = stats.per_round("effective")
            if len(samples) != args.rounds or not all(math.isfinite(t) and t > 0 for t in samples):
                raise RuntimeError(f"Invalid timing samples: {samples}")
            if not torch.equal(output.view(torch.int16), expected.view(torch.int16)):
                raise RuntimeError(f"{variant}: output changed during repeated execution")
            record = dict(
                rows=rows,
                variant=variant,
                input_sha256=input_sha,
                sources=digest(),
                device=args.device,
                memory_planner=args.memory_planner,
                rounds=args.rounds,
                warmup=args.warmup,
                tile_rows=K,
                workers=WORKERS,
                sync_start=False,
                different_bits=different_bits,
                metric="effective_us",
                median_us=statistics.median(samples),
                p10_us=float(np.percentile(samples, 10)),
                p90_us=float(np.percentile(samples, 90)),
                samples_us=samples,
            )
            results.append(record)
            (ROOT / f"results-{args.memory_planner}.json").write_text(json.dumps(results, indent=2))
            print("RESULT", rows, variant, record["median_us"], "us; BF16 bit mismatches=0", flush=True)


if __name__ == "__main__":
    main()
