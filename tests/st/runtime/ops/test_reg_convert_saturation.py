# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 regression tests for pl.reg.convert saturation under inherited SPR.CTRL state.

On Ascend A5 (dav-c310), hardware cold start and CANN ``asc_init()`` leave
``SPR.CTRL[60] = 1`` (``USE_GLOBAL``), causing the hardware to ignore the
per-instruction ``#sat`` operand of ``VCVTFF`` / ``VCVTFI`` and substitute
global ``CTRL[48]`` (float) / ``CTRL[59]`` (cast) bits instead.

These tests poison ``CTRL[60]``, ``CTRL[48]``, and ``CTRL[59]`` immediately
before ``kernel_entry``'s A5 vector reset and verify that both ``saturate=True``
and ``saturate=False`` follow the instruction-level flag across all supported
narrowing conversions. The tile-cast test starts with CTRL[60] cleared to
emulate a preceding register conversion and verify restoration of USE_GLOBAL.
"""

from dataclasses import replace

import pypto.language as pl
import pytest
import torch
from pypto import CacheConfig
from pypto.backend import BackendType, pto_backend
from pypto.runtime import ExecutionMode

LANES = 64

# Poison patterns emulated before kernel_entry's CTRL[60] reset:
# - non_sat_global: CTRL[60]=1 (USE_GLOBAL), CTRL[48]=1 (FLOAT non-sat), CTRL[59]=1 (CAST non-sat).
#   Without the entry reset, saturate=True is silently overridden to non-saturating.
# - sat_global: CTRL[60]=1 (USE_GLOBAL), CTRL[48]=0 (FLOAT sat), CTRL[59]=0 (CAST sat).
#   Without the entry reset, saturate=False is silently overridden to saturating.
_POISON_PATTERNS = [
    pytest.param(1, 1, id="poison_non_sat_global"),
    pytest.param(0, 0, id="poison_sat_global"),
]


def _poison_before_ctrl_write(monkeypatch: pytest.MonkeyPatch, write: str, poison: str) -> None:
    """Inject the inherited state inside the same A5 device guard as the entry write."""
    generate_wrapper = pto_backend._generate_kernel_wrapper

    def poison_inherited_ctrl(func, ptoas_code, **kwargs):
        source = generate_wrapper(func, ptoas_code, **kwargs)
        assert source.count(write) == 1
        return source.replace(write, poison + write, 1)

    monkeypatch.setattr(pto_backend, "_generate_kernel_wrapper", poison_inherited_ctrl)


def _install_ctrl_poison(monkeypatch: pytest.MonkeyPatch, ctrl48: int, ctrl59: int) -> None:
    op48 = "sbitset1" if ctrl48 else "sbitset0"
    op59 = "sbitset1" if ctrl59 else "sbitset0"
    _poison_before_ctrl_write(
        monkeypatch,
        "    set_ctrl(sbitset0(get_ctrl(), 60));\n",
        "    set_ctrl(sbitset1(get_ctrl(), 60));\n"
        f"    set_ctrl({op48}(get_ctrl(), 48));\n"
        f"    set_ctrl({op59}(get_ctrl(), 59));\n",
    )


def _repeat_to_lanes(values: list[float]) -> torch.Tensor:
    repeats = (LANES + len(values) - 1) // len(values)
    return torch.tensor((values * repeats)[:LANES], dtype=torch.float32).reshape(1, LANES)


def _fresh_config(test_config):
    return replace(
        test_config,
        arch=BackendType.Ascend950,
        execution_mode=ExecutionMode.ONBOARD,
        cache_config=CacheConfig(enabled=False),
    )


@pytest.mark.platforms("a5")
@pytest.mark.parametrize("ctrl48,ctrl59", _POISON_PATTERNS)
@pytest.mark.parametrize("saturate", [True, False], ids=["sat", "nosat"])
def test_reg_convert_fp32_to_fp8_clears_inherited_ctrl(test_config, monkeypatch, ctrl48, ctrl59, saturate):
    """FP32 -> FP8E4M3FN obeys per-instruction saturate despite poisoned CTRL[60]."""
    _install_ctrl_poison(monkeypatch, ctrl48, ctrl59)

    @pl.jit.incore
    def convert_kernel(
        src: pl.Tensor[[1, LANES], pl.FP32],
        out: pl.Out[pl.Tensor[[1, LANES], pl.UINT8]],
    ) -> pl.Tensor[[1, LANES], pl.UINT8]:
        x_tile = pl.reshape(pl.load(src, [0, 0], [1, LANES], target_memory=pl.Mem.Vec), [LANES])
        y_tile = pl.tile.create([LANES], dtype=pl.UINT8, target_memory=pl.Mem.Vec)
        with pl.reg_scope():
            x = pl.reg.load(x_tile, lanes=LANES)
            fp8 = pl.reg.convert(x, dtype=pl.FP8E4M3FN, rounding="R", saturate=saturate)
            raw = pl.reg.interpret_cast(fp8, dtype=pl.UINT8)
            y_tile = pl.reg.store(y_tile, value=raw)
        pl.store(pl.reshape(y_tile, [1, LANES]), [0, 0], out)
        return out

    @pl.jit
    def entry(
        src: pl.Tensor[[1, LANES], pl.FP32],
        out: pl.Out[pl.Tensor[[1, LANES], pl.UINT8]],
    ) -> pl.Tensor[[1, LANES], pl.UINT8]:
        return convert_kernel(src, out)

    # -490.0 / +490.0 / +1000.0 exceed FP8 E4M3FN max finite magnitude (448.0).
    # In E4M3FN: -448.0 = 0xFE, +448.0 = 0x7E, -1.5 = 0xBC, 0.0 = 0x00, 1.5 = 0x3C;
    # NaN has exponent=0xF and mantissa=0x7 (0x7F or 0xFF).
    values = [-490.0, 490.0, -448.0, 448.0, -1.5, 0.0, 1.5, 1000.0]
    src = _repeat_to_lanes(values)
    out = torch.zeros((1, LANES), dtype=torch.uint8)
    config = _fresh_config(test_config)

    if config.codegen_only:
        entry.compile(config=config)
        return

    compiled = entry.compile(src, out, config=config)
    compiled(src, out, config=config)

    got = out.reshape(-1)
    if saturate:
        expected_cycle = [0xFE, 0x7E, 0xFE, 0x7E, 0xBC, 0x00, 0x3C, 0x7E]
        expected = torch.tensor((expected_cycle * (LANES // len(expected_cycle)))[:LANES], dtype=torch.uint8)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
    else:
        for idx in range(LANES):
            slot = idx % len(values)
            byte = int(got[idx].item())
            if slot in (0, 1, 7):
                assert (byte & 0x7F) == 0x7F, (
                    f"lane {idx} ({values[slot]}): expected FP8 NaN, got 0x{byte:02x}"
                )
            else:
                expected_in_range = {2: 0xFE, 3: 0x7E, 4: 0xBC, 5: 0x00, 6: 0x3C}[slot]
                assert byte == expected_in_range, (
                    f"lane {idx} ({values[slot]}): expected 0x{expected_in_range:02x}, got 0x{byte:02x}"
                )


@pytest.mark.platforms("a5")
@pytest.mark.parametrize("ctrl48,ctrl59", _POISON_PATTERNS)
@pytest.mark.parametrize("saturate", [True, False], ids=["sat", "nosat"])
def test_reg_convert_fp32_to_fp16_clears_inherited_ctrl(test_config, monkeypatch, ctrl48, ctrl59, saturate):
    """FP32 -> FP16 obeys per-instruction saturate despite poisoned CTRL[60]."""
    _install_ctrl_poison(monkeypatch, ctrl48, ctrl59)

    @pl.jit.incore
    def convert_kernel(
        src: pl.Tensor[[1, LANES], pl.FP32],
        out: pl.Out[pl.Tensor[[1, LANES], pl.FP16]],
    ) -> pl.Tensor[[1, LANES], pl.FP16]:
        x_tile = pl.reshape(pl.load(src, [0, 0], [1, LANES], target_memory=pl.Mem.Vec), [LANES])
        y_tile = pl.tile.create([LANES], dtype=pl.FP16, target_memory=pl.Mem.Vec)
        with pl.reg_scope():
            x = pl.reg.load(x_tile, lanes=LANES)
            y = pl.reg.convert(x, dtype=pl.FP16, rounding="R", saturate=saturate)
            y_tile = pl.reg.store(y_tile, value=y)
        pl.store(pl.reshape(y_tile, [1, LANES]), [0, 0], out)
        return out

    @pl.jit
    def entry(
        src: pl.Tensor[[1, LANES], pl.FP32],
        out: pl.Out[pl.Tensor[[1, LANES], pl.FP16]],
    ) -> pl.Tensor[[1, LANES], pl.FP16]:
        return convert_kernel(src, out)

    values = [-70000.0, 70000.0, -65504.0, 65504.0, -1.5, 0.0, 1.5, 1.0e10]
    src = _repeat_to_lanes(values)
    out = torch.zeros((1, LANES), dtype=torch.float16)
    config = _fresh_config(test_config)

    if config.codegen_only:
        entry.compile(config=config)
        return

    compiled = entry.compile(src, out, config=config)
    compiled(src, out, config=config)

    if saturate:
        expected = torch.clamp(src, -65504.0, 65504.0).to(torch.float16)
    else:
        expected = src.to(torch.float16)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.platforms("a5")
@pytest.mark.parametrize("ctrl48,ctrl59", _POISON_PATTERNS)
@pytest.mark.parametrize("saturate", [True, False], ids=["sat", "nosat"])
def test_reg_convert_fp32_to_bf16_clears_inherited_ctrl(test_config, monkeypatch, ctrl48, ctrl59, saturate):
    """FP32 -> BF16 obeys per-instruction saturate despite poisoned CTRL[60]."""
    _install_ctrl_poison(monkeypatch, ctrl48, ctrl59)

    @pl.jit.incore
    def convert_kernel(
        src: pl.Tensor[[1, LANES], pl.FP32],
        out: pl.Out[pl.Tensor[[1, LANES], pl.BF16]],
    ) -> pl.Tensor[[1, LANES], pl.BF16]:
        x_tile = pl.reshape(pl.load(src, [0, 0], [1, LANES], target_memory=pl.Mem.Vec), [LANES])
        y_tile = pl.tile.create([LANES], dtype=pl.BF16, target_memory=pl.Mem.Vec)
        with pl.reg_scope():
            x = pl.reg.load(x_tile, lanes=LANES)
            y = pl.reg.convert(x, dtype=pl.BF16, rounding="R", saturate=saturate)
            y_tile = pl.reg.store(y_tile, value=y)
        pl.store(pl.reshape(y_tile, [1, LANES]), [0, 0], out)
        return out

    @pl.jit
    def entry(
        src: pl.Tensor[[1, LANES], pl.FP32],
        out: pl.Out[pl.Tensor[[1, LANES], pl.BF16]],
    ) -> pl.Tensor[[1, LANES], pl.BF16]:
        return convert_kernel(src, out)

    # BF16 max finite magnitude is 3.3895313892515355e38 (0x7F7F); +/-3.4e38 is finite FP32
    # that overflows BF16.
    bf16_max = float(torch.finfo(torch.bfloat16).max)
    values = [-3.4e38, 3.4e38, -bf16_max, bf16_max, -1.5, 0.0, 1.5, 3.4028234e38]
    src = _repeat_to_lanes(values)
    out = torch.zeros((1, LANES), dtype=torch.bfloat16)
    config = _fresh_config(test_config)

    if config.codegen_only:
        entry.compile(config=config)
        return

    compiled = entry.compile(src, out, config=config)
    compiled(src, out, config=config)

    if saturate:
        expected = torch.clamp(src, -bf16_max, bf16_max).to(torch.bfloat16)
    else:
        expected = src.to(torch.bfloat16)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.platforms("a5")
@pytest.mark.parametrize("ctrl48,ctrl59", _POISON_PATTERNS)
@pytest.mark.parametrize("saturate", [True, False], ids=["sat", "nosat"])
def test_reg_convert_fp32_to_int32_clears_inherited_ctrl(test_config, monkeypatch, ctrl48, ctrl59, saturate):
    """FP32 -> INT32 obeys per-instruction saturate despite poisoned CTRL[60]."""
    _install_ctrl_poison(monkeypatch, ctrl48, ctrl59)

    @pl.jit.incore
    def convert_kernel(
        src: pl.Tensor[[1, LANES], pl.FP32],
        out: pl.Out[pl.Tensor[[1, LANES], pl.INT32]],
    ) -> pl.Tensor[[1, LANES], pl.INT32]:
        x_tile = pl.reshape(pl.load(src, [0, 0], [1, LANES], target_memory=pl.Mem.Vec), [LANES])
        y_tile = pl.tile.create([LANES], dtype=pl.INT32, target_memory=pl.Mem.Vec)
        with pl.reg_scope():
            x = pl.reg.load(x_tile, lanes=LANES)
            y = pl.reg.convert(x, dtype=pl.INT32, rounding="Z", saturate=saturate)
            y_tile = pl.reg.store(y_tile, value=y)
        pl.store(pl.reshape(y_tile, [1, LANES]), [0, 0], out)
        return out

    @pl.jit
    def entry(
        src: pl.Tensor[[1, LANES], pl.FP32],
        out: pl.Out[pl.Tensor[[1, LANES], pl.INT32]],
    ) -> pl.Tensor[[1, LANES], pl.INT32]:
        return convert_kernel(src, out)

    values = [-3.0e9, 3.0e9, -2147483648.0, 1000.0, -1000.0, 0.0, 42.75, -42.75]
    src = _repeat_to_lanes(values)
    out = torch.zeros((1, LANES), dtype=torch.int32)
    config = _fresh_config(test_config)

    if config.codegen_only:
        entry.compile(config=config)
        return

    compiled = entry.compile(src, out, config=config)
    compiled(src, out, config=config)

    if saturate:
        expected_cycle = [-2147483648, 2147483647, -2147483648, 1000, -1000, 0, 42, -42]
    else:
        # With #sat=0 the two overflow slots wrap modulo 2**32.
        expected_cycle = [1294967296, -1294967296, -2147483648, 1000, -1000, 0, 42, -42]
    expected = torch.tensor(expected_cycle, dtype=torch.int32).repeat(LANES // len(expected_cycle))
    torch.testing.assert_close(out.reshape(-1), expected, rtol=0, atol=0)


@pytest.mark.platforms("a5")
@pytest.mark.parametrize("saturation_mode", ["on", "off"])
def test_bf16_to_fp16_tile_cast_restores_ctrl60_after_reg_convert(test_config, monkeypatch, saturation_mode):
    _poison_before_ctrl_write(
        monkeypatch,
        "    set_ctrl(sbitset1(get_ctrl(), 60));\n",
        "    set_ctrl(sbitset0(get_ctrl(), 60));\n",
    )

    @pl.jit.incore
    def cast_kernel(
        src: pl.Tensor[[1, LANES], pl.BF16],
        out: pl.Out[pl.Tensor[[1, LANES], pl.FP16]],
    ) -> pl.Tensor[[1, LANES], pl.FP16]:
        tile = pl.load(src, [0, 0], [1, LANES], target_memory=pl.Mem.Vec)
        converted = pl.cast(tile, pl.FP16, saturation_mode=saturation_mode)
        pl.store(converted, [0, 0], out)
        return out

    @pl.jit
    def entry(
        src: pl.Tensor[[1, LANES], pl.BF16],
        out: pl.Out[pl.Tensor[[1, LANES], pl.FP16]],
    ) -> pl.Tensor[[1, LANES], pl.FP16]:
        return cast_kernel(src, out)

    config = _fresh_config(test_config)
    if config.codegen_only:
        entry.compile(config=config)
        return

    values = [-70000.0, 70000.0, -65504.0, 65504.0, -1.5, 0.0, 1.5, 1.0e10]
    src = _repeat_to_lanes(values).to(torch.bfloat16)
    out = torch.zeros((1, LANES), dtype=torch.float16)
    compiled = entry.compile(src, out, config=config)
    compiled(src, out, config=config)

    if saturation_mode == "on":
        expected = torch.clamp(src.float(), -65504.0, 65504.0).to(torch.float16)
    else:
        expected = src.to(torch.float16)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
