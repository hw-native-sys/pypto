# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Per-transfer L2-hint finalization without a modified assembler."""

import pytest
from pypto.backend._l2_hint import apply_l2_hints
from pypto.backend._ptoas_preprocess import preprocess_ptoas_output
from pypto.pypto_core.ir import LoadL2Hint, StoreL2Hint


def _region(direction, hint, body):
    return f"// __pypto_l2_hint_begin {direction} {hint}\n{body}\n// __pypto_l2_hint_end\n"


@pytest.mark.parametrize("hint", list(LoadL2Hint))
def test_load_hints_are_explicit_including_zero(hint):
    body = "  TLOAD(tile, gm);"
    result = apply_l2_hints(_region("load", hint.name, body))
    assert result == f"  TLOAD<pto::TLoadL2Hint::{hint.name}>(tile, gm);\n"


@pytest.mark.parametrize("hint", list(StoreL2Hint))
@pytest.mark.parametrize(
    "call, tail",
    [
        ("TSTORE(tile, gm);", "(tile, gm);"),
        ("TSTORE<STPhase::Final>(gm, tile);", ", STPhase::Final>(gm, tile);"),
        (
            "TSTORE<Tile<float, 16, 32>, GlobalTensor<float, Shape<1, 2>>, AtomicType::AtomicAdd,"
            " ReluPreMode::NormalRelu>(gm, tile, quant);",
            ", Tile<float, 16, 32>, GlobalTensor<float, Shape<1, 2>>, AtomicType::AtomicAdd,"
            " ReluPreMode::NormalRelu>(gm, tile, quant);",
        ),
        ("TSTORE_FP(gm, tile, fp);", "(gm, tile, fp);"),
    ],
)
def test_store_preserves_other_template_arguments(hint, call, tail):
    result = apply_l2_hints(_region("store", hint.name, call))
    close = ">" if tail.startswith("(") else ""
    assert result == f"TSTORE<pto::TStoreL2Hint::{hint.name}{close}{tail}\n"


def test_only_marked_calls_change_and_comments_are_not_calls():
    source = (
        "TLOAD(other, gm);\n"
        + _region(
            "load",
            "NormalLastVictim",
            "// TSTORE(fake, fake);\n/* TLOAD(fake, fake); */\n"
            'printf("TLOAD(fake, fake);");\npto::TLOAD(tile, gm);',
        )
        + "TSTORE(gm, other);\n"
    )
    result = apply_l2_hints(source)
    assert result.startswith("TLOAD(other, gm);\n")
    assert result.endswith("TSTORE(gm, other);\n")
    assert "pto::TLOAD<pto::TLoadL2Hint::NormalLastVictim>(tile, gm);" in result
    assert 'printf("TLOAD(fake, fake);");' in result
    assert apply_l2_hints(result) == result


@pytest.mark.parametrize(
    "source, message",
    [
        ("// __pypto_l2_hint_end\n", "without a begin"),
        ("// __pypto_l2_hint_begin load NormalFirstVictim\n", "missing end"),
        ("// __pypto_l2_hint_begin load Unknown\n", "invalid load"),
        ("// __pypto_l2_hint_begin store NotAllocDrop\n", "invalid store"),
        ("// __pypto_l2_hint_bad\n", "malformed marker"),
        (_region("load", "NormalFirstVictim", ""), "exactly one"),
        (_region("load", "NormalFirstVictim", "TSTORE(a, b);"), "exactly one"),
        (_region("load", "NormalFirstVictim", "TLOAD(a,b); TLOAD(c,d);"), "exactly one"),
        (_region("store", "NormalFirstVictim", "TSTORE<Shape<1, 2>(a,b);"), "malformed ISA"),
        (
            _region("load", "NormalFirstVictim", "TLOAD<pto::TLoadL2Hint::NotAllocClean>(a,b);"),
            "already has a hint",
        ),
        (
            _region("load", "NormalFirstVictim", _region("load", "NormalLastVictim", "TLOAD(a,b);")),
            "nested begin",
        ),
    ],
)
def test_invalid_output_fails_closed(source, message):
    with pytest.raises(RuntimeError, match=message):
        apply_l2_hints(source)


def test_wrapper_and_rebuild_preprocessing_apply_hints():
    source = _region("store", "NormalPersistent", "TSTORE(dst, src);")
    assert preprocess_ptoas_output(source) == "TSTORE<pto::TStoreL2Hint::NormalPersistent>(dst, src);\n"


def test_emitc_statement_terminators_on_markers():
    source = "// __pypto_l2_hint_begin load NormalLastVictim;\nTLOAD(tile, gm);\n// __pypto_l2_hint_end;\n"
    assert apply_l2_hints(source) == "TLOAD<pto::TLoadL2Hint::NormalLastVictim>(tile, gm);\n"


def test_unmarked_output_is_unchanged():
    source = "TLOAD(a, b);\nTSTORE<X, Y, AtomicType::AtomicAdd>(b, a);\n"
    assert apply_l2_hints(source) == source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
