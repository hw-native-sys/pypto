# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prove `check_no_foreign_targets` can fire, and can tell code from comment.

A lint is only worth its exit code if a known-bad input makes it non-zero. These
tests seed one finding per class and assert the message, then assert the benign
shapes stay silent -- because a check that flags every mention of another vendor
would be turned off, and a check that flags nothing would never be noticed.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from check_no_foreign_targets import check_file, is_comment  # noqa: E402


def _write(tmp_path: Path, body: str) -> tuple[Path, Path]:
    """Write `body` to a .rs file inside a fake repository root."""
    root = tmp_path
    target = root / "rust" / "crates" / "probe" / "src" / "lib.rs"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return target, root


def test_fires_on_foreign_target_in_code(tmp_path):
    target, root = _write(tmp_path, 'let backend = "cuda";\n')
    findings = check_file(target, root)
    assert len(findings) == 1
    assert "cuda" in findings[0]
    assert "CANN OSL 3.1" in findings[0]


def test_silent_on_foreign_target_in_a_comment(tmp_path):
    # The distinction the whole check rests on: 42 code hits mattered in the
    # measured tree and 194 comment hits did not.
    target, root = _write(tmp_path, "// Unlike CUDA shared memory, UB is not per-block.\nlet x = 1;\n")
    assert check_file(target, root) == []


def test_fires_on_process_vocabulary_even_in_a_comment(tmp_path):
    target, root = _write(tmp_path, "/// M126: fused decode-time matvec.\nlet x = 1;\n")
    findings = check_file(target, root)
    assert len(findings) == 1
    assert "M126:" in findings[0]


def test_silent_on_apple_silicon_generation_names(tmp_path):
    # `M1 Ultra` and `M5 Max` are hardware generations, not milestone tags, and
    # a check that refused them would make honest prose unwriteable.
    target, root = _write(tmp_path, "// Measured on an M1 Ultra and an M5 Max.\nlet x = 1;\n")
    assert check_file(target, root) == []


def test_silent_on_an_apple_silicon_generation_before_a_colon(tmp_path):
    # Hardware provenance, not a milestone. This exact line is in the emitter
    # this tree carries, and the first version of the check flagged it.
    target, root = _write(
        tmp_path,
        "//! Ported from a ranked receipt on Apple M5: upstream measured -12%.\nlet x = 1;\n",
    )
    assert check_file(target, root) == []


def test_fires_on_a_single_digit_milestone_that_is_not_apple(tmp_path):
    # The counterpart, and the reason digit count cannot be the discriminator:
    # `M5:` is a milestone here and silicon above, so only the preceding word
    # separates them. Both spellings occur in this project's own sources.
    target, root = _write(tmp_path, "// Optional, deferred to M5: not in this release.\nlet x = 1;\n")
    findings = check_file(target, root)
    assert len(findings) == 1
    assert "M5:" in findings[0]


def test_fires_on_a_bare_milestone_reference(tmp_path):
    # The miss that this check shipped with. `M94` carries no colon, so the
    # original pattern passed it -- and a bare id is the WORSE case: `M126:`
    # at least labels a definition, while "sibling of M94" is a pointer with
    # no referent anywhere in this repository. Measured in `tile_std`: 23
    # colon-form lines, 85 more bare-form lines, same file.
    target, root = _write(tmp_path, "/// R1PTG=3 sibling of M94.\nlet x = 1;\n")
    findings = check_file(target, root)
    assert len(findings) == 1
    assert "M94" in findings[0]


def test_fires_on_a_sub_milestone_letter_suffix(tmp_path):
    # `M104b` is the sub-milestone spelling, and it occurred in the same file.
    target, root = _write(tmp_path, "/// f16-dst sibling of M104b.\nlet x = 1;\n")
    findings = check_file(target, root)
    assert len(findings) == 1
    assert "M104b" in findings[0]


def test_silent_on_a_bare_apple_generation(tmp_path):
    # The counterweight. Two-and-three-digit ids are refused bare because no
    # Apple part is numbered that high, but the `Apple M5`/`M1 Ultra` prose
    # this check already protected must stay writable.
    target, root = _write(
        tmp_path,
        "// Measured on an M1 Ultra, an M5 Max and an Apple M5.\nlet x = 1;\n",
    )
    assert check_file(target, root) == []


def test_silent_on_identifiers_that_merely_contain_m_and_digits(tmp_path):
    # Word boundaries carry this: a refusal that fired inside `nb12`, `ne20`
    # or `simd_sum` would make the kernel ABI comments unwriteable, and those
    # spellings are all over the carried DSL.
    target, root = _write(
        tmp_path,
        "/// Params: ne00, nb12, ne20, r2, r3; reduce via simd_sum.\nlet x = 1;\n",
    )
    assert check_file(target, root) == []


def test_fires_on_private_doc_reference_with_digits(tmp_path):
    # Regression guard for a real miss: a pattern written `docs/[A-Za-z_]*\\.md`
    # reports ZERO hits on this filename, because the class excludes digits.
    target, root = _write(tmp_path, "//! See docs/DS4_THING_PLAN.md.\nlet x = 1;\n")
    findings = check_file(target, root)
    assert len(findings) == 1
    assert "DS4_THING_PLAN.md" in findings[0]


def test_silent_on_a_docs_path_inside_a_url(tmp_path):
    # Someone else's published specification is a citation, not a pointer at an
    # unpublished local file. The first version of this check flagged the GGUF
    # format spec in this tree.
    target, root = _write(
        tmp_path,
        "//! Spec reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md\nlet x = 1;\n",
    )
    assert check_file(target, root) == []


def test_fires_on_author_absolute_path(tmp_path):
    target, root = _write(tmp_path, "//! Reads /Users/someone/model.gguf.\nlet x = 1;\n")
    findings = check_file(target, root)
    assert len(findings) == 1
    assert "author-absolute path" in findings[0]


def test_clean_file_reports_nothing(tmp_path):
    target, root = _write(tmp_path, "//! Lower a tile-rs kernel to PTO.\npub fn emit() {}\n")
    assert check_file(target, root) == []


@pytest.mark.parametrize(
    "line,expected",
    [
        ("// a line comment", True),
        ("/// a doc comment", True),
        ("  * a block continuation", True),
        ("# a toml comment", True),
        ('let backend = "cuda";', False),
        ('    note: "served by metal",', False),
    ],
)
def test_is_comment(line, expected):
    assert is_comment(line) is expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
