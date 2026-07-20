# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fail if docs/en and docs/zh-cn markdown path sets diverge.

English docs are ground truth; zh-CN must mirror the same relative file tree.
This check does not compare file contents or translation freshness.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EN = ROOT / "docs" / "en"
ZH = ROOT / "docs" / "zh-cn"


def rel_mds(base: Path) -> set[str]:
    if not base.is_dir():
        print(f"Error: missing docs tree {base}", file=sys.stderr)
        sys.exit(2)
    return {p.relative_to(base).as_posix() for p in base.rglob("*.md")}


def main() -> int:
    en, zh = rel_mds(EN), rel_mds(ZH)
    only_en, only_zh = sorted(en - zh), sorted(zh - en)
    if not only_en and not only_zh:
        print(f"OK: {len(en)} paired markdown paths under docs/en ↔ docs/zh-cn")
        return 0
    if only_en:
        print("EN-only (missing zh-cn):")
        print("\n".join(f"  {p}" for p in only_en))
    if only_zh:
        print("ZH-only (missing en):")
        print("\n".join(f"  {p}" for p in only_zh))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
