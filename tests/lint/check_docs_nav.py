# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fail if mkdocs.yml's nav and the docs/en tree disagree.

Two invariants, each of which breaks the documentation site when violated:

1. Every nav entry exists. A stale path makes `mkdocs build --strict` fail.
2. Every English page is in the nav. A page missing from the nav is reachable only
   by guessing its URL — the site never links to it.

The nav is declared with default-locale (`en/...`) paths; mkdocs-static-i18n swaps
in the docs/zh-cn counterpart for the Chinese build, so only docs/en is checked
here. That the two trees mirror each other is check_docs_en_zh_parity.py's job.

Runs without MkDocs installed (PyYAML only).
"""

import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
MKDOCS_YML = ROOT / "mkdocs.yml"
DOCS_DIR = ROOT / "docs"
DEFAULT_LOCALE = "en"


class _LenientLoader(yaml.SafeLoader):
    """YAML loader that tolerates MkDocs' Python-object and !ENV tags."""


def _ignore_unknown(loader: yaml.Loader, suffix: str, node: yaml.Node) -> None:
    """Drop tags MkDocs understands but this checker does not need."""
    del loader, suffix, node


_LenientLoader.add_multi_constructor("tag:yaml.org,2002:python/name:", _ignore_unknown)
_LenientLoader.add_multi_constructor("!", _ignore_unknown)


def collect_nav_paths(nav: Any, out: list[str]) -> None:
    """Walk a MkDocs nav structure and collect every referenced document path."""
    if isinstance(nav, str):
        if "://" not in nav:  # external links are not documents
            out.append(nav)
    elif isinstance(nav, list):
        for entry in nav:
            collect_nav_paths(entry, out)
    elif isinstance(nav, dict):
        for value in nav.values():
            collect_nav_paths(value, out)


def main() -> int:
    if not MKDOCS_YML.is_file():
        print(f"Error: missing {MKDOCS_YML}", file=sys.stderr)
        return 2

    config = yaml.load(MKDOCS_YML.read_text(encoding="utf-8"), Loader=_LenientLoader)
    nav_paths: list[str] = []
    collect_nav_paths(config.get("nav", []), nav_paths)

    missing = [p for p in nav_paths if not (DOCS_DIR / p).is_file()]
    on_disk = {md.relative_to(DOCS_DIR).as_posix() for md in (DOCS_DIR / DEFAULT_LOCALE).rglob("*.md")}
    orphans = sorted(on_disk - set(nav_paths))

    if not missing and not orphans:
        print(f"OK: {len(nav_paths)} nav entries cover all {len(on_disk)} pages under docs/{DEFAULT_LOCALE}")
        return 0

    if missing:
        print("Nav entries with no such file:")
        print("\n".join(f"  docs/{p}" for p in missing))
    if orphans:
        print("Pages missing from the mkdocs.yml nav:")
        print("\n".join(f"  docs/{p}" for p in orphans))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
