# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression coverage for the mirrored PTOAS documentation integration."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("ptoas_docs_hooks", ROOT / "scripts/mkdocs_hooks.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Cannot load the MkDocs hook module")
hooks = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = hooks
SPEC.loader.exec_module(hooks)


@pytest.fixture(autouse=True)
def snapshot(monkeypatch):
    monkeypatch.setattr(
        hooks,
        "_PTOAS",
        hooks._PtoasSnapshot(
            "a" * 40,
            {
                "docs/isa/vmi-isa/types.md": Path("types.md"),
                "docs/isa/vmi-isa/assets/layout.svg": Path("layout.svg"),
            },
        ),
    )


def test_imported_links_preserve_local_pages_and_assets():
    page = SimpleNamespace(file=SimpleNamespace(src_uri="en/reference/ptoas/source/docs/isa/vmi-isa/ops.md"))
    original = "[types](types.md#shape) ![layout](assets/layout.svg)"
    assert hooks.on_page_markdown(original, page, None, None) == original
    assert page.edit_url.endswith("/docs/isa/vmi-isa/ops.md")


def test_source_links_use_ptoas_revision_and_skip_code():
    page = SimpleNamespace(file=SimpleNamespace(src_uri="zh/reference/ptoas/source/docs/isa/vmi-isa/ops.md"))
    text = "[source](../../../include/PTO/IR/Ops.td)\n```text\n[x](../../../code)\n```"
    result = hooks.on_page_markdown(text, page, None, None)
    assert f"PTOAS/blob/{'a' * 40}/include/PTO/IR/Ops.td" in result
    assert "[x](../../../code)" in result


def test_upstream_path_cannot_escape_repository():
    with pytest.raises(ValueError, match="escapes its repository"):
        hooks._ptoas_link("../../secret", "docs/manual.md")


def test_generated_locale_urls_and_provenance(tmp_path):
    source = tmp_path / "manual.md"
    source.write_text("# Manual\n", encoding="utf-8")
    config = SimpleNamespace(
        site_dir=str(tmp_path / "site"),
        use_directory_urls=True,
        plugins=SimpleNamespace(_current_plugin="test"),
    )
    en = hooks._imported_file("docs/manual.md", source, "en", config)
    zh = hooks._imported_file("docs/manual.md", source, "zh", config)
    assert en.url == "reference/ptoas/source/docs/manual/"
    assert zh.url == "zh/reference/ptoas/source/docs/manual/"
    assert "Source: [PTOAS `aaaaaaaaaaaa`]" in en.content_string
    assert en.content_string.endswith("# Manual\n")


def test_navigation_imports_only_selected_manuals_and_designs(tmp_path, monkeypatch):
    checkout = tmp_path / ".cache/ptoas-docs"
    wanted = ["docs/PTO_IR_manual.md", "docs/isa/vmi-isa/00-overview.md", *hooks._PTOAS_DESIGNS.values()]
    excluded = ["docs/vpto-spec.md", "docs/isa/tile-op/01-overview.md", "docs/isa/micro-isa/01-sync.md"]
    for name in wanted + excluded:
        path = checkout / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Reference\n", encoding="utf-8")
    monkeypatch.setattr(hooks.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="a" * 40))
    config = SimpleNamespace(config_file_path=str(tmp_path / "mkdocs.yml"), nav=[{"PTOAS": "index.md"}])
    hooks.on_config(config)
    assert set(hooks._PTOAS.files) == set(wanted)
    navigation = str(config.nav)
    assert "Pass Designs" in navigation
    assert all(name in navigation for name in wanted)
    assert all(name not in navigation for name in excluded)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
