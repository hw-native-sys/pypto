# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression tests for the docs symbol-coverage gate."""

from pathlib import Path

import check_docs_symbol_coverage as coverage
import pytest


def test_catalog_rows_parse_api_links(tmp_path: Path) -> None:
    """The catalog's reference-link API cell contributes its pld.tensor name."""
    catalog = tmp_path / "catalog.md"
    catalog.write_text(
        "\n## Distributed\n\n"
        "| Operation | API |\n"
        "| --- | --- |\n"
        "| AllReduce | [`pld.tensor.allreduce`][pypto.language.distributed.op.tensor_ops.allreduce] |\n"
        "\n## Next\n",
        encoding="utf-8",
    )
    assert coverage.catalog_distributed_names(catalog) == {"allreduce"}


def test_catalog_rows_reject_multiple_tensor_apis(tmp_path: Path) -> None:
    """One catalog row must not stand in for two tensor operators."""
    catalog = tmp_path / "catalog.md"
    catalog.write_text(
        "\n## Distributed\n\n"
        "| Operation | API |\n"
        "| --- | --- |\n"
        "| Combined | `pld.tensor.allreduce`, `pld.tensor.allgather` |\n"
        "\n## Next\n",
        encoding="utf-8",
    )
    with pytest.raises(coverage.CatalogError):
        coverage.catalog_distributed_names(catalog)


def test_catalog_ignores_fenced_example(tmp_path: Path) -> None:
    """A fenced table example is not a catalog row."""
    catalog = tmp_path / "catalog.md"
    catalog.write_text(
        "\n## Distributed\n\n"
        "```text\n"
        "| Operation | API |\n"
        "| --- | --- |\n"
        "| AllReduce | [`pld.tensor.allreduce`][ref] |\n"
        "```\n"
        "\n## Next\n",
        encoding="utf-8",
    )
    assert coverage.catalog_distributed_names(catalog) == set()


def test_catalog_missing_section_is_reportable(tmp_path: Path) -> None:
    """A missing Distributed section is a CatalogError, not an unconditional exit."""
    catalog = tmp_path / "catalog.md"
    catalog.write_text("\n## Other\n", encoding="utf-8")
    with pytest.raises(coverage.CatalogError):
        coverage.catalog_distributed_names(catalog)


def test_catalog_missing_file_is_reportable(tmp_path: Path) -> None:
    """An unreadable catalog is a CatalogError so --report can still return 0."""
    with pytest.raises(coverage.CatalogError):
        coverage.catalog_distributed_names(tmp_path / "missing.md")


def test_qualified_pld_does_not_collapse_categories(tmp_path: Path) -> None:
    """`pld.tensor.get` must not satisfy the qualified check for `pld.tile.get`."""
    (tmp_path / "page.md").write_text("Use `pld.tensor.get`.\n", encoding="utf-8")
    assert coverage.documented_qualified_pld(tmp_path) == {"pld.tensor.get"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
