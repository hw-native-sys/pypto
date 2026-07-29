# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""MkDocs build hooks for the PyPTO documentation site.

The docs deliberately link to source files outside ``docs/`` — ``python/pypto/**``,
``include/pypto/**``, ``examples/**``, ``.claude/rules/**`` — using repo-relative
paths, so a reader browsing the repository on GitHub can click straight through to
the code. MkDocs cannot resolve those targets (they are not documentation pages),
and ``mkdocs build --strict`` turns each one into a build failure.

This hook rewrites exactly those links to absolute GitHub blob URLs while the site
is being built. The markdown on disk is untouched, so it stays the single source of
truth and stays clickable on GitHub.

See https://www.mkdocs.org/user-guide/configuration/#hooks for the hook protocol.
"""

import posixpath
import re
from typing import Any

# Matches inline markdown links and images: `[label](target)` / `![alt](target)`,
# with an optional `"title"` after the target. Group 3 is the target.
_LINK = re.compile(r'(!?)\[([^\]]*)\]\(([^)\s]+)((?:\s+"[^"]*")?)\)')

# Line ranges (`#L38`, `#L63-L81`) are GitHub anchors, not markdown headings, and
# must survive the rewrite intact.
_GITHUB_BLOB = "https://github.com/hw-native-sys/pypto/blob/main/"

# Fenced code blocks must not be rewritten — a link inside an example is content,
# not navigation.
_FENCE = re.compile(r"^(\s*)(```+|~~~+)", re.MULTILINE)


def _is_repo_relative(target: str) -> bool:
    """Return True for a relative path that is not a URL, anchor, or template var."""
    if not target or target.startswith(("#", "/", "<", "{")):
        return False
    return "://" not in target and not target.startswith("mailto:")


def _split_fragment(target: str) -> tuple[str, str]:
    path, sep, fragment = target.partition("#")
    return path, sep + fragment


def _code_spans(markdown: str) -> list[tuple[int, int]]:
    """Return (start, end) offsets of fenced code blocks, so they can be skipped."""
    spans: list[tuple[int, int]] = []
    open_at: int | None = None
    for match in _FENCE.finditer(markdown):
        if open_at is None:
            open_at = match.start()
        else:
            spans.append((open_at, match.end()))
            open_at = None
    if open_at is not None:
        spans.append((open_at, len(markdown)))
    return spans


def on_page_markdown(markdown: str, page: Any, config: Any, files: Any) -> str:
    """Rewrite repo-relative links that point outside ``docs/`` into GitHub URLs.

    Args:
        markdown: Raw markdown source of the page.
        page: The page being rendered; ``page.file.src_uri`` locates it in ``docs/``.
        config: The MkDocs config (unused).
        files: The collection of documentation files (unused).

    Returns:
        The markdown with out-of-``docs/`` links replaced by absolute GitHub URLs.
    """
    del config, files  # part of the hook signature, not needed here

    # Directory of this page relative to `docs/`, e.g. `en/dev/passes`.
    page_dir = posixpath.dirname(page.file.src_uri)
    skip = _code_spans(markdown)

    def rewrite(match: re.Match[str]) -> str:
        if any(start <= match.start() < end for start, end in skip):
            return match.group(0)

        bang, label, target, title = match.groups()
        if not _is_repo_relative(target):
            return match.group(0)

        path, fragment = _split_fragment(target)
        if not path:
            return match.group(0)

        # Resolve the target against the repo root. Anything still under `docs/`
        # is a documentation page and MkDocs resolves it itself; anything else
        # escaped the docs tree and needs an absolute URL to survive `--strict`.
        repo_path = posixpath.normpath(posixpath.join("docs", page_dir, path))
        if repo_path.startswith("docs/") or repo_path == "docs":
            return match.group(0)
        if repo_path.startswith("../"):  # escaped the repository itself — leave it
            return match.group(0)

        return f"{bang}[{label}]({_GITHUB_BLOB}{repo_path}{fragment}{title})"

    return _LINK.sub(rewrite, markdown)
