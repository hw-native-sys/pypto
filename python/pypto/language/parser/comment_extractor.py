# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Extract ``#`` comments from DSL source text.

The Python ``ast`` module discards comments, so we run ``tokenize`` independently
and key each comment by its (1-based) line number. Each entry carries the column
offset so the parser can distinguish tail-of-block comments (inside body indent)
from outer-scope comments (at the enclosing indent).
"""

import io
import tokenize


def extract_line_comments(source: str) -> dict[int, list[tuple[int, str]]]:
    """Return a mapping from 1-based line number to ``(col_offset, text)`` tuples.

    The leading ``#`` and any single space after it are stripped from each
    comment. Trailing whitespace is preserved (comments are emitted verbatim).

    Args:
        source: Python source code (same text that will be fed to :func:`ast.parse`)

    Returns:
        Dict keyed by line number. Each value is a list of ``(col_offset, text)``
        pairs in source order. Lines without comments are absent from the map.
    """
    result: dict[int, list[tuple[int, str]]] = {}
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for tok in tokens:
            if tok.type == tokenize.COMMENT:
                # Strip exactly one leading '#' and at most one following space.
                # Multi-hash forms like "## heading" keep their extra hashes so
                # the printed comment still reads "# heading" after re-emission.
                text = tok.string[1:] if tok.string.startswith("#") else tok.string
                if text.startswith(" "):
                    text = text[1:]
                result.setdefault(tok.start[0], []).append((tok.start[1], text))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # Malformed source — caller's ast.parse will surface a clearer error.
        return result
    return result
