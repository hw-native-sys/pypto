# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Private spellings for lossless IR text; not a kernel authoring API.

The AST parser also recognizes ``comm_domain``, ``window_buffer``, ``window_ref``,
``alloc_window_buffer``, and ``call`` inside printed programs. These are syntax
markers rather than Python functions: function bodies are parsed, not executed.
Window descriptors use named references, for example
``window_buffer("buf_window", base="buf", size=16)`` and
``window_ref("buf_window")``. Labels encode object sharing without adding IR
fields. ``call`` accepts any result type supported by the type parser.
"""

import re
import struct

from pypto.pypto_core import ir

# Annotation-only marker values. The parser resolves the spelling, not the value.
WindowBuffer = ir.WindowBufferType.get()
Unknown = ir.UnknownType()


def float64(bits: str) -> float:
    """Decode exactly one IEEE 754 binary64 value, retaining NaN sign and payload."""
    if not isinstance(bits, str) or re.fullmatch(r"[0-9a-fA-F]{16}", bits) is None:  # pyright: ignore[reportUnnecessaryIsInstance] -- mistyped-input guard on raw dump content
        raise ValueError("float64 requires exactly 16 hexadecimal digits")
    return struct.unpack(">d", bytes.fromhex(bits))[0]
