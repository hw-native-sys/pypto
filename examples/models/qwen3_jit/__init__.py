# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Qwen3-32B single-layer decode forward, expressed via the JIT interface.

Demonstrates cross-file utility composition:

  - ``kernels/`` holds reusable per-scope utilities decorated with
    ``@pl.jit.inline`` / ``@pl.jit.incore`` / ``@pl.jit.opaque``. Each kernel
    file is independently importable.
  - ``qwen3_decode.py`` is the ``@pl.jit`` entry that composes the kernels.
  - ``config.py`` holds shape/tile constants shared across kernels and entry.

The IR pass pipeline (``InlineFunctions`` runs first) splices Inline
utilities into the entry, then ``OutlineIncoreScopes`` extracts each
``pl.at`` block normally — the post-pass IR matches what a single monolithic
``@pl.program`` would produce.
"""
