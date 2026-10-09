# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Vendored boundary stubs for the ``simpler`` runtime package.

Only the surface pypto consumes is stubbed; the runtime package itself is
untyped. These stubs shadow the editable source install for pyright via
``stubPath`` (and make ``simpler`` resolvable in simpler-less type-check
environments). Delete this directory when the runtime ships its own
annotations/stubs.
"""
