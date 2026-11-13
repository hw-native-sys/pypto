# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Forwarding module for the pre-move import spelling of the argument directions.

The definitions live in :mod:`pypto.language.typing.arg_direction`; this path
keeps ``pypto.language.arg_direction`` importable for external consumers.
"""

from pypto.language.typing.arg_direction import (
    DIRECTION_TO_NAME,
    NAME_TO_DIRECTION,
    ArgDirection,
    inout,
    input,
    no_dep,
    output,
    output_existing,
    scalar,
)

__all__ = [
    "ArgDirection",
    "DIRECTION_TO_NAME",
    "NAME_TO_DIRECTION",
    "inout",
    "input",
    "no_dep",
    "output",
    "output_existing",
    "scalar",
]
