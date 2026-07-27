# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Immutable data models for IR pass traces."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


class IRTraceError(ValueError):
    """Report an actionable IR trace input or output error."""


@dataclass(frozen=True)
class Snapshot:
    index: int
    pass_name: str | None
    path: Path
    text: str
    lines: tuple[str, ...]
    warning_text: str | None = None


@dataclass(frozen=True)
class DiffRow:
    kind: Literal["equal", "insert", "delete", "replace"]
    before_number: int | None
    before_html: str
    after_number: int | None
    after_html: str


@dataclass(frozen=True)
class DiffHunk:
    rows: tuple[DiffRow, ...]
    collapsed: bool


@dataclass(frozen=True)
class PassTrace:
    index: int
    name: str
    before: Snapshot
    after: Snapshot
    inserted: int
    deleted: int
    hunks: tuple[DiffHunk, ...]

    @property
    def changed(self) -> bool:
        return self.inserted != 0 or self.deleted != 0
