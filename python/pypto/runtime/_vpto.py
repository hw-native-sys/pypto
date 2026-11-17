# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Build contract for separately compiled, self-contained A5 VPTO helpers."""

import re
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from . import _device_image

# These options are also included in the binary cache identity. Keep LLVM
# compilation separate from the C++ flags and retain PTOAS's validated device
# profile. Generic-address-space on/off happened to produce identical code for
# the add helper; this does not establish ABI equivalence for other helpers.
PTOAS_FLAGS = (
    "--pto-arch=a5",
    "--pto-backend=vpto",
    "--emit-vpto-llvm-ir",
    "--vmi-fastmath=false",
    "--enable-vecscope-mem-bar=true",
)
# Tile operations can leave out-of-line functions before kernel_entry in the
# caller object. Separate their sections so the linker can put the entry first;
# the persistent loader starts at byte zero of the final .text payload.
CALLER_FLAGS = ("-ffunction-sections",)
LINK_SYMBOL_ORDER = ("kernel_entry",)
DEVICE_FLAGS = (
    "--cce-aicore-arch=dav-c310-vec",
    "--cce-aicore-only",
    "-O2",
    "--cce-generic-addrspace=off",
    "-cce-bitcode-is-aicore",
    "-Wno-override-module",
    "-dc",
    "--cce-long-scbz=true",
    "-mllvm",
    "-cce-dyn-kernel-stack-size=true",
)


def prepare_llvm_for_bisheng(text: str) -> str:
    """Drop newer memory-effect annotations, as PTOAS object emission does.

    LLVM text export retains LLVM 19's memory(...) attribute, which CANN
    Bisheng cannot parse. Removing an optimization annotation is conservative;
    preserve instructions, other attributes and quoted attribute values.
    """

    empty_groups: set[str] = set()

    def clean_group(group: re.Match[str]) -> str:
        body = re.sub(
            r'"(?:[^"\\]|\\.)*"|\bmemory\([^()\n]*\) *',
            lambda token: token[0] if token[0].startswith('"') else "",
            group["body"],
        )
        if not body.strip():
            empty_groups.add(group["number"])
            return ""
        return f"attributes #{group['number']} = {{{body}}}"

    text = re.sub(
        r"^attributes #(?P<number>[0-9]+) = \{(?P<body>[^\n]*)\}$",
        clean_group,
        text,
        flags=re.MULTILINE,
    )
    # LLVM rejects empty groups. Remove references only after collecting every
    # empty group, since declarations and call sites usually precede definitions.
    # Quoted names/strings and comments are not attribute-group references.
    if empty_groups:
        text = re.sub(
            r'"(?:[^"\\]|\\.)*"|;[^\n]*|#(?P<number>[0-9]+)\b',
            lambda token: "" if token["number"] in empty_groups else token[0],
            text,
        )
    return text


def source_paths(value: object) -> tuple[str, ...]:
    """Validate config metadata without requiring the compiler or native SDK."""
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("vpto_sources must be a list of .pto source paths")
    paths: list[str] = []
    for path in cast(Sequence[object], value):
        if not isinstance(path, (str, Path)) or Path(path).suffix != ".pto":
            raise ValueError(f"vpto_sources requires .pto files, got {path!r}")
        paths.append(str(path))
    if len(paths) != len(set(paths)):
        raise ValueError("vpto_sources contains duplicate paths")
    return tuple(paths)


def validate_target(platform: str, core_type: str) -> None:
    """Reject CPU simulation and other device targets before cache lookup."""
    if platform != "a5" or core_type != "aiv":
        raise ValueError(
            f"VPTO helpers require platform='a5' and core_type='aiv', got {platform!r}/{core_type!r}. "
            "The a5sim platform is CPU simulation and cannot execute VPTO device objects."
        )


def validate_device_image(data: bytes) -> None:
    """Reject linked images whose dependencies would be lost by .text extraction.

    The persistent loader starts at byte zero and retains no ELF sections or
    relocation metadata. This structural check does not prove arbitrary code
    position-independent; helper authors must still obey the device-call ABI.
    """
    _device_image.validate_device_image(data)
