# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Build contract for separately compiled A5 VPTO helpers."""

import re
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from . import _device_image

PTOAS_FLAGS = (
    "--pto-arch=a5",
    "--pto-backend=vpto",
    "--vmi-fastmath=false",
)
# Tile operations can leave out-of-line functions before kernel_entry in the
# caller object. Separate their sections so the linker can put the entry first;
# the persistent loader starts at byte zero of the final .text payload.
CALLER_FLAGS = ("-ffunction-sections",)
LINK_SYMBOL_ORDER = ("kernel_entry",)

LLVM_IR_FLAG = "--emit-vpto-llvm-ir"
DEVICE_OBJECT_FLAG = "--emit-vpto-device-object"
# Inline-assembly SIMT entries also require PTOAS's launch-size check and
# deterministic scheduling. Ordinary register helpers use PTOAS's defaults.
SIMT_OBJECT_FLAGS = (
    "--vpto-fix-vfsimt-size=auto",
    "--bisheng-vec-misched=on",
)

# Intrinsic-only SIMT helpers still compile their exported LLVM with Bisheng.
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

REG_AUTO_SYNC_PREFIX = "// pypto.reg_auto_sync:"


def device_object_aliases(data: bytes) -> tuple[str, ...]:
    """Expose logical names for native AIV functions emitted with a CANN ABI suffix."""
    exports = _device_image.device_object_exports(data)
    variants: dict[str, list[str]] = {}
    for symbol in sorted(exports):
        for suffix in (".vector", "_mix_aiv"):
            if symbol.endswith(suffix):
                variants.setdefault(symbol.removesuffix(suffix), []).append(symbol)
                break
    # An existing logical symbol takes precedence. Leave ambiguous references
    # for the final device link, which also sees the caller and other helpers.
    return tuple(
        f"--defsym={name}={symbols[0]}"
        for name, symbols in variants.items()
        if name and name not in exports and len(symbols) == 1
    )


def snapshot_sources(sources: tuple[str, ...]) -> tuple[str, ...]:
    """Capture helper contents for one cache lookup or compilation."""
    return tuple(Path(source).read_text() for source in sources)


def ptoas_flags_for_source(source_text: str, *, emit_llvm: bool = False) -> tuple[str, ...]:
    """Select the helper's synchronization policy and output mode.

    Reserved markers must be exact standalone comments in the file header.
    Reject misplaced or malformed markers rather than silently compiling a
    manual helper with a different synchronization policy.
    """
    markers: dict[str, str] = {}
    allowed = {
        "pypto.reg_auto_sync": {"true", "false"},
        "pypto.vpto_scheduler": {"on", "off"},
        "pypto.vpto_link": {"inline"},
    }
    in_header = True
    for line in source_text.splitlines():
        for key, values in allowed.items():
            if key not in line:
                continue
            prefix = f"// {key}: "
            value = line.removeprefix(prefix)
            if not in_header or not line.startswith(prefix) or value not in values:
                raise ValueError(f"Invalid {key} header marker")
            if key in markers:
                raise ValueError(f"Duplicate {key} header markers")
            markers[key] = value
        if line.strip() and not line.lstrip().startswith("//"):
            in_header = False
    # Match the actual PTO entry attribute, not user function names, comments
    # or string attributes that happen to mention the same spelling.
    code_tokens = re.sub(
        r'"(?:[^"\\]|\\.)*"|//[^\n]*|/\*.*?\*/',
        lambda token: "pto.simt_entry" if token[0] == '"pto.simt_entry"' else " ",
        source_text,
        flags=re.DOTALL,
    )
    simt = re.search(r"[{,]\s*pto\.simt_entry\s*(?=[},=])", code_tokens)
    if "pypto.reg_auto_sync" in markers and simt:
        raise ValueError("pypto.reg_auto_sync is not valid for SIMT helpers")
    if "pypto.vpto_scheduler" in markers and simt:
        raise ValueError("pypto.vpto_scheduler is not valid for SIMT helpers")
    if "pypto.vpto_link" in markers and simt:
        raise ValueError("pypto.vpto_link is not valid for SIMT helpers")
    scheduler = (
        (f"--vpto-scheduler={markers['pypto.vpto_scheduler']}",) if "pypto.vpto_scheduler" in markers else ()
    )
    enabled = markers.get("pypto.reg_auto_sync", "true") == "true"
    mode = LLVM_IR_FLAG if simt or emit_llvm else DEVICE_OBJECT_FLAG
    return (*PTOAS_FLAGS, mode, *scheduler, f"--enable-vecscope-mem-bar={str(enabled).lower()}")


def inline_requested(source_text: str) -> bool:
    """The caller validates reserved markers with ptoas_flags_for_source first."""
    return "// pypto.vpto_link: inline" in source_text.splitlines()


def partition_sources(
    sources: tuple[str, ...], snapshot: tuple[str, ...] | None
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    """Validate source markers before selecting bitcode import or native device linking."""
    texts = snapshot if snapshot is not None else snapshot_sources(sources)
    inline: list[str] = []
    separate: list[tuple[str, str]] = []
    for path, text in zip(sources, texts, strict=True):
        ptoas_flags_for_source(text)
        if inline_requested(text):
            inline.append(text)
        else:
            separate.append((path, text))
    return tuple(inline), tuple(separate)


def inline_register_ir(text: str) -> str:
    """Request inlining on the register functions generated by PTOAS, keeping their logical names."""
    text = prepare_llvm_for_bisheng(text)
    text, count = re.subn(r"^(define void @[^\n]+) \{$", r"\1 alwaysinline {", text, flags=re.MULTILINE)
    if not count:
        raise ValueError("Inline register module has no PTOAS-generated void function definition")
    return text


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
