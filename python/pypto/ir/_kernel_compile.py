# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Internal kernel artifact producer; public compile() remains a program entry."""

import json
from pathlib import Path
from typing import TYPE_CHECKING

from pypto._kernel_abi import KernelABI
from pypto.pypto_core import codegen
from pypto.pypto_core.ir import Function, FunctionType, Program, level_to_linqu_level

from .compiled_program import _extract_func_param_infos, write_kernel_metadata
from .param_info import kernel_abi_from_params

if TYPE_CHECKING:
    from pypto.runtime._kernel_artifact import KernelArtifact


def _entry(program: Program) -> Function:
    """Select one chip entry; distributed and ambiguous entry sets are unsupported."""
    functions = list(program.functions.values())
    if any(f.level is not None and level_to_linqu_level(f.level) >= 3 for f in functions):
        raise ValueError("Kernel artifacts require a single chip entry, not a distributed program")
    entries = [f for f in functions if f.func_type == FunctionType.Orchestration]
    if not entries and len(functions) == 1:
        entries = functions
    if len(entries) != 1:
        raise ValueError("Kernel artifacts require exactly one orchestration entry")
    return entries[0]


def kernel_abi_for_program(program: Program, *, platform: str, runtime: str) -> KernelABI:
    """Derive logical pools and external return aliases before lowering rewrites IR."""
    entry = _entry(program)
    params, _, returns = _extract_func_param_infos(entry)
    aliases = codegen._returned_param_indices(entry, program)
    if len(aliases) != len(returns) or any(index is None for index in aliases):
        raise ValueError(
            "Kernel returns must alias external tensor parameters; pass every Out/InOut explicitly"
        )
    return kernel_abi_from_params(
        params, platform=platform, runtime=runtime, return_aliases=[i for i in aliases if i is not None]
    )


def finish_kernel_artifact(
    original: Program, lowered: Program, directory: Path, abi: KernelABI
) -> "KernelArtifact":
    """Stamp freshly generated code only after its chip signature matches the contract."""
    from pypto.runtime._artifact_sources import read_kernel_config  # noqa: PLC0415
    from pypto.runtime._kernel_artifact import KernelArtifact, validate_kernel_config  # noqa: PLC0415

    kernel_abi_for_program(original, platform=abi.platform, runtime=abi.runtime).require_compatible(abi)
    # A lowered entry may not introduce or reorder the externally supplied pools.
    params, _, _ = _extract_func_param_infos(_entry(lowered))
    actual = kernel_abi_from_params(
        params, platform=abi.platform, runtime=abi.runtime, return_aliases=abi.return_aliases
    )
    expected = [(p.dtype, p.direction, p.shape) for p in abi.parameters]
    if [(p.dtype, p.direction, p.shape) for p in actual.parameters] != expected:
        raise ValueError("Lowering changed the kernel entry parameter ABI")
    config = read_kernel_config(directory / "kernel_config.py")
    validate_kernel_config(config, abi)
    source = Path(config.ORCHESTRATION["source"])
    if directory.resolve() not in source.resolve().parents:
        raise ValueError("Kernel orchestration must be generated inside its private artifact directory")
    # Simpler converts the wire ChipStorageTaskArgs to ChipTaskArgs before entry.
    # Both runtime modes use this function type; keep a distinct binary contract.
    tag = json.dumps(abi.binary_tag().decode("ascii").rstrip("\0"))
    with source.open("a") as stream:
        stream.write(
            "\n#include <type_traits>\n"
            "static_assert(std::is_same_v<decltype(&aicpu_orchestration_entry), "
            "void (*)(const ChipTaskArgs&)>);\n"
            'extern "C" __attribute__((visibility("default")))\n'
            f"const char* pypto_kernel_abi_descriptor() {{ return {tag}; }}\n"
        )
    logical_params, _, _ = _extract_func_param_infos(_entry(original))
    write_kernel_metadata(directory, logical_params, abi)
    return KernelArtifact(directory, abi)
