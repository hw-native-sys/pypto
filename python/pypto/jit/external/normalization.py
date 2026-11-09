# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Plan internal SSA results independently of the provider's Python returns.

The JIT applies this plan before parsing generated call sites.
The plan is consumed before passes reorder returns; consumers must materialize
projections in IR rather than retain these slot numbers after transformation.
"""

from dataclasses import dataclass

from pypto.pypto_core.ir import ParamDirection

from .contracts import Access, Coverage, KernelContract, ScalarSpec
from .validation import validate_contract


@dataclass(frozen=True)
class CallPlan:
    parameter_directions: tuple[ParamDirection, ...]
    result_parameters: tuple[int, ...]
    public_slots: tuple[int, ...]
    writeback_slots: tuple[tuple[int, int], ...]  # (parameter, result slot)


def plan_call(contract: KernelContract) -> CallPlan:
    """Keep each written parameter once, then add any readonly public aliases."""
    validate_contract(contract)
    effects = {effect.parameter: effect for effect in contract.effects}
    directions: list[ParamDirection] = []
    for i, parameter in enumerate(contract.parameters):
        if isinstance(parameter.type, ScalarSpec):
            directions.append(ParamDirection.In)
            continue
        effect = effects[i]
        if effect.access in (Access.NONE, Access.READ):
            directions.append(ParamDirection.In)
        elif effect.access == Access.WRITE and effect.coverage == Coverage.FULL:
            directions.append(ParamDirection.Out)
        else:
            directions.append(ParamDirection.InOut)
    return plan_results(tuple(directions), tuple(binding.parameter for binding in contract.public_returns))


def plan_results(directions: tuple[ParamDirection, ...], public_parameters: tuple[int, ...]) -> CallPlan:
    """Share result ordering with legacy extern declarations after type validation."""
    written = [i for i, direction in enumerate(directions) if direction != ParamDirection.In]
    results = list(written)
    slots = {parameter: slot for slot, parameter in enumerate(results)}
    for parameter in public_parameters:
        if parameter not in slots:
            slots[parameter] = len(results)
            results.append(parameter)
    return CallPlan(
        tuple(directions),
        tuple(results),
        tuple(slots[parameter] for parameter in public_parameters),
        tuple((parameter, slots[parameter]) for parameter in written),
    )
