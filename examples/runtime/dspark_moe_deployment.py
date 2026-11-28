# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run DSpark MoE validation with caller-supplied A3 placement.

Build the DSpark model with `--compile-only`. Pass reserved local card IDs and
peer groups such as `--remote-device "192.0.2.10:9000 4,5"` before `--`.
Pass the artifact directory and matching model shape options after `--`.
"""

import argparse
import importlib
import re
import sys
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pypto.runtime import DistributedDeployment


def _load_model(lib_root: Path, world_size: int, model_args: list[str]):
    lib_root = lib_root.resolve()
    model_dir = lib_root / "models" / "deepseek_v4_flash_dspark"
    model_script = model_dir / "moe.py"
    if not model_script.is_file():
        raise ValueError(f"DSpark MoE entry not found: {model_script}")
    if any(arg == "--ep" or arg.startswith("--ep=") for arg in model_args):
        raise ValueError("EP is determined by the deployment")
    # The model specializes its tensor shapes as its module is loaded.
    sys.argv = [str(model_script), "--ep", str(world_size), *model_args]
    sys.path[:0] = [str(model_dir), str(lib_root)]
    return importlib.import_module("moe")


def _signal_slots(layouts):
    # The compiled layout may version allocation names during SSA conversion.
    slots = []
    for name in ("arrived_buf", "data_arrived_buf", "combine_arrived_buf"):
        pattern = re.compile(re.escape(name) + r"(?:__ssa_v[0-9]+)?")
        matches = [
            (layout, buffer)
            for layout in layouts
            for buffer in layout.buffers
            if pattern.fullmatch(buffer.name)
        ]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one compiled signal slot for {name!r}")
        slots.append(matches[0])
    return slots


def run_moe_replay(
    lib_root: Path,
    deployment: "DistributedDeployment",
    model_args: list[str],
    *,
    check_artifact: bool = False,
) -> None:
    """Replay one DSpark request on fresh communication and weight domains."""
    import torch  # noqa: PLC0415
    from pypto.ir import DistributedCompiledProgram  # noqa: PLC0415
    from pypto.runtime import RunConfig  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=run_moe_replay.__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--experts-per-rank", type=int)
    parser.add_argument("--token-capacity", type=int)
    parser.add_argument("--num-tokens", type=int)
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--routing", choices=("uniform", "balanced", "real"), default="uniform")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(model_args)

    moe = _load_model(lib_root, deployment.world_size, model_args)
    # These helpers also serve the standalone lib runner on main.
    from golden import TensorSpec, ratio_reldiff, validate_golden  # noqa: PLC0415
    from golden.runner import _compute_golden, _validate_compiled_spec_abi  # noqa: PLC0415

    num_tokens = moe.T if args.num_tokens is None else args.num_tokens
    if not 0 <= num_tokens <= moe.MOE_TOKEN_CAPACITY:
        parser.error("--num-tokens must be within --token-capacity")
    specs = moe.build_rounds_tensor_specs(
        layer_id=args.layer_id,
        num_tokens=num_tokens,
        routing=args.routing,
    )
    artifact = args.runtime_dir.resolve()
    compiled = DistributedCompiledProgram.from_dir(artifact)
    _validate_compiled_spec_abi(compiled, specs)
    tensor_specs = [spec for spec in specs if isinstance(spec, TensorSpec)]
    scalars = {spec.name: spec for spec in specs if not isinstance(spec, TensorSpec)}
    scalar_values = {name: spec.to_python() for name, spec in scalars.items()}
    layouts = compiled.domain_layouts(world_size=deployment.world_size, **scalar_values)
    signal_slots = _signal_slots(layouts)
    residents = [spec for spec in tensor_specs if spec.is_resident]
    for spec in residents:
        if spec.resident != "stacked" or spec.is_output or spec.shape[0] != deployment.world_size:
            raise ValueError(f"expected a rank-stacked input for resident parameter {spec.name!r}")
    if check_artifact:
        print(
            f"DSpark EP{deployment.world_size}: ABI and {len(signal_slots)} signal slots verified",
            flush=True,
        )
        return

    from simpler.task_interface import CommBufferSpec  # noqa: PLC0415

    torch.manual_seed(args.seed)
    tensors = {spec.name: spec.create_tensor() for spec in tensor_specs}
    resident_names = {spec.name for spec in residents}
    for name, tensor in tensors.items():
        if name not in resident_names:
            tensor.share_memory_()
    buffers = []
    for spec in residents:
        nbytes = prod(spec.shape[1:]) * tensors[spec.name].element_size()
        buffers.append(CommBufferSpec(spec.name, "opaque", nbytes, (nbytes + 63) // 64 * 64))

    config = RunConfig(ring_heap=moe.MOE_RING_HEAP)
    compare = {"x_next": ratio_reldiff(diff_thd=3e-3, pct_thd=0.05)}
    golden = _compute_golden(
        specs,
        tensor_specs,
        scalars,
        tensors,
        artifact,
        None,
        moe.golden_moe_rounds,
        save_data=False,
    )
    outputs = {spec.name: tensors[spec.name] for spec in tensor_specs if spec.is_output}
    for output in outputs.values():
        output.fill_(float("nan"))

    with compiled.prepare(
        config,
        deployment=deployment,
        reset_persistent_windows=False,
    ) as worker:
        comms = worker.prepare_domains(compiled, **scalar_values)
        parameters = worker.allocate_domain(
            "dspark:weights",
            window_size=sum(buffer.nbytes for buffer in buffers),
            buffers=buffers,
        )
        domains = (*comms.values(), parameters)
        with worker.request(*domains):
            # Initialize the notifications for the first dispatch on the fresh windows.
            for layout, buffer in signal_slots:
                zeros = torch.zeros(buffer.logical_nbytes, dtype=torch.uint8)
                for rank in layout.workers:
                    worker.copy_domain_to(comms[layout.name], rank, buffer.name, zeros)
            weights = {
                spec.name: worker.alloc_stacked_tensor(
                    tensors[spec.name],
                    domain=parameters,
                    buffer=spec.name,
                )
                for spec in residents
            }
            bindings = {**tensors, **scalar_values, **weights}
            arguments = [bindings[spec.name] for spec in specs]
            worker.run(compiled, *arguments, config=config)
        if any(not output.isfinite().all().item() for output in outputs.values()):
            raise AssertionError("DSpark HOST output contains unread or non-finite values")
        validate_golden(outputs, golden, rtol=1e-3, atol=1e-3, compare_fn=compare)
    print("[RUN] PASS", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lib-root", type=Path, required=True, help="Matching pypto-lib checkout")
    parser.add_argument(
        "--check-artifact", action="store_true", help="Check replay ABI and layout without devices"
    )
    parser.add_argument(
        "--local-devices", required=True, metavar="ID[,ID...]", help="Reserved local device IDs"
    )
    parser.add_argument(
        "--remote-device",
        action="append",
        default=[],
        metavar="IP:PORT ID[,ID...]",
        help="Peer daemon endpoint and reserved device IDs; repeat for other daemons",
    )
    parser.add_argument("model_args", nargs=argparse.REMAINDER, help="DSpark MoE options after --")
    args = parser.parse_args()

    from pypto.runtime import DistributedDeployment, RemoteChipTarget  # noqa: PLC0415

    model_args = args.model_args
    if model_args[:1] == ["--"]:
        model_args = model_args[1:]
    try:
        local_ids = tuple(int(device) for device in args.local_devices.split(","))
        remote_chips = []
        for target in args.remote_device:
            group = target.split(maxsplit=1)
            if len(group) == 2:
                endpoint, devices = group
                remote_chips.extend(RemoteChipTarget(endpoint, int(device)) for device in devices.split(","))
                continue

            # Keep existing single-card validation commands usable.
            endpoint, separator, device = target.rpartition("/")
            if not separator:
                raise ValueError(f"invalid remote device {target!r}; expected IP:PORT ID[,ID...]")
            remote_chips.append(RemoteChipTarget(endpoint, int(device)))
        deployment = DistributedDeployment(local_device_ids=local_ids, remote_chips=tuple(remote_chips))
        run_moe_replay(args.lib_root, deployment, model_args, check_artifact=args.check_artifact)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
