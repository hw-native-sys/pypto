# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Distributed compiled program wrapper for L3+ programs.

Provides a callable API similar to :class:`CompiledProgram` but executes
through simpler's distributed runtime (Worker level=3)::

    compiled = ir.compile(MyDistributedProgram)
    compiled(a, b, c)   # executes via simpler Worker(level=3)
"""

import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from pypto.backend import BackendType
from pypto.pypto_core.ir import ParamDirection, Program, Role, level_to_linqu_level
from pypto.runtime.device_tensor import DeviceTensor

from .compiled_program import (
    CallArg,
    _default_platform,
    _extract_param_infos,
    _param_info_from_dict,
    _param_info_to_dict,
    _ParamInfo,
    _to_torch_dtype,
    _validate_device_tensor,
)

# Filename of the small JSON sidecar persisted alongside the build artifacts so
# the program can be reconstructed (``from_dir``) without the live post-pass IR.
# Bump ``_META_SCHEMA`` on any incompatible format change.
_DISTRIBUTED_META_FILENAME = "distributed_meta.json"
_META_SCHEMA = 1

if TYPE_CHECKING:
    from pypto.runtime.distributed_runner import DistributedWorker
    from pypto.runtime.runner import RunConfig


def _extract_param_infos_from_func(func):
    """Extract parameter metadata from a specific function."""
    from pypto.pypto_core.ir import ConstInt, ParamDirection, ScalarType, ShapedType  # noqa: PLC0415

    param_infos = []
    output_indices = []

    for i, (param, direction) in enumerate(zip(func.params, func.param_directions, strict=True)):
        param_type = param.type
        shape = None

        if isinstance(param_type, ShapedType):
            dtype = param_type.dtype
            shape = [dim.value if isinstance(dim, ConstInt) else -1 for dim in param_type.shape]
        elif isinstance(param_type, ScalarType):
            dtype = param_type.dtype
        else:
            raise TypeError(
                f"Unsupported parameter type for {param.name_hint!r}: {type(param_type).__name__}"
            )

        param_infos.append(_ParamInfo(name=param.name_hint, direction=direction, shape=shape, dtype=dtype))
        if direction == ParamDirection.Out:
            output_indices.append(i)

    return param_infos, output_indices, list(func.return_types)


@dataclass
class DistributedConfig:
    """Configuration for L3 distributed execution.

    ``aicpu_thread_num=4`` matches the ``tensormap_and_ringbuffer`` runtime's
    3-scheduler-plus-1-dispatcher layout; ``block_dim=None`` lets the L2
    simpler runtime pick its own default.
    """

    device_ids: list[int] = field(default_factory=lambda: [0])
    num_sub_workers: int = 0
    runtime: str = "tensormap_and_ringbuffer"
    block_dim: int | None = None
    aicpu_thread_num: int = 4


class DistributedCompiledProgram:
    """A compiled L3+ distributed program that executes via simpler Worker(level=3).

    Returned by :func:`ir.compile` when the program contains HOST-level
    or higher hierarchy functions.

    Calling conventions match :class:`CompiledProgram`:

    **In-place** (output passed as argument)::

        compiled(a, b, c)

    **Return** (program has a return value)::

        c = compiled(a, b)
    """

    __test__ = False

    def __init__(
        self,
        program: Program | None,
        output_dir: str,
        *,
        backend_type: BackendType = BackendType.Ascend910B,
        platform: str | None = None,
        distributed_config: DistributedConfig | None = None,
        _param_infos: list[_ParamInfo] | None = None,
        _output_indices: list[int] | None = None,
        _return_types: list[Any] | None = None,
    ) -> None:
        # ``program`` is the post-pass IR. The runtime needs post-pass IR for
        # orchestrator metadata (post-SSA names that match the generated
        # host_orch.py) and to iterate Orchestration functions synthesized by
        # passes such as OutlineHierarchyScopes.
        #
        # ``program`` is ``None`` on the :meth:`from_dir` reload path: param
        # metadata is supplied pre-derived via the ``_param_infos`` /
        # ``_output_indices`` / ``_return_types`` kwargs (read back from
        # ``distributed_meta.json``), and chip-callable assembly is driven by
        # the on-disk ``next_levels/`` layout — so no live IR is needed.
        self._program = program
        self._output_dir = Path(output_dir).resolve()
        self._backend_type = backend_type
        self._platform = platform or _default_platform(backend_type)
        self._distributed_config = distributed_config or DistributedConfig()
        self._param_infos = _param_infos
        self._output_indices = _output_indices
        self._return_types = _return_types

        # Only the fresh-compile path (live IR) writes artifacts. The reload
        # path must not clobber a user's hand-edited debug/run.py or the
        # already-present metadata file.
        if program is not None:
            self._persist_metadata()
            self._emit_debug_runner()

    def _emit_debug_runner(self) -> None:
        """Write ``<output_dir>/debug/run.py`` for replaying this program.

        Best-effort: distributed programs without a clean orchestration entry
        will skip emission (the replay CLI is still usable directly).

        Disable globally by setting ``PYPTO_EMIT_DEBUG_RUNNER=0`` (also accepts
        ``false`` / ``no``).
        """
        if os.environ.get("PYPTO_EMIT_DEBUG_RUNNER", "").strip().lower() in ("0", "false", "no"):
            return

        from pypto.runtime.debug.run_script_writer import write_run_script  # noqa: PLC0415

        try:
            param_infos, _, _ = self._get_metadata()
        except (ValueError, TypeError):
            return
        write_run_script(self._output_dir, param_infos, platform=self._platform)

    def _persist_metadata(self) -> None:
        """Write ``<output_dir>/distributed_meta.json`` for :meth:`from_dir`.

        Captures exactly what :func:`execute_distributed` reads from the
        post-pass IR — the HOST-orchestrator param metadata (post-SSA names,
        directions, shapes, dtypes) plus the return-type count — alongside the
        platform / backend / :class:`DistributedConfig` so a later reload can
        reconstruct a fully functional program without re-running the pass
        chain. Chip-callable assembly is rederived from the ``next_levels/``
        layout and needs no persistence.

        Best-effort: a program without a resolvable orchestrator signature
        skips emission (mirrors :meth:`_emit_debug_runner`); :meth:`from_dir`
        then reports the missing file with a recompile hint.
        """
        try:
            param_infos, _, return_types = self._get_metadata()
        except (ValueError, TypeError):
            return
        dc = self._distributed_config
        meta = {
            "schema": _META_SCHEMA,
            "params": [_param_info_to_dict(p) for p in param_infos],
            "num_return_types": len(return_types),
            "platform": self._platform,
            "backend_type": self._backend_type.name,
            "distributed_config": {
                "device_ids": list(dc.device_ids),
                "num_sub_workers": dc.num_sub_workers,
                "runtime": dc.runtime,
                "block_dim": dc.block_dim,
                "aicpu_thread_num": dc.aicpu_thread_num,
            },
        }
        (self._output_dir / _DISTRIBUTED_META_FILENAME).write_text(json.dumps(meta, indent=2))

    @classmethod
    def from_dir(
        cls,
        output_dir: str | os.PathLike[str],
        *,
        platform: str | None = None,
        backend_type: BackendType | None = None,
        distributed_config: DistributedConfig | None = None,
    ) -> "DistributedCompiledProgram":
        """Reconstruct a distributed program from an existing ``build_output/`` dir.

        Rebuilds metadata from ``distributed_meta.json`` (written at compile
        time) so the program is callable / dispatchable **without** re-running
        the pypto compile — the basis of the ``runtime_dir`` replay workflow for
        L3 programs. Chip callables are assembled from the on-disk
        ``next_levels/`` layout at dispatch time.

        Args:
            output_dir: A build directory produced by a prior ``ir.compile`` of
                a distributed (L3+) program. Must contain
                ``distributed_meta.json``.
            platform: Override the persisted platform (e.g. swap ``a2a3sim`` →
                ``a2a3`` to replay on hardware). ``None`` keeps the persisted
                value.
            backend_type: Override the persisted codegen backend. ``None`` keeps
                the persisted value.
            distributed_config: Override the persisted run config (e.g. to
                replay on a different set of ``device_ids``). ``None`` rebuilds
                the :class:`DistributedConfig` recorded at compile time.

        Returns:
            A :class:`DistributedCompiledProgram` whose ``__call__`` / ``prepare``
            behave exactly like the freshly-compiled object.

        Raises:
            FileNotFoundError: ``distributed_meta.json`` is absent (the directory
                predates this feature or is not a distributed build).
            ValueError: ``distributed_meta.json`` records a ``schema`` version
                incompatible with this pypto build (the metadata format changed).
        """
        meta_path = Path(output_dir).resolve() / _DISTRIBUTED_META_FILENAME
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} not found — cannot reconstruct a distributed program from this "
                f"directory. It predates the L3 replay feature or is not a distributed (L3+) "
                f"build. Recompile via ir.compile() to refresh."
            )
        meta = json.loads(meta_path.read_text())
        schema = meta.get("schema")
        if schema != _META_SCHEMA:
            raise ValueError(
                f"Incompatible {_DISTRIBUTED_META_FILENAME} schema {schema!r} (expected "
                f"{_META_SCHEMA}) in {meta_path}. The metadata was written by a different "
                f"pypto version — recompile via ir.compile() to refresh."
            )
        param_infos = [_param_info_from_dict(p) for p in meta["params"]]
        output_indices = [i for i, p in enumerate(param_infos) if p.direction == ParamDirection.Out]
        # ``return_types`` contents are never inspected at runtime — only the
        # count matters (has_return = len(...) > 0), so placeholders suffice.
        return_types: list[Any] = [None] * int(meta.get("num_return_types", 0))
        dc = distributed_config or DistributedConfig(**meta.get("distributed_config", {}))
        bt = backend_type or getattr(BackendType, meta.get("backend_type", "Ascend910B"))
        return cls(
            None,
            str(output_dir),
            backend_type=bt,
            platform=platform or meta.get("platform"),
            distributed_config=dc,
            _param_infos=param_infos,
            _output_indices=output_indices,
            _return_types=return_types,
        )

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    @property
    def program(self) -> Program | None:
        """Post-pass IR, or ``None`` for a program reconstructed via :meth:`from_dir`."""
        return self._program

    @property
    def platform(self) -> str:
        return self._platform

    def __str__(self) -> str:
        return str(self._output_dir)

    def __repr__(self) -> str:
        return f"DistributedCompiledProgram({self._output_dir!s})"

    def __fspath__(self) -> str:
        return str(self._output_dir)

    def _get_metadata(self) -> tuple[list[_ParamInfo], list[int], list[Any]]:
        if self._param_infos is None:
            if self._program is None:
                # Reload path with no pre-filled metadata — should not happen
                # (``from_dir`` always supplies it); guard rather than deref None.
                raise RuntimeError(
                    "DistributedCompiledProgram has neither live IR nor persisted param "
                    "metadata; reconstruct via DistributedCompiledProgram.from_dir()."
                )
            # Find the HOST orchestrator function (post-SSA names match the
            # generated Python code).
            host_orch = None
            for func in self._program.functions.values():
                if (
                    func.level is not None
                    and level_to_linqu_level(func.level) >= 3
                    and func.role is not None
                    and func.role == Role.Orchestrator
                ):
                    host_orch = func
                    break

            if host_orch is not None:
                self._param_infos, self._output_indices, self._return_types = _extract_param_infos_from_func(
                    host_orch
                )
            else:
                self._param_infos, self._output_indices, self._return_types = _extract_param_infos(
                    self._program
                )
        assert self._output_indices is not None and self._return_types is not None
        return self._param_infos, self._output_indices, self._return_types

    def __call__(
        self,
        *args: CallArg,
        config: "RunConfig | None" = None,
    ) -> torch.Tensor | DeviceTensor | tuple[torch.Tensor | DeviceTensor, ...] | None:
        """Execute the distributed program via simpler Worker(level=3).

        ``config`` is an optional per-dispatch :class:`RunConfig`; its per-task
        ring-sizing overrides (``ring_task_window`` / ``ring_heap`` /
        ``ring_dep_pool``) size this dispatch's runtime ring buffers, and its
        runtime-diagnostic DFX flags (``enable_dump_tensor`` / ``enable_pmu`` /
        ``enable_dep_gen`` / ``enable_scope_stats`` / ``enable_l2_swimlane``) are
        written per dispatch under ``<output_dir>/dfx_outputs/rank{r}/d{k}/``
        (``d{k}`` is the card's k-th dispatch, so multiple dispatches to one card
        keep separate artifacts; swimlane co-enables dep_gen and emits
        ``merged_swimlane_*.json`` per dispatch, onboard only). Other compile-side
        fields are not consumed on the dispatch path.
        """
        from pypto.runtime.distributed_runner import execute_distributed  # noqa: PLC0415

        param_infos, output_indices, return_types = self._get_metadata()
        n_params = len(param_infos)
        n_inputs = n_params - len(output_indices)
        has_return = len(return_types) > 0
        return_style = has_return and len(args) == n_inputs

        if len(args) == n_params:
            all_args: list[CallArg] = list(args)
        elif return_style:
            all_args = self._build_full_args(args, param_infos, output_indices)
        else:
            expected = f"{n_params} (in-place)"
            if has_return:
                expected += f" or {n_inputs} (return)"
            raise TypeError(
                f"DistributedCompiledProgram expects {expected} arguments, got {len(args)}. "
                f"Parameters: {[p.name for p in param_infos]}"
            )

        # Validate and coerce args. Tensor params accept a host ``torch.Tensor``
        # or a worker-resident ``DeviceTensor`` (skips H2D/D2H), matching the L2
        # ``CompiledProgram`` calling convention.
        coerced: list[torch.Tensor | DeviceTensor] = []
        for info, arg in zip(param_infos, all_args, strict=True):
            if isinstance(arg, DeviceTensor):
                _validate_device_tensor(arg, info)
                coerced.append(arg)
                continue
            if not isinstance(arg, torch.Tensor):
                raise TypeError(
                    f"Distributed programs only support tensor parameters "
                    f"(torch.Tensor host or DeviceTensor worker-resident). "
                    f"Parameter {info.name!r} got {type(arg).__name__}"
                )
            coerced.append(arg)

        execute_distributed(self, coerced, config)

        if not return_style:
            return None
        outputs = [coerced[i] for i in output_indices]
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def prepare(
        self,
        config: Any = None,
        *,
        extra_compiled: Sequence["DistributedCompiledProgram"] = (),
        callbacks: dict[str, Callable[..., Any]] | None = None,
        sub_worker_overrides: dict[str, Callable[..., Any]] | None = None,
    ) -> "DistributedWorker":
        """Prepare a reusable L3 execution handle (setup once, dispatch many).

        Runs the expensive setup (``compile_and_assemble``, generated-module
        loading, simpler ``Worker(level=3)`` construction + registration +
        ``init()``) exactly once and returns a :class:`DistributedWorker` that
        dispatches many times on the held worker. The handle also exposes
        device-memory helpers (``alloc_tensor`` / ``malloc`` / ``copy_to`` /
        ``copy_from`` / ``free``) for building worker-resident
        :class:`DeviceTensor` buffers that survive across dispatches.

        Per-call inputs and outputs are reused-in-place **shared-memory** host
        ``torch.Tensor`` buffers (allocated before ``prepare()``) and/or
        worker-resident ``DeviceTensor`` / simpler ``Tensor`` arguments.
        Non-shared host tensors are rejected (the forked chip worker cannot see
        a buffer allocated after the fork). The convenience host-to-device
        upload of arbitrary host ``torch.Tensor`` inputs is only available on
        the one-shot ``compile(...)(*args)`` / ``execute_distributed`` path.

        Args:
            config: Optional run configuration (reserved; currently unused).
            extra_compiled: Additional compatible programs to prepare on the
                same L3 worker (multi-program serving — e.g. ``prefill`` prepares
                the worker and passes ``[decode]`` here so both share the worker
                and its device-resident KV cache). ``self`` is the *primary*
                program dispatched by ``rt(*args)``; the rest are dispatched via
                ``rt.run(other, *args)``. All must agree on platform, runtime,
                and device ids. Defaults to none (single-program).
            callbacks: Bind a callable to a SubWorker by name — e.g. a real
                sampling closure. Abstract SubWorkers (declared with a ``...``
                body) are runtime-bound callback points and MUST be supplied
                here; a missing binding raises ``ValueError``. A callback may
                also replace a concrete SubWorker's generated body. Each name
                must be a sub-worker the program declares; an unknown name
                raises ``ValueError``. In multi-program mode the callbacks apply
                to every prepared program.
            sub_worker_overrides: Deprecated alias for ``callbacks``.

        Returns:
            A :class:`DistributedWorker`; use it as a context manager or call
            ``close()`` when done.
        """
        from pypto.runtime.distributed_runner import DistributedWorker  # noqa: PLC0415

        return DistributedWorker(
            [self, *extra_compiled],
            config,
            callbacks=callbacks,
            sub_worker_overrides=sub_worker_overrides,
        )

    @staticmethod
    def _build_full_args(input_args, param_infos, output_indices):
        output_set = set(output_indices)
        all_tensors = []
        input_idx = 0

        for i, info in enumerate(param_infos):
            if i in output_set:
                if info.shape is None:
                    raise ValueError(f"Cannot allocate output tensor {info.name!r}: no shape in IR")
                if any(d < 0 for d in info.shape):
                    raise ValueError(
                        f"Cannot allocate output tensor {info.name!r}: shape {info.shape} "
                        f"contains dynamic dimensions."
                    )
                torch_dtype = _to_torch_dtype(info.dtype)
                if torch_dtype is None:
                    raise ValueError(f"Unsupported dtype {info.dtype} for output tensor {info.name!r}")
                all_tensors.append(torch.zeros(info.shape, dtype=torch_dtype))
            else:
                all_tensors.append(input_args[input_idx])
                input_idx += 1

        return all_tensors
