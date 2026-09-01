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

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from pypto.backend import BackendType
from pypto.pypto_core.ir import ParamDirection, Program, Role, level_to_linqu_level
from pypto.runtime.device_tensor import DeviceTensor, StackedDeviceTensor

from .compiled_program import (
    _DISTRIBUTED_META_FILENAME,
    CallArg,
    _default_platform,
    _drop_foreign_build_markers,
    _extract_func_param_infos,
    _extract_param_infos,
    _load_meta,
    _meta_error,
    _param_info_to_dict,
    _ParamInfo,
    _remove_meta,
    _to_torch_dtype,
    _write_debug_runner,
    _write_meta_atomically,
)

# The sidecar filename itself lives beside its L2 counterpart in
# ``compiled_program`` (both feed the build-kind marker table there) and is
# re-exported here, where the rest of the L3 metadata contract lives. Bump
# ``_META_SCHEMA`` on any incompatible format change.
_META_SCHEMA = 2

if TYPE_CHECKING:
    from pypto.runtime.distributed_runner import DistributedWorker, ReadOnlyHostTensor
    from pypto.runtime.runner import RunConfig


@dataclass
class DistributedConfig:
    """Configuration for L3 distributed execution.

    Attributes:
        device_ids: List of NPU device indices to use. Defaults to ``[0]``
            (single-card). Pass ``[0, 1, 2, 3]`` for a 4-card ring. Must be
            a non-empty list of distinct ints.
        num_sub_workers: Number of sub-workers to request per chip process.
            ``0`` (default) does not force a specific count — the runtime
            instead uses ``max(num_sub_workers, len(sub_worker_fns))``, i.e.
            exactly as many sub-workers as the program's declared sub-worker
            functions. Set this explicitly only when requesting more
            sub-workers than are declared, e.g. for multi-slice or internal
            pipelining scenarios.
        runtime: Simpler runtime flavour. ``"tensormap_and_ringbuffer"`` (default)
            enables the tensor-map helpers and the ring-buffer DMA driver.
        aicpu_thread_num: Number of AICPU threads allocated to the simpler
            runtime. ``0`` (default) selects the architecture default (a2a3: 4;
            a5: 5). Explicit normal-run values must be at least 2 and are
            validated against the selected platform's limit by the runtime.
    """

    device_ids: list[int] = field(default_factory=lambda: [0])
    num_sub_workers: int = 0
    runtime: str = "tensormap_and_ringbuffer"
    aicpu_thread_num: int = 0

    def __post_init__(self) -> None:
        """Enforce the value constraints the fields above document.

        Placed on the dataclass rather than on either caller so the live
        construction path and the ``distributed_meta.json`` reload cannot drift:
        a hand-edited sidecar must not be able to smuggle in a config the API
        itself rejects -- a duplicated device id would put two ranks on one card,
        and an empty list would reach worker startup with nothing to run on.

        Raises:
            ValueError: a field violates its documented constraint.
        """
        if not self.device_ids:
            raise ValueError("DistributedConfig.device_ids must not be empty")
        if any(d < 0 for d in self.device_ids):
            raise ValueError(
                f"DistributedConfig.device_ids must be non-negative device indices, got {self.device_ids}"
            )
        if len(set(self.device_ids)) != len(self.device_ids):
            raise ValueError(f"DistributedConfig.device_ids must be distinct, got {self.device_ids}")
        if self.num_sub_workers < 0:
            raise ValueError(
                f"DistributedConfig.num_sub_workers must be non-negative, got {self.num_sub_workers}"
            )
        # 0 selects the architecture default; 1 is below the runtime's floor.
        if self.aicpu_thread_num < 0 or self.aicpu_thread_num == 1:
            raise ValueError(
                "DistributedConfig.aicpu_thread_num must be 0 (architecture default) or at least 2, "
                f"got {self.aicpu_thread_num}"
            )


def _distributed_config_from_dict(meta: dict[str, Any], meta_path: Path) -> DistributedConfig:
    """Rebuild the :class:`DistributedConfig` a sidecar recorded.

    Validated like every other sidecar field (see
    :func:`~pypto.ir.compiled_program._load_meta`): the block must be an object
    keyed only by ``DistributedConfig`` fields, each correctly typed. A
    hand-edited ``device_ids`` therefore fails here, naming the file, instead of
    reaching ``DistributedConfig(**raw)`` as a raw ``TypeError`` or surfacing
    much later inside worker startup.
    """

    def _bad(detail: str) -> ValueError:
        return _meta_error(_DISTRIBUTED_META_FILENAME, meta_path, detail)

    # ``bool`` is an ``int`` subclass; a JSON ``true`` is never a count or index.
    def _is_int(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    raw = meta.get("distributed_config", {})
    if not isinstance(raw, dict):
        raise _bad(f"'distributed_config' must be an object, got {type(raw).__name__}")
    known = {f.name for f in fields(DistributedConfig)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise _bad(f"'distributed_config' has unknown key(s) {unknown}; known: {sorted(known)}")

    if "device_ids" in raw and not (
        isinstance(raw["device_ids"], list) and all(_is_int(d) for d in raw["device_ids"])
    ):
        raise _bad(f"'distributed_config.device_ids' must be a list of ints, got {raw['device_ids']!r}")
    for key in ("num_sub_workers", "aicpu_thread_num"):
        if key in raw and not _is_int(raw[key]):
            raise _bad(f"'distributed_config.{key}' must be an int, got {raw[key]!r}")
    if "runtime" in raw and not isinstance(raw["runtime"], str):
        raise _bad(f"'distributed_config.runtime' must be a string, got {raw['runtime']!r}")
    # Value constraints (non-empty / distinct device ids, thread-count floors)
    # live on the dataclass, so the reload is held to exactly the same bar as a
    # live DistributedConfig; only the JSON *shape* is checked above.
    try:
        return DistributedConfig(**raw)
    except ValueError as exc:
        raise _bad(f"'distributed_config' is invalid ({exc})") from exc


class DistributedCompiledProgram:
    """A compiled L3+ distributed program that executes via simpler Worker(level=3).

    Returned by :func:`ir.compile` when the program contains HOST-level
    or higher hierarchy functions.

    Calling conventions match :class:`CompiledProgram`:

    **In-place** (output passed as argument)::

        compiled(a, b, c)

    **Return** (program has a return value)::

        c = compiled(a, b)

    **One-shot dispatch**::

        compiled = ir.compile(MyProgram, platform="a2a3", distributed_config=dc)
        compiled(host_inputs, host_outputs)   # blocks until all ranks finish

    **Persistent dispatch** (repeated launches without re-registering)::

        with compiled.prepare() as rt:
            for step in steps:
                rt(host_x, host_out)
        # rt.close() on exit — releases buffers and shuts down workers
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
            # An L3 build writes no top-level ``kernel_config.py`` (per-rank
            # configs live under ``next_levels/``), so one found here is an L2
            # leftover -- and ``replay()`` keys off exactly that file, which
            # would make this build replay as the previous single-chip one.
            _drop_foreign_build_markers(
                self._output_dir, keep=(_DISTRIBUTED_META_FILENAME, "orchestration/host_orch.py")
            )
            self._persist_metadata()
            _write_debug_runner(self._output_dir, self._platform, self._get_metadata)

    def _persist_metadata(self) -> None:
        """Write ``<output_dir>/distributed_meta.json`` for :meth:`from_dir`.

        Captures exactly what :func:`_execute_distributed` reads from the
        post-pass IR — the HOST-orchestrator param metadata (post-SSA names,
        directions, shapes, dtypes) plus the return-type count — alongside the
        platform / backend / :class:`DistributedConfig` so a later reload can
        reconstruct a fully functional program without re-running the pass
        chain. Chip-callable assembly is rederived from the ``next_levels/``
        layout and needs no persistence.

        Best-effort: a program without a resolvable orchestrator signature
        emits nothing (mirrors :func:`_write_debug_runner`) *and* deletes any
        sidecar a previous compile into this directory left behind, so
        :meth:`from_dir` reports the missing file with a recompile hint instead
        of handing out a signature that no longer describes these artifacts --
        ``ir.compile`` never clears ``output_dir``, and the mismatch is one
        :meth:`from_dir` cannot detect.
        """
        try:
            param_infos, _, return_types = self._get_metadata()
        except (ValueError, TypeError):
            _remove_meta(self._output_dir, _DISTRIBUTED_META_FILENAME)
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
                "aicpu_thread_num": dc.aicpu_thread_num,
            },
        }
        _write_meta_atomically(self._output_dir / _DISTRIBUTED_META_FILENAME, meta)

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
            ValueError: ``distributed_meta.json`` is unreadable, malformed, or
                records a ``schema`` version incompatible with this pypto build.
        """
        meta_path = Path(output_dir).resolve() / _DISTRIBUTED_META_FILENAME
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} not found — cannot reconstruct a distributed program from this "
                f"directory. It predates the L3 replay feature or is not a distributed (L3+) "
                f"build. Recompile via ir.compile() to refresh."
            )
        # Shares the L2 loader, so every malformed field -- unreadable file, bad
        # JSON, bad params, a ``backend_type`` naming a class attribute instead
        # of an enum member -- surfaces as one ValueError naming the file and
        # the recompile fix, rather than whichever internal exception it raised.
        meta = _load_meta(meta_path, filename=_DISTRIBUTED_META_FILENAME, schema=_META_SCHEMA)
        param_infos = meta["param_infos"]
        output_indices = [i for i, p in enumerate(param_infos) if p.direction == ParamDirection.Out]
        # ``return_types`` contents are never inspected at runtime — only the
        # count matters (has_return = len(...) > 0), so placeholders suffice.
        return_types: list[Any] = [None] * meta["num_return_types"]
        dc = distributed_config or _distributed_config_from_dict(meta["raw"], meta_path)
        return cls(
            None,
            str(output_dir),
            backend_type=backend_type or meta["backend_type"],
            platform=platform or meta["platform"],
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
                self._param_infos, self._output_indices, self._return_types = _extract_func_param_infos(
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
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | None:
        """Execute the distributed program via simpler Worker(level=3).

        ``config`` is an optional per-dispatch :class:`RunConfig`; its per-task
        ring-sizing overrides (``ring_task_window`` / ``ring_heap`` /
        ``ring_dep_pool``) size this dispatch's runtime ring buffers, and its
        runtime-diagnostic DFX flags (``enable_dump_args`` / ``enable_pmu`` /
        ``enable_dep_gen`` / ``enable_scope_stats`` / ``enable_chip_swimlane``) are
        written per dispatch under ``<output_dir>/dfx_outputs/rank{r}/d{k}/``
        (``d{k}`` is the card's k-th dispatch, so multiple dispatches to one card
        keep separate artifacts). Onboard swimlane runs a dep-gen-only graph
        pass followed by a dep-gen-disabled timing pass and emits
        ``merged_swimlane_*.json`` per dispatch. Both passes execute the program
        and do not restore mutable arguments between them. Other compile-side
        fields are not consumed on the dispatch path.

        One-shot calls accept host ``torch.Tensor`` arguments only. A resident
        ``DeviceTensor`` needs the owner ``Buffer`` and provenance of the same
        live Worker; use :meth:`prepare`, allocate it through the returned
        ``DistributedWorker``, then call ``worker.run(...)``.
        """
        from pypto.runtime.distributed_runner import _execute_distributed  # noqa: PLC0415

        param_infos, output_indices, return_types = self._get_metadata()
        n_params = len(param_infos)
        n_inputs = n_params - len(output_indices)
        has_return = len(return_types) > 0
        return_style = has_return and len(args) == n_inputs

        if any(isinstance(arg, (DeviceTensor, StackedDeviceTensor)) for arg in args):
            raise TypeError(
                "One-shot DistributedCompiledProgram calls cannot accept DeviceTensor or "
                "StackedDeviceTensor: their Buffer/provenance must belong to the same prepared "
                "DistributedWorker. Use `with compiled.prepare() as worker:`, allocate with "
                "`worker.alloc_tensor()` / `worker.alloc_stacked_tensor()`, then call `worker.run(...)`."
            )

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

        # Validate and coerce one-shot host tensor args. Resident tensors are a
        # prepared-DistributedWorker-only calling convention (guarded above).
        coerced: list[torch.Tensor] = []
        for info, arg in zip(param_infos, all_args, strict=True):
            if not isinstance(arg, torch.Tensor):
                raise TypeError(
                    f"One-shot distributed programs only support host torch.Tensor parameters. "
                    f"Parameter {info.name!r} got {type(arg).__name__}"
                )
            coerced.append(arg)

        _execute_distributed(self, coerced, config)

        if not return_style:
            return None
        outputs = [coerced[i] for i in output_indices]
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def prepare(
        self,
        config: "RunConfig | None" = None,
        *,
        extra_compiled: Sequence["DistributedCompiledProgram"] = (),
        persistent: bool = False,
        reset_persistent_windows: bool | None = None,
        callbacks: dict[str, Callable[..., Any]] | None = None,
        sub_worker_overrides: dict[str, Callable[..., Any]] | None = None,
        inherited_host_tensors: "Sequence[torch.Tensor | ReadOnlyHostTensor] | None" = None,
        startup_timeout_s: float | None = None,
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
        worker-resident ``DeviceTensor`` arguments. Non-shared host tensors are
        rejected as direct dispatch arguments because the forked chip worker
        cannot see a buffer allocated after the fork. Explicit
        ``alloc_tensor(init=...)``, ``copy_to`` and ``copy_from`` operations may
        still use ordinary contiguous CPU tensors: the runtime stages them
        through an owned POSIX shared-memory ``Buffer``.

        Args:
            config: Optional :class:`~pypto.runtime.RunConfig` used **only** to
                prewarm the runtime-arena cache with its ring sizing, so the first
                dispatch skips the ~800ms cold build. It is not stored: every
                dispatch still takes its own ``config=``, so pass the same
                ``RunConfig`` to both. ``None`` prewarms the program's baseline
                sizing (env / compile-time fallback). The cache is single-slot, so
                dispatches that alternate ring sizings rebuild on each switch.
            extra_compiled: Additional compatible programs to prepare on the
                same L3 worker (multi-program serving — e.g. ``prefill`` prepares
                the worker and passes ``[decode]`` here so both share the worker
                and its device-resident KV cache). ``self`` is the *primary*
                program dispatched by ``rt(*args)``; the rest are dispatched via
                ``rt.run(other, *args)``. All must agree on platform, runtime,
                and device ids. Defaults to none (single-program).
            persistent: Retain each compiled program's CommDomains across
                dispatches. Every request still runs through its own
                ``Worker.run`` completion fence. Retained windows are restored
                to zero before reuse by default. Requires artifacts generated
                by a PyPTO version that supports the internal domain-provider
                hook.
            reset_persistent_windows: Whether to restore retained CommDomain
                windows to zero before every reuse. ``None`` (the default)
                enables reset in persistent mode. Set to ``False`` only when
                the caller manually clears or otherwise manages all reused
                communication-buffer state.
            callbacks: Bind a callable to a SubWorker by name — e.g. a real
                sampling closure. Abstract SubWorkers (declared with a ``...``
                body) are runtime-bound callback points and MUST be supplied
                here; a missing binding raises ``ValueError``. A callback may
                also replace a concrete SubWorker's generated body. Each name
                must be a sub-worker the program declares; an unknown name
                raises ``ValueError``. In multi-program mode the callbacks apply
                to every prepared program.
            sub_worker_overrides: Deprecated alias for ``callbacks``.
            inherited_host_tensors: Host ranges that predate the fork and that
                the caller **guarantees** are visible across processes — a
                ``MAP_SHARED`` mapping, whether torch's own shared memory or an
                external file mapping — and stay valid for the worker's lifetime.
                Listed ranges are named in place by ``copy_to`` / ``copy_from``
                instead of being staged through an owned shm ``Buffer``, which
                removes one full host-side copy of the payload. Passing a
                ``MAP_PRIVATE`` backing is unsupported and may upload stale or
                incorrect data; see :class:`DistributedWorker` for the full
                contract and for the warning emitted when the guarantee cannot
                be confirmed. An entry may be wrapped in
                :class:`~pypto.runtime.ReadOnlyHostTensor` when the backing is
                shared but not writable, which makes ``copy_from`` into it raise
                instead of faulting in the forked child.
            startup_timeout_s: Optional positive finite bound, in seconds, for
                the forked worker hierarchy to report startup readiness. ``None``
                keeps Simpler's default timeout.

        Returns:
            A :class:`DistributedWorker`; use it as a context manager or call
            ``close()`` when done.
        """
        from pypto.runtime.distributed_runner import DistributedWorker  # noqa: PLC0415

        return DistributedWorker(
            [self, *extra_compiled],
            config,
            persistent=persistent,
            reset_persistent_windows=reset_persistent_windows,
            callbacks=callbacks,
            sub_worker_overrides=sub_worker_overrides,
            inherited_host_tensors=inherited_host_tensors,
            startup_timeout_s=startup_timeout_s,
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
