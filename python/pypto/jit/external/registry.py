# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Lazy provider selection and request-owned capture/preparation lifetimes."""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock

from .contracts import KernelRequest, PreparedKernel
from .identity import _hash, preparation_key
from .protocols import CapturedInputs, CapturingKernelProvider
from .validation import ContractError, _immutable, validate_identity, validate_prepared, validate_request


class _Registration:
    def __init__(
        self, name: str, matches: Callable[[object], bool], load: Callable[[], CapturingKernelProvider]
    ) -> None:
        self.name = name
        self.matches = matches
        self._load = load
        self._provider: CapturingKernelProvider | None = None
        self._lock = RLock()
        self._loading = False

    def provider(self) -> CapturingKernelProvider:
        # Loading is serialized, but capture and compilation are request-owned.
        # RLock lets a recursive loader fail explicitly instead of deadlocking.
        with self._lock:
            if self._provider is None:
                if self._loading:
                    raise ContractError(f"provider {self.name!r}: recursive loader")
                self._loading = True
                try:
                    provider = self._load()
                    if not callable(getattr(provider, "capture", None)) or not callable(
                        getattr(provider, "prepare", None)
                    ):
                        raise ContractError(f"provider {self.name!r}: expected capture and prepare methods")
                    self._provider = provider
                finally:
                    self._loading = False
            return self._provider


@dataclass(frozen=True, eq=False)
class CapturedKernel:
    """One preparation request, independent of the original kernel's lifetime."""

    request: KernelRequest
    inputs: CapturedInputs
    _provider: CapturingKernelProvider

    @property
    def preparation_key(self) -> str | None:
        """Return no reusable identity when the provider cannot capture all inputs."""
        return preparation_key(self.request, self.inputs.identity)

    def prepare(self) -> PreparedKernel:
        """Prepare afresh; persistent reuse belongs to the existing artifact store."""
        prepared = self._provider.prepare(self.inputs, self.request)
        validate_prepared(prepared)
        # Dataclass equality would treat constant True, 1 and 1.0 as equal.
        # Compare the canonical serialization used for preparation identity.
        if _hash(prepared.request) != _hash(self.request):
            raise ContractError("prepared: provider changed the captured request")
        if prepared.identity != self.inputs.identity:
            raise ContractError("prepared: provider changed the captured input identity")
        return prepared


@dataclass(frozen=True, eq=False)
class KernelRef:
    """Strong, explicit ownership of a kernel and its selected registration.

    Registries never retain kernel objects. Dropping this reference releases the
    kernel, including unhashable/non-weak-referenceable objects. Existing refs
    remain valid after unregistering or replacing a provider registration.
    """

    _kernel: object
    _registration: _Registration

    @property
    def provider_name(self) -> str:
        return self._registration.name

    def capture(self, request: KernelRequest) -> CapturedKernel:
        """Validate the signature and target before loading optional compiler code."""
        validate_request(request)
        provider = self._registration.provider()
        inputs = provider.capture(self._kernel, request)
        _immutable(inputs, "capture", CapturedInputs)
        validate_identity(inputs.identity)
        if inputs.identity.provider != self.provider_name:
            raise ContractError(f"capture: expected provider identity {self.provider_name!r}")
        if hashlib.sha256(inputs.payload).hexdigest() != inputs.identity.kernel_digest:
            raise ContractError("capture: payload does not match the input identity digest")
        return CapturedKernel(request, inputs, provider)


class ProviderRegistry:
    """A scoped registry; importing PyPTO does not load or probe optional DSLs.

    Matchers only recognize objects and must not compile or import optional DSLs.
    Factories may import them after an explicit capture request. No global kernel
    cache or implicit entry-point scanning extends users' object lifetimes.
    """

    def __init__(self) -> None:
        self._registrations: dict[str, _Registration] = {}
        self._lock = RLock()

    def register(
        self,
        name: str,
        *,
        matches: Callable[[object], bool],
        load: Callable[[], CapturingKernelProvider],
    ) -> None:
        """Register recognition and loading separately; duplicate names are errors."""
        if type(name) is not str or not name:
            raise ContractError(f"registry: expected a nonempty provider name, got {name!r}")
        if not callable(matches) or not callable(load):
            raise ContractError(f"provider {name!r}: expected callable matcher and loader")
        with self._lock:
            if name in self._registrations:
                raise ContractError(f"registry: provider {name!r} is already registered")
            self._registrations[name] = _Registration(name, matches, load)

    def unregister(self, name: str) -> None:
        """Remove future discovery of a provider without invalidating live refs."""
        with self._lock:
            if name not in self._registrations:
                raise ContractError(f"registry: unknown provider {name!r}")
            del self._registrations[name]

    def bind(self, kernel: object, *, provider: str) -> KernelRef:
        """Select a provider explicitly, bypassing automatic recognition."""
        with self._lock:
            registration = self._registrations.get(provider)
        if registration is None:
            raise ContractError(f"registry: unknown provider {provider!r}")
        return KernelRef(kernel, registration)

    def discover(self, kernel: object) -> KernelRef | None:
        """Recognize an object without loading providers; reject every ambiguity."""
        if isinstance(kernel, KernelRef):
            return kernel
        with self._lock:
            registrations = tuple(self._registrations.values())
        matches: list[_Registration] = []
        for registration in registrations:
            matched = registration.matches(kernel)
            if type(matched) is not bool:
                raise ContractError(f"provider {registration.name!r}: matcher must return bool")
            if matched:
                matches.append(registration)
        if len(matches) > 1:
            names = sorted(registration.name for registration in matches)
            raise ContractError(f"registry: ambiguous providers {names}; bind the kernel explicitly")
        return KernelRef(kernel, matches[0]) if matches else None
