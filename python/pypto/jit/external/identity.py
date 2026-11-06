# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Preparation identities, without implementing a second artifact cache."""

import hashlib
import json
from dataclasses import fields, is_dataclass, replace
from enum import Enum
from typing import cast

from pypto.pypto_core import DataType

from .contracts import KernelRequest, PreparationIdentity, PreparedKernel
from .validation import validate_identity, validate_prepared, validate_request


def _encode(value: object) -> object:
    if isinstance(value, DataType):
        return {"dtype": value.to_string()}
    if isinstance(value, Enum):
        return {"enum": type(value).__name__, "value": value.value}
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _encode(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, (tuple, frozenset)):
        encoded = [_encode(item) for item in cast(tuple[object, ...] | frozenset[object], value)]
        return (
            sorted(encoded, key=lambda item: json.dumps(item, sort_keys=True))
            if isinstance(value, frozenset)
            else encoded
        )
    return value


def _hash(value: object) -> str:
    payload = json.dumps(_encode(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def preparation_key(request: KernelRequest, identity: PreparationIdentity) -> str | None:
    """Return no reusable key when source/dependency identity is incomplete.

    Identity must describe every provider input not already in request. This is
    a provider promise, not a claim that Python closure discovery is complete.
    Runtime addresses are intentionally absent; per-call checks still apply.
    """
    validate_request(request)
    validate_identity(identity)
    if not identity.complete:
        return None
    normalized = replace(
        request, constants=tuple(sorted(request.constants)), options=tuple(sorted(request.options))
    )
    return _hash((normalized, identity))


def prepared_digest(prepared: PreparedKernel) -> str:
    """Fingerprint validated outputs even when preparation itself cannot be reused.

    Downstream artifact keys must additionally include bridge version, runtime,
    and effective build options. This digest alone is not an executable key.
    """
    validate_prepared(prepared)
    normalized_request = replace(
        prepared.request,
        constants=tuple(sorted(prepared.request.constants)),
        options=tuple(sorted(prepared.request.options)),
    )
    artifact = replace(
        prepared.artifact, files=tuple(sorted(prepared.artifact.files, key=lambda file: file.path))
    )
    return _hash(replace(prepared, request=normalized_request, artifact=artifact))
