# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Local caller/controller control messages and alias-preserving I/O staging.

This portable module has no PyPTO dependency. The authenticated local channel
does not carry L4 task frames: the MPI controller dispatches those directly.
"""

import copy
import ctypes
import hmac
import json
import mmap
import os
import secrets
import socket
import struct
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

MAX_RECORD_BYTES = 8 * 1024 * 1024
_LENGTH = struct.Struct("!I")


def _remaining(deadline: float) -> float:
    timeout = deadline - time.monotonic()
    if timeout <= 0:
        raise TimeoutError("Direct controller authentication timed out")
    return timeout


def _read_exact(stream: socket.socket, size: int, deadline: float | None = None) -> bytes:
    data = bytearray()
    while len(data) < size:
        if deadline is not None:
            stream.settimeout(_remaining(deadline))
        part = stream.recv(size - len(data))
        if not part:
            raise EOFError("Direct controller channel closed before completion")
        data.extend(part)
    return bytes(data)


def send_record(stream: socket.socket, record: dict[str, Any]) -> None:
    """Send one bounded finite JSON control record."""
    payload = json.dumps(record, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(payload) > MAX_RECORD_BYTES:
        raise ValueError("Direct controller record exceeds size limit")
    stream.sendall(_LENGTH.pack(len(payload)) + payload)


def recv_record(stream: socket.socket) -> dict[str, Any]:
    """Read one complete record without allowing unbounded allocation."""
    size = _LENGTH.unpack(_read_exact(stream, _LENGTH.size))[0]
    if size <= 0 or size > MAX_RECORD_BYTES:
        raise ValueError("Direct controller record exceeds size limit")
    result = json.loads(_read_exact(stream, size))
    if not isinstance(result, dict):
        raise ValueError("Direct controller record must be an object")
    return result


def accept_session(listener: socket.socket, authkey: bytes, timeout: float) -> socket.socket:
    """Authenticate the one local rank-0 controller under a finite deadline."""
    deadline = time.monotonic() + timeout
    listener.settimeout(timeout)
    stream, address = listener.accept()
    try:
        stream.settimeout(_remaining(deadline))
        if address[0] != "127.0.0.1":
            raise ValueError("Direct controller authentication requires a local peer")
        challenge = secrets.token_bytes(32)
        stream.sendall(challenge)
        answer = _read_exact(stream, 32, deadline)
        stream.settimeout(_remaining(deadline))
        if not hmac.compare_digest(answer, hmac.digest(authkey, challenge, "sha256")):
            stream.sendall(b"NO")
            raise ValueError("Direct controller authentication failed")
        stream.sendall(b"OK")
        stream.settimeout(None)
        return stream
    except BaseException:
        stream.close()
        raise


def connect_session(address: Any, authkey: bytes, timeout: float) -> socket.socket:
    """Connect a rank-0 engine to its caller without placing keys in argv."""
    host, port = address
    if isinstance(authkey, str):
        authkey = bytes.fromhex(authkey)
    if host != "127.0.0.1":
        raise ValueError("Direct caller authentication requires a local address")
    deadline = time.monotonic() + timeout
    stream = socket.create_connection((host, int(port)), timeout=timeout)
    try:
        challenge = _read_exact(stream, 32, deadline)
        stream.settimeout(_remaining(deadline))
        stream.sendall(hmac.digest(authkey, challenge, "sha256"))
        if _read_exact(stream, 2, deadline) != b"OK":
            raise ValueError("Direct controller authentication failed")
        stream.settimeout(None)
        return stream
    except BaseException:
        stream.close()
        raise


class PackedValues:
    """Own local mapped backing until a run fence or confirmed launcher exit.

    Each original Torch storage is copied once. Different tensor views retain
    their common storage and element offsets in the controller process. Output
    publication is explicit and restricted to declared writable argument spans.
    """

    def __init__(self, values: list[Any], writable: list[int], directory: Path) -> None:
        self.directory = Path(directory)
        self._backings: list[tuple[Any, Any, Path, int, list[tuple[int, int]]]] = []
        self._record: dict[str, Any] = {"storages": [], "values": []}
        self._closed = False
        if any(
            isinstance(value, torch.Tensor)
            and (value.device.type != "cpu" or not value.is_contiguous() or value.numel() <= 0)
            for value in values
        ):
            raise ValueError("Cluster ordinary I/O requires nonempty contiguous CPU tensors")
        seen: dict[tuple[int, int], int] = {}
        writable_set = set(writable)
        try:
            for index, value in enumerate(values):
                if not isinstance(value, torch.Tensor):
                    scalar = value.value if isinstance(value, ctypes._SimpleCData) else value
                    if scalar is not None and type(scalar) not in (bool, int, float):
                        raise ValueError("Cluster scalar argument must be a plain numeric value")
                    self._record["values"].append(
                        {"kind": "float64", "value": struct.pack("!d", scalar).hex()}
                        if type(scalar) is float
                        else {"kind": "scalar", "value": scalar}
                    )
                    continue
                storage = value.untyped_storage()
                key = (storage.data_ptr(), storage.nbytes())
                storage_index = seen.get(key)
                if storage_index is None:
                    storage_index = len(self._backings)
                    name = "storage-" + secrets.token_hex(16) + ".bin"
                    path = self.directory / name
                    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
                    try:
                        # Write through the file before mapping it so allocation errors
                        # surface as OSError instead of a fault on a sparse mapping.
                        with os.fdopen(descriptor, "wb", closefd=False) as writer:
                            writer.write((ctypes.c_ubyte * key[1]).from_address(key[0]))
                        backing = mmap.mmap(descriptor, key[1])
                    except BaseException:
                        path.unlink()
                        raise
                    finally:
                        os.close(descriptor)
                    self._backings.append((storage, backing, path, key[1], []))
                    self._record["storages"].append({"name": name, "nbytes": key[1]})
                    seen[key] = storage_index
                offset = value.storage_offset()
                self._record["values"].append(
                    {
                        "kind": "tensor",
                        "storage": storage_index,
                        "dtype": str(value.dtype).removeprefix("torch."),
                        "shape": list(value.shape),
                        "strides": list(value.stride()),
                        "offset": offset,
                        "conjugate": value.is_conj(),
                        "negative": value.is_neg(),
                    }
                )
                if index in writable_set:
                    begin = value.data_ptr() - key[0]
                    self._backings[storage_index][4].append(
                        (begin, begin + value.numel() * value.element_size())
                    )
        except BaseException:
            self.close()
            raise

    def request(self) -> dict[str, Any]:
        """Return an independent message snapshot, never a writable journal."""
        return copy.deepcopy(self._record)

    def publish(self) -> None:
        """Publish declared outputs only after the controller reports success."""
        if self._closed:
            raise RuntimeError("Direct invocation storage is closed")
        for storage, backing, _path, _size, spans in self._backings:
            source = ctypes.addressof(ctypes.c_ubyte.from_buffer(backing))
            for begin, end in spans:
                ctypes.memmove(storage.data_ptr() + begin, source + begin, end - begin)

    def close(self) -> None:
        """Release mappings after a successful run or a proved session drain."""
        if self._closed:
            return
        for _storage, backing, path, _size, _spans in self._backings:
            backing.close()
            path.unlink(missing_ok=True)
        self._backings.clear()
        self._closed = True

    def __enter__(self) -> "PackedValues":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def _mapped_value(record: dict[str, Any], roots: list[torch.Tensor]) -> Any:
    """Decode one scalar or a bounds-checked view into mapped storage."""
    if record["kind"] == "float64":
        encoded = record["value"]
        if not isinstance(encoded, str) or len(encoded) != 16:
            raise ValueError("Invalid direct float scalar encoding")
        return struct.unpack("!d", bytes.fromhex(encoded))[0]
    if record["kind"] == "scalar":
        value = record["value"]
        if value is not None and type(value) not in (bool, int, float):
            raise ValueError("Invalid direct scalar argument")
        return value
    if record["kind"] != "tensor":
        raise ValueError("Invalid direct argument kind")
    index = record["storage"]
    dtype = getattr(torch, record["dtype"], None)
    shape, strides, offset = record["shape"], record["strides"], record["offset"]
    if (
        type(index) is not int
        or not 0 <= index < len(roots)
        or not isinstance(dtype, torch.dtype)
        or not isinstance(shape, list)
        or not isinstance(strides, list)
        or len(shape) != len(strides)
        or len(shape) > 64
        or any(type(x) is not int or x <= 0 for x in shape)
        or any(type(x) is not int or x < 0 for x in strides)
        or type(offset) is not int
        or offset < 0
    ):
        raise ValueError("Invalid direct tensor view")
    itemsize = torch.tensor([], dtype=dtype).element_size()
    end = offset + 1 + sum((dim - 1) * stride for dim, stride in zip(shape, strides, strict=True))
    root = roots[index]
    if end * itemsize > root.numel():
        raise ValueError("Direct tensor view exceeds storage backing")
    view = root[: root.numel() // itemsize * itemsize].view(dtype).as_strided(shape, strides, offset)
    if record["conjugate"]:
        view = view.conj()
    if record["negative"]:
        view = torch._neg_view(view)
    return view


@contextmanager
def map_values(request: dict[str, Any], directory: Path):
    """Map only caller-owned local staging files and validate every view."""
    roots: list[torch.Tensor] = []
    values: list[Any] = []
    root = Path(directory).resolve()
    try:
        for record in request["storages"]:
            name, size = record["name"], record["nbytes"]
            if (
                not isinstance(name, str)
                or not name.startswith("storage-")
                or Path(name).name != name
                or (root / name).resolve().parent != root
                or type(size) is not int
                or size <= 0
            ):
                raise ValueError("Invalid direct storage descriptor")
            descriptor = os.open(root / name, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
            try:
                if os.fstat(descriptor).st_size != size:
                    raise ValueError("Direct storage size does not match backing")
                backing = mmap.mmap(descriptor, size)
            finally:
                os.close(descriptor)
            roots.append(torch.frombuffer(backing, dtype=torch.uint8))
        values.extend(_mapped_value(record, roots) for record in request["values"])
        yield values
    finally:
        # Torch storage retains its mmap exporter. Releasing our references
        # lets the last tensor view close it, including a view held until drain.
        values.clear()
        roots.clear()
