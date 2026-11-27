# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime-only placement for L3 programs; never part of compiled artifact metadata."""

from dataclasses import dataclass
from ipaddress import IPv4Address


def _device_id(value: int) -> int:
    if type(value) is not int or not 0 <= value < 2**31:
        raise ValueError("device_id must be a non-negative int32")
    return value


@dataclass(frozen=True)
class RemoteChipTarget:
    """One daemon-hosted card, using the parent program's platform and runtime.

    Args:
        endpoint: Numeric IPv4 address (or localhost) and daemon port.
        device_id: Device index in the remote host's CANN namespace.
    """

    endpoint: str
    device_id: int

    def __post_init__(self) -> None:
        if not isinstance(self.endpoint, str):
            raise TypeError("endpoint must be an IPv4:port string")
        host, separator, port = self.endpoint.rpartition(":")
        if not separator or not port.isdecimal() or not 1 <= int(port) <= 65535:
            raise ValueError("endpoint must contain an IPv4 address and a port in [1, 65535]")
        if host != "localhost":
            IPv4Address(host)
        _device_id(self.device_id)


@dataclass(frozen=True)
class DistributedDeployment:
    """Ordered local cards followed by ordered remote cards.

    Logical rank is the position in this sequence, independent of the host's
    device index. Simpler's returned worker IDs are retained separately; neither
    device indices nor a presumed contiguous worker-ID range are global identity.

    Args:
        local_device_ids: Device indices on the controller host.
        remote_chips: Ordered daemon-hosted cards.
        comm_profile: Simpler communication profile, such as a3-fabric-v1 or sim.
    """

    local_device_ids: tuple[int, ...]
    remote_chips: tuple[RemoteChipTarget, ...] = ()
    comm_profile: str = "a3-fabric-v1"

    def __post_init__(self) -> None:
        local = tuple(_device_id(device) for device in self.local_device_ids)
        remote = tuple(self.remote_chips)
        if len(set(local)) != len(local):
            raise ValueError("local_device_ids must be distinct")
        if any(not isinstance(target, RemoteChipTarget) for target in remote):
            raise TypeError("remote_chips must contain RemoteChipTarget values")
        # A daemon's port does not create a second physical device namespace.
        identities = [
            (
                "127.0.0.1"
                if target.endpoint.rpartition(":")[0] == "localhost"
                else str(IPv4Address(target.endpoint.rpartition(":")[0])),
                target.device_id,
            )
            for target in remote
        ]
        if len(set(identities)) != len(identities):
            raise ValueError("a remote host/device pair may appear only once")
        if not local and not remote:
            raise ValueError("deployment must contain at least one card")
        if self.comm_profile not in ("a3-fabric-v1", "sim"):
            raise ValueError("comm_profile must be a3-fabric-v1 or sim")
        object.__setattr__(self, "local_device_ids", local)
        object.__setattr__(self, "remote_chips", remote)

    @property
    def world_size(self) -> int:
        """Number of logical ranks, including remote cards."""
        return len(self.local_device_ids) + len(self.remote_chips)
