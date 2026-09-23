# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import argparse
import ipaddress
import os
import select
import signal
import socket
import socketserver
from pathlib import Path


class RelayServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], target: tuple[str, int], allowed_cidr: str):
        self.target = target
        self.allowed_network = ipaddress.ip_network(allowed_cidr, strict=True)
        super().__init__(server_address, RelayHandler)


class RelayHandler(socketserver.BaseRequestHandler):
    server: RelayServer

    def handle(self) -> None:
        source = ipaddress.ip_address(self.client_address[0])
        if source not in self.server.allowed_network:
            return

        with socket.create_connection(self.server.target, timeout=10) as upstream:
            self.request.settimeout(None)
            upstream.settimeout(None)
            sockets = (self.request, upstream)

            while True:
                readable, _, _ = select.select(sockets, (), ())
                for reader in readable:
                    data = reader.recv(65536)
                    if not data:
                        return
                    destination = upstream if reader is self.request else self.request
                    destination.sendall(data)


def write_ready_file(path: Path, port: int) -> None:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(str(path), flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as ready_file:
        ready_file.write(f"{port}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Relay a Docker network to a fixed local TCP service")
    parser.add_argument("--listen-host", required=True)
    parser.add_argument("--listen-port", type=int, default=0)
    parser.add_argument("--allow-cidr", required=True)
    parser.add_argument("--target-host", default="127.0.0.1")
    parser.add_argument("--target-port", type=int, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with RelayServer(
        (args.listen_host, args.listen_port),
        (args.target_host, args.target_port),
        args.allow_cidr,
    ) as server:
        write_ready_file(args.ready_file, server.server_address[1])

        def stop(_signum: int, _frame: object) -> None:
            raise KeyboardInterrupt

        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        try:
            server.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
