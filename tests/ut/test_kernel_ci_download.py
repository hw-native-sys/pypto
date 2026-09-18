# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exercise CI release downloads against interrupted local HTTP transfers."""

import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / ".github/scripts/download-kernel-framework.sh"
_PAYLOAD = b"framework release bytes" * 1024
_PARTIAL_SIZE = len(_PAYLOAD) // 2


@pytest.fixture
def release_server(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    state = {"mode": "resume", "ranges": []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            request_range = self.headers.get("Range")
            state["ranges"].append(request_range)
            if state["mode"] == "unavailable":
                self.send_error(503)
                return
            if request_range and state["mode"] == "reject":
                self.send_error(416)
                return
            if request_range and state["mode"] in {"resume", "always-interrupt"}:
                offset = int(request_range.removeprefix("bytes=").removesuffix("-"))
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {offset}-{len(_PAYLOAD) - 1}/{len(_PAYLOAD)}")
                payload = _PAYLOAD[offset:]
            else:
                self.send_response(200)
                payload = _PAYLOAD
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if state["mode"] == "always-interrupt" or (
                len(state["ranges"]) == 1 and state["mode"] != "complete"
            ):
                payload = payload[: len(payload) // 2]
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass  # curl closes the connection when a server ignores Range.

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/release.whl", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _download(url, destination):
    return subprocess.run(
        ["bash", str(_SCRIPT), url, str(destination)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@pytest.mark.parametrize("mode", ["complete", "resume", "ignore", "reject"])
def test_completed_download_after_interruption(tmp_path, release_server, mode):
    url, state = release_server
    state["mode"] = mode
    destination = tmp_path / "framework.whl"
    result = _download(url, destination)
    assert result.returncode == 0, result.stderr
    assert destination.read_bytes() == _PAYLOAD
    expected_ranges: list[str | None] = [None]
    if mode != "complete":
        expected_ranges.append(f"bytes={_PARTIAL_SIZE}-")
    if mode in {"ignore", "reject"}:
        expected_ranges.append(None)
    assert state["ranges"] == expected_ranges
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("mode", ["unavailable", "always-interrupt"])
def test_failed_download_is_bounded_and_does_not_publish(tmp_path, release_server, existing, mode):
    url, state = release_server
    state["mode"] = mode
    destination = tmp_path / "framework.whl"
    if existing:
        destination.write_bytes(b"existing complete release")
    result = _download(url, destination)
    assert result.returncode != 0
    assert len(state["ranges"]) == 4
    assert "attempt 4/4 failed" in result.stderr
    if existing:
        assert destination.read_bytes() == b"existing complete release"
    else:
        assert not destination.exists()
    assert not list(tmp_path.glob(".download.*"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
