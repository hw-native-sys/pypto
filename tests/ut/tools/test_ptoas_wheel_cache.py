# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise the CI wheel-cache shell code without network or compiler imports."""

import hashlib
import os
import subprocess
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[3]
_WHEEL = "ptoas-0.72-cp310-cp310-manylinux_2_34_x86_64.whl"


def _cache_script() -> str:
    action = yaml.safe_load((_ROOT / ".github/actions/setup-ci-job/action.yml").read_text())
    step = next(s for s in action["runs"]["steps"] if s.get("name") == "Install ptoas (local cache)")
    script = step["run"]
    # Run the production download/verify/publish fragment, stopping before the
    # job activates its interpreter and installs the wheel.
    start = script.index("PTOAS_PACKAGE_VERSION=")
    end = script.index("source activate.sh", start)
    return script[start:end]


def _run_cache(tmp_path, payload: bytes, *, expected_digest: str | None = None):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    curl = fake_bin / "curl"
    curl.write_text(
        '#!/bin/sh\nwhile [ "$1" != "-o" ]; do shift; done\n'
        'cp "$CACHE_TEST_PAYLOAD" "$2"\nprintf "download\n" >> "$CACHE_TEST_LOG"\n'
    )
    curl.chmod(0o755)
    source = tmp_path / "payload"
    source.write_bytes(payload)
    digest = expected_digest or hashlib.sha256(payload).hexdigest()
    env = dict(os.environ)
    env.update(
        PATH=f"{fake_bin}{os.pathsep}{env['PATH']}",
        PTOAS_VERSION="v0.72",
        PTOAS_ARCH="x86_64",
        PTOAS_SHA256=digest,
        PTOAS_CACHE_DIR=str(tmp_path / "cache"),
        CACHE_TEST_PAYLOAD=str(source),
        CACHE_TEST_LOG=str(tmp_path / "downloads"),
    )
    return subprocess.run(
        ["bash", "-e", "-c", _cache_script()],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )


def test_republished_wheel_uses_new_pin_and_preserves_cache_hits(tmp_path):
    """A stale version-only cache must not poison a corrected release pin."""
    cache = tmp_path / "cache"
    cache.mkdir()
    stale = cache / _WHEEL
    stale.write_bytes(b"old release bytes")
    for payload in (b"first pinned release", b"second pinned release"):
        result = _run_cache(tmp_path, payload)
        assert result.returncode == 0, result.stderr
        digest = hashlib.sha256(payload).hexdigest()
        assert (cache / digest / _WHEEL).read_bytes() == payload
        # Reusing the same pin needs no second download.
        result = _run_cache(tmp_path, payload)
        assert result.returncode == 0, result.stderr
        assert "Cache hit" in result.stdout
    assert stale.read_bytes() == b"old release bytes"
    assert (tmp_path / "downloads").read_text().splitlines() == ["download", "download"]


def test_bad_download_is_never_published(tmp_path):
    digest = hashlib.sha256(b"trusted release").hexdigest()
    result = _run_cache(tmp_path, b"wrong bytes", expected_digest=digest)
    assert result.returncode != 0
    assert not (tmp_path / "cache" / digest / _WHEEL).exists()
    assert not list((tmp_path / "cache").rglob(".ptoas-*"))


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
