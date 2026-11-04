# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise the CI wheel cache using the action's shell and local download fixtures."""

import hashlib
import os
import subprocess
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _cache_script() -> str:
    action = (ROOT / ".github/actions/setup-ci-job/action.yml").read_text()
    # Execute the real download/cache block without toolchain activation or pip.
    start = action.index('        PTOAS_ARCH="$(uname -m)"')
    end = action.index("        # activate.sh gives both", start)
    return textwrap.dedent(action[start:end])


def _run_cache(cache: Path, payload: Path, arch: str, digest: str) -> subprocess.CompletedProcess[str]:
    mocks = """
uname() { printf '%s\n' "$TEST_ARCH"; }
curl() {
    printf 'download\n'
    cp "$TEST_PAYLOAD" "${@: -1}"
}
"""
    return subprocess.run(
        ["bash", "-euo", "pipefail", "-c", mocks + _cache_script()],
        env={
            **os.environ,
            "TEST_ARCH": arch,
            "TEST_PAYLOAD": str(payload),
            "PTOAS_CACHE_DIR": str(cache),
            "PTOAS_VERSION": "v0.72",
            "PTOAS_SHA256_AARCH64": digest,
            "PTOAS_SHA256_X86_64": digest,
        },
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def _wheel_name(arch: str) -> str:
    return f"ptoas-0.72-cp310-cp310-manylinux_2_34_{arch}.whl"


@pytest.mark.parametrize("arch", ["aarch64", "x86_64"])
def test_republished_wheel_ignores_old_cache_and_reuses_verified_download(tmp_path: Path, arch: str) -> None:
    cache = tmp_path / "shared cache"
    cache.mkdir()
    old = b"original release"
    new = b"republished release"
    name = _wheel_name(arch)
    # Both the historical flat cache and an older digest must remain untouched.
    legacy = cache / name
    legacy.write_bytes(old)
    old_digest = hashlib.sha256(old).hexdigest()
    old_path = cache / old_digest / name
    old_path.parent.mkdir()
    old_path.write_bytes(old)
    payload = tmp_path / "download"
    payload.write_bytes(new)
    digest = hashlib.sha256(new).hexdigest()

    result = _run_cache(cache, payload, arch, digest)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Cache miss" in result.stdout
    assert (cache / digest / name).read_bytes() == new
    assert legacy.read_bytes() == old
    assert old_path.read_bytes() == old
    assert not list(cache.rglob(".ptoas-*"))

    # A valid cache hit must work even when downloading is impossible.
    payload.unlink()
    result = _run_cache(cache, payload, arch, digest)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Cache hit" in result.stdout
    assert "download\n" not in result.stdout


@pytest.mark.parametrize("arch", ["aarch64", "x86_64"])
def test_invalid_download_is_never_published(tmp_path: Path, arch: str) -> None:
    payload = tmp_path / "download"
    payload.write_bytes(b"invalid release")
    digest = hashlib.sha256(b"expected release").hexdigest()
    cache = tmp_path / "cache"
    result = _run_cache(cache, payload, arch, digest)
    assert result.returncode != 0
    assert "FAILED" in result.stdout
    assert not list(cache.rglob("*.whl"))
    assert not list(cache.rglob(".ptoas-*"))


@pytest.mark.parametrize("same_pin", [False, True])
def test_concurrent_jobs_preserve_each_pinned_wheel(tmp_path: Path, same_pin: bool) -> None:
    cache = tmp_path / "cache"
    payloads = [tmp_path / "first", tmp_path / "second"]
    payloads[0].write_bytes(b"first release")
    payloads[1].write_bytes(b"first release" if same_pin else b"second release")
    digests = [hashlib.sha256(path.read_bytes()).hexdigest() for path in payloads]
    # Same-version jobs may share a pin or use different republished assets.
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(_run_cache, cache, payload, "aarch64", digest)
            for payload, digest in zip(payloads, digests)
        ]
        results = [future.result() for future in futures]
    for result in results:
        assert result.returncode == 0, result.stdout + result.stderr
    for payload, digest in zip(payloads, digests):
        assert (cache / digest / _wheel_name("aarch64")).read_bytes() == payload.read_bytes()
    assert not list(cache.rglob(".ptoas-*"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
