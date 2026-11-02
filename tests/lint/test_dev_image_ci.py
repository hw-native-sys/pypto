# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise the image publication boundary without Docker, credentials or NPUs."""

import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import urllib.error
from email.message import Message
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / f"docker/ci/{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def ci(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(ROOT / "docker/ci"))
    return load_module("maintain")


@pytest.fixture
def state(ci: ModuleType, tmp_path: Path) -> dict:
    data = {
        "image": ci.load_config()["images"]["arm-a2a3"],
        "identity": "arm-a2a3-123-1",
        "image_id": "sha256:" + "a" * 64,
        "local": "pypto-ci:arm-a2a3-123-1",
        "pypto": "b" * 40,
        "recipe": "f" * 40,
        "pypto_lib": "c" * 40,
        "registry": "registry.example",
        "unique": "registry.example/pypto:run-123-1",
        "rolling": "registry.example/pypto:arm-a2a3-cann9.2-dist",
        "previous": "registry.example/pypto:arm-a2a3-cann9.2-dist-previous",
    }
    ci.save(tmp_path / "state.json", data)
    return data


@pytest.mark.parametrize(
    "key,value",
    [
        ("build_jobs", 0),
        ("build_jobs", True),
        ("task_max_seconds", 0),
        ("task_timeout_seconds", 1200),
        ("task_timeout_seconds", 1800),
        ("platform", "linux/amd64"),
        ("validation_platform", "a5"),
        ("cann_digest", "sha256:missing"),
        ("patch", "unknown"),
        ("patch", "cann92-a5-aicpu-class-memaccess"),
        ("distributed", "/workspace/pypto-lib/../other.py"),
        ("distributed_args", "--ep 2"),
        ("distributed_args", ["--ep", 2]),
        ("distributed_args", ["bad\0argument"]),
        ("tag", "bad tag"),
    ],
)
def test_invalid_configuration_fails_before_scheduling(
    ci: ModuleType, tmp_path: Path, key: str, value: object
) -> None:
    config = ci.load_config()
    config["images"]["arm-a2a3"][key] = value
    path = tmp_path / "images.json"
    ci.save(path, config)
    with pytest.raises(ValueError):
        ci.load_config(path)


@pytest.mark.parametrize(
    "first,second", [("shared", "shared"), ("shared", "shared-previous"), ("shared-previous", "shared")]
)
def test_configuration_rejects_current_and_rollback_tag_collisions(
    ci: ModuleType, tmp_path: Path, first: str, second: str
) -> None:
    config = ci.load_config()
    config["images"]["arm-a2a3"]["tag"] = first
    config["images"]["arm-a5"]["tag"] = second
    path = tmp_path / "images.json"
    ci.save(path, config)
    with pytest.raises(ValueError, match="rollback tags"):
        ci.load_config(path)


def test_disabled_future_architecture_is_not_scheduled(
    ci: ModuleType, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ci.load_config()
    config["images"]["x86-a2a3"] = dict(
        config["images"]["arm-a2a3"],
        enabled=False,
        platform="linux/amd64",
        patch="",
        tag="x86-a2a3-cann9.2-dist",
    )
    monkeypatch.setattr(ci, "load_config", lambda: config)
    original = sys.argv
    try:
        sys.argv = ["maintain.py", "matrix"]
        ci.main()
    finally:
        sys.argv = original
    result = json.loads(capsys.readouterr().out)
    assert [item["name"] for item in result["include"]] == ["arm-a2a3", "arm-a5", "x86-a5"]
    assert result["include"][1]["runners"] == ["self-hosted", "Linux", "ARM64", "npu-a5"]
    assert result["include"][2]["runners"] == ["self-hosted", "Linux", "npu-a5", "X64", "pr"]


def test_annotated_tag_resolves_to_commit(ci: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    queries = []

    def remote(argv: list[str], **kwargs: object) -> str:
        queries.append(argv[-1])
        return "a" * 40 + "\trefs/tags/release^{}" if argv[-1].endswith("^{}") else ""

    monkeypatch.setattr(ci, "run", remote)
    assert ci.resolve_ref("https://github.com/org/repo.git", "release") == "a" * 40
    assert queries == ["refs/heads/release", "refs/tags/release^{}"]


@pytest.mark.parametrize(
    "event,pypto_ref", [("push", "main"), ("pull_request", "main"), ("pull_request", "4" * 40)]
)
@pytest.mark.parametrize("image_name", ["arm-a2a3", "arm-a5", "x86-a5"])
def test_build_freezes_refs_and_base_digest(
    ci: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    event: str,
    pypto_ref: str,
    image_name: str,
) -> None:
    config = ci.load_config()
    config["sources"]["pypto"]["ref"] = pypto_ref
    image = config["images"][image_name]
    image["cann_digest"] = "sha256:" + "3" * 64
    architecture = image["platform"].split("/")[1]
    monkeypatch.setattr(ci.platform, "machine", lambda: "x86_64" if architecture == "amd64" else "aarch64")
    monkeypatch.setattr(ci.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.delenv("RUNNER_NAME", raising=False)
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    monkeypatch.setenv("GITHUB_EVENT_NAME", event)
    fork_repository = "https://github.com/contributor/pypto.git"
    monkeypatch.setenv("PYPTO_PR_REPOSITORY", fork_repository)
    monkeypatch.setenv("PYPTO_BUILD_JOBS", "1")
    monkeypatch.setenv("PYPTO_DOCKER_PROXY", "http://proxy.example:7892")
    original_read = Path.read_text

    def read(path: Path, *args: object, **kwargs: object) -> str:
        if str(path) == "/usr/local/Ascend/driver/version.info":
            return f"Version={image['driver_version']}\n"
        return original_read(path)

    monkeypatch.setattr(Path, "read_text", read)
    recipe = "1" * 40
    library = "2" * 40
    base = "quay.io/ascend/cann@sha256:" + "3" * 64
    calls = []

    def command(argv: list[str], *args: object, **kwargs: object) -> str:
        calls.append(argv)
        if argv[0] == "git":
            return library + "\trefs/heads/main" if argv[1] == "ls-remote" else recipe
        if argv[1:3] == ["image", "inspect"]:
            if "--format" in argv:
                return "sha256:" + "4" * 64
            return json.dumps([{"Os": "linux", "Architecture": architecture, "RepoDigests": [base]}])
        return ""

    monkeypatch.setattr(ci, "run", command)
    ci.save(tmp_path / "validated.json", {"stale": True})
    ci.build(config, image_name, tmp_path)
    build = next(argv for argv in calls if argv[1:3] == ["buildx", "build"])
    expected_sha = recipe if pypto_ref == "main" else pypto_ref
    expected_repository = (
        fork_repository
        if event == "pull_request" and pypto_ref == "main"
        else config["sources"]["pypto"]["repository"]
    )
    assert f"PYPTO_REF={expected_sha}" in build
    assert f"PYPTO_REPO={expected_repository}" in build
    assert f"PYPTO_LIB_REF={library}" in build
    assert f"CANN_IMAGE={base}" in build
    assert "DIST_BUILD_JOBS=1" in build
    assert f"DIST_RUNTIME_TARGETS=build_package_{image['validation_platform']};build_package_sim" in build
    assert f"CANN_TAG={image['cann_tag']}" in build
    assert f"DIST_PATCH={image['patch']}" in build
    assert "HTTP_PROXY=http://proxy.example:7892" in build
    assert "--load" in build and "--push" not in build
    assert not (tmp_path / "validated.json").exists()
    record = json.loads((tmp_path / "state.json").read_text())
    assert record["base"] == base
    assert record["pypto"] == expected_sha
    assert record["pypto_repository"] == expected_repository


@pytest.mark.parametrize("repository", [None, "file:///tmp/pypto", "https://github.com@other.example/repo"])
def test_pr_build_rejects_missing_or_invalid_repository_before_network(
    ci: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, repository: str | None
) -> None:
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.delenv("PYPTO_PR_REPOSITORY", raising=False)
    if repository is not None:
        monkeypatch.setenv("PYPTO_PR_REPOSITORY", repository)
    monkeypatch.setattr(ci, "preflight", lambda *args: None)
    calls = []

    def command(argv: list[str], **kwargs: object) -> str:
        calls.append(argv)
        return "1" * 40

    monkeypatch.setattr(ci, "run", command)
    with pytest.raises(ValueError, match="PYPTO_PR_REPOSITORY"):
        ci.build(ci.load_config(), "arm-a2a3", tmp_path)
    assert calls == [["git", "-C", str(ci.ROOT), "rev-parse", "HEAD"]]


@pytest.mark.parametrize("case,devices", [("decode", "5"), ("distributed", "5,7")])
@pytest.mark.parametrize("image_name", ["arm-a2a3", "arm-a5", "x86-a5"])
def test_devices_expand_only_inside_queued_shell(
    ci: ModuleType, state: dict, tmp_path: Path, case: str, devices: str, image_name: str
) -> None:
    state["image"] = ci.load_config()["images"][image_name]
    # Execute the generated shell with a harmless Docker stand-in. Spaces in the
    # evidence path and physical-vs-logical device IDs must survive both shells.
    directory = tmp_path / "evidence with spaces"
    directory.mkdir()
    commands = tmp_path / "commands"
    commands.mkdir()
    docker = commands / "docker"
    docker.write_text('#!/bin/bash\nprintf "%s\\n" "$@" >> "$CALLS"\n')
    docker.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        PATH=f"{commands}:{environment['PATH']}", TASK_DEVICE=devices, CALLS=str(tmp_path / "calls")
    )
    argv = ci.task_command(state, directory, case)
    assert argv[argv.index("--device") + 1] == "auto"
    assert argv[argv.index("--device-num") + 1] == ("2" if case == "distributed" else "1")
    result = subprocess.run(["bash", "-c", argv[-1]], env=environment, check=False)
    assert result.returncode == 0
    calls = (tmp_path / "calls").read_text()
    assert f"ASCEND_VISIBLE_DEVICES={devices}" in calls
    assert state["image_id"] in calls
    assert ("pypto-doctor --strict" in calls) == (case == "decode")
    assert "-d 0,1" in calls if case == "distributed" else "-d 0" in calls
    assert f"-p {state['image']['validation_platform']}" in calls
    assert "PYPTO_BENCH=0" in calls
    if state["image"]["validation_platform"] == "a5":
        script = "decode_moe.py --ep 2" if case == "distributed" else "decode_swa.py"
        assert f"/workspace/pypto-lib/models/deepseek_v4_1_flash/{script} -p a5" in calls
    else:
        script = (
            "examples/advanced/allreduce.py" if case == "distributed" else "models/qwen3_14b/decode_fwd.py"
        )
        assert f"/workspace/pypto-lib/{script} -p a2a3" in calls
    assert ("--device\n/dev/ummu" in calls) == (image_name == "arm-a5")
    assert ("--device\n/dev/uburma" in calls) == (image_name == "arm-a5")
    assert f"rm\n-f\npypto-ci-{state['identity']}-{case}" in calls


@pytest.mark.parametrize(
    "failure", ["decode-exit", "distributed-exit", "missing-pass", "reported-fail", "manifest"]
)
@pytest.mark.parametrize("image_name", ["arm-a2a3", "arm-a5", "x86-a5"])
def test_failed_validation_cannot_publish(
    ci: ModuleType,
    state: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    image_name: str,
) -> None:
    # A previous successful marker must not survive a revalidation failure.
    state["image"] = ci.load_config()["images"][image_name]
    ci.save(tmp_path / "state.json", state)
    ci.save(tmp_path / "validated.json", state)

    def command(argv: list[str], log: Path | None = None, **kwargs: object) -> str:
        if log and log.stem + "-exit" == failure:
            raise RuntimeError("test failed")
        if argv[0] == "task-submit":
            if failure == "reported-fail":
                return "[RUN] FAIL\n[RUN] PASS\n"
            return "no test result" if failure == "missing-pass" else "[RUN] PASS\n"
        return f"pypto wrong\npypto-lib {state['pypto_lib']}"

    monkeypatch.setattr(ci, "run", command)
    with pytest.raises((ValueError, RuntimeError)):
        ci.validate(tmp_path)
    assert not (tmp_path / "validated.json").exists()
    with pytest.raises(FileNotFoundError):
        ci.publish(tmp_path)


def test_successful_validation_records_exact_source_and_patch(
    ci: ModuleType, state: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_record = {"pypto": state["pypto"], "runtime": "d" * 40, "patch_sha256": "e" * 64}

    def command(argv: list[str], log: Path | None = None) -> str:
        if argv[0] == "task-submit":
            return "\x1b[32m[RUN] PASS\x1b[0m\n"
        # The real image's Bash activation prints a banner when CANN sets PS1.
        # Metadata must remain parseable even with that default entrypoint.
        banner = "PyPTO dev environment\n" if "--entrypoint" not in argv else ""
        if argv[-1].endswith("dist-state.json"):
            return banner + json.dumps(build_record)
        return (
            f"{banner}pypto {state['pypto']}\npypto-lib {state['pypto_lib']}\n"
            f"runtime {build_record['runtime']}\n"
        )

    monkeypatch.setattr(ci, "run", command)
    ci.validate(tmp_path)
    verified = json.loads((tmp_path / "validated.json").read_text())
    assert verified["image_id"] == state["image_id"]
    assert verified["manifest"]["runtime"] == build_record["runtime"]
    assert json.loads((tmp_path / "dist-state.json").read_text()) == build_record


@pytest.fixture
def publication(
    ci: ModuleType, state: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> SimpleNamespace:
    ci.save(tmp_path / "validated.json", state)
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.setenv("GITHUB_REPOSITORY", "hw-native-sys/pypto")
    monkeypatch.setenv("SWR_USERNAME", "test-user")
    monkeypatch.setenv("SWR_PASSWORD", "test-password")
    monkeypatch.setattr(ci, "login", lambda registry: None)
    candidate = SimpleNamespace(digest="sha256:" + "d" * 64, config_digest=state["image_id"], content=b"new")
    tags = {}
    calls = []
    mutations = []

    def put(tag: str, manifest: SimpleNamespace) -> None:
        mutations.append(("PUT", tag, manifest.digest))
        tags[tag] = manifest

    def delete_tag(tag: str) -> None:
        mutations.append(("DELETE", tag))
        tags.pop(tag, None)

    registry = SimpleNamespace(get=tags.get, put=put, delete_tag=delete_tag)
    monkeypatch.setattr(ci, "Registry", lambda *args: registry)

    def command(argv: list[str], *args: object, **kwargs: object) -> str:
        calls.append(argv)
        if argv[0] == "git":
            assert argv[1:] == [
                "ls-remote",
                "--exit-code",
                "https://github.com/hw-native-sys/pypto.git",
                "refs/heads/main",
            ]
            return f"{state['recipe']}\trefs/heads/main"
        if argv[1] == "push":
            assert kwargs.get("direct") is True
            tags[state["unique"].rsplit(":", 1)[1]] = candidate
        return json.dumps([{"RepoDigests": [f"registry.example/pypto@{candidate.digest}"]}])

    monkeypatch.setattr(ci, "run", command)
    return SimpleNamespace(
        tags=tags, calls=calls, mutations=mutations, registry=registry, candidate=candidate
    )


@pytest.mark.parametrize("stage", [1, 2])
@pytest.mark.parametrize("failure", ["stale", "unreachable", "missing-branch"])
def test_publication_preserves_retained_tags_when_main_cannot_be_verified(
    ci: ModuleType,
    state: dict,
    tmp_path: Path,
    publication: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    stage: int,
    failure: str,
) -> None:
    retained = {
        state["rolling"].rsplit(":", 1)[1]: SimpleNamespace(digest="old", content=b"old"),
        state["previous"].rsplit(":", 1)[1]: SimpleNamespace(digest="older", content=b"older"),
    }
    publication.tags.update(retained)
    command = ci.run
    checks = 0
    logins = []
    monkeypatch.setattr(ci, "login", logins.append)

    def run(argv: list[str], *args: object, **kwargs: object) -> str:
        nonlocal checks
        if argv[0] == "git":
            checks += 1
            if checks == stage:
                if failure == "unreachable":
                    raise RuntimeError("git failed: upstream unreachable")
                return "" if failure == "missing-branch" else f"{'e' * 40}\trefs/heads/main"
        return command(argv, *args, **kwargs)

    monkeypatch.setattr(ci, "run", run)
    with pytest.raises((ValueError, RuntimeError), match="upstream"):
        ci.publish(tmp_path)
    assert checks == stage
    assert publication.tags == retained
    assert not (tmp_path / "published.json").exists()
    if stage == 1:
        assert not logins
        assert not publication.calls
        assert not publication.mutations
    else:
        assert logins == [state["registry"]]
        assert publication.mutations == [("DELETE", state["unique"].rsplit(":", 1)[1])]


@pytest.mark.parametrize("existing", [False, True])
def test_only_latest_and_previous_remain_after_publication(
    ci: ModuleType, state: dict, tmp_path: Path, publication: SimpleNamespace, existing: bool
) -> None:
    latest = state["rolling"].rsplit(":", 1)[1]
    previous = state["previous"].rsplit(":", 1)[1]
    old = SimpleNamespace(digest="sha256:" + "e" * 64, content=b"old")
    if existing:
        publication.tags[latest] = old
    ci.publish(tmp_path)
    assert [argv for argv in publication.calls if argv[1] == "tag"] == [
        ["docker", "tag", state["image_id"], state["unique"]],
    ]
    assert set(publication.tags) == ({latest, previous} if existing else {latest})
    assert publication.tags[latest] is publication.candidate
    if existing:
        assert publication.tags[previous] is old
        assert (tmp_path / "before-latest.json").read_bytes() == b"old"
    assert (
        json.loads((tmp_path / "published.json").read_text())["published_digest"]
        == publication.candidate.digest
    )


@pytest.mark.parametrize("mismatch", ["config", "manifest"])
def test_wrong_registry_digest_prevents_rolling_update(
    ci: ModuleType,
    state: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    publication: SimpleNamespace,
    mismatch: str,
) -> None:
    if mismatch == "config":
        publication.candidate.config_digest = "sha256:" + "f" * 64
    else:
        original = ci.run

        def command(argv: list[str], *args: object, **kwargs: object) -> str:
            output = original(argv, *args, **kwargs)
            return '[{"RepoDigests": []}]' if argv[1:3] == ["image", "inspect"] else output

        monkeypatch.setattr(ci, "run", command)
    with pytest.raises(ValueError, match="Published digest"):
        ci.publish(tmp_path)
    assert not any(item[0] == "PUT" for item in publication.mutations)
    assert not publication.tags


def test_retry_does_not_rotate_previous_and_legacy_tags_are_untouched(
    ci: ModuleType, state: dict, tmp_path: Path, publication: SimpleNamespace
) -> None:
    latest = state["rolling"].rsplit(":", 1)[1]
    previous = state["previous"].rsplit(":", 1)[1]
    old = SimpleNamespace(digest="sha256:" + "e" * 64, content=b"old")
    publication.tags.update({latest: publication.candidate, previous: old, "a3-dist": old})
    ci.publish(tmp_path)
    assert publication.tags == {latest: publication.candidate, previous: old, "a3-dist": old}
    assert not any(item[0] == "PUT" for item in publication.mutations)


def test_failed_previous_promotion_preserves_latest(
    ci: ModuleType, state: dict, tmp_path: Path, publication: SimpleNamespace
) -> None:
    latest = state["rolling"].rsplit(":", 1)[1]
    old = SimpleNamespace(digest="sha256:" + "e" * 64, content=b"old")
    publication.tags[latest] = old

    def fail(tag: str, manifest: object) -> None:
        raise RuntimeError("registry unavailable")

    publication.registry.put = fail
    with pytest.raises(RuntimeError, match="unavailable"):
        ci.publish(tmp_path)
    assert publication.tags == {latest: old}
    assert not (tmp_path / "published.json").exists()


def test_publication_checks_retained_tags_after_cleanup(
    ci: ModuleType, state: dict, tmp_path: Path, publication: SimpleNamespace
) -> None:
    original = publication.registry.delete_tag

    def broken_delete(tag: str) -> None:
        original(tag)
        publication.tags.pop(state["rolling"].rsplit(":", 1)[1])

    publication.registry.delete_tag = broken_delete
    with pytest.raises(ValueError, match="Latest tag changed"):
        ci.publish(tmp_path)
    assert not (tmp_path / "published.json").exists()


def test_swr_subprocess_environment_bypasses_proxy(ci: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.setenv(key, "http://proxy.invalid")
    environment = ci.direct_environment()
    assert environment["NO_PROXY"] == environment["no_proxy"] == "*"
    assert not any(key.lower() in ("http_proxy", "https_proxy", "all_proxy") for key in environment)


def test_registry_checks_manifest_bytes_and_never_deletes_digest(
    ci: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = ci.Registry.__new__(ci.Registry)
    content = json.dumps(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": {"digest": "sha256:" + "a" * 64},
        }
    ).encode()
    digest = "sha256:" + hashlib.sha256(content).hexdigest()
    original = registry.request
    monkeypatch.setattr(registry, "request", lambda *args: (200, content, digest))
    manifest = registry.get("latest")
    assert manifest.digest == digest
    monkeypatch.setattr(registry, "request", lambda *args: (200, content, "sha256:" + "b" * 64))
    with pytest.raises(ValueError, match="digest mismatch"):
        registry.get("latest")
    with pytest.raises(ValueError, match="never a digest"):
        original("DELETE", digest)


@pytest.mark.parametrize(
    "status,code,missing",
    [
        (404, "MANIFEST_UNKNOWN", True),
        (401, "UNAUTHORIZED", False),
        (500, "UNKNOWN", False),
        (404, "UNKNOWN", False),
    ],
)
def test_registry_only_treats_manifest_not_found_as_absent(
    ci: ModuleType, status: int, code: str, missing: bool
) -> None:
    registry = ci.Registry.__new__(ci.Registry)
    registry.base = "https://registry.example"
    registry.repository = "pypto"
    registry.token = "test"

    def fail(request: object, **kwargs: object) -> None:
        raise urllib.error.HTTPError(
            "https://registry.example",
            status,
            "failure",
            Message(),
            io.BytesIO(json.dumps({"errors": [{"code": code}]}).encode()),
        )

    registry.opener = SimpleNamespace(open=fail)
    if missing:
        assert registry.get("latest") is None
    else:
        with pytest.raises(RuntimeError, match=f"HTTP {status}"):
            registry.get("latest")


@pytest.mark.parametrize("accepted", [True, False])
def test_temporary_tag_deletion_preserves_shared_manifest(
    ci: ModuleType, monkeypatch: pytest.MonkeyPatch, accepted: bool
) -> None:
    registry = ci.Registry.__new__(ci.Registry)
    shared = SimpleNamespace(digest="sha256:" + "a" * 64)
    tags = {"candidate": shared, "latest": shared}
    calls = []

    def request(method: str, tag: str) -> tuple[int, bytes, str]:
        calls.append((method, tag))
        if accepted:
            tags.pop(tag)
        return (202 if accepted else 405), b"", ""

    monkeypatch.setattr(registry, "get", tags.get)
    monkeypatch.setattr(registry, "request", request)
    if accepted:
        registry.delete_tag("candidate")
        assert tags == {"latest": shared}
    else:
        with pytest.raises(RuntimeError, match="temporary tag"):
            registry.delete_tag("candidate")
        assert tags == {"candidate": shared, "latest": shared}
    assert calls == [("DELETE", "candidate")]


@pytest.mark.parametrize("reason", ["different-image", "branch", "fork"])
def test_publication_rejects_unvalidated_or_untrusted_run(
    ci: ModuleType, state: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    monkeypatch.setenv("GITHUB_REF", "refs/heads/topic" if reason == "branch" else "refs/heads/main")
    monkeypatch.setenv("GITHUB_REPOSITORY", "someone/pypto" if reason == "fork" else "hw-native-sys/pypto")
    if reason == "different-image":
        state["image_id"] = "sha256:" + "e" * 64
    ci.save(tmp_path / "validated.json", state)
    with pytest.raises(ValueError):
        ci.publish(tmp_path)


def test_cleanup_removes_only_this_runs_resources(
    ci: ModuleType, state: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def command(argv: list[str], **kwargs: object) -> str:
        calls.append(argv)
        return "task-42" if argv[:2] == ["task-submit", "--find"] else ""

    monkeypatch.setattr(ci, "run", command)
    ci.cleanup(tmp_path)
    assert calls[:2] == [
        ["docker", "rm", "-f", f"pypto-ci-{state['identity']}-decode"],
        ["docker", "rm", "-f", f"pypto-ci-{state['identity']}-distributed"],
    ]
    assert ["task-submit", "--find", str(tmp_path)] in calls
    assert ["task-submit", "--kill", "task-42"] in calls
    assert not any("prune" in argv for argv in calls)


@pytest.fixture(params=["cann92-a2a3-aicpu-class-memaccess", "cann92-a5-aicpu-class-memaccess"])
def patch_tree(tmp_path: Path, request: pytest.FixtureRequest) -> tuple[ModuleType, Path, Path]:
    module = load_module("dist_state")
    patch_name: str = getattr(request, "param")
    source = tmp_path / "pypto"
    runtime = source / "runtime"
    runtime.mkdir(parents=True)
    module.git(runtime, "init")
    original = (
        "target_compile_options(aicpu_kernel\n    PRIVATE\n"
        "        -fPIC\n        $<$<BOOL:${WERROR}>:-Werror>\n"
        "        -Wno-error=variadic-macros\n        -fno-common\n"
        "        -g\n        $<$<COMPILE_LANGUAGE:CXX>:-std=gnu++17>\n)\n"
    )
    target = runtime / module.PATCH_FILES[patch_name]
    target.parent.mkdir(parents=True)
    target.write_text(original)
    module.git(runtime, "add", ".")
    module.git(
        runtime, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "init"
    )
    patch = ROOT / "docker/patches" / f"{patch_name}.patch"
    return module, source, patch


def test_patch_applies_and_rejects_unrelated_dirty_source(patch_tree: tuple) -> None:
    module, source, patch = patch_tree
    module.apply_patch(source, patch)
    assert "-Wno-error=class-memaccess" in (source / "runtime" / module.PATCH_FILES[patch.stem]).read_text()
    with pytest.raises(ValueError, match="clean"):
        module.apply_patch(source, patch)


def test_already_upstream_patch_is_not_applied_twice(patch_tree: tuple) -> None:
    module, source, patch = patch_tree
    module.apply_patch(source, patch)
    runtime = source / "runtime"
    module.git(
        runtime, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-am", "fixed"
    )
    module.apply_patch(source, patch)
    assert not module.git(runtime, "status", "--porcelain")


def test_changed_patch_context_is_a_hard_failure(patch_tree: tuple) -> None:
    module, source, patch = patch_tree
    runtime = source / "runtime"
    (runtime / module.PATCH_FILES[patch.stem]).write_text("new build system\n")
    module.git(
        runtime, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-am", "changed"
    )
    with pytest.raises(subprocess.CalledProcessError):
        module.apply_patch(source, patch)


@pytest.mark.parametrize("change", ["extension", "source", "untracked", "off-pin", "patch"])
def test_build_record_detects_later_changes(
    patch_tree: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    module, source, patch = patch_tree
    module.git(source, "init")
    module.git(source, "add", "runtime")
    module.git(source, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "pin")
    module.apply_patch(source, patch)
    home = tmp_path / "home"
    (home / "image-tools").mkdir(parents=True)
    copied_patch = home / "image-tools" / patch.name
    copied_patch.write_bytes(patch.read_bytes())
    extension = tmp_path / "_task_interface.so"
    extension.write_bytes(b"compiled extension")
    monkeypatch.setattr(
        module.importlib.util, "find_spec", lambda name: SimpleNamespace(origin=str(extension))
    )
    module.record(source, home, patch.stem)
    module.verify(source, home)
    if change == "extension":
        extension.write_bytes(b"different build")
    elif change == "source":
        (source / "runtime" / module.PATCH_FILES[patch.stem]).write_text("uncommitted change")
    elif change == "untracked":
        (source / "runtime/new.cpp").write_text("new source")
    elif change == "patch":
        copied_patch.write_text("modified patch")
    else:
        module.git(
            source / "runtime",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-am",
            "other",
        )
    with pytest.raises(ValueError):
        module.verify(source, home)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
