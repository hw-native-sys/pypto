# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Build, queue hardware validation, and publish a configured development image.

Only Python's standard library is required on the runner. All subprocess inputs
are argv lists; the task-submit shell boundary is quoted separately, leaving only
TASK_DEVICE to expand after the scheduler has allocated cards.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from dist_state import PATCH_FILES
from swr_registry import Registry

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "docker/images.json"
SHA = re.compile(r"[0-9a-f]{40}")
DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
GITHUB_REPOSITORY_URL = re.compile(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")


def direct_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.lower() not in ("http_proxy", "https_proxy", "all_proxy")
    }
    environment.update(NO_PROXY="*", no_proxy="*")
    return environment


def run(argv: list[str], log: Path | None = None, *, check: bool = True, direct: bool = False) -> str:
    environment = direct_environment() if direct else os.environ.copy()
    proxy = environment.get("PYPTO_DOCKER_PROXY", "")
    if argv[0] == "git" and proxy:
        environment.update(dict.fromkeys(("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"), proxy))
    if log is None:
        result = subprocess.run(argv, text=True, capture_output=True, check=False, env=environment)
        if check and result.returncode:
            raise RuntimeError(f"{argv[0]} failed ({result.returncode}): {result.stderr.strip()}")
        return result.stdout.strip()
    with log.open("w") as output:
        with subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=environment
        ) as process:
            if process.stdout is None:
                raise RuntimeError(f"Cannot capture {argv[0]} output")
            for line in process.stdout:
                print(line, end="", flush=True)
                output.write(line)
            code = process.wait()
    if check and code:
        raise RuntimeError(f"{argv[0]} failed ({code}); see {log}")
    return log.read_text()


def save(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    config = json.loads(path.read_text())
    if config["schema_version"] != 1:
        raise ValueError("Unsupported images.json schema_version")
    for key in ("registry", "repository"):
        if not re.fullmatch(r"[a-z0-9][a-z0-9./:-]*", config[key]):
            raise ValueError(f"Invalid {key}: {config[key]!r}")
    for source in config["sources"].values():
        if not GITHUB_REPOSITORY_URL.fullmatch(source["repository"]):
            raise ValueError("Source repositories must be public GitHub HTTPS URLs")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./-]*", source["ref"]):
            raise ValueError(f"Invalid source ref: {source['ref']!r}")
    tags: set[str] = set()
    for name, image in config["images"].items():
        validate_image(name, image)
        image_tags = {image["tag"], f"{image['tag']}-previous"}
        if tags.intersection(image_tags) or not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,79}", image["tag"]):
            raise ValueError(f"Duplicate or invalid image tag (including rollback tags): {image['tag']}")
        tags.update(image_tags)
    return config


def validate_image(name: str, image: dict[str, Any]) -> None:
    if not re.fullmatch(r"(?:arm|x86)-(?:a2a3|a5)", name):
        raise ValueError(f"Invalid image name: {name!r}")
    expected_platform = "linux/arm64" if name.startswith("arm-") else "linux/amd64"
    if image["platform"] != expected_platform or image["validation_platform"] != name.split("-")[1]:
        raise ValueError(f"Architecture/platform mismatch for {name}")
    if type(image["enabled"]) is not bool:
        raise ValueError(f"enabled must be boolean for {name}")
    if type(image["proxy_required"]) is not bool:
        raise ValueError(f"proxy_required must be boolean for {name}")
    if not isinstance(image["runners"], list) or "self-hosted" not in image["runners"]:
        raise ValueError(f"Missing self-hosted runner labels for {name}")
    for key in ("build_jobs", "task_timeout_seconds", "task_max_seconds"):
        if type(image[key]) is not int or image[key] <= 0:
            raise ValueError(f"{name}.{key} must be a positive integer")
    if image["task_timeout_seconds"] <= image["task_max_seconds"]:
        raise ValueError(f"{name}.task_timeout_seconds must exceed task_max_seconds to allow queueing")
    for key in ("cann_repository", "cann_tag", "tag", "driver_version"):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./:-]*", image[key]):
            raise ValueError(f"Invalid {name}.{key}")
    if image["cann_digest"] is not None and not DIGEST.fullmatch(image["cann_digest"]):
        raise ValueError(f"Invalid CANN digest for {name}")
    if image["patch"] and image["patch"] not in PATCH_FILES:
        raise ValueError(f"Unsupported runtime patch for {name}")
    if image["patch"] and (
        not image["cann_tag"].startswith("9.2.0-")
        or image["patch"] != f"cann92-{image['validation_platform']}-aicpu-class-memaccess"
    ):
        raise ValueError(f"The CANN 9.2 patch must match the image platform: {name}")
    validate_cases(name, image)


def validate_cases(name: str, image: dict[str, Any]) -> None:
    """Check the single-card and distributed case paths and extra arguments."""
    for key in ("decode", "distributed"):
        path_value = image[key]
        if not path_value.startswith("/workspace/pypto-lib/") or ".." in Path(path_value).parts:
            raise ValueError(f"{name}.{key} must be a path inside pypto-lib")
    arguments = image.get("distributed_args", [])
    if not isinstance(arguments, list) or any(
        not isinstance(value, str) or "\0" in value for value in cast(list[object], arguments)
    ):
        raise ValueError(f"{name}.distributed_args must be a list of strings without NUL characters")


def resolve_ref(repository: str, ref: str) -> str:
    if SHA.fullmatch(ref):
        return ref
    # Ask for exact refs so a branch and a similarly named tag cannot be mixed.
    for candidate in (f"refs/heads/{ref}", f"refs/tags/{ref}^{{}}", f"refs/tags/{ref}"):
        output = run(["git", "ls-remote", "--exit-code", repository, candidate], check=False)
        if output:
            value = output.split()[0]
            if SHA.fullmatch(value):
                return value
    raise ValueError(f"Cannot resolve {repository} ref {ref}")


def preflight(name: str, image: dict[str, Any]) -> None:
    if image["proxy_required"] and not os.environ.get("PYPTO_DOCKER_PROXY"):
        raise ValueError("Set the PYPTO_DOCKER_PROXY environment variable before building this image")
    if not image["enabled"]:
        raise ValueError(f"Image {name} is disabled")
    if platform.machine() != {"linux/arm64": "aarch64", "linux/amd64": "x86_64"}[image["platform"]]:
        raise ValueError(f"Build {name} on its native architecture: {image['platform']}")
    runner = os.environ.get("RUNNER_NAME")
    if runner and image.get("runner_name") and runner != image["runner_name"]:
        raise ValueError(f"Expected runner {image['runner_name']}, got {runner}")
    for executable in ("docker", "git", "task-submit"):
        if shutil.which(executable) is None:
            raise ValueError(f"Required command is missing: {executable}")
    # CLI plugins can disappear when the job selects a fresh DOCKER_CONFIG.
    run(["docker", "buildx", "version"])
    # The host driver metadata is evidence, not a caller-supplied version claim.
    driver_file = Path("/usr/local/Ascend/driver/version.info")
    driver = re.search(r"^Version=(.+)$", driver_file.read_text(), re.MULTILINE)
    if driver is None or driver[1].strip() != image["driver_version"]:
        raise ValueError(f"Host driver does not match configured {image['driver_version']}")


def build(config: dict[str, Any], name: str, directory: Path) -> None:
    image = config["images"][name]
    preflight(name, image)
    for filename in ("state.json", "validated.json", "published.json"):
        (directory / filename).unlink(missing_ok=True)
    recipe_sha = run(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
    sources = config["sources"]
    # Keep the recipe and PyPTO helpers on the same event snapshot. PR jobs
    # check out the exact head commit, which may exist only in the fork.
    pypto_repository = sources["pypto"]["repository"]
    if sources["pypto"]["ref"] == "main" and os.environ.get("GITHUB_EVENT_NAME") == "pull_request":
        pypto_repository = os.environ.get("PYPTO_PR_REPOSITORY", "")
        if not GITHUB_REPOSITORY_URL.fullmatch(pypto_repository):
            raise ValueError("PR builds require PYPTO_PR_REPOSITORY to be a public GitHub HTTPS URL")
    pypto_sha = recipe_sha if sources["pypto"]["ref"] == "main" else resolve_ref(**sources["pypto"])
    lib_sha = resolve_ref(**sources["pypto_lib"])
    base = image["cann_repository"]
    base += f"@{image['cann_digest']}" if image["cann_digest"] else f":{image['cann_tag']}"
    # A successful cached build must never hide an inaccessible/nonexistent base.
    run(["docker", "pull", "--platform", image["platform"], base], directory / "pull.log")
    details = json.loads(run(["docker", "image", "inspect", base]))[0]
    if f"{details['Os']}/{details['Architecture']}" != image["platform"]:
        raise ValueError("Pulled CANN architecture does not match the configured target")
    digests = [value for value in details["RepoDigests"] if value.startswith(image["cann_repository"] + "@")]
    if not digests:
        raise ValueError("Pulled CANN image has no repository digest")
    if image["cann_digest"]:
        # RepoDigests can include an index and a platform manifest for the same
        # repository. Preserve the explicitly requested digest instead of taking
        # whichever entry Docker happened to return first.
        if base not in digests:
            raise ValueError("Pulled CANN digest differs from the configured pin")
    else:
        base = digests[0]
    run_id = os.environ.get("GITHUB_RUN_ID", datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"))
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "1")
    if not re.fullmatch(r"[0-9]+", run_id) or not re.fullmatch(r"[0-9]+", attempt):
        raise ValueError("Run ID and attempt must be numeric")
    identity = f"{name}-{run_id}-{attempt}"
    local = f"pypto-ci:{identity}"
    repository = f"{config['registry']}/{config['repository']}"
    unique = f"{repository}:{image['tag']}-{run_id}-{attempt}"
    state = {
        "name": name,
        "image": image,
        "identity": identity,
        "local": local,
        "registry": config["registry"],
        "unique": unique,
        "rolling": f"{repository}:{image['tag']}",
        "previous": f"{repository}:{image['tag']}-previous",
        "pypto": pypto_sha,
        "pypto_repository": pypto_repository,
        "pypto_lib": lib_sha,
        "recipe": recipe_sha,
        "base": base,
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
    }
    save(directory / "state.json", state)
    jobs = min(image["build_jobs"], int(os.environ.get("PYPTO_BUILD_JOBS", "2")))
    if jobs < 1:
        raise ValueError("PYPTO_BUILD_JOBS must be positive")
    arguments = {
        "CANN_IMAGE": base,
        "CANN_TAG": image["cann_tag"],
        "PYPTO_REF": pypto_sha,
        "PYPTO_LIB_REF": lib_sha,
        "PYPTO_REPO": pypto_repository,
        "PYPTO_LIB_REPO": sources["pypto_lib"]["repository"],
        "DIST_BUILD_JOBS": str(jobs),
        "DIST_PATCH": image["patch"],
        "DIST_RUNTIME_TARGETS": f"build_package_{image['validation_platform']};build_package_sim",
    }
    proxy = os.environ.get("PYPTO_DOCKER_PROXY", "")
    if proxy:
        arguments.update(dict.fromkeys(("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"), proxy))
    command = [
        "docker",
        "buildx",
        "build",
        "--builder",
        "default",
        "--network=host",
        "--load",
        "--platform",
        image["platform"],
        "--target",
        "dist",
        "--file",
        str(ROOT / "docker/pypto_dev.Dockerfile"),
        "--tag",
        local,
        "--label",
        f"org.opencontainers.image.revision={pypto_sha}",
        "--label",
        f"org.pypto.image.recipe={recipe_sha}",
    ]
    for key, value in arguments.items():
        command += ["--build-arg", f"{key}={value}"]
    run(command + [str(ROOT)], directory / "build.log")
    state["image_id"] = run(["docker", "image", "inspect", local, "--format", "{{.Id}}"])
    save(directory / "state.json", state)


def task_command(state: dict[str, Any], directory: Path, case: str) -> list[str]:
    image = state["image"]
    count = 2 if case == "distributed" else 1
    container = f"pypto-ci-{state['identity']}-{case}"
    model = shlex.join(
        [
            "python",
            image[case],
            *(image.get("distributed_args", []) if count == 2 else []),
            "-p",
            image["validation_platform"],
            "-d",
            "0,1" if count == 2 else "0",
        ]
    )
    inner = f"set -eo pipefail; {model}"
    if case == "decode":
        # A pinned older PyPTO ref can install an older doctor that silently
        # ignores unknown options. Require the capability before trusting it.
        inner = (
            "set -eo pipefail; pypto-doctor --help | grep -F -- --strict >/dev/null; "
            f"pypto-doctor --strict; {model}"
        )
    docker = [
        "docker",
        "run",
        "--rm",
        "--name",
        container,
        "--label",
        f"org.pypto.ci.run={state['identity']}",
        "--runtime=ascend",
        "--shm-size=16g",
        "-e",
        "PYPTO_BENCH=0",
        "--mount",
        "type=bind,src=/usr/local/Ascend/driver/version.info,dst=/run/pypto-driver-version.info,readonly",
        "-e",
        "PYPTO_DRIVER_VERSION_FILE=/run/pypto-driver-version.info",
    ]
    if image["platform"] == "linux/arm64" and image["validation_platform"] == "a5":
        # ARM Ascend950 images use URMA for RoCE communication.
        docker += ["--device", "/dev/ummu", "--device", "/dev/uburma"]
    # The trap removes the container before the queued shell releases its card.
    # always() cleanup is an additional fallback for cancellation/disconnection.
    command = (
        f"cd {shlex.quote(str(directory.resolve()))} || exit; "
        f"trap {shlex.quote(f'docker rm -f {container} >/dev/null 2>&1 || true')} EXIT; "
        f'{shlex.join(docker)} -e ASCEND_VISIBLE_DEVICES="$TASK_DEVICE" '
        f"{shlex.join([state['image_id'], '-c', inner])}"
    )
    return [
        "task-submit",
        "--device",
        "auto",
        "--device-num",
        str(count),
        "--timeout",
        str(image["task_timeout_seconds"]),
        "--max-time",
        str(image["task_max_seconds"]),
        "--run",
        command,
    ]


def validate(directory: Path) -> None:
    (directory / "validated.json").unlink(missing_ok=True)
    state = json.loads((directory / "state.json").read_text())
    for case in ("decode", "distributed"):
        output = run(task_command(state, directory, case), directory / f"{case}.log")
        # task-submit may echo the submitted command: require an actual PASS
        # line, not a substring embedded in a command or an error explanation.
        clean = re.sub(r"\x1b\[[0-9;]*m", "", output)
        if re.search(r"^\s*\[RUN\] FAIL\b", clean, re.MULTILINE):
            raise ValueError(f"{case} reported [RUN] FAIL")
        if not re.search(r"^\s*\[RUN\] PASS\b", clean, re.MULTILINE):
            raise ValueError(f"{case} exited without a [RUN] PASS result")
    # Read data without activating the shell: CANN can set PS1 even for bash -c,
    # causing the development banner to precede the manifest and JSON record.
    read_file = ["docker", "run", "--rm", "--entrypoint", "/bin/cat", state["image_id"]]
    manifest = run(read_file + ["/opt/pypto/dist-manifest"])
    (directory / "dist-manifest.txt").write_text(manifest + "\n")
    entries = dict(line.split(maxsplit=1) for line in manifest.splitlines())
    if entries["pypto"] != state["pypto"] or entries["pypto-lib"] != state["pypto_lib"]:
        raise ValueError("Image source manifest differs from the requested revisions")
    build_record = json.loads(run(read_file + ["/opt/pypto/dist-state.json"]))
    if build_record["pypto"] != state["pypto"] or build_record["runtime"] != entries["runtime"]:
        raise ValueError("Image patch/build record differs from the source manifest")
    save(directory / "dist-state.json", build_record)
    state["manifest"] = entries
    save(directory / "validated.json", state)


def login(registry: str) -> None:
    password = os.environ.get("SWR_PASSWORD", "")
    username = os.environ.get("SWR_USERNAME", "")
    if not username or not password:
        raise ValueError("Set SWR_USERNAME and SWR_PASSWORD secrets to publish")
    subprocess.run(
        ["docker", "login", "--username", username, "--password-stdin", registry],
        input=password + "\n",
        text=True,
        check=True,
        env=direct_environment(),
    )


def require_current_recipe(recipe: str) -> None:
    """Reject stale recipes and fail closed when upstream main cannot be verified."""
    if not SHA.fullmatch(recipe):
        raise ValueError(f"Invalid image recipe revision: {recipe!r}")
    upstream = run(
        ["git", "ls-remote", "--exit-code", "https://github.com/hw-native-sys/pypto.git", "refs/heads/main"]
    ).split()
    if upstream != [recipe, "refs/heads/main"]:
        raise ValueError(f"Image recipe {recipe} is not the current upstream main; build the latest main")


def publish(directory: Path) -> None:
    # Even direct script invocation cannot publish a build lacking both tests.
    state = json.loads((directory / "state.json").read_text())
    verified = json.loads((directory / "validated.json").read_text())
    if any(verified.get(key) != value for key, value in state.items()):
        raise ValueError("Validation record does not belong to this build")
    if (
        os.environ.get("GITHUB_REF") != "refs/heads/main"
        or os.environ.get("GITHUB_REPOSITORY") != "hw-native-sys/pypto"
    ):
        raise ValueError("Publish is restricted to the upstream main branch")
    require_current_recipe(state.get("recipe", ""))
    login(state["registry"])
    repository = state["rolling"].removeprefix(state["registry"] + "/").rsplit(":", 1)[0]
    registry = Registry(state["registry"], repository, os.environ["SWR_USERNAME"], os.environ["SWR_PASSWORD"])
    candidate_tag = state["unique"].rsplit(":", 1)[1]
    latest_tag = state["rolling"].rsplit(":", 1)[1]
    previous_tag = state["previous"].rsplit(":", 1)[1]
    current = registry.get(latest_tag)
    previous = registry.get(previous_tag)
    # Keep the exact pre-promotion manifests in the evidence for recovery if a
    # registry/network failure interrupts the two-tag rotation.
    for name, manifest in (("latest", current), ("previous", previous)):
        if manifest is not None:
            (directory / f"before-{name}.json").write_bytes(manifest.content)
    try:
        run(["docker", "tag", state["image_id"], state["unique"]])
        run(["docker", "push", state["unique"]], directory / "push-candidate.log", direct=True)
        candidate = registry.get(candidate_tag)
        if candidate is None or candidate.config_digest != state["image_id"]:
            raise ValueError("Published digest does not match the tested local image")
        local_digests = json.loads(run(["docker", "image", "inspect", state["unique"]]))[0]["RepoDigests"]
        if f"{state['registry']}/{repository}@{candidate.digest}" not in local_digests:
            raise ValueError("Published digest does not match the tested local image")
        # Main may have advanced while the image was uploading.
        require_current_recipe(state["recipe"])
        # Rerunning promotion for the same digest must not overwrite rollback.
        if current is None or current.digest != candidate.digest:
            if current is not None:
                registry.put(previous_tag, current)
            registry.put(latest_tag, candidate)
        verified["published_digest"] = candidate.digest
        previous = registry.get(previous_tag)
        verified["previous_digest"] = previous.digest if previous is not None else None
    finally:
        # Tag deletion preserves layers shared with latest/previous. Never use
        # digest deletion here, even if the registry rejects tag deletion.
        registry.delete_tag(candidate_tag)
    current = registry.get(latest_tag)
    previous = registry.get(previous_tag)
    if current is None or current.digest != verified["published_digest"]:
        raise ValueError("Latest tag changed during temporary-tag cleanup")
    if (previous.digest if previous is not None else None) != verified["previous_digest"]:
        raise ValueError("Previous tag changed during temporary-tag cleanup")
    save(directory / "published.json", verified)


def cleanup(directory: Path) -> None:
    state_path = directory / "state.json"
    if not state_path.exists():
        return
    state = json.loads(state_path.read_text())
    # Stop Docker workloads before killing the queue jobs that hold their cards.
    for case in ("decode", "distributed"):
        run(["docker", "rm", "-f", f"pypto-ci-{state['identity']}-{case}"], check=False)
    ids = run(["task-submit", "--find", str(directory.resolve())])
    for task_id in ids.splitlines():
        run(["task-submit", "--kill", task_id])
    # Unique publication tags belong to this run too. Keep the rolling local
    # tag and shared BuildKit cache, but do not accumulate one local tag per day.
    for tag in (state["local"], state["unique"]):
        run(["docker", "image", "rm", tag], check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["matrix", "build", "validate", "publish", "cleanup"])
    parser.add_argument("--image", default="arm-a2a3")
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    if args.operation == "matrix":
        config = load_config()
        print(
            json.dumps(
                {
                    "include": [
                        {"name": name, "runners": image["runners"]}
                        for name, image in config["images"].items()
                        if image["enabled"]
                    ]
                }
            )
        )
        return
    if args.directory is None:
        parser.error("--directory is required for build/validate/publish/cleanup")
    args.directory.mkdir(parents=True, exist_ok=True)
    if args.operation == "build":
        build(load_config(), args.image, args.directory)
    else:
        {"validate": validate, "publish": publish, "cleanup": cleanup}[args.operation](args.directory)


if __name__ == "__main__":
    main()
