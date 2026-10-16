# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exercise container maintenance without downloading packages or using an NPU."""

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from check_headers import PY_HEADER, check_file_header

SCRIPTS = Path(__file__).resolve().parents[2] / ".github/docker"


def executable(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/bash\nset -eu\n{body}\n")
    path.chmod(0o755)
    return path


def run(script: Path, env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", str(script), *args], env=env, text=True, capture_output=True, check=False)


@pytest.fixture
def container(tmp_path: Path) -> dict[str, str]:
    """Provide fake curl/uv programs and an isolated installation prefix."""
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("PYPTO_", "PTOAS_", "GIT_CONFIG_", "ASCEND_")) or key == "BASH_ENV":
            del env[key]
    home = tmp_path / "home"
    source = tmp_path / "source"
    commands = tmp_path / "commands"
    commands.mkdir()
    pin = source / "toolchain/versions.env"
    pin.parent.mkdir(parents=True)
    checksum = hashlib.sha256(b"wheel").hexdigest()
    pin.write_text(f"PTOAS_VERSION=v0.65\nPTOAS_SHA256_AARCH64={checksum}\nPTOAS_SHA256_X86_64={checksum}\n")
    env.update(
        ASCEND_TOOLKIT_ROOT=str(tmp_path / "missing-toolkit"),
        PYPTO_HOME=str(home),
        PYPTO_SRC=str(source),
        PYPTO_LIB_SRC=str(tmp_path / "missing-lib"),
        PTOAS_ROOT=str(home / "ptoas"),
        PATH=f"{commands}:{env['PATH']}",
        CALLS=str(tmp_path / "calls"),
    )
    executable(home / "venv/bin/python", "echo 3.10")
    executable(
        commands / "curl",
        'printf "curl %s\\n" "$*" >> "$CALLS"\nwhile [ "$1" != -o ]; do shift; done\nprintf wheel > "$2"',
    )
    executable(
        commands / "uv",
        'printf "uv %s\\n" "$*" >> "$CALLS"\n'
        'if [ "$1" = venv ]; then\n'
        '    mkdir -p "$PTOAS_ROOT/bin"\n'
        '    cp "$PYPTO_HOME/venv/bin/python" "$PTOAS_ROOT/bin/python"\n'
        'elif [[ "${*: -1}" = *.whl ]]; then\n'
        '    printf "#!/bin/bash\\necho ptoas 0.65\\n" > "$PTOAS_ROOT/bin/ptoas"\n'
        '    chmod +x "$PTOAS_ROOT/bin/ptoas"\n'
        "fi",
    )
    return env


def test_setup_installs_shared_helpers(container: dict[str, str]) -> None:
    constraints = Path(container["PYPTO_SRC"]) / "build-constraints.txt"
    constraints.write_text("nanobind==2.15.0\n")
    result = run(SCRIPTS / "pypto-setup.sh", container)
    assert result.returncode == 0, result.stderr
    home = Path(container["PYPTO_HOME"])
    assert (home / "bin/pypto-build").read_text() == (SCRIPTS / "pypto-build").read_text()
    assert (home / "toolchain/build-constraints.txt").read_text() == constraints.read_text()
    calls = Path(container["CALLS"]).read_text()
    assert "--constraint" in calls
    assert "ptoas-0.65-" in calls


@pytest.mark.parametrize("shebang", ["", "#!/bin/bash\n"])
def test_script_license_headers(tmp_path: Path, shebang: str) -> None:
    script = tmp_path / "example.sh"
    script.write_text(f"{shebang}{PY_HEADER}\necho hello\n")
    assert check_file_header(script)[0]


@pytest.mark.parametrize("arch", ["aarch64", "x86_64"])
def test_install_pinned_wheel(container: dict[str, str], arch: str) -> None:
    executable(Path(container["PATH"].split(":")[0]) / "uname", f"echo {arch}")
    result = run(SCRIPTS / "pypto-install-ptoas.sh", container)
    assert result.returncode == 0, result.stderr
    calls = Path(container["CALLS"]).read_text()
    assert f"ptoas-0.65-cp310-cp310-manylinux_2_34_{arch}.whl" in calls
    assert "uv venv" in calls
    assert "--force-reinstall" in calls


@pytest.mark.parametrize("version", ["ptoas 0.65", "ptoas version v0.65"])
def test_matching_version_skips_download(container: dict[str, str], version: str) -> None:
    executable(Path(container["PTOAS_ROOT"]) / "bin/ptoas", f"echo '{version}'")
    result = run(SCRIPTS / "pypto-install-ptoas.sh", container)
    assert result.returncode == 0, result.stderr
    assert not Path(container["CALLS"]).exists()


def test_checksum_failure_keeps_existing_install(container: dict[str, str]) -> None:
    ptoas = executable(Path(container["PTOAS_ROOT"]) / "bin/ptoas", "echo ptoas 0.61")
    pin = Path(container["PYPTO_SRC"]) / "toolchain/versions.env"
    pin.write_text(pin.read_text().replace(hashlib.sha256(b"wheel").hexdigest(), "0" * 64))
    result = run(SCRIPTS / "pypto-install-ptoas.sh", container)
    assert result.returncode != 0
    assert "uv " not in Path(container["CALLS"]).read_text()
    assert "0.61" in ptoas.read_text()


def test_invalid_pin_fails_before_download(container: dict[str, str]) -> None:
    pin = Path(container["PYPTO_SRC"]) / "toolchain/versions.env"
    pin.write_text(pin.read_text().replace("v0.65", "invalid"))
    result = run(SCRIPTS / "pypto-install-ptoas.sh", container)
    assert result.returncode != 0
    assert not Path(container["CALLS"]).exists()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-c", "core.hooksPath=/dev/null", "-C", str(repo), *args], text=True
    ).strip()


@pytest.fixture
def updater(tmp_path: Path, container: dict[str, str]) -> tuple[Path, dict[str, str]]:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy2(SCRIPTS / "pypto-update.sh", scripts / "pypto-update.sh")
    for name in ["pypto-install-ptoas.sh", "pypto-build", "pypto-doctor"]:
        executable(scripts / name, f'echo {name} >> "$CALLS"')
    return scripts / "pypto-update.sh", container


def test_updater_accepts_linked_worktree(tmp_path: Path, updater: tuple[Path, dict[str, str]]) -> None:
    script, env = updater
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "--initial-branch=main")
    git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "--allow-empty",
        "-m",
        "init",
    )
    linked = tmp_path / "linked"
    git(repo, "worktree", "add", "-b", "topic", str(linked))
    git(linked, "branch", "--set-upstream-to=main")
    env["PYPTO_SRC"] = str(linked)
    result = run(script, env, "--skip-doctor", "--jobs", "2")
    assert result.returncode == 0, result.stderr
    assert "pypto-build" in Path(env["CALLS"]).read_text()
    assert git(linked, "branch", "--show-current") == "topic"


def test_updater_preserves_dirty_tree(tmp_path: Path, updater: tuple[Path, dict[str, str]]) -> None:
    script, env = updater
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "--initial-branch=main")
    git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "--allow-empty",
        "-m",
        "init",
    )
    local = repo / "local.txt"
    local.write_text("keep this")
    env["PYPTO_SRC"] = str(repo)
    before = git(repo, "rev-parse", "HEAD")
    result = run(script, env)
    assert result.returncode != 0
    assert "local changes" in result.stderr
    assert git(repo, "rev-parse", "HEAD") == before
    assert local.read_text() == "keep this"
    assert not Path(env["CALLS"]).exists()


def test_no_pull_skips_git_operations(
    updater: tuple[Path, dict[str, str]],
) -> None:
    script, env = updater
    result = run(script, env, "--no-pull", "--skip-doctor")
    assert result.returncode == 0, result.stderr
    assert Path(env["CALLS"]).read_text().splitlines() == ["pypto-install-ptoas.sh", "pypto-build"]


def test_repeated_activation_prefers_pinned_ptoas(container: dict[str, str], tmp_path: Path) -> None:
    home = Path(container["PYPTO_HOME"])
    (home / "venv/bin/activate").write_text('export PATH="$PYPTO_HOME/venv/bin:$PATH"\n')
    cann = tmp_path / "cann"
    executable(cann / "online-build/bin/ptoas", "echo wrong assembler")
    executable(home / "ptoas/bin/ptoas", "echo pinned assembler")
    (cann / "set_env.sh").write_text(f'export PATH="{cann}/online-build/bin:$PATH"\n')
    container["ASCEND_TOOLKIT_ROOT"] = str(cann)
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; source "$1"; ptoas', "bash", str(SCRIPTS / "pypto-activate.sh")],
        env=container,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "pinned assembler"


@pytest.mark.parametrize("requested,expected", [("2", "2"), ("16", "8")])
def test_build_respects_requested_jobs_and_machine_limit(
    container: dict[str, str], requested: str, expected: str
) -> None:
    home = Path(container["PYPTO_HOME"])
    (home / "venv/bin/activate").write_text('export PATH="$PYPTO_HOME/venv/bin:$PATH"\n')
    source = Path(container["PYPTO_SRC"])
    (source / "CMakeLists.txt").touch()
    loader = source / ".claude/skills/testing/load-env.sh"
    loader.parent.mkdir(parents=True)
    loader.write_text("export PYPTO_BUILD_JOBS=8\n")
    executable(
        Path(container["PATH"].split(":")[0]) / "cmake",
        'printf "cmake %s MAX_JOBS=%s\\n" "$*" "$MAX_JOBS" >> "$CALLS"',
    )
    container["PYPTO_BUILD_JOBS"] = requested
    result = run(SCRIPTS / "pypto-build", container, "--pypto-only")
    assert result.returncode == 0, result.stderr
    assert f"--parallel {expected} MAX_JOBS={expected}" in Path(container["CALLS"]).read_text()


@pytest.mark.parametrize("args", [("--jobs", "0"), ("--jobs",), ("--proxy",)])
def test_invalid_update_options_do_not_change_environment(
    updater: tuple[Path, dict[str, str]], args: tuple[str, ...]
) -> None:
    script, env = updater
    result = run(script, env, *args)
    assert result.returncode == 2
    assert not Path(env["CALLS"]).exists()


@pytest.mark.parametrize("generator", ["Ninja", "Unix Makefiles"])
@pytest.mark.parametrize("remapped", [False, True])
def test_build_reuses_native_generator(container: dict[str, str], generator: str, remapped: bool) -> None:
    home = Path(container["PYPTO_HOME"])
    (home / "venv/bin/activate").write_text('export PATH="$PYPTO_HOME/venv/bin:$PATH"\n')
    source = Path(container["PYPTO_SRC"])
    (source / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.20)\nproject(probe NONE)\n")
    loader = source / ".claude/skills/testing/load-env.sh"
    loader.parent.mkdir(parents=True)
    loader.write_text("export PYPTO_BUILD_JOBS=2\n")
    subprocess.run(["cmake", "-S", str(source), "-B", str(source / "build"), "-G", generator], check=True)
    # The normal updater must preserve an existing native build rather than require --clean.
    marker = source / "build/keep.txt"
    marker.write_text("incremental build")
    if remapped:
        # A bind mount exposes the same cache at a different absolute path.
        original = source
        source = source.with_name("mounted-source")
        original.rename(source)
        container["PYPTO_SRC"] = str(source)
        marker = source / "build/keep.txt"
    result = run(SCRIPTS / "pypto-build", container, "--pypto-only")
    assert result.returncode == 0, result.stderr
    assert marker.read_text() == "incremental build"
    assert f"CMAKE_GENERATOR:INTERNAL={generator}" in (source / "build/CMakeCache.txt").read_text()


@pytest.fixture
def ptoas_import_environment(container: dict[str, str], tmp_path: Path) -> dict[str, str]:
    home = Path(container["PYPTO_HOME"])
    # Use real interpreters: selecting the right console script is insufficient when
    # CANN's PYTHONPATH still points at a conflicting Python package.
    (home / "venv/bin/python").unlink()
    container["VIRTUAL_ENV_DISABLE_PROMPT"] = "1"
    # The fixture uses pure Python packages, not real cp310 wheels. Keep the
    # installer's ABI-version probe independent of the pytest host interpreter.
    commands = Path(container["PATH"].split(":")[0])
    container["PYPTO_PYTHON"] = str(executable(commands / "python310-probe", "echo 3.10"))
    for prefix in [home / "venv", Path(container["PTOAS_ROOT"])]:
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(prefix)], check=True)
    purelib = subprocess.check_output(
        [str(home / "ptoas/bin/python"), "-c", 'import sysconfig; print(sysconfig.get_paths()["purelib"])'],
        text=True,
    ).strip()
    (Path(purelib) / "ptoas.py").write_text('VERSION = "ptoas 0.65"\n')
    console = home / "ptoas/bin/ptoas"
    console.write_text(f"#!{home}/ptoas/bin/python\nimport ptoas\nprint(ptoas.VERSION)\n")
    console.chmod(0o755)
    cann = tmp_path / "cann"
    (cann / "site-packages").mkdir(parents=True)
    (cann / "site-packages/ptoas.py").write_text('raise RuntimeError("CANN package shadowed the pin")\n')
    (cann / "set_env.sh").write_text(
        f'export PYTHONPATH="{cann}/site-packages${{PYTHONPATH:+:$PYTHONPATH}}"\n'
    )
    container["ASCEND_TOOLKIT_ROOT"] = str(cann)
    container["PYTHONPATH"] = str(cann / "site-packages")
    return container


def test_repeated_activation_isolates_ptoas_imports(ptoas_import_environment: dict[str, str]) -> None:
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; source "$1"; ptoas', "bash", str(SCRIPTS / "pypto-activate.sh")],
        env=ptoas_import_environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ptoas 0.65"


@pytest.mark.parametrize("force", [False, True])
def test_standalone_installer_isolates_version_probes(
    ptoas_import_environment: dict[str, str], force: bool
) -> None:
    env = ptoas_import_environment
    commands = Path(env["PATH"].split(":")[0])
    # Keep the real pinned executable; installing the fake wheel is a no-op.
    # Both the skip probe and post-install probe must survive CANN's broken package.
    executable(commands / "uv", 'printf "uv %s\\n" "$*" >> "$CALLS"')
    result = run(SCRIPTS / "pypto-install-ptoas.sh", env, *(["--force"] if force else []))
    assert result.returncode == 0, result.stderr
    if force:
        assert "ptoas 0.65" in result.stdout
        assert "--force-reinstall" in Path(env["CALLS"]).read_text()
    else:
        assert "already installed" in result.stdout
        assert not Path(env["CALLS"]).exists()


@pytest.mark.parametrize("access,expected", [("runtime", "0"), ("privileged", "3")])
def test_launcher_uses_container_visible_device(
    container: dict[str, str], access: str, expected: str
) -> None:
    commands = Path(container["PATH"].split(":")[0])
    executable(
        commands / "docker",
        'if [ "$1 $2" = "container inspect" ]; then exit 1; fi\n'
        'if [ "$1" = run ]; then printf "%s\\n" "$@" > "$CALLS"; fi',
    )
    container["PYPTO_DEVICE_ACCESS"] = access
    result = run(SCRIPTS / "pypto-dev.sh", container, "run", "a3", "3")
    assert result.returncode == 0, result.stderr
    args = Path(container["CALLS"]).read_text().splitlines()
    assert f"PYPTO_DEVICE={expected}" in args
    if access == "runtime":
        assert "ASCEND_VISIBLE_DEVICES=3" in args
        assert "--runtime=ascend" in args
    else:
        assert "--privileged" in args


@pytest.mark.parametrize("matches,dirty", [(True, False), (True, True), (False, False)])
def test_doctor_checks_installed_runtime_revision(
    container: dict[str, str], matches: bool, dirty: bool
) -> None:
    source = Path(container["PYPTO_SRC"])
    git(source, "init", "--initial-branch=main")
    runtime = source / "runtime"
    runtime.mkdir()
    git(runtime, "init", "--initial-branch=main")
    (runtime / "abi.h").write_text("original ABI")
    git(runtime, "add", "abi.h")
    git(
        runtime,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "--allow-empty",
        "-m",
        "init",
    )
    revision = git(runtime, "rev-parse", "HEAD") if matches else "0" * 40
    if dirty:
        (runtime / "abi.h").write_text("uncommitted ABI change")
    # A site-packages-style simpler imports successfully despite a stale build.
    # Exercise the real runtime-pin implementation without requiring the native core.
    packages = source / "python"
    for name in ["pypto", "pypto/runtime", "simpler"]:
        package = packages / name
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").touch()
    shutil.copy2(
        SCRIPTS.parents[1] / "python/pypto/runtime/runtime_pin.py", packages / "pypto/runtime/runtime_pin.py"
    )
    (packages / "pypto/runtime/__init__.py").write_text("from .runtime_pin import check_runtime_pin\n")
    (packages / "simpler/task_interface.py").touch()
    (packages / "_task_interface.py").write_text(f'__build_commit__ = "{revision}"\n')
    home = Path(container["PYPTO_HOME"])
    (home / "venv/bin/activate").write_text('export PATH="$PYPTO_HOME/venv/bin:$PATH"\n')
    executable(home / "venv/bin/python", f'exec "{sys.executable}" "$@"')
    commands = Path(container["PATH"].split(":")[0])
    executable(commands / "npu-smi", "exit 1")
    # Do not probe real hardware while exercising the binding diagnosis.
    executable(commands / "ls", 'if [ "${1:-}" = -d ]; then exit 1; fi; exec /bin/ls "$@"')
    result = run(SCRIPTS / "pypto-doctor", container)
    if matches and dirty:
        assert "runtime/ has uncommitted changes" in result.stdout
        assert "cannot prove the binding is current" in result.stdout
        assert "simpler binding matches its source tree" not in result.stdout
    elif matches:
        assert "simpler binding matches its source tree" in result.stdout, result.stdout + result.stderr
        assert "simpler binding rejected" not in result.stdout
    else:
        assert result.returncode != 0
        assert "simpler binding rejected" in result.stdout, result.stdout + result.stderr
        assert "rerun pypto-build" in result.stdout
        assert "environment looks good" not in result.stdout


@pytest.mark.parametrize("checkout", ["pypto", "pypto-lib"])
def test_launcher_mounts_linked_worktree_metadata(
    container: dict[str, str], tmp_path: Path, checkout: str
) -> None:
    primary = tmp_path / "primary"
    primary.mkdir()
    git(primary, "init", "--initial-branch=main")
    git(
        primary,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "--allow-empty",
        "-m",
        "init",
    )
    linked = tmp_path / "linked"
    git(primary, "worktree", "add", "-b", "topic", str(linked))
    commands = Path(container["PATH"].split(":")[0])
    executable(
        commands / "docker",
        'if [ "$1 $2" = "container inspect" ]; then exit 1; fi\n'
        'if [ "$1" = run ]; then printf "%s\\n" "$@" > "$CALLS"; fi',
    )
    host_variable = "PYPTO_SRC_HOST" if checkout == "pypto" else "PYPTO_LIB_SRC_HOST"
    container[host_variable] = str(linked)
    container["PYPTO_DEVICE_ACCESS"] = "runtime"
    result = run(SCRIPTS / "pypto-dev.sh", container, "run", "a3", "0")
    assert result.returncode == 0, result.stderr
    args = Path(container["CALLS"]).read_text().splitlines()
    assert f"{linked}:/workspace/{checkout}" in args
    common = git(linked, "rev-parse", "--path-format=absolute", "--git-common-dir")
    assert f"{common}:{common}" in args


def test_repeated_activation_prefers_mounted_checkouts(ptoas_import_environment: dict[str, str]) -> None:
    env = ptoas_import_environment
    home = Path(env["PYPTO_HOME"])
    site = Path(
        subprocess.check_output(
            [
                str(home / "venv/bin/python"),
                "-c",
                'import sysconfig; print(sysconfig.get_paths()["purelib"])',
            ],
            text=True,
        ).strip()
    )
    mounted = Path(env["PYPTO_SRC"]) / "python"
    library = home / "mounted-lib"
    library.mkdir()
    env["PYPTO_LIB_SRC"] = str(library)
    for root, origin in [(site, "installed"), (mounted, "mounted")]:
        (root / "pypto").mkdir(parents=True)
        (root / "pypto/__init__.py").write_text(f'ORIGIN = "{origin}"\n')
    (site / "container_lib_probe.py").write_text('ORIGIN = "installed"\n')
    (library / "container_lib_probe.py").write_text('ORIGIN = "mounted"\n')
    # Rcfile/BASH_ENV activation followed by helper activation must not validate
    # a stale site-packages copy instead of the checkout that was just built.
    probe = (
        "import os, sys, pypto, container_lib_probe; "
        'assert pypto.ORIGIN == "mounted"; assert container_lib_probe.ORIGIN == "mounted"; '
        'assert sys.path.count(os.environ["PYPTO_SRC"] + "/python") == 1; '
        'assert sys.path.count(os.environ["PYPTO_LIB_SRC"]) == 1'
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; source "$1"; source "$1"; python -c "$2"',
            "bash",
            str(SCRIPTS / "pypto-activate.sh"),
            probe,
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("ref", ["main", "feature/update", "release", "commit"])
def test_dist_checkout_preserves_tracking_branches(
    container: dict[str, str], tmp_path: Path, ref: str
) -> None:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    git(upstream, "init", "--initial-branch=main")
    git(
        upstream,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "--allow-empty",
        "-m",
        "init",
    )
    revision = git(upstream, "rev-parse", "HEAD")
    git(upstream, "branch", "feature/update")
    git(upstream, "tag", "release")
    env = container
    env.update(
        PYPTO_REPO=str(upstream),
        PYPTO_LIB_REPO=str(upstream),
        PYPTO_REF=revision if ref == "commit" else ref,
        PYPTO_LIB_REF=revision if ref == "commit" else ref,
        GIT_CONFIG_GLOBAL=str(tmp_path / "gitconfig"),
    )
    dockerfile = (SCRIPTS / "pypto_dev.Dockerfile").read_text()
    start = dockerfile.index('RUN set -eu; \\\n    git clone "$PYPTO_REPO"')
    clone_step = dockerfile[start : dockerfile.index("\n\n", start)]
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    # Execute the actual dist clone step against local repos, with global Git
    # writes redirected to the fixture. No Docker build or network is needed.
    command = clone_step.removeprefix("RUN ").replace("\\\n", "").replace("/workspace", str(workspace))
    result = subprocess.run(
        ["bash", "-eu", "-c", command], env=env, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    for name in ["pypto", "pypto-lib"]:
        checkout = workspace / name
        assert git(checkout, "rev-parse", "HEAD") == revision
        branch = git(checkout, "rev-parse", "--abbrev-ref", "HEAD")
        if ref in ["main", "feature/update"]:
            assert branch == ref
            assert git(checkout, "rev-parse", "--abbrev-ref", "@{upstream}") == f"origin/{ref}"
        else:
            assert branch == "HEAD"
    if ref == "main":
        # A default dist checkout must support the updater's fast-forward operation.
        git(
            upstream,
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--allow-empty",
            "-m",
            "next",
        )
        for name in ["pypto", "pypto-lib"]:
            git(workspace / name, "pull", "--ff-only")
            assert git(workspace / name, "rev-parse", "HEAD") == git(upstream, "rev-parse", "HEAD")


@pytest.mark.parametrize("device,status", [(None, 0), ("3", 0), ("3", 1)])
def test_doctor_probes_selected_device(
    container: dict[str, str], tmp_path: Path, device: str | None, status: int
) -> None:
    commands = Path(container["PATH"].split(":")[0])
    if device is not None:
        container["PYPTO_DEVICE"] = device
    container["TEST_ACL_STATUS"] = str(status)
    container["ASCEND_HOME_PATH"] = str(tmp_path / "cann")
    packages = Path(container["ASCEND_HOME_PATH"]) / "python/site-packages"
    packages.mkdir(parents=True)
    (packages / "acl.py").write_text(
        "import os\n"
        "def init(): return 0\n"
        "class Runtime:\n"
        "    def set_device(self, device):\n"
        '        with open(os.environ["CALLS"], "w") as log: log.write(str(device))\n'
        '        return int(os.environ["TEST_ACL_STATUS"])\n'
        "    def get_mem_info(self, kind):\n"
        '        assert os.environ["TEST_ACL_STATUS"] == "0", "device was not opened"\n'
        "        return (0, 1024 * 1024)\n"
        "rt = Runtime()\n"
    )
    cann_python = executable(commands / "cann-python", f'exec "{sys.executable}" "$@"')
    executable(
        commands / "ls", f'if [ "${{1:-}}" = -d ]; then echo "{cann_python}"; else exec /bin/ls "$@"; fi'
    )
    executable(commands / "npu-smi", "exit 1")
    home = Path(container["PYPTO_HOME"])
    (home / "venv/bin/activate").write_text('export PATH="$PYPTO_HOME/venv/bin:$PATH"\n')
    result = run(SCRIPTS / "pypto-doctor", container)
    assert Path(container["CALLS"]).read_text() == (device or "0")
    if status == 0:
        assert f"aclrtSetDevice({device or '0'}) ok" in result.stdout
    else:
        assert result.returncode != 0
        assert f"device access refused (device {device})" in result.stdout
        assert "could not run the ACL device probe" not in result.stdout


@pytest.mark.parametrize("dirty", [None, "tracked", "untracked", "parent", "staged-pin"])
def test_remote_submodule_updates_repeat_without_losing_edits(
    tmp_path: Path, updater: tuple[Path, dict[str, str]], dirty: str | None
) -> None:
    script, env = updater
    runtime = tmp_path / "runtime-upstream"
    parent = tmp_path / "parent-upstream"
    for repo in [runtime, parent]:
        repo.mkdir()
        git(repo, "init", "--initial-branch=main")
        git(repo, "config", "user.name", "Test")
        git(repo, "config", "user.email", "test@example.invalid")
    (runtime / "abi.h").write_text("first")
    git(runtime, "add", "abi.h")
    git(runtime, "commit", "-m", "first")
    git(parent, "-c", "protocol.file.allow=always", "submodule", "add", str(runtime), "runtime")
    git(parent, "commit", "-am", "pin runtime")
    checkout = tmp_path / "checkout"
    git(
        tmp_path,
        "-c",
        "protocol.file.allow=always",
        "clone",
        "--recurse-submodules",
        str(parent),
        str(checkout),
    )
    env.update(PYPTO_SRC=str(checkout), GIT_ALLOW_PROTOCOL="file")
    (runtime / "abi.h").write_text("second")
    git(runtime, "commit", "-am", "advance runtime")
    new_revision = git(runtime, "rev-parse", "HEAD")
    # The first remote update leaves the gitlink deliberately modified. That
    # expected commit drift must not prevent a second update from running.
    for _ in range(2):
        result = run(script, env, "--remote-submodules", "--skip-doctor")
        assert result.returncode == 0, result.stderr
        assert git(checkout / "runtime", "rev-parse", "HEAD") == new_revision
    calls = Path(env["CALLS"]).read_text()
    if dirty is not None:
        if dirty == "staged-pin":
            git(checkout, "add", "runtime")
            target = checkout / "runtime/abi.h"
            content = "second"
        else:
            target = checkout / (
                "local.txt"
                if dirty == "parent"
                else f"runtime/{'abi.h' if dirty == 'tracked' else 'local.txt'}"
            )
            content = "keep local edits"
            target.write_text(content)
        result = run(script, env, "--remote-submodules", "--skip-doctor")
        assert result.returncode != 0
        assert "local changes" in result.stderr
        assert target.read_text() == content
        assert Path(env["CALLS"]).read_text() == calls
    else:
        # Normal pinned mode still refuses off-pin submodules. Remote mode can
        # then pull a superproject commit that records the advanced gitlink.
        result = run(script, env, "--skip-doctor")
        assert result.returncode != 0
        git(parent / "runtime", "fetch", "origin")
        git(parent / "runtime", "checkout", "--detach", new_revision)
        git(parent, "add", "runtime")
        git(parent, "commit", "-m", "record advanced pin")
        result = run(script, env, "--remote-submodules", "--skip-doctor")
        assert result.returncode == 0, result.stderr
        assert git(checkout, "rev-parse", "HEAD") == git(parent, "rev-parse", "HEAD")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
