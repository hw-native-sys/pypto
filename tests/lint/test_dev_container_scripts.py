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
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
from check_headers import PY_HEADER, check_file_header

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "docker/scripts"
DOCKERFILE = REPO_ROOT / "docker/pypto_dev.Dockerfile"


def executable(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/bash\nset -eu\n{body}\n")
    path.chmod(0o755)
    return path


def fake_venv(prefix: Path, python_body: str = "echo 3.10") -> None:
    """Provide the interpreter and metadata required by the installer contract."""
    executable(prefix / "bin/python", python_body)
    (prefix / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
    (prefix / "bin/activate").write_text(f'export PATH="{prefix}/bin:$PATH"\n')


def run(script: Path, env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", str(script), *args], env=env, text=True, capture_output=True, check=False)


@pytest.fixture
def container(tmp_path: Path) -> dict[str, str]:
    """Provide fake curl/uv programs and an isolated installation prefix."""
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("PYPTO_", "PTOAS_", "GIT_CONFIG_", "ASCEND_")) or key in {
            "BASH_ENV",
            "UV_VERSION",
        }:
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
    fake_venv(home / "venv")
    executable(
        commands / "curl",
        'printf "curl %s\\n" "$*" >> "$CALLS"\nwhile [ "$1" != -o ]; do shift; done\nprintf wheel > "$2"',
    )
    executable(
        commands / "uv",
        'printf "uv %s\\n" "$*" >> "$CALLS"\n'
        'if [ "$1" = venv ]; then\n'
        '    target="${*: -1}"\n'
        '    mkdir -p "$target/bin"\n'
        '    printf "include-system-site-packages = false\\n" > "$target/pyvenv.cfg"\n'
        '    printf \'export PATH="%s/bin:$PATH"\\n\' "$target" '
        '> "$target/bin/activate"\n'
        '    if [ "$target" = "$PYPTO_HOME/venv" ]; then\n'
        '        printf "#!/usr/bin/env -S BASH_ENV= bash\\necho %s\\n" '
        '"${TEST_NEW_PYTHON_VERSION:-3.10}" > "$target/bin/python"\n'
        '        chmod +x "$target/bin/python"\n'
        '    else cp "$PYPTO_HOME/venv/bin/python" "$target/bin/python"; fi\n'
        'elif [[ "${*: -1}" = *.whl ]]; then\n'
        '    printf "#!/bin/bash\\necho ptoas 0.65\\n" > "${PTOAS_ROOT:-$PYPTO_HOME/ptoas}/bin/ptoas"\n'
        '    chmod +x "${PTOAS_ROOT:-$PYPTO_HOME/ptoas}/bin/ptoas"\n'
        "fi",
    )
    return env


@pytest.mark.parametrize("script", ["pypto-setup.sh", "pypto-install-ptoas.sh"])
def test_helpers_infer_checkout_from_scripts_directory(container: dict[str, str], script: str) -> None:
    source = Path(container.pop("PYPTO_SRC"))
    shutil.copytree(SCRIPTS, source / "docker/scripts")
    constraints = source / "build-constraints.txt"
    constraints.write_text("nanobind==2.15.0\n")
    # No installed pin fallback or PYPTO_SRC: the checkout must be inferred
    # from the helper's location, independently of the caller's working directory.
    result = run(source / "docker/scripts" / script, container)
    assert result.returncode == 0, result.stderr
    calls = Path(container["CALLS"]).read_text()
    assert "ptoas-0.65-" in calls
    if script == "pypto-setup.sh":
        assert f"--constraint {constraints}" in calls
        assert (Path(container["PYPTO_HOME"]) / "toolchain/versions.env").read_bytes() == (
            source / "toolchain/versions.env"
        ).read_bytes()


@pytest.mark.parametrize("repeat", [False, True])
def test_setup_installs_shared_helpers(container: dict[str, str], repeat: bool) -> None:
    constraints = Path(container["PYPTO_SRC"]) / "build-constraints.txt"
    constraints.write_text("nanobind==2.15.0\n")
    result = run(SCRIPTS / "pypto-setup.sh", container)
    assert result.returncode == 0, result.stderr
    home = Path(container["PYPTO_HOME"])
    if repeat:
        (home / "bin/uv").chmod(0o700)
        result = run(SCRIPTS / "pypto-setup.sh", container)
        assert result.returncode == 0, result.stderr
    assert (home / "bin/uv").stat().st_mode & 0o777 == 0o755
    assert (home / "bin/pypto-build").read_text() == (SCRIPTS / "pypto-build").read_text()
    assert (home / "bin/uv").read_bytes() == (Path(container["PATH"].split(":")[0]) / "uv").read_bytes()
    assert (home / "toolchain/build-constraints.txt").read_text() == constraints.read_text()
    calls = Path(container["CALLS"]).read_text()
    assert "--constraint" in calls
    assert "ptoas-0.65-" in calls


@pytest.mark.parametrize(
    "version",
    ["3.10", "3.11", "3.12", "unsupported", "broken", "missing", "real", "no-activate", "no-config"],
)
def test_setup_migrates_incompatible_venv(container: dict[str, str], version: str) -> None:
    home = Path(container["PYPTO_HOME"])
    (Path(container["PYPTO_SRC"]) / "build-constraints.txt").write_text("nanobind==2.15.0\n")
    marker = home / "venv/old-environment"
    marker.touch()
    python = home / "venv/bin/python"
    if version in {"no-activate", "no-config"}:
        (home / "venv" / ("bin/activate" if version == "no-activate" else "pyvenv.cfg")).unlink()
    elif version == "missing":
        python.unlink()
    elif version == "real":
        executable(python, f'exec "{sys.executable}" "$@"')
    else:
        executable(python, "exit 1" if version == "broken" else f"echo {version}")
    preserved = version == "3.10" or (
        version == "real" and sys.implementation.name == "cpython" and sys.version_info[:2] == (3, 10)
    )
    result = run(SCRIPTS / "pypto-setup.sh", container)
    assert result.returncode == 0, result.stderr
    calls = Path(container["CALLS"]).read_text().splitlines()
    recreated = [call for call in calls if call.endswith(f"cpython@3.10 {home / 'venv'}")]
    assert bool(recreated) == (not preserved)
    assert marker.exists() == preserved
    assert (
        subprocess.check_output(
            [str(python), "-I", "-c", 'import sys; print(".".join(map(str, sys.version_info[:2])))'],
            text=True,
        ).strip()
        == "3.10"
    )
    if recreated:
        assert calls.index(recreated[0]) < next(
            i for i, call in enumerate(calls) if call.startswith("uv pip install")
        )


@pytest.mark.parametrize("recreated_broken", [False, True])
@pytest.mark.parametrize("missing", [None, "Python.h", "pyconfig.h", "libpython.so", "directory"])
def test_setup_requires_development_files(
    container: dict[str, str], tmp_path: Path, missing: str | None, recreated_broken: bool
) -> None:
    home = Path(container["PYPTO_HOME"])
    (Path(container["PYPTO_SRC"]) / "build-constraints.txt").write_text("nanobind==2.15.0\n")
    development = tmp_path / "development"
    development.mkdir()
    for name in ["Python.h", "pyconfig.h", "libpython.so"]:
        (development / name).touch()
    if missing == "directory":
        (development / "Python.h").unlink()
        (development / "Python.h").mkdir()
    elif missing:
        (development / missing).unlink()
    # Execute the real probe with sysconfig pointing at controlled development
    # files, while reporting 3.10 for the independent interpreter-version probe.
    probe = (
        "import sys, sysconfig; "
        f"sysconfig.get_path = lambda key: {str(development)!r}; "
        f"values = {{'LIBDIR': {str(development)!r}, 'LDLIBRARY': 'libpython.so'}}; "
        "sysconfig.get_config_var = values.get; exec(sys.argv[1])"
    )
    executable(
        home / "venv/bin/python",
        'if [[ "${*: -1}" == *"import sysconfig"* ]]; then\n'
        f'    exec {shlex.quote(sys.executable)} -I -c {shlex.quote(probe)} "${{*: -1}}"\n'
        "fi\necho 3.10",
    )
    if recreated_broken:
        saved = development / "python"
        shutil.copy2(home / "venv/bin/python", saved)
        uv = Path(container["PATH"].split(":")[0]) / "uv"
        uv.write_text(
            uv.read_text()
            + 'if [ "$1" = venv ] && [ "${*: -1}" = "$PYPTO_HOME/venv" ]; then\n'
            + f'    cp {shlex.quote(str(saved))} "$PYPTO_HOME/venv/bin/python"\nfi\n'
        )
    marker = home / "venv/existing"
    marker.touch()
    result = run(SCRIPTS / "pypto-setup.sh", container)
    if recreated_broken and missing is not None:
        assert result.returncode != 0
        assert "Requires a complete CPython 3.10 virtual environment" in result.stderr
        assert "uv pip install" not in Path(container["CALLS"]).read_text()
    else:
        assert result.returncode == 0, result.stderr
    assert marker.exists() == (missing is None)
    calls = Path(container["CALLS"]).read_text()
    assert ("uv venv --managed-python --python cpython@3.10" in calls) == (missing is not None)


@pytest.mark.parametrize("helpers_only", [False, True])
@pytest.mark.parametrize("inferred", [False, True])
def test_setup_activation_command_preserves_paths(
    container: dict[str, str], tmp_path: Path, helpers_only: bool, inferred: bool
) -> None:
    home = tmp_path / "custom home 'quoted'"
    source = tmp_path / "checkout ; literal"
    Path(container["PYPTO_HOME"]).rename(home)
    Path(container["PYPTO_SRC"]).rename(source)
    # Refresh the fixture activation after moving its prefix.
    fake_venv(home / "venv")
    (source / "python").mkdir()
    (source / "build-constraints.txt").write_text("nanobind==2.15.0\n")
    shutil.copytree(SCRIPTS, source / "docker/scripts")
    container.pop("PYPTO_HOME")
    container.pop("PYPTO_SRC")
    container.pop("PTOAS_ROOT")
    args = ["--pypto-home", str(home)]
    if not inferred:
        args += ["--pypto-src", str(source)]
    if helpers_only:
        args += ["--helpers-only"]
    result = run(source / "docker/scripts/pypto-setup.sh", container, *args)
    assert result.returncode == 0, result.stderr
    command = result.stdout.split("Activate with: ", 1)[1].strip()
    # A different shell must not depend on exports made by the setup child.
    parent = {**container, "PYPTO_HOME": "/stale", "PYPTO_SRC": "/stale", "PTOAS_ROOT": "/stale"}
    activated = subprocess.run(
        [
            "bash",
            "-c",
            command
            + '\nprintf "%s\\n" "$PYPTO_HOME" "$PYPTO_SRC" "$PTOAS_ROOT"; '
            + "command -v python; command -v pypto-build",
        ],
        env=parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert activated.returncode == 0, activated.stderr
    assert activated.stdout.splitlines()[-5:] == [
        str(home),
        str(source),
        str(home / "ptoas"),
        str(home / "venv/bin/python"),
        str(home / "bin/pypto-build"),
    ]


def test_setup_rejects_incorrect_recreated_python(container: dict[str, str]) -> None:
    home = Path(container["PYPTO_HOME"])
    (Path(container["PYPTO_SRC"]) / "build-constraints.txt").write_text("nanobind==2.15.0\n")
    executable(home / "venv/bin/python", "echo 3.11")
    container["TEST_NEW_PYTHON_VERSION"] = "3.12"
    result = run(SCRIPTS / "pypto-setup.sh", container)
    assert result.returncode != 0
    assert "Requires a complete CPython 3.10 virtual environment" in result.stderr
    assert "uv pip install" not in Path(container["CALLS"]).read_text()


def test_setup_replaces_venv_symlink_without_deleting_target(
    container: dict[str, str], tmp_path: Path
) -> None:
    home = Path(container["PYPTO_HOME"])
    (Path(container["PYPTO_SRC"]) / "build-constraints.txt").write_text("nanobind==2.15.0\n")
    external = tmp_path / "external-venv"
    (home / "venv").rename(external)
    executable(external / "bin/python", "echo 3.11")
    marker = external / "keep-this-environment"
    marker.touch()
    (home / "venv").symlink_to(external, target_is_directory=True)
    result = run(SCRIPTS / "pypto-setup.sh", container)
    assert result.returncode == 0, result.stderr
    assert not (home / "venv").is_symlink()
    assert marker.exists()
    assert subprocess.check_output([str(external / "bin/python")], text=True).strip() == "3.11"


@pytest.mark.parametrize("option", ["argument", "environment"])
@pytest.mark.parametrize("existing", ["absent", "path", "shared", "symlink"])
def test_setup_honors_uv_version(
    container: dict[str, str], tmp_path: Path, option: str, existing: str
) -> None:
    home = Path(container["PYPTO_HOME"])
    commands = Path(container["PATH"].split(":")[0])
    (Path(container["PYPTO_SRC"]) / "build-constraints.txt").write_text("nanobind==2.15.0\n")
    downloaded = executable(
        tmp_path / "release/uv", (commands / "uv").read_text() + "\n# requested release\n"
    )
    asset = "uv-x86_64-unknown-linux-gnu"
    archive = tmp_path / "uv.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(downloaded, arcname=f"{asset}/uv")
    executable(commands / "uname", "echo x86_64")
    executable(
        commands / "curl",
        'printf "curl %s\\n" "$*" >> "$CALLS"\n'
        'while [ "$1" != -o ]; do shift; done\n'
        'case "$2" in\n'
        f'    */uv.tar.gz) cp "{archive}" "$2" ;;\n'
        '    *) printf wheel > "$2" ;;\n'
        "esac",
    )
    original = (commands / "uv").read_bytes()
    if existing == "absent":
        # Remove the fake uv and exclude the runner's environment directories.
        (commands / "uv").unlink()
        container["PATH"] = f"{commands}:/usr/bin:/bin"
    elif existing in {"shared", "symlink"}:
        (home / "bin").mkdir()
        if existing == "shared":
            shutil.copyfile(commands / "uv", home / "bin/uv")
            (home / "bin/uv").chmod(0o755)
        else:
            (home / "bin/uv").symlink_to(commands / "uv")
    container["UV_VERSION"] = "0.1.0" if option == "argument" else "0.9.2"
    args = ("--uv-version", "0.9.2") if option == "argument" else ()
    result = run(SCRIPTS / "pypto-setup.sh", container, *args)
    assert result.returncode == 0, result.stderr
    shared = home / "bin/uv"
    assert shared.is_file() and not shared.is_symlink()
    assert shared.read_bytes() == downloaded.read_bytes()
    assert shared.stat().st_mode & 0o777 == 0o755
    calls = Path(container["CALLS"]).read_text()
    assert f"releases/download/0.9.2/{asset}.tar.gz" in calls
    assert "releases/download/0.1.0/" not in calls
    assert "releases/latest/" not in calls
    if existing != "absent":
        assert (commands / "uv").read_bytes() == original


def test_setup_does_not_fall_back_when_pinned_uv_download_fails(container: dict[str, str]) -> None:
    commands = Path(container["PATH"].split(":")[0])
    (Path(container["PYPTO_SRC"]) / "build-constraints.txt").write_text("nanobind==2.15.0\n")
    executable(commands / "curl", "exit 22")
    result = run(SCRIPTS / "pypto-setup.sh", container, "--uv-version", "0.9.2")
    assert result.returncode == 22
    assert not Path(container["CALLS"]).exists()
    assert not (Path(container["PYPTO_HOME"]) / "bin/uv").exists()


@pytest.mark.parametrize("shared_symlink", [False, True])
def test_setup_shares_uv_from_root_private_home(
    container: dict[str, str], tmp_path: Path, shared_symlink: bool
) -> None:
    if os.getuid() != 0:
        pytest.skip("requires a rootful container to reproduce a root-private uv executable")
    source = Path(container["PYPTO_SRC"])
    (source / "build-constraints.txt").write_text("nanobind==2.15.0\n")
    for parent in source.parents:
        if parent == Path("/tmp"):
            break
        parent.chmod(parent.stat().st_mode | 0o111)
    for path in [source, *source.rglob("*")]:
        os.chown(path, 10001, 10001)
    private = tmp_path / "private-home"
    private.mkdir(mode=0o700)
    private_uv = private / "uv"
    commands = Path(container["PATH"].split(":")[0])
    (commands / "uv").rename(private_uv)
    container["PATH"] = f"{private}:{container['PATH']}"
    home = Path(container["PYPTO_HOME"])
    if shared_symlink:
        (home / "bin").mkdir()
        (home / "bin/uv").symlink_to(private_uv)
    denied = subprocess.run(
        ["setpriv", "--reuid=10001", "--regid=10001", "--clear-groups", str(private_uv), "--version"],
        env=container,
        text=True,
        capture_output=True,
        check=False,
    )
    assert denied.returncode != 0
    assert "Permission denied" in denied.stderr
    result = run(SCRIPTS / "pypto-setup.sh", container)
    assert result.returncode == 0, result.stderr
    assert not (home / "bin/uv").is_symlink()
    assert (home / "bin/uv").read_bytes() == private_uv.read_bytes()
    (home / "venv/bin/activate").write_text('export PATH="$PYPTO_HOME/venv/bin:$PATH"\n')
    # Exercise the same activation and UID transition used by pypto-build.
    result = subprocess.run(
        [
            "bash",
            "-eu",
            "-c",
            'source "$PYPTO_HOME/bin/pypto-activate.sh"; '
            'source "$PYPTO_HOME/bin/pypto-owner.sh"; '
            'pypto_as_checkout_owner "$PYPTO_SRC" env CALLS="$PYPTO_SRC/uv-calls" uv --version',
        ],
        env=container,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (source / "uv-calls").read_text() == "uv --version\n"
    assert (source / "uv-calls").stat().st_uid == 10001


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
    fake_venv(Path(container["PTOAS_ROOT"]))
    executable(Path(container["PTOAS_ROOT"]) / "bin/ptoas", f"echo '{version}'")
    result = run(SCRIPTS / "pypto-install-ptoas.sh", container)
    assert result.returncode == 0, result.stderr
    assert not Path(container["CALLS"]).exists()


@pytest.mark.parametrize("state", ["3.10", "3.11", "unsupported", "broken", "python", "activate", "cfg"])
def test_installer_repairs_ptoas_environment(container: dict[str, str], state: str) -> None:
    prefix = Path(container["PTOAS_ROOT"])
    python_body = {"3.11": "echo 3.11", "unsupported": "echo unsupported", "broken": "exit 1"}.get(
        state, "echo 3.10"
    )
    fake_venv(prefix, python_body)
    marker = prefix / "old-environment"
    marker.touch()
    if state in {"python", "activate", "cfg"}:
        (prefix / ("pyvenv.cfg" if state == "cfg" else f"bin/{state}")).unlink()
    # Even a matching assembler banner must not preserve an incompatible or
    # incomplete venv. A healthy environment remains a download-free no-op.
    executable(prefix / "bin/ptoas", "echo ptoas 0.65")
    result = run(SCRIPTS / "pypto-install-ptoas.sh", container)
    assert result.returncode == 0, result.stderr
    assert marker.exists() == (state == "3.10")
    assert (prefix / "pyvenv.cfg").is_file() and (prefix / "bin/activate").is_file()
    assert (
        subprocess.check_output([str(prefix / "bin/python"), "-I", "-S", "-c", ""], text=True).strip()
        == "3.10"
    )
    if state != "3.10":
        calls = Path(container["CALLS"]).read_text()
        assert "uv venv" in calls and calls.index("uv venv") < calls.index("uv pip install")
    else:
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
    shutil.copy2(SCRIPTS / "pypto-owner.sh", scripts / "pypto-owner.sh")
    for name in ["pypto-install-ptoas.sh", "pypto-build", "pypto-doctor", "pypto-setup.sh"]:
        executable(scripts / name, f'echo {name} >> "$CALLS"')
        executable(Path(container["PYPTO_SRC"]) / "docker/scripts" / name, f'echo {name} >> "$CALLS"')
    return scripts / "pypto-update.sh", container


def test_updater_accepts_linked_worktree(tmp_path: Path, updater: tuple[Path, dict[str, str]]) -> None:
    script, env = updater
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "--initial-branch=main")
    shutil.copytree(Path(env["PYPTO_SRC"]) / "docker", repo / "docker")
    git(repo, "add", "docker")
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
    assert Path(env["CALLS"]).read_text().splitlines() == [
        "pypto-setup.sh",
        "pypto-install-ptoas.sh",
        "pypto-build",
    ]


def test_updater_rejects_missing_checkout_helper(updater: tuple[Path, dict[str, str]]) -> None:
    script, env = updater
    (Path(env["PYPTO_SRC"]) / "docker/scripts/pypto-build").unlink()
    result = run(script, env, "--no-pull", "--skip-doctor")
    assert result.returncode != 0
    assert "Missing checkout helper" in result.stderr
    assert not Path(env["CALLS"]).exists()


@pytest.mark.parametrize("old_directory", [".github/docker", "docker", "docker/scripts"])
@pytest.mark.parametrize("reinstall", [False, True])
def test_installed_updater_uses_pulled_helpers(
    updater: tuple[Path, dict[str, str]], tmp_path: Path, reinstall: bool, old_directory: str
) -> None:
    script, env = updater
    upstream = Path(env["PYPTO_SRC"])
    git(upstream, "init", "--initial-branch=main")
    git(upstream, "config", "user.name", "Test")
    git(upstream, "config", "user.email", "test@example.invalid")
    helpers = ["pypto-install-ptoas.sh", "pypto-build", "pypto-doctor", "pypto-setup.sh"]
    if old_directory != "docker/scripts":
        (upstream / old_directory).mkdir(parents=True, exist_ok=True)
        for helper in helpers:
            (upstream / "docker/scripts" / helper).rename(upstream / old_directory / helper)
        (upstream / "docker/scripts").rmdir()
    git(upstream, "add", old_directory, "toolchain/versions.env")
    git(upstream, "commit", "-m", "old helpers")
    checkout = tmp_path / "checkout"
    git(upstream, "clone", str(upstream), str(checkout))
    if old_directory != "docker/scripts":
        # Resolve checkout helpers after pulling, even when the pull moves them.
        (upstream / "docker/scripts").mkdir(parents=True)
        for helper in helpers:
            git(upstream, "mv", f"{old_directory}/{helper}", f"docker/scripts/{helper}")
    for helper in helpers:
        executable(script.parent / helper, "exit 99")
        executable(upstream / "docker/scripts" / helper, f'echo "new {helper}" >> "$CALLS"')
    git(upstream, "commit", "-am", "new helpers")
    env["PYPTO_SRC"] = str(checkout)
    result = run(script, env, *(["--reinstall-deps"] if reinstall else []))
    assert result.returncode == 0, result.stderr
    assert Path(env["CALLS"]).read_text().splitlines() == [
        "new pypto-setup.sh",
        *([] if reinstall else ["new pypto-install-ptoas.sh"]),
        "new pypto-build",
        "new pypto-doctor",
    ]
    assert git(checkout, "rev-parse", "HEAD") == git(upstream, "rev-parse", "HEAD")


def test_normal_update_refreshes_installed_helpers(container: dict[str, str]) -> None:
    home = Path(container["PYPTO_HOME"])
    source = Path(container["PYPTO_SRC"])
    helpers = source / "docker/scripts"
    shutil.copytree(SCRIPTS, helpers)
    (source / "build-constraints.txt").write_text("nanobind==2.15.0\n")
    result = run(helpers / "pypto-setup.sh", container, "--helpers-only")
    assert result.returncode == 0, result.stderr
    installed_updater = home / "bin/pypto-update.sh"
    installed_updater.write_text(installed_updater.read_text() + "\necho old-launcher-tail\n")
    (helpers / "pypto-update.sh").write_text(
        (helpers / "pypto-update.sh").read_text() + "\necho new-launcher-tail\n"
    )
    activation = helpers / "pypto-activate.sh"
    activation.write_text(activation.read_text() + "\nexport CURRENT_HELPERS=1\n")
    for name in ["pypto-build", "pypto-doctor", "pypto-install-ptoas.sh"]:
        executable(home / "bin" / name, "exit 99")
        executable(helpers / name, f'echo "current {name}" >> "$CALLS"')
    marker = home / "venv/keep-dependencies"
    marker.touch()
    # An open reader retains the old inode while the installed updater refreshes
    # itself. This guards against in-place truncation of a running shell script.
    before = installed_updater.read_bytes()
    with installed_updater.open("rb") as reader:
        result = run(installed_updater, container, "--no-pull", "--skip-doctor")
        assert result.returncode == 0, result.stdout + result.stderr
        assert reader.read() == before
    assert "old-launcher-tail" in result.stdout and "new-launcher-tail" not in result.stdout
    assert "Reload this shell with: export" in result.stdout
    for name in [
        "pypto-build",
        "pypto-doctor",
        "pypto-update.sh",
        "pypto-setup.sh",
        "pypto-install-ptoas.sh",
        "pypto-activate.sh",
        "pypto-owner.sh",
        "pypto-venv.sh",
    ]:
        assert (home / "bin" / name).read_bytes() == (helpers / name).read_bytes()
    calls = Path(container["CALLS"]).read_text()
    assert "uv " not in calls and "curl " not in calls
    assert marker.exists()
    result = subprocess.run(
        [
            "bash",
            "-eu",
            "-c",
            'source "$PYPTO_HOME/bin/pypto-activate.sh"; test "$CURRENT_HELPERS" = 1; pypto-build',
        ],
        env=container,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert Path(container["CALLS"]).read_text().splitlines()[-1] == "current pypto-build"


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


@pytest.mark.parametrize("manifest", [False, True])
@pytest.mark.parametrize(
    "tag,override,expected",
    [
        ("9.0.0-a3-ubuntu22.04-py3.11-devel", None, "a2a3"),
        ("9.1.0-950-ubuntu22.04-py3.12-devel", None, "a5"),
        ("9.1.T500-950-ubuntu22.04-py3.11-devel", None, "a5"),
        ("9.0.0-a3-ubuntu22.04-py3.11-devel", "a5", "a5"),
        ("custom-tag", "a5", "a5"),
        ("9.1.0-950-ubuntu22.04-py3.12-devel", "a5sim", "a5sim"),
        ("custom-tag", None, ""),
    ],
)
def test_activation_guidance_uses_current_platform(
    container: dict[str, str], manifest: bool, tag: str, override: str | None, expected: str
) -> None:
    home = Path(container["PYPTO_HOME"])
    (home / "venv/bin/activate").write_text('export PATH="$PYPTO_HOME/venv/bin:$PATH"\n')
    if manifest:
        (home / "dist-manifest").write_text("pypto image-revision\nruntime image-runtime\n")
    container["PYPTO_CANN_TAG"] = tag
    if override is not None:
        container["PYPTO_PLATFORM"] = override
    # A retained image manifest and a fresh bind-mounted tree must not produce
    # a readiness claim. Repeated activation should show guidance only once.
    assert not list(Path(container["PYPTO_SRC"]).rglob("pypto_core*.so"))
    result = subprocess.run(
        [
            "bash",
            "--noprofile",
            "--norc",
            "-i",
            "-c",
            'source "$1"; source "$1"; printf "selected=%s\\n" "$PYPTO_PLATFORM"',
            "bash",
            str(SCRIPTS / "pypto-activate.sh"),
        ],
        env=container,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert f"selected={expected}\n" in result.stdout
    assert result.stdout.count("PyPTO dev environment") == 1
    assert "already built" not in result.stdout
    assert "pypto-doctor (check the current checkout)" in result.stdout
    assert "pypto-build (build or refresh it)" in result.stdout
    assert ("Image build revisions:" in result.stdout) == manifest
    if expected:
        assert f" -p {expected} " in result.stdout
        model = "deepseek_v4_1_flash/decode_swa.py" if expected == "a5" else "qwen3_14b/decode_fwd.py"
        assert f"$PYPTO_LIB_SRC/models/{model} -p {expected} -d ${{PYPTO_DEVICE:-0}}" in result.stdout
        if expected == "a5":
            assert "qwen3_14b" not in result.stdout
    else:
        assert "Set PYPTO_PLATFORM=a2a3 or a5" in result.stdout
        assert " -p a2a3 " not in result.stdout


@pytest.mark.parametrize("platform", ["a2a3", "a5", None])
def test_build_suggests_selected_platform(container: dict[str, str], platform: str | None) -> None:
    home = Path(container["PYPTO_HOME"])
    (home / "venv/bin/activate").write_text('export PATH="$PYPTO_HOME/venv/bin:$PATH"\n')
    source = Path(container["PYPTO_SRC"])
    (source / "CMakeLists.txt").touch()
    loader = source / ".claude/skills/testing/load-env.sh"
    loader.parent.mkdir(parents=True)
    loader.write_text("export PYPTO_BUILD_JOBS=2\n")
    executable(
        Path(container["PATH"].split(":")[0]) / "cmake",
        'if [ "$1" = -B ]; then mkdir -p build; touch build/CMakeCache.txt; fi',
    )
    if platform is not None:
        container["PYPTO_PLATFORM"] = platform
    result = run(SCRIPTS / "pypto-build", container, "--pypto-only")
    assert result.returncode == 0, result.stderr
    if platform is not None:
        assert f" -p {platform} " in result.stdout
        model = "deepseek_v4_1_flash/decode_swa.py" if platform == "a5" else "qwen3_14b/decode_fwd.py"
        assert f"$PYPTO_LIB_SRC/models/{model} -p {platform} -d ${{PYPTO_DEVICE:-0}}" in result.stdout
        if platform == "a5":
            assert "qwen3_14b" not in result.stdout
    else:
        assert "Set PYPTO_PLATFORM=a2a3 or a5" in result.stdout
        assert " -p a2a3 " not in result.stdout


@pytest.mark.parametrize(
    "profile,requested,expected", [(None, "1", "1"), (None, "16", "2"), ("8", "1", "1"), ("8", "16", "8")]
)
@pytest.mark.parametrize("linked", [False, True])
@pytest.mark.parametrize("updater", [False, True])
def test_build_respects_requested_jobs_and_machine_limit(
    container: dict[str, str], profile: str | None, requested: str, expected: str, linked: bool, updater: bool
) -> None:
    home = Path(container["PYPTO_HOME"])
    (home / "venv/bin/activate").write_text('export PATH="$PYPTO_HOME/venv/bin:$PATH"\n')
    source = Path(container["PYPTO_SRC"])
    (source / "CMakeLists.txt").touch()
    loader = source / ".claude/skills/testing/load-env.sh"
    loader.parent.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / ".claude/skills/testing/load-env.sh", loader)
    if updater:
        helpers = source / "docker/scripts"
        executable(helpers / "pypto-build", f'exec "{SCRIPTS / "pypto-build"}" --pypto-only')
        for helper in ["pypto-install-ptoas.sh", "pypto-setup.sh", "pypto-doctor"]:
            executable(helpers / helper, "exit 0")
    git(source, "init", "--initial-branch=main")
    git(source, "add", "CMakeLists.txt", ".claude/skills/testing/load-env.sh")
    if updater:
        git(source, "add", "docker")
    git(source, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "init")
    # The profile exists only in the primary checkout. Missing profiles must
    # retain the actual loader's two-job fallback, even for an explicit request.
    if profile is not None:
        (loader.parent / "testing.env").write_text(
            f"export PYPTO_BUILD_JOBS={profile}\nexport PYPTO_TEST_JOBS=2\n"
        )
    if linked:
        checkout = source.with_name("linked")
        git(source, "worktree", "add", "-b", "topic", str(checkout))
        container["PYPTO_SRC"] = str(checkout)
    executable(
        Path(container["PATH"].split(":")[0]) / "cmake",
        'printf "cmake %s MAX_JOBS=%s\\n" "$*" "$MAX_JOBS" >> "$CALLS"\n'
        'if [ "$1" = -B ]; then mkdir -p build; touch build/CMakeCache.txt; fi',
    )
    if updater:
        result = run(
            SCRIPTS / "pypto-update.sh", container, "--no-pull", "--skip-doctor", "--jobs", requested
        )
    else:
        container["PYPTO_BUILD_JOBS"] = requested
        result = run(SCRIPTS / "pypto-build", container, "--pypto-only")
    assert result.returncode == 0, result.stderr
    assert f"--parallel {expected} MAX_JOBS={expected}" in Path(container["CALLS"]).read_text()


@pytest.fixture
def runtime_wheel_environment(container: dict[str, str], rootful: bool) -> dict[str, str]:
    """Provision real uv/Python tools accessible to the runtime wheel builder."""
    if rootful and os.getuid() != 0:
        pytest.skip("requires a rootful container for the numeric-owner wheel-build regression")
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required for the offline wheel-cache regression")
    home = Path(container["PYPTO_HOME"])
    container["VIRTUAL_ENV_DISABLE_PROMPT"] = "1"
    container["PYTHONPATH"] = ""
    commands = Path(container["PATH"].split(":")[0])
    private = home / "private-tools"
    if rootful:
        # Reproduce an installation below root's mode-0700 private HOME even
        # when the runner happens to provide a globally accessible uv.
        private.mkdir(mode=0o700)
        shutil.copyfile(Path(uv).resolve(), private / "uv")
        (private / "uv").chmod(0o755)
        uv = str(private / "uv")
    shared_uv = commands / "uv-real"
    shutil.copyfile(Path(uv).resolve(), shared_uv)
    shared_uv.chmod(0o755)
    executable(commands / "uv", f'exec "{shared_uv}" --offline --no-config "$@"')
    python = sys.executable

    def probe_owner_python(candidate: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "setpriv",
                "--reuid=10001",
                "--regid=10001",
                "--clear-groups",
                "env",
                candidate,
                "-I",
                "-c",
                "import os, sysconfig, zipfile; assert os.getuid() == 10001",
            ],
            env=container,
            text=True,
            capture_output=True,
            check=False,
        )

    if rootful:
        # The test runner's Python may live below root's private HOME. Probe
        # candidate interpreters as the numeric owner before creating its venv.
        for parent in home.parents:
            if parent == Path("/tmp"):
                break
            parent.chmod(parent.stat().st_mode | 0o111)
        private_python = private / "python"
        private_python.symlink_to(sys.executable)
        candidates = [
            str(private_python),
            sys.executable,
            shutil.which("python3"),
            "/usr/bin/python3",
            "/usr/local/bin/python3",
        ]
        probes = []
        for candidate in dict.fromkeys(candidates):
            if candidate is None:
                continue
            probe = probe_owner_python(candidate)
            probes.append(probe)
            if probe.returncode == 0:
                python = candidate
                break
        else:
            pytest.fail(
                "No Python executable is runnable by UID 10001: " + "\n".join(p.stderr for p in probes)
            )
        assert probes[0].returncode != 0
    shutil.rmtree(home / "venv")
    for prefix in [home / "venv", Path(container["PTOAS_ROOT"])]:
        created = subprocess.run(
            [
                str(shared_uv),
                "venv",
                "--offline",
                "--no-config",
                "--no-python-downloads",
                "--python",
                python,
                str(prefix),
            ],
            env=container,
            text=True,
            capture_output=True,
            check=False,
        )
        assert created.returncode == 0, created.stderr
    if rootful:
        probe = probe_owner_python(str(home / "venv/bin/python"))
        assert probe.returncode == 0, probe.stderr
    return container


@pytest.mark.parametrize("rootful", [False, True])
@pytest.mark.parametrize("runtime_targets", [None, "build_package_a2a3;build_package_sim"])
def test_build_reinstalls_runtime_source_edits(
    runtime_wheel_environment: dict[str, str], rootful: bool, runtime_targets: str | None
) -> None:
    container = runtime_wheel_environment
    if runtime_targets is not None:
        container["PYPTO_RUNTIME_BUILD_TARGETS"] = runtime_targets
    home = Path(container["PYPTO_HOME"])
    commands = Path(container["PATH"].split(":")[0])
    executable(commands / "cmake", 'if [ "$1" = -B ]; then mkdir -p build; touch build/CMakeCache.txt; fi')
    source = Path(container["PYPTO_SRC"])
    (source / "CMakeLists.txt").touch()
    loader = source / ".claude/skills/testing/load-env.sh"
    loader.parent.mkdir(parents=True)
    loader.write_text("export PYPTO_BUILD_JOBS=2\n")
    packages = source / "python"
    (packages / "pypto").mkdir(parents=True)
    (packages / "pypto/__init__.py").touch()
    (packages / "pypto/language.py").write_text("__all__ = []\n")
    (packages / "nanobind.py").write_text('def cmake_dir(): return "."\n')
    runtime = source / "runtime"
    runtime.mkdir()
    (runtime / "pyproject.toml").write_text(
        '[build-system]\nrequires = []\nbuild-backend = "backend"\nbackend-path = ["."]\n'
        '[project]\nname = "simpler"\nversion = "0.1.0"\n'
    )
    # A standard-library-only wheel backend keeps this an actual uv install,
    # without compiling PyPTO, fetching dependencies, or touching a real runtime.
    (runtime / "backend.py").write_text(
        "import json\n"
        "from pathlib import Path\n"
        "from zipfile import ZipFile\n"
        "def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):\n"
        '    Path("received-settings.json").write_text(json.dumps(config_settings))\n'
        '    name = "simpler-0.1.0-py3-none-any.whl"\n'
        '    info = "simpler-0.1.0.dist-info"\n'
        "    files = {\n"
        '        "simpler/__init__.py": "",\n'
        '        "simpler/task_interface.py": Path("task_interface.py").read_text(),\n'
        '        "simpler_setup/__init__.py": "",\n'
        '        "simpler_setup/pto_isa.py": "def get_pto_isa_clone_path(): return \'fixture\'\\n",\n'
        '        f"{info}/METADATA": "Metadata-Version: 2.1\\nName: simpler\\nVersion: 0.1.0\\n",\n'
        '        f"{info}/WHEEL": "Wheel-Version: 1.0\\nRoot-Is-Purelib: true\\nTag: py3-none-any\\n",\n'
        "    }\n"
        '    files[f"{info}/RECORD"] = "".join(f"{path},,\\n" for path in [*files, f"{info}/RECORD"])\n'
        '    with ZipFile(Path(wheel_directory) / name, "w") as wheel:\n'
        "        for path, content in files.items(): wheel.writestr(path, content)\n"
        "    return name\n"
    )
    container["UV_CACHE_DIR"] = str(home / "uv-cache")
    (runtime / "task_interface.py").touch()
    if rootful:
        # The image environment remains root-owned. Only the bind-mounted
        # checkout belongs to the host's numeric user, absent from /etc/passwd.
        for parent in source.parents:
            if parent == Path("/tmp"):
                break
            parent.chmod(parent.stat().st_mode | 0o111)
        for path in [source, *source.rglob("*")]:
            os.chown(path, 10001, 10001)
    for revision in ["original", "edited-without-metadata-change"]:
        (runtime / "task_interface.py").write_text(f'REVISION = "{revision}"\n')
        result = run(SCRIPTS / "pypto-build", container)
        assert result.returncode == 0, result.stderr
        settings = json.loads((runtime / "received-settings.json").read_text()) or {}
        assert settings.get("build.targets") == runtime_targets
        installed = subprocess.check_output(
            [
                str(home / "venv/bin/python"),
                "-I",
                "-c",
                "from simpler.task_interface import REVISION; print(REVISION)",
            ],
            text=True,
        ).strip()
        assert installed == revision
        # The dist image's next /bin/sh RUN invokes the main venv directly,
        # without activation or PYTHONPATH. Its resolver must be installed there.
        subprocess.run(
            [
                str(home / "venv/bin/python"),
                "-I",
                "-c",
                "from simpler_setup.pto_isa import get_pto_isa_clone_path; "
                "assert get_pto_isa_clone_path() == 'fixture'",
            ],
            check=True,
        )
        subprocess.run(
            [
                str(home / "ptoas/bin/python"),
                "-I",
                "-c",
                "import importlib.util; assert importlib.util.find_spec('simpler') is None",
            ],
            check=True,
        )
        if rootful:
            assert all(path.lstat().st_uid == 10001 for path in source.rglob("*"))
            assert (home / "venv/bin/python").lstat().st_uid == 0


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
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\nproject(probe C CXX)\n"
        "add_executable(probe main.cpp helper.c)\n"
    )
    (source / "main.cpp").write_text(
        '#include <iostream>\nextern "C" int helper();\nint main() { std::cout << TAG + helper(); }\n'
    )
    (source / "helper.c").write_text("int helper(void) { return TAG; }\n")
    loader = source / ".claude/skills/testing/load-env.sh"
    loader.parent.mkdir(parents=True)
    loader.write_text("export PYPTO_BUILD_JOBS=2\n")
    commands = Path(container["PATH"].split(":")[0])
    for variable, compiler in [("CC", "gcc"), ("CXX", "g++")]:
        real = shutil.which(compiler)
        assert real is not None
        wrapper = executable(commands / f"host-{compiler}", f'exec "{real}" -DTAG=1 "$@"')
        container[variable] = str(wrapper)
    container["CCACHE_DISABLE"] = "1"
    subprocess.run(
        ["cmake", "-S", str(source), "-B", str(source / "build"), "-G", generator],
        env=container,
        check=True,
    )
    subprocess.run(["cmake", "--build", str(source / "build"), "--parallel", "2"], env=container, check=True)
    assert subprocess.check_output([str(source / "build/probe")], text=True) == "2"
    (source / "build/foreign-object").touch()
    # Select a different container compiler even when the source mount path is unchanged.
    for variable, compiler in [("CC", "gcc"), ("CXX", "g++")]:
        wrapper = executable(
            commands / f"container-{compiler}", f'exec "{shutil.which(compiler)}" -DTAG=2 "$@"'
        )
        container[variable] = str(wrapper)
    if remapped:
        # A bind mount exposes the same cache at a different absolute path.
        original = source
        source = source.with_name("mounted-source")
        original.rename(source)
        container["PYPTO_SRC"] = str(source)
    result = run(SCRIPTS / "pypto-build", container, "--pypto-only")
    assert result.returncode == 0, result.stderr
    assert not (source / "build/foreign-object").exists()
    assert f"CMAKE_GENERATOR:INTERNAL={generator}" in (source / "build/CMakeCache.txt").read_text()
    assert subprocess.check_output([str(source / "build/probe")], text=True) == "4"

    # Ownership/mode changes alter ctime; activation can alter PATH's spelling.
    # Neither should discard unchanged objects or an unchanged toolchain.
    container["PATH"] += ":/unused/container/path"
    for artifact in source.glob("build/CMakeFiles/probe.dir/*.o"):
        artifact.chmod(artifact.stat().st_mode)
    products = [source / "build/probe", *source.glob("build/CMakeFiles/probe.dir/*.o")]
    assert len(products) == 3
    mtimes = [path.stat().st_mtime_ns for path in products]
    result = run(SCRIPTS / "pypto-build", container, "--pypto-only")
    assert result.returncode == 0, result.stderr
    assert [path.stat().st_mtime_ns for path in products] == mtimes

    # A host build can replace objects without changing CMakeCache.txt or the
    # recorded container compiler's contents. The helper must detect that too.
    cache = (source / "build/CMakeCache.txt").read_bytes()
    for variable, compiler in [("CC", "gcc"), ("CXX", "g++")]:
        executable(Path(container[variable]), f'exec "{shutil.which(compiler)}" -DTAG=9 "$@"')
    subprocess.run(
        ["cmake", "--build", str(source / "build"), "--clean-first", "--parallel", "2"],
        env=container,
        check=True,
    )
    assert subprocess.check_output([str(source / "build/probe")], text=True) == "18"
    for variable, compiler in [("CC", "gcc"), ("CXX", "g++")]:
        executable(Path(container[variable]), f'exec "{shutil.which(compiler)}" -DTAG=2 "$@"')
    assert (source / "build/CMakeCache.txt").read_bytes() == cache
    result = run(SCRIPTS / "pypto-build", container, "--pypto-only")
    assert result.returncode == 0, result.stderr
    assert subprocess.check_output([str(source / "build/probe")], text=True) == "4"

    # A toolchain replacement at the same path must invalidate even a stamped
    # cache. CMake itself does not consider the compiler a source dependency.
    for variable, compiler in [("CC", "gcc"), ("CXX", "g++")]:
        executable(Path(container[variable]), f'exec "{shutil.which(compiler)}" -DTAG=3 "$@"')
    result = run(SCRIPTS / "pypto-build", container, "--pypto-only")
    assert result.returncode == 0, result.stderr
    assert subprocess.check_output([str(source / "build/probe")], text=True) == "6"


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
    # These pure-Python import tests run on both host versions; only the ABI
    # probe is simulated. Package imports still execute the real PTOAS venv.
    ptoas_python = home / "ptoas/bin/python"
    ptoas_python.rename(home / "ptoas/bin/python-real")
    executable(
        ptoas_python,
        'if [ "${1:-}" = -I ] && [ "${2:-}" = -S ]; then echo 3.10; '
        f'else exec "{home}/ptoas/bin/python-real" "$@"; fi',
    )
    ptoas_python.write_text(
        ptoas_python.read_text().replace("#!/bin/bash", "#!/usr/bin/env -S BASH_ENV= bash", 1)
    )
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
    home = Path(ptoas_import_environment["PYPTO_HOME"])
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; source "$1"; command -v python; python -c "import sys; print(sys.prefix)"; ptoas',
            "bash",
            str(SCRIPTS / "pypto-activate.sh"),
        ],
        env=ptoas_import_environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [str(home / "venv/bin/python"), str(home / "venv"), "ptoas 0.65"]


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


@pytest.mark.parametrize("access,expected", [("runtime", "0"), ("privileged", "3"), ("auto", "3")])
@pytest.mark.parametrize("chip,platform", [("a3", "a2a3"), ("a5", "a5")])
def test_launcher_uses_container_visible_device(
    container: dict[str, str], access: str, expected: str, chip: str, platform: str
) -> None:
    commands = Path(container["PATH"].split(":")[0])
    executable(
        commands / "docker",
        'if [ "$1 $2" = "container inspect" ]; then exit 1; fi\n'
        'if [ "$1" = run ]; then printf "%s\\n" "$@" > "$CALLS"; fi',
    )
    container["PYPTO_DEVICE_ACCESS"] = access
    result = run(SCRIPTS / "pypto-dev.sh", container, "run", chip, "3")
    assert result.returncode == 0, result.stderr
    args = Path(container["CALLS"]).read_text().splitlines()
    assert f"PYPTO_DEVICE={expected}" in args
    assert f"PYPTO_PLATFORM={platform}" in args
    if access != "runtime":
        assert "exposes every host device/card" in result.stderr
    if access == "runtime":
        assert "ASCEND_VISIBLE_DEVICES=3" in args
        assert "--runtime=ascend" in args
    else:
        assert "--privileged" in args


@pytest.mark.parametrize("command", ["build", "run"])
def test_launcher_infers_checkout_from_scripts_directory(container: dict[str, str], command: str) -> None:
    source = Path(container["PYPTO_SRC"])
    shutil.copytree(SCRIPTS, source / "docker/scripts")
    shutil.copy2(DOCKERFILE, source / "docker/pypto_dev.Dockerfile")
    executable(
        Path(container["PATH"].split(":")[0]) / "docker",
        'if [ "$1 $2" = "container inspect" ]; then exit 1; fi\nprintf "%s\\n" "$@" > "$CALLS"',
    )
    container["PYPTO_DEVICE_ACCESS"] = "runtime"
    result = run(source / "docker/scripts/pypto-dev.sh", container, command, "a3")
    assert result.returncode == 0, result.stderr
    args = Path(container["CALLS"]).read_text().splitlines()
    if command == "build":
        assert args[args.index("--file") + 1] == str(source / "docker/pypto_dev.Dockerfile")
        assert Path(args[args.index("--file") + 1]).is_file()
        assert args[-1] == str(source)
    else:
        assert f"{source}:/workspace/pypto" in args


@pytest.mark.parametrize("command", ["build", "run"])
def test_launcher_forwards_proxy_without_disclosing_credentials(
    container: dict[str, str], command: str
) -> None:
    commands = Path(container["PATH"].split(":")[0])
    executable(
        commands / "docker",
        'if [ "$1 $2" = "container inspect" ]; then exit 1; fi\n'
        'printf "%s\\n" "$@" >> "$CALLS"\n'
        'test "$HTTPS_PROXY" = "http://review-user:synthetic-password@proxy.invalid:8080"',
    )
    container.update(
        HTTPS_PROXY="http://review-user:synthetic-password@proxy.invalid:8080",
        PYPTO_DEVICE_ACCESS="runtime",
    )
    result = run(SCRIPTS / "pypto-dev.sh", container, command, "a3")
    assert result.returncode == 0, result.stderr
    arguments = Path(container["CALLS"]).read_text()
    assert "synthetic-password" not in result.stdout + result.stderr + arguments
    assert f"{'--build-arg' if command == 'build' else '-e'}\nHTTPS_PROXY\n" in arguments


@pytest.fixture
def launcher(container: dict[str, str], tmp_path: Path) -> dict[str, str]:
    container.update(
        PYPTO_SRC_HOST=container["PYPTO_SRC"],
        PYPTO_LIB_SRC_HOST=container["PYPTO_LIB_SRC"],
        PYPTO_DEVICE_ACCESS="runtime",
        DOCKER_STATE=str(tmp_path / "docker-state"),
        TEST_IMAGE_ID="sha256:first",
    )
    executable(
        Path(container["PATH"].split(":")[0]) / "docker",
        'printf "%s\\n" "$@" >> "$CALLS"\n'
        'case "$1 $2" in\n'
        '    "container inspect") cat "$DOCKER_STATE" ;;\n'
        '    "image inspect") echo "$TEST_IMAGE_ID" ;;\n'
        "    run\\ *)\n"
        '        for arg in "$@"; do\n'
        '            case "$arg" in org.pypto.dev.config=*)\n'
        '                printf "%s %s\\n" "${arg#*=}" "$TEST_IMAGE_ID" > "$DOCKER_STATE" ;;\n'
        "            esac\n"
        "        done ;;\n"
        "esac",
    )
    return container


@pytest.mark.parametrize("alias", [False, True])
def test_launcher_reenters_matching_container(launcher: dict[str, str], tmp_path: Path, alias: bool) -> None:
    first = run(SCRIPTS / "pypto-dev.sh", launcher, "run", "a3")
    assert first.returncode == 0, first.stderr
    if alias:
        link = tmp_path / "source-alias"
        link.symlink_to(launcher["PYPTO_SRC_HOST"])
        launcher["PYPTO_SRC_HOST"] = str(link)
    Path(launcher["CALLS"]).write_text("")
    again = run(SCRIPTS / "pypto-dev.sh", launcher, "run", "a3")
    assert again.returncode == 0, again.stderr
    calls = Path(launcher["CALLS"]).read_text().splitlines()
    assert calls[-3:] == ["start", "-ai", "pypto-dev-a3-0"]
    assert "run" not in calls
    assert "volume" not in calls


@pytest.mark.parametrize("change", ["src", "lib", "image", "image_id", "access", "cache", "legacy"])
def test_launcher_rejects_different_container_configuration(
    launcher: dict[str, str], tmp_path: Path, change: str
) -> None:
    first = run(SCRIPTS / "pypto-dev.sh", launcher, "run", "a3")
    assert first.returncode == 0, first.stderr
    if change in ["src", "lib"]:
        checkout = tmp_path / "other-checkout"
        checkout.mkdir()
        launcher["PYPTO_SRC_HOST" if change == "src" else "PYPTO_LIB_SRC_HOST"] = str(checkout)
    elif change == "legacy":
        Path(launcher["DOCKER_STATE"]).write_text("<no value> sha256:first\n")
    else:
        key, value = {
            "image": ("IMAGE", "different-image"),
            "image_id": ("TEST_IMAGE_ID", "sha256:rebuilt"),
            "access": ("PYPTO_DEVICE_ACCESS", "privileged"),
            "cache": ("PYPTO_CACHE_VOLUME", "different-cache"),
        }[change]
        launcher[key] = value
    Path(launcher["CALLS"]).write_text("")
    result = run(SCRIPTS / "pypto-dev.sh", launcher, "run", "a3")
    assert result.returncode != 0
    assert "docker rm pypto-dev-a3-0" in result.stderr
    calls = Path(launcher["CALLS"]).read_text().splitlines()
    assert not {"start", "run", "rm", "volume"}.intersection(calls)


@pytest.mark.parametrize("skip_guard", [False, True])
@pytest.mark.parametrize(
    "state", ["match", "dirty", "mismatch", "empty-stamp", "missing-stamp", "no-checkout"]
)
def test_doctor_checks_installed_runtime_revision(
    healthy_doctor: dict[str, str], state: str, skip_guard: bool
) -> None:
    container = healthy_doctor
    source = Path(container["PYPTO_SRC"])
    git(source, "init", "--initial-branch=main")
    runtime = source / "runtime"
    git(runtime, "init", "--initial-branch=main")
    (runtime / "abi.h").write_text("original ABI")
    git(runtime, "add", "abi.h", "pyproject.toml")
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
    revision = "0" * 40 if state == "mismatch" else git(runtime, "rev-parse", "HEAD")
    if state == "dirty":
        (runtime / "abi.h").write_text("uncommitted ABI change")
    elif state == "no-checkout":
        shutil.rmtree(runtime / ".git")
    elif state == "empty-stamp":
        revision = ""
    container["PYPTO_SKIP_RUNTIME_PIN_CHECK"] = "1" if skip_guard else "0"
    # A site-packages-style simpler imports successfully despite a stale build.
    # Exercise the real runtime-pin implementation without requiring the native core.
    packages = source / "python"
    for name in ["pypto", "pypto/runtime", "simpler"]:
        package = packages / name
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").touch()
    shutil.copy2(REPO_ROOT / "python/pypto/runtime/runtime_pin.py", packages / "pypto/runtime/runtime_pin.py")
    (packages / "pypto/runtime/__init__.py").write_text("from .runtime_pin import check_runtime_pin\n")
    (packages / "simpler/task_interface.py").touch()
    (packages / "_task_interface.py").write_text(
        "" if state == "missing-stamp" else f'__build_commit__ = "{revision}"\n'
    )
    home = Path(container["PYPTO_HOME"])
    (home / "venv/bin/activate").write_text('export PATH="$PYPTO_HOME/venv/bin:$PATH"\n')
    executable(
        home / "venv/bin/python",
        'if [[ "${2:-}" == *"sys.version_info[:3]"* ]]; then echo 3.10.0; '
        f'else exec "{sys.executable}" "$@"; fi',
    )
    result = run(SCRIPTS / "pypto-doctor", container)
    if state in ("mismatch", "missing-stamp"):
        assert result.returncode != 0
        assert "simpler binding rejected" in result.stdout, result.stdout + result.stderr
        assert "rerun pypto-build" in result.stdout
        assert "environment looks good" not in result.stdout
    elif state == "dirty":
        assert result.returncode == 0, result.stdout + result.stderr
        assert "runtime/ has uncommitted changes" in result.stdout
        assert "cannot prove the binding is current" in result.stdout
    elif state == "match":
        assert result.returncode == 0, result.stdout + result.stderr
        assert "simpler binding matches its source tree" in result.stdout, result.stdout + result.stderr
        assert "simpler binding rejected" not in result.stdout
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert "simpler binding revision is unverified" in result.stdout
    if state != "match":
        assert "simpler binding matches its source tree" not in result.stdout


@pytest.mark.parametrize("checkout", ["pypto", "pypto-lib"])
@pytest.mark.parametrize("has_profile", [False, True])
def test_launcher_mounts_linked_worktree_metadata(
    container: dict[str, str], tmp_path: Path, checkout: str, has_profile: bool
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
    runtime = tmp_path / "runtime"
    git(primary, "clone", str(primary), str(runtime))
    git(runtime, "-c", "protocol.file.allow=always", "submodule", "add", str(primary), "nested module")
    git(runtime, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-am", "nested")
    git(primary, "-c", "protocol.file.allow=always", "submodule", "add", str(runtime), "runtime")
    git(primary, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-am", "runtime")
    linked = tmp_path / "linked checkout"
    git(primary, "worktree", "add", "-b", "topic", str(linked))
    git(linked, "-c", "protocol.file.allow=always", "submodule", "update", "--init", "--recursive")
    profile = primary / ".claude/skills/testing/testing.env"
    if has_profile:
        profile.parent.mkdir(parents=True)
        profile.write_text("export PYPTO_BUILD_JOBS=8\nexport PYPTO_TEST_JOBS=2\n")
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
    assert f"{linked}:/workspace/{checkout}" not in args
    common = git(linked, "rev-parse", "--path-format=absolute", "--git-common-dir")
    assert f"{common}:{common}" in args
    assert f"{linked}:{linked}" in args
    assert f"{'PYPTO_SRC' if checkout == 'pypto' else 'PYPTO_LIB_SRC'}={linked}" in args
    if checkout == "pypto":
        assert args[args.index("--workdir") + 1] == str(linked)
    profile_mount = f"type=bind,src={profile},dst={profile},readonly"
    assert (profile_mount in args) == (has_profile and checkout == "pypto")
    git_env = {arg.split("=", 1)[0]: arg.split("=", 1)[1] for arg in args if arg.startswith("GIT_CONFIG_")}
    safe_paths = {
        git_env[f"GIT_CONFIG_VALUE_{i}"]
        for i in range(int(git_env["GIT_CONFIG_COUNT"]))
        if git_env[f"GIT_CONFIG_KEY_{i}"] == "safe.directory"
    }
    trusted = [linked, linked / "runtime", linked / "runtime/nested module"]
    assert {str(path) for path in trusted}.issubset(safe_paths)
    assert "*" not in safe_paths
    # Opt-in integration check for a host with Docker. The regular helper suite
    # runs inside a container without a Docker socket.
    if image := os.environ.get("PYPTO_TEST_DOCKER_IMAGE"):
        mounts = [f"{common}:{common}", f"{linked}:{linked}"]
        # Cached images may have global safe.directory=*. Disable inherited
        # file configuration so only the launcher's scoped entries can help.
        docker = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "-e",
            "GIT_CONFIG_GLOBAL=/dev/null",
            "-e",
            "GIT_CONFIG_NOSYSTEM=1",
            "--entrypoint",
            "git",
        ]
        for mount in mounts:
            assert mount in args
            docker += ["-v", f"{mount}:ro"]
        # The old mount remaps a relative gitdir; adding a source alias still
        # leaves that relative path invalid under /workspace.
        broken = subprocess.run(
            docker
            + [
                "-v",
                f"{linked}:/workspace/{checkout}:ro",
                image,
                "-c",
                f"safe.directory=/workspace/{checkout}/runtime",
                "-C",
                f"/workspace/{checkout}/runtime",
                "status",
                "--porcelain",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert broken.returncode != 0
        assert "not a git repository" in broken.stderr or "cannot chdir" in broken.stderr
        # Rootful Docker sees host-user ownership. Prove this image enforces
        # ownership checks before testing the actual launcher configuration.
        assert linked.stat().st_uid != 0, "Run this Docker smoke check as a non-root host user"
        untrusted = subprocess.run(
            docker + [image, "-C", str(linked), "status", "--porcelain"],
            text=True,
            capture_output=True,
            check=False,
        )
        assert untrusted.returncode != 0
        assert "dubious ownership" in untrusted.stderr
        # No independent `git -c safe.directory=...` override: use only the
        # protected configuration that the launcher passes to docker run.
        for key, value in git_env.items():
            docker += ["-e", f"{key}={value}"]
        for path in trusted:
            working = subprocess.run(
                docker + [image, "-C", str(path), "status", "--porcelain"],
                text=True,
                capture_output=True,
                check=False,
            )
            assert working.returncode == 0, working.stderr
            assert working.stdout == ""


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
        'assert sys.path.count(os.environ["PYPTO_LIB_SRC"]) == 1; '
        'assert len(os.environ["PATH"].split(":")) == len(set(os.environ["PATH"].split(":"))); '
        "assert len(sys.path) == len(set(sys.path)); "
        'assert os.environ["PYTHONDONTWRITEBYTECODE"] == "1"; '
        'assert os.environ["GIT_OPTIONAL_LOCKS"] == "0"'
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
    selected_helpers = upstream / "docker/scripts"
    shutil.copytree(SCRIPTS, selected_helpers)
    shutil.copytree(Path(container["PYPTO_SRC"]) / "toolchain", upstream / "toolchain")
    (upstream / "build-constraints.txt").write_text("nanobind==2.15.0\n")
    activation = selected_helpers / "pypto-activate.sh"
    activation.write_text(activation.read_text() + "\nexport SELECTED_ACTIVATION=1\n")
    executable(
        selected_helpers / "pypto-build",
        'test "$SELECTED_ACTIVATION" = 1\n'
        'test "$PYPTO_BUILD_JOBS" = 2\n'
        'cmp "$PYPTO_HOME/toolchain/versions.env" "$PYPTO_SRC/toolchain/versions.env"\n'
        'cmp "$PYPTO_HOME/toolchain/build-constraints.txt" "$PYPTO_SRC/build-constraints.txt"\n'
        'echo selected-build >> "$CALLS"',
    )
    git(upstream, "add", "docker", "toolchain", "build-constraints.txt")
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
    dockerfile = DOCKERFILE.read_text()
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

    home = Path(env["PYPTO_HOME"])
    (home / "venv/bin/activate").write_text('export PATH="$PYPTO_HOME/venv/bin:$PATH"\n')
    # A real Python executable ignores BASH_ENV. This shell-based probe must
    # do likewise, or activation recursively launches itself through the probe.
    python_probe = home / "venv/bin/python"
    python_probe.write_text(
        python_probe.read_text().replace("#!/bin/bash", "#!/usr/bin/env -S BASH_ENV= bash", 1)
    )
    for name in ["pypto-activate.sh", "pypto-install-ptoas.sh", "pypto-build"]:
        executable(home / "bin" / name, f'echo "stale context helper: {name}" >&2; exit 88')
    (home / "toolchain").mkdir()
    (home / "toolchain/versions.env").write_text("PTOAS_VERSION=v0.01\n")
    (home / "toolchain/build-constraints.txt").write_text("nanobind==0.0.1\n")
    env.update(
        PATH=f"{home / 'bin'}:{env['PATH']}",
        BASH_ENV=str(home / "bin/pypto-activate.sh"),
        DIST_BUILD_JOBS="2",
    )
    start = dockerfile.index("RUN set -eu; \\\n    export PYPTO_SRC=")
    build_step = dockerfile[start : dockerfile.index("\n\n", start)]
    command = build_step.removeprefix("RUN ").replace("\\\n", "").replace("/workspace", str(workspace))
    # Docker's default /bin/sh does not source BASH_ENV. Its bash helpers must
    # bootstrap without the stale activation, then use the selected activation.
    result = subprocess.run(
        ["/bin/sh", "-eu", "-c", command], env=env, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    calls = Path(env["CALLS"]).read_text()
    assert f"--constraint {workspace / 'pypto/build-constraints.txt'}" in calls
    assert "ptoas-0.65-" in calls
    assert calls.splitlines()[-1] == "selected-build"
    for name in ["pypto-build", "pypto-setup.sh", "pypto-activate.sh"]:
        assert (home / "bin" / name).read_bytes() == (workspace / "pypto/docker/scripts" / name).read_bytes()


def test_dist_rejects_ref_without_helpers(container: dict[str, str]) -> None:
    dockerfile = DOCKERFILE.read_text()
    start = dockerfile.index("RUN set -eu; \\\n    export PYPTO_SRC=")
    step = dockerfile[start : dockerfile.index("\n\n", start)]
    command = (
        step.removeprefix("RUN ").replace("\\\n", "").replace("/workspace/pypto", container["PYPTO_SRC"])
    )
    result = subprocess.run(
        ["/bin/sh", "-eu", "-c", command], env=container, text=True, capture_output=True, check=False
    )
    assert result.returncode != 0
    assert "Selected PYPTO_REF lacks docker/scripts/pypto-setup.sh" in result.stderr
    assert not Path(container["CALLS"]).exists()


@pytest.fixture
def healthy_doctor(container: dict[str, str], tmp_path: Path) -> dict[str, str]:
    """Make all unrelated checks pass without loading native code or an NPU."""
    commands = Path(container["PATH"].split(":")[0])
    cann = tmp_path / "cann"
    driver = tmp_path / "driver"
    (driver / "lib64").mkdir(parents=True)
    (driver / "version.info").write_text("Version=26.0.rc1\ncompatible_version=[V100R001C10]\n")
    (cann / "compiler").mkdir(parents=True)
    (cann / "compiler/version.info").write_text("Version=9.0.0\n")
    for tool in [
        "bin/ccec",
        "bin/ld.lld",
        "tools/hcc/bin/aarch64-target-linux-gnu-g++",
        "tools/hcc/bin/aarch64-target-linux-gnu-gcc",
    ]:
        executable(cann / tool, "exit 0")
    packages = cann / "python/site-packages"
    packages.mkdir(parents=True)
    (packages / "acl.py").write_text(
        "def init(): return 0\n"
        "class Runtime:\n"
        "    def set_device(self, device): return 0\n"
        "    def get_mem_info(self, kind): return (0, 1024 * 1024)\n"
        "rt = Runtime()\n"
    )
    cann_python = executable(commands / "cann-python", f'exec "{sys.executable}" "$@"')
    executable(
        commands / "ls",
        'if [ "${1:-}" = -d ]; then\n'
        '    test "${TEST_MISSING_INTERPRETER:-0}" = 0 || exit 1\n'
        f'    echo "{cann_python}"\n'
        'elif [[ "$1" = /dev/davinci* ]]; then echo /dev/davinci0\n'
        'else exec /bin/ls "$@"; fi',
    )
    executable(commands / "grep", 'if [ "${2:-}" = GLIBCXX_3.4.29 ]; then exit 0; fi\nexec /bin/grep "$@"')
    executable(commands / "npu-smi", "exit 1")
    compiler = executable(commands / "compiler", "echo 11")
    home = Path(container["PYPTO_HOME"])
    (home / "venv/bin/activate").write_text('export PATH="$PYPTO_HOME/venv/bin:$PATH"\n')
    executable(home / "venv/bin/python", "echo 3.10.0")
    executable(Path(container["PTOAS_ROOT"]) / "bin/ptoas", "echo ptoas 0.65")
    source = Path(container["PYPTO_SRC"])
    (source / "CMakeLists.txt").touch()
    (source / "runtime").mkdir()
    (source / "runtime/pyproject.toml").touch()
    container.update(ASCEND_HOME_PATH=str(cann), ASCEND_DRIVER_ROOT=str(driver), CXX=str(compiler))
    return container


@pytest.mark.parametrize(
    "cann,driver,valid",
    [
        ("9.0.0", "26.0.rc1", True),
        ("9.1.T500", "25.6.rc1.b108", True),
        ("9.0.0", "25.6.rc1.b108", False),
        ("9.1.T500", "26.0.rc1", False),
        ("9.0.0", "", False),
        ("", "26.0.rc1", False),
        ("9.9.0", "26.0.rc1", False),
    ],
)
def test_doctor_validates_supported_driver_pairs(
    healthy_doctor: dict[str, str], cann: str, driver: str, valid: bool
) -> None:
    env = healthy_doctor
    (Path(env["ASCEND_HOME_PATH"]) / "compiler/version.info").write_text(f"Version={cann}\n")
    (Path(env["ASCEND_DRIVER_ROOT"]) / "version.info").write_text(
        f"Version={driver}\ncompatible_version=[V100R001C10]\n"
    )
    result = run(SCRIPTS / "pypto-doctor", env)
    assert (result.returncode == 0) == valid, result.stdout + result.stderr
    assert ("environment looks good" in result.stdout) == valid
    assert ("matches supported CANN" in result.stdout) == valid


def test_doctor_warns_when_runtime_hides_driver_version(healthy_doctor: dict[str, str]) -> None:
    (Path(healthy_doctor["ASCEND_DRIVER_ROOT"]) / "version.info").unlink()
    result = run(SCRIPTS / "pypto-doctor", healthy_doctor)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "verify the supported CANN/driver pair on the host" in result.stdout


@pytest.mark.parametrize(
    "driver,chip,arch,valid",
    [
        ("26.0.rc1", "a3", "aarch64", True),
        ("26.0.rc1", "a2", "aarch64", True),
        ("26.0.rc1", "950", "aarch64", False),
        ("26.0.rc1", "unknown", "aarch64", False),
        ("25.6.rc1.b108", "a3", "aarch64", False),
        ("25.6.rc2.b023", "950", "aarch64", True),
        ("25.6.rc2.b023", "a3", "aarch64", False),
        ("25.6.rc1.b108", "950", "aarch64", False),
        ("25.7.rc1.12", "950", "x86_64", True),
        ("25.7.rc1.12", "950", "aarch64", False),
        ("25.6.rc2.b023", "950", "x86_64", False),
        ("25.7.rc1.12", "a3", "x86_64", False),
        ("26.0.rc1", "950", "x86_64", False),
    ],
)
def test_doctor_cann92_uses_mounted_host_metadata(
    healthy_doctor: dict[str, str], tmp_path: Path, driver: str, chip: str, arch: str, valid: bool
) -> None:
    env = healthy_doctor
    executable(Path(env["PATH"].split(":")[0]) / "uname", f"echo {arch}")
    env["PYPTO_CANN_TAG"] = f"9.2.0-{chip}-ubuntu22.04-py3.11-devel"
    (Path(env["ASCEND_HOME_PATH"]) / "compiler/version.info").write_text("Version=9.2.0\n")
    host_metadata = tmp_path / "host-version.info"
    host_metadata.write_text(f"Version={driver}\n")
    env["PYPTO_DRIVER_VERSION_FILE"] = str(host_metadata)
    result = run(SCRIPTS / "pypto-doctor", env)
    assert (result.returncode == 0) == valid, result.stdout + result.stderr
    assert ("image validation configuration" in result.stdout) == valid


def test_doctor_strict_rejects_incomplete_image(healthy_doctor: dict[str, str]) -> None:
    # The diagnostic's default mode warns about missing developer checkouts.
    # Publication must fail even if these warnings would normally exit zero.
    result = run(SCRIPTS / "pypto-doctor", healthy_doctor, "--strict")
    assert result.returncode != 0
    assert "no pypto-lib" in result.stdout
    assert "environment looks good" not in result.stdout


@pytest.mark.parametrize("failure", ["missing", "startup", "import", "probe"])
def test_doctor_rejects_unexecutable_acl_probe(healthy_doctor: dict[str, str], failure: str) -> None:
    env = healthy_doctor
    if failure == "missing":
        env["TEST_MISSING_INTERPRETER"] = "1"
    elif failure == "startup":
        executable(Path(env["PATH"].split(":")[0]) / "cann-python", "exit 42")
    else:
        acl = Path(env["ASCEND_HOME_PATH"]) / "python/site-packages/acl.py"
        acl.write_text(
            'raise ImportError("broken ACL installation")\n'
            if failure == "import"
            else 'def init(): raise RuntimeError("probe failed")\n'
        )
    result = run(SCRIPTS / "pypto-doctor", env)
    assert result.returncode != 0
    assert "environment looks good" not in result.stdout
    expected = "cannot probe device access" if failure == "missing" else "could not run the ACL device probe"
    assert expected in result.stdout
    assert "FAIL" in result.stdout


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
    shutil.copytree(Path(env["PYPTO_SRC"]) / "docker", parent / "docker")
    git(parent, "add", "docker")
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


def docker_script(image: str, root: Path, script: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Run an offline rootful session against only this test's disposable tree."""
    return subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--entrypoint",
            "/bin/bash",
            "-e",
            "BASH_ENV=",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            "-e",
            "GIT_CONFIG_GLOBAL=/dev/null",
            "-e",
            "GIT_CONFIG_NOSYSTEM=1",
            "--mount",
            f"type=bind,src={root},dst={root}",
            "--mount",
            f"type=bind,src={SCRIPTS},dst=/helpers,readonly",
            image,
            "-eu",
            "-c",
            script,
            "bash",
            *arguments,
        ],
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize("linked", [False, True])
def test_rootful_update_preserves_host_ownership(
    tmp_path: Path, updater: tuple[Path, dict[str, str]], linked: bool
) -> None:
    image = os.environ.get("PYPTO_TEST_DOCKER_IMAGE")
    if not image or os.getuid() == 0:
        pytest.skip("requires an opt-in Docker image and a non-root host user")
    _, env = updater
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    git(upstream, "init", "--initial-branch=main")
    git(upstream, "config", "user.name", "Test")
    git(upstream, "config", "user.email", "test@example.invalid")
    shutil.copytree(Path(env["PYPTO_SRC"]) / "docker", upstream / "docker")
    git(upstream, "add", "docker")
    git(upstream, "commit", "-m", "initial helpers")
    primary = tmp_path / "primary"
    git(tmp_path, "clone", str(upstream), str(primary))
    checkout = primary
    if linked:
        checkout = tmp_path / "linked"
        git(primary, "worktree", "add", "-b", "topic", str(checkout), "origin/main")
    (upstream / "new-directory").mkdir()
    (upstream / "new-directory/source.txt").write_text("updated")
    git(upstream, "add", "new-directory")
    git(upstream, "commit", "-m", "new source")
    library = tmp_path / "library"
    git(tmp_path, "clone", str(upstream), str(library))
    command = (
        'export PYPTO_SRC="$1" PYPTO_LIB_SRC="$2" PYPTO_HOME=/tmp/pypto-owner-test CALLS="$3"; '
        "bash /helpers/pypto-update.sh --skip-doctor"
    )
    result = docker_script(image, tmp_path, command, str(checkout), str(library), env["CALLS"])
    assert result.returncode == 0, result.stdout + result.stderr
    for repo in (checkout, primary, library):
        assert all(path.lstat().st_uid == os.getuid() for path in repo.rglob("*"))
        git(repo, "fetch", "origin")
    target = checkout / "new-directory/source.txt"
    target.write_text("host edit")
    calls = Path(env["CALLS"]).read_text()
    result = docker_script(image, tmp_path, command, str(checkout), str(library), env["CALLS"])
    assert result.returncode != 0
    assert "local changes" in result.stderr
    assert target.read_text() == "host edit"
    assert Path(env["CALLS"]).read_text() == calls


@pytest.mark.parametrize("handoff", ["bind", "image"])
def test_native_build_survives_fresh_containers(tmp_path: Path, handoff: str) -> None:
    image = os.environ.get("PYPTO_TEST_DOCKER_IMAGE")
    if not image or os.getuid() == 0:
        pytest.skip("requires an opt-in Docker image and a non-root host user")
    source = tmp_path / "source"
    source.mkdir()
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\nproject(probe C CXX)\nadd_executable(probe main.cpp)\n"
    )
    (source / "main.cpp").write_text("int main() { return 0; }\n")
    loader = source / ".claude/skills/testing/load-env.sh"
    loader.parent.mkdir(parents=True)
    loader.write_text("export PYPTO_BUILD_JOBS=2\n")
    # Each session gets a fresh hostname and root-owned installation prefix.
    # Use real CMake/compilers, mocking only imports unrelated to this probe.
    command = r"""
export PYPTO_SRC="$1" PYPTO_HOME=/tmp/pypto-owner-test PYPTO_LIB_SRC=/missing
export PATH=/opt/pypto/venv/bin:/usr/local/bin:/usr/bin:/bin
export CC=gcc CXX=g++ CCACHE_DISABLE=1
export ASCEND_TOOLKIT_ROOT=/missing
mkdir -p "$PYPTO_HOME/venv/bin" "$PYPTO_HOME/bin"
printf '#!/bin/bash\necho 3.10\n' > "$PYPTO_HOME/venv/bin/python"
chmod 755 "$PYPTO_HOME/venv/bin/python"
printf 'export PATH="$PYPTO_HOME/venv/bin:$PATH"\n' > "$PYPTO_HOME/venv/bin/activate"
printf dependency > "$PYPTO_HOME/venv/dependency"
bash /helpers/pypto-build --pypto-only
owner=$(stat -c %u "$PYPTO_SRC")
test "$(stat -c %u "$PYPTO_HOME/venv/dependency")" = 0
test "$(stat -c %u "$PYPTO_HOME/cache/ccache/$owner/pypto")" = "$owner"
"""
    result = docker_script(image, tmp_path, command, str(source))
    assert result.returncode == 0, result.stdout + result.stderr
    products = [source / "build/probe", *source.glob("build/CMakeFiles/probe.dir/*.o")]
    assert len(products) == 2
    mtimes = [path.stat().st_mtime_ns for path in products]
    if handoff == "image":
        # COPY the completed build into a real image layer, as opposed to merely
        # simulating a changed hostname/ctime on a persistent bind mount.
        shutil.copytree(SCRIPTS, tmp_path / "helpers")
        (tmp_path / "Dockerfile").write_text(f"FROM {image}\nCOPY helpers /helpers\nCOPY source {source}\n")
        tag = f"pypto-build-regression:{hashlib.sha256(str(tmp_path).encode()).hexdigest()[:16]}"
        try:
            built = subprocess.run(
                ["docker", "build", "--network", "none", "--pull=false", "-t", tag, str(tmp_path)],
                text=True,
                capture_output=True,
                check=False,
            )
            assert built.returncode == 0, built.stdout + built.stderr
            checks = (
                'before="$(stat -c %y "$1/build/probe")"; '
                + command
                + '\ntest "$before" = "$(stat -c %y "$1/build/probe")"\n'
            )
            result = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--network",
                    "none",
                    "--entrypoint",
                    "/bin/bash",
                    "-e",
                    "BASH_ENV=",
                    tag,
                    "-eu",
                    "-c",
                    checks,
                    "bash",
                    str(source),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            assert result.returncode == 0, result.stdout + result.stderr
            assert "unknown or changed toolchain" not in result.stdout
        finally:
            subprocess.run(["docker", "image", "rm", tag], capture_output=True, check=False)
        return
    # A handoff that changes ctime while retaining content and mtime is harmless.
    for path in source.rglob("*"):
        if path.is_file():
            path.chmod(path.stat().st_mode)
    result = docker_script(image, tmp_path, command, str(source))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "unknown or changed toolchain" not in result.stdout
    assert [path.stat().st_mtime_ns for path in products] == mtimes
    assert all(path.lstat().st_uid == os.getuid() for path in source.rglob("*"))
    (source / "main.cpp").write_text("int main() { return 7; }\n")
    result = docker_script(image, tmp_path, command, str(source))
    assert result.returncode == 0, result.stdout + result.stderr
    assert subprocess.run([str(source / "build/probe")], check=False).returncode == 7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
