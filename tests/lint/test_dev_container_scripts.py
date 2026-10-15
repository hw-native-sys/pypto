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
        if key.startswith(("PYPTO_", "PTOAS_", "GIT_CONFIG_")) or key == "BASH_ENV":
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
