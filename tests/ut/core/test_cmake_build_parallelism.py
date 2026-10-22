# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exercise the real ExternalProject configuration without compiling PyPTO."""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def external_project(tmp_path: Path) -> Path:
    for tool in ("cmake", "make", "ninja"):
        if shutil.which(tool) is None:
            pytest.skip(f"{tool} is required for ExternalProject tests")
    source = tmp_path / "source"
    dependency = source / "3rdparty/libbacktrace"
    dependency.mkdir(parents=True)
    # Use the production module with a tiny Autotools-shaped dependency. Its
    # recipes record actual recursive Make flags, not just generated commands.
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.15)\n"
        "project(parallelism_test NONE)\n"
        f'include("{REPO_ROOT / "cmake/libbacktrace.cmake"}")\n'
    )
    configure = dependency / "configure"
    configure.write_text('#!/bin/sh\ncp "$(dirname "$0")/Makefile.in" Makefile\n')
    configure.chmod(0o755)
    (dependency / "Makefile.in").write_text(
        ".PHONY: all install\n"
        "all:\n\t@printf '%s\\n' '$(MAKEFLAGS)' > build-flags.txt\n"
        "install:\n\t@printf '%s\\n' '$(MAKEFLAGS)' > install-flags.txt\n"
    )
    return source


def configure_project(
    source: Path,
    generator: str,
    budget: str | None = None,
    env_budget: str | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    # Isolate budget selection from the developer's resource-limit loader.
    for name in ("CMAKE_BUILD_PARALLEL_LEVEL", "MAKEFLAGS", "MFLAGS", "CMAKE_GENERATOR"):
        env.pop(name, None)
    if env_budget is not None:
        env["CMAKE_BUILD_PARALLEL_LEVEL"] = env_budget
    command = ["cmake", "-S", str(source), "-B", str(source.parent / "build"), "-G", generator]
    if budget is not None:
        command.append(f"-DPYPTO_EXTERNAL_BUILD_JOBS:STRING={budget}")
    return subprocess.run(command, env=env, text=True, capture_output=True, check=False)


def generated_commands(source: Path, generator: str) -> list[str]:
    build = source.parent / "build"
    path = build / (
        "build.ninja" if generator == "Ninja" else "CMakeFiles/project_libbacktrace.dir/build.make"
    )
    return [line for line in path.read_text().splitlines() if " && " in line and "make" in line.lower()]


@pytest.mark.parametrize(
    ("budget", "env_budget", "expected"),
    [
        (None, None, 1),
        (None, "", 1),
        (None, "2", 2),
        ("2", None, 2),
        ("2", "3", 2),
        ("2", "bad", 2),
        ("2 ", None, 2),  # CMake strips trailing whitespace from -D values.
    ],
)
def test_ninja_budget(
    external_project: Path, budget: str | None, env_budget: str | None, expected: int
) -> None:
    result = configure_project(external_project, "Ninja", budget, env_budget)
    assert result.returncode == 0, result.stdout + result.stderr
    commands = generated_commands(external_project, "Ninja")
    # Both steps must override inherited MAKEFLAGS with the selected finite cap.
    assert any(f"make -j{expected} &&" in line for line in commands)
    assert any(f"make -j{expected} install &&" in line for line in commands)
    assert not any(re.search(r"\bmake -j(?:\s|$)", line) for line in commands)


@pytest.mark.parametrize("budget", ["", "0", "-1", "1.5", "abc", "2;3", " 2"])
def test_ninja_rejects_invalid_explicit_budget(external_project: Path, budget: str) -> None:
    result = configure_project(external_project, "Ninja", budget, "2")
    assert result.returncode != 0
    assert "PYPTO_EXTERNAL_BUILD_JOBS must be a positive integer" in result.stderr


@pytest.mark.parametrize("env_budget", ["0", "-1", "1.5", "abc", "2;3", " 2", "2 "])
def test_ninja_rejects_invalid_environment_budget(external_project: Path, env_budget: str) -> None:
    result = configure_project(external_project, "Ninja", env_budget=env_budget)
    assert result.returncode != 0
    assert "CMAKE_BUILD_PARALLEL_LEVEL must be a positive integer" in result.stderr


def test_ninja_reconfigure_reads_current_environment(external_project: Path) -> None:
    for env_budget, expected in [("2", 2), ("3", 3), (None, 1)]:
        result = configure_project(external_project, "Ninja", env_budget=env_budget)
        assert result.returncode == 0, result.stdout + result.stderr
        assert any(f"make -j{expected} &&" in line for line in generated_commands(external_project, "Ninja"))


@pytest.mark.parametrize("budget", [None, "2"])
def test_make_uses_recursive_defaults(external_project: Path, budget: str | None) -> None:
    result = configure_project(external_project, "Unix Makefiles", budget, "3")
    assert result.returncode == 0, result.stdout + result.stderr
    commands = generated_commands(external_project, "Unix Makefiles")
    assert any(line.endswith("&& $(MAKE)") for line in commands)
    assert any(line.endswith("&& $(MAKE) install") for line in commands)
    assert not any(" -j" in line for line in commands)


@pytest.mark.parametrize("generator", ["Unix Makefiles", "Ninja"])
@pytest.mark.parametrize("requested_jobs", [1, 2])
def test_build_and_install_receive_budget(
    external_project: Path, generator: str, requested_jobs: int
) -> None:
    build_jobs = min(int(os.environ.get("PYPTO_BUILD_JOBS", "2")), requested_jobs)
    budget = build_jobs
    result = configure_project(external_project, generator, str(budget))
    assert result.returncode == 0, result.stdout + result.stderr
    env = os.environ.copy()
    env.pop("MFLAGS", None)
    # Simulate a different inherited budget: Ninja's inner Make must still use
    # the configured cap; recursive Make must honor the outer --parallel value.
    env["MAKEFLAGS"] = f"-j{build_jobs + 1}"
    build = external_project.parent / "build"
    result = subprocess.run(
        ["cmake", "--build", str(build), "--parallel", str(build_jobs), "--target", "project_libbacktrace"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "jobserver unavailable" not in output
    assert "resetting jobserver" not in output
    expected = build_jobs if generator == "Unix Makefiles" else budget
    for step in ("build", "install"):
        flags = (build / f"3rdparty/libbacktrace/build/{step}-flags.txt").read_text()
        if generator == "Unix Makefiles" and build_jobs > 1:
            assert "--jobserver-auth=" in flags or "--jobserver-fds=" in flags
            # Older GNU Make versions omit the count when advertising a
            # jobserver; bare -j here is safe only with that shared token pool.
            assert "-j" in flags.split() or f"-j{expected}" in flags.split(), flags
        else:
            assert f"-j{expected}" in flags.split(), flags


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
