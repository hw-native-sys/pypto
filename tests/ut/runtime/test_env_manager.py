# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for ``pypto.runtime.env_manager.get_simpler_root``.

The function has three resolution strategies and one error path. We exercise
each branch by monkey-patching ``env_manager.__file__`` and the cached
``_SIMPLER_SETUP_PROJECT_ROOT`` value, so the tests stay hermetic and never
need a real ``pip install``.
"""

import pytest
from pypto.runtime import env_manager


@pytest.fixture
def fresh_env_manager(monkeypatch):
    """Return ``env_manager`` with the resolution cache cleared per test."""
    monkeypatch.setattr(env_manager, "_simpler_root", None)
    return env_manager


class TestGetSimplerRoot:
    """Verify each resolution branch of ``get_simpler_root``."""

    def test_source_tree_install(self, fresh_env_manager, tmp_path, monkeypatch):
        """Editable install: ``<repo>/runtime`` is found via the ``__file__`` anchor.

        Hermetic: build a fake repo with the expected ``python/pypto/runtime/``
        depth so the ``parents[3] / "runtime"`` lookup hits a controlled dir,
        and ensure the test passes regardless of how/where pypto is installed.
        """
        fake_repo = tmp_path / "fake_repo"
        fake_runtime = fake_repo / "runtime"
        fake_runtime.mkdir(parents=True)
        fake_module_dir = fake_repo / "python" / "pypto" / "runtime"
        fake_module_dir.mkdir(parents=True)
        monkeypatch.setattr(fresh_env_manager, "__file__", str(fake_module_dir / "env_manager.py"))

        root = fresh_env_manager.get_simpler_root()
        assert root == fake_runtime

    def test_simpler_setup_fallback(self, fresh_env_manager, tmp_path, monkeypatch):
        """Pip-installed pypto + pip-installed simpler.

        Simulates the post-install layout that the ``pypto-lib`` CI hits:
        ``__file__`` lives under ``site-packages/pypto/runtime/`` so the
        source-tree anchor misses, and ``simpler_setup.environment.PROJECT_ROOT``
        points into ``site-packages/simpler_setup/_assets/`` where the wheel
        installer placed ``src/``.
        """
        fake_install = tmp_path / "fake_site_packages" / "pypto" / "runtime"
        fake_install.mkdir(parents=True)
        monkeypatch.setattr(fresh_env_manager, "__file__", str(fake_install / "env_manager.py"))

        fake_assets = tmp_path / "simpler_assets"
        (fake_assets / "src").mkdir(parents=True)
        monkeypatch.setattr(fresh_env_manager, "_SIMPLER_SETUP_PROJECT_ROOT", fake_assets)

        root = fresh_env_manager.get_simpler_root()
        assert root == fake_assets

    def test_cwd_walk_fallback(self, fresh_env_manager, tmp_path, monkeypatch):
        """Legacy ad-hoc layout: walk up from cwd to find ``.git`` + ``runtime/``.

        Triggered only when both the source-tree anchor and the simpler wheel
        are unavailable.
        """
        fake_install = tmp_path / "fake_site_packages" / "pypto" / "runtime"
        fake_install.mkdir(parents=True)
        monkeypatch.setattr(fresh_env_manager, "__file__", str(fake_install / "env_manager.py"))
        monkeypatch.setattr(fresh_env_manager, "_SIMPLER_SETUP_PROJECT_ROOT", None)

        fake_repo = tmp_path / "fake_repo"
        (fake_repo / ".git").mkdir(parents=True)
        (fake_repo / "runtime").mkdir(parents=True)
        monkeypatch.chdir(fake_repo)

        root = fresh_env_manager.get_simpler_root()
        assert root == fake_repo / "runtime"

    def test_simpler_setup_skipped_when_src_missing(self, fresh_env_manager, tmp_path, monkeypatch):
        """An importable ``simpler_setup`` whose ``_assets/src`` is absent must not match.

        Guards against partial installs where ``simpler_setup`` is on the path
        but the wheel's ``src/`` payload was never placed.
        """
        fake_install = tmp_path / "fake_site_packages" / "pypto" / "runtime"
        fake_install.mkdir(parents=True)
        monkeypatch.setattr(fresh_env_manager, "__file__", str(fake_install / "env_manager.py"))

        broken_assets = tmp_path / "broken_simpler"
        broken_assets.mkdir()
        monkeypatch.setattr(fresh_env_manager, "_SIMPLER_SETUP_PROJECT_ROOT", broken_assets)

        fake_repo = tmp_path / "legacy_repo"
        (fake_repo / ".git").mkdir(parents=True)
        (fake_repo / "runtime").mkdir(parents=True)
        monkeypatch.chdir(fake_repo)

        root = fresh_env_manager.get_simpler_root()
        assert root == fake_repo / "runtime"

    def test_all_paths_fail_raises(self, fresh_env_manager, tmp_path, monkeypatch):
        """When no strategy succeeds, raise ``OSError`` with actionable guidance."""
        fake_install = tmp_path / "fake_site_packages" / "pypto" / "runtime"
        fake_install.mkdir(parents=True)
        monkeypatch.setattr(fresh_env_manager, "__file__", str(fake_install / "env_manager.py"))
        monkeypatch.setattr(fresh_env_manager, "_SIMPLER_SETUP_PROJECT_ROOT", None)

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        monkeypatch.chdir(empty_dir)

        with pytest.raises(OSError, match=r"Cannot find runtime/"):
            fresh_env_manager.get_simpler_root()

    def test_result_is_cached(self, fresh_env_manager, tmp_path, monkeypatch):
        """Repeated calls return the cached path object without re-resolving.

        Hermetic: seed a guaranteed-success resolution path via the
        ``simpler_setup`` fallback so the cache check does not depend on the
        ambient environment having a real source tree or git checkout.
        """
        fake_install = tmp_path / "fake_site_packages" / "pypto" / "runtime"
        fake_install.mkdir(parents=True)
        monkeypatch.setattr(fresh_env_manager, "__file__", str(fake_install / "env_manager.py"))

        fake_assets = tmp_path / "simpler_assets"
        (fake_assets / "src").mkdir(parents=True)
        monkeypatch.setattr(fresh_env_manager, "_SIMPLER_SETUP_PROJECT_ROOT", fake_assets)

        first = fresh_env_manager.get_simpler_root()
        second = fresh_env_manager.get_simpler_root()
        assert first is second
        assert first == fake_assets
