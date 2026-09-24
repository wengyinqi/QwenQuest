"""Version bookkeeping: one version, recorded in CHANGELOG.md."""

from __future__ import annotations

import re
from importlib import metadata
from pathlib import Path

import pytest

import qwenquest

ROOT = Path(__file__).resolve().parents[1]
SEMVER = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$")


def test_version_is_semver():
    assert SEMVER.match(qwenquest.__version__)


def test_changelog_has_an_entry_for_this_version():
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    releases = re.findall(r"^## \[(\d+\.\d+\.\d+[^\]]*)\]", changelog, flags=re.M)
    assert releases, "CHANGELOG.md has no release sections"
    assert releases[0] == qwenquest.__version__, "newest CHANGELOG entry must match __version__"


def test_installed_metadata_matches():
    try:
        installed = metadata.version("qwenquest")
    except metadata.PackageNotFoundError:
        pytest.skip("package not installed (running from a source tree)")
    assert installed == qwenquest.__version__


def test_upstream_pin_is_a_full_sha():
    assert re.fullmatch(r"[0-9a-f]{40}", qwenquest.UPSTREAM_COMMIT)
