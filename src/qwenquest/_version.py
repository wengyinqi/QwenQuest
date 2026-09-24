"""Single source of truth for the package version (read by pyproject.toml).

Bump it together with a new ``## [x.y.z]`` section in CHANGELOG.md; the
``version-consistency`` CI job and ``tests/test_version.py`` fail otherwise.
"""

__version__ = "0.1.0"
