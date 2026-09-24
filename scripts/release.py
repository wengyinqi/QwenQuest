#!/usr/bin/env python3
"""Version / release helper (stdlib only, used by CI and by maintainers).

    python scripts/release.py check [--tag vX.Y.Z]   # sources agree (CI: "version" job)
    python scripts/release.py notes X.Y.Z            # CHANGELOG section -> stdout
    python scripts/release.py bump {major|minor|patch|X.Y.Z} [--date YYYY-MM-DD]

``bump`` rewrites src/qwenquest/_version.py and turns the ``[Unreleased]``
section of CHANGELOG.md into ``[X.Y.Z] - date`` (plus the compare links).
It does not commit or tag; see docs/versioning.md for the full flow.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "src" / "qwenquest" / "_version.py"
CHANGELOG = ROOT / "CHANGELOG.md"
REPO_URL = "https://github.com/wengyinqi/QwenQuest"

SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?$")
VERSION_LINE = re.compile(r'^__version__ = "([^"]+)"$', re.M)
RELEASE_HEADING = re.compile(r"^## \[(\d+\.\d+\.\d+[^\]]*)\](?: - (\d{4}-\d{2}-\d{2}))?\s*$", re.M)


def read_version() -> str:
    m = VERSION_LINE.search(VERSION_FILE.read_text(encoding="utf-8"))
    if not m:
        raise SystemExit(f"no __version__ line in {VERSION_FILE}")
    return m.group(1)


def changelog_releases(text: str) -> list[tuple[str, str | None]]:
    return [(m.group(1), m.group(2)) for m in RELEASE_HEADING.finditer(text)]


def section(text: str, version: str) -> str:
    pattern = re.compile(
        rf"^## \[{re.escape(version)}\].*?$(.*?)(?=^## \[|^\[[^\]]+\]: |\Z)", re.M | re.S
    )
    m = pattern.search(text)
    if not m:
        raise SystemExit(f"CHANGELOG.md has no section for {version}")
    return m.group(1).strip() + "\n"


def cmd_check(args: argparse.Namespace) -> int:
    errors = []
    version = read_version()
    if not SEMVER.match(version):
        errors.append(f"__version__ {version!r} is not semantic versioning")
    text = CHANGELOG.read_text(encoding="utf-8")
    if "## [Unreleased]" not in text:
        errors.append("CHANGELOG.md lost its '## [Unreleased]' section")
    releases = changelog_releases(text)
    if not releases:
        errors.append("CHANGELOG.md has no release section")
    elif releases[0][0] != version:
        errors.append(f"newest CHANGELOG release is {releases[0][0]}, __version__ is {version}")
    elif releases[0][1] is None:
        errors.append(f"CHANGELOG section {version} has no release date")
    if f"[{version}]: " not in text:
        errors.append(f"CHANGELOG.md lacks the link reference '[{version}]: ...'")
    if args.tag is not None and args.tag != f"v{version}":
        errors.append(f"tag {args.tag} does not match __version__ {version} (expected v{version})")
    for e in errors:
        print(f"::error::{e}")
    if not errors:
        print(f"version {version} is consistent" + (f" with tag {args.tag}" if args.tag else ""))
    return 1 if errors else 0


def cmd_notes(args: argparse.Namespace) -> int:
    sys.stdout.write(section(CHANGELOG.read_text(encoding="utf-8"), args.version))
    return 0


def next_version(current: str, part: str) -> str:
    if SEMVER.match(part):
        return part
    m = SEMVER.match(current)
    if not m:
        raise SystemExit(f"current version {current!r} is not semver")
    major, minor, patch = (int(x) for x in m.groups()[:3])
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    if part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise SystemExit(f"bump expects major|minor|patch|X.Y.Z, got {part!r}")


def cmd_bump(args: argparse.Namespace) -> int:
    current = read_version()
    new = next_version(current, args.part)
    date = args.date or dt.date.today().isoformat()
    text = CHANGELOG.read_text(encoding="utf-8")
    if any(v == new for v, _ in changelog_releases(text)):
        raise SystemExit(f"CHANGELOG.md already has a section for {new}")
    text = text.replace("## [Unreleased]", f"## [Unreleased]\n\n## [{new}] - {date}", 1)
    text = re.sub(
        r"^\[Unreleased\]: .*$",
        f"[Unreleased]: {REPO_URL}/compare/v{new}...HEAD\n"
        f"[{new}]: {REPO_URL}/compare/v{current}...v{new}",
        text,
        count=1,
        flags=re.M,
    )
    CHANGELOG.write_text(text, encoding="utf-8")
    VERSION_FILE.write_text(
        VERSION_LINE.sub(f'__version__ = "{new}"', VERSION_FILE.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    print(f"{current} -> {new}: edit the new CHANGELOG section, commit, merge, then tag v{new}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--tag")
    c.set_defaults(func=cmd_check)
    n = sub.add_parser("notes")
    n.add_argument("version")
    n.set_defaults(func=cmd_notes)
    b = sub.add_parser("bump")
    b.add_argument("part")
    b.add_argument("--date")
    b.set_defaults(func=cmd_bump)
    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
