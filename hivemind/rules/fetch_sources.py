"""Download the pinned upstream data into ``data/sources/`` (docs/contracts.md section 1).

Usage::

    python -m hivemind.rules.fetch_sources [--sources-json data/sources.json] [--dest data/sources]

What it does, per source in ``data/sources.json``:

- ``raw``: downloads each listed file from raw.githubusercontent.com at the pinned commit
  and checks its sha256 against the pin (a pin of ``null`` is reported so it can be added).
- ``git-sparse``: makes a shallow, sparse, blob-filtered checkout of exactly the pinned
  commit (only the listed directories) and checks that HEAD is that commit and that the
  checkout is unmodified.

The script is idempotent: files that already exist and match their pin are not fetched
again. It exits with status 1 if anything fails verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCES_JSON = REPO_ROOT / "data" / "sources.json"
DEFAULT_DEST = REPO_ROOT / "data" / "sources"

USER_AGENT = "hivemind-fetch-sources/1"
TIMEOUT_S = 60


@dataclass
class Report:
    """What the fetch did, for the summary printed at the end."""

    fetched: list[str] = field(default_factory=list)
    verified: list[str] = field(default_factory=list)
    unpinned: list[tuple[str, str]] = field(default_factory=list)  # (path, sha256)
    errors: list[str] = field(default_factory=list)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pins(path: Path = DEFAULT_SOURCES_JSON) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _download(url: str, out: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp = out.with_name(out.name + ".part")
    out.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp, tmp.open("wb") as f:
        shutil.copyfileobj(resp, f)
    tmp.replace(out)


def fetch_raw(name: str, spec: dict, dest: Path, report: Report) -> None:
    """Fetch the files of a ``raw`` source and check their sha256 pins."""
    root = dest / spec["dir"]
    for rel, pinned in sorted(spec["files"].items()):
        out = root / rel
        label = f"{name}:{rel}"
        if out.exists() and pinned is not None and sha256_file(out) == pinned:
            report.verified.append(label)
            continue
        url = spec["url"].format(commit=spec["commit"], path=rel)
        try:
            _download(url, out)
        except Exception as exc:  # network errors are reported, not raised
            report.errors.append(f"{label}: download failed: {exc}")
            continue
        digest = sha256_file(out)
        if pinned is None:
            report.unpinned.append((label, digest))
            report.fetched.append(label)
        elif digest != pinned:
            report.errors.append(f"{label}: sha256 {digest} does not match pin {pinned}")
        else:
            report.fetched.append(label)


def _git(args: list[str], cwd: Path) -> str:
    out = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return out.stdout.strip()


def _git_checkout_ok(root: Path, spec: dict) -> bool:
    """True if ``root`` is an unmodified checkout of the pinned commit with all paths."""
    if not (root / ".git").exists():
        return False
    try:
        head = _git(["rev-parse", "HEAD"], root)
        dirty = _git(["status", "--porcelain"], root)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return head == spec["commit"] and not dirty and all((root / p).is_dir() for p in spec["paths"])


def fetch_git_sparse(name: str, spec: dict, dest: Path, report: Report) -> None:
    """Shallow, sparse checkout of exactly the pinned commit (not of whatever the tag says now)."""
    root = dest / spec["dir"]
    label = f"{name}@{spec.get('tag', '')}"
    if _git_checkout_ok(root, spec):
        report.verified.append(f"{label} ({spec['commit'][:12]}, {', '.join(spec['paths'])})")
        return
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    try:
        _git(["init", "-q"], root)
        _git(["remote", "add", "origin", spec["repo"]], root)
        _git(["sparse-checkout", "set", "--cone", *spec["paths"]], root)
        _git(["fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", spec["commit"]], root)
        _git(["checkout", "-q", "FETCH_HEAD"], root)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        stderr = getattr(exc, "stderr", "") or ""
        report.errors.append(f"{label}: git failed: {exc} {stderr.strip()}")
        return
    if _git_checkout_ok(root, spec):
        report.fetched.append(f"{label} ({spec['commit'][:12]}, {', '.join(spec['paths'])})")
    else:
        report.errors.append(f"{label}: checkout does not match commit {spec['commit']}")


def fetch_all(sources_json: Path = DEFAULT_SOURCES_JSON, dest: Path = DEFAULT_DEST) -> Report:
    pins = load_pins(sources_json)
    report = Report()
    dest.mkdir(parents=True, exist_ok=True)
    for name, spec in sorted(pins["sources"].items()):
        if spec["method"] == "raw":
            fetch_raw(name, spec, dest, report)
        elif spec["method"] == "git-sparse":
            fetch_git_sparse(name, spec, dest, report)
        else:
            report.errors.append(f"{name}: unknown method {spec['method']!r}")
    return report


def verify_all(sources_json: Path = DEFAULT_SOURCES_JSON, dest: Path = DEFAULT_DEST) -> list[str]:
    """Check what is on disk against the pins without downloading. Returns problems."""
    pins = load_pins(sources_json)
    problems: list[str] = []
    for name, spec in sorted(pins["sources"].items()):
        root = dest / spec["dir"]
        if spec["method"] == "raw":
            for rel, pinned in sorted(spec["files"].items()):
                path = root / rel
                if not path.exists():
                    problems.append(f"{name}:{rel}: missing")
                elif pinned is None or sha256_file(path) != pinned:
                    problems.append(f"{name}:{rel}: sha256 does not match pin")
        elif not _git_checkout_ok(root, spec):
            problems.append(f"{name}: not a clean checkout of {spec['commit']}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sources-json", type=Path, default=DEFAULT_SOURCES_JSON)
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = ap.parse_args(argv)

    report = fetch_all(args.sources_json, args.dest)
    for label in report.fetched:
        print(f"fetched   {label}")
    for label in report.verified:
        print(f"verified  {label}")
    for label, digest in report.unpinned:
        print(f"UNPINNED  {label}: sha256 {digest} (add it to {args.sources_json})")
    for err in report.errors:
        print(f"ERROR     {err}", file=sys.stderr)
    print(
        f"{len(report.fetched)} fetched, {len(report.verified)} already present and verified, "
        f"{len(report.unpinned)} unpinned, {len(report.errors)} errors -> {args.dest}"
    )
    return 1 if report.errors else 0


if __name__ == "__main__":
    sys.exit(main())
