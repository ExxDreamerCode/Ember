#!/usr/bin/env python3
"""Download and validate all six Ember release archives from one CI run."""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from urllib.parse import urljoin, urlparse
import zipfile
from pathlib import Path

from package_release_archive import ARCHIVE_RE, verify_release_archive


EXPECTED_ARTIFACTS = {
    "ember-release-linux-windows",
    "ember-release-macos",
}
EXPECTED_PLATFORMS = {
    "linux-amd64",
    "linux-arm64",
    "windows-amd64",
    "windows-arm64",
    "macos-amd64",
    "macos-arm64",
}
GITHUB_API_VERSION = "2026-03-10"
REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}
MAX_REDIRECTS = 10


def repository_from_remote(remote: str) -> str:
    patterns = [
        r"^(?:https?://|ssh://git@)github\.com/(?P<repo>[^/]+/[^/]+?)(?:\.git)?$",
        r"^git@github\.com:(?P<repo>[^/]+/[^/]+?)(?:\.git)?$",
    ]
    for pattern in patterns:
        match = re.fullmatch(pattern, remote.strip())
        if match is not None:
            return match.group("repo")
    raise ValueError(f"cannot infer GitHub repository from remote: {remote}")


def default_repository() -> str:
    from_environment = os.environ.get("GITHUB_REPOSITORY")
    if from_environment:
        return from_environment
    remote = subprocess.run(
        ["git", "config", "--get", "remote.origin.url"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return repository_from_remote(remote)


def github_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "ember-release-downloader",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
    }
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token.strip()}"
    return headers


def redirect_headers(
    previous_url: str,
    next_url: str,
    previous_headers: dict[str, str],
) -> dict[str, str]:
    previous_origin = urlparse(previous_url).netloc
    next_origin = urlparse(next_url).netloc
    if previous_origin == next_origin:
        return previous_headers
    return {"User-Agent": previous_headers["User-Agent"]}


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request_bytes(url: str) -> bytes:
    opener = urllib.request.build_opener(NoRedirectHandler)
    headers = github_headers()
    current_url = url
    for _ in range(MAX_REDIRECTS + 1):
        request = urllib.request.Request(current_url, headers=headers)
        try:
            with opener.open(request, timeout=120) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code not in REDIRECT_STATUS_CODES:
                raise
            location = error.headers.get("Location")
            if not location:
                raise
            next_url = urljoin(current_url, location)
            headers = redirect_headers(current_url, next_url, headers)
            current_url = next_url
    raise RuntimeError(f"too many redirects while downloading {url}")



def select_artifacts(payload: dict[str, object]) -> dict[str, str]:
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("GitHub response does not contain an artifact list")

    selected: dict[str, str] = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        name = artifact.get("name")
        download_url = artifact.get("archive_download_url")
        expired = artifact.get("expired")
        if name not in EXPECTED_ARTIFACTS:
            continue
        if expired:
            raise ValueError(f"CI artifact has expired: {name}")
        if not isinstance(download_url, str):
            raise ValueError(f"CI artifact has no download URL: {name}")
        if name in selected:
            raise ValueError(f"duplicate CI artifact: {name}")
        selected[name] = download_url

    missing = EXPECTED_ARTIFACTS - set(selected)
    if missing:
        raise ValueError(f"missing CI artifacts: {', '.join(sorted(missing))}")
    return selected


def release_files_from_actions_zip(contents: bytes) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    with zipfile.ZipFile(io.BytesIO(contents)) as bundle:
        for info in bundle.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            if ARCHIVE_RE.fullmatch(name) is None:
                continue
            if name in result:
                raise ValueError(f"duplicate release archive in CI artifact: {name}")
            result[name] = bundle.read(info)
    return result


def validate_release_set(paths: list[Path]) -> None:
    metadata = [verify_release_archive(path) for path in paths]
    if len(metadata) != len(EXPECTED_PLATFORMS):
        raise ValueError(
            f"expected {len(EXPECTED_PLATFORMS)} release archives, got {len(metadata)}"
        )
    platforms = {item.platform for item in metadata}
    if platforms != EXPECTED_PLATFORMS:
        missing = EXPECTED_PLATFORMS - platforms
        extra = platforms - EXPECTED_PLATFORMS
        raise ValueError(
            "wrong release platform set"
            f"; missing={','.join(sorted(missing)) or '-'}"
            f"; extra={','.join(sorted(extra)) or '-'}"
        )
    versions = {item.version for item in metadata}
    commits = {item.commit for item in metadata}
    if len(versions) != 1 or len(commits) != 1:
        raise ValueError("release archives do not share one version and commit")


def download_release_archives(
    repository: str,
    run_id: int,
    output_dir: Path,
    force: bool,
) -> list[Path]:
    api_url = (
        f"https://api.github.com/repos/{repository}/actions/runs/"
        f"{run_id}/artifacts?per_page=100"
    )
    payload = json.loads(request_bytes(api_url))
    selected = select_artifacts(payload)

    release_files: dict[str, bytes] = {}
    for name in sorted(selected):
        for archive_name, contents in release_files_from_actions_zip(
            request_bytes(selected[name])
        ).items():
            if archive_name in release_files:
                raise ValueError(
                    f"duplicate release archive across CI artifacts: {archive_name}"
                )
            release_files[archive_name] = contents

    with tempfile.TemporaryDirectory(prefix="ember-release-") as temporary:
        temporary_dir = Path(temporary)
        temporary_paths = []
        for name, contents in release_files.items():
            path = temporary_dir / name
            path.write_bytes(contents)
            temporary_paths.append(path)
        validate_release_set(temporary_paths)

        output_dir.mkdir(parents=True, exist_ok=True)
        destinations = []
        for source in sorted(temporary_paths):
            destination = output_dir / source.name
            if destination.exists() and not force:
                raise FileExistsError(f"{destination} already exists; pass --force to replace it")
            shutil.copyfile(source, destination)
            destinations.append(destination)
    return destinations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", type=int, help="GitHub Actions workflow run ID")
    parser.add_argument("--repo", default=None, help="GitHub OWNER/REPOSITORY")
    parser.add_argument("--output-dir", type=Path, default=Path("release"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repository = args.repo or default_repository()
    paths = download_release_archives(repository, args.run_id, args.output_dir, args.force)
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
