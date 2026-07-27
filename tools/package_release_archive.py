#!/usr/bin/env python3
"""Create and validate one release-ready Ember binary archive."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import re
import tarfile
import tomllib
import zipfile
from dataclasses import dataclass
from pathlib import Path


PLATFORM_RE = re.compile(r"^(linux|windows|macos)-(amd64|arm64)$")
ARCHIVE_RE = re.compile(
    r"^ember-(?P<version>\d+\.\d+\.\d+)-(?P<commit>[0-9a-f]{8})-"
    r"(?P<platform>(?:linux|windows|macos)-(?:amd64|arm64))"
    r"(?P<extension>\.zip|\.tar\.gz)$"
)


@dataclass(frozen=True)
class ReleaseMetadata:
    version: str
    commit: str
    platform: str
    binary_sha256: str

    @property
    def commit_prefix(self) -> str:
        return self.commit[:8]

    @property
    def binary_name(self) -> str:
        return "ember.exe" if self.platform.startswith("windows-") else "ember"

    @property
    def archive_extension(self) -> str:
        return ".zip" if self.platform.startswith("windows-") else ".tar.gz"

    @property
    def root_name(self) -> str:
        return f"ember-{self.version}-{self.commit_prefix}-{self.platform}"

    @property
    def archive_name(self) -> str:
        return f"{self.root_name}{self.archive_extension}"

    def build_info(self) -> bytes:
        os_name, arch = self.platform.split("-", 1)
        text = "\n".join(
            [
                f"platform={self.platform}",
                f"os={os_name}",
                f"arch={arch}",
                f"version={self.version}",
                f"git_commit={self.commit}",
                f"git_commit_prefix={self.commit_prefix}",
                f"binary_sha256={self.binary_sha256}",
                "",
            ]
        )
        return text.encode("utf-8")

    def checksums(self) -> bytes:
        return f"{self.binary_sha256}  {self.binary_name}\n".encode()


def cargo_version(cargo_toml: Path) -> str:
    with cargo_toml.open("rb") as stream:
        document = tomllib.load(stream)
    version = document.get("package", {}).get("version")
    if not isinstance(version, str) or not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError(f"invalid package version in {cargo_toml}")
    return version


def validate_commit(commit: str) -> str:
    normalized = commit.lower()
    if not re.fullmatch(r"[0-9a-f]{8,64}", normalized):
        raise ValueError("commit must be an 8-64 character hexadecimal hash")
    return normalized


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def zip_member(name: str, data: bytes, mode: int) -> tuple[zipfile.ZipInfo, bytes]:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = mode << 16
    return info, data


def tar_member(name: str, data: bytes, mode: int) -> tuple[tarfile.TarInfo, bytes]:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = mode
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info, data


def create_release_archive(
    binary: Path,
    platform: str,
    commit: str,
    cargo_toml: Path,
    output_dir: Path,
) -> Path:
    if not PLATFORM_RE.fullmatch(platform):
        raise ValueError(f"unsupported release platform: {platform}")
    if not binary.is_file():
        raise FileNotFoundError(binary)

    metadata = ReleaseMetadata(
        version=cargo_version(cargo_toml),
        commit=validate_commit(commit),
        platform=platform,
        binary_sha256=file_sha256(binary),
    )
    binary_data = binary.read_bytes()
    members = [
        (f"{metadata.root_name}/{metadata.binary_name}", binary_data, 0o755),
        (f"{metadata.root_name}/BUILD-INFO.txt", metadata.build_info(), 0o644),
        (f"{metadata.root_name}/SHA256SUMS.txt", metadata.checksums(), 0o644),
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / metadata.archive_name
    if metadata.archive_extension == ".zip":
        with zipfile.ZipFile(archive, "w") as bundle:
            for name, data, mode in members:
                bundle.writestr(*zip_member(name, data, mode))
    else:
        with archive.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w") as bundle:
                    for name, data, mode in members:
                        info, contents = tar_member(name, data, mode)
                        bundle.addfile(info, io.BytesIO(contents))

    verify_release_archive(archive)
    return archive


def parse_build_info(contents: bytes) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in contents.decode("utf-8").splitlines():
        if not line:
            continue
        key, separator, value = line.partition("=")
        if not separator or not key or key in result:
            raise ValueError("invalid BUILD-INFO.txt")
        result[key] = value
    return result


def archive_contents(path: Path) -> dict[str, tuple[bytes, int]]:
    if path.name.endswith(".zip"):
        with zipfile.ZipFile(path) as bundle:
            result = {}
            for info in bundle.infolist():
                if info.is_dir():
                    continue
                result[info.filename] = (bundle.read(info), info.external_attr >> 16)
            return result

    with tarfile.open(path, mode="r:gz") as bundle:
        result = {}
        for info in bundle.getmembers():
            if not info.isfile():
                continue
            extracted = bundle.extractfile(info)
            if extracted is None:
                raise ValueError(f"cannot read {info.name}")
            result[info.name] = (extracted.read(), info.mode)
        return result


def verify_release_archive(path: Path) -> ReleaseMetadata:
    match = ARCHIVE_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"invalid release archive name: {path.name}")

    platform = match.group("platform")
    expected_extension = ".zip" if platform.startswith("windows-") else ".tar.gz"
    if match.group("extension") != expected_extension:
        raise ValueError(f"wrong archive format for {platform}")

    root_name = path.name[: -len(expected_extension)]
    binary_name = "ember.exe" if platform.startswith("windows-") else "ember"
    expected_names = {
        f"{root_name}/{binary_name}",
        f"{root_name}/BUILD-INFO.txt",
        f"{root_name}/SHA256SUMS.txt",
    }
    contents = archive_contents(path)
    if set(contents) != expected_names:
        raise ValueError(f"unexpected archive members in {path.name}")

    build_info = parse_build_info(contents[f"{root_name}/BUILD-INFO.txt"][0])
    required = {
        "platform",
        "os",
        "arch",
        "version",
        "git_commit",
        "git_commit_prefix",
        "binary_sha256",
    }
    if set(build_info) != required:
        raise ValueError(f"incomplete build information in {path.name}")

    os_name, arch = platform.split("-", 1)
    full_commit = validate_commit(build_info["git_commit"])
    if (
        build_info["platform"] != platform
        or build_info["os"] != os_name
        or build_info["arch"] != arch
        or build_info["version"] != match.group("version")
        or build_info["git_commit_prefix"] != match.group("commit")
        or not full_commit.startswith(match.group("commit"))
    ):
        raise ValueError(f"archive metadata does not match {path.name}")

    binary_data, binary_mode = contents[f"{root_name}/{binary_name}"]
    if not platform.startswith("windows-") and binary_mode & 0o111 == 0:
        raise ValueError(f"binary is not executable in {path.name}")
    binary_sha256 = hashlib.sha256(binary_data).hexdigest()
    if build_info["binary_sha256"] != binary_sha256:
        raise ValueError(f"binary checksum mismatch in {path.name}")
    expected_checksum = f"{binary_sha256}  {binary_name}\n".encode()
    if contents[f"{root_name}/SHA256SUMS.txt"][0] != expected_checksum:
        raise ValueError(f"SHA256SUMS.txt mismatch in {path.name}")

    return ReleaseMetadata(
        version=match.group("version"),
        commit=full_commit,
        platform=platform,
        binary_sha256=binary_sha256,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--cargo-toml", type=Path, default=Path("Cargo.toml"))
    parser.add_argument("--output-dir", type=Path, default=Path("release"))
    args = parser.parse_args()

    archive = create_release_archive(
        args.binary,
        args.platform,
        args.commit,
        args.cargo_toml,
        args.output_dir,
    )
    print(archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
