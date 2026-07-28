import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

from download_release_archives import (  # noqa: E402
    redirect_headers,
    release_files_from_actions_zip,
    repository_from_remote,
    select_artifacts,
    validate_release_set,
)
from package_release_archive import create_release_archive  # noqa: E402


class ReleaseSetTests(unittest.TestCase):
    def test_complete_release_set_requires_one_version_and_commit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "input"
            binary.write_bytes(b"ember-test-binary")
            cargo_toml = root / "Cargo.toml"
            cargo_toml.write_text('[package]\nversion = "1.2.3"\n', encoding="utf-8")
            paths = [
                create_release_archive(
                    binary,
                    platform,
                    "0123456789abcdef",
                    cargo_toml,
                    root,
                )
                for platform in (
                    "linux-amd64",
                    "linux-arm64",
                    "windows-amd64",
                    "windows-arm64",
                    "macos-amd64",
                    "macos-arm64",
                )
            ]

            validate_release_set(paths)


class ActionsArtifactTests(unittest.TestCase):
    def test_selects_both_unexpired_build_artifacts(self):
        payload = {
            "artifacts": [
                {
                    "name": "ember-release-linux-windows",
                    "archive_download_url": "https://example/linux",
                    "expired": False,
                },
                {
                    "name": "ember-release-macos",
                    "archive_download_url": "https://example/macos",
                    "expired": False,
                },
                {
                    "name": "unrelated",
                    "archive_download_url": "https://example/other",
                    "expired": False,
                },
            ]
        }

        self.assertEqual(
            select_artifacts(payload),
            {
                "ember-release-linux-windows": "https://example/linux",
                "ember-release-macos": "https://example/macos",
            },
        )

    def test_extracts_only_release_archives_from_actions_wrapper(self):
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as bundle:
            bundle.writestr("release/ember-1.2.3-01234567-linux-amd64.tar.gz", b"linux")
            bundle.writestr("notes.txt", b"ignored")

        self.assertEqual(
            release_files_from_actions_zip(output.getvalue()),
            {"ember-1.2.3-01234567-linux-amd64.tar.gz": b"linux"},
        )

    def test_parses_https_and_ssh_github_remotes(self):
        expected = "ExxDreamerCode/Ember"
        self.assertEqual(
            repository_from_remote("https://github.com/ExxDreamerCode/Ember.git"),
            expected,
        )
        self.assertEqual(
            repository_from_remote("git@github.com:ExxDreamerCode/Ember.git"),
            expected,
        )

    def test_cross_host_artifact_redirect_drops_github_auth_headers(self):
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": "Bearer secret",
            "User-Agent": "ember-release-downloader",
            "X-GitHub-Api-Version": "2026-03-10",
        }

        self.assertEqual(
            redirect_headers(
                "https://api.github.com/repos/starius/Ember/actions/artifacts/1/zip",
                "https://productionresultssa1.blob.core.windows.net/actions-results/1",
                headers,
            ),
            {"User-Agent": "ember-release-downloader"},
        )
        self.assertEqual(
            redirect_headers(
                "https://api.github.com/repos/starius/Ember/actions/artifacts/1/zip",
                "https://api.github.com/repos/starius/Ember/actions/artifacts/2/zip",
                headers,
            ),
            headers,
        )


if __name__ == "__main__":
    unittest.main()
