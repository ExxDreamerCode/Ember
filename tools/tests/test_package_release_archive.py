import sys
import tempfile
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

from package_release_archive import (  # noqa: E402
    create_release_archive,
    verify_release_archive,
)


class ReleaseArchiveTests(unittest.TestCase):
    def test_platform_archive_formats_and_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "input"
            binary.write_bytes(b"ember-test-binary")
            cargo_toml = root / "Cargo.toml"
            cargo_toml.write_text('[package]\nversion = "1.2.3"\n', encoding="utf-8")

            linux = create_release_archive(
                binary,
                "linux-amd64",
                "0123456789abcdef",
                cargo_toml,
                root,
            )
            windows = create_release_archive(
                binary,
                "windows-arm64",
                "0123456789abcdef",
                cargo_toml,
                root,
            )

            self.assertEqual(linux.name, "ember-1.2.3-01234567-linux-amd64.tar.gz")
            self.assertEqual(windows.name, "ember-1.2.3-01234567-windows-arm64.zip")
            self.assertEqual(verify_release_archive(linux).platform, "linux-amd64")
            self.assertEqual(verify_release_archive(windows).platform, "windows-arm64")

if __name__ == "__main__":
    unittest.main()
