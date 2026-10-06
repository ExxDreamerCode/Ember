"""The independent test tools must come from Ember's locked fork revision."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pyrrhic_tools  # noqa: E402


REVISION = "a" * 40
SOURCE = (
    "git+https://github.com/starius/pyrrhic-rs"
    f"?rev={REVISION}#{REVISION}"
)


class PyrrhicToolsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()
        (self.root / "Cargo.toml").write_text(
            '[dependencies]\npyrrhic-rs = { git = '
            f'"https://github.com/starius/pyrrhic-rs", rev = "{REVISION}" }}\n',
            encoding="utf-8",
        )
        (self.root / "Cargo.lock").write_text(
            f'[[package]]\nname = "pyrrhic-rs"\nsource = "{SOURCE}"\n',
            encoding="utf-8",
        )

    def test_uses_cargo_checkout_with_the_locked_source(self):
        result = Mock()
        result.stdout = json.dumps({
            "packages": [{
                "name": "pyrrhic-rs", "source": SOURCE,
                "manifest_path": str(self.checkout / "Cargo.toml"),
            }]
        })
        with patch.object(pyrrhic_tools, "ROOT", self.root), \
                patch.object(pyrrhic_tools.subprocess, "run", return_value=result) as run:
            self.assertEqual(pyrrhic_tools.pyrrhic_root(), self.checkout.resolve(strict=True))
        self.assertEqual(run.call_args.args[0][:3], ["cargo", "metadata", "--locked"])
        self.assertEqual(run.call_args.kwargs["env"]["CARGO_NET_GIT_FETCH_WITH_CLI"], "true")

    def test_rejects_a_different_cargo_source(self):
        result = Mock()
        result.stdout = json.dumps({
            "packages": [{
                "name": "pyrrhic-rs", "source": SOURCE[:-40] + "b" * 40,
                "manifest_path": str(self.checkout / "Cargo.toml"),
            }]
        })
        with patch.object(pyrrhic_tools, "ROOT", self.root), \
                patch.object(pyrrhic_tools.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(ValueError, "pinned Pyrrhic fork"):
                pyrrhic_tools.pyrrhic_root()

    def test_rejects_a_stale_lock_before_resolving_tools(self):
        bad_source = SOURCE[:-40] + "b" * 40
        (self.root / "Cargo.lock").write_text(
            f'[[package]]\nname = "pyrrhic-rs"\nsource = "{bad_source}"\n',
            encoding="utf-8",
        )
        with patch.object(pyrrhic_tools, "ROOT", self.root), \
                patch.object(pyrrhic_tools.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "Cargo.lock does not pin"):
                pyrrhic_tools.pyrrhic_root()
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
