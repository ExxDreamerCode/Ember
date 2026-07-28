import sys
import tempfile
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

from smoke_test_uci import run_smoke, validate_uci_output  # noqa: E402


class UciSmokeTests(unittest.TestCase):
    def test_accepts_complete_uci_exchange(self):
        validate_uci_output(
            "\n".join(
                [
                    "id name Ember 1.2.3",
                    "id author ExxDreamerCode",
                    "uciok",
                    "readyok",
                    "bestmove e2e4",
                ]
            ),
            "1.2.3",
        )

    def test_rejects_missing_best_move(self):
        with self.assertRaisesRegex(ValueError, "legal best move"):
            validate_uci_output(
                "id name Ember 1.2.3\nuciok\nreadyok\nbestmove 0000\n",
                "1.2.3",
            )

    def test_runs_command_and_checks_cargo_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cargo_toml = root / "Cargo.toml"
            cargo_toml.write_text(
                '[package]\nversion = "1.2.3"\n',
                encoding="utf-8",
            )
            fake_engine = root / "fake_engine.py"
            fake_engine.write_text(
                "\n".join(
                    [
                        "import sys",
                        "for line in sys.stdin:",
                        "    if line.rstrip() == 'uci':",
                        "        print('id name Ember 1.2.3', flush=True)",
                        "        print('uciok', flush=True)",
                        "    elif line.rstrip() == 'isready':",
                        "        print('readyok', flush=True)",
                        "    elif line.startswith('go '):",
                        "        print('bestmove e2e4', flush=True)",
                        "    elif line.rstrip() == 'quit':",
                        "        break",
                    ]
                ),
                encoding="utf-8",
            )

            output = run_smoke(
                [sys.executable, str(fake_engine)],
                cargo_toml,
                timeout=5.0,
            )

            self.assertIn("bestmove e2e4", output)


if __name__ == "__main__":
    unittest.main()
