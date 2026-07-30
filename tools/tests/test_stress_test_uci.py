import sys
import tempfile
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

from stress_test_uci import assert_no_crash_output, run_stress  # noqa: E402


class UciStressTests(unittest.TestCase):
    def test_rejects_crash_markers(self):
        with self.assertRaisesRegex(RuntimeError, "overflowed"):
            assert_no_crash_output("thread 'main' has overflowed its stack")

    def test_runs_all_stress_cases(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_engine = root / "fake_engine.py"
            fake_engine.write_text(
                "\n".join(
                    [
                        "import sys",
                        "for line in sys.stdin:",
                        "    command = line.rstrip()",
                        "    if command == 'uci':",
                        "        print('id name Ember 1.2.3', flush=True)",
                        "        print('uciok', flush=True)",
                        "    elif command == 'isready':",
                        "        print('readyok', flush=True)",
                        "    elif command.startswith('go '):",
                        "        print('bestmove e2e4', flush=True)",
                        "    elif command == 'quit':",
                        "        break",
                    ]
                ),
                encoding="utf-8",
            )

            outputs = run_stress([sys.executable, str(fake_engine)], timeout=5.0)

            self.assertEqual(
                set(outputs),
                {"malformed-input", "queued-quit", "search-eof"},
            )
            self.assertIn("bestmove e2e4", outputs["malformed-input"])


if __name__ == "__main__":
    unittest.main()
