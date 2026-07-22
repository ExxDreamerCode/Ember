import json
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

from measure_elo import smoke  # noqa: E402


class EloSmokeProtocolTests(unittest.TestCase):
    def test_waits_for_async_bestmove_before_quit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            engine = root / "async_engine"
            engine.write_text(
                textwrap.dedent(
                    f"""\
                    #!{sys.executable}
                    import sys
                    import threading
                    import time

                    search_finished = threading.Event()

                    def search():
                        time.sleep(0.1)
                        print("bestmove e2e4", flush=True)
                        search_finished.set()

                    for command in sys.stdin:
                        command = command.strip()
                        if command == "uci":
                            print("uciok", flush=True)
                        elif command == "isready":
                            print("readyok", flush=True)
                        elif command.startswith("go "):
                            threading.Thread(target=search, daemon=True).start()
                        elif command == "quit":
                            if not search_finished.is_set():
                                raise SystemExit(2)
                            break
                    """
                ),
                encoding="utf-8",
            )
            engine.chmod(0o755)
            opponents = root / "opponents.toml"
            opponents.write_text("", encoding="utf-8")
            config = root / "elo.toml"
            config.write_text(
                textwrap.dedent(
                    f"""\
                    [run]

                    [ember]
                    name = "Async Ember"
                    binary = "{engine}"
                    proto = "uci"
                    book = "<embedded>"

                    [ember.options]
                    Hash = "64"
                    Threads = "1"

                    [selection]
                    opponent_file = "{opponents}"
                    """
                ),
                encoding="utf-8",
            )

            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                (root / "results/async-smoke").mkdir(parents=True)
                smoke(config, "async-smoke")
            finally:
                os.chdir(previous_cwd)

            run_dir = root / "results/async-smoke"
            metadata = json.loads(
                (run_dir / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertTrue(metadata["smoke_ok"])
            self.assertIn(
                "bestmove e2e4",
                (run_dir / "smoke.log").read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
