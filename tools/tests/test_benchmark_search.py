import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

from benchmark_search import bench_once  # noqa: E402


class SearchBenchmarkProtocolTests(unittest.TestCase):
    def test_waits_for_bestmove_before_sending_quit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = Path(temp_dir) / "async_engine"
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
                        print("info depth 7 score cp 12 nodes 1234 nps 12000", flush=True)
                        search_finished.set()
                        print("bestmove e2e4", flush=True)

                    for command in sys.stdin:
                        command = command.strip()
                        if command.startswith("go "):
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

            sample = bench_once(
                engine,
                ("fake", "startpos"),
                depth=7,
                hash_mb=64,
                threads=1,
                timeout=5.0,
                disable_book=True,
            )

        self.assertEqual(sample["reported_depth"], 7)
        self.assertEqual(sample["nodes"], 1234)
        self.assertEqual(sample["nps"], 12000)
        self.assertGreaterEqual(sample["wall_seconds"], 0.1)


if __name__ == "__main__":
    unittest.main()
