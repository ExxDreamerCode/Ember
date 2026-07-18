import sys
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

from hunt_stupidities import parse_stockfish  # noqa: E402


class ParseStockfishTests(unittest.TestCase):
    def test_preserves_each_multipv_rank(self):
        bestmove, infos = parse_stockfish(
            [
                "info depth 18 seldepth 22 multipv 1 score cp 71 nodes 100 pv e2e4 e7e5",
                "info depth 18 seldepth 23 multipv 2 score cp 55 nodes 100 pv d2d4 d7d5",
                "info depth 18 seldepth 24 multipv 3 score mate -7 nodes 100 pv g1f3 d7d5",
                "bestmove e2e4 ponder e7e5",
            ]
        )

        self.assertEqual(bestmove, "e2e4")
        self.assertEqual([info["multipv"] for info in infos], [1, 2, 3])
        self.assertEqual([info["move"] for info in infos], ["e2e4", "d2d4", "g1f3"])
        self.assertEqual([info["score_cp"] for info in infos], [71, 55, -99_993])


if __name__ == "__main__":
    unittest.main()
