import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch


TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))
SCRIPT = TOOLS_DIR / "compare_ember_syzygy.py"
SPEC = importlib.util.spec_from_file_location("compare_ember_syzygy", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class EmberSyzygyComparisonTests(unittest.TestCase):
    def test_fifty_move_boundary_withholds_ambiguous_score(self):
        request = {"operation": "dtz", "fen": "7k/8/8/8/8/8/8/1Q2K3 w - - 87 1"}
        reference = {"wdl": 2, "dtz": {"value": 13}, "wdl50": "MaybeWin"}
        candidate = {"selected_move": None}
        ember = {"wdl": 2, "dtz": 13, "wdl50": None, "search_score": None}
        self.assertEqual(MODULE.compare_ember(request, reference, candidate, ember), [])
        ember["search_score"] = 1
        self.assertIn(
            "ember_ambiguous_score",
            MODULE.compare_ember(request, reference, candidate, ember),
        )

    def test_root_translation_detects_selected_move_disagreement(self):
        request = {"operation": "root", "fen": "7k/8/8/8/8/8/8/1Q2K3 w - - 0 1"}
        reference = {"wdl": 2, "dtz": {"value": 13}, "wdl50": "Win"}
        candidate = {"selected_move": "b1b7"}
        ember = {"wdl": 2, "dtz": 13, "wdl50": "Win", "search_score": 1,
                 "selected_move": "b1g6"}
        self.assertIn(
            "ember_selected_move",
            MODULE.compare_ember(request, reference, candidate, ember),
        )


class EmberSyzygyDriverTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.fork = self.root / "fork"
        self.tables = self.root / "tables"
        self.tables.mkdir()
        self.ember = self.root / "ember"
        self.output = self.root / "comparison"
        self.cases = self.fork / "tools/syzygy_compact_cases.jsonl"
        for path in (
            self.fork / "tools/compare_syzygy.py",
            self.fork / "target/debug/examples/syzygy_probe",
            self.fork / "tools/syzygy-reference/target/debug/pyrrhic-syzygy-reference",
            self.fork / "nix/syzygy-3-4-5.json",
            self.ember,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"test input\n")
        self.requests = [
            {"version": 1, "id": f"case-{index}", "operation": "wdl",
             "fen": "7k/8/8/8/8/8/8/1Q2K3 w - - 0 1", "required_files": []}
            for index in range(2)
        ]
        self.batch_requests = [self.requests]
        self.summary_count = 2
        self.reverse_reference = False
        self.truncate_ember = False
        self.write_jsonl(self.cases, self.requests)

    @staticmethod
    def write_jsonl(path, values):
        path.write_bytes(b"".join(json.dumps(value).encode() + b"\n" for value in values))

    @staticmethod
    def replies(requests):
        return [
            {"version": 1, "id": request["id"], "status": "ok", "wdl": 2}
            for request in requests
        ]

    def run_probe(self, command, **kwargs):
        if command[0] == sys.executable:
            backend = Path(command[command.index("--output-dir") + 1])
            backend.mkdir()
            (backend / "summary.json").write_text(json.dumps({
                "compared": self.summary_count, "mismatches": 0,
            }), encoding="utf-8")
            for index, requests in enumerate(self.batch_requests):
                batch = backend / f"batch-{index:06d}"
                batch.mkdir()
                self.write_jsonl(batch / "requests.jsonl", requests)
                replies = self.replies(requests)
                self.write_jsonl(batch / "candidate.stdout.jsonl", replies)
                self.write_jsonl(
                    batch / "reference.stdout.jsonl",
                    replies[::-1] if self.reverse_reference else replies,
                )
            return Mock(returncode=0, stdout=b"backend complete\n", stderr=b"")
        requests = [json.loads(line) for line in kwargs["input"].splitlines()]
        replies = self.replies(requests)
        if self.truncate_ember:
            replies = replies[:-1]
        return Mock(
            returncode=0, stderr=b"",
            stdout=b"".join(json.dumps(reply).encode() + b"\n" for reply in replies),
        )

    def run_driver(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(MODULE.subprocess, "run", side_effect=self.run_probe) as run, \
                redirect_stdout(stdout), redirect_stderr(stderr):
            status = MODULE.main([
                "--pyrrhic-root", str(self.fork), "--ember", str(self.ember),
                "--tables", str(self.tables), "--output-dir", str(self.output),
            ])
        return status, stdout.getvalue(), stderr.getvalue(), run.call_count

    def test_accepts_complete_corpus_with_a_different_size_and_multiple_batches(self):
        self.batch_requests = [[request] for request in self.requests]
        status, stdout, stderr, calls = self.run_driver()
        self.assertEqual((status, calls), (0, 3))
        self.assertEqual(json.loads(stdout), {"compared": 2, "mismatches": 0})
        self.assertEqual(stderr, "")
        invocation = json.loads((self.output / "invocation.json").read_text())
        self.assertEqual(invocation["case_count"], 2)

    def test_rejects_empty_corpus_before_launching_probes(self):
        self.cases.write_bytes(b"")
        status, _, stderr, calls = self.run_driver()
        self.assertEqual((status, calls), (1, 0))
        self.assertIn("compact case file is empty", stderr)
        self.assertFalse(self.output.exists())

    def test_rejects_truncated_backend_summary(self):
        self.summary_count = 1
        status, _, stderr, calls = self.run_driver()
        self.assertEqual((status, calls), (1, 1))
        self.assertIn("backend compared 1 of 2 compact cases", stderr)

    def test_rejects_missing_request_from_backend_batches(self):
        self.batch_requests = [self.requests[:1]]
        status, _, stderr, calls = self.run_driver()
        self.assertEqual((status, calls), (1, 1))
        self.assertIn("backend request batches do not match the compact corpus", stderr)
        self.assertIn("1 of 2 requests", stderr)

    def test_rejects_duplicate_request_even_when_the_count_matches(self):
        self.batch_requests = [[self.requests[0], self.requests[0]]]
        status, _, stderr, calls = self.run_driver()
        self.assertEqual((status, calls), (1, 1))
        self.assertIn("backend request batches do not match the compact corpus", stderr)

    def test_rejects_changed_request_with_the_original_id_and_count(self):
        changed = dict(self.requests[0], fen="7k/8/8/8/8/8/8/1Q2K3 w - - 1 1")
        self.batch_requests = [[changed, self.requests[1]]]
        status, _, stderr, calls = self.run_driver()
        self.assertEqual((status, calls), (1, 1))
        self.assertIn("backend request batches do not match the compact corpus", stderr)

    def test_rejects_reordered_reference_replies_before_running_ember(self):
        self.reverse_reference = True
        status, _, stderr, calls = self.run_driver()
        self.assertEqual((status, calls), (1, 1))
        self.assertIn("invalid reply for case-0", stderr)

    def test_rejects_truncated_ember_replies(self):
        self.truncate_ember = True
        status, _, stderr, calls = self.run_driver()
        self.assertEqual((status, calls), (1, 2))
        self.assertIn("returned 1 of 2 replies", stderr)
