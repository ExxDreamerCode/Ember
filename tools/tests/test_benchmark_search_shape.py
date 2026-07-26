import importlib.util
import pathlib
import tempfile
import unittest


MODULE_PATH = pathlib.Path(__file__).parents[1] / "benchmark_search_shape.py"
SPEC = importlib.util.spec_from_file_location("benchmark_search_shape", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SearchShapeArtifactTests(unittest.TestCase):
    def test_safe_component_is_portable_and_nonempty(self):
        self.assertEqual(MODULE.safe_component("candidate / startpos"), "candidate-startpos")
        self.assertEqual(MODULE.safe_component("***"), "sample")

    def test_sha256_file_records_the_exact_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "engine"
            path.write_bytes(b"ember")
            self.assertEqual(
                MODULE.sha256_file(path),
                "7cadc15d609c4ae9b4be6265b8e1cace16e6fa78a81ab0c7db82e687a7c867a5",
            )


if __name__ == "__main__":
    unittest.main()
