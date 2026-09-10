import tomllib
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FILE_KEYS = {"opening_file", "opponent_file"}


def configured_files(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FILE_KEYS and isinstance(child, str):
                yield child
            yield from configured_files(child)
    elif isinstance(value, list):
        for child in value:
            yield from configured_files(child)


class ConfigAssetTests(unittest.TestCase):
    def test_committed_config_file_inputs_exist(self):
        for config_path in sorted((REPO_ROOT / "configs").rglob("*.toml")):
            config = tomllib.loads(config_path.read_text(encoding="utf-8"))
            for configured_path in configured_files(config):
                with self.subTest(config=config_path, input=configured_path):
                    path = Path(configured_path)
                    if not path.is_absolute():
                        path = REPO_ROOT / path
                    self.assertTrue(path.is_file(), f"missing configured input: {path}")


if __name__ == "__main__":
    unittest.main()
