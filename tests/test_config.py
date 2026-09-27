import json
import tempfile
import unittest
from pathlib import Path

from aicover.config import resolve_config
from aicover.errors import ConfigurationError


class ConfigTests(unittest.TestCase):
    def test_precedence_and_explicit_false(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"pipeline": {"mix": {"enabled": False}, "master": {"enabled": False}}}))
            project = {"pipeline": {"mix": {"enabled": True}}}
            cli = {"pipeline": {"mix": {"enabled": False}}}
            config = resolve_config(default_path=path, project=project, cli=cli)
            self.assertFalse(config.mix.enabled)
            self.assertFalse(config.master.enabled)

    def test_master_without_mix_is_invalid(self):
        with self.assertRaises(ConfigurationError):
            resolve_config(cli={"pipeline": {"mix": {"enabled": False}, "master": {"enabled": True}}})

    def test_mix_without_master_is_valid(self):
        config = resolve_config(cli={"pipeline": {"mix": {"enabled": True}, "master": {"enabled": False}}})
        self.assertTrue(config.mix.enabled)
        self.assertFalse(config.master.enabled)


if __name__ == "__main__":
    unittest.main()
