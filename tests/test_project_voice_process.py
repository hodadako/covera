import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path

from aicover.engine import load_voice
from aicover.errors import ConfigurationError
from aicover.process import ProcessRunner


class VoiceAndProcessTests(unittest.TestCase):
    def test_voice_load_and_missing_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "voice.json").write_text(json.dumps({"schema_version": 1, "model": "model.pth", "index": "model.index"}))
            with self.assertRaises(ConfigurationError):
                load_voice(root)
            (root / "model.pth").write_bytes(b"model")
            (root / "model.index").write_bytes(b"index")
            voice = load_voice(root)
            self.assertEqual(len(voice["model_sha256"]), 64)

    def test_process_runner_success_and_failure(self):
        runner = ProcessRunner(logging.getLogger("process-test"))
        success = runner.run([sys.executable, "-c", "print('ok')"])
        failure = runner.run([sys.executable, "-c", "import sys; print('bad', file=sys.stderr); sys.exit(7)"])
        self.assertEqual(success.exit_code, 0)
        self.assertIn("ok", success.stdout)
        self.assertEqual(failure.exit_code, 7)
        self.assertIn("bad", failure.stderr)


if __name__ == "__main__":
    unittest.main()
