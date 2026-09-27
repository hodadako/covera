import unittest

from aicover.cli import build_parser


class CliTests(unittest.TestCase):
    def test_boolean_optional_action_distinguishes_unspecified(self):
        parser = build_parser()
        base = parser.parse_args(["generate", "--input", "song.wav", "--voice", "voice"])
        disabled = parser.parse_args(["generate", "--input", "song.wav", "--voice", "voice", "--no-mix"])
        self.assertIsNone(base.mix)
        self.assertFalse(disabled.mix)


if __name__ == "__main__":
    unittest.main()
