import logging
import shutil
import tempfile
import unittest
import wave
from pathlib import Path

from aicover.config import resolve_config
from aicover.external import FFmpeg
from aicover.process import ProcessRunner


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is not installed")
class FFmpegIntegrationTests(unittest.TestCase):
    def test_normalize_mix_master_and_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "tone.wav"
            with wave.open(str(source), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(44100)
                output.writeframes(b"\x00\x00" * 44100)
            config = resolve_config()
            ffmpeg = FFmpeg(config, ProcessRunner(logging.getLogger("ffmpeg-test")))
            normalized = root / "normalized.wav"
            mixed = root / "mixed.wav"
            mastered = root / "mastered.wav"
            exported = root / "exported.wav"
            ffmpeg.normalize(source, normalized, step="normalize")
            ffmpeg.mix([normalized, normalized, normalized], mixed, step="mix")
            ffmpeg.master(mixed, mastered, step="master")
            ffmpeg.export(mastered, exported, step="export")
            probe = ffmpeg.probe(exported, step="probe")
            self.assertTrue(probe["streams"])
            self.assertGreater(exported.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
