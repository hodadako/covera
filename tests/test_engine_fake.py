import json
import tempfile
import unittest
from pathlib import Path

from aicover.engine import CoverEngine
from aicover.process import CommandResult
from aicover.project import StepStatus


class FakeRunner:
    def __init__(self):
        self.commands = []

    def run(self, command, *, cwd=None, env=None, step="external"):
        command = [str(value) for value in command]
        self.commands.append((step, command))
        executable = Path(command[0]).name
        if "ffprobe" in executable:
            return CommandResult(tuple(command), 0, 0.01, '{"streams":[{"codec_name":"pcm_s24le"}],"format":{"duration":"1"}}', "")
        if "audio-separator" in executable:
            output_dir = Path(command[command.index("--output_dir") + 1])
            names = json.loads(command[command.index("--custom_output_names") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            for name in names.values():
                (output_dir / f"{name}.wav").write_bytes(b"fake wav")
        elif "core.py" in " ".join(command):
            Path(command[command.index("--output-path") + 1]).write_bytes(b"fake wav")
        else:
            Path(command[-1]).write_bytes(b"fake wav")
        return CommandResult(tuple(command), 0, 0.01, "", "")


class EngineFakeTests(unittest.TestCase):
    def create_inputs(self, root):
        song = root / "song with spaces.wav"
        song.write_bytes(b"source")
        voice = root / "voice with spaces"
        voice.mkdir()
        (voice / "model.pth").write_bytes(b"model")
        (voice / "model.index").write_bytes(b"index")
        (voice / "voice.json").write_text(json.dumps({"schema_version": 1}))
        applio = root / "Applio"
        applio.mkdir()
        (applio / "core.py").write_text("# fake")
        config = root / "config.json"
        config.write_text(json.dumps({"tools": {
            "ffmpeg": "fake-ffmpeg", "ffprobe": "fake-ffprobe",
            "audio_separator": "fake-audio-separator",
            "applio_root": str(applio), "applio_python": "fake-python"
        }}))
        return song, voice, config

    def test_full_pipeline_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            song, voice, config = self.create_inputs(root)
            runner = FakeRunner()
            engine = CoverEngine(runner=runner)
            project = engine.generate(song, voice, project_dir=root / "project", default_config=config)
            self.assertTrue((project.root / "output/final.wav").is_file())
            self.assertTrue(all(record.status is StepStatus.COMPLETED for record in project.steps.values()))
            count = len(runner.commands)
            engine.run(project.root, default_config=config)
            self.assertEqual(len(runner.commands), count)

    def test_deferred_mix_and_master(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            song, voice, config = self.create_inputs(root)
            runner = FakeRunner()
            engine = CoverEngine(runner=runner)
            project = engine.generate(
                song, voice, project_dir=root / "manual", default_config=config,
                overrides={"pipeline": {"mix": {"enabled": False}, "master": {"enabled": False}}},
            )
            self.assertEqual(project.steps["mix"].status, StepStatus.SKIPPED)
            self.assertTrue((project.root / "stems/instrumental.wav").is_file())
            self.assertTrue((project.root / "stems/backing.wav").is_file())
            self.assertTrue((project.root / "inference/lead_converted.wav").is_file())
            runner.commands.clear()
            resumed = engine.run(
                project.root, default_config=config,
                overrides={"pipeline": {"mix": {"enabled": True}, "master": {"enabled": True}}},
            )
            self.assertTrue((project.root / "output/final.wav").is_file())
            self.assertEqual(resumed.steps["mix"].status, StepStatus.COMPLETED)
            self.assertTrue(all(step in {"mix", "master", "export"} for step, _ in runner.commands))


if __name__ == "__main__":
    unittest.main()
