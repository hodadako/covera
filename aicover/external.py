from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Sequence

from .config import ResolvedConfig
from .errors import ArtifactError, ConfigurationError, ExternalCommandError
from .process import CommandResult, ProcessRunner


def temporary_path(final_path: Path) -> Path:
    return final_path.with_name(f"{final_path.stem}.{uuid.uuid4().hex}.tmp{final_path.suffix}")


def require_success(result: CommandResult, label: str) -> None:
    if result.exit_code != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no output"
        raise ExternalCommandError(f"{label} failed ({result.exit_code}): {detail[-2000:]}", exit_code=result.exit_code)


def commit_artifact(temporary: Path, final: Path) -> None:
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise ArtifactError(f"expected output was not created: {temporary}")
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, final)


class FFmpeg:
    def __init__(self, config: ResolvedConfig, runner: ProcessRunner):
        self.config = config
        self.runner = runner

    def probe(self, path: Path, *, step: str) -> dict[str, Any]:
        command = [
            self.config.tools.ffprobe,
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_name,sample_rate,channels,duration:format=duration",
            "-of", "json",
            str(path),
        ]
        result = self.runner.run(command, step=step)
        require_success(result, "ffprobe")
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise ArtifactError(f"ffprobe returned invalid JSON for {path}") from exc
        if not data.get("streams"):
            raise ArtifactError(f"artifact has no audio stream: {path}")
        return data

    def normalize(self, source: Path, destination: Path, *, step: str) -> None:
        audio = self.config.audio
        target = temporary_path(destination)
        filter_value = f"loudnorm=I={audio.import_lufs}:LRA={audio.import_lra}:TP={audio.import_true_peak}"
        result = self.runner.run([
            self.config.tools.ffmpeg, "-nostdin", "-y", "-i", str(source),
            "-vn", "-af", filter_value, "-ar", str(audio.sample_rate), "-ac", str(audio.channels),
            "-c:a", "pcm_s24le", str(target),
        ], step=step)
        require_success(result, "FFmpeg normalize")
        self.probe(target, step=step)
        commit_artifact(target, destination)

    def mix(self, inputs: Sequence[Path], destination: Path, *, step: str) -> None:
        if len(inputs) != 3:
            raise ConfigurationError("mix requires instrumental, backing, and converted lead")
        mix = self.config.mix
        target = temporary_path(destination)
        lead_filters = [f"volume={mix.lead_gain_db}dB"]
        if mix.lead_eq:
            lead_filters.append(mix.lead_eq)
        if mix.lead_reverb:
            lead_filters.append(mix.lead_reverb)
        graph = (
            f"[0:a]volume={mix.instrumental_gain_db}dB[instrumental];"
            f"[1:a]volume={mix.backing_gain_db}dB[backing];"
            f"[2:a]{','.join(lead_filters)}[lead];"
            "[instrumental][backing][lead]amix=inputs=3:duration=longest:dropout_transition=0:normalize=0[mix]"
        )
        command = [self.config.tools.ffmpeg, "-nostdin", "-y"]
        for path in inputs:
            command.extend(["-i", str(path)])
        command.extend([
            "-filter_complex", graph, "-map", "[mix]", "-ar", str(self.config.audio.sample_rate),
            "-ac", str(self.config.audio.channels), "-c:a", "pcm_s24le", str(target),
        ])
        result = self.runner.run(command, step=step)
        require_success(result, "FFmpeg mix")
        self.probe(target, step=step)
        commit_artifact(target, destination)

    def master(self, source: Path, destination: Path, *, step: str) -> None:
        master = self.config.master
        target = temporary_path(destination)
        filters = (
            f"loudnorm=I={master.integrated_lufs}:LRA={master.loudness_range}:TP={master.true_peak},"
            f"alimiter=limit={master.limiter}"
        )
        result = self.runner.run([
            self.config.tools.ffmpeg, "-nostdin", "-y", "-i", str(source), "-vn", "-af", filters,
            "-ar", str(self.config.audio.sample_rate), "-ac", str(self.config.audio.channels),
            "-c:a", "pcm_s24le", str(target),
        ], step=step)
        require_success(result, "FFmpeg master")
        self.probe(target, step=step)
        commit_artifact(target, destination)

    def export(self, source: Path, destination: Path, *, step: str) -> None:
        target = temporary_path(destination)
        result = self.runner.run([
            self.config.tools.ffmpeg, "-nostdin", "-y", "-i", str(source), "-vn",
            "-c:a", "pcm_s24le", str(target),
        ], step=step)
        require_success(result, "FFmpeg export")
        self.probe(target, step=step)
        commit_artifact(target, destination)


class AudioSeparator:
    def __init__(self, config: ResolvedConfig, runner: ProcessRunner):
        self.config = config
        self.runner = runner

    def separate(
        self,
        source: Path,
        model: str,
        outputs: dict[str, Path],
        *,
        step: str,
    ) -> None:
        if not outputs:
            raise ConfigurationError("separator requires at least one output")
        temporaries = {label: temporary_path(path) for label, path in outputs.items()}
        output_dirs = {path.parent.resolve() for path in temporaries.values()}
        if len(output_dirs) != 1:
            raise ConfigurationError("separator outputs must share one directory")
        names = {label: path.stem for label, path in temporaries.items()}
        command = [
            self.config.tools.audio_separator,
            str(source),
            "--model_filename", model,
            "--output_dir", str(next(iter(output_dirs))),
            "--output_format", "WAV",
            "--model_file_dir", str(Path(self.config.tools.audio_separator_model_dir).expanduser()),
            "--custom_output_names", json.dumps(names, separators=(",", ":")),
        ]
        result = self.runner.run(command, step=step)
        require_success(result, "audio-separator")
        for label, final in outputs.items():
            temporary = temporaries[label]
            if not temporary.exists():
                raise ArtifactError(f"audio-separator did not produce {label}: {temporary}")
            commit_artifact(temporary, final)


class ApplioRVC:
    def __init__(self, config: ResolvedConfig, runner: ProcessRunner):
        self.config = config
        self.runner = runner

    def convert(self, source: Path, destination: Path, voice: dict[str, Any], *, step: str) -> None:
        tools = self.config.tools
        if not tools.applio_root or not tools.applio_python:
            raise ConfigurationError("tools.applio_root and tools.applio_python are required")
        root = Path(tools.applio_root).expanduser().resolve()
        core = root / "core.py"
        if not core.is_file():
            raise ConfigurationError(f"Applio core.py not found: {core}")
        target = temporary_path(destination)
        command = [
            str(Path(tools.applio_python).expanduser()), str(core), "infer",
            "--input-path", str(source), "--output-path", str(target),
            "--pth-path", str(voice["model_path"]), "--index-path", str(voice["index_path"]),
            "--pitch", str(voice.get("pitch", 0)),
            "--index-rate", str(voice.get("index_rate", 0.3)),
            "--volume-envelope", str(voice.get("volume_envelope", 1.0)),
            "--protect", str(voice.get("protect", 0.33)),
            "--f0-method", str(voice.get("f0_method", "rmvpe")),
            "--embedder-model", str(voice.get("embedder_model", "contentvec")),
            "--export-format", "WAV",
        ]
        result = self.runner.run(command, cwd=root, step=step)
        require_success(result, "Applio inference")
        commit_artifact(target, destination)
