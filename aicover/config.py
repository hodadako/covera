from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .errors import ConfigurationError


@dataclass(frozen=True)
class ToolConfig:
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    audio_separator: str = "audio-separator"
    audio_separator_model_dir: str = "~/.aicover/models/audio-separator"
    applio_root: str | None = None
    applio_python: str | None = None


@dataclass(frozen=True)
class SeparationConfig:
    stem_model: str = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
    backing_model: str = "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 44100
    channels: int = 2
    import_lufs: float = -18.0
    import_lra: float = 11.0
    import_true_peak: float = -1.5


@dataclass(frozen=True)
class MixConfig:
    enabled: bool = True
    instrumental_gain_db: float = 0.0
    backing_gain_db: float = -3.0
    lead_gain_db: float = 0.0
    lead_eq: str | None = None
    lead_reverb: str | None = None


@dataclass(frozen=True)
class MasterConfig:
    enabled: bool = True
    integrated_lufs: float = -14.0
    loudness_range: float = 11.0
    true_peak: float = -1.0
    limiter: float = 0.95


@dataclass(frozen=True)
class ResolvedConfig:
    workspace_root: str = "./workspace"
    tools: ToolConfig = field(default_factory=ToolConfig)
    separation: SeparationConfig = field(default_factory=SeparationConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    mix: MixConfig = field(default_factory=MixConfig)
    master: MasterConfig = field(default_factory=MasterConfig)

    def validate(self) -> None:
        if self.master.enabled and not self.mix.enabled:
            raise ConfigurationError("master requires mix; use --mix --master or --no-mix --no-master")
        if self.audio.sample_rate <= 0 or self.audio.channels <= 0:
            raise ConfigurationError("audio sample_rate and channels must be positive")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["pipeline"] = {
            "mix": {"enabled": data.pop("mix")["enabled"]},
            "master": {"enabled": data.pop("master")["enabled"]},
        }
        data["mix"] = asdict(self.mix)
        data["master"] = asdict(self.master)
        return data

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ResolvedConfig":
        pipeline = raw.get("pipeline", {})
        mix_raw = dict(raw.get("mix", {}))
        master_raw = dict(raw.get("master", {}))
        if "mix" in pipeline:
            value = pipeline["mix"]
            mix_raw["enabled"] = value.get("enabled", True) if isinstance(value, dict) else bool(value)
        if "master" in pipeline:
            value = pipeline["master"]
            master_raw["enabled"] = value.get("enabled", True) if isinstance(value, dict) else bool(value)
        config = cls(
            workspace_root=str(raw.get("workspace_root", "./workspace")),
            tools=ToolConfig(**raw.get("tools", {})),
            separation=SeparationConfig(**raw.get("separation", {})),
            audio=AudioConfig(**raw.get("audio", {})),
            mix=MixConfig(**mix_raw),
            master=MasterConfig(**master_raw),
        )
        config.validate()
        return config


def _merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def default_config_path() -> Path:
    override = os.environ.get("AICOVER_CONFIG")
    return Path(override).expanduser() if override else Path("~/.aicover/config.json").expanduser()


def read_json_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"cannot read config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError(f"config root must be an object: {path}")
    return value


def resolve_config(
    *,
    default_path: Path | None = None,
    project: Mapping[str, Any] | None = None,
    cli: Mapping[str, Any] | None = None,
) -> ResolvedConfig:
    built_in = ResolvedConfig().to_dict()
    merged = _merge(built_in, read_json_config(default_path or default_config_path()))
    if project:
        merged = _merge(merged, project)
    if cli:
        merged = _merge(merged, cli)
    return ResolvedConfig.from_mapping(merged)
