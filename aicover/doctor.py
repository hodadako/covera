from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import ResolvedConfig
from .engine import load_voice


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def _command_path(command: str) -> str | None:
    expanded = str(Path(command).expanduser())
    if Path(expanded).is_file():
        return expanded
    return shutil.which(command)


def _run(command: list[str], *, cwd: Path | None = None) -> tuple[bool, str]:
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=60, shell=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    output = (result.stdout or result.stderr).strip().splitlines()
    return result.returncode == 0, output[0] if output else f"exit {result.returncode}"


def doctor(config: ResolvedConfig, voice_dir: Path | None = None) -> list[Check]:
    checks = [Check("Python", sys.version_info >= (3, 10), f"{platform.python_version()} ({platform.platform()})")]
    ffmpeg = _command_path(config.tools.ffmpeg)
    ffprobe = _command_path(config.tools.ffprobe)
    checks.append(Check("FFmpeg", ffmpeg is not None, ffmpeg or "not found"))
    checks.append(Check("ffprobe", ffprobe is not None, ffprobe or "not found"))
    if ffmpeg:
        try:
            result = subprocess.run(
                [ffmpeg, "-hide_banner", "-filters"], capture_output=True, text=True,
                timeout=60, shell=False,
            )
            required = ("loudnorm", "amix", "alimiter", "equalizer", "aecho")
            missing = [name for name in required if name not in result.stdout]
            checks.append(Check("FFmpeg filters", result.returncode == 0 and not missing, "missing: " + ", ".join(missing) if missing else "available"))
        except (OSError, subprocess.TimeoutExpired) as exc:
            checks.append(Check("FFmpeg filters", False, str(exc)))

    separator = _command_path(config.tools.audio_separator)
    if separator:
        ok, detail = _run([separator, "--version"])
        checks.append(Check("audio-separator", ok, detail))
        try:
            models = subprocess.run(
                [separator, "--list_models", "--list_format=json"], capture_output=True, text=True,
                timeout=60, shell=False,
            )
            parsed = json.loads(models.stdout) if models.returncode == 0 else {}
            text = json.dumps(parsed)
            wanted = (config.separation.stem_model, config.separation.backing_model)
            checks.append(Check("separation models", all(model in text for model in wanted), ", ".join(wanted)))
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            checks.append(Check("separation models", False, str(exc)))
    else:
        checks.append(Check("audio-separator", False, "not found"))

    root = Path(config.tools.applio_root).expanduser() if config.tools.applio_root else None
    python = _command_path(config.tools.applio_python) if config.tools.applio_python else None
    core = root / "core.py" if root else None
    applio_ok = bool(root and python and core and core.is_file())
    checks.append(Check("Applio", applio_ok, str(core) if core else "tools.applio_root/python not configured"))
    if applio_ok and core and root and python:
        ok, detail = _run([python, str(core), "infer", "--help"], cwd=root)
        checks.append(Check("Applio inference", ok, detail))
        ok, detail = _run([
            python, "-c",
            "import torch; print('cuda='+str(torch.cuda.is_available())+' mps='+str(getattr(torch.backends,'mps',None) is not None and torch.backends.mps.is_available()))",
        ], cwd=root)
        checks.append(Check("Acceleration", ok, detail))

    if voice_dir:
        try:
            voice = load_voice(voice_dir)
            checks.append(Check("Voice", True, voice["directory"]))
        except Exception as exc:
            checks.append(Check("Voice", False, str(exc)))
    workspace = Path(config.workspace_root).expanduser()
    writable = workspace.exists() and os_access_writable(workspace) if workspace.exists() else os_access_writable(workspace.parent)
    checks.append(Check("Workspace", writable, str(workspace)))
    return checks


def os_access_writable(path: Path) -> bool:
    import os
    return path.exists() and os.access(path, os.W_OK)
