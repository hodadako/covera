from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Mapping

from .config import ResolvedConfig, resolve_config
from .errors import ConfigurationError
from .pipeline import Pipeline, PipelineContext
from .process import ProcessRunner
from .project import Project, StepRecord
from .steps import default_steps


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_voice(directory: Path) -> dict[str, Any]:
    directory = directory.expanduser().resolve()
    config_path = directory / "voice.json"
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"cannot read voice config {config_path}: {exc}") from exc
    if raw.get("schema_version", 1) != 1:
        raise ConfigurationError(f"unsupported voice schema: {raw.get('schema_version')}")
    model = directory / raw.get("model", "model.pth")
    index = directory / raw.get("index", "model.index")
    if not model.is_file():
        raise ConfigurationError(f"voice model not found: {model}")
    if not index.is_file():
        raise ConfigurationError(f"voice index not found: {index}")
    result = dict(raw)
    result.update({
        "directory": str(directory),
        "model_path": str(model.resolve()),
        "index_path": str(index.resolve()),
        "model_sha256": sha256_file(model),
        "index_sha256": sha256_file(index),
    })
    return result


class CoverEngine:
    def __init__(self, *, runner: ProcessRunner | None = None, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger("aicover")
        self.runner = runner or ProcessRunner(self.logger)

    def generate(
        self,
        input_path: Path,
        voice_dir: Path,
        *,
        project_dir: Path | None = None,
        default_config: Path | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> Project:
        source = input_path.expanduser().resolve()
        if not source.is_file():
            raise ConfigurationError(f"input audio not found: {source}")
        config = resolve_config(default_path=default_config, cli=overrides)
        voice = load_voice(voice_dir)
        if project_dir is None:
            safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", source.stem).strip("-_") or "cover"
            project_dir = Path(config.workspace_root) / f"{safe_stem}-{uuid.uuid4().hex[:8]}"
        root = project_dir.expanduser().resolve()
        if (root / "project.json").exists():
            raise ConfigurationError(f"project already exists: {root}")
        project = Project(
            root=root,
            project_id=root.name,
            config=config.to_dict(),
            source={"original_path": str(source), "sha256": sha256_file(source)},
            voice=voice,
            steps={step.name: StepRecord() for step in default_steps()},
        )
        project.ensure_directories()
        self._attach_file_logger(project)
        project.save()
        self._run(project, config)
        return project

    def run(
        self,
        project_dir: Path,
        *,
        default_config: Path | None = None,
        overrides: Mapping[str, Any] | None = None,
        from_step: str | None = None,
        force: bool = False,
    ) -> Project:
        project = Project.load(project_dir)
        config = resolve_config(default_path=default_config, project=project.config, cli=overrides)
        project.config = config.to_dict()
        self._attach_file_logger(project)
        project.save()
        self._run(project, config, from_step=from_step, force=force)
        return project

    def status(self, project_dir: Path) -> Project:
        return Project.load(project_dir)

    def _run(self, project: Project, config: ResolvedConfig, *, from_step: str | None = None, force: bool = False) -> None:
        context = PipelineContext(project, config, self.runner, self.logger)
        Pipeline(default_steps()).run(context, from_step=from_step, force=force)

    def _attach_file_logger(self, project: Project) -> None:
        project.ensure_directories()
        path = (project.root / "logs/pipeline.log").resolve()
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename).resolve() == path:
                return
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s level=%(levelname)s %(message)s"))
        self.logger.addHandler(handler)
