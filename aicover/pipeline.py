from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import ResolvedConfig
from .errors import ArtifactError, ExternalCommandError
from .process import ProcessRunner
from .project import Project, StepRecord, StepStatus, utc_now


@dataclass
class PipelineContext:
    project: Project
    config: ResolvedConfig
    runner: ProcessRunner
    logger: logging.Logger


class Step:
    name = "step"
    version = 1

    def enabled(self, context: PipelineContext) -> bool:
        return True

    def outputs(self, context: PipelineContext) -> list[Path]:
        return []

    def fingerprint_data(self, context: PipelineContext) -> dict[str, Any]:
        return {}

    def run(self, context: PipelineContext) -> None:
        raise NotImplementedError


class Pipeline:
    def __init__(self, steps: Iterable[Step]):
        self.steps = list(steps)

    def run(self, context: PipelineContext, *, from_step: str | None = None, force: bool = False) -> None:
        names = [step.name for step in self.steps]
        if from_step is not None and from_step not in names:
            raise ValueError(f"unknown step {from_step}; choose one of: {', '.join(names)}")
        force_index = names.index(from_step) if from_step else (0 if force else None)

        for step in self.steps:
            if not step.enabled(context):
                record = context.project.steps.setdefault(step.name, StepRecord())
                record.status = StepStatus.SKIPPED
                record.finished_at = utc_now()
                record.error = None
        context.project.save()

        upstream = "root"
        for index, step in enumerate(self.steps):
            fingerprint = self._fingerprint(step, context, upstream)
            upstream = fingerprint
            record = context.project.steps.setdefault(step.name, StepRecord())
            if not step.enabled(context):
                record.fingerprint = fingerprint
                context.logger.info("step=%s status=SKIPPED", step.name)
                context.project.save()
                continue
            expected = step.outputs(context)
            forced = force_index is not None and index >= force_index
            if not forced and self._can_skip(record, fingerprint, expected):
                context.logger.info("step=%s status=SKIPPED reason=resume", step.name)
                continue

            started = time.monotonic()
            record.status = StepStatus.RUNNING
            record.fingerprint = fingerprint
            record.attempts += 1
            record.outputs = [context.project.relative(path) for path in expected]
            record.started_at = utc_now()
            record.finished_at = None
            record.error = None
            record.exit_code = None
            context.project.save()
            try:
                step.run(context)
                missing = [str(path) for path in expected if not path.is_file() or path.stat().st_size == 0]
                if missing:
                    raise ArtifactError(f"step outputs are missing or empty: {', '.join(missing)}")
            except Exception as exc:
                record.status = StepStatus.FAILED
                record.finished_at = utc_now()
                record.duration_seconds = time.monotonic() - started
                record.exit_code = exc.exit_code if isinstance(exc, ExternalCommandError) else None
                record.error = str(exc)
                context.project.save()
                context.logger.exception("step=%s status=FAILED error=%s", step.name, exc)
                raise
            record.status = StepStatus.COMPLETED
            record.finished_at = utc_now()
            record.duration_seconds = time.monotonic() - started
            record.exit_code = 0
            context.project.save()
            context.logger.info("step=%s status=COMPLETED duration=%.3f", step.name, record.duration_seconds)

    @staticmethod
    def _fingerprint(step: Step, context: PipelineContext, upstream: str) -> str:
        value = {
            "step": step.name,
            "version": step.version,
            "upstream": upstream,
            "config": step.fingerprint_data(context),
            "source_sha256": context.project.source.get("sha256"),
            "model_sha256": context.project.voice.get("model_sha256"),
            "index_sha256": context.project.voice.get("index_sha256"),
        }
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _can_skip(record: StepRecord, fingerprint: str, outputs: list[Path]) -> bool:
        return (
            record.status is StepStatus.COMPLETED
            and record.fingerprint == fingerprint
            and all(path.is_file() and path.stat().st_size > 0 for path in outputs)
        )
