from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from .errors import ConfigurationError


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StepStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass
class StepRecord:
    status: StepStatus = StepStatus.PENDING
    fingerprint: str | None = None
    attempts: int = 0
    outputs: list[str] = field(default_factory=list)
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float | None = None
    exit_code: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        return value

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "StepRecord":
        value = dict(raw)
        value["status"] = StepStatus(value.get("status", "PENDING"))
        return cls(**value)


@dataclass
class Project:
    root: Path
    project_id: str
    config: dict[str, Any]
    source: dict[str, Any]
    voice: dict[str, Any]
    steps: dict[str, StepRecord] = field(default_factory=dict)
    schema_version: int = 1
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    @property
    def json_path(self) -> Path:
        return self.root / "project.json"

    def artifact(self, relative: str) -> Path:
        return self.root / Path(relative)

    def ensure_directories(self) -> None:
        for name in ("input", "stems", "inference", "mix", "output", "logs"):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def save(self) -> None:
        self.ensure_directories()
        self.updated_at = utc_now()
        data = {
            "schema_version": self.schema_version,
            "project_id": self.project_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source": self.source,
            "voice": self.voice,
            "pipeline": self.config.get("pipeline", {}),
            "config": self.config,
            "steps": {key: value.to_dict() for key, value in self.steps.items()},
        }
        temporary = self.json_path.with_name(f"project.{uuid.uuid4().hex}.tmp.json")
        temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, self.json_path)

    @classmethod
    def load(cls, root: Path) -> "Project":
        root = root.expanduser().resolve()
        path = root / "project.json"
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"cannot load project {path}: {exc}") from exc
        if raw.get("schema_version") != 1:
            raise ConfigurationError(f"unsupported project schema: {raw.get('schema_version')}")
        return cls(
            root=root,
            project_id=raw["project_id"],
            config=raw.get("config", {"pipeline": raw.get("pipeline", {})}),
            source=raw.get("source", {}),
            voice=raw.get("voice", {}),
            steps={key: StepRecord.from_dict(value) for key, value in raw.get("steps", {}).items()},
            schema_version=raw["schema_version"],
            created_at=raw.get("created_at", utc_now()),
            updated_at=raw.get("updated_at", utc_now()),
        )
