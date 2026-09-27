from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from .config import resolve_config
from .doctor import doctor
from .engine import CoverEngine
from .errors import AICoverError


STEP_NAMES = ("import", "separate_stem", "separate_backing_vocal", "voice_conversion", "mix", "master", "export")


def _add_switches(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mix", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--master", action=argparse.BooleanOptionalAction, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aicover", description="Resumable local AI cover pipeline")
    parser.add_argument("--config", type=Path, help="default config JSON")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser("doctor", help="check local dependencies")
    doctor_parser.add_argument("--voice", type=Path)

    generate = subparsers.add_parser("generate", help="create and run a project")
    generate.add_argument("--input", type=Path, required=True)
    generate.add_argument("--voice", type=Path, required=True)
    generate.add_argument("--project-dir", type=Path)
    _add_switches(generate)

    run = subparsers.add_parser("run", help="resume an existing project")
    run.add_argument("project", type=Path)
    run.add_argument("--from", dest="from_step", choices=STEP_NAMES)
    run.add_argument("--force", action="store_true")
    _add_switches(run)

    status = subparsers.add_parser("status", help="show project status")
    status.add_argument("project", type=Path)

    voices = subparsers.add_parser("voices", help="list voice directories")
    voices.add_argument("root", nargs="?", type=Path, default=Path("./voices"))
    return parser


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    pipeline: dict[str, Any] = {}
    if getattr(args, "mix", None) is not None:
        pipeline["mix"] = {"enabled": args.mix}
    if getattr(args, "master", None) is not None:
        pipeline["master"] = {"enabled": args.master}
    return {"pipeline": pipeline} if pipeline else {}


def _configure_logging() -> logging.Logger:
    logger = logging.getLogger("aicover")
    logger.setLevel(logging.INFO)
    if not any(type(handler) is logging.StreamHandler for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
    return logger


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logger = _configure_logging()
    engine = CoverEngine(logger=logger)
    try:
        if args.command == "doctor":
            config = resolve_config(default_path=args.config)
            checks = doctor(config, args.voice)
            for check in checks:
                print(f"{'OK' if check.ok else 'FAIL':4} {check.name}: {check.detail}")
            return 0 if all(check.ok for check in checks) else 1
        if args.command == "generate":
            project = engine.generate(
                args.input, args.voice, project_dir=args.project_dir,
                default_config=args.config, overrides=_overrides(args),
            )
            print(project.root)
            return 0
        if args.command == "run":
            project = engine.run(
                args.project, default_config=args.config, overrides=_overrides(args),
                from_step=args.from_step, force=args.force,
            )
            print(project.root)
            return 0
        if args.command == "status":
            project = engine.status(args.project)
            print(f"project: {project.project_id}")
            for name, record in project.steps.items():
                suffix = f" ({record.error})" if record.error else ""
                print(f"{name:24} {record.status.value}{suffix}")
            return 0
        if args.command == "voices":
            root = args.root.expanduser()
            if root.is_dir():
                for path in sorted(root.iterdir()):
                    if path.is_dir() and (path / "voice.json").is_file():
                        print(path)
            return 0
    except (AICoverError, ValueError) as exc:
        logger.error("%s", exc)
        return 2
    return 2
