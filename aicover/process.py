from __future__ import annotations

import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class CommandResult:
    command: tuple[str, ...]
    exit_code: int
    duration_seconds: float
    stdout: str
    stderr: str


class ProcessRunner:
    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger("aicover.process")
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        step: str = "external",
    ) -> CommandResult:
        argv = [str(item) for item in command]
        self.logger.info("step=%s status=RUNNING command=%r", step, argv)
        started = time.monotonic()
        process = subprocess.Popen(
            argv,
            cwd=str(cwd) if cwd else None,
            env=dict(env) if env else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            shell=False,
        )
        with self._lock:
            self._process = process
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        def consume(stream, target: list[str], level: int) -> None:
            assert stream is not None
            for line in iter(stream.readline, ""):
                target.append(line)
                self.logger.log(level, "step=%s %s", step, line.rstrip())
            stream.close()

        threads = [
            threading.Thread(target=consume, args=(process.stdout, stdout_lines, logging.INFO), daemon=True),
            threading.Thread(target=consume, args=(process.stderr, stderr_lines, logging.WARNING), daemon=True),
        ]
        for thread in threads:
            thread.start()
        exit_code = process.wait()
        for thread in threads:
            thread.join()
        with self._lock:
            self._process = None
        duration = time.monotonic() - started
        self.logger.info("step=%s status=%s duration=%.3f exit_code=%d", step, "COMPLETED" if exit_code == 0 else "FAILED", duration, exit_code)
        return CommandResult(tuple(argv), exit_code, duration, "".join(stdout_lines), "".join(stderr_lines))

    def cancel(self) -> bool:
        with self._lock:
            process = self._process
        if process is None or process.poll() is not None:
            return False
        process.terminate()
        return True
