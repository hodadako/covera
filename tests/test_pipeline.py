import logging
import tempfile
import unittest
from pathlib import Path

from aicover.config import resolve_config
from aicover.pipeline import Pipeline, PipelineContext, Step
from aicover.project import Project, StepRecord, StepStatus


class RecordingStep(Step):
    def __init__(self, name, events, *, active=True, fail_once=False):
        self.name = name
        self.events = events
        self.active = active
        self.fail_once = fail_once

    def enabled(self, context):
        return self.active

    def outputs(self, context):
        return [context.project.artifact(f"output/{self.name}.txt")]

    def run(self, context):
        self.events.append(self.name)
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("planned failure")
        output = self.outputs(context)[0]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.name)


class PipelineTests(unittest.TestCase):
    def make_context(self, root, config=None):
        project = Project(Path(root), "test", (config or resolve_config()).to_dict(), {"sha256": "source"}, {"model_sha256": "m", "index_sha256": "i"})
        project.ensure_directories()
        return PipelineContext(project, config or resolve_config(), object(), logging.getLogger("test"))

    def test_order_disabled_and_persistence(self):
        with tempfile.TemporaryDirectory() as directory:
            events = []
            steps = [RecordingStep("one", events), RecordingStep("two", events, active=False), RecordingStep("three", events)]
            context = self.make_context(directory)
            Pipeline(steps).run(context)
            self.assertEqual(events, ["one", "three"])
            self.assertEqual(context.project.steps["two"].status, StepStatus.SKIPPED)
            loaded = Project.load(Path(directory))
            self.assertEqual(loaded.steps["one"].outputs, ["output/one.txt"])

    def test_completed_steps_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            events = []
            pipeline = Pipeline([RecordingStep("one", events), RecordingStep("two", events)])
            context = self.make_context(directory)
            pipeline.run(context)
            pipeline.run(context)
            self.assertEqual(events, ["one", "two"])

    def test_resume_after_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            events = []
            steps = [RecordingStep("one", events), RecordingStep("two", events, fail_once=True), RecordingStep("three", events)]
            context = self.make_context(directory)
            with self.assertRaises(RuntimeError):
                Pipeline(steps).run(context)
            self.assertEqual(context.project.steps["one"].status, StepStatus.COMPLETED)
            self.assertEqual(context.project.steps["two"].status, StepStatus.FAILED)
            Pipeline(steps).run(context)
            self.assertEqual(events, ["one", "two", "two", "three"])

    def test_from_and_force(self):
        with tempfile.TemporaryDirectory() as directory:
            events = []
            steps = [RecordingStep("one", events), RecordingStep("two", events), RecordingStep("three", events)]
            context = self.make_context(directory)
            pipeline = Pipeline(steps)
            pipeline.run(context)
            pipeline.run(context, from_step="two")
            pipeline.run(context, force=True)
            self.assertEqual(events, ["one", "two", "three", "two", "three", "one", "two", "three"])


if __name__ == "__main__":
    unittest.main()
