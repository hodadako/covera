from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from .external import ApplioRVC, AudioSeparator, FFmpeg
from .pipeline import PipelineContext, Step


class ImportAudioStep(Step):
    name = "import"

    def outputs(self, context: PipelineContext) -> list[Path]:
        suffix = Path(context.project.source["original_path"]).suffix.lower() or ".audio"
        return [context.project.artifact(f"input/original{suffix}"), context.project.artifact("input/source.wav")]

    def fingerprint_data(self, context: PipelineContext):
        return {"audio": context.config.audio.__dict__}

    def run(self, context: PipelineContext) -> None:
        original, normalized = self.outputs(context)
        source = Path(context.project.source["original_path"])
        copied = original.with_name(f"{original.stem}.{uuid.uuid4().hex}.tmp{original.suffix}")
        shutil.copyfile(source, copied)
        FFmpeg(context.config, context.runner).normalize(copied, normalized, step=self.name)
        os.replace(copied, original)


class SeparateStemStep(Step):
    name = "separate_stem"

    def outputs(self, context: PipelineContext) -> list[Path]:
        return [context.project.artifact("stems/instrumental.wav"), context.project.artifact("stems/vocals.wav")]

    def fingerprint_data(self, context: PipelineContext):
        return {"model": context.config.separation.stem_model}

    def run(self, context: PipelineContext) -> None:
        instrumental, vocals = self.outputs(context)
        AudioSeparator(context.config, context.runner).separate(
            context.project.artifact("input/source.wav"),
            context.config.separation.stem_model,
            {"Instrumental": instrumental, "Vocals": vocals},
            step=self.name,
        )


class SeparateBackingVocalStep(Step):
    name = "separate_backing_vocal"

    def outputs(self, context: PipelineContext) -> list[Path]:
        return [context.project.artifact("stems/lead.wav"), context.project.artifact("stems/backing.wav")]

    def fingerprint_data(self, context: PipelineContext):
        return {"model": context.config.separation.backing_model}

    def run(self, context: PipelineContext) -> None:
        lead, backing = self.outputs(context)
        AudioSeparator(context.config, context.runner).separate(
            context.project.artifact("stems/vocals.wav"),
            context.config.separation.backing_model,
            {"Lead Vocals": lead, "Backing Vocals": backing},
            step=self.name,
        )


class VoiceConversionStep(Step):
    name = "voice_conversion"

    def outputs(self, context: PipelineContext) -> list[Path]:
        return [context.project.artifact("inference/lead_converted.wav")]

    def fingerprint_data(self, context: PipelineContext):
        return {key: value for key, value in context.project.voice.items() if key not in {"model_sha256", "index_sha256"}}

    def run(self, context: PipelineContext) -> None:
        ApplioRVC(context.config, context.runner).convert(
            context.project.artifact("stems/lead.wav"), self.outputs(context)[0], context.project.voice, step=self.name
        )


class MixStep(Step):
    name = "mix"

    def enabled(self, context: PipelineContext) -> bool:
        return context.config.mix.enabled

    def outputs(self, context: PipelineContext) -> list[Path]:
        return [context.project.artifact("mix/premaster.wav")]

    def fingerprint_data(self, context: PipelineContext):
        return context.config.mix.__dict__

    def run(self, context: PipelineContext) -> None:
        FFmpeg(context.config, context.runner).mix([
            context.project.artifact("stems/instrumental.wav"),
            context.project.artifact("stems/backing.wav"),
            context.project.artifact("inference/lead_converted.wav"),
        ], self.outputs(context)[0], step=self.name)


class MasterStep(Step):
    name = "master"

    def enabled(self, context: PipelineContext) -> bool:
        return context.config.master.enabled

    def outputs(self, context: PipelineContext) -> list[Path]:
        return [context.project.artifact("mix/master.wav")]

    def fingerprint_data(self, context: PipelineContext):
        return context.config.master.__dict__

    def run(self, context: PipelineContext) -> None:
        FFmpeg(context.config, context.runner).master(
            context.project.artifact("mix/premaster.wav"), self.outputs(context)[0], step=self.name
        )


class ExportStep(Step):
    name = "export"

    def enabled(self, context: PipelineContext) -> bool:
        return context.config.mix.enabled

    def outputs(self, context: PipelineContext) -> list[Path]:
        name = "final.wav" if context.config.master.enabled else "premaster.wav"
        return [context.project.artifact(f"output/{name}")]

    def fingerprint_data(self, context: PipelineContext):
        return {"master": context.config.master.enabled, "codec": "pcm_s24le"}

    def run(self, context: PipelineContext) -> None:
        source = context.project.artifact("mix/master.wav" if context.config.master.enabled else "mix/premaster.wav")
        FFmpeg(context.config, context.runner).export(source, self.outputs(context)[0], step=self.name)


def default_steps() -> list[Step]:
    return [
        ImportAudioStep(), SeparateStemStep(), SeparateBackingVocalStep(), VoiceConversionStep(),
        MixStep(), MasterStep(), ExportStep(),
    ]
