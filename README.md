# Covera

Covera is a small, resumable local AI cover pipeline. It keeps every intermediate artifact and calls FFmpeg, [audio-separator](https://github.com/nomadkaraoke/python-audio-separator), and [Applio](https://github.com/IAHispano/Applio) through explicit subprocess commands.

```text
Input → Normalize → Vocal/Instrumental → Lead/Backing → RVC → Mix → Master → Export
```

## Requirements

- Python 3.12+
- FFmpeg and ffprobe
- audio-separator 0.47.0
- Applio 3.6.5 in its own Python environment
- A voice directory containing `model.pth`, `model.index`, and `voice.json`

The application itself has no third-party runtime Python dependency. External ML tools should be installed separately; Covera only discovers and invokes them.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install "audio-separator[cpu]==0.47.0"
```

Create `~/.aicover/config.json`:

```json
{
  "tools": {
    "ffmpeg": "ffmpeg",
    "ffprobe": "ffprobe",
    "audio_separator": "audio-separator",
    "applio_root": "/absolute/path/to/Applio",
    "applio_python": "/absolute/path/to/Applio/.venv/bin/python"
  }
}
```

Example `voices/uruha/voice.json`:

```json
{
  "schema_version": 1,
  "model": "model.pth",
  "index": "model.index",
  "pitch": 0,
  "f0_method": "rmvpe",
  "index_rate": 0.3,
  "volume_envelope": 1.0,
  "protect": 0.33,
  "embedder_model": "contentvec"
}
```

## Usage

```bash
python -m aicover doctor --voice ./voices/uruha
python -m aicover generate --input ./song.wav --voice ./voices/uruha
python -m aicover generate --input ./song.wav --voice ./voices/uruha --no-mix --no-master
python -m aicover run ./workspace/project
python -m aicover run ./workspace/project --mix --master
python -m aicover run ./workspace/project --from voice_conversion
python -m aicover status ./workspace/project
```

`--mix/--no-mix` and `--master/--no-master` override project settings. Mastering without mixing is rejected. `--from STEP` reruns that Step and every enabled Step after it; `--force` reruns the full pipeline.

With both mix and master disabled, the DAW-ready files are:

```text
stems/instrumental.wav
stems/backing.wav
inference/lead_converted.wav
```

Pipeline state lives in `project.json`; command logs live in `logs/pipeline.log`.
