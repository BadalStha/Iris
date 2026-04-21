# Iris

Iris is a local voice assistant built in Python. It uses Whisper for speech-to-text, Ollama for chat and memory extraction, and Kokoro for text-to-speech.

## What this repo includes

This repository should contain only source code and documentation. Private runtime data and large local assets are ignored so they do not get pushed to GitHub.

Ignored local files and folders include:

- `.iris_memory.json`
- `.iris_conversations.jsonl`
- `.env`
- `venv/` or `.venv/`
- `kokoro_models/`
- large audio or model files such as `*.mp3`, `*.onnx`, `*.npz`

## Requirements

- Python 3.10 or newer
- Ollama running locally
- Linux desktop environment with microphone and audio output
- Recommended system packages:
  - `ffmpeg`
  - `portaudio`
  - `mpv`
  - `brightnessctl`
  - `pactl` or PulseAudio / PipeWire utilities
  - `qdbus6` or `qdbus-qt6` for KDE Wayland window snapping
  - `tkinter` if you want the GUI window

## Python packages

Install the Python dependencies with pip:

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install ollama sounddevice numpy faster-whisper psutil kokoro-onnx duckduckgo_search
```

If your Linux distribution splits Tkinter into a separate package, install it from the system package manager, for example `python3-tk`.

## Ollama models

Iris expects local Ollama models for chat and memory:

- `IRIS_MODEL_FAST` defaults to `phi3.5`
- `IRIS_MODEL_SMART` defaults to `llama3.1:8b`
- `IRIS_MEMORY_MODEL` defaults to `phi3.5`

Pull them with Ollama if you do not already have them:

```bash
ollama pull phi3.5
ollama pull llama3.1:8b
```

## Voice and TTS assets

Kokoro is the documented text-to-speech path for this repo. Place these files locally:

- `kokoro_models/kokoro-v0_19.onnx`
- `kokoro_models/voices.json`

If you want to use a different local Kokoro voice, set:

- `IRIS_KOKORO_VOICE`

Other useful environment variables:

- `IRIS_GUI`
- `IRIS_MODEL_FAST`
- `IRIS_MODEL_SMART`
- `IRIS_MEMORY_MODEL`
- `IRIS_WHISPER_MODEL`

## Run

```bash
source venv/bin/activate
python iris.py
```

## Privacy notes

- Personal memory is stored in your home directory, not inside the repo.
- Do not commit `.env`, conversation logs, or local model files.
- If you accidentally staged private files, remove them before pushing.

## First-time setup checklist

1. Install the system packages you need.
2. Create and activate the Python virtual environment.
3. Install the Python dependencies.
4. Pull the Ollama models.
5. Download the local Kokoro assets if you want speech output.
6. Run `python iris.py`.