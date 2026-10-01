# Speech-2-Text

A simple voice-input text editor for organizing your thoughts. It sends your speech to a locally running Qwen3-ASR server, inserts the transcript at the cursor, and lets you edit it freely.
Runs independently of Hermes Agent.

## Requirements

- A Qwen3-ASR server (`qwen-asr-serve`, default `http://localhost:8000/v1`)
  - Start it with `Start-Qwen3-ASR.bat` on the desktop
- Python 3.11+

## Launch

Double-click `run.bat`. On the first run it creates `.venv` and installs the dependencies.

## Usage

| Action | Key |
|---|---|
| Start / stop recording → insert transcript at the cursor | **F9** (or the "● 録音" button) |
| New / Open / Save / Save As | Ctrl+N / Ctrl+O / Ctrl+S / Ctrl+Shift+S |
| Change text size | Ctrl + mouse wheel |

- **Autosave**: a saved file is overwritten every `autosave_seconds`. An untitled document is saved as a draft to `autosave/draft.md` and restored on the next launch.
- Recordings are sent as WAV directly to the ASR server, so the format-conversion proxy for Hermes (:8001) is not needed.

## Settings (`config.json`)

Created with default values on first launch. Restart the app after changing it.

| Key | Default | Description |
|---|---|---|
| `server_url` | `http://localhost:8000/v1` | ASR server URL |
| `model` | `Qwen/Qwen3-ASR-1.7B` | Model name passed to `qwen-asr-serve` |
| `language` | `""` | `""` = auto-detect, or an ISO code such as `"ja"` / `"en"` (names like `"Japanese"` are rejected) |
| `hotkey` | `F9` | Key to start / stop recording |
| `autosave_seconds` | `30` | Autosave interval (seconds) |
| `max_recording_seconds` | `600` | Maximum length of one recording (seconds) |
| `request_timeout` | `120` | Transcription request timeout (seconds) |
| `input_device` | `null` | Microphone. `null` = system default, or a device index / name |
| `font_family` / `font_size` | `Yu Gothic UI` / `14` | Editor font |
