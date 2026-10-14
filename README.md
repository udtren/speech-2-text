# Speech-2-Text

A simple voice-input text editor for organizing your thoughts. It sends your speech to a locally running Qwen3-ASR server, inserts the transcript at the cursor, and lets you edit it freely.
Runs independently of Hermes Agent.

## Requirements

- A Qwen3-ASR server (`qwen-asr-serve`, default `http://localhost:8000/v1`)
  - Start it with `Start-Qwen3-ASR.bat` on the desktop
  - vLLM rejects uploads over 25 MB by default; the bat raises this with `VLLM_MAX_AUDIO_CLIP_FILESIZE_MB=200`
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
- Recordings are sent as 16 kHz WAV directly to the ASR server, so the format-conversion proxy for Hermes (:8001) is not needed.
- **Long recordings**: while recording, audio is cut every `chunk_seconds` (at the quietest point in the 5 seconds before the boundary, so words are not split) and transcribed in the background in order. The pieces are joined at the cursor position where recording started. If a piece fails, its audio is kept in `autosave/failed/`.

## Settings (`config.json`)

Created with default values on first launch. Restart the app after changing it.

| Key | Default | Description |
|---|---|---|
| `server_url` | `http://localhost:8000/v1` | ASR server URL |
| `model` | `Qwen/Qwen3-ASR-1.7B` | Model name passed to `qwen-asr-serve` |
| `language` | `""` | `""` = auto-detect, or an ISO code such as `"ja"` / `"en"` (names like `"Japanese"` are rejected) |
| `hotkey` | `F9` | Key to start / stop recording |
| `autosave_seconds` | `30` | Autosave interval (seconds) |
| `max_recording_seconds` | `3600` | Maximum length of one recording (seconds) |
| `chunk_seconds` | `120` | Length of each piece sent for transcription during recording (seconds) |
| `request_timeout` | `300` | Transcription request timeout (seconds) |
| `input_device` | `null` | Microphone. `null` = system default, or a device index / name |
| `font_family` / `font_size` | `Yu Gothic UI` / `14` | Editor font |
