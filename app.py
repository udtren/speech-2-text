"""Speech-2-Text: 思考整理向けのシンプルな音声入力エディター.

ローカルで稼働中の Qwen3-ASR (OpenAI 互換 /v1/audio/transcriptions) に録音を送り、
文字起こし結果をカーソル位置に挿入する。テキストは自由に編集・保存できる。
"""
from __future__ import annotations

import io
import json
import sys
import time
import wave
from pathlib import Path

import httpx
import numpy as np
import sounddevice as sd
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QAction, QCloseEvent, QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (QApplication, QFileDialog, QLabel, QMainWindow, QMessageBox,
                               QPlainTextEdit, QPushButton, QToolBar)

APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "config.json"
DRAFT_PATH = APP_DIR / "autosave" / "draft.md"

DEFAULT_CONFIG = {
    "server_url": "http://localhost:8000/v1",
    "model": "Qwen/Qwen3-ASR-1.7B",
    "language": "",            # "" = 自動判定, "ja" / "en" など ISO コード
    "hotkey": "F9",
    "autosave_seconds": 30,
    "max_recording_seconds": 600,
    "request_timeout": 120,
    "input_device": None,      # None = 既定のマイク。番号かデバイス名
    "font_family": "Yu Gothic UI",
    "font_size": 14,
}


def load_config() -> dict:
    config = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        config.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    else:
        CONFIG_PATH.write_text(json.dumps(DEFAULT_CONFIG, ensure_ascii=False, indent=2), encoding="utf-8")
    return config


def strip_asr_prefix(text: str) -> str:
    # Qwen3-ASR の生出力 "language Japanese<asr_text>..." から本文だけを取り出す
    return text.split("<asr_text>", 1)[-1].strip()


# ---- Recording --------------------------------------------------------------

class Recorder:
    """マイク入力をメモリに貯め、停止時に WAV バイト列を返す."""

    def __init__(self, device):
        self.device = device
        self.stream: sd.InputStream | None = None
        self.chunks: list[np.ndarray] = []
        self.samplerate = 16000

    def start(self):
        info = sd.query_devices(self.device, kind="input")
        self.samplerate = int(info["default_samplerate"])
        self.chunks = []
        self.stream = sd.InputStream(device=self.device, channels=1, samplerate=self.samplerate,
                                     dtype="int16", callback=self._callback)
        self.stream.start()

    def _callback(self, indata, frames, time_info, status):
        self.chunks.append(indata.copy())

    def stop(self) -> bytes | None:
        if self.stream is None:
            return None
        self.stream.stop()
        self.stream.close()
        self.stream = None
        if not self.chunks:
            return None
        pcm = np.concatenate(self.chunks).reshape(-1)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.samplerate)
            w.writeframes(pcm.tobytes())
        return buf.getvalue()


# ---- Transcription (background thread) --------------------------------------

class TranscribeSignals(QObject):
    done = Signal(str)
    failed = Signal(str)


class TranscribeJob(QRunnable):
    def __init__(self, wav: bytes, config: dict):
        super().__init__()
        self.wav = wav
        self.config = config
        self.signals = TranscribeSignals()

    def run(self):
        data = {"model": self.config["model"], "response_format": "json"}
        if self.config.get("language"):
            data["language"] = self.config["language"]
        try:
            r = httpx.post(f"{self.config['server_url'].rstrip('/')}/audio/transcriptions",
                           data=data, files={"file": ("audio.wav", self.wav, "audio/wav")},
                           timeout=self.config["request_timeout"])
            if r.status_code != 200:
                try:
                    message = r.json()["error"]["message"]
                except Exception:
                    message = r.text[:300]
                self.signals.failed.emit(f"HTTP {r.status_code}: {message}")
                return
            self.signals.done.emit(strip_asr_prefix(r.json().get("text", "")))
        except httpx.HTTPError as exc:
            self.signals.failed.emit(f"サーバーに接続できません: {exc}")


# ---- Main window ------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.recorder = Recorder(config.get("input_device"))
        self.pool = QThreadPool.globalInstance()
        self.file_path: Path | None = None
        self.record_started = 0.0
        self.transcribing = False
        self.last_draft = ""

        self.editor = QPlainTextEdit()
        self.editor.setFont(QFont(config["font_family"], config["font_size"]))
        self.editor.setTabStopDistance(self.editor.fontMetrics().horizontalAdvance(" ") * 4)
        self.editor.document().modificationChanged.connect(self._update_title)
        self.setCentralWidget(self.editor)

        self._build_toolbar()
        self.status = QLabel()
        self.statusBar().addWidget(self.status, 1)

        self.tick = QTimer(self, interval=250, timeout=self._on_tick)
        self.autosave = QTimer(self, interval=max(5, int(config["autosave_seconds"])) * 1000,
                               timeout=self._autosave)
        self.autosave.start()

        self._restore_draft()
        self._update_title()
        self.resize(900, 700)
        QTimer.singleShot(0, self._check_server)

    # -- UI ---------------------------------------------------------------
    def _build_toolbar(self):
        bar = QToolBar(movable=False)
        self.addToolBar(bar)
        hotkey = self.config["hotkey"]
        self.record_btn = QPushButton(f"● 録音 ({hotkey})")
        self.record_btn.setMinimumWidth(150)
        self.record_btn.clicked.connect(self.toggle_recording)
        bar.addWidget(self.record_btn)
        bar.addSeparator()
        for label, key, slot in [("新規", QKeySequence.New, self.new_file),
                                 ("開く", QKeySequence.Open, self.open_file),
                                 ("保存", QKeySequence.Save, self.save_file),
                                 ("名前を付けて保存", QKeySequence.SaveAs, self.save_file_as)]:
            action = QAction(label, self, shortcut=key, triggered=slot)
            bar.addAction(action)
        shortcut = QShortcut(QKeySequence(hotkey), self, activated=self.toggle_recording)
        shortcut.setContext(Qt.ApplicationShortcut)

    def _set_status(self, text: str, color: str = ""):
        self.status.setText(text)
        self.status.setStyleSheet(f"color: {color};" if color else "")

    def _update_title(self, *_):
        name = self.file_path.name if self.file_path else "無題"
        mark = "*" if self.editor.document().isModified() else ""
        self.setWindowTitle(f"{mark}{name} - Speech-2-Text")

    # -- Server -----------------------------------------------------------
    def _check_server(self):
        try:
            httpx.get(f"{self.config['server_url'].rstrip('/')}/models", timeout=3).raise_for_status()
            self._set_status(f"準備完了 — {self.config['hotkey']} で録音開始 / 停止")
        except httpx.HTTPError:
            self._set_status(f"⚠ ASR サーバーに接続できません ({self.config['server_url']})。"
                             "Start-Qwen3-ASR.bat で起動してください。", "#c0392b")

    # -- Recording --------------------------------------------------------
    def toggle_recording(self):
        if self.recorder.stream is not None:
            self._stop_recording()
        elif not self.transcribing:
            self._start_recording()

    def _start_recording(self):
        try:
            self.recorder.start()
        except Exception as exc:
            QMessageBox.critical(self, "録音エラー", f"マイクを開けませんでした:\n{exc}")
            return
        self.record_started = time.monotonic()
        self.record_btn.setText(f"■ 停止 ({self.config['hotkey']})")
        self.record_btn.setStyleSheet("background:#c0392b; color:white; font-weight:bold;")
        self.tick.start()
        self._on_tick()

    def _stop_recording(self):
        self.tick.stop()
        wav = self.recorder.stop()
        self.record_btn.setStyleSheet("")
        if not wav or time.monotonic() - self.record_started < 0.3:
            self._reset_record_button()
            self._set_status("録音が短すぎます")
            return
        self.transcribing = True
        self.record_btn.setText("文字起こし中…")
        self.record_btn.setEnabled(False)
        self._set_status("文字起こし中…", "#2471a3")
        job = TranscribeJob(wav, self.config)
        job.signals.done.connect(self._on_transcript)
        job.signals.failed.connect(self._on_transcribe_failed)
        self.pool.start(job)

    def _on_tick(self):
        elapsed = time.monotonic() - self.record_started
        self._set_status(f"● 録音中 {int(elapsed // 60)}:{int(elapsed % 60):02d}", "#c0392b")
        if elapsed >= self.config["max_recording_seconds"]:
            self._stop_recording()

    def _reset_record_button(self):
        self.transcribing = False
        self.record_btn.setEnabled(True)
        self.record_btn.setText(f"● 録音 ({self.config['hotkey']})")

    def _on_transcript(self, text: str):
        self._reset_record_button()
        if not text:
            self._set_status("音声を検出できませんでした")
            return
        cursor = self.editor.textCursor()
        # 行の途中なら改行してから挿入し、挿入後も改行して次の入力に備える
        if cursor.positionInBlock() > 0:
            text = "\n" + text
        cursor.insertText(text + "\n")
        self.editor.setTextCursor(cursor)
        self.editor.ensureCursorVisible()
        self.editor.setFocus()
        self._set_status(f"挿入しました ({len(text.strip())} 文字)")

    def _on_transcribe_failed(self, message: str):
        self._reset_record_button()
        self._set_status(f"⚠ 文字起こし失敗: {message}", "#c0392b")

    # -- Files ------------------------------------------------------------
    def _confirm_discard(self) -> bool:
        if not self.editor.document().isModified() or (self.file_path is None and not self.editor.toPlainText()):
            return True
        answer = QMessageBox.question(self, "未保存の変更", "変更を保存しますか？",
                                      QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if answer == QMessageBox.Save:
            return self.save_file()
        return answer == QMessageBox.Discard

    def _load(self, path: Path | None, text: str):
        self.file_path = path
        self.editor.setPlainText(text)
        self.editor.document().setModified(False)
        self._update_title()

    def new_file(self):
        if self._confirm_discard():
            self._load(None, "")
            DRAFT_PATH.unlink(missing_ok=True)

    def open_file(self):
        if not self._confirm_discard():
            return
        name, _ = QFileDialog.getOpenFileName(self, "開く", str(self._default_dir()),
                                              "テキスト (*.md *.txt);;すべてのファイル (*)")
        if name:
            path = Path(name)
            self._load(path, path.read_text(encoding="utf-8-sig"))

    def save_file(self) -> bool:
        if self.file_path is None:
            return self.save_file_as()
        self.file_path.write_text(self.editor.toPlainText(), encoding="utf-8")
        self.editor.document().setModified(False)
        DRAFT_PATH.unlink(missing_ok=True)
        self._set_status(f"保存しました: {self.file_path}")
        return True

    def save_file_as(self) -> bool:
        suggested = self._default_dir() / time.strftime("memo-%Y%m%d-%H%M.md")
        name, _ = QFileDialog.getSaveFileName(self, "名前を付けて保存", str(suggested),
                                              "Markdown (*.md);;テキスト (*.txt)")
        if not name:
            return False
        self.file_path = Path(name)
        return self.save_file()

    def _default_dir(self) -> Path:
        return self.file_path.parent if self.file_path else Path.home() / "Documents"

    # -- Autosave ---------------------------------------------------------
    def _autosave(self):
        if not self.editor.document().isModified():
            return
        if self.file_path is not None:
            self.save_file()
        else:
            # 無題の文書は下書きとしてアプリフォルダに退避し、次回起動時に復元する
            text = self.editor.toPlainText()
            if text == self.last_draft:
                return
            DRAFT_PATH.parent.mkdir(exist_ok=True)
            DRAFT_PATH.write_text(text, encoding="utf-8")
            self.last_draft = text
            self._set_status(f"下書きを自動保存しました ({time.strftime('%H:%M:%S')})")

    def _restore_draft(self):
        if DRAFT_PATH.exists():
            self.last_draft = DRAFT_PATH.read_text(encoding="utf-8")
            self.editor.setPlainText(self.last_draft)
            self.editor.document().setModified(True)
            self.editor.moveCursor(self.editor.textCursor().MoveOperation.End)

    def closeEvent(self, event: QCloseEvent):
        if self.recorder.stream is not None:
            self.recorder.stop()
        if self.file_path is None:
            # 無題は確認せず下書きとして残す（次回起動時に復元）
            if self.editor.toPlainText():
                DRAFT_PATH.parent.mkdir(exist_ok=True)
                DRAFT_PATH.write_text(self.editor.toPlainText(), encoding="utf-8")
            event.accept()
        elif self._confirm_discard():
            event.accept()
        else:
            event.ignore()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Speech-2-Text")
    window = MainWindow(load_config())
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
