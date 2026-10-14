"""Speech-2-Text: 思考整理向けのシンプルな音声入力エディター.

ローカルで稼働中の Qwen3-ASR (OpenAI 互換 /v1/audio/transcriptions) に録音を送り、
文字起こし結果をカーソル位置に挿入する。テキストは自由に編集・保存できる。
"""
from __future__ import annotations

import io
import json
import re
import sys
import threading
import time
import wave
from pathlib import Path

import httpx
import numpy as np
import sounddevice as sd
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QAction, QCloseEvent, QFont, QKeySequence, QShortcut, QTextCursor
from PySide6.QtWidgets import (QApplication, QFileDialog, QLabel, QMainWindow, QMessageBox,
                               QPlainTextEdit, QPushButton, QToolBar)

APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "config.json"
DRAFT_PATH = APP_DIR / "autosave" / "draft.md"
FAILED_DIR = APP_DIR / "autosave" / "failed"

DEFAULT_CONFIG = {
    "server_url": "http://localhost:8000/v1",
    "model": "Qwen/Qwen3-ASR-1.7B",
    "language": "",            # "" = 自動判定, "ja" / "en" など ISO コード
    "hotkey": "F9",
    "autosave_seconds": 30,
    "max_recording_seconds": 3600,
    "chunk_seconds": 120,      # 録音中、この長さごとに区切って順次文字起こしする
    "request_timeout": 300,
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


# Qwen3-ASR の生出力には "language Japanese<asr_text>" が付く。長い音声はサーバー側で
# チャンク分割され、各チャンクにこのラベルが付いたまま連結されるので全箇所を除去する
ASR_LABEL = re.compile(r"\s*(?:language\s+[A-Za-z]+\s*)?<asr_text>\s*")
CJK = re.compile(r"[　-ヿ㐀-鿿＀-￯]")


def strip_asr_prefix(text: str) -> str:
    def join(m: re.Match) -> str:
        before = text[m.start() - 1] if m.start() > 0 else ""
        after = text[m.end()] if m.end() < len(text) else ""
        # 日本語同士の継ぎ目は詰め、英単語同士の継ぎ目には空白を残す
        if not before or not after or CJK.match(before) or CJK.match(after):
            return ""
        return " "

    return ASR_LABEL.sub(join, text).strip()


# ---- Recording --------------------------------------------------------------

def to_wav(pcm: np.ndarray, samplerate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(samplerate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class Recorder:
    """マイク入力をメモリに貯め、区間ごと / 停止時に WAV バイト列を返す."""

    # Qwen3-ASR は内部で 16 kHz に変換するので、録音時点で 16 kHz にしてサイズを抑える
    TARGET_RATE = 16000
    # 区間の切れ目を探す範囲（末尾から）と、音量を比べる単位
    CUT_SEARCH_SECONDS = 5.0
    CUT_FRAME_SECONDS = 0.1

    def __init__(self, device):
        self.device = device
        self.stream: sd.InputStream | None = None
        self.chunks: list[np.ndarray] = []
        self.lock = threading.Lock()
        self.samplerate = self.TARGET_RATE

    def start(self):
        try:
            sd.check_input_settings(device=self.device, channels=1, samplerate=self.TARGET_RATE, dtype="int16")
            self.samplerate = self.TARGET_RATE
        except Exception:
            self.samplerate = int(sd.query_devices(self.device, kind="input")["default_samplerate"])
        self.chunks = []
        self.stream = sd.InputStream(device=self.device, channels=1, samplerate=self.samplerate,
                                     dtype="int16", callback=self._callback)
        self.stream.start()

    def _callback(self, indata, frames, time_info, status):
        with self.lock:
            self.chunks.append(indata[:, 0].copy())

    def buffered_seconds(self) -> float:
        with self.lock:
            return sum(len(c) for c in self.chunks) / self.samplerate

    def take_segment(self, seconds: float) -> bytes:
        """先頭から約 seconds 秒を、その直前 5 秒で最も静かな位置で切り出す。残りは次の区間へ持ち越す."""
        with self.lock:
            pcm = np.concatenate(self.chunks) if self.chunks else np.zeros(0, np.int16)
            self.chunks = []
        frame = int(self.samplerate * self.CUT_FRAME_SECONDS)
        end = min(len(pcm), int(self.samplerate * seconds))
        start = max(0, end - int(self.samplerate * self.CUT_SEARCH_SECONDS))
        count = (end - start) // frame
        cut = end
        if count > 1:
            window = pcm[start:start + count * frame].astype(np.float32).reshape(count, frame)
            cut = start + int(np.argmin((window ** 2).mean(axis=1))) * frame + frame // 2
        with self.lock:
            # 切り出し中に届いた音声より前に、持ち越し分を戻す
            self.chunks.insert(0, pcm[cut:])
        return to_wav(pcm[:cut], self.samplerate)

    def stop(self) -> bytes | None:
        if self.stream is None:
            return None
        self.stream.stop()
        self.stream.close()
        self.stream = None
        with self.lock:
            pcm = np.concatenate(self.chunks) if self.chunks else np.zeros(0, np.int16)
            self.chunks = []
        return to_wav(pcm, self.samplerate) if len(pcm) else None


# ---- Transcription (background thread) --------------------------------------

class TranscribeSignals(QObject):
    done = Signal(object, str)     # (job, text)
    failed = Signal(object, str)   # (job, message)


class TranscribeJob(QRunnable):
    def __init__(self, wav: bytes, config: dict, final: bool = False):
        super().__init__()
        self.wav = wav
        self.config = config
        self.final = final
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
                self.signals.failed.emit(self, f"HTTP {r.status_code}: {message}")
                return
            self.signals.done.emit(self, strip_asr_prefix(r.json().get("text", "")))
        except httpx.HTTPError as exc:
            self.signals.failed.emit(self, f"サーバーに接続できません: {exc}")


# ---- Main window ------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.recorder = Recorder(config.get("input_device"))
        # 1 スレッドに限定して区間を録音順に文字起こしし、結果も順番どおりに届ける
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.jobs: list[TranscribeJob] = []
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
        # 録音開始時のカーソル位置に区間ごとの結果を順に挿入する。QTextCursor は
        # ユーザーが録音中に他の場所を編集しても位置が追従する
        self.insert_cursor = QTextCursor(self.editor.textCursor())
        self.inserted_chars = 0
        self.last_char = ""
        self.segments_sent = 0
        self.segments_done = 0
        self.failures: list[str] = []
        self.record_btn.setText(f"■ 停止 ({self.config['hotkey']})")
        self.record_btn.setStyleSheet("background:#c0392b; color:white; font-weight:bold;")
        self.tick.start()
        self._on_tick()

    def _stop_recording(self):
        self.tick.stop()
        wav = self.recorder.stop()
        self.record_btn.setStyleSheet("")
        if self.segments_sent == 0 and (not wav or time.monotonic() - self.record_started < 0.3):
            self._reset_record_button()
            self._set_status("録音が短すぎます")
            return
        self.transcribing = True
        self.record_btn.setEnabled(False)
        if wav:
            self._submit_segment(wav, final=True)
        else:
            self._finish_session()
        self._show_progress()

    def _submit_segment(self, wav: bytes, final: bool = False):
        self.segments_sent += 1
        job = TranscribeJob(wav, self.config, final)
        # 結果が届くまで参照を保持する（自動削除させるとシグナルが失われることがある）
        job.setAutoDelete(False)
        self.jobs.append(job)
        # ウィンドウのメソッドに接続し、結果をメインスレッドで受け取る
        job.signals.done.connect(self._on_segment)
        job.signals.failed.connect(self._on_segment_failed)
        self.pool.start(job)

    def _show_progress(self):
        progress = f"{self.segments_done}/{self.segments_sent}"
        if self.recorder.stream is None:
            self.record_btn.setText(f"文字起こし中… ({progress})")
            self._set_status(f"文字起こし中… 区間 {progress}", "#2471a3")

    def _on_tick(self):
        elapsed = time.monotonic() - self.record_started
        status = f"● 録音中 {int(elapsed // 60)}:{int(elapsed % 60):02d}"
        if self.segments_sent:
            status += f" — 区間 {self.segments_done}/{self.segments_sent} 文字起こし済み"
        self._set_status(status, "#c0392b")
        if elapsed >= self.config["max_recording_seconds"]:
            self._stop_recording()
        elif self.recorder.buffered_seconds() >= self.config["chunk_seconds"]:
            self._submit_segment(self.recorder.take_segment(self.config["chunk_seconds"]))

    def _reset_record_button(self):
        self.transcribing = False
        self.record_btn.setEnabled(True)
        self.record_btn.setText(f"● 録音 ({self.config['hotkey']})")

    def _insert_segment(self, text: str):
        cursor = self.insert_cursor
        if not self.inserted_chars:
            # 行の途中なら改行してから挿入する
            if cursor.positionInBlock() > 0:
                text = "\n" + text
        elif not (CJK.match(self.last_char) or CJK.match(text[0])):
            # 英単語同士の継ぎ目だけ空白でつなぐ
            text = " " + text
        cursor.insertText(text)
        self.inserted_chars += len(text.strip())
        self.last_char = text[-1]

    def _on_segment(self, job: TranscribeJob, text: str):
        self.jobs.remove(job)
        self.segments_done += 1
        if text:
            self._insert_segment(text)
        if job.final:
            self._finish_session()
        else:
            self._show_progress()

    def _on_segment_failed(self, job: TranscribeJob, message: str):
        self.jobs.remove(job)
        self.segments_done += 1
        # 失敗した区間の音声は捨てずに保存しておく
        FAILED_DIR.mkdir(parents=True, exist_ok=True)
        path = FAILED_DIR / time.strftime(f"%Y%m%d-%H%M%S-{self.segments_done}.wav")
        path.write_bytes(job.wav)
        self.failures.append(f"{message}（音声: {path}）")
        if job.final:
            self._finish_session()
        else:
            self._show_progress()

    def _finish_session(self):
        self._reset_record_button()
        if self.inserted_chars:
            # 挿入後に改行して次の入力に備える
            self.insert_cursor.insertText("\n")
            self.editor.setTextCursor(self.insert_cursor)
            self.editor.ensureCursorVisible()
            self.editor.setFocus()
        if self.failures:
            self._set_status(f"⚠ {len(self.failures)}/{self.segments_sent} 区間の文字起こし失敗: "
                             f"{self.failures[-1]}", "#c0392b")
        elif self.inserted_chars:
            self._set_status(f"挿入しました ({self.inserted_chars} 文字, {self.segments_sent} 区間)")
        else:
            self._set_status("音声を検出できませんでした")

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
