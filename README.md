# Speech-2-Text

ローカルで稼働している Qwen3-ASR を使って音声をテキストにし、そのまま編集できる思考整理向けのシンプルなエディター。
Hermes Agent とは独立して動作します。

## 必要なもの

- Qwen3-ASR サーバー（`qwen-asr-serve`、既定 `http://localhost:8000/v1`）
  - デスクトップの `Start-Qwen3-ASR.bat` で起動
- Python 3.11+

## 起動

`run.bat` をダブルクリック（初回のみ `.venv` を作成して依存関係をインストール）。

## 使い方

| 操作 | キー |
|---|---|
| 録音開始 / 停止 → カーソル位置に挿入 | **F9**（または「● 録音」ボタン） |
| 新規 / 開く / 保存 / 名前を付けて保存 | Ctrl+N / Ctrl+O / Ctrl+S / Ctrl+Shift+S |
| 文字サイズ変更 | Ctrl+マウスホイール |

- **自動保存**: 保存済みのファイルは `autosave_seconds` ごとに上書き保存。無題の文書は `autosave/draft.md` に下書き保存され、次回起動時に復元されます。
- 録音は WAV で直接 ASR サーバーへ送るため、Hermes 用の変換プロキシ（:8001）は不要です。

## 設定 (`config.json`)

初回起動時に既定値で作成されます。変更後はアプリを再起動してください。

| キー | 既定値 | 説明 |
|---|---|---|
| `server_url` | `http://localhost:8000/v1` | ASR サーバーの URL |
| `model` | `Qwen/Qwen3-ASR-1.7B` | `qwen-asr-serve` に渡したモデル名 |
| `language` | `""` | `""` = 自動判定、`"ja"` / `"en"` など ISO コード（`"Japanese"` は不可） |
| `hotkey` | `F9` | 録音の開始 / 停止キー |
| `autosave_seconds` | `30` | 自動保存の間隔（秒） |
| `max_recording_seconds` | `600` | 1 回の録音の上限（秒） |
| `request_timeout` | `120` | 文字起こしリクエストのタイムアウト（秒） |
| `input_device` | `null` | マイク。`null` = 既定、番号またはデバイス名 |
| `font_family` / `font_size` | `Yu Gothic UI` / `14` | エディターのフォント |
