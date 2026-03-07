# 🎬 Video Subtitle Tool

> Local, privacy-first AI subtitle generation — no cloud, no API keys, no data leaving your machine.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Whisper](https://img.shields.io/badge/Powered%20by-Whisper-green?logo=openai)](https://github.com/openai/whisper)

Upload a video → transcribe with Whisper → translate → edit subtitles → export SRT or burned-in MP4. Everything runs locally using open-source models.

---

## ✨ Features

- **Multi-format support** — MP4, MKV, AVI, MOV
- **Local transcription** — [faster-whisper](https://github.com/guillaumekln/faster-whisper) with 5 model sizes (tiny → large)
- **Offline translation** — English ↔ Hebrew via [Helsinki-NLP](https://huggingface.co/Helsinki-NLP) transformer models
- **Interactive subtitle editor** — edit text, adjust timestamps, add/delete rows live
- **SRT import** — load and edit an existing `.srt` file without re-transcribing
- **Grammar improvement** — AI-powered cleanup of transcribed text
- **Export options** — download `.srt` or burn subtitles permanently into the video via FFmpeg
- **Right-to-left support** — Hebrew subtitles render correctly in both SRT and burned video

---

## 🚀 Quick Start

### Prerequisites

**macOS**
```bash
brew install ffmpeg
```

**Ubuntu / Debian**
```bash
sudo apt update && sudo apt install -y ffmpeg
```

**Windows**
Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

### Install & Run

```bash
git clone https://github.com/stavmush/video-subtitle-tool.git
cd video-subtitle-tool

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt  # ~1–2 GB, takes a few minutes first time

streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧠 How It Works

```
Video file
    ↓  FFmpeg extracts audio
Whisper model
    ↓  Transcribes speech → raw subtitle segments
Helsinki-NLP (optional)
    ↓  Translates to English or Hebrew
Subtitle editor
    ↓  Edit text & timestamps in-browser
Export
    ↓  .srt file  OR  burned-in MP4 (FFmpeg + libass)
```

---

## 📦 Model Reference

Models are downloaded automatically on first use and cached to disk. After that, everything works offline.

| Model | Size | Notes |
|---|---|---|
| Whisper tiny | ~75 MB | Fastest, lowest accuracy |
| Whisper base | ~140 MB | Good for clear audio |
| Whisper small | ~460 MB | **Default** — best speed/accuracy tradeoff |
| Whisper medium | ~1.5 GB | Better accuracy |
| Whisper large | ~3 GB | Best accuracy, slow |
| Helsinki-NLP (en→he) | ~300 MB | Downloaded on first Hebrew translation |

---

## 🌐 Hebrew / RTL Subtitles

For burned-in Hebrew subtitles, you need a font with Hebrew glyph support:

**macOS (recommended):**
```bash
brew install font-noto-sans-hebrew
# Font path: /Library/Fonts/NotoSansHebrew-Regular.ttf
```

**Linux:**
```bash
sudo apt install fonts-noto
```

Enter the font path in the app's burn settings before exporting.

---

## 🗂 Project Structure

```
video-subtitle-tool/
├── app.py                  # Main Streamlit app
├── requirements.txt
└── utils/
    ├── transcribe.py       # Whisper transcription
    ├── translate.py        # Helsinki-NLP translation
    ├── improve.py          # Grammar improvement
    ├── srt_utils.py        # SRT parsing / formatting
    └── video.py            # FFmpeg burn-in
```

---

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) to get started.

Ideas for what to work on:

- [ ] Additional translation language pairs
- [ ] Subtitle styling options (font, size, color, position)
- [ ] Batch processing multiple videos
- [ ] Docker image for zero-setup deployment
- [ ] Word-level timestamp highlighting
- [ ] Speaker diarization

---

## 📄 License

MIT — see [LICENSE](LICENSE).
