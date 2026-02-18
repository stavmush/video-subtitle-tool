# Video Subtitle Tool

A local Streamlit app that transcribes MP4 videos with Whisper, translates subtitles to English or Hebrew using offline AI models, lets you edit them, and exports as SRT or a burned-in MP4.

## Features

- Upload MP4, MKV, AVI, or MOV videos
- Transcribe audio with [OpenAI Whisper](https://github.com/openai/whisper) (runs fully locally)
- Translate to **English** or **Hebrew** using offline [Helsinki-NLP](https://huggingface.co/Helsinki-NLP) models
- Edit subtitles in an interactive table (adjust text and timestamps, add/delete rows)
- Export as `.srt` file or burn subtitles permanently into the video

## Setup

### 1. System dependencies

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt install -y ffmpeg
```

### 2. Python environment

```bash
cd video-subtitle-tool
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Note:** PyTorch is a large dependency (~1–2 GB). The first `pip install` will take a while.

### 3. Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Model Download Notes

Models are downloaded automatically on first use and cached locally:

| Model | Size | When |
|---|---|---|
| Whisper (size you choose) | 75 MB – 1.5 GB | On first Transcribe |
| Helsinki-NLP opus-mt-en-he | ~300 MB | On first Hebrew translation |

After the first run, everything works offline.

## Hebrew Subtitles

Hebrew text renders right-to-left automatically in the exported SRT and burned video (handled by FFmpeg's libass renderer). For burned-in subtitles you need a font with Hebrew glyph support:

- **Noto Sans Hebrew** (recommended): `brew install font-noto-sans-hebrew` (macOS)
  Font path: `/Library/Fonts/NotoSansHebrew-Regular.ttf`
- **Arial Unicode MS**: usually at `/Library/Fonts/Arial Unicode.ttf` on macOS

Enter the font path in the app before burning.

## Whisper Model Sizes

| Size | VRAM | Speed | Accuracy |
|---|---|---|---|
| tiny | ~1 GB | fastest | lowest |
| base | ~1 GB | fast | low |
| small | ~2 GB | moderate | good |
| medium | ~5 GB | slow | great |
| large | ~10 GB | slowest | best |

`small` is the default — a good balance for most videos.
