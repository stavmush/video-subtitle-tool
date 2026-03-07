# Contributing to Video Subtitle Tool

Thanks for your interest in contributing! This is a local-first tool built with Streamlit and open-source AI models — contributions of all kinds are welcome.

## Ways to Contribute

- **Bug reports** — found something broken? Open an issue.
- **Feature requests** — have an idea? Open an issue and describe it.
- **Code** — fix a bug, implement a feature, or improve performance.
- **Docs** — improve the README, add examples, or clarify setup steps.

---

## Getting Started

### 1. Fork & clone

```bash
git clone https://github.com/your-username/video-subtitle-tool.git
cd video-subtitle-tool
```

### 2. Set up the environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

You'll also need `ffmpeg` installed — see [README.md](README.md#-quick-start) for platform-specific instructions.

### 3. Run the app

```bash
streamlit run app.py
```

---

## Project Structure

```
app.py              — Main Streamlit UI and session state
utils/
  transcribe.py     — Whisper transcription logic
  translate.py      — Helsinki-NLP translation
  improve.py        — Grammar improvement pass
  srt_utils.py      — SRT read/write helpers
  video.py          — FFmpeg burn-in
```

---

## Submitting a Pull Request

1. Create a branch: `git checkout -b feature/your-feature-name`
2. Make your changes and test locally
3. Keep commits focused — one logical change per commit
4. Open a PR against `main` with a clear description of what changed and why

---

## Coding Style

- Python 3.9+ compatible
- Keep functions small and single-purpose
- Avoid adding new dependencies unless necessary — this project prioritizes a lean, local-first setup
- Session state keys should be added to `STATE_DEFAULTS` in `app.py` so they're always initialized

---

## Reporting Bugs

Please include:
- OS and Python version
- The video format and approximate duration
- Which Whisper model size you used
- The full error message or traceback from the terminal
