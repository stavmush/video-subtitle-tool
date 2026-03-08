"""
Autosave / session persistence for the subtitle tool.

Saves the current subtitle DataFrame plus metadata to a JSON file in
~/.subtitle_tool/autosave.json so work can be restored after a
dropped connection, browser close, or server restart.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

AUTOSAVE_PATH = Path.home() / ".subtitle_tool" / "autosave.json"


def save_session(
    df: pd.DataFrame,
    source_filename: str | None,
    video_path: str | None,
    target_language: str | None,
) -> None:
    """Write current session to disk. Silently ignores write errors."""
    try:
        AUTOSAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "source_filename": source_filename or "",
            "video_path": video_path or "",
            "target_language": target_language or "en",
            "subtitles": df.to_dict(orient="records"),
        }
        AUTOSAVE_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass


def load_session() -> dict | None:
    """Return the autosave dict, or None if no valid save exists."""
    if not AUTOSAVE_PATH.exists():
        return None
    try:
        return json.loads(AUTOSAVE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def clear_session() -> None:
    """Delete the autosave file."""
    AUTOSAVE_PATH.unlink(missing_ok=True)
