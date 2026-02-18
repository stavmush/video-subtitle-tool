"""
Whisper transcription utilities.

- transcribe_video: transcribe audio in its original language
- transcribe_to_english: use Whisper's built-in translate task to get English output
  (used as the first stage of the any→Hebrew pipeline)
"""

import streamlit as st
import whisper
from typing import Any

Segment = dict[str, Any]


@st.cache_resource(show_spinner=False)
def _load_whisper_model(model_size: str) -> whisper.Whisper:
    """Download (first run) and cache the Whisper model in memory."""
    return whisper.load_model(model_size)


def _parse_segments(raw_segments: list[dict]) -> list[Segment]:
    """Normalize Whisper's raw segment dicts to our internal format."""
    result = []
    for seg in raw_segments:
        text = seg["text"].strip()
        if not text:
            continue
        result.append({
            "id":    int(seg["id"]) + 1,  # 1-based for SRT
            "start": float(seg["start"]),
            "end":   float(seg["end"]),
            "text":  text,
        })
    return result


def transcribe_video(video_path: str, model_size: str) -> tuple[list[Segment], str]:
    """
    Transcribe video audio in its original language.

    Returns:
        (segments, detected_language)  where detected_language is an ISO 639-1 code
    """
    model = _load_whisper_model(model_size)
    result = model.transcribe(
        video_path,
        task="transcribe",
        verbose=False,
        fp16=False,
    )
    detected_lang: str = result.get("language", "unknown")
    segments = _parse_segments(result["segments"])
    return segments, detected_lang


def transcribe_to_english(video_path: str, model_size: str) -> list[Segment]:
    """
    Use Whisper's built-in translate task to produce English-language segments
    from a video in any source language.

    This is used as stage 1 of the any-language → Hebrew pipeline.
    """
    model = _load_whisper_model(model_size)
    result = model.transcribe(
        video_path,
        task="translate",
        verbose=False,
        fp16=False,
    )
    return _parse_segments(result["segments"])
