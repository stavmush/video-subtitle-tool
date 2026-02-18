"""
Whisper transcription using faster-whisper (CTranslate2 backend).

Advantages over openai-whisper:
  - ~4x less RAM via int8 quantization
  - ~2-4x faster on CPU
  - Native segment-level progress callbacks (no tqdm patching needed)
  - Medium model uses ~770MB instead of 1.5GB
"""

import streamlit as st
from faster_whisper import WhisperModel
from typing import Any, Callable

Segment = dict[str, Any]


@st.cache_resource(show_spinner=False)
def _load_model(model_size: str) -> WhisperModel:
    """
    Download (first run) and cache the faster-whisper model.
    Uses int8 quantization on CPU for minimal RAM usage.
    compute_type="int8" halves memory vs float32 with negligible accuracy loss.
    """
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def _collect_segments(
    segments_iter,
    total_duration: float,
    on_progress: Callable[[float], None] | None,
) -> list[Segment]:
    """
    Consume the faster-whisper segment generator, optionally reporting progress.
    faster-whisper is lazy — segments are produced as decoding advances, giving
    us natural progress checkpoints without any tqdm patching.
    """
    result = []
    idx = 1
    for seg in segments_iter:
        text = seg.text.strip()
        if text:
            result.append({
                "id":    idx,
                "start": float(seg.start),
                "end":   float(seg.end),
                "text":  text,
            })
            idx += 1
        if on_progress and total_duration > 0:
            on_progress(min(seg.end / total_duration, 1.0))
    return result


def transcribe_video(
    video_path: str,
    model_size: str,
    on_progress: Callable[[float], None] | None = None,
) -> tuple[list[Segment], str]:
    """
    Transcribe video audio in its original language.

    Returns:
        (segments, detected_language)  — detected_language is an ISO 639-1 code
    """
    model = _load_model(model_size)
    segments_iter, info = model.transcribe(
        video_path,
        task="transcribe",
        beam_size=5,
        condition_on_previous_text=False,
        vad_filter=True,        # skip silent sections — faster and fewer empty segments
        vad_parameters={"min_silence_duration_ms": 500},
    )
    segments = _collect_segments(segments_iter, info.duration, on_progress)
    return segments, info.language


def transcribe_to_english(
    video_path: str,
    model_size: str,
    on_progress: Callable[[float], None] | None = None,
) -> list[Segment]:
    """
    Use Whisper's built-in translate task to produce English-language segments
    from a video in any source language.

    Used as stage 1 of the any-language → Hebrew pipeline.
    """
    model = _load_model(model_size)
    segments_iter, info = model.transcribe(
        video_path,
        task="translate",
        beam_size=5,
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    return _collect_segments(segments_iter, info.duration, on_progress)
