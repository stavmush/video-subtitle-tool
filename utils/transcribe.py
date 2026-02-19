"""
Whisper transcription using faster-whisper (CTranslate2 backend).

Advantages over openai-whisper:
  - ~4x less RAM via int8 quantization
  - ~2-4x faster on CPU
  - Native segment-level progress callbacks (no tqdm patching needed)
  - Medium model uses ~770MB instead of 1.5GB
"""

import gc
import os
import subprocess
import tempfile

import streamlit as st
from faster_whisper import WhisperModel
from typing import Any, Callable

Segment = dict[str, Any]

# Large models use more memory during beam search; lower beam_size keeps
# quality high while cutting inference time ~30-40% on CPU.
_BEAM_SIZE: dict[str, int] = {
    "tiny":     5,
    "base":     5,
    "small":    5,
    "medium":   5,
    "large-v2": 2,
    "large-v3": 2,
}


@st.cache_resource(show_spinner=False)
def _load_model(model_size: str) -> WhisperModel:
    """
    Download (first run) and cache the faster-whisper model.
    Uses int8 quantization on CPU for minimal RAM usage.
    compute_type="int8" halves memory vs float32 with negligible accuracy loss.
    """
    gc.collect()  # free any lingering objects before allocating the model
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def _extract_audio(video_path: str) -> str:
    """
    Extract audio from video to a temporary 16 kHz mono WAV file.
    Returns the temp file path — caller is responsible for deleting it.

    Benefits for long videos:
      - Whisper only reads the small audio file, not the full video
      - If both transcribe and translate tasks run, audio is decoded once
      - More robust across exotic video codecs
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", video_path,
            "-vn",                        # drop video stream
            "-acodec", "pcm_s16le",       # uncompressed PCM — Whisper's native format
            "-ar", "16000",               # 16 kHz — Whisper's expected sample rate
            "-ac", "1",                   # mono
            tmp.name,
        ],
        capture_output=True,
        check=True,
    )
    return tmp.name


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
    audio_path = _extract_audio(video_path)
    try:
        model = _load_model(model_size)
        segments_iter, info = model.transcribe(
            audio_path,
            task="transcribe",
            beam_size=_BEAM_SIZE.get(model_size, 5),
            condition_on_previous_text=False,
            vad_filter=True,        # skip silent sections — faster and fewer empty segments
            vad_parameters={"min_silence_duration_ms": 500},
        )
        segments = _collect_segments(segments_iter, info.duration, on_progress)
        return segments, info.language
    finally:
        os.unlink(audio_path)


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
    audio_path = _extract_audio(video_path)
    try:
        model = _load_model(model_size)
        segments_iter, info = model.transcribe(
            audio_path,
            task="translate",
            beam_size=_BEAM_SIZE.get(model_size, 5),
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        return _collect_segments(segments_iter, info.duration, on_progress)
    finally:
        os.unlink(audio_path)
