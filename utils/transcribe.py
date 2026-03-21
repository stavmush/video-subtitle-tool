"""
Whisper transcription using faster-whisper (CTranslate2 backend).

Advantages over openai-whisper:
  - ~4x less RAM via int8 quantization
  - ~2-4x faster on CPU
  - Native segment-level progress callbacks (no tqdm patching needed)
  - Medium model uses ~770MB instead of 1.5GB

Memory strategy for large files:
  Audio is processed in 10-minute chunks extracted directly from the
  video. Each chunk is ~50 MB in memory vs ~460 MB for a full 2-hour
  WAV. The model stays loaded across chunks; gc.collect() runs between
  them to release intermediate allocations.
"""

import gc
import json
import os
import subprocess
import tempfile
from typing import Any, Callable

import numpy as np
import noisereduce as nr
import soundfile as sf

import streamlit as st
from faster_whisper import WhisperModel

Segment = dict[str, Any]

# 10-minute chunks keep peak audio memory ~50 MB regardless of video length.
_CHUNK_SECONDS = 600

# beam_size=1 (greedy) for large models: ~20% faster, meaningfully less
# decoder memory. Accuracy is still excellent at that model size.
_BEAM_SIZE: dict[str, int] = {
    "tiny":     5,
    "base":     5,
    "small":    5,
    "medium":   4,
    "large-v2": 1,
    "large-v3": 1,
}


@st.cache_resource(show_spinner=False)
def _load_model(model_size: str) -> WhisperModel:
    """Download (first run) and cache the faster-whisper model."""
    gc.collect()
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def _get_duration(video_path: str) -> float:
    """Return video/audio duration in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", video_path,
        ],
        capture_output=True, check=True, text=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


_FREQ_BANDS = {
    "bass":   (20,   300),   # rumble, traffic, AC hum
    "speech": (300,  3000),  # voice frequencies
    "treble": (3000, 8000),  # hiss, fan noise, electronics
}


def _band_reductions(data_before: np.ndarray, data_after: np.ndarray, rate: int) -> dict[str, float]:
    """Return % energy removed per frequency band using FFT magnitude comparison."""
    freqs = np.fft.rfftfreq(len(data_before), 1.0 / rate)
    mag_before = np.abs(np.fft.rfft(data_before))
    mag_after  = np.abs(np.fft.rfft(data_after))
    result = {}
    for name, (lo, hi) in _FREQ_BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        rms_b = float(np.sqrt(np.mean(mag_before[mask] ** 2)))
        rms_a = float(np.sqrt(np.mean(mag_after[mask] ** 2)))
        result[name] = (1.0 - rms_a / rms_b) * 100.0 if rms_b > 0 else 0.0
    return result


def _denoise_audio(audio_path: str) -> tuple[float, float, dict[str, float]]:
    """Apply spectral gating noise reduction in-place on a WAV file.

    Returns (rms_before, rms_after, band_reductions) where band_reductions is
    a dict of {band_name: % energy removed} for bass, speech, and treble bands.
    """
    data, rate = sf.read(audio_path)
    rms_before = float(np.sqrt(np.mean(data ** 2)))
    reduced = nr.reduce_noise(y=data, sr=rate, stationary=False)
    rms_after = float(np.sqrt(np.mean(reduced ** 2)))
    bands = _band_reductions(data, reduced, rate)
    sf.write(audio_path, reduced, rate, subtype="PCM_16")
    return rms_before, rms_after, bands


def _extract_audio_chunk(video_path: str, start: float, duration: float) -> str:
    """
    Extract [start, start+duration] seconds of audio from video to a temp WAV.

    Placing -ss before -i uses FFmpeg's fast keyframe seek so we don't decode
    the entire video up to `start` — important for chunks deep into long files.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(start),        # seek before -i for fast seek
            "-i", video_path,
            "-t", str(duration),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            tmp.name,
        ],
        capture_output=True,
        check=True,
    )
    return tmp.name


def _transcribe_chunked(
    video_path: str,
    task: str,                                   # "transcribe" or "translate"
    model_size: str,
    on_progress: Callable[[float], None] | None,
    denoise: bool = False,
) -> tuple[list[Segment], str, float | None]:
    """
    Transcribe or translate video audio in memory-bounded 10-minute chunks.

    Each chunk is extracted fresh from the video, processed, then deleted.
    Timestamps are offset so the returned segments have global video times.

    Returns (segments, detected_language, denoise_reduction_pct).
    denoise_reduction_pct is None when denoise=False, otherwise the average
    percentage of audio energy removed across all chunks (0–100).
    """
    total = _get_duration(video_path)
    model = _load_model(model_size)
    beam_size = _BEAM_SIZE.get(model_size, 5)

    all_segments: list[Segment] = []
    detected_lang = "unknown"
    seg_id = 1
    chunk_start = 0.0
    rms_pairs: list[tuple[float, float]] = []
    band_accum: dict[str, list[float]] = {b: [] for b in _FREQ_BANDS}

    while chunk_start < total:
        chunk_dur = min(_CHUNK_SECONDS, total - chunk_start)
        audio_path = _extract_audio_chunk(video_path, chunk_start, chunk_dur)
        if denoise:
            rms_before, rms_after, bands = _denoise_audio(audio_path)
            rms_pairs.append((rms_before, rms_after))
            for b, v in bands.items():
                band_accum[b].append(v)
        try:
            segments_iter, info = model.transcribe(
                audio_path,
                task=task,
                beam_size=beam_size,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
            )
            if chunk_start == 0:
                detected_lang = info.language

            for seg in segments_iter:
                text = seg.text.strip()
                if text:
                    all_segments.append({
                        "id":    seg_id,
                        "start": float(seg.start) + chunk_start,
                        "end":   float(seg.end) + chunk_start,
                        "text":  text,
                    })
                    seg_id += 1
                if on_progress and total > 0:
                    on_progress(min((chunk_start + float(seg.end)) / total, 1.0))
        finally:
            os.unlink(audio_path)

        gc.collect()
        chunk_start += chunk_dur

    reduction_pct: float | None = None
    band_reductions: dict[str, float] | None = None
    if rms_pairs:
        avg_before = sum(b for b, _ in rms_pairs) / len(rms_pairs)
        avg_after = sum(a for _, a in rms_pairs) / len(rms_pairs)
        if avg_before > 0:
            reduction_pct = (1.0 - avg_after / avg_before) * 100.0
        band_reductions = {
            b: sum(vals) / len(vals) for b, vals in band_accum.items() if vals
        }

    return all_segments, detected_lang, reduction_pct, band_reductions


def transcribe_video(
    video_path: str,
    model_size: str,
    on_progress: Callable[[float], None] | None = None,
    denoise: bool = False,
) -> tuple[list[Segment], str, float | None, dict[str, float] | None]:
    """
    Transcribe video audio in its original language.

    Returns:
        (segments, detected_language, denoise_reduction_pct, band_reductions)
        denoise_reduction_pct and band_reductions are None when denoise=False.
    """
    return _transcribe_chunked(video_path, "transcribe", model_size, on_progress, denoise)


def transcribe_to_english(
    video_path: str,
    model_size: str,
    on_progress: Callable[[float], None] | None = None,
    denoise: bool = False,
) -> list[Segment]:
    """
    Use Whisper's built-in translate task to produce English-language segments
    from a video in any source language.

    Used as stage 1 of the any-language → Hebrew pipeline.
    """
    segments, _, _, _ = _transcribe_chunked(video_path, "translate", model_size, on_progress, denoise)
    return segments
