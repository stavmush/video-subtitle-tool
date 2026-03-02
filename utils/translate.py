"""
Offline translation using Helsinki-NLP MarianMT models from HuggingFace.

Supported target languages:
  - English ("en"): handled upstream via Whisper's translate task, or via opus-mt-he-en
  - Hebrew ("he"): uses opus-mt-en-he (~300MB, downloaded on first use)

For any-language → Hebrew, the caller must supply whisper_segments_english
(English segments produced by transcribe_to_english()).
"""

import streamlit as st
import torch
from transformers import MarianMTModel, MarianTokenizer
from typing import Any

Segment = dict[str, Any]

HELSINKI_EN_HE = "Helsinki-NLP/opus-mt-en-he"
HELSINKI_HE_EN = "Helsinki-NLP/opus-mt-tc-big-he-en"


@st.cache_resource(show_spinner=False)
def _load_marian_model(model_name: str) -> tuple[MarianTokenizer, MarianMTModel]:
    """Download (first run) and cache a Helsinki-NLP MarianMT model."""
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def _translate_texts(texts: list[str], model_name: str, batch_size: int = 16) -> list[str]:
    """
    Translate a list of strings using a MarianMT model, processing in batches
    to avoid out-of-memory errors on long videos.
    """
    tokenizer, model = _load_marian_model(model_name)
    results: list[str] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            translated_tokens = model.generate(**inputs)
        decoded = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        results.extend(decoded)
    return results


def translate_segments(
    segments: list[Segment],
    source_lang: str,
    target_lang: str,
    whisper_segments_english: list[Segment] | None = None,
) -> list[Segment]:
    """
    Translate segments to the target language.

    Routing logic:
      - target=en:              return segments as-is (Whisper already produced English)
      - target=he, source=en:  translate directly with opus-mt-en-he
      - target=he, source!=en: use whisper_segments_english (English stage already done)
        and translate with opus-mt-en-he

    Raises:
        ValueError: if target_lang is unsupported or whisper_segments_english is
                    missing when required
    """
    if target_lang == "en":
        return segments

    if target_lang == "he":
        english_segs = whisper_segments_english if whisper_segments_english is not None else segments
        if not english_segs:
            raise ValueError("No English segments to translate to Hebrew.")
        source_texts = [seg["text"] for seg in english_segs]
        translated_texts = _translate_texts(source_texts, HELSINKI_EN_HE)
        return [
            {**seg, "text": translated_texts[i]}
            for i, seg in enumerate(english_segs)
        ]

    raise ValueError(f"Unsupported target language: '{target_lang}'. Supported: 'en', 'he'.")


def translate_text_list(texts: list[str], src: str, tgt: str) -> list[str]:
    """
    Translate a plain list of strings between supported language pairs.
    Currently supported: 'en'↔'he'.

    Raises:
        ValueError: if the language pair is not supported
    """
    if src == tgt:
        return texts
    if src == "en" and tgt == "he":
        return _translate_texts(texts, HELSINKI_EN_HE)
    if src == "he" and tgt == "en":
        return _translate_texts(texts, HELSINKI_HE_EN)
    raise ValueError(f"Unsupported language pair: '{src}' → '{tgt}'.")
