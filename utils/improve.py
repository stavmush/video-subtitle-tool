"""Grammar and fluency improvement for English subtitles.

Uses vennify/t5-base-grammar-correction (T5-base, ~240 MB).
Downloaded on first use and cached locally by HuggingFace.
Requires torch >= 2.4 (transformers 5.x dependency).
"""

from __future__ import annotations

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

_tokenizer = None
_model = None
_MODEL = "vennify/t5-base-grammar-correction"


def _load_model():
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = T5Tokenizer.from_pretrained(_MODEL)
        _model = T5ForConditionalGeneration.from_pretrained(_MODEL)
        _model.eval()
    return _tokenizer, _model


def improve_text_list(texts: list[str]) -> list[str]:
    """Return grammar/fluency-improved versions of English subtitle texts."""
    if not texts:
        return texts
    tokenizer, model = _load_model()
    improved = []
    for text in texts:
        inputs = tokenizer(f"grammar: {text}", return_tensors="pt", max_length=128, truncation=True)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=128)
        improved.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
    return improved
