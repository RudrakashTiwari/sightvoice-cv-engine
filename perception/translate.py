"""Offline multilingual output with MarianMT (pretrained, no API).

Models are loaded lazily and cached per language. Unsupported languages fall
back to the original English text instead of erroring.
"""
import torch
from transformers import MarianMTModel, MarianTokenizer
import config


class Translator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._cache = {}

    def _load(self, lang):
        name = config.TRANSLATION_MODELS.get(lang)
        if name is None:
            return None
        if lang not in self._cache:
            tok = MarianTokenizer.from_pretrained(name)
            model = MarianMTModel.from_pretrained(name).to(self.device).eval()
            self._cache[lang] = (tok, model)
        return self._cache[lang]

    def translate(self, text, lang):
        if lang == "en" or not text:
            return text
        pair = self._load(lang)
        if pair is None:
            return text      # language not configured -> keep English
        tok, model = pair
        batch = tok([text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            gen = model.generate(**batch, max_new_tokens=80)
        return tok.decode(gen[0], skip_special_tokens=True).strip()
