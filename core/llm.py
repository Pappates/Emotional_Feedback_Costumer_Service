"""Gemini API sarmalayıcısı: JSON çıktılı metin üretimi, ses anlama ve TTS.

Bir model aşırı yüklenirse (503/504/429) ya da kapatılmışsa (404) sıradaki modele geçilir.
Model listeleri config.TEXT_MODELS / config.TTS_MODELS içindedir.
"""
from __future__ import annotations

import io
import re
import time
import wave

from pydantic import BaseModel

from . import config

_client = None
_preferred: dict[str, str] = {}   # en son başarılı model (her tür için)
_cooldown: dict[str, float] = {}  # kotası dolan model -> tekrar denenebileceği zaman
COOLDOWN_SN = 600

RETRYABLE = ("429", "500", "503", "504", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE", "overloaded",
             "high demand", "timed out", "Timeout", "404", "NOT_FOUND")


def client():
    global _client
    if _client is None:
        from google import genai
        from google.genai import types
        _client = genai.Client(
            api_key=config.API_KEY,
            # SDK'nın kendi yeniden denemeleri kapalı: model değiştirme mantığı aşağıda (_with_fallback)
            http_options=types.HttpOptions(api_version="v1beta", timeout=config.REQUEST_TIMEOUT_MS,
                                           retry_options=types.HttpRetryOptions(attempts=1)),
        )
    return _client


def _with_fallback(kind: str, models: list[str], fn):
    """fn(model) çağırır; geçici hata ya da kullanım dışı modelde sıradaki modeli dener."""
    order = ([_preferred[kind]] if kind in _preferred else []) + [m for m in models if m != _preferred.get(kind)]
    last = RuntimeError(f"{kind}: kullanılabilir model yok (hepsinin kotası dolu)")
    for rnd in range(2):                       # tüm listeyi en fazla iki tur dene
        for m in order:
            if _cooldown.get(m, 0) > time.time():
                continue
            try:
                out = fn(m)
                _preferred[kind] = m
                return out
            except Exception as e:  # noqa: BLE001
                last = e
                msg = str(e)
                if not any(x in msg or x in type(e).__name__ for x in RETRYABLE):
                    raise
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "404" in msg:
                    _cooldown[m] = time.time() + COOLDOWN_SN   # kota dolu / model kapalı: bir süre atla
                print(f"[gemini] {kind} modeli {m} başarısız ({msg[:80]}), sıradaki deneniyor…", flush=True)
        if all(_cooldown.get(m, 0) > time.time() for m in order):
            break                              # hepsi kota dolu: beklemenin anlamı yok
        time.sleep(2 * (rnd + 1))
    raise last


def generate_json(system: str, prompt: str, schema: type[BaseModel], audio_wav: bytes | None = None,
                  temperature: float = 0.7) -> BaseModel:
    from google.genai import types

    parts = []
    if audio_wav:
        parts.append(types.Part.from_bytes(data=audio_wav, mime_type="audio/wav"))
    parts.append(types.Part.from_text(text=prompt))

    def call(model):
        resp = client().models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        if resp.parsed is not None:
            return resp.parsed
        return schema.model_validate_json(resp.text)

    return _with_fallback("metin", config.TEXT_MODELS, call)


def pcm_to_wav(pcm: bytes, rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def tts(text: str, style: str, voice: str) -> bytes:
    """Metni verilen tonlamayla seslendirir, WAV döner."""
    from google.genai import types

    def call(model):
        resp = client().models.generate_content(
            model=model,
            contents=f"{style}: {text}",
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                    )
                ),
            ),
        )
        part = resp.candidates[0].content.parts[0].inline_data
        rate = 24000
        m = re.search(r"rate=(\d+)", part.mime_type or "")
        if m:
            rate = int(m.group(1))
        return pcm_to_wav(part.data, rate)

    return _with_fallback("ses", config.TTS_MODELS, call)


def active_models():
    return {"metin": _preferred.get("metin", config.TEXT_MODELS[0]), "ses": _preferred.get("ses", config.TTS_MODELS[0])}
