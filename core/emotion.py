"""Müşterinin sesinden ve sözlerinden duygu analizi.

İki kanal birleştirilir:
  1. Akustik özellikler (yerel, numpy, GPU gerekmez): ses enerjisi, perde (pitch),
     perde dalgalanması, konuşma hızı. Bunlar uyarılmışlığı (arousal) iyi yakalar.
  2. Gemini ses anlama: sesi doğrudan dinleyip hem metne çevirir hem de tonlamaya ve
     söyleme bakarak duygu, valens (olumlu/olumsuz) ve churn riskini çıkarır.
"""
from __future__ import annotations

import io
import wave
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from . import llm

EMOTIONS = ["öfke", "hayal kırıklığı", "endişe", "nötr", "rahatlama", "memnuniyet"]


class GeminiEmotion(BaseModel):
    transkript: str = Field(description="Müşterinin söylediklerinin Türkçe dökümü")
    duygu: Literal["öfke", "hayal kırıklığı", "endişe", "nötr", "rahatlama", "memnuniyet"]
    valens: float = Field(description="-1 (çok olumsuz) ile +1 (çok olumlu) arası")
    uyarilma: float = Field(description="0 (çok sakin) ile 1 (çok gergin/yüksek enerjili) arası")
    churn_riski: float = Field(description="0-1 arası: müşterinin operatörü bırakma olasılığı")
    konu: Literal["internet_kesintisi", "hiz_sorunu", "fatura_itirazi", "nakil", "diger"]
    ses_ipuclari: str = Field(description="Ses tonuna dair kısa gözlem (hız, yükseklik, gerginlik)")
    metin_ipuclari: str = Field(description="Sözlerdeki duygu ipuçları (kısa)")


SYSTEM = (
    "Sen bir çağrı merkezi için duygu analizi uzmanısın. Müşterinin konuşmasını hem SÖZLERİNE hem de "
    "SES TONUNA (hız, yükseklik, gerginlik, iç çekme) bakarak analiz edersin. Değerlendirmeyi yalnızca "
    "müşterinin bu son konuşmasına göre yap. Tüm açıklamaları Türkçe yaz. Olumlu bir gelişmede valensi cömertçe yükselt, olumsuzda düşür."
)


def acoustic_features(wav_bytes: bytes, transcript: str = "") -> dict:
    with wave.open(io.BytesIO(wav_bytes)) as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if x.size == 0:
        return {}
    dur = x.size / sr
    frame = int(0.04 * sr)
    hop = frame // 2
    n = max(1, (x.size - frame) // hop)
    rms = np.array([np.sqrt(np.mean(x[i * hop:i * hop + frame] ** 2) + 1e-12) for i in range(n)])
    voiced = rms >= max(0.02, np.percentile(rms, 40))
    pitches = []
    lo, hi = int(sr / 400), int(sr / 70)  # 70-400 Hz
    for i in np.where(voiced)[0][::2]:
        seg = x[i * hop:i * hop + frame]
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, mode="full")[frame - 1:]
        if ac[0] <= 0:
            continue
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] / ac[0] > 0.35:
            pitches.append(sr / lag)
    pitches = np.array(pitches)
    energy_db = float(20 * np.log10(np.mean(rms[voiced]) + 1e-9)) if voiced.any() else -60.0
    words = len(transcript.split())
    speech_time = voiced.sum() * hop / sr
    feats = {
        "sure_sn": round(dur, 2),
        "enerji_db": round(energy_db, 1),
        "perde_hz": round(float(np.median(pitches)), 0) if pitches.size else None,
        "perde_dalgalanma": round(float(np.std(pitches) / (np.mean(pitches) + 1e-9)), 3) if pitches.size > 3 else None,
        "konusma_hizi": round(float(words / speech_time), 2) if words and speech_time > 0.3 else None,
    }
    # Basit akustik uyarılmışlık skoru (0-1): yüksek enerji, hızlı konuşma ve dalgalı perde = gerginlik
    score = 0.0
    score += np.clip((energy_db + 30) / 18, 0, 1) * 0.4
    if feats["konusma_hizi"]:
        score += np.clip((feats["konusma_hizi"] - 1.8) / 2.0, 0, 1) * 0.3
    if feats["perde_dalgalanma"]:
        score += np.clip((feats["perde_dalgalanma"] - 0.08) / 0.2, 0, 1) * 0.3
    feats["akustik_uyarilma"] = round(float(score), 3)
    return feats


def analyze_audio(wav_bytes: bytes) -> dict:
    if llm.config.PROVIDER == "lokal":
        from . import local
        text = local.transcribe(wav_bytes)
        feats = acoustic_features(wav_bytes, text)
        out = local.analyze(text, wav_bytes, feats)
        out["akustik"] = feats
        out["uyarilma_gemini"] = None      # lokal modda Gemini kanalı yok; füzyon local.analyze içinde
        out["kaynak"] = "lokal: ses + metin"
        return out
    r: GeminiEmotion = llm.generate_json(
        SYSTEM, "Bu ses kaydındaki müşteriyi analiz et.", GeminiEmotion, audio_wav=wav_bytes, temperature=0.2
    )
    out = r.model_dump()
    feats = acoustic_features(wav_bytes, out["transkript"])
    out["akustik"] = feats
    # Füzyon: valens Gemini'den (söz + ton); uyarılmışlık iki kanalın ağırlıklı ortalaması
    out["uyarilma_gemini"] = out["uyarilma"]
    if feats.get("akustik_uyarilma") is not None:
        out["uyarilma"] = round(0.6 * out["uyarilma"] + 0.4 * feats["akustik_uyarilma"], 3)
    out["kaynak"] = "ses + metin"
    return out


def analyze_text(text: str) -> dict:
    if llm.config.PROVIDER == "lokal":
        from . import local
        out = local.analyze(text, None, None)
        out["akustik"] = {}
        out["kaynak"] = "lokal: yalnızca metin"
        return out
    r: GeminiEmotion = llm.generate_json(
        SYSTEM, f"Müşterinin söyledikleri (yalnızca metin, ses yok):\n\"{text}\"", GeminiEmotion, temperature=0.2
    )
    out = r.model_dump()
    out["transkript"] = text
    out["akustik"] = {}
    out["kaynak"] = "yalnızca metin"
    return out


def mock_analysis(true_mood: float, text: str, issue: str, rng: np.random.Generator) -> dict:
    """API anahtarı yokken: gerçek ruh haline gürültü ekleyerek 'kusurlu bir algı' üretir."""
    v = float(np.clip(true_mood + rng.normal(0, 0.2), -1, 1))   # gerçek analizdeki kadar gürültülü
    ar = float(np.clip(abs(min(true_mood, 0)) * 0.9 + rng.normal(0, 0.08), 0, 1))
    if v < -0.6:
        emo = "öfke"
    elif v < -0.25:
        emo = "hayal kırıklığı"
    elif v < 0.1:
        emo = "nötr"
    elif v < 0.45:
        emo = "rahatlama"
    else:
        emo = "memnuniyet"
    return {
        "transkript": text, "duygu": emo, "valens": round(v, 3), "uyarilma": round(ar, 3),
        "churn_riski": round(float(1 / (1 + np.exp(4 * (v + 0.35)))), 3), "konu": issue,
        "ses_ipuclari": "(demo modu: ses analizi simüle edildi)", "metin_ipuclari": "",
        "akustik": {}, "kaynak": "demo (simülasyon)",
    }
