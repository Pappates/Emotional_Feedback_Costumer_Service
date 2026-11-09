"""Ortam ayarları. Anahtar yoksa uygulama 'demo (mock)' modunda çalışır."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio"
MODEL_FILE = DATA_DIR / "global_model.json"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
# Virgülle ayrılmış model listeleri: ilki tercih edilir, hata olursa sıradakine geçilir.
TEXT_MODELS = [m.strip() for m in os.getenv(
    "GEMINI_TEXT_MODEL", "gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-flash-lite-latest,gemini-3.5-flash,gemini-3.8-flash").split(",") if m.strip()]
TTS_MODELS = [m.strip() for m in os.getenv(
    "GEMINI_TTS_MODEL", "gemini-3.8-flash-tts,gemini-3.8-flash-lite-tts,gemini-3.1-flash-tts-preview,gemini-2.5-flash-preview-tts").split(",") if m.strip()]
TEXT_MODEL, TTS_MODEL = TEXT_MODELS[0], TTS_MODELS[0]
REQUEST_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "30000"))
AGENT_VOICE = os.getenv("AGENT_VOICE", "Sulafat")

# Bir görüşmenin en fazla kaç tur (asistan + müşteri) süreceği
MAX_TURNS = int(os.getenv("MAX_TURNS", "9"))


def has_gemini() -> bool:
    return bool(API_KEY) and os.getenv("FORCE_MOCK", "") != "1"
