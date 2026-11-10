"""Ortam ayarları. Anahtar yoksa uygulama 'demo (mock)' modunda çalışır."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio"
MODEL_FILE = DATA_DIR / "global_model.json"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def _load_dotenv():
    """Proje kökündeki .env dosyasını (varsa) okur. Ortamda zaten tanımlı değerleri ezmez."""
    f = ROOT / ".env"
    if not f.exists():
        return
    for line in f.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

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


# Model sağlayıcısı: "gemini" (bulut) ya da "lokal" (Ollama + Hugging Face modelleri, kota yok, GPU gerekmez)
PROVIDER = os.getenv("SAGLAYICI", "gemini").strip().lower()


def default_mode() -> str:
    """Görüşmelerin varsayılan modu: lokal | gemini | mock (demo)."""
    if PROVIDER == "lokal":
        return "lokal"
    return "gemini" if has_gemini() else "mock"


def has_gemini() -> bool:
    return bool(API_KEY) and os.getenv("FORCE_MOCK", "") != "1"
