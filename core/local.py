"""Lokal sağlayıcı (Aşama 0): bulut ve kota olmadan, GPU gerektirmeden çalışan hazır açık modeller.

  Diyalog (asistan + müşteri) : Ollama üzerinden lokal LLM (varsayılan qwen2.5:3b), JSON şemalı çıktı
  Ses -> metin                : faster-whisper (CPU, int8)
  Sesten duygu                : wav2vec2 tabanlı konuşma duygu modeli (varsayılan superb/wav2vec2-base-superb-er)
  Metinden duygu              : Türkçe BERT duygu (sentiment) modeli (varsayılan savasy/bert-base-turkish-sentiment-cased)
  Churn niyeti ve konu        : anahtar kelime kuralları (Aşama 2'de BERTurk çok görevli modelle değiştirilecek)
  Seslendirme                 : Piper (Türkçe sesler), tonlama hız/ses yüksekliği/perde ile

Modeller ilk kullanımda Hugging Face'ten indirilir ve önbelleğe alınır (sonraki açılışlar internetsiz çalışır).
Bütün model adları .env / ortam değişkenleriyle değiştirilebilir (bkz. README).
"""
from __future__ import annotations

import io
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
import wave

import numpy as np
from pydantic import BaseModel

# Hugging Face'in "Xet" indirme protokolü ayrı sunuculara gidiyor; kurumsal proxy'lerde sorun çıkarabildiği için
# klasik HTTP indirmeyi kullan (kullanıcı isterse HF_HUB_DISABLE_XET=0 ile açabilir).
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
# Testte qwen2.5:3b Türkçede rolleri karıştırıp yer tutucular ("[adres]") yazdı; gemma3:4b çok daha tutarlı.
LLM_MODEL = os.getenv("LOKAL_LLM", "gemma3:4b")
WHISPER_MODEL = os.getenv("LOKAL_WHISPER", "small")
SER_MODEL = os.getenv("LOKAL_SES_DUYGU", "superb/wav2vec2-base-superb-er")
TEXT_MODEL = os.getenv("LOKAL_METIN_DUYGU", "savasy/bert-base-turkish-sentiment-cased")
PIPER_REPO = "rhasspy/piper-voices"
# Piper Türkçe sesleri: rol -> ses adı (tr_TR-<ad>-medium)
# Piper'da şu an tek Türkçe ses var (tr_TR-dfki-medium). Konuşmacıları ayırmak için aynı sesi farklı perdelerde
# kullanıyoruz. Farklı/klonlanmış sesler için sonraki aşamada XTTS veya Chatterbox gibi bir model gerekir.
AGENT_PIPER_VOICE = os.getenv("LOKAL_ASISTAN_SESI", "dfki")
AGENT_PITCH = float(os.getenv("LOKAL_ASISTAN_PERDE", "1.0"))
CUSTOMER_PIPER_VOICES = {  # Gemini ses adı (catalog.py'daki persona sesi) -> (Piper sesi, perde çarpanı)
    "Fenrir": ("dfki", 0.86),   # Ahmet Bey: kalın
    "Puck": ("dfki", 0.94),     # Mert
    "Leda": ("dfki", 1.24),     # Zeynep Hanım: ince
    "Kore": ("dfki", 1.16),     # Selin Hanım
}

_lock = threading.Lock()
_models: dict = {}
STATUS: dict[str, str] = {}   # bileşen -> "hazır" / "yükleniyor" / hata mesajı (panelde gösterilir)


# ============================================================ yardımcılar
def wav_to_float(wav_bytes: bytes, target_sr: int = 16000) -> np.ndarray:
    with wave.open(io.BytesIO(wav_bytes)) as w:
        sr, ch = w.getframerate(), w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != target_sr and x.size:
        n = int(round(x.size * target_sr / sr))
        x = np.interp(np.linspace(0, x.size - 1, n), np.arange(x.size), x).astype(np.float32)
    return x


def float_to_wav(x: np.ndarray, sr: int) -> bytes:
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return buf.getvalue()


def _tr_lower(s: str) -> str:
    return s.replace("I", "ı").replace("İ", "i").lower()


def _get(name, loader):
    """Modeli bir kez yükler (iş parçacığı güvenli) ve durumunu STATUS'a yazar."""
    if name in _models:
        return _models[name]
    with _lock:
        if name not in _models:
            STATUS[name] = "yükleniyor"
            t = time.time()
            try:
                _models[name] = loader()
                STATUS[name] = f"hazır ({time.time() - t:.0f} sn)"
            except Exception as e:  # noqa: BLE001
                STATUS[name] = f"HATA: {e}"
                raise
    return _models[name]


# ============================================================ LLM (Ollama)
def _ollama(path: str, payload: dict | None = None, timeout: float = 300):
    req = urllib.request.Request(
        OLLAMA_URL + path, data=None if payload is None else json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="GET" if payload is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def check_ollama() -> str:
    try:
        tags = _ollama("/api/tags", timeout=5)
    except Exception:  # noqa: BLE001
        STATUS["llm"] = (f"HATA: Ollama çalışmıyor ({OLLAMA_URL}). https://ollama.com/download adresinden kurun, "
                         f"sonra terminalde: ollama pull {LLM_MODEL}")
        return STATUS["llm"]
    names = {m.get("name", "") for m in tags.get("models", [])}
    want = LLM_MODEL if ":" in LLM_MODEL else LLM_MODEL + ":latest"
    if want not in names and LLM_MODEL not in names:
        STATUS["llm"] = f"HATA: Ollama'da '{LLM_MODEL}' modeli yok. Terminalde: ollama pull {LLM_MODEL}"
    else:
        STATUS["llm"] = f"hazır ({LLM_MODEL})"
    return STATUS["llm"]


def _merge_roles(history: list[dict], prompt: str) -> list[dict]:
    """Ardışık aynı rollü mesajları birleştirir ve ilk mesajın 'user' olmasını sağlar
    (Gemma gibi bazı sohbet şablonları rollerin sırayla gelmesini ister)."""
    msgs: list[dict] = []
    for m in [*history, {"role": "user", "content": prompt}]:
        if msgs and msgs[-1]["role"] == m["role"]:
            msgs[-1]["content"] += "\n\n" + m["content"]
        else:
            msgs.append({"role": m["role"], "content": m["content"]})
    if msgs and msgs[0]["role"] != "user":
        msgs.insert(0, {"role": "user", "content": "(Telefon görüşmesi başladı.)"})
    return msgs


def chat_json(system: str, prompt: str, schema: type[BaseModel], temperature: float = 0.7,
              history: list[dict] | None = None) -> BaseModel:
    """history: konuşmanın bu ajanın bakış açısından rol tabanlı geçmişi
    ([{"role": "user"|"assistant", "content": ...}]). Küçük modeller tüm konuşmayı tek metin olarak alınca
    rolleri karıştırıp karşı tarafın cümlelerini tekrarlıyor; gerçek sohbet rolleri bunu büyük ölçüde önlüyor."""
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "system", "content": system}, *_merge_roles(history or [], prompt)],
        "format": schema.model_json_schema(),      # Ollama yapılandırılmış çıktı: JSON şemasına uyar
        "stream": False,
        "keep_alive": "30m",
        "options": {"temperature": temperature, "num_ctx": 4096, "top_p": 0.9,
                    "repeat_penalty": 1.15, "num_predict": 260},
    }
    try:
        out = _ollama("/api/chat", payload)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "ignore")[:200]
        if e.code == 404:
            raise RuntimeError(f"Ollama'da '{LLM_MODEL}' modeli yok. Terminalde: ollama pull {LLM_MODEL}") from e
        raise RuntimeError(f"Ollama hatası {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama'ya bağlanılamadı ({OLLAMA_URL}). Ollama açık mı? ({e.reason})") from e
    STATUS["llm"] = f"hazır ({LLM_MODEL})"
    # strict=False: küçük modeller bazen metnin içine çıplak satır sonu koyuyor; JSON yine de okunabilsin
    return schema.model_validate(json.loads(out["message"]["content"], strict=False))


# ============================================================ ses -> metin
def transcribe(wav_bytes: bytes) -> str:
    def load():
        from faster_whisper import WhisperModel
        return WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    model = _get("whisper", load)
    segs, _ = model.transcribe(wav_to_float(wav_bytes), language="tr", beam_size=1, condition_on_previous_text=False)
    return " ".join(s.text.strip() for s in segs).strip()


# ============================================================ sesten duygu
# Model etiketleri -> (valens, uyarılma). superb-er: neu / hap / ang / sad
SER_MAP = {
    "neu": (0.0, 0.35), "neutral": (0.0, 0.35),
    "hap": (0.7, 0.65), "happy": (0.7, 0.65), "joy": (0.7, 0.65),
    "ang": (-0.75, 0.9), "angry": (-0.75, 0.9), "anger": (-0.75, 0.9),
    "sad": (-0.6, 0.2), "sadness": (-0.6, 0.2),
    "fear": (-0.6, 0.75), "fearful": (-0.6, 0.75), "disgust": (-0.65, 0.6), "surprise": (0.2, 0.8),
    "calm": (0.2, 0.15),
}


def speech_emotion(wav_bytes: bytes) -> dict:
    def load():
        from transformers import pipeline
        return pipeline("audio-classification", model=SER_MODEL, device=-1)
    clf = _get("ses_duygu", load)
    res = clf({"raw": wav_to_float(wav_bytes), "sampling_rate": 16000}, top_k=None)
    probs = {r["label"].lower(): float(r["score"]) for r in res}
    tot = sum(p for k, p in probs.items() if k in SER_MAP) or 1.0
    v = sum(SER_MAP[k][0] * p for k, p in probs.items() if k in SER_MAP) / tot
    a = sum(SER_MAP[k][1] * p for k, p in probs.items() if k in SER_MAP) / tot
    return {"olasiliklar": {k: round(p, 3) for k, p in probs.items()}, "valens": round(v, 3), "uyarilma": round(a, 3)}


# ============================================================ metinden duygu, churn, konu
SENT_SCALE = float(os.getenv("LOKAL_METIN_OLCEK", "0.6"))
# İngilizce oyunculu konuşmayla eğitilmiş ses duygu modeli, Türkçe öfkeli müşteriyi %80-95 "mutlu" sınıflıyor ve
# valensi yukarı çekiyordu (gerçek kullanıcı raporunda ortalama sapma +0,20). Valensteki payı %40'tan %10'a indirildi;
# uyarılmışlıkta (sesin gerginliği) kullanılmaya devam ediyor. Türkçe veriyle eğitilince artırılabilir.
SER_VALENCE_WEIGHT = float(os.getenv("LOKAL_SES_VALENS_AGIRLIK", "0.1"))


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?…])\s+|\s+(?=(?:ama|ancak|fakat|yalnız) )", text.strip())
    return [p for p in parts if len(p) > 2] or [text]


def text_sentiment(text: str) -> dict:
    """Hazır ikili (olumlu/olumsuz) model bütün mesaja bakınca en güçlü kelimeye göre ±1'e doyuyor; karışık
    mesajlarda ('Tamam, düzeldi. Ama bu kadar bekletmeyin!') çok yanılıyordu. Bu yüzden mesaj cümlelere ve
    'ama/ancak/fakat' öncesine bölünüp her parça ayrı puanlanır, ortalanır ve ölçek daraltılır (×0,6).
    Test (16 cümle, gerçek ruh haline göre): MAE 0,63 -> 0,30, korelasyon 0,57 -> 0,59."""
    def load():
        from transformers import pipeline
        return pipeline("text-classification", model=TEXT_MODEL, device=-1)
    clf = _get("metin_duygu", load)
    parts = _split_sentences(text[:1000])
    vals, probs_all = [], []
    for part in parts:
        probs = {r["label"].lower(): float(r["score"]) for r in clf(part, top_k=None)}
        pos = sum(p for k, p in probs.items() if k.startswith(("pos", "olumlu")) or k in ("label_1",))
        neg = sum(p for k, p in probs.items() if k.startswith(("neg", "olumsuz")) or k in ("label_0",))
        vals.append(pos - neg)
        probs_all.append({"parca": part[:80], "valens": round(pos - neg, 3)})
    return {"parcalar": probs_all, "valens": round(float(np.mean(vals)) * SENT_SCALE, 3)}


CHURN_PATTERNS = [r"iptal", r"başka (bir )?operat", r"operatör(ümü)? değiştir", r"geçece[ğk]", r"geçiyorum", r"ayrıl",
                  r"aboneli[ğk]i?(mi)? (iptal|sonlandır|bitir)", r"bir daha (aramayın|sizinle)", r"kapatıyorum", r"yeter artık"]
TOPICS = {
    "internet_kesintisi": [r"internet(im)? (yok|gitti|kesik|çalışmıyor)", r"kesinti", r"kesildi", r"bağlantı yok", r"\bpos\b", r"modem"],
    "hiz_sorunu": [r"yavaş", r"\bhız", r"mbps", r"kopuyor", r"donuyor", r"ping"],
    "fatura_itirazi": [r"fatura", r"ücret", r"tutar", r"\blira\b", r"\btl\b", r"iade", r"ödeme", r"para"],
    "nakil": [r"taşın", r"nakil", r"adres", r"yeni ev", r"ofis(imi)? taşı"],
}
POSITIVE = [r"teşekkür", r"sağ ?ol", r"memnun", r"sevindim", r"harika", r"süper", r"rahatladım", r"çok iyi"]
NEGATIVE = [r"rezalet", r"saçma", r"bıktım", r"yeter", r"sinir", r"mağdur", r"kabul edilemez", r"utanç", r"berbat"]


def text_cues(text: str) -> dict:
    t = _tr_lower(text)
    churn_hits = [p for p in CHURN_PATTERNS if re.search(p, t)]
    scores = {k: sum(1 for p in pats if re.search(p, t)) for k, pats in TOPICS.items()}
    konu = max(scores, key=scores.get) if max(scores.values()) > 0 else "diger"
    pos = [p.strip("\\b") for p in POSITIVE if re.search(p, t)]
    neg = [p.strip("\\b") for p in NEGATIVE if re.search(p, t)]
    return {"churn_ifadeleri": churn_hits, "konu": konu, "olumlu": pos, "olumsuz": neg}


# ============================================================ birleşik analiz (Gemini analizinin lokal karşılığı)
def _label(v: float, a: float) -> str:
    if v < -0.45 and a >= 0.55:
        return "öfke"
    if v < -0.3:
        return "hayal kırıklığı"
    if v < -0.05 and a >= 0.55:
        return "endişe"
    if v < 0.15:
        return "nötr"
    if v < 0.5:
        return "rahatlama"
    return "memnuniyet"


def _voice_cues(feats: dict, ser: dict | None) -> str:
    parts = []
    if feats.get("enerji_db") is not None:
        parts.append("yüksek ses" if feats["enerji_db"] > -16 else "alçak ses" if feats["enerji_db"] < -26 else "orta ses")
    if feats.get("konusma_hizi"):
        parts.append("hızlı konuşma" if feats["konusma_hizi"] > 3.2 else "yavaş konuşma" if feats["konusma_hizi"] < 2.0 else "normal hız")
    if feats.get("perde_dalgalanma") is not None:
        parts.append("dalgalı/gergin perde" if feats["perde_dalgalanma"] > 0.18 else "düz perde")
    if ser:
        top = max(ser["olasiliklar"], key=ser["olasiliklar"].get)
        parts.append(f"ses modeli: {top} (%{ser['olasiliklar'][top] * 100:.0f})")
    return ", ".join(parts) or "–"


def analyze(text: str, wav_bytes: bytes | None, acoustic: dict | None) -> dict:
    """Gemini analiziyle aynı alanları üretir. Füzyon:
       valens    = 0,9 × metin duygu modeli + 0,1 × ses duygu modeli (ses yoksa yalnızca metin) + kelime ipuçları
       uyarılma  = 0,5 × ses duygu modeli + 0,5 × akustik gerginlik
       churn     = sigmoid(−3·(valens + 0,2)); açık ayrılma ifadesi varsa en az 0,85"""
    tx = text_sentiment(text)
    cues = text_cues(text)
    ser = speech_emotion(wav_bytes) if wav_bytes else None
    v_text = tx["valens"] + 0.08 * len(cues["olumlu"]) - 0.08 * len(cues["olumsuz"])
    v = (1 - SER_VALENCE_WEIGHT) * v_text + SER_VALENCE_WEIGHT * ser["valens"] if ser else v_text
    v = float(np.clip(v, -1, 1))
    ac = (acoustic or {}).get("akustik_uyarilma")
    if ser and ac is not None:
        a = 0.5 * ser["uyarilma"] + 0.5 * ac
    elif ser:
        a = ser["uyarilma"]
    elif ac is not None:
        a = ac
    else:
        a = float(np.clip(0.35 + 0.5 * max(0.0, -v), 0, 1))
    churn = float(1 / (1 + np.exp(3 * (v + 0.2))))
    if cues["churn_ifadeleri"]:
        churn = max(churn, 0.85)
    metin_ip = []
    if cues["churn_ifadeleri"]:
        metin_ip.append("ayrılma/iptal ifadesi")
    if cues["olumsuz"]:
        metin_ip.append("olumsuz: " + ", ".join(cues["olumsuz"]))
    if cues["olumlu"]:
        metin_ip.append("olumlu: " + ", ".join(cues["olumlu"]))
    metin_ip.append(f"metin modeli valensi {tx['valens']:+.2f}")
    return {
        "transkript": text, "duygu": _label(v, a), "valens": round(v, 3), "uyarilma": round(float(a), 3),
        "churn_riski": round(churn, 3), "konu": cues["konu"],
        "ses_ipuclari": _voice_cues(acoustic or {}, ser), "metin_ipuclari": "; ".join(metin_ip),
        "lokal_ayrinti": {"metin_modeli": tx, "ses_modeli": ser, "kelime_ipuclari": cues,
                          "valens_metin": round(v_text, 3)},
    }


# ============================================================ seslendirme (Piper)
def _piper_voice(name: str):
    def load():
        from huggingface_hub import hf_hub_download
        from piper import PiperVoice
        base = f"tr/tr_TR/{name}/medium/tr_TR-{name}-medium.onnx"
        onnx = hf_hub_download(PIPER_REPO, base)
        hf_hub_download(PIPER_REPO, base + ".json")
        return PiperVoice.load(onnx)
    return _get(f"piper_{name}", load)


def style_params(style: str) -> dict:
    """Gemini tarzı tonlama talimatını Piper parametrelerine çevirir.
    length_scale > 1 yavaşlatır; volume ses yüksekliği; pitch perde çarpanı."""
    s = style.lower()
    table = [
        ("angry", 0.86, 1.0, 1.04), ("irritated", 0.92, 0.95, 1.02), ("tired", 1.05, 0.75, 0.98),
        ("calmer", 1.08, 0.8, 1.0), ("happy", 0.98, 0.9, 1.05), ("warm, slow", 1.18, 0.8, 0.98),
        ("upbeat", 0.9, 0.95, 1.06), ("professional", 0.98, 0.9, 1.0), ("friendly", 1.0, 0.85, 1.03),
    ]
    for key, ls, vol, pitch in table:
        if key in s:
            return {"length_scale": ls, "volume": vol, "pitch": pitch}
    return {"length_scale": 1.0, "volume": 0.9, "pitch": 1.0}


def _synth(voice, text: str, length_scale: float) -> tuple[np.ndarray, int]:
    """piper-tts sürümleri arasında API farkı var: ikisini de destekle."""
    sr = voice.config.sample_rate
    try:  # piper-tts >= 1.3
        from piper import SynthesisConfig
        chunks = voice.synthesize(text, syn_config=SynthesisConfig(length_scale=length_scale))
        arrs = [c.audio_float_array for c in chunks]
        return (np.concatenate(arrs) if arrs else np.zeros(1, np.float32)), sr
    except ImportError:  # eski sürüm (1.2): synthesize(text, wav_file, length_scale=...)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            voice.synthesize(text, w, length_scale=length_scale)
        return wav_to_float(buf.getvalue(), sr), sr


def tts(text: str, style: str, voice_name: str) -> bytes:
    from . import config
    if voice_name == config.AGENT_VOICE:
        piper_name, base_pitch = AGENT_PIPER_VOICE, AGENT_PITCH
    else:
        piper_name, base_pitch = CUSTOMER_PIPER_VOICES.get(voice_name, ("dfki", 0.9))
    p = style_params(style)
    pitch = base_pitch * p["pitch"]
    voice = _piper_voice(piper_name)
    # Perde kaydırma: sesi `pitch` kadar yavaş üretip örnekleme hızını `pitch` kadar artırınca süre aynı kalır, perde yükselir
    x, sr = _synth(voice, text, p["length_scale"] * pitch)
    x = x / (np.max(np.abs(x)) + 1e-6) * 0.9 * p["volume"]
    return float_to_wav(x.astype(np.float32), int(round(sr * pitch)))


# ============================================================ ısınma
def warmup():
    """Uygulama açılırken modelleri arka planda yükler (ilk görüşme beklemesin)."""
    def run():
        check_ollama()
        for name, fn in (("whisper", lambda: transcribe(float_to_wav(np.zeros(16000, np.float32), 16000))),
                         ("metin_duygu", lambda: text_sentiment("merhaba")),
                         ("ses_duygu", lambda: speech_emotion(float_to_wav(np.zeros(16000, np.float32), 16000))),
                         ("piper", lambda: [_piper_voice(v) for v in {AGENT_PIPER_VOICE, *[x[0] for x in CUSTOMER_PIPER_VOICES.values()]}])):
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                STATUS.setdefault(name, f"HATA: {e}")
                print(f"[lokal] {name} yüklenemedi: {e}", flush=True)
        print("[lokal] model durumu:", json.dumps(STATUS, ensure_ascii=False), flush=True)
    threading.Thread(target=run, daemon=True).start()
