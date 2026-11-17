"""Chatterbox Multilingual (Resemble AI, MIT lisansı) ile Türkçe seslendirme — ONNX Runtime, GPU gerektirmez.

Piper'dan belirgin şekilde daha doğal; ayrıca 'exaggeration' (duygu yoğunluğu) ayarı var. Bedeli hız: 4 çekirdekli
test makinesinde cümle başına ~15 sn (gerçek zamanın ~2,4 katı). Bu yüzden isteğe bağlı: `.env` -> LOKAL_TTS=chatterbox.
Yalnızca asistanın sesi için kullanılır; müşteri sesleri hızlı Piper'da kalır.

Model kartındaki çıkarım betiğinden uyarlandı (onnx-community/chatterbox-multilingual-ONNX). İki düzeltme:
  * Metin Türkçe kurallarıyla küçük harfe çevrilip NFKD ile ayrıştırılıyor (orijinal ön işleme). Yapılmazsa
    büyük 'İ' gibi harfler modelin sözlüğü dışında kalıyor ve çıkarım hata veriyor.
  * Oturumlar ve konuşmacı koşulu bir kez yükleniyor (betik her cümlede yeniden yüklüyordu).
"""
from __future__ import annotations

import os
import threading
import unicodedata

import numpy as np

REPO = "onnx-community/chatterbox-multilingual-ONNX"
SR = 24000
START, STOP = 6561, 6562
LM_VARIANT = os.getenv("LOKAL_CHATTERBOX_LM", "language_model_q4")   # q4: ~350 MB, CPU'da en hızlısı
REF_VOICE = os.getenv("LOKAL_CHATTERBOX_SES", "")                      # boşsa modelin varsayılan sesi

_lock = threading.Lock()
_state: dict = {}


def _read_wav_any(path: str) -> tuple[np.ndarray, int]:
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32")
    return (x.mean(axis=1) if x.ndim > 1 else x), sr


def _load():
    if _state:
        return _state
    with _lock:
        if _state:
            return _state
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        def dl(f):
            return hf_hub_download(REPO, f)
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, (os.cpu_count() or 4))
        sessions = {}
        for base in ("speech_encoder", "embed_tokens", "conditional_decoder", LM_VARIANT):
            path = dl(f"onnx/{base}.onnx")
            dl(f"onnx/{base}.onnx_data")
            sessions[base] = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        ref, sr = _read_wav_any(REF_VOICE or dl("default_voice.wav"))
        if sr != SR:
            ref = np.interp(np.linspace(0, len(ref) - 1, int(len(ref) * SR / sr)), np.arange(len(ref)), ref)
        cond = sessions["speech_encoder"].run(None, {"audio_values": ref[None, :].astype(np.float32)})
        _state.update(sessions=sessions, tok=AutoTokenizer.from_pretrained(REPO), cond=cond,
                      past_names=[i.name for i in sessions[LM_VARIANT].get_inputs() if i.name.startswith("past_key_values")])
    return _state


def preprocess(text: str) -> str:
    return unicodedata.normalize("NFKD", text.replace("I", "ı").replace("İ", "i").lower())


def synthesize(text: str, exaggeration: float = 0.5, max_new_tokens: int = 500) -> tuple[np.ndarray, int]:
    st = _load()
    S, tok = st["sessions"], st["tok"]
    lm = S[LM_VARIANT]
    ids = tok(f"[tr]{preprocess(text)}", return_tensors="np")["input_ids"].astype(np.int64)
    pos = np.where(ids >= START, 0, np.arange(ids.shape[1])[None, :] - 1).astype(np.int64)
    feed = {"input_ids": ids, "position_ids": pos, "exaggeration": np.array([exaggeration], np.float32)}
    cond_emb, prompt_token, xvec, pfeat = st["cond"]
    gen = np.array([[START]], dtype=np.int64)
    past, mask = None, None
    for i in range(max_new_tokens):
        emb = S["embed_tokens"].run(None, feed)[0]
        if i == 0:
            emb = np.concatenate((cond_emb, emb), axis=1)
            past = {n: np.zeros([1, 16, 0, 64], np.float32) for n in st["past_names"]}
            mask = np.ones((1, emb.shape[1]), np.int64)
        logits, *present = lm.run(None, dict(inputs_embeds=emb, attention_mask=mask, **past))
        lg = logits[:, -1, :].astype(np.float32)
        sc = np.take_along_axis(lg, gen, axis=1)                       # tekrar cezası 1,2
        np.put_along_axis(lg, gen, np.where(sc < 0, sc * 1.2, sc / 1.2), axis=1)
        nxt = np.argmax(lg, axis=-1, keepdims=True).astype(np.int64)
        gen = np.concatenate((gen, nxt), axis=-1)
        if nxt.item() == STOP:
            break
        feed["input_ids"] = nxt
        feed["position_ids"] = np.full((1, 1), i + 1, np.int64)
        mask = np.concatenate([mask, np.ones((1, 1), np.int64)], axis=1)
        for j, k in enumerate(past):
            past[k] = present[j]
    toks = np.concatenate([prompt_token, gen[:, 1:-1]], axis=1)
    wav = S["conditional_decoder"].run(None, {"speech_tokens": toks, "speaker_embeddings": xvec,
                                              "speaker_features": pfeat})[0]
    return np.squeeze(wav).astype(np.float32), SR


def warmup():
    _load()
    synthesize("merhaba.", max_new_tokens=40)
