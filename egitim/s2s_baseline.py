"""Aşama 2 baseline: eğitimden önce TR-Omni-E4B (sesten sese) ve emotion2vec+ base ne kadar iyi?

Eğitim yok; modeller hazır hâliyle ölçülür. Sonraki ince ayarların geçmesi gereken çıta budur.

Bölümler (her biri ayrı .jsonl dosyasına satır satır yazılır; Colab koparsa kaldığı yerden devam eder):
  e2v_algi    TurEV-DB dengeli örneklem: emotion2vec+ base hazır etiketi (4 duygu)
  omni_algi   Aynı örneklem: TR-Omni'ye sesle birlikte "duygu hangisi?" diye sorulur (yalnızca metin cevap)
  omni_ofke   ElevenLabs müşteri tepkileri (data/enjeksiyon_ses): TR-Omni'den 0-4 öfke puanı;
              mevcut hattın ölçtüğü öfkeyle (ses_ofke) sıra korelasyonu
  omni_sohbet Aynı tepkiler: TR-Omni sesli cevap verir. Süre, cevap metni, Türkçe yazım denetimi,
              cevabın sesinin emotion2vec etiketi (asistan sakin mi duyuluyor?). Ses dosyaları saklanır.

Kullanım (Colab, L4 veya A100; TR-Omni ~17 GB GPU belleği ister, T4 yetmez):
  python egitim/s2s_baseline.py --turev "<TurEV Sound Source>" --model <TR-Omni klasörü> \
      --musteri <enjeksiyon_ses klasörü> --cikti <çıktı klasörü> [--adet 40] [--bolum e2v_algi omni_algi ...]
Özet: <çıktı>/ozet.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CLASSES = ["Angry", "Calm", "Happy", "Sad"]
E2V_MAP = {"angry": 0, "neutral": 1, "happy": 2, "sad": 3}
# TR-Omni'nin serbest metin cevabından sınıf: ilk geçen anahtar kelime kazanır
OMNI_WORDS = [(0, ("kızgın", "öfkeli", "sinirli", "kızgınlık", "öfke")), (1, ("sakin", "nötr", "durgun")),
              (2, ("mutlu", "neşeli", "sevinçli", "heyecanlı")), (3, ("üzgün", "hüzünlü", "kederli", "mutsuz"))]
PROMPT_DUYGU = ("Bu kaydı yalnızca ses tonuna göre değerlendir, sözlerin anlamına bakma. Konuşanın duygusu hangisi? "
                "Yalnızca tek kelime yaz: kızgın, sakin, mutlu ya da üzgün.")
PROMPT_OFKE = ("Bu müşterinin sesindeki ve sözlerindeki öfkeyi 0 ile 4 arasında puanla. 0 hiç öfkeli değil, "
               "4 çok öfkeli. Yalnızca tek bir rakam yaz.")
BOLUMLER = ["e2v_algi", "omni_algi", "omni_ofke", "omni_sohbet"]


def tr_lower(s: str) -> str:
    return s.replace("I", "ı").replace("İ", "i").lower()


def load16(path: str) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    return librosa.resample(x, orig_sr=sr, target_sr=16000) if sr != 16000 else x


def omni_class(text: str) -> int | None:
    t = tr_lower(text)
    hits = [(t.find(w), c) for c, ws in OMNI_WORDS for w in ws if w in t]
    return min(hits)[1] if hits else None


def omni_score(text: str) -> int | None:
    m = re.search(r"[0-4]", text)
    return int(m.group()) if m else None


def uar(y, p, n=4) -> float:
    """Dengeli doğruluk; tanınamayan cevap (None) yanlış sayılır."""
    rec = [np.mean([pi == c for yi, pi in zip(y, p) if yi == c]) for c in range(n) if c in y]
    return float(np.mean(rec))


def spearman(a, b) -> float | None:
    if len(a) < 3:
        return None
    ra, rb = (np.argsort(np.argsort(v)) for v in (np.asarray(a), np.asarray(b)))
    return float(np.corrcoef(ra, rb)[0, 1])


class Jsonl:
    """Satır satır sonuç dosyası; var olan anahtarlar atlanır (kaldığı yerden devam)."""

    def __init__(self, path: Path):
        self.path = path
        self.rows = [json.loads(l) for l in open(path, encoding="utf-8")] if path.exists() else []
        self.done = {r["anahtar"] for r in self.rows}

    def add(self, row: dict):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.rows.append(row)
        self.done.add(row["anahtar"])


def turev_sample(src: str, per_class: int) -> list[tuple[str, int, str]]:
    files = sorted(glob.glob(os.path.join(src, "*", "*.wav")))
    by = {c: [f for f in files if os.path.basename(os.path.dirname(f)) == c] for c in CLASSES}
    rng = random.Random(0)
    out = []
    for c, fs in by.items():
        for f in rng.sample(fs, min(per_class, len(fs))):
            out.append((f, CLASSES.index(c), os.path.basename(f).split("_")[0]))
    return out


_E2V = None


def e2v(x: np.ndarray) -> dict[str, float]:
    global _E2V
    if _E2V is None:
        from funasr import AutoModel
        _E2V = AutoModel(model="emotion2vec/emotion2vec_plus_base", hub="hf", disable_update=True)
    r = _E2V.generate(x, granularity="utterance", extract_embedding=False, disable_pbar=True)[0]
    return {l.split("/")[-1].lower(): float(s) for l, s in zip(r["labels"], r["scores"])}


def e2v_class(scores: dict[str, float]) -> int:
    return max((s, E2V_MAP[l]) for l, s in scores.items() if l in E2V_MAP)[1]


_OMNI = None


def omni(model_dir: str):
    global _OMNI
    if _OMNI is None:
        sys.path.insert(0, model_dir)
        from tr_omni import TROmni
        t0 = time.time()
        _OMNI = TROmni(model_dir)
        print(f"TR-Omni yüklendi ({time.time() - t0:.0f} sn)", flush=True)
    return _OMNI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turev", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--musteri", required=True)
    ap.add_argument("--cikti", required=True)
    ap.add_argument("--adet", type=int, default=40, help="TurEV'den sınıf başına kayıt")
    ap.add_argument("--bolum", nargs="+", default=BOLUMLER, choices=BOLUMLER)
    a = ap.parse_args()
    out = Path(a.cikti)
    (out / "sohbet_ses").mkdir(parents=True, exist_ok=True)
    R = {b: Jsonl(out / f"{b}.jsonl") for b in BOLUMLER}

    sample = turev_sample(a.turev, a.adet)
    musteri = [json.loads(l) for l in open(Path(a.musteri) / "sonuc.jsonl", encoding="utf-8")]
    for m in musteri:
        m["anahtar"] = f"{m['persona']}_g{m['gorusme']}_t{m['tur']}_{m['varyant']}"
        m["wav"] = str(Path(a.musteri) / f"{m['anahtar']}.wav")
    musteri = [m for m in musteri if os.path.exists(m["wav"])]
    print(f"TurEV örneklem: {len(sample)} kayıt, müşteri tepkisi: {len(musteri)} ses", flush=True)

    if "e2v_algi" in a.bolum:
        for f, y, spk in sample:
            if f in R["e2v_algi"].done:
                continue
            sc = e2v(load16(f))
            R["e2v_algi"].add({"anahtar": f, "y": y, "konusmaci": spk, "tahmin": e2v_class(sc), "skorlar": sc})

    if "omni_algi" in a.bolum:
        for i, (f, y, spk) in enumerate(sample):
            if f in R["omni_algi"].done:
                continue
            t0 = time.time()
            cevap, _ = omni(a.model).chat(audio=load16(f), text=PROMPT_DUYGU, speak=False, max_new_tokens=32)
            R["omni_algi"].add({"anahtar": f, "y": y, "konusmaci": spk, "cevap": cevap, "tahmin": omni_class(cevap),
                                "sure": round(time.time() - t0, 2)})
            if i % 20 == 0:
                print(f"omni_algi {i}/{len(sample)}: {cevap!r}", flush=True)

    if "omni_ofke" in a.bolum:
        for m in musteri:
            if m["anahtar"] in R["omni_ofke"].done:
                continue
            cevap, _ = omni(a.model).chat(audio=load16(m["wav"]), text=PROMPT_OFKE, speak=False, max_new_tokens=24)
            R["omni_ofke"].add({"anahtar": m["anahtar"], "olculen_ofke": m["ses_ofke"], "cevap": cevap,
                                "puan": omni_score(cevap)})

    if "omni_sohbet" in a.bolum:
        from core.tr_check import bad_words
        for m in musteri:
            if m["anahtar"] in R["omni_sohbet"].done:
                continue
            t0 = time.time()
            cevap, wav = omni(a.model).chat(audio=load16(m["wav"]), speak=True)
            sure = time.time() - t0
            path = out / "sohbet_ses" / f"{m['anahtar']}.wav"
            sf.write(path, wav, omni(a.model).sr)
            sc = e2v(librosa.resample(wav.astype(np.float32), orig_sr=omni(a.model).sr, target_sr=16000))
            R["omni_sohbet"].add({"anahtar": m["anahtar"], "musteri": m["tepki"], "musteri_ofke": m["ses_ofke"],
                                  "cevap": cevap, "sure": round(sure, 2), "ses_suresi": round(len(wav) / omni(a.model).sr, 2),
                                  "bozuk_kelime": bad_words(cevap), "cevap_sesi_duygu": max(sc, key=sc.get),
                                  "cevap_sesi_skor": sc})
            print(f"omni_sohbet {m['anahtar']} ({sure:.1f} sn): {cevap[:80]!r}", flush=True)

    ozet = {}
    rows = R["e2v_algi"].rows
    if rows:
        y, p = [r["y"] for r in rows], [r["tahmin"] for r in rows]
        ozet["e2v_algi"] = {"n": len(rows), "uar4": uar(y, p),
                            "kizgin_mi_uar": uar([int(v == 0) for v in y], [int(v == 0) for v in p], 2)}
    rows = R["omni_algi"].rows
    if rows:
        y, p = [r["y"] for r in rows], [r["tahmin"] for r in rows]
        ozet["omni_algi"] = {"n": len(rows), "uar4": uar(y, p),
                             "kizgin_mi_uar": uar([int(v == 0) for v in y], [int(v == 0) for v in p], 2),
                             "taninamayan_cevap": sum(v is None for v in p),
                             "ort_sure": float(np.mean([r["sure"] for r in rows]))}
    rows = [r for r in R["omni_ofke"].rows if r["puan"] is not None]
    if R["omni_ofke"].rows:
        ozet["omni_ofke"] = {"n": len(R["omni_ofke"].rows), "taninamayan_cevap": len(R["omni_ofke"].rows) - len(rows),
                             "spearman_olculen_ofke": spearman([r["puan"] for r in rows], [r["olculen_ofke"] for r in rows]),
                             "ort_mutlak_fark": float(np.mean([abs(r["puan"] - r["olculen_ofke"]) for r in rows])) if rows else None}
    rows = R["omni_sohbet"].rows
    if rows:
        sesler = [r["cevap_sesi_duygu"] for r in rows]
        ozet["omni_sohbet"] = {"n": len(rows), "ort_sure": float(np.mean([r["sure"] for r in rows])),
                               "ort_ses_suresi": float(np.mean([r["ses_suresi"] for r in rows])),
                               "bozuk_kelimeli_cevap": sum(bool(r["bozuk_kelime"]) for r in rows),
                               "cevap_sesi_duygu_dagilimi": {k: sesler.count(k) for k in sorted(set(sesler))}}
    json.dump(ozet, open(out / "ozet.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(ozet, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
