"""Aşama 2 baseline: eğitimden önce TR-Omni-E4B (sesten sese) ve emotion2vec+ base ne kadar iyi?

Eğitim yok; modeller hazır hâliyle ölçülür. Sonraki ince ayarların geçmesi gereken çıta budur.

Bölümler (her biri ayrı .jsonl dosyasına satır satır yazılır; Colab koparsa kaldığı yerden devam eder):
  e2v_algi    TurEV-DB dengeli örneklem: emotion2vec+ base hazır etiketi (4 duygu)
  omni_algi   Aynı örneklem: TR-Omni'ye sesle birlikte "duygu hangisi?" diye sorulur (yalnızca metin cevap)
  omni_ofke   ElevenLabs müşteri tepkileri (data/enjeksiyon_ses): TR-Omni'den 0-4 öfke puanı;
              mevcut hattın ölçtüğü öfkeyle (ses_ofke) sıra korelasyonu
  omni_sohbet Aynı tepkiler: TR-Omni sesli cevap verir. Süre, cevap metni, Türkçe yazım denetimi,
              cevabın sesinin emotion2vec etiketi (asistan sakin mi duyuluyor?). Ses dosyaları saklanır.
  omni_gecikme Aynı tepkiler: akışlı kullanımda ilk sesin ne zaman çıkacağı. TR-Omni cevabı cümle cümle seslendirir;
              ilk_ses = düşünürün ilk cümleyi bitirme süresi + konuşucunun o cümleyi seslendirme süresi. Ayrıca
              konuşucunun gerçek zaman oranı (seslendirme süresi / ses süresi). Satırlar GPU adıyla anahtarlanır;
              L4 ve A100 sonuçları aynı klasörde yan yana durur.
  omni_enjeksiyon Aynı tepkiler, üç koşul: yok | sakin | ofkeli. Öfke bilgisi kullanıcı turuna metin olarak eklenir
              (core/call.py biçiminde). Cevap metni ve modelin kendi seçtiği ton etiketi karşılaştırılır: ton,
              kelime sayısı, empati/özür, bilginin müşteriye sızması, "yok" cevabına benzerlik.
  omni_konusucu Konuşucuyu (VoxCPM2) hızlandırma denemesi. omni_gecikme'deki cevaplar aynı cümlelerle üç ayarda
              seslendirilir; hepsinde ses akışla (parça parça) çözülür:
                akis             derleme kapalı (TR-Omni'nin varsayılanı), difüzyon adımı model ayarındaki kadar
                akis_derleme     + torch.compile (VoxCPM2'nin kendi optimize() yolu)
                akis_derleme_yarim + difüzyon adımı yarıya iner (kalite düşebilir; sesler dinlemek için saklanır)
              Ölçülen: ilk ses parçasının süresi, gerçek zaman oranı, tahmini ilk ses
              (düşünürün ilk cümlesi + durumlar + ilk parça).

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
from egitim.enjeksiyon_deneyi import EMPATHY  # noqa: E402  (aynı empati/özür ölçütü)

CLASSES = ["Angry", "Calm", "Happy", "Sad"]
E2V_MAP = {"angry": 0, "neutral": 1, "happy": 2, "sad": 3}
# TR-Omni'nin serbest metin cevabından sınıf: ilk geçen anahtar kelime kazanır
OMNI_WORDS = [(0, ("kızgın", "öfkeli", "sinirli", "kızgınlık", "öfke")), (1, ("sakin", "nötr", "durgun")),
              (2, ("mutlu", "neşeli", "sevinçli", "heyecanlı")), (3, ("üzgün", "hüzünlü", "kederli", "mutsuz"))]
PROMPT_DUYGU = ("Bu kaydı yalnızca ses tonuna göre değerlendir, sözlerin anlamına bakma. Konuşanın duygusu hangisi? "
                "Yalnızca tek kelime yaz: kızgın, sakin, mutlu ya da üzgün.")
PROMPT_OFKE = ("Bu müşterinin sesindeki ve sözlerindeki öfkeyi 0 ile 4 arasında puanla. 0 hiç öfkeli değil, "
               "4 çok öfkeli. Yalnızca tek bir rakam yaz.")
BOLUMLER = ["e2v_algi", "omni_algi", "omni_ofke", "omni_sohbet", "omni_gecikme", "omni_enjeksiyon", "omni_konusucu"]
AYARLAR = ["akis", "akis_derleme", "akis_derleme_yarim"]   # sıra önemli: derleme geri alınamaz
# Öfke bilgisi core/call.py'deki _psych_hint biçiminde; başındaki not müşteriye okunmasın diye
NOT = "[Sistem notu, müşteriye okuma] Ses analizi: "
KOSULLAR = {"yok": None,
            "sakin": NOT + "Müşterinin öfkesi: 0,5/4 (sakin), aynı.",
            "ofkeli": NOT + "Müşterinin öfkesi: 3,5/4 (çok öfkeli), yükseliyor. "
                            "Önce öfkeyi yatıştır: kısa cümle, savunmaya geçme, somut adım."}
TON_RE = re.compile(r"^\s*\[\s*ton\s*:\s*([^\]]*)\]\s*")
SIZINTI = re.compile(r"öfke|/4|ses analiz|sistem notu")


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


def omni_turn(audio: np.ndarray, text: str | None) -> tuple[dict, list[dict]]:
    """TROmni.chat ile aynı mesajlar: (sistem, [kullanıcı turu])."""
    T = sys.modules["tr_omni"]
    content = ([{"type": "text", "text": text}] if text else []) + \
              [{"type": "audio", "audio": a} for a in T.split_long(np.asarray(audio, np.float32))]
    return {"role": "system", "content": [{"type": "text", "text": T.SYSTEM}]}, content


def gecikme(o, audio: np.ndarray) -> dict:
    """chat(speak=True) adımları zamanlanarak: düşünür akışla üretir, cevap cümle cümle seslendirilir."""
    T = sys.modules["tr_omni"]
    from tr_speak import speakable, split_sentences
    system, content = omni_turn(audio, None)
    x = o.thinker._inputs([system, {"role": "user", "content": content}], T.TONE)
    t0, ids, ilk_token, ilk_cumle = time.time(), [], None, None
    for tok in o.thinker.fast.generate(x, 1024):
        ids.append(tok)
        if ilk_token is None:
            ilk_token = time.time() - t0
        # ikinci cümle başladıysa birincisi bitmiştir
        if ilk_cumle is None and len(split_sentences(o.thinker.tok.decode(ids, skip_special_tokens=True))) >= 2:
            ilk_cumle = time.time() - t0
    dusunur = time.time() - t0
    cevap = TON_RE.sub("", o.thinker.tok.decode(ids, skip_special_tokens=True)).strip()
    ctx = [system, {"role": "user", "content": [{"type": "audio", "audio": np.asarray(audio, np.float32)[-T.CTX_SEC * 16000:]}]}]
    said, parca = [], []
    for s in split_sentences(cevap):
        t = speakable(s)
        if not re.search(r"\w", t):
            continue
        t1 = time.time()
        st, goffs = o.thinker.states(ctx, " ".join(said), t)
        w = o.talker.speak(t, st, goffs)
        parca.append((time.time() - t1, len(w) / o.sr))
        said.append(t)
    konusucu, ses = sum(p[0] for p in parca), sum(p[1] for p in parca)
    ilk_cumle = ilk_cumle if ilk_cumle is not None else dusunur
    return {"cevap": cevap, "cumle": len(parca), "ilk_token": round(ilk_token or dusunur, 2),
            "ilk_cumle": round(ilk_cumle, 2), "ilk_ses": round(ilk_cumle + parca[0][0], 2) if parca else None,
            "dusunur": round(dusunur, 2), "konusucu": round(konusucu, 2), "ses_suresi": round(ses, 2),
            "konusucu_rtf": round(konusucu / ses, 2) if ses else None, "toplam": round(dusunur + konusucu, 2)}


def speak_akis(tk, text: str, states, goffs, timesteps: int, seed: int = 42) -> tuple[np.ndarray, float, float]:
    """tr_omni.Talker.speak'in akışlı kopyası: hazırlık aynı, ses her adımda VoxCPM2'nin akışlı VAE çözücüsüyle
    parça parça çıkar. Döner: (ses, ilk parçanın süresi, toplam süre)."""
    import torch
    T = sys.modules["tr_omni"]
    t0 = time.time()
    ids, t2g = tk.align(text, goffs, states.shape[0])
    seen, run = {}, {}
    for g in t2g:
        run[g] = run.get(g, 0) + 1
    sub = []
    for g in t2g:
        sub.append(seen.get(g, 0))
        seen[g] = seen.get(g, 0) + 1
    npc = [run[g] for g in t2g]
    with torch.inference_mode():
        s = states.cuda()
        pieces = tk.mapper(s[torch.tensor(t2g, device="cuda")], torch.tensor(sub, device="cuda"),
                           torch.tensor(npc, device="cuda")).float()
        m = tk.m
        e = m.base_lm.embed_tokens.emb.weight[torch.tensor(ids, device="cuda")].float()
        x = tk.fusion(e, pieces)
        P, n, Ap = len(tk.prompt_ids), len(ids), tk.prompt_feat.shape[0]
        L = P + n + 1 + Ap
        toks = torch.zeros(1, L, dtype=torch.long, device="cuda")
        toks[0, :P] = torch.tensor(tk.prompt_ids, device="cuda")
        toks[0, P:P + n] = torch.tensor(ids, device="cuda")
        toks[0, P + n] = T.AUDIO_START
        feat = torch.zeros(1, L, 4, 64, device="cuda", dtype=torch.bfloat16)
        feat[0, P + n + 1:] = tk.prompt_feat.to("cuda", torch.bfloat16)
        am = torch.zeros(1, L, dtype=torch.int32, device="cuda")
        am[0, P + n + 1:] = 1
        mask = torch.zeros(1, L, device="cuda")
        mask[0, P:P + n] = 1
        full = torch.zeros(1, L, x.shape[-1], device="cuda")
        full[0, P:P + n] = x
        m.base_lm.embed_tokens.rep = (mask, full)
        torch.manual_seed(seed)
        parts, ilk = [], None
        try:
            gen = m._inference(toks, 1 - am, feat, am, min_len=2, max_len=min(n * 6 + 10, 2048),
                               inference_timesteps=timesteps, cfg_value=tk.cfg["cfg_value"], streaming=True)
            with m.audio_vae.streaming_decode() as dec:
                for lat, _, _ in gen:
                    parts.append(dec.decode_chunk(lat.to(torch.float32)).squeeze().float().cpu().numpy())
                    if ilk is None:
                        ilk = time.time() - t0
        finally:
            m.base_lm.embed_tokens.rep = None
    wav = np.concatenate(parts) if parts else np.zeros(0, np.float32)
    return wav, (ilk if ilk is not None else time.time() - t0), time.time() - t0


def konusucu(o, audio: np.ndarray, cevap: str, timesteps: int) -> tuple[dict, np.ndarray]:
    """Cevabı cümle cümle akışlı seslendirir (chat ile aynı bağlam). Döner: (ölçümler, ilk cümlenin sesi)."""
    T = sys.modules["tr_omni"]
    from tr_speak import speakable, split_sentences
    system, _ = omni_turn(audio, None)
    ctx = [system, {"role": "user", "content": [{"type": "audio", "audio": np.asarray(audio, np.float32)[-T.CTX_SEC * 16000:]}]}]
    said, sure, ses, ilk, ilk_wav = [], 0.0, 0.0, None, None
    for s in split_sentences(cevap):
        t = speakable(s)
        if not re.search(r"\w", t):
            continue
        t0 = time.time()
        st, goffs = o.thinker.states(ctx, " ".join(said), t)
        durum = time.time() - t0
        wav, ilk_parca, toplam = speak_akis(o.talker, t, st, goffs, timesteps)
        if ilk is None:
            ilk, ilk_wav = durum + ilk_parca, wav
        sure += durum + toplam
        ses += len(wav) / o.sr
        said.append(t)
    return {"ilk_parca": round(ilk, 3) if ilk is not None else None, "konusucu": round(sure, 2),
            "ses_suresi": round(ses, 2), "rtf": round(sure / ses, 3) if ses else None}, ilk_wav


def enjeksiyon(o, audio: np.ndarray, text: str | None) -> tuple[str, str]:
    """Yalnızca metin cevap; ton etiketini (chat'te sabit '[ton: nötr]') bu kez model kendisi seçer."""
    system, content = omni_turn(audio, text)
    x = o.thinker._inputs([system, {"role": "user", "content": content}], "[ton:")
    out = "[ton:" + o.thinker.tok.decode(list(o.thinker.fast.generate(x, 400)), skip_special_tokens=True)
    m = TON_RE.match(out)
    return (tr_lower(m.group(1).strip()) if m else ""), TON_RE.sub("", out).strip()


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

    if "omni_gecikme" in a.bolum:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        kalan = [m for m in musteri if f"{gpu}|{m['anahtar']}" not in R["omni_gecikme"].done]
        for m in kalan[:2]:                                    # ısınma: derleme süresi ölçüme girmesin
            gecikme(omni(a.model), load16(m["wav"]))           # (tek kayıtla ilk ölçüm yine ~18 sn çıktı)
        for m in kalan:
            r = gecikme(omni(a.model), load16(m["wav"]))
            R["omni_gecikme"].add({"anahtar": f"{gpu}|{m['anahtar']}", "gpu": gpu, **r})
            print(f"omni_gecikme {gpu} {m['anahtar']}: ilk ses {r['ilk_ses']} sn, toplam {r['toplam']} sn", flush=True)

    if "omni_enjeksiyon" in a.bolum:
        for m in musteri:
            x16 = None
            for k, text in KOSULLAR.items():
                if f"{m['anahtar']}|{k}" in R["omni_enjeksiyon"].done:
                    continue
                x16 = load16(m["wav"]) if x16 is None else x16
                ton, cevap = enjeksiyon(omni(a.model), x16, text)
                R["omni_enjeksiyon"].add({"anahtar": f"{m['anahtar']}|{k}", "tepki": m["anahtar"], "kosul": k,
                                          "musteri_ofke": m["ses_ofke"], "ton": ton, "cevap": cevap,
                                          "kelime": len(cevap.split()), "empati": bool(EMPATHY.search(cevap)),
                                          "sizinti": bool(SIZINTI.search(tr_lower(cevap)))})
                print(f"omni_enjeksiyon {m['anahtar']} {k}: [{ton}] {cevap[:70]!r}", flush=True)

    if "omni_konusucu" in a.bolum:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        # aynı cümleler: omni_gecikme'deki cevaplar ve düşünürün ilk cümle süresi (önce bu GPU'nun satırları)
        gec = {}
        for r in sorted(R["omni_gecikme"].rows, key=lambda r: r["gpu"] == gpu):
            gec[r["anahtar"].split("|", 1)[1]] = r
        hedef = [m for m in musteri if m["anahtar"] in gec]
        (out / "konusucu_ses").mkdir(exist_ok=True)
        for ayar in AYARLAR:
            kalan = [m for m in hedef if f"{gpu}|{ayar}|{m['anahtar']}" not in R["omni_konusucu"].done]
            if not kalan:
                continue
            o = omni(a.model)
            tm = o.talker.m
            if ayar != "akis" and not hasattr(tm, "_feat_encoder_raw"):
                tm.optimize()
            derleme = hasattr(tm, "_feat_encoder_raw")
            adim = o.talker.cfg["inference_timesteps"]
            adim = max(1, adim // 2) if ayar.endswith("yarim") else adim
            for m in kalan[:2]:                                # ısınma: derleme ve CUDA grafikleri ölçüme girmesin
                konusucu(o, load16(m["wav"]), gec[m["anahtar"]]["cevap"], adim)
            for i, m in enumerate(kalan):
                g = gec[m["anahtar"]]
                r, ilk_wav = konusucu(o, load16(m["wav"]), g["cevap"], adim)
                if i < 5 and ilk_wav is not None:
                    sf.write(out / "konusucu_ses" / f"{ayar}_{m['anahtar']}.wav", ilk_wav, o.sr)
                ilk_ses = round(g["ilk_cumle"] + r["ilk_parca"], 2) if r["ilk_parca"] is not None else None
                R["omni_konusucu"].add({"anahtar": f"{gpu}|{ayar}|{m['anahtar']}", "gpu": gpu, "ayar": ayar,
                                        "derleme": derleme, "adim": adim, "ilk_cumle": g["ilk_cumle"],
                                        "ilk_ses_tahmini": ilk_ses, **r})
                print(f"omni_konusucu {gpu} {ayar} {m['anahtar']}: ilk parça {r['ilk_parca']} sn, "
                      f"rtf {r['rtf']}, tahmini ilk ses {ilk_ses} sn", flush=True)

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
    rows = R["omni_gecikme"].rows
    if rows:
        ozet["omni_gecikme"] = {}
        for gpu in sorted({r["gpu"] for r in rows}):
            g = [r for r in rows if r["gpu"] == gpu and r["ilk_ses"] is not None]
            ilk = [r["ilk_ses"] for r in g]
            ozet["omni_gecikme"][gpu] = {
                "n": len(g), "ilk_ses_ort": float(np.mean(ilk)), "ilk_ses_medyan": float(np.median(ilk)),
                "ilk_ses_max": float(np.max(ilk)), "ilk_token_ort": float(np.mean([r["ilk_token"] for r in g])),
                "ilk_cumle_ort": float(np.mean([r["ilk_cumle"] for r in g])),
                "konusucu_rtf_ort": float(np.mean([r["konusucu_rtf"] for r in g])),
                "toplam_ort": float(np.mean([r["toplam"] for r in g]))}
    rows = R["omni_enjeksiyon"].rows
    if rows:
        from difflib import SequenceMatcher
        cev = {(r["tepki"], r["kosul"]): r["cevap"] for r in rows}
        ozet["omni_enjeksiyon"] = {}
        for k in KOSULLAR:
            g = [r for r in rows if r["kosul"] == k]
            if not g:
                continue
            tonlar = [r["ton"] for r in g]
            benz = [SequenceMatcher(None, r["cevap"], cev[(r["tepki"], "yok")]).ratio()
                    for r in g if (r["tepki"], "yok") in cev]
            ozet["omni_enjeksiyon"][k] = {
                "n": len(g), "ton_dagilimi": {t: tonlar.count(t) for t in sorted(set(tonlar))},
                "ort_kelime": float(np.mean([r["kelime"] for r in g])),
                "empati_orani": float(np.mean([r["empati"] for r in g])),
                "sizinti": sum(r["sizinti"] for r in g),
                "yok_ile_benzerlik": float(np.mean(benz)) if benz else None}
    rows = R["omni_konusucu"].rows
    if rows:
        ozet["omni_konusucu"] = {}
        for gpu in sorted({r["gpu"] for r in rows}):
            ozet["omni_konusucu"][gpu] = {}
            for ayar in AYARLAR:
                g = [r for r in rows if r["gpu"] == gpu and r["ayar"] == ayar and r["ilk_ses_tahmini"] is not None]
                if not g:
                    continue
                ilk = [r["ilk_ses_tahmini"] for r in g]
                ozet["omni_konusucu"][gpu][ayar] = {
                    "n": len(g), "derleme": g[0]["derleme"], "adim": g[0]["adim"],
                    "ilk_parca_ort": float(np.mean([r["ilk_parca"] for r in g])),
                    "ilk_ses_tahmini_ort": float(np.mean(ilk)), "ilk_ses_tahmini_medyan": float(np.median(ilk)),
                    "rtf_ort": float(np.mean([r["rtf"] for r in g])),
                    "rtf_max": float(np.max([r["rtf"] for r in g]))}
    json.dump(ozet, open(out / "ozet.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(ozet, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
