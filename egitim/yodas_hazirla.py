"""YODAS2 Türkçe (CC BY 3.0 YouTube konuşmaları) -> ön etiketli kesitler -> /etiket test/eğitim seti.

Neden: Türkçe, doğal (oyunculu olmayan) ve ticari kullanıma açık duygu verisi yok. YODAS2'nin Türkçe bölümü CC BY 3.0
lisanslı YouTube videolarından oluşuyor (kaynak gösterilerek ticari kullanılabilir). Duygu etiketi yok: kesitlere
emotion2vec+ (ses), Türkçe BERT (metin) ve psikolojik katmanla ÖN etiket verilir, duygu yönünden çeşitli olanlar seçilir
ve /etiket sayfasında insanlar tarafından (ön etiketi görmeden) puanlanır. Böylece insan doğrulamalı bir set oluşur.

Adımlar
  1) topla: parçaları akış hâlinde okur (diske tam arşiv yazmaz), her videodan rastgele 3-8 sn'lik pencereler alır,
     ön etiketler, kesitleri ve aday listesini kaydeder.
       python egitim/yodas_hazirla.py topla --bolum tr100 --parcalar 0 1 --cikti data/yodas
  2) sec: çeşitli ve duygu yoğun kesitleri seçip /etiket kayıtlarına aktarır, kaynak listesini yazar.
       python egitim/yodas_hazirla.py sec --cikti data/yodas --adet 300
     --paket yodas_tr.zip verilirse aktarmak yerine taşınabilir bir paket üretir (başka bilgisayarda kullanmak için).
  3) ice-aktar: paketi bu bilgisayardaki /etiket kayıtlarına ekler (aynı kesit iki kez eklenmez).
       python egitim/yodas_hazirla.py ice-aktar yodas_tr.zip

Bölümler: tr000 = elle yazılmış altyazı (çoğunlukla belgesel/eğitim), tr100 = otomatik altyazı (vlog, sohbet; daha doğal).
Kaynak gösterme: data/yodas/KAYNAKLAR.md (video bağlantıları, CC BY 3.0). Veri git'e girmez (data/ altında).
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import os
import random
import sys
import tarfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

BASE = "https://huggingface.co/datasets/espnet/yodas2/resolve/main/data"
SR = 16000


def _segments(text_field) -> list[tuple[float, float, str]]:
    """'VID-00003-00001023-00001268': 'metin' -> (başlangıç sn, bitiş sn, metin). Zamanlar santisaniye."""
    d = text_field if isinstance(text_field, dict) else ast.literal_eval(text_field)
    out = []
    for k, v in d.items():
        p = k.rsplit("-", 2)
        try:
            out.append((int(p[1]) / 100, int(p[2]) / 100, str(v).strip()))
        except (ValueError, IndexError):
            continue
    return sorted(out)


def _windows(segs, lo=3.0, hi=8.0, max_gap=0.6):
    """Ardışık parçaları 3-8 sn'lik pencerelerde birleştirir (otomatik altyazı cümle ortasından bölünüyor)."""
    out, cur = [], []
    for s in segs:
        if cur and (s[0] - cur[-1][1] > max_gap or s[1] - cur[0][0] > hi):
            if cur[-1][1] - cur[0][0] >= lo:
                out.append((cur[0][0], cur[-1][1], " ".join(x[2] for x in cur)))
            cur = []
        cur.append(s)
    if cur and cur[-1][1] - cur[0][0] >= lo:
        out.append((cur[0][0], cur[-1][1], " ".join(x[2] for x in cur)))
    # "[Müzik]", "[Alkış]" gibi işaretli bölümler konuşma değil
    return [w for w in out if len(w[2].split()) >= 4 and "[" not in w[2]]


def _load_wav(raw: bytes) -> np.ndarray:
    import soundfile as sf
    x, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    return x


def topla(a):
    from core import local, psych
    out = Path(a.cikti)
    (out / "ses").mkdir(parents=True, exist_ok=True)
    cand = out / "adaylar.jsonl"
    prev = [json.loads(l) for l in cand.open(encoding="utf-8")] if cand.exists() else []
    done = {r["parca"] for r in prev if r.get("bitti")}
    seen_vid = {r["video"] for r in prev if "video" in r}
    rng = random.Random(a.tohum)
    for part in a.parcalar:
        tag = f"{a.bolum}/{part:08d}"
        if tag in done:
            print(f"{tag}: zaten işlendi", flush=True)
            continue
        meta = json.load(urllib.request.urlopen(f"{BASE}/{a.bolum}/text/{part:08d}.json"))
        segs = {m["audio_id"]: _segments(m["text"]) for m in meta}
        n_vid = n_clip = 0
        for attempt in range(4):   # bağlantı koparsa baştan açılır, işlenmiş videolar atlanır (gzip akışı ortadan sürdürülemez)
            try:
                print(f"{tag}: {len(segs)} video, ses akışı açılıyor (deneme {attempt + 1})…", flush=True)
                resp = urllib.request.urlopen(f"{BASE}/{a.bolum}/audio/{part:08d}.tar.gz", timeout=120)
                n_vid, n_clip = _stream(resp, segs, seen_vid, cand, out, a, rng, tag, local, psych, n_vid, n_clip)
                break
            except (tarfile.ReadError, OSError, EOFError) as e:
                print(f"{tag}: akış kesildi ({str(e)[:60]}); {n_vid} video işlendi, yeniden deneniyor", flush=True)
        else:
            print(f"{tag}: 4 denemede tamamlanamadı; işlenen kesitler saklandı, parça bitti sayılmadı", flush=True)
            continue
        # parça bittiğini işaretle (yarıda kesilirse yeniden işlenir)
        with cand.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"parca": tag, "bitti": True}) + "\n")
        print(f"{tag}: bitti ({n_vid} video, {n_clip} kesit)", flush=True)


def _stream(resp, segs, seen_vid, cand, out, a, rng, tag, local, psych, n_vid, n_clip):
    with tarfile.open(fileobj=resp, mode="r|gz") as tf, cand.open("a", encoding="utf-8") as f:
            for m in tf:
                if not m.isfile():
                    continue
                vid = Path(m.name).stem
                if vid in seen_vid:
                    continue
                wins = _windows(segs.get(vid, []))
                if not wins:
                    seen_vid.add(vid)
                    continue
                x = _load_wav(tf.extractfile(m).read())
                rng.shuffle(wins)
                for (s, e, txt) in wins[: a.pencere]:
                    clip = x[int(s * SR): int(e * SR)]
                    if clip.size < SR * 2 or np.sqrt(np.mean(clip ** 2)) < 0.005:
                        continue
                    wav = local.float_to_wav(clip / (np.max(np.abs(clip)) + 1e-6) * 0.9, SR)
                    ser = local.speech_emotion(wav)
                    tx = local.text_sentiment(txt)
                    v = 0.7 * tx["valens"] + 0.3 * ser["valens"]
                    ps = psych.assess(txt, v, ser["uyarilma"], None)
                    cid = f"{vid}_{int(s * 100):07d}"
                    (out / "ses" / f"{cid}.wav").write_bytes(wav)
                    probs = ser["olasiliklar"]
                    f.write(json.dumps({
                        "id": cid, "parca": tag, "video": vid, "baslangic": s, "bitis": e, "metin": txt,
                        "ses_olasiliklari": probs, "ses_ust": max(probs, key=probs.get), "notr": probs.get("neutral", 0.0),
                        "valens_ses": ser["valens"], "uyarilma_ses": ser["uyarilma"], "valens_metin": tx["valens"],
                        "kategori_on": ps["kategori"], "baskinlik_on": ps["baskinlik"],
                    }, ensure_ascii=False) + "\n")
                    n_clip += 1
                f.flush()
                seen_vid.add(vid)
                n_vid += 1
                if n_vid % 10 == 0:
                    print(f"{tag}: {n_vid} video, {n_clip} kesit", flush=True)
    return n_vid, n_clip


def sec(a):
    from core import etiket
    out = Path(a.cikti)
    rows = [json.loads(l) for l in (out / "adaylar.jsonl").open(encoding="utf-8")]
    rows = [r for r in rows if "id" in r and (out / "ses" / f"{r['id']}.wav").exists()]
    existing = {r.get("kaynak_id") for r in etiket._read(etiket.KAYIT)}
    rows = [r for r in rows if r["id"] not in existing]
    rng = random.Random(a.tohum)
    rows = [r for r in rows if "[" not in r["metin"]]
    # Duygu yoğunluğu: önce ses (nötr olmayan olasılık); otomatik altyazı cümle ortasından bölündüğü için metnin payı küçük
    for r in rows:
        r["yogunluk"] = (1 - r["notr"]) + 0.2 * abs(r["valens_metin"])
    # Katman: ses modelinin üst sınıfı (ana) × metin valensinin yönü; her katmandan en yoğunlar, video başına sınır
    groups = defaultdict(list)
    for r in rows:
        groups[(r["ses_ust"], "olumsuz" if r["valens_metin"] < -0.2 else "olumlu" if r["valens_metin"] > 0.2 else "orta")].append(r)
    for g in groups.values():
        g.sort(key=lambda r: -r["yogunluk"])
    per_video, picked = Counter(), []
    quota_neutral = int(a.adet * 0.15)          # nötr de olmalı (her şey duygulu olmasın), ama az
    while len(picked) < a.adet and any(groups.values()):
        for key in sorted(groups, key=lambda k: rng.random()):
            g = groups[key]
            while g and per_video[g[0]["video"]] >= a.video_basina:
                g.pop(0)
            if not g:
                continue
            r = g[0]
            is_neutral = r["ses_ust"] == "neutral"
            if is_neutral and sum(1 for p in picked if p["ses_ust"] == "neutral") >= quota_neutral:
                g.pop(0)
                continue
            picked.append(g.pop(0)); per_video[r["video"]] += 1
            if len(picked) >= a.adet:
                break
    manifest = [_manifest_row(r) for r in picked]
    vids = sorted({r["video"] for r in picked})
    lines = ["# Kaynaklar (YODAS2 Türkçe, CC BY 3.0)", "",
             "Ses kesitleri, YODAS2 veri setindeki (espnet/yodas2) Creative Commons Attribution 3.0 lisanslı YouTube "
             "videolarından alınmıştır. Her video:", ""]
    lines += [f"- https://www.youtube.com/watch?v={v}" for v in vids]
    if a.paket:
        import zipfile
        with zipfile.ZipFile(a.paket, "w", zipfile.ZIP_DEFLATED) as z:
            for r in picked:
                z.write(out / "ses" / f"{r['id']}.wav", f"ses/{r['id']}.wav")
            z.writestr("kesitler.jsonl", "\n".join(json.dumps(m, ensure_ascii=False) for m in manifest) + "\n")
            z.writestr("KAYNAKLAR.md", "\n".join(lines) + "\n")
        print(f"Paket: {a.paket} ({os.path.getsize(a.paket) / 1e6:.1f} MB)")
    else:
        for r, m in zip(picked, manifest):
            _import_one((out / "ses" / f"{r['id']}.wav").read_bytes(), m)
        with (out / "KAYNAKLAR.md").open("a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    c1 = Counter(r["ses_ust"] for r in picked)
    c2 = Counter(r["kategori_on"] for r in picked)
    print(f"{len(picked)} kesit seçildi ({len(per_video)} farklı video).\nSes modeli üst sınıf: {dict(c1)}\n"
          f"Psikolojik ön kategori: {dict(c2)}\nKaynak listesi: {out / 'KAYNAKLAR.md'}")


def _manifest_row(r: dict) -> dict:
    return {"id": r["id"], "video": r["video"], "parca": r["parca"], "baslangic": r["baslangic"], "bitis": r["bitis"],
            "metin": r["metin"], "on_etiket": {"kategori": r["kategori_on"], "ses_ust": r["ses_ust"],
                                               "valens_ses": r["valens_ses"], "valens_metin": r["valens_metin"],
                                               "uyarilma_ses": r["uyarilma_ses"]}}


def _import_one(wav: bytes, m: dict):
    from core import etiket
    etiket.save_recording(wav, f"yt:{m['video']}", None, None,
                          f"YODAS2 {m['parca']} · youtu.be/{m['video']}?t={int(m['baslangic'])} · CC BY 3.0",
                          extra={"kaynak": "yodas2", "kaynak_id": m["id"], "lisans": "CC BY 3.0",
                                 "video_url": f"https://www.youtube.com/watch?v={m['video']}&t={int(m['baslangic'])}s",
                                 "metin": m["metin"], "on_etiket": m["on_etiket"]})


def ice_aktar(a):
    import zipfile
    from core import etiket
    existing = {r.get("kaynak_id") for r in etiket._read(etiket.KAYIT)}
    n = 0
    with zipfile.ZipFile(a.paket) as z:
        for line in z.read("kesitler.jsonl").decode("utf-8").splitlines():
            if not line.strip():
                continue
            m = json.loads(line)
            if m["id"] in existing:
                continue
            _import_one(z.read(f"ses/{m['id']}.wav"), m)
            n += 1
        (ROOT / "data" / "etiket").mkdir(parents=True, exist_ok=True)
        (ROOT / "data" / "etiket" / "KAYNAKLAR_yodas.md").write_bytes(z.read("KAYNAKLAR.md"))
    print(f"{n} kesit /etiket kayıtlarına eklendi (zaten olanlar atlandı). Kaynak listesi: data/etiket/KAYNAKLAR_yodas.md")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="komut", required=True)
    t = sub.add_parser("topla")
    t.add_argument("--bolum", default="tr100", choices=["tr000", "tr100"])
    t.add_argument("--parcalar", type=int, nargs="+", default=[0])
    t.add_argument("--pencere", type=int, default=12, help="video başına ön etiketlenecek pencere sayısı")
    t.add_argument("--cikti", default=str(ROOT / "data" / "yodas"))
    t.add_argument("--tohum", type=int, default=0)
    s = sub.add_parser("sec")
    s.add_argument("--cikti", default=str(ROOT / "data" / "yodas"))
    s.add_argument("--adet", type=int, default=300)
    s.add_argument("--video-basina", dest="video_basina", type=int, default=4)
    s.add_argument("--tohum", type=int, default=0)
    s.add_argument("--paket", help="aktarmak yerine bu zip dosyasına paketle")
    i = sub.add_parser("ice-aktar")
    i.add_argument("paket")
    a = p.parse_args()
    {"topla": topla, "sec": sec, "ice-aktar": ice_aktar}[a.komut](a)


if __name__ == "__main__":
    main()
