"""Enjeksiyon deneyinin sesli doğrulaması: kızgın durumlardaki müşteri tepkileri ElevenLabs ile seslendirilir ve
öfke sesten + metinden (canlı görüşmedeki gibi) yeniden ölçülür.

enjeksiyon_deneyi.py tepki öfkesini yalnızca metinden ölçer. Burada her tepki, simülasyonun gizli ruh haline uygun ses
etiketiyle ([angry], [frustrated]…) seslendirilir; ses dosyaları saklanır, böylece öfke ölçümü değişince kredi
harcamadan yeniden analiz edilebilir (--yeniden).

Kullanım (ELEVENLABS_API_KEY ortam değişkeni gerekli; yalnızca erkek personalar, Adam sesi):
  python egitim/enjeksiyon_ses.py              (girdi: egitim/sonuc_enjeksiyon.jsonl, sesler: data/enjeksiyon_ses)
  python egitim/enjeksiyon_ses.py --yeniden     (kredi harcamadan kayıtlı sesleri yeniden analiz et)
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SAGLAYICI", "lokal")
os.environ.setdefault("LOKAL_MUSTERI_SESI", "elevenlabs")
VARIANTS = ["yok", "ofke", "neden", "tam"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--girdi", default=str(ROOT / "egitim" / "sonuc_enjeksiyon.jsonl"))
    p.add_argument("--klasor", default=str(ROOT / "data" / "enjeksiyon_ses"))
    p.add_argument("--esik", type=float, default=1.5, help="müşteri öfkesi bu değerin üstündeki durumlar")
    p.add_argument("--yeniden", action="store_true", help="seslendirme yapma, kayıtlı sesleri yeniden analiz et")
    a = p.parse_args()
    logging.disable(logging.WARNING)
    from core import emotion, local
    from core.catalog import PERSONAS, mood_delivery

    rows = [json.loads(x) for x in open(a.girdi, encoding="utf-8")]
    rows = [r for r in rows if (r.get("musteri_ofke") or 0) >= a.esik
            and local.ELEVEN_VOICES.get(PERSONAS[r["persona"]]["ses"])]
    out_dir = Path(a.klasor)
    out_dir.mkdir(parents=True, exist_ok=True)
    chars = sum(len(r["tepki"]) for r in rows)
    print(f"{len(rows)} tepki, {chars} karakter", flush=True)
    res = []
    for r in rows:
        f = out_dir / f"{r['persona']}_g{r['gorusme']}_t{r['tur']}_{r['varyant']}.wav"
        if not f.exists():
            if a.yeniden:
                continue
            wav = local.eleven_tts(r["tepki"], mood_delivery(r["gizli_ruh_hali"]),
                                   local.ELEVEN_VOICES[PERSONAS[r["persona"]]["ses"]])
            if not wav:
                print("ElevenLabs seslendiremedi, duruyorum", flush=True)
                break
            f.write_bytes(wav)
        an = emotion.analyze_audio(f.read_bytes())
        d = an.get("ofke_detay") or {}
        res.append({**r, "ses_ofke": an["ofke"], "ses_kanal": d.get("ses"), "metin_kanal": d.get("metin"),
                    "duyulan": an.get("transkript", "")})
        print(f"{f.name}: metin {r['tepki_ofke']:.2f} -> ses+metin {an['ofke']:.2f} (ses {d.get('ses')})", flush=True)
    (out_dir / "sonuc.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in res), encoding="utf-8")

    by = collections.defaultdict(dict)
    for x in res:
        by[(x["persona"], x["gorusme"], x["tur"])][x["varyant"]] = x
    keys = [k for k in by if all(v in by[k] for v in VARIANTS)]
    print(f"\n{len(keys)} durum (dört varyant da sesli)")
    if not keys:
        return
    for col, name in (("tepki_ofke", "yalnızca metin"), ("ses_ofke", "ses + metin")):
        print(f"-- {name}")
        for v in VARIANTS:
            vals = [by[k][v][col] for k in keys]
            diff = [by[k][v][col] - by[k]["yok"][col] for k in keys]
            print(f"  {v:6} ort {st.mean(vals):.2f}  yok'a göre {st.mean(diff):+.2f}  "
                  f"(daha az {sum(d < -0.1 for d in diff)} / daha çok {sum(d > 0.1 for d in diff)})")


if __name__ == "__main__":
    main()
