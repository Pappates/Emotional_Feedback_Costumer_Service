"""Prompt enjeksiyonu deneyi: dört varyant aynı görüşme durumunda eşleştirilmiş olarak karşılaştırılır.

Neden eşleştirilmiş: simülasyonun gizli ruh hali asistanın SÖZLERİNE değil strateji/ton etiketine ve işlemlere tepki
verir; enjeksiyon ise yalnızca sözleri değiştirir. Bu yüzden her asistan turunda aynı durumdan (aynı strateji, ton,
geçmiş, ruh hali) dört varyantla ayrı cevap üretilir ve her cevaba müşteri ajanının TEPKİSİ üretilir. Tepkideki öfke
(metinden, psych.anger_score) asıl ölçüttür; müşteri ajanı asistanın cevabını okuduğu için sözlere tepki verebilir.

Varyantlar: yok | ofke (düzey + eğilim) | neden (+ değerlendirme) | tam (+ önerilen yaklaşım)
Ek ölçütler (cevap başına): kelime sayısı, empati/özür ifadesi, somut adım (işlem), kural dışı söz, bozuk Türkçe.

Kullanım (lokal mod, Ollama ya da uyumlu sunucu açık):
  python egitim/enjeksiyon_deneyi.py --personalar ahmet mert zeynep selin --gorusme 1 --cikti data/enjeksiyon.jsonl
Görüşme "tam" varyantının cevabıyla ilerler (diğerleri yalnızca karşılaştırma için üretilir).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SAGLAYICI", "lokal")

VARIANTS = ["yok", "ofke", "neden", "tam"]
EMPATHY = re.compile(r"\b(üzgünüm|özür|anlıyorum|haklısınız|kusura bakmayın|sizi anlıyorum|mağduriyet)", re.I)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--personalar", nargs="+", default=["ahmet", "mert", "zeynep", "selin"])
    p.add_argument("--gorusme", type=int, default=1, help="persona başına görüşme")
    p.add_argument("--tur", type=int, default=4, help="görüşme başına en fazla asistan turu")
    p.add_argument("--cikti", default=str(ROOT / "data" / "enjeksiyon.jsonl"))
    a = p.parse_args()
    logging.disable(logging.WARNING)
    from core import call, emotion, tr_check
    from core.call import Call, promise_violations
    from core.learner import GlobalModel

    out = open(a.cikti, "a", encoding="utf-8")
    for persona in a.personalar:
        for g in range(a.gorusme):
            c = Call(persona, GlobalModel(), audio=False, mode="lokal", seed=1000 + g, save_report=False)
            c.customer_turn()
            for turn in range(a.tur):
                ctx = c._agent_prepare()
                if ctx["son"]:
                    break
                an = ctx["an"]
                replies = {}
                for v in VARIANTS:
                    call.INJECTION = v
                    r = c._agent_reply(ctx["s"], ctx["t"], an, ctx["director"])
                    # tepki: aynı ruh hali ve geçmiş, son asistan sözü bu varyantın cevabı
                    c.history.append({"rol": "asistan", "metin": r["cevap"]})
                    try:
                        react = c._customer_text(None, c.current_issue is None)
                    finally:
                        c.history.pop()
                    ra = emotion.analyze_text(react)
                    replies[v] = r
                    out.write(json.dumps({
                        "persona": persona, "gorusme": g, "tur": turn, "varyant": v,
                        "strateji": ctx["s"], "ton": ctx["t"], "musteri_ofke": an.get("ofke"),
                        "gizli_ruh_hali": round(c.mood, 3), "cevap": r["cevap"], "tepki": react,
                        "tepki_ofke": ra.get("ofke"), "kelime": len(r["cevap"].split()),
                        "empati": bool(EMPATHY.search(r["cevap"])), "islem": len(r.get("araclar") or []),
                        "kural_disi": promise_violations(r["cevap"]), "bozuk": tr_check.bad_words(r["cevap"]),
                    }, ensure_ascii=False) + "\n")
                    out.flush()
                call.INJECTION = "tam"
                reply = replies["tam"]
                c._agent_apply(ctx, reply, None)
                c.customer_turn()
                print(f"{persona} g{g} t{turn}: öfke {an.get('ofke')} -> tepkiler "
                      + ", ".join(f"{v}" for v in VARIANTS), flush=True)
                if c.ended:
                    break
    out.close()


if __name__ == "__main__":
    main()
