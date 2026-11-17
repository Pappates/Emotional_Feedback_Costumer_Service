"""Gerçek insan sesinden test seti (Aşama 1, A2): kayıt, etiketleme, uyum ve model değerlendirmesi.

Akış: ekip arkadaşları senaryo kartlarını canlandırıp kaydeder -> en az 3 kişi her kaydı SAM ölçeğiyle (1-9) valens,
uyarılma, baskınlık; kategori ve değerlendirme boyutlarıyla etiketler -> etiketleyiciler arası uyum (Krippendorff alfa)
-> mevcut model bu sette ölçülür (CCC, kategori UAR). Aşama 2'de her yeni model aynı sette yarışır.

Dosyalar: data/etiket/ses/<id>.wav, data/etiket/kayitlar.jsonl, data/etiket/puanlar.jsonl
"""
from __future__ import annotations

import csv
import io
import json
import time
import uuid
from pathlib import Path

import numpy as np

from . import config

ROOT = config.DATA_DIR / "etiket"
AUDIO = ROOT / "ses"
KAYIT = ROOT / "kayitlar.jsonl"
PUAN = ROOT / "puanlar.jsonl"

KATEGORILER = ["öfke", "hayal kırıklığı", "kaygı", "kafa karışıklığı", "küskünlük", "nötr", "rahatlama", "memnuniyet"]
SUC = ["kurum", "durum", "kendisi", "yok"]

# Senaryo kartları: her kategori ve değerlendirme birleşimi için canlandırma yönergesi + örnek cümle.
# Konuşmacı cümleyi kendi sözleriyle söyleyebilir; önemli olan tarif edilen ruh hali.
KARTLAR = [
    ("K01", "öfke", "Suç kurumda, talepkâr", "Üç gündür internetim yok ve kimse arayıp sormuyor! Hemen bir teknisyen gönderin."),
    ("K02", "öfke", "Haksızlık, yüksek ses", "Kullanmadığım bir paketin parasını neden ödüyorum? Bu düpedüz soygun!"),
    ("K03", "öfke", "Tekrarlanan sorun", "Bu ay üçüncü kez arıyorum, her seferinde aynı şeyi söylüyorsunuz!"),
    ("K04", "öfke", "Ayrılma tehdidi", "Yeter artık, sözleşmemi iptal edin, başka operatöre geçiyorum."),
    ("K05", "hayal kırıklığı", "Beklenti boşa çıktı, ses alçak", "Dün düzelecek demiştiniz ama hâlâ aynı, açıkçası çok bozuldum."),
    ("K06", "hayal kırıklığı", "Uzun bekleme", "Kırk dakikadır hattayım, sonunda bağlandım ama bilmiyorum, umudum azaldı."),
    ("K07", "hayal kırıklığı", "Hizmet kalitesi", "Bu kadar yıllık müşterinizim, böyle bir hizmet beklemezdim."),
    ("K08", "kaygı", "Belirsizlik, düşük kontrol", "Yarın evden sınava gireceğim, internet gelmezse ne yapacağım bilmiyorum."),
    ("K09", "kaygı", "Para kaygısı", "Faturam iki katına çıkmış, bunu ödeyemem, ne olacak şimdi?"),
    ("K10", "kaygı", "İş kaybı", "POS cihazım çalışmıyor, müşteriler kapıdan dönüyor, ne zaman düzelir?"),
    ("K11", "kafa karışıklığı", "Anlamıyor, soruyor", "Modemde hangi ışığa bakmam gerekiyor? Kırmızı mı turuncu mu, anlamadım."),
    ("K12", "kafa karışıklığı", "Fatura kalemleri", "Faturada üç ayrı ücret var, hangisi ne için, açıklar mısınız?"),
    ("K13", "küskünlük", "Vazgeçmiş, düz ses", "Neyse, boş verin. Zaten bir şey değişmeyecek."),
    ("K14", "küskünlük", "Sessiz churn", "Tamam. Ben ne yapacağımı biliyorum artık, teşekkürler."),
    ("K15", "küskünlük", "Bıkkınlık", "Bıktım açıkçası, her ay aynı terane. Yorgunum artık."),
    ("K16", "nötr", "Bilgi talebi", "Merhaba, yeni adresime nakil yaptırmak istiyorum, ne gerekiyor?"),
    ("K17", "nötr", "Rutin işlem", "Fatura adresimi güncellemek istiyorum, ad soyad bilgilerim aynı."),
    ("K18", "nötr", "Kimlik doğrulama", "Tabii, kimlik numaramın son dört hanesi bir iki üç dört."),
    ("K19", "rahatlama", "Sorun çözüldü", "Oh, çok şükür. Işık yeşile döndü, internet geldi galiba."),
    ("K20", "rahatlama", "İade kabul", "Tamam, iade yapılacaksa sorun yok, içim rahatladı."),
    ("K21", "rahatlama", "Hâlâ temkinli", "Peki, teknisyen gelirse mesele kalmaz. Yine de bir mesaj atarsanız sevinirim."),
    ("K22", "memnuniyet", "Teşekkür", "Çok teşekkür ederim, gerçekten çok yardımcı oldunuz."),
    ("K23", "memnuniyet", "Kampanya sevinci", "Harika! Bir ay indirim de güzel oldu, sağ olun."),
    ("K24", "memnuniyet", "Övgü", "Sizin gibi ilgili birine denk geldiğim için şanslıyım, iyi günler."),
]


def _ensure():
    AUDIO.mkdir(parents=True, exist_ok=True)


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _append(path: Path, row: dict):
    _ensure()
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_recording(wav: bytes, konusmaci: str, kart: str | None, hedef: str | None, not_: str = "") -> dict:
    _ensure()
    kid = uuid.uuid4().hex[:10]
    (AUDIO / f"{kid}.wav").write_bytes(wav)
    row = {"id": kid, "konusmaci": konusmaci.strip()[:40] or "anonim", "kart": kart, "hedef_kategori": hedef,
           "not": not_[:200], "tarih": time.strftime("%Y-%m-%dT%H:%M:%S")}
    _append(KAYIT, row)
    return row


def recordings() -> list[dict]:
    puan = _read(PUAN)
    out = []
    for r in _read(KAYIT):
        if not (AUDIO / f"{r['id']}.wav").exists():
            continue
        raters = {p["etiketleyici"] for p in puan if p["kayit"] == r["id"]}
        out.append({**r, "ses": f"/etiket-ses/{r['id']}.wav", "etiketleyici_sayisi": len(raters),
                    "etiketleyiciler": sorted(raters)})
    return out


def save_rating(d: dict) -> dict:
    row = {"kayit": d["kayit"], "etiketleyici": str(d["etiketleyici"]).strip()[:40] or "anonim",
           "valens": int(d["valens"]), "uyarilma": int(d["uyarilma"]), "baskinlik": int(d["baskinlik"]),
           "kategori": d["kategori"] if d.get("kategori") in KATEGORILER else "nötr",
           "suc": d.get("suc") if d.get("suc") in SUC else "yok",
           "belirsizlik": int(d.get("belirsizlik", 1)), "kontrol_kaybi": int(d.get("kontrol_kaybi", 1)),
           "haksizlik": int(d.get("haksizlik", 1)), "guven": int(d.get("guven", 3)),
           "tarih": time.strftime("%Y-%m-%dT%H:%M:%S")}
    for k in ("valens", "uyarilma", "baskinlik"):
        row[k] = min(9, max(1, row[k]))
    _append(PUAN, row)
    return row


def _latest_ratings() -> list[dict]:
    """Aynı etiketleyici aynı kaydı birden çok kez puanladıysa sonuncusu geçerli."""
    last = {}
    for p in _read(PUAN):
        last[(p["kayit"], p["etiketleyici"])] = p
    return list(last.values())


# --------------------------------------------------------------------------- uyum (Krippendorff alfa)
def krippendorff_alpha(units: dict[str, list], level: str = "interval") -> float | None:
    """units: kayıt -> o kayda verilen puanlar listesi. level: 'interval' (SAM 1-9) ya da 'nominal' (kategori).
    Yalnızca en az 2 puan almış kayıtlar sayılır. 1 = tam uyum, 0 = şans düzeyi; ≥0,667 kabul edilebilir."""
    units = {k: v for k, v in units.items() if len(v) >= 2}
    if not units:
        return None
    vals = [x for v in units.values() for x in v]
    n = len(vals)
    if level == "nominal":
        def d(a, b): return 0.0 if a == b else 1.0
    else:
        def d(a, b): return (float(a) - float(b)) ** 2
    do = sum(sum(d(a, b) for i, a in enumerate(v) for j, b in enumerate(v) if i != j) / (len(v) - 1) for v in units.values())
    de = sum(d(a, b) for i, a in enumerate(vals) for j, b in enumerate(vals) if i != j) / (n - 1)
    if de == 0:
        return 1.0
    return round(1 - do / de, 3)


def _sam_to_unit(x: float) -> float:
    """SAM 1..9 -> -1..+1 (5 = nötr)."""
    return (x - 5) / 4


def summary() -> dict:
    rec = {r["id"]: r for r in recordings()}
    rat = [p for p in _latest_ratings() if p["kayit"] in rec]
    by = {}
    for p in rat:
        by.setdefault(p["kayit"], []).append(p)
    alpha = {k: krippendorff_alpha({u: [p[k] for p in ps] for u, ps in by.items()}) for k in ("valens", "uyarilma", "baskinlik")}
    alpha["kategori"] = krippendorff_alpha({u: [p["kategori"] for p in ps] for u, ps in by.items()}, "nominal")
    items = []
    for u, ps in by.items():
        cats = [p["kategori"] for p in ps]
        items.append({"kayit": u, "n": len(ps), **{k: round(float(np.mean([p[k] for p in ps])), 2) for k in
                     ("valens", "uyarilma", "baskinlik", "belirsizlik", "kontrol_kaybi", "haksizlik")},
                      "kategori": max(set(cats), key=cats.count), "hedef": rec[u].get("hedef_kategori"),
                      "konusmaci": rec[u]["konusmaci"]})
    hedef_uyum = [i["kategori"] == i["hedef"] for i in items if i["hedef"]]
    return {"kayit_sayisi": len(rec), "etiketli_kayit": len(by), "en_az_3_etiketli": sum(1 for v in by.values() if len(v) >= 3),
            "etiketleyiciler": sorted({p["etiketleyici"] for p in rat}), "konusmacilar": sorted({r["konusmaci"] for r in rec.values()}),
            "alfa": alpha, "hedefle_uyum": round(float(np.mean(hedef_uyum)), 3) if hedef_uyum else None, "kayitlar": items}


def to_csv() -> str:
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=";")
    cols = ["kayit", "konusmaci", "kart", "hedef_kategori", "etiketleyici", "valens", "uyarilma", "baskinlik", "kategori",
            "suc", "belirsizlik", "kontrol_kaybi", "haksizlik", "guven", "tarih"]
    w.writerow(cols)
    rec = {r["id"]: r for r in _read(KAYIT)}
    for p in _latest_ratings():
        r = rec.get(p["kayit"], {})
        w.writerow([p["kayit"], r.get("konusmaci"), r.get("kart"), r.get("hedef_kategori")] + [p.get(c) for c in cols[4:]])
    return "﻿" + buf.getvalue()


# --------------------------------------------------------------------------- mevcut modeli test setinde ölç
def ccc(x, y) -> float | None:
    """Uyum korelasyon katsayısı (Lin): hem korelasyonu hem ölçek/kaymayı cezalandırır; duygu regresyonunda standart."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return None
    mx, my = x.mean(), y.mean()
    cov = ((x - mx) * (y - my)).mean()
    return round(float(2 * cov / (x.var() + y.var() + (mx - my) ** 2 + 1e-12)), 3)


def evaluate() -> dict:
    """Mevcut lokal analiz hattını etiketli kayıtlarda çalıştırır (en az 2 etiketleyici)."""
    from . import emotion
    s = summary()
    items = [i for i in s["kayitlar"] if i["n"] >= 2]
    rows = []
    for i in items:
        wav = (AUDIO / f"{i['kayit']}.wav").read_bytes()
        an = emotion.analyze_audio(wav)
        ps = an.get("psikoloji") or {}
        rows.append({"kayit": i["kayit"], "valens_insan": _sam_to_unit(i["valens"]), "valens_model": an["valens"],
                     "uyarilma_insan": (i["uyarilma"] - 1) / 8, "uyarilma_model": an["uyarilma"],
                     "baskinlik_insan": _sam_to_unit(i["baskinlik"]), "baskinlik_model": ps.get("baskinlik"),
                     "kategori_insan": i["kategori"], "kategori_model": ps.get("kategori") or an.get("duygu"),
                     "transkript": an.get("transkript")})
    out = {"n": len(rows)}
    for k in ("valens", "uyarilma", "baskinlik"):
        pairs = [(r[f"{k}_insan"], r[f"{k}_model"]) for r in rows if r[f"{k}_model"] is not None]
        out[f"ccc_{k}"] = ccc(*zip(*pairs)) if len(pairs) >= 3 else None
    cats = sorted({r["kategori_insan"] for r in rows})
    if cats:
        recall = [np.mean([r["kategori_model"] == c for r in rows if r["kategori_insan"] == c]) for c in cats]
        out["kategori_uar"] = round(float(np.mean(recall)), 3)
    out["satirlar"] = rows
    return out
