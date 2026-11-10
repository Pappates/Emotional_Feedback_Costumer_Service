"""Görüşme başına ayrıntılı duygu analizi raporu (HTML + CSV + JSON) ve tüm görüşmelerin özet sayfası.

Rapor her tur için şunları kaydeder:
  * Müşterinin gizli ruh halinin nasıl değiştiği ve bileşenleri (strateji/ton etkisi, çözüm bonusu, sabır, gürültü)
  * Asistanın algısı: Gemini (söz + ses tonu) çıktısı, akustik özellikler, uyarılmışlık füzyonu, algı hatası
  * Ödülün adım adım hesaplanışı (duygu değişimi + ilerleme bonusu, kırpma)
  * Strateji/ton seçimi: Thompson örnekleri, seçim öncesi model durumu, gerekçe, yönetmen uyarısı
Dosyalar: data/raporlar/<görüşme id>/rapor.html | rapor.csv | rapor.json
"""
from __future__ import annotations

import csv
import datetime as dt
import html
import io
import json
from pathlib import Path

import numpy as np

from . import config
from .catalog import ISSUES, PERSONAS, STRATEGIES, TONES

REPORT_DIR = config.DATA_DIR / "raporlar"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SONUC = {"kaldi": "KALDI", "ayrildi": "KAYIP (churn)", None: "yarıda kaldı"}


# ======================================================================== veri
def _turns(events: list[dict]) -> list[dict]:
    """Olay listesini (müşteri, asistan) çiftlerine dönüştürür. Tur k: müşterinin k. konuşması ve asistanın ona tepkisi."""
    turns, cur = [], None
    for e in events:
        if e["tip"] == "musteri":
            cur = {"tur": len(turns), "musteri": e, "asistan": None}
            turns.append(cur)
        elif cur is not None:
            cur["asistan"] = e
    return turns


def _pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() < 1e-9 or b.std() < 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a, b):
    if len(a) < 3:
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return _pearson(ra, rb)


def build(c) -> dict:
    turns = _turns(c.events)
    analyzed = [t for t in turns if t["asistan"]]
    per = [t["asistan"]["algi"]["valens"] for t in analyzed]
    true = [t["musteri"]["gercek_ruh_hali"] for t in analyzed]
    err = [p - q for p, q in zip(per, true)]

    # Yön uyumu: algılanan değişim gerçek değişimle aynı yönde mi? (gerçek değişim anlamlıysa)
    agree = total = 0
    for i in range(1, len(per)):
        dm = true[i] - true[i - 1]
        if abs(dm) > 0.03:
            total += 1
            agree += np.sign(per[i] - per[i - 1]) == np.sign(dm)

    rewards = [t["asistan"]["odul"] for t in analyzed if t["asistan"].get("odul")]
    usage = {"strateji": {}, "ton": {}}
    for t in analyzed:
        sec = t["asistan"].get("secim")
        if sec:
            usage["strateji"][sec["strateji"]] = usage["strateji"].get(sec["strateji"], 0) + 1
            usage["ton"][sec["ton"]] = usage["ton"].get(sec["ton"], 0) + 1

    # Öğrenilen (son model) ile gizli tercihlerin karşılaştırması
    son = next((t["asistan"].get("beyin_son") for t in reversed(analyzed) if t["asistan"].get("beyin_son")), None)
    if son is None:
        son = c.learner.snapshot()
    p = PERSONAS[c.persona_id]
    learned = {}
    for dim, prefs in (("strateji", p["tercih_strateji"]), ("ton", p["tercih_ton"])):
        rows = []
        for arm, hidden in prefs.items():
            s = son[dim][arm]
            rows.append({"kol": arm, "gizli": hidden, "tahmin": s["tahmin"], "belirsizlik": s["belirsizlik"],
                         "gorusme_n": s["gorusme_n"], "gorusme_ort": s["gorusme_ort"], "genel_n": s["genel_n"]})
        tried = [r for r in rows if r["gorusme_n"] > 0]
        learned[dim] = {
            "satirlar": rows,
            "sira_korelasyonu": _spearman([r["tahmin"] for r in rows], [r["gizli"] for r in rows]),
            "en_iyi_gizli": max(rows, key=lambda r: r["gizli"])["kol"],
            "en_iyi_tahmin": max(rows, key=lambda r: r["tahmin"])["kol"],
            "denenen": len(tried),
        }

    start = turns[0]["musteri"]["gercek_ruh_hali"] if turns else None
    end = turns[-1]["musteri"]["gercek_ruh_hali"] if turns else None
    return {
        "id": c.id, "tarih": dt.datetime.fromtimestamp(c.started_at).isoformat(timespec="seconds"),
        "musteri": p["ad"], "persona": c.persona_id, "ozet_profil": p["ozet"], "mod": c.mode, "sesli": c.audio,
        "politika": c.policy, "baglam": c.learner.ctx, "sonuc": c.outcome, "asistan_turu": c.agent_turns,
        "sorunlar": [ISSUES[i]["ad"] for i in p["sorunlar"]], "cozulen": [ISSUES[i]["ad"] for i in c.resolved],
        "metrikler": {
            "gercek_baslangic": start, "gercek_son": end,
            "algilanan_baslangic": per[0] if per else None, "algilanan_son": per[-1] if per else None,
            "algi_mae": round(float(np.mean(np.abs(err))), 4) if err else None,
            "algi_sapma": round(float(np.mean(err)), 4) if err else None,
            "algi_korelasyon": _pearson(per, true),
            "yon_uyumu": round(agree / total, 3) if total else None, "yon_uyumu_n": total,
            "ort_ham_odul": round(float(np.mean([r["ham_odul"] for r in rewards])), 4) if rewards else None,
            "kirpilan_odul": sum(1 for r in rewards if r["ham_odul"] != r["ogrenmeye_giren"]),
            "kullanim": usage,
        },
        "ogrenilen_vs_gizli": learned,
        "turlar": turns,
    }


# ======================================================================== CSV
def _num(x):
    if x is None:
        return ""
    if isinstance(x, (bool, np.bool_)):
        return "evet" if x else "hayır"
    if isinstance(x, (int, float, np.floating)):
        return f"{float(x):.4f}".rstrip("0").rstrip(".").replace(".", ",")   # Türkçe Excel: ondalık virgül
    return str(x)


CSV_COLS = [
    "tur", "musteri_metni", "konu_gercek", "gercek_ruh_hali", "ruh_hali_degisimi", "strateji_etkisi", "ton_etkisi",
    "cozum_bonusu", "kampanya_etkisi", "sabir_cezasi", "gurultu", "algi_kaynagi", "algilanan_duygu", "algilanan_valens",
    "algi_hatasi", "uyarilma_gemini", "uyarilma_akustik", "uyarilma_fuzyon", "churn_riski", "algilanan_konu",
    "ses_ipuclari", "metin_ipuclari", "sure_sn", "enerji_db", "perde_hz", "perde_dalgalanma", "konusma_hizi_kelime_sn",
    "odul_icin_strateji", "odul_icin_ton", "onceki_valens", "yeni_valens", "duygu_degisimi", "ilerleme_bonusu",
    "ham_odul", "ogrenmeye_giren_odul", "yonetmen_uyarisi", "secilen_strateji", "secilen_ton", "strateji_gerekcesi",
    "ton_gerekcesi", "asistan_cevabi", "asistan_ic_sesi", "araclar",
]


def to_csv(rep: dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=";")
    w.writerow(CSV_COLS)
    for t in rep["turlar"]:
        m, a = t["musteri"], t["asistan"] or {}
        b = m.get("ruh_hali_bilesenleri") or {}
        an = a.get("algi") or {}
        ak = an.get("akustik") or {}
        o = a.get("odul") or {}
        s = a.get("secim") or {}
        w.writerow([_num(v) for v in [
            t["tur"], m["metin"], ISSUES.get(m.get("konu") or "", {}).get("ad", ""), m["gercek_ruh_hali"],
            b.get("toplam"), b.get("strateji_etkisi"), b.get("ton_etkisi"), b.get("cozum_bonusu"),
            b.get("kampanya_etkisi"), b.get("sabir_cezasi"), b.get("gurultu"), an.get("kaynak"), an.get("duygu"),
            an.get("valens"), a.get("algi_hatasi"), an.get("uyarilma_gemini"), ak.get("akustik_uyarilma"),
            an.get("uyarilma"), an.get("churn_riski"), an.get("konu"), an.get("ses_ipuclari"), an.get("metin_ipuclari"),
            ak.get("sure_sn"), ak.get("enerji_db"), ak.get("perde_hz"), ak.get("perde_dalgalanma"), ak.get("konusma_hizi"),
            o.get("strateji"), o.get("ton"), o.get("onceki_valens"), o.get("yeni_valens"), o.get("duygu_degisimi"),
            o.get("ilerleme_bonusu"), o.get("ham_odul"), o.get("ogrenmeye_giren"), a.get("yonetmen"),
            s.get("strateji"), s.get("ton"), (s.get("neden") or {}).get("strateji"), (s.get("neden") or {}).get("ton"),
            a.get("cevap"), a.get("ic_ses"), " | ".join(f"{x['arac']}: {x['sonuc']}" for x in a.get("araclar") or []),
        ]])
    return "﻿" + buf.getvalue()   # BOM: Excel Türkçe karakterleri doğru açsın


# ======================================================================== HTML
E = html.escape


def _f(x, d=2, sign=True):
    if x is None:
        return "–"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def _svg_mood(rep) -> str:
    rows = [t for t in rep["turlar"] if t["asistan"]]
    if not rows:
        return ""
    W, H, L, R, T, B = 720, 230, 44, 16, 14, 30
    n = len(rows)
    x = lambda i: L + (W - L - R) * (i / max(1, n - 1))
    y = lambda v: T + (H - T - B) * (1 - (v + 1) / 2)
    grid = "".join(
        f'<line x1="{L}" x2="{W-R}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="{"zero" if v == 0 else "grid"}"/>'
        f'<text x="{L-8}" y="{y(v)+4:.1f}" class="ax" text-anchor="end">{v:+.1f}</text>'.replace("+0.0", "0")
        for v in (-1, -0.5, 0, 0.5, 1))
    xt = "".join(f'<text x="{x(i):.1f}" y="{H-8}" class="ax" text-anchor="middle">T{i}</text>' for i in range(n))

    def series(vals, cls, label):
        pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(vals))
        dots = "".join(f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="4" class="{cls}"><title>T{i} · {label}: {v:+.2f}</title></circle>'
                       for i, v in enumerate(vals))
        return f'<polyline points="{pts}" class="{cls}"/>{dots}'

    per = [t["asistan"]["algi"]["valens"] for t in rows]
    tru = [t["musteri"]["gercek_ruh_hali"] for t in rows]
    return (f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="Algılanan ve gerçek duygu">'
            f'{grid}{xt}{series(tru, "s2 dash", "gerçek ruh hali")}{series(per, "s1", "algılanan valens")}</svg>')


def _svg_churn(rep) -> str:
    rows = [t for t in rep["turlar"] if t["asistan"]]
    if not rows:
        return ""
    W, H, L, R, T, B = 720, 120, 44, 16, 10, 26
    n = len(rows)
    bw = min(46, (W - L - R) / n - 8)
    x = lambda i: L + (W - L - R) * ((i + 0.5) / n)
    y = lambda v: T + (H - T - B) * (1 - v)
    grid = "".join(f'<line x1="{L}" x2="{W-R}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="grid"/>'
                   f'<text x="{L-8}" y="{y(v)+4:.1f}" class="ax" text-anchor="end">%{int(v*100)}</text>' for v in (0, 0.5, 1))
    bars = ""
    for i, t in enumerate(rows):
        v = float(t["asistan"]["algi"]["churn_riski"])
        bars += (f'<rect x="{x(i)-bw/2:.1f}" y="{y(v):.1f}" width="{bw:.1f}" height="{y(0)-y(v):.1f}" rx="3" class="bar">'
                 f'<title>T{i} · churn riski %{v*100:.0f}</title></rect>'
                 f'<text x="{x(i):.1f}" y="{H-8}" class="ax" text-anchor="middle">T{i}</text>')
    return f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="Churn riski">{grid}{bars}</svg>'


METHOD_GEMINI = """
<ol class="method">
<li><b>Ses → metin + duygu (Gemini, ses anlama).</b> Müşterinin ses kaydı doğrudan Gemini'ye verilir. Model hem dökümü çıkarır
hem de <i>sözlere</i> ve <i>ses tonuna</i> (hız, yükseklik, gerginlik, iç çekme) bakarak şunları döner: duygu etiketi,
<b>valens</b> (−1 çok olumsuz … +1 çok olumlu), <b>uyarılmışlık</b> (0 sakin … 1 gergin), <b>churn riski</b> (0–1), konu,
ses ve söz ipuçları. Ses yoksa (TTS kotası dolduysa) aynı analiz yalnızca metinden yapılır; rapordaki "algı kaynağı" bunu gösterir.</li>
<li><b>Akustik özellikler (yerel, numpy).</b> 40 ms'lik pencerelerde enerji (RMS → dB), otokorelasyonla perde (70–400 Hz),
perde dalgalanması (std/ort) ve konuşma hızı (kelime / sesli süre) ölçülür. Bunlardan bir <b>akustik gerginlik</b> skoru çıkar:
<code>0,4·enerji + 0,3·hız + 0,3·perde dalgalanması</code> (her biri 0–1'e ölçeklenir).</li>
<li><b>Füzyon.</b> Valens Gemini'den alınır (söz + ton birlikte değerlendirildiği için). Uyarılmışlık iki kanalın ağırlıklı ortalamasıdır:
<code>uyarılma = 0,6·Gemini + 0,4·akustik</code>.</li>
<li><b>Ödül.</b> Asistanın bir önceki hamlesi (strateji + ton) şöyle puanlanır:
<code>ödül = (yeni valens − önceki valens) + 0,15 × (o hamlede çözülen sorun sayısı)</code>.
Tek bir gürültülü ölçümün öğrenmeyi ele geçirmemesi için ödül öğrenmeye <code>[−0,5, +0,5]</code> aralığına kırpılarak girer.</li>
<li><b>Öğrenme (Thompson Sampling, iki seviyeli).</b> Her strateji ve ton için ortalama ödül tahmini tutulur.
Görüşme içi gözlemler 2 kat ağırlıklıdır; benzer bağlamdaki (konu · gerilim) genel bilgi en fazla 6 gözlem ağırlığında ön bilgi olarak girer;
eski genel gözlemler her güncellemede %3 unutulur. Her turda her kol için <code>N(tahmin, σ/√n)</code> dağılımından bir örnek çekilir,
en yüksek örnek seçilir (σ = 0,25). Böylece az denenen kollar da arada denenir (keşif).</li>
<li><b>Yönetmen.</b> Churn riski üst üste 2 tur %75'in üzerindeyse asistana "müşteri kaçmak üzere" uyarısı geçer.</li>
<li><b>Simülasyonun gizli gerçeği (yalnızca test için).</b> Müşteri ajanının ruh hali her turda
<code>strateji etkisi + ton etkisi + 0,15×çözülen sorun + kampanya etkisi − sabır cezası + gürültü</code> kadar değişir.
Asistan bu tabloyu görmez; bu rapor, asistanın algısının bu gerçeğe ne kadar yaklaştığını ölçmek için onu da gösterir.</li>
</ol>"""


METHOD_LOKAL = """
<ol class="method">
<li><b>Ses → metin (faster-whisper, yerel).</b> Müşterinin ses kaydı bilgisayarda Whisper ile Türkçe metne çevrilir.</li>
<li><b>Metinden duygu (Türkçe BERT, yerel).</b> Mesaj cümlelere ve "ama/ancak/fakat" öncesine bölünür, her parça ayrı puanlanır
(olumlu olasılığı − olumsuz olasılığı), ortalanır ve ×0,6 ile ölçeklenir. Üzerine kelime ipuçları eklenir
(olumlu kelime başına +0,08, olumsuz kelime başına −0,08).</li>
<li><b>Sesten duygu (wav2vec2, yerel).</b> Model öfke / mutluluk / üzüntü / nötr olasılıkları verir; bunlar valens ve uyarılmışlığa
çevrilir. Model İngilizce oyunculu konuşmayla eğitildiği için Türkçede güvenilir değildir; bu yüzden valense yalnızca %10 katılır.</li>
<li><b>Akustik özellikler (yerel, numpy).</b> 40 ms'lik pencerelerde enerji (RMS → dB), otokorelasyonla perde (70–400 Hz),
perde dalgalanması ve konuşma hızı ölçülür; <code>0,4·enerji + 0,3·hız + 0,3·perde dalgalanması</code> ile akustik gerginlik çıkar.</li>
<li><b>Füzyon.</b> <code>valens = 0,9·metin + 0,1·ses modeli</code>, <code>uyarılma = 0,5·ses modeli + 0,5·akustik</code>,
<code>churn = sigmoid(−3·(valens + 0,2))</code>; "iptal", "başka operatöre geçeceğim" gibi açık ayrılma ifadesi varsa en az 0,85.
Konu ve ayrılma niyeti Türkçe anahtar kelime kurallarıyla bulunur.</li>
<li><b>Ödül.</b> Asistanın bir önceki hamlesi (strateji + ton) şöyle puanlanır:
<code>ödül = (yeni valens − önceki valens) + 0,15 × (o hamlede çözülen sorun sayısı)</code>.
Tek bir gürültülü ölçümün öğrenmeyi ele geçirmemesi için ödül öğrenmeye <code>[−0,5, +0,5]</code> aralığına kırpılarak girer.</li>
<li><b>Öğrenme (Thompson Sampling, iki seviyeli).</b> Her strateji ve ton için ortalama ödül tahmini tutulur.
Görüşme içi gözlemler 2 kat ağırlıklıdır; benzer bağlamdaki (konu · gerilim) genel bilgi en fazla 6 gözlem ağırlığında ön bilgi olarak girer;
eski genel gözlemler her güncellemede %3 unutulur. Her turda her kol için <code>N(tahmin, σ/√n)</code> dağılımından bir örnek çekilir,
en yüksek örnek seçilir (σ = 0,25). Böylece az denenen kollar da arada denenir (keşif).</li>
<li><b>Yönetmen.</b> Churn riski üst üste 2 tur %75'in üzerindeyse asistana "müşteri kaçmak üzere" uyarısı geçer.</li>
<li><b>Simülasyonun gizli gerçeği (yalnızca test için).</b> Müşteri ajanının ruh hali her turda
<code>strateji etkisi + ton etkisi + 0,15×çözülen sorun + kampanya etkisi − sabır cezası + gürültü</code> kadar değişir.
Asistan bu tabloyu görmez; bu rapor, asistanın algısının bu gerçeğe ne kadar yaklaştığını ölçmek için onu da gösterir.</li>
</ol>"""


def _kpi(label, value, hint=""):
    return f'<div class="kpi"><div class="l">{E(label)}</div><div class="v">{value}</div><div class="h">{E(hint)}</div></div>'


def _bar(v, scale=0.5):
    if v is None:
        return ""
    m = max(-scale, min(scale, v))
    w = abs(m) / scale * 50
    left = 50 if m >= 0 else 50 - w
    return (f'<div class="track"><div class="zero"></div><div class="fill {"pos" if m >= 0 else "neg"}" '
            f'style="left:{left:.1f}%;width:{w:.1f}%"></div></div>')


def to_html(rep: dict) -> str:
    m = rep["metrikler"]
    names = {**{k: v["ad"] for k, v in STRATEGIES.items()}, **{k: v["ad"] for k, v in TONES.items()}}

    kpis = "".join([
        _kpi("Gerçek ruh hali", f'{_f(m["gercek_baslangic"])} → {_f(m["gercek_son"])}', "simülasyonun gizli değeri"),
        _kpi("Algılanan valens", f'{_f(m["algilanan_baslangic"])} → {_f(m["algilanan_son"])}', "asistanın ölçtüğü"),
        _kpi("Algı hatası (MAE)", _f(m["algi_mae"], sign=False), f'ortalama sapma {_f(m["algi_sapma"])}'),
        _kpi("Algı–gerçek korelasyonu", _f(m["algi_korelasyon"]), "Pearson r (≥3 tur)"),
        _kpi("Yön uyumu", "–" if m["yon_uyumu"] is None else f'%{m["yon_uyumu"]*100:.0f}',
             f'{m["yon_uyumu_n"]} anlamlı değişimde doğru yön'),
        _kpi("Ortalama ödül", _f(m["ort_ham_odul"]), f'{m["kirpilan_odul"]} ödül kırpıldı'),
    ])

    # Öğrenilen vs gizli
    lv = ""
    for dim, title in (("strateji", "Söylem stratejisi"), ("ton", "Ses tonu")):
        d = rep["ogrenilen_vs_gizli"][dim]
        rows = "".join(
            f'<tr><td>{E(names[r["kol"]])}{" ★" if r["kol"] == d["en_iyi_gizli"] else ""}</td>'
            f'<td class="num">{_f(r["gizli"])}</td><td>{_bar(r["gizli"], 0.3)}</td>'
            f'<td class="num">{_f(r["tahmin"])}</td><td>{_bar(r["tahmin"])}</td>'
            f'<td class="num">{r["gorusme_n"]}</td><td class="num">{r["genel_n"]}</td></tr>' for r in d["satirlar"])
        ok = d["en_iyi_gizli"] == d["en_iyi_tahmin"]
        lv += (f'<h3>{title}</h3><p class="note">En iyi gizli kol: <b>{E(names[d["en_iyi_gizli"]])}</b> · '
               f'asistanın en iyi tahmini: <b>{E(names[d["en_iyi_tahmin"]])}</b> '
               f'{"<span class=ok>✔ eşleşti</span>" if ok else "<span class=bad>✖ eşleşmedi</span>"} · '
               f'sıra korelasyonu (Spearman): <b>{_f(d["sira_korelasyonu"])}</b> · bu görüşmede denenen kol: {d["denenen"]}/{len(d["satirlar"])}</p>'
               f'<div class="tw"><table><thead><tr><th>Kol</th><th colspan="2">Gizli etki / tur</th><th colspan="2">Asistanın tahmini ödülü</th>'
               f'<th>Görüşmede deneme</th><th>Genel gözlem</th></tr></thead><tbody>{rows}</tbody></table></div>')

    # Tur tur
    tblocks = ""
    for t in rep["turlar"]:
        mu, a = t["musteri"], t["asistan"]
        b = mu.get("ruh_hali_bilesenleri")
        comp = ""
        if b and b.get("eksik"):
            comp = ('<table class="mini"><tbody>'
                    f'<tr class="sum"><td>Toplam değişim</td><td class="num">{_f(b["toplam"], 3)}</td></tr>'
                    f'<tr><td>Ruh hali</td><td class="num">{_f(b["onceki"])} → {_f(b["sonraki"])}</td></tr></tbody></table>'
                    '<p class="note">Bu rapor kayıtlı görüşmeden üretildi; kayıtta değişimin bileşenleri '
                    '(strateji/ton etkisi, bonuslar, gürültü) yalnızca toplam olarak tutuluyor.</p>')
        elif b:
            parts = [("Strateji etkisi", b["strateji_etkisi"]), ("Ton etkisi", b["ton_etkisi"]),
                     ("Çözüm bonusu", b["cozum_bonusu"]), ("Kampanya etkisi", b["kampanya_etkisi"]),
                     ("Sabır cezası", b["sabir_cezasi"]), ("Gürültü", b["gurultu"])]
            comp = ('<table class="mini"><tbody>' + "".join(
                f'<tr><td>{k}</td><td class="num">{_f(v, 3)}</td></tr>' for k, v in parts if abs(v) > 1e-9) +
                f'<tr class="sum"><td>Toplam değişim</td><td class="num">{_f(b["toplam"], 3)}</td></tr>'
                f'<tr><td>Ruh hali</td><td class="num">{_f(b["onceki"])} → {_f(b["sonraki"])}{" (sınırda kırpıldı)" if b["kirpildi"] else ""}</td></tr>'
                '</tbody></table>')
        cust = (f'<div class="bubble cust"><div class="who">Müşteri · T{t["tur"]}'
                f'{" · " + E(ISSUES[mu["konu"]]["ad"]) if mu.get("konu") else ""}</div>{E(mu["metin"])}'
                f'<div class="meta">Gerçek ruh hali: <b>{_f(mu["gercek_ruh_hali"])}</b> ({E(mu["gercek_etiket"])}) · '
                f'{"<b>Model bozuk cevap verdi; hazır müşteri cümlesi kullanıldı.</b> · " if mu.get("yedek_cumle") else ""}'
                f'TTS tonlaması: <code>{E(mu["tts_tonu"])}</code>{" · <a href=" + chr(34) + E(mu["ses"]) + chr(34) + ">ses kaydı</a>" if mu.get("ses") else ""}</div>'
                + (f'<details><summary>Ruh hali neden değişti? (asistanın önceki hamlesinin etkisi)</summary>{comp}</details>' if comp else "")
                + '</div>')
        agent = ""
        if a:
            an = a["algi"]
            ak = an.get("akustik") or {}
            ac_rows = [("Süre", ak.get("sure_sn"), "sn"), ("Enerji", ak.get("enerji_db"), "dB"), ("Perde (medyan)", ak.get("perde_hz"), "Hz"),
                       ("Perde dalgalanması", ak.get("perde_dalgalanma"), ""), ("Konuşma hızı", ak.get("konusma_hizi"), "kelime/sn"),
                       ("Akustik gerginlik (0–1)", ak.get("akustik_uyarilma"), "")]
            ac = "".join(f'<tr><td>{k}</td><td class="num">{v} {u}</td></tr>' for k, v, u in ac_rows if v is not None)
            fusion = ""
            if an.get("uyarilma_gemini") is not None and ak.get("akustik_uyarilma") is not None:
                fusion = (f'<tr class="sum"><td>Uyarılma füzyonu</td><td class="num">0,6×{an["uyarilma_gemini"]:.2f} + 0,4×'
                          f'{ak["akustik_uyarilma"]:.2f} = <b>{an["uyarilma"]:.2f}</b></td></tr>')
            perc = (f'<div class="grid2"><div><h4>Algı ({E(an.get("kaynak", ""))})</h4><table class="mini"><tbody>'
                    f'<tr><td>Duygu</td><td class="num"><b>{E(an["duygu"])}</b></td></tr>'
                    f'<tr><td>Valens</td><td class="num"><b>{_f(an["valens"])}</b> (gerçek {_f(mu["gercek_ruh_hali"])}, hata {_f(a["algi_hatasi"])})</td></tr>'
                    f'<tr><td>Uyarılma</td><td class="num">{an["uyarilma"]:.2f}</td></tr>'
                    f'<tr><td>Churn riski</td><td class="num">%{an["churn_riski"]*100:.0f}</td></tr>'
                    f'<tr><td>Algılanan konu</td><td class="num">{E(ISSUES.get(an["konu"], {}).get("ad", "Diğer"))}</td></tr>{fusion}</tbody></table>'
                    f'<p class="cue"><b>Ses ipuçları:</b> {E(an.get("ses_ipuclari") or "–")}</p>'
                    f'<p class="cue"><b>Söz ipuçları:</b> {E(an.get("metin_ipuclari") or "–")}</p></div>'
                    f'<div><h4>Akustik özellikler</h4>'
                    + (f'<table class="mini"><tbody>{ac}</tbody></table>' if ac else '<p class="note">Bu turda ses kaydı yok (metin analizi).</p>')
                    + '</div></div>')
            o = a.get("odul")
            rew = ""
            if o:
                rew = (f'<h4>Önceki hamlenin ödülü: {E(names[o["strateji"]])} + {E(names[o["ton"]])}</h4>'
                       f'<p class="formula">({_f(o["yeni_valens"])} − {_f(o["onceki_valens"])}) + 0,15 × {o["cozulen_sorun"]} = '
                       f'{_f(o["duygu_degisimi"])} + {_f(o["ilerleme_bonusu"])} = <b>{_f(o["ham_odul"])}</b>'
                       + (f' → öğrenmeye <b>{_f(o["ogrenmeye_giren"])}</b> olarak girdi (kırpıldı)' if o["ham_odul"] != o["ogrenmeye_giren"] else "")
                       + '</p>')
            sec = a.get("secim")
            choice = ""
            if sec:
                def samples(dim):
                    ex = sec["ornekler"][dim]
                    pre = (a.get("beyin_secim_oncesi") or {}).get(dim, {})
                    if not ex:
                        return '<p class="note">Rastgele politika: örnek yok.</p>'
                    rows = "".join(
                        f'<tr class="{"sel" if k == sec[dim] else ""}"><td>{E(names[k])}</td>'
                        f'<td class="num">{_f(pre.get(k, {}).get("tahmin"))}</td><td class="num">±{pre.get(k, {}).get("belirsizlik", 0):.2f}</td>'
                        f'<td class="num">{pre.get(k, {}).get("gorusme_n", 0)}</td><td class="num"><b>{_f(v)}</b></td></tr>'
                        for k, v in sorted(ex.items(), key=lambda kv: -kv[1]))
                    return ('<table class="mini"><thead><tr><th>Kol</th><th>Tahmin</th><th>σ</th><th>n</th><th>Örnek</th></tr></thead>'
                            f'<tbody>{rows}</tbody></table>')
                choice = (f'<h4>Seçim: <b>{E(names[sec["strateji"]])}</b> + <b>{E(names[sec["ton"]])}</b></h4>'
                          f'<p class="note">{E(names[sec["strateji"]])}: {E(sec["neden"]["strateji"])}<br>'
                          f'{E(names[sec["ton"]])}: {E(sec["neden"]["ton"])}</p>'
                          + (f'<p class="director">⚠ Yönetmen: {E(a["yonetmen"])}</p>' if a.get("yonetmen") else "")
                          + f'<details><summary>Thompson örnekleri (en yüksek örnek seçilir)</summary><div class="grid2">'
                            f'<div><h5>Strateji</h5>{samples("strateji")}</div><div><h5>Ton</h5>{samples("ton")}</div></div></details>')
                tools = "".join(f'<li><code>{E(x["arac"])}</code>: {E(x["sonuc"])}{" <b>(çözdü: " + E(x["cozdu"]) + ")</b>" if x["cozdu"] else ""}'
                                f'{" <i>(model aracı çağırmadı; cevabın metninden çıkarıldı)</i>" if x.get("metinden") else ""}</li>'
                                for x in a.get("araclar") or [])
                choice += (f'<div class="bubble agent"><div class="who">Asistan</div>{E(a.get("cevap", ""))}'
                           + (f'<div class="director">Kural dışı söz yakalandı: {E(", ".join(a["kural_disi_soz"]))} '
                              '(ilgili cümle müşteriye söylenmeden çıkarıldı)</div>' if a.get("kural_disi_soz") else "")
                           + f'<div class="meta">İç ses: <i>{E(a.get("ic_ses", ""))}</i>'
                           f'{" · <a href=" + chr(34) + E(a["ses"]) + chr(34) + ">ses kaydı</a>" if a.get("ses") else ""}</div>'
                           + (f'<ul class="tools">{tools}</ul>' if tools else "") + '</div>')
            agent = f'<div class="analysis">{perc}{rew}{choice}</div>'
        tblocks += f'<section class="turn"><h3>Tur {t["tur"]}</h3>{cust}{agent}</section>'

    use = rep["metrikler"]["kullanim"]
    usage = " · ".join(f"{E(names[k])} ×{v}" for k, v in sorted(use["strateji"].items(), key=lambda kv: -kv[1])) or "–"
    usage_t = " · ".join(f"{E(names[k])} ×{v}" for k, v in sorted(use["ton"].items(), key=lambda kv: -kv[1])) or "–"

    return f"""<!doctype html><html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Duygu Raporu · {E(rep["musteri"])}</title><style>{CSS}</style></head><body><main>
<p class="crumb"><a href="/raporlar">← Tüm raporlar</a> · <a href="/raporlar/{rep["id"]}/html">HTML indir</a> · <a href="/raporlar/{rep["id"]}/csv">CSV indir</a> · <a href="/raporlar/{rep["id"]}/json">JSON</a></p>
<h1>Duygu analizi raporu: {E(rep["musteri"])}</h1>
<p class="sub">{E(rep["ozet_profil"])}<br>{"<b>Kayıtlı görüşmeden üretildi.</b> " if rep.get("kayittan") else ""}Görüşme <code>{rep["id"]}</code> · {E(rep["tarih"].replace("T", " "))} · mod: {E(rep["mod"])}{" (sesli)" if rep["sesli"] else ""}
 · bağlam: {E(rep["baglam"] or "–")} · {rep["asistan_turu"]} asistan turu · sonuç: <b class="{"ok" if rep["sonuc"] == "kaldi" else "bad"}">{SONUC.get(rep["sonuc"], rep["sonuc"])}</b>
 · çözülen: {len(rep["cozulen"])}/{len(rep["sorunlar"])} ({E(", ".join(rep["sorunlar"]))})</p>
<div class="kpis">{kpis}</div>
<section class="card"><h2>Duygu gidişatı</h2>
<div class="legend"><span><i class="l1"></i>Asistanın algıladığı valens</span><span><i class="l2"></i>Müşterinin gerçek (gizli) ruh hali</span></div>
{_svg_mood(rep)}<h2 style="margin-top:14px">Algılanan churn riski</h2>{_svg_churn(rep)}
<p class="note">Strateji kullanımı: {usage}<br>Ton kullanımı: {usage_t}</p></section>
<section class="card"><h2>Öğrenilen tercihler ve gizli tercihler</h2>{lv}</section>
<section class="card"><h2>Yöntem: duygu değişimini nasıl analiz ediyoruz?</h2>{METHOD_LOKAL if rep["mod"] == "lokal" else METHOD_GEMINI}</section>
<h2 class="sect">Tur tur ayrıntı</h2>{tblocks}
</main></body></html>"""


CSS = """
:root{color-scheme:light;--bg:#f4f3f0;--surface:#fcfcfb;--surface-2:#efeeea;--border:#dedcd6;--text:#0b0b0b;--text-2:#52514e;--muted:#8a8984;
--s1:#2a78d6;--s2:#eb6834;--pos:#2a78d6;--neg:#e34948;--good:#1a8f63;--warn:#b07800;--cust:#fbe9df;--agent:#e3eefb}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#121211;--surface:#1a1a19;--surface-2:#242422;--border:#33332f;
--text:#f0efec;--text-2:#c3c2b7;--muted:#8d8c84;--s1:#3987e5;--s2:#d95926;--pos:#3987e5;--neg:#e66767;--good:#2bb07f;--warn:#d9a000;--cust:#3a2519;--agent:#17283d}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#121211;--surface:#1a1a19;--surface-2:#242422;--border:#33332f;--text:#f0efec;--text-2:#c3c2b7;--muted:#8d8c84;
--s1:#3987e5;--s2:#d95926;--pos:#3987e5;--neg:#e66767;--good:#2bb07f;--warn:#d9a000;--cust:#3a2519;--agent:#17283d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1080px;margin:0 auto;padding:20px 16px 60px}a{color:var(--s1)}h1{font-size:24px;margin:6px 0}h2{font-size:15px;margin:0 0 10px}
h2.sect{margin:26px 0 10px}h3{font-size:14px;margin:4px 0 8px}h4{font-size:13px;margin:12px 0 6px;color:var(--text-2)}h5{font-size:12px;margin:4px 0;color:var(--text-2)}
.sub,.note,.crumb{color:var(--text-2)}.note{font-size:12.5px}.ok{color:var(--good)}.bad{color:var(--neg)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:16px 0}
.kpi,.card,.turn{background:var(--surface);border:1px solid var(--border);border-radius:12px}.kpi{padding:10px 12px}
.kpi .l{font-size:12px;color:var(--text-2)}.kpi .v{font-size:18px;font-weight:650;font-variant-numeric:tabular-nums}.kpi .h{font-size:11.5px;color:var(--muted)}
.card,.turn{padding:14px 16px;margin-bottom:14px}.chart{width:100%;height:auto;display:block}
.chart .grid{stroke:var(--border)}.chart .zero{stroke:var(--muted)}.chart .ax{font-size:11px;fill:var(--muted)}
.chart polyline{fill:none;stroke-width:2}.chart polyline.s1{stroke:var(--s1)}.chart polyline.s2{stroke:var(--s2)}.chart .dash{stroke-dasharray:5 4}
.chart circle.s1{fill:var(--s1);stroke:var(--surface);stroke-width:2}.chart circle.s2{fill:var(--s2);stroke:var(--surface);stroke-width:2}
.chart .bar{fill:var(--s1)}.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--text-2);margin-bottom:4px}
.legend i{display:inline-block;width:16px;height:3px;border-radius:2px;vertical-align:middle;margin-right:6px}.legend .l1{background:var(--s1)}
.legend .l2{background:repeating-linear-gradient(90deg,var(--s2) 0 5px,transparent 5px 8px)}
.tw{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:12.5px}th,td{padding:5px 8px;border-bottom:1px solid var(--border);text-align:left;vertical-align:middle}
th{color:var(--text-2);font-weight:600}td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
table.mini{width:auto;min-width:260px}tr.sum td{font-weight:600;border-top:1px solid var(--muted)}tr.sel td{background:var(--surface-2);font-weight:600}
.track{position:relative;height:12px;min-width:90px;background:var(--surface-2);border-radius:4px}.track .zero{position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:var(--muted)}
.track .fill{position:absolute;top:2px;bottom:2px;border-radius:3px}.track .pos{background:var(--pos)}.track .neg{background:var(--neg)}
.bubble{padding:10px 12px;border-radius:10px;margin:8px 0}.bubble.cust{background:var(--cust)}.bubble.agent{background:var(--agent)}
.bubble .who{font-size:12px;font-weight:600;color:var(--text-2);margin-bottom:3px}.bubble .meta{font-size:12px;color:var(--text-2);margin-top:6px}
.analysis{border-left:3px solid var(--border);padding-left:12px;margin-left:4px}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:700px){.grid2{grid-template-columns:1fr}}.cue{font-size:12.5px;margin:6px 0}
.formula{font-family:ui-monospace,Consolas,monospace;font-size:12.5px;background:var(--surface-2);padding:6px 10px;border-radius:6px;overflow-x:auto}
.director{border:1px solid var(--warn);border-radius:8px;padding:6px 10px;font-size:12.5px}details{margin-top:6px}summary{cursor:pointer;color:var(--text-2);font-size:12.5px}
ol.method li{margin-bottom:8px}code{font-size:12px;background:var(--surface-2);padding:1px 4px;border-radius:4px}ul.tools{margin:6px 0 0;padding-left:18px;font-size:12.5px}
"""


# ======================================================================== kaydet / listele
def save(c) -> Path:
    d = save_report_dict(build(c))
    print(f"[rapor] kaydedildi: {d}", flush=True)
    return d


def from_recording(rec: dict, started_at: float) -> dict:
    """Kayıtlı görüşmeden (data/kayitlar/<id>.json) rapor üretir: panelin aldığı tur verilerinden
    rapor olaylarını yeniden kurar. Eksik olan tek şey gizli ruh hali değişiminin bileşenleridir."""
    from types import SimpleNamespace

    from .learner import REWARD_CLIP

    issue_by_name = {v["ad"]: k for k, v in ISSUES.items()}
    events, resolved = [], set()
    prev_mood = prev_val = None
    prev_solved = 0
    last_brain, ctx, agent_turns = None, None, 0
    for e in rec["events"]:
        d = e["data"]
        if e["tip"] == "musteri":
            b = None
            if d.get("ruh_hali_degisimi") is not None and prev_mood is not None:
                b = {"eksik": True, "onceki": prev_mood, "toplam": d["ruh_hali_degisimi"], "sonraki": d["gercek_ruh_hali"],
                     "kirpildi": False}
                for k in ("strateji_etkisi", "ton_etkisi", "cozum_bonusu", "kampanya_etkisi", "sabir_cezasi", "gurultu"):
                    b[k] = None
            events.append({"tip": "musteri", "metin": d["metin"], "ses": d.get("ses"), "gercek_ruh_hali": d["gercek_ruh_hali"],
                           "gercek_etiket": d.get("gercek_etiket", ""), "tts_tonu": d.get("tts_tonu", ""), "konu": None,
                           "ruh_hali_bilesenleri": b, "bitti": d.get("bitti"), "sonuc": d.get("sonuc")})
            prev_mood = d["gercek_ruh_hali"]
            continue
        an = d["analiz"]
        v = float(an["valens"])
        odul = None
        if d.get("odul") and prev_val is not None:
            raw = d["odul"]["odul"]
            odul = {"strateji": d["odul"]["strateji"], "ton": d["odul"]["ton"], "onceki_valens": prev_val, "yeni_valens": v,
                    "duygu_degisimi": round(v - prev_val, 4), "cozulen_sorun": prev_solved,
                    "ilerleme_bonusu": round(0.15 * prev_solved, 4), "ham_odul": raw,
                    "ogrenmeye_giren": round(max(-REWARD_CLIP, min(REWARD_CLIP, raw)), 4)}
        ev = {"tip": "asistan", "algi": an, "gercek_ruh_hali": prev_mood, "algi_hatasi": round(v - prev_mood, 4),
              "odul": odul, "baglam": d.get("baglam")}
        ctx = d.get("baglam") or ctx
        last_brain = d.get("beyin") or last_brain
        if d.get("bitti"):
            ev["beyin_son"] = d.get("beyin")
        else:
            agent_turns += 1
            ev.update({"beyin_secim_oncesi": d.get("beyin"), "yonetmen": d.get("yonetmen"), "ses": d.get("ses"),
                       "secim": {"strateji": d["strateji"], "ton": d["ton"], "ornekler": d.get("ornekler") or {"strateji": {}, "ton": {}},
                                 "neden": d.get("neden") or {"strateji": "", "ton": ""}},
                       "cevap": d.get("metin", ""), "ic_ses": d.get("ic_ses", ""), "araclar": d.get("araclar") or []})
            prev_solved = 0
            for x in d.get("araclar") or []:
                if x.get("cozdu"):
                    prev_solved += 1
                    resolved.add(issue_by_name.get(x["cozdu"], x["cozdu"]))
        events.append(ev)
        prev_val = v

    fake = SimpleNamespace(
        id=rec["id"], events=events, persona_id=rec["persona"], started_at=started_at, mode=rec.get("mod", "gemini"),
        audio=bool(rec.get("ses")), policy="ogrenen", outcome=rec.get("sonuc"), agent_turns=agent_turns,
        resolved={r for r in resolved if r in ISSUES},
        learner=SimpleNamespace(ctx=ctx, snapshot=lambda: last_brain),
    )
    rep = build(fake)
    rep["kayittan"] = True
    return rep


def save_report_dict(rep: dict) -> Path:
    d = REPORT_DIR / rep["id"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "rapor.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (d / "rapor.csv").write_text(to_csv(rep), encoding="utf-8")
    (d / "rapor.html").write_text(to_html(rep), encoding="utf-8")
    return d


def ensure_from_recordings(rec_dir: Path) -> int:
    """Raporu olmayan kayıtlı görüşmeler için rapor üretir. Üretilen rapor sayısını döner."""
    n = 0
    for f in rec_dir.glob("*.json"):
        if (REPORT_DIR / f.stem / "rapor.html").exists():
            continue
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
            save_report_dict(from_recording(rec, f.stat().st_mtime))
            n += 1
            print(f"[rapor] kayıtlı görüşmeden üretildi: {REPORT_DIR / f.stem}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[rapor] {f.name} için rapor üretilemedi: {e}", flush=True)
    return n


def list_reports() -> list[dict]:
    out = []
    for f in REPORT_DIR.glob("*/rapor.json"):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        out.append(r)
    return sorted(out, key=lambda r: r["tarih"], reverse=True)


def index_html() -> str:
    reps = list_reports()
    names = {**{k: v["ad"] for k, v in STRATEGIES.items()}, **{k: v["ad"] for k, v in TONES.items()}}
    rows = "".join(
        f'<tr><td><a href="/raporlar/{r["id"]}">{E(r["tarih"].replace("T", " "))}</a></td><td>{E(r["musteri"])}</td>'
        f'<td>{E(r["mod"])}{" · sesli" if r["sesli"] else ""}</td><td class="{"ok" if r["sonuc"] == "kaldi" else "bad"}">{SONUC.get(r["sonuc"], r["sonuc"])}</td>'
        f'<td class="num">{r["asistan_turu"]}</td><td class="num">{_f(r["metrikler"]["gercek_baslangic"])} → {_f(r["metrikler"]["gercek_son"])}</td>'
        f'<td class="num">{_f(r["metrikler"]["algi_mae"], sign=False)}</td><td class="num">{_f(r["metrikler"]["algi_korelasyon"])}</td>'
        f'<td class="num">{"–" if r["metrikler"]["yon_uyumu"] is None else "%" + format(r["metrikler"]["yon_uyumu"] * 100, ".0f")}</td>'
        f'<td class="num">{_f(r["metrikler"]["ort_ham_odul"])}</td>'
        f'<td><a href="/raporlar/{r["id"]}/html">HTML</a> · <a href="/raporlar/{r["id"]}/csv">CSV</a></td></tr>' for r in reps)

    # Toplu istatistik
    agg = ""
    if reps:
        maes = [r["metrikler"]["algi_mae"] for r in reps if r["metrikler"]["algi_mae"] is not None]
        all_per, all_true = [], []
        arm_rewards: dict[str, list[float]] = {}
        persona = {}
        for r in reps:
            persona.setdefault(r["musteri"], []).append(r["sonuc"] == "kaldi")
            for t in r["turlar"]:
                a = t.get("asistan")
                if a:
                    all_per.append(a["algi"]["valens"])
                    all_true.append(t["musteri"]["gercek_ruh_hali"])
                    if a.get("odul"):
                        for k in (a["odul"]["strateji"], a["odul"]["ton"]):
                            arm_rewards.setdefault(k, []).append(a["odul"]["ham_odul"])
        corr = _pearson(all_per, all_true)
        arms = "".join(f'<tr><td>{E(names[k])}</td><td class="num">{len(v)}</td><td class="num">{_f(float(np.mean(v)))}</td><td>{_bar(float(np.mean(v)))}</td></tr>'
                       for k, v in sorted(arm_rewards.items(), key=lambda kv: -np.mean(kv[1])))
        pers = " · ".join(f"{E(k)}: %{100 * np.mean(v):.0f} kaldı ({len(v)})" for k, v in persona.items())
        agg = (f'<div class="kpis">{_kpi("Görüşme", str(len(reps)))}'
               f'{_kpi("Ortalama algı hatası", _f(float(np.mean(maes)), sign=False) if maes else "–", "MAE, valens birimi")}'
               f'{_kpi("Algı–gerçek korelasyonu", _f(corr), f"{len(all_per)} turun tamamı")}'
               f'{_kpi("Müşteriyi tutma", "%" + format(100 * np.mean([r["sonuc"] == "kaldi" for r in reps]), ".0f"))}</div>'
               f'<p class="note">{pers}</p>'
               f'<section class="card"><h2>Strateji ve tonların gözlenen ortalama ödülü (tüm görüşmeler)</h2>'
               f'<div class="tw"><table><thead><tr><th>Kol</th><th>Gözlem</th><th>Ort. ödül</th><th></th></tr></thead><tbody>{arms}</tbody></table></div></section>')
    body = (f'<div class="tw"><table><thead><tr><th>Tarih</th><th>Müşteri</th><th>Mod</th><th>Sonuç</th><th>Tur</th><th>Gerçek ruh hali</th>'
            f'<th>Algı MAE</th><th>Korelasyon</th><th>Yön uyumu</th><th>Ort. ödül</th><th></th></tr></thead><tbody>{rows}</tbody></table></div>'
            if reps else '<p class="note">Henüz rapor yok. Panelden bir görüşme yapın; bittiğinde rapor otomatik oluşur.</p>')
    return f"""<!doctype html><html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Duygu Raporları</title><style>{CSS}</style></head><body><main>
<p class="crumb"><a href="/">← Panele dön</a></p><h1>Duygu analizi raporları</h1>
<p class="sub">Her görüşme bittiğinde ayrıntılı rapor otomatik oluşturulur (data/raporlar/). Kayıtlı görüşmelerin raporları da
otomatik üretilir. Hızlı simülasyon görüşmeleri rapora girmez.</p>
{agg}<section class="card"><h2>Görüşmeler</h2>{body}</section></main></body></html>"""
