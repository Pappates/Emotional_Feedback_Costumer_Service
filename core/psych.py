"""Psikolojik katman: valens/uyarılma ölçümünü değerlendirme (appraisal) kuramıyla yorumlar.

Kuramsal temel
* Boyutsal model (Russell 1980; Mehrabian PAD 1996): valens, uyarılma ve BASKINLIK. Baskınlık, kişinin durum
  üzerinde kendini ne kadar güçlü/kontrol sahibi hissettiğidir: talepkâr, sert müşteri (yüksek) ile çaresiz müşteri
  (düşük) aynı valensle bambaşka davranır.
* Değerlendirme kuramı (Scherer, Bileşen Süreç Modeli 2001; Lazarus 1991): duygu, olayın nasıl yorumlandığından
  doğar. Çağrı merkezinde belirleyici dört değerlendirme:
    - suç yükleme (kim sorumlu?): kuruma yüklenen suç öfkeyi ve ayrılma niyetini besler
    - belirsizlik (ne olacağını biliyor mu?): kaygının kaynağı; net bilgi ve süre verilince azalır
    - kontrol / başa çıkma gücü: kontrol kaybı çaresizlik ve küskünlüğe dönüşür; somut çözüm geri verir
    - haksızlık algısı (norm ihlali): "hak etmedim" duygusu; telafi ve kabul ile onarılır
* Çağrı merkezi kategorileri bu değerlendirmelerden türetilir (örn. öfke = olumsuz + yüksek uyarılma + kuruma suç;
  kaygı = olumsuz + belirsizlik + düşük baskınlık; küskünlük = olumsuz + düşük uyarılma + düşük baskınlık).
  Küskünlük sessiz churn sinyalidir: müşteri bağırmaz ama vazgeçmiştir.

Bu modül Aşama 1'de Türkçe sözcük kalıplarıyla çalışır (CPU'da milisaniyeler). Aşama 2'de ince ayarlı modelin
baskınlık çıktısı ve etiketli veriyle doğrulanacak.
"""
from __future__ import annotations

import re

_LOW = str.maketrans("İIŞĞÜÖÇ", "iışğüöç")


def _low(s: str) -> str:
    return (s or "").translate(_LOW).lower()


BLAME_ORG = [r"\bsiz(in|e|i|den)?\b", r"şirketiniz", r"operatör", r"firmanız", r"\byine\b", r"\bhep (böyle|aynı)",
             r"her seferinde", r"bir türlü", r"rezalet", r"saçmalık", r"(nasıl|ne biçim) (bir )?(hizmet|şirket)"]
BLAME_SELF = [r"benim hatam", r"ben yanlış", r"yanlışlıkla ben", r"ben unuttum", r"bilmeden"]
UNCERTAIN = [r"ne zaman", r"\bneden\b", r"\bniye\b", r"nasıl olacak", r"belli değil", r"bilmiyorum", r"emin değil",
             r"anlamadım", r"\bne demek\b", r"nereden çıktı", r"neyin nesi", r"ne olacak"]
LOW_CONTROL = [r"ne yapacağımı bilmiyorum", r"elimden (bir şey|hiçbir şey) gelmiyor", r"çaresiz", r"mecbur",
               r"bekliyorum", r"bekleyip duruyorum", r"iş(ler)?im durdu", r"kaybediyorum", r"yapamıyorum", r"deliye döndüm"]
UNFAIR = [r"haksızlık", r"kullanmadım", r"hak etmiyorum", r"neden (ben )?ödeyeyim", r"hiç (almadım|istemedim)",
          r"(ücret|para|fatura|paket)\w*[^.!?]{0,30}(neden|niye|haksız|kullanmadı)", r"kandır", r"\bhile\b"]
DEMAND = [r"\bhemen\b", r"\bderhal\b", r"istiyorum", r"\b(yapın|halledin|düzeltin|gönderin|bağlayın)\b",
          r"\bşimdi\b", r"bekleyemem", r"kabul etmiyorum", r"\bmutlaka\b"]
RESIGNED = [r"(boş ?ver|neyse|ne fark eder|zaten|artık bir şey beklemiyorum|umudum kalmadı|bıktım|yoruldum)"]

# Değerlendirme -> önerilen yaklaşım (bandit kollarıyla aynı adlar: catalog.STRATEGIES)
_ADVICE = {
    "suc": ("cozum_odakli", "Suç kuruma yükleniyor: savunmaya geçmeden sorumluluğu üstlen, hemen somut adım at."),
    "belirsizlik": ("teknik_aciklama", "Belirsizlik yüksek: ne olacağını ve ne zaman olacağını net söyle."),
    "kontrol": ("cozum_odakli", "Kontrol kaybı var: müşteriye somut bir çözüm ve sonraki adımı ver."),
    "haksizlik": ("telafi_teklifi", "Haksızlık algısı var: hatayı kabul et, telafi sun."),
    "kuskunluk": ("empati", "Küskünlük (sessiz churn riski): önce dinlediğini ve önemsediğini göster."),
}


def _score(t: str, pats: list[str], per: float = 0.34) -> tuple[float, list[str]]:
    hits = [p for p in pats if re.search(p, t)]
    return min(1.0, per * len(hits)), hits


def assess(text: str, valence: float, arousal: float, acoustic: dict | None = None) -> dict:
    t = _low(text)
    blame_org, h1 = _score(t, BLAME_ORG)
    blame_self, _ = _score(t, BLAME_SELF, 0.5)
    unc, h2 = _score(t, UNCERTAIN, 0.4)
    unc = min(1.0, unc + 0.1 * t.count("?"))
    low_ctrl, h3 = _score(t, LOW_CONTROL, 0.45)
    unfair, h4 = _score(t, UNFAIR, 0.45)
    demand, h5 = _score(t, DEMAND, 0.3)
    resigned, h6 = _score(t, RESIGNED, 0.5)

    if valence > 0.1:          # olumlu konuşmada suç/haksızlık kalıpları ('sizin sayenizde') anlam taşımaz
        blame_org *= 0.3; unfair *= 0.3
    # Baskınlık (−1..+1): talep ve sertlik artırır; çaresizlik, belirsizlik ve vazgeçmişlik azaltır; ses yüksekliği katkı
    dom = 0.5 * demand + 0.25 * blame_org - 0.45 * low_ctrl - 0.25 * unc - 0.4 * resigned
    e = (acoustic or {}).get("enerji_db")
    if e is not None:
        dom += 0.15 * max(-1.0, min(1.0, (e + 19) / 6))
    dom = max(-1.0, min(1.0, dom))

    suc = "kurum" if blame_org >= 0.34 and blame_org > blame_self else ("kendisi" if blame_self > 0 else "durum")
    if valence >= 0.15:
        kat = "memnuniyet" if valence >= 0.5 else "rahatlama"
    elif resigned >= 0.5 or (valence < -0.2 and arousal < 0.45 and dom < -0.1):
        kat = "küskünlük"
    elif valence < -0.35 and arousal >= 0.55 and (suc == "kurum" or dom > 0.2):
        kat = "öfke"
    elif valence < -0.1 and unc >= 0.4 and dom <= 0.1:
        kat = "kaygı"
    elif unc >= 0.5 and valence >= -0.25:
        kat = "kafa karışıklığı"
    elif valence < -0.15:
        kat = "hayal kırıklığı"
    else:
        kat = "nötr"

    # En baskın olumsuz değerlendirme -> öneri
    cand = {"suc": blame_org if suc == "kurum" else 0.0, "belirsizlik": unc, "kontrol": low_ctrl, "haksizlik": unfair}
    if kat == "küskünlük":
        key = "kuskunluk"
    else:
        key = max(cand, key=cand.get) if max(cand.values()) >= 0.34 and valence < 0.15 else None
    oneri = _ADVICE.get(key) if key else None
    gerekce = []
    if suc == "kurum":
        gerekce.append("suçu kuruma yüklüyor")
    if unc >= 0.4:
        gerekce.append("belirsizlik yüksek")
    if low_ctrl >= 0.45:
        gerekce.append("kontrol kaybı/çaresizlik")
    if unfair >= 0.45:
        gerekce.append("haksızlık algısı")
    if demand >= 0.6:
        gerekce.append("talepkâr")
    if resigned >= 0.5:
        gerekce.append("vazgeçmişlik")
    return {
        "kategori": kat, "baskinlik": round(dom, 2), "suc_yukleme": suc,
        "belirsizlik": round(unc, 2), "kontrol_kaybi": round(low_ctrl, 2), "haksizlik": round(unfair, 2),
        "talep": round(demand, 2), "vazgecmislik": round(resigned, 2),
        "gerekce": ", ".join(gerekce) or "belirgin değerlendirme yok",
        "oneri_strateji": oneri[0] if oneri else None, "oneri": oneri[1] if oneri else None,
    }
