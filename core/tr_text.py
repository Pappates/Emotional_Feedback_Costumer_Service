"""Seslendirme öncesi Türkçe metin normalleştirme.

TTS modelleri rakam, saat, yüzde, para birimi ve kısaltmaları ya hiç okuyamıyor ya da İngilizce/harf harf okuyor
("16:00", "%30", "240 TL", "50 GB", "Mbps'e"). Bu modül bunları konuşma diline çevirir:

    "Teknisyen 16:00-18:00 arası gelecek, 240 TL iade ve %30 indirim tanımladım."
 -> "Teknisyen on altı ile on sekiz arası gelecek, iki yüz kırk lira iade ve yüzde otuz indirim tanımladım."

Rakamdan sonra kesme işaretiyle gelen ekler korunur: "10'dan" -> "ondan", "16'sı" -> "on altısı", "100'e" -> "yüze".
Aynı fonksiyon, seslendirme kalitesini ölçerken Whisper dökümünü de normalleştirmek için kullanılır.
"""
from __future__ import annotations

import re

_ONES = ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
_TENS = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]


def _below_thousand(n: int) -> list[str]:
    words = []
    h, rest = divmod(n, 100)
    if h:
        words += (["yüz"] if h == 1 else [_ONES[h], "yüz"])
    t, o = divmod(rest, 10)
    if t:
        words.append(_TENS[t])
    if o:
        words.append(_ONES[o])
    return words


def number_to_words(n: int) -> str:
    """Türkçe sayı okunuşu (0 - 999 milyar): 240 -> 'iki yüz kırk', 1000 -> 'bin', 2025 -> 'iki bin yirmi beş'."""
    if n == 0:
        return "sıfır"
    if n < 0:
        return "eksi " + number_to_words(-n)
    words = []
    for value, name in ((10**9, "milyar"), (10**6, "milyon"), (1000, "bin")):
        q, n = divmod(n, value)
        if q:
            if value == 1000 and q == 1:
                words.append("bin")          # "bir bin" değil "bin"
            else:
                words += _below_thousand(q) + [name]
    words += _below_thousand(n)
    return " ".join(words)


def _num(s: str) -> str:
    s = s.replace(".", "") if re.fullmatch(r"\d{1,3}(\.\d{3})+", s) else s   # 1.500 -> 1500
    if "," in s:
        a, b = s.split(",", 1)
        return f"{number_to_words(int(a))} virgül {number_to_words(int(b))}"
    return number_to_words(int(s))


# Kısaltmalar ve birimler (okunuş). Birimden sonra kesme işaretli ek gelebilir: "Mbps'e" -> "megabite"
_UNITS = {
    "tl": "lira", "₺": "lira", "gb": "gigabayt", "mb": "megabayt", "tb": "terabayt",
    "mbps": "megabit", "mbit": "megabit", "gbps": "gigabit", "kbps": "kilobit", "dk": "dakika", "sn": "saniye", "km": "kilometre",
}
_ABBR = {"POS": "pos", "SMS": "es em es", "ADSL": "a de es el", "VDSL": "ve de es el", "IP": "ay pi",
         "Wi-Fi": "vay fay", "WiFi": "vay fay", "wifi": "vay fay", "AVM": "a ve me", "vb.": "ve benzeri", "vs.": "ve saire"}


def normalize_tr(text: str) -> str:
    t = text
    for k, v in _ABBR.items():
        t = re.sub(rf"(?<![\w]){re.escape(k)}(?![\w])", v, t)
    # Saat aralığı: 16:00-18:00 / 16.00 - 18.00
    t = re.sub(r"\b(\d{1,2})[:.](\d{2})\s*[-–]\s*(\d{1,2})[:.](\d{2})\b", r"\1:\2 ile \3:\4", t)

    def clock(m):
        h, mm, suf = int(m.group(1)), int(m.group(2)), m.group(3) or ""
        w = number_to_words(h) + ("" if mm == 0 else " " + number_to_words(mm))
        return w + suf
    t = re.sub(r"\b(\d{1,2})[:.](\d{2})(?!\d)(?:'(\w+))?", clock, t)
    # Yüzde: %30, % 30, %30'luk
    t = re.sub(r"%\s*(\d+(?:,\d+)?)(?:'(\w+))?", lambda m: "yüzde " + _num(m.group(1)) + (m.group(2) or ""), t)
    # Aralık: 5-10 Mbps -> beş ile on Mbps
    t = re.sub(r"\b(\d+)\s*[-–]\s*(\d+)\b", lambda m: f"{_num(m.group(1))} ile {_num(m.group(2))}", t)
    # Sayı + birim (+ ek): 240 TL, 50 GB'lık, 100 Mbps'e
    unit_re = "|".join(sorted((re.escape(u) for u in _UNITS), key=len, reverse=True))
    t = re.sub(rf"\b(\d+(?:[.,]\d+)?)\s*({unit_re})(?:'(\w+))?(?![\w])",
               lambda m: f"{_num(m.group(1))} {_UNITS[m.group(2).lower()]}{m.group(3) or ''}", t, flags=re.I)
    # Sayı + kesme işaretli ek: 10'dan, 16'sı
    t = re.sub(r"\b(\d+(?:,\d+)?)'(\w+)", lambda m: _num(m.group(1)) + m.group(2), t)
    # Sayıdan sonra boşluksuz gelen birim olmayan rakamlar ve yalın sayılar
    t = re.sub(r"\b\d{1,3}(?:\.\d{3})+(?:,\d+)?\b|\b\d+(?:,\d+)?\b", lambda m: _num(m.group(0)), t)
    # Sayısız kalan birimler: "Mbps'e düştü", "TL"
    t = re.sub(rf"(?<![\w])({unit_re})(?:'(\w+))?(?![\w])",
               lambda m: _UNITS[m.group(1).lower()] + (m.group(2) or ""), t, flags=re.I)
    t = re.sub(r"[#*_~`<>|]", " ", t)
    return re.sub(r"\s+", " ", t).strip()
