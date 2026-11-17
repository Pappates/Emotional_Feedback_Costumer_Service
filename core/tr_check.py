"""Türkçe yazım denetimi: biçimbilimsel çözümleme (zeyrek / Zemberek kuralları) + kelime sıklığı (wordfreq).

Küçük dil modelleri zaman zaman Türkçede olmayan kelime biçimleri üretiyor ("oldığınızı", "kaybetiyorum",
"faturalımdaki"). Bir kelime ancak İKİ denetimden de geçemezse bozuk sayılır: zeyrek çözümleyemiyor VE Türkçe
kelime sıklığı listesinde hiç geçmiyor. (zeyrek tek başına "edeceğim", "düzelir" gibi kelimeleri tanımıyordu: 784
kelimelik doğru metinde 4 yanlış alarm; sıklık listesi tek başına ise "yönlendiriyorum" gibi nadir ama doğru çekimleri
tanımıyor.) Bozuk kelimeli cümle yeniden ürettirilir ya da söylenmez. Dilbilgisi olarak geçerli ama anlamsız yapıları ("neyin nesine") yakalamaz.

zeyrek kurulu değilse denetim sessizce devre dışı kalır (uygulama çalışmaya devam eder).
Kapatmak için: LOKAL_YAZIM_DENETIMI=0
"""
from __future__ import annotations

import os
import re
import threading

ENABLED = os.getenv("LOKAL_YAZIM_DENETIMI", "1") != "0"
_LOCK = threading.Lock()
_AN = None
_FAILED = False
_CACHE: dict[str, bool] = {}

# Alan terimleri, kısaltmalar, özel adlar ve sözlükte olmayan yaygın günlük kelimeler
KNOWN = {
    "pos", "tl", "gb", "mb", "mbps", "mbit", "sms", "fiber", "modem", "internet", "vdsl", "adsl", "wifi", "router",
    "deniz", "ahmet", "zeynep", "selin", "mert", "bey", "hanım", "rv", "tt", "ft", "tamamdır", "hmm", "hı", "ha",
    "valla", "vallahi", "yahu", "ya", "be", "ee", "eh", "of", "off", "ay", "aa", "tabii", "tabi", "peki", "okey",
    "online", "app", "uygulamadan", "taahhüt", "taahhüdüm", "taahhüdünüz", "mesaj", "operatör", "ok",
}
_WORD = re.compile(r"[A-Za-zÇĞİÖŞÜçğıöşüâîû]+")


def _analyzer():
    global _AN, _FAILED
    if _AN is not None or _FAILED or not ENABLED:
        return _AN
    with _LOCK:
        if _AN is None and not _FAILED:
            try:
                import logging
                logging.getLogger("zeyrek").setLevel(logging.ERROR)
                import zeyrek
                _AN = zeyrek.MorphAnalyzer()
            except Exception as e:  # noqa: BLE001
                _FAILED = True
                print(f"[lokal] Türkçe yazım denetimi kapalı (zeyrek yüklenemedi: {str(e)[:80]})", flush=True)
    return _AN


_FREQ = None


def _freq(w: str) -> float:
    """Türkçe kelime sıklığı (Zipf ölçeği, 0 = listede yok). wordfreq kurulu değilse 1 (yani denetimi geçer)."""
    global _FREQ
    if _FREQ is None:
        try:
            from wordfreq import zipf_frequency
            _FREQ = lambda x: zipf_frequency(x, "tr")  # noqa: E731
        except Exception:  # noqa: BLE001
            _FREQ = lambda x: 1.0  # noqa: E731
    return _FREQ(w)


_FRONT, _BACK = set("eiöü"), set("aıou")
_FUT = re.compile(r"^(.+?)(y?)([ae])c\3[kğ].*$")


def _future_ok(an, w: str) -> bool:
    """zeyrek gelecek zaman çekimlerinin bir kısmını tanımıyor ("çözülecektir", "inceleyecek"). Kök geçerli bir fiilse ve
    ek ünlü uyumuna uyuyorsa ("çözül-ecek" doğru, "düzel-acak" yanlış) kelime doğru sayılır."""
    m = _FUT.match(w)
    if not m:
        return False
    stem, vowel = m.group(1), m.group(3)
    last = next((c for c in reversed(stem) if c in _FRONT | _BACK), None)
    if last is None or (vowel == "e") != (last in _FRONT):
        return False
    inf = stem + ("mek" if last in _FRONT else "mak")
    try:
        return bool(an._parse(inf)) or _freq(inf) > 0
    except Exception:  # noqa: BLE001
        return False


_NEC = re.compile(r"^(.+?)m([ae])l[ıi].*$")          # gereklilik: bul-malı-sınız
_LIK = re.compile(r"^(.+?)(l[ıiuü]k|l[ıiuü]ğ).*$")    # yapım eki: ilgisiz-lik-le


def _known(an, w: str) -> bool:
    try:
        return bool(an._parse(w)) or _freq(w) > 0
    except Exception:  # noqa: BLE001
        return False


def _suffix_ok(an, w: str) -> bool:
    """zeyrek'in tanımadığı bazı çekim/yapım ekleri: kök geçerliyse ve ünlü uyumu doğruysa kelime doğru sayılır."""
    m = _NEC.match(w)
    if m:
        stem, v = m.group(1), m.group(2)
        last = next((c for c in reversed(stem) if c in _FRONT | _BACK), None)
        if last and (v == "e") == (last in _FRONT) and _known(an, stem + ("mek" if v == "e" else "mak")):
            return True
    m = _LIK.match(w)
    if m and len(m.group(1)) >= 3 and _known(an, m.group(1)):
        return True
    return False


def _lower(w: str) -> str:
    return w.replace("I", "ı").replace("İ", "i").lower()


def bad_words(text: str) -> list[str]:
    """Türkçe olarak çözümlenemeyen kelimeler. Kısaltma (tamamı büyük harf), tek harf ve bilinen terimler atlanır.
    Kesme işaretinden sonraki ek ("POS'u") ayrı kelime sayılmaz."""
    an = _analyzer()
    if an is None:
        return []
    text = re.sub(r"(\w)['’]\w+", r"\1", text)
    out = []
    for w in _WORD.findall(text):
        if len(w) < 2 or (w.isupper() and len(w) <= 6):
            continue
        lw = _lower(w)
        if lw in KNOWN:
            continue
        ok = _CACHE.get(lw)
        if ok is None:
            try:
                ok = bool(an._parse(lw))
            except Exception:  # noqa: BLE001
                ok = True
            if not ok:
                ok = _freq(lw) > 0 or _future_ok(an, lw) or _suffix_ok(an, lw)
            _CACHE[lw] = ok
        if not ok:
            out.append(w)
    return out


def warmup():
    _analyzer()
    bad_words("Merhaba, internetim çalışmıyor.")
