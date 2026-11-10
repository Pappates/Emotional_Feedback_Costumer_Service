"""Bir görüşmenin yönetimi: simüle müşteri ajanı <-> öğrenen çağrı merkezi ajanı.

Tur sırası:  müşteri açılış -> [asistan -> müşteri] x N

  Asistan turu: müşterinin son konuşmasını DİNLER (ses -> duygu analizi), önceki hamlesinin
  ödülünü hesaplar ve öğrenir, sonra yeni strateji+ton seçip cevap verir ve gerekirse araç çalıştırır.
  Müşteri turu: ruh hali asistanın hamlesine göre değişir (gizli tercihler), sonra bunu
  sözleriyle ve ses tonuyla ifade eder.
"""
from __future__ import annotations

import random
import re
import uuid
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from . import config, emotion, llm
from .catalog import ISSUES, PERSONAS, STRATEGIES, TONES, TOOLS, mood_delivery, mood_label
from .learner import REWARD_CLIP, CallLearner, GlobalModel

PROGRESS_WEIGHT = 0.15
RESOLVE_MOOD_BONUS = 0.15
STAY_THRESHOLD = 0.3
TOOL_NAMES = tuple(TOOLS)


class AgentReply(BaseModel):
    # Sıra bilinçli: önce düşün, sonra işlem kararı ver, en son konuş (küçük modellerde tutarlılığı artırıyor)
    ic_ses: str = Field(description="Kısa iç ses: müşteri hakkında ne düşünüyorsun, neden bu şekilde cevap veriyorsun (1 cümle)")
    araclar: list[Literal[TOOL_NAMES]] = Field(default_factory=list, description="Bu turda çalıştırılacak araçlar (0-2)")
    cevap: str = Field(description="Müşteriye söyleyeceğin Türkçe cevap (en fazla 3 kısa cümle)")


class LocalAgentReply(BaseModel):
    """Lokal (küçük) model için aynı şema, uzunluk sınırlı: uzun 'iç ses' JSON'u yarıda kesiyordu."""
    ic_ses: str = Field(max_length=150, description="Tek kısa cümle: neden böyle cevap veriyorsun")
    araclar: list[Literal[TOOL_NAMES]] = Field(default_factory=list, max_length=2, description="Çalıştırılacak araçlar (0-2)")
    cevap: str = Field(max_length=330, description="Müşteriye söyleyeceğin Türkçe cevap (en fazla 3 kısa cümle)")


class LocalCustomerReply(BaseModel):
    konusma: str = Field(max_length=300, description="Müşterinin söyleyeceği 1-3 kısa Türkçe cümle")


class CustomerReply(BaseModel):
    konusma: str = Field(description="Müşterinin telefonda söyleyeceği Türkçe cümleler (1-3 cümle)")


# ---------------------------------------------------------------- demo (mock) metinleri
MOCK_CUSTOMER = {
    "internet_kesintisi": [
        "İki gündür internetim yok! Dükkânda POS çalışmıyor, müşteri kaçırıyorum. Bu nasıl iş?",
        "Hâlâ internet yok, ne zaman düzelecek bunu bilmek istiyorum.",
        "Tamam, kaydı açtınız ama gerçekten bugün gelecekler mi?",
    ],
    "hiz_sorunu": [
        "Akşamları internet hızı yerlere düşüyor, toplantılarım kopuyor. Ödediğim hızın yarısını bile alamıyorum.",
        "Hız hâlâ düşük. Gerçek sorun ne, onu söyleyin bana.",
        "Peki bu ayar kalıcı mı, yarın akşam yine düşmeyecek değil mi?",
    ],
    "fatura_itirazi": [
        "Bu ayki faturam neden bu kadar yüksek geldi? Kullanmadığım bir ücret yansıtılmış.",
        "Faturadaki o ücretin ne olduğunu hâlâ anlamadım.",
        "İade yapılacaksa ne zaman yansıyacak?",
    ],
    "nakil": [
        "Bir de taşınıyorum, hattımı yeni adrese aldırmam lazım. Bu iş ne kadar sürer?",
        "Nakil işlemini başlatabilir misiniz, vaktim çok kısıtlı.",
        "Yeni adreste hat ne zaman aktif olur?",
    ],
}
MOCK_CHURN = "Yeter artık! Hattımı iptal ettirip başka operatöre geçeceğim."
MOCK_STAY = "Tamam, teşekkür ederim, ilginiz için sağ olun. İyi günler."
MOCK_AGENT = {
    "empati": "Yaşadığınız durum için gerçekten çok üzgünüm, ne kadar zor olduğunu anlıyorum. Hemen ilgileniyorum.",
    "cozum_odakli": "Hemen işlemi başlatıyorum, kaydınız şu an oluşturuldu ve size net bir süre veriyorum.",
    "teknik_aciklama": "Kontrol ettim, sorun bölgenizdeki santral portundan kaynaklanıyor. Şimdi adım adım düzeltiyoruz.",
    "telafi_teklifi": "Yaşadığınız mağduriyet için hesabınıza telafi kampanyası tanımlıyorum.",
    "dinle_sor": "Anlıyorum. Sorunu tam netleştirmek için sorayım: ne zamandan beri bu şekilde?",
}
# Demo modunda asistanın bir turda sorunu çözme olasılığı (stratejiye göre)
MOCK_RESOLVE_P = {"cozum_odakli": 0.55, "teknik_aciklama": 0.5, "dinle_sor": 0.4, "empati": 0.35, "telafi_teklifi": 0.3}


# ---------------------------------------------------------------- küçük model korumaları
AGENT_PHRASES = ["size nasıl yardımcı", "yardımcı olabilirim", "yardımcı olmak istiyorum", "yardımcı olabilecek",
                 "kaydı açacağım", "kaydınızı açıyorum", "randevu oluşturacağım", "iade ediyorum", "değerli müşterimiz",
                 "müşterilerimiz", "temsilciniz", "size en uygun çözüm"]
TOOL_HINTS = {  # lokal modda model aracı çağırmayı unutup "açıyorum" derse, cevaptan işlemi çıkar
    # Desenler Türkçe karaktersiz yazılır; metin de karaktersizleştirilip aranır ("ariza" = "arıza")
    "ariza_kaydi_ac": r"ariza (kayd|bildirim)",
    "teknisyen_randevusu": r"teknisyen|randevu|saha ekib",
    "fatura_incele_iade": r"\biade|fatura\w* (incel|duzelt)|tutar\w* (duzelt|geri)",
    "nakil_islemi": r"nakil|nakl|tasima islem|yeni adres",
    "hat_optimizasyonu": r"hat test|optimiz|port\w* (ayar|yenile|sifirla)",
    "kampanya_tanimla": r"kampanya|indirim|hediye|telafi",
}
_ASCII = str.maketrans("ıİşŞğĞüÜöÖçÇâÂ", "iissgguuoocc" + "aa")


def ascii_tr(s: str) -> str:
    return (s or "").translate(_ASCII).lower()


TOOL_SPOKEN = {"ariza_kaydi_ac": "arıza kaydı", "teknisyen_randevusu": "teknisyen randevusu",
               "hat_optimizasyonu": "hat iyileştirmesi", "fatura_incele_iade": "fatura incelemesi",
               "nakil_islemi": "nakil işlemi", "kampanya_tanimla": "telafi kampanyası"}


# Küçük modeller Türkçe hecelerin yerine okunuşu aynı Kiril harflerini koyabiliyor ("Haklıсыз" = "Haklısız")
_CYRILLIC = str.maketrans({**dict(zip("абвгдежзийклмнопрстуфхцчшыэюяі", "abvgdejziyklmnoprstufhcçşıeüai")),
                           **dict(zip("АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЫЭЮЯІ", "ABVGDEJZİYKLMNOPRSTUFHCÇŞIEÜAI"))})


def clean_utterance(text: str, max_sentences: int = 3, max_chars: int = 420) -> str:
    """Model çıktısını telefon konuşmasına uygun hale getirir: '#1,' gibi kalıntılar, satır sonları, aşırı uzunluk,
    müşteriye okunan araç adları ('fatura_incele_iade aracını kullanacağım')."""
    t = re.sub(r"^\s*#\d+\s*[,.:-]?\s*", "", text or "")
    for k, v in TOOL_SPOKEN.items():
        t = re.sub(rf"`?{k}`?(\s+aracını)?", v, t)
    t = re.sub(r"\[[^\]]{1,25}\]|\{[^}]{1,25}\}", "", t)          # "[adres]", "{mesaj}" gibi şablon kalıntıları
    t = re.sub(r"\((?:işlem sonucu|islem sonucu|not|sistem)[^)]*\)", "", t, flags=re.I)  # geçmişteki notları okumasın
    t = t.translate(_CYRILLIC)                                       # "Haklıсыз" gibi karışan Kiril harfleri
    t = re.sub(r"\s+([,.!?])", r"\1", t)
    t = re.sub(r"\s+", " ", t).strip().strip('"').strip()
    sents = re.split(r"(?<=[.!?…])\s+", t)
    out = []
    for s_ in sents:   # aynı cümleyi tekrar etme
        if s_ and s_ not in out:
            out.append(s_)
    t = " ".join(out[:max_sentences])
    return t[:max_chars].rsplit(" ", 1)[0] + "…" if len(t) > max_chars else t


def _similar(a: str, b: str) -> float:
    from difflib import SequenceMatcher
    return SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()


class Call:
    def __init__(self, persona_id: str, global_model: GlobalModel, audio: bool = True,
                 mode: str | None = None, seed: int | None = None, policy: str = "ogrenen",
                 save_report: bool = True):
        self.policy = policy
        self.save_report = save_report
        self.events: list[dict] = []          # ayrıntılı rapor için tur tur kayıt (core/report.py)
        self.started_at = __import__("time").time()
        self.id = uuid.uuid4().hex[:10]
        self.persona_id = persona_id
        self.p = PERSONAS[persona_id]
        self.g = global_model
        self.mode = mode or config.default_mode()
        self.audio = audio and self.mode in ("gemini", "lokal")
        self.rng = np.random.default_rng(seed)
        self.pyrng = random.Random(seed)
        self.learner = CallLearner(global_model, self.rng)

        self.mood = self.p["baslangic"]
        self.issue_idx = 0                    # müşterinin şu an konuştuğu sorun
        self.resolved: set[str] = set()
        self.kampanya = False
        self.tool_counter = 48200 + self.pyrng.randint(0, 700)

        self.history: list[dict] = []         # {rol, metin}
        self.timeline: list[dict] = []        # grafik için
        self.last_agent: dict | None = None   # son asistan hamlesi (ödül için)
        self.prev_valence: float | None = None
        self.agent_turns = 0
        self.ended = False
        self.outcome: str | None = None
        self.rewards: list[float] = []
        self.low_streak = 0
        self.last_mood_delta = 0.0
        self.mentioned: set[str] = set()
        self.done_since: int | None = None    # tüm sorunların çözüldüğü asistan turu

    # ------------------------------------------------------------ yardımcılar
    @property
    def current_issue(self) -> str | None:
        if self.issue_idx < len(self.p["sorunlar"]):
            return self.p["sorunlar"][self.issue_idx]
        return None

    def revealed_issues(self):
        return self.p["sorunlar"][: self.issue_idx + 1]

    def _audio_path(self, tag):
        name = f"{self.id}_{len(self.history):02d}_{tag}.wav"
        return config.AUDIO_DIR / name, f"/audio/{name}"

    def _speak(self, text, style, voice, tag):
        if not self.audio:
            return None, None
        try:
            wav = llm.tts(text, style, voice)
        except Exception as e:  # noqa: BLE001  TTS kotası bitse bile görüşme sürsün (tarayıcı sesi devreye girer)
            print(f"[tts] seslendirme yapılamadı, tarayıcı sesine düşülüyor: {str(e)[:100]}", flush=True)
            return None, None
        path, url = self._audio_path(tag)
        path.write_bytes(wav)
        return wav, url

    def _transcript(self, last=12):
        return "\n".join(f"{h['rol'].upper()}: {h['metin']}" for h in self.history[-last:])

    # ------------------------------------------------------------ müşteri turu
    def customer_turn(self) -> dict:
        info = {}
        bilesen = None
        if self.last_agent is not None:
            a = self.last_agent
            bilesen = {
                "onceki": round(self.mood, 4),
                "strateji_etkisi": self.p["tercih_strateji"][a["strateji"]],
                "ton_etkisi": self.p["tercih_ton"][a["ton"]],
                "cozum_bonusu": RESOLVE_MOOD_BONUS * len(a["cozulen"]),
                "kampanya_etkisi": self.p["kampanya_etkisi"] if a["kampanya_ilk"] else 0.0,
                "sabir_cezasi": -0.06 if (not a["cozulen"] and self.agent_turns >= 3) else 0.0,  # sabır azalıyor
                "gurultu": float(self.rng.normal(0, 0.04)),
            }
            d = sum(v for k, v in bilesen.items() if k != "onceki")
            self.mood = float(np.clip(self.mood + d, -1, 1))
            bilesen["toplam"] = round(d, 4)
            bilesen["sonraki"] = round(self.mood, 4)
            bilesen["kirpildi"] = abs((bilesen["onceki"] + d) - self.mood) > 1e-9
            self.last_mood_delta = d
            info["ruh_hali_degisimi"] = round(d, 3)
            while self.current_issue and self.current_issue in self.resolved:
                self.issue_idx += 1

        all_done = self.current_issue is None
        if all_done and self.done_since is None:
            self.done_since = self.agent_turns
        ending = None
        if self.mood <= -0.95 and self.agent_turns >= 3:   # en az 3 şans tanı
            ending = "ayrildi"
        elif all_done and self.mood >= STAY_THRESHOLD:
            ending = "kaldi"
        elif all_done and self.agent_turns - self.done_since >= 2:
            # Sorunlar çözüldükten sonra asistana iki tur gönül alma şansı verilir, sonra görüşme biter.
            # (Önceden müşteri vedalaştığı halde görüşme uzuyor, müşteri bundan sinirleniyordu.)
            ending = "kaldi" if self.mood >= -0.2 else "ayrildi"
        elif self.agent_turns >= config.MAX_TURNS:
            ending = "kaldi" if self.mood > 0 else "ayrildi"

        text = self._customer_text(ending, all_done)
        wav, url = self._speak(text, mood_delivery(self.mood), self.p["ses"], "musteri")
        self._last_customer_wav = wav
        self.history.append({"rol": "müşteri", "metin": text})
        if self.current_issue:
            self.mentioned.add(self.current_issue)
        if ending:
            self.ended = True
            self.outcome = ending

        self.events.append({
            "tip": "musteri", "metin": text, "ses": url, "gercek_ruh_hali": round(self.mood, 4),
            "gercek_etiket": mood_label(self.mood), "tts_tonu": mood_delivery(self.mood),
            "konu": self.current_issue, "ruh_hali_bilesenleri": bilesen, "bitti": self.ended, "sonuc": self.outcome,
            "yedek_cumle": getattr(self, "_customer_fallback", False),
        })
        return {
            "rol": "musteri", "metin": text, "ses": url, "bitti": self.ended, "sonuc": self.outcome,
            "gercek_ruh_hali": round(self.mood, 3), "gercek_etiket": mood_label(self.mood),
            "tts_tonu": mood_delivery(self.mood), **info,
        }

    def _customer_text(self, ending, all_done) -> str:
        if self.mode == "mock":
            if ending == "ayrildi":
                return MOCK_CHURN
            if ending == "kaldi":
                return MOCK_STAY if self.mood >= STAY_THRESHOLD else "Sorunlarım çözüldü, hâlâ biraz kırgınım ama şimdilik devam ediyorum. İyi günler."
            if all_done:
                return "Sorun çözüldü ama açıkçası hâlâ çok kırgınım."
            lines = MOCK_CUSTOMER[self.current_issue]
            bucket = 0 if self.mood < -0.5 else (1 if self.mood < 0 else 2)
            return lines[bucket]

        if self.last_mood_delta > 0.05:
            yon = "Asistanın son cevabı seni biraz rahatlattı, bu sözlerine yansısın."
        elif self.last_mood_delta < -0.05:
            yon = "Asistanın son cevabı seni daha da gerdi ya da sinirlendirdi, bu sözlerine yansısın."
        else:
            yon = "Asistanın son cevabı seni pek etkilemedi."
        if not self.history:
            yon = "Görüşmeyi sen başlatıyorsun: kendini kısaca tanıt ve sorununu anlat."

        if ending == "ayrildi":
            gorev = "Artık sabrın tükendi: hattını iptal ettirip rakip operatöre geçeceğini söyle ve görüşmeyi bitir."
        elif ending == "kaldi" and self.mood >= STAY_THRESHOLD:
            gorev = "Sorunların çözüldü ve memnunsun: teşekkür et, vedalaş ve görüşmeyi bitir."
        elif ending == "kaldi":
            gorev = ("Sorunların çözüldü; hâlâ biraz kırgınsın ama operatörde kalmaya karar verdin. "
                     "Bunu kısaca belli ederek vedalaş ve görüşmeyi bitir.")
        elif all_done:
            gorev = ("Sorunların çözüldü ama hâlâ kırgınsın. Bunu dile getir. "
                     "Henüz vedalaşma ve telefonu kapatacağını söyleme.")
        else:
            iss = ISSUES[self.current_issue]["ad"]
            yeni = self.issue_idx > 0 and self.current_issue not in self.mentioned
            gorev = (f"Şu an konuştuğun sorun: {iss}." + (" Önceki sorunun çözüldü; şimdi bu YENİ sorunu ilk kez gündeme getir." if yeni else ""))
            if self.mood < -0.6:
                gorev += " Başka operatöre geçmeyi düşündüğünü ima edebilirsin."

        system = (
            f"Sen bir telekom operatörünün çağrı merkezini arayan müşteri {self.p['ad']}'sin. {self.p['ozet']} "
            f"Karakterin: {self.p['karakter']} Telefonda konuşur gibi doğal, günlük Türkçe kullan. "
            "Sadece müşterinin söyleyeceklerini yaz; sahne yönergesi, parantez, emoji ya da sayı ile duygu belirtme."
        )
        prompt = (
            f"Şu anki ruh halin: {mood_label(self.mood)}.\n{yon}\n{gorev}\n"
            f"Çözülmüş sorunların: {', '.join(ISSUES[i]['ad'] for i in self.resolved) or 'yok'}.\n\n"
            f"Görüşme şu ana kadar:\n{self._transcript() or '(henüz başlamadı)'}\n\n1-3 cümleyle cevap ver."
        )
        if self.mode == "lokal":
            # Küçük model: konuşma, müşterinin bakış açısından gerçek sohbet rolleriyle verilir
            system = (f"Bir telekom şirketinin çağrı merkezini arayan müşteri {self.p['ad']}'sin. {self.p['ozet']} "
                      f"Konuşma tarzın şöyle: {self.p['karakter']} Telefonda konuşur gibi doğal, günlük Türkçe kullan. "
                      "ÖNEMLİ: Sen MÜŞTERİSİN, temsilci değilsin. Temsilcinin cümlelerini tekrarlama; 'size nasıl "
                      "yardımcı olabilirim' gibi temsilci cümleleri kurma. Karakter tarifini değil, söyleyeceğin sözleri yaz.")
            if not self.history:   # ilk tur: küçük model karakter tanımını tekrar etmesin, gerçekten aradığını bilsin
                yon = ("Temsilci telefonu açtı ve 'Merhaba, ben Deniz, size nasıl yardımcı olabilirim?' dedi. "
                       "Ona kendi sorununu anlat.")
            prompt = (f"[YÖNERGE] Ruh halin: {mood_label(self.mood)}. {yon} {gorev} "
                      f"Çözülmüş sorunların: {', '.join(ISSUES[i]['ad'] for i in self.resolved) or 'yok'}. "
                      "Müşteri olarak 1-3 kısa cümleyle cevap ver.")
            hist = self._chat_history("müşteri")
        else:
            hist = None
        last_agent = next((h["metin"] for h in reversed(self.history) if h["rol"] == "asistan"), "")
        self._customer_fallback = False
        schema = LocalCustomerReply if self.mode == "lokal" else CustomerReply
        for attempt in range(2):
            try:
                r = llm.generate_json(system, prompt, schema, temperature=0.9 if attempt == 0 else 1.0,
                                      tr_fields=("konusma",), history=hist)
            except Exception as e:  # noqa: BLE001  bozuk/yarım JSON: bir daha dene, olmazsa hazır cümle
                if "bağlanılamadı" in str(e) or "modeli yok" in str(e):
                    raise
                print(f"[koruma] müşteri cevabı okunamadı: {str(e)[:100]}", flush=True)
                continue
            text = clean_utterance(r.konusma)
            low = text.lower()
            bad = (len(text.split()) < (4 if not self.history else 1) or _similar(text, last_agent) > 0.5
                   or (self.mode == "lokal" and low.rstrip(".!") in self.p["karakter"].lower())
                   or any(p in low for p in AGENT_PHRASES) or low.startswith(self.p["ad"].lower()))
            if not bad:
                return text
            print(f"[koruma] müşteri cevabı reddedildi (tekrar/rol karışması): {text[:80]!r}", flush=True)
            prompt += (" DİKKAT: Önceki denemende temsilcinin sözlerini tekrarladın ya da temsilci gibi konuştun. "
                       "Sen müşterisin; kendi derdini kendi ağzından anlat.")
        self._customer_fallback = True       # iki deneme de bozuksa hazır müşteri cümlesine düş
        if ending == "ayrildi":
            return MOCK_CHURN
        if ending == "kaldi" or all_done:
            return MOCK_STAY if self.mood >= STAY_THRESHOLD else "Sorunlarım çözüldü ama açıkçası hâlâ kırgınım."
        bucket = 0 if self.mood < -0.5 else (1 if self.mood < 0 else 2)
        return MOCK_CUSTOMER[self.current_issue][bucket]

    def _chat_history(self, me: str) -> list[dict]:
        """Konuşmayı `me` ajanının bakış açısından sohbet rollerine çevirir (kendi sözleri: assistant, karşı taraf: user).
        İşlem sonuçları asistanın kendi mesajına not olarak eklenir."""
        msgs: list[dict] = []
        for h in self.history:
            if h["rol"] == "sistem":
                if me == "asistan" and msgs and msgs[-1]["role"] == "assistant":
                    msgs[-1]["content"] += f" (İşlem sonucu: {h['metin']})"
                continue
            role = "assistant" if h["rol"] == me else "user"
            msgs.append({"role": role, "content": h["metin"]})
        return msgs

    # ------------------------------------------------------------ asistan turu
    def agent_turn(self) -> dict:
        last_text = self.history[-1]["metin"]
        # 1) Dinle: duygu analizi
        if self.mode == "mock":
            an = emotion.mock_analysis(self.mood, last_text, self.current_issue or "diger", self.rng)
        elif getattr(self, "_last_customer_wav", None):
            an = emotion.analyze_audio(self._last_customer_wav)
        else:
            an = emotion.analyze_text(last_text)
        v = float(an["valens"])

        # 2) Bağlam (ilk turda belirlenir; genel bilgiyi buna göre çağırır)
        if self.learner.ctx is None:
            gerilim = "yüksek gerilim" if an["uyarilma"] >= 0.5 else "düşük gerilim"
            self.learner.ctx = f"{an['konu']} · {gerilim}"

        # 3) Öğren: önceki hamlenin ödülü
        reward_info = None
        odul_detay = None
        if self.last_agent is not None and self.prev_valence is not None:
            r = (v - self.prev_valence) + PROGRESS_WEIGHT * len(self.last_agent["cozulen"])
            odul_detay = {
                "strateji": self.last_agent["strateji"], "ton": self.last_agent["ton"],
                "onceki_valens": round(self.prev_valence, 4), "yeni_valens": round(v, 4),
                "duygu_degisimi": round(v - self.prev_valence, 4),
                "cozulen_sorun": len(self.last_agent["cozulen"]),
                "ilerleme_bonusu": round(PROGRESS_WEIGHT * len(self.last_agent["cozulen"]), 4),
                "ham_odul": round(r, 4), "ogrenmeye_giren": round(max(-REWARD_CLIP, min(REWARD_CLIP, r)), 4),
            }
            self.rewards.append(r)
            for dim in ("strateji", "ton"):
                self.learner.update(dim, self.last_agent[dim], r)
            reward_info = {"odul": round(r, 3), "strateji": self.last_agent["strateji"], "ton": self.last_agent["ton"]}
        self.prev_valence = v

        self.timeline.append({
            "tur": len(self.timeline), "algilanan": round(v, 3), "gercek": round(self.mood, 3),
            "churn": round(float(an["churn_riski"]), 3), "uyarilma": round(float(an["uyarilma"]), 3),
        })

        base = {"rol": "asistan", "analiz": an, "odul": reward_info, "zaman_cizelgesi": self.timeline}
        ev = {"tip": "asistan", "algi": an, "gercek_ruh_hali": round(self.mood, 4),
              "algi_hatasi": round(v - self.mood, 4), "odul": odul_detay, "baglam": self.learner.ctx}
        self.events.append(ev)
        if self.ended:
            ev["beyin_son"] = self.learner.snapshot()
            self._finish()
            return {**base, "bitti": True, "sonuc": self.outcome, "ozet": self.summary(),
                    "rapor": f"/raporlar/{self.id}" if self.save_report else None,
                    "beyin": self.learner.snapshot(), "baglam": self.learner.ctx}

        # 4) Yönetmen (Director): müşteri kaçmak üzereyse uyarı
        self.low_streak = self.low_streak + 1 if an["churn_riski"] >= 0.75 else 0
        director = None
        if self.low_streak >= 2:
            director = "Müşteri kaçmak üzere! Kalıcı bir çözüm taahhüdü ya da somut bir telafi düşün."

        # 5) Strateji ve ton seç (Thompson Sampling)
        ev["beyin_secim_oncesi"] = self.learner.snapshot()
        ev["yonetmen"] = director
        if self.policy == "rastgele":
            s, s_samples = self.pyrng.choice(list(STRATEGIES)), {}
            t, t_samples = self.pyrng.choice(list(TONES)), {}
        else:
            s, s_samples = self.learner.select("strateji")
            t, t_samples = self.learner.select("ton")

        # 6) Cevap üret ve araçları çalıştır
        reply = self._agent_reply(s, t, an, director)
        actions, cozulen, kampanya_ilk = [], [], False
        for tool in reply["araclar"][:2]:
            self.tool_counter += 1
            res = TOOLS[tool]["sonuc"].format(n=self.tool_counter)
            solved = None
            for iss in self.revealed_issues():
                if iss not in self.resolved and tool in ISSUES[iss]["cozen"]:
                    self.resolved.add(iss)
                    cozulen.append(iss)
                    solved = ISSUES[iss]["ad"]
                    break
            if tool == "kampanya_tanimla" and not self.kampanya:
                self.kampanya = kampanya_ilk = True
            actions.append({"arac": tool, "sonuc": res, "cozdu": solved,
                            "metinden": tool in reply.get("metinden_araclar", [])})

        self.agent_turns += 1
        self.last_agent = {"strateji": s, "ton": t, "cozulen": cozulen, "kampanya_ilk": kampanya_ilk}
        ev.update({
            "secim": {"strateji": s, "ton": t, "ornekler": {"strateji": s_samples, "ton": t_samples},
                      "neden": {"strateji": self.learner.explain("strateji", s), "ton": self.learner.explain("ton", t)}},
            "cevap": reply["cevap"], "ic_ses": reply["ic_ses"], "araclar": actions,
        })
        wav, url = self._speak(reply["cevap"], TONES[t]["tts"], config.AGENT_VOICE, "asistan")
        ev["ses"] = url
        self.history.append({"rol": "asistan", "metin": reply["cevap"]})
        for a in actions:
            self.history.append({"rol": "sistem", "metin": f"[{a['arac']}] {a['sonuc']}"})

        return {
            **base, "bitti": False, "metin": reply["cevap"], "ses": url, "ic_ses": reply["ic_ses"],
            "strateji": s, "ton": t, "strateji_ad": STRATEGIES[s]["ad"], "ton_ad": TONES[t]["ad"],
            "tts_tonu": TONES[t]["tts"], "rate": TONES[t]["rate"], "pitch": TONES[t]["pitch"],
            "neden": {"strateji": self.learner.explain("strateji", s), "ton": self.learner.explain("ton", t)},
            "ornekler": {"strateji": s_samples, "ton": t_samples},
            "araclar": actions, "yonetmen": director,
            "beyin": self.learner.snapshot(), "baglam": self.learner.ctx,
        }

    def _agent_reply(self, s, t, an, director) -> dict:
        if self.mode == "mock":
            tools = []
            cur = self.current_issue
            if cur and self.pyrng.random() < MOCK_RESOLVE_P[s]:
                tools.append(ISSUES[cur]["cozen"][0])
            if s == "telafi_teklifi":
                tools.append("kampanya_tanimla")
            return {"cevap": MOCK_AGENT[s], "ic_ses": f"{STRATEGIES[s]['ad']} deniyorum.", "araclar": tools}

        tools_txt = "\n".join(f"- {k}: {v['aciklama']}" for k, v in TOOLS.items())
        done = [h["metin"] for h in self.history if h["rol"] == "sistem"]
        system = (
            "Sen 'Deniz', bir telekom operatörünün yapay zekâ destekli sesli müşteri temsilcisisin. "
            "Nihai amacın müşterinin sorununu GERÇEKTEN çözmek ve onu memnun etmek. Telefonda konuşuyorsun: "
            "kısa, doğal, akıcı Türkçe kullan (Türkçe karakterleri ç, ğ, ı, ö, ş, ü doğru yaz). Madde işareti, emoji, parantez kullanma. "
            "Yalnızca müşterinin dile getirdiği sorunlar için araç çalıştır; gereksiz araç kullanma, "
            "aynı işlemi iki kez yapma. Araç çalıştırdıysan sonucunu cevabında kısaca belirt."
        )
        prompt = (
            f"BU TURDAKİ SÖYLEM STRATEJİN: {STRATEGIES[s]['ad']} -> {STRATEGIES[s]['talimat']}\n"
            f"BU TURDAKİ TONUN: {TONES[t]['ad']} -> {TONES[t]['yazim']}\n\n"
            f"Müşterinin son konuşmasının duygu analizi: duygu={an['duygu']}, valens={an['valens']:+.2f}, "
            f"gerginlik={an['uyarilma']:.2f}, churn riski={an['churn_riski']:.2f}. "
            f"Ses ipuçları: {an['ses_ipuclari']}. Söz ipuçları: {an['metin_ipuclari']}\n"
            + (f"YÖNETMEN UYARISI: {director}\n" if director else "")
            + f"\nKullanabileceğin araçlar:\n{tools_txt}\n\n"
            f"Daha önce yapılan işlemler: {'; '.join(done) or 'yok'}\n\n"
            f"Görüşme:\n{self._transcript()}\n\nSıradaki cevabını ver."
        )
        if self.mode == "lokal":
            system += " Araç adlarını (ör. ariza_kaydi_ac) müşteriye söyleme; yaptığın işlemi doğal cümleyle anlat."
            prompt = prompt.replace(f"Görüşme:\n{self._transcript()}\n\nSıradaki cevabını ver.",
                                    "[YÖNERGE] Müşterinin son mesajına temsilci olarak cevap ver. Müşterinin cümlelerini "
                                    "tekrarlama, aynı cümleyi iki kez kurma, en fazla 3 kısa cümle kullan.")
            hist = self._chat_history("asistan")
        else:
            hist = None
        last_cust = next((h["metin"] for h in reversed(self.history) if h["rol"] == "müşteri"), "")
        schema = LocalAgentReply if self.mode == "lokal" else AgentReply
        out = None
        for attempt in range(2):
            try:
                r = llm.generate_json(system, prompt, schema, temperature=0.6 if attempt == 0 else 0.8,
                                      tr_fields=("cevap", "ic_ses"), history=hist)
            except Exception as e:  # noqa: BLE001
                if "bağlanılamadı" in str(e) or "modeli yok" in str(e):
                    raise
                print(f"[koruma] asistan cevabı okunamadı: {str(e)[:100]}", flush=True)
                continue
            out = r.model_dump()
            out["cevap"] = clean_utterance(out["cevap"])
            if self.agent_turns > 0:   # her tura "Merhaba" ile başlamasın
                out["cevap"] = re.sub(r"^merhaba[,!.]?\s*", "", out["cevap"], flags=re.I) or out["cevap"]
            if len(out["cevap"]) >= 3 and _similar(out["cevap"], last_cust) <= 0.55:
                break
            print(f"[koruma] asistan cevabı reddedildi (müşteriyi tekrar): {out['cevap'][:80]!r}", flush=True)
            prompt += " DİKKAT: Müşterinin sözlerini tekrarlama; kendi cevabını ver."
        else:
            out = out or {"ic_ses": "", "araclar": []}
            out["cevap"] = MOCK_AGENT[s]
        if len(out.get("ic_ses", "")) < 10 or _similar(out["ic_ses"], out["cevap"]) > 0.7:
            out["ic_ses"] = f"{STRATEGIES[s]['ad']} stratejisini uyguluyorum."
        out["metinden_araclar"] = []
        if self.mode == "lokal" and not out["araclar"]:
            low = ascii_tr(out["cevap"])
            open_issues = [i for i in self.revealed_issues() if i not in self.resolved]
            raw = ascii_tr(r.cevap if out else "")
            for tool, pat in TOOL_HINTS.items():
                if (re.search(pat, low) or tool in raw) and len(out["araclar"]) < 2:
                    useful = any(tool in ISSUES[i]["cozen"] for i in open_issues) or tool == "kampanya_tanimla"
                    if useful:
                        out["araclar"].append(tool)
                        out["metinden_araclar"].append(tool)
        return out

    # ------------------------------------------------------------ bitiş
    def _finish(self):
        if getattr(self, "_finished", False):
            return
        self._finished = True
        self.g.episodes.append(self.summary())
        self.g.save()
        if self.save_report:
            try:
                from . import report
                report.save(self)
            except Exception as e:  # noqa: BLE001  rapor hatası görüşmeyi bozmasın
                print(f"[rapor] oluşturulamadı: {e}", flush=True)

    def summary(self):
        return {
            "musteri": self.p["ad"], "persona": self.persona_id, "mod": self.mode,
            "tur": self.agent_turns, "son_ruh_hali": round(self.mood, 3), "sonuc": self.outcome,
            "cozulen": len(self.resolved), "sorun": len(self.p["sorunlar"]),
            "ort_odul": round(float(np.mean(self.rewards)), 3) if self.rewards else None,
            "baglam": self.learner.ctx, "politika": self.policy,
        }


def run_fast(global_model: GlobalModel, n: int, persona: str | None = None, seed: int = 0,
             policy: str = "ogrenen") -> list[dict]:
    """LLM ve ses kullanmadan hızlı eğitim: sadece öğrenme algoritmasını çalıştırır.
    policy="rastgele" karşılaştırma için her turda rastgele strateji/ton seçen temsilcidir."""
    if policy == "rastgele":
        global_model = GlobalModel(persist=False)
    out = []
    rng = random.Random(seed)
    for i in range(n):
        pid = persona or rng.choice(list(PERSONAS))
        c = Call(pid, global_model, audio=False, mode="mock", seed=rng.randint(0, 10**9), policy=policy,
                 save_report=False)
        c.customer_turn()
        while True:
            a = c.agent_turn()
            if a["bitti"]:
                break
            c.customer_turn()
        out.append(c.summary())
    global_model.save()
    return out
