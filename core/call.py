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
from .learner import CallLearner, GlobalModel

PROGRESS_WEIGHT = 0.15
RESOLVE_MOOD_BONUS = 0.15
STAY_THRESHOLD = 0.3
TOOL_NAMES = tuple(TOOLS)


class AgentReply(BaseModel):
    ic_ses: str = Field(description="Kısa iç ses: müşteri hakkında ne düşünüyorsun, neden bu şekilde cevap veriyorsun (1 cümle)")
    cevap: str = Field(description="Müşteriye söyleyeceğin Türkçe cevap (en fazla 3 kısa cümle)")
    araclar: list[Literal[TOOL_NAMES]] = Field(default_factory=list, description="Bu turda çalıştırılacak araçlar (0-2)")


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


class Call:
    def __init__(self, persona_id: str, global_model: GlobalModel, audio: bool = True,
                 mode: str | None = None, seed: int | None = None, policy: str = "ogrenen"):
        self.policy = policy
        self.id = uuid.uuid4().hex[:10]
        self.persona_id = persona_id
        self.p = PERSONAS[persona_id]
        self.g = global_model
        self.mode = mode or ("gemini" if config.has_gemini() else "mock")
        self.audio = audio and self.mode == "gemini"
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
        if self.last_agent is not None:
            a = self.last_agent
            d = self.p["tercih_strateji"][a["strateji"]] + self.p["tercih_ton"][a["ton"]]
            d += RESOLVE_MOOD_BONUS * len(a["cozulen"])
            if a["kampanya_ilk"]:
                d += self.p["kampanya_etkisi"]
            if not a["cozulen"] and self.agent_turns >= 3:
                d -= 0.06                      # sabır azalıyor
            d += float(self.rng.normal(0, 0.04))
            self.mood = float(np.clip(self.mood + d, -1, 1))
            self.last_mood_delta = d
            info["ruh_hali_degisimi"] = round(d, 3)
            while self.current_issue and self.current_issue in self.resolved:
                self.issue_idx += 1

        all_done = self.current_issue is None
        ending = None
        if self.mood <= -0.95 and self.agent_turns >= 3:   # en az 3 şans tanı
            ending = "ayrildi"
        elif all_done and self.mood >= STAY_THRESHOLD:
            ending = "kaldi"
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

        return {
            "rol": "musteri", "metin": text, "ses": url, "bitti": self.ended, "sonuc": self.outcome,
            "gercek_ruh_hali": round(self.mood, 3), "gercek_etiket": mood_label(self.mood),
            "tts_tonu": mood_delivery(self.mood), **info,
        }

    def _customer_text(self, ending, all_done) -> str:
        if self.mode == "mock":
            if ending == "ayrildi":
                return MOCK_CHURN
            if ending == "kaldi" or all_done:
                return MOCK_STAY if self.mood >= STAY_THRESHOLD else "Sorun çözüldü ama açıkçası hâlâ çok kırgınım."
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
        elif ending == "kaldi":
            gorev = "Sorunların çözüldü ve memnunsun: teşekkür et, vedalaş ve görüşmeyi bitir."
        elif all_done:
            gorev = "Sorunların çözüldü ama hâlâ kırgınsın. Bunu dile getir."
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
        r: CustomerReply = llm.generate_json(system, prompt, CustomerReply, temperature=0.9)
        return r.konusma.strip()

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
        if self.last_agent is not None and self.prev_valence is not None:
            r = (v - self.prev_valence) + PROGRESS_WEIGHT * len(self.last_agent["cozulen"])
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
        if self.ended:
            self._finish()
            return {**base, "bitti": True, "sonuc": self.outcome, "ozet": self.summary(),
                    "beyin": self.learner.snapshot(), "baglam": self.learner.ctx}

        # 4) Yönetmen (Director): müşteri kaçmak üzereyse uyarı
        self.low_streak = self.low_streak + 1 if an["churn_riski"] >= 0.75 else 0
        director = None
        if self.low_streak >= 2:
            director = "Müşteri kaçmak üzere! Kalıcı bir çözüm taahhüdü ya da somut bir telafi düşün."

        # 5) Strateji ve ton seç (Thompson Sampling)
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
            actions.append({"arac": tool, "sonuc": res, "cozdu": solved})

        self.agent_turns += 1
        self.last_agent = {"strateji": s, "ton": t, "cozulen": cozulen, "kampanya_ilk": kampanya_ilk}
        wav, url = self._speak(reply["cevap"], TONES[t]["tts"], config.AGENT_VOICE, "asistan")
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
        r: AgentReply = llm.generate_json(system, prompt, AgentReply, temperature=0.6)
        tr = re.compile("[çğıöşüÇĞİÖŞÜ]")
        if any(len(x) > 40 and not tr.search(x) for x in (r.cevap, r.ic_ses)):
            # Hafif modeller uzun bağlamda bazen Türkçe karakterleri atlıyor: bir kez yeniden üret
            r = llm.generate_json(system, prompt + "\nDİKKAT: Türkçe karakterleri (ç, ğ, ı, ö, ş, ü) mutlaka kullan.",
                                  AgentReply, temperature=0.4)
        return r.model_dump()

    # ------------------------------------------------------------ bitiş
    def _finish(self):
        if getattr(self, "_finished", False):
            return
        self._finished = True
        self.g.episodes.append(self.summary())
        self.g.save()

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
        c = Call(pid, global_model, audio=False, mode="mock", seed=rng.randint(0, 10**9), policy=policy)
        c.customer_turn()
        while True:
            a = c.agent_turn()
            if a["bitti"]:
                break
            c.customer_turn()
        out.append(c.summary())
    global_model.save()
    return out
