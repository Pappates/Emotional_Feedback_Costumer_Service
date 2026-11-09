"""Stratejiler, ses tonları, sorunlar, araçlar ve simüle müşteri profilleri.

Asistan ("avcı") hangi stratejinin ve tonun bir müşteride işe yaradığını BİLMEZ.
Bunu yalnızca müşterinin sesinden ve sözlerinden çıkardığı duygu değişimiyle öğrenir.
Müşteri profillerindeki `tercih_*` alanları simülasyonun "gizli gerçeği"dir
(Alien'daki oyuncunun saklanma alışkanlıkları gibi) ve yalnızca test düzeneği
ile paneldeki "gizli tercihleri göster" düğmesi tarafından kullanılır.
"""

# --- Söylem stratejileri (ne söylendiği) ---------------------------------
STRATEGIES = {
    "empati": {
        "ad": "Empati & özür",
        "talimat": "Müşterinin duygusunu adıyla kabul et, içtenlikle özür dile ve yaşadığı zorluğu anladığını göster.",
    },
    "cozum_odakli": {
        "ad": "Doğrudan çözüm",
        "talimat": "Lafı uzatma. Hemen somut bir aksiyon al ve net bir zaman ya da sonuç ver.",
    },
    "teknik_aciklama": {
        "ad": "Teknik şeffaflık",
        "talimat": "Sorunun olası teknik nedenini açık ve şeffaf biçimde açıkla, ne yapılacağını adım adım anlat.",
    },
    "telafi_teklifi": {
        "ad": "Telafi teklifi",
        "talimat": "Yaşanan mağduriyet için somut bir telafi ya da kampanya öner (kampanya_tanimla aracını kullanabilirsin).",
    },
    "dinle_sor": {
        "ad": "Dinle & sor",
        "talimat": "Kısa bir onaylamadan sonra durumu netleştirmek için tek ve açık uçlu bir soru sor, müşteriyi konuşmaya davet et.",
    },
}

# --- Ses tonları (nasıl söylendiği) ----------------------------------------
# tts: Gemini TTS'e verilen stil talimatı. rate/pitch: tarayıcı yedek seslendirmesi için.
TONES = {
    "sicak_yavas": {
        "ad": "Sıcak & yavaş",
        "tts": "Say in a warm, slow, calm and caring voice",
        "yazim": "Sıcak, sakin ve yumuşak cümleler kur.",
        "rate": 0.9, "pitch": 1.0,
    },
    "enerjik": {
        "ad": "Enerjik & pozitif",
        "tts": "Say in an upbeat, energetic, positive and confident voice",
        "yazim": "Enerjik, pozitif ve kendinden emin cümleler kur.",
        "rate": 1.12, "pitch": 1.1,
    },
    "resmi_net": {
        "ad": "Resmi & net",
        "tts": "Say in a professional, crisp, clear and matter-of-fact voice",
        "yazim": "Resmi, kısa ve net cümleler kur.",
        "rate": 1.05, "pitch": 0.95,
    },
    "samimi": {
        "ad": "Samimi & rahat",
        "tts": "Say in a friendly, casual, relaxed conversational voice",
        "yazim": "Samimi ve rahat bir konuşma dili kullan (yine de saygılı ol).",
        "rate": 1.0, "pitch": 1.05,
    },
}

# --- Müşteri sorunları ve onları çözen araçlar ------------------------------
ISSUES = {
    "internet_kesintisi": {"ad": "İnternet kesintisi", "cozen": ["ariza_kaydi_ac", "teknisyen_randevusu"]},
    "hiz_sorunu": {"ad": "İnternet hızı düşük", "cozen": ["hat_optimizasyonu", "ariza_kaydi_ac"]},
    "fatura_itirazi": {"ad": "Fatura itirazı", "cozen": ["fatura_incele_iade"]},
    "nakil": {"ad": "Adres nakli", "cozen": ["nakil_islemi"]},
}

TOOLS = {
    "ariza_kaydi_ac": {
        "aciklama": "Arıza kaydı açar (internet kesintisi ya da hız sorunu için).",
        "sonuc": "Arıza kaydı #TT-{n} açıldı. Saha ekibi 4 saat içinde müdahale edecek.",
    },
    "teknisyen_randevusu": {
        "aciklama": "Adrese teknisyen randevusu oluşturur (internet kesintisi için).",
        "sonuc": "Teknisyen randevusu oluşturuldu: bugün 16:00-18:00 arası, kayıt #RV-{n}.",
    },
    "hat_optimizasyonu": {
        "aciklama": "Uzaktan hat testi yapar ve port/profil optimizasyonu uygular (hız sorunu için).",
        "sonuc": "Hat testi tamamlandı: port profili 100 Mbps'e yeniden ayarlandı (kayıt #OP-{n}).",
    },
    "fatura_incele_iade": {
        "aciklama": "Faturayı inceler ve hatalı tutarı iade eder (fatura itirazı için).",
        "sonuc": "Fatura incelendi: hatalı 240 TL bir sonraki faturaya iade olarak yansıtılacak (kayıt #FT-{n}).",
    },
    "nakil_islemi": {
        "aciklama": "Aboneliği yeni adrese taşır (adres nakli için).",
        "sonuc": "Nakil talebi oluşturuldu: yeni adreste hat 2 iş günü içinde aktif olacak (kayıt #NK-{n}).",
    },
    "kampanya_tanimla": {
        "aciklama": "Müşteriye telafi kampanyası tanımlar (1 ay %30 indirim + 50 GB). Sorunu çözmez, gönül alır.",
        "sonuc": "Telafi kampanyası tanımlandı: 1 ay %30 indirim ve 50 GB ek internet.",
    },
}

# --- Simüle müşteri profilleri ----------------------------------------------
# tercih_strateji / tercih_ton: her turda müşterinin ruh haline eklenen etki (gizli).
PERSONAS = {
    "ahmet": {
        "ad": "Ahmet Bey",
        "ozet": "Kafe işleten esnaf. İki gündür internet yok, POS cihazı çalışmıyor, iş kaybediyor.",
        "karakter": "Sabırsız ve öfkeli. Lafı dolandırmaktan nefret eder, hemen net aksiyon ister. Saygılı ama sert konuşur.",
        "baslangic": -0.75,
        "sorunlar": ["internet_kesintisi", "fatura_itirazi"],
        "ses": "Fenrir",
        "tercih_strateji": {"empati": -0.10, "cozum_odakli": 0.24, "teknik_aciklama": 0.04, "telafi_teklifi": 0.05, "dinle_sor": -0.16},
        "tercih_ton": {"sicak_yavas": -0.10, "enerjik": -0.05, "resmi_net": 0.12, "samimi": 0.00},
        "kampanya_etkisi": 0.08,
    },
    "zeynep": {
        "ad": "Zeynep Hanım",
        "ozet": "Emekli öğretmen, 12 yıllık müşteri. Faturası beklenmedik şekilde yüksek gelmiş, ayrıca yakında taşınacak.",
        "karakter": "Kırgın ve hayal kırıklığına uğramış. Kendini değersiz hissediyor, ilgi ve anlayış bekliyor. Teknik terimlerden hoşlanmaz.",
        "baslangic": -0.55,
        "sorunlar": ["fatura_itirazi", "nakil"],
        "ses": "Leda",
        "tercih_strateji": {"empati": 0.22, "cozum_odakli": 0.05, "teknik_aciklama": -0.14, "telafi_teklifi": 0.12, "dinle_sor": 0.08},
        "tercih_ton": {"sicak_yavas": 0.14, "enerjik": -0.12, "resmi_net": -0.08, "samimi": 0.05},
        "kampanya_etkisi": 0.15,
    },
    "mert": {
        "ad": "Mert",
        "ozet": "Evden çalışan yazılımcı. Akşamları internet hızı çok düşüyor, toplantıları kopuyor.",
        "karakter": "Teknik bilgisi yüksek, şüpheci. Kalıp cümlelerden ve pazarlamadan hoşlanmaz, gerçek nedeni öğrenmek ister.",
        "baslangic": -0.55,
        "sorunlar": ["hiz_sorunu", "internet_kesintisi"],
        "ses": "Puck",
        "tercih_strateji": {"empati": -0.05, "cozum_odakli": 0.08, "teknik_aciklama": 0.24, "telafi_teklifi": -0.10, "dinle_sor": 0.04},
        "tercih_ton": {"sicak_yavas": -0.10, "enerjik": 0.08, "resmi_net": -0.04, "samimi": 0.12},
        "kampanya_etkisi": -0.05,
    },
    "selin": {
        "ad": "Selin Hanım",
        "ozet": "Yoğun çalışan bir avukat. Ofisini taşıyor, ayrıca son faturasında hatalı bir ücret var.",
        "karakter": "Aceleci ve gergin. Zamanı çok değerli, kısa ve profesyonel bir iletişim bekliyor.",
        "baslangic": -0.6,
        "sorunlar": ["nakil", "fatura_itirazi"],
        "ses": "Kore",
        "tercih_strateji": {"empati": 0.00, "cozum_odakli": 0.20, "teknik_aciklama": -0.08, "telafi_teklifi": 0.04, "dinle_sor": -0.14},
        "tercih_ton": {"sicak_yavas": -0.12, "enerjik": 0.04, "resmi_net": 0.12, "samimi": -0.05},
        "kampanya_etkisi": 0.04,
    },
}


def mood_delivery(mood: float) -> str:
    """Müşterinin ruh haline göre TTS tonlaması (müşterinin sesi duygusunu taşır)."""
    if mood < -0.6:
        return "Say in an angry, loud, fast and very frustrated voice"
    if mood < -0.25:
        return "Say in an irritated, tense and impatient voice"
    if mood < 0.15:
        return "Say in a tired, neutral, slightly skeptical voice"
    if mood < 0.5:
        return "Say in a calmer, relieved and softer voice"
    return "Say in a happy, warm and grateful voice"


def mood_label(mood: float) -> str:
    if mood < -0.6:
        return "çok öfkeli"
    if mood < -0.25:
        return "gergin ve sinirli"
    if mood < 0.15:
        return "kararsız, temkinli"
    if mood < 0.5:
        return "sakinleşmiş, biraz rahatlamış"
    return "memnun ve minnettar"
