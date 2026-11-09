# Duygu Avcısı: Müşteriyi sesinden öğrenen sesli çağrı merkezi asistanı (PoC)

Alien: Isolation'daki yaratık, oyuncunun saklanma ve kaçış alışkanlıklarını öğrenip aramasını
bunlara göre yönlendiriyordu. Bu PoC aynı mekaniği çağrı merkezine uyarlıyor:

* **Avcı** = çağrı merkezi asistanı, **av** = müşteri.
* Asistan her turda müşterinin **sesinden ve sözlerinden** duygusunu analiz eder.
* Hangi **söylem stratejisinin** (empati, doğrudan çözüm, teknik açıklama, telafi, dinle-sor) ve hangi
  **ses tonunun** (sıcak-yavaş, enerjik, resmi-net, samimi) müşteride olumlu tepki yarattığını öğrenir
  ve konuşmayı o yöne çeker.
* Nihai amaç sorunu **çözmek**: ödül = duygu iyileşmesi + çözülen sorun. Böylece asistan yalnızca
  gönül alan, sorunu çözmeyen bir şeye dönüşmez.

Demo için müşteriyi de bir AI ajan oynar. Her müşteri profilinin asistanın bilmediği **gizli tercihleri**
vardır (örneğin Ahmet Bey lafı dolandırmaktan nefret eder, Zeynep Hanım empati bekler). Panelde
"Gizli gerçeği göster" açılınca, asistanın tahminlerinin bu gizli tercihlere ne kadar yaklaştığı görülür.

## Mimari

```
 Müşteri ajanı (Gemini)                         Çağrı merkezi ajanı ("avcı")
 ─────────────────────                          ──────────────────────────────
 gizli tercihler -> ruh hali                    1. DİNLE: ses -> Gemini ses anlama (döküm + duygu
 ruh hali -> sözler (LLM)        ── ses ──►        + churn) + yerel akustik özellikler (enerji,
 ruh hali -> tonlama (TTS)                         perde, konuşma hızı; numpy, GPU gerekmez)
                                                2. ÖĞREN: ödül = Δduygu + 0.15 x çözülen sorun
                                                   -> Thompson Sampling (görüşme içi + genel)
                                                3. SEÇ: strateji + ton
                                ◄── ses ──      4. CEVAPLA: Gemini (strateji/ton talimatıyla)
                                                   + araçlar (arıza kaydı, iade, nakil, kampanya)
                                                5. KONUŞ: Gemini TTS, tonlama stratejiye göre
```

* **İki seviyeli öğrenme:** görüşme içi ("bu müşteri neyi seviyor", hızlı) + bağlama göre genel
  ("internet kesintisi yaşayan, gergin müşterilerde ne işe yarıyor"). Yeni görüşmeler genel bilgiyle
  ısınarak başlar. Genel model `data/global_model.json` dosyasında saklanır.
* **Yönetmen (Director):** churn riski üst üste iki tur yüksekse asistana uyarı geçer.
* **Öğrenme kanıtı:** "60 görüşme simüle et" düğmesi öğrenen asistanı her turda rastgele strateji
  seçen bir asistanla karşılaştırır (LLM kullanmaz, saniyeler sürer).

## Kurulum (GPU gerekmez)

1. Python 3.10+ kurulu olmalı.
2. Ücretsiz bir Gemini API anahtarı al: https://aistudio.google.com/apikey
   (Gemini uygulama üyeliği API anahtarı yerine geçmez; anahtar AI Studio'dan ayrıca alınır.)
3. Kur ve çalıştır:

```bash
cd duygu-avcisi
pip install -r requirements.txt
```

Windows (PowerShell):
```powershell
$env:GEMINI_API_KEY="AIza..."
python app.py
```

macOS / Linux:
```bash
GEMINI_API_KEY="AIza..." python app.py
```

4. Tarayıcıda aç: http://localhost:8000

Anahtar verilmezse uygulama **demo modunda** çalışır: müşteri ve asistan hazır cümleler kullanır,
seslendirmeyi tarayıcı yapar, duygu analizi simüle edilir. Öğrenme algoritması ve panel aynen çalışır.

### İsteğe bağlı ayarlar (ortam değişkenleri)

| Değişken | Varsayılan | Açıklama |
|---|---|---|
| `GEMINI_TEXT_MODEL` | `gemini-3.5-flash-lite,gemini-3.1-flash-lite,…` | Diyalog ve ses analizi modelleri (virgülle sıralı yedek listesi) |
| `GEMINI_TTS_MODEL` | `gemini-3.8-flash-tts,gemini-3.8-flash-lite-tts,…` | Seslendirme modelleri (virgülle sıralı yedek listesi) |
| `GEMINI_TIMEOUT_MS` | `30000` | Tek bir Gemini isteği için zaman aşımı |
| `AGENT_VOICE` | `Sulafat` | Asistanın sesi (Gemini ses adları: Kore, Puck, Charon, Aoede…) |
| `MAX_TURNS` | `9` | Bir görüşmedeki en fazla asistan turu |
| `FORCE_MOCK` | | `1` verilirse anahtar olsa bile demo modunda çalışır |

## Kota, yedek modeller ve kayıt (ÖNEMLİ)

* **Ücretsiz katmanda TTS kotası çok düşük:** model başına günde yalnızca birkaç seslendirme. Bir sesli
  görüşme yaklaşık 15-20 seslendirme kullanır. Kota dolunca uygulama sıradaki TTS modeline geçer.
  Hepsi dolarsa görüşme durmaz: seslendirmeyi tarayıcı yapar, duygu analizi metinden yapılır.
  Kota günlük yenilenir. Sunum günü rahat etmek istiyorsan AI Studio'da faturalandırmayı açabilirsin;
  bu demo birkaç kuruşluk kullanım yapar.
* **Metin modelleri yoğun saatlerde aşırı yüklenebiliyor (503/504).** Uygulama otomatik olarak sıradaki
  modele geçer. Kotası dolan model 10 dakika atlanır.
* **Kayıtlı görüşmeler:** her gerçek (Gemini) görüşme `data/kayitlar/` klasörüne sesleriyle kaydedilir.
  Panelden **"Kayıtlı görüşmeyi oynat"** ile API'ye hiç gitmeden tekrar oynatılır. Sunumdan önce
  1-2 iyi görüşme kaydet; internet ya da kota sorun çıkarırsa onu oynatırsın. Pakette örnek olarak bir
  Zeynep Hanım görüşmesi var.

## Demo akışı (yaklaşık 5 dakika)

0. Sunumdan önce: **"60 görüşme simüle et"**e bir kez bas. Böylece asistan canlı görüşmeye genel bilgiyle
   ısınmış başlar (soğuk başlayan asistan ilk turlarda yanlış strateji deneyebilir).
1. **"60 görüşme simüle et"**: öğrenen asistan müşterilerin yaklaşık %80-95'ini tutarken rastgele asistan
   yaklaşık %20-35'ini tutar. Bu, mekaniğin işe yaradığının sayısal kanıtı.
2. Müşteri olarak **Ahmet Bey**'i seç, **Görüşmeyi başlat**. İki AI ajanı sesli konuşur.
   Sağ panelde algılanan duygu, churn riski ve "avcının beyni" canlı güncellenir.
3. **"Gizli gerçeği göster"**: turuncu kesikli çizgi müşterinin gerçek ruh halidir, siyah işaretler
   ise gizli tercihleridir. Asistanın bunları hiç görmeden yalnızca ses ve sözlerden yakaladığını göster.
4. Aynı şeyi **Zeynep Hanım** ile tekrarla: bu kez empati ve sıcak ton öne çıkar. Aynı asistan farklı
   bir insana farklı davranmayı öğrenir.

## Dosyalar

| Dosya | İçerik |
|---|---|
| `core/learner.py` | Thompson Sampling öğrenicisi (avcının beyni) |
| `core/emotion.py` | Akustik özellikler + Gemini ses ve duygu analizi |
| `core/call.py` | Görüşme döngüsü, müşteri ve asistan ajanları, araçlar |
| `core/catalog.py` | Stratejiler, tonlar, sorunlar, araçlar, müşteri profilleri |
| `core/llm.py` | Gemini sarmalayıcısı (JSON çıktı, ses girişi, TTS) |
| `app.py` | FastAPI sunucusu |
| `static/index.html` | Panel |

## Sınırlamalar ve sonraki adımlar

* Müşteri simülasyonunda ruh hali kurala dayalıdır (gizli tercih tablosu), müşteri bunu LLM ile
  sözlere ve tona döker. Bu tasarım, öğrenmenin ölçülebilir olması için bilinçli olarak seçildi.
* Gerçek bir görüşme Gemini ile yaklaşık 1,5-2 dakika sürer (tur başına ~7 sn; sesler çalınırken bir
  sonraki tur arka planda hazırlanır).
* Asistanın duyduğu ses, müşteri ajanının TTS çıktısıdır. Gerçek kullanımda burası canlı mikrofon ya da
  çağrı kaydı olur; mimari değişmez (`emotion.analyze_audio`).
* Sonraki adımlar: canlı mikrofon modu, Gemini Live API ile gerçek zamanlı akış, gerçek çağrı
  kayıtlarıyla duygu modelinin doğrulanması, segment bazlı bağlamın CRM verisiyle zenginleştirilmesi.
