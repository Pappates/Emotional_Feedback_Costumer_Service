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

## Hızlı başlangıç (tek tık)

1. Python 3.10+ kurulu olsun (Windows'ta kurulumda **"Add python.exe to PATH"** kutusunu işaretle).
2. Depoyu indir: `git clone https://github.com/Pappates/Emotional_Feedback_Costumer_Service`
   (ya da GitHub'da **Code → Download ZIP**).
3. **Windows:** `baslat.bat` dosyasına çift tıkla. **macOS / Linux:** terminalde `./baslat.sh`.

İlk çalıştırmada script sanal ortamı kurar, paketleri yükler ve Gemini API anahtarını sorar
(boş bırakırsan demo modunda açılır). Anahtar proje klasöründeki `.env` dosyasına yazılır
(git'e gönderilmez). Sonraki açılışlarda doğrudan başlar ve tarayıcıda http://localhost:8000 açılır.
Anahtarı değiştirmek için `.env` dosyasını düzenle ya da sil.

## Kurulum (elle, GPU gerekmez)

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

## Lokal mod: kota yok, bulut yok (Aşama 0)

`.env` dosyasına `SAGLAYICI=lokal` satırını eklersen uygulama hiçbir bulut servisine gitmez; bütün modeller
bilgisayarında çalışır. GPU gerekmez. İlk kurulumda `baslat.bat` anahtar sorduğunda `lokal` yazarak da seçebilirsin.

| İş | Lokal model | CPU'da süre (4 çekirdek, ölçülen) |
|---|---|---|
| Diyalog (asistan + müşteri) | **Ollama** üzerinden `gemma3:4b` (JSON şemalı çıktı) | cevap başına ~17-23 sn (4 çekirdekli test makinesi; daha güçlü CPU'da daha hızlı) |
| Ses → metin | **faster-whisper** `small` (int8) | ~1,4 sn / cümle |
| Sesten duygu | `superb/wav2vec2-base-superb-er` | ~0,1 sn |
| Metinden duygu | `savasy/bert-base-turkish-sentiment-cased`, cümle cümle puanlanıp ortalanır (×0,6) | ~0,1 sn |
| Churn niyeti, konu | Türkçe anahtar kelime kuralları | anlık |
| Seslendirme | **Piper** `tr_TR-dfki-medium` (hız, ses yüksekliği, perde ile tonlama) | ~0,2 sn |

Tam ses analizi (Whisper + iki duygu modeli + akustik) tur başına ~1,5-2,5 sn sürer.

**Kurulum (Windows):**
1. Ollama'yı kur: https://ollama.com/download
2. `.env` dosyasına `SAGLAYICI=lokal` satırını ekle.
3. `baslat.bat`'ı çalıştır. Lokal paketleri (torch, whisper, transformers, piper) kurar ve Ollama modelini indirir.
   Uygulama açılınca diğer modeller arka planda Hugging Face'ten iner; panelde "Modeller yükleniyor…" rozeti görünür.
   İlk açılışta toplam ~3 GB indirme olur, sonraki açılışlar internetsiz çalışır.

Şirket ağında `huggingface.co`, `*.hf.co` ve `ollama.com` erişimi gerekebilir. Proxy varsa `.env`'e
`HTTPS_PROXY=http://proxy:port` eklenebilir. Modeller indikten sonra tamamen çevrimdışı çalışmak için `.env`'e
`HF_HUB_OFFLINE=1` eklenebilir.

**Ayarlar (`.env`):**

| Değişken | Varsayılan | Açıklama |
|---|---|---|
| `SAGLAYICI` | `gemini` | `lokal` yapınca bulut kullanılmaz |
| `LOKAL_LLM` | `gemma3:4b` | Ollama modeli. Daha hızlı ama daha zayıf: `gemma3:1b`; daha iyi ama yavaş: `gemma3:12b` |
| `LOKAL_WHISPER` | `small` | `base` daha hızlı, `medium` daha doğru |
| `LOKAL_SES_DUYGU` | `superb/wav2vec2-base-superb-er` | Hugging Face'teki herhangi bir audio-classification modeli |
| `LOKAL_METIN_DUYGU` | `savasy/bert-base-turkish-sentiment-cased` | Hugging Face'teki herhangi bir Türkçe text-classification modeli |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama başka makinedeyse |

**Küçük modellerle yaşanan sorunlar ve alınan önlemler** (gerçek modellerle test edildi):

| Sorun | Önlem |
|---|---|
| `qwen2.5:3b` Türkçede rolleri karıştırdı (müşteri temsilci gibi konuştu), karşı tarafın cümlelerini aynen tekrarladı, "[adres]" gibi yer tutucular yazdı | Varsayılan model `gemma3:4b` yapıldı; aynı testte doğal ve tutarlı Türkçe konuştu |
| Konuşma tek metin bloğu olarak verilince küçük model metni "devam ettiriyor" | Her ajan konuşmayı kendi bakış açısından gerçek sohbet rolleriyle (user/assistant) görüyor |
| Müşteri, temsilcinin sözünü tekrarlıyor ya da temsilci gibi konuşuyor | Benzerlik ve kalıp kontrolü; bozuksa yeniden üretim, yine bozuksa hazır müşteri cümlesi (raporda işaretlenir) |
| Model "arıza kaydı açıyorum" deyip aracı çağırmıyor | Lokal modda işlem, cevabın metninden çıkarılıyor (raporda "metinden çıkarıldı" diye işaretlenir) |
| Uzun "iç ses" JSON'u yarıda kesiyor; satır sonu JSON'u bozuyor | Uzunluk sınırlı şema, hata toleranslı okuma, iki deneme |
| Araç adını müşteriye okuma ("fatura_incele_iade aracını..."), Kiril harfleri ("Haklıсыз"), "(İşlem sonucu: ...)" notunu okuma | Cevap temizleme |
| Metin duygu modeli karışık mesajlarda ±1'e doyuyor | Cümle cümle puanlama + ölçek daraltma: MAE 0,63 → 0,30 |

Bu önlemlerden sonra gerçek `gemma3:4b` ile yapılan testte iki görüşme de **müşteriyi tutarak ve iki sorunu da
çözerek** bitti (Zeynep Hanım: algı −0,38 → +0,26, gerçek −0,55 → +0,04; Ahmet Bey: algı −0,33 → +0,38,
gerçek −0,75 → +0,38).

**Hâlâ var olan sınırlamalar (Aşama 2'de eğitilecek kısımlar):**
* **Metin duygu modeli** ürün yorumlarıyla eğitilmiş; kelimelere bakıyor. İyileştirmeden sonra bile gerçek ruh haliyle
  korelasyonu ~0,6. Çağrı merkezi verisiyle ince ayar gerekiyor. (Gemma'nın kendisini duygu puanlayıcı olarak denedik:
  hem yavaş, 7 sn/cümle, hem de daha kötü sonuç verdi.)
* **Hız:** 4 çekirdekli test makinesinde LLM cevabı ~20 sn, tur başına (asistan + müşteri) ~35-45 sn.
  Demo için "Sesleri çal" kapalı metin modu ve önceden kaydedilmiş görüşmeler önerilir.
* **Ses duygu modeli** İngilizce oyunculu konuşmayla eğitilmiş; Türkçe öfkeli müşteriyi "mutlu" (%59) sınıfladı.
  Bu yüzden füzyonda ağırlığı düşük (valensin %40'ı). Türkçe duygulu konuşma verisiyle ince ayar gerekiyor.
* **Piper'da tek Türkçe ses** var; konuşmacılar perde farkıyla ayrılıyor ve ses duyguyu gerçekten taşımıyor
  (yalnızca hız ve ses yüksekliği değişiyor). Duygulu ve farklı sesler için XTTS veya Chatterbox gibi bir model gerekir.
* Churn niyeti ve konu şimdilik anahtar kelime kurallarıyla bulunuyor.

Raporlarda lokal analiz "lokal: ses + metin" olarak görünür. Metin ve ses modellerinin ayrı çıktıları da
`rapor.json` içinde `lokal_ayrinti` alanında durur; böylece iki sağlayıcıyı raporlarla karşılaştırabilirsin.

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

## Ayrıntılı duygu raporları

Panelden yapılan her görüşme bittiğinde arka tarafta otomatik bir rapor oluşur:
`data/raporlar/<görüşme id>/` içinde **rapor.html** (okunabilir rapor), **rapor.csv** (Excel için, `;` ayraçlı,
ondalık virgüllü) ve **rapor.json** (ham veri). Rapor sayfasındaki **HTML indir** linki raporu tek başına açılabilen bir dosya
olarak indirir (grafikler ve stil dosyanın içinde; yalnızca ses kaydı linkleri uygulama açıkken çalışır), böylece
e-postayla ya da sohbette paylaşılabilir. Görüşme bitince panelde "Detaylı duygu raporunu aç" linki çıkar;
tüm görüşmelerin karşılaştırması için panelin sağ üstündeki **Tüm raporlar** sayfası (`/raporlar`) kullanılır.

Raporda her tur için:

| Bölüm | İçerik |
|---|---|
| Müşteri | Söylenen cümle, gizli ruh hali, TTS tonlaması, ses kaydı linki |
| Ruh hali değişimi | Asistanın önceki hamlesinin etkisi bileşen bileşen: strateji etkisi, ton etkisi, çözüm bonusu, kampanya, sabır cezası, gürültü |
| Algı | Duygu etiketi, valens, algı hatası (algılanan − gerçek), churn riski, konu, ses ve söz ipuçları, algının kaynağı (ses+metin / yalnızca metin) |
| Akustik | Süre, enerji (dB), perde (Hz), perde dalgalanması, konuşma hızı, akustik gerginlik ve uyarılmışlık füzyonunun hesabı |
| Ödül | `(yeni valens − önceki valens) + 0,15 × çözülen sorun` hesabının sayılarla açılımı ve kırpma |
| Seçim | Seçilen strateji ve ton, gerekçesi, yönetmen uyarısı, her kol için tahmin / belirsizlik / Thompson örneği |
| Asistan | Cevap, iç ses, çalıştırılan araçlar ve sonuçları |

Görüşme düzeyinde metrikler: algı hatası (MAE ve ortalama sapma), algı ile gerçek ruh hali arasındaki Pearson
korelasyonu, **yön uyumu** (gerçek ruh hali anlamlı değiştiğinde algının doğru yönde değişme oranı), ortalama ödül,
öğrenilen tercihlerle gizli tercihlerin karşılaştırması (en iyi kol eşleşmesi ve Spearman sıra korelasyonu).
Raporun "Yöntem" bölümü, duygu değişiminin nasıl analiz edildiğini formülleriyle anlatır.

Hızlı simülasyon ("60 görüşme simüle et") görüşmeleri rapor üretmez.

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
| `core/report.py` | Görüşme başına ayrıntılı duygu raporu (HTML/CSV/JSON) ve özet sayfası |
| `core/emotion.py` | Akustik özellikler + Gemini ses ve duygu analizi |
| `core/call.py` | Görüşme döngüsü, müşteri ve asistan ajanları, araçlar |
| `core/catalog.py` | Stratejiler, tonlar, sorunlar, araçlar, müşteri profilleri |
| `core/llm.py` | Gemini sarmalayıcısı (JSON çıktı, ses girişi, TTS); lokal modda `core/local.py`'a yönlendirir |
| `core/local.py` | Lokal sağlayıcı: Ollama, faster-whisper, duygu modelleri, Piper |
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
