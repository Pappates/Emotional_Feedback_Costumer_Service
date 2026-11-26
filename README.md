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
| Diyalog (asistan + müşteri) | **Ollama** üzerinden `gemma3:4b` (JSON şemalı çıktı) | asistan cevabı ~14-20 sn (4 çekirdekli test makinesi, önbellek açık; daha güçlü CPU'da daha hızlı) |
| Ses → metin | **faster-whisper** `small` (int8) | ~1,4 sn / cümle |
| Sesten duygu | `superb/wav2vec2-base-superb-er` | ~0,1 sn |
| Metinden duygu | `savasy/bert-base-turkish-sentiment-cased`, cümle cümle puanlanıp ortalanır (×0,6) | ~0,1 sn |
| Churn niyeti, konu | Türkçe anahtar kelime kuralları | anlık |
| Seslendirme | **Piper**, her konuşmacıya ayrı Türkçe ses + Türkçe metin normalleştirme | ~0,15 sn |
| Seslendirme (isteğe bağlı, asistan) | **Chatterbox Multilingual** (ONNX, q4) — çok daha doğal, `LOKAL_TTS=chatterbox` | ~15 sn / cümle |

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
| `LOKAL_SES_DUYGU` | `emotion2vec/emotion2vec_plus_base` | Ses duygu modeli (Aşama 1 karşılaştırmasının kazananı, FunASR ile). Eski model: `superb/wav2vec2-base-superb-er`; Hugging Face'teki herhangi bir audio-classification modeli de verilebilir |
| `LOKAL_SES_VALENS_AGIRLIK` | `0.3` | Valens füzyonunda sesin payı (kalanı metin) |
| `LOKAL_SES_UYARILMA_AGIRLIK` | `0.7` | Uyarılma füzyonunda ses modelinin payı (kalanı akustik ölçüm) |
| `LOKAL_METIN_DUYGU` | `savasy/bert-base-turkish-sentiment-cased` | Hugging Face'teki herhangi bir Türkçe text-classification modeli |
| `OLLAMA_URL` | (otomatik) | Verilirse uygulama kendi Ollama sunucusunu açmaz, bu adresi kullanır |
| `LOKAL_GECMIS` | `4` | Lokal modda modele verilen son mesaj sayısı (fazlası promptu ve bekleme süresini büyütür) |
| `LOKAL_OZEL_OLLAMA` | `1` | `0` yapılırsa önbellek ayarlı özel Ollama sunucusu başlatılmaz |
| `LOKAL_TTS` | `piper` | Asistan sesi motoru: `piper` (hızlı) ya da `chatterbox` (doğal ama yavaş) |
| `LOKAL_ASISTAN_SESI` | `dfki` | Asistanın Piper sesi: `dfki`, `fahrettin`, `99eren99` (erkek) ya da `fettah` (kadın) |
| `LOKAL_CHATTERBOX_SES` | (modelin sesi) | Chatterbox için ses örneği (WAV yolu); asistanın sesi bu örnekten klonlanır |
| `LOKAL_ASR_SOZLUK` | `1` | Whisper'a alan sözlüğü verilir (POS, iade, fatura…); `0` kapatır |
| `LOKAL_TEKRAR_CEZASI` | `1.0` | Dil modelinin tekrar cezası. Türkçede ekleri cezalandırdığı için kapalı (eskiden 1,15) |
| `LOKAL_SICAKLIK_CARPANI` | `0.7` | Tüm sıcaklıkların çarpanı (düşük = daha tutarlı, daha az yaratıcı) |
| `LOKAL_YAZIM_DENETIMI` | `1` | Türkçe yazım denetimi (zeyrek + wordfreq); `0` kapatır |
| `ODUL` | `ofke` | Öğrenen beynin ödülü: `ofke` (kızgınken öfke azalması, sakinken valens artışı) ya da `valens` (eski) |
| `ENJEKSIYON` | `neden` | Her müşteri dönüşünde talimata eklenen durum: `yok`, `ofke` (düzey + eğilim), `neden` (+ değerlendirme), `tam` (+ önerilen yaklaşım) |
| `LOKAL_MUSTERI_SESI` | `piper` | Müşteri sesi: `piper` (yerel) ya da `elevenlabs` (duygulu, bulut; kredi harcar) |
| `ELEVENLABS_API_KEY` | (boş) | ElevenLabs anahtarı; yalnızca `.env`'de tutun, git'e girmez |
| `ELEVEN_ERKEK_SES` / `ELEVEN_KADIN_SES` | Adam / (boş) | ElevenLabs ses kimlikleri; kadın sesi boşsa kadın müşteriler Piper ile konuşur |

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
| Asistan uydurma söz veriyor ("faturanız iptal edildi", "para yarın 10:00'da hesabınızda") | İşlemlerin gerçek sonuçları "şirket kuralları" olarak asistana önceden veriliyor; uydurma tarih/saat/tutar yasak |
| Asistan "işlemi tamamladım" deyip aracı çağırmıyor | Lokal modda asistanın kendi algıladığı konuyu çözen araç yönergede açıkça söyleniyor; "sil/kaldır/düzelt" gibi fiiller de metinden işlem çıkarımına eklendi |
| Ses duygu modeli öfkeli Türkçe müşteriyi %80-95 "mutlu" sınıflıyor, algıyı yukarı çekiyor (ortalama sapma +0,20) | Ses modelinin valensteki payı %40 → %10 (`LOKAL_SES_VALENS_AGIRLIK`); uyarılmışlıkta kullanılmaya devam ediyor |
| Müşteri ısrar edince asistan kurallar dışı süre veriyor ("yarım saat içinde gönderiyorum") | Kural dışı süre/saat sözü yakalanınca cevap yeniden üretiliyor; yine varsa o cümle müşteriye söylenmeden çıkarılıyor (raporda gösterilir) |
| "Randevuyu iptal etmem gerekiyor" churn sayıldı (%85) | Ayrılma ifadeleri yalnızca abonelik/hat/operatörle ilgili olunca sayılıyor |

Bu önlemlerden sonra gerçek `gemma3:4b` ile yapılan testte iki görüşme de **müşteriyi tutarak ve iki sorunu da
çözerek** bitti (Zeynep Hanım: algı −0,38 → +0,26, gerçek −0,55 → +0,04; Ahmet Bey: algı −0,33 → +0,38,
gerçek −0,75 → +0,38).

**Hâlâ var olan sınırlamalar (Aşama 2'de eğitilecek kısımlar):**
* **Metin duygu modeli** ürün yorumlarıyla eğitilmiş; kelimelere bakıyor. İyileştirmeden sonra bile gerçek ruh haliyle
  korelasyonu ~0,6. Çağrı merkezi verisiyle ince ayar gerekiyor. (Gemma'nın kendisini duygu puanlayıcı olarak denedik:
  hem yavaş, 7 sn/cümle, hem de daha kötü sonuç verdi.)
* **Hız:** bkz. "Hız ve ses kalitesi". 4 çekirdekli test makinesinde asistan cevabı ~14-20 sn.
  Demo için "Sesleri çal" kapalı metin modu ve önceden kaydedilmiş görüşmeler önerilir.
* **Ses duygu modeli** İngilizce oyunculu konuşmayla eğitilmiş; Türkçe öfkeli müşteriyi "mutlu" (%59) sınıfladı.
  Bu yüzden füzyonda ağırlığı düşük (valensin %10'u). Türkçe duygulu konuşma verisiyle ince ayar gerekiyor.
* **Piper sesleri duyguyu gerçekten taşımıyor**; ton yalnızca hız, ses yüksekliği ve küçük perde farklarıyla veriliyor
  (yalnızca hız ve ses yüksekliği değişiyor). Duygulu ve farklı sesler için XTTS veya Chatterbox gibi bir model gerekir.
* Churn niyeti ve konu şimdilik anahtar kelime kurallarıyla bulunuyor.

Raporlarda lokal analiz "lokal: ses + metin" olarak görünür. Metin ve ses modellerinin ayrı çıktıları da
`rapor.json` içinde `lokal_ayrinti` alanında durur; böylece iki sağlayıcıyı raporlarla karşılaştırabilirsin.

### Hız ve ses kalitesi (ölçümler)

**Dil modeli hızı.** CPU'da sürenin yarısı prompt işleme (saniyede ~95 token), yarısı üretim (saniyede ~6,5 token).
Asistan ve müşteri sırayla aynı modeli kullandığı için her çağrı bir öncekinin önbelleğini siliyor ve büyüyen konuşma
geçmişi her seferinde baştan işleniyordu. Yapılanlar:

| Değişiklik | Etki |
|---|---|
| Sabit kısımlar (kurallar, araç listesi) sistem promptunda, bu tura özgü kısa yönerge en sonda | Konuşma geçmişi önbellekten kullanılabiliyor |
| Ollama'da `OLLAMA_NUM_PARALLEL=2`: asistan ve müşteri ayrı önbellek "slot"u kullanıyor | 1016 tokenlik promptun yalnızca ~250'si yeniden işleniyor |
| Lokal modda "iç ses" modele yazdırılmıyor, cevaplar en fazla 2 cümle | Üretilen token 95-126 → 50-67 |
| **Toplam** | **Asistan cevabı 32-43 sn → 14-20 sn** (aynı makine, aynı model) |

**Kısa prompt (lokal mod, varsayılan):** kullanıcı makinesindeki ölçümlerde Ollama, Gemma 3 ile konuşma geçmişini turdan
tura önbellekten güvenilir şekilde kullanamadı (yalnızca aynı rolün art arda çağrılarında kullanıldı); büyüyen geçmiş
her turda baştan işlendi (1124 token = 16 sn). Bu yüzden lokal modda sistem promptu ~480 → ~200 tokene indirildi, yalnızca
son 4 mesaj tutuluyor (`LOKAL_GECMIS`), daha eskisi tek satırlık "yapılanlar" özetiyle veriliyor. Prompt artık turdan
tura büyümüyor (testte 300-670 token).

**Önbellek ayarını uygulama kendisi yapar:** `OLLAMA_NUM_PARALLEL=2` ile kendi Ollama sunucusunu 11435 portunda
başlatır (modeller aynı klasörden okunur, yeniden indirilmez). Ortam değişkeni ayarlamak ya da Ollama'yı yeniden
başlatmak gerekmez. Konsolda `[lokal] Ollama önbellek ayarıyla ayrı sunucuda başlatıldı` satırı görünür. Kapatmak için
`.env`'e `LOKAL_OZEL_OLLAMA=0` yazılabilir; `OLLAMA_URL` verilirse de özel sunucu açılmaz. Not: Ollama'nın kendi
uygulaması da modeli bellekte tutuyorsa bir süre iki kopya bellekte kalabilir; RAM azsa Ollama simgesinden "Quit" yapılabilir.

Önbelleğin çalıştığını konsoldaki satırlardan görebilirsin: `[lokal] LLM 15 sn = prompt 250 token (...)`. Buradaki
token sayısı her turda birkaç yüz düzeyinde kalıyorsa önbellek çalışıyor; turdan tura büyüyorsa çalışmıyor demektir
(uygulama bu durumda konsola bir uyarı yazar, nedeni `data/ollama_ozel.log` dosyasında olur).

**Cümle cümle seslendirme (lokal + sesli mod, varsayılan).** Asistan artık cevabın tamamını beklemiyor: model yazdıkça
her tamamlanan cümle denetleniyor (kural dışı söz, tekrar, müşteriyi yankılama), seslendiriliyor ve hemen çalınıyor;
geri kalan cümleler bu sırada üretiliyor. Müşterinin cevabı da asistanın son cümleleri çalarken hazırlanmaya başlıyor.
Akış `POST /api/call/{id}/agent_stream` (SSE) üzerinden: önce `analiz` (seçilen strateji/ton, duygu analizi), sonra her
cümle için `cumle`, en sonda `son` (araç sonuçları, iç ses, rapor için birleştirilmiş ses). Akış şemasında `cevap`
alanı `araclar`dan önce geldiği için ilk cümle araç listesini beklemez. Modelin "kontrol ediyorum" deyip aracı çağırmayı
unutması durumunda araç yine metinden çıkarılır. Test makinesinde (4 çekirdek, saniyede ~6,5 token):

| | Eski (tam cevap) | Cümle cümle |
|---|---|---|
| Müşteri sustuktan sonra asistanın ilk sesi | 11-18 sn | **1-5 sn** (müşteri konuşurken düşünmeye başlıyor) |
| Modelin ilk parçası | – | 2-4 sn |

Gemini ve demo modu eskisi gibi tek seferde çalışır. Kayıttan tekrar oynatma da aynı kalır.

**Türkçe okuma ve doğallık.** 12 cümle × 2 tekrar seslendirildi, ses Whisper medium ile geri dinlenip kelime hata oranı
(WER) ölçüldü:

| Seçenek | WER | Cümle başı süre |
|---|---|---|
| Eski: dfki, ham metin | %9,4 | 0,15 sn |
| **dfki + Türkçe normalleştirme** | **%2,9** | 0,15 sn |
| fahrettin / fettah / 99eren99 + normalleştirme | %4,5-5,7 | 0,15 sn |
| MMS-TTS (Meta) + normalleştirme | %16,7 | 0,94 sn (lisansı ticari kullanıma kapalı, elendi) |
| Chatterbox Multilingual (q4) + normalleştirme | 4 cümlede 1 hata | ~15 sn |

* **Normalleştirme** (`core/tr_text.py`): "16:00-18:00" → "on altı ile on sekiz", "%30" → "yüzde otuz",
  "240 TL" → "iki yüz kırk lira", "50 GB" → "elli gigabayt", "Mbps'e" → "megabite", "10'dan" → "ondan".
  Eskiden saatler "16-00-18-00", GB "gebe" diye okunuyordu.
* **Yapaylık:** eskiden tek ses farklı perdelere kaydırılarak (×0,86 - ×1,24) farklı konuşmacılar üretiliyordu; bu sesi
  belirgin şekilde yapaylaştırıyordu. Artık her konuşmacının kendi sesi var (asistan dfki, Ahmet fahrettin, Mert 99eren99,
  Zeynep ve Selin fettah). Cinsiyet, ölçülen temel frekanstan belirlendi.
* **En doğal ses** Chatterbox: modern bir model, duygu yoğunluğu ayarı (asistanın tonuna bağlandı) ve ses klonlama var.
  CPU'da yavaş olduğu için isteğe bağlı; sunum için önceden kayıt almaya uygun (kayıtlar API'siz tekrar oynatılır).

## Aşama 1: ölçüm temeli ve model seçimi

Amaç: ince ayardan (Aşama 2) önce **neyi, neyle ölçeceğimizi** ve **hangi modeli** eğiteceğimizi belirlemek.

### Psikolojik katman (`core/psych.py`)
Valens/uyarılmaya PAD modelinin üçüncü boyutu **baskınlık** (talepkâr ↔ çaresiz) ve değerlendirme (appraisal) kuramına
göre dört yorum boyutu eklendi: **suç yükleme** (kurum/durum/kendisi), **belirsizlik**, **kontrol kaybı**, **haksızlık algısı**
(+ vazgeçmişlik). Bunlardan çağrı merkezi kategorisi türetilir: öfke, hayal kırıklığı, kaygı, kafa karışıklığı,
**küskünlük** (sessiz churn), rahatlama, memnuniyet. Her kategori için bir yaklaşım önerilir (örn. belirsizlik → teknik
açıklama, haksızlık → telafi). Strateji seçimini yine öğrenen asistan yapar; öneri yalnızca konuşma biçimini etkiler
ve raporda seçimle karşılaştırılır. Raporda her turda "Psikolojik değerlendirme" bölümü, CSV'de yeni sütunlar var.

### Gerçek ses test seti (`/etiket`)
Paneldeki **Test seti etiketleme** bağlantısı: ekip arkadaşları 24 senaryo kartını (her kategori ve değerlendirme
birleşimi) tarayıcıdan kaydeder, en az 3 kişi her kaydı SAM ölçeğiyle (1-9) valens/uyarılma/baskınlık, kategori ve
değerlendirme boyutlarıyla puanlar. Sayfa etiketleyiciler arası uyumu (Krippendorff α, ≥0,667 kabul edilebilir) ve
mevcut modelin bu setteki başarısını (CCC, kategori UAR) gösterir. Kayıtlar yalnızca `data/etiket` altında kalır,
GitHub'a gönderilmez. Hedef: 5-8 konuşmacı × 24 kart ≈ 120-200 kayıt.

### İnternetten gerçek Türkçe konuşma (YODAS2, `egitim/yodas_hazirla.py`)
CC BY 3.0 lisanslı YouTube konuşmaları (YODAS2 Türkçe, otomatik altyazılı `tr100` bölümü). 3 parçadan 1.322 kesit
(3-8 sn) ön etiketlendi: %80'i nötr (beklenen), duygulu kesitler ses modeline göre: üzgün 119, mutlu 79, öfkeli 34,
tiksinti 13, korku 11, şaşkınlık 10. Seçim: 241 kesit (196 duygulu + 45 nötr, 101 farklı video, video başına en fazla 4).
Ön etiket etiketleyiciye gösterilmez; özet sayfası "ön etiket ↔ insan" uyumunu ölçer. Paketi başka bilgisayara almak:
`python egitim/yodas_hazirla.py ice-aktar yodas_tr_300.zip` (kaynak listesi `data/etiket/KAYNAKLAR_yodas.md`).
Daha fazla duygulu kesit için daha çok parça: `topla --parcalar 3 4 5` (parça başına ~12 dk, ~65 duygulu kesit).

### Simülasyon ve ses düzeltmeleri
* **Müşteri metni ruh haliyle tutarlı:** sakinleşen müşteri hâlâ bağırıyordu (raporda gerçek +0,29, algı −0,43).
  Her ruh hali için somut konuşma tarzı veriliyor; üretilen cümle metin duygu modeliyle denetleniyor, uyumsuzsa yeniden üretiliyor.
* **Duyguya göre Piper ayarı:** hız, ses seviyesi (artık tepe değere değil konuşma seviyesine göre) ve canlılık
  (noise) ruh haline bağlandı. eGeMAPS ölçümü: öfkeli ile kararsız arasındaki ses yüksekliği farkı 1,29/0,96 → 1,22/0,57.
  Sınır: Piper'da tonlama eğrisi kontrol edilemiyor, perde değişkenliği duyguyla artmıyor (Aşama 3: duygulu TTS eğitimi).
* **Whisper alan sözlüğü:** "POS" → "Post", "iade" → "Yadem" hataları için kısa bağlam cümlesi. 3 ses × 10 cümlede
  kelime hata oranı %30,7 → %16,9.

### Omurga (backbone) seçimi (`egitim/`)
`ser_bench.py` + `ser_eval.py`: aday modeller TurEV-DB'de (6 konuşmacı, 1735 kayıt, 4 duygu) donmuş öznitelik +
lojistik regresyonla, **konuşmacı-dışarıda** doğrulamayla karşılaştırılır (katman seçimi test konuşmacısını görmez).
`egitim/ser_model_secimi.ipynb` aynı karşılaştırmayı Colab GPU'sunda tüm adaylarla tekrarlar.

**Sonuçlar** (TurEV-DB, 4 çekirdekli CPU; ham sonuç `egitim/sonuc_cpu_turev.json`). UAR şans düzeyi 0,25;
uyarılma/valens ikili görevlerde 0,50. RTF: işlem süresi / ses süresi (CPU, aynı anda başka işler de çalışırken).

| Omurga | 4 duygu UAR | ±konuşmacı | Uyarılma | Valens | RTF (CPU) | Hazır tahmini |
|---|---|---|---|---|---|---|
| **emotion2vec+ base** (~90M) | **0,773** | **±0,063** | **0,880** | **0,831** | 0,09 | UAR 0,79 |
| emotion2vec+ large (~300M) | 0,708 | ±0,072 | 0,858 | 0,773 | 0,15 | UAR 0,72 |
| whisper-base (30 sn dolgu) | 0,550 | ±0,130 | 0,813 | 0,609 | 0,95 | – |
| whisper-small (4 sn pencere) | 0,528 | ±0,060 | 0,784 | 0,646 | 0,14 | – |
| XLS-R 300M | 0,522 | ±0,113 | 0,784 | 0,651 | 0,25 | – |
| whisper-base (4 sn pencere) | 0,509 | ±0,081 | 0,762 | 0,611 | 0,05 | – |
| WavLM base+ | 0,475 | ±0,142 | 0,754 | 0,660 | 0,09 | – |
| audeering (valens/uyarılma/baskınlık) | 0,470 | ±0,077 | 0,723 | 0,635 | 0,14 | uyarılma AUC 0,82, valens AUC 0,50 |
| mHuBERT-147 | 0,440 | ±0,079 | 0,749 | 0,613 | 0,09 | – |
| superb wav2vec2 (şu an kullanılan) | 0,436 | ±0,108 | 0,766 | 0,611 | 0,14 | **UAR 0,27** |
| eGeMAPS (88 özellik) | 0,431 | ±0,103 | 0,755 | 0,528 | 0,03 | – |

**Colab (T4 GPU) tekrarı** (`egitim/sonuc_colab_turev.json`): sonuçlar CPU ölçümüyle aynı çıktı (emotion2vec+ base 0,774,
large 0,708; superb 0,464), yani karşılaştırma tekrarlanabilir. Yalnızca Colab'da ölçülebilen **whisper-small (30 sn dolgu)**:
UAR 0,603, ±0,108, uyarılma 0,814, valens 0,683; genel konuşma modellerinin en iyisi ama emotion2vec+ base'in gerisinde.

**Uygulamaya alındı (eğitimsiz hâli):** lokal modda ses duygu modeli artık emotion2vec+ base. Hazır olasılıkları valens ve
uyarılmaya çevrildiğinde TurEV'de valens AUC 0,90, uyarılma AUC 0,95 (eski model 0,46 / 0,60). Bu yüzden füzyon
`valens = 0,7·metin + 0,3·ses`, `uyarılma = 0,7·ses + 0,3·akustik` oldu. Kayıt başına ~0,1 sn (CPU). Model
yüklenemezse (ör. `funasr` kurulu değilse) uygulama eski modele düşer ve konsola yazar. Not: simülasyondaki Piper
müşteri sesi duygu taşımadığından model onu doğru şekilde %90-100 "nötr" duyuyor; ses kanalının simülasyonda da
işe yaraması için duygulu TTS (Aşama 3) gerekiyor.

**Karar: ince ayar emotion2vec+ base ile yapılacak.** Hem en doğru hem konuşmacılar arası en tutarlı, CPU'da gerçek
zamanın ~11 katı hızlı; large sürümden de iyi. Diğer bulgular:
* Şu an kullanılan ses modeli Türkçede şans düzeyinde (0,27): rapordaki "ses modeli: hap" çıktısının nedeni.
* Duyguya özel ön eğitim, genel konuşma modellerinden (WavLM, XLS-R, mHuBERT, Whisper) açık ara iyi; boyut değil ön eğitim belirleyici.
* İngilizce eğitimli audeering modeli Türkçede uyarılmayı tanıyor (AUC 0,82) ama valensi hiç tanımıyor (0,50):
  ses dilden bağımsız olarak uyarılmayı taşır, valens dile ve kültüre bağlıdır. Valens Türkçe veriyle öğretilmeli ve metinle birleştirilmeli.
* Sınırlar: TurEV tek kelimelik, oyunculu kayıtlardan oluşuyor; gerçek konuşmada sayılar düşer. Asıl ölçüm `/etiket`
  test setinde yapılacak. emotion2vec+ lisansı "FunASR model lisansı"dır; ticari kullanımdan önce hukuk kontrolü gerekir.

### Türkçe kalitesi (lokal mod)
Gemma 3 4B zaman zaman bozuk Türkçe üretiyor. Ölçüm: gerçek bir görüşmeden alınan 15 istek, 3 ayar × 2 örnek (90 yanıt).
* **Yazım denetimi** (`core/tr_check.py`): kelime biçimbilimsel olarak çözümlenemiyor (zeyrek) VE Türkçe sıklık
  listesinde yoksa (wordfreq) bozuk sayılır; gelecek zaman, gereklilik ve -lik ekleri için ek kontroller var.
  Test: 15 zor ama doğru kelimede yanlış alarm yok, 12 bozuk kelimenin hepsi yakalandı ("düzelacak", "söyleyorsunuz",
  "almıssınız"…). Müşteri cümlesi bozuksa bir kez yeniden üretilir; asistanın bozuk cümlesi söylenmez; raporda görünür.
* **Örnekleme:** tekrar cezası kaldırıldı (Türkçe ekleri cezalandırıyordu), sıcaklık ×0,7. Yazım hatası oranı üç
  ayarda da %0-0,8; asıl sorun dilbilgisi/anlam ("faturala dalıyor musunuz") ve modelin boyutundan kaynaklanıyor.
* **JSON/yönerge sızıntısı:** model metni erken kapatıp "”} ```json" ile sürdürebiliyordu; ilk artıkta kesiliyor.
  Müşterinin "Sakin ve yumuşak" gibi yönerge kelimelerini konuşma diye söylemesi engellendi.
* **Denenen alternatif:** Kumru-2B (Türkçe için sıfırdan eğitilmiş, Apache-2.0). Asistan rolünde akıcı ama müşteri
  rolünü oynayamıyor (temsilci gibi konuşuyor), yönergeyi sızdırıyor, kelimeleri bitiştiriyor ("ikigündür"); elendi.

## Revize plan, Aşama 1: öfke odağı

Toplantı kararı: hedef "müşteri kızgın mı, ne kadar kızgın, öfkeyi hangi yöntem azaltıyor".

* **Öfke düzeyi 0–4** (`core/psych.py`, `anger_score`): ses kanalı emotion2vec+ "kızgın" olasılığı × gerginlik (kızgın mı
  normal mi), metin kanalı olumsuzluk + talep, suçu kuruma yükleme, ünlem, ayrılma tehdidi (derece). Raporda öfke eğrisi
  (ölçülen ve gizli ruh halinin karşılığı), her turda ses/metin ayrımıyla öfke satırı; panelde duygu yanında öfke.
* **Ödül:** müşteri kızgınken `0,25 × (önceki öfke − yeni öfke) + 0,15 × çözülen sorun`; iki tur da sakinse valens
  artışı. Hızlı eğitim (3 tohum × 120 görüşme, son 60): kalma oranı öfke+valens %92,2, yalnızca öfke %87,8,
  valens %92,8, rastgele %38,9. Bağlam artık "konu · kızgın/sakin"; yönetmen öfke ≥ 3'te de uyarır.
* **Prompt enjeksiyonu** (`ENJEKSIYON`): her müşteri dönüşünden sonra talimata öfke düzeyi ve eğilimi, nedeni ve önerilen
  yaklaşım eklenir. Eşleştirilmiş deney (`egitim/enjeksiyon_deneyi.py`, 4 persona × 2 görüşme, 29 durum × 4 varyant):
  müşteri tepkisindeki öfke `yok` 0,94 · `ofke` 0,84 · **`neden` 0,60** · `tam` 0,81. Yalnızca `neden` belirgin
  (yok'a göre −0,33, standart hatanın ~2,7 katı; müşteri kızgınken −1,15). Öneri cümlesi (`tam`) cevabı genel kalıplara
  kaydırıyor. Varsayılan `neden`.
* **ElevenLabs müşteri sesi** (`LOKAL_MUSTERI_SESI=elevenlabs`): eleven_v3, ruh haline göre ses etiketi ([angry],
  [frustrated], [calm]…). Test: öfkeli cümlede öfke 2,9 (ses 3,8), sakin cümlede 0. Ücretsiz hesapta ayda 10.000
  karakter; bir görüşme ~400–700 karakter. Hata ya da kota dolunca otomatik Piper'a döner.
* **Etiketleme:** `/etiket`'te "öfke düzeyi 0–4" sorusu, öfke uyumu (α) ve "kızgın mı?" doğruluğu.

## Aşama 2: sesten sese model (hazırlık)

Hedef: metin aracılığı olmadan sesi doğrudan anlayan ve sesle cevap veren model. Saha araştırması:
`docs/sesten_sese_saha_arastirmasi.md`. Seçilen temel **TR-Omni-E4B** (Türkçe, Apache-2.0); öfke ölçümü ayrı model
olarak **emotion2vec+ base** ile sürer. Çağrı kaydı olmadığından eğitim açık ve sentetik veriyle yapılacak.

* **Baseline** (`egitim/s2s_baseline.ipynb`, Colab L4/A100): eğitimden önce TR-Omni'nin TurEV'de duygu algısı, ElevenLabs
  müşteri tepkilerinde 0-4 öfke puanı ve sesli cevapları (süre, Türkçe yazım, cevap sesinin duygusu) ile emotion2vec+
  base'in aynı örneklemdeki hazır etiketi ölçülür. Betik: `egitim/s2s_baseline.py`; sonuçlar `ozet.json`.
* **Baseline sonucu (24.11.2026, L4)** `egitim/sonuc_s2s_baseline.json`:
  * emotion2vec+ base: 4 duygu UAR 0,775, "kızgın mı?" UAR 0,875.
  * TR-Omni: 160 kaydın hepsine "sakin" dedi (4 duygu UAR 0,24, "kızgın mı?" 0,50, yani şans düzeyi).
  * TR-Omni'nin öfke puanı ölçülen öfkeyle ilişkisiz (Spearman −0,32, n=20).
  * TR-Omni'nin sesli cevabı ortalama 24 sn sürüyor (cevap sesi 13 sn, akış yok). Cevap sesinin tonu 20/20 nötr,
    1/20 cevapta bozuk kelime var.
  * Sonuç: öfke ayrı modelle (emotion2vec+) ölçülüp TR-Omni'ye metin olarak verilir. TR-Omni'ye duygu algısı
    öğretmek birinci öncelik değil. Asıl darboğaz gecikme; sıradaki ölçüm akışla ilk ses süresi ve A100 süresi.
  * Gecikme dökümü (L4): düşünürün ilk cümlesi 2 sn, ama ilk ses ortalama 9,4 sn. Darboğaz konuşucu (VoxCPM2,
    gerçek zaman oranı 1,65). Sıradaki deneme `omni_konusucu`: akışlı çözme, derleme, yarı difüzyon adımı.
  * Konuşucu akışla çalışınca ilk ses ~1,4 sn'ye iniyor. Ama tek L4'te düşünür aynı anda çalışırken konuşucu gerçek
    zamana yetişmiyor (oran 1,11); difüzyon adımını azaltmak az kazandırıyor (4 adımda 0,90). Seçenekler: önce
    düşünür sonra konuşucu, iki GPU, ya da daha hızlı bir Türkçe TTS.
  * Ses kalitesi: akış ve derleme sesi bozmuyor (UTMOS 3,71-3,77, Whisper ile 45 cümlenin hiçbiri bozuk değil).
    8 difüzyon adımı çok az düşürüyor (3,68), 4 adım belirgin bozuyor (2,83). Önerilen: akış + derleme + 8 adım.
  * Önce düşünür sonra konuşucu (GPU paylaşılmadan): öfke notu + "en fazla iki kısa cümle" ile ilk ses ~2,0 sn,
    ses kesilmiyor, empati %75. Tek L4 için önerilen kurulum bu.
  * Enjeksiyon çalışıyor: öfkeli notta empati %40'tan %85'e çıkıyor, cevaplar kısalıyor.
    Ayrıntı: `docs/sesten_sese_saha_arastirmasi.md` bölüm 7.

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
| `core/psych.py` | Psikolojik katman: baskınlık, değerlendirme boyutları, çağrı merkezi kategorileri |
| `core/etiket.py` | Gerçek ses test seti: kayıt, etiket, Krippendorff alfa, CCC ölçümü |
| `app.py` | FastAPI sunucusu |
| `static/index.html` | Panel |
| `static/etiket.html` | Test seti kayıt ve etiketleme sayfası (`/etiket`) |
| `egitim/` | Model seçimi (omurga karşılaştırması) betikleri ve Colab not defteri |

## Sınırlamalar ve sonraki adımlar

* Müşteri simülasyonunda ruh hali kurala dayalıdır (gizli tercih tablosu), müşteri bunu LLM ile
  sözlere ve tona döker. Bu tasarım, öğrenmenin ölçülebilir olması için bilinçli olarak seçildi.
* Gerçek bir görüşme Gemini ile yaklaşık 1,5-2 dakika sürer (tur başına ~7 sn; sesler çalınırken bir
  sonraki tur arka planda hazırlanır).
* Asistanın duyduğu ses, müşteri ajanının TTS çıktısıdır. Gerçek kullanımda burası canlı mikrofon ya da
  çağrı kaydı olur; mimari değişmez (`emotion.analyze_audio`).
* Sonraki adımlar: canlı mikrofon modu, Gemini Live API ile gerçek zamanlı akış, gerçek çağrı
  kayıtlarıyla duygu modelinin doğrulanması, segment bazlı bağlamın CRM verisiyle zenginleştirilmesi.
