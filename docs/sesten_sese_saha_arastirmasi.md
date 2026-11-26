# Sesten sese (speech-to-speech) modeller: saha araştırması

Tarih: 24 Kasım 2026. Amaç: Duygu Avcısı'ndaki "ses → metin → LLM → metin → ses" hattını, analizi ve konuşma
üretimini doğrudan sesten yapan modellerle değiştirmek için seçenekleri çıkarmak. Model eğitimi planı bu
araştırmanın sonrasına bırakıldı.

> Not: Liderlik tablosu sayıları üçüncü taraf ölçümleridir ve İngilizce senaryolarda alınmıştır. Hiçbir
> kaynakta Türkçe sesten sese karşılaştırması yok; Türkçe kalitesini kendimiz ölçmemiz gerekiyor.

## 1. Şu anki hat (karşılaştırma için)

`core/local.py` ve `core/llm.py`: müşteri sesi → faster-whisper (döküm) → emotion2vec+ (ses duygusu) + metin
duygu analizi → Ollama/Gemini metin LLM'i → Piper / ElevenLabs / Gemini TTS. Asistanın "duyduğu" şey
pratikte dökümdür; ses yalnızca emotion2vec ve akustik özellikler üzerinden yan kanal olarak girer.
Makinede NVIDIA GPU yok.

## 2. En önemli bulgu: sesten sese modeller duyuyor ama dinlemiyor

"Real-Time Voice AI Hears but Does Not Listen" (Bartelds, Bianchi, Zou; Haziran 2026) dört gerçek zamanlı
sesten sese API'yi aynı sözler, farklı söyleyişlerle (ağlayan / korkmuş / alaycı ve nötr) test etti:
GPT Realtime 2, Gemini 3.1 Flash Live, Qwen3.5 Omni Plus Realtime, Qwen3.5 Omni Flash Realtime.

* **Algı var:** doğrudan sorulunca GPT Realtime 2, Gemini Live ve Qwen3.5 Omni Plus duyguyu yakalıyor.
  Qwen3.5 Omni Plus en iyisi (korkulu ses 20/20, sakin kontrol 0/20). Qwen3.5 Omni Flash ayırt edemiyor.
* **Karar yok:** konuşma içinde dördü de sözlere göre davranıyor. Ağlayan arayanın görüşmesini 5/5
  kapattılar, korkmuş sesle istenen havale onayını 4-5/5 verdiler, alaycı gönüllüyü 5/5 kaydettiler.
* **Prompt kısmen işe yarıyor:** "sese dikkat et, yalnızca sözlere göre karar verme" talimatı söylenenleri
  değiştiriyor ama kararları ancak kısmen değiştiriyor.

**Bizim için anlamı:** sesten sese modele geçmek, ayrı bir duygu ölçüm katmanını gereksiz kılmıyor. Kendi
deneyimiz de aynı yönde: öfkeyi ve **nedenini** talimata enjekte etmek (`neden` varyantı) müşteri öfkesini
0,94'ten 0,60'a indirdi. Mimari bu iki bulguyu birleştirmeli: ses doğal olarak modele girsin, ama öfke
sesten ayrıca ölçülüp karara açıkça bağlansın.

## 3. Ticari sesten sese modeller

| Model | Türkçe | Duygu / ifade | Ölçümler (İngilizce) | Not |
|---|---|---|---|---|
| **Google Gemini 3.8 Live** (`gemini-3.8-live`) | Live API 99 dil, `tr` listede | Affective dialog 3.8'de **kaldırıldı**; proactive audio açık | AA STS 74,5; Coval p95 1,25 sn; Cekura pass³ %41,5; AA bench maliyeti ~0,84 $/saat | Live API önizlemede. Forumda "Türkçe 3.8'de 2.5'e göre daha düz ve tutarsız" şikâyeti var |
| Gemini 2.5 Flash native audio | Var | Affective dialog **var** | | **13 Aralık 2026'da emekli** oluyor, üzerine kurulmamalı |
| **OpenAI GPT Realtime 2 / 2.1** | Azure listesinde Türkçe var | Ses tonu talimatla ayarlanıyor | Cekura pass³ %79,3, 0,085 $/dk; AA STS 73,6 | Makalede algısı iyi, kararı sözlere bağlı |
| **OpenAI GPT-Live-1 / 1.1** | Resmi dil listesi yok; 1.1'deki yeni seslerde Türkçe yok | "Konuşanın tonunu ve duygusunu yansıtabilir" | AA STS 80-83; Cekura %75,6, 0,055 $/dk | En yeni OpenAI ses modeli |
| **Alibaba Qwen3.5-Omni Realtime (Plus)** | Alibaba tablosunda Türkçe ses çıkışı var | Duygu, hız, ses düzeyi uçtan uca kontrol; ses duygu sınıflandırma örneği | AA (Qwen Audio 3.0 RT Plus) konuşma dinamiği %98,4; liste fiyatı çok düşük | Makalede en iyi duygu algısı. Veri Çin bulutuna gidiyor |
| Qwen3.8-Omni-Flash | Girişte 74, çıkışta 29 dil; Türkçe ikisinde de var | Raporda duygu ölçümü yok | Coval p95 1,40 sn, yönerge uyumu %100 | Açık ağırlık açıklaması yok |
| **xAI Grok Voice 3.0** | Resmi listede `tr` var | Bilgi yok | AA STS 86,9; Cekura %81,7, 0,08 $/dk | |
| Phonic 1 / 1.1 | 51 dil; Türkçe teyit edilmedi | Bilgi yok | AA STS 87-89; Cekura 1. sıra (%90,2) | |
| ElevenLabs Realtime v1 | 90+ dil | Güçlü ifadeli ses | AA STS 77,3, TTFA 0,57 sn | |
| Amazon Nova 2 Sonic | Teyit edilmedi | | Cekura %50, ses kalitesi 5/5 | |
| Hume EVI 3 / 4-mini | **Yok** | Duyguya odaklı tek ürün | | **EVI API 13 Kasım 2026'da kapandı** |

## 4. Açık ağırlıklı modeller

| Model | Türkçe | Yapı | Duygu | Donanım / lisans |
|---|---|---|---|---|
| **TR-Omni-E4B** (furkanbekmezci) | **Yalnızca Türkçe** | Gemma 4 E4B "düşünür" sesi doğrudan alıyor, ASR yok. "Konuşan" VoxCPM2 / Trendyol-TTS'ten türetilmiş ve düşünürün gizli durumlarıyla besleniyor. Tam uçtan uca değil, hibrit | Cinsiyet %92,5, ses olayları %96,6; **duygu zayıf** (6 sınıf %36,8, n=19). Tek ses, prozodi konuşmaya uymuyor | 24 GB GPU rahat. Yarım dupleks. Apache-2.0. Ölçümler yazarın kendi ölçümü |
| Qwen3-Omni (açık) | Türkçe girişi var, **çıkışı yok** | Uçtan uca düşünür-konuşan | Duygu anlama iyi | Büyük MoE, çok GPU |
| Step-Audio 2 mini | Çince/İngilizce | Uçtan uca ses tokenı | Paralinguistik 80, duygu 82 (kendi testleri) | Apache-2.0 |
| MiniCPM-o 4.5 | Konuşma yalnızca İngilizce/Çince | Tam dupleks, omni | | Açık |
| NVIDIA PersonaPlex 7B | Yalnızca İngilizce | Moshi tabanlı, tam dupleks, ses ve persona kontrolü | | A100, NVIDIA açık model lisansı |
| NVIDIA NemotronLabs VoiceChat 11B | Bilinmiyor | Tam dupleks, araç çağırma | | Açık |
| KRAFTON Raon-SpeechChat 9B | İngilizce (aile EN/KO) | Qwen3 + Mimi; TTFA 0,04 sn | | 16 GB+; **ticari kullanım yok** (CC BY-NC) |
| Kyutai Moshi | İngilizce | İlk açık tam dupleks model | | Yeni dile uyarlanabiliyor (aşağıda) |

**Sonuç:** Türkçe konuşan, tam dupleks, uçtan uca açık bir model yok. En yakını TR-Omni-E4B, ama duygu
tarafı zayıf ve konuşan kısmı ayrı bir TTS türevi.

## 5. Yeni dile kendi modelini eğitmenin maliyeti (referans)

* **Japonca J-Moshi (2025):** 344 saat gerçek stereo diyalog + 602 saat sentetik diyalog ile ince ayar. Ön
  eğitim J-CHAT üzerinde.
* **Hintçe Human-1 (2026):** 26.000 saat iki kanallı gerçek konuşmayla ön eğitim, ~990 saat ince ayar. Mimi
  kodlayıcı donduruldu.
* **Urduca topluluk denemesi:** 38 saat sentetik veri anlamlı konuşma üretmedi. 200+ saat gerçek veri
  gerekiyor.
* **PersonaPlex:** ~2.250 saat sentetik diyalog, ~48 A100-saat ince ayar.

Telekom açısından kritik: çağrı merkezi kayıtları zaten iki kanallı (müşteri / temsilci). Bu, tam dupleks
eğitimin en pahalı girdisi. Birkaç yüz saat etiketli kayıt bile Türkçe ince ayar için anlamlı başlangıç olur.

## 6. Önerilen yön

**Mimari (Aşama 2 önerisi):**

```
müşteri sesi ──► [Sesten sese asistan oturumu]  ──► asistan sesi
     │               ▲  (Gemini Live / GPT Realtime / Qwen-Omni Realtime)
     │               │  oturum talimatı her turda güncellenir: öfke düzeyi + neden + strateji/ton
     ▼               │
[Sesten duygu analizi, paralel]  ──►  [Öğrenici: Thompson Sampling]
 ses-yerli model: öfke 0–4, neden, churn       (mevcut core/learner.py)
 (Qwen-Omni Plus'a doğrudan sor + emotion2vec çapraz kontrol)
```

* **Konuşma üretimi:** sesten sese modelin kendisi. Strateji ve ton metin kalıbı olarak değil, oturum
  talimatı olarak verilir (Gemini Live sistem talimatı, OpenAI `session.update`).
* **Analiz:** döküm analiz girdisi olmaktan çıkar (yalnızca rapor ve kayıt için tutulur). Duygu ve öfke,
  sesi doğrudan dinleyen bir modelle ölçülür. Makaleye göre bunun için en güçlü aday Qwen3.5 Omni Plus;
  emotion2vec+ ucuz çapraz kontrol olarak kalır.
* **Müşteri ajanı:** iki seçenek var.
  * (a) ElevenLabs v3 kalır. Duygu etiketiyle kontrol edilebildiği için ölçülebilir "gizli gerçek" sağlar.
  * (b) Müşteri de sesten sese bir oturum olur (ör. Qwen-Omni'nin duygu kontrolüyle). Daha doğal olur, ama
    ruh hali kontrolü ve ölçülebilirlik zayıflar.

  Öneri: değerlendirme için (a), demo için (b).

**Hemen yapılacak:** Türkçe karşılaştırma testi. Aynı 4 persona ve kızgın / sakin senaryolar, ElevenLabs
müşteri sesiyle. Adaylar: Gemini 3.8 Live, GPT Realtime 2.1 (ve GPT-Live-1), Qwen3.5-Omni Plus Realtime,
Grok Voice 3.0. Ölçülecekler:

1. Türkçe telaffuz ve doğallık (insan puanı + WER)
2. Yanıt gecikmesi
3. Doğrudan sorulduğunda öfke algısı
4. Öfke enjeksiyonuna karar düzeyinde uyum (aynı sözler, kızgın ve sakin ses)
5. Dakika maliyeti

**Sonra (model eğitimi, ayrı planlanacak):** KVKK nedeniyle müşteri sesinin yurt dışı API'lere gitmesi
canlı kullanımda büyük olasılıkla kabul edilmez. Uzun vadede yerinde (on-prem) bir model gerekir. Aday
temel TR-Omni-E4B. Gerçek, etiketli Telekom kayıtlarıyla duygu ve öfke ince ayarı yapılır; tam dupleks daha
sonra Moshi tarzı uyarlamayla gelir. Bu iş GPU gerektirir: çıkarım için 24 GB, eğitim için A100/H100 sınıfı.

## 7. TR-Omni-E4B baseline ölçümü (24.11.2026)

Eğitimsiz modeller, Colab L4. Betik ve defter: `egitim/s2s_baseline.py`, `egitim/s2s_baseline.ipynb`.
Ham sonuç: `egitim/sonuc_s2s_baseline.json`.

| Ölçüm | Sonuç | Şans |
|---|---|---|
| emotion2vec+ base, TurEV 4 duygu UAR (n=160) | 0,775 | 0,25 |
| emotion2vec+ base, "kızgın mı?" UAR | 0,875 | 0,50 |
| TR-Omni, TurEV 4 duygu UAR (sesle sorulunca) | 0,244 | 0,25 |
| TR-Omni, "kızgın mı?" UAR | 0,50 | 0,50 |
| TR-Omni 0-4 öfke puanı ile ölçülen öfke, Spearman (n=20) | −0,32 | 0 |
| TR-Omni sesli cevap üretim süresi, ortalama (n=20) | 24,4 sn (ses 13 sn) | — |
| Cevap sesinin tonu (emotion2vec+) | 20/20 nötr | — |

* TR-Omni 160 kaydın hepsine "sakin" dedi. Soruyu anlıyor (tanınamayan cevap 0), ama sesteki duyguyu
  kullanmıyor. Bu, 2. bölümdeki "duyuyor ama dinlemiyor" bulgusunun daha güçlü bir hâli.
* Öfke ölçümü ayrı ses modeliyle (emotion2vec+) yapılır, sonuç TR-Omni'ye metin olarak verilir.
* Cevap içeriği empatik ama kalıplı. 20 cevabın 1'inde konu kaymış, 1'inde bozuk kelime var.
* En büyük engel gecikme: akış olmadan cevabın tamamı ortalama 24 sn, en uzun 63 sn.

**Gecikme dökümü (L4, n=20, `omni_gecikme`).** Cevap cümle cümle seslendirildiğinde ilk sesin süresi:

| Adım | Süre |
|---|---|
| Düşünürün ilk kelimesi | 0,22 sn |
| Düşünürün ilk cümlesi | 1,96 sn |
| İlk ses (ilk cümle + seslendirmesi) | ort 9,4 sn, medyan 8,7 sn, en çok 23,6 sn |
| Konuşucunun gerçek zaman oranı | 1,65 (konuşmadan yavaş) |

Darboğaz konuşucu (VoxCPM2): bir cümleyi seslendirmek ~7 sn sürüyor. TR-Omni konuşucuyu derleme kapalı
(`optimize=False`) ve akışsız (`streaming=False`) çağırıyor. Sıradaki deneme (`omni_konusucu`): akışlı çözme,
derleme ve yarı difüzyon adımı. Bunlar yetmezse düşünür kalır, konuşucu daha hızlı bir Türkçe TTS ile değiştirilir.

**Konuşucu hızlandırma (L4, n=20, `omni_konusucu`).** Aynı cevaplar, ses VoxCPM2'nin akışlı çözücüsüyle parça parça:

| Ayar | İlk ses parçası | Tahmini ilk ses* | Gerçek zaman oranı |
|---|---|---|---|
| Akış yok (`omni_gecikme`) | — | 9,4 sn | 1,65 |
| Akış | 0,48 sn | 1,60 sn | 1,74 |
| Akış + derleme | 0,31 sn | 1,42 sn | 1,07 |
| Akış + derleme + yarı difüzyon adımı (16 → 8) | 0,29 sn | 1,40 sn | 0,95 |
| Akış + derleme, 6 adım | 0,28 sn | 1,39 sn (medyan) | 0,93 |
| Akış + derleme, 4 adım | 0,27 sn | 1,40 sn (medyan) | 0,90 |
| Akış + derleme, 8 adım, **düşünür aynı GPU'da çalışırken** | 0,48 sn | 1,58 sn (medyan) | **1,11** |

\* Düşünürün ilk cümlesi + ilk ses parçası. Isınma hatası yüzünden ~18 sn çıkan tek kayıt hariç ortalama
(ısınma sonradan iki kayda çıkarıldı).

* İlk ses sorunu akışla çözülüyor: ~1,4 sn, hedef 1,5 sn. Kalan süre çoğunlukla düşünürün ilk cümlesi.
* Akışın kesintisiz olması için oran < 1 gerekir. Difüzyon adımı darboğaz değil: 16 → 4 adımda oran yalnızca
  1,07 → 0,90. Süre çoğunlukla konuşucunun dil modeli adımlarında ve ses çözücüde.
* GPU paylaşımı belirleyici: düşünür arka planda üretirken konuşucunun oranı 0,95 → 1,11 (ses kesilir), düşünür de
  saniyede ~4 token'a düşüyor. (Arka plan üretimi derlemesiz yoldan yapıldı; kaba bir tahmin, ama yön açık.)
* Sonuç: tek L4'te TR-Omni'nin düşünürü ve konuşucusu aynı anda canlı akışa yetişmiyor. Seçenekler: önce düşünür
  sonra konuşucu (ilk ses ~3 sn), iki GPU, ya da düşünür kalıp konuşucunun daha hızlı akışlı bir Türkçe TTS ile
  değiştirilmesi.

**Ses kalitesi (L4, `omni_kalite` + `omni_mos`).** Her ayarda aynı 45 cümle sentezlendi. Anlaşılırlık Whisper
large-v3-turbo ile geri yazımın karakter hata oranıyla (CER), doğallık UTMOS ile (1-5, referanssız) ölçüldü:

| Ayar | Bozuk cümle (CER > 0,3) | UTMOS ort. | En düşük | < 3,0 |
|---|---|---|---|---|
| Akışsız (TR-Omni'nin kendi yolu) | 0/45 | 3,71 | 3,01 | 0 |
| Akış | 0/45 | 3,71 | 3,01 | 0 |
| Akışsız + derleme | 0/45 | 3,77 | 3,26 | 0 |
| Akış + derleme | 0/45 | 3,77 | 3,26 | 0 |
| Akış + derleme, 8 adım | 0/45 | 3,68 | 3,07 | 0 |

Hız ölçümündeki ilk cümleler (ayar başına 5) ayrıca puanlandı: 16 adım 3,88, 8 adım 3,78, 6 adım 3,61,
**4 adım 2,83 (5 örneğin 3'ü 3,0'ın altında)**.

* Akış ve derleme sesi bozmuyor; 8 adım çok az düşürüyor (3,77 → 3,68), sözler her ayarda anlaşılır.
* 4 adım belirgin bozuyor, 6 adım sınırda; hızları 8 adımdan pek farklı olmadığı için ikisi de elendi.
  Kulakla bozuk bulunan örneklerin çoğu düşük puan aldı (4 adım "neden" 2,42; 6 adım "öfke" 3,37).
* Önerilen ayar: akış + derleme + 8 adım. Hız sorunu (GPU paylaşımında oran 1,11) bununla çözülmüyor.
* UTMOS İngilizce konuşmayla eğitilmiş; Türkçede mutlak puan kaba, ayarlar arası kıyas güvenilir.

**Önce düşünür, sonra konuşucu (L4, 20 müşteri × 3 koşul, `omni_sirali`).** Düşünür cevabın tamamını üretir, ardından
konuşucu akış + derleme + 8 adımla seslendirir; GPU aynı anda paylaşılmaz. Takılma: ses parçaları çalmaya
yetişmediğinde dinleyicinin beklediği süre.

| Koşul | Düşünür | İlk ses ort. (medyan / en çok) | Oran | Takılma | Kelime | Empati |
|---|---|---|---|---|---|---|
| Not yok | 2,79 sn | 3,08 sn (2,80 / 6,82) | 0,96 | 0 | 24,3 | %25 |
| Öfke notu | 2,07 sn | 2,35 sn (2,36 / 3,00) | 0,96 | 0 | 17,2 | %65 |
| Öfke notu + "en fazla iki kısa cümle" | 1,73 sn | **2,01 sn (1,98 / 2,44)** | 0,97 | ~0 | 12,8 | %75 |

* GPU paylaşılmayınca konuşucu gerçek zamana yetişiyor: 60 cevabın hiçbirinde ses kesilmedi.
* Kısa cevap talimatı ilk sesi ~2 sn'ye indiriyor ve empatiyi düşürmüyor (%65 → %75); sızıntı yok.
* Sonuç: tek L4'te çalışan kurulum öfke notu + kısa cevap + akış + derleme + 8 adım; ilk ses ~2 sn (ses tanıma ve
  sessizlik algısı hariç). 1,5 sn'nin altı gerekirse iki GPU ya da daha hızlı bir Türkçe TTS gerekir.

**Enjeksiyon (L4, 20 müşteri × 3 koşul, `omni_enjeksiyon`).** Öfke bilgisi kullanıcı turuna metin olarak
verildi (`core/call.py` biçimi):

| | yok | sakin (0,5/4) | öfkeli (3,5/4) |
|---|---|---|---|
| Empati/özür içeren cevap | %40 | %65 | %85 |
| Ortalama kelime | 22,7 | 18,3 | 17,0 |
| "yok" cevabına benzerlik | 1,00 | 0,34 | 0,29 |
| Modelin seçtiği ton | 14 sakin, 3 nötr, 2 ciddi, 1 şefkatli | 18 sakin | 20 sakin |
| Not müşteriye sızdı | 0 | 0 | 1 |

* Enjeksiyon cevabın sözlerini değiştiriyor: öfkeli notta daha empatik ve daha kısa. Mimarinin dayanağı doğrulandı.
* Etkinin bir kısmı notun varlığından: sakin notu da empatiyi artırıyor. Öfke düzeyinin kendi etkisi ~20 puan.
* Tek sızıntı sınırda: "sizi ne kadar öfkelendirdiğini hissedebiliyorum". Kızgın müşteriye öfkesini söylemek riskli.
* Ton etiketi bilgi taşımıyor (neredeyse hep "sakin"); konuşucu da etiketi kullanmıyor.

## Kaynaklar

* [Real-Time Voice AI Hears but Does Not Listen (arXiv 2606.26083)](https://arxiv.org/abs/2606.26083)
* [Artificial Analysis, speech-to-speech](https://artificialanalysis.ai/speech-to-speech)
* [Cekura S2S benchmark](https://benchmarks.cekura.ai/speech-to-speech)
* [Coval, voice AI models 2026](https://www.coval.ai/blog/voice-ai-models-2026/)
* [Gradium, cascade vs speech-to-speech 2026](https://gradium.ai/content/cascaded-voice-agent-vs-speech-to-speech-2026)
* [Gemini Live API capabilities](https://aistudio.google.com/docs/live-api/capabilities)
* [Gemini 3.8 Live Türkçe forum başlığı](https://discuss.ai.google.dev/t/gemini-3-8-live-turkish-speech-flatter-and-less-consistent-than-2-5-native-audio-same-voice/199256)
* [Azure Voice Live dil desteği (gpt-realtime)](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/voice-live-language-support)
* [GPT-Live-1 duyuru (OpenAI Devs)](https://x.com/OpenAIDevs/status/2098099427357724870)
* [xAI speech-to-speech dokümanı](https://docs.x.ai/developers/model-capabilities/audio/speech-to-speech)
* [Phonic desteklenen diller](https://docs.phonic.ai/docs/build/agents/supported_languages)
* [Alibaba Model Studio S2S tablosu](https://help.aliyun.com/en/model-studio/s2s-model)
* [Qwen3.5-Omni teknik raporu](https://arxiv.org/html/2604.15804v2)
* [Qwen3.8-Omni (arXiv 2609.25611)](https://arxiv.org/abs/2609.25611)
* [Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni)
* [Step-Audio 2 teknik raporu](https://arxiv.org/abs/2507.16632)
* [Hume EVI dokümanı](https://dev.hume.ai/docs/speech-to-speech-evi)
* [TR-Omni-E4B](https://huggingface.co/furkanbekmezci/TR-Omni-E4B-Turkish-Native-Speech-to-Speech-Model)
* [Awesome Full-Duplex SDM listesi](https://github.com/Ruiqi-Yan/Awesome-Full-Duplex-SDM)
* [MiniCPM-o 4.5](https://huggingface.co/openbmb/MiniCPM-o-4_5)
* [NVIDIA PersonaPlex](https://huggingface.co/nvidia/personaplex-7b-v1)
* [KRAFTON Raon-SpeechChat](https://huggingface.co/KRAFTON/Raon-SpeechChat-9B)
* [J-Moshi](https://arxiv.org/abs/2506.02979)
* [Human-1 (Hintçe)](https://arxiv.org/abs/2604.23295)
* [Urduca Moshi denemesi](https://github.com/SBasit8/full-duplex-urdu-voice-poc)
