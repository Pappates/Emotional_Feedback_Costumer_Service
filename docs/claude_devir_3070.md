# Devir notu: Aşama 2'yi RTX 3070'li bilgisayarda sürdürmek

Bu belge, şirket bilgisayarındaki Claude Code oturumunun özetidir. Yeni makinedeki Claude Code'a "bu dosyayı oku,
buradan devam ediyoruz" demek için yazıldı. Oturum geçmişi makineler arasında taşınmaz; gereken bağlam burada.

Tarih: 26.11.2026. Dal: `asama2/s2s-baseline` (main'e birleştirilmedi, PR açılmadı).

---

## 1. Proje ve kurallar

**Duygu Avcısı**: Telekom için çağrı merkezi sesli asistan PoC'u. Müşterinin öfkesini ölçer, asistanın cevabını
buna göre yönlendirir. Aşama 1 (metin üzerinden hat: Gemma + Piper, Ollama) çalışıyor; ayrıntılar README'de.
Aşama 2: sesten sese modele geçiş. Seçilen temel **TR-Omni-E4B** (Türkçe, Apache-2.0); öfke ölçümü ayrı model
olarak **emotion2vec+ base** ile yapılır.

Kesin kurallar:
* **Yalnızca açık ve sentetik veri.** Gerçek Telekom çağrı kayıtları KVKK onayı olmadan Colab'a, buluta, yabancı
  API'lere ya da kişisel bilgisayarlara gitmez.
* Depoya eklenmeyecekler: `data/kayitlar/`, `.env`, kimlik bilgisi içeren her dosya (ör. `sd.py`), depo içindeki
  iç içe kopya `Emotional_Feedback_Costumer_Service/`. Commit öncesi `git status` ile kontrol et.
* Kullanıcının e-posta adresi ve kişisel bilgileri dış servislere gönderilmez.
* OAuth, Drive izni, şifre gibi onayları kullanıcı kendisi verir.

Çalışma alışkanlıkları:
* Kullanıcıyla Türkçe, kısa ve öz konuşulur (kullanıcı "caveman" kısa üslubunu açmıştı). Kod, commit mesajı ve
  belgeler düz, açık Türkçe yazılır.
* Commit mesajları Türkçe, kısa başlık + gerekirse gövde; sonuna
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` eklenir. Push yalnızca kullanıcı isteyince.
* Kod değişikliklerinden önce sahte modelle (stub) test edildi; aynı alışkanlık sürmeli (bkz. bölüm 6).

---

## 2. Ölçülen sonuçlar (L4 GPU, eğitimsiz modeller)

Ayrıntı: `docs/sesten_sese_saha_arastirmasi.md` bölüm 7; sayılar: `egitim/sonuc_s2s_baseline.json`.
Ölçüm betiği: `egitim/s2s_baseline.py` (bölümler `--bolum` ile seçilir), defter: `egitim/s2s_baseline.ipynb`.

* **Öfke algısı:** TR-Omni sesten öfkeyi duymuyor (160 kaydın hepsine "sakin", UAR şans düzeyi). emotion2vec+
  base "kızgın mı?" UAR 0,875. Karar: öfkeyi emotion2vec+ ölçer, TR-Omni'ye metin notu olarak verilir.
* **Enjeksiyon:** öfke notu empatik cevabı %40 → %85'e çıkarıyor, cevaplar kısalıyor.
* **Gecikme:** TR-Omni'nin kendi yolunda ilk ses ~9,4 sn. Konuşucu (VoxCPM2) akışla + `torch.compile` ile ilk
  parçayı ~0,3 sn'de veriyor.
* **GPU paylaşımı sorunu:** tek L4'te düşünür ve konuşucu aynı anda çalışınca konuşucu gerçek zamana yetişmiyor
  (oran 1,11). Difüzyon adımı 16 → 4 yalnızca 1,07 → 0,90 kazandırıyor.
* **Ses kalitesi:** akış ve derleme sesi bozmuyor (Whisper ile 45 cümlede 0 bozuk; UTMOS 3,71-3,77). 8 adım
  3,68 (kabul), 4 adım 2,83 (bozuk). Önerilen: akış + derleme + 8 adım.
* **Önce düşünür, sonra konuşucu (`omni_sirali`):** öfke notu + "en fazla iki kısa cümle" ile ilk ses ortalama
  **2,0 sn**, takılma yok, empati %75, sızıntı yok. **Toplantıda ~2 sn kabul edildi.** Önerilen kurulum bu.
* **Uçtan uca demo (`omni_demo`):** kod hazır ve stub testinden geçti, ama Colab'da L4 bulunamadığı için hiç
  çalışmadı. Whisper + emotion2vec+ ile canlı öfke ölçümü (`core/psych.anger_score`) → not + kısa cevap →
  önce düşünür sonra konuşucu; adım süreleri ve `demo_ses/` altında cevap sesleri.

---

## 3. Şimdiki hedef: TR-Omni'yi RTX 3070 (8 GB) üzerinde çalıştırmak

Ev bilgisayarı: RTX 3070 8 GB, Ryzen 5 5600 (6 çekirdek), 32 GB RAM. (İkinci laptop RTX 4060 Laptop 8 GB +
16 GB RAM; RAM az olduğu için 3070'li makine tercih.) Şirketin Jupyter sunucusu internete kapalı; 21 GB'lık
modeli oraya taşımak şimdilik mümkün değil.

### 3.1 Bellek nereye gidiyor (TR-Omni dosyaları, Hugging Face'teki boyutlar)

| Parça | Dosya | Not |
|---|---|---|
| Düşünür (Gemma-4 E4B tabanlı) | `thinker/` 4 parça, toplam ~15,9 GB (bf16) | 42 katman, gizli boyut 2560, MLP 10240, sözlük 262.144 |
| ...bunun içinde katman başına gömme tablosu (PLE) | ~5,6 GB | `tr_omni.py` zaten **CPU RAM'de** tutuyor (`HostEmbedding`) |
| Konuşucu (VoxCPM2) | `talker/model.safetensors` ~4,6 GB + `audiovae.pth` ~0,38 GB + adaptör ~0,07 GB | bf16 |

L4'te ölçülen toplam GPU kullanımı ~17 GB: düşünürün GPU'daki kısmı ~10 GB, konuşucu ~5 GB, kalanı KV önbelleği
ve ara değerler. **Aşağıdaki rakamlar kaba tahmindir, ölçülmedi.**

| Senaryo | Tahmini GPU belleği | 8 GB'a sığar mı |
|---|---|---|
| Hiç değişiklik yok | ~17 GB | Hayır |
| Düşünür katmanları 4-bit, gerisi bf16 | ~4-5 GB (düşünür) + ~5 GB (konuşucu) + ~1 GB = ~10 GB | Hayır |
| + görüntü kodlayıcı GPU'dan çıkarılır (resim kullanmıyoruz) | ~9-9,5 GB | Sınırda değil, taşar |
| + konuşucunun dil modeli int8 | ~7-8 GB | **Sınırda** |
| + düşünürün sözlük gömmesi/`lm_head` (~1,3 GB) de 8-bit | ~6,5-7 GB | Muhtemelen |

### 3.2 `tr_omni.py` hakkında bilinmesi gerekenler

Dosya model klasöründe gelir (`tr_omni.py`, `tr_speak.py`). Önemli yerler:

* `Thinker.__init__`: `AutoModelForMultimodalLM.from_pretrained(path, dtype=torch.bfloat16)`, PLE tablosu
  `HostEmbedding` ile CPU'ya alınıyor, sonra `model.cuda()`. **4-bit yükleme buraya eklenecek** (ör.
  `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)` ve
  `.cuda()` yerine `device_map`; ya da yükledikten sonra torchao `int4_weight_only`). PLE CPU'da kalmalı.
* `FastDecoder`: statik KV önbelleği + `torch.compile(mode="reduce-overhead")` ile CUDA grafiği. bitsandbytes
  4-bit ile çalışmayabilir; çalışmazsa `TROmni(path, fast=False)` (HF `generate`, daha yavaş). torchao
  ağırlık-yalnızca nicemleme `torch.compile` ile uyumludur, önce o denenebilir.
* `Thinker.states()`: konuşucu her cümlede düşünürün 14/28/42. katman gizli durumlarını kullanıyor. Yani
  konuşma sırasında düşünür de GPU'da olmalı; iki modeli sırayla GPU'ya alıp çıkarmak bu yüzden pahalı.
* `Talker.__init__`: `VoxCPM2Model.from_local(path, optimize=False, device="cuda")`. HF modeli değil; int8 için
  torchao `quantize_(m.base_lm, int8_weight_only())` gibi genel bir yol gerekir. Ses çözücü (`audio_vae`) bf16
  kalsın.
* Her şey `"cuda"` sabitine yazılmış; tek GPU olduğu için sorun değil.

### 3.3 Ortam

* **WSL2 + Ubuntu önerilir.** `torch.compile` (Triton) ve bitsandbytes Linux'ta sorunsuz; Windows'ta ikisi de
  sorunlu olabilir. NVIDIA'nın Windows sürücüsü WSL2'ye GPU'yu verir; WSL içinde ayrıca sürücü kurulmaz.
* Python 3.11-3.13, PyTorch CUDA sürümü (Colab'da `torch 2.14.1+cu130` kullanıldı).
* Paketler: `pip install "transformers>=5.17" voxcpm==2.0.3 funasr librosa soundfile zeyrek wordfreq bitsandbytes torchao`
* Model (~21 GB): `huggingface-cli download furkanbekmezci/TR-Omni-E4B-Turkish-Native-Speech-to-Speech-Model --local-dir ~/tr_omni`
* Depo: `git clone -b asama2/s2s-baseline https://github.com/Pappates/Emotional_Feedback_Costumer_Service`
* Müşteri sesleri (git'te yok): Google Drive `DuyguAvcisi/enjeksiyon_ses/` (20 `.wav` + `sonuc.jsonl`) →
  `data/enjeksiyon_ses/`. ElevenLabs ile üretilmiş sentetik sesler, kişisel veri yok.

### 3.4 Önerilen adımlar

1. **Bellek yoklaması** (yeni küçük betik, ör. `egitim/omni_8gb.py`): modeli yükle, her adımdan sonra
   `torch.cuda.memory_allocated()` ve `max_memory_allocated()` yaz. Sıra: düşünür 4-bit → görüntü kodlayıcıyı
   CPU'ya al → konuşucu bf16 → tek bir kısa cevap üret ve seslendir. Taşarsa konuşucunun dil modelini int8 yap.
   Hedef: tepe kullanım ≤ ~7,3 GB (Windows/WSL ekran için pay bırakır).
2. **Hız:** sığarsa `s2s_baseline.py`'deki `sirali()` mantığıyla (önce düşünür, sonra konuşucu, akış + 8 adım)
   20 müşteri sesinde ilk ses ve takılma ölç. Kıyas: L4'te `ofkeli_kisa` ilk ses 2,01 sn, oran 0,97, takılma 0.
   3070'in bant genişliği (608 GB/s) L4'ten (300 GB/s) yüksek; sığarsa hız benzer ya da iyi olabilir.
3. **Kalite:** nicemleme düşünürün Türkçesini bozabilir. Aynı 20 seste cevapları L4 sonuçlarıyla karşılaştır:
   empati oranı, sızıntı, bozuk Türkçe (`core/tr_check.bad_words`), kelime sayısı. Konuşucu int8 olduysa ses
   kalitesini Whisper CER ve UTMOS ile ölç (`omni_kalite`, `omni_mos` bölümlerinin mantığı).
4. **Entegrasyon:** en kolayı `s2s_baseline.py`'ye nicemleme seçeneği eklemek (ör. ortam değişkeni ya da
   `--nicem` argümanı, `omni()` fonksiyonu içinde) ve `omni_sirali` / `omni_demo` bölümlerini evde
   çalıştırmak. Sonuçlar GPU adıyla anahtarlandığı için L4 satırlarıyla aynı dosyada yan yana durur.
5. Sonuçları bölüm 7'ye ve `egitim/sonuc_s2s_baseline.json`'a ekle; kullanıcı isterse push'la.

Başarı ölçütü: 3070'te tepe bellek < 8 GB, ilk ses ≤ ~2,5 sn, takılma yok, empati ve Türkçe kalitesi L4'e
yakın. Sığmazsa denenecek son seçenek: modelin bir kısmını `accelerate` ile CPU RAM'e taşımak (32 GB var), ama
bu ciddi yavaşlatır; demo için anlamlı olmayabilir.

---

## 4. Asıl ince ayar (henüz başlanmadı)

Aşama 1'deki karar: **ince ayar emotion2vec+ base ile yapılacak** (README, "Omurga seçimi"). TR-Omni'ye ince
ayar şimdilik gerekmiyor; öfke notu yaklaşımı çalışıyor.

* emotion2vec+ base ~90M parametre, ~360 MB; 3070'te rahat eğitilir.
* Veri: TurEV-DB (açık, 1.735 kayıt, 4 duygu), YODAS2 kesitleri (`egitim/yodas_hazirla.py`), ElevenLabs sentetik
  müşteri sesleri, varsa `/etiket` sayfasından insan etiketleri.
* Doğrulama konuşmacı-dışarıda (test konuşmacısı eğitimde yok). Bugünkü çıta: TurEV 4 duygu UAR 0,775,
  "kızgın mı?" UAR 0,875.
* Eğitim betiği (`egitim/ser_ince_ayar.py`) henüz yazılmadı; kullanıcı TR-Omni'nin 3070 denemesini öne aldı.
* Şirket Jupyter sunucusu internete kapalı: gerçek veriyle eğitim ileride orada, çevrim dışı paketle (kod + ~360
  MB model + `.whl` dosyaları, Jupyter Upload ile) yapılır.

---

## 5. Açık işler

* `omni_demo`'yu bir kez çalıştırmak (3070'te ya da Colab L4'te).
* TR-Omni'yi 3070'e sığdırma denemesi (bölüm 3).
* PR: `asama2/s2s-baseline` → main. Şirket bilgisayarında `gh` kurulu değil, panelde GitHub oturumu yoktu.
* emotion2vec+ ince ayar betiği ve eğitimi (bölüm 4).
* Mikrofonla canlı demo betiği (yok).
* Uygulamaya (`core/call.py`) bağlama: 24 GB+ GPU'lu bir sunucu kararı gerektirir; kiralık GPU şimdilik istenmedi.

---

## 6. Test alışkanlığı

`s2s_baseline.py` değişiklikleri GPU'suz bir makinede sahte modelle denenir: `tr_omni.py` ve `tr_speak.py`'nin
küçük taklitleri (düşünür sabit token üretir, `states` None döner, konuşucu sıfır ses verir) bir klasöre konur,
`PYTHONPATH` ile öne alınır; `speak_akis`, `asr`, `e2v` gibi ağır fonksiyonlar betik içinde lambda ile
değiştirilir ve `main()` `--bolum <yeni bölüm>` ile çalıştırılır. Çıktı `.jsonl`, `ozet.json` ve ses dosyaları
kontrol edilir. Defterde değişiklikten sonra her kod hücresi (`!` satırları `pass` yapılarak) derlenip denetlenir.

---

## 7. Colab notları (gerekirse)

* Colab Pro hesabı. T4 (15 GB) TR-Omni'ye yetmez. L4 sık sık müsait olmuyor; olmayınca Colab sessizce T4'e düşer.
  O zaman model inmeden çalışma zamanı silinmeli.
* Defter GitHub'dan açılır; Colab bazen eski kopyayı gösterir, betik ise `git pull` ile her zaman günceldir.
* Sonuçlar Drive'da `DuyguAvcisi/s2s_baseline/`; çalıştırma kaldığı yerden devam eder (biten satırlar atlanır).
* İşi biten L4'ü hemen kapatma: arka arkaya iş varsa tekrar L4 bulmak saatler sürebiliyor.
