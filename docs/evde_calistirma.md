# Aşama 2 demosunu başka bir bilgisayarda çalıştırma

Bu not, Aşama 2 çalışmasını (TR-Omni-E4B sesten sese) şirket bilgisayarı dışındaki bir makinede sürdürmek için
yazıldı. Claude Code oturumu makineler arasında taşınmaz; yeni oturumda bu dosyayı ve aşağıdaki belgeleri okutmak
bağlamı verir.

## Nerede kaldık (26.11.2026)

* Dal: `asama2/s2s-baseline` (main'e henüz birleştirilmedi, PR açılmadı).
* Ölçüm sonuçları: `docs/sesten_sese_saha_arastirmasi.md` bölüm 7, sayılar `egitim/sonuc_s2s_baseline.json`.
* Varılan kurulum (tek L4 GPU): öfkeyi emotion2vec+ ölçer, kullanıcı turuna metin notu olarak eklenir, cevap en
  fazla iki kısa cümle istenir; TR-Omni'nin düşünürü cevabı bitirince konuşucu akış + `torch.compile` + 8 difüzyon
  adımıyla seslendirir. İlk ses ~2,0 sn, ses kesilmiyor, empatik cevap %75. Toplantıda ~2 sn kabul edildi.
* `omni_demo` (uçtan uca tur) Colab L4'te çalıştı ve sunumda gösterildi: müşteri sustuktan sonra ilk ses 2,53 sn,
  takılma yok, empati %55. Cevap ve müşteri sesleri GPU'suz oynatmak için `demo/omni_demo/demo.html`
  (tarayıcıda aç). Ayrıntı bölüm 7'nin sonunda.

## Donanım şartı

TR-Omni-E4B bf16 hâliyle ~17 GB GPU belleği ister (düşünür + konuşucu birlikte).

| GPU | Bellek | Durum |
|---|---|---|
| RTX 3090 / 4090 / 5090, L4, A100 | 22-80 GB | Çalışır |
| RTX 3070 / 3070 Ti | 8 GB | **Olduğu gibi sığmaz.** Nicemlemeyle sığdırma planı: `docs/claude_devir_3070.md` |
| CPU | - | Pratik değil: cevap başına tahminen 1-2 dk (ölçülmedi) |

8 GB'lık bir kartla seçenekler:

1. **Demo çıktısını Colab'da üret, başka makinede oynat.** `omni_demo` Colab L4'te bir kez çalışır; `demo_ses/`
   altındaki cevap sesleri, müşteri sesleri ve `omni_demo.jsonl`'deki süreler herhangi bir bilgisayarda (GPU'suz)
   gösterilebilir. En az riskli yol.
2. **Aşama 1 hattını (Gemma + Piper, Ollama) GPU'lu makinede çalıştırmak.** Bu hat 8 GB'a sığar ve şirket
   bilgisayarındaki CPU'ya göre çok daha hızlı cevap verir. Kurulum: README "Lokal mod". Sesten sese değildir.
3. **TR-Omni'yi kuantize edip denemek.** Düşünürü 4-bit, konuşucuyu bf16 yükleyip belleğin yetip yetmediği ve
   kalitenin korunup korunmadığı ölçülmeli. Belirsiz bir deneme; demo için önerilmez.

## 24 GB'lık bir GPU ile kurulum

Linux ya da Windows'ta WSL2 (Ubuntu) önerilir: ölçülen hızlandırma `torch.compile` kullanır ve Windows'ta sorunlu.
Doğrudan Windows'ta derlemesiz yol çalışır ama daha yavaştır (L4'te gerçek zaman oranı 1,07, yani sınırda).

1. Depoyu klonla ve dala geç: `git clone -b asama2/s2s-baseline https://github.com/Pappates/Emotional_Feedback_Costumer_Service`
2. Müşteri seslerini kopyala: Drive'daki `DuyguAvcisi/enjeksiyon_ses/` (20 `.wav` + `sonuc.jsonl`) ->
   `data/enjeksiyon_ses/`. Bu sesler ElevenLabs ile üretilmiş sentetik seslerdir, git'te tutulmaz.
3. Paketler (PyTorch CUDA sürümü kurulduktan sonra): `pip install "transformers>=5.17" voxcpm==2.0.3 funasr librosa soundfile zeyrek wordfreq`
4. Modeli indir (~21 GB): `huggingface-cli download furkanbekmezci/TR-Omni-E4B-Turkish-Native-Speech-to-Speech-Model --local-dir tr_omni`
5. Demoyu çalıştır:
   `python egitim/s2s_baseline.py --turev yok --model tr_omni --musteri data/enjeksiyon_ses --cikti s2s_cikti --bolum omni_demo`
   (`--turev` yalnızca TurEV bölümlerinde kullanılır; bu bölümde boş bir yol vermek yeterli.)
6. Sonuç: `s2s_cikti/ozet.json` içinde `omni_demo` özeti, `s2s_cikti/demo_ses/` altında cevap sesleri.

## Veri kuralları

* Yalnızca açık ve sentetik veri. Gerçek çağrı kayıtları KVKK onayı olmadan bu makinelere ya da buluta gitmez.
* `data/kayitlar/`, `.env` ve kimlik bilgisi içeren dosyalar depoya eklenmez.

## Sıradaki işler

* Mikrofonla canlı demo için küçük bir betik (henüz yok).
* PR'ı açıp main'e almak.
* TR-Omni'yi uygulamaya (`core/call.py`) bağlamak: GPU'lu bir sunucu kararı gerektirir.
