"""Sesten duygu (SER) omurga karşılaştırması, adım 1: öznitelik çıkarımı.

Her aday modelin her katmanından, kayıt başına ortalama alınmış gömme çıkarılır (model donmuş, eğitim yok).
Ardından ser_eval.py konuşmacı-dışarıda (LOSO) lojistik regresyonla hangi omurganın duyguyu en iyi taşıdığını ölçer.
İnce ayar (Aşama 2) için omurga bu ölçüme göre seçilir.

Kullanım: python ser_bench.py <TurEV "Sound Source" klasörü> <önbellek klasörü> model1 model2 ...
GPU varsa (Colab) otomatik kullanılır. Veri: https://github.com/Xeonen/TurEV-DB (CC BY-NC-ND: araştırma/PoC).
"""
import glob, json, os, sys, time
import numpy as np, soundfile as sf, torch
import librosa

torch.set_num_threads(max(1, os.cpu_count() or 1))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SRC, CACHE, MODELS = sys.argv[1], sys.argv[2], sys.argv[3:]
os.makedirs(CACHE, exist_ok=True)
CLASSES = ["Angry", "Calm", "Happy", "Sad"]
files = sorted(glob.glob(os.path.join(SRC, "*", "*.wav")))
y = np.array([CLASSES.index(os.path.basename(os.path.dirname(f))) for f in files])
spk = np.array([os.path.basename(f).split("_")[0] for f in files])


def load(f):
    x, sr = sf.read(f, dtype="float32")
    return librosa.resample(x, orig_sr=sr, target_sr=16000)


def hf_layers(name, kind):
    from transformers import AutoFeatureExtractor, AutoModel
    fe = AutoFeatureExtractor.from_pretrained(name)
    m = AutoModel.from_pretrained(name).eval().to(DEV)

    def f(x):
        inp = {k: v.to(DEV) for k, v in fe(x, sampling_rate=16000, return_tensors="pt").items()}
        with torch.no_grad():
            hs = m(**inp, output_hidden_states=True).hidden_states
        return np.stack([h[0].mean(0).cpu().numpy() for h in hs])
    return f


def whisper_layers(name, seconds=None):
    from transformers import WhisperFeatureExtractor, WhisperModel
    fe = WhisperFeatureExtractor.from_pretrained(name)
    enc = WhisperModel.from_pretrained(name).encoder.eval().to(DEV)
    if seconds:  # kısa pencere: konum gömmeleri kırpılır (30 sn dolgudan ~7 kat hızlı; temsil biraz farklı)
        npos = int(seconds * 50)
        enc.embed_positions.weight.data = enc.embed_positions.weight.data[:npos]
        enc.config.max_source_positions = npos

    def f(x):
        # 30 sn dolgu: üretimde transkripsiyon sırasında zaten hesaplanan encoder çıktısıyla aynı koşul
        feats = fe(x, sampling_rate=16000, return_tensors="pt").input_features
        if seconds:
            feats = feats[:, :, : int(seconds * 100)]
        n = max(1, int(np.ceil(len(x) / 16000 / 0.02)))
        with torch.no_grad():
            hs = enc(feats.to(DEV), output_hidden_states=True).hidden_states
        return np.stack([h[0, :n].mean(0).cpu().numpy() for h in hs])
    return f


def egemaps():
    import opensmile
    sm = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals)
    return lambda x: sm.process_signal(x, 16000).values[0][None, :]


def emotion2vec(name):
    from funasr import AutoModel
    m = AutoModel(model=name, hub="hf", disable_update=True, device=DEV)
    zs = {}

    def f(x, key=None):
        r = m.generate(x, granularity="utterance", extract_embedding=True)[0]
        zs[key] = (r["labels"], r["scores"])
        return np.asarray(r["feats"])[None, :]
    f.zs = zs
    return f


def audeering_dim():
    # Wav2vec2-large-robust, MSP-Podcast üzerinde uyarılma/baskınlık/valens için ince ayarlı (CC BY-NC-SA)
    import torch.nn as nn
    from transformers import Wav2Vec2Processor
    from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

    class Head(nn.Module):
        def __init__(s, c):
            super().__init__(); s.dense = nn.Linear(c.hidden_size, c.hidden_size); s.dropout = nn.Dropout(0.0)
            s.out_proj = nn.Linear(c.hidden_size, c.num_labels)

        def forward(s, x):
            return s.out_proj(torch.tanh(s.dense(x)))

    class EM(Wav2Vec2PreTrainedModel):
        def __init__(s, c):
            super().__init__(c); s.wav2vec2 = Wav2Vec2Model(c); s.classifier = Head(c); s.post_init()

        def forward(s, v):
            o = s.wav2vec2(v, output_hidden_states=True)
            return o.hidden_states, s.classifier(o.last_hidden_state.mean(1))
    name = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    p = Wav2Vec2Processor.from_pretrained(name); m = EM.from_pretrained(name).eval().to(DEV)
    zs = {}

    def f(x, key=None):
        v = torch.tensor(p(x, sampling_rate=16000).input_values[0])[None].to(DEV)
        with torch.no_grad():
            hs, dim = m(v)
        zs[key] = dim[0].tolist()          # [uyarılma, baskınlık, valens]
        return np.stack([h[0].mean(0).cpu().numpy() for h in hs])
    f.zs = zs
    return f


def superb_er():
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    name = "superb/wav2vec2-base-superb-er"
    fe = AutoFeatureExtractor.from_pretrained(name); m = AutoModelForAudioClassification.from_pretrained(name).eval().to(DEV)
    zs = {}

    def f(x, key=None):
        inp = {k: v.to(DEV) for k, v in fe(x, sampling_rate=16000, return_tensors="pt").items()}
        with torch.no_grad():
            o = m(**inp, output_hidden_states=True)
        zs[key] = o.logits[0].softmax(-1).tolist()
        return np.stack([h[0].mean(0).cpu().numpy() for h in o.hidden_states])
    f.zs = zs
    f.labels = [m.config.id2label[i] for i in range(len(m.config.id2label))]
    return f


BUILD = {
    "egemaps": egemaps,
    "superb_er": superb_er,
    "wavlm_base_plus": lambda: hf_layers("microsoft/wavlm-base-plus", "ssl"),
    "xlsr_300m": lambda: hf_layers("facebook/wav2vec2-xls-r-300m", "ssl"),
    "mhubert_147": lambda: hf_layers("utter-project/mHuBERT-147", "ssl"),
    "whisper_base": lambda: whisper_layers("openai/whisper-base"),
    "whisper_small": lambda: whisper_layers("openai/whisper-small"),
    "whisper_small_4sn": lambda: whisper_layers("openai/whisper-small", seconds=4),
    "whisper_base_4sn": lambda: whisper_layers("openai/whisper-base", seconds=4),
    "emotion2vec_plus_base": lambda: emotion2vec("emotion2vec/emotion2vec_plus_base"),
    "emotion2vec_plus_large": lambda: emotion2vec("emotion2vec/emotion2vec_plus_large"),
    "audeering_dim": audeering_dim,
}

for name in MODELS:
    out = os.path.join(CACHE, name + ".npz")
    if os.path.exists(out):
        print(f"{name}: önbellekte", flush=True); continue
    f = BUILD[name]()
    feats, t_audio, t0 = [], 0.0, time.time()
    for i, path in enumerate(files):
        x = load(path); t_audio += len(x) / 16000
        feats.append(f(x, key=i) if hasattr(f, "zs") else f(x))
        if i % 300 == 0:
            print(f"{name}: {i}/{len(files)}", flush=True)
    el = time.time() - t0
    extra = {}
    if hasattr(f, "zs"):
        extra["zs"] = json.dumps({str(k): v for k, v in f.zs.items()})
    if hasattr(f, "labels"):
        extra["labels"] = json.dumps(f.labels)
    np.savez(out, X=np.stack(feats), y=y, spk=spk, rtf=el / t_audio, device=DEV, **extra)
    print(f"{name}: bitti, RTF {el / t_audio:.3f} ({el:.0f} sn)", flush=True)
