"""Sesten duygu omurga karşılaştırması, adım 2: değerlendirme.

ser_bench.py önbelleğini konuşmacı-dışarıda (LOSO) ölçer; katman seçimi iç LOSO ile yapılır (test konuşmacısı hiç
görülmez, iyimser sonuç çıkmaz). Görevler:
  * 4 sınıf (Öfke/Sakin/Mutlu/Üzgün) dengeli doğruluk (UAR)
  * uyarılma ikili: Öfke+Mutlu / Sakin+Üzgün  (Russell'ın çember modeli)
  * valens ikili:   Mutlu+Sakin / Öfke+Üzgün
Hazır duygu modellerinin kendi (eğitimsiz) tahminleri de raporlanır. RTF: işlem süresi / ses süresi (CPU'da ölçüldüyse).
Kullanım: python ser_eval.py <önbellek klasörü>
"""
import glob, json, os, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

CACHE = sys.argv[1]
AROUSAL = np.array([1, 0, 1, 0])   # Angry, Calm, Happy, Sad
VALENCE = np.array([0, 1, 1, 0])


def clf(dim):
    steps = [StandardScaler()]
    if dim > 256:
        steps.append(PCA(128, random_state=0))
    steps.append(LogisticRegression(C=0.5, max_iter=3000))
    return make_pipeline(*steps)


def loso(X, y, spk, layers):
    """Dış LOSO: her test konuşmacısı için katman, kalan konuşmacılarda iç LOSO ile seçilir."""
    pred = np.zeros_like(y); chosen = []
    for s in np.unique(spk):
        tr, te = spk != s, spk == s
        if len(layers) == 1:
            best = layers[0]
        else:
            scores = []
            for l in layers:
                acc = []
                for s2 in np.unique(spk[tr]):
                    a, b = tr & (spk != s2), tr & (spk == s2)
                    m = clf(X.shape[-1]).fit(X[a, l], y[a]); acc.append(balanced_accuracy_score(y[b], m.predict(X[b, l])))
                scores.append(np.mean(acc))
            best = layers[int(np.argmax(scores))]
        chosen.append(best)
        m = clf(X.shape[-1]).fit(X[tr, best], y[tr]); pred[te] = m.predict(X[te, best])
    per_spk = [balanced_accuracy_score(y[spk == s], pred[spk == s]) for s in np.unique(spk)]
    return balanced_accuracy_score(y, pred), np.std(per_spk), chosen


rows = []
for path in sorted(glob.glob(os.path.join(CACHE, "*.npz"))):
    name = os.path.basename(path)[:-4]
    d = np.load(path, allow_pickle=True)
    X, y, spk, rtf = d["X"], d["y"], d["spk"], float(d["rtf"])
    L = X.shape[1]
    # büyük modellerde her katmanı denemek yerine her ikinci katman (süre)
    layers = list(range(L)) if L <= 13 else list(range(0, L, 2))
    uar, sd, chosen = loso(X, y, spk, layers)
    ua, _, _ = loso(X, AROUSAL[y], spk, [max(set(chosen), key=chosen.count)])
    uv, _, _ = loso(X, VALENCE[y], spk, [max(set(chosen), key=chosen.count)])
    row = {"model": name, "uar4": uar, "sd_spk": sd, "arousal": ua, "valence": uv, "rtf": rtf,
           "layer": max(set(chosen), key=chosen.count), "layers": L}
    if "zs" in d:  # hazır modelin kendi tahmini (eğitimsiz)
        zs = json.loads(str(d["zs"]))
        if name == "audeering_dim":
            A = np.array([zs[str(i)] for i in range(len(y))])   # uyarılma, baskınlık, valens
            row["zs_arousal_auc"] = roc_auc_score(AROUSAL[y], A[:, 0])
            row["zs_valence_auc"] = roc_auc_score(VALENCE[y], A[:, 2])
        elif name == "superb_er":
            P = np.array([zs[str(i)] for i in range(len(y))]); labs = json.loads(str(d["labels"]))
            mp = {"ang": 0, "neu": 1, "hap": 2, "sad": 3}
            row["zs_uar4"] = balanced_accuracy_score(y, [mp[labs[k]] for k in P.argmax(1)])
        elif name.startswith("emotion2vec"):
            mp = {"angry": 0, "neutral": 1, "happy": 2, "sad": 3}
            pr = []
            for i in range(len(y)):
                labs, sc = zs[str(i)]
                labs = [l.split("/")[-1] for l in labs]
                best = max((s, mp[l]) for l, s in zip(labs, sc) if l in mp)
                pr.append(best[1])
            row["zs_uar4"] = balanced_accuracy_score(y, pr)
    rows.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)

print("\nmodel                    UAR(4) ±konuşmacı  uyarılma  valens   RTF    katman  hazır tahmin")
for r in sorted(rows, key=lambda r: -r["uar4"]):
    zs = " ".join(f"{k[3:]}={v:.2f}" for k, v in r.items() if k.startswith("zs_"))
    print(f"{r['model']:24s} {r['uar4']:.3f}  ±{r['sd_spk']:.3f}     {r['arousal']:.3f}    {r['valence']:.3f}  {r['rtf']:.3f}  "
          f"{r['layer']:>2}/{r['layers'] - 1}   {zs}")
json.dump(rows, open(os.path.join(CACHE, "sonuc.json"), "w"), ensure_ascii=False, indent=1)
