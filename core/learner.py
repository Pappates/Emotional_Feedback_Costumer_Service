"""Avcının beyni: iki seviyeli Thompson Sampling (contextual bandit).

Alien'daki yaratık, oyuncunun saklandığı yerlerin ısı haritasını tutar ve
aramaya en "sıcak" yerden başlar. Burada ısı haritası, her söylem stratejisinin
ve ses tonunun getirdiği ortalama ödüldür:

    ödül = (müşterinin algılanan duygusundaki değişim) + 0.15 * (bu hamlede çözülen sorun sayısı)

İki seviye vardır:
  * Görüşme içi: "BU müşteri neye olumlu tepki veriyor?" (hızlı öğrenir, ağırlığı yüksek)
  * Genel (bağlama göre): "İnternet kesintisi yaşayan, öfkeli müşterilerde genelde ne işe yarıyor?"
    Yeni bir görüşme, kendi bağlamının genel bilgisiyle ısınarak başlar.
"""
from __future__ import annotations

import json
import math
import threading
from collections import defaultdict

import numpy as np

from .catalog import STRATEGIES, TONES
from .config import MODEL_FILE

DIMS = {"strateji": list(STRATEGIES), "ton": list(TONES)}

SIGMA = 0.25          # ödül gürültüsü tahmini (gerçek ses analizinde algı oldukça gürültülü)
REWARD_CLIP = 0.5     # tek bir şanslı/şanssız algı sıçramasının öğrenmeyi ele geçirmesini önler
GLOBAL_CAP = 6.0      # genel bilginin görüşme içi bilgiyi en fazla kaç gözlem kadar bastırabileceği
CALL_WEIGHT = 2.0     # görüşme içi gözlemlerin ağırlığı
DECAY = 0.97          # unutma: eski genel gözlemler zamanla etkisini yitirir

_lock = threading.Lock()


def _empty():
    return {"n": 0.0, "sum": 0.0}


class GlobalModel:
    """Tüm görüşmeler boyunca, bağlam başına strateji/ton istatistikleri."""

    def __init__(self, persist: bool = True):
        self.persist = persist
        self.stats: dict = defaultdict(lambda: {d: {a: _empty() for a in arms} for d, arms in DIMS.items()})
        self.episodes: list[dict] = []
        if persist:
            self.load()

    def load(self):
        if MODEL_FILE.exists():
            raw = json.loads(MODEL_FILE.read_text(encoding="utf-8"))
            for ctx, dims in raw.get("stats", {}).items():
                for d, arms in dims.items():
                    for a, s in arms.items():
                        if d in DIMS and a in DIMS[d]:
                            self.stats[ctx][d][a] = s
            self.episodes = raw.get("episodes", [])

    def save(self):
        if not self.persist:
            return
        with _lock:
            MODEL_FILE.write_text(
                json.dumps({"stats": self.stats, "episodes": self.episodes[-500:]}, ensure_ascii=False, indent=1),
                encoding="utf-8",
            )

    def reset(self):
        self.stats.clear()
        self.episodes = []
        if MODEL_FILE.exists():
            MODEL_FILE.unlink()

    def get(self, ctx, dim, arm):
        return self.stats[ctx][dim][arm]

    def update(self, ctx, dim, arm, reward):
        with _lock:
            arms = self.stats[ctx][dim]
            for s in arms.values():          # unutma
                s["n"] *= DECAY
                s["sum"] *= DECAY
            arms[arm]["n"] += 1.0
            arms[arm]["sum"] += reward

    def snapshot(self, ctx):
        out = {}
        for d, arms in DIMS.items():
            out[d] = {}
            for a in arms:
                s = self.stats[ctx][d][a] if ctx in self.stats else _empty()
                out[d][a] = {"n": round(s["n"], 2), "ort": round(s["sum"] / s["n"], 3) if s["n"] > 0.05 else None}
        return out

    def contexts(self):
        return list(self.stats)


class CallLearner:
    """Tek bir görüşmenin öğrenicisi; genel modelden ısınarak başlar."""

    def __init__(self, global_model: GlobalModel, rng: np.random.Generator):
        self.g = global_model
        self.rng = rng
        self.ctx: str | None = None
        self.call = {d: {a: _empty() for a in arms} for d, arms in DIMS.items()}

    def _posterior(self, dim, arm):
        c = self.call[dim][arm]
        g = self.g.get(self.ctx, dim, arm) if self.ctx else _empty()
        g_mean = g["sum"] / g["n"] if g["n"] > 0.05 else 0.0
        k = min(g["n"], GLOBAL_CAP)
        n = 1.0 + k + CALL_WEIGHT * c["n"]          # 1: sıfır ortalamalı zayıf ön bilgi
        mean = (g_mean * k + CALL_WEIGHT * c["sum"]) / n
        return mean, SIGMA / math.sqrt(n)

    def select(self, dim):
        samples = {}
        for arm in DIMS[dim]:
            mean, std = self._posterior(dim, arm)
            samples[arm] = float(self.rng.normal(mean, std))
        best = max(samples, key=samples.get)
        return best, samples

    def update(self, dim, arm, reward):
        reward = max(-REWARD_CLIP, min(REWARD_CLIP, reward))
        self.call[dim][arm]["n"] += 1
        self.call[dim][arm]["sum"] += reward
        if self.ctx:
            self.g.update(self.ctx, dim, arm, reward)

    def snapshot(self):
        out = {}
        for d, arms in DIMS.items():
            out[d] = {}
            for a in arms:
                c = self.call[d][a]
                g = self.g.get(self.ctx, d, a) if self.ctx else _empty()
                mean, std = self._posterior(d, a)
                out[d][a] = {
                    "gorusme_ort": round(c["sum"] / c["n"], 3) if c["n"] else None,
                    "gorusme_n": int(c["n"]),
                    "genel_ort": round(g["sum"] / g["n"], 3) if g["n"] > 0.05 else None,
                    "genel_n": round(g["n"], 1),
                    "tahmin": round(mean, 3),
                    "belirsizlik": round(std, 3),
                }
        return out

    def explain(self, dim, arm):
        c = self.call[dim][arm]
        g = self.g.get(self.ctx, dim, arm) if self.ctx else _empty()
        parts = []
        if c["n"]:
            parts.append(f"bu görüşmede ort. {c['sum'] / c['n']:+.2f} ({int(c['n'])} deneme)")
        else:
            parts.append("bu görüşmede henüz denenmedi")
        if g["n"] > 0.05:
            parts.append(f"benzer müşterilerde ort. {g['sum'] / g['n']:+.2f} ({g['n']:.0f} gözlem)")
        return ", ".join(parts)
