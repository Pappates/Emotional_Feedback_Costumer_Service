"""Duygu Avcısı - web sunucusu.  Çalıştırma:  python app.py   ->  http://localhost:8000"""
from __future__ import annotations

import json
import random
from contextlib import asynccontextmanager
import threading
import traceback

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core import config, report
from core.call import Call, run_fast
from core.catalog import ISSUES, PERSONAS, STRATEGIES, TONES
from core.learner import GlobalModel

@asynccontextmanager
async def lifespan(_app):
    if config.default_mode() == "lokal":
        from core import local
        print("[lokal] modeller arka planda yükleniyor (ilk açılışta indirme birkaç dakika sürebilir)…", flush=True)
        local.warmup()
    yield


app = FastAPI(title="Duygu Avcısı", lifespan=lifespan)
GLOBAL = GlobalModel()
CALLS: dict[str, Call] = {}
RECORDINGS: dict[str, dict] = {}        # görüşme id -> {persona, events}
REC_DIR = config.DATA_DIR / "kayitlar"
REC_DIR.mkdir(exist_ok=True)


def _record(c: Call, tip: str, data: dict):
    """Sesli görüşmeleri kaydeder; panelden API kullanmadan tekrar oynatılabilir (demo güvencesi)."""
    rec = RECORDINGS.setdefault(c.id, {"id": c.id, "persona": c.persona_id, "ses": c.audio, "mod": c.mode, "events": []})
    rec["events"].append({"tip": tip, "data": data})
    if c.ended and tip == "asistan" and c.mode != "mock":
        rec["sonuc"] = c.outcome
        (REC_DIR / f"{c.id}.json").write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
LOCK = threading.Lock()


@app.exception_handler(Exception)
async def on_error(_, exc: Exception):
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"hata": f"{type(exc).__name__}: {exc}"})


class StartReq(BaseModel):
    persona: str = "rastgele"
    ses: bool = True


class TrainReq(BaseModel):
    n: int = 60


MODE_NAMES = {"gemini": "GEMINI", "lokal": "LOKAL (Ollama + Hugging Face, kota yok)", "mock": "DEMO (API anahtarı yok)"}


def _models():
    if config.default_mode() == "lokal":
        from core import local
        return {"metin": local.LLM_MODEL, "ses": "Piper"}
    return {"metin": config.TEXT_MODEL, "ses": config.TTS_MODEL}


def _local_status():
    if config.default_mode() != "lokal":
        return None
    from core import local
    return local.STATUS


@app.get("/")
def index():
    return FileResponse(config.ROOT / "static" / "index.html")


@app.get("/api/info")
def info():
    return {
        "mod": config.default_mode(),
        "metin_modeli": _models()["metin"], "tts_modeli": _models()["ses"],
        "lokal_durum": _local_status(),
        "kayitlar": recordings(),
        "personalar": {k: {"ad": v["ad"], "ozet": v["ozet"], "sorunlar": [ISSUES[i]["ad"] for i in v["sorunlar"]],
                           "tercih_strateji": v["tercih_strateji"], "tercih_ton": v["tercih_ton"]}
                       for k, v in PERSONAS.items()},
        "stratejiler": {k: v["ad"] for k, v in STRATEGIES.items()},
        "tonlar": {k: v["ad"] for k, v in TONES.items()},
        "bolumler": GLOBAL.episodes[-200:],
    }


def _get(cid) -> Call:
    if cid not in CALLS:
        raise HTTPException(404, "Görüşme bulunamadı")
    return CALLS[cid]


@app.post("/api/call/start")
def start(req: StartReq):
    pid = req.persona if req.persona in PERSONAS else random.choice(list(PERSONAS))
    c = Call(pid, GLOBAL, audio=req.ses)
    CALLS[c.id] = c
    turn = c.customer_turn()
    _record(c, "musteri", turn)
    return {"id": c.id, "persona": pid, "mod": c.mode, "ses": c.audio, "tur": turn}


@app.post("/api/call/{cid}/agent")
def agent(cid: str):
    c = _get(cid)
    with LOCK:
        out = c.agent_turn()
    _record(c, "asistan", out)
    return out


@app.post("/api/call/{cid}/customer")
def customer(cid: str):
    c = _get(cid)
    if c.ended:
        raise HTTPException(400, "Görüşme bitti")
    with LOCK:
        out = c.customer_turn()
    _record(c, "musteri", out)
    return out


@app.get("/api/recordings")
def recordings():
    out = []
    for f in sorted(REC_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
        r = json.loads(f.read_text(encoding="utf-8"))
        out.append({"id": r["id"], "persona": r["persona"], "ad": PERSONAS[r["persona"]]["ad"], "sonuc": r.get("sonuc"),
                    "ses": r.get("ses"), "tur": sum(e["tip"] == "asistan" for e in r["events"]), "zaman": f.stat().st_mtime})
    return out


@app.get("/api/recordings/{rid}")
def recording(rid: str):
    f = REC_DIR / f"{rid}.json"
    if not f.exists():
        raise HTTPException(404, "Kayıt bulunamadı")
    report.ensure_from_recordings(REC_DIR)              # oynatılan kaydın raporu hazır olsun
    rec = json.loads(f.read_text(encoding="utf-8"))
    rec["rapor"] = f"/raporlar/{rid}"
    return rec


@app.get("/raporlar", response_class=HTMLResponse)
def reports_index():
    report.ensure_from_recordings(REC_DIR)
    return report.index_html()


def _report_file(rid: str, name: str):
    if rid.isalnum() and not (report.REPORT_DIR / rid / "rapor.html").exists() and (REC_DIR / f"{rid}.json").exists():
        report.ensure_from_recordings(REC_DIR)          # kayıtlı görüşmenin raporu henüz yoksa üret
    f = report.REPORT_DIR / rid / name
    if not rid.isalnum() or not f.exists():
        raise HTTPException(404, "Rapor bulunamadı")
    return f


@app.get("/raporlar/{rid}", response_class=HTMLResponse)
def report_html(rid: str):
    return _report_file(rid, "rapor.html").read_text(encoding="utf-8")


@app.get("/raporlar/{rid}/html")
def report_html_download(rid: str):
    """Raporu tek başına açılabilen HTML dosyası olarak indirir (CSS ve grafikler dosyanın içinde)."""
    data = _report_file(rid, "rapor.html").read_bytes()
    return Response(data, media_type="text/html; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="duygu-raporu-{rid}.html"'})


@app.get("/raporlar/{rid}/csv")
def report_csv(rid: str):
    data = _report_file(rid, "rapor.csv").read_bytes()
    return Response(data, media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="duygu-raporu-{rid}.csv"'})


@app.get("/raporlar/{rid}/json")
def report_json(rid: str):
    return FileResponse(_report_file(rid, "rapor.json"), media_type="application/json")


@app.post("/api/train")
def train(req: TrainReq):
    n = max(1, min(req.n, 500))
    seed = random.randint(0, 10**6)
    with LOCK:
        learned = run_fast(GLOBAL, n, seed=seed)
        baseline = run_fast(GLOBAL, n, seed=seed, policy="rastgele")
    return {"ogrenen": learned, "rastgele": baseline, "bolumler": GLOBAL.episodes[-200:]}


@app.post("/api/reset")
def reset():
    GLOBAL.reset()
    return {"ok": True}


app.mount("/static", StaticFiles(directory=config.ROOT / "static"), name="static")
app.mount("/audio", StaticFiles(directory=config.AUDIO_DIR), name="audio")

if __name__ == "__main__":
    print(f"Mod: {MODE_NAMES[config.default_mode()]} -> http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
