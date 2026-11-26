"""demo_ses klasörü için müşteri / asistan sayfası üretir.

Kullanım: python demo_sayfa.py <demo_ses klasörü> <enjeksiyon_ses klasörü>
omni_demo.jsonl demo_ses klasöründe varsa asistan metni, ölçülen öfke ve süreler de yazılır.
"""
import html
import json
import shutil
import sys
from pathlib import Path

demo = Path(sys.argv[1])
enj = Path(sys.argv[2])

musteri_dir = demo / "musteri"
musteri_dir.mkdir(exist_ok=True)

meta = {}
for line in open(enj / "sonuc.jsonl", encoding="utf-8"):
    r = json.loads(line)
    meta[f"{r['persona']}_g{r['gorusme']}_t{r['tur']}_{r['varyant']}"] = r

demo_kayit = {}
jl = demo / "omni_demo.jsonl"
if jl.exists():
    for line in open(jl, encoding="utf-8"):
        r = json.loads(line)
        demo_kayit[r["tepki"]] = r

ozet = None
oz = demo / "ozet.json"
if oz.exists():
    ozet = json.load(open(oz, encoding="utf-8")).get("omni_demo", {}).get("NVIDIA L4")
if ozet is None:  # L4 ölçümü, 26.11.2026
    ozet = {"n": 20, "asr_ort": 0.379, "e2v_ort": 0.028, "dusunur_ort": 1.873, "toplam_ilk_ses_ort": 2.531,
            "toplam_ilk_ses_medyan": 2.44, "toplam_ilk_ses_max": 2.96, "takilmali": 0, "empati_orani": 0.55,
            "sizinti": 0}

e = html.escape
kartlar = []
for wav in sorted(demo.glob("*.wav")):
    ad = wav.stem
    kaynak = enj / f"{ad}.wav"
    if kaynak.exists():
        shutil.copyfile(kaynak, musteri_dir / wav.name)
    m = meta.get(ad, {})
    d = demo_kayit.get(ad, {})
    musteri_metin = d.get("musteri") or m.get("tepki", "")
    olcum = ""
    if "ofke" in d:
        ofke = float(d["ofke"])
        sinif = "kizgin" if ofke >= 1.5 else "sakin"
        olcum = (f'<span class="rozet {sinif}">öfke {ofke:.2f}/4 · {e(str(d.get("ofke_ad", "")))}</span>'
                 f'<span class="rozet">ilk ses {float(d.get("toplam_ilk_ses", 0)):.2f} sn</span>')
    cevap = d.get("cevap")
    cevap_html = f'<p class="metin">{e(cevap)}</p>' if cevap else ""
    kartlar.append(f"""
<article class="kart">
  <div class="baslik"><span class="kod">{e(ad)}</span>{olcum}</div>
  <div class="satir musteri">
    <div class="kim">Müşteri</div>
    <div class="govde"><p class="metin">{e(musteri_metin)}</p>
      <audio controls preload="none" src="musteri/{e(wav.name)}"></audio></div>
  </div>
  <div class="satir asistan">
    <div class="kim">Asistan</div>
    <div class="govde">{cevap_html}
      <audio controls preload="none" src="{e(wav.name)}"></audio></div>
  </div>
</article>""")

algi = ozet["asr_ort"] + ozet["e2v_ort"]
sayfa = f"""<!doctype html>
<html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Duygu Avcısı Demo</title>
<style>
:root {{ --zemin:#f6f7f9; --kart:#fff; --yazi:#1d2330; --soluk:#5d6677; --cizgi:#e2e5ea;
        --musteri:#fdf1ee; --asistan:#eef4fd; --vurgu:#e2001a; --sakin:#2f7d4f; }}
@media (prefers-color-scheme: dark) {{ :root {{ --zemin:#14171c; --kart:#1d2128; --yazi:#e8eaee; --soluk:#9aa3b2;
        --cizgi:#2c313a; --musteri:#2b1f1d; --asistan:#1b2433; --sakin:#5cc488; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--zemin); color:var(--yazi);
       font:16px/1.5 system-ui, "Segoe UI", sans-serif; }}
main {{ max-width:860px; margin:0 auto; padding:24px 16px 48px; }}
h1 {{ font-size:1.5rem; margin:0 0 4px; }}
.alt {{ color:var(--soluk); margin:0 0 20px; }}
.ozet {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin-bottom:24px; }}
.kutu {{ background:var(--kart); border:1px solid var(--cizgi); border-radius:10px; padding:12px; }}
.kutu b {{ display:block; font-size:1.35rem; }}
.kutu span {{ color:var(--soluk); font-size:.85rem; }}
.kart {{ background:var(--kart); border:1px solid var(--cizgi); border-radius:12px; padding:14px; margin-bottom:14px; }}
.baslik {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-bottom:10px; }}
.kod {{ font-family:ui-monospace, Consolas, monospace; color:var(--soluk); font-size:.85rem; margin-right:auto; }}
.rozet {{ font-size:.8rem; padding:2px 8px; border-radius:99px; border:1px solid var(--cizgi); }}
.rozet.kizgin {{ color:var(--vurgu); border-color:var(--vurgu); }}
.rozet.sakin {{ color:var(--sakin); border-color:var(--sakin); }}
.satir {{ display:flex; gap:12px; padding:10px; border-radius:8px; margin-top:8px; }}
.musteri {{ background:var(--musteri); }} .asistan {{ background:var(--asistan); }}
.kim {{ width:70px; flex:none; font-weight:600; font-size:.9rem; }}
.govde {{ flex:1; min-width:0; }}
.metin {{ margin:0 0 6px; }}
audio {{ width:100%; height:36px; }}
@media (max-width:520px) {{ .satir {{ flex-direction:column; gap:4px; }} }}
</style></head>
<body><main>
<h1>Duygu Avcısı: sesten sese demo</h1>
<p class="alt">TR-Omni-E4B + emotion2vec+. Müşteri sesinden öfke canlı ölçülür, cevaba not olarak eklenir, cevap
kısa tutulur; önce düşünür, sonra konuşucu. NVIDIA L4, {ozet['n']} sentetik müşteri tepkisi.</p>
<section class="ozet">
  <div class="kutu"><b>{algi:.2f} sn</b><span>öfke algısı (Whisper + emotion2vec+)</span></div>
  <div class="kutu"><b>{ozet['dusunur_ort']:.2f} sn</b><span>cevabın yazılması</span></div>
  <div class="kutu"><b>{ozet['toplam_ilk_ses_ort']:.2f} sn</b><span>müşteri sustuktan sonra ilk ses (en çok {ozet['toplam_ilk_ses_max']:.2f})</span></div>
  <div class="kutu"><b>{ozet['takilmali']}</b><span>takılan cevap</span></div>
  <div class="kutu"><b>%{ozet['empati_orani'] * 100:.0f}</b><span>empatik cevap</span></div>
</section>
{''.join(kartlar)}
</main></body></html>
"""
(demo / "demo.html").write_text(sayfa, encoding="utf-8")
print(f"{len(kartlar)} örnek, cevap metni olan: {sum(1 for w in demo.glob('*.wav') if w.stem in demo_kayit)}")
