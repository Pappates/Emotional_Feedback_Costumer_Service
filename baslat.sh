#!/usr/bin/env bash
# Duygu Avcısı - macOS / Linux tek tık kurulum ve başlatma
set -e
cd "$(dirname "$0")"

PY=$(command -v python3 || command -v python || true)
if [ -z "$PY" ]; then
  echo "[HATA] Python bulunamadı. Python 3.10+ kurun: https://www.python.org/downloads/"; exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
  echo "İlk kurulum yapılıyor, bu bir-iki dakika sürebilir..."
  "$PY" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip -q
  .venv/bin/python -m pip install -r requirements.txt -q
fi

if [ ! -f ".env" ]; then
  echo
  echo "Gemini API anahtarınızı yapıştırın (https://aistudio.google.com/apikey)."
  echo "Boş bırakıp Enter'a basarsanız uygulama demo modunda çalışır."
  echo "Bulut yerine bilgisayardaki lokal modelleri kullanmak için \"lokal\" yazın."
  read -r -p "Anahtar: " KEY
  if [ "$KEY" = "lokal" ]; then echo "SAGLAYICI=lokal" > .env
  elif [ -n "$KEY" ]; then echo "GEMINI_API_KEY=$KEY" > .env; else echo "# GEMINI_API_KEY=buraya_anahtar" > .env; fi
  chmod 600 .env
fi

if grep -qi '^SAGLAYICI=lokal' .env 2>/dev/null; then
  if [ ! -f .venv/lokal-kurulu.txt ]; then
    echo "Lokal model paketleri kuruluyor (torch, whisper, transformers, piper). İlk seferde 5-10 dakika sürebilir..."
    if [ "$(uname)" = "Linux" ]; then
      .venv/bin/python -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu || true   # GPU'suz küçük torch
    fi
    .venv/bin/python -m pip install -q -r requirements-lokal.txt
    echo ok > .venv/lokal-kurulu.txt
  fi
  LLM=$(grep -i '^LOKAL_LLM=' .env | cut -d= -f2-); LLM=${LLM:-qwen2.5:3b}
  if command -v ollama >/dev/null; then
    ollama list | grep -qi "$LLM" || { echo "$LLM modeli indiriliyor (ilk sefer, ~2 GB)..."; ollama pull "$LLM"; }
  else
    echo "[UYARI] Ollama bulunamadı. https://ollama.com/download adresinden kurun, sonra: ollama pull $LLM"
  fi
fi

echo
echo "Uygulama başlatılıyor: http://localhost:8000  (durdurmak için Ctrl+C)"
( sleep 3; (command -v open >/dev/null && open http://localhost:8000) || (command -v xdg-open >/dev/null && xdg-open http://localhost:8000) || true ) &
exec .venv/bin/python app.py
