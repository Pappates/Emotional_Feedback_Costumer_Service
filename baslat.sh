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
  read -r -p "Anahtar: " KEY
  if [ -n "$KEY" ]; then echo "GEMINI_API_KEY=$KEY" > .env; else echo "# GEMINI_API_KEY=buraya_anahtar" > .env; fi
  chmod 600 .env
fi

echo
echo "Uygulama başlatılıyor: http://localhost:8000  (durdurmak için Ctrl+C)"
( sleep 3; (command -v open >/dev/null && open http://localhost:8000) || (command -v xdg-open >/dev/null && xdg-open http://localhost:8000) || true ) &
exec .venv/bin/python app.py
